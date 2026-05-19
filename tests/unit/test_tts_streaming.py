"""Stage E.2 E2/E3 — TTSStreamingOutput unit tests.

PROTOTYPE caveat: these tests verify the WIRE shape (chunks written to
mpv stdin in order, AudioStop ends the stream, done callback fires, mpv
respawn on dead subprocess) but do NOT verify on-hardware playback. The
real first-audio-latency check needs a Piper container at the configured
WYOMING_PIPER_URI and an mpv binary that can decode rawaudio.
"""

from __future__ import annotations

import asyncio
import subprocess
from typing import List
from unittest.mock import MagicMock

import pytest

from linux_voice_assistant import tts_streaming
from linux_voice_assistant.tts_streaming import (
    TTSStreamingError,
    TTSStreamingOutput,
)


class _FakeMpvProc:
    """Stand-in for subprocess.Popen returned from _spawn_mpv."""

    def __init__(self, *, returncode_after: int | None = None) -> None:
        self.stdin = _FakeStdin()
        self._returncode_after = returncode_after
        self.returncode: int | None = None
        self.terminated = False
        self.killed = False

    def poll(self) -> int | None:
        return self._returncode_after

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True

    def wait(self, timeout: float | None = None) -> int:
        self.returncode = self._returncode_after or 0
        return self.returncode


class _FakeStdin:
    def __init__(self) -> None:
        self.writes: List[bytes] = []
        self.flushes: int = 0
        self.closed = False

    def write(self, data: bytes) -> int:
        if self.closed:
            raise BrokenPipeError("stdin closed")
        self.writes.append(bytes(data))
        return len(data)

    def flush(self) -> None:
        self.flushes += 1

    def close(self) -> None:
        self.closed = True


class _FakeEvent:
    def __init__(self, event_type: str, **payload):
        self.type = event_type
        self._payload = payload


class _FakeWyomingClient:
    """Plays back a scripted sequence of events on read_event."""

    def __init__(self, events):
        self._events = list(events)
        self.written = []

    @classmethod
    def from_uri(cls, _uri):
        return cls.factory_for_uri

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def write_event(self, event):
        self.written.append(event)

    async def read_event(self):
        if not self._events:
            return None
        return self._events.pop(0)


@pytest.fixture
def patched_streaming(monkeypatch):
    """Patch subprocess.Popen + the wyoming bits used by TTSStreamingOutput."""

    spawned: List[_FakeMpvProc] = []

    def _spawn(*args, **kwargs):
        proc = _FakeMpvProc()
        spawned.append(proc)
        return proc

    monkeypatch.setattr(tts_streaming.subprocess, "Popen", _spawn)

    # Patch AudioStart/AudioChunk/AudioStop type-checks + .from_event so we
    # can hand back hand-built event objects without depending on wyoming's
    # event-name strings.
    def _is_type_factory(want):
        return lambda etype: etype == want

    monkeypatch.setattr(tts_streaming.AudioStart, "is_type", _is_type_factory("audio-start"))
    monkeypatch.setattr(tts_streaming.AudioChunk, "is_type", _is_type_factory("audio-chunk"))
    monkeypatch.setattr(tts_streaming.AudioStop, "is_type", _is_type_factory("audio-stop"))

    def _audio_start_from_event(event):
        return MagicMock(
            rate=event._payload["rate"],
            width=event._payload["width"],
            channels=event._payload["channels"],
        )

    def _audio_chunk_from_event(event):
        return MagicMock(audio=event._payload["audio"])

    monkeypatch.setattr(tts_streaming.AudioStart, "from_event", _audio_start_from_event)
    monkeypatch.setattr(tts_streaming.AudioChunk, "from_event", _audio_chunk_from_event)

    return spawned


def _make_client(events):
    client = _FakeWyomingClient(events)
    return client


@pytest.fixture
def patch_client(monkeypatch):
    """Replace AsyncClient.from_uri with a factory returning our fake."""
    holder = {"client": None}

    def _factory_for_events(events):
        client = _FakeWyomingClient(events)
        holder["client"] = client

        class _CtxFactory:
            @staticmethod
            def from_uri(_uri):
                return client

        return _CtxFactory

    def _install(events):
        ctx = _factory_for_events(events)
        monkeypatch.setattr(tts_streaming, "AsyncClient", ctx)
        return holder

    return _install


def test_speak_writes_chunks_to_mpv_stdin_in_order(patched_streaming, patch_client):
    events = [
        _FakeEvent("audio-start", rate=22050, width=2, channels=1),
        _FakeEvent("audio-chunk", audio=b"\x01\x02"),
        _FakeEvent("audio-chunk", audio=b"\x03\x04"),
        _FakeEvent("audio-stop"),
    ]
    holder = patch_client(events)

    out = TTSStreamingOutput(uri="tcp://piper:10200", voice="en_GB-jenny_dioco-medium")
    asyncio.run(out.speak(None, text="Hi there.", done_callback=None))

    proc = patched_streaming[0]
    assert proc.stdin.writes == [b"\x01\x02", b"\x03\x04"]
    assert proc.stdin.flushes == 2
    assert holder["client"] is not None


def test_speak_empty_text_short_circuits(patched_streaming, patch_client):
    """Empty text must not spawn mpv or open a Wyoming connection."""
    out = TTSStreamingOutput(uri="tcp://piper:10200")
    cb = MagicMock()

    asyncio.run(out.speak(None, text="", done_callback=cb))

    assert patched_streaming == []
    cb.assert_called_once()


def test_speak_schedules_done_callback_after_drain(patched_streaming, patch_client):
    events = [
        _FakeEvent("audio-start", rate=22050, width=2, channels=1),
        _FakeEvent("audio-chunk", audio=b"\x00\x00"),
        _FakeEvent("audio-stop"),
    ]
    patch_client(events)
    out = TTSStreamingOutput(uri="tcp://piper:10200")

    cb = MagicMock()

    async def _run():
        await out.speak(None, text="x", done_callback=cb)
        # Done callback is scheduled via loop.call_later(0.6); wait it out.
        await asyncio.sleep(0.7)

    asyncio.run(_run())
    cb.assert_called_once()


def test_speak_respawns_mpv_when_subprocess_dies(patched_streaming, patch_client, monkeypatch):
    """If the persistent mpv has died, the next speak() must spawn a fresh
    process rather than write into a dead stdin."""
    out = TTSStreamingOutput(uri="tcp://piper:10200")

    # Force the first spawn to return a "dead" proc (poll() returns 1).
    counter = {"i": 0}

    def _spawn(*args, **kwargs):
        proc = _FakeMpvProc(returncode_after=1 if counter["i"] == 0 else None)
        counter["i"] += 1
        patched_streaming.append(proc)
        return proc

    monkeypatch.setattr(tts_streaming.subprocess, "Popen", _spawn)

    # Manually inject the "dead" proc via _ensure_mpv first.
    asyncio.run(out._ensure_mpv())
    assert len(patched_streaming) == 1
    # _ensure_mpv was called; mpv proc has returncode=1 → it's dead.
    # Call again — should respawn.
    asyncio.run(out._ensure_mpv())
    assert len(patched_streaming) == 2


def test_speak_wraps_piper_exception_in_streaming_error(patched_streaming, monkeypatch):
    """A wyoming-side failure must surface as TTSStreamingError, not the
    underlying exception type."""

    class _BrokenClient:
        @staticmethod
        def from_uri(_uri):
            class _Ctx:
                async def __aenter__(self):
                    raise OSError("connection refused")

                async def __aexit__(self, *exc):
                    return False

            return _Ctx()

    monkeypatch.setattr(tts_streaming, "AsyncClient", _BrokenClient)

    out = TTSStreamingOutput(uri="tcp://piper:10200")
    with pytest.raises(TTSStreamingError):
        asyncio.run(out.speak(None, text="x"))


def test_spawn_mpv_args_include_rawaudio_demuxer(patched_streaming, monkeypatch):
    """Regression: the spawn args must include the rawaudio demuxer + rate
    + channels matching the Piper voice config."""
    captured_args = []

    def _spawn(args, *_a, **_kw):
        captured_args.append(args)
        return _FakeMpvProc()

    monkeypatch.setattr(tts_streaming.subprocess, "Popen", _spawn)

    out = TTSStreamingOutput(uri="tcp://piper:10200", sample_rate=22050, channels=1, sample_width=2)
    asyncio.run(out._ensure_mpv())

    assert len(captured_args) == 1
    args = captured_args[0]
    assert "--demuxer=rawaudio" in args
    assert any(a.startswith("--demuxer-rawaudio-rate=22050") for a in args)
    assert any(a.startswith("--demuxer-rawaudio-channels=1") for a in args)
    assert any(a.startswith("--demuxer-rawaudio-format=s16le") for a in args)
    assert "-" in args  # stdin sentinel


def test_broken_pipe_marks_proc_dead(patched_streaming, patch_client, monkeypatch):
    """A BrokenPipeError mid-stream must not propagate — instead the
    subprocess is marked dead so the next speak() respawns."""
    events = [
        _FakeEvent("audio-start", rate=22050, width=2, channels=1),
        _FakeEvent("audio-chunk", audio=b"\x00"),
        _FakeEvent("audio-stop"),
    ]
    patch_client(events)
    out = TTSStreamingOutput(uri="tcp://piper:10200")

    async def _run():
        await out._ensure_mpv()
        # Forcibly close stdin so write raises.
        proc = patched_streaming[0]
        proc.stdin.closed = True
        await out.speak(None, text="x")

    # Should NOT raise; the BrokenPipeError is swallowed and proc is nulled.
    asyncio.run(_run())
    assert out._mpv_proc is None


def test_close_terminates_subprocess(patched_streaming, patch_client):
    out = TTSStreamingOutput(uri="tcp://piper:10200")
    asyncio.run(out._ensure_mpv())
    proc = patched_streaming[0]

    out.close()

    assert proc.terminated is True
    assert proc.stdin.closed is True
    assert out._mpv_proc is None
