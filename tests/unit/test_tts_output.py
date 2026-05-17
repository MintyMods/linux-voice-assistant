"""Stage B3 — TTSOutput unit tests against a fake Wyoming Piper server.

Round-trips a Synthesize event → AudioStart/AudioChunk+/AudioStop, asserts
the resulting tempfile contains a valid WAV, and that the mpv player receives
it for playback.
"""

from __future__ import annotations

import asyncio
import io
import os
import wave
from typing import Optional

import pytest
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import async_read_event, async_write_event
from wyoming.tts import Synthesize

from linux_voice_assistant.tts_output import TTSError, TTSOutput


class _FakePiper:
    def __init__(self, pcm_ms: int = 200, rate: int = 22050) -> None:
        self.pcm_ms = pcm_ms
        self.rate = rate
        self.synth_seen: Optional[Synthesize] = None
        self.empty_audio = False
        self.host = "127.0.0.1"
        self.port = 0
        self._server: Optional[asyncio.AbstractServer] = None

    async def start(self) -> None:
        self._server = await asyncio.start_server(self._handle, self.host, 0)
        self.port = self._server.sockets[0].getsockname()[1]

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

    @property
    def uri(self) -> str:
        return f"tcp://{self.host}:{self.port}"

    async def _handle(self, reader, writer):
        try:
            event = await async_read_event(reader)
            if event is None or not Synthesize.is_type(event.type):
                return
            self.synth_seen = Synthesize.from_event(event)
            if self.empty_audio:
                # Close without sending audio — TTSError should fire.
                return
            n_samples = (self.rate * self.pcm_ms) // 1000
            pcm = bytes(n_samples * 2)
            await async_write_event(
                AudioStart(rate=self.rate, width=2, channels=1).event(), writer
            )
            await async_write_event(
                AudioChunk(rate=self.rate, width=2, channels=1, audio=pcm).event(), writer
            )
            await async_write_event(AudioStop().event(), writer)
        except (ConnectionResetError, asyncio.IncompleteReadError):
            return
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass


class _FakePlayer:
    def __init__(self) -> None:
        self.played: list[str] = []
        self.done_callback = None

    def play(self, url, done_callback=None, stop_first: bool = False) -> None:
        self.played.append(url)
        self.done_callback = done_callback


@pytest.mark.asyncio
async def test_speak_round_trip_writes_wav_and_queues_play():
    server = _FakePiper()
    await server.start()
    player = _FakePlayer()
    try:
        out = TTSOutput(server.uri, voice="en_GB-alba-medium")
        await out.speak(player, text="hello there")
    finally:
        await server.stop()

    assert player.played, "mpv should have received a play() call"
    path = player.played[0]
    assert os.path.exists(path)
    with wave.open(path, "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == 22050
    assert server.synth_seen is not None
    assert server.synth_seen.text == "hello there"
    assert server.synth_seen.voice is not None
    assert server.synth_seen.voice.name == "en_GB-alba-medium"

    # Trigger done_callback to exercise cleanup of the *previous* tempfile.
    # (There is no previous on a first speak, so this is a no-op smoke test.)
    assert player.done_callback is not None
    player.done_callback()
    # Cleanup the active file ourselves to avoid leaving tmpfiles around.
    try:
        os.unlink(path)
    except OSError:
        pass


@pytest.mark.asyncio
async def test_empty_text_skips_synth_fires_callback():
    out = TTSOutput("tcp://127.0.0.1:1")
    player = _FakePlayer()
    fired = []
    await out.speak(player, text="   ", done_callback=lambda: fired.append(True))
    assert player.played == []
    assert fired == [True]


@pytest.mark.asyncio
async def test_empty_audio_raises_tts_error():
    server = _FakePiper()
    server.empty_audio = True
    await server.start()
    try:
        out = TTSOutput(server.uri)
        with pytest.raises(TTSError):
            await out.speak(_FakePlayer(), text="hello")
    finally:
        await server.stop()
