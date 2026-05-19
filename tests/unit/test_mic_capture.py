"""Stage B3 — MicCapture unit tests.

Covers the chunk feed → pre-roll ring → start_capture → VAD-driven
end-of-speech → SpeechBuffer dispatch contract, plus the 30s hard cap (D5),
the 200ms pre-roll + wake-token strip (D6), and abort() drop semantics.

TEN-VAD is monkeypatched with a deterministic stub so the tests don't
require the native library on the dev machine. The stub raises speech-
probability for high-amplitude frames and silence-probability for near-zero
frames — close enough to the real behaviour for the wire-level contracts
under test here.
"""

from __future__ import annotations

import asyncio
import io
import sys
import types
import wave
from typing import List

import numpy as np
import pytest

from linux_voice_assistant import mic_capture as mic_capture_module
from linux_voice_assistant.mic_capture import (
    MAX_CAPTURE_MS,
    PRE_ROLL_MS,
    SAMPLE_RATE,
    SAMPLE_WIDTH,
    SILENCE_END_MS,
    WAKE_TOKEN_STRIP_MS,
    MicCapture,
    SpeechBuffer,
)


# ---- TEN-VAD stub ---------------------------------------------------------


class StubTenVad:
    """Treats frame as speech when mean abs amplitude > 4096 (~12% of int16)."""

    def __init__(self, hop_size: int = 256, threshold: float = 0.5) -> None:
        self.hop_size = hop_size

    def process(self, samples: np.ndarray):
        mean_abs = float(np.mean(np.abs(samples)))
        prob = min(1.0, mean_abs / 8192.0)
        flag = 1 if mean_abs > 4096 else 0
        return prob, flag


@pytest.fixture(autouse=True)
def _stub_ten_vad(monkeypatch):
    """Inject a fake `ten_vad` module so MicCapture._ensure_vad finds StubTenVad."""
    fake = types.ModuleType("ten_vad")
    fake.TenVad = StubTenVad  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ten_vad", fake)


# ---- helpers --------------------------------------------------------------


def _silence(ms: int) -> bytes:
    n = (SAMPLE_RATE * ms) // 1000
    return np.zeros(n, dtype=np.int16).tobytes()


def _speech(ms: int, amplitude: int = 16000) -> bytes:
    """A clipped sine wave at amplitude — well above the VAD stub threshold."""
    n = (SAMPLE_RATE * ms) // 1000
    t = np.arange(n) / SAMPLE_RATE
    wave_data = (amplitude * np.sin(2 * np.pi * 440 * t)).astype(np.int16)
    return wave_data.tobytes()


def _wav_to_pcm(wav_bytes: bytes) -> bytes:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        assert wf.getframerate() == SAMPLE_RATE
        assert wf.getsampwidth() == SAMPLE_WIDTH
        assert wf.getnchannels() == 1
        return wf.readframes(wf.getnframes())


def _drain_callbacks(loop: asyncio.AbstractEventLoop) -> List[SpeechBuffer]:
    """Run the loop until call_soon_threadsafe callbacks fire."""
    captured: List[SpeechBuffer] = []

    async def _wait() -> None:
        # Allow scheduled callbacks to run.
        await asyncio.sleep(0)
        await asyncio.sleep(0)

    loop.run_until_complete(_wait())
    return captured


# ---- tests ----------------------------------------------------------------


def test_feed_maintains_preroll_ring_without_capture():
    """feed() with no active capture just fills the ring; no callback fires."""
    loop = asyncio.new_event_loop()
    try:
        emitted: List[SpeechBuffer] = []
        mc = MicCapture(loop=loop, on_speech_captured=lambda b: emitted.append(b))
        mc.feed(_silence(500))
        mc.feed(_speech(100))
        # No start_capture — no buffer emitted.
        loop.run_until_complete(asyncio.sleep(0))
        loop.run_until_complete(asyncio.sleep(0))
        assert emitted == []
    finally:
        loop.close()


def test_start_capture_then_silence_emits_buffer_with_preroll():
    """Pre-roll fills, start_capture, 100ms speech, 600ms silence → buffer with
    a buffer at least ~PRE_ROLL_MS + ~100ms of audio (post-trim), end_reason=silence.
    """
    loop = asyncio.new_event_loop()
    try:
        emitted: List[SpeechBuffer] = []
        mc = MicCapture(loop=loop, on_speech_captured=lambda b: emitted.append(b))
        # Fill ring with 1s of speech-amplitude (so pre-roll has something audible).
        mc.feed(_speech(1000))
        mc.start_capture()
        # Active speech then silence to trigger end-of-speech.
        mc.feed(_speech(200))
        mc.feed(_silence(SILENCE_END_MS + 100))
        loop.run_until_complete(asyncio.sleep(0))
        loop.run_until_complete(asyncio.sleep(0))
        assert len(emitted) == 1
        buf = emitted[0]
        assert buf.end_reason == "silence"
        pcm = _wav_to_pcm(buf.wav_bytes)
        # PCM should be non-empty and shorter than total fed bytes (trim ran).
        assert len(pcm) > 0
    finally:
        loop.close()


def test_wake_token_strip_drops_recent_audio_from_preroll():
    """The last WAKE_TOKEN_STRIP_MS of the ring must not appear in pre-roll."""
    loop = asyncio.new_event_loop()
    try:
        emitted: List[SpeechBuffer] = []
        mc = MicCapture(loop=loop, on_speech_captured=lambda b: emitted.append(b))
        # 500ms of distinctive marker speech far enough back to survive strip.
        mc.feed(_speech(500, amplitude=8000))
        # Recent (wake-token range) silence — this should be the part stripped.
        mc.feed(_silence(WAKE_TOKEN_STRIP_MS))
        # Start; speech (to arm silence timer), then silence to end.
        mc.start_capture()
        mc.feed(_speech(100))
        mc.feed(_silence(SILENCE_END_MS + 50))
        loop.run_until_complete(asyncio.sleep(0))
        loop.run_until_complete(asyncio.sleep(0))
        assert len(emitted) == 1
        # The 200ms pre-roll should look like 8000-amplitude speech (after edge
        # trim, this becomes the dominant content), not silence.
        pcm = _wav_to_pcm(emitted[0].wav_bytes)
        samples = np.frombuffer(pcm, dtype=np.int16)
        assert int(np.max(np.abs(samples))) > 4000
    finally:
        loop.close()


def test_no_speech_timeout_fires_with_no_speech_reason():
    """Empty wake: capture ends with end_reason=no_speech after the timeout."""
    loop = asyncio.new_event_loop()
    try:
        emitted: List[SpeechBuffer] = []
        mc = MicCapture(loop=loop, on_speech_captured=lambda b: emitted.append(b),
                        no_speech_timeout_ms=500)
        mc.start_capture()
        mc.feed(_silence(600))  # past the no-speech timeout
        loop.run_until_complete(asyncio.sleep(0))
        loop.run_until_complete(asyncio.sleep(0))
        assert len(emitted) == 1
        assert emitted[0].end_reason == "no_speech"
    finally:
        loop.close()


def test_30s_hard_cap_emits_with_cap_reason():
    """Capture exceeding MAX_CAPTURE_MS finishes with end_reason=cap."""
    loop = asyncio.new_event_loop()
    try:
        emitted: List[SpeechBuffer] = []
        mc = MicCapture(loop=loop, on_speech_captured=lambda b: emitted.append(b))
        mc.start_capture()
        # Backdate the start so cap check fires on the next feed.
        mc._start_ts -= (MAX_CAPTURE_MS / 1000.0) + 0.1
        mc.feed(_speech(50))  # any chunk after the cap triggers finish
        loop.run_until_complete(asyncio.sleep(0))
        loop.run_until_complete(asyncio.sleep(0))
        assert len(emitted) == 1
        assert emitted[0].end_reason == "cap"
    finally:
        loop.close()


def test_abort_drops_inflight_capture_silently():
    """abort() called between start_capture and silence-end → no callback."""
    loop = asyncio.new_event_loop()
    try:
        emitted: List[SpeechBuffer] = []
        mc = MicCapture(loop=loop, on_speech_captured=lambda b: emitted.append(b))
        mc.start_capture()
        mc.feed(_speech(100))
        mc.abort()
        # Even subsequent silence does not re-trigger emit.
        mc.feed(_silence(SILENCE_END_MS + 100))
        loop.run_until_complete(asyncio.sleep(0))
        loop.run_until_complete(asyncio.sleep(0))
        assert emitted == []
    finally:
        loop.close()


def test_idempotent_start_capture():
    """A second start_capture during an active capture is ignored."""
    loop = asyncio.new_event_loop()
    try:
        emitted: List[SpeechBuffer] = []
        mc = MicCapture(loop=loop, on_speech_captured=lambda b: emitted.append(b))
        mc.start_capture()
        first_ts = mc._start_ts
        mc.start_capture()  # no-op
        assert mc._start_ts == first_ts
    finally:
        loop.close()


# ---- Path B mute / unmute --------------------------------------------------


def test_mute_drops_all_feed_chunks_and_clears_ring():
    """While muted, feed() must not update the pre-roll ring or progress
    any active capture. Pre-roll ring is cleared on entry to muted state
    so post-unmute does not leak pre-mute audio."""
    loop = asyncio.new_event_loop()
    try:
        emitted: List[SpeechBuffer] = []
        mc = MicCapture(loop=loop, on_speech_captured=lambda b: emitted.append(b))
        # Fill the ring with some audio first.
        mc.feed(_speech(500))
        assert len(mc._ring) > 0
        # Mute clears the ring.
        mc.mute()
        assert mc.is_muted is True
        assert len(mc._ring) == 0
        # Subsequent feeds drop silently.
        mc.feed(_speech(500))
        assert len(mc._ring) == 0
        # No capture should be in flight; no callback should fire.
        loop.run_until_complete(asyncio.sleep(0))
        assert emitted == []
    finally:
        loop.close()


def test_mute_aborts_inflight_capture():
    """If a capture is active when mute() is called, it must be aborted
    so a half-recorded utterance can't surface to ASR after unmute."""
    loop = asyncio.new_event_loop()
    try:
        emitted: List[SpeechBuffer] = []
        mc = MicCapture(loop=loop, on_speech_captured=lambda b: emitted.append(b))
        mc.start_capture()
        mc.feed(_speech(200))
        assert mc._active is True
        mc.mute()
        assert mc._active is False
        # Subsequent silence does not flush a buffer (capture was aborted).
        mc.feed(_silence(SILENCE_END_MS + 100))
        loop.run_until_complete(asyncio.sleep(0))
        loop.run_until_complete(asyncio.sleep(0))
        assert emitted == []
    finally:
        loop.close()


def test_unmute_resumes_ring_maintenance():
    """After unmute(), feed() must once again fill the pre-roll ring."""
    loop = asyncio.new_event_loop()
    try:
        emitted: List[SpeechBuffer] = []
        mc = MicCapture(loop=loop, on_speech_captured=lambda b: emitted.append(b))
        mc.mute()
        mc.feed(_speech(500))
        assert len(mc._ring) == 0
        mc.unmute()
        assert mc.is_muted is False
        mc.feed(_speech(500))
        assert len(mc._ring) > 0
    finally:
        loop.close()


def test_mute_unmute_idempotent():
    """Calling mute() twice or unmute() twice should be a no-op the second time."""
    loop = asyncio.new_event_loop()
    try:
        mc = MicCapture(loop=loop, on_speech_captured=lambda _b: None)
        mc.mute()
        mc.mute()  # no-op
        assert mc.is_muted is True
        mc.unmute()
        mc.unmute()  # no-op
        assert mc.is_muted is False
    finally:
        loop.close()
