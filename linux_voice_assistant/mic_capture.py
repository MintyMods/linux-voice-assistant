"""Stage B3 — MicCapture: local audio capture, VAD-gated, ASR-ready buffer.

Replaces the v0 HA-side STT streaming path. The `process_audio` thread feeds
each 16 kHz int16 PCM chunk into MicCapture.feed(); MicCapture maintains:

  * A 3 s rolling pre-roll ring buffer (D6 — feeds the 200 ms that precedes
    wake-fire into the speech buffer once start_capture is called).
  * On start_capture(): emits 200 ms of pre-roll (post wake-token strip, D6),
    then runs TEN-VAD (D3) hop-by-hop over the incoming stream.
  * End-of-speech: 400 ms of continuous TEN-VAD-silent frames (D4).
  * Hard cap: 30 s total capture (D5).
  * Continuous-capture trim: keep all VAD-active frames, trim leading +
    trailing silence to ~200 ms each at the edges (D8).
  * Optional denoise hook (D7) — passthrough by default; DeepFilterNet 3
    plumbed in Stage F once a Rust-built wheel is available on the host.

On end-of-speech (or 30 s cap), schedules `DeviceSession.on_speech_captured`
on the asyncio loop with a SpeechBuffer(wav_bytes, start_ts, end_ts, asr_hint).

Threading: feed() is called from the sync `process_audio` thread. Hand-off
to asyncio uses `loop.call_soon_threadsafe`. State is protected by a lock.
"""

from __future__ import annotations

import asyncio
import io
import logging
import threading
import time
import wave
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional

import numpy as np

if TYPE_CHECKING:
    from ten_vad import TenVad

_LOGGER = logging.getLogger(__name__)

# Audio constants — fixed by the upstream pipeline (process_audio uses 16 kHz).
SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2  # bytes (int16)
CHANNELS = 1

# TEN-VAD operates on 256-sample hops = 16 ms at 16 kHz.
VAD_HOP = 256

# D6 — 200 ms pre-roll prepended to the capture buffer once VAD starts.
PRE_ROLL_MS = 200
# D6 — wake-word audible portion stripped: ~250 ms covers a short wake word
# (e.g. "Alexa", "Calisto"). Tunable in Stage F via M.5.
WAKE_TOKEN_STRIP_MS = 250

# D4 — silence-end threshold (only counted AFTER first speech detected).
SILENCE_END_MS = 400
# D5 — hard cap on capture duration.
MAX_CAPTURE_MS = 30_000
# Max wait between start_capture and first VAD-detected speech. Without
# this, an empty wake (user wakes but says nothing) would run for 30 s.
# Tuned to "if the user hasn't started talking within 2 s of the chime, abort".
NO_SPEECH_TIMEOUT_MS = 2_000
# D8 — leave ~200 ms of leading/trailing silence either side of speech.
EDGE_KEEP_MS = 200

# Pre-roll ring buffer length — 3 s covers any wake delay + the 200 ms
# pre-roll we emit; D6.
PRE_ROLL_RING_MS = 3_000


def _ms_to_samples(ms: int) -> int:
    return (SAMPLE_RATE * ms) // 1000


def _ms_to_bytes(ms: int) -> int:
    return _ms_to_samples(ms) * SAMPLE_WIDTH * CHANNELS


@dataclass
class SpeechBuffer:
    """Captured speech ready for ASR submission."""

    wav_bytes: bytes
    """WAV-format bytes (16 kHz mono int16, full RIFF header)."""

    start_ts: float
    """monotonic() at capture start (pre-roll begin)."""

    end_ts: float
    """monotonic() at capture end (silence-end or 30 s cap)."""

    asr_confidence_hint: float = 0.0
    """Mean TEN-VAD speech-probability across captured frames; informational
    only — Whisper produces the real confidence at L.1 submit time (C6)."""

    end_reason: str = "silence"
    """One of: silence, cap, abort."""


class MicCapture:
    """Synchronous mic-side capture, async-deliverable buffer.

    `feed(chunk)` runs on the `process_audio` thread; `start_capture()` is
    safe to call from either thread. Hand-off to the asyncio loop uses
    `loop.call_soon_threadsafe`.

    Denoising (D7) is currently a passthrough — the upstream `process_audio`
    already cleans audio via WebRTC NS. Hook is kept so Stage F can drop in
    DeepFilterNet 3 with no caller changes.
    """

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        on_speech_captured: Callable[[SpeechBuffer], None],
        *,
        vad_threshold: float = 0.5,
        silence_end_ms: int = SILENCE_END_MS,
        max_capture_ms: int = MAX_CAPTURE_MS,
        no_speech_timeout_ms: int = NO_SPEECH_TIMEOUT_MS,
        pre_roll_ms: int = PRE_ROLL_MS,
        wake_token_strip_ms: int = WAKE_TOKEN_STRIP_MS,
        denoise: Optional[Callable[[bytes], bytes]] = None,
        require_vad: bool = True,
    ) -> None:
        self._loop = loop
        self._cb = on_speech_captured
        self._silence_end_ms = silence_end_ms
        self._max_capture_ms = max_capture_ms
        self._no_speech_timeout_ms = no_speech_timeout_ms
        self._pre_roll_ms = pre_roll_ms
        self._wake_token_strip_ms = wake_token_strip_ms
        self._denoise = denoise

        self._vad_threshold = vad_threshold
        self._vad: "Optional[TenVad]" = None
        self._vad_init_failed = False

        # Ring buffer (3 s) and active-capture state.
        self._ring_max = _ms_to_bytes(PRE_ROLL_RING_MS)
        self._ring = bytearray()
        # vad_carry buffers partial hops that don't fit a 256-sample frame.
        self._vad_carry = bytearray()

        self._lock = threading.Lock()
        self._active = False
        self._aborted = False
        self._capture_buf = bytearray()
        self._start_ts: float = 0.0
        self._first_speech_ts: float = 0.0
        self._speech_seen: bool = False
        self._silent_run_ms = 0.0
        self._vad_prob_sum = 0.0
        self._vad_prob_count = 0

        # Fail loud on construction if TEN-VAD can't load. With require_vad=False
        # callers opt into the silence-only degraded path (tests use this).
        self._ensure_vad()
        if require_vad and self._vad is None:
            raise RuntimeError(
                "TEN-VAD failed to load — MicCapture refuses to wire the v1 audio path. "
                "Install `ten-vad` (PyPI wheel ships the native lib for Linux x86_64). "
                "Pass require_vad=False to bypass for tests."
            )

    # ---- VAD initialisation ----------------------------------------------

    def _ensure_vad(self) -> "Optional[TenVad]":
        if self._vad is not None or self._vad_init_failed:
            return self._vad
        try:
            from ten_vad import TenVad  # type: ignore
            self._vad = TenVad(hop_size=VAD_HOP, threshold=self._vad_threshold)
            _LOGGER.info("TEN-VAD initialised (hop=%d, threshold=%.2f)", VAD_HOP, self._vad_threshold)
        except Exception:
            self._vad_init_failed = True
            _LOGGER.exception("TEN-VAD init failed")
        return self._vad

    # ---- feed (sync; process_audio thread) -------------------------------

    def feed(self, chunk: bytes) -> None:
        """Push a 16 kHz int16 PCM chunk. Always maintains the pre-roll ring.

        If a capture is active, also routes the chunk through VAD and the
        speech buffer, with end-of-speech / 30 s cap detection.
        """
        if not chunk:
            return
        with self._lock:
            # Always maintain the pre-roll ring.
            self._ring.extend(chunk)
            overflow = len(self._ring) - self._ring_max
            if overflow > 0:
                del self._ring[:overflow]

            if not self._active:
                return

            # 30 s cap.
            elapsed_ms = (time.monotonic() - self._start_ts) * 1000.0
            if elapsed_ms >= self._max_capture_ms:
                self._finish_locked(end_reason="cap")
                return

            self._capture_buf.extend(chunk)
            # Run TEN-VAD hop-by-hop. Two phases:
            #   1. PRE-SPEECH: silence_run_ms is NOT counted (user hasn't
            #      started talking yet). If we hit `no_speech_timeout_ms`
            #      since start_capture without ever seeing speech → abort
            #      this capture with end_reason="no_speech".
            #   2. POST-SPEECH (after first speech hop): each silent hop ticks
            #      silence_run_ms; SILENCE_END_MS continuous silence ends it.
            # End-of-speech triggers on the first hop that crosses the
            # threshold — we stop processing further hops in this chunk to
            # avoid clipping into the next utterance.
            hop_ms = (VAD_HOP / SAMPLE_RATE) * 1000.0
            for is_speech, prob in self._iter_vad_hops(chunk):
                if prob > 0:
                    self._vad_prob_sum += prob
                    self._vad_prob_count += 1
                if is_speech:
                    if not self._speech_seen:
                        self._speech_seen = True
                        self._first_speech_ts = time.monotonic()
                    self._silent_run_ms = 0.0
                elif self._speech_seen:
                    self._silent_run_ms += hop_ms
                if self._speech_seen and self._silent_run_ms >= self._silence_end_ms:
                    self._finish_locked(end_reason="silence")
                    return
            # No-speech timeout: compute elapsed from captured bytes (so
            # synthetic tests with one big silence chunk fire just like
            # real-time mic chunks of a sleeping user).
            if not self._speech_seen:
                buffered_ms = (len(self._capture_buf) / SAMPLE_WIDTH / SAMPLE_RATE) * 1000.0
                if buffered_ms >= self._no_speech_timeout_ms:
                    self._finish_locked(end_reason="no_speech")
                    return

    def _iter_vad_hops(self, chunk: bytes):
        """Yield (is_speech, prob) per VAD_HOP-sized hop in `chunk` (plus carry).

        If TEN-VAD is unavailable, every hop yields (False, 0.0) so the
        silence timer can still drive end-of-speech.
        """
        vad = self._ensure_vad()
        self._vad_carry.extend(chunk)
        hop_bytes = VAD_HOP * SAMPLE_WIDTH
        offset = 0
        try:
            while len(self._vad_carry) - offset >= hop_bytes:
                frame = bytes(self._vad_carry[offset:offset + hop_bytes])
                offset += hop_bytes
                if vad is None:
                    yield False, 0.0
                    continue
                try:
                    samples = np.frombuffer(frame, dtype=np.int16)
                    prob, flag = vad.process(samples)
                except Exception:
                    _LOGGER.exception("TEN-VAD process failed; treating frame as silence")
                    yield False, 0.0
                    continue
                yield bool(flag), float(prob)
        finally:
            if offset:
                del self._vad_carry[:offset]

    # ---- capture lifecycle (callable from any thread) --------------------

    def start_capture(self) -> None:
        """Begin a new capture. Emits 200 ms pre-roll (post wake-strip)."""
        with self._lock:
            if self._active:
                _LOGGER.debug("start_capture called while already active; ignoring")
                return
            self._active = True
            self._aborted = False
            self._capture_buf = bytearray()
            self._vad_carry = bytearray()
            self._silent_run_ms = 0.0
            self._vad_prob_sum = 0.0
            self._vad_prob_count = 0
            self._speech_seen = False
            self._first_speech_ts = 0.0
            self._start_ts = time.monotonic()

            # D6 — emit the trailing 200 ms of the pre-roll ring AFTER
            # stripping the most-recent ~250 ms (the wake word itself).
            strip_bytes = _ms_to_bytes(self._wake_token_strip_ms)
            preroll_bytes = _ms_to_bytes(self._pre_roll_ms)
            ring_len = len(self._ring)
            if ring_len > strip_bytes:
                upper = ring_len - strip_bytes
                lower = max(0, upper - preroll_bytes)
                pre = bytes(self._ring[lower:upper])
            else:
                pre = b""
            self._capture_buf.extend(pre)

    def abort(self) -> None:
        """Drop the in-flight capture silently. Used on cancel (gen bump)."""
        with self._lock:
            if not self._active:
                return
            self._active = False
            self._aborted = True
            self._capture_buf = bytearray()
            self._vad_carry = bytearray()

    # ---- internal: finish + dispatch -------------------------------------

    def _finish_locked(self, *, end_reason: str) -> None:
        """Hold _lock when calling. Emits the captured buffer to the loop."""
        self._active = False
        raw = bytes(self._capture_buf)
        self._capture_buf = bytearray()
        self._vad_carry = bytearray()
        end_ts = time.monotonic()
        mean_prob = (self._vad_prob_sum / self._vad_prob_count) if self._vad_prob_count else 0.0

        # D8 — trim leading + trailing silence so Whisper sees a tight buffer.
        trimmed = _trim_edges(raw, EDGE_KEEP_MS)
        if self._denoise is not None:
            try:
                trimmed = self._denoise(trimmed)
            except Exception:
                _LOGGER.exception("Denoiser raised; falling back to raw audio")

        wav_bytes = _pcm_to_wav(trimmed)

        buf = SpeechBuffer(
            wav_bytes=wav_bytes,
            start_ts=self._start_ts,
            end_ts=end_ts,
            asr_confidence_hint=mean_prob,
            end_reason=end_reason,
        )
        _LOGGER.info(
            "Capture finished: end_reason=%s, raw=%d bytes, trimmed=%d bytes, dur=%.2fs, mean_vad=%.2f",
            end_reason,
            len(raw),
            len(trimmed),
            (end_ts - self._start_ts),
            mean_prob,
        )

        # Hand off to the asyncio loop. The DeviceSession callback is gen-checked.
        try:
            self._loop.call_soon_threadsafe(self._cb, buf)
        except RuntimeError:
            # Loop is shutting down. Drop silently — gen will be invalid anyway.
            _LOGGER.debug("Event loop closed during capture dispatch; dropping buffer")


def _pcm_to_wav(pcm: bytes) -> bytes:
    """Wrap raw 16-bit mono 16 kHz PCM in a WAV container."""
    out = io.BytesIO()
    with wave.open(out, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm)
    return out.getvalue()


def _trim_edges(pcm: bytes, edge_keep_ms: int) -> bytes:
    """D8 — find first/last non-silent sample, keep `edge_keep_ms` either side.

    Cheap silence detector: amplitude threshold. The real VAD already did
    its job; this is just a final cosmetic trim so Whisper doesn't waste
    cycles on leading/trailing silence that slipped past the 200 ms
    silence run-up.
    """
    if not pcm:
        return pcm
    samples = np.frombuffer(pcm, dtype=np.int16)
    # Threshold scales with signal — use a fraction of peak abs.
    peak = int(np.max(np.abs(samples))) if samples.size else 0
    if peak == 0:
        return pcm
    threshold = max(200, peak // 50)  # ~1/50 of peak, floor 200/32768
    mask = np.abs(samples) > threshold
    if not mask.any():
        return pcm
    first_idx = int(np.argmax(mask))
    last_idx = int(samples.size - 1 - np.argmax(mask[::-1]))
    keep_samples = _ms_to_samples(edge_keep_ms)
    start = max(0, first_idx - keep_samples)
    end = min(samples.size, last_idx + 1 + keep_samples)
    trimmed = samples[start:end].tobytes()
    return trimmed
