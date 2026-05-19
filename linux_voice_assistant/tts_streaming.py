"""Stage E.2 E2 / E3 — Wyoming streaming TTS via dedicated mpv subprocess.

The Stage B3 `TTSOutput` (tts_output.py) is the minimal path: collect the
full WAV, write to tempfile, hand to `tts_player` (libmpv via Python
binding). That keeps the one-mpv-per-player invariant of the v0 fork but
pays the round-trip latency cost — first audio not heard until Piper has
finished synthesising the entire sentence.

`TTSStreamingOutput` is the v1 target per architecture-v1-decisions §E2/E3:

  Piper (Wyoming TCP)  →  PCM AudioChunks  →  mpv stdin (rawaudio demuxer)
                          ───────────────       ───────────────────────────
                          ~150 ms latency       persistent mpv subprocess

Why a separate mpv subprocess rather than the existing libmpv binding:
the binding controls a single MPV instance per `LibMpvPlayer`, and that
instance is already configured for file/URL playback. Reconfiguring it
for `--demuxer=rawaudio` would break short-clip playback on the same
channel. The cleanest split per Section O.2 is a dedicated mpv #1 for
TTS streaming, separate from the `tts_player` (which becomes the chime
channel in spirit).

PROTOTYPE STATUS (2026-05-19): the wire shape is implemented and unit-
tested but NOT verified on hardware. Enable via `TTS_STREAMING=1`; the
default deploy keeps using the existing `TTSOutput` tempfile path.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
import time
from typing import Callable, Optional

from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.client import AsyncClient
from wyoming.tts import Synthesize, SynthesizeVoice

_LOGGER = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_S = 15.0
_DEFAULT_RATE = 22050  # Piper default for medium-quality voices
_DEFAULT_CHANNELS = 1
_DEFAULT_WIDTH = 2  # s16le


class TTSStreamingError(Exception):
    """Raised when Piper is unreachable, mpv is missing, or the stream dies."""


class TTSStreamingOutput:
    """Streams Wyoming-Piper PCM to a dedicated mpv subprocess via stdin.

    API matches the minimal `TTSOutput`:

        out = TTSStreamingOutput(uri, voice=..., audio_device=...)
        await out.speak(player_unused, text="...", done_callback=cb)

    The `player_unused` arg is the historical `MpvMediaPlayer` reference;
    it's ignored here since this class owns its own mpv. Kept on the
    signature for drop-in compat with `TTSOutput`.
    """

    def __init__(
        self,
        uri: str,
        *,
        voice: Optional[str] = None,
        audio_device: Optional[str] = None,
        sample_rate: int = _DEFAULT_RATE,
        channels: int = _DEFAULT_CHANNELS,
        sample_width: int = _DEFAULT_WIDTH,
        timeout_s: float = _DEFAULT_TIMEOUT_S,
        mpv_path: Optional[str] = None,
    ) -> None:
        self.uri = uri
        self.voice = voice
        self.audio_device = audio_device
        self.sample_rate = sample_rate
        self.channels = channels
        self.sample_width = sample_width
        self.timeout_s = timeout_s
        self.mpv_path = mpv_path or shutil.which("mpv") or "mpv"

        # Persistent mpv subprocess: launched lazily on first speak() and
        # kept alive across calls. mpv on stdin in rawaudio mode treats
        # EOF as end-of-track; we re-spawn on subprocess death.
        self._mpv_proc: Optional[subprocess.Popen] = None
        self._mpv_lock = asyncio.Lock()

    async def speak(
        self,
        _player_unused,
        text: str,
        *,
        done_callback: Optional[Callable[[], None]] = None,
    ) -> None:
        """Synthesize `text` via Piper, stream PCM to mpv stdin.

        Returns once the synthesis stream is closed (AudioStop received).
        `done_callback` fires on the asyncio loop after mpv has drained
        the pipe (best-effort — we don't have a true playback-finished
        signal without IPC, so we use a "stream closed + small drain
        sleep" heuristic). The drain delay is hold-the-line for a
        future IPC-based completion event.
        """
        if not text.strip():
            _LOGGER.debug("TTSStreamingOutput.speak: empty text; firing done callback")
            if done_callback is not None:
                done_callback()
            return

        # Spawn or revive mpv subprocess (idempotent).
        try:
            await self._ensure_mpv()
        except TTSStreamingError:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            raise TTSStreamingError(f"mpv spawn failed: {exc}") from exc

        loop = asyncio.get_running_loop()
        start_ts = time.monotonic()
        first_chunk_ts: Optional[float] = None

        async def _stream_synthesis() -> None:
            nonlocal first_chunk_ts
            voice = SynthesizeVoice(name=self.voice) if self.voice else None
            async with AsyncClient.from_uri(self.uri) as client:
                await client.write_event(Synthesize(text=text, voice=voice).event())
                while True:
                    event = await client.read_event()
                    if event is None:
                        break
                    if AudioStart.is_type(event.type):
                        # Piper announces rate/width/channels here. If the
                        # mpv we spawned was configured for different
                        # values, log it — the heard pitch will be off.
                        start = AudioStart.from_event(event)
                        if (start.rate, start.width, start.channels) != (
                            self.sample_rate, self.sample_width, self.channels
                        ):
                            _LOGGER.warning(
                                "Piper stream params %s differ from mpv config %s — pitch may be wrong",
                                (start.rate, start.width, start.channels),
                                (self.sample_rate, self.sample_width, self.channels),
                            )
                        continue
                    if AudioChunk.is_type(event.type):
                        chunk = AudioChunk.from_event(event)
                        if first_chunk_ts is None:
                            first_chunk_ts = time.monotonic()
                            _LOGGER.info(
                                "TTS first audio chunk: %.0f ms after speak() start",
                                (first_chunk_ts - start_ts) * 1000.0,
                            )
                        self._write_chunk(chunk.audio)
                        continue
                    if AudioStop.is_type(event.type):
                        break

        try:
            async with asyncio.timeout(self.timeout_s):
                await _stream_synthesis()
        except asyncio.TimeoutError as exc:
            raise TTSStreamingError(f"Piper streaming timeout after {self.timeout_s}s") from exc
        except TTSStreamingError:
            raise
        except Exception as exc:
            raise TTSStreamingError(f"Piper streaming failed: {exc}") from exc

        # Schedule the done callback after a small drain delay.
        # mpv's audio-buffer is 0.8s by default in our config; rawaudio mode
        # is similar. 600ms is a safe heuristic; replace with an IPC-based
        # `playback-restart` event listener for a tighter loop.
        if done_callback is not None:
            def _fire() -> None:
                try:
                    done_callback()
                except Exception:
                    _LOGGER.exception("TTSStreamingOutput done_callback raised")

            loop.call_later(0.6, _fire)

    def close(self) -> None:
        """Tear down the persistent mpv subprocess. Idempotent."""
        proc = self._mpv_proc
        if proc is None:
            return
        try:
            if proc.stdin is not None:
                proc.stdin.close()
            proc.terminate()
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                proc.kill()
        except Exception:
            _LOGGER.exception("TTSStreamingOutput.close: subprocess teardown raised")
        self._mpv_proc = None

    # -- internals ---------------------------------------------------------

    async def _ensure_mpv(self) -> None:
        """Spawn or revive the persistent mpv subprocess (under a lock)."""
        async with self._mpv_lock:
            proc = self._mpv_proc
            if proc is not None and proc.poll() is None:
                return
            # Stale or never-spawned. Build fresh.
            if proc is not None:
                _LOGGER.warning(
                    "TTS mpv subprocess died (returncode=%s); respawning",
                    proc.returncode,
                )
            self._mpv_proc = self._spawn_mpv()

    def _spawn_mpv(self) -> subprocess.Popen:
        args = [
            self.mpv_path,
            "--no-config",
            "--no-terminal",
            "--keep-open=no",
            "--idle=no",
            "--audio-buffer=0.2",  # smaller buffer than the file mpv (lower first-audio latency)
            "--audio-stream-silence=yes",
            f"--demuxer-rawaudio-rate={self.sample_rate}",
            f"--demuxer-rawaudio-channels={self.channels}",
            f"--demuxer-rawaudio-format={'s16le' if self.sample_width == 2 else 'u8'}",
            "--demuxer=rawaudio",
            "-",  # read from stdin
        ]
        if self.audio_device:
            args.insert(2, f"--audio-device={self.audio_device}")
        _LOGGER.debug("Spawning mpv for TTS streaming: %s", " ".join(args))
        try:
            return subprocess.Popen(
                args,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
        except FileNotFoundError as exc:
            raise TTSStreamingError(
                f"mpv binary not found at {self.mpv_path!r}; install mpv or set MPV_PATH"
            ) from exc

    def _write_chunk(self, audio: bytes) -> None:
        proc = self._mpv_proc
        if proc is None or proc.stdin is None:
            _LOGGER.warning("TTSStreamingOutput._write_chunk: no mpv stdin")
            return
        try:
            proc.stdin.write(audio)
            proc.stdin.flush()
        except (BrokenPipeError, ValueError):
            _LOGGER.warning("TTS mpv stdin closed mid-stream; chunk dropped")
            # Mark proc as dead so the next speak() respawns.
            try:
                proc.terminate()
            except Exception:
                pass
            self._mpv_proc = None
