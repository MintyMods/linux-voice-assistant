"""Stage B3 — TTSOutput: minimal Wyoming-Piper TCP client → WAV → mpv.

Renders bridge reply text via wyoming-piper, writes the result to a tempfile,
and plays it through the existing `state.tts_player` (MpvMediaPlayer). This
is the *minimal* B3 path; Stage E does the proper stdin-stream split (E2)
with first-audio in ~150 ms.

API:
  out = TTSOutput(uri="tcp://minty-ai-300:10200", voice="en_GB-alba-medium")
  await out.speak(state.tts_player, text="Hello.", done_callback=cb)

`done_callback` is invoked on the asyncio loop when mpv finishes playing.
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import tempfile
import wave
from typing import TYPE_CHECKING, Callable, Optional

from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.client import AsyncClient
from wyoming.tts import Synthesize, SynthesizeVoice

if TYPE_CHECKING:
    from .mpv_player import MpvMediaPlayer

_LOGGER = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_S = 15.0


class TTSError(Exception):
    """Raised when Piper is unreachable or returns no audio."""


class TTSOutput:
    """Wyoming-Piper TCP client. Renders to WAV; plays via mpv.

    Stateless: opens a fresh connection per speak() call. Piper sessions
    are short (<1 s typical).
    """

    def __init__(
        self,
        uri: str,
        *,
        voice: Optional[str] = None,
        timeout_s: float = _DEFAULT_TIMEOUT_S,
    ) -> None:
        self.uri = uri
        self.voice = voice
        self.timeout_s = timeout_s
        self._last_temp: Optional[str] = None

    async def speak(
        self,
        player: "MpvMediaPlayer",
        text: str,
        *,
        done_callback: Optional[Callable[[], None]] = None,
    ) -> None:
        """Synthesize text via Piper, write to temp WAV, queue mpv playback.

        Returns once playback has been *queued* (mpv runs in its own thread);
        `done_callback` fires when mpv signals end-of-playback. Raises TTSError
        on Piper-side failure (no audio is queued).
        """
        if not text.strip():
            _LOGGER.debug("TTSOutput.speak: empty text; firing done callback immediately")
            if done_callback is not None:
                done_callback()
            return

        try:
            async with asyncio.timeout(self.timeout_s):
                wav_bytes = await self._synthesize(text)
        except asyncio.TimeoutError as exc:
            raise TTSError(f"Piper timeout after {self.timeout_s}s") from exc
        except TTSError:
            raise
        except Exception as exc:
            raise TTSError(f"Piper connection failed: {exc}") from exc

        # Persist to a tempfile that survives until the next speak() call,
        # then hand the path to mpv. mpv reads the file lazily — we must
        # NOT delete it before playback completes.
        fd, path = tempfile.mkstemp(prefix="lva_tts_", suffix=".wav")
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(wav_bytes)
        except Exception:
            os.close(fd) if not fh else None  # type: ignore[possibly-undefined]
            raise

        # Stage E refines this — for B3 we keep one previous file around and
        # rotate. We delete the *previous* temp on each new speak(), so the
        # currently-playing one stays on disk.
        prev = self._last_temp
        self._last_temp = path

        loop = asyncio.get_running_loop()

        def _on_finished() -> None:
            # mpv invokes this on its own thread; bounce to the asyncio loop
            # before firing the caller's done_callback, and clean up the
            # previous tempfile (current one stays for any restart-of-play).
            try:
                if prev and os.path.exists(prev):
                    os.unlink(prev)
            except OSError:
                pass
            if done_callback is not None:
                try:
                    loop.call_soon_threadsafe(done_callback)
                except RuntimeError:
                    pass

        player.play(path, done_callback=_on_finished)

    async def _synthesize(self, text: str) -> bytes:
        voice = SynthesizeVoice(name=self.voice) if self.voice else None
        async with AsyncClient.from_uri(self.uri) as client:
            await client.write_event(Synthesize(text=text, voice=voice).event())

            # Collect AudioStart/AudioChunk+/AudioStop into a single WAV.
            rate: Optional[int] = None
            width: Optional[int] = None
            channels: Optional[int] = None
            pcm = bytearray()
            while True:
                event = await client.read_event()
                if event is None:
                    break
                if AudioStart.is_type(event.type):
                    start = AudioStart.from_event(event)
                    rate, width, channels = start.rate, start.width, start.channels
                    continue
                if AudioChunk.is_type(event.type):
                    chunk = AudioChunk.from_event(event)
                    pcm.extend(chunk.audio)
                    if rate is None:
                        rate, width, channels = chunk.rate, chunk.width, chunk.channels
                    continue
                if AudioStop.is_type(event.type):
                    break
                # Ignore synthesize-start / synthesize-chunk / synthesize-stop /
                # synthesize-stopped streaming events; we collect a single WAV.

        if not pcm or rate is None or width is None or channels is None:
            raise TTSError("Piper returned no audio")

        return _pcm_to_wav(bytes(pcm), rate=rate, width=width, channels=channels)


def _pcm_to_wav(pcm: bytes, *, rate: int, width: int, channels: int) -> bytes:
    out = io.BytesIO()
    with wave.open(out, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(width)
        wf.setframerate(rate)
        wf.writeframes(pcm)
    return out.getvalue()
