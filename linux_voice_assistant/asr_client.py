"""Stage B3 — ASRClient: Wyoming-TCP submission to wyoming-whisper.

Replaces v0's HA-side Wyoming/ESPHome ASR streaming. Speaks Wyoming protocol
directly to the shared wyoming-whisper service (default tcp://minty-ai-300:10300).

API:
  client = ASRClient(uri="tcp://minty-ai-300:10300", language="en")
  result = await client.transcribe(wav_bytes)   # ASRResult(text, confidence, language)

C1–C8 mapped:
  C1 — Whisper model (medium-int8) is configured server-side; we just speak.
  C2 — Streaming events: Transcribe → AudioStart → AudioChunk+ → AudioStop → Transcript.
  C3 — Beam_size=1: server-side config (already medium-int8 beam=1).
  C4 — VAD inside Whisper: OFF (D3 owns VAD upstream); server-side config.
  C5 — Single shared whisper service; per-request `Transcribe.context` carries
       hotwords + language (Wyoming-Whisper passes context to faster-whisper's
       `initial_prompt`).
  C6 — asr_confidence: Wyoming doesn't expose per-token logprobs over the wire;
       we return a coarse hint (1.0 on success, 0.0 on empty). L.1 keeps the
       field optional and the bridge tolerates a missing value.
  C7 — Language pinned per call (default en).
  C8 — Hotwords: comma-separated, joined into `initial_prompt` via `context`.

Errors: connection failures and timeouts raise ASRError; the caller (DeviceSession)
treats this as a cancellation chain trigger (no bridge call made).
"""

from __future__ import annotations

import asyncio
import io
import logging
import wave
from dataclasses import dataclass
from typing import List, Optional

from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.client import AsyncClient

_LOGGER = logging.getLogger(__name__)

# Conservative per-request timeout. Whisper-medium-int8 on a CPU typically
# returns in <3 s for a 5 s utterance; 15 s is a safe upper bound that
# still leaves room inside the ≤4 s end-to-end goal when streaming overlaps.
_DEFAULT_TIMEOUT_S = 15.0

# Wyoming-Whisper expects 16 kHz, 16-bit, mono PCM chunks.
_CHUNK_SAMPLES = 1024  # 64 ms at 16 kHz — small enough to keep streaming smooth


@dataclass
class ASRResult:
    text: str
    confidence: float
    language: str = ""


class ASRError(Exception):
    """Raised when the Whisper service is unreachable or returns no transcript."""


class ASRClient:
    """Wyoming-TCP client for wyoming-whisper.

    Stateless: opens a fresh connection per transcribe() call. Whisper
    sessions are short-lived (~1 s + audio length) so a per-call connect
    is simpler than pooling and matches the Wyoming reference design.
    """

    def __init__(
        self,
        uri: str,
        *,
        language: str = "en",
        hotwords: Optional[List[str]] = None,
        timeout_s: float = _DEFAULT_TIMEOUT_S,
        model_name: Optional[str] = None,
    ) -> None:
        self.uri = uri
        self.language = language
        self.hotwords = list(hotwords or [])
        self.timeout_s = timeout_s
        self.model_name = model_name

    async def transcribe(self, wav_bytes: bytes) -> ASRResult:
        """Send a WAV-encoded utterance, await the Transcript event.

        Raises ASRError on connection failure, timeout, or empty transcript.
        """
        rate, width, channels, pcm = _unwrap_wav(wav_bytes)
        if rate != 16000 or width != 2 or channels != 1:
            raise ASRError(
                f"Unexpected WAV format rate={rate} width={width} channels={channels} "
                "— Wyoming-Whisper expects 16 kHz/16-bit/mono"
            )

        try:
            async with asyncio.timeout(self.timeout_s):
                async with AsyncClient.from_uri(self.uri) as client:
                    await self._stream(client, pcm)
                    transcript = await self._await_transcript(client)
        except asyncio.TimeoutError as exc:
            raise ASRError(f"ASR timeout after {self.timeout_s}s") from exc
        except ASRError:
            raise
        except Exception as exc:
            raise ASRError(f"ASR connection failed: {exc}") from exc

        if not transcript.text.strip():
            raise ASRError("ASR returned an empty transcript")

        return ASRResult(
            text=transcript.text.strip(),
            confidence=1.0,
            language=transcript.language or self.language,
        )

    async def _stream(self, client: AsyncClient, pcm: bytes) -> None:
        context: dict = {}
        if self.hotwords:
            # Wyoming-Whisper forwards `context` into faster-whisper's
            # `initial_prompt`; a space-joined hotword list is the canonical form.
            context["initial_prompt"] = " ".join(self.hotwords)
        await client.write_event(
            Transcribe(name=self.model_name, language=self.language, context=context or None).event()
        )
        await client.write_event(AudioStart(rate=16000, width=2, channels=1).event())

        chunk_bytes = _CHUNK_SAMPLES * 2  # 16-bit
        for start in range(0, len(pcm), chunk_bytes):
            data = pcm[start:start + chunk_bytes]
            await client.write_event(
                AudioChunk(rate=16000, width=2, channels=1, audio=data).event()
            )

        await client.write_event(AudioStop().event())

    async def _await_transcript(self, client: AsyncClient) -> Transcript:
        while True:
            event = await client.read_event()
            if event is None:
                raise ASRError("ASR stream closed before Transcript")
            if Transcript.is_type(event.type):
                return Transcript.from_event(event)
            # Streaming partials (transcript-start / transcript-chunk) ignored;
            # we wait for the final Transcript event.


def _unwrap_wav(wav_bytes: bytes) -> tuple[int, int, int, bytes]:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        rate = wf.getframerate()
        width = wf.getsampwidth()
        channels = wf.getnchannels()
        pcm = wf.readframes(wf.getnframes())
    return rate, width, channels, pcm
