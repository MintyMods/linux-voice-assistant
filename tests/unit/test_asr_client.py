"""Stage B3 — ASRClient unit tests against a fake Wyoming TCP server.

Spins up an in-process asyncio TCP server that speaks Wyoming protocol with
just enough fidelity to round-trip a Transcribe → AudioStart → AudioChunk+
→ AudioStop sequence and reply with a Transcript. Verifies:
  * Language + hotwords reach the server via the Transcribe event.
  * WAV unwrap rejects non-16k/16-bit/mono inputs.
  * Empty Transcript → ASRError.
  * Server EOF before Transcript → ASRError.
"""

from __future__ import annotations

import asyncio
import io
import wave

import pytest
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import async_read_event, async_write_event

from linux_voice_assistant.asr_client import ASRClient, ASRError


def _make_wav(rate: int = 16000, width: int = 2, channels: int = 1, duration_ms: int = 200) -> bytes:
    n = (rate * duration_ms) // 1000
    pcm = bytes(n * width * channels)
    out = io.BytesIO()
    with wave.open(out, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(width)
        wf.setframerate(rate)
        wf.writeframes(pcm)
    return out.getvalue()


class _FakeWhisper:
    """In-process Wyoming server. Records the Transcribe event it sees,
    consumes audio chunks, then replies with `text` and EOF.
    """

    def __init__(self, text: str = "what time is it", language: str = "en") -> None:
        self.text = text
        self.language = language
        self.transcribe_seen: Transcribe | None = None
        self.audio_bytes_seen = 0
        self._server: asyncio.AbstractServer | None = None
        self.host = "127.0.0.1"
        self.port = 0
        self.eof_before_transcript = False

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

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            while True:
                event = await async_read_event(reader)
                if event is None:
                    return
                if Transcribe.is_type(event.type):
                    self.transcribe_seen = Transcribe.from_event(event)
                elif AudioStart.is_type(event.type):
                    pass
                elif AudioChunk.is_type(event.type):
                    chunk = AudioChunk.from_event(event)
                    self.audio_bytes_seen += len(chunk.audio)
                elif AudioStop.is_type(event.type):
                    if self.eof_before_transcript:
                        return
                    await async_write_event(
                        Transcript(text=self.text, language=self.language).event(), writer
                    )
                    return
        except (ConnectionResetError, asyncio.IncompleteReadError):
            return
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass


@pytest.mark.asyncio
async def test_transcribe_round_trip_with_hotwords():
    server = _FakeWhisper(text="what time is it", language="en")
    await server.start()
    try:
        client = ASRClient(server.uri, language="en", hotwords=["Calisto", "Alexa"])
        result = await client.transcribe(_make_wav())
    finally:
        await server.stop()

    assert result.text == "what time is it"
    assert result.language == "en"
    assert server.transcribe_seen is not None
    assert server.transcribe_seen.language == "en"
    assert server.transcribe_seen.context is not None
    assert "Calisto" in server.transcribe_seen.context["initial_prompt"]
    assert server.audio_bytes_seen > 0


@pytest.mark.asyncio
async def test_empty_transcript_raises_asr_error():
    server = _FakeWhisper(text="   ")
    await server.start()
    try:
        client = ASRClient(server.uri, language="en")
        with pytest.raises(ASRError):
            await client.transcribe(_make_wav())
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_eof_before_transcript_raises_asr_error():
    server = _FakeWhisper()
    server.eof_before_transcript = True
    await server.start()
    try:
        client = ASRClient(server.uri, language="en")
        with pytest.raises(ASRError):
            await client.transcribe(_make_wav())
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_unexpected_wav_format_rejected():
    client = ASRClient("tcp://127.0.0.1:1", language="en")  # never connects
    with pytest.raises(ASRError):
        await client.transcribe(_make_wav(rate=8000))
