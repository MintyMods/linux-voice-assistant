"""Stage B3 — bridge routing integration test.

Exercises DeviceSession._run_turn end-to-end with fake ASR / Bridge / TTS
collaborators:
  * Verifies the bridge receives `device=<room>` so per-room routing (J2)
    works for multi-Calisto deploys.
  * Verifies the SPEAKING → IDLE transition fires on TTS done.
  * Verifies a mid-turn cancel (gen bump) silently drops both the bridge
    reply and the TTS — §7.6 #2 and #3.
"""

from __future__ import annotations

import asyncio
import io
import wave
from typing import List, Optional

import pytest

from linux_voice_assistant.asr_client import ASRResult
from linux_voice_assistant.bridge_client import ChatReply
from linux_voice_assistant.mic_capture import SpeechBuffer
from linux_voice_assistant.session import DeviceSession, State


def _make_wav() -> bytes:
    out = io.BytesIO()
    with wave.open(out, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 1600)
    return out.getvalue()


class _FakeASR:
    def __init__(self, text: str = "what time is it") -> None:
        self.text = text
        self.calls = 0

    async def transcribe(self, wav_bytes: bytes) -> ASRResult:
        self.calls += 1
        return ASRResult(text=self.text, confidence=1.0, language="en")


class _FakeBridge:
    def __init__(self, reply: str = "two thirty PM", slow_s: float = 0.0,
                 continue_conversation: bool = False) -> None:
        self.reply = reply
        self.slow_s = slow_s
        self.continue_conversation = continue_conversation
        self.chat_calls: List[dict] = []
        self.cancel_calls: List[dict] = []

    async def chat(self, *, device: str, generation: int, session_id: str,
                   text: str, caller_area=None, user_id=None, asr_confidence=None):
        self.chat_calls.append({
            "device": device,
            "generation": generation,
            "session_id": session_id,
            "text": text,
            "asr_confidence": asr_confidence,
        })
        if self.slow_s:
            await asyncio.sleep(self.slow_s)
        return ChatReply(
            generation=generation,
            session_id=session_id,
            reply=self.reply,
            continue_conversation=self.continue_conversation,
            model="sonnet-4.6",
        )

    async def cancel(self, *, device: str, generation: int, reason=None):
        self.cancel_calls.append({"device": device, "generation": generation, "reason": reason})
        return None


class _FakeTTS:
    def __init__(self) -> None:
        self.speak_calls = 0
        self.fire_done_immediately = True

    async def speak(self, player, text: str, *, done_callback=None) -> None:
        self.speak_calls += 1
        if self.fire_done_immediately and done_callback is not None:
            done_callback()


def _make_state(room: str = "lounge"):
    from tests.conftest import make_server_state
    state = make_server_state()
    state.room = room
    return state


@pytest.mark.asyncio
async def test_run_turn_routes_to_bridge_with_correct_device():
    state = _make_state(room="bedroom")
    session = DeviceSession(state, ha_bridge=None)
    asr = _FakeASR()
    bridge = _FakeBridge()
    tts = _FakeTTS()
    state.asr_client = asr
    state.bridge_client = bridge
    state.tts_output = tts

    # Move into LISTENING (mints a session_id) before on_speech_captured.
    session.transition_to(State.LISTENING, reason="wake_chime_finished")
    buf = SpeechBuffer(wav_bytes=_make_wav(), start_ts=0.0, end_ts=0.5)

    captured_gen = session.generation
    await session._run_turn(captured_gen, buf)

    assert asr.calls == 1
    assert len(bridge.chat_calls) == 1
    assert bridge.chat_calls[0]["device"] == "bedroom"
    assert bridge.chat_calls[0]["text"] == "what time is it"
    assert tts.speak_calls == 1
    # TTS done_callback drove the SPEAKING → IDLE transition.
    assert session.state_value == State.IDLE


@pytest.mark.asyncio
async def test_run_turn_drops_reply_when_gen_moves_mid_chat():
    """§7.6 #3 — bridge slow + cancel mid-chat → reply silently dropped."""
    state = _make_state(room="lounge")
    session = DeviceSession(state, ha_bridge=None)
    asr = _FakeASR()
    bridge = _FakeBridge(slow_s=0.05)  # short slowness for test speed
    tts = _FakeTTS()
    state.asr_client = asr
    state.bridge_client = bridge
    state.tts_output = tts

    session.transition_to(State.LISTENING, reason="wake")
    buf = SpeechBuffer(wav_bytes=_make_wav(), start_ts=0.0, end_ts=0.5)
    captured_gen = session.generation

    task = asyncio.create_task(session._run_turn(captured_gen, buf))
    # Wait for ASR to finish + bridge to start, then bump gen (cancel).
    await asyncio.sleep(0.01)
    session.bump_gen()
    session.transition_to(State.IDLE, reason="cancel_force", cancel_reason="RED_BUTTON_SOFT")
    await task

    # ASR + bridge.chat both ran, but TTS never spoke (gen-check after bridge).
    assert asr.calls == 1
    assert len(bridge.chat_calls) == 1
    assert tts.speak_calls == 0


@pytest.mark.asyncio
async def test_run_turn_drops_when_gen_moves_after_asr():
    """Cancel during ASR → bridge never called."""
    state = _make_state(room="lounge")
    session = DeviceSession(state, ha_bridge=None)

    class _SlowASR:
        def __init__(self) -> None:
            self.calls = 0

        async def transcribe(self, wav_bytes: bytes) -> ASRResult:
            self.calls += 1
            await asyncio.sleep(0.05)
            return ASRResult(text="hi", confidence=1.0, language="en")

    asr = _SlowASR()
    bridge = _FakeBridge()
    tts = _FakeTTS()
    state.asr_client = asr
    state.bridge_client = bridge
    state.tts_output = tts

    session.transition_to(State.LISTENING, reason="wake")
    buf = SpeechBuffer(wav_bytes=_make_wav(), start_ts=0.0, end_ts=0.5)
    captured_gen = session.generation

    task = asyncio.create_task(session._run_turn(captured_gen, buf))
    await asyncio.sleep(0.01)
    session.bump_gen()
    session.transition_to(State.IDLE, reason="cancel_force", cancel_reason="RED_BUTTON_SOFT")
    await task

    assert asr.calls == 1
    assert len(bridge.chat_calls) == 0
    assert tts.speak_calls == 0


@pytest.mark.asyncio
async def test_run_turn_handles_empty_bridge_reply():
    state = _make_state()
    session = DeviceSession(state, ha_bridge=None)
    state.asr_client = _FakeASR()
    state.bridge_client = _FakeBridge(reply="")
    state.tts_output = _FakeTTS()

    session.transition_to(State.LISTENING, reason="wake")
    buf = SpeechBuffer(wav_bytes=_make_wav(), start_ts=0.0, end_ts=0.5)
    await session._run_turn(session.generation, buf)

    # Empty reply → straight to IDLE, no TTS.
    assert session.state_value == State.IDLE
    assert state.tts_output.speak_calls == 0
