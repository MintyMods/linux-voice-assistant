"""Stage B — DeviceSession: §6 state machine + generation counter + K.1 publish hook.

DeviceSession is the lifecycle layer above VoiceSatelliteProtocol.
- It owns the generation counter (sourced here, mirrored back to the satellite).
- It owns the §6 state machine (10 states per K.1).
- Every state transition flows through `transition_to(...)`, which optionally
  publishes K.1 via HABridge (publish-on-transition).
- It mints a session_id (UUIDv4) on entry to WAKING and clears it on entry to IDLE.

Stage B scope is *publish-on-transition only*. The 30s slow re-assert,
drift detection, and startup state assertion all land in Stage F (see
v1-spec-O-lva-integration.md §O.2 and v1-progress.md Stage F).
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import uuid
from enum import Enum
from typing import TYPE_CHECKING, Optional, Union

if TYPE_CHECKING:
    from .ha_bridge import HABridge
    from .mic_capture import SpeechBuffer
    from .models import ServerState

_LOGGER = logging.getLogger(__name__)


class State(str, Enum):
    """K.1 canonical state enum. Values are wire-format strings."""

    STARTING = "STARTING"
    IDLE = "IDLE"
    WAKING = "WAKING"
    LISTENING = "LISTENING"
    THINKING = "THINKING"
    SPEAKING = "SPEAKING"
    FOLLOWUP = "FOLLOWUP"
    CANCELLING = "CANCELLING"
    OFFLINE = "OFFLINE"
    DEGRADED = "DEGRADED"


def _coerce_state(value: Union[State, str]) -> State:
    if isinstance(value, State):
        return value
    return State(str(value))


class DeviceSession:
    """Owns the §6 state machine, generation counter and session_id lifecycle.

    For Stage B the satellite still drives audio. DeviceSession is a parallel
    state holder: every `_set_state_label` / cancel in satellite.py forwards
    here, which (a) mirrors back to satellite for back-compat reads, and (b)
    publishes K.1 via HABridge when one is attached.

    session_id mint point: a UUIDv4 is generated when the state first leaves
    IDLE (i.e. transition into WAKING) and cleared on the transition back to
    IDLE. K.1 spec phrases this as "one per IDLE → LISTENING transition";
    minting on entry to WAKING is the same identity since WAKING is the
    on-ramp to LISTENING and there is no path to LISTENING that skips WAKING.
    """

    def __init__(self, state: "ServerState", ha_bridge: "Optional[HABridge]" = None) -> None:
        self.state = state
        self.ha_bridge = ha_bridge
        self._lock = threading.Lock()
        self._generation: int = 0
        self._state: State = State.IDLE
        self._session_id: Optional[str] = None
        self._last_change_ts: float = time.monotonic()
        state.device_session = self
        state.device_state = self._state.value
        state.session_id = None
        state.generation = 0

    # -- generation counter -------------------------------------------------

    def bump_gen(self) -> int:
        with self._lock:
            self._generation += 1
            gen = self._generation
        self.state.generation = gen
        sat = getattr(self.state, "satellite", None)
        if sat is not None:
            try:
                sat._generation = gen
            except AttributeError:
                pass
        return gen

    def gen_check(self, captured: int) -> bool:
        return captured == self._generation

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def session_id(self) -> Optional[str]:
        return self._session_id

    @property
    def state_value(self) -> State:
        return self._state

    # -- state machine -----------------------------------------------------

    def transition_to(
        self,
        new_state: Union[State, str],
        *,
        reason: str = "transition",
        cancel_reason: Optional[str] = None,
    ) -> None:
        """Move into new_state, mint/clear session_id as needed, publish K.1.

        Idempotent for same-state transitions: no-op (no republish), matching
        the existing satellite `_set_state_label` semantics.
        """
        target = _coerce_state(new_state)
        prev = self._state
        if target == prev and reason == "transition":
            return

        # session_id lifecycle: any move out of IDLE that lacks a session_id
        # mints one; move into IDLE clears it. Stage C: prev_session_id is
        # captured before clearing so wake_capture.update_label can resolve
        # the sidecar by the session_id that's about to disappear.
        prev_session_id = self._session_id
        minted_session_id = False
        if prev == State.IDLE and target != State.IDLE and self._session_id is None:
            self._session_id = str(uuid.uuid4())
            minted_session_id = True
        if target == State.IDLE:
            self._session_id = None

        self._state = target
        self._last_change_ts = time.monotonic()
        self.state.device_state = target.value
        self.state.session_id = self._session_id

        # Mirror back to satellite for back-compat reads (Stage A tests + any
        # callers still reading sat._state_label / sat._session_id directly).
        # On entry to IDLE also reset v0-legacy lifecycle flags. _pipeline_active
        # is set in satellite.wakeup() and cleared by sat.stop() / _tts_finished
        # on the v0 path; the v1 (B3) path drives the turn through DS._run_turn
        # and never clears it, so wake → no_speech / empty_reply / reply_done
        # used to leave _pipeline_active=True and silently swallow every
        # subsequent wake at satellite.wakeup() L764. Resetting here makes
        # _pipeline_active derived from "DS.state == IDLE".
        sat = getattr(self.state, "satellite", None)
        if sat is not None:
            try:
                sat._state_label = target.value
                sat._session_id = self._session_id
                sat._last_state_change_ts = self._last_change_ts
                if target == State.IDLE:
                    sat._pipeline_active = False
                    sat._is_streaming_audio = False
                    sat._ha_pipeline_started = False
                    sat._continue_conversation = False
            except AttributeError:
                pass

        self._publish(reason=reason, cancel_reason=cancel_reason)

        # Stage C — wake_capture bind / label hooks. Bind happens when a new
        # session_id has just been minted (IDLE → non-IDLE) and the satellite
        # has a pending wake_id from the most recent wake-fire. Label happens
        # when a session_id is being cleared (any → IDLE) — we use the
        # prev_session_id captured above. Both are best-effort; failures
        # never break the state machine.
        wake_capture = getattr(self.state, "wake_capture", None)
        if wake_capture is not None:
            if minted_session_id and sat is not None:
                pending_wake_id = getattr(sat, "_pending_wake_id", None)
                if pending_wake_id and self._session_id is not None:
                    try:
                        wake_capture.bind_session(
                            pending_wake_id, self._session_id, self._generation
                        )
                    except Exception:
                        _LOGGER.exception("wake_capture.bind_session raised; swallowing")
                    try:
                        sat._pending_wake_id = None
                    except AttributeError:
                        pass
            if target == State.IDLE and prev_session_id is not None:
                try:
                    wake_capture.update_label(
                        prev_session_id, reason=reason, cancel_reason=cancel_reason
                    )
                except Exception:
                    _LOGGER.exception("wake_capture.update_label raised; swallowing")

    def publish_current(self, *, reason: str = "transition", cancel_reason: Optional[str] = None) -> None:
        """Re-publish current state (used by Stage F watchdog; kept here as
        a stable API even though Stage B doesn't call it)."""
        self._publish(reason=reason, cancel_reason=cancel_reason)

    def _publish(self, *, reason: str, cancel_reason: Optional[str]) -> None:
        hb = self.ha_bridge
        if hb is None:
            return
        try:
            hb.publish_state(
                state=self._state,
                generation=self._generation,
                session_id=self._session_id,
                reason=reason,
                cancel_reason=cancel_reason,
                since_ms=0,
            )
        except Exception:
            _LOGGER.exception("HABridge.publish_state raised; swallowing to keep state machine alive")

    # -- B3 lifecycle hooks (audio path) -----------------------------------

    def on_wake_chime_finished(self, captured_gen: int) -> bool:
        """v1 path: chime done, start local mic capture.

        Returns True when the transition fired; False when gen has moved
        (i.e. cancel happened during the chime) and the caller should not
        proceed with the wake. Called from satellite._on_wakeup_sound_finished
        when state.mic_capture is wired; replaces the v0 HA streaming start.
        """
        if not self.gen_check(captured_gen):
            _LOGGER.debug("on_wake_chime_finished: gen moved (captured=%d, current=%d); dropping",
                          captured_gen, self._generation)
            return False
        self.transition_to(State.LISTENING, reason="wake_chime_finished")
        mic = getattr(self.state, "mic_capture", None)
        if mic is not None:
            try:
                mic.start_capture()
            except Exception:
                _LOGGER.exception("MicCapture.start_capture raised; cancelling wake")
                self._cancel_via_satellite("MIC_CAPTURE_FAILED")
                return False
        else:
            _LOGGER.warning("on_wake_chime_finished: no mic_capture wired; v1 path stalled")
        return True

    def on_speech_captured(self, buf: "SpeechBuffer") -> None:
        """Receives a captured speech buffer from MicCapture (loop callback).

        Schedules the ASR → bridge → TTS coroutine; gen-checks at every await.
        Safe to call when state has moved (will short-circuit at the first
        gen-check).
        """
        if self._state == State.IDLE:
            _LOGGER.debug("on_speech_captured but state is IDLE; dropping")
            return
        captured_gen = self._generation
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            _LOGGER.warning("on_speech_captured called outside a running loop; dropping")
            return
        loop.create_task(self._run_turn(captured_gen, buf))

    async def _run_turn(self, captured_gen: int, buf: "SpeechBuffer") -> None:
        """ASR → bridge → TTS. Each await gen-checks; cancel = silent abort."""
        if not self.gen_check(captured_gen):
            return

        # An empty wake (user said nothing within the no-speech timeout)
        # short-circuits: no ASR call, just unduck and return to IDLE so the
        # next wake works. Mirrors how Alexa handles a "where'd you go" wake.
        if getattr(buf, "end_reason", "") == "no_speech":
            _LOGGER.info("Wake captured no speech; returning to IDLE without ASR call")
            self.transition_to(State.IDLE, reason="no_speech")
            sat = getattr(self.state, "satellite", None)
            if sat is not None:
                try:
                    sat.unduck()
                except Exception:
                    pass
            return

        asr = getattr(self.state, "asr_client", None)
        bridge = getattr(self.state, "bridge_client", None)
        tts = getattr(self.state, "tts_output", None)
        if asr is None or bridge is None or tts is None:
            _LOGGER.warning("_run_turn: missing component(s) asr=%s bridge=%s tts=%s; aborting",
                            asr is not None, bridge is not None, tts is not None)
            self._cancel_via_satellite("MISSING_COMPONENT")
            return

        self.transition_to(State.THINKING, reason="speech_captured")

        # ---- ASR -----------------------------------------------------------
        try:
            asr_result = await asr.transcribe(buf.wav_bytes)
        except Exception as exc:
            _LOGGER.warning("ASR failed: %s", exc)
            if self.gen_check(captured_gen):
                self._cancel_via_satellite("ASR_FAILED")
            return
        if not self.gen_check(captured_gen):
            _LOGGER.debug("_run_turn: gen moved after ASR; dropping reply path")
            return

        # ---- Bridge --------------------------------------------------------
        try:
            reply = await bridge.chat(
                device=self.state.room,
                generation=captured_gen,
                session_id=self._session_id or "",
                text=asr_result.text,
                asr_confidence=asr_result.confidence,
            )
        except Exception as exc:
            _LOGGER.warning("Bridge /chat failed: %s", exc)
            if self.gen_check(captured_gen):
                self._cancel_via_satellite("BRIDGE_TIMEOUT")
            return
        if not self.gen_check(captured_gen):
            _LOGGER.debug("_run_turn: gen moved after bridge; dropping TTS")
            return
        if not reply.reply.strip():
            _LOGGER.info("Bridge returned empty reply; returning to IDLE")
            self.transition_to(State.IDLE, reason="empty_reply")
            sat = getattr(self.state, "satellite", None)
            if sat is not None:
                try:
                    sat.unduck()
                except Exception:
                    pass
            return

        # ---- TTS -----------------------------------------------------------
        self.transition_to(State.SPEAKING, reason="bridge_reply")
        sat = getattr(self.state, "satellite", None)
        player = getattr(self.state, "tts_player", None)
        if player is None:
            _LOGGER.error("_run_turn: tts_player missing; cannot speak reply")
            self._cancel_via_satellite("MISSING_COMPONENT")
            return

        speak_done = asyncio.Event()

        def _on_speak_done() -> None:
            speak_done.set()

        try:
            await tts.speak(player, text=reply.reply, done_callback=_on_speak_done)
        except Exception as exc:
            _LOGGER.warning("TTS speak failed: %s", exc)
            if self.gen_check(captured_gen):
                self._cancel_via_satellite("TTS_FAILED")
            return

        # Wait for mpv playback to complete; gen-check after.
        try:
            await speak_done.wait()
        except asyncio.CancelledError:
            return
        if not self.gen_check(captured_gen):
            _LOGGER.debug("_run_turn: gen moved during TTS playback; not transitioning")
            return

        # Continue-conversation handling (D-bridge follow-up marker, Roadmap §2).
        if reply.continue_conversation:
            self.transition_to(State.FOLLOWUP, reason="follow_up")
            mic = getattr(self.state, "mic_capture", None)
            if mic is not None:
                mic.start_capture()
            self.transition_to(State.LISTENING, reason="follow_up_listening")
        else:
            self.transition_to(State.IDLE, reason="reply_done")
            # Mirror the satellite-level "unduck after TTS" UX so music returns.
            if sat is not None:
                try:
                    sat.unduck()
                except Exception:
                    pass

    def _cancel_via_satellite(self, reason: str) -> None:
        """Trigger the same cancel chain a red-button press would, with a
        component-specific reason. Bumps gen → satellite.stop() → IDLE +
        K.1 publish + bridge cancel are handled there.
        """
        sat = getattr(self.state, "satellite", None)
        if sat is None:
            self.transition_to(State.IDLE, reason="cancel_no_sat", cancel_reason=reason)
            return
        try:
            sat.stop(cancel_reason=reason)
        except Exception:
            _LOGGER.exception("satellite.stop raised during _cancel_via_satellite")
