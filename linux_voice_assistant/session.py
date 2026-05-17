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

import logging
import threading
import time
import uuid
from enum import Enum
from typing import TYPE_CHECKING, Optional, Union

if TYPE_CHECKING:
    from .ha_bridge import HABridge
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
        # mints one; move into IDLE clears it.
        if prev == State.IDLE and target != State.IDLE and self._session_id is None:
            self._session_id = str(uuid.uuid4())
        if target == State.IDLE:
            self._session_id = None

        self._state = target
        self._last_change_ts = time.monotonic()
        self.state.device_state = target.value
        self.state.session_id = self._session_id

        # Mirror back to satellite for back-compat reads (Stage A tests + any
        # callers still reading sat._state_label / sat._session_id directly).
        sat = getattr(self.state, "satellite", None)
        if sat is not None:
            try:
                sat._state_label = target.value
                sat._session_id = self._session_id
                sat._last_state_change_ts = self._last_change_ts
            except AttributeError:
                pass

        self._publish(reason=reason, cancel_reason=cancel_reason)

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
