"""Stage F1 — CancelCoordinator: single entry-point for every cancel trigger.

Per H1 + H2: cancel is dumb on purpose. It signals teardown to each
component, then returns. Healing is the supervisor's problem (H2 L1-L6),
not the coordinator's.

Triggers route through `CancelCoordinator.cancel(reason, scope=..., ...)`:

  * HID red button   (led/hid.py → __main__._phone_cancel)
  * MQTT K.3 + K.4   (calisto/<room>/cancel + calisto/all/cancel)
  * Voice stop-word  (process_audio, gated to THINKING/SPEAKING per H1)
  * DeviceSession internal triggers (BRIDGE_TIMEOUT, SILENCE_TIMEOUT, ...)
  * HA dashboard     (DASHBOARD payload via K.3)
  * LLM cancel tool  (bridge → MQTT → K.3)

Thread-safety: `cancel()` is callable from any thread. When invoked off
the asyncio loop, the dispatch is marshalled via `loop.call_soon_threadsafe`.

Tier resolution (soft/hard) follows the K.3 reason table. Hard tier
"touches music" — coordinator stops the media player in addition to the
voice teardown that soft does. EXTERNAL inherits tier from its scope:
scope=all → hard, scope in {voice, media, alarm} → soft.

Scope determines fan-out:
  voice  : satellite voice pipeline (sat.stop)
  alarm  : alarm controller stop_alarm()
  media  : music_player.stop()
  all    : voice + alarm + media

The 5-minute rolling cancel count is exposed to F2 heartbeat via
`recent_cancel_count(window_s)`.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections import deque
from enum import Enum
from typing import TYPE_CHECKING, Any, Deque, Optional, Tuple

if TYPE_CHECKING:
    from .models import ServerState

_LOGGER = logging.getLogger(__name__)


class CancelTier(str, Enum):
    SOFT = "soft"
    HARD = "hard"


class CancelScope(str, Enum):
    VOICE = "voice"
    ALARM = "alarm"
    MEDIA = "media"
    ALL = "all"


# K.3 canonical reason → (default tier, default scope) per the K-spec table.
# EXTERNAL is special-cased: tier inferred from scope at runtime.
_REASON_TABLE: dict = {
    "RED_BUTTON_SOFT":   (CancelTier.SOFT, CancelScope.VOICE),
    "RED_BUTTON_HARD":   (CancelTier.HARD, CancelScope.ALL),
    "MIC_MUTE_SOFT":     (CancelTier.SOFT, CancelScope.VOICE),
    "MIC_MUTE_HARD":     (CancelTier.HARD, CancelScope.ALL),
    "VOICE_STOP_WORD":   (CancelTier.SOFT, CancelScope.VOICE),
    "STOP_EVERYTHING":   (CancelTier.HARD, CancelScope.ALL),
    "STOP_WORD_INPROCESS": (CancelTier.SOFT, CancelScope.VOICE),
    "SILENCE_TIMEOUT":   (CancelTier.SOFT, CancelScope.VOICE),
    "BRIDGE_TIMEOUT":    (CancelTier.HARD, CancelScope.VOICE),
    "DASHBOARD":         (CancelTier.SOFT, CancelScope.VOICE),
}

VALID_REASONS = set(_REASON_TABLE.keys()) | {"EXTERNAL"}


def resolve_reason(reason: Optional[str], scope: Optional[str]) -> Tuple[str, CancelTier, CancelScope]:
    """Return canonical (reason, tier, scope).

    Unknown reasons → EXTERNAL, tier inferred from scope (all → HARD, else SOFT).
    Caller-supplied scope overrides the table's default scope.
    """
    if reason is None or reason not in _REASON_TABLE:
        canonical = "EXTERNAL"
        s = _coerce_scope(scope, CancelScope.VOICE)
        tier = CancelTier.HARD if s == CancelScope.ALL else CancelTier.SOFT
        return canonical, tier, s
    default_tier, default_scope = _REASON_TABLE[reason]
    s = _coerce_scope(scope, default_scope)
    # If caller widened scope to ALL but reason is SOFT-default, escalate tier.
    tier = CancelTier.HARD if s == CancelScope.ALL else default_tier
    return reason, tier, s


def _coerce_scope(scope: Optional[str], fallback: CancelScope) -> CancelScope:
    if scope is None:
        return fallback
    try:
        return CancelScope(str(scope).lower())
    except ValueError:
        return fallback


class CancelCoordinator:
    """Single dispatch point for all cancel triggers.

    Thread-safe `cancel()` marshals to the asyncio loop. Per H1+H2 the
    fan-out is non-blocking and fire-and-forget: each component's teardown
    is wrapped in try/except — a dead component does not stall cancel.

    Rolling 5-minute cancel counter is tracked for K.2 heartbeat
    (`cancel_count_5m`).
    """

    def __init__(
        self,
        state: "ServerState",
        loop: "Optional[asyncio.AbstractEventLoop]" = None,
    ) -> None:
        self.state = state
        self.loop = loop
        self._lock = threading.Lock()
        self._recent: Deque[Tuple[float, str, str]] = deque()
        self._total = 0
        state.cancel_coordinator = self

    # ------------------------------------------------------------------ API

    def cancel(
        self,
        reason: Optional[str],
        *,
        scope: Optional[str] = None,
        source: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> None:
        """Fire a cancel. Safe to call from any thread.

        Off-loop callers marshal via `loop.call_soon_threadsafe`. On-loop
        callers (or callers without a loop, e.g. tests) execute inline.
        """
        canonical, tier, resolved_scope = resolve_reason(reason, scope)
        self._record(canonical, resolved_scope.value)

        loop = self.loop or getattr(self.state, "loop", None)
        if loop is None:
            self._dispatch(canonical, tier, resolved_scope, source, request_id)
            return
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is loop:
            self._dispatch(canonical, tier, resolved_scope, source, request_id)
            return
        try:
            loop.call_soon_threadsafe(
                self._dispatch, canonical, tier, resolved_scope, source, request_id,
            )
        except RuntimeError:
            # Loop closed — best-effort: run inline so nothing silently drops.
            self._dispatch(canonical, tier, resolved_scope, source, request_id)

    # ----------------------------------------------------------- internal

    def _dispatch(
        self,
        reason: str,
        tier: CancelTier,
        scope: CancelScope,
        source: Optional[str],
        request_id: Optional[str],
    ) -> None:
        _LOGGER.info(
            "cancel reason=%s tier=%s scope=%s source=%s req=%s",
            reason, tier.value, scope.value, source or "-", request_id or "-",
        )

        touches_voice = scope in (CancelScope.VOICE, CancelScope.ALL)
        touches_alarm = scope in (CancelScope.ALARM, CancelScope.ALL)
        touches_media = scope in (CancelScope.MEDIA, CancelScope.ALL) or tier == CancelTier.HARD

        if touches_voice:
            self._teardown_voice(reason)
        if touches_alarm:
            self._teardown_alarm()
        if touches_media:
            self._teardown_media()

    def _teardown_voice(self, reason: str) -> None:
        """Call sat.stop() if a satellite is connected, else fall back to
        DeviceSession-only transition. Both paths idempotent."""
        sat = getattr(self.state, "satellite", None)
        if sat is not None:
            try:
                sat.stop(cancel_reason=reason)
            except Exception:
                _LOGGER.exception("CancelCoordinator: sat.stop raised")
            return
        # No satellite — drive DeviceSession directly so the state machine
        # converges to IDLE even when the wire-side satellite is absent.
        ds = getattr(self.state, "device_session", None)
        if ds is not None:
            try:
                ds.transition_to("IDLE", reason="cancel_no_sat", cancel_reason=reason)
            except Exception:
                _LOGGER.exception("CancelCoordinator: DeviceSession.transition_to raised")

    def _teardown_alarm(self) -> None:
        ac = getattr(self.state, "alarm_controller", None)
        if ac is None:
            return
        try:
            ac.stop_alarm()
        except Exception:
            _LOGGER.exception("CancelCoordinator: alarm.stop_alarm raised")

    def _teardown_media(self) -> None:
        player = getattr(self.state, "music_player", None)
        if player is None:
            return
        try:
            player.stop()
        except Exception:
            _LOGGER.exception("CancelCoordinator: music_player.stop raised")

    # ----------------------------------------------------------- counters

    def _record(self, reason: str, scope: str) -> None:
        now = time.monotonic()
        with self._lock:
            self._recent.append((now, reason, scope))
            self._total += 1
            self._prune_locked(now)

    def _prune_locked(self, now: float, *, window_s: float = 300.0) -> None:
        cutoff = now - window_s
        while self._recent and self._recent[0][0] < cutoff:
            self._recent.popleft()

    def recent_cancel_count(self, window_s: float = 300.0) -> int:
        """Number of cancels fired in the last `window_s` seconds (default 5m)."""
        now = time.monotonic()
        with self._lock:
            self._prune_locked(now, window_s=window_s)
            return len(self._recent)

    @property
    def total_cancels(self) -> int:
        with self._lock:
            return self._total
