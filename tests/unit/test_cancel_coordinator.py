"""Stage F1 — CancelCoordinator unit tests.

Covers reason-table resolution, tier + scope routing, EXTERNAL fallback,
rolling 5m counter, satellite + alarm + media teardown fan-out, and
thread-safe dispatch via call_soon_threadsafe.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock

import pytest

from linux_voice_assistant.cancel import (
    CancelCoordinator,
    CancelScope,
    CancelTier,
    resolve_reason,
)

from tests.conftest import make_server_state


# -- resolve_reason --------------------------------------------------------


def test_resolve_reason_canonical_table_entries():
    reason, tier, scope = resolve_reason("RED_BUTTON_SOFT", None)
    assert (reason, tier, scope) == ("RED_BUTTON_SOFT", CancelTier.SOFT, CancelScope.VOICE)


def test_resolve_reason_hard_red_button_defaults_to_all():
    _, tier, scope = resolve_reason("RED_BUTTON_HARD", None)
    assert tier == CancelTier.HARD
    assert scope == CancelScope.ALL


def test_resolve_reason_stop_everything_is_hard_all():
    _, tier, scope = resolve_reason("STOP_EVERYTHING", None)
    assert (tier, scope) == (CancelTier.HARD, CancelScope.ALL)


def test_resolve_reason_unknown_falls_back_to_external_soft():
    reason, tier, scope = resolve_reason("WHATEVER", None)
    assert reason == "EXTERNAL"
    assert tier == CancelTier.SOFT
    assert scope == CancelScope.VOICE


def test_resolve_reason_external_all_scope_is_hard():
    _, tier, scope = resolve_reason("EXTERNAL", "all")
    assert (tier, scope) == (CancelTier.HARD, CancelScope.ALL)


def test_resolve_reason_caller_widens_scope_to_all_escalates_tier():
    """Caller-supplied scope=all upgrades a soft-default reason to HARD."""
    _, tier, scope = resolve_reason("DASHBOARD", "all")
    assert (tier, scope) == (CancelTier.HARD, CancelScope.ALL)


def test_resolve_reason_invalid_scope_falls_back_to_default():
    _, tier, scope = resolve_reason("RED_BUTTON_SOFT", "weird")
    assert scope == CancelScope.VOICE


# -- routing ---------------------------------------------------------------


def _make_state(with_satellite=True, with_alarm=True):
    state = make_server_state()
    if with_satellite:
        state.satellite = MagicMock()
    else:
        state.satellite = None
    if with_alarm:
        state.alarm_controller = MagicMock()
    else:
        state.alarm_controller = None
    state.music_player = MagicMock()
    state.device_session = MagicMock()
    state.device_session.state_value = MagicMock(value="IDLE")
    return state


def test_soft_voice_cancel_routes_only_to_satellite():
    state = _make_state()
    coord = CancelCoordinator(state, loop=None)

    coord.cancel("RED_BUTTON_SOFT")

    state.satellite.stop.assert_called_once_with(cancel_reason="RED_BUTTON_SOFT")
    state.alarm_controller.stop_alarm.assert_not_called()
    state.music_player.stop.assert_not_called()


def test_hard_red_button_touches_voice_alarm_media():
    state = _make_state()
    coord = CancelCoordinator(state, loop=None)

    coord.cancel("RED_BUTTON_HARD")

    state.satellite.stop.assert_called_once_with(cancel_reason="RED_BUTTON_HARD")
    state.alarm_controller.stop_alarm.assert_called_once()
    state.music_player.stop.assert_called_once()


def test_bridge_timeout_is_hard_voice_touches_media():
    state = _make_state()
    coord = CancelCoordinator(state, loop=None)

    coord.cancel("BRIDGE_TIMEOUT")

    state.satellite.stop.assert_called_once_with(cancel_reason="BRIDGE_TIMEOUT")
    # HARD tier touches media even when scope is voice.
    state.music_player.stop.assert_called_once()


def test_scope_alarm_only_stops_alarm():
    state = _make_state()
    coord = CancelCoordinator(state, loop=None)

    coord.cancel("EXTERNAL", scope="alarm")

    state.alarm_controller.stop_alarm.assert_called_once()
    state.satellite.stop.assert_not_called()
    state.music_player.stop.assert_not_called()


def test_scope_media_only_stops_media():
    state = _make_state()
    coord = CancelCoordinator(state, loop=None)

    coord.cancel("EXTERNAL", scope="media")

    state.music_player.stop.assert_called_once()
    state.satellite.stop.assert_not_called()


def test_dashboard_default_is_soft_voice():
    state = _make_state()
    coord = CancelCoordinator(state, loop=None)

    coord.cancel("DASHBOARD")

    state.satellite.stop.assert_called_once_with(cancel_reason="DASHBOARD")
    state.music_player.stop.assert_not_called()
    state.alarm_controller.stop_alarm.assert_not_called()


def test_no_satellite_falls_back_to_device_session_transition():
    state = _make_state(with_satellite=False)
    state.device_session.transition_to = MagicMock()
    coord = CancelCoordinator(state, loop=None)

    coord.cancel("RED_BUTTON_SOFT")

    state.device_session.transition_to.assert_called_once()


def test_teardown_swallows_component_exceptions():
    state = _make_state()
    state.satellite.stop.side_effect = RuntimeError("boom")
    state.alarm_controller.stop_alarm.side_effect = RuntimeError("nope")
    state.music_player.stop.side_effect = RuntimeError("nope")
    coord = CancelCoordinator(state, loop=None)

    coord.cancel("RED_BUTTON_HARD")  # must not raise


# -- counters --------------------------------------------------------------


def test_recent_cancel_count_within_window():
    state = _make_state()
    coord = CancelCoordinator(state, loop=None)

    for _ in range(3):
        coord.cancel("RED_BUTTON_SOFT")

    assert coord.recent_cancel_count(window_s=300.0) == 3
    assert coord.total_cancels == 3


def test_recent_cancel_count_drops_after_window(monkeypatch):
    state = _make_state()
    coord = CancelCoordinator(state, loop=None)
    base = time.monotonic()

    # Fire 2 cancels "now".
    coord.cancel("RED_BUTTON_SOFT")
    coord.cancel("RED_BUTTON_SOFT")
    # Synthesise an aged entry.
    coord._recent.appendleft((base - 600.0, "RED_BUTTON_SOFT", "voice"))

    # Window=5m drops the aged entry; total stays 2 + the synthetic one is
    # not counted in total (we appended directly to the deque without
    # bumping _total — intentional, this tests prune separately).
    assert coord.recent_cancel_count(window_s=300.0) == 2


# -- threading -------------------------------------------------------------


def test_cancel_from_thread_marshals_to_loop():
    state = _make_state()
    loop = asyncio.new_event_loop()
    try:
        coord = CancelCoordinator(state, loop=loop)

        # Simulate off-loop caller: call cancel() from this thread (no
        # running loop), with loop set. Should marshal via call_soon_threadsafe.
        coord.cancel("RED_BUTTON_SOFT")

        # Run the loop briefly to drain the queued callback.
        loop.call_later(0.05, loop.stop)
        loop.run_forever()
        state.satellite.stop.assert_called_once_with(cancel_reason="RED_BUTTON_SOFT")
    finally:
        loop.close()
