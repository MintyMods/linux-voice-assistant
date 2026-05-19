"""Stage F integration tests — HABridge K.3/K.4 routing, K.13 admin/restart,
DeviceSession 30s state-watchdog re-assert, end-to-end cancel through
the coordinator."""

from __future__ import annotations

import asyncio
import json
import sys
from unittest.mock import MagicMock

import pytest

from linux_voice_assistant.cancel import CancelCoordinator
from linux_voice_assistant.ha_bridge import HABridge
from linux_voice_assistant.session import DeviceSession, State

from tests.conftest import make_server_state


# -- K.3 / K.4 routing through HABridge -----------------------------------


def _build_bridge(fake_paho):
    created, factory = fake_paho
    bridge = HABridge(
        room="lounge", host="127.0.0.1", port=1883, client_factory=factory,
    )
    bridge.start()
    return bridge, created[0]


def test_habridge_routes_k3_to_coordinator(fake_paho):
    bridge, fake = _build_bridge(fake_paho)
    state = make_server_state()
    state.satellite = MagicMock()
    state.device_session = MagicMock()
    state.device_session.state_value = MagicMock(value="THINKING")
    coord = CancelCoordinator(state, loop=None)
    bridge.attach_cancel_coordinator(coord)

    msg = MagicMock(
        topic="calisto/lounge/cancel",
        payload=json.dumps({"reason": "DASHBOARD"}).encode(),
    )
    bridge._on_message(fake, None, msg)

    state.satellite.stop.assert_called_once_with(cancel_reason="DASHBOARD")


def test_habridge_routes_k4_broadcast_cancel(fake_paho):
    bridge, fake = _build_bridge(fake_paho)
    state = make_server_state()
    state.satellite = MagicMock()
    state.device_session = MagicMock()
    state.alarm_controller = MagicMock()
    state.music_player = MagicMock()
    coord = CancelCoordinator(state, loop=None)
    bridge.attach_cancel_coordinator(coord)

    msg = MagicMock(
        topic="calisto/all/cancel",
        payload=json.dumps({"reason": "STOP_EVERYTHING", "scope": "all"}).encode(),
    )
    bridge._on_message(fake, None, msg)

    state.satellite.stop.assert_called_once_with(cancel_reason="STOP_EVERYTHING")
    state.alarm_controller.stop_alarm.assert_called_once()
    state.music_player.stop.assert_called_once()


def test_habridge_cancel_subscribed_at_connect(fake_paho):
    bridge, fake = _build_bridge(fake_paho)
    subs = {t for t, _qos in fake.subscriptions}
    assert "calisto/lounge/cancel" in subs
    assert "calisto/all/cancel" in subs
    assert "calisto/lounge/admin/restart" in subs


def test_habridge_admin_restart_invokes_hook(fake_paho):
    bridge, fake = _build_bridge(fake_paho)
    invoked = {"count": 0}

    def _hook():
        invoked["count"] += 1
    bridge.attach_restart_hook(_hook)

    msg = MagicMock(topic="calisto/lounge/admin/restart", payload=b"{}")
    bridge._on_message(fake, None, msg)
    assert invoked["count"] == 1


# -- DeviceSession state watchdog -----------------------------------------


@pytest.mark.asyncio
async def test_state_watchdog_republishes_current(fake_paho):
    bridge, fake = _build_bridge(fake_paho)
    state = make_server_state()
    session = DeviceSession(state, ha_bridge=bridge)
    session.transition_to("LISTENING", reason="startup")

    fake.publishes.clear()
    loop = asyncio.get_event_loop()
    session.start_state_watchdog(loop, interval_s=0)
    # 0s interval: each iteration yields once and re-publishes immediately.
    await asyncio.sleep(0.05)
    session.stop_state_watchdog()

    reassertions = [
        p for (t, p, _, _) in fake.publishes
        if t == "calisto/lounge/session/state" and "watchdog_reassert" in p
    ]
    assert len(reassertions) >= 1


# -- end-to-end cancel ----------------------------------------------------


def test_coordinator_records_cancel_count():
    state = make_server_state()
    state.satellite = MagicMock()
    coord = CancelCoordinator(state, loop=None)

    coord.cancel("RED_BUTTON_SOFT", source="hid")
    coord.cancel("DASHBOARD", source="ha")
    coord.cancel("STOP_EVERYTHING", scope="all", source="voice_tool")

    assert coord.recent_cancel_count(window_s=300.0) == 3
    assert coord.total_cancels == 3
