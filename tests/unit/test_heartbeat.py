"""Stage F2 — HeartbeatPublisher unit tests.

Covers K.2 payload shape, channel-health snapshot, rolling counter
plumbing (wake_count_5m, cancel_count_5m), mic_active / hidraw_ok edge
cases, and the bridge_reachable 3-strike health-ping behaviour.
"""

from __future__ import annotations

import asyncio
import json
import time
from unittest.mock import MagicMock

import pytest

from linux_voice_assistant.cancel import CancelCoordinator
from linux_voice_assistant.heartbeat import HeartbeatPublisher

from tests.conftest import make_server_state


def _publisher(state=None, **kwargs):
    state = state or make_server_state()
    ha_bridge = MagicMock()
    ha_bridge.publish.return_value = True
    return HeartbeatPublisher(
        state, ha_bridge=ha_bridge, room="lounge", **kwargs
    ), ha_bridge, state


def _payload_of(ha_bridge):
    """Extract the JSON payload from the last `ha_bridge.publish` call."""
    args, _ = ha_bridge.publish.call_args
    return json.loads(args[1])


def test_payload_has_all_k2_keys():
    pub, ha, _ = _publisher()
    pub.publish_once()
    payload = _payload_of(ha)
    required = {
        "ts", "uptime_s", "state", "generation", "bridge_reachable",
        "mic_active", "hidraw_ok", "mpv_channels", "wake_count_5m",
        "cancel_count_5m", "subsystems",
    }
    assert required.issubset(set(payload.keys()))


def test_topic_uses_room():
    pub, ha, _ = _publisher()
    assert pub.topic == "calisto/lounge/heartbeat"
    pub.publish_once()
    args, _ = ha.publish.call_args
    assert args[0] == "calisto/lounge/heartbeat"


def test_mpv_channels_snapshot_reads_channel_status():
    state = make_server_state()
    state.tts_player.channel_status = "degraded"
    state.chime_player.channel_status = "ok"
    state.music_player.channel_status = "dead"
    state.alarm_player.channel_status = "ok"
    pub, ha, _ = _publisher(state=state)
    pub.publish_once()
    payload = _payload_of(ha)
    assert payload["mpv_channels"]["tts"] == "degraded"
    assert payload["mpv_channels"]["media"] == "dead"
    assert payload["mpv_channels"]["chime"] == "ok"


def test_mpv_channels_missing_player_reports_dead():
    state = make_server_state()
    state.tts_player = None
    pub, ha, _ = _publisher(state=state)
    pub.publish_once()
    payload = _payload_of(ha)
    assert payload["mpv_channels"]["tts"] == "dead"


def test_mic_active_false_when_no_recent_frame():
    pub, ha, state = _publisher()
    state.last_mic_frame_ts = 0.0
    pub.publish_once()
    assert _payload_of(ha)["mic_active"] is False


def test_mic_active_true_when_frame_within_window():
    pub, ha, state = _publisher()
    state.last_mic_frame_ts = time.monotonic()
    pub.publish_once()
    assert _payload_of(ha)["mic_active"] is True


def test_hidraw_ok_uses_led_controller_last_event():
    pub, ha, state = _publisher()
    state.led_controller = MagicMock()
    state.led_controller.last_hid_event_ts = time.monotonic()
    pub.publish_once()
    assert _payload_of(ha)["hidraw_ok"] is True


def test_hidraw_ok_false_when_event_stale():
    pub, ha, state = _publisher()
    state.led_controller = MagicMock()
    state.led_controller.last_hid_event_ts = time.monotonic() - 10.0
    pub.publish_once()
    assert _payload_of(ha)["hidraw_ok"] is False


def test_wake_count_uses_state_wake_events():
    pub, ha, state = _publisher()
    now = time.monotonic()
    state.wake_events = [now - 60.0, now - 30.0, now - 1.0]
    pub.publish_once()
    assert _payload_of(ha)["wake_count_5m"] == 3


def test_wake_count_trims_stale_events():
    pub, ha, state = _publisher()
    now = time.monotonic()
    state.wake_events = [now - 600.0, now - 30.0]
    pub.publish_once()
    assert _payload_of(ha)["wake_count_5m"] == 1
    assert len(state.wake_events) == 1  # stale removed in-place


def test_cancel_count_reads_coordinator():
    state = make_server_state()
    state.satellite = MagicMock()
    coord = CancelCoordinator(state, loop=None)
    coord.cancel("RED_BUTTON_SOFT")
    coord.cancel("RED_BUTTON_SOFT")
    pub, ha, _ = _publisher(state=state)
    pub.publish_once()
    assert _payload_of(ha)["cancel_count_5m"] == 2


def test_subsystem_health_marks_failed_when_components_missing():
    state = make_server_state()
    state.asr_client = None
    state.tts_output = None
    pub, ha, _ = _publisher(state=state)
    pub.publish_once()
    payload = _payload_of(ha)
    assert payload["subsystems"]["asr_client"] == "failed"
    assert payload["subsystems"]["tts_engine"] == "failed"


def test_publish_count_tracks_calls():
    pub, ha, _ = _publisher()
    pub.publish_once()
    pub.publish_once()
    assert pub.published_count == 2


def test_publish_no_op_when_ha_bridge_absent():
    state = make_server_state()
    pub = HeartbeatPublisher(state, ha_bridge=None, room="lounge")
    assert pub.publish_once() is False
    assert pub.published_count == 0


# -- bridge ping sidecar ---------------------------------------------------


@pytest.mark.asyncio
async def test_bridge_ping_marks_reachable_on_first_ok():
    state = make_server_state()
    bc = MagicMock()
    bc.health = MagicMock(return_value=_async(MagicMock(ok=True)))
    state.bridge_client = bc
    loop = asyncio.get_event_loop()
    pub = HeartbeatPublisher(
        state, ha_bridge=MagicMock(publish=MagicMock(return_value=True)),
        room="lounge", bridge_ping_interval_s=0.01,
    )
    pub.start(loop)
    await asyncio.sleep(0.05)
    pub.stop()
    assert pub.bridge_reachable is True


@pytest.mark.asyncio
async def test_bridge_ping_three_strikes_marks_unreachable():
    state = make_server_state()
    bc = MagicMock()
    bc.health = MagicMock(return_value=_async(MagicMock(ok=False)))
    state.bridge_client = bc
    loop = asyncio.get_event_loop()
    pub = HeartbeatPublisher(
        state, ha_bridge=MagicMock(publish=MagicMock(return_value=True)),
        room="lounge", bridge_ping_interval_s=0.01, bridge_ping_strikes=3,
    )
    pub.start(loop)
    await asyncio.sleep(0.1)
    pub.stop()
    assert pub.bridge_reachable is False


def _async(value):
    """Wrap a value in a coroutine for MagicMock.return_value."""
    async def _co():
        return value
    return _co()
