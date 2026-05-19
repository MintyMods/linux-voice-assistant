"""Stage D — SpeakerVerifierDiscovery unit tests.

Drives publish_configs / publish_state / handle_*_command directly with a
fake HABridge so the wire-level Discovery payloads + tunable command
handling stay locked to N.2 templates B + C.
"""

from __future__ import annotations

import json
from typing import Any, List, Tuple

import pytest

from linux_voice_assistant.discovery import (
    SV_THRESHOLD_MAX,
    SV_THRESHOLD_MIN,
    SV_THRESHOLD_STEP,
    SpeakerVerifierDiscovery,
)


class _FakeBridge:
    def __init__(self) -> None:
        self.publishes: List[Tuple[str, str, bool]] = []

    def publish(self, topic: str, payload: str, *, qos: int = 1, retain: bool = True) -> bool:
        self.publishes.append((topic, payload, retain))
        return True

    def publishes_to(self, topic: str) -> List[str]:
        return [p for t, p, _ in self.publishes if t == topic]


class _FakeState:
    sv_threshold: float = 0.70
    sv_audible_notify: bool = True


class _FakeEnrollments:
    def __init__(self) -> None:
        self.threshold: float = 0.70
        self.saved = 0

    def save(self) -> None:
        self.saved += 1


def _make() -> Tuple[SpeakerVerifierDiscovery, _FakeBridge, _FakeState, _FakeEnrollments]:
    bridge = _FakeBridge()
    state = _FakeState()
    enrollments = _FakeEnrollments()
    sv = SpeakerVerifierDiscovery(
        ha_bridge=bridge,
        state=state,
        enrollments=enrollments,
        room="lounge",
    )
    return sv, bridge, state, enrollments


# -- configs -------------------------------------------------------------------


def test_publish_configs_emits_threshold_number_per_template_b():
    sv, bridge, _, _ = _make()
    n = sv.publish_configs()
    assert n == 2

    cfg_topic = "homeassistant/number/calisto_lounge_sv_threshold/config"
    payloads = bridge.publishes_to(cfg_topic)
    assert len(payloads) == 1
    cfg = json.loads(payloads[0])
    assert cfg["unique_id"] == "calisto_lounge_sv_threshold"
    assert cfg["state_topic"] == "calisto/lounge/tunable/sv_threshold/state"
    assert cfg["command_topic"] == "calisto/lounge/tunable/sv_threshold/set"
    assert cfg["min"] == SV_THRESHOLD_MIN
    assert cfg["max"] == SV_THRESHOLD_MAX
    assert cfg["step"] == SV_THRESHOLD_STEP
    assert cfg["mode"] == "slider"
    assert cfg["value_template"] == "{{ value_json.value }}"
    assert cfg["device"]["identifiers"] == ["calisto_lounge"]


def test_publish_configs_emits_audible_notify_switch_per_template_c():
    sv, bridge, _, _ = _make()
    sv.publish_configs()

    cfg_topic = "homeassistant/switch/calisto_lounge_sv_audible_notify/config"
    payloads = bridge.publishes_to(cfg_topic)
    assert len(payloads) == 1
    cfg = json.loads(payloads[0])
    assert cfg["unique_id"] == "calisto_lounge_sv_audible_notify"
    assert cfg["state_topic"] == "calisto/lounge/tunable/sv_audible_notify/state"
    assert cfg["command_topic"] == "calisto/lounge/tunable/sv_audible_notify/set"
    assert cfg["payload_on"] == "{\"value\": true}"
    assert cfg["payload_off"] == "{\"value\": false}"
    assert cfg["state_on"] is True
    assert cfg["state_off"] is False


def test_publish_configs_marks_retained():
    sv, bridge, _, _ = _make()
    sv.publish_configs()
    for t, _p, retain in bridge.publishes:
        assert retain is True, f"Discovery config {t} must be retained"


# -- state ---------------------------------------------------------------------


def test_publish_state_writes_current_threshold_and_notify():
    sv, bridge, state, _ = _make()
    state.sv_threshold = 0.65
    state.sv_audible_notify = False

    n = sv.publish_state()
    assert n == 2

    threshold_state = bridge.publishes_to("calisto/lounge/tunable/sv_threshold/state")
    assert len(threshold_state) == 1
    assert json.loads(threshold_state[0]) == {"value": 0.65}

    notify_state = bridge.publishes_to("calisto/lounge/tunable/sv_audible_notify/state")
    assert len(notify_state) == 1
    assert json.loads(notify_state[0]) == {"value": False}


def test_start_publishes_configs_then_state():
    sv, bridge, _, _ = _make()
    sv.start()
    topics = [t for t, _p, _r in bridge.publishes]
    assert "homeassistant/number/calisto_lounge_sv_threshold/config" in topics
    assert "homeassistant/switch/calisto_lounge_sv_audible_notify/config" in topics
    assert "calisto/lounge/tunable/sv_threshold/state" in topics
    assert "calisto/lounge/tunable/sv_audible_notify/state" in topics


# -- threshold command ---------------------------------------------------------


def test_handle_threshold_command_clamps_to_range_and_persists():
    sv, bridge, state, enrollments = _make()

    assert sv.handle_threshold_command(b'{"value": 0.55}') is True
    assert state.sv_threshold == pytest.approx(0.55)
    assert enrollments.threshold == pytest.approx(0.55)
    assert enrollments.saved == 1

    state_payloads = bridge.publishes_to("calisto/lounge/tunable/sv_threshold/state")
    assert json.loads(state_payloads[-1]) == {"value": 0.55}


def test_handle_threshold_command_clamps_below_minimum():
    sv, _, state, _ = _make()
    sv.handle_threshold_command(b'{"value": 0.10}')
    assert state.sv_threshold == pytest.approx(SV_THRESHOLD_MIN)


def test_handle_threshold_command_clamps_above_maximum():
    sv, _, state, _ = _make()
    sv.handle_threshold_command(b'{"value": 0.99}')
    assert state.sv_threshold == pytest.approx(SV_THRESHOLD_MAX)


def test_handle_threshold_command_rejects_malformed_payload():
    sv, _, state, enrollments = _make()
    assert sv.handle_threshold_command(b"not-json") is False
    assert sv.handle_threshold_command(b'{"oops": 0.5}') is False
    assert sv.handle_threshold_command(b'{"value": "high"}') is False
    assert state.sv_threshold == 0.70
    assert enrollments.saved == 0


def test_handle_threshold_keeps_in_memory_value_when_save_raises():
    sv, _, state, enrollments = _make()

    def boom() -> None:
        raise OSError("disk full")

    enrollments.save = boom  # type: ignore[assignment]
    assert sv.handle_threshold_command(b'{"value": 0.62}') is True
    assert state.sv_threshold == pytest.approx(0.62)
    assert enrollments.threshold == pytest.approx(0.62)


# -- audible_notify command ----------------------------------------------------


def test_handle_audible_notify_command_accepts_bool():
    sv, bridge, state, _ = _make()
    assert sv.handle_audible_notify_command(b'{"value": false}') is True
    assert state.sv_audible_notify is False
    payloads = bridge.publishes_to("calisto/lounge/tunable/sv_audible_notify/state")
    assert json.loads(payloads[-1]) == {"value": False}

    assert sv.handle_audible_notify_command(b'{"value": true}') is True
    assert state.sv_audible_notify is True


def test_handle_audible_notify_command_accepts_string_truthy():
    sv, _, state, _ = _make()
    sv.handle_audible_notify_command(b'{"value": "on"}')
    assert state.sv_audible_notify is True
    sv.handle_audible_notify_command(b'{"value": "off"}')
    assert state.sv_audible_notify is False


def test_handle_audible_notify_rejects_malformed():
    sv, _, state, _ = _make()
    assert sv.handle_audible_notify_command(b"not-json") is False
    assert sv.handle_audible_notify_command(b'{"oops": true}') is False
    assert state.sv_audible_notify is True


# ============================================================================
# Stage G — EntitySurface + StatePublishCounter
# ============================================================================


from linux_voice_assistant.discovery import EntitySurface, StatePublishCounter


class _FakeState:  # type: ignore[no-redef]
    sv_threshold: float = 0.70
    sv_audible_notify: bool = True


def _make_surface() -> Tuple[EntitySurface, _FakeBridge]:
    bridge = _FakeBridge()
    state = _FakeState()
    surface = EntitySurface(ha_bridge=bridge, room="lounge", state=state)
    return surface, bridge


def _published_configs(bridge: _FakeBridge):
    """Return {component+thing: payload dict} for every Discovery config publish."""
    out: dict = {}
    for topic, payload, retain in bridge.publishes:
        if not topic.startswith("homeassistant/"):
            continue
        # homeassistant/<component>/<unique_id>/config
        parts = topic.split("/")
        assert retain is True, f"Discovery config {topic} must be retained"
        assert parts[-1] == "config"
        unique_id = parts[-2]
        out[unique_id] = json.loads(payload)
    return out


def test_publish_configs_emits_session_state_sensors():
    surface, bridge = _make_surface()
    n = surface.publish_configs()
    assert n >= 3
    cfgs = _published_configs(bridge)
    state_cfg = cfgs["calisto_lounge_state"]
    assert state_cfg["state_topic"] == "calisto/lounge/session/state"
    assert state_cfg["value_template"] == "{{ value_json.state }}"
    assert state_cfg["device"]["identifiers"] == ["calisto_lounge"]
    assert state_cfg["availability_topic"] == "calisto/lounge/heartbeat"

    gen_cfg = cfgs["calisto_lounge_generation"]
    assert gen_cfg["value_template"] == "{{ value_json.generation }}"

    cancel_cfg = cfgs["calisto_lounge_last_cancel_reason"]
    assert "cancel_reason" in cancel_cfg["value_template"]


def test_publish_configs_emits_connectivity_binary_sensors():
    surface, bridge = _make_surface()
    surface.publish_configs()
    cfgs = _published_configs(bridge)

    online = cfgs["calisto_lounge_online"]
    assert online["state_topic"] == "calisto/lounge/heartbeat"
    assert online["device_class"] == "connectivity"
    assert online["expire_after"] == 300
    assert online["payload_on"] == "online"
    assert online["payload_off"] == "offline"

    bridge_cfg = cfgs["calisto_lounge_bridge_reachable"]
    assert bridge_cfg["device_class"] == "connectivity"
    assert "bridge_reachable" in bridge_cfg["value_template"]


def test_publish_configs_emits_per_channel_mpv_health():
    surface, bridge = _make_surface()
    surface.publish_configs()
    cfgs = _published_configs(bridge)
    for channel in ("tts", "chime", "media", "alarm"):
        uid = f"calisto_lounge_mpv_{channel}_health"
        assert uid in cfgs, f"missing {uid}"
        assert cfgs[uid]["state_topic"] == "calisto/lounge/heartbeat"
        assert channel in cfgs[uid]["value_template"]


def test_publish_configs_emits_audio_health_aggregate():
    surface, bridge = _make_surface()
    surface.publish_configs()
    cfgs = _published_configs(bridge)
    audio = cfgs["calisto_lounge_audio_health"]
    tmpl = audio["value_template"]
    assert "dead" in tmpl
    assert "degraded" in tmpl
    assert "ok" in tmpl


def test_publish_configs_emits_volume_sensor():
    surface, bridge = _make_surface()
    surface.publish_configs()
    cfgs = _published_configs(bridge)
    vol = cfgs["calisto_lounge_volume"]
    assert vol["state_topic"] == "calisto/lounge/volume/state"
    assert vol["unit_of_measurement"] == "%"


def test_publish_configs_all_marked_retained():
    surface, bridge = _make_surface()
    surface.publish_configs()
    for topic, _payload, retain in bridge.publishes:
        assert retain is True, f"Discovery config {topic} must be retained"


def test_publish_state_no_tunables_yet_returns_zero():
    surface, _bridge = _make_surface()
    assert surface.publish_state() == 0


def test_subscription_topics_empty_in_read_only_commit():
    surface, _bridge = _make_surface()
    assert surface.subscription_topics() == []


def test_route_returns_false_for_unknown_topic():
    surface, _bridge = _make_surface()
    assert surface.route("calisto/lounge/tunable/unknown/set", b"{}") is False


def test_start_logs_and_does_not_raise():
    surface, bridge = _make_surface()
    surface.start()
    # Configs published; no exception.
    assert any(t.startswith("homeassistant/") for t, _p, _r in bridge.publishes)


# -- StatePublishCounter ------------------------------------------------------


def test_state_publish_counter_records_within_window():
    counter = StatePublishCounter(window_s=60.0)
    assert counter.count() == 0
    counter.record()
    counter.record()
    counter.record()
    assert counter.count() == 3


def test_register_tunable_number_publishes_template_b_config():
    surface, bridge = _make_surface()
    value = {"x": 0.5}
    surface.register_tunable_number(
        thing="wake_sensitivity",
        name_suffix="Wake Sensitivity",
        min_value=0.1, max_value=0.9, step=0.05,
        getter=lambda: value["x"],
        setter=lambda v: value.__setitem__("x", v),
    )
    surface.publish_configs()
    cfgs = _published_configs(bridge)
    cfg = cfgs["calisto_lounge_wake_sensitivity"]
    assert cfg["state_topic"] == "calisto/lounge/tunable/wake_sensitivity/state"
    assert cfg["command_topic"] == "calisto/lounge/tunable/wake_sensitivity/set"
    assert cfg["min"] == 0.1
    assert cfg["max"] == 0.9
    assert cfg["step"] == 0.05
    assert cfg["mode"] == "slider"
    assert cfg["value_template"] == "{{ value_json.value }}"


def test_register_tunable_number_emits_state_on_publish_state():
    surface, bridge = _make_surface()
    value = {"x": 0.55}
    surface.register_tunable_number(
        thing="wake_sensitivity",
        name_suffix="Wake Sensitivity",
        min_value=0.1, max_value=0.9, step=0.05,
        getter=lambda: value["x"],
        setter=lambda v: value.__setitem__("x", v),
    )
    n = surface.publish_state()
    assert n == 1
    state_msgs = [(t, p) for t, p, _ in bridge.publishes if t == "calisto/lounge/tunable/wake_sensitivity/state"]
    assert len(state_msgs) == 1
    assert json.loads(state_msgs[0][1]) == {"value": 0.55}


def test_register_tunable_number_clamps_and_invokes_setter():
    surface, bridge = _make_surface()
    value = {"x": 0.5}
    surface.register_tunable_number(
        thing="wake_sensitivity",
        name_suffix="Wake Sensitivity",
        min_value=0.1, max_value=0.9, step=0.05,
        getter=lambda: value["x"],
        setter=lambda v: value.__setitem__("x", v),
    )

    set_topic = "calisto/lounge/tunable/wake_sensitivity/set"
    assert surface.route(set_topic, b'{"value": 0.42}') is True
    assert value["x"] == pytest.approx(0.42)

    # Clamps high.
    assert surface.route(set_topic, b'{"value": 5.0}') is True
    assert value["x"] == pytest.approx(0.9)
    # Clamps low.
    assert surface.route(set_topic, b'{"value": -1.0}') is True
    assert value["x"] == pytest.approx(0.1)


def test_register_tunable_number_int_cast():
    surface, _bridge = _make_surface()
    value = {"x": 30}
    surface.register_tunable_number(
        thing="duck_floor_pct",
        name_suffix="Duck Floor",
        min_value=10, max_value=80, step=5,
        getter=lambda: value["x"],
        setter=lambda v: value.__setitem__("x", v),
        is_int=True,
    )
    surface.route("calisto/lounge/tunable/duck_floor_pct/set", b'{"value": 42.7}')
    assert value["x"] == 43
    assert isinstance(value["x"], int)


def test_register_tunable_number_rejects_malformed_payloads():
    surface, _bridge = _make_surface()
    value = {"x": 0.5}
    surface.register_tunable_number(
        thing="wake_sensitivity",
        name_suffix="Wake Sensitivity",
        min_value=0.1, max_value=0.9, step=0.05,
        getter=lambda: value["x"],
        setter=lambda v: value.__setitem__("x", v),
    )
    set_topic = "calisto/lounge/tunable/wake_sensitivity/set"
    surface.route(set_topic, b"not-json")
    surface.route(set_topic, b'{"oops": 0.4}')
    surface.route(set_topic, b'{"value": "nope"}')
    assert value["x"] == 0.5  # unchanged


def test_register_tunable_number_route_appears_in_subscriptions():
    surface, _bridge = _make_surface()
    surface.register_tunable_number(
        thing="duck_attack_ms",
        name_suffix="Duck Attack",
        min_value=50, max_value=500, step=10,
        getter=lambda: 150,
        setter=lambda v: None,
        is_int=True,
    )
    assert "calisto/lounge/tunable/duck_attack_ms/set" in surface.subscription_topics()


def test_publish_configs_includes_state_publishes_per_min_sensor():
    surface, bridge = _make_surface()
    surface.publish_configs()
    cfgs = _published_configs(bridge)
    spm = cfgs["calisto_lounge_state_publishes_per_min"]
    assert spm["state_topic"] == "calisto/lounge/heartbeat"
    assert "state_publishes_per_min" in spm["value_template"]
    assert spm["unit_of_measurement"] == "/min"


def test_register_tunable_switch_template_c_json_envelope():
    surface, bridge = _make_surface()
    value = {"x": True}
    surface.register_tunable_switch(
        thing="audible_notify",
        name_suffix="Audible Notify",
        getter=lambda: value["x"],
        setter=lambda v: value.__setitem__("x", v),
    )
    surface.publish_configs()
    cfgs = _published_configs(bridge)
    cfg = cfgs["calisto_lounge_audible_notify"]
    assert cfg["payload_on"] == "{\"value\": true}"
    assert cfg["payload_off"] == "{\"value\": false}"
    assert cfg["state_on"] is True
    assert cfg["state_off"] is False
    assert cfg["value_template"] == "{{ value_json.value }}"


def test_register_tunable_switch_handles_set_command():
    surface, bridge = _make_surface()
    value = {"x": True}
    surface.register_tunable_switch(
        thing="audible_notify",
        name_suffix="Audible Notify",
        getter=lambda: value["x"],
        setter=lambda v: value.__setitem__("x", v),
    )
    set_topic = "calisto/lounge/tunable/audible_notify/set"
    assert surface.route(set_topic, b'{"value": false}') is True
    assert value["x"] is False

    # String coercion
    surface.route(set_topic, b'{"value": "on"}')
    assert value["x"] is True
    surface.route(set_topic, b'{"value": "off"}')
    assert value["x"] is False

    # Malformed silently rejected
    surface.route(set_topic, b"not-json")
    surface.route(set_topic, b'{"oops": true}')
    assert value["x"] is False  # unchanged


def test_register_tunable_switch_emits_state():
    surface, bridge = _make_surface()
    value = {"x": False}
    surface.register_tunable_switch(
        thing="audible_notify",
        name_suffix="Audible Notify",
        getter=lambda: value["x"],
        setter=lambda v: value.__setitem__("x", v),
    )
    surface.publish_state()
    state_msgs = [(t, p) for t, p, _ in bridge.publishes if t == "calisto/lounge/tunable/audible_notify/state"]
    assert len(state_msgs) == 1
    assert json.loads(state_msgs[0][1]) == {"value": False}


def test_register_passthrough_switch_publishes_config_only():
    surface, bridge = _make_surface()
    surface.register_passthrough_switch(
        thing="mute",
        name_suffix="Mute",
        state_topic="calisto/lounge/mute/state",
        set_topic="calisto/lounge/mute/set",
    )
    surface.publish_configs()
    cfgs = _published_configs(bridge)
    cfg = cfgs["calisto_lounge_mute"]
    assert cfg["state_topic"] == "calisto/lounge/mute/state"
    assert cfg["command_topic"] == "calisto/lounge/mute/set"
    assert cfg["payload_on"] == "on"
    assert cfg["payload_off"] == "off"
    # value_template intentionally absent — state is a plain string.
    assert "value_template" not in cfg

    # No subscription claimed; no state emitter; no route owned.
    assert "calisto/lounge/mute/set" not in surface.subscription_topics()
    assert surface.publish_state() == 0
    assert surface.route("calisto/lounge/mute/set", b"on") is False


def test_register_tunable_select_template_d():
    surface, bridge = _make_surface()
    value = {"x": "Alarm clock.ogg"}
    surface.register_tunable_select(
        thing="alarm_ringtone",
        name_suffix="Alarm Ringtone",
        options=["Alarm clock.ogg", "Beep.ogg", "Ringer.ogg"],
        getter=lambda: value["x"],
        setter=lambda v: value.__setitem__("x", v),
    )
    surface.publish_configs()
    cfgs = _published_configs(bridge)
    cfg = cfgs["calisto_lounge_alarm_ringtone"]
    assert cfg["options"] == ["Alarm clock.ogg", "Beep.ogg", "Ringer.ogg"]
    assert cfg["command_template"] == "{\"value\": \"{{ value }}\"}"


def test_register_tunable_select_rejects_unknown_option():
    surface, _bridge = _make_surface()
    value = {"x": "Alarm clock.ogg"}
    surface.register_tunable_select(
        thing="alarm_ringtone",
        name_suffix="Alarm Ringtone",
        options=["Alarm clock.ogg", "Beep.ogg"],
        getter=lambda: value["x"],
        setter=lambda v: value.__setitem__("x", v),
    )
    set_topic = "calisto/lounge/tunable/alarm_ringtone/set"

    # Valid option accepted.
    surface.route(set_topic, b'{"value": "Beep.ogg"}')
    assert value["x"] == "Beep.ogg"

    # Unknown option rejected.
    surface.route(set_topic, b'{"value": "Tubular.ogg"}')
    assert value["x"] == "Beep.ogg"  # unchanged


def test_register_button_publishes_config():
    surface, bridge = _make_surface()
    surface.register_button(
        thing="stop",
        name_suffix="Stop",
        command_topic="calisto/lounge/cancel",
        press_payload='{"reason":"DASHBOARD","source":"ha_dashboard"}',
    )
    surface.publish_configs()
    cfgs = _published_configs(bridge)
    btn = cfgs["calisto_lounge_stop"]
    assert btn["command_topic"] == "calisto/lounge/cancel"
    assert btn["payload_press"] == '{"reason":"DASHBOARD","source":"ha_dashboard"}'
    assert "availability_template" not in btn  # buttons don't need it
    # Button doesn't claim a subscription (LVA already subscribes to cancel).
    assert "calisto/lounge/cancel" not in surface.subscription_topics()


def test_register_tunable_number_with_unit_in_payload():
    surface, bridge = _make_surface()
    surface.register_tunable_number(
        thing="duck_floor_pct",
        name_suffix="Duck Floor",
        min_value=10, max_value=80, step=5,
        getter=lambda: 30,
        setter=lambda v: None,
        unit="%", is_int=True,
    )
    surface.publish_configs()
    cfgs = _published_configs(bridge)
    assert cfgs["calisto_lounge_duck_floor_pct"]["unit_of_measurement"] == "%"


def test_state_publish_counter_trims_outside_window(monkeypatch):
    counter = StatePublishCounter(window_s=60.0)
    base = [1000.0]

    def fake_monotonic():
        return base[0]

    monkeypatch.setattr("linux_voice_assistant.discovery.time.monotonic", fake_monotonic)

    counter.record()  # at t=1000
    base[0] = 1059.0
    counter.record()  # at t=1059, both still in window
    assert counter.count() == 2
    base[0] = 1061.0  # t=1061: first event (at 1000) is now outside 60s window
    assert counter.count() == 1
    base[0] = 1120.0  # both expired
    assert counter.count() == 0


# ============================================================================
# Stage H — N.2 wake-arbitration entries
# ============================================================================


def test_entity_surface_emits_stage_h_wake_arbitration_sensors():
    surface, bridge = _make_surface()
    surface.publish_configs()
    cfgs = _published_configs(bridge)

    won = cfgs["calisto_lounge_wake_arbitrations_won_24h"]
    assert won["state_topic"] == "calisto/lounge/heartbeat"
    assert "wake_arb_stats.won_24h" in won["value_template"]

    lost = cfgs["calisto_lounge_wake_arbitrations_lost_24h"]
    assert "wake_arb_stats.lost_24h" in lost["value_template"]

    margin = cfgs["calisto_lounge_wake_arbitration_avg_margin_24h"]
    assert "wake_arb_stats.avg_margin_24h" in margin["value_template"]
