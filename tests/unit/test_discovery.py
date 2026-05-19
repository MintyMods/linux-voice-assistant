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
