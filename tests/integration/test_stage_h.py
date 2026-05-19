"""Stage H integration tests — WakeArbiter wired through HABridge.

Covers:
  - HABridge subscribes to `calisto/wake_arb` on connect (K.5).
  - Inbound wake_arb payloads dispatch into the attached arbiter.
  - The arbiter's own publishes go out via HABridge's paho client (QoS 0,
    not retained per K.5).
  - heartbeat fold-in: `wake_arb_stats` shows up in the K.2 payload.
"""

from __future__ import annotations

import json
import time
from unittest.mock import MagicMock

import pytest

from linux_voice_assistant.ha_bridge import HABridge
from linux_voice_assistant.heartbeat import HeartbeatPublisher
from linux_voice_assistant.wake_arbiter import WAKE_ARB_TOPIC, WakeArbiter

from tests.conftest import make_server_state


def _build_bridge(fake_paho):
    created, factory = fake_paho
    bridge = HABridge(
        room="bedroom", host="127.0.0.1", port=1883, client_factory=factory,
    )
    bridge.start()
    return bridge, created[0]


def test_habridge_subscribes_to_wake_arb_on_connect(fake_paho):
    bridge, fake = _build_bridge(fake_paho)
    subs = {t for t, _qos in fake.subscriptions}
    assert WAKE_ARB_TOPIC in subs
    qos_map = dict(fake.subscriptions)
    assert qos_map[WAKE_ARB_TOPIC] == 0  # K.5 QoS 0


def test_habridge_dispatches_wake_arb_to_arbiter(fake_paho):
    bridge, fake = _build_bridge(fake_paho)
    arbiter = WakeArbiter(room="bedroom", device_id="bedroom-host", wait_ms=50)
    bridge.attach_wake_arbiter(arbiter)

    peer_payload = json.dumps({
        "room": "bathroom", "wake_id": 1,
        "score": 0.75, "peak_score": 0.78,
        "ts": time.time(), "device_id": "bathroom-host",
    }).encode("utf-8")
    msg = MagicMock(topic=WAKE_ARB_TOPIC, payload=peer_payload)
    bridge._on_message(fake, None, msg)

    with arbiter._lock:
        assert len(arbiter._events) == 1
        assert arbiter._events[0].room == "bathroom"


def test_arbiter_publishes_via_habridge_at_qos0(fake_paho):
    bridge, fake = _build_bridge(fake_paho)
    arbiter = WakeArbiter(room="bedroom", device_id="bedroom-host", wait_ms=10)
    bridge.attach_wake_arbiter(arbiter)

    arbiter.arbitrate(score=0.6, peak_score=0.6)
    wake_publishes = [
        (topic, payload, qos, retain)
        for (topic, payload, qos, retain) in fake.publishes
        if topic == WAKE_ARB_TOPIC
    ]
    assert len(wake_publishes) == 1
    _topic, payload_raw, qos, retain = wake_publishes[0]
    assert qos == 0
    assert retain is False
    decoded = json.loads(payload_raw)
    assert decoded["room"] == "bedroom"
    assert decoded["device_id"] == "bedroom-host"


def test_heartbeat_payload_includes_wake_arb_stats(fake_paho):
    bridge, _fake = _build_bridge(fake_paho)
    state = make_server_state()
    arbiter = WakeArbiter(room="bedroom", device_id="bedroom-host", wait_ms=5)
    state.wake_arbiter = arbiter
    # Record one win + one loss so the snapshot is non-default.
    arbiter._record_outcome(won=True, margin=0.5)
    arbiter._record_outcome(won=False, margin=0.1)

    hb = HeartbeatPublisher(state, ha_bridge=bridge, room="bedroom", interval_s=60)
    payload = hb._build_payload()
    assert "wake_arb_stats" in payload
    stats = payload["wake_arb_stats"]
    assert stats["won_24h"] == 1
    assert stats["lost_24h"] == 1
    assert stats["avg_margin_24h"] == pytest.approx(0.30, abs=1e-6)


def test_heartbeat_payload_wake_arb_stats_default_when_arbiter_missing(fake_paho):
    bridge, _fake = _build_bridge(fake_paho)
    state = make_server_state()
    state.wake_arbiter = None
    hb = HeartbeatPublisher(state, ha_bridge=bridge, room="bedroom", interval_s=60)
    payload = hb._build_payload()
    assert payload["wake_arb_stats"] == {
        "won_24h": 0, "lost_24h": 0, "avg_margin_24h": 0.0,
    }
