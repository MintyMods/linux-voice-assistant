"""Stage B — DeviceSession + HABridge integration tests.

Covers the publish-on-transition contract, the K.1 payload schema, LWT
arming, generation lifecycle, and session_id lifecycle. Uses the fake paho
client wired by `fake_paho` (top-level conftest) — no real broker.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from linux_voice_assistant.ha_bridge import HABridge
from linux_voice_assistant.session import DeviceSession, State


def _make_state():
    # Import inside fn so the conftest stubs are in place first.
    from tests.conftest import make_server_state
    return make_server_state()


def _last_publish(client) -> dict:
    """Return the most recent K.1 state-topic publish as a dict.

    B3 added a stop-gap LED mirror publish to `calisto/<room>/led/set` (plain
    string), so the last entry in client.publishes is NOT always the K.1
    JSON we care about. Filter to the state topic before parsing.
    """
    state_topics = [p for p in client.publishes if p[0].endswith("/session/state")]
    assert state_topics, "expected at least one /session/state publish"
    return json.loads(state_topics[-1][1])


# ---- HABridge LWT + publish ------------------------------------------------


def test_habridge_arms_lwt_before_connect(fake_paho):
    created, factory = fake_paho
    hb = HABridge(room="lounge", host="127.0.0.1", port=1883, username="u", password="p", client_factory=factory)
    hb.start()
    assert len(created) == 1
    client = created[0]

    # K.1.4 / H5: will_set must be called before connect_async. The fake
    # raises if the order is reversed.
    assert client.will is not None
    will_topic, will_payload, will_qos, will_retain = client.will
    assert will_topic == "calisto/lounge/session/state"
    assert will_qos == 1
    assert will_retain is True
    lwt = json.loads(will_payload)
    assert lwt["state"] == "OFFLINE"
    assert lwt["reason"] == "lwt_triggered"
    assert lwt["generation"] == -1
    assert lwt["session_id"] is None

    assert client.connect_args == ("127.0.0.1", 1883, 60)
    assert client.loop_started is True
    assert client.username == "u"


def test_habridge_publish_state_emits_k1_payload(fake_paho):
    created, factory = fake_paho
    hb = HABridge(room="lounge", host="127.0.0.1", client_factory=factory)
    hb.start()
    client = created[0]

    hb.publish_state(
        state=State.LISTENING,
        generation=42,
        session_id="sess-xyz",
        reason="transition",
        cancel_reason=None,
    )

    payload = _last_publish(client)
    assert payload["state"] == "LISTENING"
    assert payload["generation"] == 42
    assert payload["session_id"] == "sess-xyz"
    assert payload["reason"] == "transition"
    assert payload["cancel_reason"] is None
    assert payload["since_ms"] == 0
    assert "ts" in payload
    # Must be published retained QoS1 per K.1. Filter past the B3 LED mirror.
    state_pubs = [p for p in client.publishes if p[0].endswith("/session/state")]
    topic, _, qos, retain = state_pubs[-1]
    assert topic == "calisto/lounge/session/state"
    assert qos == 1
    assert retain is True


# ---- DeviceSession state machine + K.1 wiring ------------------------------


def test_device_session_transitions_publish_to_habridge(fake_paho):
    created, factory = fake_paho
    state = _make_state()
    hb = HABridge(room="lounge", host="127.0.0.1", client_factory=factory)
    hb.start()
    client = created[0]
    ds = DeviceSession(state, ha_bridge=hb)
    # Construction itself does NOT publish — DeviceSession leaves the initial
    # IDLE silent; the bootstrap in __main__ explicitly publishes STARTING+IDLE
    # at startup so tests can isolate transition publishes.
    pre_count = len(client.publishes)

    ds.transition_to(State.WAKING)
    ds.transition_to(State.LISTENING)
    ds.transition_to(State.THINKING)
    ds.transition_to(State.SPEAKING)
    ds.transition_to(State.IDLE)

    # Filter past the B3 LED mirror publishes on calisto/<room>/led/set.
    new_publishes = [p for p in client.publishes[pre_count:] if p[0].endswith("/session/state")]
    assert [json.loads(p[1])["state"] for p in new_publishes] == [
        "WAKING", "LISTENING", "THINKING", "SPEAKING", "IDLE",
    ]
    # ALL state-topic publishes are retained QoS1.
    for _, _, qos, retain in new_publishes:
        assert qos == 1
        assert retain is True


def test_device_session_mints_session_id_on_leaving_idle():
    state = _make_state()
    ds = DeviceSession(state)
    assert ds.session_id is None
    ds.transition_to(State.WAKING)
    sid = ds.session_id
    assert sid is not None
    assert state.session_id == sid

    # Sticky through the lifecycle.
    ds.transition_to(State.LISTENING)
    ds.transition_to(State.THINKING)
    assert ds.session_id == sid

    # Cleared on entry to IDLE.
    ds.transition_to(State.IDLE)
    assert ds.session_id is None
    assert state.session_id is None

    # New session → new UUID.
    ds.transition_to(State.WAKING)
    assert ds.session_id is not None
    assert ds.session_id != sid


def test_device_session_idempotent_on_same_state():
    state = _make_state()
    ds = DeviceSession(state)
    ds.transition_to(State.WAKING)
    sid_first = ds.session_id
    ds.transition_to(State.WAKING)
    # Same-state transition with default reason is a no-op (no session_id churn).
    assert ds.session_id == sid_first


def test_device_session_generation_increments_and_mirrors_to_state():
    state = _make_state()
    ds = DeviceSession(state)
    assert ds.generation == 0
    assert state.generation == 0

    g1 = ds.bump_gen()
    g2 = ds.bump_gen()
    g3 = ds.bump_gen()

    assert (g1, g2, g3) == (1, 2, 3)
    assert ds.generation == 3
    assert state.generation == 3
    assert ds.gen_check(3) is True
    assert ds.gen_check(2) is False


def test_device_session_cancel_publish_carries_cancel_reason(fake_paho):
    created, factory = fake_paho
    state = _make_state()
    hb = HABridge(room="lounge", host="127.0.0.1", client_factory=factory)
    hb.start()
    client = created[0]
    ds = DeviceSession(state, ha_bridge=hb)

    ds.transition_to(State.WAKING)
    ds.transition_to(State.LISTENING)
    ds.transition_to(State.IDLE, reason="cancel_force", cancel_reason="RED_BUTTON_SOFT")

    payload = _last_publish(client)
    assert payload["state"] == "IDLE"
    assert payload["reason"] == "cancel_force"
    assert payload["cancel_reason"] == "RED_BUTTON_SOFT"
    assert payload["session_id"] is None


def test_device_session_no_habridge_is_safe():
    """If HABridge isn't configured (no MQTT broker env), state machine still
    runs; publishes are silently dropped."""
    state = _make_state()
    ds = DeviceSession(state, ha_bridge=None)
    ds.transition_to(State.WAKING)
    ds.transition_to(State.LISTENING)
    ds.transition_to(State.IDLE)
    assert state.device_state == "IDLE"


def test_transition_to_idle_clears_v0_pipeline_flags_on_satellite():
    """Regression: pre-fix, sat._pipeline_active was set True in wakeup() and
    only cleared by sat.stop() / v0 _tts_finished. The v1 (B3) path drives
    turns via DS._run_turn, so no_speech / empty_reply / reply_done all left
    _pipeline_active=True and the next wake bailed at satellite.wakeup() L764
    ('Ignoring wake word - pipeline already active'). This test pins that
    every DS IDLE transition resets the v0-legacy flags on the satellite."""
    state = _make_state()
    sat = MagicMock()
    sat._pipeline_active = True
    sat._is_streaming_audio = True
    sat._ha_pipeline_started = True
    sat._continue_conversation = True
    state.satellite = sat

    ds = DeviceSession(state, ha_bridge=None)
    ds.transition_to(State.WAKING)
    ds.transition_to(State.LISTENING)
    # Simulate the no_speech path: DS goes back to IDLE directly.
    ds.transition_to(State.IDLE, reason="no_speech")

    assert sat._pipeline_active is False, "next wake would silently bail in wakeup()"
    assert sat._is_streaming_audio is False
    assert sat._ha_pipeline_started is False
    assert sat._continue_conversation is False


def test_transition_to_non_idle_does_not_clear_v0_pipeline_flags():
    """Sanity: clearing must be IDLE-entry only, not every transition (else a
    WAKING→LISTENING transition would clear the flag set by wakeup()).
    """
    state = _make_state()
    sat = MagicMock()
    sat._pipeline_active = True
    sat._is_streaming_audio = True
    state.satellite = sat

    ds = DeviceSession(state, ha_bridge=None)
    ds.transition_to(State.WAKING)
    ds.transition_to(State.LISTENING)

    assert sat._pipeline_active is True
    assert sat._is_streaming_audio is True


# ---- Satellite shim integration -------------------------------------------


def test_satellite_shim_routes_through_device_session(fake_paho):
    """When DeviceSession is attached to state, satellite's _set_state_label
    and _bump_gen must flow through it (K.1 publishes on every transition,
    generation sourced from DS)."""
    import threading
    import time
    from unittest.mock import MagicMock

    from aioesphomeapi.api_pb2 import VoiceAssistantAnnounceFinished, VoiceAssistantRequest  # type: ignore

    from linux_voice_assistant.satellite import VoiceSatelliteProtocol

    created, factory = fake_paho
    state = _make_state()
    hb = HABridge(room="lounge", host="127.0.0.1", client_factory=factory)
    hb.start()
    client = created[0]
    ds = DeviceSession(state, ha_bridge=hb)

    # Bypass __init__ — same approach as Stage A tests, but with the real
    # ServerState wired in so state.device_session is the DS we built.
    sat = object.__new__(VoiceSatelliteProtocol)
    sat.state = state
    sat._generation = 0
    sat._gen_lock = threading.Lock()
    sat._pipeline_start_gen = 0
    sat._session_id = None
    sat._state_label = "IDLE"
    sat._last_state_change_ts = time.monotonic()
    sat._is_streaming_audio = False
    sat._tts_url = None
    sat._tts_played = False
    sat._continue_conversation = False
    sat._timer_finished = False
    sat._timer_ring_start = None
    sat._processing = False
    sat._pipeline_active = False
    sat._ha_pipeline_started = False
    sat._external_wake_words = {}
    sat._disconnect_event = MagicMock()
    sat.send_messages = MagicMock()
    state.satellite = sat
    state.stop_word.id = "stop_word_id"
    state.active_wake_words = set()
    state.muted = False
    state.wakeup_sound = "wakeup.flac"

    pre_count = len(client.publishes)

    # Drive a wake. The shim must route the WAKING transition through DS,
    # which mints session_id and publishes K.1.
    wake_word = MagicMock()
    wake_word.wake_word = "okay_nabu"
    sat.wakeup(wake_word)

    # WAKING publish landed (filter past the B3 LED mirror publishes).
    new_pubs = [p for p in client.publishes[pre_count:] if p[0].endswith("/session/state")]
    transitions = [json.loads(p[1])["state"] for p in new_pubs]
    assert "WAKING" in transitions
    # session_id minted via DS and mirrored to satellite.
    assert ds.session_id is not None
    assert sat._session_id == ds.session_id

    # Now cancel. Single _set_state_label call must publish IDLE with
    # cancel_force reason — and not double-publish (legacy path stays off
    # when DS is wired).
    pre_cancel = len(client.publishes)
    sat.stop(cancel_reason="RED_BUTTON_SOFT")

    cancel_publishes = [p for p in client.publishes[pre_cancel:] if p[0].endswith("/session/state")]
    # Exactly one publish for the cancel transition (no double-publish).
    idle_payloads = [json.loads(p[1]) for p in cancel_publishes if json.loads(p[1])["state"] == "IDLE"]
    assert len(idle_payloads) == 1, f"expected exactly one IDLE publish, got {idle_payloads}"
    idle = idle_payloads[0]
    assert idle["reason"] == "cancel_force"
    assert idle["cancel_reason"] == "RED_BUTTON_SOFT"
    assert idle["session_id"] is None

    # Generation bumped through DS.
    assert sat._generation == ds.generation
    assert state.generation == ds.generation
    assert ds.generation > 0

    # HA release sequence still emitted (Stage A invariant preserved).
    sent = [m for call in sat.send_messages.call_args_list for m in (call.args[0] if call.args else [])]
    assert any(isinstance(m, VoiceAssistantRequest) and not m.start for m in sent), \
        "Stage A invariant: stop() must emit VoiceAssistantRequest(start=False)"
    assert any(isinstance(m, VoiceAssistantAnnounceFinished) for m in sent), \
        "Stage A invariant: stop() must emit VoiceAssistantAnnounceFinished"
