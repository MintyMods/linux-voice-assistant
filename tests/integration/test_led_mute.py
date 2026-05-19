"""Stage E.1 — integration tests for LED + mute + HID + back-compat MQTT.

Covers the absorbed subsystem end-to-end without touching real hardware:

  * `FakeHidWriter` replaces `linux_voice_assistant.led.hid.HidWriter` —
    records every byte sequence + write_one in order.
  * Path B mute uses two injected callables (`mic_capture_mute` /
    `mic_capture_unmute`) to gate `MicCapture`'s frame flow. The tests
    use simple counter callables — the audio claim is NOT touched, so
    there's no recorder simulator needed any more.
  * `HABridge` is wired with the `fake_paho` factory from
    `tests/conftest.py` so the MQTT subscribe/publish path runs without
    a real broker.

Scope per Stage E.1 plan §12 (post Path-B pivot 2026-05-18):
  1. `enter_private` paints the cosmetic red palette + invokes the
     MicCapture mute gate.
  2. `exit_private` clears the cosmetic palette + invokes the unmute
     gate; LedController re-renders the last K.1 state.
  3. K.1 state transitions render the correct cosmetic palette.
  4. Muted state suppresses subsequent `on_state` rendering (M2).
  5. While muted, `apply_legacy` suppresses non-mute legacy verbs.
  6. Inbound MQTT routes:
       calisto/<room>/led/set        → apply_legacy
       calisto/<room>/volume/set     → bar.apply
       calisto/<room>/ring/set       → ring.start/.stop
       calisto/<room>/mute/set       → set_private (M3, new)
  7. Outbound publishers (`mute/state`, `volume/state`, phone-button)
     fire with the right retain flag.

The write log + gate counters let assertions reach the actual sequence
the firmware would receive — see `mute_button_probe_results.md` for why
these bytes were chosen.
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any, Callable, List, Optional, Tuple

import pytest

from linux_voice_assistant.ha_bridge import HABridge
from linux_voice_assistant.led import LedController
from linux_voice_assistant.led.hid import (
    REPORT_HOLD_LED,
    REPORT_MUTE_LED,
    REPORT_OFFHOOK_LED,
    REPORT_RING_LED,
)
from linux_voice_assistant.session import State


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class FakeHidWriter:
    """Records every HID write. Thread-safe."""

    def __init__(self, return_ok: bool = True) -> None:
        self._lock = threading.Lock()
        self.sequences: List[List[Tuple[int, ...]]] = []
        self.flat_writes: List[Tuple[int, ...]] = []
        self.return_ok = return_ok

    def write_seq(self, *payloads, pause: float = 0.05) -> bool:  # noqa: ARG002
        with self._lock:
            seq = [tuple(p) for p in payloads]
            self.sequences.append(seq)
            self.flat_writes.extend(seq)
        return self.return_ok

    def write_one(self, report_id: int, on: bool) -> bool:
        return self.write_seq((report_id, 1 if on else 0), pause=0)

    def reset(self) -> None:
        with self._lock:
            self.sequences.clear()
            self.flat_writes.clear()


class MicGateRecorder:
    """Counts mute/unmute callback invocations. Wired into LedController
    as the `mic_capture_mute` / `mic_capture_unmute` callables."""

    def __init__(self) -> None:
        self.mute_count = 0
        self.unmute_count = 0

    def mute(self) -> None:
        self.mute_count += 1

    def unmute(self) -> None:
        self.unmute_count += 1


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def asyncio_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def writer():
    return FakeHidWriter()


@pytest.fixture
def mic_gate():
    return MicGateRecorder()


@pytest.fixture
def controller(asyncio_loop, writer, mic_gate):
    """LedController wired with fakes everywhere."""
    return LedController(
        loop=asyncio_loop,
        mic_capture_mute=mic_gate.mute,
        mic_capture_unmute=mic_gate.unmute,
        writer=writer,
    )


# ---------------------------------------------------------------------------
# §12.1  Path B mute: enter_private / exit_private
# ---------------------------------------------------------------------------


def test_enter_private_paints_cosmetic_red_palette(controller, writer, mic_gate):
    """Path B enter: `17 01 + 09 01` in that order, MicCapture gated."""
    assert controller.set_private(True, source="test") is True
    assert controller.is_private() is True

    # Path B writes the cosmetic palette in a single write_seq.
    mute_seqs = [
        s for s in writer.sequences
        if (REPORT_OFFHOOK_LED, 0x01) in s and (REPORT_MUTE_LED, 0x01) in s
    ]
    assert mute_seqs, f"Expected a sequence with OFFHOOK on + MUTE on; got {writer.sequences}"
    seq = mute_seqs[0]
    # Order within the seq matters — OFFHOOK before MUTE (firmware
    # unlocks the red mute palette only once OFFHOOK is asserted).
    offhook_idx = seq.index((REPORT_OFFHOOK_LED, 0x01))
    mute_idx = seq.index((REPORT_MUTE_LED, 0x01))
    assert offhook_idx < mute_idx, (
        f"OFFHOOK must precede MUTE in cosmetic enter; got {seq}"
    )

    # MicCapture frame-gate fired once.
    assert mic_gate.mute_count == 1
    assert mic_gate.unmute_count == 0


def test_exit_private_clears_cosmetic_palette(controller, writer, mic_gate):
    """Path B exit: `09 00 + 17 00`, MicCapture ungated."""
    controller.set_private(True, source="test")
    assert mic_gate.mute_count == 1
    writer.reset()

    assert controller.set_private(False, source="test") is True
    assert controller.is_private() is False

    # First written sequence after reset should be the cosmetic clear.
    seq = writer.sequences[0]
    assert seq == [
        (REPORT_MUTE_LED, 0x00),
        (REPORT_OFFHOOK_LED, 0x00),
    ], f"Cosmetic exit must clear MUTE then OFFHOOK; got {seq}"

    assert mic_gate.unmute_count == 1


def test_enter_private_audio_claim_not_touched(controller, writer, mic_gate):
    """Path B regression: NO telephony-page writes (`0x0A/0x0E/0x46`) —
    those would only land in firmware after an audio-claim release, and
    Path B keeps the claim open. The mute gate is the load-bearing piece."""
    controller.set_private(True, source="test")
    # No CALL_STATE writes anywhere in the mute path.
    for seq in writer.sequences:
        for report_id, _value in seq:
            assert report_id not in (0x0A, 0x0E, 0x46), (
                f"Path B must not write telephony page; got {seq}"
            )


def test_set_private_idempotent(controller, mic_gate):
    """Calling set_private(True) twice should only fire one gate event."""
    controller.set_private(True, source="test")
    controller.set_private(True, source="test")
    assert mic_gate.mute_count == 1


# ---------------------------------------------------------------------------
# §12.2  K.1 state → palette rendering
# ---------------------------------------------------------------------------


def test_on_state_renders_listening_pulse_on_waking(controller, writer):
    writer.reset()
    controller.on_state(State.WAKING)
    # listening_pulse() calls `RING off` then starts a thread pulsing OFFHOOK.
    # We just check at least one OFFHOOK_LED on was written before we tear down.
    time.sleep(0.05)
    assert any(w == (REPORT_OFFHOOK_LED, 1) for w in writer.flat_writes)
    controller.on_state(State.IDLE)


def test_on_state_renders_processing_steady(controller, writer):
    writer.reset()
    controller.on_state(State.THINKING)
    # processing_steady: OFFHOOK off, RING on. Both should appear.
    assert (REPORT_OFFHOOK_LED, 0) in writer.flat_writes
    assert (REPORT_RING_LED, 1) in writer.flat_writes


def test_on_state_renders_speaking_pulse(controller, writer):
    writer.reset()
    controller.on_state(State.SPEAKING)
    time.sleep(0.05)
    assert any(w == (REPORT_RING_LED, 1) for w in writer.flat_writes)
    controller.on_state(State.IDLE)


def test_on_state_on_degraded_writes_hold_led(controller, writer):
    writer.reset()
    controller.on_state(State.DEGRADED)
    assert (REPORT_HOLD_LED, 1) in writer.flat_writes


def test_muted_suppresses_on_state_rendering(controller, writer):
    """M2: while muted, on_state must not paint state-palette LEDs —
    the mute overlay is the visual source of truth."""
    controller.set_private(True, source="test")
    writer.reset()
    controller.on_state(State.WAKING)
    time.sleep(0.05)
    assert (REPORT_OFFHOOK_LED, 1) not in writer.flat_writes


def test_muted_suppresses_apply_legacy_non_mute_verbs(controller, writer):
    """M2: while muted, legacy verbs other than mute/unmute must be
    suppressed — they would overwrite the red overlay."""
    controller.set_private(True, source="test")
    writer.reset()
    controller.apply_legacy("wake")
    controller.apply_legacy("processing")
    time.sleep(0.05)
    assert (REPORT_OFFHOOK_LED, 1) not in writer.flat_writes
    assert (REPORT_RING_LED, 1) not in writer.flat_writes


def test_apply_legacy_unmute_still_routes_while_muted(controller, mic_gate):
    """The mute-overlay guard must NOT block `unmute` — otherwise an
    HA recovery script (`led/set unmute`) could not escape."""
    controller.set_private(True, source="test")
    assert controller.is_private() is True
    controller.apply_legacy("unmute")
    assert controller.is_private() is False
    assert mic_gate.unmute_count == 1


def test_unmute_replays_last_state(controller, writer):
    """Going private then back should re-render the K.1 state we were in."""
    controller.on_state(State.THINKING)
    controller.set_private(True, source="test")
    writer.reset()
    controller.set_private(False, source="test")
    time.sleep(0.05)
    # exit_private clears LEDs then re-renders THINKING → RING on.
    assert (REPORT_RING_LED, 1) in writer.flat_writes


# ---------------------------------------------------------------------------
# §12.3  Cancel visuals (F4)
# ---------------------------------------------------------------------------


def test_hard_cancel_paints_red_overlay(controller, writer):
    writer.reset()
    controller.on_state(State.CANCELLING, cancel_reason="RED_BUTTON_HARD")
    time.sleep(0.05)
    # The cosmetic mute palette is OFFHOOK on + MUTE on.
    assert (REPORT_OFFHOOK_LED, 0x01) in writer.flat_writes
    assert (REPORT_MUTE_LED, 0x01) in writer.flat_writes


def test_soft_cancel_is_silent(controller, writer):
    writer.reset()
    controller.on_state(State.CANCELLING, cancel_reason="RED_BUTTON_SOFT")
    time.sleep(0.05)
    # SOFT should not paint the red overlay.
    assert (REPORT_MUTE_LED, 0x01) not in writer.flat_writes


# ---------------------------------------------------------------------------
# §12.4  Phone button dispatch
# ---------------------------------------------------------------------------


def test_phone_short_press_routes_to_red_button_soft(asyncio_loop, writer, mic_gate):
    cancels: List[str] = []
    actions: List[str] = []
    controller = LedController(
        loop=asyncio_loop,
        mic_capture_mute=mic_gate.mute,
        mic_capture_unmute=mic_gate.unmute,
        writer=writer,
        on_phone_cancel=lambda r: cancels.append(r),
        on_phone_button=lambda a: actions.append(a),
    )
    controller._dispatch_phone_press("short")
    controller._dispatch_phone_press("long")
    asyncio_loop.run_until_complete(asyncio.sleep(0.05))
    assert cancels == ["RED_BUTTON_SOFT", "RED_BUTTON_HARD"]
    assert actions == ["short", "long"]


# ---------------------------------------------------------------------------
# §12.5  HABridge subscribe routing (with fake paho)
# ---------------------------------------------------------------------------


class _Msg:
    def __init__(self, topic: str, payload: bytes) -> None:
        self.topic = topic
        self.payload = payload


def _make_bridge(fake_paho):
    created, factory = fake_paho
    bridge = HABridge(
        room="lounge",
        host="127.0.0.1",
        port=1883,
        client_factory=factory,
    )
    bridge.start()
    return bridge, created[0]


def test_ha_bridge_routes_led_set_to_apply_legacy(fake_paho, asyncio_loop, writer, mic_gate):
    bridge, fake = _make_bridge(fake_paho)
    assert fake.connect_args == ("127.0.0.1", 1883, 60)
    controller = LedController(
        loop=asyncio_loop,
        mic_capture_mute=mic_gate.mute,
        mic_capture_unmute=mic_gate.unmute,
        writer=writer,
    )
    bridge.attach_led_controller(controller)

    writer.reset()
    bridge._on_message(fake, None, _Msg(bridge.led_set_room_topic, b"processing"))
    # apply_legacy("processing") → RING on, OFFHOOK off.
    assert (REPORT_RING_LED, 1) in writer.flat_writes
    assert (REPORT_OFFHOOK_LED, 0) in writer.flat_writes


def test_ha_bridge_routes_volume_set_to_bar(fake_paho, asyncio_loop, writer, mic_gate, monkeypatch):
    bridge, fake = _make_bridge(fake_paho)
    controller = LedController(
        loop=asyncio_loop,
        mic_capture_mute=mic_gate.mute,
        mic_capture_unmute=mic_gate.unmute,
        writer=writer,
    )
    bridge.attach_led_controller(controller)

    applied: List[str] = []
    monkeypatch.setattr(controller.bar, "apply", lambda payload: applied.append(payload) or True)

    bridge._on_message(fake, None, _Msg(bridge.volume_set_topic, b"up"))
    bridge._on_message(fake, None, _Msg(bridge.volume_set_topic, b"50%"))
    assert applied == ["up", "50%"]


def test_ha_bridge_routes_ring_set(fake_paho, asyncio_loop, writer, mic_gate):
    bridge, fake = _make_bridge(fake_paho)
    controller = LedController(
        loop=asyncio_loop,
        mic_capture_mute=mic_gate.mute,
        mic_capture_unmute=mic_gate.unmute,
        writer=writer,
    )
    bridge.attach_led_controller(controller)

    writer.reset()
    bridge._on_message(fake, None, _Msg(bridge.ring_set_topic, b"on"))
    time.sleep(0.05)
    # Ring loop fires OFFHOOK on at least once.
    assert (REPORT_OFFHOOK_LED, 1) in writer.flat_writes
    bridge._on_message(fake, None, _Msg(bridge.ring_set_topic, b"off"))
    # And publish_ring_state should have fired for both transitions.
    topics = [t for t, *_ in fake.publishes]
    assert bridge.ring_state_topic in topics


def test_ha_bridge_routes_mute_set_on(fake_paho, asyncio_loop, writer, mic_gate):
    """M3: calisto/<room>/mute/set = 'on' → controller.set_private(True)."""
    bridge, fake = _make_bridge(fake_paho)
    controller = LedController(
        loop=asyncio_loop,
        mic_capture_mute=mic_gate.mute,
        mic_capture_unmute=mic_gate.unmute,
        writer=writer,
    )
    bridge.attach_led_controller(controller)

    bridge._on_message(fake, None, _Msg(bridge.mute_set_topic, b"on"))
    assert controller.is_private() is True
    assert mic_gate.mute_count == 1


def test_ha_bridge_routes_mute_set_off(fake_paho, asyncio_loop, writer, mic_gate):
    """M3: calisto/<room>/mute/set = 'off' → controller.set_private(False)."""
    bridge, fake = _make_bridge(fake_paho)
    controller = LedController(
        loop=asyncio_loop,
        mic_capture_mute=mic_gate.mute,
        mic_capture_unmute=mic_gate.unmute,
        writer=writer,
    )
    bridge.attach_led_controller(controller)

    controller.set_private(True, source="test")
    bridge._on_message(fake, None, _Msg(bridge.mute_set_topic, b"off"))
    assert controller.is_private() is False
    assert mic_gate.unmute_count == 1


def test_hidraw_mute_press_toggles_set_private(asyncio_loop, writer, mic_gate):
    """When the firmware reports a hardware mute-button press on hidraw
    0x0B 0x01, LedController must toggle the private state. Each
    physical press is a single momentary event (press/release pair
    filtered to press-only by the listener); LVA mirrors firmware's
    internal toggle by flipping `_muted`."""
    controller = LedController(
        loop=asyncio_loop,
        mic_capture_mute=mic_gate.mute,
        mic_capture_unmute=mic_gate.unmute,
        writer=writer,
    )
    # First press → muted.
    controller._dispatch_mute_hidraw_press()
    asyncio_loop.run_until_complete(asyncio.sleep(0))
    assert controller.is_private() is True
    assert mic_gate.mute_count == 1

    # Second press → unmuted.
    controller._dispatch_mute_hidraw_press()
    asyncio_loop.run_until_complete(asyncio.sleep(0))
    assert controller.is_private() is False
    assert mic_gate.unmute_count == 1

    # Third press → muted again.
    controller._dispatch_mute_hidraw_press()
    asyncio_loop.run_until_complete(asyncio.sleep(0))
    assert controller.is_private() is True
    assert mic_gate.mute_count == 2


def test_ha_bridge_routes_mute_set_legacy_verbs(fake_paho, asyncio_loop, writer, mic_gate):
    """M3: accept legacy 'mute'/'unmute' verb payloads for back-compat
    with HA automations that historically used the led/set topic."""
    bridge, fake = _make_bridge(fake_paho)
    controller = LedController(
        loop=asyncio_loop,
        mic_capture_mute=mic_gate.mute,
        mic_capture_unmute=mic_gate.unmute,
        writer=writer,
    )
    bridge.attach_led_controller(controller)

    bridge._on_message(fake, None, _Msg(bridge.mute_set_topic, b"mute"))
    assert controller.is_private() is True
    bridge._on_message(fake, None, _Msg(bridge.mute_set_topic, b"unmute"))
    assert controller.is_private() is False


# ---------------------------------------------------------------------------
# §12.6  HABridge publish helpers
# ---------------------------------------------------------------------------


def test_publish_helpers_use_correct_retain_flag(fake_paho):
    bridge, fake = _make_bridge(fake_paho)
    fake.publishes.clear()

    bridge.publish_volume_state(42)
    bridge.publish_mute_state(True)
    bridge.publish_ring_state(False)
    bridge.publish_phone_button("short")
    bridge.publish_phone_button("long")

    by_topic = {t: (p, retain) for (t, p, _, retain) in fake.publishes}
    assert by_topic[bridge.volume_state_topic] == ("42", True)
    assert by_topic[bridge.mute_state_topic] == ("on", True)
    assert by_topic[bridge.ring_state_topic] == ("off", True)
    # Phone-button events are momentary, must not be retained.
    assert by_topic[bridge.phone_short_topic] == ("press", False)
    assert by_topic[bridge.phone_long_topic] == ("press", False)


# ---------------------------------------------------------------------------
# Regression coverage
# ---------------------------------------------------------------------------


def test_on_connect_subscribes_to_all_back_compat_topics(fake_paho):
    """Regression: the back-compat control topics must be subscribed
    on connect — including M3 mute/set and the E.2 G5 calisto/all/*
    fleet broadcast mirrors."""
    bridge, fake = _make_bridge(fake_paho)
    subscribed_topics = {t for t, _qos in fake.subscriptions}
    assert subscribed_topics == {
        bridge.led_set_room_topic,
        bridge.led_set_all_topic,
        bridge.volume_set_topic,
        bridge.volume_set_all_topic,
        bridge.ring_set_topic,
        bridge.ring_set_all_topic,
        bridge.mute_set_topic,
        bridge.mute_set_all_topic,
        bridge.alarm_set_topic,
        bridge.alarm_set_all_topic,
        bridge.alarm_stop_topic,
        bridge.say_topic,
        bridge.say_all_topic,
        # Stage F5 — consolidated K.3/K.4/K.13 subscriptions.
        bridge.cancel_topic,
        bridge.cancel_all_topic,
        bridge.admin_restart_topic,
        # Stage D — speaker enrollment trigger.
        bridge.enroll_capture_topic,
        # Stage D — SV live tunables (N.2 rows 143 + 153).
        bridge.sv_threshold_set_topic,
        bridge.sv_audible_notify_set_topic,
        # Stage H J1 — fleet-wide wake arbitration (K.5).
        bridge.wake_arb_topic,
    }
    # All at QoS 1 except K.5 wake_arb which is QoS 0 per spec.
    qos_by_topic = dict(fake.subscriptions)
    assert qos_by_topic[bridge.wake_arb_topic] == 0
    other_qos = {qos for t, qos in fake.subscriptions if t != bridge.wake_arb_topic}
    assert other_qos == {1}


def test_ha_bridge_routes_calisto_all_volume_set(fake_paho, asyncio_loop, writer, mic_gate, monkeypatch):
    """G5 regression: a fleet-wide volume command must hit the bar just
    like the per-room equivalent."""
    bridge, fake = _make_bridge(fake_paho)
    controller = LedController(
        loop=asyncio_loop,
        mic_capture_mute=mic_gate.mute,
        mic_capture_unmute=mic_gate.unmute,
        writer=writer,
    )
    apply_calls = []
    monkeypatch.setattr(controller.bar, "apply", lambda p: apply_calls.append(p))
    bridge.attach_led_controller(controller)

    bridge._on_message(fake, None, _Msg(bridge.volume_set_all_topic, b"75%"))

    assert apply_calls == ["75%"]


def test_ha_bridge_routes_calisto_all_mute_set(fake_paho, asyncio_loop, writer, mic_gate, monkeypatch):
    """G5 regression: a fleet-wide mute=on broadcast mutes every device
    that hears it (same routing as the per-room topic)."""
    bridge, fake = _make_bridge(fake_paho)
    controller = LedController(
        loop=asyncio_loop,
        mic_capture_mute=mic_gate.mute,
        mic_capture_unmute=mic_gate.unmute,
        writer=writer,
    )
    set_private_calls = []
    monkeypatch.setattr(
        controller,
        "set_private",
        lambda v, source="": set_private_calls.append((v, source)),
    )
    bridge.attach_led_controller(controller)

    bridge._on_message(fake, None, _Msg(bridge.mute_set_all_topic, b"on"))
    bridge._on_message(fake, None, _Msg(bridge.mute_set_all_topic, b"off"))

    assert set_private_calls == [(True, "mqtt"), (False, "mqtt")]


def test_ha_bridge_routes_calisto_all_ring_set(fake_paho, asyncio_loop, writer, mic_gate, monkeypatch):
    """G5 regression: fleet-wide ring/set hits the ring controller."""
    bridge, fake = _make_bridge(fake_paho)
    controller = LedController(
        loop=asyncio_loop,
        mic_capture_mute=mic_gate.mute,
        mic_capture_unmute=mic_gate.unmute,
        writer=writer,
    )
    started = []
    stopped = []
    monkeypatch.setattr(controller.ring, "start", lambda: started.append(True))
    monkeypatch.setattr(controller.ring, "stop", lambda: stopped.append(True))
    bridge.attach_led_controller(controller)

    bridge._on_message(fake, None, _Msg(bridge.ring_set_all_topic, b"on"))
    bridge._on_message(fake, None, _Msg(bridge.ring_set_all_topic, b"off"))

    assert len(started) == 1
    assert len(stopped) == 1


def test_state_publish_skips_led_mirror_when_controller_attached(
    fake_paho, asyncio_loop, writer, mic_gate
):
    """Regression: with LedController attached, publish_state must NOT
    echo to `calisto/<room>/led/set` — we subscribe to that topic, so
    every K.1 transition would round-trip through `apply_legacy` and
    re-paint the palette out of order."""
    bridge, fake = _make_bridge(fake_paho)
    controller = LedController(
        loop=asyncio_loop,
        mic_capture_mute=mic_gate.mute,
        mic_capture_unmute=mic_gate.unmute,
        writer=writer,
    )
    bridge.attach_led_controller(controller)
    fake.publishes.clear()

    bridge.publish_state(
        state=State.WAKING,
        generation=1,
        session_id="sid-test",
    )
    # K.1 state topic publish present.
    topics = [t for t, *_ in fake.publishes]
    assert bridge.state_topic in topics
    # But NOT the legacy LED control topic — that would echo back.
    assert bridge.led_topic not in topics


def test_state_publish_still_mirrors_led_when_no_controller(fake_paho):
    """Inverse: with no LedController attached (pre-Stage-E.1 deploys),
    the v0 LED mirror must still fire so external calisto-led.service
    instances keep working."""
    bridge, fake = _make_bridge(fake_paho)
    fake.publishes.clear()

    bridge.publish_state(state=State.WAKING, generation=1, session_id="sid")
    topics = [t for t, *_ in fake.publishes]
    assert bridge.led_topic in topics


def test_startup_assertion_clears_full_telephony_state(asyncio_loop, writer, mic_gate):
    """Regression: a mid-mute crash could leave `09 01` + `0A 04 / 0E 01`
    latched in firmware from a Path A era / external write. Startup
    runs the full end-call sequence so the device comes up clean."""
    from linux_voice_assistant.led.hid import (
        CALL_STATE_ENDED,
        REPORT_AUX_INDICATOR,
        REPORT_CALL_STATE,
    )

    controller = LedController(
        loop=asyncio_loop,
        mic_capture_mute=mic_gate.mute,
        mic_capture_unmute=mic_gate.unmute,
        writer=writer,
    )
    writer.reset()
    controller.start()
    try:
        # First writer.write_seq from start() must be the end-call sequence.
        # Order matters: AUX off → CALL_STATE_ENDED → MUTE off → OFFHOOK off → HOLD off.
        assert writer.sequences[0] == [
            (REPORT_AUX_INDICATOR, 0x00),
            (REPORT_CALL_STATE, CALL_STATE_ENDED),
            (REPORT_MUTE_LED, 0x00),
            (REPORT_OFFHOOK_LED, 0x00),
            (REPORT_HOLD_LED, 0x00),
        ]
    finally:
        controller.stop()
