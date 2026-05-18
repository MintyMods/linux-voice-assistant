"""Stage E.1 — integration tests for LED + mute + HID + back-compat MQTT.

Covers the absorbed subsystem end-to-end without touching real hardware:

  * `FakeHidWriter` replaces `linux_voice_assistant.led.hid.HidWriter` —
    records every byte sequence + write_one in order.
  * A daemon "recorder simulator" drives `AudioControl` like the real
    `process_audio` loop: polls `is_pause_desired`, confirms paused on
    request, waits for resume, confirms resumed.
  * `HABridge` is wired with the `fake_paho` factory from
    `tests/conftest.py` so the MQTT subscribe/publish path runs without
    a real broker.

Scope per Stage E.1 plan §12:
  1. `enter_private` performs the full Path A handshake (pause →
     settle → wake seq → 09 01) in the right order.
  2. `exit_private` runs the canonical end-call sequence then resumes.
  3. K.1 state transitions render the correct cosmetic palette.
  4. Muted state suppresses subsequent `on_state` rendering.
  5. Inbound MQTT (`calisto/<room>/led/set`, `volume/set`, `ring/set`)
     routes to the right controller method.
  6. Outbound publishers (`mute/state`, `volume/state`, phone-button)
     fire with the right retain flag.

The recorder simulator and write logs let assertions reach the actual
sequence the firmware would receive on a real device — see
`mute_button_probe_results.md` for why these bytes were chosen.
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any, List, Optional, Tuple

import pytest

from linux_voice_assistant.audio_control import AudioControl
from linux_voice_assistant.ha_bridge import HABridge
from linux_voice_assistant.led import LedController
from linux_voice_assistant.led import mute as mute_mod
from linux_voice_assistant.led.hid import (
    CALL_STATE_ACTIVE,
    CALL_STATE_ENDED,
    CALL_STATE_IN_CALL,
    REPORT_AUX_INDICATOR,
    REPORT_CALL_STATE,
    REPORT_HOLD_LED,
    REPORT_MUTE_LED,
    REPORT_OFFHOOK_LED,
    REPORT_PULSE,
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


class RecorderSimulator:
    """Models `process_audio`'s pause/resume cooperation."""

    def __init__(self, ac: AudioControl) -> None:
        self._ac = ac
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        # Track lifecycle for assertions.
        self.pause_count = 0
        self.resume_count = 0

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            if self._ac.is_pause_desired():
                self.pause_count += 1
                self._ac.confirm_paused()
                self._ac.wait_for_resume()
                self.resume_count += 1
                self._ac.confirm_resumed()
            time.sleep(0.005)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fast_mute(monkeypatch):
    """Shrink the mute-path sleeps so tests run in ~50ms instead of ~1.2s."""
    monkeypatch.setattr(mute_mod, "_STREAM_CLOSE_WAIT_S", 0.005)
    monkeypatch.setattr(mute_mod, "_WAKE_SEQ_SETTLE_S", 0.005)


@pytest.fixture
def asyncio_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def audio_control():
    return AudioControl()


@pytest.fixture
def recorder(audio_control):
    sim = RecorderSimulator(audio_control)
    sim.start()
    yield sim
    sim.stop()


@pytest.fixture
def writer():
    return FakeHidWriter()


@pytest.fixture
def controller(asyncio_loop, audio_control, recorder, writer, fast_mute):
    """LedController wired with fakes everywhere."""
    return LedController(
        loop=asyncio_loop,
        audio_control=audio_control,
        writer=writer,
    )


# ---------------------------------------------------------------------------
# §12.1  enter_private / exit_private handshake
# ---------------------------------------------------------------------------


def test_enter_private_runs_full_path_a_sequence(controller, writer, recorder):
    """Path A: pause → settle → wake seq groups → 09 01 last."""
    assert controller.set_private(True, source="test") is True
    assert controller.is_private() is True
    assert recorder.pause_count == 1

    # The wake sequence is split into three writer.write_seq calls in mute.py.
    # Extract them after the startup `phone.off()` writes (which use write_one
    # under the hood and land as individual single-tuple sequences).
    seqs = [s for s in writer.sequences if any(t[0] in {
        REPORT_OFFHOOK_LED, REPORT_CALL_STATE, REPORT_PULSE,
        REPORT_AUX_INDICATOR, REPORT_MUTE_LED,
    } and len(t) == 2 and t[1] != 0 for t in s)]
    # Sequence 1: OFFHOOK on + CALL_STATE active + PULSE 1 + PULSE 0
    # Sequence 2: CALL_STATE in-call + AUX on
    # Sequence 3: MUTE_LED on (the functional mute)
    by_first = [s[0] for s in seqs]
    assert (REPORT_OFFHOOK_LED, 0x01) in by_first or any(
        t == (REPORT_OFFHOOK_LED, 0x01) for s in seqs for t in s
    )

    # The final HID write must be MUTE_LED on — that's what physically mutes
    # the mic at the firmware level.
    last_writes = writer.flat_writes[-3:]
    assert (REPORT_MUTE_LED, 0x01) in last_writes, (
        f"Expected MUTE_LED=1 in last 3 writes, got {last_writes}"
    )

    # Wake sequence order check: OFFHOOK 0x01 must precede CALL_STATE ACTIVE
    # which must precede CALL_STATE IN_CALL which must precede MUTE_LED 0x01.
    order = []
    for w in writer.flat_writes:
        if w == (REPORT_OFFHOOK_LED, 0x01):
            order.append("offhook_on")
        elif w == (REPORT_CALL_STATE, CALL_STATE_ACTIVE):
            order.append("call_active")
        elif w == (REPORT_CALL_STATE, CALL_STATE_IN_CALL):
            order.append("call_in")
        elif w == (REPORT_AUX_INDICATOR, 0x01):
            order.append("aux_on")
        elif w == (REPORT_MUTE_LED, 0x01):
            order.append("mute_on")
    assert order == ["offhook_on", "call_active", "call_in", "aux_on", "mute_on"], (
        f"Wake-sequence order wrong: {order}"
    )


def test_exit_private_runs_end_call_then_resumes(controller, writer, recorder):
    # Enter, then clear writer log to focus on exit.
    controller.set_private(True, source="test")
    assert recorder.pause_count == 1
    writer.reset()

    assert controller.set_private(False, source="test") is True
    assert controller.is_private() is False
    assert recorder.resume_count == 1

    # End-call sequence — order matters: AUX off → CALL_STATE_ENDED →
    # MUTE_LED off → OFFHOOK off → HOLD off, all before resume.
    seq = writer.sequences[0]  # mute.exit_private writes one big sequence
    assert seq == [
        (REPORT_AUX_INDICATOR, 0x00),
        (REPORT_CALL_STATE, CALL_STATE_ENDED),
        (REPORT_MUTE_LED, 0x00),
        (REPORT_OFFHOOK_LED, 0x00),
        (REPORT_HOLD_LED, 0x00),
    ]


def test_pause_timeout_converges_to_unmuted(asyncio_loop, audio_control, writer, fast_mute):
    """If the recorder never confirms pause, LedMute must converge to
    unmuted (clear LEDs + try to force resume) and return False."""
    # No RecorderSimulator — audio_control will never see confirm_paused.
    controller = LedController(
        loop=asyncio_loop,
        audio_control=audio_control,
        writer=writer,
    )
    # Shrink the LedMute timeout so this test is fast.
    import linux_voice_assistant.led.mute as mm
    saved_to = mm._PAUSE_TIMEOUT_S
    mm._PAUSE_TIMEOUT_S = 0.05
    try:
        assert controller.set_private(True, source="test") is False
    finally:
        mm._PAUSE_TIMEOUT_S = saved_to
    # _muted should have been reverted to False.
    assert controller.is_private() is False


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


def test_muted_suppresses_on_state_rendering(controller, writer, recorder):
    controller.set_private(True, source="test")
    writer.reset()
    # While muted, on_state(WAKING) should not paint cosmetic OFFHOOK pulse.
    controller.on_state(State.WAKING)
    time.sleep(0.05)
    assert (REPORT_OFFHOOK_LED, 1) not in writer.flat_writes


def test_unmute_replays_last_state(controller, writer, recorder):
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


def test_phone_short_press_routes_to_red_button_soft(asyncio_loop, audio_control, recorder, writer, fast_mute):
    cancels: List[str] = []
    actions: List[str] = []
    controller = LedController(
        loop=asyncio_loop,
        audio_control=audio_control,
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


def test_ha_bridge_routes_led_set_to_apply_legacy(fake_paho, asyncio_loop, audio_control, recorder, writer, fast_mute):
    created, factory = fake_paho
    bridge = HABridge(
        room="lounge",
        host="127.0.0.1",
        port=1883,
        client_factory=factory,
    )
    bridge.start()
    fake = created[0]
    assert fake.connect_args == ("127.0.0.1", 1883, 60)
    # Subscribe call recorded via the on_connect path — but FakeMqttClient
    # doesn't implement subscribe(), so we just verify routing works by
    # invoking _on_message directly.
    controller = LedController(
        loop=asyncio_loop, audio_control=audio_control, writer=writer
    )
    bridge.attach_led_controller(controller)

    class _Msg:
        def __init__(self, topic, payload):
            self.topic = topic
            self.payload = payload

    writer.reset()
    bridge._on_message(fake, None, _Msg(bridge.led_set_room_topic, b"processing"))
    # apply_legacy("processing") → RING on, OFFHOOK off.
    assert (REPORT_RING_LED, 1) in writer.flat_writes
    assert (REPORT_OFFHOOK_LED, 0) in writer.flat_writes


def test_ha_bridge_routes_volume_set_to_bar(fake_paho, asyncio_loop, audio_control, recorder, writer, fast_mute, monkeypatch):
    created, factory = fake_paho
    bridge = HABridge(room="lounge", host="127.0.0.1", port=1883, client_factory=factory)
    bridge.start()
    fake = created[0]
    controller = LedController(
        loop=asyncio_loop, audio_control=audio_control, writer=writer
    )
    bridge.attach_led_controller(controller)

    # Mock LedBar.apply so we don't shell out to amixer.
    applied: List[str] = []
    monkeypatch.setattr(controller.bar, "apply", lambda payload: applied.append(payload) or True)

    class _Msg:
        def __init__(self, topic, payload):
            self.topic = topic
            self.payload = payload

    bridge._on_message(fake, None, _Msg(bridge.volume_set_topic, b"up"))
    bridge._on_message(fake, None, _Msg(bridge.volume_set_topic, b"50%"))
    assert applied == ["up", "50%"]


def test_ha_bridge_routes_ring_set(fake_paho, asyncio_loop, audio_control, recorder, writer, fast_mute):
    created, factory = fake_paho
    bridge = HABridge(room="lounge", host="127.0.0.1", port=1883, client_factory=factory)
    bridge.start()
    fake = created[0]
    controller = LedController(
        loop=asyncio_loop, audio_control=audio_control, writer=writer
    )
    bridge.attach_led_controller(controller)

    class _Msg:
        def __init__(self, topic, payload):
            self.topic = topic
            self.payload = payload

    writer.reset()
    bridge._on_message(fake, None, _Msg(bridge.ring_set_topic, b"on"))
    time.sleep(0.05)
    # Ring loop fires OFFHOOK on at least once.
    assert (REPORT_OFFHOOK_LED, 1) in writer.flat_writes
    bridge._on_message(fake, None, _Msg(bridge.ring_set_topic, b"off"))
    # And publish_ring_state should have fired for both transitions.
    topics = [t for t, *_ in fake.publishes]
    assert bridge.ring_state_topic in topics


# ---------------------------------------------------------------------------
# §12.6  HABridge publish helpers
# ---------------------------------------------------------------------------


def test_publish_helpers_use_correct_retain_flag(fake_paho):
    created, factory = fake_paho
    bridge = HABridge(room="lounge", host="127.0.0.1", port=1883, client_factory=factory)
    bridge.start()
    fake = created[0]
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
# Advisor regressions — both caught in the dry-run review pass.
# ---------------------------------------------------------------------------


def test_on_connect_subscribes_to_all_back_compat_topics(fake_paho):
    """Regression: the four back-compat control topics must be subscribed
    on connect. Caught when subscribe() was silently AttributeError-ing
    on FakeMqttClient and the try/except in HABridge swallowed it."""
    created, factory = fake_paho
    bridge = HABridge(room="lounge", host="127.0.0.1", port=1883, client_factory=factory)
    bridge.start()
    fake = created[0]
    subscribed_topics = {t for t, _qos in fake.subscriptions}
    assert subscribed_topics == {
        bridge.led_set_room_topic,
        bridge.led_set_all_topic,
        bridge.volume_set_topic,
        bridge.ring_set_topic,
    }
    # All at QoS 1.
    qos_values = {qos for _t, qos in fake.subscriptions}
    assert qos_values == {1}


def test_state_publish_skips_led_mirror_when_controller_attached(
    fake_paho, asyncio_loop, audio_control, recorder, writer, fast_mute
):
    """Regression: with LedController attached, publish_state must NOT
    echo to `calisto/<room>/led/set` — we subscribe to that topic, so
    every K.1 transition would round-trip through `apply_legacy` and
    re-paint the palette out of order. Caught by advisor review."""
    created, factory = fake_paho
    bridge = HABridge(room="lounge", host="127.0.0.1", port=1883, client_factory=factory)
    bridge.start()
    fake = created[0]
    controller = LedController(
        loop=asyncio_loop, audio_control=audio_control, writer=writer
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
    created, factory = fake_paho
    bridge = HABridge(room="lounge", host="127.0.0.1", port=1883, client_factory=factory)
    bridge.start()
    fake = created[0]
    fake.publishes.clear()

    bridge.publish_state(state=State.WAKING, generation=1, session_id="sid")
    topics = [t for t, *_ in fake.publishes]
    assert bridge.led_topic in topics


def test_startup_assertion_clears_full_telephony_state(
    asyncio_loop, audio_control, recorder, writer, fast_mute
):
    """Regression: a mid-mute crash leaves `09 01` + `0A 04 / 0E 01`
    latched in firmware. Cosmetic-only `phone.off()` doesn't clear those.
    Startup must run the full end-call sequence."""
    controller = LedController(
        loop=asyncio_loop, audio_control=audio_control, writer=writer
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
