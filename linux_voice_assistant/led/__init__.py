"""Stage E.1 — LED + mute + HID absorbed into LVA.

Replaces the v0 `calisto-led` systemd service. HID writes, hidraw button
reads (vol± / phone short-long), evdev KEY_MICMUTE, telephony-mute wake
sequence, and bidirectional volume sync all live in-process here, owned
by `DeviceSession` via `LedController`.

Public surface is `LedController` — instantiate once at startup, call
`start()` after MicCapture is built, route DeviceSession state
transitions through `on_state()`, and shut down via `stop()`.

Empirical basis: `calisto-led/tools/mute_button_probe_results.md`.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import TYPE_CHECKING, Callable, Optional

from .bar import LedBar
from .evdev import EvdevMuteListener
from .hid import (
    ALL_COSMETIC_REPORTS,
    CALL_STATE_ENDED,
    HidButtonListener,
    HidWriter,
    REPORT_AUX_INDICATOR,
    REPORT_CALL_STATE,
    REPORT_HOLD_LED,
    REPORT_MUTE_LED,
    REPORT_OFFHOOK_LED,
    REPORT_RING_LED,
)
from .mute import LedMute
from .phone import LedPhone
from .ring import LedRing

if TYPE_CHECKING:
    from ..session import State

_LOGGER = logging.getLogger(__name__)


PhoneCancelCallback = Callable[[str], None]
"""Receives `"RED_BUTTON_SOFT"` or `"RED_BUTTON_HARD"` — the K.3 cancel reason."""

PhoneButtonCallback = Callable[[str], None]
"""Receives `"short"` or `"long"` — the raw v0 phone-button action for the
back-compat `calisto/<room>/button/phone/{short,long}` MQTT publish."""

VolumePublishCallback = Callable[[int], None]
"""Receives the new volume percentage — K.11 retained publish hook."""

PrivateToggleCallback = Callable[[], None]
"""Invoked when the hardware mute button is pressed. Caller decides what
"go private" means in the current session context (typically: defer to
`LedController.toggle_private()` from the asyncio loop)."""


# Hard-cancel visual — paint the cosmetic mute palette for this long
# then clear. Visible but transient; matches F4 ("hard all-red"). Uses
# the cosmetic-only path (no audio release) since it's a momentary
# acknowledgement, not a functional mute.
_HARD_CANCEL_FLASH_S = 1.2


class LedController:
    """Facade for the LED + mute + HID subsystem.

    Lifecycle:
        ctrl = LedController(loop=loop, ...)
        ctrl.start()         # spawns listener threads, writes known-clean state
        ...
        ctrl.on_state(State.WAKING)
        ctrl.on_state(State.SPEAKING)
        ctrl.on_state(State.CANCELLING, cancel_reason="RED_BUTTON_HARD")
        ctrl.on_state(State.IDLE)
        ...
        ctrl.stop()

    Mute layer is orthogonal to the K.1 state machine: while muted, the
    bar/mic/phone LEDs stay red regardless of `on_state` calls. The
    underlying state is still tracked, so on unmute the controller
    re-renders whatever state the session has reached in the meantime.

    Path B (2026-05-18) — `set_private` paints the cosmetic mute palette
    (`17 01 + 09 01`) and gates the `MicCapture` frame flow via the
    injected `mic_capture_mute` / `mic_capture_unmute` callables. The
    USB Audio Class claim stays held by `process_audio` throughout, so
    the hardware mute button keeps emitting `KEY_MICMUTE` on evdev and
    wake-word framing stays alive.
    """

    def __init__(
        self,
        *,
        loop: asyncio.AbstractEventLoop,
        mic_capture_mute: Optional[Callable[[], None]] = None,
        mic_capture_unmute: Optional[Callable[[], None]] = None,
        on_phone_cancel: Optional[PhoneCancelCallback] = None,
        on_phone_button: Optional[PhoneButtonCallback] = None,
        on_volume_change: Optional[VolumePublishCallback] = None,
        on_private_toggle: Optional[PrivateToggleCallback] = None,
        on_mute_state_changed: Optional[Callable[[bool], None]] = None,
        writer: Optional[HidWriter] = None,
    ) -> None:
        self._loop = loop
        self._on_phone_cancel = on_phone_cancel or (lambda _reason: None)
        self._on_phone_button = on_phone_button or (lambda _action: None)
        self._on_volume_change = on_volume_change or (lambda _pct: None)
        self._on_private_toggle = on_private_toggle
        self._on_mute_state_changed = on_mute_state_changed or (lambda _muted: None)

        # `writer` is dependency-injected for tests. Production passes
        # nothing and we build the real (hidraw-backed) writer.
        self._writer = writer if writer is not None else HidWriter()
        self.bar = LedBar(on_state_changed=self._dispatch_volume)
        self.phone = LedPhone(self._writer)
        self.ring = LedRing(self._writer)
        # Path B cosmetic mute + MicCapture frame-gate. Always present —
        # there is no fallback path. mic_capture_mute / _unmute can be
        # None for tests that don't wire MicCapture; in that case the
        # mute LEDs paint but mic frames are not gated (the test fakes
        # already prove the gate semantics independently).
        self.mute = LedMute(
            self._writer,
            mic_capture_mute=mic_capture_mute,
            mic_capture_unmute=mic_capture_unmute,
            on_state=self._dispatch_mute_state,
        )

        self._buttons = HidButtonListener(
            on_phone_press=self._dispatch_phone_press,
            on_volume_press=self._dispatch_button_volume,
            on_mute_press=self._dispatch_mute_hidraw_press,
        )
        # Evdev listener remains as defence-in-depth — if a firmware update
        # ever starts surfacing KEY_MICMUTE again (as in the original Test 3
        # measurement), this path will still trigger the in-process toggle.
        # On current firmware in PC Media mode it stays silent — the hidraw
        # `0x0B` press path above is the load-bearing trigger.
        self._mute_listener = EvdevMuteListener(
            on_mute_press=self._dispatch_mute_press,
        )

        self._state_lock = threading.Lock()
        self._muted = False
        self._last_state: Optional["State"] = None
        self._hard_cancel_timer: Optional[threading.Timer] = None

    @property
    def last_hid_event_ts(self) -> float:
        """Monotonic timestamp of the last hidraw read (F2 — heartbeat)."""
        return getattr(self._buttons, "last_event_ts", 0.0)

    @property
    def hidraw_device_present(self) -> bool:
        """True iff the ButtonListener currently has the Calisto hidraw open.
        Heartbeat uses this to flip `hidraw_ok` when the USB phone is
        unplugged but LVA itself keeps publishing heartbeats."""
        return getattr(self._buttons, "is_open", False)

    # ---- lifecycle ----------------------------------------------------

    def start(self) -> None:
        """Write a known-clean LED state and spawn listener threads.

        Startup assertion runs the full Hub end-call HID sequence
        (`0E 00 / 0A 08 / 09 00 / 17 00 / 19 00`) before clearing the
        cosmetic palette. Without the telephony-page clears, a prior
        run that died mid-mute would leave `09 01` and `0A 04 / 0E 01`
        latched in firmware; just clearing OFFHOOK/RING/HOLD wouldn't
        unmute the bar. User chose "always come up unmuted" as the
        crash-survival policy — this is where it's enforced.
        """
        _LOGGER.info("LedController starting")
        # Force-clear firmware state from any prior mid-mute crash.
        # Same byte sequence as `LedMute.exit_private`; safe to write at
        # startup before the recorder has reacquired the audio claim
        # (telephony page accepts writes during that window).
        self._writer.write_seq(
            (REPORT_AUX_INDICATOR, 0x00),
            (REPORT_CALL_STATE, CALL_STATE_ENDED),
            (REPORT_MUTE_LED, 0x00),
            (REPORT_OFFHOOK_LED, 0x00),
            (REPORT_HOLD_LED, 0x00),
            pause=0.05,
        )
        self.phone.off()
        # Publish current volume so the K.11 retained topic is fresh.
        pct = self.bar.read_pct()
        if pct is not None:
            self._dispatch_volume(pct)
        self._buttons.start()
        self._mute_listener.start()

    def stop(self) -> None:
        _LOGGER.info("LedController stopping")
        self._buttons.stop()
        self._mute_listener.stop()
        if self._hard_cancel_timer is not None:
            self._hard_cancel_timer.cancel()
            self._hard_cancel_timer = None
        self.ring.stop()
        self.phone.off()
        # Cosmetic mute palette must also be cleared in case we leave
        # mid-private. Functional unmute (Path A) is owned by LedMute and
        # may not be safe to drive at shutdown; the cosmetic clear keeps
        # the visible LEDs honest.
        self._writer.write_seq(*((r, 0) for r in ALL_COSMETIC_REPORTS), pause=0)

    # ---- K.1 state rendering ------------------------------------------

    def on_state(
        self,
        new_state: "State",
        *,
        cancel_reason: Optional[str] = None,
    ) -> None:
        """Render the visual palette for the given K.1 state.

        Idempotent for repeat calls in the same state — animation
        threads handle re-entry. Frozen when muted (state is recorded
        but not rendered; the visual stays red until unmute).
        """
        from ..session import State  # local import to avoid cycle at module load

        # DIAG (idle-reset hunt 2026-05-23): log entry so we can attribute
        # the 30s phone.off() cycle to either on_state or apply_legacy.
        _LOGGER.debug("on_state called: new_state=%s cancel_reason=%s", new_state, cancel_reason)
        with self._state_lock:
            self._last_state = new_state
            if self._muted:
                _LOGGER.debug(
                    "on_state(%s) suppressed — muted; will render on unmute",
                    new_state,
                )
                return

        if new_state in (State.STARTING, State.IDLE, State.OFFLINE):
            self.phone.off()
        elif new_state in (State.WAKING, State.LISTENING, State.FOLLOWUP):
            self.phone.listening_pulse()
        elif new_state == State.THINKING:
            self.phone.processing_steady()
        elif new_state == State.SPEAKING:
            self.phone.speaking_pulse()
        elif new_state == State.CANCELLING:
            # F4: SOFT silent (no extra LED change — the subsequent IDLE
            # transition clears via `phone.off`). HARD flashes the red
            # cosmetic palette as a transient acknowledgement.
            # Stage D — GATE2_REJECT (speaker verification fail) flashes
            # the same red palette so the user sees the rejection visually
            # alongside the audible chime.
            if cancel_reason in ("RED_BUTTON_HARD", "GATE2_REJECT"):
                self._hard_cancel_overlay()
            # otherwise SOFT — fall through, no LED action here
        elif new_state == State.DEGRADED:
            self.phone.error_hold()
        else:
            _LOGGER.warning("on_state: unknown K.1 state %r — ignored", new_state)

    def apply_legacy(self, payload: str) -> None:
        """Back-compat entry point for `calisto/<room>/led/set` MQTT
        publishes coming from HA or v0 automations.

        The v0 payload vocabulary is:
            off | wake | processing | speaking | complete | error
            mute | unmute

        Voice-cycle terms map to a small synthetic state palette. Mute /
        unmute toggle the private state via the same path the hardware
        button uses.
        """
        # DIAG (idle-reset hunt 2026-05-23): log every MQTT-driven LED command.
        _LOGGER.debug("apply_legacy called: payload=%r", payload)
        normalised = payload.strip().lower()
        if normalised == "mute":
            self.set_private(True, source="mqtt")
            return
        if normalised == "unmute":
            self.set_private(False, source="mqtt")
            return
        # While muted, suppress all other legacy verbs — they would write
        # phone/ring/bar LEDs and dim the red overlay. Same gate as
        # `on_state`: mute is the visual source of truth.
        if self.is_private():
            _LOGGER.debug(
                "apply_legacy(%r) suppressed — muted; mute overlay is source of truth",
                payload,
            )
            return
        if normalised == "off":
            self.phone.off()
        elif normalised == "wake":
            self.phone.listening_pulse()
        elif normalised == "processing":
            self.phone.processing_steady()
        elif normalised == "speaking":
            self.phone.speaking_pulse()
        elif normalised == "complete":
            self.phone.off()
            self.phone.complete_flash()
        elif normalised == "error":
            self.phone.error_hold()
        else:
            _LOGGER.warning("apply_legacy: unknown payload %r", payload)

    # ---- mute / private ----------------------------------------------

    def toggle_private(self, *, source: str = "hardware-button") -> None:
        """Toggle the private (muted) state."""
        self.set_private(not self.is_private(), source=source)

    def set_private(self, want: bool, *, source: str) -> bool:
        """Enter or exit private mode. Returns True on success.

        Path B: paints the cosmetic red palette (`17 01 + 09 01` on
        enter, `09 00 + 17 00` on exit) and gates the MicCapture frame
        flow via the injected callables. The USB Audio Class claim is
        NOT released — the hardware mute button + wake-word listener
        stay alive throughout. See [[calisto-cosmetic-vs-functional-
        mute]] for the boundary.
        """
        with self._state_lock:
            if want == self._muted:
                _LOGGER.info(
                    "set_private(%s, source=%s) — already in that state",
                    want,
                    source,
                )
                return True

        _LOGGER.info("private %s (source=%s)", "ON" if want else "OFF", source)

        # Suppress any in-flight cosmetic state palette before painting
        # red — on_state suppression-while-muted prevents future writes
        # but in-flight pulse threads (LedPhone.listening_pulse) keep
        # writing their own bytes until told to stop.
        if want:
            self.phone.off()

        ok = self.mute.enter_private() if want else self.mute.exit_private()
        with self._state_lock:
            self._muted = want if ok else not want
        if not ok:
            _LOGGER.error(
                "set_private(%s): LedMute transition failed; staying %s",
                want,
                "muted" if self._muted else "unmuted",
            )
        if not want and ok:
            self._render_last_state()
        return ok

    def is_private(self) -> bool:
        with self._state_lock:
            return self._muted

    def _render_last_state(self) -> None:
        """Re-apply the most recently received K.1 state — used on
        unmute so visuals catch up to whatever the session is doing now."""
        with self._state_lock:
            state = self._last_state
        if state is not None:
            self.on_state(state)

    # ---- F4 hard-cancel visual ---------------------------------------

    def _hard_cancel_overlay(self) -> None:
        """Paint the cosmetic mute palette for `_HARD_CANCEL_FLASH_S`,
        then clear. Non-blocking — runs on a Timer."""
        if self._hard_cancel_timer is not None:
            self._hard_cancel_timer.cancel()
        # Stop any voice-cycle animation so the red doesn't fight a pulse.
        self.phone.off()
        self._writer.write_seq(
            (REPORT_OFFHOOK_LED, 0x01),
            (REPORT_MUTE_LED, 0x01),
            pause=0.05,
        )

        def clear() -> None:
            # If a transition has muted us in the meantime, leave red on.
            if self.is_private():
                return
            self._writer.write_seq(
                (REPORT_MUTE_LED, 0x00),
                (REPORT_OFFHOOK_LED, 0x00),
                pause=0.05,
            )
            self._render_last_state()

        self._hard_cancel_timer = threading.Timer(_HARD_CANCEL_FLASH_S, clear)
        self._hard_cancel_timer.daemon = True
        self._hard_cancel_timer.start()

    # ---- listener dispatch (always invoked from listener threads) -----

    def _dispatch_phone_press(self, action: str) -> None:
        # K.3 vocabulary uses RED_BUTTON_SOFT / _HARD.
        k3 = "RED_BUTTON_HARD" if action == "long" else "RED_BUTTON_SOFT"
        _LOGGER.info("phone-button %s (k3=%s)", action, k3)
        self._loop.call_soon_threadsafe(self._on_phone_cancel, k3)
        self._loop.call_soon_threadsafe(self._on_phone_button, action)

    def _dispatch_button_volume(self, direction: str) -> None:
        # Apply synchronously on the listener thread — amixer is local
        # subprocess, the volume bar callback fires after.
        self.bar.apply(direction)

    def _dispatch_mute_press(self) -> None:
        cb = self._on_private_toggle
        if cb is None:
            # No external orchestrator wired — toggle in-process.
            self._loop.call_soon_threadsafe(self.toggle_private)
            return
        self._loop.call_soon_threadsafe(cb)

    def _dispatch_mute_hidraw_press(self) -> None:
        """Hardware mute-button press observed via hidraw `0x0B 0x01`.
        Each physical press toggles firmware's internal mute state
        (firmware does the functional mute itself); we mirror by
        toggling LVA's `_muted` flag and re-driving the cosmetic +
        MicCapture-gate path so HA / MQTT stay in sync.

        Press/release are emitted as a pair by the firmware — the
        listener already filters to press-only with debounce, so each
        call here corresponds to a real button press.
        """
        # call_soon_threadsafe takes positional args only — wrap in a
        # closure so we can pass the `source` keyword.
        self._loop.call_soon_threadsafe(
            lambda: self.set_private(
                not self.is_private(),
                source="hardware-button",
            )
        )

    def _dispatch_volume(self, pct: int) -> None:
        # bar callback fires from whichever thread called bar.apply().
        # The publish hook is generally cheap and thread-safe, but route
        # through the loop so MQTT publishes don't race with each other.
        try:
            self._loop.call_soon_threadsafe(self._on_volume_change, pct)
        except RuntimeError:
            # Loop closed during shutdown — best-effort.
            pass

    def _dispatch_mute_state(self, muted: bool) -> None:
        """LedMute fires this after every transition (including converge-
        to-unmuted). Forwards to the caller-supplied mirror so MQTT /
        `ServerState.muted` stay in sync. Routed through the loop because
        LedMute runs on the listener thread for hardware-button toggles."""
        try:
            self._loop.call_soon_threadsafe(self._on_mute_state_changed, muted)
        except RuntimeError:
            pass


__all__ = ["LedController", "PhoneCancelCallback", "VolumePublishCallback", "PrivateToggleCallback"]
