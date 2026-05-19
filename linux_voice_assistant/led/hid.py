"""Stage E.1 — HID layer: hidraw discovery, locked writer, button listener.

Ports the hidraw side of `calisto-led/led_service.py`:

  * `find_calisto_hidraw()` — locate the Calisto P7200 hidraw node by
    VID/PID match against the sysfs `uevent`.
  * `HidWriter` — thread-safe multi-payload write. All LED state changes
    funnel through `write_seq()` which holds a lock so concurrent writes
    from animation threads can't interleave their byte sequences.
  * `HidButtonListener` — daemon thread that reads `0x02 0xNN` reports
    off the hidraw and dispatches:
        - phone press/release with short/long classification (>=500 ms
          held = long → RED_BUTTON_HARD; otherwise → RED_BUTTON_SOFT)
        - vol± press → bar delta callback
        - sporadic `0x0B` (mute) → ignored; mute trigger is on evdev
          (see `led/evdev.py` and the Stage E.1 evidence pack).

Both classes use an outer reconnect loop — USB re-enumerates on every
LVA cycle / mute toggle / kernel hidraw re-bind, and a single blocking
read can't survive that without a re-open. See [[hid-listener-must-
reconnect]] and v0 commit history for the failure modes.

Callbacks are intentionally semantic and free of MQTT / DeviceSession
coupling — `LedController` wires them to the asyncio loop via
`loop.call_soon_threadsafe`.
"""

from __future__ import annotations

import glob
import logging
import os
import threading
import time
from typing import Callable, Iterable, Optional, Sequence

_LOGGER = logging.getLogger(__name__)

# Calisto P7200 — Plantronics VID / PID. Matched case-insensitively against
# the sysfs `uevent` HID_ID line; sysfs reports them upper-case.
_CALISTO_VID = "047F"
_CALISTO_PID = "1200"

# HID LED page — cosmetic, accepted while the USB Audio Class stream is
# held open (PC Media mode). See `calisto_hid_protocol.md §6b` and
# `mute_button_probe_results.md` cosmetic-trial section.
REPORT_OFFHOOK_LED = 0x17
REPORT_RING_LED = 0x18
REPORT_HOLD_LED = 0x19
REPORT_MUTE_LED = 0x09

# HID telephony page — only effective once the audio stream is released
# (Path A audio-claim dance, driven by `led/mute.py`).
REPORT_CALL_STATE = 0x0A
REPORT_AUX_INDICATOR = 0x0E
REPORT_PULSE = 0x46

CALL_STATE_ACTIVE = 0x01
CALL_STATE_IN_CALL = 0x04
CALL_STATE_ENDED = 0x08

ALL_COSMETIC_REPORTS = (
    REPORT_OFFHOOK_LED,
    REPORT_RING_LED,
    REPORT_HOLD_LED,
    REPORT_MUTE_LED,
)

# Button-input report layout for `0x02 0xNN`:
_BTN_REPORT_ID = 0x02
_BTN_RELEASE = 0x00
_BTN_VOL_UP = 0x02
_BTN_VOL_DOWN = 0x04
_BTN_PHONE = 0x80

# Firmware mute-button report (`0x0b 0xNN`). Empirically confirmed
# 2026-05-18 on the lounge unit: when the hardware mute button is
# pressed in PC Media mode, the Calisto firmware handles the mute
# itself (paints LEDs red + clips mic audio + beeps) and emits a
# press/release PAIR on hidraw 0x0B — `KEY_MICMUTE` on evdev does
# NOT fire. The hidraw probe captured each physical press as
# `0x0b 0x01` (press) immediately followed by `0x0b 0x00` (release).
#
# Treat `0x0b 0x01` as a single momentary press event (button-click
# semantics, NOT an absolute state); ignore the release. Each press
# toggles the LVA mute state. Debounced at 0.5 s to absorb the
# membrane bounce.
_MUTE_REPORT_ID = 0x0B
_MUTE_PRESS = 0x01
_MUTE_RELEASE = 0x00
_MUTE_BUTTON_DEBOUNCE_S = 0.5

# >= 500 ms held = "long" press → RED_BUTTON_HARD; < 500 ms = "short" →
# RED_BUTTON_SOFT. v0 constant; kept verbatim to preserve calibration.
PHONE_LONG_PRESS_MS = 500

# Outer reconnect cadence — when the device disappears (USB re-enum, LVA
# cycle), retry the open at this interval. Matches v0; 5 s is comfortable
# below the longest LVA restart window without busy-looping.
_RECONNECT_BACKOFF_S = 5.0


PhonePressCallback = Callable[[str], None]
"""Receives `"short"` or `"long"`. Invoked from the listener thread."""

MutePressCallback = Callable[[], None]
"""Invoked once per debounced hardware-mute-button press, observed via
the hidraw `0x0B 0x01` report. Caller decides the semantics — typically
a state toggle. Invoked from the listener thread."""

VolumePressCallback = Callable[[str], None]
"""Receives `"up"` or `"down"`. Invoked from the listener thread."""


def find_calisto_hidraw() -> Optional[str]:
    """Return `/dev/hidrawN` of the Calisto P7200, or None if not present.

    Scans `/sys/class/hidraw/hidraw*/device/uevent` for a HID_ID line
    containing both the VID and PID. Returns the first match by sorted
    path order (deterministic across reboots given udev's enumeration).
    """
    for sysfs_path in sorted(glob.glob("/sys/class/hidraw/hidraw*")):
        uevent_path = os.path.join(sysfs_path, "device", "uevent")
        try:
            with open(uevent_path) as fh:
                uevent = fh.read()
        except OSError:
            continue
        if "HID_ID=" not in uevent:
            continue
        upper = uevent.upper()
        if _CALISTO_VID in upper and _CALISTO_PID in upper:
            return f"/dev/{os.path.basename(sysfs_path)}"
    return None


class HidWriter:
    """Locked, multi-payload writer for the Calisto hidraw node.

    Every LED state change goes through `write_seq`. The lock prevents
    interleaving between concurrent animation threads (listening pulse,
    speaking pulse, ring loop) and between animations and one-shot
    transitions (mute on, complete-flash, error-hold).

    Each call re-discovers the hidraw path and opens fresh — matches the
    v0 `_hid_seq` pattern and survives USB re-enumeration without holding
    a stale fd. Open-write-close per sequence is ~ms-cheap and not on a
    hot loop.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()

    def write_seq(
        self,
        *payloads: Sequence[int],
        pause: float = 0.05,
    ) -> bool:
        """Write each payload to the Calisto hidraw, atomically vs. other
        writers. Returns True on success.

        Returns False (and logs) if the device is absent or the open/write
        fails — caller decides whether to propagate or swallow. State
        machines should treat False as "transition not applied; retry on
        next reconciliation tick".
        """
        if not payloads:
            return True
        dev = find_calisto_hidraw()
        if dev is None:
            _LOGGER.warning(
                "Calisto hidraw not present; skipping %d-payload sequence",
                len(payloads),
            )
            return False
        with self._lock:
            try:
                with open(dev, "wb", buffering=0) as fh:
                    for p in payloads:
                        fh.write(bytes(p))
                        if pause:
                            time.sleep(pause)
            except OSError as exc:
                _LOGGER.error("HID write to %s failed: %s", dev, exc)
                return False
        return True

    def write_one(self, report_id: int, on: bool) -> bool:
        """Convenience: write a single 2-byte `(report_id, 0 or 1)`."""
        return self.write_seq((report_id, 1 if on else 0), pause=0)

    def all_cosmetic_off(self) -> bool:
        """Clear every cosmetic LED. Used on startup, idle, and after
        cancel-soft to reach a known-clean visual state."""
        return self.write_seq(
            *((r, 0) for r in ALL_COSMETIC_REPORTS),
            pause=0,
        )


class HidButtonListener:
    """Daemon thread reading `0x02 0xNN` reports off the Calisto hidraw.

    Dispatches semantic events via callbacks supplied at construction
    time. The listener is intentionally protocol-bound and free of any
    higher-level coupling — `LedController` wires the callbacks to the
    asyncio loop via `loop.call_soon_threadsafe`.

    Outer reconnect: on OSError or device-absent, the loop sleeps
    `_RECONNECT_BACKOFF_S` and tries again. A press in flight at the time
    of reconnect is dropped (the per-connection `phone_press_at` is
    reset).
    """

    def __init__(
        self,
        *,
        on_phone_press: PhonePressCallback,
        on_volume_press: VolumePressCallback,
        on_mute_press: Optional[MutePressCallback] = None,
        stop_event: Optional[threading.Event] = None,
    ) -> None:
        self._on_phone_press = on_phone_press
        self._on_volume_press = on_volume_press
        self._on_mute_press = on_mute_press
        self._stop = stop_event or threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_mute_press_at = 0.0
        # Stage F2 — last successful hidraw read timestamp (monotonic).
        # Heartbeat reads this to assert hidraw_ok (event seen <5s ago).
        self.last_event_ts: float = 0.0

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="calisto-hid-buttons",
            daemon=True,
        )
        self._thread.start()

    def stop(self, *, join_timeout: float = 2.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=join_timeout)
            self._thread = None

    def _run(self) -> None:
        while not self._stop.is_set():
            dev = find_calisto_hidraw()
            if dev is None:
                _LOGGER.warning(
                    "hidraw button listener: device not found, retrying in %.0fs",
                    _RECONNECT_BACKOFF_S,
                )
                if self._stop.wait(_RECONNECT_BACKOFF_S):
                    return
                continue
            _LOGGER.info("hidraw button listener: opening %s", dev)
            try:
                fh = open(dev, "rb", buffering=0)
            except OSError as exc:
                _LOGGER.error(
                    "hidraw open %s failed: %s — retry in %.0fs",
                    dev,
                    exc,
                    _RECONNECT_BACKOFF_S,
                )
                if self._stop.wait(_RECONNECT_BACKOFF_S):
                    return
                continue
            try:
                self._read_loop(fh)
            finally:
                try:
                    fh.close()
                except OSError:
                    pass
            # USB likely just re-enumerated. Short pause before re-opening.
            if self._stop.wait(2.0):
                return

    def _read_loop(self, fh) -> None:
        """Per-connection read loop. Returns on OSError or stop signal so
        the outer loop can reconnect; resets press state on entry so a
        press straddling a reconnect doesn't survive."""
        phone_press_at: Optional[float] = None
        while not self._stop.is_set():
            try:
                buf = fh.read(64)
            except OSError as exc:
                _LOGGER.error("hidraw read failed: %s — reopening", exc)
                return
            if not buf or len(buf) < 2:
                continue
            # Stage F2 — heartbeat-side liveness probe.
            self.last_event_ts = time.monotonic()
            if buf[0] == _MUTE_REPORT_ID:
                # 0x0b 0x01 = press; 0x0b 0x00 = release. Each physical
                # press fires both rapidly — only act on the press edge.
                if self._on_mute_press is None or buf[1] != _MUTE_PRESS:
                    continue
                now = time.monotonic()
                if now - self._last_mute_press_at < _MUTE_BUTTON_DEBOUNCE_S:
                    _LOGGER.debug("hardware mute button: debounced")
                    continue
                self._last_mute_press_at = now
                _LOGGER.info("hardware mute button pressed (hidraw 0x0B)")
                self._safe_invoke_nullary(self._on_mute_press)
                continue
            if buf[0] != _BTN_REPORT_ID:
                # Other report (e.g. 0x07 LED state echo) — ignore.
                continue
            code = buf[1]
            if code == _BTN_PHONE:
                if phone_press_at is None:
                    phone_press_at = time.monotonic()
                    _LOGGER.info("phone button press detected")
            elif code == _BTN_RELEASE and phone_press_at is not None:
                held_ms = int((time.monotonic() - phone_press_at) * 1000)
                phone_press_at = None
                action = "long" if held_ms >= PHONE_LONG_PRESS_MS else "short"
                _LOGGER.info(
                    "phone button released after %dms (%s)", held_ms, action
                )
                self._safe_invoke(self._on_phone_press, action)
            elif code in (_BTN_VOL_UP, _BTN_VOL_DOWN):
                direction = "up" if code == _BTN_VOL_UP else "down"
                _LOGGER.info("vol-%s button press detected", direction)
                self._safe_invoke(self._on_volume_press, direction)
            # else: trailing release with no in-flight press, or unknown
            # code — silently ignore.

    @staticmethod
    def _safe_invoke(cb: Callable[[str], None], arg: str) -> None:
        try:
            cb(arg)
        except Exception:
            _LOGGER.exception("hid listener callback raised")

    @staticmethod
    def _safe_invoke_nullary(cb: Callable[[], None]) -> None:
        try:
            cb()
        except Exception:
            _LOGGER.exception("hid listener mute-press callback raised")


__all__ = [
    "REPORT_OFFHOOK_LED",
    "REPORT_RING_LED",
    "REPORT_HOLD_LED",
    "REPORT_MUTE_LED",
    "REPORT_CALL_STATE",
    "REPORT_AUX_INDICATOR",
    "REPORT_PULSE",
    "CALL_STATE_ACTIVE",
    "CALL_STATE_IN_CALL",
    "CALL_STATE_ENDED",
    "ALL_COSMETIC_REPORTS",
    "PHONE_LONG_PRESS_MS",
    "find_calisto_hidraw",
    "HidWriter",
    "HidButtonListener",
    "PhonePressCallback",
    "VolumePressCallback",
]
