"""Stage E.1 — `LedPhone`: cosmetic LED animations for voice-cycle states.

Ports the animation primitives from `calisto-led/led_service.py:LedDriver`:

  * `listening_pulse()` — slow even pulse on OFFHOOK_LED (0.55s / 0.55s).
    Used while the wake word is firing and the user is being recorded.
  * `speaking_pulse()` — asymmetric pulse on RING_LED (0.9s on / 0.3s off).
    Used while LVA is speaking — long-on weighted to read as "active".
  * `processing_steady()` — RING_LED on, OFFHOOK_LED off. Used while the
    bridge is generating a reply.
  * `complete_flash()` — two quick green flashes on RING_LED (~120ms /
    120ms × 2), independent of the pulse track so a complete event can
    overlay without re-cancelling pulse logic.
  * `error_hold()` — HOLD_LED on for 2 s then auto-off.
  * `off()` — all cosmetic LEDs off.

Threading model matches v0: two cancel-event tracks (pulse vs. flash) so
they can be controlled independently. Each animation method cancels its
own track before launching a fresh thread, so concurrent state changes
converge cleanly to the latest call.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

from .hid import (
    HidWriter,
    REPORT_HOLD_LED,
    REPORT_OFFHOOK_LED,
    REPORT_RING_LED,
)

_LOGGER = logging.getLogger(__name__)

# Pulse timings. Listening = symmetric, slow — reads as "I'm listening,
# you have time". Speaking = long-on / short-off — reads as continuous
# activity. Constants kept verbatim from v0; tweaks belong in Stage F.
_LISTENING_ON_S = 0.55
_LISTENING_OFF_S = 0.55
_SPEAKING_ON_S = 0.9
_SPEAKING_OFF_S = 0.3

# Complete-flash: two quick green flashes, ~240 ms total.
_FLASH_ON_S = 0.120
_FLASH_OFF_S = 0.120
_FLASH_COUNT = 2

# Error-hold: amber HOLD LED on for 2 s.
_ERROR_HOLD_S = 2.0

# Join timeout for cancelling a previous animation. Long enough that a
# pending `time.sleep()` inside the thread can wake from a `wait()` and
# exit cleanly.
_ANIM_JOIN_S = 1.0


class LedPhone:
    """Cosmetic LED animation tracks for voice-cycle states.

    All methods are thread-safe and idempotent: starting the same
    animation twice is a no-op; starting a different animation cancels
    the previous one first.
    """

    def __init__(self, writer: HidWriter) -> None:
        self._writer = writer
        self._pulse_cancel = threading.Event()
        self._pulse_thread: Optional[threading.Thread] = None
        self._flash_cancel = threading.Event()
        self._flash_thread: Optional[threading.Thread] = None
        self._error_timer: Optional[threading.Timer] = None
        self._lock = threading.Lock()

    # ---- public API ---------------------------------------------------

    def off(self) -> None:
        """Cancel any animation and clear all cosmetic LEDs."""
        with self._lock:
            self._cancel_pulse_locked()
            self._cancel_flash_locked()
            self._cancel_error_locked()
            self._writer.write_one(REPORT_OFFHOOK_LED, False)
            self._writer.write_one(REPORT_RING_LED, False)
            self._writer.write_one(REPORT_HOLD_LED, False)

    def listening_pulse(self) -> None:
        with self._lock:
            self._cancel_pulse_locked()
            self._writer.write_one(REPORT_RING_LED, False)
            self._start_pulse_locked(
                report_id=REPORT_OFFHOOK_LED,
                on_s=_LISTENING_ON_S,
                off_s=_LISTENING_OFF_S,
                name="listening-pulse",
            )

    def speaking_pulse(self) -> None:
        with self._lock:
            self._cancel_pulse_locked()
            self._writer.write_one(REPORT_OFFHOOK_LED, False)
            self._start_pulse_locked(
                report_id=REPORT_RING_LED,
                on_s=_SPEAKING_ON_S,
                off_s=_SPEAKING_OFF_S,
                name="speaking-pulse",
            )

    def processing_steady(self) -> None:
        with self._lock:
            self._cancel_pulse_locked()
            self._writer.write_one(REPORT_OFFHOOK_LED, False)
            self._writer.write_one(REPORT_RING_LED, True)

    def complete_flash(self) -> None:
        """Two-flash overlay on RING_LED. Runs independently of pulse."""
        with self._lock:
            self._cancel_flash_locked()
            self._flash_cancel.clear()

            def run() -> None:
                for _ in range(_FLASH_COUNT):
                    if self._flash_cancel.is_set():
                        return
                    self._writer.write_one(REPORT_RING_LED, True)
                    if self._flash_cancel.wait(_FLASH_ON_S):
                        return
                    self._writer.write_one(REPORT_RING_LED, False)
                    if self._flash_cancel.wait(_FLASH_OFF_S):
                        return

            self._flash_thread = threading.Thread(
                target=run, name="led-complete-flash", daemon=True
            )
            self._flash_thread.start()

    def error_hold(self) -> None:
        with self._lock:
            self._cancel_pulse_locked()
            self._cancel_flash_locked()
            self._cancel_error_locked()
            self._writer.write_one(REPORT_OFFHOOK_LED, False)
            self._writer.write_one(REPORT_RING_LED, False)
            self._writer.write_one(REPORT_HOLD_LED, True)
            self._error_timer = threading.Timer(
                _ERROR_HOLD_S,
                lambda: self._writer.write_one(REPORT_HOLD_LED, False),
            )
            self._error_timer.daemon = True
            self._error_timer.start()

    # ---- internal: lock must be held when called ----------------------

    def _start_pulse_locked(
        self, *, report_id: int, on_s: float, off_s: float, name: str
    ) -> None:
        self._pulse_cancel.clear()

        def run() -> None:
            while not self._pulse_cancel.is_set():
                self._writer.write_one(report_id, True)
                if self._pulse_cancel.wait(on_s):
                    break
                self._writer.write_one(report_id, False)
                if self._pulse_cancel.wait(off_s):
                    break
            self._writer.write_one(report_id, False)

        self._pulse_thread = threading.Thread(target=run, name=name, daemon=True)
        self._pulse_thread.start()

    def _cancel_pulse_locked(self) -> None:
        if self._pulse_thread is not None and self._pulse_thread.is_alive():
            self._pulse_cancel.set()
            self._pulse_thread.join(timeout=_ANIM_JOIN_S)
        self._pulse_thread = None
        self._pulse_cancel.clear()

    def _cancel_flash_locked(self) -> None:
        if self._flash_thread is not None and self._flash_thread.is_alive():
            self._flash_cancel.set()
            self._flash_thread.join(timeout=_ANIM_JOIN_S)
        self._flash_thread = None
        self._flash_cancel.clear()

    def _cancel_error_locked(self) -> None:
        if self._error_timer is not None:
            self._error_timer.cancel()
            self._error_timer = None
            self._writer.write_one(REPORT_HOLD_LED, False)


__all__ = ["LedPhone"]
