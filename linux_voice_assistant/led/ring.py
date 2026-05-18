"""Stage E.1 — `LedRing`: alarm ring pattern (two quick pulses + pause).

Ports `LedDriver.start_ring` / `stop_ring` from `calisto-led/
led_service.py`. The pattern — two ~150ms pulses then 800ms gap, looping
— is the v0 alarm-active visual.

Scope here is the LED hook only: caller starts/stops in response to
alarm state. Alarm orchestration (state machine, MQTT, audible-notify
coexistence) belongs in Stage E.2.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

from .hid import HidWriter, REPORT_OFFHOOK_LED

_LOGGER = logging.getLogger(__name__)

_PULSE_ON_S = 0.15
_PULSE_OFF_S = 0.15
_PATTERN_PAUSE_S = 0.8
_PULSES_PER_CYCLE = 2

_RING_JOIN_S = 1.5


class LedRing:
    """Alarm ring loop on OFFHOOK_LED. Idempotent start/stop."""

    def __init__(self, writer: HidWriter) -> None:
        self._writer = writer
        self._cancel = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._cancel.clear()
            self._thread = threading.Thread(
                target=self._run, name="led-ring", daemon=True
            )
            self._thread.start()

    def stop(self) -> None:
        with self._lock:
            self._cancel.set()
            if self._thread is not None:
                self._thread.join(timeout=_RING_JOIN_S)
            self._thread = None
            self._writer.write_one(REPORT_OFFHOOK_LED, False)

    def is_active(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _run(self) -> None:
        while not self._cancel.is_set():
            for _ in range(_PULSES_PER_CYCLE):
                if self._cancel.is_set():
                    break
                self._writer.write_one(REPORT_OFFHOOK_LED, True)
                if self._cancel.wait(_PULSE_ON_S):
                    break
                self._writer.write_one(REPORT_OFFHOOK_LED, False)
                if self._cancel.wait(_PULSE_OFF_S):
                    break
            if self._cancel.wait(_PATTERN_PAUSE_S):
                break
        self._writer.write_one(REPORT_OFFHOOK_LED, False)


__all__ = ["LedRing"]
