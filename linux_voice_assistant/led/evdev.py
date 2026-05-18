"""Stage E.1 — `EvdevMuteListener`: KEY_MICMUTE → "go private".

Ports the mute-button half of `calisto-led/led_service.py:
evdev_listener_loop`. The hidraw mute report (`0x0B`) is ~25% reliable
in PC Media mode while evdev KEY_MICMUTE is 100% reliable (verified
2026-05-18, results in `mute_button_probe_results.md` §"Test 3").

Vol± is intentionally **not** handled here — `led/hid.py` covers it
directly off hidraw `0x02 0x02` / `0x02 0x04`, which is also 100%
reliable. The v0 evdev vol± path was defence-in-depth that the Stage
E.1 design dropped.

Triggers `on_mute_press()` (semantic — caller decides whether that means
"enter private" or "exit private" by inspecting current state). 1.0 s
debounce to filter bounce from the membrane button.
"""

from __future__ import annotations

import logging
import os
import struct
import threading
import time
from typing import Callable, Optional

_LOGGER = logging.getLogger(__name__)

# Linux input_event struct: 16-byte timeval + u16 type + u16 code + u32 value.
_INPUT_EVENT_STRUCT = "@llHHi"
_INPUT_EVENT_SIZE = struct.calcsize(_INPUT_EVENT_STRUCT)

_EV_KEY = 0x01
_KEY_MICMUTE = 0xF8

# Calisto P7200 evdev device name as exposed via `/proc/bus/input/devices`.
# The duplicated "Plantronics" prefix is firmware-supplied, not a typo.
_CALISTO_INPUT_NAME = "Plantronics Plantronics Calisto 7200"

_HW_BUTTON_DEBOUNCE_S = 1.0
_RECONNECT_BACKOFF_S = 5.0


MutePressCallback = Callable[[], None]


def find_calisto_input() -> Optional[str]:
    """Return `/dev/input/eventN` for the Calisto, or None if absent."""
    try:
        with open("/proc/bus/input/devices") as fh:
            blocks = fh.read().split("\n\n")
    except OSError:
        return None
    for block in blocks:
        if _CALISTO_INPUT_NAME not in block:
            continue
        for line in block.splitlines():
            if not line.startswith("H:") or "event" not in line:
                continue
            for token in line.split():
                if token.startswith("event"):
                    return f"/dev/input/{token}"
    return None


class EvdevMuteListener:
    """Daemon thread reading KEY_MICMUTE events.

    Survives device re-enumeration via an outer reconnect loop. Drops
    repeats within `_HW_BUTTON_DEBOUNCE_S` to absorb membrane bounce.
    """

    def __init__(
        self,
        *,
        on_mute_press: MutePressCallback,
        stop_event: Optional[threading.Event] = None,
    ) -> None:
        self._on_mute_press = on_mute_press
        self._stop = stop_event or threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_press_at = 0.0

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="calisto-evdev-mute",
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
            path = find_calisto_input()
            if path is None:
                _LOGGER.warning(
                    "Calisto evdev not found; retry in %.0fs", _RECONNECT_BACKOFF_S
                )
                if self._stop.wait(_RECONNECT_BACKOFF_S):
                    return
                continue
            _LOGGER.info("evdev listener: opening %s for KEY_MICMUTE", path)
            try:
                fd = os.open(path, os.O_RDONLY)
            except OSError as exc:
                _LOGGER.error(
                    "could not open %s: %s — retry in %.0fs",
                    path,
                    exc,
                    _RECONNECT_BACKOFF_S,
                )
                if self._stop.wait(_RECONNECT_BACKOFF_S):
                    return
                continue
            try:
                self._read_loop(fd)
            finally:
                try:
                    os.close(fd)
                except OSError:
                    pass

    def _read_loop(self, fd: int) -> None:
        while not self._stop.is_set():
            try:
                data = os.read(fd, _INPUT_EVENT_SIZE)
            except OSError as exc:
                _LOGGER.error("evdev read failed: %s — reopening", exc)
                return
            if len(data) < _INPUT_EVENT_SIZE:
                return
            _, _, etype, ecode, evalue = struct.unpack(_INPUT_EVENT_STRUCT, data)
            if etype != _EV_KEY or ecode != _KEY_MICMUTE or evalue != 1:
                continue
            now = time.monotonic()
            if now - self._last_press_at < _HW_BUTTON_DEBOUNCE_S:
                _LOGGER.info("hardware mute button: debounced")
                continue
            self._last_press_at = now
            _LOGGER.info("hardware mute button pressed")
            try:
                self._on_mute_press()
            except Exception:
                _LOGGER.exception("mute-press callback raised")


__all__ = ["EvdevMuteListener", "MutePressCallback", "find_calisto_input"]
