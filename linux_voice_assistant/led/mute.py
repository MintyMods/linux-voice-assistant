"""Stage E.1 — `LedMute`: Path B cosmetic mute + MicCapture frame-gate.

**Path B** (chosen 2026-05-18 after on-hardware pre-test, results in
`calisto-led/tools/mute_button_probe_results.md` §"Path B pre-test"):

  1. Write the cosmetic mute palette — `REPORT_OFFHOOK_LED 0x17 = 0x01`
     then `REPORT_MUTE_LED 0x09 = 0x01`. Paints the full red palette
     on the bar / mic / phone via firmware's cosmetic page. No
     telephony wake sequence; no audio-claim release.
  2. Tell `MicCapture` to drop incoming frames + clear the pre-roll
     ring. The USB Audio Class claim stays held by `process_audio`, so:
       - wake-word detection keeps receiving audio (but MicCapture
         won't act on a wake while muted — the gate is at feed());
       - the hardware mute button keeps emitting `KEY_MICMUTE` on
         evdev (Test 3 found 100% reliability while the claim is held);
       - no FU=0 firmware quirk on entry or exit (cosmetic page
         doesn't trigger the v0 M4 condition).
  3. Exit reverses the writes — `09 00 + 17 00` — and unmutes MicCapture.

Path A (v0 audio-release + telephony wake sequence) was the design until
Stage E.1 deploy on 2026-05-18 — it produced a true firmware-level
mic-stream stop, but released the audio claim, which left the hardware
mute button dead and broke wake-word listening for the duration. Path B
trades "bits-still-on-USB-but-discarded-in-software" for hardware mute
button working bidirectionally and wake-word detection alive
post-unmute. See [[calisto-cosmetic-vs-functional-mute]] for the boundary.

If functional firmware mute (Path C: Hub-handshake bypass + always-on
telephony mode) ever lands, this module is the single seam to replace.
"""

from __future__ import annotations

import logging
import threading
from typing import Callable, Optional

from .hid import (
    HidWriter,
    REPORT_MUTE_LED,
    REPORT_OFFHOOK_LED,
)

_LOGGER = logging.getLogger(__name__)


MuteStateCallback = Callable[[bool], None]
"""Called with the final muted state (True/False) after every
transition, including failure-recovered transitions. Useful for
mirroring to MQTT / `ServerState.muted` / a metrics surface."""

MicCaptureGate = Callable[[], None]
"""Gate hook on `MicCapture` — `mute()` drops frames + clears pre-roll;
`unmute()` resumes frame delivery. The USB Audio Class claim is NOT
touched on either side; only the in-process frame flow into VAD / ASR /
wake-word is gated."""


class LedMute:
    """Path B cosmetic mute controller.

    Thread-safe: `enter_private` / `exit_private` serialise on an
    internal lock so a hardware-button press during an MQTT-driven
    transition can't interleave HID writes.
    """

    def __init__(
        self,
        writer: HidWriter,
        *,
        mic_capture_mute: Optional[MicCaptureGate] = None,
        mic_capture_unmute: Optional[MicCaptureGate] = None,
        on_state: Optional[MuteStateCallback] = None,
    ) -> None:
        self._writer = writer
        self._mic_capture_mute = mic_capture_mute
        self._mic_capture_unmute = mic_capture_unmute
        self._on_state = on_state or (lambda _muted: None)
        self._lock = threading.Lock()

    def enter_private(self) -> bool:
        """Paint cosmetic red palette + gate the mic frame flow.

        Returns True on HID write success. The frame-gate is best-effort
        (if it raises we still flip muted=True — the red is visible and
        we'd rather over-mute than mismatch state with visuals)."""
        with self._lock:
            _LOGGER.info("LedMute: entering private (Path B cosmetic)")
            ok = self._writer.write_seq(
                (REPORT_OFFHOOK_LED, 0x01),
                (REPORT_MUTE_LED, 0x01),
                pause=0.05,
            )
            if not ok:
                _LOGGER.error("LedMute.enter: HID writes failed")
            if self._mic_capture_mute is not None:
                try:
                    self._mic_capture_mute()
                except Exception:
                    _LOGGER.exception("LedMute.enter: mic_capture_mute raised; continuing")
            self._on_state(True)
            return ok

    def exit_private(self) -> bool:
        """Clear cosmetic red palette + un-gate the mic frame flow."""
        with self._lock:
            _LOGGER.info("LedMute: exiting private (Path B cosmetic)")
            ok = self._writer.write_seq(
                (REPORT_MUTE_LED, 0x00),
                (REPORT_OFFHOOK_LED, 0x00),
                pause=0.05,
            )
            if not ok:
                _LOGGER.warning(
                    "LedMute.exit: HID writes failed — LEDs may be inconsistent until next reconciliation"
                )
            if self._mic_capture_unmute is not None:
                try:
                    self._mic_capture_unmute()
                except Exception:
                    _LOGGER.exception("LedMute.exit: mic_capture_unmute raised; continuing")
            self._on_state(False)
            return ok


__all__ = ["LedMute", "MuteStateCallback", "MicCaptureGate"]
