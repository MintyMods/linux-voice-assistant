"""Stage E.1 — `LedMute`: Path A firmware mute via audio-release + wake seq.

Ports `LedDriver._enter_telephony_mute` / `_exit_telephony_mute` from
`calisto-led/led_service.py`, replacing the v0 `systemctl stop linux-
voice-assistant.service` with an in-process `AudioControl.request_pause`
handshake.

Sequence on `enter_private`:

  1. `AudioControl.request_pause` — `process_audio` exits its
     `with mic.recorder(...)` block; soundcard / ALSA / PipeWire release
     the USB Audio Class claim.
  2. Sleep `_STREAM_CLOSE_WAIT_S` (0.6 s) — firmware sensing lag.
     Without this the telephony writes below are dropped.
  3. Wake sequence on the telephony page (`0x0A / 0x0E / 0x46`) — moves
     the firmware state machine into call-active + in-call so it accepts
     the mute LED.
  4. `09 01` — paints the red mute palette and (critically, this is the
     functional half) tells firmware to stop emitting mic audio over USB.

`exit_private` runs the canonical Hub call-end sequence (`0E 00 / 0A 08
/ 09 00 / 17 00 / 19 00`) **before** re-claiming audio — once the
recorder reopens, firmware reverts to PC Media mode and the telephony
writes are silently dropped again. Then `AudioControl.request_resume`
brings the recorder back online.

Convergence-to-sane: any failure (audio pause timeout, partial HID
writes, firmware state mismatch) triggers `_converge_to_unmuted` —
best-effort clear + force resume — and the method returns False so
`LedController` can revert its in-memory `_muted` flag.

Crash survival: out of scope here. `LedController.start()` writes a
known-clean cosmetic state and force-resumes the recorder; if LVA
crashed mid-mute it comes up unmuted on restart, matching the user's
[[deployment-robustness-principle]] choice.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Optional

from ..audio_control import AudioControl
from .hid import (
    CALL_STATE_ACTIVE,
    CALL_STATE_ENDED,
    CALL_STATE_IN_CALL,
    HidWriter,
    REPORT_AUX_INDICATOR,
    REPORT_CALL_STATE,
    REPORT_HOLD_LED,
    REPORT_MUTE_LED,
    REPORT_OFFHOOK_LED,
    REPORT_PULSE,
)

_LOGGER = logging.getLogger(__name__)

# v0 STREAM_CLOSE_WAIT_SEC — firmware sense lag between userspace closing
# the USB Audio claim and the device transitioning out of PC Media mode.
# Kept verbatim from v0 calibration.
_STREAM_CLOSE_WAIT_S = 0.6

# Intra-sequence settle pauses between the three wake-sequence bursts.
# v0 used 0.3 s between groups; preserved exactly.
_WAKE_SEQ_SETTLE_S = 0.3

# Audio handshake timeouts. Pause is fast (process_audio polls every
# block, ~16 ms). Resume can be slower (soundcard may take a few
# hundred ms to re-open the ALSA/PipeWire stream).
_PAUSE_TIMEOUT_S = 2.0
_RESUME_TIMEOUT_S = 3.0


MuteStateCallback = Callable[[bool], None]
"""Called with the final muted state (True/False) after every
transition, including failure-recovered transitions. Useful for
mirroring to MQTT / `ServerState.muted` / a metrics surface."""


class LedMute:
    """Path A firmware-mute controller.

    Thread-safe: `enter_private` / `exit_private` serialise on an
    internal lock so a hardware-button press during an MQTT-driven
    transition can't interleave HID writes.
    """

    def __init__(
        self,
        writer: HidWriter,
        audio_control: AudioControl,
        *,
        on_state: Optional[MuteStateCallback] = None,
        on_mic_capture_abort: Optional[Callable[[], None]] = None,
    ) -> None:
        self._writer = writer
        self._audio_control = audio_control
        self._on_state = on_state or (lambda _muted: None)
        self._on_mic_capture_abort = on_mic_capture_abort
        self._lock = threading.Lock()

    def enter_private(self) -> bool:
        """Release audio claim, run wake sequence, paint red. Returns
        True on success; False if any step failed (LEDs and recorder
        will have been converged back to unmuted)."""
        with self._lock:
            _LOGGER.info("LedMute: entering private")
            # Drop any in-flight VAD capture so a half-recorded utterance
            # doesn't trigger ASR after we unmute.
            if self._on_mic_capture_abort is not None:
                try:
                    self._on_mic_capture_abort()
                except Exception:
                    _LOGGER.exception("mic_capture.abort raised; continuing")

            paused = self._audio_control.request_pause(timeout=_PAUSE_TIMEOUT_S)
            if not paused:
                _LOGGER.error(
                    "LedMute.enter: AudioControl did not confirm pause within %.1fs",
                    _PAUSE_TIMEOUT_S,
                )
                self._converge_to_unmuted_locked()
                return False

            # Firmware sensing lag — without this the telephony writes
            # below land in a state machine that hasn't yet noticed the
            # audio claim was released.
            time.sleep(_STREAM_CLOSE_WAIT_S)

            ok = self._writer.write_seq(
                (REPORT_OFFHOOK_LED, 0x01),
                (REPORT_CALL_STATE, CALL_STATE_ACTIVE),
                (REPORT_PULSE, 0x01),
                (REPORT_PULSE, 0x00),
                pause=0.05,
            )
            time.sleep(_WAKE_SEQ_SETTLE_S)
            ok = self._writer.write_seq(
                (REPORT_CALL_STATE, CALL_STATE_IN_CALL),
                (REPORT_AUX_INDICATOR, 0x01),
                pause=0.05,
            ) and ok
            time.sleep(_WAKE_SEQ_SETTLE_S)
            ok = self._writer.write_seq((REPORT_MUTE_LED, 0x01)) and ok

            if not ok:
                _LOGGER.error(
                    "LedMute.enter: HID writes failed; converging to unmuted"
                )
                self._converge_to_unmuted_locked()
                return False

            self._on_state(True)
            return True

    def exit_private(self) -> bool:
        """Run end-call sequence (telephony page, still effective while
        audio is released), then re-claim audio. Returns True on success."""
        with self._lock:
            _LOGGER.info("LedMute: exiting private")
            ok = self._writer.write_seq(
                (REPORT_AUX_INDICATOR, 0x00),
                (REPORT_CALL_STATE, CALL_STATE_ENDED),
                (REPORT_MUTE_LED, 0x00),
                (REPORT_OFFHOOK_LED, 0x00),
                (REPORT_HOLD_LED, 0x00),
                pause=0.05,
            )
            if not ok:
                _LOGGER.warning(
                    "LedMute.exit: end-call HID writes failed — LEDs may be inconsistent until next reconciliation"
                )

            resumed = self._audio_control.request_resume(timeout=_RESUME_TIMEOUT_S)
            if not resumed:
                _LOGGER.error(
                    "LedMute.exit: AudioControl did not confirm resume within %.1fs — wake-word path may be dead",
                    _RESUME_TIMEOUT_S,
                )
                # State still flips to "unmuted" — pretending we're muted
                # while audio is gone would be worse. The L1/L4 watchdog
                # in Stage F is the layer that should respawn here.
            self._on_state(False)
            return ok and resumed

    # ---- recovery -----------------------------------------------------

    def _converge_to_unmuted_locked(self) -> None:
        """Best-effort cleanup. Called while `self._lock` is held."""
        _LOGGER.warning("LedMute: converging to unmuted")
        self._writer.write_seq(
            (REPORT_AUX_INDICATOR, 0x00),
            (REPORT_CALL_STATE, CALL_STATE_ENDED),
            (REPORT_MUTE_LED, 0x00),
            (REPORT_OFFHOOK_LED, 0x00),
            (REPORT_HOLD_LED, 0x00),
            pause=0.05,
        )
        self._audio_control.request_resume(timeout=_RESUME_TIMEOUT_S)
        self._on_state(False)


__all__ = ["LedMute", "MuteStateCallback"]
