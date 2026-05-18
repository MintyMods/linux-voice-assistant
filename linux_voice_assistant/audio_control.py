"""Stage E.1 — `AudioControl`: pause/resume coordinator for the mic recorder.

The Calisto P7200 firmware silently drops HID telephony writes
(`0x0A / 0x0E / 0x46 / 0x09`) while any process holds the USB Audio
Class claim open. To functionally mute the microphone — not just paint
red LEDs — `led/mute.py` must release the claim before the wake
sequence, then re-claim afterwards. See [[calisto-cosmetic-vs-
functional-mute]] for the empirical basis.

The audio claim is held by `__main__.py:process_audio` inside its
`with mic.recorder(...) as mic_in:` block (soundcard library). Exiting
that block releases the claim at the kernel/PipeWire level. This module
brokers the handshake:

  * `request_pause()`: led/mute side. Sets `_pause_desired`, then blocks
    on the condition until `process_audio` confirms it has exited the
    recorder context (`_is_paused == True`). Returns False on timeout
    so the caller can converge to sane (force resume + clear LEDs).
  * `is_pause_desired()` / `confirm_paused()`: process_audio side.
    Checked between record iterations; the loop breaks out of the
    `with` block, then calls `confirm_paused()` so the requester
    unblocks.
  * `request_resume()` / `wait_for_resume()` / `confirm_resumed()`:
    mirror for the resume path.

Threading model: a single `threading.Condition` guards two booleans.
Both sides notify on every state change so neither end spins.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

_LOGGER = logging.getLogger(__name__)

# Default timeouts. `request_pause` is fast because the recorder loop
# polls every block (~16 ms at 16 kHz / 256 samples). Resume can be
# slower because mic.recorder() opens a fresh ALSA/PipeWire stream.
_DEFAULT_PAUSE_TIMEOUT_S = 2.0
_DEFAULT_RESUME_TIMEOUT_S = 3.0


class AudioControl:
    """Bidirectional handshake between mute logic and the audio recorder.

    Construct one instance at startup and inject into both
    `ServerState.audio_control` and `LedMute`. `process_audio` polls
    `is_pause_desired()` between recorder reads.
    """

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._pause_desired = False
        self._is_paused = False

    # ---- led/mute side -----------------------------------------------

    def request_pause(self, *, timeout: float = _DEFAULT_PAUSE_TIMEOUT_S) -> bool:
        """Ask `process_audio` to release the audio claim. Block until
        `confirm_paused()` fires or `timeout` elapses. Returns True on
        confirmation, False on timeout."""
        with self._cond:
            self._pause_desired = True
            self._cond.notify_all()
            deadline = time.monotonic() + timeout
            while not self._is_paused:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    _LOGGER.error(
                        "AudioControl.request_pause: timed out after %.1fs (recorder still active)",
                        timeout,
                    )
                    return False
                self._cond.wait(remaining)
            return True

    def request_resume(self, *, timeout: float = _DEFAULT_RESUME_TIMEOUT_S) -> bool:
        """Ask `process_audio` to re-claim the audio stream. Block until
        `confirm_resumed()` fires. Returns True on confirmation, False
        on timeout."""
        with self._cond:
            self._pause_desired = False
            self._cond.notify_all()
            deadline = time.monotonic() + timeout
            while self._is_paused:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    _LOGGER.error(
                        "AudioControl.request_resume: timed out after %.1fs (recorder did not re-open)",
                        timeout,
                    )
                    return False
                self._cond.wait(remaining)
            return True

    # ---- process_audio side ------------------------------------------

    def is_pause_desired(self) -> bool:
        """Cheap poll between recorder reads. No lock contention in the
        common case — just an atomic bool read under a brief lock."""
        with self._cond:
            return self._pause_desired

    def confirm_paused(self) -> None:
        """Recorder context has exited; the audio claim is released as
        far as soundcard / ALSA / PipeWire is concerned. Firmware may
        take an additional ~0.6 s to sense the release (v0
        STREAM_CLOSE_WAIT_SEC). Callers needing firmware-level release
        sleep after this call returns."""
        with self._cond:
            self._is_paused = True
            self._cond.notify_all()

    def confirm_resumed(self) -> None:
        """Recorder context is open again."""
        with self._cond:
            self._is_paused = False
            self._cond.notify_all()

    def wait_for_resume(self, *, timeout: Optional[float] = None) -> bool:
        """Block while pause is still desired. process_audio calls this
        between an `exit context` and `re-enter context` so it doesn't
        spin waiting for the resume request. Returns True if pause was
        cleared, False on timeout (caller should re-loop)."""
        with self._cond:
            while self._pause_desired:
                if not self._cond.wait(timeout):
                    return False
            return True

    # ---- introspection (tests, watchdog) ------------------------------

    @property
    def is_paused(self) -> bool:
        with self._cond:
            return self._is_paused


__all__ = ["AudioControl"]
