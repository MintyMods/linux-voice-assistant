"""Stage E.1 — `LedBar`: USB-Audio-Class Feature Unit volume control.

Ports the volume side of `calisto-led/led_service.py`:

  * Reads / writes the Calisto P7200 USB-Audio Feature Unit (`amixer
    numid=4`, integer range 0..15, 1 dB / step). This is the device's
    *real* volume knob; PipeWire / pulse percentages map to it through a
    logarithmic curve that wastes half the range below ~65%.
  * Parses the v0 payload grammar: `up` / `down` / `louder` / `quieter`,
    absolute `N` / `N%`, relative `+N%` / `-N%`.
  * Calls `on_state_changed(pct)` after every successful mutation so the
    `LedController` can republish the K.11 retained `calisto/<room>/
    volume/state` topic.

`pactl set-sink-volume … 100%` pinning is **deliberately not ported** —
v0 commit FU_DIRECT_PATCH_v2 (2026-05-15) proved that pinning pactl
clamps the FU on every relative step (current always reads 15 → 15-step
collapses to FU=13 stuck on vol-down). PipeWire tracks our direct FU
writes without re-clamping.
"""

from __future__ import annotations

import logging
import re
import subprocess
from typing import Callable, Optional

_LOGGER = logging.getLogger(__name__)

# Calisto P7200 Feature Unit range — 0..15 (4-bit), 1 dB per step.
FU_MIN = 0
FU_MAX = 15

# Default step for `up` / `down` / vol± hardware buttons.
_VOLUME_STEP_FU = 2

_AMIXER_CARD = "hw:CARD=P7200"
_AMIXER_NUMID = "numid=4"
_AMIXER_TIMEOUT_S = 2.0

_NUMID_VALUE_RE = re.compile(r"^\s*:\s*values=(\d+)", re.MULTILINE)


VolumeStateCallback = Callable[[int], None]
"""Invoked with the new percentage after every successful mutation."""


def _clamp_fu(value: int) -> int:
    return max(FU_MIN, min(FU_MAX, int(value)))


def _pct_to_fu(pct: int) -> int:
    pct = max(0, min(100, int(pct)))
    return round(pct * FU_MAX / 100)


def _fu_to_pct(fu: int) -> int:
    return round(_clamp_fu(fu) * 100 / FU_MAX)


class LedBar:
    """Volume controller.

    State lives in the hardware FU register, not the class — every read
    re-queries amixer. The callback fires after every successful write so
    the K.11 retained topic stays consistent with hardware reality even
    when the source of change is a hardware button press.
    """

    def __init__(self, on_state_changed: Optional[VolumeStateCallback] = None) -> None:
        self._on_state_changed = on_state_changed or (lambda _pct: None)

    # ---- read --------------------------------------------------------

    def read_fu(self) -> Optional[int]:
        try:
            result = subprocess.run(
                ["amixer", "-D", _AMIXER_CARD, "cget", _AMIXER_NUMID],
                check=False,
                timeout=_AMIXER_TIMEOUT_S,
                capture_output=True,
                text=True,
            )
        except subprocess.TimeoutExpired:
            _LOGGER.error("amixer cget timed out")
            return None
        except FileNotFoundError:
            _LOGGER.error("amixer not on PATH")
            return None
        match = _NUMID_VALUE_RE.search(result.stdout)
        if match is None:
            _LOGGER.warning("could not parse FU from amixer output: %r", result.stdout)
            return None
        return _clamp_fu(int(match.group(1)))

    def read_pct(self) -> Optional[int]:
        fu = self.read_fu()
        if fu is None:
            return None
        return _fu_to_pct(fu)

    # ---- write -------------------------------------------------------

    def _write_fu(self, value: int) -> bool:
        target = _clamp_fu(value)
        try:
            subprocess.run(
                ["amixer", "-D", _AMIXER_CARD, "cset", _AMIXER_NUMID, str(target)],
                check=False,
                timeout=_AMIXER_TIMEOUT_S,
                capture_output=True,
            )
        except subprocess.TimeoutExpired:
            _LOGGER.error("amixer cset timed out")
            return False
        except FileNotFoundError:
            _LOGGER.error("amixer not on PATH")
            return False
        return True

    def apply(self, payload: str) -> bool:
        """Apply the v0 volume grammar. Returns True if a write was made
        and the state callback fired.

        Grammar (case-insensitive, whitespace-stripped):
            up | louder        → +_VOLUME_STEP_FU (2 FU)
            down | quieter     → −_VOLUME_STEP_FU
            N                  → absolute pct  (0..100, clamped)
            N%                 → absolute pct
            +N%                → relative +N% (rendered through pct→FU)
            -N%                → relative −N%
        """
        normalised = payload.strip().lower()
        current = self.read_fu()
        if current is None:
            current = FU_MAX  # match v0 fallback: treat unknown as max

        target_fu = self._resolve_target(normalised, current)
        if target_fu is None:
            _LOGGER.error("unparseable volume payload %r", payload)
            return False
        target_fu = _clamp_fu(target_fu)

        _LOGGER.info(
            "FU set numid=4 %d (=%d%%) from payload=%r",
            target_fu,
            _fu_to_pct(target_fu),
            payload,
        )
        if not self._write_fu(target_fu):
            return False
        # Re-read rather than trusting our target — confirms the write
        # landed and survives an amixer that silently clamps.
        confirmed = self.read_fu()
        pct = _fu_to_pct(confirmed if confirmed is not None else target_fu)
        try:
            self._on_state_changed(pct)
        except Exception:
            _LOGGER.exception("volume state callback raised")
        return True

    @staticmethod
    def _resolve_target(payload: str, current_fu: int) -> Optional[int]:
        if payload in ("up", "louder"):
            return current_fu + _VOLUME_STEP_FU
        if payload in ("down", "quieter"):
            return current_fu - _VOLUME_STEP_FU
        if payload.endswith("%") and payload.lstrip("+-").rstrip("%").isdigit():
            body = payload.rstrip("%")
            if body.startswith("+"):
                return current_fu + _pct_to_fu(int(body[1:]))
            if body.startswith("-"):
                return current_fu - _pct_to_fu(int(body[1:]))
            return _pct_to_fu(int(body))
        try:
            return _pct_to_fu(int(float(payload)))
        except ValueError:
            return None


__all__ = [
    "FU_MIN",
    "FU_MAX",
    "LedBar",
    "VolumeStateCallback",
]
