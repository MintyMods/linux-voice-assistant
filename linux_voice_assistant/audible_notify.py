"""Stage E.2 E6 + E.2-h — audible notifications (chime + say) with
priority/preemption.

Three concerns share this file because they're tightly coupled and small:

  1. resolve_chime_sound() / resolve_alarm_sound() — env-var-driven path
     lookup for the Ubuntu Touch sound library (E1, E4). Used by both
     `alarm.py` and `ChimeController`.
  2. ChimeController       — short FX (wake / reject / confirm / arm) on
     `chime_player`. Suppressed when the priority hierarchy says so.
  3. AudibleNotifyArbiter  — pure-logic decision: given the current device
     state + alarm state + the audible_notify on/off toggle, should this
     chime / say play? Sits between MQTT routing and the channel players.

Priority hierarchy (E6):

  alarm   (paused music, kept-alive TTS, 100% volume on alarm channel)
  tts     (voice replies, K.10 say with scope=tts — survives chime)
  chime   (short FX, scope=chime — suppressed during tts or alarm)

A `say` with scope=tts always goes through; a `say` with scope=chime
defers to the arbiter.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from .alarm import AlarmController
    from .models import ServerState
    from .mpv_player import MpvMediaPlayer
    from .session import DeviceSession, State

_LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sound library path resolution (E1 / E4 — Ubuntu Touch chimes + alarms)
# ---------------------------------------------------------------------------

_DEFAULT_CHIME_DIRS = (
    "/usr/share/sounds/ubuntu-touch/notifications/single",
    "/usr/share/sounds/freedesktop/stereo",
)
_DEFAULT_ALARM_DIRS = (
    "/usr/share/sounds/ubuntu-touch/ringtones",
    "/usr/share/sounds/alsa",
)


def _candidate_dirs(env_var: str, defaults: tuple) -> tuple:
    """Return (env_var split by os.pathsep, then defaults). Caller checks
    existence — we don't error on missing dirs since a deploy might use
    only the env var (or only the defaults)."""
    extra = os.environ.get(env_var)
    if extra:
        return tuple(p for p in extra.split(os.pathsep) if p) + defaults
    return defaults


def _resolve_in_dirs(name: str, dirs: tuple) -> Optional[str]:
    p = Path(name)
    if p.is_absolute() and p.exists():
        return str(p)
    for d in dirs:
        candidate = Path(d) / name
        if candidate.exists():
            return str(candidate)
    return None


def resolve_chime_sound(name: Optional[str]) -> Optional[str]:
    """Map a chime slug to a path. Returns None when nothing matches —
    the caller (ChimeController.play) treats that as 'silently skip'.
    Env override: CHIME_SOUNDS_DIR (colon/semicolon-separated)."""
    if not name:
        return None
    return _resolve_in_dirs(name, _candidate_dirs("CHIME_SOUNDS_DIR", _DEFAULT_CHIME_DIRS))


def list_alarm_ringtones() -> list:
    """Return a deduplicated, sorted list of alarm ringtone slugs available
    on this host. Honoured by ``ALARM_SOUNDS_DIR`` plus the
    ``_DEFAULT_ALARM_DIRS`` fallbacks. Caller (Stage G N.2 select) uses the
    list as the HA select options.

    On a fresh host with no alarm library installed, returns an empty
    list — the caller is expected to fall back to a sensible default
    rather than block startup."""
    found = set()
    for d in _candidate_dirs("ALARM_SOUNDS_DIR", _DEFAULT_ALARM_DIRS):
        try:
            for entry in Path(d).iterdir():
                if entry.is_file() and entry.suffix.lower() in (".ogg", ".wav", ".mp3", ".flac"):
                    found.add(entry.name)
        except (FileNotFoundError, NotADirectoryError, PermissionError):
            continue
    return sorted(found)


def resolve_alarm_sound(name: Optional[str]) -> str:
    """Map an alarm slug to a path. Falls back through:

      1. literal path or env-listed dirs
      2. _DEFAULT_ALARM_DIRS
      3. first alarm-shaped file in any of the dirs (so "ring whatever you
         have" still works on a fresh host)
      4. the raw name as-passed (lets mpv attempt to load it and surface
         its own error — easier to debug than swallowing)

    Unlike chime, an alarm MUST produce sound, so we never return None.
    Env override: ALARM_SOUNDS_DIR (colon/semicolon-separated)."""
    dirs = _candidate_dirs("ALARM_SOUNDS_DIR", _DEFAULT_ALARM_DIRS)
    resolved = _resolve_in_dirs(name or "", dirs) if name else None
    if resolved is not None:
        return resolved

    for d in dirs:
        try:
            for entry in Path(d).iterdir():
                if entry.is_file() and entry.suffix.lower() in (".ogg", ".wav", ".mp3", ".flac"):
                    _LOGGER.warning(
                        "alarm: ringtone %r not found; falling back to %s", name, entry
                    )
                    return str(entry)
        except (FileNotFoundError, NotADirectoryError, PermissionError):
            continue
    _LOGGER.warning("alarm: no ringtone found for %r and no fallbacks available", name)
    return name or "alarm.ogg"


# ---------------------------------------------------------------------------
# Priority arbitration
# ---------------------------------------------------------------------------


class AudibleNotifyArbiter:
    """Pure-logic decision: should this audible notification play right now?

    Holds no audio state itself — instead it's handed (a) a callable that
    returns the current `State` (from DeviceSession), and (b) a callable
    returning whether an alarm is ringing. Both are read lazily so the
    arbiter survives component restarts.

    Tunable: `enabled` flag mirrors `switch.calisto_<room>_audible_notify`
    (Section N) for global chime suppression. Defaults to True.
    """

    def __init__(
        self,
        *,
        state_getter: Callable[[], "State"],
        alarm_ringing_getter: Callable[[], bool],
        enabled: bool = True,
    ) -> None:
        self._state_getter = state_getter
        self._alarm_ringing_getter = alarm_ringing_getter
        self.enabled = enabled

    def allow_chime(self) -> bool:
        """Chime tier: suppressed if disabled, if voice is actively speaking,
        or if an alarm is ringing."""
        from .session import State

        if not self.enabled:
            return False
        if self._alarm_ringing_getter():
            return False
        try:
            current = self._state_getter()
        except Exception:
            return True
        return current not in (State.SPEAKING, State.THINKING)

    def allow_say_tts(self) -> bool:
        """TTS tier: only suppressed during an active alarm. Voice replies
        and ad-hoc K.10 announcements both go through during normal
        conversation; the existing cancel chain handles barge-in."""
        return not self._alarm_ringing_getter()


# ---------------------------------------------------------------------------
# ChimeController — thin wrapper over chime_player that consults the arbiter
# ---------------------------------------------------------------------------


class ChimeController:
    """Plays short FX through `chime_player`, gated by the arbiter."""

    def __init__(
        self,
        *,
        chime_player: "MpvMediaPlayer",
        arbiter: AudibleNotifyArbiter,
    ) -> None:
        self._player = chime_player
        self._arbiter = arbiter

    def play(self, name: str) -> bool:
        """Resolve `name`, consult the arbiter, fire-and-forget play.
        Returns True when the chime was queued, False on any skip reason."""
        if not self._arbiter.allow_chime():
            _LOGGER.debug("chime %r suppressed by arbiter", name)
            return False
        path = resolve_chime_sound(name)
        if path is None:
            _LOGGER.warning("chime %r: no sound file found in CHIME_SOUNDS_DIR or defaults", name)
            return False
        try:
            self._player.play(path)
            return True
        except Exception:
            _LOGGER.exception("chime_player.play raised for %s", path)
            return False
