"""Stage E.2 G3 / K.8 / K.9 — alarm orchestration.

AlarmController owns the `alarm_player` (Section O.2 / G1) and the rules
around playing an alarm alongside other audio channels per G3:

  alarm_player → @ volume_pct, plays the ringtone on loop until stopped
  music_player → paused for the duration of the alarm (snapshot before)
  tts_player   → keeps playing alongside (voice always wins through alarm)
  chime_player → suppressed by the audible_notify arbiter (E6)

The controller is owned by `ServerState.alarm_controller`. Wiring:

  __main__.py        constructs + attaches it.
  ha_bridge.py       routes calisto/<room>/alarm/set + /all/alarm/set here.
  audible_notify.py  reads `is_ringing` to suppress chimes during alarm.

K.9 `alarm/state` is published retained on every start/stop transition.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from .ha_bridge import HABridge
    from .mpv_player import MpvMediaPlayer

_LOGGER = logging.getLogger(__name__)


@dataclass
class AlarmRequest:
    """Parsed K.8 payload — defaults applied for omitted fields."""

    ringtone_path: str
    duration_s: int
    volume_pct: int
    source: str
    alarm_id: str
    schedule_ts: Optional[str]


class AlarmController:
    """Owns the alarm playback lifecycle + K.9 state publish.

    Thread-safety: `set_alarm` / `stop_alarm` are called from the asyncio
    loop (HABridge routes message bytes through `loop.call_soon_threadsafe`),
    but the duration timer fires on a background thread. A single lock
    serialises the start/stop critical sections; mpv calls themselves are
    already thread-safe via the libmpv binding.
    """

    def __init__(
        self,
        *,
        alarm_player: "MpvMediaPlayer",
        music_player: "MpvMediaPlayer",
        room: str,
        ha_bridge: "Optional[HABridge]" = None,
        default_duration_s: int = 300,
        default_volume_pct: int = 100,
    ) -> None:
        self._alarm_player = alarm_player
        self._music_player = music_player
        self._room = room
        self._ha_bridge = ha_bridge
        self._default_duration_s = default_duration_s
        self._default_volume_pct = default_volume_pct

        self._lock = threading.Lock()
        self._current: Optional[AlarmRequest] = None
        self._started_ts: Optional[float] = None
        self._media_was_playing: bool = False
        self._stop_timer: Optional[threading.Timer] = None

        self._state_topic = f"calisto/{room}/alarm/state"

    # -- public API --------------------------------------------------------

    @property
    def is_ringing(self) -> bool:
        return self._current is not None

    def attach_ha_bridge(self, ha_bridge: "HABridge") -> None:
        """Wire the publisher after construction (deferred bridge start)."""
        self._ha_bridge = ha_bridge

    def set_alarm(self, payload: bytes) -> None:
        """Handle a K.8 `alarm/set` message. Idempotent on alarm_id —
        repeating the same alarm_id is a no-op while it's ringing."""
        try:
            req = self._parse_payload(payload)
        except ValueError as exc:
            _LOGGER.warning("alarm/set: bad payload (%s); dropped", exc)
            return

        with self._lock:
            if self._current is not None and self._current.alarm_id == req.alarm_id:
                _LOGGER.debug("alarm/set: %s already ringing; ignoring duplicate", req.alarm_id)
                return
            # A second alarm during an active one replaces the first. The
            # snapshot is preserved (we only restore at the *final* stop).
            self._cancel_timer_locked()
            if self._current is None:
                self._media_was_playing = bool(getattr(self._music_player, "is_playing", False))
                if self._media_was_playing:
                    try:
                        self._music_player.pause()
                    except Exception:
                        _LOGGER.exception("alarm/set: music_player.pause() raised")
            self._current = req
            self._started_ts = time.time()

            try:
                self._alarm_player.set_volume(float(req.volume_pct))
            except Exception:
                _LOGGER.exception("alarm/set: alarm_player.set_volume raised")
            try:
                self._alarm_player.play(req.ringtone_path, done_callback=self._on_alarm_finished)
            except Exception:
                _LOGGER.exception("alarm/set: alarm_player.play raised")
                # Roll back snapshot so a failed start doesn't leave the
                # music paused forever.
                self._current = None
                self._started_ts = None
                if self._media_was_playing:
                    try:
                        self._music_player.resume()
                    except Exception:
                        pass
                return

            self._arm_timer_locked(req.duration_s)

        self._publish_state()

    def stop_alarm(self) -> None:
        """Stop the currently-ringing alarm. No-op when idle."""
        with self._lock:
            if self._current is None:
                return
            self._cancel_timer_locked()
            try:
                self._alarm_player.stop()
            except Exception:
                _LOGGER.exception("stop_alarm: alarm_player.stop raised")
            self._restore_media_locked()
            self._current = None
            self._started_ts = None

        self._publish_state()

    # -- internals --------------------------------------------------------

    def _on_alarm_finished(self) -> None:
        """mpv signals end-of-playback. The ringtone is short; we loop by
        re-issuing play() if the alarm is still active. The duration timer
        is the authoritative stop signal."""
        with self._lock:
            req = self._current
            if req is None:
                return
            try:
                self._alarm_player.play(req.ringtone_path, done_callback=self._on_alarm_finished)
            except Exception:
                _LOGGER.exception("alarm loop: alarm_player.play raised")

    def _arm_timer_locked(self, duration_s: int) -> None:
        timer = threading.Timer(float(duration_s), self._on_duration_elapsed)
        timer.daemon = True
        timer.start()
        self._stop_timer = timer

    def _cancel_timer_locked(self) -> None:
        timer = self._stop_timer
        if timer is not None:
            timer.cancel()
            self._stop_timer = None

    def _on_duration_elapsed(self) -> None:
        _LOGGER.info("alarm duration elapsed; auto-stopping")
        self.stop_alarm()

    def _restore_media_locked(self) -> None:
        if not self._media_was_playing:
            return
        try:
            self._music_player.resume()
        except Exception:
            _LOGGER.exception("alarm restore: music_player.resume raised")
        self._media_was_playing = False

    def _publish_state(self) -> None:
        bridge = self._ha_bridge
        if bridge is None:
            return
        with self._lock:
            req = self._current
            ts = self._started_ts
            media_was_playing = self._media_was_playing if req is not None else False
        if req is not None:
            payload = {
                "ringing": True,
                "alarm_id": req.alarm_id,
                "ringtone": req.ringtone_path,
                "started_ts": (
                    datetime.fromtimestamp(ts).astimezone().isoformat(timespec="milliseconds")
                    if ts is not None
                    else None
                ),
                "pre_alarm_snapshot": (
                    {"media_was_playing": True} if media_was_playing else None
                ),
            }
        else:
            payload = {
                "ringing": False,
                "alarm_id": None,
                "ringtone": None,
                "started_ts": None,
                "pre_alarm_snapshot": None,
            }
        try:
            bridge.publish(self._state_topic, json.dumps(payload), qos=1, retain=True)
        except Exception:
            _LOGGER.exception("alarm/state publish raised for %s", self._state_topic)

    def _parse_payload(self, payload: bytes) -> AlarmRequest:
        try:
            obj: Any = json.loads(payload.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise ValueError(f"not JSON: {exc}") from exc
        if not isinstance(obj, dict):
            raise ValueError("payload must be a JSON object")

        ringtone = obj.get("ringtone")
        ringtone_path = self._resolve_ringtone(ringtone)
        duration_s = int(obj.get("duration_s") or self._default_duration_s)
        if duration_s <= 0 or duration_s > 24 * 3600:
            duration_s = self._default_duration_s
        volume_pct = int(obj.get("volume_pct") or self._default_volume_pct)
        volume_pct = max(0, min(100, volume_pct))
        source = str(obj.get("source") or "unknown")
        alarm_id = str(obj.get("alarm_id") or f"alarm-{int(time.time())}")
        schedule_ts = obj.get("ts")
        return AlarmRequest(
            ringtone_path=ringtone_path,
            duration_s=duration_s,
            volume_pct=volume_pct,
            source=source,
            alarm_id=alarm_id,
            schedule_ts=schedule_ts if isinstance(schedule_ts, str) else None,
        )

    def _resolve_ringtone(self, name: Optional[str]) -> str:
        """Map a ringtone slug to a playable path.

        Accepts:
          - an absolute path → returned verbatim
          - a bare filename → resolved against ALARM_SOUNDS_DIR env var
          - None / empty → default ringtone
        """
        from .audible_notify import resolve_alarm_sound

        return resolve_alarm_sound(name)
