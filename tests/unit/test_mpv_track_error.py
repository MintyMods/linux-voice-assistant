"""H5 §3 — per-channel track-error recovery tests.

The Python mpv binding is stubbed in tests/conftest.py; we attach a
capable fake `MPV` here that records `command()` calls and lets us
trigger synthetic `end-file` events. The fake also implements
`observe_property` so the media-role time-pos resume path can be
exercised end-to-end.

Coverage:
  - tts/chime: silent recovery (done_callback fires, no reload)
  - media: reload last_url with `start=<position>` from cached time-pos
  - alarm: reload last_url with `loop-playlist=inf` + full volume
  - heartbeat demotion ladder: ok → degraded (≥3 in 30s) → dead (≥6 in 30s)
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Tuple

import pytest

import mpv as _mpv_module  # the bare stub from conftest


class _EndFileEvent:
    def __init__(self, reason: int) -> None:
        self.data = type("EndFileData", (), {"reason": reason})()


class _FakeMpv:
    """Capable fake supporting event_callback, observe_property, command,
    play/stop, and option assignment. Records every command + every
    explicit volume write."""

    def __init__(self, **_kwargs) -> None:
        self.options: Dict[str, Any] = {}
        self.volume_history: List[float] = []
        self.pause = False
        self._event_callbacks: Dict[str, Any] = {}
        self._property_observers: Dict[str, Any] = {}
        self.commands: List[Tuple[Any, ...]] = []
        self.plays: List[str] = []

    def __setitem__(self, key, value) -> None:
        self.options[key] = value

    def __setattr__(self, name, value) -> None:
        if name == "volume":
            self.volume_history.append(float(value))
        super().__setattr__(name, value)

    def event_callback(self, name: str):
        def deco(fn):
            self._event_callbacks[name] = fn
            return fn

        return deco

    def observe_property(self, name: str, callback) -> None:
        self._property_observers[name] = callback

    def play(self, url: str) -> None:
        self.plays.append(url)

    def stop(self) -> None:
        pass

    def command(self, *args) -> None:
        self.commands.append(tuple(args))

    # -- helpers for tests --

    def fire_end_file(self, reason: int) -> None:
        cb = self._event_callbacks.get("end-file")
        assert cb is not None, "end-file callback not registered"
        cb(_EndFileEvent(reason))

    def fire_time_pos(self, value: float) -> None:
        cb = self._property_observers.get("time-pos")
        assert cb is not None, "time-pos observer not registered"
        cb("time-pos", value)


@pytest.fixture
def libmpv_module(monkeypatch):
    monkeypatch.setattr(_mpv_module, "MPV", _FakeMpv, raising=False)
    from linux_voice_assistant.player import libmpv

    return libmpv


@pytest.fixture
def mpv_player_module(monkeypatch):
    monkeypatch.setattr(_mpv_module, "MPV", _FakeMpv, raising=False)
    from linux_voice_assistant import mpv_player

    return mpv_player


# -- LibMpvPlayer: per-role recovery ------------------------------------------


def test_tts_role_track_error_fires_done_callback_silently(libmpv_module):
    p = libmpv_module.LibMpvPlayer(role="tts")
    captured: List[str] = []
    p.play("file:///tmp/reply.wav", done_callback=lambda: captured.append("fired"))
    p._mpv.commands.clear()

    p._mpv.fire_end_file(reason=4)

    assert captured == ["fired"], "tts must invoke captured done_callback on track error"
    assert p._mpv.commands == [], "tts must NOT reload — silent recovery"


def test_chime_role_track_error_fires_done_callback_silently(libmpv_module):
    p = libmpv_module.LibMpvPlayer(role="chime")
    captured: List[str] = []
    p.play("file:///tmp/ping.ogg", done_callback=lambda: captured.append("fired"))
    p._mpv.commands.clear()

    p._mpv.fire_end_file(reason=4)
    assert captured == ["fired"]
    assert p._mpv.commands == []


def test_media_role_track_error_reloads_with_start_position(libmpv_module):
    p = libmpv_module.LibMpvPlayer(role="media")
    captured: List[str] = []
    p.play("https://stream/song.mp3", done_callback=lambda: captured.append("fired"))
    # Simulate playback progressing.
    p._mpv.fire_time_pos(42.5)
    p._mpv.commands.clear()

    p._mpv.fire_end_file(reason=4)

    assert len(p._mpv.commands) == 1
    cmd = p._mpv.commands[0]
    assert cmd[0] == "loadfile"
    assert cmd[1] == "https://stream/song.mp3"
    assert cmd[2] == "replace"
    assert cmd[3] == "start=42.50"
    # done_callback must NOT fire — playback is being recovered, not finished.
    assert captured == []


def test_media_role_track_error_with_no_url_fires_callback_so_channel_not_stuck(libmpv_module):
    p = libmpv_module.LibMpvPlayer(role="media")
    captured: List[str] = []
    # Wire a callback by injecting it directly — error fires before any play().
    p._done_callback = lambda: captured.append("fired")
    p._mpv.fire_end_file(reason=4)
    assert captured == ["fired"]
    assert p._mpv.commands == []


def test_alarm_role_track_error_reloads_with_loop_at_full_volume(libmpv_module):
    p = libmpv_module.LibMpvPlayer(role="alarm")
    p.play("file:///usr/share/sounds/alarm.ogg")
    p._mpv.commands.clear()
    p._mpv.volume_history.clear()

    p._mpv.fire_end_file(reason=4)

    assert len(p._mpv.commands) == 1
    cmd = p._mpv.commands[0]
    assert cmd[:3] == ("loadfile", "file:///usr/share/sounds/alarm.ogg", "replace")
    assert cmd[3] == "loop-playlist=inf"
    assert p._mpv.volume_history[-1] == pytest.approx(100.0)


def test_eof_reason_zero_path_still_works_unchanged(libmpv_module):
    """Regression: reason=0 (eof) is the happy path — must still fire
    the callback exactly once and clear it."""
    p = libmpv_module.LibMpvPlayer(role="media")
    captured: List[str] = []
    p.play("file:///tmp/song.mp3", done_callback=lambda: captured.append("fired"))

    p._mpv.fire_end_file(reason=0)
    assert captured == ["fired"]
    # Subsequent non-eof events must not re-fire.
    p._mpv.fire_end_file(reason=1)
    assert captured == ["fired"]


def test_non_eof_non_error_end_file_is_ignored(libmpv_module):
    """Reason 1 (stop), 2 (abort), 3 (quit) must not trigger recovery."""
    p = libmpv_module.LibMpvPlayer(role="media")
    p.play("https://stream/song.mp3")
    p._mpv.commands.clear()
    for reason in (1, 2, 3):
        p._mpv.fire_end_file(reason=reason)
    assert p._mpv.commands == []


def test_observe_property_only_registered_for_media_role(libmpv_module):
    media = libmpv_module.LibMpvPlayer(role="media")
    tts = libmpv_module.LibMpvPlayer(role="tts")
    alarm = libmpv_module.LibMpvPlayer(role="alarm")
    assert "time-pos" in media._mpv._property_observers
    assert "time-pos" not in tts._mpv._property_observers
    assert "time-pos" not in alarm._mpv._property_observers


def test_time_pos_observer_ignores_none_and_garbage(libmpv_module):
    p = libmpv_module.LibMpvPlayer(role="media")
    p.play("https://stream/song.mp3")
    p._on_time_pos_changed("time-pos", None)
    assert p._last_position == 0.0
    p._on_time_pos_changed("time-pos", "not-a-float")
    assert p._last_position == 0.0
    p._on_time_pos_changed("time-pos", 12.3)
    assert p._last_position == pytest.approx(12.3)


# -- MpvMediaPlayer: heartbeat demotion ladder --------------------------------


def test_channel_status_starts_ok_and_demotes_after_three_errors(mpv_player_module):
    player = mpv_player_module.MpvMediaPlayer(role="media")
    player._player.play("https://stream/song.mp3")
    assert player.channel_status == "ok"

    for _ in range(2):
        player._player._mpv.fire_end_file(reason=4)
    assert player.channel_status == "ok"

    player._player._mpv.fire_end_file(reason=4)
    assert player.channel_status == "degraded"


def test_channel_status_demotes_to_dead_after_six_errors(mpv_player_module):
    player = mpv_player_module.MpvMediaPlayer(role="alarm")
    player._player.play("file:///alarm.ogg")
    for _ in range(6):
        player._player._mpv.fire_end_file(reason=4)
    assert player.channel_status == "dead"


def test_channel_status_recovers_after_30s_window_clears(mpv_player_module):
    player = mpv_player_module.MpvMediaPlayer(role="media")
    # Synthesize older events outside the 30s window.
    now = time.time()
    player._respawn_events = [now - 60.0, now - 50.0, now - 45.0]
    # record_respawn trims old entries; only the new one counts.
    count = player.record_respawn(now)
    assert count == 1


def test_on_track_error_callback_is_role_aware(mpv_player_module):
    """Each role wires its own on_track_error → record_respawn callback,
    so a media error and an alarm error increment independent counters."""
    media = mpv_player_module.MpvMediaPlayer(role="media")
    alarm = mpv_player_module.MpvMediaPlayer(role="alarm")
    media._player.play("https://stream/song.mp3")
    alarm._player.play("file:///alarm.ogg")
    for _ in range(3):
        media._player._mpv.fire_end_file(reason=4)
    assert media.channel_status == "degraded"
    assert alarm.channel_status == "ok"
