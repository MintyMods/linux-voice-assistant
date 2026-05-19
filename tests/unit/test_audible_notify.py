"""Stage E.2 E6 — audible-notify arbiter + chime controller tests.

Pure-logic tests: the arbiter takes two callables (state getter +
alarm-ringing getter), so we can drive every priority case without
mocking mpv or session machinery.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from linux_voice_assistant.audible_notify import (
    AudibleNotifyArbiter,
    ChimeController,
    resolve_chime_sound,
    resolve_alarm_sound,
)
from linux_voice_assistant.session import State


@pytest.fixture
def arbiter_factory():
    def make(*, state: State = State.IDLE, alarm: bool = False, enabled: bool = True):
        return AudibleNotifyArbiter(
            state_getter=lambda: state,
            alarm_ringing_getter=lambda: alarm,
            enabled=enabled,
        )

    return make


class TestArbiterChime:
    def test_chime_allowed_at_idle_no_alarm_enabled(self, arbiter_factory):
        arb = arbiter_factory(state=State.IDLE, alarm=False)
        assert arb.allow_chime() is True

    def test_chime_blocked_when_disabled(self, arbiter_factory):
        arb = arbiter_factory(enabled=False)
        assert arb.allow_chime() is False

    def test_chime_blocked_during_alarm(self, arbiter_factory):
        arb = arbiter_factory(state=State.IDLE, alarm=True)
        assert arb.allow_chime() is False

    @pytest.mark.parametrize("state", [State.SPEAKING, State.THINKING])
    def test_chime_blocked_while_voice_busy(self, arbiter_factory, state):
        arb = arbiter_factory(state=state, alarm=False)
        assert arb.allow_chime() is False

    @pytest.mark.parametrize(
        "state",
        [State.WAKING, State.LISTENING, State.FOLLOWUP, State.IDLE],
    )
    def test_chime_allowed_in_non_voice_busy_states(self, arbiter_factory, state):
        arb = arbiter_factory(state=state, alarm=False)
        assert arb.allow_chime() is True


class TestArbiterSayTts:
    def test_say_tts_blocked_during_alarm(self, arbiter_factory):
        arb = arbiter_factory(alarm=True)
        assert arb.allow_say_tts() is False

    def test_say_tts_allowed_when_no_alarm(self, arbiter_factory):
        arb = arbiter_factory(state=State.SPEAKING, alarm=False)
        # TTS during SPEAKING is permitted — barge-in handles the existing
        # voice session, the arbiter only enforces alarm precedence.
        assert arb.allow_say_tts() is True


class TestResolveSound:
    def test_resolve_chime_returns_none_for_unknown(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CHIME_SOUNDS_DIR", str(tmp_path))
        assert resolve_chime_sound("nonexistent.ogg") is None
        assert resolve_chime_sound("") is None
        assert resolve_chime_sound(None) is None

    def test_resolve_chime_finds_file_in_env_dir(self, monkeypatch, tmp_path):
        sound = tmp_path / "Ping.ogg"
        sound.write_bytes(b"fake-ogg")
        monkeypatch.setenv("CHIME_SOUNDS_DIR", str(tmp_path))
        assert resolve_chime_sound("Ping.ogg") == str(sound)

    def test_resolve_chime_accepts_absolute_path(self, tmp_path):
        sound = tmp_path / "Bell.ogg"
        sound.write_bytes(b"fake-ogg")
        assert resolve_chime_sound(str(sound)) == str(sound)

    def test_resolve_alarm_falls_back_to_first_file_in_dir(self, monkeypatch, tmp_path):
        existing = tmp_path / "DefaultRing.ogg"
        existing.write_bytes(b"fake-ogg")
        monkeypatch.setenv("ALARM_SOUNDS_DIR", str(tmp_path))
        # Requested name doesn't exist; should fall back to existing.
        result = resolve_alarm_sound("MissingTone.ogg")
        assert result == str(existing)

    def test_resolve_alarm_returns_name_when_no_files_anywhere(
        self, monkeypatch, tmp_path
    ):
        # tmp_path is empty; defaults likely missing on a Windows dev box.
        monkeypatch.setenv("ALARM_SOUNDS_DIR", str(tmp_path))
        result = resolve_alarm_sound("PassThrough.ogg")
        assert result == "PassThrough.ogg"


class TestChimeController:
    def test_play_invokes_player_when_allowed(self, arbiter_factory, monkeypatch, tmp_path):
        sound = tmp_path / "Wake.ogg"
        sound.write_bytes(b"fake-ogg")
        monkeypatch.setenv("CHIME_SOUNDS_DIR", str(tmp_path))

        player = MagicMock()
        chime = ChimeController(chime_player=player, arbiter=arbiter_factory())

        assert chime.play("Wake.ogg") is True
        player.play.assert_called_once_with(str(sound))

    def test_play_skipped_when_arbiter_blocks(self, arbiter_factory):
        player = MagicMock()
        chime = ChimeController(chime_player=player, arbiter=arbiter_factory(alarm=True))

        assert chime.play("Wake.ogg") is False
        player.play.assert_not_called()

    def test_play_skipped_when_sound_missing(self, arbiter_factory, monkeypatch, tmp_path):
        monkeypatch.setenv("CHIME_SOUNDS_DIR", str(tmp_path))
        player = MagicMock()
        chime = ChimeController(chime_player=player, arbiter=arbiter_factory())

        assert chime.play("nonexistent.ogg") is False
        player.play.assert_not_called()

    def test_play_swallows_player_exceptions(self, arbiter_factory, monkeypatch, tmp_path):
        sound = tmp_path / "Wake.ogg"
        sound.write_bytes(b"fake-ogg")
        monkeypatch.setenv("CHIME_SOUNDS_DIR", str(tmp_path))

        player = MagicMock()
        player.play.side_effect = RuntimeError("mpv blew up")
        chime = ChimeController(chime_player=player, arbiter=arbiter_factory())

        # Must not raise — chime failures are best-effort.
        assert chime.play("Wake.ogg") is False
