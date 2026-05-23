"""Stage E.2 G3 + K.10 — HABridge alarm/set + say routing tests.

Drives the bridge's _on_message directly with realistic K.8 and K.10
payloads, asserts that the right controller method was invoked. Lets us
verify the wire-level shape (topic → handler → controller method) without
spinning up paho.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from linux_voice_assistant.ha_bridge import HABridge


def _make_bridge(fake_paho):
    created, factory = fake_paho
    bridge = HABridge(
        room="living_room",
        host="mqtt.example",
        port=1883,
        client_factory=factory,
    )
    bridge.start()
    fake = created[-1]
    return bridge, fake


class _Msg:
    def __init__(self, topic: str, payload: bytes) -> None:
        self.topic = topic
        self.payload = payload


def test_alarm_set_routes_raw_payload_to_controller(fake_paho):
    bridge, fake = _make_bridge(fake_paho)
    alarm = MagicMock(name="alarm_controller")
    bridge.attach_audio_controllers(
        alarm=alarm, chime=None, tts_player=None, tts_output=None,
        arbiter=None, loop=None,
    )
    raw = json.dumps({"alarm_id": "x", "ringtone": "A.ogg"}).encode("utf-8")

    bridge._on_message(fake, None, _Msg(bridge.alarm_set_topic, raw))

    alarm.set_alarm.assert_called_once_with(raw)


def test_alarm_set_all_topic_routes_same_handler(fake_paho):
    bridge, fake = _make_bridge(fake_paho)
    alarm = MagicMock()
    bridge.attach_audio_controllers(
        alarm=alarm, chime=None, tts_player=None, tts_output=None,
        arbiter=None, loop=None,
    )
    raw = json.dumps({"alarm_id": "fleet"}).encode("utf-8")

    bridge._on_message(fake, None, _Msg(bridge.alarm_set_all_topic, raw))

    alarm.set_alarm.assert_called_once_with(raw)


def test_alarm_stop_topic_calls_stop_alarm(fake_paho):
    bridge, fake = _make_bridge(fake_paho)
    alarm = MagicMock()
    bridge.attach_audio_controllers(
        alarm=alarm, chime=None, tts_player=None, tts_output=None,
        arbiter=None, loop=None,
    )

    bridge._on_message(fake, None, _Msg(bridge.alarm_stop_topic, b""))

    alarm.stop_alarm.assert_called_once()


def test_alarm_set_without_attached_controller_is_dropped_silently(fake_paho):
    """A bridge constructed but never attached must NOT raise on inbound
    alarm messages — the controllers might not be wired yet at first
    connect, and a silent drop is the right degrade path."""
    bridge, fake = _make_bridge(fake_paho)
    bridge._on_message(fake, None, _Msg(bridge.alarm_set_topic, b'{"alarm_id":"x"}'))


def test_say_chime_scope_routes_through_chime_controller(fake_paho):
    bridge, fake = _make_bridge(fake_paho)
    chime = MagicMock()
    chime.play.return_value = True
    bridge.attach_audio_controllers(
        alarm=None, chime=chime, tts_player=None, tts_output=None,
        arbiter=None, loop=None,
    )
    raw = json.dumps({"text": "Ping.ogg", "scope": "chime"}).encode("utf-8")

    bridge._on_message(fake, None, _Msg(bridge.say_topic, raw))

    chime.play.assert_called_once_with("Ping.ogg")


def test_say_tts_scope_schedules_speak_via_loop(fake_paho):
    bridge, fake = _make_bridge(fake_paho)
    tts_output = MagicMock()
    # speak is async; provide a coroutine return.
    async def _speak(*_a, **_kw):
        return None
    tts_output.speak.side_effect = _speak
    tts_player = MagicMock()
    arbiter = MagicMock()
    arbiter.allow_say_tts.return_value = True

    loop = asyncio.new_event_loop()
    try:
        bridge.attach_audio_controllers(
            alarm=None, chime=None,
            tts_player=tts_player, tts_output=tts_output,
            arbiter=arbiter, loop=loop,
        )
        raw = json.dumps({"text": "Hello there.", "scope": "tts"}).encode("utf-8")

        bridge._on_message(fake, None, _Msg(bridge.say_topic, raw))

        # call_soon_threadsafe schedules a callback; one loop tick runs it.
        loop.run_until_complete(asyncio.sleep(0.05))

        # tts_output.speak called with the right text on the right player.
        tts_output.speak.assert_called_once()
        kwargs = tts_output.speak.call_args.kwargs
        args = tts_output.speak.call_args.args
        assert args[0] is tts_player
        assert kwargs.get("text") == "Hello there."
    finally:
        loop.close()


def test_say_tts_suppressed_by_arbiter_during_alarm(fake_paho):
    bridge, fake = _make_bridge(fake_paho)
    tts_output = MagicMock()
    tts_player = MagicMock()
    arbiter = MagicMock()
    arbiter.allow_say_tts.return_value = False

    bridge.attach_audio_controllers(
        alarm=None, chime=None,
        tts_player=tts_player, tts_output=tts_output,
        arbiter=arbiter, loop=asyncio.new_event_loop(),
    )

    raw = json.dumps({"text": "x", "scope": "tts"}).encode("utf-8")
    bridge._on_message(fake, None, _Msg(bridge.say_topic, raw))

    tts_output.speak.assert_not_called()


def test_say_empty_text_is_dropped(fake_paho):
    bridge, fake = _make_bridge(fake_paho)
    chime = MagicMock()
    bridge.attach_audio_controllers(
        alarm=None, chime=chime, tts_player=None, tts_output=None,
        arbiter=None, loop=None,
    )

    bridge._on_message(fake, None, _Msg(bridge.say_topic, b'{"text":"","scope":"chime"}'))
    bridge._on_message(fake, None, _Msg(bridge.say_topic, b'{"scope":"chime"}'))

    chime.play.assert_not_called()


def test_say_malformed_payload_dropped(fake_paho):
    bridge, fake = _make_bridge(fake_paho)
    chime = MagicMock()
    bridge.attach_audio_controllers(
        alarm=None, chime=chime, tts_player=None, tts_output=None,
        arbiter=None, loop=None,
    )

    bridge._on_message(fake, None, _Msg(bridge.say_topic, b"not-json"))
    bridge._on_message(fake, None, _Msg(bridge.say_topic, b"[1,2,3]"))  # not a dict

    chime.play.assert_not_called()


def test_say_all_topic_routes_same_handler(fake_paho):
    bridge, fake = _make_bridge(fake_paho)
    chime = MagicMock()
    chime.play.return_value = True
    bridge.attach_audio_controllers(
        alarm=None, chime=chime, tts_player=None, tts_output=None,
        arbiter=None, loop=None,
    )

    raw = json.dumps({"text": "Fleet.ogg", "scope": "chime"}).encode("utf-8")
    bridge._on_message(fake, None, _Msg(bridge.say_all_topic, raw))

    chime.play.assert_called_once_with("Fleet.ogg")


def test_attach_audio_controllers_wires_back_publisher_on_alarm(fake_paho):
    """The HABridge.attach_audio_controllers helper should retro-attach
    the bridge onto the alarm controller so K.9 publishes start working
    without manual wiring."""
    bridge, _fake = _make_bridge(fake_paho)
    alarm = MagicMock()

    bridge.attach_audio_controllers(
        alarm=alarm, chime=None, tts_player=None, tts_output=None,
        arbiter=None, loop=None,
    )

    alarm.attach_ha_bridge.assert_called_once_with(bridge)
