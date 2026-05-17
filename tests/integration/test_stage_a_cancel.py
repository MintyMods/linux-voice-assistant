"""Stage A acceptance tests — §7.6 #1 (false-wake silent abort) and #4 (TTS interrupt).

These tests exercise the generation-counter contract added in Stage A:
every observable callback that survives a cancel captures the generation at
schedule-time and re-checks at fire-time. After stop() bumps the gen, every
stale callback must short-circuit.

We bypass VoiceSatelliteProtocol.__init__ to avoid the entity-bootstrap dance;
the gen-counter logic is independent of the ESPHome entity wiring.
"""

import threading
import time
from unittest.mock import MagicMock

from aioesphomeapi.api_pb2 import VoiceAssistantRequest  # type: ignore

from linux_voice_assistant.satellite import VoiceSatelliteProtocol


def _make_bare_satellite() -> VoiceSatelliteProtocol:
    sat = object.__new__(VoiceSatelliteProtocol)
    sat.state = MagicMock()
    sat.state.tts_player = MagicMock()
    sat.state.music_player = MagicMock()
    sat.state.stop_word = MagicMock()
    sat.state.stop_word.id = "stop_word_id"
    sat.state.active_wake_words = set()
    sat.state.muted = False
    sat.state.wakeup_sound = "wakeup.flac"
    sat.state.mqtt_client = None
    sat.state.mqtt_state_topic = None
    sat.state.generation = 0
    sat.state.last_cancel_reason = None
    sat.state.last_cancel_ts = None
    sat._generation = 0
    sat._gen_lock = threading.Lock()
    sat._pipeline_start_gen = 0
    sat._session_id = None
    sat._state_label = "IDLE"
    sat._last_state_change_ts = time.monotonic()
    sat._is_streaming_audio = False
    sat._tts_url = None
    sat._tts_played = False
    sat._continue_conversation = False
    sat._timer_finished = False
    sat._timer_ring_start = None
    sat._processing = False
    sat._pipeline_active = False
    sat._external_wake_words = {}
    sat._disconnect_event = MagicMock()
    sat.send_messages = MagicMock()
    return sat


def _capture_done_callbacks(tts_player_mock):
    captured = []

    def fake_play(*args, done_callback=None, **_kwargs):
        if done_callback is not None:
            captured.append(done_callback)

    tts_player_mock.play.side_effect = fake_play
    return captured


def _started_streams(send_messages_mock):
    """Return all VoiceAssistantRequest(start=True) submissions seen."""
    out = []
    for call in send_messages_mock.call_args_list:
        msgs = call.args[0] if call.args else []
        for m in msgs:
            if isinstance(m, VoiceAssistantRequest) and getattr(m, "start", False):
                out.append(m)
    return out


# ----------------------------------------------------------------------------
# §7.6 #1 — false wake then red button within 500ms → silent abort
# ----------------------------------------------------------------------------


def test_false_wake_then_red_button_silent_abort():
    """Wake fires → chime starts → red button at +200ms → chime done_callback
    fires AFTER the cancel and must NOT submit a streaming-start request."""
    sat = _make_bare_satellite()
    chime_callbacks = _capture_done_callbacks(sat.state.tts_player)

    wake_word = MagicMock()
    wake_word.wake_word = "okay_nabu"

    # Wake fires (the audio thread invokes this).
    sat.wakeup(wake_word)

    assert sat._pipeline_active is True
    assert sat._session_id is not None
    assert sat._state_label == "WAKING"
    gen_at_wake = sat._generation
    assert len(chime_callbacks) == 1, "wakeup() must schedule exactly one chime done_callback"

    # User slams the red button before the chime finishes. MQTT subscriber
    # eventually calls sat.stop("RED_BUTTON_SOFT") on the event loop thread.
    sat.stop(cancel_reason="RED_BUTTON_SOFT")

    assert sat._generation == gen_at_wake + 1, "stop() must bump _generation"
    assert sat._pipeline_active is False
    assert sat._session_id is None
    assert sat._state_label == "IDLE"
    assert sat.state.last_cancel_reason == "RED_BUTTON_SOFT"

    # Belatedly, mpv finishes the chime and invokes the captured done_callback.
    # This is the late-callback class of bug Stage A neutralises.
    chime_callbacks[0]()

    assert sat._is_streaming_audio is False, "Late chime callback wrongly enabled streaming after cancel"
    assert _started_streams(sat.send_messages) == [], "Late chime callback wrongly submitted a stream-start"


def test_two_false_wakes_in_succession():
    """§7.6 #10 — two false wakes 1s apart, both cancelled, no orphan listeners."""
    sat = _make_bare_satellite()
    chime_callbacks = _capture_done_callbacks(sat.state.tts_player)

    wake_word = MagicMock()
    wake_word.wake_word = "okay_nabu"

    sat.wakeup(wake_word)
    gen_1 = sat._generation
    sat.stop(cancel_reason="RED_BUTTON_SOFT")
    assert sat._generation == gen_1 + 1

    sat.wakeup(wake_word)
    gen_2 = sat._generation
    sat.stop(cancel_reason="RED_BUTTON_SOFT")
    assert sat._generation == gen_2 + 1

    # Fire BOTH stale chime callbacks. Neither must start a stream.
    assert len(chime_callbacks) == 2
    for cb in chime_callbacks:
        cb()

    assert _started_streams(sat.send_messages) == []
    assert sat._pipeline_active is False
    assert sat._is_streaming_audio is False


# ----------------------------------------------------------------------------
# §7.6 #4 — red button mid-TTS → TTS stops, late _tts_finished is dropped
# ----------------------------------------------------------------------------


def test_red_button_mid_tts_drops_late_finished_callback():
    """play_tts has scheduled _tts_finished with the in-flight gen captured.
    After stop() the captured gen is stale, so _tts_finished must short-circuit
    instead of transitioning to FOLLOWUP or sending VoiceAssistantAnnounceFinished."""
    sat = _make_bare_satellite()
    tts_callbacks = _capture_done_callbacks(sat.state.tts_player)

    # Simulate the state just after VOICE_ASSISTANT_TTS_END dispatched play_tts.
    sat._tts_url = "http://piper/say.wav"
    sat._pipeline_active = True
    sat._pipeline_start_gen = sat._generation
    sat._continue_conversation = True  # would have triggered FOLLOWUP

    sat.play_tts()

    assert sat._tts_played is True
    assert sat._state_label == "SPEAKING"
    assert len(tts_callbacks) == 1, "play_tts() must schedule exactly one _tts_finished callback"

    pre_cancel_send_count = sat.send_messages.call_count

    # Red button hits mid-playback.
    sat.stop(cancel_reason="RED_BUTTON_SOFT")
    assert sat._pipeline_active is False
    assert sat._state_label == "IDLE"

    # mpv's stop() invokes the captured done_callback shortly after cancel.
    tts_callbacks[0]()

    # _tts_finished should have been dropped:
    #   - no VoiceAssistantAnnounceFinished sent post-cancel
    #   - no FOLLOWUP transition (state stays IDLE, no new stream started)
    assert sat._state_label == "IDLE", "Late _tts_finished wrongly transitioned to FOLLOWUP"
    assert sat._is_streaming_audio is False, "Late _tts_finished wrongly restarted streaming"
    assert _started_streams(sat.send_messages) == [], "Late _tts_finished wrongly started follow-up stream"
    # send_messages may have been called by stop() to send stream-end, but
    # nothing additional should have happened AFTER the late callback ran.
    # (Conservative assertion: at most one new send for stream-end during stop.)
    assert sat.send_messages.call_count <= pre_cancel_send_count + 1


def test_play_tts_blocked_when_pipeline_start_gen_stale():
    """If TTS_END arrives after a cancel, _pipeline_start_gen is stale and
    play_tts must drop the URL instead of speaking."""
    sat = _make_bare_satellite()

    # Pipeline ran, gen captured at RUN_START.
    sat._tts_url = "http://piper/say.wav"
    sat._pipeline_active = True
    sat._pipeline_start_gen = sat._generation

    # Cancel happens before TTS_END dispatch.
    sat.stop(cancel_reason="DASHBOARD")

    # HA's TTS_END arrives late and play_tts is invoked.
    sat.play_tts()

    assert sat._tts_played is False, "play_tts must not start TTS after a cancel"
    assert sat._tts_url is None, "play_tts must drop the URL"
    assert sat.state.tts_player.play.call_count == 0, "tts_player.play must not be called after stale gen"
