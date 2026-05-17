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

from aioesphomeapi.api_pb2 import (  # type: ignore
    VoiceAssistantAnnounceFinished,
    VoiceAssistantAudio,
    VoiceAssistantRequest,
)

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
    sat._ha_pipeline_started = False
    sat._external_wake_words = {}
    sat._disconnect_event = MagicMock()
    sat.send_messages = MagicMock()
    return sat


def _sent_messages(send_messages_mock):
    """Flatten all messages passed across all send_messages() calls."""
    out = []
    for call in send_messages_mock.call_args_list:
        msgs = call.args[0] if call.args else []
        for m in msgs:
            out.append(m)
    return out


def _has_message_of(send_messages_mock, msg_type) -> bool:
    return any(isinstance(m, msg_type) for m in _sent_messages(send_messages_mock))


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


def _release_messages(send_messages_mock):
    """Return all VoiceAssistantRequest(start=False) submissions seen."""
    out = []
    for call in send_messages_mock.call_args_list:
        msgs = call.args[0] if call.args else []
        for m in msgs:
            if isinstance(m, VoiceAssistantRequest) and not getattr(m, "start", False):
                out.append(m)
    return out


def test_false_wake_then_red_button_silent_abort():
    """Wake fires → chime starts → red button at +200ms → chime done_callback
    fires AFTER the cancel and must NOT submit a streaming-start request."""
    sat = _make_bare_satellite()
    chime_callbacks = _capture_done_callbacks(sat.state.tts_player)

    wake_word = MagicMock()
    wake_word.wake_word = "okay_nabu"

    sat.wakeup(wake_word)

    assert sat._pipeline_active is True
    assert sat._session_id is not None
    assert sat._state_label == "WAKING"
    gen_at_wake = sat._generation
    assert len(chime_callbacks) == 1, "wakeup() must schedule exactly one chime done_callback"

    sat.stop(cancel_reason="RED_BUTTON_SOFT")

    assert sat._generation == gen_at_wake + 1, "stop() must bump _generation"
    assert sat._pipeline_active is False
    assert sat._session_id is None
    assert sat._state_label == "IDLE"
    assert sat.state.last_cancel_reason == "RED_BUTTON_SOFT"

    chime_callbacks[0]()

    assert sat._is_streaming_audio is False, "Late chime callback wrongly enabled streaming after cancel"
    assert _started_streams(sat.send_messages) == [], "Late chime callback wrongly submitted a stream-start"

    # stop() emits the two HA release messages unconditionally. Both are safe
    # no-ops on HA when no pipeline_run is in flight (Request(start=False)
    # routes to _abort_pipeline which handles _pipeline_task is None;
    # AnnounceFinished sets state=IDLE on an entity already idle). Unconditional
    # emission is what makes double-press / rapid cancel deterministic.
    assert _release_messages(sat.send_messages), \
        "stop() must always emit VoiceAssistantRequest(start=False)"
    assert _has_message_of(sat.send_messages, VoiceAssistantAnnounceFinished), \
        "stop() must always emit VoiceAssistantAnnounceFinished"


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

    sat._tts_url = "http://piper/say.wav"
    sat._pipeline_active = True
    sat._pipeline_start_gen = sat._generation
    sat._continue_conversation = True  # would have triggered FOLLOWUP

    sat.play_tts()

    assert sat._tts_played is True
    assert sat._state_label == "SPEAKING"
    assert len(tts_callbacks) == 1, "play_tts() must schedule exactly one _tts_finished callback"

    sat._ha_pipeline_started = True

    sat.stop(cancel_reason="RED_BUTTON_SOFT")
    assert sat._pipeline_active is False
    assert sat._state_label == "IDLE"
    assert sat._ha_pipeline_started is False, "stop() must clear _ha_pipeline_started"
    # Two-message HA release sequence — Request(start=False) cancels HA's
    # pipeline_task; AnnounceFinished forces tts_response_finished → IDLE.
    # Order matters: Request(start=False) MUST be sent first so no further
    # PipelineEvent (e.g. TTS_START) re-transitions state to RESPONDING after
    # AnnounceFinished sets it IDLE.
    assert _release_messages(sat.send_messages), \
        "stop() mid-TTS must send VoiceAssistantRequest(start=False) to cancel HA's pipeline_task"
    assert _has_message_of(sat.send_messages, VoiceAssistantAnnounceFinished), \
        "stop() mid-TTS must send AnnounceFinished to release HA's responding state"

    # mpv's stop() invokes the captured done_callback shortly after cancel.
    tts_callbacks[0]()

    # _tts_finished should have been dropped:
    #   - no second VoiceAssistantAnnounceFinished after the cancel
    #   - no FOLLOWUP transition (state stays IDLE, no new stream started)
    assert sat._state_label == "IDLE", "Late _tts_finished wrongly transitioned to FOLLOWUP"
    assert sat._is_streaming_audio is False, "Late _tts_finished wrongly restarted streaming"
    assert _started_streams(sat.send_messages) == [], "Late _tts_finished wrongly started follow-up stream"


def test_cancel_mid_thinking_emits_release_sequence():
    """User wakes, asks a question, presses red button while HA is THINKING.
    HA's pipeline_run is in flight on its side. stop() must send the two-
    message release sequence so HA's assist_satellite returns to idle and
    the Calisto LED resets."""
    sat = _make_bare_satellite()

    sat._pipeline_active = True
    sat._is_streaming_audio = False
    sat._ha_pipeline_started = True
    sat._pipeline_start_gen = sat._generation

    sat.stop(cancel_reason="RED_BUTTON_SOFT")

    assert sat._ha_pipeline_started is False
    assert _release_messages(sat.send_messages), \
        "stop() during THINKING must send VoiceAssistantRequest(start=False) to cancel HA's pipeline_task"
    assert _has_message_of(sat.send_messages, VoiceAssistantAnnounceFinished), \
        "stop() during THINKING must send AnnounceFinished to release HA"
    # No Audio(end=True) — that's the SOFT-stop path which lets HA's pipeline
    # run to completion, exactly the wrong direction on cancel.
    audio_msgs = [m for m in _sent_messages(sat.send_messages) if isinstance(m, VoiceAssistantAudio)]
    assert audio_msgs == [], "stop() must not send VoiceAssistantAudio (soft-stop) on cancel"


def test_cancel_mid_listening_emits_release_sequence():
    """User wakes, presses red button while still talking (LISTENING phase,
    mic streaming). stop() emits the same two-message release sequence —
    Request(start=False) is the hard-abort path that cancels HA's
    pipeline_task regardless of phase."""
    sat = _make_bare_satellite()

    sat._pipeline_active = True
    sat._is_streaming_audio = True
    sat._ha_pipeline_started = True
    sat._pipeline_start_gen = sat._generation

    sat.stop(cancel_reason="RED_BUTTON_SOFT")

    assert _release_messages(sat.send_messages), \
        "stop() during LISTENING must send VoiceAssistantRequest(start=False)"
    assert _has_message_of(sat.send_messages, VoiceAssistantAnnounceFinished), \
        "stop() during LISTENING must send AnnounceFinished"
    audio_msgs = [m for m in _sent_messages(sat.send_messages) if isinstance(m, VoiceAssistantAudio)]
    assert audio_msgs == [], "stop() must not send Audio(end=True) — Request(start=False) is the abort signal"


def test_repeat_cancel_emits_release_sequence_each_time():
    """Forced cancel (double-press) UX: each stop() call must emit the full
    release sequence. Previous bug: first stop() captured _ha_pipeline_started
    and cleared it; subsequent stops sent nothing on the wire, leaving HA
    stuck if the first AnnounceFinished raced with HA's TTS_START."""
    sat = _make_bare_satellite()

    sat._pipeline_active = True
    sat._is_streaming_audio = False
    sat._ha_pipeline_started = True
    sat._pipeline_start_gen = sat._generation

    sat.stop(cancel_reason="RED_BUTTON_SOFT")
    after_first = sat.send_messages.call_count

    # Forced cancel: second press immediately after.
    sat.stop(cancel_reason="RED_BUTTON_SOFT")
    after_second = sat.send_messages.call_count

    assert after_second > after_first, \
        "Second cancel must emit messages on the wire (regression of the silent repeat-cancel bug)"
    assert len(_release_messages(sat.send_messages)) >= 2, \
        "Each cancel press must emit VoiceAssistantRequest(start=False)"
    finishes = [m for m in _sent_messages(sat.send_messages) if isinstance(m, VoiceAssistantAnnounceFinished)]
    assert len(finishes) >= 2, "Each cancel press must emit AnnounceFinished"


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
