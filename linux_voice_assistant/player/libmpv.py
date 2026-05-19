import logging
import threading
from typing import Any, Callable, Optional

import mpv

from linux_voice_assistant.gen_check import gen_independent
from linux_voice_assistant.player.base import AudioPlayer
from linux_voice_assistant.player.state import PlayerState


class LibMpvPlayer(AudioPlayer):
    """
    AudioPlayer implementation for Linux Voice Assistant using libmpv.

    Responsibilities:
    - mpv lifecycle and playback control
    - thread-safe state management
    - volume handling with ducking support
    """

    def __init__(
        self,
        device: Optional[str] = None,
        role: str = "media",
        on_track_error: Optional[Callable[[str], None]] = None,
    ) -> None:
        self._log = logging.getLogger(self.__class__.__name__)
        self.role = role
        self._state: PlayerState = PlayerState.IDLE
        self._state_lock = threading.Lock()

        # Stage F / H5 §3 — per-channel track-error recovery state.
        # `on_track_error` is invoked AFTER recovery so the owning
        # MpvMediaPlayer can update channel_status via record_respawn.
        # In-process libmpv can't be "respawned" (a real crash takes LVA
        # down and L1/L3 handle it); what we recover from here is mpv
        # `end-file reason=error` — track-level failures (404, codec,
        # transient I/O). The owner gets a single string arg describing
        # the role for logging convenience.
        self._on_track_error = on_track_error
        self._last_url: Optional[str] = None
        self._last_position: float = 0.0

        # Volume handling
        self._user_volume: float = 100.0  # 0.0 – 100.0
        self._duck_factor: float = 1.0  # 0.0 – 1.0

        # Stage E.2 G2 — ducking envelope. duck() now ramps linearly from the
        # current factor to the target over the configured attack ms; unduck()
        # ramps back to 1.0 over the release ms. A new call cancels any
        # pending ramp. Bounds match the live tunables in M.5.
        self._duck_thread: Optional[threading.Thread] = None
        self._duck_stop: threading.Event = threading.Event()
        self._duck_attack_ms: int = 150
        self._duck_release_ms: int = 300
        self._duck_floor_pct: int = 30  # used to clamp duck() callers passing >floor

        # mpv setup
        self._mpv = mpv.MPV(
            audio_display=False,
            log_handler=self._on_mpv_log,
            loglevel="error",
        )

        if device:
            self._mpv["audio-device"] = device

        # Pre-buffer audio before the sink starts clocking samples out.
        # The default (0.2 s) is too tight for short notification sounds on
        # PulseAudio/PipeWire: the sink stream takes a few ms to initialise
        # and the very first samples are dropped before it is ready, making
        # short files (<1 s) appear to start mid-way through.
        # 0.8 s gives the output pipeline enough headroom without adding any
        # noticeable latency for a user-facing notification sound.
        self._mpv["audio-buffer"] = 0.8

        # Keep the PulseAudio/PipeWire stream open between files by outputting
        # silence when idle.  This eliminates the per-play sink re-initialisation
        # penalty entirely, so back-to-back short sounds (wakeup → TTS, mute →
        # unmute) never lose their first samples regardless of system load.
        self._mpv["audio-stream-silence"] = True

        # Stage E.2 G4 — mpv-native source resilience for the MediaPlayer
        # channel only. TTS/chime/alarm play local files; reconnect logic is
        # irrelevant and the larger cache hurts first-audio latency.
        if role == "media":
            try:
                self._mpv["network-timeout"] = 10
                self._mpv["cache"] = "yes"
                self._mpv["cache-secs"] = 10
                self._mpv["stream-lavf-o"] = (
                    "reconnect=1,"
                    "reconnect_streamed=1,"
                    "reconnect_delay_max=30,"
                    "reconnect_on_network_error=1,"
                    "reconnect_on_http_error=4xx,5xx"
                )
                self._log.debug("Source-resilience options applied (role=media)")
            except Exception:
                # Old mpv builds may not accept every option; degrade gracefully
                # rather than refuse to start.
                self._log.warning("Failed to apply some source-resilience options", exc_info=True)

        # Callback Handling
        self._done_callback: Optional[Callable[[], None]] = None
        self._mpv.event_callback("end-file")(self._on_end_file)
        self._mpv.event_callback("start-file")(self._on_start_file)

        # H5 §3 media-resume — cache the most recent playback position so a
        # mid-track error can re-issue `loadfile <url> start=<pos>`. We only
        # observe for the media role; TTS/chime are short enough that
        # position-resume would land mid-syllable, and alarm always
        # restarts from the top.
        if role == "media":
            try:
                self._mpv.observe_property("time-pos", self._on_time_pos_changed)
            except Exception:
                self._log.warning(
                    "observe_property(time-pos) failed; media resume disabled",
                    exc_info=True,
                )

    # -------- Playback control --------

    def play(
        self,
        url: str,
        done_callback: Optional[Callable[[], None]] = None,
        stop_first: bool = True,
    ) -> None:
        """
        Start playback of a media URL.

        Args:
            url: Media URL or local file path.
            done_callback: Optional callback invoked when playback finishes.
            stop_first: If True, start playback in paused state.
        """
        with self._state_lock:
            self._log.debug("play: current_state=%s", self._state)
            self._done_callback = done_callback
            self._set_state(PlayerState.LOADING)
            self._last_url = url
            self._last_position = 0.0
        self._mpv.pause = stop_first
        self._mpv.play(url)

    def pause(self) -> None:
        """Pause playback."""
        with self._state_lock:
            self._mpv.pause = True
            self._set_state(PlayerState.PAUSED)

    def resume(self) -> None:
        """Resume playback if paused."""
        self._log.debug("unduck() called")
        with self._state_lock:
            self._mpv.pause = False
            self._set_state(PlayerState.PLAYING)

    def stop(self, for_replacement: bool = False) -> None:
        """
        Stop playback.

        If called for track replacement, clears the callback to prevent
        it from being invoked during the transition.
        """
        self._log.debug("unduck() called")
        with self._state_lock:
            if for_replacement:
                # Clear callback to prevent invocation during track transition
                self._done_callback = None
            self._mpv.stop()

    def state(self) -> PlayerState:
        """Return the current player state."""
        with self._state_lock:
            return self._state

    # -------- Volume / Ducking --------

    def set_volume(self, volume: float) -> None:
        """
        Set user volume.

        Args:
            volume: Volume level (0.0–100.0).
        """
        self._log.debug("unduck() called")
        with self._state_lock:
            self._user_volume = max(0.0, min(100.0, float(volume)))
            self._apply_volume()

    def duck(self, factor: float = 0.3) -> None:
        """Ramp volume down to `factor` over `_duck_attack_ms` (G2)."""
        target = max(0.0, min(1.0, float(factor)))
        self._log.debug("duck(target=%.2f, attack=%dms)", target, self._duck_attack_ms)
        self._start_ramp(target, self._duck_attack_ms)

    def unduck(self) -> None:
        """Ramp volume back to 1.0 over `_duck_release_ms` (G2)."""
        self._log.debug("unduck(release=%dms)", self._duck_release_ms)
        self._start_ramp(1.0, self._duck_release_ms)

    def configure_duck_envelope(
        self,
        *,
        floor_pct: Optional[int] = None,
        attack_ms: Optional[int] = None,
        release_ms: Optional[int] = None,
    ) -> None:
        """Update the live-tunable envelope params (G2 number entities)."""
        if floor_pct is not None:
            self._duck_floor_pct = max(0, min(100, int(floor_pct)))
        if attack_ms is not None:
            self._duck_attack_ms = max(0, int(attack_ms))
        if release_ms is not None:
            self._duck_release_ms = max(0, int(release_ms))

    def _start_ramp(self, target: float, duration_ms: int) -> None:
        """Cancel any in-flight ramp; start a new one toward `target`."""
        self._cancel_ramp()
        if duration_ms <= 0:
            with self._state_lock:
                self._duck_factor = target
                self._apply_volume()
            return

        with self._state_lock:
            start = self._duck_factor

        stop = threading.Event()
        self._duck_stop = stop
        thread = threading.Thread(
            target=self._ramp_run,
            args=(start, target, duration_ms, stop),
            name=f"mpv-duck-ramp-{self.role}",
            daemon=True,
        )
        self._duck_thread = thread
        thread.start()

    def _cancel_ramp(self) -> None:
        prev_stop = self._duck_stop
        prev_thread = self._duck_thread
        if prev_thread is not None and prev_thread.is_alive():
            prev_stop.set()

    def _ramp_run(self, start: float, target: float, duration_ms: int, stop: threading.Event) -> None:
        # 50 Hz update rate — smooth enough at the timescales we use
        # (150–1000 ms) and cheap enough that overlap with mpv's audio
        # thread is harmless.
        step_ms = 20
        steps = max(1, duration_ms // step_ms)
        delta = (target - start) / steps
        for i in range(1, steps + 1):
            if stop.wait(step_ms / 1000.0):
                # Pre-empted by another duck/unduck call. Leave the volume
                # wherever the new ramp will pick it up from.
                return
            with self._state_lock:
                self._duck_factor = start + delta * i
                self._apply_volume()
        # Snap to exact target on final step to avoid float drift.
        with self._state_lock:
            self._duck_factor = target
            self._apply_volume()

    # -------- Internal helpers --------

    def _apply_volume(self) -> None:
        """Apply effective volume (user volume × duck factor) to mpv."""
        self._log.debug("unduck() called")
        effective = self._user_volume * self._duck_factor
        self._mpv.volume = max(0.0, min(100.0, effective))

    @gen_independent
    def _on_end_file(self, event) -> None:
        callback: Optional[Callable[[], None]] = None
        is_error: bool = False

        with self._state_lock:
            # mpv events: event.data is a MpvEventEndFile object with a 'reason' attribute
            # The reason is an integer constant (see mpv.END_FILE_REASON_*)
            end_file_data = event.data
            reason = getattr(end_file_data, "reason", -1) if end_file_data else -1

            # mpv END_FILE_REASON constants:
            # 0 = eof (end of file), 1 = stop, 2 = abort, 3 = quit, 4 = error
            is_eof = reason == 0
            is_error = reason == 4

            self._log.debug(
                "_on_end_file: reason=%s (is_eof=%s, is_error=%s), state=%s, has_callback=%s",
                reason,
                is_eof,
                is_error,
                self._state,
                self._done_callback is not None,
            )

            if is_error:
                # H5 §3 — track-level error. Per-role recovery dispatched
                # outside the state lock; we capture the done_callback now
                # so a recovery reload that succeeds + completes still
                # delivers a single playback_finished signal upstream.
                self._set_state(PlayerState.IDLE)
                callback = self._done_callback
            elif is_eof:
                self._set_state(PlayerState.IDLE)
                callback = self._done_callback
                self._done_callback = None
            else:
                self._log.debug("_on_end_file: ignoring non-eof event (reason=%s)", reason)
                return

        if is_error:
            self._handle_track_error(callback)
            return

        if callback is not None:
            self._log.debug("_on_end_file: invoking callback")
            try:
                callback()
            except RuntimeError:
                # Callback errors must never break the player
                pass

    def _handle_track_error(self, captured_callback: Optional[Callable[[], None]]) -> None:
        """H5 §3 per-channel recovery for an `end-file reason=error` event.

        - tts / chime: silent — fire the captured done_callback so the
          DeviceSession lifecycle advances past SPEAKING / WAKING normally.
          The user gets silence rather than a stale state.
        - media: re-issue `loadfile <last_url> start=<last_position>` so
          the user hears "the music came back". When no last_url is
          available (error before any play) we fall through to firing the
          callback so the channel doesn't sit stuck in LOADING.
        - alarm: re-issue `loadfile <last_url> loop-playlist=inf` at full
          volume immediately. An alarm MUST NOT be silenced by a crash.

        Owner notification (`on_track_error`) always fires last so the
        heartbeat surface (channel_status / record_respawn) is updated
        regardless of which branch was taken.
        """
        role = self.role
        url = self._last_url
        position = self._last_position
        recovered = False

        try:
            if role in ("tts", "chime"):
                self._done_callback = None
                if captured_callback is not None:
                    try:
                        captured_callback()
                    except RuntimeError:
                        pass
            elif role == "media" and url:
                try:
                    start_arg = f"start={max(0.0, float(position)):.2f}"
                    self._mpv.command("loadfile", url, "replace", start_arg)
                    recovered = True
                    with self._state_lock:
                        self._set_state(PlayerState.LOADING)
                except Exception:
                    self._log.warning(
                        "media track-error recovery: loadfile raised; firing done_callback",
                        exc_info=True,
                    )
                    self._done_callback = None
                    if captured_callback is not None:
                        try:
                            captured_callback()
                        except RuntimeError:
                            pass
            elif role == "alarm" and url:
                try:
                    self._mpv.command("loadfile", url, "replace", "loop-playlist=inf")
                    try:
                        self._mpv.volume = 100.0
                    except Exception:
                        pass
                    recovered = True
                    with self._state_lock:
                        self._user_volume = 100.0
                        self._duck_factor = 1.0
                        self._set_state(PlayerState.LOADING)
                except Exception:
                    self._log.warning(
                        "alarm track-error recovery: loadfile raised; firing done_callback",
                        exc_info=True,
                    )
                    self._done_callback = None
                    if captured_callback is not None:
                        try:
                            captured_callback()
                        except RuntimeError:
                            pass
            else:
                # role is media/alarm but no url cached — nothing to reload.
                self._done_callback = None
                if captured_callback is not None:
                    try:
                        captured_callback()
                    except RuntimeError:
                        pass
        finally:
            self._log.info(
                "_handle_track_error: role=%s recovered=%s url=%s pos=%.2f",
                role, recovered, url, position,
            )
            cb = self._on_track_error
            if cb is not None:
                try:
                    cb(role)
                except Exception:
                    self._log.exception("on_track_error callback raised")

    @gen_independent
    def _on_time_pos_changed(self, _name: str, value: Any) -> None:
        """Cache the most recent playback position for media-role resume."""
        if value is None:
            return
        try:
            self._last_position = float(value)
        except (TypeError, ValueError):
            return

    @gen_independent
    def _on_start_file(self, event) -> None:
        """Called when mpv starts playing a file."""
        self._log.debug("unduck() called")
        with self._state_lock:
            self._log.debug("_on_start_file: state=%s", self._state)
            self._set_state(PlayerState.PLAYING)

    @gen_independent
    def _on_mpv_log(self, level: str, prefix: str, text: str) -> None:
        """
        Handle mpv log messages.

        Error and fatal messages transition the player into ERROR state.
        """
        if level in ("error", "fatal"):
            with self._state_lock:
                self._set_state(PlayerState.ERROR)

    def _set_state(self, new_state: PlayerState) -> None:
        """Update internal player state."""
        self._state = new_state
