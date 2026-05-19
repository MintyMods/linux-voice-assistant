# mpv_player.py
import logging
from typing import Callable, List, Optional, Union

from .gen_check import gen_independent
from .player.libmpv import LibMpvPlayer
from .player.state import PlayerState


class MpvMediaPlayer:
    """
    Linux Voice Assistant MediaPlayer implementation based on libmpv.

    This class provides the MediaPlayer interface expected by LVA and
    delegates all playback logic to LibMpvPlayer.
    """

    def __init__(self, device: str | None = None, role: str = "media") -> None:
        self._log = logging.getLogger(self.__class__.__name__)
        self.role = role
        self._player = LibMpvPlayer(device=device, role=role)
        self._done_callback: Optional[Callable[[], None]] = None
        self._playlist: List[str] = []
        # Stage F5 — per-channel supervisor metadata (G1 + H5).
        # `channel_status` is read by the K.2 heartbeat publisher; the
        # F5 follow-up adds the respawn-on-crash policy that drives it
        # (silent for tts/chime, position-resume for media, immediate-
        # resume-full-volume for alarm). For now `ok` baseline + manual
        # demote via `mark_channel_degraded` / `_dead` keeps the heartbeat
        # honest while the actual content-recovery is staged in.
        self.channel_status: str = "ok"
        self._respawn_events: List[float] = []

        self._log.debug("MpvMediaPlayer initialized (device=%s, role=%s)", device, role)

    def mark_channel_degraded(self) -> None:
        """Mark this channel as degraded (heartbeat surface). Idempotent."""
        if self.channel_status != "dead":
            self.channel_status = "degraded"

    def mark_channel_dead(self) -> None:
        self.channel_status = "dead"

    def mark_channel_ok(self) -> None:
        self.channel_status = "ok"

    def record_respawn(self, ts: float) -> int:
        """Append a respawn timestamp and return rolling 30s count.

        Per H5 §3 mpv supervisor policy: >3 respawns in 30s → degraded.
        Caller decides what to do with the count; supervisor demotes
        `channel_status` accordingly.
        """
        cutoff = ts - 30.0
        self._respawn_events = [t for t in self._respawn_events if t >= cutoff]
        self._respawn_events.append(ts)
        return len(self._respawn_events)

    def play(
        self,
        url: Union[str, List[str]],
        done_callback: Optional[Callable[[], None]] = None,
        stop_first: bool = False,
    ) -> None:
        """
        Play a media URL.

        Args:
            url: Media URL or list of URLs for sequential playback.
            done_callback: Optional callback invoked when playback finishes.
            stop_first: Kept for API compatibility.
        """
        # Handle single URL vs list
        if isinstance(url, str):
            urls = [url]
        else:
            urls = list(url)  # Copy the list

        if not urls:
            self._log.warning("play() called with empty URL list")
            return

        # Track is changing - stop if needed
        if self._done_callback is not None:
            if self._player.state() != PlayerState.IDLE:
                self._log.debug("Stopping active playback before starting new media")
                self._player.stop(for_replacement=True)
            self._done_callback = None

        self._log.info("Playing %d URL(s): %s", len(urls), urls[0])

        # Store playlist and callback
        self._playlist = urls
        self._done_callback = done_callback

        # Start playing first URL
        next_url = self._playlist.pop(0)
        self._player.play(next_url, done_callback=self._on_track_finished, stop_first=stop_first)

    @gen_independent
    def _on_track_finished(self) -> None:
        """Called when a track finishes - plays next or invokes done callback.

        Per H3 §exception: MediaPlayer is gen-independent; music plays
        across voice sessions and is not invalidated by a voice cancel."""
        if self._playlist:
            # More tracks to play
            next_url = self._playlist.pop(0)
            self._log.debug("Playing next URL from playlist: %s", next_url)
            self._player.play(next_url, done_callback=self._on_track_finished, stop_first=False)
        else:
            # Playlist finished
            callback = self._done_callback
            self._done_callback = None

            if callback:
                self._log.debug("Playlist finished, invoking done_callback")
                try:
                    callback()
                except Exception as e:
                    self._log.exception("Error in done_callback: %s", e)

    def pause(self) -> None:
        """Pause playback."""
        self._log.debug("pause() called")
        self._player.pause()

    def resume(self) -> None:
        """Resume playback."""
        self._log.debug("resume() called")
        self._player.resume()

    def stop(self) -> None:
        """Stop playback and invoke the done callback if present."""
        self._log.debug("stop() called")

        self._player.stop()

        if self._done_callback:
            self._log.debug("Invoking done_callback due to stop()")
            try:
                self._done_callback()
            finally:
                self._done_callback = None

    @property
    def is_playing(self) -> bool:
        """Check if the player is currently playing or paused."""
        state = self._player.state()
        return state in (PlayerState.PLAYING, PlayerState.PAUSED, PlayerState.LOADING)

    def set_volume(self, volume: float) -> None:
        """
        Set playback volume.

        Args:
            volume: Volume in percent (0.0-100.0).
        """
        self._log.debug("set_volume(volume=%.2f)", volume)
        self._player.set_volume(volume)

    def duck(self, factor: float = 0.3) -> None:
        """Ramp volume down to `factor` (default 0.3 per G2)."""
        self._log.debug("duck(factor=%.2f)", factor)
        self._player.duck(factor)

    def unduck(self) -> None:
        """Ramp volume back to full over the configured release time."""
        self._log.debug("unduck() called")
        self._player.unduck()

    def configure_duck_envelope(
        self,
        *,
        floor_pct: Optional[int] = None,
        attack_ms: Optional[int] = None,
        release_ms: Optional[int] = None,
    ) -> None:
        """Propagate G2 live-tunable values to the underlying mpv backend."""
        self._player.configure_duck_envelope(
            floor_pct=floor_pct,
            attack_ms=attack_ms,
            release_ms=release_ms,
        )
