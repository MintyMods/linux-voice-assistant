"""Stage F2 — K.2 heartbeat publisher (H2 L4 liveness signal).

Publishes `calisto/<room>/heartbeat` every `interval_s` (default 60s,
tunable via env / live-tunable per K.2). Payload schema mirrors K.2:

    {
      "ts": "...",
      "uptime_s": ...,
      "state": "IDLE",
      "generation": ...,
      "bridge_reachable": ...,
      "mic_active": ...,
      "hidraw_ok": ...,
      "mpv_channels": {"tts": "ok", ...},
      "wake_count_5m": ...,
      "cancel_count_5m": ...,
      "subsystems": {"wake_word": "ok", ...}
    }

The heartbeat is independent of state-watchdog (K.1). It speaks to "I am
alive AND my voice path is functional" — health-oriented payload that
drives L4 HA automation.

`bridge_reachable` is updated by a sidecar ping task (10s cadence, 3
strikes per H5 §4) calling `BridgeClient.health()`. Strikes are reset on
the first successful ping. When BridgeClient is not wired the field
mirrors False (bridge_url not configured → bridge isn't reachable).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from .ha_bridge import HABridge
    from .models import ServerState

_LOGGER = logging.getLogger(__name__)


_MIC_LIVE_S = 5.0   # mic_active = frame seen in last 5s
_HID_LIVE_S = 5.0   # hidraw_ok = listener saw a packet in last 5s
_BRIDGE_PING_INTERVAL_S = 10.0
_BRIDGE_PING_STRIKES = 3


class HeartbeatPublisher:
    """Owns the K.2 publish loop + the bridge-reachability ping sidecar."""

    def __init__(
        self,
        state: "ServerState",
        *,
        ha_bridge: "Optional[HABridge]",
        room: str,
        interval_s: int = 60,
        bridge_ping_interval_s: float = _BRIDGE_PING_INTERVAL_S,
        bridge_ping_strikes: int = _BRIDGE_PING_STRIKES,
    ) -> None:
        self._state = state
        self._ha_bridge = ha_bridge
        self._room = room
        self._interval_s = interval_s
        self._bridge_ping_interval_s = bridge_ping_interval_s
        self._bridge_ping_strikes = bridge_ping_strikes
        self._topic = f"calisto/{room}/heartbeat"
        self._start_ts = time.monotonic()
        self._publish_task: "Optional[asyncio.Task[None]]" = None
        self._ping_task: "Optional[asyncio.Task[None]]" = None
        self._stop = False
        self._bridge_reachable = False
        self._bridge_strikes = 0
        self._published_count = 0

    @property
    def topic(self) -> str:
        return self._topic

    @property
    def published_count(self) -> int:
        return self._published_count

    @property
    def bridge_reachable(self) -> bool:
        return self._bridge_reachable

    @property
    def interval_s(self) -> int:
        return self._interval_s

    def set_interval(self, value: int) -> None:
        """Update the heartbeat cadence live (N.2 heartbeat_interval_s).
        Takes effect after the next tick — the current sleep is honoured."""
        self._interval_s = max(15, min(300, int(value)))

    # ------------------------------------------------------------------ API

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        """Schedule the 60s publish loop + the bridge-ping sidecar on `loop`."""
        self._stop = False
        if self._publish_task is None or self._publish_task.done():
            self._publish_task = loop.create_task(self._publish_loop())
        if self._ping_task is None or self._ping_task.done():
            self._ping_task = loop.create_task(self._bridge_ping_loop())

    def stop(self) -> None:
        self._stop = True
        for t in (self._publish_task, self._ping_task):
            if t is not None and not t.done():
                t.cancel()
        self._publish_task = None
        self._ping_task = None

    def publish_once(self) -> bool:
        """Emit a single heartbeat now. Returns True on enqueued publish.

        Public so tests + supervisor recovery paths can force an out-of-band
        emit without waiting for the next 60s tick.
        """
        payload = self._build_payload()
        ha_bridge = self._ha_bridge
        if ha_bridge is None:
            _LOGGER.debug("Heartbeat: no HABridge wired; skipping publish")
            return False
        ok = bool(ha_bridge.publish(self._topic, json.dumps(payload), qos=0, retain=False))
        if ok:
            self._published_count += 1
        return ok

    # -------------------------------------------------------- internal

    async def _publish_loop(self) -> None:
        # Fire an immediate heartbeat at startup so HA's online-binary-sensor
        # transitions ON without waiting a full minute.
        try:
            self.publish_once()
        except Exception:
            _LOGGER.exception("Heartbeat: initial publish raised")
        try:
            while not self._stop:
                try:
                    await asyncio.sleep(self._interval_s)
                except asyncio.CancelledError:
                    break
                if self._stop:
                    break
                try:
                    self.publish_once()
                except Exception:
                    _LOGGER.exception("Heartbeat publish raised; continuing loop")
        finally:
            _LOGGER.debug("Heartbeat publish loop exiting")

    async def _bridge_ping_loop(self) -> None:
        """Update `bridge_reachable` every `bridge_ping_interval_s`.

        3 failed pings in a row → False. Any success → True + reset strikes.
        Gracefully degrades to False when no BridgeClient is wired.
        """
        while not self._stop:
            try:
                await asyncio.sleep(self._bridge_ping_interval_s)
            except asyncio.CancelledError:
                break
            if self._stop:
                break
            bc = getattr(self._state, "bridge_client", None)
            if bc is None:
                self._bridge_reachable = False
                self._bridge_strikes = self._bridge_ping_strikes
                continue
            try:
                result = await bc.health()
                if result.ok:
                    self._bridge_reachable = True
                    self._bridge_strikes = 0
                else:
                    self._bridge_strikes += 1
                    if self._bridge_strikes >= self._bridge_ping_strikes:
                        self._bridge_reachable = False
            except Exception:
                self._bridge_strikes += 1
                if self._bridge_strikes >= self._bridge_ping_strikes:
                    self._bridge_reachable = False
                _LOGGER.debug("BridgeClient.health raised; strike %d/%d",
                              self._bridge_strikes, self._bridge_ping_strikes)

    def _build_payload(self) -> dict:
        now = time.monotonic()
        uptime_s = int(now - self._start_ts)
        ts = datetime.now().astimezone().isoformat(timespec="milliseconds")

        ds = getattr(self._state, "device_session", None)
        if ds is not None:
            current_state = ds.state_value.value
            generation = ds.generation
        else:
            current_state = self._state.device_state or "IDLE"
            generation = self._state.generation

        mic_active = (now - (self._state.last_mic_frame_ts or 0.0)) < _MIC_LIVE_S
        hidraw_ok = self._hidraw_ok(now)

        return {
            "ts": ts,
            "uptime_s": uptime_s,
            "state": current_state,
            "generation": generation,
            "bridge_reachable": self._bridge_reachable,
            "mic_active": mic_active,
            "hidraw_ok": hidraw_ok,
            "mpv_channels": self._mpv_channel_health(),
            "wake_count_5m": self._wake_count_5m(now),
            "cancel_count_5m": self._cancel_count_5m(),
            "state_publishes_per_min": self._state_publishes_per_min(),
            "wake_arb_stats": self._wake_arb_stats(),
            "subsystems": self._subsystem_health(mic_active, hidraw_ok),
        }

    def _wake_arb_stats(self) -> dict:
        """Stage H J1 — fold the 24h won/lost/avg-margin counters into the
        heartbeat payload so the N.2 sensors can read them without owning
        a separate publish loop. Empty dict when no arbiter is wired."""
        arb = getattr(self._state, "wake_arbiter", None)
        if arb is None:
            return {"won_24h": 0, "lost_24h": 0, "avg_margin_24h": 0.0}
        try:
            return arb.stats_snapshot()
        except Exception:
            _LOGGER.exception("WakeArbiter.stats_snapshot raised")
            return {"won_24h": 0, "lost_24h": 0, "avg_margin_24h": 0.0}

    def _state_publishes_per_min(self) -> int:
        counter = getattr(self._state, "state_publish_counter", None)
        if counter is None:
            return 0
        try:
            return int(counter.count())
        except Exception:
            return 0

    def _mpv_channel_health(self) -> dict:
        """Per-channel mpv health: ok | degraded | dead.

        Stage F2 baseline returns `ok` if the player object exists. Stage F5
        per-channel supervisor will set `degraded` / `dead` based on respawn
        history.
        """
        result: dict = {}
        for name, attr in (
            ("tts", "tts_player"),
            ("chime", "chime_player"),
            ("media", "music_player"),
            ("alarm", "alarm_player"),
        ):
            player = getattr(self._state, attr, None)
            if player is None:
                result[name] = "dead"
                continue
            status = getattr(player, "channel_status", None)
            if isinstance(status, str):
                result[name] = status
            else:
                result[name] = "ok"
        return result

    def _hidraw_ok(self, now: float) -> bool:
        led = getattr(self._state, "led_controller", None)
        if led is None:
            return False
        # LedController exposes its HID listener's last-read timestamp via
        # `last_hid_event_ts` when wired (Stage F1 addition).
        last = getattr(led, "last_hid_event_ts", None)
        if last is None or not isinstance(last, (int, float)) or last <= 0:
            # No event yet — fall back to "controller exists" liveness so a
            # quiet host doesn't trip OFFLINE; the HID listener thread itself
            # is the load-bearing assertion (process exit kills heartbeat).
            return True
        return (now - float(last)) < _HID_LIVE_S

    def _wake_count_5m(self, now: float) -> int:
        events = getattr(self._state, "wake_events", None)
        if not events:
            return 0
        cutoff = now - 300.0
        # In-place trim so the list doesn't grow unbounded on a chatty host.
        while events and events[0] < cutoff:
            events.pop(0)
        return len(events)

    def _cancel_count_5m(self) -> int:
        coord = getattr(self._state, "cancel_coordinator", None)
        if coord is None:
            return 0
        try:
            return coord.recent_cancel_count(window_s=300.0)
        except Exception:
            return 0

    def _subsystem_health(self, mic_active: bool, hidraw_ok: bool) -> dict:
        """Per-subsystem ok | degraded | failed snapshot."""
        return {
            "wake_word": "ok" if mic_active else "degraded",
            "asr_client": "ok" if getattr(self._state, "asr_client", None) is not None else "failed",
            "tts_engine": "ok" if getattr(self._state, "tts_output", None) is not None else "failed",
            "led_hid": "ok" if hidraw_ok else "degraded",
        }
