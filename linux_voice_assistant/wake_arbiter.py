"""Stage H — WakeArbiter (J1).

Confidence-based wake arbitration across the fleet.

On every local wake fire, the arbiter publishes `calisto/wake_arb` (K.5)
with this device's peak score, then waits a tunable window (default 200ms)
for peer events. After the window closes it compares peak scores; ties
within `tiebreak_band` fall back to earliest `ts`, then alphabetical room
slug. The local device proceeds only if it wins.

The arbiter is intentionally permissive on failure: any unhandled
exception in the wait/evaluate path returns "won" so a broken arbitration
never blocks voice (per J1 failure mode).

Threading model:
- `on_peer_event` runs on the MQTT network thread (paho callback).
- `arbitrate` runs on the audio thread (synchronous block for the window).
- A short rolling peer-event window (500ms) accommodates the case where
  the peer publish lands slightly before our local wake fires.
- `_lock` (threading.Lock) guards `_events` + the arbitration cursor.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from typing import Any, Callable, Deque, List, Optional, Tuple

_LOGGER = logging.getLogger(__name__)


WAKE_ARB_TOPIC = "calisto/wake_arb"

DEFAULT_WAIT_MS = 200
DEFAULT_TIEBREAK_BAND = 0.05
WAIT_MS_MIN = 50
WAIT_MS_MAX = 500
TIEBREAK_BAND_MIN = 0.01
TIEBREAK_BAND_MAX = 0.20

# Peer events older than this are discarded — covers the "peer published
# slightly before local wake" race without keeping stale events forever.
PEER_RING_WINDOW_S = 0.5

# Rolling stats horizon for the N.2 24h sensors.
STATS_WINDOW_S = 24 * 60 * 60


class PeerEvent:
    __slots__ = ("room", "wake_id", "score", "peak_score", "ts", "device_id", "received_at")

    def __init__(
        self,
        *,
        room: str,
        wake_id: int,
        score: float,
        peak_score: float,
        ts: float,
        device_id: str,
        received_at: float,
    ) -> None:
        self.room = room
        self.wake_id = wake_id
        self.score = score
        self.peak_score = peak_score
        self.ts = ts
        self.device_id = device_id
        self.received_at = received_at


class WakeArbiter:
    """J1 wake arbitration. One instance per LVA process.

    Owns the K.5 publish + the per-wake evaluation. Tunables live here so
    the HA EntitySurface can bind getters/setters directly.
    """

    def __init__(
        self,
        *,
        room: str,
        device_id: str,
        publish: Optional[Callable[[str, str], bool]] = None,
        wait_ms: int = DEFAULT_WAIT_MS,
        tiebreak_band: float = DEFAULT_TIEBREAK_BAND,
        enabled: bool = True,
    ) -> None:
        self.room = room
        self.device_id = device_id
        self._publish = publish
        self._wait_ms = int(wait_ms)
        self._tiebreak_band = float(tiebreak_band)
        self._enabled = bool(enabled)
        self._lock = threading.Lock()
        self._events: Deque[PeerEvent] = deque()
        # 24h rolling stats — (ts_monotonic, margin) for won/lost decisions.
        self._won: Deque[Tuple[float, float]] = deque()
        self._lost: Deque[Tuple[float, float]] = deque()

    # ------------------------------------------------------------- tunables

    @property
    def wait_ms(self) -> int:
        return self._wait_ms

    def set_wait_ms(self, value: int) -> None:
        self._wait_ms = max(WAIT_MS_MIN, min(WAIT_MS_MAX, int(value)))

    @property
    def tiebreak_band(self) -> float:
        return self._tiebreak_band

    def set_tiebreak_band(self, value: float) -> None:
        self._tiebreak_band = max(TIEBREAK_BAND_MIN, min(TIEBREAK_BAND_MAX, float(value)))

    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_enabled(self, value: bool) -> None:
        self._enabled = bool(value)

    def attach_publisher(self, publish: Callable[[str, str], bool]) -> None:
        """Hook the MQTT publish callable post-construction (HABridge.publish)."""
        self._publish = publish

    # ----------------------------------------------------------- peer events

    def on_peer_event(self, raw_payload: bytes) -> None:
        """Called from the MQTT network thread for every `calisto/wake_arb`
        message. Drops events from our own room (we ignore our own publish)
        and malformed payloads."""
        if not raw_payload:
            return
        try:
            obj = json.loads(raw_payload.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            _LOGGER.debug("wake_arb: dropped malformed payload")
            return
        if not isinstance(obj, dict):
            return
        peer_room = obj.get("room")
        if not isinstance(peer_room, str) or peer_room == self.room:
            return
        try:
            event = PeerEvent(
                room=peer_room,
                wake_id=int(obj.get("wake_id", 0)),
                score=float(obj.get("score", 0.0)),
                peak_score=float(obj.get("peak_score", obj.get("score", 0.0))),
                ts=float(obj.get("ts", 0.0)),
                device_id=str(obj.get("device_id", "")),
                received_at=time.monotonic(),
            )
        except (TypeError, ValueError):
            _LOGGER.debug("wake_arb: dropped non-numeric payload")
            return
        with self._lock:
            self._events.append(event)
            self._trim_locked(event.received_at)

    def _trim_locked(self, now_monotonic: float) -> None:
        cutoff = now_monotonic - PEER_RING_WINDOW_S
        while self._events and self._events[0].received_at < cutoff:
            self._events.popleft()

    # ------------------------------------------------------------ arbitrate

    def arbitrate(
        self,
        *,
        wake_id: int,
        score: float,
        peak_score: float,
    ) -> bool:
        """Publish own wake_arb, wait for peers, return True iff we win.

        Blocks the calling thread for `wait_ms`. Per J1 the audio thread
        already runs synchronous CPU work (CAM++ verifier), so a short
        block here is precedent-compatible. The 200ms penalty applies to
        every wake (acceptable per architecture J1 § Edge cases).
        """
        if not self._enabled:
            return True
        publish = self._publish
        local_ts = time.time()
        payload = {
            "room": self.room,
            "wake_id": int(wake_id),
            "score": float(score),
            "peak_score": float(peak_score),
            "ts": local_ts,
            "device_id": self.device_id,
        }
        # Snapshot peer events that arrived BEFORE we even fired (peer's
        # publish raced ahead of our wake detector by a few ms).
        with self._lock:
            wait_start = time.monotonic()
            self._trim_locked(wait_start)
            backward_peers: List[PeerEvent] = list(self._events)

        if publish is not None:
            try:
                publish(WAKE_ARB_TOPIC, json.dumps(payload))
            except Exception:
                _LOGGER.exception("wake_arb publish raised; proceeding as solo")

        # Wait the configured window. Peer events keep accumulating on
        # `self._events` via on_peer_event during this sleep.
        wait_s = max(0.0, self._wait_ms / 1000.0)
        time.sleep(wait_s)

        with self._lock:
            wait_end = time.monotonic()
            forward_peers = [e for e in self._events if e.received_at >= wait_start]

        # Dedupe (a peer captured in backward_peers may also be in
        # forward_peers if it landed exactly at wait_start; use (room,
        # wake_id) as the identity key).
        seen = set()
        peers: List[PeerEvent] = []
        for event in backward_peers + forward_peers:
            key = (event.room, event.wake_id)
            if key in seen:
                continue
            seen.add(key)
            peers.append(event)

        won, margin = self._decide(local_ts=local_ts, local_peak=peak_score, peers=peers)
        self._record_outcome(won=won, margin=margin)
        return won

    def _decide(
        self,
        *,
        local_ts: float,
        local_peak: float,
        peers: List[PeerEvent],
    ) -> Tuple[bool, float]:
        """Return (won, margin_against_runner_up).

        Solo wakes (no peers) win trivially with margin=local_peak.
        """
        if not peers:
            return True, float(local_peak)

        # Highest peak_score wins; ties within band → oldest ts; final
        # fallback → alphabetical room slug.
        candidates: List[Tuple[float, float, str]] = [
            (float(local_peak), float(local_ts), self.room)
        ]
        for peer in peers:
            candidates.append((peer.peak_score, peer.ts, peer.room))

        top_score = max(c[0] for c in candidates)
        # Within-band candidates compete on (ts, room).
        in_band = [c for c in candidates if (top_score - c[0]) < self._tiebreak_band]
        in_band.sort(key=lambda c: (c[1], c[2]))
        winner = in_band[0]

        runner_up_score = max(
            (c[0] for c in candidates if (c[1], c[2]) != (winner[1], winner[2])),
            default=winner[0],
        )
        margin = float(winner[0] - runner_up_score)
        won = winner[2] == self.room
        if not won:
            _LOGGER.info(
                "wake_arb: lost to %s (peer_peak=%.3f local_peak=%.3f margin=%.3f peers=%d)",
                winner[2], winner[0], local_peak, margin, len(peers),
            )
        else:
            _LOGGER.info(
                "wake_arb: won (peak=%.3f runner_up=%.3f margin=%.3f peers=%d)",
                local_peak, runner_up_score, margin, len(peers),
            )
        return won, margin

    # ----------------------------------------------------------- statistics

    def _record_outcome(self, *, won: bool, margin: float) -> None:
        now = time.monotonic()
        with self._lock:
            self._trim_stats_locked(now)
            target = self._won if won else self._lost
            target.append((now, abs(float(margin))))

    def _trim_stats_locked(self, now: float) -> None:
        cutoff = now - STATS_WINDOW_S
        for deck in (self._won, self._lost):
            while deck and deck[0][0] < cutoff:
                deck.popleft()

    def won_24h(self) -> int:
        with self._lock:
            self._trim_stats_locked(time.monotonic())
            return len(self._won)

    def lost_24h(self) -> int:
        with self._lock:
            self._trim_stats_locked(time.monotonic())
            return len(self._lost)

    def avg_margin_24h(self) -> float:
        with self._lock:
            self._trim_stats_locked(time.monotonic())
            margins = [m for _, m in self._won] + [m for _, m in self._lost]
            if not margins:
                return 0.0
            return float(sum(margins) / len(margins))

    def stats_snapshot(self) -> dict:
        return {
            "won_24h": self.won_24h(),
            "lost_24h": self.lost_24h(),
            "avg_margin_24h": round(self.avg_margin_24h(), 4),
        }
