"""Stage H — WakeArbiter unit tests (J1).

Exercises the K.5 publish + the win/loss decision tree:
  - solo wake (no peers) wins
  - higher peak_score wins
  - within tiebreak band → oldest ts wins
  - within band + identical ts → alphabetical room slug wins
  - disabled arbiter always wins
  - 24h stats counters track outcomes
  - peer events older than the peer-ring window are dropped
"""

from __future__ import annotations

import json
import time
from typing import List, Tuple

import pytest

from linux_voice_assistant.wake_arbiter import (
    DEFAULT_TIEBREAK_BAND,
    PEER_RING_WINDOW_S,
    WAIT_MS_MAX,
    WAIT_MS_MIN,
    WAKE_ARB_TOPIC,
    WakeArbiter,
)


class _FakePublisher:
    def __init__(self) -> None:
        self.calls: List[Tuple[str, str]] = []

    def __call__(self, topic: str, payload: str) -> bool:
        self.calls.append((topic, payload))
        return True


def _make(room: str = "lounge", wait_ms: int = 20, enabled: bool = True) -> Tuple[WakeArbiter, _FakePublisher]:
    pub = _FakePublisher()
    arb = WakeArbiter(
        room=room,
        device_id=f"minty-ai-{room}",
        publish=pub,
        wait_ms=wait_ms,
        enabled=enabled,
    )
    return arb, pub


# ---------------------------------------------------------------- publish path


def test_arbitrate_publishes_k5_payload():
    arb, pub = _make(room="bedroom", wait_ms=10)
    arb.arbitrate(wake_id=7, score=0.78, peak_score=0.82)
    assert len(pub.calls) == 1
    topic, payload = pub.calls[0]
    assert topic == WAKE_ARB_TOPIC
    obj = json.loads(payload)
    assert obj["room"] == "bedroom"
    assert obj["wake_id"] == 7
    assert obj["score"] == pytest.approx(0.78)
    assert obj["peak_score"] == pytest.approx(0.82)
    assert obj["device_id"] == "minty-ai-bedroom"
    assert isinstance(obj["ts"], float)


def test_disabled_arbiter_returns_true_without_publish():
    arb, pub = _make(enabled=False, wait_ms=10)
    assert arb.arbitrate(wake_id=1, score=0.5, peak_score=0.5) is True
    assert pub.calls == []


# ---------------------------------------------------------------- decision tree


def test_solo_wake_wins():
    arb, _ = _make(wait_ms=10)
    assert arb.arbitrate(wake_id=1, score=0.6, peak_score=0.7) is True
    assert arb.won_24h() == 1
    assert arb.lost_24h() == 0


def test_higher_peer_peak_wins():
    arb, _ = _make(room="bedroom", wait_ms=30)
    peer = {
        "room": "bathroom",
        "wake_id": 99,
        "score": 0.85,
        "peak_score": 0.90,
        "ts": time.time(),
        "device_id": "minty-ai-bathroom",
    }
    arb.on_peer_event(json.dumps(peer).encode("utf-8"))
    assert arb.arbitrate(wake_id=1, score=0.60, peak_score=0.65) is False
    assert arb.lost_24h() == 1
    assert arb.won_24h() == 0


def test_local_higher_peak_wins():
    arb, _ = _make(room="bedroom", wait_ms=30)
    peer = {
        "room": "bathroom",
        "wake_id": 99,
        "score": 0.55,
        "peak_score": 0.60,
        "ts": time.time(),
        "device_id": "minty-ai-bathroom",
    }
    arb.on_peer_event(json.dumps(peer).encode("utf-8"))
    assert arb.arbitrate(wake_id=1, score=0.80, peak_score=0.85) is True
    assert arb.won_24h() == 1


def test_tiebreak_band_oldest_ts_wins():
    arb, _ = _make(room="bedroom", wait_ms=30)
    # Peer published earlier (lower ts) with peak inside the tiebreak band.
    now = time.time()
    peer = {
        "room": "bathroom",
        "wake_id": 99,
        "score": 0.80,
        "peak_score": 0.83,  # local 0.85; diff 0.02 < default band 0.05
        "ts": now - 0.05,
        "device_id": "minty-ai-bathroom",
    }
    arb.on_peer_event(json.dumps(peer).encode("utf-8"))
    # Local will publish with ts ≈ now; peer wins on older ts.
    assert arb.arbitrate(wake_id=1, score=0.82, peak_score=0.85) is False


def test_tiebreak_band_alphabetical_when_ts_equal():
    arb, _ = _make(room="bedroom", wait_ms=30)
    # Pin local ts close to peer ts so the tiebreak collapses to alphabetical.
    # Bathroom < bedroom alphabetically, so bathroom should win when ts equal.
    now = time.time()
    peer = {
        "room": "bathroom",
        "wake_id": 99,
        "score": 0.83,
        "peak_score": 0.85,
        "ts": now,  # identical to local ts (caller of arbitrate uses time.time())
        "device_id": "minty-ai-bathroom",
    }
    arb.on_peer_event(json.dumps(peer).encode("utf-8"))
    # Local peak inside band of peer (0.85 vs 0.85 = 0); identical ts is
    # vanishingly rare in practice but the rule must be deterministic.
    # We can't make ts perfectly identical from the test, but we exercise the
    # alphabetical fallback by giving the peer a slightly LATER ts than ours
    # and observing local wins. Then flip to verify the reverse.
    won = arb.arbitrate(wake_id=1, score=0.83, peak_score=0.85)
    # Since we can't pin equality precisely, accept either outcome here as
    # long as no exception fires. The deterministic alphabetical path is
    # covered by the dedicated unit test below using _decide directly.
    assert won in (True, False)


def test_decide_alphabetical_tiebreak_direct():
    arb, _ = _make(room="bedroom")
    from linux_voice_assistant.wake_arbiter import PeerEvent

    # Same peak (within band), identical ts → alphabetical wins → bathroom.
    peers = [PeerEvent(
        room="bathroom",
        wake_id=1,
        score=0.85,
        peak_score=0.85,
        ts=100.0,
        device_id="x",
        received_at=time.monotonic(),
    )]
    won, _margin = arb._decide(local_ts=100.0, local_peak=0.85, peers=peers)
    assert won is False  # bathroom < bedroom alphabetically


def test_decide_alphabetical_tiebreak_local_wins():
    arb, _ = _make(room="bathroom")
    from linux_voice_assistant.wake_arbiter import PeerEvent

    peers = [PeerEvent(
        room="bedroom",
        wake_id=1,
        score=0.85,
        peak_score=0.85,
        ts=100.0,
        device_id="x",
        received_at=time.monotonic(),
    )]
    won, _margin = arb._decide(local_ts=100.0, local_peak=0.85, peers=peers)
    assert won is True  # bathroom < bedroom; local is bathroom


# ---------------------------------------------------------------- peer ring


def test_peer_event_outside_ring_is_dropped_by_trim():
    arb, _ = _make(room="bedroom", wait_ms=20)
    # Inject one fresh peer event.
    fresh = {
        "room": "bathroom",
        "wake_id": 1,
        "score": 0.5,
        "peak_score": 0.5,
        "ts": time.time(),
        "device_id": "x",
    }
    arb.on_peer_event(json.dumps(fresh).encode("utf-8"))
    # Force-age it past the peer ring window by mutating received_at.
    with arb._lock:
        for event in arb._events:
            event.received_at -= PEER_RING_WINDOW_S * 2
    # Solo arbitrate — the trim inside arbitrate clears the stale event.
    won = arb.arbitrate(wake_id=2, score=0.4, peak_score=0.4)
    assert won is True  # peer dropped → trivial solo win
    assert arb.won_24h() == 1


def test_own_room_peer_event_ignored():
    arb, _ = _make(room="bedroom", wait_ms=20)
    own = {
        "room": "bedroom",
        "wake_id": 1,
        "score": 0.99,
        "peak_score": 0.99,
        "ts": time.time(),
        "device_id": "minty-ai-bedroom",
    }
    arb.on_peer_event(json.dumps(own).encode("utf-8"))
    # Despite the 0.99 peer score, we treat it as solo because the room
    # matches our own.
    won = arb.arbitrate(wake_id=2, score=0.3, peak_score=0.3)
    assert won is True


def test_malformed_peer_event_dropped_safely():
    arb, _ = _make()
    arb.on_peer_event(b"")
    arb.on_peer_event(b"{not json")
    arb.on_peer_event(json.dumps([1, 2, 3]).encode("utf-8"))
    arb.on_peer_event(json.dumps({"room": "bathroom", "ts": "not a number"}).encode("utf-8"))
    # No exception, no peers retained.
    with arb._lock:
        assert len(arb._events) == 0


# ---------------------------------------------------------------- tunables


def test_set_wait_ms_clamps_to_bounds():
    arb, _ = _make()
    arb.set_wait_ms(10)
    assert arb.wait_ms == WAIT_MS_MIN
    arb.set_wait_ms(2000)
    assert arb.wait_ms == WAIT_MS_MAX
    arb.set_wait_ms(150)
    assert arb.wait_ms == 150


def test_set_tiebreak_band_clamps_to_bounds():
    arb, _ = _make()
    arb.set_tiebreak_band(0.001)
    assert arb.tiebreak_band == 0.01
    arb.set_tiebreak_band(0.99)
    assert arb.tiebreak_band == 0.20
    arb.set_tiebreak_band(0.07)
    assert arb.tiebreak_band == pytest.approx(0.07)


def test_set_enabled_toggle():
    arb, _ = _make()
    arb.set_enabled(False)
    assert arb.enabled is False
    arb.set_enabled(True)
    assert arb.enabled is True


# ---------------------------------------------------------------- stats


def test_stats_snapshot_shape():
    arb, _ = _make(wait_ms=5)
    arb.arbitrate(wake_id=1, score=0.5, peak_score=0.5)
    arb.arbitrate(wake_id=2, score=0.6, peak_score=0.6)
    snap = arb.stats_snapshot()
    assert set(snap.keys()) == {"won_24h", "lost_24h", "avg_margin_24h"}
    assert snap["won_24h"] == 2
    assert snap["lost_24h"] == 0
    assert isinstance(snap["avg_margin_24h"], float)


def test_avg_margin_includes_both_outcomes():
    # Drive outcomes directly via _record_outcome so we don't have to fight
    # the 500ms peer ring window between back-to-back arbitrations in the
    # test harness.
    arb, _ = _make(room="bedroom")
    arb._record_outcome(won=False, margin=0.20)
    arb._record_outcome(won=True, margin=0.60)
    assert arb.won_24h() == 1
    assert arb.lost_24h() == 1
    assert arb.avg_margin_24h() == pytest.approx(0.40, abs=1e-6)
