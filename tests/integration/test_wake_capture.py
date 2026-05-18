"""Stage C — WakeCapture integration tests.

Covers:
  - ring buffer trimming + snapshot
  - on_wake_fire atomic WAV + sidecar write (executor path bypassed)
  - bind_session patches sidecar with session_id / generation
  - update_label resolves cancel_reason / transition reason via D1 mapping
  - orphan sweep flips unbound captures to ambiguous after timeout
  - retention sweep deletes oldest above max_files
  - manual_label round-trip used by the HTTP endpoint
  - list_captures default-filters to (label is null or ambiguous)
  - stats_24h aggregates correctly
  - DeviceSession.transition_to wires through to bind_session + update_label
"""

from __future__ import annotations

import json
import struct
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from linux_voice_assistant.session import DeviceSession, State
from linux_voice_assistant.wake_capture import (
    CancelLabelMap,
    SAMPLE_RATE,
    SAMPLE_WIDTH_BYTES,
    TransitionLabelMap,
    WakeCapture,
    derive_label,
)


def _make_state(wake_capture=None):
    from tests.conftest import make_server_state
    state = make_server_state()
    state.satellite = MagicMock(_pending_wake_id=None)
    state.wake_capture = wake_capture
    return state


def _silence_chunk(ms: int) -> bytes:
    nsamples = int(SAMPLE_RATE * ms / 1000)
    return struct.pack("<" + "h" * nsamples, *([0] * nsamples))


@pytest.fixture
def capture_dir(tmp_path: Path) -> Path:
    d = tmp_path / "wake_captures"
    d.mkdir()
    return d


@pytest.fixture
def clock():
    """Mutable test clock — tests bump `clock.now` to advance time."""
    class _Clock:
        now: float = 1_700_000_000.0
        def __call__(self) -> float:
            return self.now
    return _Clock()


def _make_wc(capture_dir: Path, clock, *, max_files: int = 5000, orphan: float = 60.0) -> WakeCapture:
    return WakeCapture(
        capture_dir=capture_dir,
        room="lounge",
        device_id="minty-test-01",
        max_files=max_files,
        orphan_timeout_s=orphan,
        clock=clock,
    )


# ---- ring buffer ------------------------------------------------------------


def test_ring_buffer_trims_to_three_seconds(capture_dir, clock):
    wc = _make_wc(capture_dir, clock)
    # Feed 500ms chunks: trim pops whole chunks, so any size that divides the
    # ring cleanly verifies the trim loop.
    for _ in range(4):
        wc.feed(_silence_chunk(500))
    # After 4 chunks the deque hits exactly the cap (4 * 500ms = 2000ms still
    # under 3000ms cap).
    assert wc._ring_bytes == 4 * 500 * SAMPLE_RATE // 1000 * SAMPLE_WIDTH_BYTES
    # Two more chunks pushes total to 3000ms, then 3500ms — trim drops one.
    wc.feed(_silence_chunk(500))
    wc.feed(_silence_chunk(500))
    assert wc._ring_bytes <= wc._ring_max_bytes
    assert wc._ring_bytes >= wc._ring_max_bytes - 500 * SAMPLE_RATE // 1000 * SAMPLE_WIDTH_BYTES


def test_ring_buffer_snapshot_is_bytes(capture_dir, clock):
    wc = _make_wc(capture_dir, clock)
    wc.feed(b"\x01\x02" * 100)
    snap = wc._snapshot_ring()
    assert snap == b"\x01\x02" * 100


# ---- on_wake_fire write -----------------------------------------------------


def test_on_wake_fire_writes_wav_and_sidecar(capture_dir, clock):
    wc = _make_wc(capture_dir, clock)
    wc.feed(_silence_chunk(3000))
    wake_id = wc.on_wake_fire(score=0.87, model="alexa.tflite", sensitivity=0.5)

    assert wake_id

    wav_files = list(capture_dir.glob("*.wav"))
    sidecars = list(capture_dir.glob("*.wav.json"))
    assert len(wav_files) == 1
    assert len(sidecars) == 1

    with open(sidecars[0]) as f:
        data = json.load(f)
    assert data["version"] == 1
    assert data["room"] == "lounge"
    assert data["device_id"] == "minty-test-01"
    assert data["score"] == 0.87
    assert data["model"] == "alexa.tflite"
    assert data["sensitivity"] == 0.5
    assert data["label"] is None
    assert data["session_id"] is None
    assert data["audio_meta"]["sample_rate"] == SAMPLE_RATE
    assert data["audio_meta"]["channels"] == 1
    assert data["audio_meta"]["duration_ms"] > 0


def test_on_wake_fire_returns_unique_wake_ids(capture_dir, clock):
    wc = _make_wc(capture_dir, clock)
    wc.feed(_silence_chunk(1000))
    a = wc.on_wake_fire(score=0.6)
    clock.now += 0.001
    b = wc.on_wake_fire(score=0.6)
    assert a != b


# ---- bind + label flow ------------------------------------------------------


def test_bind_session_patches_sidecar(capture_dir, clock):
    wc = _make_wc(capture_dir, clock)
    wc.feed(_silence_chunk(1000))
    wake_id = wc.on_wake_fire(score=0.7)
    wc.bind_session(wake_id, session_id="sess-1", generation=42)

    sidecar = next(capture_dir.glob("*.wav.json"))
    with open(sidecar) as f:
        data = json.load(f)
    assert data["session_id"] == "sess-1"
    assert data["generation"] == 42
    assert data["label"] is None


def test_update_label_positive_for_reply_done(capture_dir, clock):
    wc = _make_wc(capture_dir, clock)
    wc.feed(_silence_chunk(1000))
    wake_id = wc.on_wake_fire(score=0.7)
    wc.bind_session(wake_id, "sess-1", 1)
    wc.update_label("sess-1", reason="reply_done")

    sidecar = next(capture_dir.glob("*.wav.json"))
    with open(sidecar) as f:
        data = json.load(f)
    assert data["label"] == "positive"
    assert data["label_reason"] == "pipeline_complete"
    assert data["label_updated_ts"] is not None


def test_update_label_negative_on_red_button(capture_dir, clock):
    wc = _make_wc(capture_dir, clock)
    wc.feed(_silence_chunk(1000))
    wake_id = wc.on_wake_fire(score=0.7)
    wc.bind_session(wake_id, "sess-1", 1)
    wc.update_label("sess-1", reason="cancel_force", cancel_reason="RED_BUTTON_SOFT")

    sidecar = next(capture_dir.glob("*.wav.json"))
    with open(sidecar) as f:
        data = json.load(f)
    assert data["label"] == "negative"
    assert data["label_reason"] == "RED_BUTTON_SOFT"


def test_update_label_ambiguous_for_empty_reply(capture_dir, clock):
    wc = _make_wc(capture_dir, clock)
    wc.feed(_silence_chunk(1000))
    wake_id = wc.on_wake_fire(score=0.7)
    wc.bind_session(wake_id, "sess-1", 1)
    wc.update_label("sess-1", reason="empty_reply")

    sidecar = next(capture_dir.glob("*.wav.json"))
    with open(sidecar) as f:
        data = json.load(f)
    assert data["label"] == "ambiguous"
    assert data["label_reason"] == "empty_reply"


def test_unknown_session_id_update_label_is_noop(capture_dir, clock):
    wc = _make_wc(capture_dir, clock)
    wc.feed(_silence_chunk(1000))
    wake_id = wc.on_wake_fire(score=0.7)
    wc.update_label("never-bound", reason="reply_done")

    sidecar = next(capture_dir.glob("*.wav.json"))
    with open(sidecar) as f:
        data = json.load(f)
    assert data["label"] is None  # unchanged


# ---- mapping table ----------------------------------------------------------


def test_cancel_label_map_covers_K3_reasons():
    must_have = {
        "RED_BUTTON_SOFT", "RED_BUTTON_HARD", "STOP_WORD_INPROCESS",
        "SILENCE_TIMEOUT", "DASHBOARD", "EXTERNAL", "STOP_EVERYTHING",
    }
    assert must_have.issubset(set(CancelLabelMap.keys()))


def test_transition_label_map_covers_idle_reasons():
    assert "reply_done" in TransitionLabelMap
    assert "no_speech" in TransitionLabelMap
    assert "empty_reply" in TransitionLabelMap


def test_derive_label_unknown_cancel_falls_back_to_ambiguous():
    label, reason = derive_label("cancel_force", "MARS_LANDING")
    assert label == "ambiguous"
    assert reason.startswith("unknown_cancel:MARS_LANDING")


def test_derive_label_unknown_reason_falls_back_to_ambiguous():
    label, reason = derive_label("freak_state", None)
    assert label == "ambiguous"
    assert reason.startswith("unknown_reason:")


# ---- orphan sweep -----------------------------------------------------------


def test_orphan_sweep_marks_unbound_after_timeout(capture_dir, clock):
    wc = _make_wc(capture_dir, clock, orphan=60.0)
    wc.feed(_silence_chunk(1000))
    wake_id = wc.on_wake_fire(score=0.6)
    clock.now += 61.0
    wc._sweep_orphans_once()

    sidecar = next(capture_dir.glob("*.wav.json"))
    with open(sidecar) as f:
        data = json.load(f)
    assert data["label"] == "ambiguous"
    assert data["label_reason"] == "orphan_unbound"


def test_orphan_sweep_marks_bound_no_terminal_as_orphan(capture_dir, clock):
    wc = _make_wc(capture_dir, clock, orphan=60.0)
    wc.feed(_silence_chunk(1000))
    wake_id = wc.on_wake_fire(score=0.6)
    wc.bind_session(wake_id, "sess-2", 1)
    clock.now += 61.0
    wc._sweep_orphans_once()

    sidecar = next(capture_dir.glob("*.wav.json"))
    with open(sidecar) as f:
        data = json.load(f)
    assert data["label"] == "ambiguous"
    assert data["label_reason"] == "orphan"


# ---- retention sweep --------------------------------------------------------


def test_retention_sweep_drops_oldest_above_max(capture_dir, clock):
    wc = _make_wc(capture_dir, clock, max_files=3)
    wc.feed(_silence_chunk(1000))
    ids = []
    for i in range(5):
        clock.now += 1.0
        wid = wc.on_wake_fire(score=0.5)
        ids.append(wid)
        # bump the mtime so retention sweep can rank by it deterministically
        wav = capture_dir / f"{wid}.wav"
        sidecar = capture_dir / f"{wid}.wav.json"
        if wav.exists():
            t = time.time() + i  # ascending mtimes
            import os
            os.utime(wav, (t, t))
            os.utime(sidecar, (t, t))

    assert len(list(capture_dir.glob("*.wav.json"))) == 5
    deleted = wc._enforce_retention()
    assert deleted == 2
    remaining = sorted(p.stem.removesuffix(".wav") for p in capture_dir.glob("*.wav.json"))
    expected = sorted(ids[2:])
    assert remaining == expected


# ---- manual_label + list_captures + stats -----------------------------------


def test_manual_label_round_trip(capture_dir, clock):
    wc = _make_wc(capture_dir, clock)
    wc.feed(_silence_chunk(1000))
    wid = wc.on_wake_fire(score=0.7)
    ok = wc.manual_label(wid, "positive", user="rob")
    assert ok is True

    sidecar = next(capture_dir.glob("*.wav.json"))
    with open(sidecar) as f:
        data = json.load(f)
    assert data["label"] == "positive"
    assert data["label_reason"] == "manual_rob"


def test_manual_label_rejects_unknown_label(capture_dir, clock):
    wc = _make_wc(capture_dir, clock)
    wc.feed(_silence_chunk(1000))
    wid = wc.on_wake_fire(score=0.7)
    assert wc.manual_label(wid, "great", user="rob") is False


def test_delete_capture_removes_wav_and_sidecar(capture_dir, clock):
    wc = _make_wc(capture_dir, clock)
    wc.feed(_silence_chunk(1000))
    wid = wc.on_wake_fire(score=0.7)
    assert (capture_dir / f"{wid}.wav").exists()
    assert (capture_dir / f"{wid}.wav.json").exists()
    assert wc.delete_capture(wid) is True
    assert not (capture_dir / f"{wid}.wav").exists()
    assert not (capture_dir / f"{wid}.wav.json").exists()


def test_delete_capture_unknown_wake_id_is_false(capture_dir, clock):
    wc = _make_wc(capture_dir, clock)
    assert wc.delete_capture("20260518T120000_000000_deadbe") is False


def test_delete_capture_cleans_pending_dicts(capture_dir, clock):
    wc = _make_wc(capture_dir, clock)
    wc.feed(_silence_chunk(1000))
    wid = wc.on_wake_fire(score=0.7)
    wc.bind_session(wid, "sess-DEL", 5)
    assert "sess-DEL" in wc._session_to_wake
    assert wid in wc._pending
    assert wc.delete_capture(wid) is True
    assert "sess-DEL" not in wc._session_to_wake
    assert wid not in wc._pending


def test_list_captures_filters_to_triage_bucket_by_default(capture_dir, clock):
    wc = _make_wc(capture_dir, clock)
    wc.feed(_silence_chunk(1000))
    wid_unlabelled = wc.on_wake_fire(score=0.7)
    clock.now += 1.0
    wid_positive = wc.on_wake_fire(score=0.7)
    wc.manual_label(wid_positive, "positive")
    clock.now += 1.0
    wid_ambig = wc.on_wake_fire(score=0.7)
    wc.manual_label(wid_ambig, "ambiguous")

    result = wc.list_captures()
    ids = [it["wake_id_str"] for it in result["items"]]
    assert wid_unlabelled in ids
    assert wid_ambig in ids
    assert wid_positive not in ids

    result_all = wc.list_captures(include_all=True)
    assert {it["wake_id_str"] for it in result_all["items"]} == {wid_unlabelled, wid_positive, wid_ambig}


def test_stats_24h_aggregates(capture_dir, clock):
    wc = _make_wc(capture_dir, clock)
    wc.feed(_silence_chunk(1000))
    a = wc.on_wake_fire(score=0.7)
    b = wc.on_wake_fire(score=0.7)
    c = wc.on_wake_fire(score=0.7)
    wc.manual_label(a, "positive")
    wc.manual_label(b, "negative")
    # c remains unlabelled → counts as pending_triage_count

    stats = wc.stats_24h()
    assert stats["total_24h"] == 3
    assert stats["positive_24h"] == 1
    assert stats["negative_24h"] == 1
    assert stats["pending_triage_count"] == 1


# ---- DeviceSession wiring --------------------------------------------------


def test_ds_binds_wake_capture_on_waking_transition(capture_dir, clock):
    wc = _make_wc(capture_dir, clock)
    wc.feed(_silence_chunk(1000))
    wake_id = wc.on_wake_fire(score=0.7)

    state = _make_state(wake_capture=wc)
    state.satellite._pending_wake_id = wake_id
    ds = DeviceSession(state)
    ds.transition_to(State.WAKING, reason="wake")

    sidecar = next(capture_dir.glob("*.wav.json"))
    with open(sidecar) as f:
        data = json.load(f)
    assert data["session_id"] == ds.session_id
    assert state.satellite._pending_wake_id is None


def test_ds_labels_wake_capture_on_idle_transition(capture_dir, clock):
    wc = _make_wc(capture_dir, clock)
    wc.feed(_silence_chunk(1000))
    wake_id = wc.on_wake_fire(score=0.7)

    state = _make_state(wake_capture=wc)
    state.satellite._pending_wake_id = wake_id
    ds = DeviceSession(state)
    ds.transition_to(State.WAKING, reason="wake")
    ds.transition_to(State.IDLE, reason="cancel_force", cancel_reason="RED_BUTTON_SOFT")

    sidecar = next(capture_dir.glob("*.wav.json"))
    with open(sidecar) as f:
        data = json.load(f)
    assert data["label"] == "negative"
    assert data["label_reason"] == "RED_BUTTON_SOFT"


def test_ds_label_resolves_positive_for_clean_reply_done(capture_dir, clock):
    wc = _make_wc(capture_dir, clock)
    wc.feed(_silence_chunk(1000))
    wake_id = wc.on_wake_fire(score=0.7)

    state = _make_state(wake_capture=wc)
    state.satellite._pending_wake_id = wake_id
    ds = DeviceSession(state)
    ds.transition_to(State.WAKING, reason="wake")
    ds.transition_to(State.LISTENING, reason="wake_chime_finished")
    ds.transition_to(State.THINKING, reason="speech_captured")
    ds.transition_to(State.SPEAKING, reason="bridge_reply")
    ds.transition_to(State.IDLE, reason="reply_done")

    sidecar = next(capture_dir.glob("*.wav.json"))
    with open(sidecar) as f:
        data = json.load(f)
    assert data["label"] == "positive"
    assert data["label_reason"] == "pipeline_complete"


# ---- bind-before-executor race regression (advisor-flagged) ----------------


def test_bind_session_before_write_lands_in_sidecar(capture_dir, clock):
    """Bind happens BEFORE the executor write picks up the job.

    In production this is the common case: ``on_wake_fire`` queues the write
    to the loop's executor, returns synchronously; the audio thread then
    runs ``satellite.wakeup(wake_id=...)`` which triggers
    ``DS.transition_to(WAKING)`` and ``bind_session(...)`` — all of this
    completes before the executor wakes up to do the disk write. The sidecar
    JSON must still end up with the bound ``session_id`` and ``generation``.
    """
    import asyncio

    async def run() -> dict:
        loop = asyncio.get_running_loop()
        wc = _make_wc(capture_dir, clock)
        wc.attach_loop(loop)
        wc.feed(_silence_chunk(1000))
        wake_id = wc.on_wake_fire(score=0.7)
        wc.bind_session(wake_id, "sess-RACE", 99)
        # Yield control so call_soon_threadsafe + run_in_executor can drain.
        # 50ms is generous; the executor pickup is sub-ms in practice.
        await asyncio.sleep(0.05)
        await loop.run_in_executor(None, lambda: None)
        sidecar = next(capture_dir.glob("*.wav.json"))
        with open(sidecar) as f:
            return json.load(f)

    data = asyncio.run(run())
    assert data["session_id"] == "sess-RACE"
    assert data["generation"] == 99


def test_bind_session_after_write_still_patches(capture_dir, clock):
    """Bind happens AFTER the executor has materialised the sidecar.

    The legacy patch path must still kick in so the field updates land. Easy
    to verify by skipping the loop attach entirely — without an executor,
    the on_wake_fire path writes synchronously, so any subsequent bind sees
    the file already on disk.
    """
    wc = _make_wc(capture_dir, clock)
    wc.feed(_silence_chunk(1000))
    wake_id = wc.on_wake_fire(score=0.7)
    assert (capture_dir / f"{wake_id}.wav.json").exists()
    wc.bind_session(wake_id, "sess-LATE", 7)

    sidecar = next(capture_dir.glob("*.wav.json"))
    with open(sidecar) as f:
        data = json.load(f)
    assert data["session_id"] == "sess-LATE"
    assert data["generation"] == 7


def test_bind_session_during_write_disk_io_lands_in_sidecar(capture_dir, clock, monkeypatch):
    """Regression: bind_session cannot slot a stale patch into the window
    between _write_capture marking ``written=True`` and the sidecar actually
    being on disk.

    Pre-fix flow that bit production:
      T0  _write_capture acquired lock, captured sidecar dict with session_id=None,
          marked written=True, RELEASED lock.
      T1  bind_session acquired lock, saw written=True, released lock.
      T2  bind_session called _patch_sidecar — file not on disk yet —
          FileNotFoundError, silent return.
      T3  _write_capture wrote sidecar with session_id=None.

    Post-fix: _write_capture holds the lock through the disk write. This test
    reproduces the race by gating the disk write on a Barrier and verifying
    bind_session blocks until the write completes.
    """
    import threading

    wc = _make_wc(capture_dir, clock)
    wc.feed(_silence_chunk(1000))
    wake_id = wc.on_wake_fire(score=0.7)

    # Reset state — on_wake_fire wrote synchronously since no loop is attached.
    # We need to simulate the executor path: rewind so _write_capture can be
    # re-invoked with controlled timing.
    for p in capture_dir.glob("*"):
        p.unlink()
    with wc._pending_lock:
        wc._pending[wake_id] = {
            "session_id": None,
            "generation": None,
            "created_ts": clock(),
            "sidecar_path": capture_dir / f"{wake_id}.wav.json",
            "wav_path": capture_dir / f"{wake_id}.wav",
            "bound_ts": None,
            "sidecar_template": {
                "version": 1,
                "session_id": None,
                "generation": None,
                "wake_id_str": wake_id,
                "label": None,
            },
            "written": False,
        }

    write_started = threading.Event()
    release_write = threading.Event()
    original_write_json = wc._write_capture.__globals__["_atomic_write_json"]

    def gated_write_json(path, payload):
        write_started.set()
        release_write.wait(timeout=2.0)
        original_write_json(path, payload)

    monkeypatch.setattr(
        "linux_voice_assistant.wake_capture._atomic_write_json",
        gated_write_json,
    )

    def run_write():
        wc._write_capture(wake_id, _silence_chunk(100))

    writer = threading.Thread(target=run_write)
    writer.start()
    assert write_started.wait(timeout=2.0), "_write_capture never reached disk-write phase"

    # bind_session should block on _pending_lock until the writer releases it.
    bound_event = threading.Event()

    def run_bind():
        wc.bind_session(wake_id, "sess-RACE", 42)
        bound_event.set()

    binder = threading.Thread(target=run_bind)
    binder.start()
    # bind must NOT complete while writer holds the lock.
    assert not bound_event.wait(timeout=0.2), "bind_session ran while _write_capture held the lock"

    # Release the writer; both threads should finish cleanly.
    release_write.set()
    writer.join(timeout=2.0)
    binder.join(timeout=2.0)
    assert not writer.is_alive()
    assert not binder.is_alive()

    sidecar_path = capture_dir / f"{wake_id}.wav.json"
    assert sidecar_path.exists()
    with open(sidecar_path) as f:
        data = json.load(f)
    # Either ordering converges on the bound session_id ending up in the file:
    # if writer ran first, bind_session patched after; if bind beat the writer
    # to the lock, the writer's snapshot picked up the bind data.
    assert data["session_id"] == "sess-RACE"
    assert data["generation"] == 42
