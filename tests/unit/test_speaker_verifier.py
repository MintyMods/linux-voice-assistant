"""Stage D — SpeakerVerifier + EnrollmentsStore unit tests.

The CAM++ ONNX model isn't shipped in the repo, so these tests use a
stubbed embedder that returns deterministic 512-dim vectors. That
exercises the gate logic, enrollment store, outlier check, and accept-
all fallback path without requiring onnxruntime at test time.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List
from unittest.mock import MagicMock

import numpy as np
import pytest

from linux_voice_assistant.speaker_verifier import (
    DEFAULT_THRESHOLD,
    EnrolledUser,
    Enrollment,
    EnrollmentsStore,
    SpeakerVerifier,
    VerificationResult,
    _cosine,
    _l2_normalize,
)


def _silence_pcm(seconds: float = 1.5) -> bytes:
    return (np.zeros(int(seconds * 16000), dtype=np.int16)).tobytes()


def _tone_pcm(seconds: float = 1.5, freq: int = 440) -> bytes:
    t = np.arange(int(seconds * 16000), dtype=np.float32) / 16000.0
    samples = (np.sin(2 * np.pi * freq * t) * 10000).astype(np.int16)
    return samples.tobytes()


def _stub_vec(seed: int) -> List[float]:
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(512).astype(np.float32)
    vec /= np.linalg.norm(vec)
    return vec.tolist()


# -- EnrollmentsStore -----------------------------------------------------


def test_store_load_missing_file_silent(tmp_path):
    store = EnrollmentsStore(tmp_path / "enrollments.json", room="living_room")
    store.load()
    assert store.users == []
    assert store.has_enrollments is False
    assert store.load_failed is False


def test_store_load_parses_users(tmp_path):
    payload = {
        "version": 1, "room": "living_room", "threshold": 0.65,
        "users": [{
            "user_id": "rob", "display_name": "Rob",
            "embeddings": [{"vec": _stub_vec(1), "captured_ts": "2026-05-15T10:00:00+01:00"}],
        }],
    }
    p = tmp_path / "enrollments.json"
    p.write_text(json.dumps(payload))
    store = EnrollmentsStore(p, room="living_room")
    store.load()
    assert store.threshold == 0.65
    assert len(store.users) == 1
    assert store.users[0].user_id == "rob"
    assert len(store.users[0].embeddings) == 1
    assert store.has_enrollments


def test_store_load_corrupt_falls_back_to_bak(tmp_path):
    p = tmp_path / "enrollments.json"
    bak = tmp_path / "enrollments.json.bak"
    p.write_text("{ this is not json")
    bak.write_text(json.dumps({
        "version": 1, "room": "living_room",
        "users": [{"user_id": "rob", "display_name": "Rob",
                   "embeddings": [{"vec": _stub_vec(2), "captured_ts": ""}]}],
    }))
    store = EnrollmentsStore(p, room="living_room")
    store.load()
    assert len(store.users) == 1
    assert store.load_failed is False


def test_store_load_both_corrupt_marks_load_failed(tmp_path):
    p = tmp_path / "enrollments.json"
    bak = tmp_path / "enrollments.json.bak"
    p.write_text("garbage")
    bak.write_text("more garbage")
    store = EnrollmentsStore(p, room="living_room")
    store.load()
    assert store.load_failed is True
    assert store.users == []


def test_store_add_enrollment_persists(tmp_path):
    p = tmp_path / "enrollments.json"
    store = EnrollmentsStore(p, room="living_room")
    store.add_enrollment("rob", _stub_vec(3), display_name="Rob")
    assert p.exists()
    data = json.loads(p.read_text())
    assert data["users"][0]["user_id"] == "rob"
    assert len(data["users"][0]["embeddings"]) == 1


def test_store_save_rotates_bak(tmp_path):
    p = tmp_path / "enrollments.json"
    store = EnrollmentsStore(p, room="living_room")
    store.add_enrollment("rob", _stub_vec(4))
    store.add_enrollment("rob", _stub_vec(5))
    assert (tmp_path / "enrollments.json.bak").exists()


def test_store_delete_enrollment(tmp_path):
    p = tmp_path / "enrollments.json"
    store = EnrollmentsStore(p, room="living_room")
    store.add_enrollment("rob", _stub_vec(6))
    store.add_enrollment("rob", _stub_vec(7))
    assert store.delete_enrollment("rob", 0)
    assert len(store.find_user("rob").embeddings) == 1
    # Last sample deletion removes the user entirely.
    store.delete_enrollment("rob", 0)
    assert store.find_user("rob") is None


# -- Verifier (disabled / fallback paths) ---------------------------------


def test_verify_with_no_enrollments_returns_accept_all(tmp_path):
    store = EnrollmentsStore(tmp_path / "x.json", room="living_room")
    verifier = SpeakerVerifier(store=store, model_path=None)
    result = verifier.verify(_silence_pcm())
    assert result.gate1_pass is True
    assert result.gate2_pass is True
    assert result.reason == "disabled"


def test_verify_disabled_flag_short_circuits(tmp_path):
    store = EnrollmentsStore(tmp_path / "x.json", room="living_room")
    store.add_enrollment("rob", _stub_vec(8))
    verifier = SpeakerVerifier(store=store, model_path=None, enabled=False)
    result = verifier.verify(_silence_pcm())
    assert result.reason == "disabled"


def test_verify_missing_model_falls_back_to_accept_all(tmp_path):
    store = EnrollmentsStore(tmp_path / "x.json", room="living_room")
    store.add_enrollment("rob", _stub_vec(9))
    verifier = SpeakerVerifier(store=store, model_path=tmp_path / "missing.onnx")
    result = verifier.verify(_silence_pcm())
    assert result.gate1_pass is True
    assert result.gate2_pass is True
    assert result.reason == "disabled"
    assert verifier.is_active() is False


# -- Verifier (gate logic with stubbed model) -----------------------------


def _make_verifier_with_stub(tmp_path, embed_vec: List[float], *, gate1: bool = True):
    """Build a SpeakerVerifier with the ONNX path stubbed and Gate 1 forced."""
    store = EnrollmentsStore(tmp_path / "x.json", room="living_room")
    verifier = SpeakerVerifier(store=store, model_path=None)
    # Pretend the model + VAD loaded successfully.
    verifier._init_attempted = True
    verifier._init_failed = False
    verifier._session = MagicMock()
    verifier._input_name = "input"
    verifier._vad = MagicMock()
    verifier._embed = lambda pcm, _v=embed_vec: np.asarray(_v, dtype=np.float32)
    verifier._gate1_vad = lambda pcm, _g=gate1: _g
    return verifier, store


def test_verify_gate1_fail(tmp_path):
    verifier, store = _make_verifier_with_stub(tmp_path, _stub_vec(10), gate1=False)
    store.add_enrollment("rob", _stub_vec(11))
    result = verifier.verify(_silence_pcm())
    assert result.gate1_pass is False
    assert result.gate2_pass is False
    assert result.reason == "gate1_fail_vad"
    assert result.matched_user_id is None


def test_verify_gate2_pass_when_match_exceeds_threshold(tmp_path):
    target_vec = _stub_vec(12)
    verifier, store = _make_verifier_with_stub(tmp_path, target_vec, gate1=True)
    store.add_enrollment("rob", target_vec)  # exact match → cosine = 1.0
    result = verifier.verify(_silence_pcm())
    assert result.gate2_pass is True
    assert result.matched_user_id == "rob"
    assert result.score > 0.99
    assert result.reason == "verified"


def test_verify_gate2_fail_when_score_below_threshold(tmp_path):
    # Stub embed returns a vector orthogonal to the enrolled one.
    vec_a = [1.0] + [0.0] * 511
    vec_b = [0.0, 1.0] + [0.0] * 510
    verifier, store = _make_verifier_with_stub(tmp_path, vec_a, gate1=True)
    store.add_enrollment("rob", vec_b)
    result = verifier.verify(_silence_pcm())
    assert result.gate2_pass is False
    assert result.reason == "gate2_fail_score"
    assert result.score == pytest.approx(0.0, abs=1e-5)


def test_verify_matches_user_with_max_cosine(tmp_path):
    """Aggregation: max(cosine) over a user's embeddings (D2)."""
    target = _stub_vec(13)
    verifier, store = _make_verifier_with_stub(tmp_path, target, gate1=True)
    # Rob has two embeddings, one orthogonal and one exact match.
    store.add_enrollment("rob", [1.0] + [0.0] * 511)
    store.add_enrollment("rob", target)
    # Leigh has just one orthogonal embedding.
    store.add_enrollment("leigh", [0.0, 1.0] + [0.0] * 510)
    result = verifier.verify(_silence_pcm())
    assert result.matched_user_id == "rob"
    assert result.score > 0.99


# -- outlier_check (enrollment-time guard) --------------------------------


def test_outlier_check_first_sample_always_accepts(tmp_path):
    store = EnrollmentsStore(tmp_path / "x.json", room="living_room")
    verifier = SpeakerVerifier(store=store, model_path=None)
    assert verifier.outlier_check("rob", _stub_vec(14)) is True


def test_outlier_check_accepts_when_similar(tmp_path):
    base = _stub_vec(15)
    store = EnrollmentsStore(tmp_path / "x.json", room="living_room")
    store.add_enrollment("rob", base)
    verifier = SpeakerVerifier(store=store, model_path=None)
    # Same vector trivially > 0.5.
    assert verifier.outlier_check("rob", base) is True


def test_outlier_check_rejects_when_dissimilar(tmp_path):
    store = EnrollmentsStore(tmp_path / "x.json", room="living_room")
    store.add_enrollment("rob", [1.0] + [0.0] * 511)
    verifier = SpeakerVerifier(store=store, model_path=None)
    # Orthogonal candidate (cosine 0) — below 0.5.
    assert verifier.outlier_check("rob", [0.0, 1.0] + [0.0] * 510) is False


# -- Helpers --------------------------------------------------------------


def test_cosine_identical_vectors():
    v = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
    assert _cosine(v, v) == pytest.approx(1.0)


def test_cosine_orthogonal_vectors():
    a = np.asarray([1.0, 0.0], dtype=np.float32)
    b = np.asarray([0.0, 1.0], dtype=np.float32)
    assert _cosine(a, b) == pytest.approx(0.0)


def test_cosine_zero_vector_safe():
    assert _cosine(np.zeros(3, dtype=np.float32), np.ones(3, dtype=np.float32)) == 0.0


def test_l2_normalize():
    v = np.asarray([3.0, 4.0], dtype=np.float32)
    n = _l2_normalize(v)
    assert np.linalg.norm(n) == pytest.approx(1.0)
