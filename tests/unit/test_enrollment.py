"""Stage D — EnrollmentHandler unit tests."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from linux_voice_assistant.enrollment import EnrollmentHandler
from linux_voice_assistant.speaker_verifier import EnrollmentsStore, SpeakerVerifier

from tests.conftest import make_server_state


def _stub_verifier(tmp_path, *, embed_vec=None, outlier_pass=True):
    store = EnrollmentsStore(tmp_path / "enrollments.json", room="lounge")
    verifier = SpeakerVerifier(store=store, model_path=None)
    verifier.embed = MagicMock(return_value=list(embed_vec) if embed_vec is not None else [0.1] * 512)
    verifier.outlier_check = MagicMock(return_value=outlier_pass)
    return verifier, store


def _make_state(tmp_path, verifier=None, wake_capture=None):
    state = make_server_state()
    state.room = "lounge"
    state.speaker_verifier = verifier
    state.wake_capture = wake_capture or MagicMock(
        snapshot_recent_pcm=MagicMock(return_value=b"\x00\x00" * 24000),
    )
    return state


def test_handle_missing_user_id_publishes_failure(tmp_path):
    verifier, _ = _stub_verifier(tmp_path)
    state = _make_state(tmp_path, verifier=verifier)
    bridge = MagicMock()
    handler = EnrollmentHandler(state, ha_bridge=bridge)
    handler.handle(json.dumps({}).encode())
    args, _ = bridge.publish.call_args
    body = json.loads(args[1])
    assert body["ok"] is False
    assert body["reason"] == "missing_user_id"


def test_handle_malformed_json_publishes_failure(tmp_path):
    verifier, _ = _stub_verifier(tmp_path)
    state = _make_state(tmp_path, verifier=verifier)
    bridge = MagicMock()
    handler = EnrollmentHandler(state, ha_bridge=bridge)
    handler.handle(b"not json {")
    args, _ = bridge.publish.call_args
    body = json.loads(args[1])
    assert body["reason"] == "malformed_payload"


def test_handle_accepts_first_sample(tmp_path):
    verifier, store = _stub_verifier(tmp_path, embed_vec=[0.1] * 512)
    state = _make_state(tmp_path, verifier=verifier)
    bridge = MagicMock()
    handler = EnrollmentHandler(state, ha_bridge=bridge)
    handler.handle(json.dumps({
        "user_id": "rob", "display_name": "Rob", "phrase": "test", "capture_ms": 3000,
    }).encode())
    user = store.find_user("rob")
    assert user is not None
    assert len(user.embeddings) == 1
    args, _ = bridge.publish.call_args
    body = json.loads(args[1])
    assert body["ok"] is True
    assert body["sample_count"] == 1


def test_handle_rejects_outlier(tmp_path):
    verifier, store = _stub_verifier(tmp_path, embed_vec=[0.1] * 512, outlier_pass=False)
    # Pre-seed one enrollment so outlier_pass=False is meaningful.
    store.add_enrollment("rob", [0.5] * 512)
    state = _make_state(tmp_path, verifier=verifier)
    bridge = MagicMock()
    handler = EnrollmentHandler(state, ha_bridge=bridge)
    handler.handle(json.dumps({
        "user_id": "rob", "display_name": "Rob", "phrase": "test",
    }).encode())
    user = store.find_user("rob")
    assert len(user.embeddings) == 1  # not appended
    args, _ = bridge.publish.call_args
    body = json.loads(args[1])
    assert body["ok"] is False
    assert body["reason"] == "outlier_rejected"


def test_handle_insufficient_audio(tmp_path):
    verifier, _ = _stub_verifier(tmp_path)
    state = _make_state(
        tmp_path, verifier=verifier,
        wake_capture=MagicMock(snapshot_recent_pcm=MagicMock(return_value=b"\x00\x00" * 100)),
    )
    bridge = MagicMock()
    handler = EnrollmentHandler(state, ha_bridge=bridge)
    handler.handle(json.dumps({"user_id": "rob"}).encode())
    args, _ = bridge.publish.call_args
    body = json.loads(args[1])
    assert body["reason"] == "insufficient_audio"


def test_handle_model_unavailable(tmp_path):
    verifier, _ = _stub_verifier(tmp_path)
    verifier.embed = MagicMock(return_value=None)
    state = _make_state(tmp_path, verifier=verifier)
    bridge = MagicMock()
    handler = EnrollmentHandler(state, ha_bridge=bridge)
    handler.handle(json.dumps({"user_id": "rob"}).encode())
    args, _ = bridge.publish.call_args
    body = json.loads(args[1])
    assert body["reason"] == "model_unavailable"


def test_capture_ms_clamped():
    """Out-of-range capture_ms values are clamped to [500, 10000]."""
    state = MagicMock()
    state.loop = None
    bridge = MagicMock()
    handler = EnrollmentHandler(state, ha_bridge=bridge)
    captured: dict = {}

    def _do(user_id, display_name, phrase, capture_ms):
        captured["capture_ms"] = capture_ms
    handler._do_enroll = _do
    handler.handle(json.dumps({"user_id": "x", "capture_ms": 999999}).encode())
    assert captured["capture_ms"] == 10000
    handler.handle(json.dumps({"user_id": "x", "capture_ms": 100}).encode())
    assert captured["capture_ms"] == 500
