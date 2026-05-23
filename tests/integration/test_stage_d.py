"""Stage D integration tests — HABridge enroll/capture routing and the
WakeCapture snapshot + sidecar plumbing the SpeakerVerifier relies on."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from linux_voice_assistant.discovery import SpeakerVerifierDiscovery
from linux_voice_assistant.enrollment import EnrollmentHandler
from linux_voice_assistant.ha_bridge import HABridge
from linux_voice_assistant.speaker_verifier import EnrollmentsStore, SpeakerVerifier
from linux_voice_assistant.wake_capture import WakeCapture

from tests.conftest import make_server_state


def _build_bridge(fake_paho):
    created, factory = fake_paho
    bridge = HABridge(room="living_room", host="127.0.0.1", port=1883, client_factory=factory)
    bridge.start()
    return bridge, created[0]


def test_habridge_subscribes_to_enroll_capture(fake_paho):
    bridge, fake = _build_bridge(fake_paho)
    subs = {t for t, _ in fake.subscriptions}
    assert "calisto/living_room/enroll/capture" in subs


def test_habridge_subscribes_to_sv_tunables(fake_paho):
    _bridge, fake = _build_bridge(fake_paho)
    subs = {t for t, _ in fake.subscriptions}
    assert "calisto/living_room/tunable/sv_threshold/set" in subs
    assert "calisto/living_room/tunable/sv_audible_notify/set" in subs


def test_habridge_routes_sv_threshold_set_to_discovery_handler(fake_paho, tmp_path):
    bridge, fake = _build_bridge(fake_paho)
    state = make_server_state()
    state.sv_threshold = 0.70
    enrollments = EnrollmentsStore(tmp_path / "enrollments.json", room="living_room")
    sv_disco = SpeakerVerifierDiscovery(
        ha_bridge=bridge,
        state=state,
        enrollments=enrollments,
        room="living_room",
    )
    bridge.attach_speaker_verifier_discovery(sv_disco)

    msg = MagicMock(
        topic="calisto/living_room/tunable/sv_threshold/set",
        payload=b'{"value": 0.58}',
    )
    bridge._on_message(fake, None, msg)
    assert state.sv_threshold == pytest.approx(0.58)
    assert enrollments.threshold == pytest.approx(0.58)


def test_habridge_routes_sv_audible_notify_set_to_discovery_handler(fake_paho, tmp_path):
    bridge, fake = _build_bridge(fake_paho)
    state = make_server_state()
    state.sv_audible_notify = True
    enrollments = EnrollmentsStore(tmp_path / "enrollments.json", room="living_room")
    sv_disco = SpeakerVerifierDiscovery(
        ha_bridge=bridge,
        state=state,
        enrollments=enrollments,
        room="living_room",
    )
    bridge.attach_speaker_verifier_discovery(sv_disco)

    msg = MagicMock(
        topic="calisto/living_room/tunable/sv_audible_notify/set",
        payload=b'{"value": false}',
    )
    bridge._on_message(fake, None, msg)
    assert state.sv_audible_notify is False


def test_habridge_sv_tunable_without_attached_discovery_is_silent(fake_paho):
    bridge, fake = _build_bridge(fake_paho)
    msg = MagicMock(
        topic="calisto/living_room/tunable/sv_threshold/set",
        payload=b'{"value": 0.55}',
    )
    # No discovery attached — must not raise.
    bridge._on_message(fake, None, msg)


def test_habridge_routes_enroll_capture_to_handler(fake_paho, tmp_path):
    bridge, fake = _build_bridge(fake_paho)
    state = make_server_state()
    state.room = "living_room"
    state.wake_capture = MagicMock(
        snapshot_recent_pcm=MagicMock(return_value=b"\x00\x00" * 24000),
    )
    store = EnrollmentsStore(tmp_path / "enrollments.json", room="living_room")
    verifier = SpeakerVerifier(store=store, model_path=None)
    verifier.embed = MagicMock(return_value=[0.1] * 512)
    verifier.outlier_check = MagicMock(return_value=True)
    state.speaker_verifier = verifier
    handler = EnrollmentHandler(state, ha_bridge=bridge)
    bridge.attach_enrollment_handler(handler)

    msg = MagicMock(
        topic="calisto/living_room/enroll/capture",
        payload=json.dumps({"user_id": "rob", "phrase": "test"}).encode(),
    )
    bridge._on_message(fake, None, msg)

    user = store.find_user("rob")
    assert user is not None
    assert len(user.embeddings) == 1


# -- WakeCapture snapshot + sidecar plumbing ------------------------------


def test_wake_capture_snapshot_recent_pcm(tmp_path):
    wc = WakeCapture(
        capture_dir=tmp_path, room="living_room", device_id="dev",
    )
    # 1s @ 16kHz mono int16 = 32000 bytes = 16000 two-byte samples.
    wc.feed(b"\x01\x00" * 16000)
    wc.feed(b"\x02\x00" * 16000)
    pcm = wc.snapshot_recent_pcm(1.0)
    assert len(pcm) == 32000
    # The tail-slice should be the most-recent 1s — the \x02 chunk.
    assert pcm.startswith(b"\x02\x00")


def test_wake_capture_snapshot_capped_at_ring_extent(tmp_path):
    wc = WakeCapture(capture_dir=tmp_path, room="living_room", device_id="dev")
    wc.feed(b"\x01\x00" * 8000)  # 0.5s only (16000 bytes)
    pcm = wc.snapshot_recent_pcm(5.0)  # ask for 5s — ring has 0.5s
    assert len(pcm) == 16000


def test_update_speaker_match_writes_to_sidecar(tmp_path):
    wc = WakeCapture(capture_dir=tmp_path, room="living_room", device_id="dev")
    wc.feed(b"\x00" * 32000)
    wake_id = wc.on_wake_fire(score=0.8)
    # Sidecar written synchronously since no executor loop attached.
    wc.update_speaker_match(wake_id, {"matched_user_id": "rob", "score": 0.91})
    sidecar = json.loads(Path(tmp_path / f"{wake_id}.wav.json").read_text())
    assert sidecar["speaker_match"]["matched_user_id"] == "rob"


def test_update_wake_label_gate2_reject(tmp_path):
    wc = WakeCapture(capture_dir=tmp_path, room="living_room", device_id="dev")
    wc.feed(b"\x00" * 32000)
    wake_id = wc.on_wake_fire(score=0.8)
    assert wc.update_wake_label(wake_id, "gate2_reject", "gate2_reject") is True
    sidecar = json.loads(Path(tmp_path / f"{wake_id}.wav.json").read_text())
    assert sidecar["label"] == "gate2_reject"
    assert sidecar["label_reason"] == "gate2_reject"


def test_update_wake_label_invalid_label_rejected(tmp_path):
    wc = WakeCapture(capture_dir=tmp_path, room="living_room", device_id="dev")
    wc.feed(b"\x00" * 32000)
    wake_id = wc.on_wake_fire(score=0.8)
    assert wc.update_wake_label(wake_id, "purple", "nope") is False
