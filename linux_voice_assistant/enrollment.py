"""Stage D — HA-driven speaker enrollment handler.

Receives `calisto/<room>/enroll/capture` payloads from the HA enrollment
script (ha-config/scripts.yaml :: calisto_speaker_enrollment). For each
trigger:

  1. Snapshot the last 3s from the wake-capture ring (the user just spoke
     in response to a TTS prompt issued by HA).
  2. Run SpeakerVerifier.embed() on the snapshot.
  3. Outlier-check against existing samples for the same user (D2 — new
     embedding's cosine to ALL existing same-user embeddings must reach
     0.5; otherwise treat as poisoned/noisy capture and skip).
  4. On accept: append to enrollments JSON via EnrollmentsStore.
  5. Publish the outcome to `calisto/<room>/enroll/result` for HA's UI.

Failures are best-effort: a bad capture is logged and reported on the
result topic; the device keeps running normally.

Threading: invoked from HABridge's MQTT thread. Marshals to the asyncio
loop for the disk write so the MQTT callback returns promptly.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from .ha_bridge import HABridge
    from .models import ServerState

_LOGGER = logging.getLogger(__name__)


_DEFAULT_CAPTURE_MS = 3000


class EnrollmentHandler:
    """Routes `calisto/<room>/enroll/capture` MQTT messages into the
    SpeakerVerifier embedding + EnrollmentsStore append path."""

    def __init__(
        self,
        state: "ServerState",
        *,
        ha_bridge: "Optional[HABridge]",
        loop: "Optional[asyncio.AbstractEventLoop]" = None,
    ) -> None:
        self._state = state
        self._ha_bridge = ha_bridge
        self._loop = loop
        self.result_topic = f"calisto/{state.room}/enroll/result"

    def attach_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def handle(self, raw_payload: bytes) -> None:
        """Process one MQTT trigger. Safe to call from the MQTT thread.

        Marshals the actual embed + write onto the asyncio loop. When no
        loop is wired (tests, cold-start), runs inline.
        """
        try:
            obj = json.loads(raw_payload.decode("utf-8")) if raw_payload else {}
        except (UnicodeDecodeError, ValueError):
            _LOGGER.warning("Enrollment: malformed JSON payload")
            self._publish_result({"ok": False, "reason": "malformed_payload"})
            return
        if not isinstance(obj, dict):
            _LOGGER.warning("Enrollment: payload is not a JSON object")
            self._publish_result({"ok": False, "reason": "malformed_payload"})
            return

        user_id = str(obj.get("user_id") or "").strip()
        if not user_id:
            self._publish_result({"ok": False, "reason": "missing_user_id"})
            return
        display_name = str(obj.get("display_name") or user_id)
        phrase = str(obj.get("phrase") or "")
        try:
            capture_ms = int(obj.get("capture_ms", _DEFAULT_CAPTURE_MS))
        except (TypeError, ValueError):
            capture_ms = _DEFAULT_CAPTURE_MS
        capture_ms = max(500, min(10000, capture_ms))

        loop = self._loop or getattr(self._state, "loop", None)
        if loop is None:
            self._do_enroll(user_id, display_name, phrase, capture_ms)
            return
        try:
            loop.call_soon_threadsafe(
                self._do_enroll, user_id, display_name, phrase, capture_ms,
            )
        except RuntimeError:
            self._do_enroll(user_id, display_name, phrase, capture_ms)

    # ---- internals -----------------------------------------------------

    def _do_enroll(
        self,
        user_id: str,
        display_name: str,
        phrase: str,
        capture_ms: int,
    ) -> None:
        verifier = getattr(self._state, "speaker_verifier", None)
        wake_capture = getattr(self._state, "wake_capture", None)
        if verifier is None or wake_capture is None:
            _LOGGER.warning("Enrollment: verifier or wake_capture missing")
            self._publish_result({
                "ok": False, "user_id": user_id, "reason": "components_unavailable",
            })
            return
        # Snapshot the user's just-spoken phrase from the ring. The HA
        # script paces the prompts so the response audio is the last
        # `capture_ms` of the ring at trigger-receipt time.
        pcm = wake_capture.snapshot_recent_pcm(capture_ms / 1000.0)
        if len(pcm) < int(0.5 * 16000 * 2):  # < 0.5s of audio
            _LOGGER.warning("Enrollment: insufficient ring audio (%d bytes)", len(pcm))
            self._publish_result({
                "ok": False, "user_id": user_id, "reason": "insufficient_audio",
            })
            return
        vec = verifier.embed(pcm)
        if vec is None:
            _LOGGER.warning("Enrollment: embed() returned None (model unavailable?)")
            self._publish_result({
                "ok": False, "user_id": user_id, "reason": "model_unavailable",
            })
            return
        # Outlier check — only after the user already has at least one
        # sample. First sample always accepts.
        if not verifier.outlier_check(user_id, vec, min_sim=0.5):
            _LOGGER.info("Enrollment: outlier rejected for %s (phrase=%r)", user_id, phrase)
            self._publish_result({
                "ok": False, "user_id": user_id, "reason": "outlier_rejected",
                "phrase": phrase,
            })
            return
        duration_ms = int((len(pcm) / 2 / 16000) * 1000)
        try:
            entry = verifier.store.add_enrollment(
                user_id, vec,
                display_name=display_name,
                source_audio=None,
                duration_ms=duration_ms,
            )
        except Exception:
            _LOGGER.exception("Enrollment: add_enrollment raised")
            self._publish_result({
                "ok": False, "user_id": user_id, "reason": "store_write_failed",
            })
            return
        user = verifier.store.find_user(user_id)
        sample_count = len(user.embeddings) if user is not None else 0
        _LOGGER.info(
            "Enrollment: accepted phrase=%r for %s (total samples=%d)",
            phrase, user_id, sample_count,
        )
        self._publish_result({
            "ok": True, "user_id": user_id, "display_name": display_name,
            "phrase": phrase, "sample_count": sample_count,
            "captured_ts": entry.captured_ts, "duration_ms": duration_ms,
        })

    def _publish_result(self, body: dict) -> None:
        bridge = self._ha_bridge
        if bridge is None:
            return
        body.setdefault("ts", time.time())
        try:
            bridge.publish(self.result_topic, json.dumps(body), qos=1, retain=False)
        except Exception:
            _LOGGER.exception("Enrollment: result publish failed")


__all__ = ["EnrollmentHandler"]
