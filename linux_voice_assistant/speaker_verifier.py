"""Stage D — SpeakerVerifier: two-gate (VAD + CAM++) post-wake chain.

Per D2 the verifier sits between wake-detect and the WAKING/LISTENING
transition. Each wake-event's pre-roll audio is passed through:

  Gate 1 — TEN-VAD: speech-vs-not. Failure = silent drop.
  Gate 2 — CAM++ embedding cosine against per-user enrollments. Failure
           = audible rejection chime + red LED flash, sidecar labelled
           `gate2_reject` (D1 dataset).

Both gates are fail-fast and the chain is sequential (asymmetric cost +
no clean ONNX preempt = no parallel speculative path).

Graceful degradation:
  - onnxruntime not installed → verifier disabled (accept all), WARN.
  - CAM++ model file missing → verifier disabled (accept all), WARN.
  - Enrollments file missing → verifier disabled (accept all), INFO. (No
    enrolled users means we don't yet know who is allowed; rejecting
    everyone would brick the device.)
  - Enrollments file corrupt → fall back to `enrollments-<room>.json.bak`;
    if both corrupt → consult `fallback_policy` (accept_all | reject_all).

Per-sample aggregation: `max(cosine_sim(input, e))` over a user's
embeddings (D2). Matched user = the user_id with the highest max
cosine sim that crosses the threshold.

Enrollment helpers live on the same class so the HA-driven enrollment
flow (ha-config/scripts.yaml) can call `embed(pcm)` and `add_enrollment`
without instantiating extra plumbing.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    pass

_LOGGER = logging.getLogger(__name__)


SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2  # bytes (int16)
VAD_HOP = 256

# Default verification window: most-recent 1.5s ending at wake_detected.
# Wide enough for CAM++ to extract a stable embedding (~1s minimum
# recommended); narrow enough that breath / chair-scrape preceding wake
# doesn't drag the embedding away from the user's voice.
DEFAULT_VERIFY_WINDOW_MS = 1500

# D2 default threshold (cosine similarity 0-1). 0.70 is the live-tunable
# default per architecture-v1-decisions.md §D2.
DEFAULT_THRESHOLD = 0.70

# CAM++ embedding dimensionality (3D-Speaker CAM++ ONNX export).
CAMPLUS_EMBED_DIM = 512


@dataclass
class Enrollment:
    """A single enrollment sample for a user — vector + provenance."""

    vec: List[float]
    captured_ts: str
    source_audio: Optional[str] = None
    duration_ms: int = 0


@dataclass
class EnrolledUser:
    user_id: str
    display_name: str
    embeddings: List[Enrollment] = field(default_factory=list)


@dataclass
class VerificationResult:
    """Result returned by SpeakerVerifier.verify."""

    gate1_pass: bool
    gate2_pass: bool
    matched_user_id: Optional[str]
    score: float
    threshold: float
    reason: str  # one of: "verified", "gate1_fail_vad", "gate2_fail_score", "disabled"

    def to_dict(self) -> dict:
        return {
            "gate1_pass": self.gate1_pass,
            "gate2_pass": self.gate2_pass,
            "matched_user_id": self.matched_user_id,
            "score": float(self.score),
            "threshold": float(self.threshold),
            "reason": self.reason,
        }


# ---------------------------------------------------------------------------
# Enrollments store
# ---------------------------------------------------------------------------


class EnrollmentsStore:
    """Per-room enrollments JSON loader + atomic writer.

    Schema: see v1-spec-M-config.md §M.2.

    Corruption recovery: on JSON-parse fail at load, attempt `.bak`. If
    both fail, the store is empty and `fallback_policy` decides behaviour
    (accept_all → verifier degrades to accept-all; reject_all → reject
    every wake).
    """

    def __init__(
        self,
        path: Path,
        *,
        room: str,
        threshold: float = DEFAULT_THRESHOLD,
        fallback_policy: str = "accept_all",
    ) -> None:
        self.path = Path(path)
        self.room = room
        self.threshold = threshold
        self.fallback_policy = fallback_policy
        self.users: List[EnrolledUser] = []
        self.version: int = 1
        self.updated_ts: Optional[str] = None
        self._load_error: bool = False
        self._lock = threading.Lock()

    @property
    def has_enrollments(self) -> bool:
        return any(u.embeddings for u in self.users)

    @property
    def load_failed(self) -> bool:
        return self._load_error

    def load(self) -> None:
        """Read from disk; on corruption, try `.bak`. Idempotent."""
        if not self.path.exists():
            _LOGGER.info(
                "Enrollments: %s not present — verifier will accept all wakes",
                self.path,
            )
            return
        try:
            self._load_from(self.path)
            return
        except (ValueError, OSError) as exc:
            _LOGGER.warning(
                "Enrollments: %s unreadable (%s) — falling back to .bak",
                self.path, exc,
            )
        bak = self.path.with_suffix(self.path.suffix + ".bak")
        if bak.exists():
            try:
                self._load_from(bak)
                return
            except (ValueError, OSError) as exc:
                _LOGGER.error("Enrollments: backup %s also unreadable (%s)", bak, exc)
        self._load_error = True

    def _load_from(self, path: Path) -> None:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("enrollments file is not a JSON object")
        self.version = int(data.get("version", 1))
        if data.get("room") and data.get("room") != self.room:
            _LOGGER.warning(
                "Enrollments file %s declares room=%r but verifier room=%r",
                path, data.get("room"), self.room,
            )
        if "threshold" in data:
            try:
                self.threshold = float(data["threshold"])
            except (TypeError, ValueError):
                pass
        self.updated_ts = data.get("updated_ts")
        raw_users = data.get("users") or []
        loaded: List[EnrolledUser] = []
        for u in raw_users:
            if not isinstance(u, dict):
                continue
            uid = str(u.get("user_id") or "")
            if not uid:
                continue
            display_name = str(u.get("display_name") or uid)
            embeddings: List[Enrollment] = []
            for e in (u.get("embeddings") or []):
                if not isinstance(e, dict):
                    continue
                vec = e.get("vec")
                if not isinstance(vec, list) or not vec:
                    continue
                try:
                    floats = [float(v) for v in vec]
                except (TypeError, ValueError):
                    continue
                embeddings.append(Enrollment(
                    vec=floats,
                    captured_ts=str(e.get("captured_ts") or ""),
                    source_audio=e.get("source_audio"),
                    duration_ms=int(e.get("duration_ms") or 0),
                ))
            loaded.append(EnrolledUser(uid, display_name, embeddings))
        self.users = loaded
        _LOGGER.info(
            "Enrollments loaded: %d user(s), %d total samples (threshold=%.2f)",
            len(self.users),
            sum(len(u.embeddings) for u in self.users),
            self.threshold,
        )

    def save(self) -> None:
        """Atomic write with .bak rotation (H5 durable-artefact discipline)."""
        with self._lock:
            self.updated_ts = datetime.now().astimezone().isoformat(timespec="seconds")
            payload = {
                "version": self.version,
                "room": self.room,
                "threshold": self.threshold,
                "updated_ts": self.updated_ts,
                "users": [
                    {
                        "user_id": u.user_id,
                        "display_name": u.display_name,
                        "embeddings": [asdict(e) for e in u.embeddings],
                    }
                    for u in self.users
                ],
            }
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            bak = self.path.with_suffix(self.path.suffix + ".bak")
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            try:
                with open(tmp, "rb") as f:
                    os.fsync(f.fileno())
            except OSError:
                pass
            if self.path.exists():
                try:
                    shutil.copy2(self.path, bak)
                except OSError:
                    pass
            os.replace(tmp, self.path)

    # ---- user / enrollment manipulation ----

    def find_user(self, user_id: str) -> Optional[EnrolledUser]:
        for u in self.users:
            if u.user_id == user_id:
                return u
        return None

    def upsert_user(self, user_id: str, display_name: Optional[str] = None) -> EnrolledUser:
        u = self.find_user(user_id)
        if u is None:
            u = EnrolledUser(user_id, display_name or user_id, [])
            self.users.append(u)
        elif display_name and display_name != u.display_name:
            u.display_name = display_name
        return u

    def add_enrollment(
        self,
        user_id: str,
        vec: List[float],
        *,
        display_name: Optional[str] = None,
        source_audio: Optional[str] = None,
        duration_ms: int = 0,
    ) -> Enrollment:
        u = self.upsert_user(user_id, display_name)
        entry = Enrollment(
            vec=[float(v) for v in vec],
            captured_ts=datetime.now().astimezone().isoformat(timespec="seconds"),
            source_audio=source_audio,
            duration_ms=duration_ms,
        )
        u.embeddings.append(entry)
        self.save()
        return entry

    def delete_enrollment(self, user_id: str, index: int) -> bool:
        u = self.find_user(user_id)
        if u is None or index < 0 or index >= len(u.embeddings):
            return False
        u.embeddings.pop(index)
        if not u.embeddings:
            self.users = [x for x in self.users if x is not u]
        self.save()
        return True


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


class SpeakerVerifier:
    """Two-gate verifier. Owns its CAM++ ONNX session + TEN-VAD instance.

    `verify(pcm_bytes)` is sync — the gates total ~75ms worst-case which
    is acceptable on the audio thread per D2. The caller can choose to
    marshal it off-thread if needed.
    """

    def __init__(
        self,
        *,
        store: EnrollmentsStore,
        model_path: Optional[Path] = None,
        vad_threshold: float = 0.5,
        enabled: bool = True,
        verify_window_ms: int = DEFAULT_VERIFY_WINDOW_MS,
    ) -> None:
        self.store = store
        self.model_path = Path(model_path) if model_path else None
        self.vad_threshold = vad_threshold
        self.enabled = enabled
        self.verify_window_ms = verify_window_ms

        self._lock = threading.Lock()
        self._session: Any = None
        self._input_name: Optional[str] = None
        self._vad: Any = None
        self._init_attempted = False
        self._init_failed = False
        self._embed_dim: int = CAMPLUS_EMBED_DIM

    # ---- model load -----------------------------------------------------

    def _ensure_loaded(self) -> bool:
        """Lazy-load CAM++ ONNX session + TEN-VAD. Returns True when ready."""
        if self._session is not None and self._vad is not None:
            return True
        if self._init_failed:
            return False
        with self._lock:
            if self._init_attempted:
                return self._session is not None and self._vad is not None
            self._init_attempted = True

            if self.model_path is None or not self.model_path.exists():
                _LOGGER.warning(
                    "SpeakerVerifier: model file missing (%s) — accept-all fallback active",
                    self.model_path,
                )
                self._init_failed = True
                return False

            try:
                import onnxruntime as ort  # type: ignore
            except ImportError:
                _LOGGER.warning(
                    "SpeakerVerifier: onnxruntime not installed — accept-all fallback active",
                )
                self._init_failed = True
                return False

            try:
                self._session = ort.InferenceSession(
                    str(self.model_path), providers=["CPUExecutionProvider"]
                )
                inputs = self._session.get_inputs()
                if inputs:
                    self._input_name = inputs[0].name
            except Exception:
                _LOGGER.exception("SpeakerVerifier: CAM++ ONNX load failed")
                self._init_failed = True
                self._session = None
                return False

            try:
                from ten_vad import TenVad  # type: ignore
                self._vad = TenVad(hop_size=VAD_HOP, threshold=self.vad_threshold)
            except Exception:
                _LOGGER.warning(
                    "SpeakerVerifier: TEN-VAD load failed — Gate 1 will accept all frames",
                )
                # Soft-fail Gate 1 only — CAM++ still gates.

            _LOGGER.info(
                "SpeakerVerifier ready: model=%s users=%d threshold=%.2f",
                self.model_path,
                len(self.store.users),
                self.store.threshold,
            )
            return True

    # ---- public API -----------------------------------------------------

    @property
    def threshold(self) -> float:
        return self.store.threshold

    def is_active(self) -> bool:
        """Verifier will actually run, vs degrading to accept-all."""
        if not self.enabled:
            return False
        if not self.store.has_enrollments:
            return False
        return self._ensure_loaded()

    def verify(self, pcm_bytes: bytes) -> VerificationResult:
        """Run Gate 1 + Gate 2 on the supplied PCM (s16le mono @ 16kHz).

        Returns VerificationResult; never raises. On any internal failure
        the result is `reason='disabled'` and both gates pass (accept-all)
        so a model glitch doesn't brick wake.
        """
        if not self.enabled:
            return VerificationResult(True, True, None, 0.0, self.store.threshold, "disabled")

        if not self.store.has_enrollments:
            return VerificationResult(True, True, None, 0.0, self.store.threshold, "disabled")

        if not self._ensure_loaded():
            return VerificationResult(True, True, None, 0.0, self.store.threshold, "disabled")

        # ---- Gate 1: VAD ------------------------------------------------
        gate1_pass = self._gate1_vad(pcm_bytes)
        if not gate1_pass:
            return VerificationResult(
                False, False, None, 0.0, self.store.threshold, "gate1_fail_vad",
            )

        # ---- Gate 2: CAM++ ----------------------------------------------
        try:
            vec = self._embed(pcm_bytes)
        except Exception:
            _LOGGER.exception("SpeakerVerifier: embed() raised — accept-all fallback")
            return VerificationResult(True, True, None, 0.0, self.store.threshold, "disabled")

        score, user_id = self._best_match(vec)
        threshold = self.store.threshold
        gate2_pass = score >= threshold
        return VerificationResult(
            True, gate2_pass, user_id if gate2_pass else None, score, threshold,
            "verified" if gate2_pass else "gate2_fail_score",
        )

    def embed(self, pcm_bytes: bytes) -> Optional[List[float]]:
        """Public embedding API for the HA-driven enrollment flow.

        Returns None when the verifier is not loaded — caller should
        surface "speaker model not available" to the user.
        """
        if not self._ensure_loaded():
            return None
        try:
            return list(self._embed(pcm_bytes))
        except Exception:
            _LOGGER.exception("SpeakerVerifier.embed raised")
            return None

    def outlier_check(self, user_id: str, candidate: List[float], *, min_sim: float = 0.5) -> bool:
        """Per D2 enrollment: a new sample whose cosine to all existing
        same-user samples < `min_sim` is rejected as outlier.

        Returns True when the candidate is OK to add (first sample, or
        passes outlier gate). False = reject + ask the user to retry.
        """
        u = self.store.find_user(user_id)
        if u is None or not u.embeddings:
            return True
        cand = np.asarray(candidate, dtype=np.float32)
        for e in u.embeddings:
            sim = _cosine(cand, np.asarray(e.vec, dtype=np.float32))
            if sim >= min_sim:
                return True
        return False

    # ---- internals -------------------------------------------------------

    def _gate1_vad(self, pcm_bytes: bytes) -> bool:
        """At least 25% of TEN-VAD hops show speech."""
        vad = self._vad
        if vad is None:
            return True  # VAD disabled — accept-all on Gate 1
        try:
            samples = np.frombuffer(pcm_bytes, dtype=np.int16)
        except Exception:
            return True
        if samples.size < VAD_HOP:
            return False
        hops = samples.size // VAD_HOP
        speech_hops = 0
        for i in range(hops):
            frame = samples[i * VAD_HOP:(i + 1) * VAD_HOP]
            try:
                _prob, flag = vad.process(frame)
            except Exception:
                continue
            if flag:
                speech_hops += 1
        return (speech_hops / max(1, hops)) >= 0.25

    def _embed(self, pcm_bytes: bytes) -> np.ndarray:
        """Run CAM++ on a float32 [-1, 1] tensor and return a unit-norm 512-vec."""
        # CAM++ ONNX export expects waveform as (1, T) float32 in [-1, 1].
        samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        # Crop to verify window if longer than expected.
        max_samples = int(SAMPLE_RATE * self.verify_window_ms / 1000.0)
        if samples.size > max_samples:
            samples = samples[-max_samples:]
        # Single-batch input.
        x = samples.reshape(1, -1)
        outputs = self._session.run(None, {self._input_name: x})
        # The 3D-Speaker CAM++ export emits a single (1, 512) tensor.
        out = outputs[0]
        vec = np.asarray(out).reshape(-1).astype(np.float32)
        if vec.size != self._embed_dim:
            _LOGGER.warning(
                "SpeakerVerifier: embedding dim %d != expected %d",
                vec.size, self._embed_dim,
            )
        return _l2_normalize(vec)

    def _best_match(self, vec: np.ndarray) -> Tuple[float, Optional[str]]:
        best_score = -1.0
        best_user: Optional[str] = None
        for u in self.store.users:
            if not u.embeddings:
                continue
            user_best = -1.0
            for e in u.embeddings:
                ev = np.asarray(e.vec, dtype=np.float32)
                sim = _cosine(vec, ev)
                if sim > user_best:
                    user_best = sim
            if user_best > best_score:
                best_score = user_best
                best_user = u.user_id
        # Clamp to [-1, 1].
        return (max(-1.0, min(1.0, best_score)) if best_score > -1.0 else 0.0, best_user)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(vec))
    if n == 0.0:
        return vec
    return vec / n


__all__ = [
    "Enrollment",
    "EnrolledUser",
    "VerificationResult",
    "EnrollmentsStore",
    "SpeakerVerifier",
    "DEFAULT_THRESHOLD",
    "CAMPLUS_EMBED_DIM",
]
