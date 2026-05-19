"""Stage C — WakeCapture: rolling ring buffer + on-wake atomic write.

Captures the 3 seconds leading up to every wake event, writes WAV + sidecar
JSON per M.1, auto-labels via session terminal-state mapping per D1.

Threading model
---------------
- ``feed(audio_chunk)`` runs in the audio worker thread (fast deque append).
- ``on_wake_fire(...)`` runs in the audio worker thread; copies the ring
  buffer to a bytes blob synchronously (fast) then queues the actual
  WAV + sidecar write onto the asyncio loop's default executor. Returns
  ``wake_id`` so ``__main__`` can hand it to ``satellite.wakeup(wake_id=...)``.
- ``bind_session(wake_id, session_id, generation)`` runs on the loop thread,
  called by ``DeviceSession.transition_to`` when session_id is minted.
- ``update_label(session_id, reason, cancel_reason)`` runs on the loop thread,
  called by ``DeviceSession.transition_to`` on entry to IDLE.

The orphan + retention sweeps run as periodic asyncio tasks owned by the
WakeCapture instance.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import struct
import threading
import time
import uuid
import wave
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple

_LOGGER = logging.getLogger(__name__)


SAMPLE_RATE = 16000
SAMPLE_WIDTH_BYTES = 2
CHANNELS = 1
RING_BUFFER_SECONDS = 3.0
ORPHAN_TIMEOUT_SECONDS = 60.0
SCHEMA_VERSION = 1


CancelLabelMap = {
    "RED_BUTTON_SOFT": ("negative", "RED_BUTTON_SOFT"),
    "RED_BUTTON_HARD": ("negative", "RED_BUTTON_HARD"),
    "MIC_MUTE_SOFT": ("negative", "MIC_MUTE_SOFT"),
    "MIC_MUTE_HARD": ("negative", "MIC_MUTE_HARD"),
    "STOP_WORD_INPROCESS": ("negative", "STOP_WORD_INPROCESS"),
    "SILENCE_TIMEOUT": ("negative", "SILENCE_TIMEOUT"),
    "GATE2_REJECT": ("gate2_reject", "gate2_reject"),
    "DASHBOARD": ("ambiguous", "DASHBOARD"),
    "EXTERNAL": ("ambiguous", "EXTERNAL"),
    "STOP_EVERYTHING": ("ambiguous", "STOP_EVERYTHING"),
    "MIC_CAPTURE_FAILED": ("ambiguous", "MIC_CAPTURE_FAILED"),
    "MISSING_COMPONENT": ("ambiguous", "MISSING_COMPONENT"),
}


TransitionLabelMap = {
    "reply_done": ("positive", "pipeline_complete"),
    "no_speech": ("negative", "no_speech"),
    "empty_reply": ("ambiguous", "empty_reply"),
    "stale_generation": ("ambiguous", "stale_generation"),
}


def derive_label(
    reason: Optional[str], cancel_reason: Optional[str]
) -> Tuple[str, str]:
    """Map (IDLE-transition reason, K.3 cancel_reason) → (label, label_reason).

    `cancel_reason` wins when present. Falls back to `reason`. Unknown reasons
    yield `("ambiguous", f"unknown:{reason}")` rather than crashing — the
    triage UI can sort these into the manual bucket.
    """
    if cancel_reason and cancel_reason in CancelLabelMap:
        return CancelLabelMap[cancel_reason]
    if cancel_reason:
        return ("ambiguous", f"unknown_cancel:{cancel_reason}")
    if reason and reason in TransitionLabelMap:
        return TransitionLabelMap[reason]
    return ("ambiguous", f"unknown_reason:{reason or 'none'}")


def _utc_iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    data = json.dumps(payload, sort_keys=True, indent=2).encode("utf-8")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _atomic_write_wav(path: Path, pcm_bytes: bytes) -> None:
    """Render WAV to bytes, write atomically with fsync.

    Building the WAV in memory first lets us own the file lifecycle: open
    once for writing, write all bytes, flush, fsync, close, rename. Avoids
    the Windows "fsync after wave.open closed" gotcha.
    """
    import io
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH_BYTES)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm_bytes)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        f.write(buf.getvalue())
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass
    os.replace(tmp, path)


class WakeCapture:
    """Owns the wake-capture ring buffer + sidecar JSON store.

    One instance per LVA process (one room). Sidecars live under
    ``capture_dir``; WAVs sit next to them. The presence of the WAV's sidecar
    JSON is the success marker (H5 durable-artefact discipline).
    """

    def __init__(
        self,
        *,
        capture_dir: Path,
        room: str,
        device_id: str,
        max_files: int = 5000,
        ring_seconds: float = RING_BUFFER_SECONDS,
        orphan_timeout_s: float = ORPHAN_TIMEOUT_SECONDS,
        wake_model_name: str = "alexa.tflite",
        wake_sensitivity: float = 0.5,
        clock: Any = time.time,
    ) -> None:
        self.capture_dir = Path(capture_dir)
        self.capture_dir.mkdir(parents=True, exist_ok=True)
        self.room = room
        self.device_id = device_id
        self.max_files = max_files
        self.orphan_timeout_s = orphan_timeout_s
        self.wake_model_name = wake_model_name
        self.wake_sensitivity = wake_sensitivity
        self._clock = clock

        # Ring buffer: bytes deque of audio chunks, trimmed to ring_seconds.
        # Stored as raw PCM s16le mono @ 16kHz.
        self._ring_lock = threading.Lock()
        self._ring: Deque[bytes] = deque()
        self._ring_bytes = 0
        self._ring_max_bytes = int(
            ring_seconds * SAMPLE_RATE * SAMPLE_WIDTH_BYTES * CHANNELS
        )

        # wake_id → session_id binding state. Captured-but-unbound entries
        # have session_id=None until DS.transition_to mints one. The entry
        # also carries the *unwritten* sidecar dict so `_write_capture` can
        # merge any bind_session updates that landed before the executor
        # got around to the disk write (advisor-flagged race).
        self._pending_lock = threading.Lock()
        self._pending: Dict[str, Dict[str, Any]] = {}

        # session_id → wake_id reverse lookup, set by bind_session.
        self._session_to_wake: Dict[str, str] = {}

        self._executor_loop: Optional[asyncio.AbstractEventLoop] = None
        self._orphan_task: Optional[asyncio.Task] = None
        self._retention_task: Optional[asyncio.Task] = None

    # -- audio path ---------------------------------------------------------

    def feed(self, audio_chunk: bytes) -> None:
        """Append a PCM chunk to the ring buffer. Audio-thread hot path —
        keep this O(1) per chunk."""
        with self._ring_lock:
            self._ring.append(audio_chunk)
            self._ring_bytes += len(audio_chunk)
            while self._ring_bytes > self._ring_max_bytes and self._ring:
                dropped = self._ring.popleft()
                self._ring_bytes -= len(dropped)

    def _snapshot_ring(self) -> bytes:
        with self._ring_lock:
            return b"".join(self._ring)

    def on_wake_fire(
        self,
        *,
        score: float,
        peak_score: Optional[float] = None,
        model: Optional[str] = None,
        sensitivity: Optional[float] = None,
    ) -> str:
        """Snapshot the ring buffer + queue WAV + sidecar write. Returns
        ``wake_id`` synchronously so the satellite can stash it.

        Runs in the audio worker thread. Buffer copy is in-thread (fast);
        disk I/O is punted to the asyncio loop's default executor so we
        don't block wake-word inference.
        """
        ts = self._clock()
        wake_id_str = f"{datetime.fromtimestamp(ts, tz=timezone.utc).strftime('%Y%m%dT%H%M%S_%f')}_{uuid.uuid4().hex[:6]}"
        pcm_bytes = self._snapshot_ring()

        sidecar = {
            "version": SCHEMA_VERSION,
            "ts": _utc_iso(ts),
            "ts_epoch": ts,
            "room": self.room,
            "device_id": self.device_id,
            "generation": None,
            "session_id": None,
            "score": float(score),
            "peak_score": float(peak_score) if peak_score is not None else float(score),
            "model": model or self.wake_model_name,
            "sensitivity": float(sensitivity) if sensitivity is not None else self.wake_sensitivity,
            "audio_meta": {
                "sample_rate": SAMPLE_RATE,
                "channels": CHANNELS,
                "duration_ms": int(len(pcm_bytes) / (SAMPLE_RATE * SAMPLE_WIDTH_BYTES * CHANNELS) * 1000),
                "format": "pcm_s16le",
            },
            "label": None,
            "label_reason": None,
            "label_updated_ts": None,
            "speaker_match": None,
            "asr_text": None,
            "wake_id_str": wake_id_str,
        }

        with self._pending_lock:
            self._pending[wake_id_str] = {
                "session_id": None,
                "generation": None,
                "created_ts": ts,
                "sidecar_path": self.capture_dir / f"{wake_id_str}.wav.json",
                "wav_path": self.capture_dir / f"{wake_id_str}.wav",
                "bound_ts": None,
                "sidecar_template": sidecar,
                "written": False,
            }

        loop = self._executor_loop
        if loop is not None:
            loop.call_soon_threadsafe(self._schedule_write, wake_id_str, pcm_bytes)
        else:
            try:
                self._write_capture(wake_id_str, pcm_bytes)
            except Exception:
                _LOGGER.exception("Direct wake-capture write failed for %s", wake_id_str)

        return wake_id_str

    def _schedule_write(self, wake_id_str: str, pcm_bytes: bytes) -> None:
        loop = self._executor_loop
        if loop is None:
            return
        loop.run_in_executor(None, self._write_capture, wake_id_str, pcm_bytes)

    def _write_capture(self, wake_id_str: str, pcm_bytes: bytes) -> None:
        """Materialise WAV + sidecar to disk. Holds `_pending_lock` through
        the entire disk write so `bind_session` cannot observe ``written=True``
        before the file exists. Without that, this race fired in production:

          T0  _write_capture acquires lock, marks written=True, RELEASES lock
          T1  bind_session acquires lock, sees written=True, releases lock
          T2  bind_session calls _patch_sidecar — file not on disk yet —
              FileNotFoundError, silent return
          T3  _write_capture finally writes the sidecar (session_id=None)

        Result: sidecar permanently missing the bind data. Holding the lock
        across the ~10ms disk write closes that window. The disk write is
        small (a few KB) and serialised against the audio thread only via
        `on_wake_fire`'s brief append, which is fine.
        """
        try:
            with self._pending_lock:
                entry = self._pending.get(wake_id_str)
                if entry is None:
                    _LOGGER.debug("_write_capture: entry vanished for %s", wake_id_str)
                    return
                sidecar = dict(entry["sidecar_template"])
                sidecar["session_id"] = entry.get("session_id")
                sidecar["generation"] = entry.get("generation")
                wav_path = entry["wav_path"]
                sidecar_path = entry["sidecar_path"]
                _atomic_write_wav(wav_path, pcm_bytes)
                _atomic_write_json(sidecar_path, sidecar)
                entry["written"] = True
            _LOGGER.debug("Wake capture written: %s (%d bytes)", wake_id_str, len(pcm_bytes))
        except Exception:
            _LOGGER.exception("Wake-capture write failed for %s", wake_id_str)

    # -- session binding ---------------------------------------------------

    def bind_session(self, wake_id_str: str, session_id: str, generation: int) -> None:
        """Associate a wake_id with the DS-minted session_id. Loop-thread.

        When the sidecar already exists on disk (executor write landed
        first), patches it in place. When it doesn't (bind raced ahead of
        the executor), just stashes the values — `_write_capture` reads them
        out of the pending entry at materialise-time so the on-disk JSON is
        correct regardless of ordering.
        """
        with self._pending_lock:
            entry = self._pending.get(wake_id_str)
            if entry is None:
                _LOGGER.debug("bind_session: unknown wake_id_str=%s; dropping", wake_id_str)
                return
            entry["session_id"] = session_id
            entry["generation"] = generation
            entry["bound_ts"] = self._clock()
            self._session_to_wake[session_id] = wake_id_str
            sidecar_path = entry["sidecar_path"]
            already_written = entry.get("written", False)

        if already_written:
            self._patch_sidecar(sidecar_path, {"session_id": session_id, "generation": generation})

    def update_label(
        self, session_id: str, reason: Optional[str], cancel_reason: Optional[str] = None
    ) -> None:
        """Resolve and write the final label for a session. Loop-thread.

        Called from DS.transition_to(IDLE) with the session_id that's being
        cleared. If no wake_id was ever bound (rare — DS minted session_id
        without WAKING), this is a no-op.
        """
        with self._pending_lock:
            wake_id_str = self._session_to_wake.pop(session_id, None)
            if wake_id_str is None:
                return
            entry = self._pending.pop(wake_id_str, None)
            if entry is None:
                return
            sidecar_path = entry["sidecar_path"]

        label, label_reason = derive_label(reason, cancel_reason)
        self._patch_sidecar(
            sidecar_path,
            {
                "label": label,
                "label_reason": label_reason,
                "label_updated_ts": _utc_iso(self._clock()),
            },
        )

    # ---- Stage D — SpeakerVerifier integration --------------------------

    def snapshot_recent_pcm(self, duration_s: float) -> bytes:
        """Return the most-recent `duration_s` seconds of the ring as PCM.

        Used by SpeakerVerifier in `process_audio` (audio thread). The ring
        is mutated only at the tail by feed(), so a tail-slice can race
        with one append — but the slice we extract under the lock is a
        consistent snapshot of bytes that arrived strictly before this
        call. Bounded by the ring's actual extent (3s).
        """
        max_bytes = int(duration_s * SAMPLE_RATE * SAMPLE_WIDTH_BYTES * CHANNELS)
        with self._ring_lock:
            joined = b"".join(self._ring)
            if max_bytes >= len(joined):
                return joined
            return joined[-max_bytes:]

    def update_speaker_match(self, wake_id_str: str, match: Dict[str, Any]) -> None:
        """Set `speaker_match` on a wake's sidecar. Safe to call before or
        after the disk write — pending entries cache the update; on-disk
        entries are patched in place."""
        sidecar_path = self.capture_dir / f"{wake_id_str}.wav.json"
        with self._pending_lock:
            entry = self._pending.get(wake_id_str)
            if entry is not None:
                entry["sidecar_template"]["speaker_match"] = match
                if not entry.get("written", False):
                    return
        self._patch_sidecar(sidecar_path, {"speaker_match": match})

    def update_wake_label(
        self, wake_id_str: str, label: str, label_reason: str,
    ) -> bool:
        """Set the sidecar label directly by wake_id (no session_id required).

        Used by the Stage D pre-LISTENING gates (Gate 1 silent drop, Gate 2
        reject) which fire before any session_id is minted. Returns True on
        a successful patch."""
        if label not in ("positive", "negative", "ambiguous", "gate2_reject"):
            return False
        sidecar_path = self.capture_dir / f"{wake_id_str}.wav.json"
        with self._pending_lock:
            entry = self._pending.get(wake_id_str)
            if entry is not None and not entry.get("written", False):
                entry["sidecar_template"]["label"] = label
                entry["sidecar_template"]["label_reason"] = label_reason
                entry["sidecar_template"]["label_updated_ts"] = _utc_iso(self._clock())
                # Drop from pending — Stage D wake events never bind a
                # session, so leaving them in `_pending` would block the
                # orphan sweep from cleaning up disk.
                self._pending.pop(wake_id_str, None)
                return True
        if not sidecar_path.exists():
            return False
        self._patch_sidecar(
            sidecar_path,
            {
                "label": label,
                "label_reason": label_reason,
                "label_updated_ts": _utc_iso(self._clock()),
            },
        )
        with self._pending_lock:
            self._pending.pop(wake_id_str, None)
        return True

    def manual_label(self, wake_id_str: str, label: str, user: str = "manual") -> bool:
        """HTTP-endpoint entry: rewrite sidecar with a manually-chosen label."""
        if label not in ("positive", "negative", "ambiguous", "gate2_reject"):
            return False
        sidecar_path = self.capture_dir / f"{wake_id_str}.wav.json"
        if not sidecar_path.exists():
            return False
        self._patch_sidecar(
            sidecar_path,
            {
                "label": label,
                "label_reason": f"manual_{user}",
                "label_updated_ts": _utc_iso(self._clock()),
            },
        )
        with self._pending_lock:
            entry = self._pending.pop(wake_id_str, None)
            if entry is not None and entry.get("session_id"):
                self._session_to_wake.pop(entry["session_id"], None)
        return True

    def delete_capture(self, wake_id_str: str) -> bool:
        """HTTP-endpoint entry: hard-delete WAV + sidecar. Used for triage
        items the user wants to exclude from training entirely (neither
        positive nor negative). Idempotent — missing files are not an error.
        Returns True when at least one of the two files was on disk.
        """
        sidecar_path = self.capture_dir / f"{wake_id_str}.wav.json"
        wav_path = self.capture_dir / f"{wake_id_str}.wav"
        removed = False
        for path in (sidecar_path, wav_path):
            try:
                path.unlink()
                removed = True
            except FileNotFoundError:
                continue
            except Exception:
                _LOGGER.exception("delete_capture: unlink failed for %s", path)
                return False
        with self._pending_lock:
            entry = self._pending.pop(wake_id_str, None)
            if entry is not None and entry.get("session_id"):
                self._session_to_wake.pop(entry["session_id"], None)
        return removed

    def _patch_sidecar(self, path: Path, updates: Dict[str, Any]) -> None:
        try:
            with open(path) as f:
                data = json.load(f)
        except FileNotFoundError:
            _LOGGER.debug("Sidecar missing during patch: %s", path)
            return
        except Exception:
            _LOGGER.exception("Sidecar read failed: %s", path)
            return
        data.update(updates)
        try:
            _atomic_write_json(path, data)
        except Exception:
            _LOGGER.exception("Sidecar patch write failed: %s", path)

    # -- background sweeps -------------------------------------------------

    def attach_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Hand the WakeCapture the asyncio loop used to offload disk writes
        + run periodic sweeps. Called once at startup from __main__."""
        self._executor_loop = loop

    async def start_background_sweeps(self) -> None:
        """Spawn the orphan + retention sweep tasks. Idempotent."""
        if self._orphan_task is None:
            self._orphan_task = asyncio.create_task(self._orphan_sweep_loop())
        if self._retention_task is None:
            self._retention_task = asyncio.create_task(self._retention_sweep_loop())

    async def stop_background_sweeps(self) -> None:
        for task in (self._orphan_task, self._retention_task):
            if task is not None:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                except Exception:
                    _LOGGER.exception("Sweep task raised on shutdown")
        self._orphan_task = None
        self._retention_task = None

    async def _orphan_sweep_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(10.0)
                self._sweep_orphans_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                _LOGGER.exception("Orphan sweep raised; continuing")

    def _sweep_orphans_once(self) -> None:
        now = self._clock()
        expired: List[Tuple[str, Optional[str], bool]] = []
        with self._pending_lock:
            for wake_id_str, entry in list(self._pending.items()):
                age = now - entry["created_ts"]
                if age <= self.orphan_timeout_s:
                    continue
                expired.append((wake_id_str, entry.get("session_id"), entry["bound_ts"] is None))
                sid = entry.get("session_id")
                if sid:
                    self._session_to_wake.pop(sid, None)
                self._pending.pop(wake_id_str, None)

        for wake_id_str, session_id, unbound in expired:
            sidecar_path = self.capture_dir / f"{wake_id_str}.wav.json"
            label_reason = "orphan_unbound" if unbound else "orphan"
            self._patch_sidecar(
                sidecar_path,
                {
                    "label": "ambiguous",
                    "label_reason": label_reason,
                    "label_updated_ts": _utc_iso(self._clock()),
                },
            )
            _LOGGER.info("Wake-capture orphaned: %s (%s)", wake_id_str, label_reason)

    async def _retention_sweep_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(300.0)
                self._enforce_retention()
            except asyncio.CancelledError:
                raise
            except Exception:
                _LOGGER.exception("Retention sweep raised; continuing")

    def _enforce_retention(self) -> int:
        """Delete oldest WAV+sidecar pairs until count ≤ max_files. Returns
        the number of pairs deleted."""
        sidecars = sorted(
            self.capture_dir.glob("*.wav.json"),
            key=lambda p: p.stat().st_mtime,
        )
        excess = len(sidecars) - self.max_files
        if excess <= 0:
            return 0
        deleted = 0
        for path in sidecars[:excess]:
            wav = path.with_suffix("")  # strips .json → .wav
            try:
                if wav.exists():
                    wav.unlink()
                path.unlink()
                deleted += 1
            except Exception:
                _LOGGER.exception("Retention delete failed for %s", path)
        if deleted:
            _LOGGER.info("Retention sweep deleted %d wake captures", deleted)
        return deleted

    # -- read paths (HTTP endpoint + Discovery) ----------------------------

    def list_captures(
        self,
        *,
        include_all: bool = False,
        page: int = 0,
        page_size: int = 50,
    ) -> Dict[str, Any]:
        """Return a paginated list of captures. Default filter: label is
        null or ambiguous (the triage bucket); ``include_all=True`` returns
        everything."""
        sidecars = sorted(
            self.capture_dir.glob("*.wav.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        items: List[Dict[str, Any]] = []
        for path in sidecars:
            try:
                with open(path) as f:
                    data = json.load(f)
            except Exception:
                continue
            if not include_all and data.get("label") not in (None, "ambiguous"):
                continue
            items.append(self._summarise(data))
        start = page * page_size
        end = start + page_size
        return {
            "total": len(items),
            "page": page,
            "page_size": page_size,
            "items": items[start:end],
        }

    @staticmethod
    def _summarise(sidecar: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "wake_id_str": sidecar.get("wake_id_str"),
            "wake_id": sidecar.get("wake_id"),
            "ts": sidecar.get("ts"),
            "room": sidecar.get("room"),
            "score": sidecar.get("score"),
            "label": sidecar.get("label"),
            "label_reason": sidecar.get("label_reason"),
            "duration_ms": (sidecar.get("audio_meta") or {}).get("duration_ms"),
            "asr_text": sidecar.get("asr_text"),
        }

    def read_wav_bytes(self, wake_id_str: str) -> Optional[bytes]:
        wav_path = self.capture_dir / f"{wake_id_str}.wav"
        if not wav_path.exists():
            return None
        try:
            return wav_path.read_bytes()
        except Exception:
            _LOGGER.exception("Failed to read %s", wav_path)
            return None

    def stats_24h(self) -> Dict[str, int]:
        """Aggregate counts for HA Discovery sensors."""
        cutoff = self._clock() - 86400.0
        counts = {
            "total_24h": 0,
            "positive_24h": 0,
            "negative_24h": 0,
            "ambiguous_24h": 0,
            "gate2_reject_24h": 0,
            "pending_triage_count": 0,
        }
        for path in self.capture_dir.glob("*.wav.json"):
            try:
                with open(path) as f:
                    data = json.load(f)
            except Exception:
                continue
            ts_epoch = data.get("ts_epoch")
            label = data.get("label")
            if isinstance(ts_epoch, (int, float)) and ts_epoch >= cutoff:
                counts["total_24h"] += 1
                if label == "positive":
                    counts["positive_24h"] += 1
                elif label == "negative":
                    counts["negative_24h"] += 1
                elif label == "ambiguous":
                    counts["ambiguous_24h"] += 1
                elif label == "gate2_reject":
                    counts["gate2_reject_24h"] += 1
            if label in (None, "ambiguous"):
                counts["pending_triage_count"] += 1
        return counts
