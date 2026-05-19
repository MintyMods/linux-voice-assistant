"""Stage C / D — MQTT Discovery surface.

Module hosts every Discovery publisher per N (HA MQTT Discovery manifest):

  Stage C — wake-capture stats sensors (``WakeCaptureDiscovery``).
  Stage D — speaker-verification live tunables (``SpeakerVerifierDiscovery``):
    - number.calisto_<room>_sv_threshold      (Template B, 0.40–0.90 step 0.01)
    - switch.calisto_<room>_sv_audible_notify (Template C, boolean)

The full N.2 manifest (every voice entity per room) lands incrementally —
publishers are added to this module rather than scattered. All publishes go
through ``HABridge.publish`` so we share a single MQTT client; mqtt_router
(K.15) owns the subscription matrix.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, Optional

_LOGGER = logging.getLogger(__name__)


DISCOVERY_PREFIX = "homeassistant"
STATE_TOPIC_PREFIX = "calisto"


SENSOR_DEFS = [
    ("wake_captures_24h", "Wake Captures 24h", "total_24h", "mdi:waveform"),
    ("wake_positives_24h", "Wake Positives 24h", "positive_24h", "mdi:check-circle"),
    ("wake_negatives_24h", "Wake Negatives 24h", "negative_24h", "mdi:close-circle"),
    ("wake_ambiguous_24h", "Wake Ambiguous 24h", "ambiguous_24h", "mdi:help-circle"),
    ("wake_gate2_rejects_24h", "Wake Gate-2 Rejects 24h", "gate2_reject_24h", "mdi:account-cancel"),
    ("wake_pending_triage", "Wake Pending Triage", "pending_triage_count", "mdi:inbox-multiple"),
]


class WakeCaptureDiscovery:
    """Publishes Discovery configs + periodic state updates for wake-capture
    stats sensors. One instance per LVA process."""

    def __init__(
        self,
        *,
        ha_bridge: Any,
        wake_capture: Any,
        room: str,
        device_id: str,
        http_base_url: Optional[str] = None,
        publish_interval_s: float = 60.0,
    ) -> None:
        self.ha_bridge = ha_bridge
        self.wake_capture = wake_capture
        self.room = room
        self.device_id = device_id
        self.http_base_url = http_base_url
        self.publish_interval_s = publish_interval_s
        self._task: Optional[asyncio.Task] = None
        self._configs_published = False

    @property
    def state_topic(self) -> str:
        return f"{STATE_TOPIC_PREFIX}/{self.room}/wake_capture/stats"

    def _device_block(self) -> Dict[str, Any]:
        return {
            "identifiers": [f"calisto_{self.room}"],
            "name": f"Calisto {self.room.title()}",
            "manufacturer": "Minty",
            "model": "Calisto v1",
        }

    def publish_configs(self) -> int:
        """Push Discovery configs for every sensor. Idempotent; safe to call
        on every reconnect (paho will dedupe at the broker via retained).
        Returns the number of publishes successfully queued."""
        published = 0
        for obj_id, name, attr, icon in SENSOR_DEFS:
            unique_id = f"calisto_{self.room}_{obj_id}"
            cfg_topic = f"{DISCOVERY_PREFIX}/sensor/{unique_id}/config"
            payload = {
                "name": name,
                "unique_id": unique_id,
                "object_id": unique_id,
                "state_topic": self.state_topic,
                "value_template": "{{ value_json." + attr + " | default(0) }}",
                "icon": icon,
                "device": self._device_block(),
            }
            if self.ha_bridge.publish(cfg_topic, json.dumps(payload), retain=True):
                published += 1

        if self.http_base_url:
            unique_id = f"calisto_{self.room}_wake_capture_http_url"
            cfg_topic = f"{DISCOVERY_PREFIX}/sensor/{unique_id}/config"
            url_topic = f"{STATE_TOPIC_PREFIX}/{self.room}/wake_capture/http_url"
            payload = {
                "name": "Wake Capture HTTP URL",
                "unique_id": unique_id,
                "object_id": unique_id,
                "state_topic": url_topic,
                "icon": "mdi:link",
                "device": self._device_block(),
            }
            if self.ha_bridge.publish(cfg_topic, json.dumps(payload), retain=True):
                self.ha_bridge.publish(url_topic, self.http_base_url, retain=True)
                published += 1

        self._configs_published = True
        return published

    def publish_state(self) -> bool:
        stats = self.wake_capture.stats_24h()
        return self.ha_bridge.publish(self.state_topic, json.dumps(stats), retain=True)

    async def start(self) -> None:
        if self._task is not None:
            return
        try:
            n = self.publish_configs()
            _LOGGER.info("WakeCaptureDiscovery published %d configs on startup", n)
            self.publish_state()
        except Exception:
            _LOGGER.exception("WakeCaptureDiscovery initial publish raised")
        self._task = asyncio.create_task(self._publish_loop())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            except Exception:
                _LOGGER.exception("Discovery loop raised on shutdown")
        self._task = None

    async def _publish_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(self.publish_interval_s)
                self.publish_state()
            except asyncio.CancelledError:
                raise
            except Exception:
                _LOGGER.exception("Discovery state publish raised; continuing")


# ---------------------------------------------------------------------------
# Stage D — SpeakerVerifier live-tunables (N.2 entries 143 + 153)
# ---------------------------------------------------------------------------


# N.2 row 143 — number, range / step locked by spec.
SV_THRESHOLD_MIN = 0.40
SV_THRESHOLD_MAX = 0.90
SV_THRESHOLD_STEP = 0.01


class SpeakerVerifierDiscovery:
    """Publishes the two SV live-tunable HA entities + routes their command
    topics back into ServerState / EnrollmentsStore.

    Unlike WakeCaptureDiscovery there is no periodic publish loop — state
    only changes when HA writes a new value, so we publish state on
    startup + on every accepted command. Discovery configs are republished
    on every MQTT (re)connect via ``HABridge.attach_speaker_verifier_discovery``.
    """

    def __init__(
        self,
        *,
        ha_bridge: Any,
        state: Any,
        enrollments: Any,
        room: str,
    ) -> None:
        self.ha_bridge = ha_bridge
        self.state = state
        self.enrollments = enrollments
        self.room = room

    # ---- topics ----

    @property
    def threshold_state_topic(self) -> str:
        return f"{STATE_TOPIC_PREFIX}/{self.room}/tunable/sv_threshold/state"

    @property
    def threshold_set_topic(self) -> str:
        return f"{STATE_TOPIC_PREFIX}/{self.room}/tunable/sv_threshold/set"

    @property
    def audible_notify_state_topic(self) -> str:
        return f"{STATE_TOPIC_PREFIX}/{self.room}/tunable/sv_audible_notify/state"

    @property
    def audible_notify_set_topic(self) -> str:
        return f"{STATE_TOPIC_PREFIX}/{self.room}/tunable/sv_audible_notify/set"

    @property
    def availability_topic(self) -> str:
        return f"{STATE_TOPIC_PREFIX}/{self.room}/heartbeat"

    def _device_block(self) -> Dict[str, Any]:
        return {
            "identifiers": [f"calisto_{self.room}"],
            "name": f"Calisto {self.room.title()}",
            "manufacturer": "Plantronics",
            "model": "Calisto P7200",
        }

    # ---- publishes ----

    def publish_configs(self) -> int:
        published = 0

        threshold_uid = f"calisto_{self.room}_sv_threshold"
        threshold_cfg_topic = f"{DISCOVERY_PREFIX}/number/{threshold_uid}/config"
        threshold_payload = {
            "name": f"{self.room.title()} Speaker Verify Threshold",
            "unique_id": threshold_uid,
            "object_id": threshold_uid,
            "state_topic": self.threshold_state_topic,
            "command_topic": self.threshold_set_topic,
            "value_template": "{{ value_json.value }}",
            "command_template": "{\"value\": {{ value }} }",
            "min": SV_THRESHOLD_MIN,
            "max": SV_THRESHOLD_MAX,
            "step": SV_THRESHOLD_STEP,
            "mode": "slider",
            "icon": "mdi:tune",
            "availability_topic": self.availability_topic,
            "availability_template": "{{ 'online' if value_json else 'offline' }}",
            "device": self._device_block(),
        }
        if self.ha_bridge.publish(threshold_cfg_topic, json.dumps(threshold_payload), retain=True):
            published += 1

        notify_uid = f"calisto_{self.room}_sv_audible_notify"
        notify_cfg_topic = f"{DISCOVERY_PREFIX}/switch/{notify_uid}/config"
        notify_payload = {
            "name": f"{self.room.title()} Speaker Verify Audible Notify",
            "unique_id": notify_uid,
            "object_id": notify_uid,
            "state_topic": self.audible_notify_state_topic,
            "command_topic": self.audible_notify_set_topic,
            "value_template": "{{ value_json.value }}",
            "payload_on": "{\"value\": true}",
            "payload_off": "{\"value\": false}",
            "state_on": True,
            "state_off": False,
            "icon": "mdi:bell-ring",
            "availability_topic": self.availability_topic,
            "availability_template": "{{ 'online' if value_json else 'offline' }}",
            "device": self._device_block(),
        }
        if self.ha_bridge.publish(notify_cfg_topic, json.dumps(notify_payload), retain=True):
            published += 1

        return published

    def publish_state(self) -> int:
        published = 0
        threshold = float(self.state.sv_threshold)
        if self.ha_bridge.publish(
            self.threshold_state_topic,
            json.dumps({"value": threshold}),
            retain=True,
        ):
            published += 1
        notify = bool(self.state.sv_audible_notify)
        if self.ha_bridge.publish(
            self.audible_notify_state_topic,
            json.dumps({"value": notify}),
            retain=True,
        ):
            published += 1
        return published

    def start(self) -> None:
        try:
            n_cfg = self.publish_configs()
            n_state = self.publish_state()
            _LOGGER.info(
                "SpeakerVerifierDiscovery published %d config(s) + %d state(s)",
                n_cfg, n_state,
            )
        except Exception:
            _LOGGER.exception("SpeakerVerifierDiscovery initial publish raised")

    # ---- command handlers (called by HABridge._on_message) ----

    def handle_threshold_command(self, raw_payload: bytes) -> bool:
        try:
            obj = json.loads(raw_payload.decode("utf-8")) if raw_payload else {}
        except (ValueError, UnicodeDecodeError):
            _LOGGER.warning("SV threshold/set: malformed payload %r", raw_payload[:80])
            return False
        if not isinstance(obj, dict) or "value" not in obj:
            _LOGGER.warning("SV threshold/set: payload missing 'value'")
            return False
        try:
            value = float(obj["value"])
        except (TypeError, ValueError):
            _LOGGER.warning("SV threshold/set: non-numeric value %r", obj.get("value"))
            return False
        clamped = max(SV_THRESHOLD_MIN, min(SV_THRESHOLD_MAX, value))
        self.state.sv_threshold = clamped
        try:
            self.enrollments.threshold = clamped
            self.enrollments.save()
        except Exception:
            _LOGGER.exception("SV threshold/set: enrollments persist raised; in-memory value kept")
        self.publish_state()
        _LOGGER.info("SV threshold updated to %.2f", clamped)
        return True

    def handle_audible_notify_command(self, raw_payload: bytes) -> bool:
        try:
            obj = json.loads(raw_payload.decode("utf-8")) if raw_payload else {}
        except (ValueError, UnicodeDecodeError):
            _LOGGER.warning("SV audible_notify/set: malformed payload %r", raw_payload[:80])
            return False
        if not isinstance(obj, dict) or "value" not in obj:
            _LOGGER.warning("SV audible_notify/set: payload missing 'value'")
            return False
        value = obj["value"]
        if isinstance(value, str):
            normalized = value.strip().lower() in ("true", "1", "on", "yes")
        else:
            normalized = bool(value)
        self.state.sv_audible_notify = normalized
        self.publish_state()
        _LOGGER.info("SV audible_notify updated to %s", normalized)
        return True
