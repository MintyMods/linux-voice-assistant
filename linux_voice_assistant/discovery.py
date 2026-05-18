"""Stage C — minimal MQTT Discovery surface for wake-capture stats sensors.

Publishes HA Discovery config payloads (retained) for the metrics that the
triage dashboard binds to:

  - sensor.calisto_<room>_wake_captures_24h          (total wake fires / day)
  - sensor.calisto_<room>_wake_positives_24h         (label=positive)
  - sensor.calisto_<room>_wake_negatives_24h         (label=negative)
  - sensor.calisto_<room>_wake_ambiguous_24h         (label=ambiguous)
  - sensor.calisto_<room>_wake_gate2_rejects_24h     (Stage D Gate-2 rejects)
  - sensor.calisto_<room>_wake_pending_triage        (label is null OR ambiguous)
  - sensor.calisto_<room>_wake_capture_http_url      (URL prefix HA's REST hits)

The full N.2 Discovery manifest (every voice entity per room) lands in
Stage G; this module is the earliest Discovery surface and will be expanded
in-place rather than rewritten. State updates are pushed every 60s while the
LVA process is up.

All publishes go through ``HABridge.publish`` so we share a single MQTT
client until ``mqtt_router`` consolidates in Stage F.
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
