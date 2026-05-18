"""Stage B — HABridge: K.1 session/state publisher (publish-on-transition).

Owns its own paho-mqtt client, separate from the cancel subscriber's. That
keeps two concerns decoupled (cancel subscriber existed pre-Stage-A, HABridge
is new) and lets the LWT be armed via will_set() before connect_async() per
K.1.4 / H5. The two clients will be consolidated into mqtt_router in Stage F.

Stage B publishes on transitions only. The 30s slow re-assert timer, drift
detection vs. HA's reported state, and startup state assertion all land in
Stage F.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Optional, Union

from .session import State

_LOGGER = logging.getLogger(__name__)


class HABridge:
    """K.1 session/state publisher with LWT armed before connect.

    Construction does NOT connect to MQTT. Call `start()` to (a) build the
    paho client, (b) arm LWT, (c) connect_async + loop_start. `publish_state`
    is callable any time after start; if MQTT isn't yet connected, paho
    queues the publish locally.
    """

    def __init__(
        self,
        *,
        room: str,
        host: str,
        port: int = 1883,
        username: Optional[str] = None,
        password: Optional[str] = None,
        client_id: Optional[str] = None,
        client_factory: "Optional[Any]" = None,
    ) -> None:
        self.room = room
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.client_id = client_id or f"lva-ha-bridge-{room}"
        self.state_topic = f"calisto/{room}/session/state"
        # B3 stop-gap: also drive the legacy `calisto/<room>/led/set` control
        # plane so the existing calisto-led service lights the red phone LED.
        # Stage E absorbs LED control into LVA proper.
        self.led_topic = f"calisto/{room}/led/set"
        # client_factory(client_id, clean_session) -> client.  When None,
        # start() imports paho.mqtt.client and uses its real Client class.
        # Tests inject a factory returning FakeMqttClient.
        self._client_factory = client_factory
        self._client: Any = None
        self._connected = False

    def _lwt_payload(self) -> str:
        return json.dumps(
            {
                "state": State.OFFLINE.value,
                "ts": None,
                "reason": "lwt_triggered",
                "generation": -1,
                "session_id": None,
                "cancel_reason": None,
                "since_ms": 0,
            }
        )

    def start(self) -> None:
        if self._client_factory is not None:
            client = self._client_factory(client_id=self.client_id, clean_session=True)
        else:
            try:
                import paho.mqtt.client as mqtt  # type: ignore
            except ImportError:
                _LOGGER.error("paho-mqtt not installed; HABridge K.1 publisher disabled")
                return
            client = mqtt.Client(
                mqtt.CallbackAPIVersion.VERSION2,  # type: ignore[attr-defined]
                client_id=self.client_id,
                clean_session=True,
            )
        if self.username:
            client.username_pw_set(self.username, self.password)
        # LWT must be set BEFORE connect_async (K.1.4, H5 ordering).
        client.will_set(self.state_topic, self._lwt_payload(), qos=1, retain=True)
        client.on_connect = self._on_connect
        client.on_disconnect = self._on_disconnect
        client.reconnect_delay_set(min_delay=1, max_delay=30)
        try:
            client.connect_async(self.host, self.port, keepalive=60)
            client.loop_start()
        except Exception:
            _LOGGER.exception("HABridge connect_async failed; K.1 publish will be unavailable")
            return
        self._client = client

    def stop(self) -> None:
        client = self._client
        if client is None:
            return
        try:
            client.loop_stop()
            client.disconnect()
        except Exception:
            _LOGGER.exception("HABridge stop raised")
        self._client = None

    def _on_connect(self, _c, _ud, _flags, rc, _props=None) -> None:
        if rc == 0:
            self._connected = True
            _LOGGER.info(
                "HABridge connected to %s:%d, publishing K.1 on %s",
                self.host,
                self.port,
                self.state_topic,
            )
        else:
            _LOGGER.error("HABridge connect failed rc=%s", rc)

    def _on_disconnect(self, _c, _ud, *_args, **_kwargs) -> None:
        self._connected = False
        _LOGGER.warning("HABridge disconnected from MQTT broker; LWT may have fired")

    def publish(self, topic: str, payload: str, *, qos: int = 1, retain: bool = True) -> bool:
        """Publish an arbitrary MQTT message via the HABridge's connected
        client. Returns True when the publish was queued (paho handles
        offline buffering); False when there is no client yet.

        Added for Stage C Discovery — keeps Stage F's mqtt_router consolidation
        from being blocked on multiple paho clients in B3.
        """
        client = self._client
        if client is None:
            return False
        try:
            client.publish(topic, payload, qos=qos, retain=retain)
            return True
        except Exception:
            _LOGGER.exception("HABridge.publish failed for topic %s", topic)
            return False

    def publish_state(
        self,
        *,
        state: Union[State, str],
        generation: int,
        session_id: Optional[str],
        reason: str = "transition",
        cancel_reason: Optional[str] = None,
        since_ms: int = 0,
    ) -> None:
        client = self._client
        if client is None:
            return
        state_value = state.value if isinstance(state, State) else str(state)
        payload = {
            "state": state_value,
            "generation": generation,
            "session_id": session_id,
            "ts": datetime.now().astimezone().isoformat(timespec="milliseconds"),
            "reason": reason,
            "cancel_reason": cancel_reason,
            "since_ms": since_ms,
        }
        try:
            client.publish(self.state_topic, json.dumps(payload), qos=1, retain=True)
        except Exception:
            _LOGGER.exception("HABridge.publish_state failed for topic %s", self.state_topic)

        # B3 stop-gap LED mirror — the v0 calisto-led service consumes:
        #   wake|processing|complete|off|error  on calisto/<room>/led/set
        # Stage E will absorb LED control into LVA proper (E1..E6).
        led_cmd = _STATE_TO_LED.get(state_value)
        if led_cmd is not None:
            try:
                client.publish(self.led_topic, led_cmd, qos=1, retain=False)
            except Exception:
                _LOGGER.exception("HABridge LED mirror failed for topic %s", self.led_topic)


# Map K.1 states to legacy calisto-led `led/set` commands.  See
# calisto-led/led_service.py:LedDriver.apply for the verbs.
_STATE_TO_LED = {
    State.WAKING.value: "wake",
    State.LISTENING.value: "wake",
    State.THINKING.value: "processing",
    State.SPEAKING.value: "complete",
    State.FOLLOWUP.value: "wake",
    State.IDLE.value: "off",
}
