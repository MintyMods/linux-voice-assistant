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
        # Stage E.1 absorbs the consumer side; the stop-gap mirror still
        # fires here for any HA automation that historically listened on
        # this topic. Once `calisto-led.service` is decommissioned (Stage
        # E.1 step 13) the mirror is harmless — nothing is listening.
        self.led_topic = f"calisto/{room}/led/set"
        # Stage E.1 — back-compat control topics consumed from HA / v0
        # automations. LedController is attached via `attach_led_controller`.
        self.led_set_room_topic = f"calisto/{room}/led/set"
        self.led_set_all_topic = "calisto/all/led/set"
        self.volume_set_topic = f"calisto/{room}/volume/set"
        # Stage E.2 G5 — fleet broadcast mirrors. Any LVA subscribes to its
        # own room topic AND the corresponding `calisto/all/*` topic so HA
        # "set all" commands fan out without per-device routing.
        self.volume_set_all_topic = "calisto/all/volume/set"
        self.ring_set_topic = f"calisto/{room}/ring/set"
        self.ring_set_all_topic = "calisto/all/ring/set"
        # M3 (Path B) — first-class MQTT mute recovery topic. Lets HA /
        # automations toggle the mute state directly instead of routing
        # through the back-compat `led/set` legacy verbs. Payload is
        # `on` | `off` (also accepts the legacy `mute` | `unmute`).
        self.mute_set_topic = f"calisto/{room}/mute/set"
        self.mute_set_all_topic = "calisto/all/mute/set"
        # Stage E.2 K.8/K.9 + K.10 — alarm + ad-hoc TTS announce.
        self.alarm_set_topic = f"calisto/{room}/alarm/set"
        self.alarm_set_all_topic = "calisto/all/alarm/set"
        self.alarm_stop_topic = f"calisto/{room}/alarm/stop"
        self.say_topic = f"calisto/{room}/say"
        self.say_all_topic = "calisto/all/say"
        # Retained state topics — Lovelace cards + v0 automations read these.
        self.led_state_topic = f"calisto/{room}/led/state"
        self.volume_state_topic = f"calisto/{room}/volume/state"
        self.mute_state_topic = f"calisto/{room}/mute/state"
        self.ring_state_topic = f"calisto/{room}/ring/state"
        self.availability_topic = f"calisto/{room}/availability"
        # Phone-button events (not retained — these are momentary).
        self.phone_short_topic = f"calisto/{room}/button/phone/short"
        self.phone_long_topic = f"calisto/{room}/button/phone/long"
        # client_factory(client_id, clean_session) -> client.  When None,
        # start() imports paho.mqtt.client and uses its real Client class.
        # Tests inject a factory returning FakeMqttClient.
        self._client_factory = client_factory
        self._client: Any = None
        self._connected = False
        self._led_controller: Any = None
        # Stage E.2 — alarm/chime/say surface. attach_audio_controllers wires
        # these post-construction so HABridge can be tested in isolation.
        self._alarm: Any = None
        self._chime: Any = None
        self._tts_player: Any = None
        self._tts_output: Any = None
        self._arbiter: Any = None
        self._loop: Any = None

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
        client.on_message = self._on_message
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
            # Publish online availability for any v0 automation that
            # tracks `calisto/<room>/availability`. LWT on the K.1 state
            # topic still fires on ungraceful disconnect; the availability
            # topic is the v0-era equivalent for the LED control plane.
            try:
                _c.publish(self.availability_topic, "online", qos=1, retain=True)
            except Exception:
                _LOGGER.exception("HABridge availability publish failed")
            # Stage E.1 — subscribe to back-compat control topics so HA
            # scripts / Lovelace cards keep working. Routing is via the
            # attached LedController.
            try:
                _c.subscribe(
                    [
                        (self.led_set_room_topic, 1),
                        (self.led_set_all_topic, 1),
                        (self.volume_set_topic, 1),
                        (self.volume_set_all_topic, 1),
                        (self.ring_set_topic, 1),
                        (self.ring_set_all_topic, 1),
                        (self.mute_set_topic, 1),
                        (self.mute_set_all_topic, 1),
                        (self.alarm_set_topic, 1),
                        (self.alarm_set_all_topic, 1),
                        (self.alarm_stop_topic, 1),
                        (self.say_topic, 1),
                        (self.say_all_topic, 1),
                    ]
                )
            except Exception:
                _LOGGER.exception("HABridge subscribe failed")
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
        #   wake|processing|complete|off|error  on calisto/<room>/led/set.
        # Stage E.1 absorbs LED control in-process; once a LedController
        # is attached we MUST NOT publish here too, because we also
        # subscribe to the same topic — every K.1 transition would echo
        # back through `apply_legacy` and re-paint the palette out of
        # order on rapid sequences (visible flicker).
        if self._led_controller is None:
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


# ---------------------------------------------------------------------------
# Stage E.1 — back-compat MQTT surface
#
# Method patches below extend HABridge with the publish + subscribe handlers
# the absorbed LED + mute + volume system needs to mirror v0 behaviour.
# Kept after the class to keep the original Stage B class shape readable.
# ---------------------------------------------------------------------------


def _attach_led_controller(self: HABridge, controller: Any) -> None:
    """Wire a LedController to receive routed MQTT control messages.

    Idempotent — calling twice replaces the previous controller. Useful
    for tests; production sets it once at startup.
    """
    self._led_controller = controller


def _on_message(self: HABridge, _client: Any, _userdata: Any, msg: Any) -> None:
    """Route a subscribed control topic to the attached LedController or
    Stage E.2 audio controllers.

    Topics (LED, lowercased payload):
        calisto/<room>/led/set      → controller.apply_legacy(payload)
        calisto/all/led/set         → controller.apply_legacy(payload)
        calisto/<room>/volume/set   → controller.bar.apply(payload)
        calisto/<room>/ring/set     → controller.ring.start() / .stop()
        calisto/<room>/mute/set     → controller.set_private(payload, source="mqtt")

    Topics (alarm/say — payload kept as raw bytes for JSON parsing):
        calisto/<room>/alarm/set    → alarm.set_alarm(raw_bytes) (K.8)
        calisto/all/alarm/set       → alarm.set_alarm(raw_bytes)
        calisto/<room>/alarm/stop   → alarm.stop_alarm()
        calisto/<room>/say          → tts/chime per scope (K.10)
        calisto/all/say             → tts/chime per scope
    """
    try:
        topic = msg.topic
        raw_payload = bytes(msg.payload) if msg.payload is not None else b""
    except Exception:
        _LOGGER.exception("HABridge: malformed MQTT message")
        return

    # Alarm + say handlers want the raw bytes (JSON); the LED back-compat
    # handlers want a lowercased string. Dispatch by topic first.
    if topic in (self.alarm_set_topic, self.alarm_set_all_topic):
        if self._alarm is not None:
            try:
                self._alarm.set_alarm(raw_payload)
            except Exception:
                _LOGGER.exception("HABridge: alarm/set routing raised")
        return
    if topic == self.alarm_stop_topic:
        if self._alarm is not None:
            try:
                self._alarm.stop_alarm()
            except Exception:
                _LOGGER.exception("HABridge: alarm/stop routing raised")
        return
    if topic in (self.say_topic, self.say_all_topic):
        self._route_say(raw_payload)
        return

    # LED back-compat surface — lower-case the payload for legacy verbs.
    controller = self._led_controller
    if controller is None:
        return
    try:
        payload = raw_payload.decode("utf-8", errors="replace").strip().lower()
    except Exception:
        _LOGGER.exception("HABridge: malformed LED payload")
        return
    _LOGGER.debug("HABridge recv %s = %r", topic, payload)
    try:
        if topic in (self.led_set_room_topic, self.led_set_all_topic):
            controller.apply_legacy(payload)
        elif topic in (self.volume_set_topic, self.volume_set_all_topic):
            controller.bar.apply(payload)
        elif topic in (self.ring_set_topic, self.ring_set_all_topic):
            if payload in ("on", "start", "1", "true"):
                controller.ring.start()
                self.publish_ring_state(True)
            else:
                controller.ring.stop()
                self.publish_ring_state(False)
        elif topic in (self.mute_set_topic, self.mute_set_all_topic):
            # M3 — first-class mute control. Accepts `on`/`off` (preferred)
            # plus the legacy `mute`/`unmute` verbs for back-compat with
            # automations that historically wrote those values.
            if payload in ("on", "mute", "1", "true"):
                controller.set_private(True, source="mqtt")
            elif payload in ("off", "unmute", "0", "false"):
                controller.set_private(False, source="mqtt")
            else:
                _LOGGER.warning(
                    "HABridge: mute/set unknown payload %r — expected on|off",
                    payload,
                )
    except Exception:
        _LOGGER.exception("HABridge: routing %s = %r raised", topic, payload)


def _attach_audio_controllers(
    self: HABridge,
    *,
    alarm: Any,
    chime: Any,
    tts_player: Any,
    tts_output: Any,
    arbiter: Any,
    loop: Any,
) -> None:
    """Wire Stage E.2 audio controllers post-construction.

    `alarm` may be None for tests that don't exercise alarms — say/chime
    still work without it. Same for `tts_output`/`tts_player` when only the
    chime tier is being tested.
    """
    self._alarm = alarm
    self._chime = chime
    self._tts_player = tts_player
    self._tts_output = tts_output
    self._arbiter = arbiter
    self._loop = loop
    # Now that the bridge has a publisher reference, retro-attach so the
    # alarm controller's first publish_state goes out cleanly.
    if alarm is not None and hasattr(alarm, "attach_ha_bridge"):
        try:
            alarm.attach_ha_bridge(self)
        except Exception:
            _LOGGER.exception("HABridge.attach_audio_controllers: alarm.attach_ha_bridge raised")


def _route_say(self: HABridge, raw_payload: bytes) -> None:
    """Parse a K.10 say payload and dispatch through tts_output or chime.

    {"text": "...", "scope": "tts"|"chime", "interrupt": bool, "voice": "..."}

    On unknown scope or empty text, log + drop. interrupt is not honoured
    here — barge-in is the existing cancel chain's responsibility.
    """
    import json

    try:
        obj = json.loads(raw_payload.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        _LOGGER.warning("HABridge say: malformed payload %r", raw_payload[:80])
        return
    if not isinstance(obj, dict):
        _LOGGER.warning("HABridge say: payload must be JSON object")
        return
    text = str(obj.get("text") or "").strip()
    if not text:
        _LOGGER.debug("HABridge say: empty text; dropping")
        return
    scope = str(obj.get("scope") or "tts").lower()

    arbiter = self._arbiter
    if scope == "chime":
        chime = self._chime
        if chime is None:
            _LOGGER.warning("HABridge say(chime): no ChimeController wired")
            return
        # For chime scope, `text` is a chime sound slug (e.g. "Ping.ogg")
        # — not synthesized speech. K.10 treats the field uniformly.
        chime.play(text)
        return

    if arbiter is not None and not arbiter.allow_say_tts():
        _LOGGER.info("HABridge say(tts): suppressed by arbiter (alarm ringing)")
        return

    tts_output = self._tts_output
    tts_player = self._tts_player
    loop = self._loop
    if tts_output is None or tts_player is None or loop is None:
        _LOGGER.warning(
            "HABridge say(tts): missing components (output=%s player=%s loop=%s)",
            tts_output is not None,
            tts_player is not None,
            loop is not None,
        )
        return

    voice = obj.get("voice")
    voice_override = str(voice) if isinstance(voice, str) and voice else None
    if voice_override is not None and hasattr(tts_output, "voice"):
        # tts_output.speak() uses self.voice; override transiently. The
        # client is recreated per-call in tts_output.py so this is safe.
        try:
            tts_output.voice = voice_override
        except Exception:
            pass

    async def _do_speak() -> None:
        try:
            await tts_output.speak(tts_player, text=text)
        except Exception:
            _LOGGER.exception("HABridge say(tts): tts_output.speak raised")

    try:
        loop.call_soon_threadsafe(lambda: loop.create_task(_do_speak()))
    except RuntimeError:
        _LOGGER.warning("HABridge say(tts): loop not running")


def _publish_volume_state(self: HABridge, pct: int) -> None:
    """Publish retained `calisto/<room>/volume/state` = `<pct>`."""
    self.publish(self.volume_state_topic, str(int(pct)), qos=1, retain=True)


def _publish_mute_state(self: HABridge, muted: bool) -> None:
    """Publish retained `calisto/<room>/mute/state` = `on` | `off`."""
    self.publish(self.mute_state_topic, "on" if muted else "off", qos=1, retain=True)


def _publish_ring_state(self: HABridge, ringing: bool) -> None:
    """Publish retained `calisto/<room>/ring/state` = `on` | `off`."""
    self.publish(self.ring_state_topic, "on" if ringing else "off", qos=1, retain=True)


def _publish_phone_button(self: HABridge, action: str) -> None:
    """Publish a (non-retained) phone-button event for v0 HA automations
    that gated on `calisto/<room>/button/phone/{short,long}`."""
    if action == "long":
        topic = self.phone_long_topic
    else:
        topic = self.phone_short_topic
    self.publish(topic, "press", qos=1, retain=False)


HABridge.attach_led_controller = _attach_led_controller  # type: ignore[attr-defined]
HABridge.attach_audio_controllers = _attach_audio_controllers  # type: ignore[attr-defined]
HABridge._on_message = _on_message  # type: ignore[attr-defined]
HABridge._route_say = _route_say  # type: ignore[attr-defined]
HABridge.publish_volume_state = _publish_volume_state  # type: ignore[attr-defined]
HABridge.publish_mute_state = _publish_mute_state  # type: ignore[attr-defined]
HABridge.publish_ring_state = _publish_ring_state  # type: ignore[attr-defined]
HABridge.publish_phone_button = _publish_phone_button  # type: ignore[attr-defined]
