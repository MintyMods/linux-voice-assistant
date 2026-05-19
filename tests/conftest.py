"""Top-level test bootstrap.

Mirrors tests/integration/conftest.py for stubs that must load before any
test imports, and adds Stage B fixtures (fake MQTT broker, factory for a
minimal ServerState).
"""

from __future__ import annotations

import json
import sys
import types
from queue import Queue
from typing import Any, Dict, List, Tuple
from unittest.mock import MagicMock

import pytest


def _install_stub(name: str) -> None:
    if name in sys.modules:
        return
    sys.modules[name] = types.ModuleType(name)


# libmpv + netifaces are pulled in transitively by linux_voice_assistant
# imports (entity, util). The tests in this tree don't exercise the real
# backends, so stubbing at sys.modules lets the suite run on dev machines.
_install_stub("mpv")
_install_stub("netifaces")


class FakeMqttClient:
    """In-memory paho-style client. Records publishes + will_set.

    Mimics enough of paho.mqtt.client.Client for HABridge:
      - will_set(topic, payload, qos, retain)
      - username_pw_set(user, pass)
      - on_connect / on_disconnect attribute slots
      - reconnect_delay_set(...)
      - connect_async(host, port, keepalive)
      - loop_start() / loop_stop() / disconnect()
      - publish(topic, payload, qos, retain)

    HABridge calls these in this order; the fake records the calls for
    assertion.
    """

    def __init__(self) -> None:
        self.will: Tuple[str, str, int, bool] | None = None
        self.username: str | None = None
        self.password: str | None = None
        self.on_connect = None
        self.on_disconnect = None
        self.reconnect_delay: Tuple[int, int] | None = None
        self.connect_args: Tuple[str, int, int] | None = None
        self.loop_started = False
        self.loop_stopped = False
        self.disconnected = False
        self.publishes: List[Tuple[str, str, int, bool]] = []
        # Stage E.1 — HABridge._on_connect subscribes to back-compat
        # control topics. Without a `subscribe` method here, the
        # AttributeError was swallowed silently in HABridge's try/except,
        # masking regressions to the topic list or QoS.
        self.subscriptions: List[Tuple[str, int]] = []

    def will_set(self, topic: str, payload: str, qos: int = 0, retain: bool = False) -> None:
        if self.connect_args is not None:
            raise AssertionError("will_set() must be called BEFORE connect_async() (K.1.4 / H5)")
        self.will = (topic, payload, qos, retain)

    def username_pw_set(self, user: str, password: str | None = None) -> None:
        self.username = user
        self.password = password

    def reconnect_delay_set(self, min_delay: int, max_delay: int) -> None:
        self.reconnect_delay = (min_delay, max_delay)

    def connect_async(self, host: str, port: int, keepalive: int = 60) -> None:
        self.connect_args = (host, port, keepalive)

    def loop_start(self) -> None:
        self.loop_started = True
        # Simulate immediate connect for test-ergonomics.
        if self.on_connect is not None:
            self.on_connect(self, None, None, 0)

    def loop_stop(self) -> None:
        self.loop_stopped = True

    def disconnect(self) -> None:
        self.disconnected = True
        if self.on_disconnect is not None:
            self.on_disconnect(self, None)

    def subscribe(self, topics: Any) -> None:
        """Record subscriptions. Accepts paho's list-of-(topic, qos)
        form or a single (topic, qos) tuple."""
        if isinstance(topics, list):
            for t in topics:
                self.subscriptions.append(tuple(t))  # type: ignore[arg-type]
        else:
            self.subscriptions.append(tuple(topics))  # type: ignore[arg-type]

    def publish(self, topic: str, payload: str, qos: int = 0, retain: bool = False):
        self.publishes.append((topic, payload, qos, retain))
        return None

    # -- helpers for assertions --

    def state_publishes(self, topic: str) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for t, p, _, _ in self.publishes:
            if t == topic:
                try:
                    out.append(json.loads(p))
                except json.JSONDecodeError:
                    out.append({"_raw": p})
        return out


class FakePahoModule:
    """Drop-in for `import paho.mqtt.client as mqtt`.

    HABridge does `import paho.mqtt.client as mqtt; mqtt.Client(...)` inside
    its `start()`, so the test patches `sys.modules["paho.mqtt.client"]` with
    this object before instantiating HABridge.
    """

    class CallbackAPIVersion:
        VERSION2 = "VERSION2"

    @staticmethod
    def Client(api_version, client_id: str, clean_session: bool = True) -> FakeMqttClient:  # noqa: N802
        return FakeMqttClient()


@pytest.fixture
def fake_paho():
    """Yields a (created_list, factory) tuple.

    `factory` is the `client_factory` callable to pass to HABridge; each
    invocation produces a new FakeMqttClient and appends it to
    `created_list` so tests can assert on the recorded calls. This sidesteps
    the import-system gymnastics that monkey-patching `paho.mqtt.client`
    would require.

    Tests typically destructure: `created, factory = fake_paho`.
    """
    created: List[FakeMqttClient] = []

    def factory(client_id: str = "", clean_session: bool = True) -> FakeMqttClient:
        c = FakeMqttClient()
        created.append(c)
        return c

    return created, factory


def make_server_state(**overrides: Any) -> Any:
    """Build a minimal ServerState shell suitable for DeviceSession tests.

    Constructs a real ServerState (not a MagicMock) so dataclass attribute
    defaults apply. Heavy fields that the tests never touch (wake models,
    mpv players, entity list) are stubbed with MagicMock.
    """
    from linux_voice_assistant.models import Preferences, ServerState

    defaults: Dict[str, Any] = {
        "name": "lva-test",
        "friendly_name": "Test LVA",
        "mac_address": "aa:bb:cc:dd:ee:ff",
        "ip_address": "127.0.0.1",
        "network_interface": "lo",
        "version": "test",
        "esphome_version": "test",
        "audio_queue": Queue(),
        "entities": [],
        "available_wake_words": {},
        "wake_words": {},
        "active_wake_words": set(),
        "stop_word": MagicMock(id="stop_word"),
        "music_player": MagicMock(),
        "tts_player": MagicMock(),
        "chime_player": MagicMock(),
        "alarm_player": MagicMock(),
        "wakeup_sound": "",
        "timer_finished_sound": "",
        "processing_sound": "",
        "mute_sound": "",
        "unmute_sound": "",
        "preferences": Preferences(),
        "preferences_path": "preferences.json",
        "download_dir": ".",
    }
    defaults.update(overrides)
    return ServerState(**defaults)  # type: ignore[arg-type]
