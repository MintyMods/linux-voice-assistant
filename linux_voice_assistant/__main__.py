#!/usr/bin/env python3
import argparse
import asyncio
import errno
import json
import logging
import sys
import threading
import time
from pathlib import Path
from queue import Queue
from typing import List, Optional, Union

import numpy as np
import soundcard as sc
from aioesphomeapi.api_pb2 import NumberStateResponse  # type: ignore  # pylint: disable=no-name-in-module
from getmac import get_mac_address  # type: ignore
from pymicro_wakeword import MicroWakeWord, MicroWakeWordFeatures
from pyopen_wakeword import OpenWakeWord, OpenWakeWordFeatures

from .alarm import AlarmController
from .asr_client import ASRClient
from .audible_notify import AudibleNotifyArbiter, ChimeController
from .audio_control import AudioControl
from .bridge_client import BridgeClient
from .cancel import CancelCoordinator
from .enrollment import EnrollmentHandler
from .ha_bridge import HABridge
from .heartbeat import HeartbeatPublisher
from .speaker_verifier import EnrollmentsStore, SpeakerVerifier
from .led import LedController
from .mic_capture import MicCapture, SpeechBuffer
from .models import Preferences, ServerState
from .mpv_player import MpvMediaPlayer
from .satellite import VoiceSatelliteProtocol
from .session import DeviceSession
from .tts_output import TTSOutput
from .discovery import EntitySurface, SpeakerVerifierDiscovery, StatePublishCounter, WakeCaptureDiscovery
from .wake_capture import WakeCapture
from .wake_capture_http import WakeCaptureHTTP
from .util import (
    get_default_interface,
    get_default_ipv4,
    get_esphome_version,
    get_version,
)
from .wake_word import find_available_wake_words, load_stop_model, load_wake_models
from .webrtc import WebRTCProcessor
from .zeroconf import HomeAssistantZeroconf

_LOGGER = logging.getLogger(__name__)
_MODULE_DIR = Path(__file__).parent
_REPO_DIR = _MODULE_DIR.parent
_WAKEWORDS_DIR = _REPO_DIR / "wakewords"
_SOUNDS_DIR = _REPO_DIR / "sounds"


# -----------------------------------------------------------------------------


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name",
        help="Real name for the device",
    )
    parser.add_argument(
        "--audio-input-device",
        help="Name for the audio input device (see --list-input-devices)",
    )
    parser.add_argument(
        "--list-input-devices",
        action="store_true",
        help="List audio input devices and exit",
    )
    parser.add_argument(
        "--audio-input-block-size",
        type=int,
        default=1024,
        # todo
    )
    parser.add_argument(
        "--audio-output-device",
        help="Name for the audio output device (see --list-output-devices)",
    )
    parser.add_argument(
        "--list-output-devices",
        action="store_true",
        help="List audio output devices and exit",
    )
    parser.add_argument("--mic-auto-gain", type=int, default=0, choices=list(range(32)))
    parser.add_argument("--mic-noise-suppression", type=int, default=0, choices=(0, 1, 2, 3, 4))
    parser.add_argument(
        "--wake-word-dir",
        default=[_WAKEWORDS_DIR],
        action="append",
        help="Directory with wake word models (.tflite) and configuration (.json)",
    )
    parser.add_argument(
        "--wake-model",
        default="okay_nabu",
        help="File name of the first active wake model",
    )
    parser.add_argument(
        "--stop-model",
        default="stop",
        help="File name of the stop model",
    )
    parser.add_argument(
        "--download-dir",
        default=_REPO_DIR / "local",
        help="Directory to download custom wake word models to",
    )
    parser.add_argument(
        "--refractory-seconds",
        default=2.0,
        type=float,
        help="Seconds before wake word can be activated again",
    )
    parser.add_argument(
        "--wakeup-sound",
        default=str(_SOUNDS_DIR / "wake_word_triggered.flac"),
        help="Directory and file name for wake sound (when you say the wake word)",
    )
    parser.add_argument(
        "--timer-finished-sound",
        default=str(_SOUNDS_DIR / "timer_finished.flac"),
        help="Directory and file name for timer finished sound",
    )
    parser.add_argument(
        "--processing-sound",
        default=str(_SOUNDS_DIR / "processing.wav"),
        help="Short sound to play while assistant is processing (thinking)",
    )
    parser.add_argument(
        "--mute-sound",
        default=str(_SOUNDS_DIR / "mute_switch_on.flac"),
        help="Sound to play when muting the assistant",
    )
    parser.add_argument(
        "--unmute-sound",
        default=str(_SOUNDS_DIR / "mute_switch_off.flac"),
        help="Sound to play when unmuting the assistant",
    )
    parser.add_argument(
        "--preferences-file",
        default=_REPO_DIR / "preferences.json",
        help="Directory and file name for the file where the preferences are stored in JSON format",
    )
    parser.add_argument(
        "--host",
        help="Optional host IP address to bind to (default: Autodetected by network interface)",  # 0.0.0.0 is IPv4, None is all interfaces
    )
    parser.add_argument(
        "--network-interface",
        help="Network interface the application will be listening on (default: will be automatically detected by gateway)",
    )
    # Note that default port is also set in docker-entrypoint.sh
    parser.add_argument(
        "--port",
        type=int,
        default=6053,
        help="Port the application is listenening on (default: 6053)",
    )
    parser.add_argument(
        "--enable-thinking-sound",
        action="store_true",
        help="Enable thinking finish sound, when the assistant is done thinking and needed more time to process",
    )
    parser.add_argument(
        "--timer-max-ring-seconds",
        type=float,
        default=900.0,  # 15 minutes
        help="Seconds before a ringing timer auto-stops (default: 900)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Add this to enable debug logging",
    )
    parser.add_argument(
        "--output-only",
        action="store_true",
        help="Enable output only mode",
    )
    args = parser.parse_args()

    if args.list_input_devices:
        print("Audio Input devices:")
        print("=" * 13)
        for idx, mic in enumerate(sc.all_microphones()):
            print(f"[{idx}]", mic.name)
        return

    if args.list_output_devices:
        from mpv import MPV

        player = MPV()
        print("Audio output devices:")
        print("=" * 14)

        for speaker in player.audio_device_list:  # type: ignore
            print(speaker["name"] + ":", speaker["description"])
        return

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    _LOGGER.debug(args)

    # Resolve network interface for mac-address detection
    if not args.network_interface:
        print("No network interface specified, try to detect default interface")
        network_interface = get_default_interface()
        print(f"Default interface detected: {network_interface}")
    else:
        print("Network interface specified")
        network_interface = args.network_interface
        print(f"Using network interface: {network_interface}")

    # Resolve ip_address where the application will be listening
    if not args.host:
        print("No host (ip-address) specified, try to detect IP-Address")
        host_ip_address = get_default_ipv4(network_interface)
        print(f"IP-Address detected: {host_ip_address}")
    else:
        print("Host specified")
        print(f"Using host: {args.host}")
        host_ip_address = args.host

    # Resolve mac
    if not (mac_address := get_mac_address(interface=network_interface)):
        print("No Mac address was found, app stopped.")
        sys.exit(1)
    mac_address_clean = mac_address.replace(":", "").lower()

    # Resolve name
    if not args.name:
        print("No friendly name specified, try to autogenerate name")
        friendly_name = f"LVA - {mac_address_clean}"
        print(f"Friendly name autogenerated: {friendly_name}")
    else:
        print("Friendly name specified")
        print(f"Using friendly name: {args.name}")
        friendly_name = args.name

    device_name = f"lva-{mac_address_clean}"

    print(f"Device name: {device_name}")

    # Resolve version
    version = get_version()
    print(f"Version: {version}")

    # Resolve esphome version
    esphome_version = get_esphome_version()
    print(f"ESPHome api version: {esphome_version}")

    # Resolve download dir
    args.download_dir = Path(args.download_dir)
    args.download_dir.mkdir(parents=True, exist_ok=True)

    # Resolve microphone
    if args.audio_input_device is not None:
        try:
            args.audio_input_device = int(args.audio_input_device)
        except ValueError:
            pass

        mic = sc.get_microphone(args.audio_input_device)
    else:
        mic = sc.default_microphone()

    # Load available wake words
    wake_word_dirs = [Path(ww_dir) for ww_dir in args.wake_word_dir]
    wake_word_dirs.append(args.download_dir / "external_wake_words")
    available_wake_words = find_available_wake_words(wake_word_dirs, args.stop_model)

    # Load preferences
    preferences_path = Path(args.preferences_file)
    if preferences_path.exists():
        _LOGGER.debug("Loading preferences: %s", preferences_path)
        with open(preferences_path, "r", encoding="utf-8") as preferences_file:
            preferences_dict = json.load(preferences_file)
            preferences = Preferences(**preferences_dict)
    else:
        preferences = Preferences()

    # Load volume from preferences on startup, and ensure it's between 0.0 and 1.0
    initial_volume = preferences.volume if preferences.volume is not None else 1.0
    initial_volume = max(0.0, min(1.0, float(initial_volume)))
    preferences.volume = initial_volume

    if args.enable_thinking_sound:
        preferences.thinking_sound = 1

    if args.mic_auto_gain or args.mic_noise_suppression:
        try:
            import webrtc_noise_gain  # type: ignore[import-untyped] # noqa: F401
        except ImportError:
            _LOGGER.exception("Extras for webrtc are not installed")
            sys.exit(1)

    if args.mic_auto_gain > 0:
        preferences.mic_auto_gain = args.mic_auto_gain

    if args.mic_noise_suppression > 0:
        preferences.mic_noise_suppression = args.mic_noise_suppression

    # Load wake/stop models
    wake_models, active_wake_words, fallback_used = load_wake_models(available_wake_words, [word for word in preferences.active_wake_words if word is not None], args.wake_model)

    # TODO: allow openWakeWord for "stop"
    stop_model = load_stop_model(wake_word_dirs, args.stop_model)
    assert stop_model is not None

    state = ServerState(
        name=device_name,
        friendly_name=friendly_name,
        network_interface=network_interface,
        mac_address=mac_address,
        ip_address=host_ip_address,
        version=version,
        esphome_version=esphome_version,
        audio_queue=Queue(),
        entities=[],
        available_wake_words=available_wake_words,
        wake_words=wake_models,
        active_wake_words=active_wake_words,
        stop_word=stop_model,
        music_player=MpvMediaPlayer(device=args.audio_output_device, role="media"),
        tts_player=MpvMediaPlayer(device=args.audio_output_device, role="tts"),
        chime_player=MpvMediaPlayer(device=args.audio_output_device, role="chime"),
        alarm_player=MpvMediaPlayer(device=args.audio_output_device, role="alarm"),
        wakeup_sound=args.wakeup_sound,
        timer_finished_sound=args.timer_finished_sound,
        processing_sound=args.processing_sound,
        mute_sound=args.mute_sound,
        unmute_sound=args.unmute_sound,
        preferences=preferences,
        preferences_path=preferences_path,
        refractory_seconds=args.refractory_seconds,
        output_only=args.output_only,
        download_dir=args.download_dir,
        volume=initial_volume,
        mic_volume=preferences.mic_volume,
        mic_auto_gain=preferences.mic_auto_gain,
        mic_noise_suppression=preferences.mic_noise_suppression,
        timer_max_ring_seconds=args.timer_max_ring_seconds,
    )

    if fallback_used:
        # Fallback to the default model was used, save as active wake words
        _LOGGER.debug("Fallback was used, save default wake words in Preferences.")
        state.preferences.active_wake_words = list(active_wake_words)
        state.active_wake_words = active_wake_words
        state.wake_words = wake_models
        state.save_preferences()
        state.wake_words_changed = True

    if args.enable_thinking_sound or args.mic_auto_gain or args.mic_noise_suppression:
        state.save_preferences()

    initial_volume_percent = int(round(initial_volume * 100))
    state.music_player.set_volume(initial_volume_percent)
    state.tts_player.set_volume(initial_volume_percent)
    state.chime_player.set_volume(initial_volume_percent)
    state.alarm_player.set_volume(100.0)

    # Stage E.2 G2 — push ducking envelope onto every channel. Only
    # music_player ever ducks in practice today, but threading the values
    # uniformly keeps the live-tunable wire-up trivial when Stage E.2-f
    # adds chime/tts pre-emption.
    for _ducker in (state.music_player, state.tts_player, state.chime_player, state.alarm_player):
        _ducker.configure_duck_envelope(
            floor_pct=state.duck_floor_pct,
            attack_ms=state.duck_attack_ms,
            release_ms=state.duck_release_ms,
        )

    loop = asyncio.get_running_loop()
    max_attempts = 15
    attempt = 1
    server = None

    # Validate VoiceSatelliteProtocol initialization BEFORE starting server
    # This catches errors like missing imports or broken initialization immediately
    # instead of failing silently only when first client connects
    _LOGGER.debug("Validating VoiceSatelliteProtocol initialization...")
    try:
        # Create test instance to run complete __init__ code path
        test_protocol = VoiceSatelliteProtocol(state)
        # Cleanup state reference
        test_protocol.state.satellite = None
        del test_protocol
        _LOGGER.debug("✅ VoiceSatelliteProtocol validation successful")
    except Exception:
        _LOGGER.critical("❌ FATAL ERROR in VoiceSatelliteProtocol initialization!", exc_info=True)
        _LOGGER.critical("Program will exit immediately - fix the error above first!")
        sys.exit(1)

    while attempt <= max_attempts:
        try:
            server = await loop.create_server(lambda: VoiceSatelliteProtocol(state), host=host_ip_address, port=args.port)
            break  # connect successful, exit the loop
        except OSError as err:
            message = err.strerror or str(err)
            if err.errno == errno.EADDRINUSE:
                message = "address already in use"
            if attempt < max_attempts:
                _LOGGER.warning("Attempt %d/%d failed to bind on address (%s, %s): %s. Retrying in 1 second...", attempt, max_attempts, host_ip_address, args.port, message)
                await asyncio.sleep(1)
                attempt += 1
            else:
                _LOGGER.exception("All %d attempts failed to bind on address (%s, %s): %s", max_attempts, host_ip_address, args.port, message)
                sys.exit(1)

    # Stage E.1 — AudioControl must be on state BEFORE process_audio_thread
    # starts (the thread captures `state.audio_control` once at entry). The
    # LedController itself is constructed after Stage B so it can reference
    # the DeviceSession for cancel routing.
    state.audio_control = AudioControl()

    process_audio_thread = threading.Thread(
        target=process_audio,
        args=(state, mic, args.audio_input_block_size),
        daemon=True,
    )
    process_audio_thread.start()

    # Expose the loop on state BEFORE wiring subscribers — satellite.stop()
    # reads state.loop to schedule bridge cancels thread-safely.
    state.loop = loop

    # MQTT cancel side-channel — lets an external trigger (red Calisto button
    # via led_service.py, HA automation) abort the current voice pipeline
    # without restarting the process. The native ESPHome protocol has no
    # inbound abort message, so this is the cleanest cross-process hook.
    _start_mqtt_cancel_subscriber(state, loop)

    # Stage B — DeviceSession + HABridge + BridgeClient. The satellite still
    # drives the v0 HA pipeline in B2; DeviceSession is a parallel state
    # holder + K.1 publisher. B3 wires MicCapture/ASR/TTS so the satellite
    # hands off the ASR path to DeviceSession.
    _start_stage_b_components(state, loop)

    # Stage E.1 — LED + mute + HID absorbed into LVA. Build LedController
    # after DeviceSession so cancel callbacks can route through it.
    _start_stage_e1_led(state, loop)

    # Stage E.2 — alarm / chime orchestration + K.10 say. Constructs the
    # AudibleNotifyArbiter, AlarmController, and ChimeController and wires
    # the alarm/say MQTT routing into HABridge.
    _start_stage_e2_audio(state, loop)

    # Auto discovery (zeroconf, mDNS)
    discovery = HomeAssistantZeroconf(port=args.port, name=state.name, mac_address=state.mac_address, host_ip_address=host_ip_address)
    await discovery.register_server()

    try:
        async with server:  # type: ignore
            _LOGGER.info("Server started (host=%s, port=%s)", host_ip_address, args.port)
            await server.serve_forever()  # type: ignore
    except KeyboardInterrupt:
        pass
    finally:
        state.audio_queue.put_nowait(None)
        process_audio_thread.join()

    _LOGGER.debug("Server stopped")


# -----------------------------------------------------------------------------


from .cancel import VALID_REASONS as _K3_REASONS


def _parse_cancel_payload(payload: bytes) -> dict:
    """Map K.3 cancel payload → dict {reason, scope, source, request_id}.

    Unknown reasons surface as `EXTERNAL`; v0 publishers send bare `'1'`,
    which is tolerated. Scope/source/request_id default to None.
    """
    parsed: dict = {"reason": "EXTERNAL", "scope": None, "source": None, "request_id": None}
    if not payload:
        return parsed
    try:
        obj = json.loads(payload.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return parsed
    if not isinstance(obj, dict):
        return parsed
    reason = obj.get("reason")
    if isinstance(reason, str) and reason in _K3_REASONS:
        parsed["reason"] = reason
    scope = obj.get("scope")
    if isinstance(scope, str) and scope.lower() in ("voice", "alarm", "media", "all"):
        parsed["scope"] = scope.lower()
    src = obj.get("source")
    if isinstance(src, str):
        parsed["source"] = src
    rid = obj.get("request_id")
    if isinstance(rid, str):
        parsed["request_id"] = rid
    return parsed


def _start_mqtt_cancel_subscriber(state: ServerState, loop: asyncio.AbstractEventLoop) -> None:
    import os
    host = os.environ.get("LVA_MQTT_HOST")
    if not host:
        _LOGGER.info("LVA_MQTT_HOST not set; voice-cancel MQTT subscriber disabled")
        return
    room = os.environ.get("ROOM", "living_room")
    port = int(os.environ.get("LVA_MQTT_PORT", "1883"))
    username = os.environ.get("LVA_MQTT_USER") or None
    password = os.environ.get("LVA_MQTT_PASS") or None
    # K.3 + K.4: scoped + broadcast cancel topics.
    cancel_topic = f"calisto/{room}/cancel"
    cancel_topic_all = "calisto/all/cancel"
    # K.1: where satellite.stop() will publish session/state.
    state_topic = f"calisto/{room}/session/state"

    try:
        import paho.mqtt.client as mqtt  # type: ignore
    except ImportError:
        _LOGGER.error("paho-mqtt not installed; voice-cancel subscriber disabled")
        return

    cancel_beep = str(_SOUNDS_DIR / "mute_switch_on.flac")

    def _cancel_pipeline(parsed: dict) -> None:
        # Stage F1 — every MQTT-driven cancel routes through the
        # coordinator. The coordinator owns scope/tier resolution +
        # fan-out to voice/alarm/media. Audible "I heard you" beep is
        # scheduled separately so it survives the voice teardown.
        coord = getattr(state, "cancel_coordinator", None)
        if coord is None:
            _LOGGER.debug(
                "cancel received (reason=%s) but no CancelCoordinator wired; dropping",
                parsed.get("reason"),
            )
            return
        coord.cancel(
            parsed.get("reason"),
            scope=parsed.get("scope"),
            source=parsed.get("source") or "mqtt",
            request_id=parsed.get("request_id"),
        )
        try:
            loop.call_later(0.15, lambda: state.tts_player.play(cancel_beep))
        except Exception:
            _LOGGER.exception("cancel beep schedule failed")

    def _on_connect(c, _ud, _flags, rc, _props=None):
        if rc == 0:
            c.subscribe(cancel_topic, qos=1)
            c.subscribe(cancel_topic_all, qos=1)
            _LOGGER.info(
                "MQTT cancel subscriber connected to %s:%d, subscribed to %s, %s",
                host,
                port,
                cancel_topic,
                cancel_topic_all,
            )
        else:
            _LOGGER.error("MQTT cancel subscriber connect failed rc=%s", rc)

    def _on_message(_c, _ud, msg):
        _LOGGER.debug("MQTT cancel msg topic=%s payload=%r", msg.topic, msg.payload)
        parsed = _parse_cancel_payload(msg.payload)
        loop.call_soon_threadsafe(_cancel_pipeline, parsed)

    client = mqtt.Client(
        mqtt.CallbackAPIVersion.VERSION2,  # type: ignore[attr-defined]
        client_id=f"lva-cancel-{room}",
        clean_session=True,
    )
    if username:
        client.username_pw_set(username, password)
    client.on_connect = _on_connect
    client.on_message = _on_message
    client.reconnect_delay_set(min_delay=1, max_delay=30)
    try:
        client.connect_async(host, port, keepalive=60)
        client.loop_start()
        # Expose the client + topic so satellite.stop() can publish K.1.
        state.room = room
        state.mqtt_client = client
        state.mqtt_state_topic = state_topic
    except Exception:
        _LOGGER.exception("MQTT cancel subscriber failed to start")


def _parse_mqtt_broker(url: str) -> tuple:
    """Parse `mqtt://host:port` into (host, port). Returns (host, 1883) on failure."""
    from urllib.parse import urlparse
    try:
        parsed = urlparse(url if "://" in url else f"mqtt://{url}")
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or 1883
        return host, port
    except Exception:
        return "127.0.0.1", 1883


def _start_stage_b_components(state: ServerState, loop: asyncio.AbstractEventLoop) -> None:
    """Construct DeviceSession + HABridge + BridgeClient and attach to state.

    Env vars (per M.5; LVA_MQTT_* names retained as fallback for v0 deploys):
      ROOM                       — device room slug
      MQTT_BROKER | LVA_MQTT_HOST/PORT — broker host:port
      MQTT_USER   | LVA_MQTT_USER     — broker username (optional)
      MQTT_PASSWORD | LVA_MQTT_PASS   — broker password (optional)
      BRIDGE_URL                 — claude-bridge base URL (optional in B2)
      STATE_REASSERT_INTERVAL_S  — Stage F (read into state for future use)
      HEARTBEAT_INTERVAL_S       — Stage F (read into state for future use)

    All components are best-effort: missing env vars produce log warnings
    and disable that component; the satellite still runs the v0 wire path.
    """
    import os

    room = os.environ.get("ROOM", state.room or "living_room")
    state.room = room

    # MQTT broker resolution: M.5 form takes priority.
    broker_url = os.environ.get("MQTT_BROKER")
    if broker_url:
        host, port = _parse_mqtt_broker(broker_url)
    else:
        host = os.environ.get("LVA_MQTT_HOST", "")
        port = int(os.environ.get("LVA_MQTT_PORT", "1883"))
    username = os.environ.get("MQTT_USER") or os.environ.get("LVA_MQTT_USER") or None
    password = os.environ.get("MQTT_PASSWORD") or os.environ.get("LVA_MQTT_PASS") or None

    ha_bridge: Optional[HABridge] = None
    if host:
        ha_bridge = HABridge(
            room=room,
            host=host,
            port=port,
            username=username,
            password=password,
        )
        try:
            ha_bridge.start()
            state.ha_bridge = ha_bridge
        except Exception:
            _LOGGER.exception("HABridge.start failed; K.1 publishes disabled")
            ha_bridge = None
    else:
        _LOGGER.info("No MQTT broker configured; HABridge disabled (K.1 publishes off)")

    # DeviceSession wraps the satellite once state is wired. Constructing it
    # attaches it to state.device_session, after which the satellite shim
    # routes _bump_gen / _set_state_label through it.
    session = DeviceSession(state, ha_bridge=ha_bridge)
    # Publish initial STARTING then IDLE per K.1.3 (startup transition).
    session.transition_to("STARTING", reason="startup")
    session.transition_to("IDLE", reason="startup")

    # Stage F1 — CancelCoordinator is the single entry-point for every
    # cancel trigger (HID, MQTT K.3/K.4, voice stop-word, internal). It
    # must exist before LedController is constructed so the phone-cancel
    # callback can route through it.
    coord = CancelCoordinator(state, loop=loop)
    if ha_bridge is not None:
        try:
            ha_bridge.attach_cancel_coordinator(coord)
        except Exception:
            _LOGGER.exception("Stage F1: ha_bridge.attach_cancel_coordinator raised")

        # Stage F5 — K.13 admin/restart: clean exit so systemd / docker
        # restart=unless-stopped respawns. Implemented as `sys.exit(0)`
        # scheduled on the loop so paho's on_message handler returns
        # cleanly before the process dies.
        def _restart_hook() -> None:
            _LOGGER.warning("K.13 admin/restart received — exiting for L3 respawn")
            loop.call_soon_threadsafe(lambda: sys.exit(0))
        try:
            ha_bridge.attach_restart_hook(_restart_hook)
        except Exception:
            _LOGGER.exception("Stage F5: ha_bridge.attach_restart_hook raised")

    # Stage F2 — Heartbeat publisher (K.2 every 60s, H2 L4).
    try:
        hb_interval = int(os.environ.get("HEARTBEAT_INTERVAL_S", "60"))
    except ValueError:
        hb_interval = 60
    heartbeat = HeartbeatPublisher(
        state,
        ha_bridge=ha_bridge,
        room=room,
        interval_s=hb_interval,
    )
    heartbeat.start(loop)
    state.heartbeat = heartbeat

    # Stage F4 — DeviceSession 30s slow re-assert watchdog (H4).
    try:
        reassert_interval = int(os.environ.get("STATE_REASSERT_INTERVAL_S", "30"))
    except ValueError:
        reassert_interval = 30
    session.start_state_watchdog(loop, interval_s=reassert_interval)

    # BridgeClient — wired in B3 as the THINKING-phase outbound HTTP.
    bridge_url = os.environ.get("BRIDGE_URL")
    if bridge_url:
        try:
            state.bridge_client = BridgeClient(bridge_url, device_session=session)
            _LOGGER.info("BridgeClient initialised for %s", bridge_url)
        except Exception:
            _LOGGER.exception("BridgeClient construction failed")
    else:
        _LOGGER.warning("BRIDGE_URL not set; v1 audio path disabled — wake will stall in WAKING")

    # ---- Stage B3 audio path: MicCapture + ASRClient + TTSOutput ----------
    whisper_uri = os.environ.get("WYOMING_WHISPER_URI")
    piper_uri = os.environ.get("WYOMING_PIPER_URI")
    asr_language = os.environ.get("ASR_LANGUAGE", "en")
    # E1 — default Piper voice is en_GB-jenny_dioco-medium per
    # architecture-v1-decisions §E. Deploys can override via PIPER_VOICE.
    piper_voice = os.environ.get("PIPER_VOICE") or "en_GB-jenny_dioco-medium"
    hotwords_raw = os.environ.get("ASR_HOTWORDS", "")
    hotwords = [w.strip() for w in hotwords_raw.split(",") if w.strip()]

    if whisper_uri:
        try:
            state.asr_client = ASRClient(
                whisper_uri,
                language=asr_language,
                hotwords=hotwords,
            )
            _LOGGER.info("ASRClient initialised (uri=%s lang=%s hotwords=%d)",
                         whisper_uri, asr_language, len(hotwords))
        except Exception:
            _LOGGER.exception("ASRClient construction failed")
    else:
        _LOGGER.warning("WYOMING_WHISPER_URI not set; v1 audio path disabled (ASR off)")

    if piper_uri:
        # Stage E.2 E2/E3 — opt into the streaming PCM path via TTS_STREAMING=1.
        # Default remains the Stage B3 tempfile path (TTSOutput); flipping the
        # flag on hardware lets us measure first-audio latency without
        # touching code.
        use_streaming = os.environ.get("TTS_STREAMING", "0") in ("1", "true", "yes", "on")
        try:
            if use_streaming:
                from .tts_streaming import TTSStreamingOutput

                state.tts_output = TTSStreamingOutput(
                    piper_uri,
                    voice=piper_voice,
                    audio_device=args.audio_output_device,
                )
                _LOGGER.info(
                    "TTSStreamingOutput initialised (uri=%s voice=%s device=%s)",
                    piper_uri, piper_voice, args.audio_output_device,
                )
            else:
                state.tts_output = TTSOutput(piper_uri, voice=piper_voice)
                _LOGGER.info("TTSOutput initialised (uri=%s voice=%s)", piper_uri, piper_voice)
        except Exception:
            _LOGGER.exception("TTSOutput construction failed")
    else:
        _LOGGER.warning("WYOMING_PIPER_URI not set; v1 audio path disabled (TTS off)")

    # MicCapture wires only when ASR + bridge are both up — otherwise the v0
    # path is preferable to a half-broken v1 path. MicCapture refuses to
    # construct without TEN-VAD (otherwise capture degrades silently to a
    # 400ms cap); we log+leave the v0 path active in that case.
    if state.asr_client is not None and state.bridge_client is not None and state.tts_output is not None:
        try:
            state.mic_capture = MicCapture(
                loop=loop,
                on_speech_captured=session.on_speech_captured,
            )
            _LOGGER.warning("MicCapture initialised — v1 audio path ACTIVE (LVA no longer streams to HA)")
        except RuntimeError as exc:
            _LOGGER.error("MicCapture refused to wire (%s); v1 audio path DISABLED, v0 HA-streaming retained", exc)
        except Exception:
            _LOGGER.exception("MicCapture construction failed; v1 audio path DISABLED")
    else:
        _LOGGER.warning(
            "MicCapture disabled — asr=%s bridge=%s tts=%s; v0 HA-streaming path retained",
            state.asr_client is not None,
            state.bridge_client is not None,
            state.tts_output is not None,
        )

    # Stage C — wake-capture retraining loop. Always-on once env is present;
    # decoupled from MicCapture wiring so we still gather captures even when
    # the v1 audio path is degraded.
    wake_capture_dir = os.environ.get(
        "WAKE_CAPTURE_DIR",
        str(Path.home() / "wake_captures"),
    )
    try:
        wake_capture_max = int(os.environ.get("WAKE_CAPTURE_MAX_FILES", "5000"))
    except ValueError:
        wake_capture_max = 5000
    device_id = os.environ.get("DEVICE_ID")
    if not device_id and hasattr(os, "uname"):
        try:
            device_id = os.uname().nodename
        except Exception:
            device_id = None
    device_id = device_id or room
    try:
        wake_capture = WakeCapture(
            capture_dir=Path(wake_capture_dir),
            room=room,
            device_id=device_id,
            max_files=wake_capture_max,
        )
        wake_capture.attach_loop(loop)
        state.wake_capture = wake_capture
        loop.create_task(wake_capture.start_background_sweeps())
        _LOGGER.info(
            "WakeCapture initialised (dir=%s max=%d)",
            wake_capture_dir, wake_capture_max,
        )
    except Exception:
        _LOGGER.exception("WakeCapture construction failed; Stage C disabled")

    if state.wake_capture is not None:
        try:
            http_port = int(os.environ.get("WAKE_CAPTURE_HTTP_PORT", "8770"))
        except ValueError:
            http_port = 8770
        http_host = os.environ.get("WAKE_CAPTURE_HTTP_HOST", "0.0.0.0")
        try:
            wake_http = WakeCaptureHTTP(
                wake_capture=state.wake_capture,
                host=http_host,
                port=http_port,
            )
            wake_http.start()
            state.wake_capture_http = wake_http
        except Exception:
            _LOGGER.exception("WakeCaptureHTTP failed to start; triage endpoint disabled")

    if state.wake_capture is not None and state.ha_bridge is not None:
        http_advertise = os.environ.get("WAKE_CAPTURE_HTTP_ADVERTISE_URL")
        if not http_advertise and state.wake_capture_http is not None:
            http_advertise = f"http://{device_id}:{state.wake_capture_http.port}"
        try:
            discovery = WakeCaptureDiscovery(
                ha_bridge=state.ha_bridge,
                wake_capture=state.wake_capture,
                room=room,
                device_id=device_id,
                http_base_url=http_advertise,
            )
            loop.create_task(discovery.start())
            state.wake_capture_discovery = discovery
        except Exception:
            _LOGGER.exception("WakeCaptureDiscovery construction failed")

    _LOGGER.info(
        "Stage F components wired: STATE_REASSERT_INTERVAL_S=%d HEARTBEAT_INTERVAL_S=%d",
        reassert_interval,
        hb_interval,
    )

    # Stage D — SpeakerVerifier (D2). Constructed even when no model file
    # is present so the gate hook in process_audio can always call into a
    # live object; the verifier degrades to accept-all when its CAM++
    # session is unavailable.
    sv_enabled_env = os.environ.get("SV_ENABLED", "1") not in ("0", "false", "no", "off")
    state.sv_enabled = sv_enabled_env
    try:
        state.sv_threshold = float(os.environ.get("SV_THRESHOLD", str(state.sv_threshold)))
    except ValueError:
        pass
    enrollments_default = Path.home() / "calisto-led" / "data" / f"enrollments-{room}.json"
    enrollments_path = Path(os.environ.get("SV_ENROLLMENTS_PATH", enrollments_default))
    model_path_env = os.environ.get("CAMPLUS_MODEL_PATH")
    model_path = Path(model_path_env) if model_path_env else (_REPO_DIR / "models" / "campplus.onnx")
    fallback_policy = os.environ.get("SV_FALLBACK_POLICY", "accept_all")
    enrollments = EnrollmentsStore(
        enrollments_path,
        room=room,
        threshold=state.sv_threshold,
        fallback_policy=fallback_policy,
    )
    try:
        enrollments.load()
    except Exception:
        _LOGGER.exception("EnrollmentsStore.load raised; verifier will accept all")
    # Threshold from the file (if set) overrides the env default.
    state.sv_threshold = enrollments.threshold
    state.speaker_verifier = SpeakerVerifier(
        store=enrollments,
        model_path=model_path,
        enabled=state.sv_enabled,
    )
    _LOGGER.info(
        "Stage D wired: SpeakerVerifier enabled=%s users=%d model=%s threshold=%.2f",
        state.sv_enabled, len(enrollments.users), model_path, state.sv_threshold,
    )

    # Stage H J1 — WakeArbiter. Constructed even when HABridge isn't wired
    # (solo dev box) so the audio thread's `wake_arbiter.arbitrate(...)` is
    # never None. Without an MQTT publisher the arbiter still waits its
    # window but treats every wake as solo, which is the correct fallback
    # for a single-device deploy.
    try:
        from .wake_arbiter import WakeArbiter as _WakeArbiter

        state.wake_arbiter = _WakeArbiter(
            room=room,
            device_id=os.environ.get("DEVICE_ID") or room,
        )
        if ha_bridge is not None:
            try:
                ha_bridge.attach_wake_arbiter(state.wake_arbiter)
            except Exception:
                _LOGGER.exception("Stage H: ha_bridge.attach_wake_arbiter raised")
    except Exception:
        _LOGGER.exception("Stage H: WakeArbiter construction failed")

    if ha_bridge is not None:
        enrollment_handler = EnrollmentHandler(state, ha_bridge=ha_bridge, loop=loop)
        try:
            ha_bridge.attach_enrollment_handler(enrollment_handler)
        except Exception:
            _LOGGER.exception("Stage D: ha_bridge.attach_enrollment_handler raised")

        try:
            sv_discovery = SpeakerVerifierDiscovery(
                ha_bridge=ha_bridge,
                state=state,
                enrollments=enrollments,
                room=room,
            )
            ha_bridge.attach_speaker_verifier_discovery(sv_discovery)
            sv_discovery.start()
            state.speaker_verifier_discovery = sv_discovery
        except Exception:
            _LOGGER.exception("Stage D: SpeakerVerifierDiscovery construction failed")

        # Stage G — N.2 static per-room catalogue. Read-only sensors mirror
        # session/state + heartbeat; tunables (numbers / switches / select /
        # button) are registered below. Counter feeds the H4
        # state_publishes_per_min telemetry through the heartbeat payload.
        try:
            state.state_publish_counter = StatePublishCounter()
            ha_bridge.attach_state_publish_counter(state.state_publish_counter)
        except Exception:
            _LOGGER.exception("Stage G: StatePublishCounter wiring failed")

        try:
            entity_surface = EntitySurface(
                ha_bridge=ha_bridge,
                room=room,
                state=state,
            )
            ha_bridge.attach_entity_surface(entity_surface)
            _register_stage_g_tunables(entity_surface, state)
            entity_surface.start()
            state.entity_surface = entity_surface
        except Exception:
            _LOGGER.exception("Stage G: EntitySurface construction failed")


def _register_stage_g_tunables(surface: "EntitySurface", state: "ServerState") -> None:
    """Register the N.2 live-tunable rows on the EntitySurface.

    Setters write the backing field, then propagate to any in-process
    consumer (configure_duck_envelope on each mpv player, save_preferences
    for wake_sensitivity, HeartbeatPublisher.set_interval for the cadence).
    """

    def _set_wake_sensitivity(value: float) -> None:
        state.wake_word_1_threshold = float(value)
        state.preferences.wake_word_1_sensitivity = float(value)
        try:
            state.save_preferences()
        except Exception:
            _LOGGER.exception("Stage G: save_preferences raised after wake_sensitivity update")

    surface.register_tunable_number(
        thing="wake_sensitivity",
        name_suffix="Wake Sensitivity",
        min_value=0.1, max_value=0.9, step=0.05,
        getter=lambda: float(state.wake_word_1_threshold),
        setter=_set_wake_sensitivity,
        icon="mdi:waveform",
    )

    def _apply_duck_envelope() -> None:
        for player in (state.music_player, state.tts_player, state.chime_player, state.alarm_player):
            if player is None:
                continue
            try:
                player.configure_duck_envelope(
                    floor_pct=state.duck_floor_pct,
                    attack_ms=state.duck_attack_ms,
                    release_ms=state.duck_release_ms,
                )
            except Exception:
                _LOGGER.exception("Stage G: configure_duck_envelope raised")

    def _set_duck_floor(value: int) -> None:
        state.duck_floor_pct = int(value)
        _apply_duck_envelope()

    def _set_duck_attack(value: int) -> None:
        state.duck_attack_ms = int(value)
        _apply_duck_envelope()

    def _set_duck_release(value: int) -> None:
        state.duck_release_ms = int(value)
        _apply_duck_envelope()

    surface.register_tunable_number(
        thing="duck_floor_pct",
        name_suffix="Duck Floor %",
        min_value=10, max_value=80, step=5,
        getter=lambda: int(state.duck_floor_pct),
        setter=_set_duck_floor,
        unit="%", icon="mdi:volume-medium", is_int=True,
    )
    surface.register_tunable_number(
        thing="duck_attack_ms",
        name_suffix="Duck Attack",
        min_value=50, max_value=500, step=10,
        getter=lambda: int(state.duck_attack_ms),
        setter=_set_duck_attack,
        unit="ms", icon="mdi:arrow-down-bold", is_int=True,
    )
    surface.register_tunable_number(
        thing="duck_release_ms",
        name_suffix="Duck Release",
        min_value=100, max_value=1000, step=50,
        getter=lambda: int(state.duck_release_ms),
        setter=_set_duck_release,
        unit="ms", icon="mdi:arrow-up-bold", is_int=True,
    )

    def _set_heartbeat_interval(value: int) -> None:
        hb = state.heartbeat
        if hb is None:
            return
        hb.set_interval(int(value))

    def _get_heartbeat_interval() -> int:
        hb = state.heartbeat
        return int(hb.interval_s) if hb is not None else 60

    surface.register_tunable_number(
        thing="heartbeat_interval_s",
        name_suffix="Heartbeat Interval",
        min_value=15, max_value=300, step=15,
        getter=_get_heartbeat_interval,
        setter=_set_heartbeat_interval,
        unit="s", icon="mdi:heart-pulse", is_int=True,
    )

    # Top-level audible_notify (E6 ChimeController gate). Distinct from
    # sv_audible_notify (D2) which gates the post-wake rejection chime.
    def _get_audible_notify() -> bool:
        arbiter = state.audible_notify_arbiter
        return bool(getattr(arbiter, "enabled", True)) if arbiter is not None else True

    def _set_audible_notify(value: bool) -> None:
        arbiter = state.audible_notify_arbiter
        if arbiter is not None:
            arbiter.enabled = bool(value)

    surface.register_tunable_switch(
        thing="audible_notify",
        name_suffix="Audible Notify",
        getter=_get_audible_notify,
        setter=_set_audible_notify,
        icon="mdi:bell-ring",
    )

    # Mute: command topic + state topic are owned by LedController +
    # HABridge.publish_mute_state. EntitySurface only publishes the
    # Discovery config so the entity appears on the dashboard.
    surface.register_passthrough_switch(
        thing="mute",
        name_suffix="Mute",
        state_topic=f"calisto/{state.room}/mute/state",
        set_topic=f"calisto/{state.room}/mute/set",
        payload_on="on",
        payload_off="off",
        icon="mdi:microphone-off",
    )

    # Alarm ringtone select. Options enumerated from the sound library on
    # this host; falls back to the current state value when the library
    # is empty so HA always has at least one valid option.
    from .audible_notify import list_alarm_ringtones
    ringtone_options = list_alarm_ringtones()
    if not ringtone_options:
        ringtone_options = [state.alarm_ringtone]
    if state.alarm_ringtone not in ringtone_options:
        ringtone_options = [state.alarm_ringtone] + ringtone_options

    def _set_alarm_ringtone(value: str) -> None:
        state.alarm_ringtone = value

    surface.register_tunable_select(
        thing="alarm_ringtone",
        name_suffix="Alarm Ringtone",
        options=ringtone_options,
        getter=lambda: state.alarm_ringtone,
        setter=_set_alarm_ringtone,
        icon="mdi:music-note",
    )

    # Stage H J1 — wake-arbitration tunables. When the arbiter wasn't
    # constructed (e.g. test stubs) the setters become no-ops.
    arbiter = getattr(state, "wake_arbiter", None)
    if arbiter is not None:
        surface.register_tunable_number(
            thing="wake_arbitration_wait_ms",
            name_suffix="Wake Arbitration Wait",
            min_value=50, max_value=500, step=25,
            getter=lambda: int(arbiter.wait_ms),
            setter=lambda v: arbiter.set_wait_ms(int(v)),
            unit="ms", icon="mdi:timer-sand", is_int=True,
        )
        surface.register_tunable_number(
            thing="wake_arbitration_tiebreak_band",
            name_suffix="Wake Arbitration Tiebreak Band",
            min_value=0.01, max_value=0.20, step=0.01,
            getter=lambda: float(arbiter.tiebreak_band),
            setter=lambda v: arbiter.set_tiebreak_band(float(v)),
            icon="mdi:vector-difference",
        )
        surface.register_tunable_switch(
            thing="wake_arbitration_enabled",
            name_suffix="Wake Arbitration Enabled",
            getter=lambda: bool(arbiter.enabled),
            setter=lambda v: arbiter.set_enabled(bool(v)),
            icon="mdi:swap-horizontal-bold",
        )

    # I2 dashboard cancel button — publishes the K.3 cancel payload that
    # CancelCoordinator (already subscribed) parses and acts on.
    surface.register_button(
        thing="stop",
        name_suffix="Stop",
        command_topic=f"calisto/{state.room}/cancel",
        press_payload="{\"reason\":\"DASHBOARD\",\"source\":\"ha_dashboard\"}",
        icon="mdi:stop-circle",
    )


def _start_stage_e1_led(state: ServerState, loop: asyncio.AbstractEventLoop) -> None:
    """Construct LedController + wire its callbacks into DeviceSession / HABridge.

    DeviceSession must already be on `state` (built by
    `_start_stage_b_components`). Path B no longer requires AudioControl —
    mute is cosmetic LEDs + MicCapture frame-gate, no audio-claim release.

    All callbacks are best-effort: a failure in any one path logs and
    swallows so a malformed dependency doesn't take down the LED surface.
    """
    session = state.device_session
    if session is None:
        _LOGGER.error("Stage E.1: state.device_session missing — LED disabled")
        return

    def _phone_cancel(reason: str) -> None:
        # Stage F1 — route every red-button cancel through the coordinator
        # so soft/hard tier resolution and music-touches semantics live
        # in one place (cancel.py), not split between callers.
        coord = state.cancel_coordinator
        if coord is None:
            try:
                session._cancel_via_satellite(reason)
            except Exception:
                _LOGGER.exception("Stage E.1: phone-cancel fallback raised")
            return
        try:
            coord.cancel(reason, source="hid_phone_button")
        except Exception:
            _LOGGER.exception("Stage F1: phone-cancel coordinator raised")

    def _phone_button(action: str) -> None:
        # v0 back-compat — publish the legacy `calisto/<room>/button/
        # phone/{short,long}` event for HA automations that still listen.
        bridge = state.ha_bridge
        publish = getattr(bridge, "publish_phone_button", None) if bridge else None
        if publish is None:
            return
        try:
            publish(action)
        except Exception:
            _LOGGER.exception("Stage E.1: phone-button publish raised")

    def _mic_capture_mute() -> None:
        mc = state.mic_capture
        if mc is not None:
            try:
                mc.mute()
            except Exception:
                _LOGGER.exception("Stage E.1: mic_capture.mute raised")

    def _mic_capture_unmute() -> None:
        mc = state.mic_capture
        if mc is not None:
            try:
                mc.unmute()
            except Exception:
                _LOGGER.exception("Stage E.1: mic_capture.unmute raised")

    def _volume_publish(pct: int) -> None:
        # ha_bridge MQTT publish wired in Stage E.1 step 11. Until that
        # lands, surface the change in logs so deploy verification can
        # confirm the hardware → MQTT path is alive.
        bridge = state.ha_bridge
        publish = getattr(bridge, "publish_volume_state", None) if bridge else None
        if publish is None:
            _LOGGER.info("Stage E.1: volume → %d%% (no HABridge publisher wired yet)", pct)
            return
        try:
            publish(pct)
        except Exception:
            _LOGGER.exception("Stage E.1: HABridge.publish_volume_state raised")

    def _mute_state_changed(muted: bool) -> None:
        # Mirror to ServerState.muted so any other audio path consumer
        # can defend (Path B keeps the USB claim open; MicCapture.feed
        # is the load-bearing gate).
        state.muted = muted
        _LOGGER.info("Stage E.1: _mute_state_changed fired (muted=%s)", muted)
        # Drive the recorder pause/resume handshake so WebRTC + uWW stop
        # running while muted (~10% CPU saving). request_pause/resume can
        # block up to 2-3s on the AudioControl condition; LedController
        # dispatches this callback via call_soon_threadsafe onto the
        # asyncio loop, so we offload to the default executor to avoid
        # stalling the loop.
        audio_ctrl = state.audio_control
        if audio_ctrl is not None:
            def _drive_audio_ctrl() -> None:
                try:
                    if muted:
                        ok = audio_ctrl.request_pause()
                        _LOGGER.info("Stage E.1: AudioControl.request_pause → %s", ok)
                    else:
                        ok = audio_ctrl.request_resume()
                        _LOGGER.info("Stage E.1: AudioControl.request_resume → %s", ok)
                except Exception:
                    _LOGGER.exception(
                        "Stage E.1: AudioControl.%s raised",
                        "request_pause" if muted else "request_resume",
                    )
            loop.run_in_executor(None, _drive_audio_ctrl)
        bridge = state.ha_bridge
        publish = getattr(bridge, "publish_mute_state", None) if bridge else None
        if publish is None:
            _LOGGER.info("Stage E.1: mute → %s (no HABridge publisher wired yet)", muted)
            return
        try:
            publish(muted)
        except Exception:
            _LOGGER.exception("Stage E.1: HABridge.publish_mute_state raised")

    controller = LedController(
        loop=loop,
        mic_capture_mute=_mic_capture_mute,
        mic_capture_unmute=_mic_capture_unmute,
        on_phone_cancel=_phone_cancel,
        on_phone_button=_phone_button,
        on_volume_change=_volume_publish,
        on_mute_state_changed=_mute_state_changed,
    )
    try:
        controller.start()
    except Exception:
        _LOGGER.exception("Stage E.1: LedController.start raised — LED surface degraded")
        return
    state.led_controller = controller
    # Wire the back-compat MQTT control plane: HABridge subscribes on
    # connect, routes inbound topics to controller methods.
    bridge = state.ha_bridge
    if bridge is not None:
        try:
            bridge.attach_led_controller(controller)
        except Exception:
            _LOGGER.exception("Stage E.1: HABridge.attach_led_controller raised")
    _LOGGER.info(
        "Stage E.1: LedController started (mute=Path B cosmetic + mic-gate, mqtt=%s)",
        "wired" if bridge is not None else "absent",
    )


def _start_stage_e2_audio(state: ServerState, loop: asyncio.AbstractEventLoop) -> None:
    """Construct AlarmController + AudibleNotifyArbiter + ChimeController.

    DeviceSession must already be on `state`. HABridge is optional — when
    absent the controllers still work locally (no K.9 publish, no /set
    routing); useful for headless test boxes that drive alarms by direct
    method call rather than MQTT.
    """
    from .session import State as _State

    session = state.device_session
    if session is None:
        _LOGGER.error("Stage E.2: state.device_session missing — alarm/chime disabled")
        return

    def _current_state() -> _State:
        ds = state.device_session
        if ds is None:
            return _State.IDLE
        return ds.state_value

    def _alarm_ringing() -> bool:
        ac = state.alarm_controller
        return bool(ac is not None and ac.is_ringing)

    arbiter = AudibleNotifyArbiter(
        state_getter=_current_state,
        alarm_ringing_getter=_alarm_ringing,
    )
    state.audible_notify_arbiter = arbiter

    state.chime_controller = ChimeController(
        chime_player=state.chime_player,
        arbiter=arbiter,
    )

    state.alarm_controller = AlarmController(
        alarm_player=state.alarm_player,
        music_player=state.music_player,
        room=state.room,
        ha_bridge=state.ha_bridge,
    )

    # HABridge needs to know about the alarm + audible-notify surface so the
    # alarm/set + say MQTT topics route through. attach is best-effort.
    bridge = state.ha_bridge
    if bridge is not None:
        try:
            bridge.attach_audio_controllers(
                alarm=state.alarm_controller,
                chime=state.chime_controller,
                tts_player=state.tts_player,
                tts_output=state.tts_output,
                arbiter=arbiter,
                loop=loop,
            )
        except Exception:
            _LOGGER.exception("Stage E.2: HABridge.attach_audio_controllers raised")

    _LOGGER.info(
        "Stage E.2: alarm + audible-notify wired (alarm=%s chime=%s arbiter=%s ha_bridge=%s)",
        state.alarm_controller is not None,
        state.chime_controller is not None,
        arbiter is not None,
        bridge is not None,
    )


def _run_speaker_gates(state: ServerState, wake_id: Optional[str]) -> bool:
    """Stage D — run Gate 1 + Gate 2 against the wake's pre-roll audio.

    Returns True when the verifier passes (or is disabled / accept-all).
    Returns False when either gate rejects — in which case it has already
    handled the rejection UX (chime, LED flash) and patched the sidecar
    label. The caller skips `satellite.wakeup`.
    """
    verifier = getattr(state, "speaker_verifier", None)
    if verifier is None:
        return True
    wake_capture = getattr(state, "wake_capture", None)
    if wake_capture is None:
        # No ring to snapshot. Without an audio source, the gates can't
        # decide — accept all and rely on the LLM-side cancel tools.
        return True
    if not verifier.is_active():
        return True
    pcm = wake_capture.snapshot_recent_pcm(verifier.verify_window_ms / 1000.0)
    if not pcm:
        return True
    try:
        result = verifier.verify(pcm)
    except Exception:
        _LOGGER.exception("SpeakerVerifier.verify raised; accept-all fallback")
        return True
    if wake_id is not None:
        try:
            wake_capture.update_speaker_match(wake_id, result.to_dict())
        except Exception:
            _LOGGER.exception("WakeCapture.update_speaker_match raised")
    if result.gate1_pass and result.gate2_pass:
        return True
    # Reject UX + sidecar labelling. Gate 1 fails silently per D2;
    # Gate 2 fires the audible+visible rejection.
    if not result.gate1_pass:
        _LOGGER.info("Speaker gates: Gate 1 (VAD) failed — silent drop")
        if wake_id is not None:
            try:
                wake_capture.update_wake_label(wake_id, "negative", "gate1_fail_vad")
            except Exception:
                _LOGGER.exception("WakeCapture.update_wake_label raised")
        return False
    _LOGGER.info(
        "Speaker gates: Gate 2 (CAM++) failed score=%.3f threshold=%.3f",
        result.score, result.threshold,
    )
    # Audible rejection chime — gated by sv_audible_notify.
    if state.sv_audible_notify:
        chime = getattr(state, "chime_controller", None)
        if chime is not None:
            try:
                chime.play("dialog-error.ogg")
            except Exception:
                _LOGGER.exception("ChimeController.play raised for rejection chime")
    # Visible red flash via the LED hard-cancel overlay.
    led = getattr(state, "led_controller", None)
    if led is not None:
        from .session import State as _State
        try:
            led.on_state(_State.CANCELLING, cancel_reason="GATE2_REJECT")
        except Exception:
            _LOGGER.exception("LedController.on_state raised for GATE2_REJECT flash")
    if wake_id is not None:
        try:
            wake_capture.update_wake_label(wake_id, "gate2_reject", "gate2_reject")
        except Exception:
            _LOGGER.exception("WakeCapture.update_wake_label raised")
    return False


def process_audio(state: ServerState, mic, block_size: int):
    """Process audio chunks from the microphone."""

    wake_words: List[Union[MicroWakeWord, OpenWakeWord]] = []
    micro_features: Optional[MicroWakeWordFeatures] = None
    micro_inputs: List[np.ndarray] = []

    oww_features: Optional[OpenWakeWordFeatures] = None
    oww_inputs: List[np.ndarray] = []
    has_oww = False

    last_active: Optional[float] = None
    webrtc: Optional[WebRTCProcessor] = None

    # Stage E.1 — AudioControl coordinates pause/resume of the recorder
    # context so led/mute.py can release the USB Audio Class claim and
    # let firmware enter telephony mode for true firmware-level mute.
    # `None` when AudioControl is not wired (pre-Stage-E.1 startup, tests)
    # — the loop then behaves exactly as before.
    audio_ctrl = getattr(state, "audio_control", None)

    try:
        while True:  # outer pause/resume cycle
            if audio_ctrl is not None and audio_ctrl.is_pause_desired():
                # Pause requested before we (re)opened the recorder. Confirm
                # and wait for resume without ever holding the audio claim.
                _LOGGER.info(
                    "process_audio: pause requested pre-open; confirming + waiting"
                )
                audio_ctrl.confirm_paused()
                audio_ctrl.wait_for_resume()

            _LOGGER.debug("Opening audio input device: %s", mic.name)
            with mic.recorder(samplerate=16000, channels=1, blocksize=block_size) as mic_in:
                if audio_ctrl is not None:
                    audio_ctrl.confirm_resumed()
                while True:
                    if audio_ctrl is not None and audio_ctrl.is_pause_desired():
                        _LOGGER.info(
                            "process_audio: pause requested; exiting recorder context"
                        )
                        break
                    audio_chunk_array = mic_in.record(block_size).reshape(-1)
                    # little-endian 16-bit signed
                    mic_vol_scalar = max(0.1, min(1.0, state.mic_volume / 100.0))
                    audio_chunk = (np.clip(audio_chunk_array * mic_vol_scalar, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
                    agc = state.preferences.mic_auto_gain or 0
                    ns = state.preferences.mic_noise_suppression or 0

                    if agc > 0 or ns > 0:
                        if webrtc is None:
                            webrtc = WebRTCProcessor(agc_level=agc, ns_level=ns)
                        else:
                            webrtc.update_settings(agc, ns)
                        audio_chunk = webrtc.process(audio_chunk)
                        if not audio_chunk:
                            continue

                    # Stage F2 — mic-frame liveness for K.2 heartbeat (`mic_active`).
                    # Stamp every frame regardless of streaming state: an active
                    # mic stream is what we want to assert, not the streaming
                    # decision downstream.
                    state.last_mic_frame_ts = time.monotonic()

                    if state.satellite is None or not hasattr(state.satellite, "_is_streaming_audio"):
                        continue

                    # WAKE WORD
                    if (not wake_words) or (state.wake_words_changed and state.wake_words):
                        # Update list of wake word models to process
                        state.wake_words_changed = False
                        wake_words = [ww for ww in state.wake_words.values() if ww.id in state.active_wake_words]

                        # TODO: Load default stop word value from json into state and preferences missing.

                        has_oww = False
                        for idx, wake_word in enumerate(wake_words):

                            # Load default threshold from model json
                            wake_word_id = wake_word.id if hasattr(wake_word, "id") else next(iter(state.wake_words.keys()))
                            available_word = state.available_wake_words.get(wake_word_id)
                            # _LOGGER.debug("word= %s", state.available_wake_words.get(wake_word_id))
                            default_threshold = available_word.probability_cutoff if available_word else 0.7
                            _LOGGER.debug("Using default threshold %.3f for wake word '%s' from model config", default_threshold, wake_word_id)
                            # Check preferences override
                            if idx == 0:
                                old_val = state.wake_word_1_threshold
                                if state.preferences.wake_word_1_sensitivity is not None:
                                    state.wake_word_1_threshold = state.preferences.wake_word_1_sensitivity
                                else:
                                    state.wake_word_1_threshold = default_threshold
                                _LOGGER.debug("Wake Word 1 threshold set to %.3f (was %.3f, preferences: %s)", state.wake_word_1_threshold, old_val, state.preferences.wake_word_1_sensitivity)
                            elif idx == 1:
                                old_val = state.wake_word_2_threshold
                                if state.preferences.wake_word_2_sensitivity is not None:
                                    state.wake_word_2_threshold = state.preferences.wake_word_2_sensitivity
                                else:
                                    state.wake_word_2_threshold = default_threshold
                                _LOGGER.debug("Wake Word 2 threshold set to %.3f (was %.3f, preferences: %s)", state.wake_word_2_threshold, old_val, state.preferences.wake_word_2_sensitivity)

                            if isinstance(wake_word, OpenWakeWord):
                                has_oww = True

                        # Sync entity states after threshold values were updated
                        if state.satellite is not None:
                            _LOGGER.debug("Updating WebUI entities with new threshold values")

                            # Wake Word 1
                            if state.satellite.state.sensitivity_1_number_entity is not None:
                                _LOGGER.debug("  → Syncing Wake Word 1 entity to value %.3f", state.wake_word_1_threshold)
                                state.satellite.state.sensitivity_1_number_entity.sync_with_state()
                                _LOGGER.debug("  ✅ Wake Word 1 entity now has value %.3f", state.satellite.state.sensitivity_1_number_entity.value)

                            # Wake Word 2
                            if state.satellite.state.sensitivity_2_number_entity is not None:
                                _LOGGER.debug("  → Syncing Wake Word 2 entity to value %.3f", state.wake_word_2_threshold)
                                state.satellite.state.sensitivity_2_number_entity.sync_with_state()
                                _LOGGER.debug("  ✅ Wake Word 2 entity now has value %.3f", state.satellite.state.sensitivity_2_number_entity.value)

                            # Stop Word
                            if state.satellite.state.stop_sensitivity_number_entity is not None:
                                _LOGGER.debug("  → Syncing Stop Word entity to value %.3f", state.stop_word_threshold)
                                state.satellite.state.stop_sensitivity_number_entity.sync_with_state()
                                _LOGGER.debug("  ✅ Stop Word entity now has value %.3f", state.satellite.state.stop_sensitivity_number_entity.value)

                            _LOGGER.debug("All sensitivity entities synced successfully")

                            # Force push new state to connected Home Assistant instance
                            if state.satellite is not None:
                                try:
                                    _LOGGER.debug("Pushing updated state values to Home Assistant")
                                    for entity in [
                                        state.satellite.state.sensitivity_1_number_entity,
                                        state.satellite.state.sensitivity_2_number_entity,
                                        state.satellite.state.stop_sensitivity_number_entity,
                                    ]:
                                        if entity is not None:
                                            state.satellite.send_messages([NumberStateResponse(key=entity.key, state=entity.value)])  # type: ignore[attr-defined]
                                            _LOGGER.debug("  → Pushed value %.3f for entity %d", entity.value, entity.key)
                                except Exception as e:
                                    _LOGGER.debug("Could not push state (no client connected yet): %s", e)

                        # TODO: Save settings: At this moment settings are only saved when changed in the UI. Means that the default value can change while updating since its not saved in preferences.

                        if micro_features is None:
                            micro_features = MicroWakeWordFeatures()

                        if has_oww and (oww_features is None):
                            oww_features = OpenWakeWordFeatures.from_builtin()

                    try:
                        state.satellite.handle_audio(audio_chunk)

                        # Stage B3 — feed the v1 mic-capture path. The ring buffer
                        # always fills (for pre-roll); capture-active state is
                        # internal to MicCapture. Safe no-op when not wired.
                        if state.mic_capture is not None:
                            try:
                                state.mic_capture.feed(audio_chunk)
                            except Exception:
                                _LOGGER.exception("MicCapture.feed raised")

                        # Stage C — feed the wake-capture ring buffer. Independent
                        # of MicCapture: WakeCapture always tracks the 3s window
                        # leading up to a wake fire, while MicCapture only buffers
                        # the post-wake utterance. O(1) per chunk; safe no-op
                        # when wake_capture isn't wired.
                        if state.wake_capture is not None:
                            try:
                                state.wake_capture.feed(audio_chunk)
                            except Exception:
                                _LOGGER.exception("WakeCapture.feed raised")

                        assert micro_features is not None
                        micro_inputs.clear()
                        micro_inputs.extend(micro_features.process_streaming(audio_chunk))

                        if has_oww:
                            assert oww_features is not None
                            oww_inputs.clear()
                            oww_inputs.extend(oww_features.process_streaming(audio_chunk))

                        for wake_word_index, wake_word in enumerate(wake_words):
                            activated = False
                            activation_score = 0.0

                            # Set dynamic threshold depending on wake word index
                            if wake_word_index == 0:
                                threshold = state.wake_word_1_threshold
                                # _LOGGER.debug("Set wake word %d probability cutoff to %.3f", wake_word_index+1, state.wake_word_1_threshold)
                            elif wake_word_index == 1:
                                threshold = state.wake_word_2_threshold
                                # _LOGGER.debug("Set wake word %d probability cutoff to %.3f", wake_word_index+1, state.wake_word_2_threshold)
                            else:
                                threshold = 0.7
                                # _LOGGER.debug("Set wake word %d probability cutoff to fallback value 0.7", wake_word_index+1)

                            if isinstance(wake_word, MicroWakeWord):
                                # No debugging when no detection
                                wake_word.debug_probabilities = False

                                # set microWakeWord cutoff
                                wake_word.probability_cutoff = threshold

                                for micro_input in micro_inputs:
                                    if wake_word.process_streaming(micro_input):
                                        wake_word.debug_probabilities = True
                                        activated = True
                                        # MicroWakeWord doesn't expose prob over
                                        # process_streaming — use threshold as a
                                        # floor estimate for the sidecar.
                                        activation_score = max(activation_score, threshold)
                            elif isinstance(wake_word, OpenWakeWord):
                                for oww_input in oww_inputs:
                                    for prob in wake_word.process_streaming(oww_input):
                                        if prob > threshold:
                                            _LOGGER.debug("Wake word '%s' activated (probability %.3f exceeded threshold %.3f)", wake_word.wake_word, prob, threshold)  # type: ignore[attr-defined]
                                            activated = True
                                            if prob > activation_score:
                                                activation_score = float(prob)

                            if activated and not state.muted:
                                # Check refractory
                                now = time.monotonic()
                                if (last_active is None) or ((now - last_active) > state.refractory_seconds):
                                    # Stage C — capture the 3s window for the
                                    # retraining dataset before wakeup() runs, so
                                    # the WAV's audio matches the wake-fire moment.
                                    wake_id = None
                                    if state.wake_capture is not None:
                                        try:
                                            wake_id = state.wake_capture.on_wake_fire(
                                                score=activation_score,
                                                peak_score=activation_score,
                                                model=getattr(wake_word, "wake_word", None),
                                                sensitivity=threshold,
                                            )
                                        except Exception:
                                            _LOGGER.exception("WakeCapture.on_wake_fire raised")
                                    # Stage H J1 — wake arbitration BEFORE speaker
                                    # verification. Losers abort silently: no
                                    # chime, no LED, no generation bump. The
                                    # wake_capture sidecar is labelled negative
                                    # so the retraining set captures the room
                                    # acoustics for the loss case.
                                    arbiter = getattr(state, "wake_arbiter", None)
                                    if arbiter is not None:
                                        try:
                                            won = arbiter.arbitrate(
                                                score=activation_score,
                                                peak_score=activation_score,
                                            )
                                        except Exception:
                                            _LOGGER.exception("WakeArbiter.arbitrate raised; treating as solo")
                                            won = True
                                        if not won:
                                            if state.wake_capture is not None and wake_id is not None:
                                                try:
                                                    state.wake_capture.update_wake_label(
                                                        wake_id, "negative", "arbitration_lost",
                                                    )
                                                except Exception:
                                                    _LOGGER.exception("WakeCapture.update_wake_label raised")
                                            last_active = now
                                            continue
                                    # Stage D — two-gate speaker verification.
                                    # Runs BEFORE wake chime per D2. Accept-all
                                    # fallback applies when no model / enrollments.
                                    verified = _run_speaker_gates(state, wake_id)
                                    if not verified:
                                        # Verifier already handled the user-
                                        # visible rejection (silent on Gate 1,
                                        # chime+red-flash on Gate 2) and the
                                        # sidecar label. Skip wakeup entirely.
                                        last_active = now
                                        continue
                                    state.satellite.wakeup(wake_word, wake_id=wake_id)
                                    # Stage F2 — record for K.2 wake_count_5m.
                                    try:
                                        state.wake_events.append(now)
                                        if len(state.wake_events) > 1024:
                                            del state.wake_events[: len(state.wake_events) - 512]
                                    except Exception:
                                        pass
                                    last_active = now

                        # Always process to keep state correct
                        stopped = False

                        # No debugging when no detection
                        state.stop_word.debug_probabilities = False

                        # Apply stop word sensitivity threshold
                        state.stop_word.probability_cutoff = state.stop_word_threshold
                        # _LOGGER.debug("Set stop word probability cutoff to %.3f", state.stop_word_threshold)
                        for micro_input in micro_inputs:
                            if state.stop_word.process_streaming(micro_input):
                                state.stop_word.debug_probabilities = True
                                stopped = True

                        if stopped and (state.stop_word.id in state.active_wake_words) and not state.muted:
                            # Stage F1 H1 — barge-in stop-word fires only while
                            # LVA is producing output the user might want to
                            # interrupt (THINKING or SPEAKING). In LISTENING /
                            # FOLLOWUP the utterance must reach ASR/LLM intact.
                            ds = getattr(state, "device_session", None)
                            current_state = ds.state_value.value if ds is not None else state.device_state
                            if current_state in ("THINKING", "SPEAKING"):
                                _LOGGER.debug("Stop word detected (state=%s) — firing barge-in", current_state)
                                coord = state.cancel_coordinator
                                if coord is not None:
                                    coord.cancel(
                                        "STOP_WORD_INPROCESS",
                                        source="voice_barge_in",
                                    )
                                else:
                                    state.satellite.stop(cancel_reason="STOP_WORD_INPROCESS")
                            else:
                                _LOGGER.debug(
                                    "Stop word detected but H1-gated out (state=%s); dropping",
                                    current_state,
                                )
                    except Exception:
                        _LOGGER.exception("Unexpected error handling audio")
            # Inner `while` broke out — pause was requested. The `with`
            # block has now exited; the audio claim is released as far
            # as soundcard / ALSA / PipeWire are concerned. led/mute.py
            # still sleeps a short settle (STREAM_CLOSE_WAIT_SEC) after
            # `request_pause` returns, because firmware sensing the
            # release lags the userspace context exit.
            if audio_ctrl is not None:
                audio_ctrl.confirm_paused()
                audio_ctrl.wait_for_resume()
                # fall through to outer `while True` → re-enter recorder
            else:
                # No AudioControl wired but we broke out of the inner
                # loop somehow (shouldn't happen with the current guards
                # — only an exception would have left the inner loop).
                break
    except Exception:
        _LOGGER.exception("Unexpected error processing audio")
        sys.exit(1)


# -----------------------------------------------------------------------------


def run():
    asyncio.run(main())


if __name__ == "__main__":
    run()
