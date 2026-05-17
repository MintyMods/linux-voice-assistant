"""Stage B — BridgeClient unit tests (L.1 / L.2 / L.3 + L.5/L.6 error model).

Uses httpx.MockTransport: no network. Covers happy paths, every status code
in the L.5 table, the 503 single-retry behaviour, and timeout mapping.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from linux_voice_assistant.bridge_client import (
    BridgeClient,
    BridgeInternalError,
    BridgeTimeout,
    StaleGeneration,
    SubprocessUnavailable,
)


def _ok_chat_response(generation: int = 142, session_id: str = "sess-1") -> httpx.Response:
    body = {
        "generation": generation,
        "session_id": session_id,
        "reply": "It's two thirty-two PM.",
        "continue_conversation": False,
        "tool_calls_made": ["GetLiveContext"],
        "turn_count": 2,
        "bridge_latency_ms": 1872,
        "queue_wait_ms": 0,
        "model": "sonnet-4.6",
    }
    return httpx.Response(
        200,
        json=body,
        headers={
            "X-Bridge-Generation": "9001",
            "X-Bridge-Request-Id": "req-abc",
        },
    )


def _run(coro):
    return asyncio.get_event_loop_policy().new_event_loop().run_until_complete(coro)


def test_chat_happy_path_parses_l1_response():
    seen: list = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _ok_chat_response()

    client = BridgeClient("http://bridge.test", transport=httpx.MockTransport(handler))

    async def run():
        return await client.chat(
            device="lounge",
            generation=142,
            session_id="sess-1",
            text="what time is it?",
        )

    try:
        reply = _run(run())
    finally:
        _run(client.aclose())

    assert reply.reply.endswith("PM.")
    assert reply.generation == 142
    assert reply.session_id == "sess-1"
    assert reply.tool_calls_made == ["GetLiveContext"]
    assert reply.x_bridge_generation == 9001
    assert reply.x_bridge_request_id == "req-abc"

    # Request body must include the L.1 required fields.
    sent_body = json.loads(seen[0].content.decode("utf-8"))
    assert sent_body["device"] == "lounge"
    assert sent_body["generation"] == 142
    assert sent_body["session_id"] == "sess-1"
    assert sent_body["text"] == "what time is it?"
    assert "client_ts" in sent_body  # L.1: telemetry timestamp always sent


def test_chat_408_raises_bridge_timeout():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(408, json={
            "error": "bridge_timeout",
            "generation": 142,
            "session_id": "sess-1",
            "elapsed_ms": 30000,
            "phase": "subprocess_wait",
        })

    client = BridgeClient("http://bridge.test", transport=httpx.MockTransport(handler))

    async def run():
        await client.chat(device="lounge", generation=142, session_id="sess-1", text="hi")

    try:
        with pytest.raises(BridgeTimeout) as exc:
            _run(run())
        assert exc.value.phase == "subprocess_wait"
        assert exc.value.elapsed_ms == 30000
    finally:
        _run(client.aclose())


def test_chat_409_raises_stale_generation():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(409, json={
            "error": "stale_generation",
            "current_generation_for_device": 143,
            "submitted_generation": 142,
        })

    client = BridgeClient("http://bridge.test", transport=httpx.MockTransport(handler))

    async def run():
        await client.chat(device="lounge", generation=142, session_id="sess-1", text="hi")

    try:
        with pytest.raises(StaleGeneration) as exc:
            _run(run())
        assert exc.value.submitted == 142
        assert exc.value.current == 143
    finally:
        _run(client.aclose())


def test_chat_503_retries_once_then_raises_subprocess_unavailable():
    call_count = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        call_count["n"] += 1
        # Two 503s in a row → single retry exhausted → SubprocessUnavailable.
        return httpx.Response(503, json={"error": "subprocess_unavailable", "retry_after_ms": 1})

    client = BridgeClient("http://bridge.test", transport=httpx.MockTransport(handler))

    async def run():
        await client.chat(device="lounge", generation=142, session_id="sess-1", text="hi")

    try:
        with pytest.raises(SubprocessUnavailable) as exc:
            _run(run())
        assert exc.value.retry_after_ms == 1
        assert call_count["n"] == 2  # original + one retry per L.5
    finally:
        _run(client.aclose())


def test_chat_503_then_200_succeeds_on_retry():
    call_count = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        call_count["n"] += 1
        if call_count["n"] == 1:
            return httpx.Response(503, json={"retry_after_ms": 1})
        return _ok_chat_response()

    client = BridgeClient("http://bridge.test", transport=httpx.MockTransport(handler))

    async def run():
        return await client.chat(device="lounge", generation=142, session_id="sess-1", text="hi")

    try:
        reply = _run(run())
    finally:
        _run(client.aclose())

    assert call_count["n"] == 2
    assert reply.generation == 142


def test_chat_500_raises_internal_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "internal", "detail": "boom"})

    client = BridgeClient("http://bridge.test", transport=httpx.MockTransport(handler))

    async def run():
        await client.chat(device="lounge", generation=142, session_id="sess-1", text="hi")

    try:
        with pytest.raises(BridgeInternalError) as exc:
            _run(run())
        assert exc.value.status == 500
        assert "boom" in exc.value.detail
    finally:
        _run(client.aclose())


def test_chat_connect_timeout_maps_to_bridge_timeout():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("connect timed out")

    client = BridgeClient("http://bridge.test", transport=httpx.MockTransport(handler))

    async def run():
        await client.chat(device="lounge", generation=142, session_id="sess-1", text="hi")

    try:
        with pytest.raises(BridgeTimeout) as exc:
            _run(run())
        assert exc.value.phase == "connect"
    finally:
        _run(client.aclose())


# ---- L.2 /cancel -----------------------------------------------------------


def test_cancel_happy_path_parses_l2_response():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        assert body["device"] == "lounge"
        assert body["generation"] == 142
        assert body["reason"] == "RED_BUTTON_SOFT"
        return httpx.Response(200, json={
            "cancelled": True,
            "generation": 142,
            "phase_at_cancel": "subprocess_wait",
            "synthetic_note_queued": True,
        })

    client = BridgeClient("http://bridge.test", transport=httpx.MockTransport(handler))

    async def run():
        return await client.cancel(device="lounge", generation=142, reason="RED_BUTTON_SOFT")

    try:
        result = _run(run())
    finally:
        _run(client.aclose())

    assert result.cancelled is True
    assert result.phase_at_cancel == "subprocess_wait"
    assert result.synthetic_note_queued is True


def test_cancel_swallows_timeout_returns_cancelled_false():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("nope")

    client = BridgeClient("http://bridge.test", transport=httpx.MockTransport(handler))

    async def run():
        return await client.cancel(device="lounge", generation=142)

    try:
        result = _run(run())
    finally:
        _run(client.aclose())

    # L.6: /cancel is best-effort; local cancel chain already fired.
    assert result.cancelled is False
    assert result.generation == 142


# ---- L.3 /health -----------------------------------------------------------


def test_health_happy_path():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "ok": True,
            "subprocess_alive": True,
            "subprocess_uptime_s": 12834,
            "in_flight_room": None,
            "queue_depth": 0,
            "model": "sonnet-4.6",
        })

    client = BridgeClient("http://bridge.test", transport=httpx.MockTransport(handler))

    async def run():
        return await client.health()

    try:
        result = _run(run())
    finally:
        _run(client.aclose())

    assert result.ok is True
    assert result.subprocess_alive is True
    assert result.http_status == 200


def test_health_503_returns_not_ok():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={
            "ok": False,
            "subprocess_alive": False,
            "respawn_eta_ms": 800,
        })

    client = BridgeClient("http://bridge.test", transport=httpx.MockTransport(handler))

    async def run():
        return await client.health()

    try:
        result = _run(run())
    finally:
        _run(client.aclose())

    assert result.ok is False
    assert result.subprocess_alive is False
    assert result.respawn_eta_ms == 800
    assert result.http_status == 503
