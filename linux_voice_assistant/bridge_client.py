"""Stage B — BridgeClient: async HTTP client for claude-bridge (Section L).

Implements L.1 (/chat), L.2 (/cancel), L.3 (/health) plus the L.6 timeout
matrix. Returns plain dataclasses so callers don't import httpx; tests can
construct identical objects without touching the network.

Stage B builds the client but does not call /chat from any code path. B3
wires it into MicCapture → ASRClient → BridgeClient → DeviceSession.

L.5 error policy summary:
  200 → ChatReply
  408 → BridgeTimeout (BRIDGE_TIMEOUT cancel chain on caller)
  409 → StaleGeneration (drop silently)
  503 → SubprocessUnavailable; caller retries once after retry_after_ms
        then escalates to BRIDGE_TIMEOUT
  500/4xx other → BridgeInternalError (treat as BRIDGE_TIMEOUT)
  socket/timeout → BridgeTimeout
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, List, Optional

import httpx

if TYPE_CHECKING:
    from .session import DeviceSession

_LOGGER = logging.getLogger(__name__)


# L.6 timeouts
_CONNECT_S = 2.0
_TOTAL_CHAT_S = 30.0
_CANCEL_S = 2.0
_HEALTH_S = 2.0


@dataclass
class ChatReply:
    generation: int
    session_id: str
    reply: str
    continue_conversation: bool
    tool_calls_made: List[str] = field(default_factory=list)
    turn_count: int = 0
    bridge_latency_ms: int = 0
    queue_wait_ms: int = 0
    model: str = ""
    x_bridge_generation: Optional[int] = None
    x_bridge_request_id: Optional[str] = None


@dataclass
class CancelResult:
    cancelled: bool
    generation: int
    phase_at_cancel: Optional[str] = None
    synthetic_note_queued: bool = False


@dataclass
class HealthResult:
    ok: bool
    subprocess_alive: bool
    subprocess_uptime_s: int = 0
    in_flight_room: Optional[str] = None
    queue_depth: int = 0
    model: str = ""
    respawn_eta_ms: Optional[int] = None
    http_status: int = 0


class BridgeError(Exception):
    """Base class for BridgeClient-raised errors."""


class BridgeTimeout(BridgeError):
    def __init__(self, phase: str, elapsed_ms: int = 0) -> None:
        super().__init__(f"bridge_timeout phase={phase} elapsed_ms={elapsed_ms}")
        self.phase = phase
        self.elapsed_ms = elapsed_ms


class StaleGeneration(BridgeError):
    def __init__(self, submitted: int, current: int) -> None:
        super().__init__(f"stale_generation submitted={submitted} current={current}")
        self.submitted = submitted
        self.current = current


class SubprocessUnavailable(BridgeError):
    def __init__(self, retry_after_ms: int) -> None:
        super().__init__(f"subprocess_unavailable retry_after_ms={retry_after_ms}")
        self.retry_after_ms = retry_after_ms


class BridgeInternalError(BridgeError):
    def __init__(self, status: int, detail: str = "") -> None:
        super().__init__(f"bridge_internal status={status} detail={detail}")
        self.status = status
        self.detail = detail


class BridgeClient:
    """Async HTTP client for claude-bridge.

    The `transport` kwarg is for tests (httpx.MockTransport). Production code
    passes nothing and gets a real AsyncHTTPTransport.
    """

    def __init__(
        self,
        base_url: str,
        *,
        transport: "Optional[httpx.AsyncBaseTransport]" = None,
        device_session: "Optional[DeviceSession]" = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.device_session = device_session
        timeout = httpx.Timeout(connect=_CONNECT_S, read=_TOTAL_CHAT_S, write=_TOTAL_CHAT_S, pool=_CONNECT_S)
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=timeout, transport=transport)

    async def aclose(self) -> None:
        await self._client.aclose()

    # ---- L.1 /chat -------------------------------------------------------

    async def chat(
        self,
        *,
        device: str,
        generation: int,
        session_id: str,
        text: str,
        caller_area: Optional[str] = None,
        user_id: Optional[str] = None,
        asr_confidence: Optional[float] = None,
    ) -> ChatReply:
        """Submit a turn. Returns ChatReply on 200; raises on every other path.

        Caller is expected to gen-check on return: if the device's generation
        has advanced since this coroutine was scheduled, the reply must be
        dropped (D-bridge-7 + H3). BridgeClient does NOT auto-drop; that
        decision belongs in the caller's await-site.
        """
        body: dict = {
            "device": device,
            "generation": generation,
            "session_id": session_id,
            "text": text,
            "client_ts": datetime.now().astimezone().isoformat(timespec="milliseconds"),
        }
        if caller_area is not None:
            body["caller_area"] = caller_area
        if user_id is not None:
            body["user_id"] = user_id
        if asr_confidence is not None:
            body["asr_confidence"] = asr_confidence

        attempts = 0
        while True:
            attempts += 1
            try:
                resp = await self._client.post("/chat", json=body, timeout=_TOTAL_CHAT_S)
            except httpx.ConnectTimeout as exc:
                raise BridgeTimeout(phase="connect") from exc
            except httpx.ReadTimeout as exc:
                raise BridgeTimeout(phase="subprocess_wait", elapsed_ms=int(_TOTAL_CHAT_S * 1000)) from exc
            except httpx.HTTPError as exc:
                raise BridgeTimeout(phase="connect") from exc

            x_bridge_gen = _parse_int_header(resp.headers.get("X-Bridge-Generation"))
            x_bridge_req = resp.headers.get("X-Bridge-Request-Id")

            if resp.status_code == 200:
                data = _safe_json(resp)
                return ChatReply(
                    generation=int(data.get("generation", generation)),
                    session_id=str(data.get("session_id", session_id)),
                    reply=str(data.get("reply", "")),
                    continue_conversation=bool(data.get("continue_conversation", False)),
                    tool_calls_made=list(data.get("tool_calls_made", []) or []),
                    turn_count=int(data.get("turn_count", 0)),
                    bridge_latency_ms=int(data.get("bridge_latency_ms", 0)),
                    queue_wait_ms=int(data.get("queue_wait_ms", 0)),
                    model=str(data.get("model", "")),
                    x_bridge_generation=x_bridge_gen,
                    x_bridge_request_id=x_bridge_req,
                )
            if resp.status_code == 408:
                data = _safe_json(resp)
                raise BridgeTimeout(
                    phase=str(data.get("phase", "subprocess_wait")),
                    elapsed_ms=int(data.get("elapsed_ms", 0)),
                )
            if resp.status_code == 409:
                data = _safe_json(resp)
                raise StaleGeneration(
                    submitted=int(data.get("submitted_generation", generation)),
                    current=int(data.get("current_generation_for_device", generation + 1)),
                )
            if resp.status_code == 503:
                data = _safe_json(resp)
                retry_after_ms = int(data.get("retry_after_ms", 1500))
                if attempts < 2:
                    # L.5: single retry after retry_after_ms.
                    import asyncio as _asyncio
                    await _asyncio.sleep(retry_after_ms / 1000.0)
                    continue
                raise SubprocessUnavailable(retry_after_ms=retry_after_ms)
            # 500 or other 4xx
            data = _safe_json(resp)
            raise BridgeInternalError(status=resp.status_code, detail=str(data.get("detail", "")))

    # ---- L.2 /cancel -----------------------------------------------------

    async def cancel(self, *, device: str, generation: int, reason: Optional[str] = None) -> CancelResult:
        body: dict = {"device": device, "generation": generation}
        if reason is not None:
            body["reason"] = reason
        try:
            resp = await self._client.post("/cancel", json=body, timeout=_CANCEL_S)
        except (httpx.TimeoutException, httpx.HTTPError):
            # Best-effort per L.6; local cancel chain already fired.
            return CancelResult(cancelled=False, generation=generation)
        if resp.status_code == 200:
            data = _safe_json(resp)
            return CancelResult(
                cancelled=bool(data.get("cancelled", False)),
                generation=int(data.get("generation", generation)),
                phase_at_cancel=data.get("phase_at_cancel"),
                synthetic_note_queued=bool(data.get("synthetic_note_queued", False)),
            )
        # 400 malformed or anything else — treat as not cancelled remotely.
        return CancelResult(cancelled=False, generation=generation)

    # ---- L.3 /health -----------------------------------------------------

    async def health(self) -> HealthResult:
        try:
            resp = await self._client.get("/health", timeout=_HEALTH_S)
        except (httpx.TimeoutException, httpx.HTTPError):
            return HealthResult(ok=False, subprocess_alive=False, http_status=0)
        data = _safe_json(resp)
        return HealthResult(
            ok=bool(data.get("ok", False)) and resp.status_code == 200,
            subprocess_alive=bool(data.get("subprocess_alive", False)),
            subprocess_uptime_s=int(data.get("subprocess_uptime_s", 0)),
            in_flight_room=data.get("in_flight_room"),
            queue_depth=int(data.get("queue_depth", 0)),
            model=str(data.get("model", "")),
            respawn_eta_ms=data.get("respawn_eta_ms"),
            http_status=resp.status_code,
        )


def _safe_json(resp: "httpx.Response") -> dict:
    try:
        data = resp.json()
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _parse_int_header(value: Optional[str]) -> Optional[int]:
    if not value:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
