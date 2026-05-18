"""Stage C — LAN-only HTTP endpoint for wake-capture triage.

Routes
------
- ``GET  /wake_captures/list``                      → JSON, paginated, filtered
- ``GET  /wake_captures/<wake_id_str>.wav``         → WAV bytes
- ``POST /wake_captures/<wake_id_str>/label``       → JSON `{label, user?}`
- ``DELETE /wake_captures/<wake_id_str>``           → JSON `{ok, wake_id_str}`
- ``GET  /wake_captures/stats``                     → JSON `stats_24h()`
- ``GET  /healthz``                                 → ``"ok"``

Runs in a daemon ``ThreadingHTTPServer`` thread so the asyncio loop is never
blocked by HTTP I/O. No auth — the endpoint is bound to the LAN address and
matches the bridge's local-trust pattern. Stage G will fold this into the
HA Discovery surface; until then HA reaches it directly via REST commands.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Optional
from urllib.parse import parse_qs, urlparse

_LOGGER = logging.getLogger(__name__)


_WAKE_ID_RE = re.compile(r"^[A-Za-z0-9_\-]+$")


class _Handler(BaseHTTPRequestHandler):

    wake_capture: Any = None

    def log_message(self, format: str, *args: Any) -> None:
        _LOGGER.debug("wake_capture_http: " + format, *args)

    def _send_json(self, status: int, body: Any) -> None:
        data = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _send_text(self, status: int, body: str, ctype: str = "text/plain") -> None:
        data = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query or "")
        wc = self.wake_capture
        if wc is None:
            self._send_text(503, "wake_capture unavailable")
            return

        if path == "/healthz":
            self._send_text(200, "ok")
            return
        if path == "/wake_captures/list":
            include_all = qs.get("all", ["0"])[0] in ("1", "true", "yes")
            try:
                page = max(0, int(qs.get("page", ["0"])[0]))
            except ValueError:
                page = 0
            try:
                page_size = max(1, min(200, int(qs.get("page_size", ["50"])[0])))
            except ValueError:
                page_size = 50
            self._send_json(200, wc.list_captures(
                include_all=include_all, page=page, page_size=page_size,
            ))
            return
        if path == "/wake_captures/stats":
            self._send_json(200, wc.stats_24h())
            return

        m = re.match(r"^/wake_captures/([^/]+)\.wav$", path)
        if m:
            wake_id = m.group(1)
            if not _WAKE_ID_RE.match(wake_id):
                self._send_text(400, "bad wake_id")
                return
            wav = wc.read_wav_bytes(wake_id)
            if wav is None:
                self._send_text(404, "not found")
                return
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(wav)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(wav)
            return

        self._send_text(404, "not found")

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        wc = self.wake_capture
        if wc is None:
            self._send_text(503, "wake_capture unavailable")
            return

        m = re.match(r"^/wake_captures/([^/]+)/label$", path)
        if not m:
            self._send_text(404, "not found")
            return
        wake_id = m.group(1)
        if not _WAKE_ID_RE.match(wake_id):
            self._send_text(400, "bad wake_id")
            return

        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0 or length > 4096:
            self._send_text(400, "missing or oversized body")
            return
        try:
            raw = self.rfile.read(length)
            payload = json.loads(raw.decode("utf-8"))
        except Exception:
            self._send_text(400, "invalid json")
            return

        label = payload.get("label")
        user = payload.get("user") or "manual"
        if not isinstance(label, str) or not isinstance(user, str):
            self._send_text(400, "missing label")
            return
        if len(user) > 64:
            self._send_text(400, "user too long")
            return

        ok = wc.manual_label(wake_id, label, user=user)
        if not ok:
            self._send_text(400, "label rejected (unknown wake_id or invalid label)")
            return
        self._send_json(200, {"ok": True, "wake_id_str": wake_id, "label": label})

    def do_DELETE(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        wc = self.wake_capture
        if wc is None:
            self._send_text(503, "wake_capture unavailable")
            return

        m = re.match(r"^/wake_captures/([^/]+)$", path)
        if not m:
            self._send_text(404, "not found")
            return
        wake_id = m.group(1)
        if not _WAKE_ID_RE.match(wake_id):
            self._send_text(400, "bad wake_id")
            return

        ok = wc.delete_capture(wake_id)
        if not ok:
            self._send_text(404, "wake_id not found")
            return
        self._send_json(200, {"ok": True, "wake_id_str": wake_id})


class WakeCaptureHTTP:

    def __init__(
        self,
        *,
        wake_capture: Any,
        host: str = "0.0.0.0",
        port: int = 8770,
    ) -> None:
        self.wake_capture = wake_capture
        self.host = host
        self.port = port
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._server is not None:
            return

        capture_ref = self.wake_capture

        class _BoundHandler(_Handler):
            wake_capture = capture_ref

        server = ThreadingHTTPServer((self.host, self.port), _BoundHandler)
        thread = threading.Thread(
            target=server.serve_forever,
            name="wake_capture_http",
            daemon=True,
        )
        thread.start()
        self._server = server
        self._thread = thread
        _LOGGER.info("WakeCaptureHTTP listening on %s:%d", self.host, self.port)

    def stop(self) -> None:
        if self._server is not None:
            try:
                self._server.shutdown()
                self._server.server_close()
            except Exception:
                _LOGGER.exception("WakeCaptureHTTP shutdown raised")
        self._server = None
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
