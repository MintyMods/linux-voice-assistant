"""Stage C — WakeCaptureHTTP handler tests.

Spins up the stdlib ``ThreadingHTTPServer`` on port 0, exercises each route
via ``urllib.request``, asserts status + body. Catches URL pattern typos,
JSON shape regressions, and the ``_WAKE_ID_RE`` allowlist.
"""

from __future__ import annotations

import json
import struct
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from linux_voice_assistant.wake_capture import SAMPLE_RATE, SAMPLE_WIDTH_BYTES, WakeCapture
from linux_voice_assistant.wake_capture_http import WakeCaptureHTTP


def _silence_chunk(ms: int) -> bytes:
    nsamples = int(SAMPLE_RATE * ms / 1000)
    return struct.pack("<" + "h" * nsamples, *([0] * nsamples))


class _Clock:
    now: float = 1_700_000_000.0
    def __call__(self) -> float:
        return self.now


@pytest.fixture
def http_server(tmp_path: Path):
    capture_dir = tmp_path / "wake_captures"
    capture_dir.mkdir()
    clock = _Clock()
    wc = WakeCapture(
        capture_dir=capture_dir,
        room="lounge",
        device_id="minty-test-01",
        max_files=100,
        clock=clock,
    )
    wc.feed(_silence_chunk(1000))
    a = wc.on_wake_fire(score=0.7)
    clock.now += 1.0
    b = wc.on_wake_fire(score=0.8)
    wc.manual_label(b, "positive")
    clock.now += 1.0
    c = wc.on_wake_fire(score=0.6)
    wc.manual_label(c, "ambiguous")

    server = WakeCaptureHTTP(wake_capture=wc, host="127.0.0.1", port=0)
    # Capture the OS-assigned port by initialising the server manually.
    from http.server import ThreadingHTTPServer
    import threading

    capture_ref = wc

    from linux_voice_assistant.wake_capture_http import _Handler

    class _BoundHandler(_Handler):
        wake_capture = capture_ref

    real_server = ThreadingHTTPServer(("127.0.0.1", 0), _BoundHandler)
    server._server = real_server
    server.port = real_server.server_address[1]
    server._thread = threading.Thread(target=real_server.serve_forever, daemon=True)
    server._thread.start()
    base = f"http://127.0.0.1:{server.port}"

    yield base, wc, a, b, c

    real_server.shutdown()
    real_server.server_close()


def _get(url: str, timeout: float = 2.0) -> tuple[int, bytes, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status, resp.read(), resp.headers.get("Content-Type", "")
    except urllib.error.HTTPError as e:
        return e.code, e.read(), e.headers.get("Content-Type", "")


def _post(url: str, body: dict, timeout: float = 2.0) -> tuple[int, bytes]:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST",
                                  headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def test_healthz_returns_ok(http_server):
    base, _, *_ = http_server
    status, body, _ = _get(base + "/healthz")
    assert status == 200
    assert body == b"ok"


def test_list_default_returns_triage_bucket(http_server):
    base, _, a, b, c = http_server
    status, body, ctype = _get(base + "/wake_captures/list")
    assert status == 200
    assert ctype.startswith("application/json")
    payload = json.loads(body)
    ids = {it["wake_id_str"] for it in payload["items"]}
    assert a in ids       # unlabelled
    assert c in ids       # ambiguous
    assert b not in ids   # positive — excluded by default filter


def test_list_all_returns_everything(http_server):
    base, _, a, b, c = http_server
    status, body, _ = _get(base + "/wake_captures/list?all=true")
    assert status == 200
    payload = json.loads(body)
    ids = {it["wake_id_str"] for it in payload["items"]}
    assert ids == {a, b, c}


def test_list_pagination_respects_page_size(http_server):
    base, *_ = http_server
    status, body, _ = _get(base + "/wake_captures/list?page=0&page_size=1")
    assert status == 200
    payload = json.loads(body)
    assert payload["page_size"] == 1
    assert len(payload["items"]) == 1


def test_wav_endpoint_returns_audio_bytes(http_server):
    base, _, a, *_ = http_server
    status, body, ctype = _get(f"{base}/wake_captures/{a}.wav")
    assert status == 200
    assert ctype == "audio/wav"
    assert body[:4] == b"RIFF"  # WAV header magic
    assert len(body) > 100


def test_wav_endpoint_404s_on_unknown_wake_id(http_server):
    base, *_ = http_server
    status, _, _ = _get(base + "/wake_captures/does_not_exist.wav")
    assert status == 404


def test_wav_endpoint_400s_on_bad_chars(http_server):
    base, *_ = http_server
    status, _, _ = _get(base + "/wake_captures/has%20a%20space.wav")
    assert status == 400


def test_label_endpoint_round_trip(http_server):
    base, wc, a, *_ = http_server
    status, body = _post(
        f"{base}/wake_captures/{a}/label",
        {"label": "negative", "user": "ha_user"},
    )
    assert status == 200
    payload = json.loads(body)
    assert payload["ok"] is True
    assert payload["label"] == "negative"

    sidecar = next(p for p in wc.capture_dir.glob("*.wav.json") if a in p.name)
    with open(sidecar) as f:
        data = json.load(f)
    assert data["label"] == "negative"
    assert data["label_reason"] == "manual_ha_user"


def test_label_endpoint_rejects_unknown_label(http_server):
    base, _, a, *_ = http_server
    status, _ = _post(f"{base}/wake_captures/{a}/label", {"label": "questionable"})
    assert status == 400


def test_label_endpoint_400s_on_invalid_body(http_server):
    base, _, a, *_ = http_server
    req = urllib.request.Request(
        f"{base}/wake_captures/{a}/label",
        data=b"not json",
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            status = resp.status
    except urllib.error.HTTPError as e:
        status = e.code
    assert status == 400


def test_stats_endpoint(http_server):
    base, *_ = http_server
    status, body, _ = _get(base + "/wake_captures/stats")
    assert status == 200
    payload = json.loads(body)
    for key in (
        "total_24h", "positive_24h", "negative_24h", "ambiguous_24h",
        "gate2_reject_24h", "pending_triage_count",
    ):
        assert key in payload


def test_unknown_route_404s(http_server):
    base, *_ = http_server
    status, _, _ = _get(base + "/wake_captures/nope")
    assert status == 404
