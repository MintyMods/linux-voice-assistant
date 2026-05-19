"""Stage E.2 G2 — ducking envelope tests.

The audio backend (`mpv`) is stubbed in tests/conftest.py, so we work
against `LibMpvPlayer` directly with a hand-rolled fake `MPV` recorder.
The fake records every volume assignment in order so a test can assert
the shape of a ramp (monotonic, starts at 1.0, ends at floor, etc.) without
caring about exact float values or wall-clock timing.

Why this lives at the libmpv layer rather than MpvMediaPlayer:
the higher-level wrapper just forwards `duck()` / `unduck()` /
`configure_duck_envelope()` — the *envelope shape* logic is entirely in
`LibMpvPlayer._ramp_run`, so testing there is closer to the bug surface.
"""

from __future__ import annotations

import sys
import time
import types
from typing import List

import pytest


class _FakeMpv:
    """Records every option set + every volume write."""

    def __init__(self, **kwargs):
        self.options: dict = {}
        self.volume_history: List[float] = []
        self.pause = False
        self._event_callbacks: dict = {}

    def __setitem__(self, key, value):
        self.options[key] = value

    def __setattr__(self, name, value):
        # Real mpv property assignment is via attribute (e.g. `mpv.volume = 50`),
        # but pause/volume are dispatched through __setattr__. Record only the
        # ones the tests assert on.
        if name == "volume":
            self.volume_history.append(float(value))
        super().__setattr__(name, value)

    def event_callback(self, _name):
        def deco(fn):
            self._event_callbacks[_name] = fn
            return fn

        return deco

    def play(self, _url):  # pragma: no cover - not exercised here
        pass

    def stop(self):  # pragma: no cover
        pass


@pytest.fixture
def libmpv_module(monkeypatch):
    """Attach a fake `MPV` class onto whichever `mpv` stub is in sys.modules.

    The top-level `tests/conftest.py` installs a bare `mpv` module before any
    test runs so the package imports succeed. We can't replace that module
    wholesale here because `linux_voice_assistant.player.libmpv` has already
    bound `mpv` as a module-level name via `import mpv` — replacing
    `sys.modules["mpv"]` does nothing for that binding. Instead we monkey-
    patch the missing `MPV` attribute onto the already-imported stub.
    """
    import mpv as _mpv_module  # the bare stub from conftest

    monkeypatch.setattr(_mpv_module, "MPV", _FakeMpv, raising=False)
    from linux_voice_assistant.player import libmpv

    return libmpv


def _wait_for_ramp(player, timeout_s: float = 2.0):
    """Block briefly while the duck/unduck thread runs to completion."""
    end = time.monotonic() + timeout_s
    while player._duck_thread is not None and player._duck_thread.is_alive():
        if time.monotonic() > end:
            raise AssertionError("ramp did not finish within timeout")
        time.sleep(0.01)


class TestDuckEnvelope:
    def test_duck_ramps_to_target_then_unduck_ramps_back(self, libmpv_module):
        p = libmpv_module.LibMpvPlayer(role="media")
        p.set_volume(100.0)
        p.configure_duck_envelope(attack_ms=80, release_ms=80)
        baseline_history_len = len(p._mpv.volume_history)

        p.duck(0.3)
        _wait_for_ramp(p)
        ramp_down = p._mpv.volume_history[baseline_history_len:]
        assert len(ramp_down) >= 3, f"expected multiple ramp steps, got {len(ramp_down)}"
        assert ramp_down[0] > ramp_down[-1], "ramp should descend"
        assert ramp_down[-1] == pytest.approx(30.0, abs=0.5), f"final volume should hit floor (got {ramp_down[-1]})"

        before_unduck = len(p._mpv.volume_history)
        p.unduck()
        _wait_for_ramp(p)
        ramp_up = p._mpv.volume_history[before_unduck:]
        assert ramp_up[0] < ramp_up[-1], "ramp should ascend"
        assert ramp_up[-1] == pytest.approx(100.0, abs=0.5), f"final volume should restore to 100 (got {ramp_up[-1]})"

    def test_duck_then_unduck_mid_ramp_preempts(self, libmpv_module):
        """A new ramp must cancel the previous thread, not stack."""
        p = libmpv_module.LibMpvPlayer(role="media")
        p.set_volume(100.0)
        p.configure_duck_envelope(attack_ms=400, release_ms=400)

        p.duck(0.3)
        # Pre-empt almost immediately.
        time.sleep(0.05)
        p.unduck()
        _wait_for_ramp(p)

        final = p._mpv.volume_history[-1]
        assert final == pytest.approx(100.0, abs=0.5), f"unduck must win after pre-empt (got {final})"
        # And there should be no zombie ramp thread.
        assert p._duck_thread is None or not p._duck_thread.is_alive()

    def test_zero_duration_snaps_immediately(self, libmpv_module):
        p = libmpv_module.LibMpvPlayer(role="media")
        p.set_volume(100.0)
        p.configure_duck_envelope(attack_ms=0, release_ms=0)

        p.duck(0.3)
        # No ramp thread should be started for duration=0.
        assert p._duck_thread is None or not p._duck_thread.is_alive()
        assert p._mpv.volume_history[-1] == pytest.approx(30.0, abs=0.01)

    def test_configure_duck_envelope_clamps_inputs(self, libmpv_module):
        p = libmpv_module.LibMpvPlayer(role="media")
        p.configure_duck_envelope(floor_pct=999, attack_ms=-100, release_ms=10_000)
        assert p._duck_floor_pct == 100
        assert p._duck_attack_ms == 0
        assert p._duck_release_ms == 10_000


class TestSourceResilience:
    def test_media_role_sets_reconnect_options(self, libmpv_module):
        p = libmpv_module.LibMpvPlayer(role="media")
        assert "stream-lavf-o" in p._mpv.options
        lavf_o = p._mpv.options["stream-lavf-o"]
        assert "reconnect=1" in lavf_o
        assert "reconnect_streamed=1" in lavf_o
        assert "reconnect_delay_max=30" in lavf_o
        assert p._mpv.options.get("cache") == "yes"
        assert p._mpv.options.get("cache-secs") == 10
        assert p._mpv.options.get("network-timeout") == 10

    @pytest.mark.parametrize("role", ["tts", "chime", "alarm"])
    def test_non_media_roles_skip_reconnect_options(self, libmpv_module, role):
        """Short-clip channels must not pay the reconnect/cache latency cost."""
        p = libmpv_module.LibMpvPlayer(role=role)
        assert "stream-lavf-o" not in p._mpv.options
        assert "cache" not in p._mpv.options
        assert "cache-secs" not in p._mpv.options
        assert "network-timeout" not in p._mpv.options
