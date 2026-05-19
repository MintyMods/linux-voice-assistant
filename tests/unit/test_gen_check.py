"""Stage F4 — gen_checked decorator unit tests."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from linux_voice_assistant.gen_check import (
    GEN_CHECKED_ATTR,
    GEN_INDEPENDENT_ATTR,
    gen_checked,
    gen_independent,
)


class _Handler:
    def __init__(self, gen=5):
        self.session = MagicMock(generation=gen)
        self.calls = []

    @gen_checked
    def sync_call(self, x):
        self.calls.append(x)
        return x

    @gen_checked
    async def async_call(self, x):
        self.calls.append(x)
        return x


def test_sync_passes_when_gen_matches():
    h = _Handler(gen=5)
    h.sync_call("a", generation=5)
    assert h.calls == ["a"]


def test_sync_drops_when_gen_stale():
    h = _Handler(gen=7)
    result = h.sync_call("a", generation=3)
    assert result is None
    assert h.calls == []


def test_async_drops_when_gen_stale():
    h = _Handler(gen=10)

    async def run():
        return await h.async_call("a", generation=2)

    result = asyncio.run(run())
    assert result is None
    assert h.calls == []


def test_bound_generation_is_consulted_when_kwarg_missing():
    h = _Handler(gen=4)
    h._bound_generation = 4
    h.sync_call("a")
    assert h.calls == ["a"]


def test_bound_generation_stale_drops():
    h = _Handler(gen=99)
    h._bound_generation = 1
    h.sync_call("a")
    assert h.calls == []


def test_decorator_marks_attribute():
    h = _Handler()
    assert getattr(h.sync_call, GEN_CHECKED_ATTR, False) is True


def test_gen_independent_marks_attribute():
    @gen_independent
    def f():
        pass

    assert getattr(f, GEN_INDEPENDENT_ATTR, False) is True


def test_no_gen_info_runs_method():
    """If neither generation= nor _bound_generation is provided, the
    wrapper runs the method (developer's responsibility to fix)."""
    h = _Handler(gen=1)
    # No generation kwarg, no _bound_generation attr.
    h.sync_call("a")
    assert h.calls == ["a"]
