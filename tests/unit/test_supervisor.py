"""Stage F3 — TaskSupervisor unit tests."""

from __future__ import annotations

import asyncio

import pytest

from linux_voice_assistant.supervisor import SupervisorStorm, TaskSupervisor


@pytest.mark.asyncio
async def test_respawns_after_clean_exit():
    loop = asyncio.get_event_loop()
    counter = {"runs": 0}

    async def worker():
        counter["runs"] += 1

    sup = TaskSupervisor(loop, respawn_delay_s=0.01)
    sup.spawn("worker", worker)
    await asyncio.sleep(0.1)
    sup.stop_all()
    await asyncio.sleep(0.02)
    assert counter["runs"] >= 2


@pytest.mark.asyncio
async def test_respawns_after_exception():
    loop = asyncio.get_event_loop()
    counter = {"runs": 0}

    async def worker():
        counter["runs"] += 1
        raise RuntimeError("boom")

    sup = TaskSupervisor(loop, respawn_delay_s=0.01)
    sup.spawn("worker", worker)
    await asyncio.sleep(0.1)
    sup.stop_all()
    await asyncio.sleep(0.02)
    assert counter["runs"] >= 2


@pytest.mark.asyncio
async def test_storm_raises_when_budget_exceeded():
    loop = asyncio.get_event_loop()
    storm_raised = {"event": None}

    async def worker():
        raise RuntimeError("crash")

    sup = TaskSupervisor(
        loop, respawn_delay_s=0.001, storm_window_s=10.0, storm_budget=2,
    )

    # The supervisor raises SupervisorStorm synchronously inside the
    # call_later callback that does respawn bookkeeping. asyncio's default
    # exception handler is the one to catch it.
    def _capture_exc(loop, context):
        exc = context.get("exception")
        if isinstance(exc, SupervisorStorm):
            storm_raised["event"] = exc

    loop.set_exception_handler(_capture_exc)
    sup.spawn("worker", worker)
    await asyncio.sleep(0.2)
    sup.stop_all()
    loop.set_exception_handler(None)
    assert isinstance(storm_raised["event"], SupervisorStorm)
    assert storm_raised["event"].child == "worker"


@pytest.mark.asyncio
async def test_stop_prevents_respawn():
    loop = asyncio.get_event_loop()
    counter = {"runs": 0}

    async def worker():
        counter["runs"] += 1
        await asyncio.sleep(0.01)

    sup = TaskSupervisor(loop, respawn_delay_s=0.01)
    sup.spawn("worker", worker)
    await asyncio.sleep(0.02)
    sup.stop("worker")
    await asyncio.sleep(0.1)
    # After stop(), the worker should not be respawned. Allow one extra
    # run for the in-flight callback to drain.
    captured = counter["runs"]
    await asyncio.sleep(0.1)
    assert counter["runs"] == captured
