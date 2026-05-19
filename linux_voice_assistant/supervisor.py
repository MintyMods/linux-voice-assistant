"""Stage F3 — L1 in-process task supervisor (H2).

Wraps long-running asyncio tasks (heartbeat, state-watchdog, bridge-ping,
future per-channel mpv supervisors) in a respawn-on-failure loop. Per H2
"timings deliberately relaxed" — respawn delay defaults to 2s; a storm
counter (>5 respawns in 60s) escalates by exiting the LVA process so L2
(systemd) takes over.

Usage:

    supervisor = TaskSupervisor(loop)
    supervisor.add("heartbeat", lambda: heartbeat.publish_once())  # no
    # — but the typical use is to wrap a coroutine factory:
    supervisor.spawn("heartbeat", heartbeat.run_forever)

`run_forever` should be an `async def` callable that returns a coroutine.
When the coroutine exits or raises, the supervisor logs and respawns.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from typing import Callable, Coroutine, Deque, Dict

from .gen_check import gen_independent

_LOGGER = logging.getLogger(__name__)


_DEFAULT_RESPAWN_DELAY_S = 2.0
_STORM_WINDOW_S = 60.0
_STORM_BUDGET = 5


class _ChildState:
    __slots__ = ("name", "factory", "task", "respawns", "stop")

    def __init__(self, name: str, factory: Callable[[], Coroutine]) -> None:
        self.name = name
        self.factory = factory
        self.task: asyncio.Task | None = None
        self.respawns: Deque[float] = deque()
        self.stop: bool = False


class TaskSupervisor:
    """Spawns and respawns long-running tasks.

    Each supervised child is a `(name, async-callable)` pair. The
    supervisor calls the callable to get a fresh coroutine, schedules it
    on the loop, and reschedules a new one if the prior one exited or
    raised.

    Storm-detection: if a single child's respawns exceed `storm_budget`
    in `storm_window_s`, the supervisor raises a `SupervisorStorm`. Caller
    (typically `__main__`) treats this as fatal — exits with non-zero
    code so systemd (L3) takes over.
    """

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        *,
        respawn_delay_s: float = _DEFAULT_RESPAWN_DELAY_S,
        storm_window_s: float = _STORM_WINDOW_S,
        storm_budget: int = _STORM_BUDGET,
    ) -> None:
        self._loop = loop
        self._respawn_delay_s = respawn_delay_s
        self._storm_window_s = storm_window_s
        self._storm_budget = storm_budget
        self._children: Dict[str, _ChildState] = {}

    def spawn(self, name: str, factory: Callable[[], Coroutine]) -> None:
        """Register and immediately schedule `factory()` as a supervised task."""
        if name in self._children:
            raise ValueError(f"supervised child {name!r} already registered")
        child = _ChildState(name, factory)
        self._children[name] = child
        self._schedule(child)

    def stop(self, name: str) -> None:
        child = self._children.get(name)
        if child is None:
            return
        child.stop = True
        if child.task is not None and not child.task.done():
            child.task.cancel()

    def stop_all(self) -> None:
        for name in list(self._children.keys()):
            self.stop(name)

    def child_respawn_count(self, name: str) -> int:
        c = self._children.get(name)
        return len(c.respawns) if c is not None else 0

    # ---------------------------------------------------------------- impl

    def _schedule(self, child: _ChildState) -> None:
        try:
            coro = child.factory()
        except Exception:
            _LOGGER.exception("Supervisor: %s factory raised; respawn in %.1fs",
                              child.name, self._respawn_delay_s)
            self._loop.call_later(self._respawn_delay_s, self._on_failed, child)
            return
        if not asyncio.iscoroutine(coro):
            raise TypeError(
                f"supervised child {child.name!r} factory must return a coroutine"
            )
        child.task = self._loop.create_task(coro, name=f"supervised-{child.name}")
        child.task.add_done_callback(lambda t, c=child: self._on_done(c, t))

    @gen_independent
    def _on_done(self, child: _ChildState, task: asyncio.Task) -> None:
        if child.stop:
            _LOGGER.debug("Supervisor: %s stopped cleanly", child.name)
            return
        try:
            exc = task.exception()
        except asyncio.CancelledError:
            exc = None
        if exc is None:
            _LOGGER.info("Supervisor: %s exited cleanly; respawn in %.1fs",
                         child.name, self._respawn_delay_s)
        else:
            _LOGGER.warning("Supervisor: %s raised %s; respawn in %.1fs",
                            child.name, exc, self._respawn_delay_s)
        self._loop.call_later(self._respawn_delay_s, self._on_failed, child)

    @gen_independent
    def _on_failed(self, child: _ChildState) -> None:
        if child.stop:
            return
        now = time.monotonic()
        child.respawns.append(now)
        cutoff = now - self._storm_window_s
        while child.respawns and child.respawns[0] < cutoff:
            child.respawns.popleft()
        if len(child.respawns) > self._storm_budget:
            raise SupervisorStorm(child.name, len(child.respawns), self._storm_window_s)
        self._schedule(child)


class SupervisorStorm(RuntimeError):
    """Raised when a supervised child exceeds its respawn budget."""

    def __init__(self, name: str, respawns: int, window_s: float) -> None:
        super().__init__(
            f"supervised child {name!r} respawned {respawns} times in {window_s}s — "
            "exceeding L2 storm budget; raising to caller for L3 escalation"
        )
        self.child = name
        self.respawns = respawns
        self.window_s = window_s
