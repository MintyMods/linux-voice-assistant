"""Stage F4 — `@gen_checked` decorator + `@gen_independent` allowlist marker.

Per H3 every async callback with a user-observable side-effect MUST consult
the DeviceSession generation counter before acting:

    @gen_checked
    async def _on_bridge_reply(self, captured_gen, reply):
        ...  # only runs if captured_gen == ds.generation

For methods bound to a DeviceSession instance, the decorator reads
`self.session.generation` (or `self.device_session.generation`, or
`self.state.device_session.generation` — first match wins). The captured
generation can be passed explicitly via the `generation=` kwarg OR carried
via `self._bound_generation` set at session start.

For functions that don't fit the bound-method pattern, pass `generation=`
explicitly and the decorator does the right thing.

`@gen_independent` is a no-op marker used by the static auditor
(`script/lint_gen_checks.py`) to signal "this callback is gen-agnostic
on purpose" (e.g. heartbeat publisher, music player callbacks).
"""

from __future__ import annotations

import functools
import inspect
import logging
from typing import Any, Awaitable, Callable, Optional, TypeVar

_LOGGER = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

GEN_CHECKED_ATTR = "__gen_checked__"
GEN_INDEPENDENT_ATTR = "__gen_independent__"


def _resolve_current_generation(self_obj: Any) -> Optional[int]:
    """Walk the common attribute paths to find DeviceSession.generation."""
    for path in ("session", "device_session"):
        ds = getattr(self_obj, path, None)
        if ds is not None:
            gen = getattr(ds, "generation", None)
            if isinstance(gen, int):
                return gen
    state = getattr(self_obj, "state", None)
    if state is not None:
        ds = getattr(state, "device_session", None)
        if ds is not None:
            gen = getattr(ds, "generation", None)
            if isinstance(gen, int):
                return gen
    return None


def gen_checked(method: F) -> F:
    """Wrap a callback to no-op when its captured generation is stale.

    Works on both sync and async callables. The captured generation is
    read in priority order:

        1. `generation=` kwarg passed by the caller.
        2. `self._bound_generation` set at session start.

    If neither is present the wrapper logs once and runs the inner method
    (no gen-check possible — caller's bug to fix).
    """
    is_coro = inspect.iscoroutinefunction(method)

    if is_coro:
        @functools.wraps(method)
        async def async_wrapper(self_obj: Any, *args: Any, **kwargs: Any) -> Any:
            captured = kwargs.pop("generation", None)
            if captured is None:
                captured = getattr(self_obj, "_bound_generation", None)
            current = _resolve_current_generation(self_obj)
            if captured is not None and current is not None and captured != current:
                _LOGGER.debug(
                    "gen_drop handler=%s stale=%d current=%d",
                    method.__name__, captured, current,
                )
                return None
            return await method(self_obj, *args, **kwargs)
        setattr(async_wrapper, GEN_CHECKED_ATTR, True)
        return async_wrapper  # type: ignore[return-value]

    @functools.wraps(method)
    def sync_wrapper(self_obj: Any, *args: Any, **kwargs: Any) -> Any:
        captured = kwargs.pop("generation", None)
        if captured is None:
            captured = getattr(self_obj, "_bound_generation", None)
        current = _resolve_current_generation(self_obj)
        if captured is not None and current is not None and captured != current:
            _LOGGER.debug(
                "gen_drop handler=%s stale=%d current=%d",
                method.__name__, captured, current,
            )
            return None
        return method(self_obj, *args, **kwargs)
    setattr(sync_wrapper, GEN_CHECKED_ATTR, True)
    return sync_wrapper  # type: ignore[return-value]


def gen_independent(method: F) -> F:
    """No-op marker — declares the callback is intentionally gen-agnostic.

    Used by the static auditor to allow handlers like heartbeat publish
    and music-player callbacks (per H3 §"gen-independent surfaces").
    """
    setattr(method, GEN_INDEPENDENT_ATTR, True)
    return method


__all__ = [
    "gen_checked",
    "gen_independent",
    "GEN_CHECKED_ATTR",
    "GEN_INDEPENDENT_ATTR",
]
