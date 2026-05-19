"""Stage F5 — MQTT subscription matrix (K.15) as code.

HABridge is the single transport. This module is the documented K.15
subscription matrix: every topic LVA subscribes to is listed here, with
the in-process handler that owns it. HABridge.start() consults this list
when subscribing.

Why a separate module: the matrix is the contract between LVA and HA.
Keeping it in code lets a CI lint (Stage F follow-up) compare it against
the K-spec markdown and fail on drift.

Adding a new subscription:
    1. Add the topic to `K15_SUBSCRIPTION_MATRIX` here.
    2. Add the handler dispatch in `HABridge._on_message`.
    3. Document it in `docs/v1-spec-K-mqtt.md`.

Topics are listed per-room (`<room>`) and global (`all`). The router
substitutes `<room>` at subscribe time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class SubscriptionSpec:
    """One row of the K.15 matrix."""

    topic: str  # may contain `<room>` placeholder
    qos: int
    handler: str  # human-readable: where the dispatch lives
    section: str  # K-spec section reference


# K.15 — the complete subscription matrix for LVA. Order is K-spec order
# for readability; HABridge subscribes all of them at connect time.
K15_SUBSCRIPTION_MATRIX: List[SubscriptionSpec] = [
    # K.3 + K.4 — cancel control plane.
    SubscriptionSpec("calisto/<room>/cancel", 1, "CancelCoordinator", "K.3"),
    SubscriptionSpec("calisto/all/cancel", 1, "CancelCoordinator", "K.4"),
    # K.8 — alarm control plane.
    SubscriptionSpec("calisto/<room>/alarm/set", 1, "AlarmController.set_alarm", "K.8"),
    SubscriptionSpec("calisto/all/alarm/set", 1, "AlarmController.set_alarm", "K.8"),
    SubscriptionSpec("calisto/<room>/alarm/stop", 1, "AlarmController.stop_alarm", "K.8"),
    # K.10 — ad-hoc TTS announce.
    SubscriptionSpec("calisto/<room>/say", 1, "HABridge._route_say", "K.10"),
    SubscriptionSpec("calisto/all/say", 1, "HABridge._route_say", "K.10"),
    # K.11 — volume bidirectional sync.
    SubscriptionSpec("calisto/<room>/volume/set", 1, "LedController.bar.apply", "K.11"),
    SubscriptionSpec("calisto/all/volume/set", 1, "LedController.bar.apply", "K.11"),
    # K.12 — LED control plane (back-compat).
    SubscriptionSpec("calisto/<room>/led/set", 1, "LedController.apply_legacy", "K.12"),
    SubscriptionSpec("calisto/all/led/set", 1, "LedController.apply_legacy", "K.12"),
    SubscriptionSpec("calisto/<room>/ring/set", 1, "LedController.ring", "K.12"),
    SubscriptionSpec("calisto/all/ring/set", 1, "LedController.ring", "K.12"),
    SubscriptionSpec("calisto/<room>/mute/set", 1, "LedController.set_private", "M3"),
    SubscriptionSpec("calisto/all/mute/set", 1, "LedController.set_private", "M3"),
    # K.13 — admin control plane.
    SubscriptionSpec("calisto/<room>/admin/restart", 1, "HABridge restart_hook", "K.13"),
]


def expand(matrix: List[SubscriptionSpec], room: str) -> List[SubscriptionSpec]:
    """Substitute `<room>` in every topic and return a fresh list."""
    return [
        SubscriptionSpec(spec.topic.replace("<room>", room), spec.qos, spec.handler, spec.section)
        for spec in matrix
    ]


__all__ = ["SubscriptionSpec", "K15_SUBSCRIPTION_MATRIX", "expand"]
