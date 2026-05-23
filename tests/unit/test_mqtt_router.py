"""Stage F5 — mqtt_router matrix smoke tests."""

from __future__ import annotations

from linux_voice_assistant.mqtt_router import K15_SUBSCRIPTION_MATRIX, expand


def test_matrix_contains_required_topics():
    topics = {s.topic for s in K15_SUBSCRIPTION_MATRIX}
    assert "calisto/<room>/cancel" in topics
    assert "calisto/all/cancel" in topics
    assert "calisto/<room>/alarm/set" in topics
    assert "calisto/<room>/say" in topics
    assert "calisto/<room>/volume/set" in topics
    assert "calisto/<room>/admin/restart" in topics


def test_expand_substitutes_room():
    expanded = expand(K15_SUBSCRIPTION_MATRIX, "living_room")
    topics = {s.topic for s in expanded}
    assert "calisto/living_room/cancel" in topics
    assert "calisto/all/cancel" in topics  # unchanged
    assert all("<room>" not in s.topic for s in expanded)


def test_all_subscriptions_qos1():
    for s in K15_SUBSCRIPTION_MATRIX:
        assert s.qos == 1, f"{s.topic} has qos={s.qos}, expected 1"
