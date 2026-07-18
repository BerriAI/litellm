"""Unit tests for multi-pod model peer-reload readiness (no live proxy)."""

from __future__ import annotations

from proxy_client import peer_reload_ready


def test_peer_reload_zero_ready_on_first_list() -> None:
    ready, seen = peer_reload_ready(
        listed=True,
        first_seen_at=None,
        now=100.0,
        peer_reload_seconds=0.0,
    )
    assert ready is True
    assert seen == 100.0


def test_peer_reload_not_ready_before_grace() -> None:
    ready, seen = peer_reload_ready(
        listed=True,
        first_seen_at=10.0,
        now=20.0,
        peer_reload_seconds=35.0,
    )
    assert ready is False
    assert seen == 10.0


def test_peer_reload_ready_after_grace() -> None:
    ready, seen = peer_reload_ready(
        listed=True,
        first_seen_at=10.0,
        now=45.0,
        peer_reload_seconds=35.0,
    )
    assert ready is True
    assert seen == 10.0


def test_peer_reload_miss_resets_first_seen() -> None:
    ready, seen = peer_reload_ready(
        listed=False,
        first_seen_at=10.0,
        now=40.0,
        peer_reload_seconds=35.0,
    )
    assert ready is False
    assert seen is None


def test_peer_reload_records_first_seen_on_first_list() -> None:
    ready, seen = peer_reload_ready(
        listed=True,
        first_seen_at=None,
        now=50.0,
        peer_reload_seconds=35.0,
    )
    assert ready is False
    assert seen == 50.0


def test_peer_reload_after_reset_must_complete_full_grace() -> None:
    _, seen = peer_reload_ready(
        listed=True,
        first_seen_at=None,
        now=100.0,
        peer_reload_seconds=35.0,
    )
    ready, seen2 = peer_reload_ready(
        listed=True,
        first_seen_at=seen,
        now=134.0,
        peer_reload_seconds=35.0,
    )
    assert ready is False
    assert seen2 == 100.0
    ready2, _ = peer_reload_ready(
        listed=True,
        first_seen_at=seen,
        now=135.0,
        peer_reload_seconds=35.0,
    )
    assert ready2 is True
