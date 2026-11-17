import io
import json
import logging
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from itertools import count
from typing import Final

import pytest

from litellm.proxy.control_plane.audit.event import redact_path as redact_event_path
from litellm.proxy.control_plane.audit.log import build_audit_log, redact_path
from litellm.proxy.control_plane.audit.ocsf import verify

T0: Final = datetime(2026, 10, 13, 12, 0, tzinfo=UTC)


def _ticking() -> Callable[[], datetime]:
    ticks: Final = count()
    return lambda: T0 + timedelta(seconds=next(ticks))


def test_recent_is_newest_first_filtered_and_limited() -> None:
    audit: Final = build_audit_log(capacity=10, now=_ticking())
    audit.record("alice", None, "eu", "GET", "/key/list", "allowed", 200, "service_token")
    audit.record("bob", "bob@x.test", "us", "POST", "/key/generate", "allowed", 200, "oidc_exchange")
    audit.record("alice", None, "us", "DELETE", "/key/delete", "denied", detail="nope")
    assert [(e.actor, e.path) for e in audit.recent(10)] == [
        ("alice", "/key/delete"),
        ("bob", "/key/generate"),
        ("alice", "/key/list"),
    ]
    assert [e.path for e in audit.recent(10, dataplane_ids=frozenset({"us"}), actor="alice")] == ["/key/delete"]
    assert [e.path for e in audit.recent(1)] == ["/key/delete"]
    assert audit.recent(10)[0].at == T0 + timedelta(seconds=2)


def test_the_ring_buffer_drops_the_oldest_event_at_capacity() -> None:
    audit: Final = build_audit_log(capacity=2, now=_ticking())
    for path in ("/a", "/b", "/c"):
        audit.record("alice", None, "eu", "GET", path, "allowed", 200)
    assert [e.path for e in audit.recent(10)] == ["/c", "/b"]


def test_unauthenticated_failures_do_not_evict_authenticated_history() -> None:
    audit: Final = build_audit_log(capacity=2, now=_ticking())
    audit.record("alice", None, "eu", "GET", "/one", "allowed", 200)
    audit.record("alice", None, "eu", "GET", "/two", "allowed", 200)
    for index in range(2):
        audit.record_login(f"bad-{index}", "denied")
    assert [event.path for event in audit.recent(10)] == ["/auth/login", "/auth/login", "/two", "/one"]


def test_free_text_fields_are_safe_to_encode() -> None:
    audit: Final = build_audit_log(capacity=2, now=_ticking())
    event: Final = audit.record("\ud800", "\ud800", "eu", "GET", "/\ud800", "denied", detail="\ud800")
    assert (event.actor, event.actor_email, event.path, event.detail) == (
        "\\ud800",
        "\\ud800",
        "/\\ud800",
        "\\ud800",
    )


def test_caller_controlled_audit_fields_are_bounded_before_redaction() -> None:
    audit: Final = build_audit_log(capacity=2, now=_ticking())
    path: Final = "/" + "x" * 32_768

    event: Final = audit.record("a" * 32_768, "e" * 32_768, "d" * 32_768, "GET", path, "denied", detail="z" * 32_768)

    assert all(
        len(value) <= 2048
        for value in (event.actor, event.actor_email, event.dataplane_id, event.path, event.detail)
        if value is not None
    )
    malformed_path: Final = "/" + "-----BEGIN PRIVATE KEY-----" * 256
    bounded: Final = audit.record("alice", None, "eu", "GET", malformed_path, "denied")
    assert len(bounded.path) <= 2048


def test_unknown_secret_looking_dataplane_ids_are_redacted() -> None:
    audit: Final = build_audit_log(known_dataplane_ids=frozenset({"registered"}))
    unknown_id: Final = "sk-unknown-dataplane"

    event: Final = audit.record("alice", None, unknown_id, "GET", "/key/list", "allowed", 200)

    assert event.dataplane_id != unknown_id
    assert "sk-" not in event.dataplane_id


def test_dropped_unauthenticated_records_use_an_injected_info_logger() -> None:
    stream: Final = io.StringIO()
    audit: Final = build_audit_log(capacity=2, now=_ticking(), stream=stream)
    proxy_logger: Final = logging.getLogger("LiteLLM Proxy")
    previous_level: Final = proxy_logger.level
    proxy_logger.setLevel(logging.WARNING)
    try:
        audit.record_login("alice", "denied")
    finally:
        proxy_logger.setLevel(previous_level)
    assert "control_plane.audit" in stream.getvalue()


def test_failed_login_redacts_before_sealing_and_logging() -> None:
    stream: Final = io.StringIO()
    audit: Final = build_audit_log(capacity=2, now=_ticking(), stream=stream)
    username: Final = "sk-" + "A" * 24

    audit.record_login(username, "denied")

    (record,) = audit.recent_records(1)
    emitted_body: Final = stream.getvalue().split("control_plane.audit ", 1)[1].rstrip("\n")
    stored_body: Final = json.dumps(record.body, default=str, separators=(",", ":"))
    assert emitted_body == stored_body
    assert username not in emitted_body
    assert record.event.actor != username
    assert verify((record,)).valid


def test_virtual_keys_in_paths_are_redacted_before_they_are_stored_or_logged() -> None:
    audit: Final = build_audit_log(capacity=10, now=_ticking())
    event: Final = audit.record(
        "alice", None, "eu", "POST", "/key/sk-live-123/reset_spend", "denied", detail="no scope"
    )
    assert event.path == "/key/sk-[redacted]/reset_spend"
    assert audit.recent(10)[0].path == "/key/sk-[redacted]/reset_spend"
    assert audit.record("alice", None, "eu", "GET", "/key/list", "allowed", 200).path == "/key/list"


def test_recent_with_an_empty_dataplane_set_shows_nothing_and_none_shows_everything() -> None:
    audit: Final = build_audit_log(capacity=10, now=_ticking())
    audit.record("alice", None, "eu", "GET", "/key/list", "allowed", 200)
    audit.record("alice", None, "us", "GET", "/key/list", "allowed", 200)
    assert audit.recent(10, dataplane_ids=frozenset()) == ()
    assert [e.dataplane_id for e in audit.recent(10, dataplane_ids=None)] == ["us", "eu"]
    assert [e.dataplane_id for e in audit.recent(10, dataplane_ids=frozenset({"eu", "us"}))] == ["us", "eu"]


@pytest.mark.parametrize(
    "path",
    [
        "/key/SK-live-123/reset_spend",
        "/key/sk%2Dlive-123/reset_spend",
        "/key/sk%252Dlive-123/reset_spend",
        "/key/%2573k-live-123/team/secret/reset_spend",
    ],
)
def test_encoded_virtual_keys_are_redacted_after_repeated_decoding(path: str) -> None:
    assert redact_path(path) == "/key/sk-[redacted]/reset_spend"
    assert redact_event_path(path) == "/key/sk-[redacted]/reset_spend"


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("/key/sk-live-123/team/secret/reset_spend", "/key/sk-[redacted]/reset_spend"),
        ("/key/sk-live-123/team/secret", "/key/sk-[redacted]"),
    ],
)
def test_virtual_key_paths_redact_all_segments_before_a_known_final_action(path: str, expected: str) -> None:
    assert redact_path(path) == expected
