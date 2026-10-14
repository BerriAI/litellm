from datetime import UTC, datetime

from litellm.proxy.control_plane.audit.event import AuditEvent
from litellm.proxy.control_plane.audit.ocsf import (
    ChainCheck,
    AuditRecord,
    ChainState,
    Product,
    chain,
    section,
    to_ocsf,
    verify,
)

T0 = datetime(2026, 10, 13, 12, 0, tzinfo=UTC)
PRODUCT = Product(version="1.2.3")


def _event(**overrides: object) -> AuditEvent:
    base: dict[str, object] = {
        "at": T0,
        "actor": "alice",
        "actor_email": "alice@x.test",
        "dataplane_id": "eu",
        "method": "POST",
        "path": "/key/generate",
        "decision": "allowed",
        "status": 200,
        "credential": "service_token",
        "detail": None,
        "trace_id": "0af7651916cd43dd8448eb211c80319c",
        "span_id": "b7ad6b7169203331",
    }
    return AuditEvent(**{**base, **overrides})  # pyright: ignore[reportArgumentType]  # test literal


def test_an_allowed_write_is_an_api_activity_create_with_success_status() -> None:
    body = to_ocsf(_event(), PRODUCT, ChainState())
    assert (body["class_uid"], body["category_uid"], body["type_uid"], body["activity_id"]) == (6003, 6, 600301, 1)
    assert (body["status_id"], body["severity_id"], body["time"]) == (1, 1, int(T0.timestamp() * 1000))
    assert body["actor"] == {
        "user": {"uid": "alice", "email_addr": "alice@x.test"},
        "session": {"credential_uid": "service_token"},
        "authorizations": [{"decision": "allowed"}],
    }
    assert body["api"] == {
        "operation": "POST",
        "service": {"name": "eu", "uid": "dataplane"},
        "response": {"code": 200},
    }
    assert body["http_request"] == {"http_method": "POST", "url": {"path": "/key/generate"}}
    assert body["metadata"] == {
        "version": "1.5.0",
        "product": {"name": "LiteLLM Control Plane", "vendor_name": "LiteLLM", "version": "1.2.3"},
        "sequence": 1,
        "correlation_uid": "0af7651916cd43dd8448eb211c80319c",
    }


def test_denied_and_failed_calls_are_failures_with_medium_severity() -> None:
    denied = to_ocsf(_event(decision="denied", status=None, detail="no scope"), PRODUCT, ChainState())
    failed = to_ocsf(_event(status=400), PRODUCT, ChainState())
    unreachable = to_ocsf(_event(decision="unreachable", status=None, method="GET"), PRODUCT, ChainState())
    assert (denied["status_id"], denied["severity_id"], denied["status_detail"]) == (2, 3, "no scope")
    assert (failed["status_id"], failed["status_code"]) == (2, "400")
    assert (unreachable["status_id"], unreachable["activity_id"], unreachable["activity_name"]) == (2, 2, "Read")


def test_a_login_is_an_authentication_logon_event() -> None:
    body = to_ocsf(_event(kind="authentication", method="POST", path="/auth/login", status=None), PRODUCT, ChainState())
    assert (body["class_uid"], body["category_uid"], body["type_uid"], body["activity_name"]) == (
        3002,
        3,
        300201,
        "Logon",
    )
    assert body["user"] == {"uid": "alice"}


def test_records_chain_and_verify_detects_a_tampered_or_missing_link() -> None:
    state0 = ChainState()
    state1, first = chain(_event(), PRODUCT, state0)
    state2, second = chain(_event(method="GET", path="/key/list"), PRODUCT, state1)
    _, third = chain(_event(method="DELETE", path="/key/delete"), PRODUCT, state2)
    assert (first.sequence, second.sequence, third.sequence) == (1, 2, 3)
    assert section(first.body, "unmapped", "litellm")["prev_hash"] is None
    assert section(second.body, "unmapped", "litellm")["prev_hash"] == first.hash
    assert verify((first, second, third)) == ChainCheck(valid=True, gaps=0)
    assert verify((first, third)) == ChainCheck(valid=True, gaps=1), "a missing record is a gap, not tampering"
    assert verify((second, third)) == ChainCheck(valid=True, gaps=0), "a page may start after the first record"
    tampered_body = {**second.body, "actor": {"user": {"uid": "mallory"}}}
    tampered = AuditRecord(second.event, second.sequence, second.hash, tampered_body)
    assert verify((first, tampered, third)).valid is False, "an edited record no longer matches its hash"
    _, forged = chain(_event(method="GET", path="/key/list"), PRODUCT, ChainState(1, "sha256:" + "0" * 64))
    assert verify((first, forged)).valid is False, "a re-hashed record that points at the wrong predecessor"
    assert verify((forged,)).valid is True, "alone it is consistent, only the link exposes it"
    assert verify(()) == ChainCheck(valid=True, gaps=0)
