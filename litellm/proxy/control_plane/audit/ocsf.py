"""Audit events as OCSF 1.5 objects: API Activity (6003) for calls through the control plane, Authentication (3002)
for logins. Each record carries a per-process sequence and a sha256 chain over the previous record so a dropped or
edited line is detectable in whatever sink holds it."""

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from functools import reduce
from typing import Final, Literal, TypeAlias

from pydantic import TypeAdapter, ValidationError

from litellm.proxy.control_plane.audit.log import AuditEvent

OCSF_VERSION: Final = "1.5.0"
API_ACTIVITY_CLASS: Final = 6003
AUTHENTICATION_CLASS: Final = 3002
APPLICATION_ACTIVITY_CATEGORY: Final = 6
IAM_CATEGORY: Final = 3
HASH_PREFIX: Final = "sha256:"
VENDOR: Final = "LiteLLM"
PRODUCT: Final = "LiteLLM Control Plane"
Json: TypeAlias = Mapping[str, object]
_JSON: Final = TypeAdapter(Mapping[str, object])
_ACTIVITY_BY_METHOD: Final[Mapping[str, tuple[int, str]]] = {
    "POST": (1, "Create"),
    "GET": (2, "Read"),
    "PUT": (3, "Update"),
    "PATCH": (3, "Update"),
    "DELETE": (4, "Delete"),
}
_LOGON: Final = (1, "Logon")
_UNKNOWN: Final = (0, "Unknown")


@dataclass(frozen=True, slots=True)
class ChainState:
    sequence: int = 0
    prev_hash: str | None = None


@dataclass(frozen=True, slots=True)
class AuditRecord:
    event: AuditEvent
    sequence: int
    hash: str
    body: Json


@dataclass(frozen=True, slots=True)
class Product:
    version: str


Outcome: TypeAlias = Literal["success", "failure"]


def _outcome(event: AuditEvent) -> Outcome:
    if event.decision != "allowed":
        return "failure"
    return "failure" if event.status is not None and event.status >= 400 else "success"


def _status(event: AuditEvent) -> Json:
    outcome: Final = _outcome(event)
    return {
        "status_id": 1 if outcome == "success" else 2,
        "status": "Success" if outcome == "success" else "Failure",
        **({} if event.status is None else {"status_code": str(event.status)}),
        **({} if event.detail is None else {"status_detail": event.detail}),
    }


def _severity(event: AuditEvent) -> Json:
    if event.decision == "allowed":
        return {"severity_id": 1, "severity": "Informational"}
    return {"severity_id": 3, "severity": "Medium"}


def _actor(event: AuditEvent) -> Json:
    user: Final = {"uid": event.actor, **({} if event.actor_email is None else {"email_addr": event.actor_email})}
    return {
        "user": user,
        **({} if event.credential is None else {"session": {"credential_uid": event.credential}}),
        "authorizations": [{"decision": event.decision}],
    }


def _metadata(event: AuditEvent, product: Product, state: ChainState) -> Json:
    return {
        "version": OCSF_VERSION,
        "product": {"name": PRODUCT, "vendor_name": VENDOR, "version": product.version},
        "sequence": state.sequence + 1,
        **({} if event.trace_id is None else {"correlation_uid": event.trace_id}),
    }


def _litellm(event: AuditEvent, state: ChainState) -> Json:
    return {
        "decision": event.decision,
        "sequence": state.sequence + 1,
        "prev_hash": state.prev_hash,
        **({} if event.trace_id is None else {"trace_id": event.trace_id}),
        **({} if event.span_id is None else {"span_id": event.span_id}),
    }


def _api_activity(event: AuditEvent, product: Product, state: ChainState) -> Json:
    activity_id, activity_name = _ACTIVITY_BY_METHOD.get(event.method, _UNKNOWN)
    return {
        "class_uid": API_ACTIVITY_CLASS,
        "class_name": "API Activity",
        "category_uid": APPLICATION_ACTIVITY_CATEGORY,
        "category_name": "Application Activity",
        "type_uid": API_ACTIVITY_CLASS * 100 + activity_id,
        "activity_id": activity_id,
        "activity_name": activity_name,
        "time": int(event.at.timestamp() * 1000),
        **_severity(event),
        **_status(event),
        "metadata": _metadata(event, product, state),
        "actor": _actor(event),
        "api": {
            "operation": event.method,
            "service": {"name": event.dataplane_id, "uid": "dataplane"},
            **({} if event.status is None else {"response": {"code": event.status}}),
        },
        "http_request": {"http_method": event.method, "url": {"path": event.path}},
        **({} if event.status is None else {"http_response": {"code": event.status}}),
        "unmapped": {"litellm": {"dataplane_id": event.dataplane_id, **_litellm(event, state)}},
    }


def _authentication(event: AuditEvent, product: Product, state: ChainState) -> Json:
    activity_id, activity_name = _LOGON
    return {
        "class_uid": AUTHENTICATION_CLASS,
        "class_name": "Authentication",
        "category_uid": IAM_CATEGORY,
        "category_name": "Identity & Access Management",
        "type_uid": AUTHENTICATION_CLASS * 100 + activity_id,
        "activity_id": activity_id,
        "activity_name": activity_name,
        "time": int(event.at.timestamp() * 1000),
        **_severity(event),
        **_status(event),
        "metadata": _metadata(event, product, state),
        "actor": _actor(event),
        "user": {"uid": event.actor},
        "service": {"name": PRODUCT},
        "http_request": {"http_method": event.method, "url": {"path": event.path}},
        "unmapped": {"litellm": _litellm(event, state)},
    }


def to_ocsf(event: AuditEvent, product: Product, state: ChainState) -> Json:
    match event.kind:
        case "api_activity":
            return _api_activity(event, product, state)
        case "authentication":
            return _authentication(event, product, state)


def canonical(body: Json) -> bytes:
    return json.dumps(body, sort_keys=True, separators=(",", ":"), default=str).encode()


def chain(event: AuditEvent, product: Product, state: ChainState) -> tuple[ChainState, AuditRecord]:
    """The next record and the state after it. The hash covers the whole body including the previous hash."""
    body: Final = to_ocsf(event, product, state)
    digest: Final = HASH_PREFIX + hashlib.sha256(canonical(body)).hexdigest()
    litellm_block: Final = dict(_unmapped_litellm(body))
    sealed: Final = {**body, "unmapped": {"litellm": {**litellm_block, "hash": digest}}}
    record: Final = AuditRecord(event=event, sequence=state.sequence + 1, hash=digest, body=sealed)
    return ChainState(sequence=state.sequence + 1, prev_hash=digest), record


def _child(body: Json, key: str) -> Json:
    try:
        return _JSON.validate_python(body.get(key))
    except ValidationError:
        return {}


def section(body: Json, *keys: str) -> Json:
    """The nested object at ``keys`` or an empty mapping when any step is missing or not an object."""
    return reduce(_child, keys, body)


def _unmapped_litellm(body: Json) -> Json:
    return section(body, "unmapped", "litellm")


@dataclass(frozen=True, slots=True)
class ChainCheck:
    """``valid`` is False only on tampering. ``gaps`` counts places where consecutive records are not consecutive
    sequences, which is expected for a filtered or limited page and a dropped record for a complete export."""

    valid: bool
    gaps: int


def verify(records: tuple[AuditRecord, ...]) -> ChainCheck:
    """Every record's hash must match its body, the first record of a process names no predecessor, and records
    with adjacent sequences must link. Records separated by a gap are checked on their own only."""
    pairs: Final = tuple(zip(records, records[1:], strict=False))
    adjacent: Final = tuple(pair for pair in pairs if pair[1].sequence == pair[0].sequence + 1)
    return ChainCheck(
        valid=all(_sealed(record) for record in records) and all(_linked(prev, record) for prev, record in adjacent),
        gaps=len(pairs) - len(adjacent),
    )


def _sealed(record: AuditRecord) -> bool:
    litellm_block: Final = _unmapped_litellm(record.body)
    unsealed: Final = {
        **record.body,
        "unmapped": {"litellm": {key: value for key, value in litellm_block.items() if key != "hash"}},
    }
    hashed: Final = litellm_block.get("hash") == HASH_PREFIX + hashlib.sha256(canonical(unsealed)).hexdigest()
    return hashed and (litellm_block.get("prev_hash") is None) == (record.sequence == 1)


def _linked(prev: AuditRecord, record: AuditRecord) -> bool:
    return _unmapped_litellm(record.body).get("prev_hash") == prev.hash
