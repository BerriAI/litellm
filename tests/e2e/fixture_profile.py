from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal, DecimalException
from typing import Final, Literal
from urllib.parse import parse_qsl, urlsplit

from pydantic import BaseModel, JsonValue, TypeAdapter

type MatchProfile = Literal["legacy", "stateless_v1"]
type ExactJson = dict[str, ExactJson] | list[ExactJson] | str | bool | Decimal | int | None

SEMANTIC_HEADERS: Final = frozenset({"content-type", "accept", "anthropic-version", "anthropic-beta", "openai-beta"})
AUTH_HEADERS: Final = frozenset({"authorization", "x-api-key"})
EXCLUDED_HEADERS: Final = frozenset(
    {
        "host",
        "content-length",
        "transfer-encoding",
        "connection",
        "accept-encoding",
        "user-agent",
        "traceparent",
        "tracestate",
        "x-request-id",
        "x-client-request-id",
        "cookie",
    }
)
CREDENTIAL_QUERY: Final = frozenset(
    {
        "api_key",
        "api-key",
        "apikey",
        "key",
        "token",
        "access_token",
        "signature",
        "password",
        "secret",
        "credentials",
        "authorization",
        "sig",
        "client_secret",
        "aws_access_key_id",
        "aws_secret_access_key",
        "aws_session_token",
    }
)
JSON_VALUE: Final[TypeAdapter[ExactJson]] = TypeAdapter(ExactJson)


def match_profile() -> MatchProfile:
    raw: Final = os.environ.get("E2E_REPLAY_MATCH_PROFILE", "legacy")
    if raw in ("legacy", "stateless_v1"):
        return raw
    raise ValueError("E2E_REPLAY_MATCH_PROFILE must be legacy or stateless_v1")


class StrictIdentity(BaseModel):
    upstream: str
    mount: str
    query: tuple[tuple[str, str], ...]
    headers: dict[str, str]
    auth: dict[str, str]
    body_present: bool
    body: JsonValue


@dataclass(frozen=True, slots=True)
class IneligibleRequest:
    reason: str


def _unique_object(pairs: list[tuple[str, ExactJson]]) -> dict[str, ExactJson]:
    if len({key for key, _ in pairs}) != len(pairs):
        raise ValueError("duplicate JSON object keys")
    return dict(pairs)


def _invalid_constant(value: str) -> ExactJson:
    raise ValueError("nonfinite JSON number")


def _exact_value(value: ExactJson) -> JsonValue:
    match value:
        case dict():
            return {"object": {key: _exact_value(item) for key, item in value.items()}}
        case list():
            return {"array": [_exact_value(item) for item in value]}
        case bool():
            return {"boolean": value}
        case int() | Decimal():
            return {"number": str(value)}
        case str():
            return {"string": value}
        case None:
            return None


def strict_identity(
    *,
    method: str,
    path: str,
    query: str,
    headers: Mapping[str, str],
    body: bytes | None,
    mount: str,
    upstream_base: str,
) -> StrictIdentity | IneligibleRequest:
    if (mount, path, method.upper()) not in {
        ("openai", "/openai/v1/chat/completions", "POST"),
        ("anthropic", "/anthropic/v1/messages", "POST"),
    }:
        return IneligibleRequest("unsupported endpoint or method")
    lowered: Final = {key.lower(): value for key, value in headers.items()}
    if len(lowered) != len(headers):
        return IneligibleRequest("duplicate header names")
    if any(
        key not in SEMANTIC_HEADERS | AUTH_HEADERS | EXCLUDED_HEADERS and not key.startswith("x-stainless-")
        for key in lowered
    ):
        return IneligibleRequest("unsupported semantic header")
    if "transfer-encoding" in lowered:
        return IneligibleRequest("unsupported request transfer-encoding; send a content-length framed JSON body")
    authorization: Final = lowered.get("authorization")
    if authorization is not None and authorization.partition(" ")[0].lower() not in {"bearer", "basic", "digest"}:
        return IneligibleRequest("unsupported authorization scheme")
    destination: Final = urlsplit(upstream_base)
    if destination.username or destination.password or destination.query or destination.fragment:
        return IneligibleRequest("upstream destination contains credentials, query or fragment")
    if destination.scheme not in ("http", "https") or not destination.netloc:
        return IneligibleRequest("unsupported upstream destination")
    if body and lowered.get("content-type", "").split(";", 1)[0].strip().lower() != "application/json":
        return IneligibleRequest("unsupported body content-type; stateless_v1 requires JSON")
    try:
        parsed: Final = (
            JSON_VALUE.validate_python(
                json.loads(
                    body, object_pairs_hook=_unique_object, parse_constant=_invalid_constant, parse_float=Decimal
                )
            )
            if body
            else None
        )
    except (ValueError, UnicodeError, DecimalException):
        return IneligibleRequest("invalid JSON or duplicate JSON object keys")
    if body and not isinstance(parsed, dict):
        return IneligibleRequest("stateless inference requires a JSON object")
    try:
        query_pairs: Final = tuple(parse_qsl(query, keep_blank_values=True, errors="strict"))
    except UnicodeError:
        return IneligibleRequest("invalid UTF-8 query encoding")
    return StrictIdentity(
        upstream=upstream_base,
        mount=mount,
        query=tuple((key, "<credential>" if key.lower() in CREDENTIAL_QUERY else value) for key, value in query_pairs),
        headers={key: value for key, value in lowered.items() if key in SEMANTIC_HEADERS},
        auth={
            key: (value.partition(" ")[0] if key == "authorization" else "present")
            for key, value in lowered.items()
            if key in AUTH_HEADERS
        },
        body_present=bool(body),
        body=_exact_value(parsed),
    )
