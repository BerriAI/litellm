"""Canonical request identity for replay matching (LIT-5741).

Matching a replayed call against the raw recorded request never hits: unique
markers salt prompts, model names, and tags; every run mints fresh virtual
keys; request ids and timestamps differ on every call. Matching on transport
verb + path alone collides: two different requests to the same route silently
swap responses, which passes when it should miss. The canonicalizer strips
exactly the volatile material (volatile headers, credential fields, markers,
generated ids, timestamps) and hashes what remains with sorted object keys, so
identity is content-based and stable across runs and machines.

Every rewrite rule lives in this module, next to the transports that apply it:
a new volatile header, credential field name, or generated-id shape is one
edit here, never a per-suite change.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from functools import reduce
from typing import Final

from pydantic import JsonValue

from fixture_bundle import RecordedRequest

VOLATILE_HEADER_NAMES: Final[frozenset[str]] = frozenset(
    {
        "authorization",
        "x-litellm-api-key",
        "x-api-key",
        "x-goog-api-key",
        "x-request-id",
        "traceparent",
        "tracestate",
    }
)

SECRET_FIELD_NAMES: Final[frozenset[str]] = frozenset(
    {"api_key", "aws_access_key_id", "static_headers", "vertex_credentials"}
)
SECRET_FIELD_SUFFIXES: Final[tuple[str, ...]] = (
    "_api_key",
    "_secret_key",
    "_secret_access_key",
    "_session_token",
    "_credentials",
    "_password",
)
SECRET_PLACEHOLDER: Final = "<secret>"

PLACEHOLDER_RULES: Final[tuple[tuple[re.Pattern[str], str], ...]] = (
    (re.compile(r"(?<![0-9a-fA-F])[0-9a-f]{64}(?![0-9a-fA-F])"), "<sha256>"),
    (
        re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"),
        "<uuid>",
    ),
    (re.compile(r"sk-[A-Za-z0-9_-]{16,}"), "<key>"),
    (
        re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?"),
        "<timestamp>",
    ),
    (re.compile(r"(?<!\d)\d{4}-\d{2}-\d{2}(?!\d)"), "<date>"),
    (
        re.compile(r"\b(?:chatcmpl|msgbatch|msg|resp|batch|call|req|ftjob|gen|file)[-_][A-Za-z0-9]{8,}\b"),
        "<id>",
    ),
    (re.compile(r"(?<![0-9a-fA-F])[0-9a-f]{12}(?![0-9a-fA-F])"), "<marker>"),
)


def is_secret_field(name: str) -> bool:
    lowered: Final = name.lower()
    return lowered in SECRET_FIELD_NAMES or lowered.endswith(SECRET_FIELD_SUFFIXES)


def canonical_string(value: str) -> str:
    return reduce(lambda acc, rule: rule[0].sub(rule[1], acc), PLACEHOLDER_RULES, value)


def _canonical_flat(fields: dict[str, str]) -> dict[str, JsonValue]:
    return {
        key: SECRET_PLACEHOLDER if is_secret_field(key) else canonical_string(value)
        for key, value in fields.items()
    }


def _canonical_value(value: JsonValue) -> JsonValue:
    match value:
        case str():
            return canonical_string(value)
        case dict():
            return {
                key: SECRET_PLACEHOLDER
                if is_secret_field(key) and item is not None
                else _canonical_value(item)
                for key, item in value.items()
            }
        case list():
            return [_canonical_value(item) for item in value]
        case _:
            return value


@dataclass(frozen=True, slots=True)
class CanonicalRequest:
    method: str
    path: str
    content: str

    @property
    def key(self) -> str:
        digest: Final = hashlib.sha256(
            f"{self.method} {self.path}\n{self.content}".encode()
        ).hexdigest()[:16]
        return f"{self.method} {self.path} #{digest}"

    def pretty_content(self) -> str:
        return json.dumps(json.loads(self.content), indent=2, sort_keys=True)


def canonicalize(request: RecordedRequest) -> CanonicalRequest:
    file_identity: Final[JsonValue | None] = (
        None
        if request.file_name is None and request.file_sha256 is None
        else {
            "name": None if request.file_name is None else canonical_string(request.file_name),
            "sha256": request.file_sha256,
        }
    )
    content: Final[dict[str, JsonValue]] = {
        "headers": {
            name.lower(): canonical_string(value)
            for name, value in request.headers.items()
            if name.lower() not in VOLATILE_HEADER_NAMES
        },
        "params": _canonical_flat(request.params),
        "body": _canonical_value(request.body),
        "form": None if request.form is None else _canonical_flat(request.form),
        "file": file_identity,
    }
    return CanonicalRequest(
        method=request.method,
        path=canonical_string(request.path),
        content=json.dumps(content, sort_keys=True, separators=(",", ":")),
    )
