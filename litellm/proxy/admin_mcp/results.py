import hashlib
import hmac
import json
import re
import secrets
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import reduce
from typing import Final

from pydantic import JsonValue

from ._types import JSON_VALUE, JsonObject

PAGE_BYTES: Final = 16_000
ITEM_BYTES: Final = PAGE_BYTES - 1000
PREVIEW_BYTES: Final = 6_000
MAX_RESULT_BYTES: Final = 16_000_000
SECRET_RESULTS: Final = frozenset(
    {"create_key", "create_service_account_key", "regenerate_key", "regenerate_key_by_id"}
)


class ResultError(ValueError):
    pass


def encode(value: JsonValue | JsonObject) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def pointer(parent: str, key: str | int) -> str:
    return parent + "/" + str(key).replace("~", "~0").replace("/", "~1")


def _select(value: JsonValue, parts: tuple[str, ...]) -> JsonValue:
    if not parts:
        return value
    part: Final = parts[0].replace("~1", "/").replace("~0", "~")
    if isinstance(value, list) and re.fullmatch(r"0|[1-9][0-9]*", part):
        return _select(value[int(part)], parts[1:])
    if isinstance(value, dict):
        return _select(value[part], parts[1:])
    raise ValueError("Invalid JSON Pointer segment")


def select(value: JsonValue, path: str) -> JsonValue:
    if not path:
        return value
    if not path.startswith("/") or re.search(r"~(?![01])", path):
        raise ResultError("Use a JSON Pointer path returned by the result index.")
    try:
        return _select(value, tuple(path[1:].split("/")))
    except (KeyError, IndexError, ValueError):
        raise ResultError("The result does not contain that path; browse its parent first.") from None


def kind(value: JsonValue) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    if isinstance(value, str):
        return "string"
    return "number"


def _credential_field(key: str, child: JsonValue, source: str, path: tuple[str | int, ...]) -> bool:
    catalog_key: Final = (
        source in {"list_models", "get_model"}
        and len(path) == 3
        and path[0] == "data"
        and isinstance(path[1], int)
        and path[2] == "model_info"
        and key == "key"
    )
    return bool(
        not catalog_key
        and key.lower()
        in {
            "key",
            "api_key",
            "token",
            "access_token",
            "refresh_token",
            "password",
            "secret",
            "client_secret",
            "authorization",
            "authentication_token",
        }
        and isinstance(child, str)
        and child
        and child not in {"[redacted]", "[credential redacted]"}
        and not re.fullmatch(r"[a-f0-9]{64}", child)
    )


def contains_credential(value: JsonValue, *, source: str = "", path: tuple[str | int, ...] = ()) -> bool:
    if isinstance(value, dict):
        return any(
            _credential_field(key, child, source, path) or contains_credential(child, source=source, path=(*path, key))
            for key, child in value.items()
        )
    if isinstance(value, list):
        return any(contains_credential(child, source=source, path=(*path, i)) for i, child in enumerate(value))
    if not isinstance(value, str):
        return False
    if re.search(r"\bsk-[A-Za-z0-9_+-]+", value):
        return True
    try:
        nested: Final = JSON_VALUE.validate_json(value)
    except ValueError:
        return False
    return contains_credential(nested, source=source, path=path) if isinstance(nested, (dict, list)) else False


@dataclass(frozen=True, slots=True)
class Snapshot:
    owner: bytes
    principal: str
    source: str
    raw: bytes
    deadline: float
    expires_at: float


class ResultStore:
    def __init__(
        self,
        *,
        ttl: int = 300,
        max_bytes: int = 64_000_000,
        max_per_owner: int = 16,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
    ) -> None:
        self.ttl = ttl
        self.max_bytes = max_bytes
        self.max_per_owner = max_per_owner
        self._clock = clock
        self._wall_clock = wall_clock
        self._salt = secrets.token_bytes(32)
        self._items: Mapping[str, Snapshot] = {}

    def _owner(self, credential: str) -> bytes:
        return hmac.digest(self._salt, credential.encode(), hashlib.sha256)

    def _expire(self) -> None:
        now: Final = self._clock()
        self._items = {token: item for token, item in self._items.items() if item.deadline > now}

    def clear(self) -> None:
        self._items = {}

    def put(self, source: str, value: JsonValue, credential: str, principal: str) -> str | None:
        self._expire()
        raw: Final = encode(value)
        if source in SECRET_RESULTS or contains_credential(value, source=source):
            return None
        owner: Final = self._owner(credential)
        if (
            len(raw) > MAX_RESULT_BYTES
            or sum(len(item.raw) for item in self._items.values()) + len(raw) > self.max_bytes
            or sum(item.owner == owner for item in self._items.values()) >= self.max_per_owner
        ):
            return None
        token: Final = secrets.token_urlsafe(24)
        self._items = {
            **self._items,
            token: Snapshot(
                owner,
                principal,
                source,
                raw,
                self._clock() + self.ttl,
                self._wall_clock() + self.ttl,
            ),
        }
        return token

    def get(self, token: str, credential: str, principal: str) -> Snapshot:
        self._expire()
        item: Final = self._items.get(token)
        if item is None or item.owner != self._owner(credential) or item.principal != principal:
            raise ResultError(
                "Result unavailable to this caller or expired. Do not repeat a write to retrieve its response."
            )
        return item


@dataclass(frozen=True, slots=True)
class Preview:
    fields: JsonObject
    omitted: tuple[str, ...] = ()
    omitted_count: int = 0
    remaining: int = PREVIEW_BYTES


def _preview_field(preview: Preview, field: tuple[str, JsonValue]) -> Preview:
    key, value = field
    cost: Final = len(encode({key: value}))
    if cost <= preview.remaining:
        return Preview({**preview.fields, key: value}, preview.omitted, preview.omitted_count, preview.remaining - cost)
    return Preview(preview.fields, (*preview.omitted, key)[:20], preview.omitted_count + 1, preview.remaining)


def _entry(key: str | int, child: JsonValue, path: str, token: str) -> JsonObject:
    child_path: Final = pointer(path, key)
    base: Final = {"key" if isinstance(key, str) else "index": key, "path": child_path, "type": kind(child)}
    if len(encode(child)) <= ITEM_BYTES:
        return {**base, "value": child, "complete": True}
    preview: Final = reduce(_preview_field, child.items(), Preview({})) if isinstance(child, dict) else None
    return {
        **base,
        "complete": False,
        "read": {"result_id": token, "path": child_path},
        **({"total": len(child)} if isinstance(child, (dict, list, str)) else {}),
        **(
            {
                "preview": dict(preview.fields),
                "omitted_fields": list(preview.omitted),
                "omitted_count": preview.omitted_count,
            }
            if preview is not None
            else {}
        ),
    }


def _entries(
    children: tuple[tuple[str | int, JsonValue], ...],
    path: str,
    token: str,
    remaining: int = PAGE_BYTES,
    first: bool = True,
) -> tuple[JsonObject, ...]:
    if not children:
        return ()
    key, child = children[0]
    if not first and len(encode(child)) > ITEM_BYTES and remaining < PREVIEW_BYTES + 1000:
        return ()
    entry: Final = _entry(key, child, path, token)
    cost: Final = len(encode(entry))
    if not first and cost > remaining:
        return ()
    return (entry, *_entries(children[1:], path, token, remaining - cost, False))


def page(
    item: Snapshot,
    token: str,
    *,
    path: str = "",
    offset: int = 0,
    limit: int = 20,
    max_chars: int = 4000,
) -> JsonValue:
    value: Final = select(JSON_VALUE.validate_json(item.raw), path)
    result: Final[JsonObject] = {
        "format": "litellm-admin-page-v1",
        "result_id": token,
        "source_tool": item.source,
        "expires_at": item.expires_at,
        "path": path,
        "type": kind(value),
        "read_tool": "read_admin_result",
        "full_read": {"result_id": token, "path": path, "view": "full"},
    }
    if not isinstance(value, (dict, list, str)):
        if offset:
            raise ResultError("Scalar values do not have offsets.")
        return {**result, "value": value, "complete": True}
    total: Final = len(value)
    if offset > total:
        raise ResultError("Offset exceeds this node's length.")
    end, contents = _page_contents(value, token, path, offset, limit, max_chars)
    return {
        **result,
        "total": total,
        "offset": offset,
        **contents,
        "has_more": end < total,
        "next": {"result_id": token, "path": path, "offset": end, "limit": limit, "max_chars": max_chars}
        if end < total
        else None,
    }


def _page_contents(
    value: JsonValue,
    token: str,
    path: str,
    offset: int,
    limit: int,
    max_chars: int,
) -> tuple[int, JsonObject]:
    if isinstance(value, str):
        chunk: Final = value[offset : offset + max_chars]
        end: Final = offset + len(chunk)
        return end, {
            "value": chunk,
            "offset_unit": "unicode_code_points",
            "complete": offset == 0 and end == len(value),
        }
    assert isinstance(value, (dict, list))
    children: Final = tuple(value.items()) if isinstance(value, dict) else tuple(enumerate(value))
    entries: Final = _entries(children[offset : offset + limit], path, token)
    last: Final = offset + len(entries)
    return last, {
        "entries": [dict(entry) for entry in entries],
        "complete": offset == 0 and last == len(value) and all(entry["complete"] for entry in entries),
    }


RESPONSE_SCHEMA: Final[JsonObject] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "view": {
            "type": "string",
            "enum": ["full", "compact"],
            "description": "full preserves the original result. compact keeps small results whole and pages large results "
            "without discarding data. Never repeat a write for a different view; use read_admin_result.",
        }
    },
}
READ_SCHEMA: Final[JsonObject] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["result_id"],
    "properties": {
        "result_id": {"type": "string"},
        "path": {
            "type": "string",
            "default": "",
            "description": "Exact JSON Pointer from an index; empty string selects the entire original result.",
        },
        "view": {"type": "string", "enum": ["page", "full"], "default": "page"},
        "offset": {"type": "integer", "minimum": 0, "default": 0},
        "limit": {
            "type": "integer",
            "minimum": 1,
            "maximum": 100,
            "default": 20,
            "description": "Object fields or array items per page.",
        },
        "max_chars": {
            "type": "integer",
            "minimum": 1,
            "maximum": 16000,
            "default": 4000,
            "description": "Unicode code points per string chunk; offset uses the same units.",
        },
    },
}
