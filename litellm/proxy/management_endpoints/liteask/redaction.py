import json
import re
from collections.abc import Mapping
from functools import reduce
from types import MappingProxyType
from typing import Final, TypeAlias

from pydantic import JsonValue

from litellm.litellm_core_utils.secret_redaction import REDACTED, redact_string
from litellm.litellm_core_utils.sensitive_data_masker import SensitiveDataMasker

_MASKER: Final = SensitiveDataMasker()
_HASH: Final = re.compile(r"[a-fA-F0-9]{64}")
_SAFE_FIELDS: Final = frozenset(
    (
        "key_alias",
        "key_name",
        "key_id",
        "key_hash",
        "api_key_hash",
        "token_hash",
        "key_type",
        "key_rotation_interval",
        "access_groups",
        "access_group_ids",
        "api_key_alias",
        "api_key_name",
    )
)
_MAX_DEPTH: Final = 16
_MAX_ITEMS: Final = 100
_MAX_TEXT: Final = 8_000
_MAX_RESULT_BYTES: Final = 32_000
_Path: TypeAlias = tuple[str | None, ...]
_RESPONSE_HASH_PATHS: Final[Mapping[str, tuple[_Path, ...]]] = MappingProxyType(
    {
        "keys_list": (("keys", None, "token"),),
        "key_info": (("key",), ("info", "token")),
        "key_update": (("key",), ("token",)),
        "key_block": (("token",),),
        "key_unblock": (("token",),),
        "team_info": (("keys", None, "token"),),
        "user_info": (("keys", None, "token"),),
        "request_logs": (("data", None, "api_key"),),
        "spend_report": ((None, "api_key"),),
        "team_spend_report": ((None, "api_key"),),
        "key_spend_report": ((None, "api_key"),),
    }
)
_ARGUMENT_HASH_PATHS: Final[Mapping[str, tuple[_Path, ...]]] = MappingProxyType(
    {
        "key_info": (("query", "key"),),
        "key_update": (("body", "key"),),
        "key_delete": (("body", "keys", None),),
        "key_block": (("body", "key"),),
        "key_unblock": (("body", "key"),),
        "request_logs": (("query", "api_key"),),
        "spend_report": (("query", "api_key"),),
        "key_spend_report": (("query", "api_key"),),
    }
)


def _text(value: str, secrets: tuple[str, ...]) -> str:
    exact: Final = reduce(lambda text, secret: text.replace(secret, REDACTED) if secret else text, secrets, value)
    clean: Final = redact_string(exact)
    return clean if len(clean) <= _MAX_TEXT else clean[:_MAX_TEXT] + " [truncated]"


def _walk(value: JsonValue, secrets: tuple[str, ...], path: _Path, hash_paths: tuple[_Path, ...]) -> JsonValue:
    if path in hash_paths and isinstance(value, str) and _HASH.fullmatch(value):
        return _text(value, secrets)
    name: Final = path[-1] if path else None
    if (
        isinstance(name, str)
        and name.lower() not in _SAFE_FIELDS
        and _MASKER.is_sensitive_key(name)
        and value is not None
    ):
        return REDACTED
    if len(path) >= _MAX_DEPTH:
        return "[nested data omitted]"
    if isinstance(value, dict):
        return {  # mutable-ok: JSON response boundary
            _text(field, secrets): _walk(item, secrets, (*path, field), hash_paths) for field, item in value.items()
        }
    if isinstance(value, list):
        notice: Final[JsonValue] = {"truncated": True, "omitted_items": len(value) - _MAX_ITEMS}
        return [  # mutable-ok: JSON response arrays preserve the original shape
            *(_walk(item, secrets, (*path, None), hash_paths) for item in value[:_MAX_ITEMS]),
            *((notice,) if len(value) > _MAX_ITEMS else ()),
        ]
    if isinstance(value, str):
        return _text(value, secrets)
    return value


def sanitize(
    value: JsonValue,
    secrets: tuple[str, ...] = (),
    *,
    response_operation: str = "",
    argument_operation: str = "",
) -> JsonValue:
    hash_paths: Final = (
        *_RESPONSE_HASH_PATHS.get(response_operation, ()),
        *_ARGUMENT_HASH_PATHS.get(argument_operation, ()),
    )
    cleaned: Final = _walk(value, secrets, (), hash_paths)
    if len(json.dumps(cleaned).encode()) > _MAX_RESULT_BYTES:
        notice: Final[JsonValue] = {
            "truncated": True,
            "message": "Result exceeds the size limit. Use pagination or a narrower query.",
        }
        return notice
    return cleaned
