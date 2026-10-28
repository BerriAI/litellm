"""SDK-free half of the result conversion boundary.

``openapi_to_mcp_generator`` and ``contracts`` must import without the ``mcp``
package installed, so the compatibility enum and the tagged handler outcomes
live here; ``result_conversion`` turns them into SDK results.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Final

from mcp_types.version import MODERN_PROTOCOL_VERSIONS
from pydantic import JsonValue, TypeAdapter, ValidationError

_JSON_VALUE: Final = TypeAdapter(JsonValue)


class WireCompat(str, Enum):
    LEGACY = "legacy"
    MODERN = "modern"


def wire_compat_for(protocol_version: str) -> WireCompat:
    return WireCompat.MODERN if protocol_version in MODERN_PROTOCOL_VERSIONS else WireCompat.LEGACY


@dataclass(frozen=True, slots=True)
class TextResult:
    text: str


@dataclass(frozen=True, slots=True)
class JsonResult:
    value: JsonValue
    original_text: str


def parse_http_body(body: str) -> TextResult | JsonResult:
    if not body.strip():
        return TextResult(body)
    try:
        value: Final = _JSON_VALUE.validate_json(body)
    except ValidationError:
        return TextResult(body)
    if value is None:
        return TextResult(body)
    return JsonResult(value=value, original_text=body)


def handler_outcome(value: object) -> TextResult | JsonResult:
    """Normalize what a registered tool handler returned; OpenAPI handlers already return a tagged outcome."""
    if isinstance(value, (TextResult, JsonResult)):
        return value
    return TextResult(str(value))
