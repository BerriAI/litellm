from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal, TypeAlias

from pydantic import StrictStr, TypeAdapter, ValidationError
from typing_extensions import ReadOnly, TypedDict

NativeProtocol: TypeAlias = Literal["responses", "messages"]
_Items: TypeAlias = tuple[Mapping[str, object], ...]
_ITEMS: Final = TypeAdapter(_Items)
_WIRE_ITEMS: Final = TypeAdapter(list[Mapping[str, object]])
_HEADERS: Final = TypeAdapter(Mapping[StrictStr, StrictStr])
_COMPACTION_BETA: Final = "compact-2026-09-04"


class _CompactionOperation(TypedDict):
    type: ReadOnly[Literal["summarize"]]


@dataclass(frozen=True, slots=True)
class CompactionHistoryError:
    message: str
    kind: Literal["compaction_history_error"] = "compaction_history_error"


@dataclass(frozen=True, slots=True)
class NativeHistory:
    field: str
    prefix: _Items
    tail: _Items


def _copy_items(items: object) -> _Items:
    return tuple(MappingProxyType(item) for item in deepcopy(_ITEMS.validate_python(items)))


def _native_items(value: object) -> _Items | CompactionHistoryError:
    try:
        return _copy_items(value)
    except ValidationError:
        return CompactionHistoryError("Native history requires an array of objects")


def _is_user_request(item: Mapping[str, object], protocol: NativeProtocol) -> bool:
    if item.get("role") != "user":
        return False
    content: Final = item.get("content")
    if protocol == "responses" or isinstance(content, str):
        return True
    blocks: Final = _native_items(content)
    return isinstance(blocks, tuple) and any(block.get("type") != "tool_result" for block in blocks)


def split_native_history(
    payload: Mapping[str, object], protocol: NativeProtocol
) -> NativeHistory | CompactionHistoryError:
    if payload.get("previous_response_id") is not None or payload.get("conversation") is not None:
        return CompactionHistoryError("Server-side conversation history is unavailable for native compaction")
    field: Final = "input" if protocol == "responses" else "messages"
    items: Final = _native_items(payload.get(field))
    if isinstance(items, CompactionHistoryError):
        return items
    start: Final = next(
        (index for index in range(len(items) - 1, -1, -1) if _is_user_request(items[index], protocol)),
        -1,
    )
    if start < 0:
        return CompactionHistoryError("Native history has no actual user request")
    if start == 0:
        return CompactionHistoryError("Native history has no compactable prefix")
    return NativeHistory(field=field, prefix=items[:start], tail=items[start:])


def compaction_headers(existing: object) -> Mapping[str, str]:
    headers: Final = _HEADERS.validate_python(MappingProxyType({}) if existing is None else existing)
    betas: Final = tuple(
        beta.strip()
        for name, value in headers.items()
        if name.lower() == "anthropic-beta"
        for beta in value.split(",")
        if beta.strip()
    )
    preserved: Final = MappingProxyType(
        {name: value for name, value in headers.items() if name.lower() != "anthropic-beta"}
    )
    return MappingProxyType(
        {
            **preserved,
            "anthropic-beta": ",".join(dict.fromkeys((*betas, _COMPACTION_BETA))),
        }
    )


def native_compaction_payload(
    history: NativeHistory, payload: Mapping[str, object], protocol: NativeProtocol
) -> Mapping[str, object]:
    prefix: Final = deepcopy(_WIRE_ITEMS.validate_python(history.prefix))
    parameters: Final = ("instructions",) if protocol == "responses" else ("system", "tools")
    context: Final = MappingProxyType({key: deepcopy(payload[key]) for key in parameters if key in payload})
    if protocol == "responses":
        return MappingProxyType({"input": prefix, **context})
    operation: Final[_CompactionOperation] = {"type": "summarize"}
    return MappingProxyType(
        {
            "messages": prefix,
            **context,
            "compaction": operation,
            "max_tokens": 4096,
            "extra_headers": _HEADERS.validate_python(compaction_headers(payload.get("extra_headers"))),
        }
    )


def _has_text(item: Mapping[str, object], field: str) -> bool:
    value: Final = item.get(field)
    return isinstance(value, str) and bool(value.strip())


def native_compaction_result(
    response: Mapping[str, object], history: NativeHistory, protocol: NativeProtocol
) -> _Items | CompactionHistoryError:
    if protocol == "responses":
        output: Final = _native_items(response.get("output"))
        if isinstance(output, CompactionHistoryError):
            return output
        compacted: Final = tuple(item for item in output if item.get("type") == "compaction")
        if not compacted or any(not _has_text(item, "encrypted_content") for item in compacted):
            return CompactionHistoryError("Native Responses compaction requires encrypted compaction output")
        canonical: Final = tuple(
            MappingProxyType({key: value for key, value in item.items() if key != "created_by" or value is not None})
            if item.get("type") == "compaction"
            else item
            for item in output
        )
        return _ITEMS.validate_python((*canonical, *_copy_items(history.tail)))
    blocks: Final = _native_items(response.get("content"))
    if response.get("stop_reason") != "compaction" or isinstance(blocks, CompactionHistoryError) or len(blocks) != 1:
        return CompactionHistoryError("Native Messages compaction requires one signed compaction block")
    block: Final = blocks[0]
    if block.get("type") != "compaction" or not _has_text(block, "content") or not _has_text(block, "signature"):
        return CompactionHistoryError("Native Messages compaction requires content and signature")
    return _ITEMS.validate_python(
        (
            MappingProxyType({"role": "assistant", "content": _WIRE_ITEMS.validate_python(blocks)}),
            *_copy_items(history.tail),
        )
    )
