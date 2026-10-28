import base64
import copy
import json
import struct
from binascii import crc32
from collections.abc import AsyncGenerator, AsyncIterable, Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial, reduce
from itertools import accumulate, chain
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final, Optional, Protocol, TypeAlias, TypeVar

from pydantic import InstanceOf, TypeAdapter, ValidationError
from typing_extensions import ReadOnly, TypedDict

from litellm._logging import verbose_proxy_logger
from litellm.llms.anthropic.chat.guardrail_translation.handler import AnthropicMessagesHandler
from litellm.llms.base_llm.guardrail_translation.base_translation import (
    BaseTranslation,
    StreamingScanKey,
    StreamTransformSink,
)
from litellm.llms.base_llm.guardrail_translation.utils import (
    effective_scan_only_tool_results_for_guardrail,
    effective_skip_system_message_for_guardrail,
    effective_skip_tool_message_for_guardrail,
    scoped_structured_message_indices,
    unappliable_request_rewrite,
)
from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM
from litellm.llms.bedrock.chat.converse_to_openai import (
    ConverseRow,
    JsonPath,
    as_json_array,
    as_json_object,
    converse_rows,
    converse_tools,
    parse_converse_request,
)
from litellm.llms.bedrock.passthrough.guardrail_translation.structured_edits import (
    Edit,
    ReplacedTools,
    ReturnedTool,
    ToolResultText,
    structured_edits,
    tool_edits,
)
from litellm.proxy.guardrails.auto_router_compression import is_compression_guardrail
from litellm.types.utils import GenericGuardrailAPIInputs

if TYPE_CHECKING:
    from litellm.integrations.custom_guardrail import CustomGuardrail, ModifyResponseException
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.llms.pass_through.guardrail_translation.handler import (
        PassThroughEndpointHandler,
    )
    from litellm.proxy.utils import ProxyLogging
    from litellm.types.proxy.auth.user_api_key_auth import UserAPIKeyAuth

_CONVERSE_ACTIONS: Final = frozenset({"converse", "converse-stream"})
_INVOKE_STREAM_ACTION: Final = "invoke-with-response-stream"
_EVENT_STREAM_CONTENT_TYPE: Final = "vnd.amazon.eventstream"
_EVENT_STREAM_MEDIA_TYPE: Final = "application/vnd.amazon.eventstream"


def _is_converse_endpoint(endpoint: str) -> bool:
    parts: Final = endpoint.rstrip("/").split("/")
    return bool(parts) and parts[-1] in _CONVERSE_ACTIONS


def _is_invoke_stream_endpoint(endpoint: str) -> bool:
    return endpoint.rstrip("/").split("/")[-1] == _INVOKE_STREAM_ACTION


def _generic_passthrough_handler() -> "PassThroughEndpointHandler":
    """
    Fallback for non-Converse Bedrock routes (e.g. invoke). The generic
    handler scans the full request/response payload so blocking guardrails
    still run, matching how other passthrough providers are guarded.
    """
    from litellm.llms.pass_through.guardrail_translation.handler import (
        PassThroughEndpointHandler,
    )

    return PassThroughEndpointHandler()


def _reads_flat_texts_only(guardrail: "CustomGuardrail") -> bool:
    """Whether `guardrail` gets only the flat texts on Converse, without rows or tools."""
    return is_compression_guardrail(guardrail) or not guardrail.supports_structured_messages()


_StringHolder = tuple[Any, str | int]
_LeafIdentity: TypeAlias = tuple[int, str | int]


@dataclass(frozen=True, slots=True)
class _Assign:
    holder: _StringHolder
    value: object


@dataclass(frozen=True, slots=True)
class _Remove:
    holder: _StringHolder


_Write: TypeAlias = _Assign
_ContentHolder: TypeAlias = tuple[object, str]
_BlankDrop: TypeAlias = tuple[_ContentHolder, frozenset[int]]
_BODY_BY_REFERENCE: Final = TypeAdapter(InstanceOf[dict[str, object]])
_OBJECT_BY_REFERENCE: Final = TypeAdapter(InstanceOf[Mapping[str, object]])
_ARRAY_BY_REFERENCE: Final = TypeAdapter(InstanceOf[list[object]])


def _collect_strings(node: object, holders: list[_StringHolder]) -> None:
    """
    Record a (container, key) holder for every non-empty string value nested
    under an arbitrary JSON node, so prompt content a caller hides in fields
    like ``toolUse.input`` or ``toolResult.content[].json`` is still scanned
    and can be written back in place. Iterative to avoid unbounded recursion
    on deeply nested payloads.
    """
    stack: Final[list[object]] = [node]
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            for key, value in current.items():
                if isinstance(value, str):
                    if value:
                        holders.append((current, key))
                else:
                    stack.append(value)
        elif isinstance(current, list):
            for index, value in enumerate(current):
                if isinstance(value, str):
                    if value:
                        holders.append((current, index))
                else:
                    stack.append(value)


def _collect_block_text(block: dict, holders: list[_StringHolder]) -> None:
    text: Final = block.get("text")
    if isinstance(text, str) and text:
        holders.append((block, "text"))


def _request_field_holders(body: Mapping[str, object]) -> tuple[_StringHolder, ...]:
    holders: Final = list[_StringHolder]()
    _collect_strings(body.get("additionalModelRequestFields"), holders)
    return tuple(holders)


def _extract_converse_texts(
    body: dict,
    skip_system: bool,
    skip_tool: bool,
    scan_only_tool_results: bool = False,
) -> tuple[list[str], list[_StringHolder]]:
    """
    Walk a Bedrock Converse request body and collect text content.

    Returns (texts, holders) where each holder is the (container, key) pair
    that owns the extracted string, so write-back mutates it in place. Besides
    top-level ``text`` blocks this scans the arbitrary-JSON fields a caller can
    hide prompt content in -- ``toolUse.input`` and
    ``toolResult.content[].json`` (alongside ``toolResult.content[].text``) --
    as well as ``additionalModelRequestFields``, a free-form model-parameter bag
    with no schema that a caller can route blocked content through.

    ``toolConfig.tools`` is deliberately NOT scanned. Tool definitions are
    app-authored config, so their names, descriptions and JSON-schema strings
    ("object", property names, titles, type names, enum values) would each reach
    the guardrail as a separate INPUT item, producing false positives and
    inflating guardrail usage for a request whose only prompt is one user
    message. No other guardrail translation handler puts tool definitions in
    ``texts``; the chat and messages handlers carry them in the structured
    ``tools`` input instead, which this handler also populates by converting
    each Bedrock ``toolSpec`` to the OpenAI tool shape.

    ``additionalModelRequestFields`` is treated differently on purpose. Bedrock
    gives ``toolConfig.tools`` a fixed schema whose contents are tool metadata by
    contract, while ``additionalModelRequestFields`` is free-form and defined by
    the target model, so what it carries cannot be classified without knowing
    that model. Scanning it stays the fail-closed default. Its strings come
    last, in the same order as they follow the latest user message in the rows.
    """
    system_holders: Final = list[_StringHolder]()
    if not skip_system and not scan_only_tool_results:
        for block in body.get("system") or []:
            if isinstance(block, dict):
                _collect_block_text(block, system_holders)

    request_field_holders: Final = () if scan_only_tool_results else _request_field_holders(body)

    message_holders: Final = list[_StringHolder]()
    for message in body.get("messages") or []:
        if not isinstance(message, dict):
            continue
        for block in message.get("content") or []:
            if not isinstance(block, dict):
                continue
            if skip_tool and ("toolUse" in block or "toolResult" in block):
                continue
            if scan_only_tool_results and "toolResult" not in block:
                continue
            _collect_block_text(block, message_holders)
            tool_use = block.get("toolUse")
            if isinstance(tool_use, dict):
                _collect_strings(tool_use.get("input"), message_holders)
            tool_result = block.get("toolResult")
            if isinstance(tool_result, dict):
                for inner in tool_result.get("content") or []:
                    if isinstance(inner, dict):
                        _collect_block_text(inner, message_holders)
                        _collect_strings(inner.get("json"), message_holders)

    holders: Final = [*system_holders, *message_holders, *request_field_holders]
    texts: Final = [container[key] for container, key in holders]
    return texts, holders


@dataclass(frozen=True, slots=True)
class _Scope:
    skip_system: bool
    skip_tool: bool
    scan_only_tool_results: bool


def _texts_applied_rows(body: Mapping[str, object], texts: Sequence[str], scope: _Scope) -> tuple[ConverseRow, ...]:
    rewritten: Final = _BODY_BY_REFERENCE.validate_python(json.loads(json.dumps(body)))
    _, holders = _extract_converse_texts(rewritten, scope.skip_system, scope.skip_tool, scope.scan_only_tool_results)
    _write_back_texts(texts, holders)
    request: Final = parse_converse_request(rewritten)
    return converse_rows(request, include_tool_calls=not scope.skip_tool) if request else ()


def _texts_follow_rows(
    body: Mapping[str, object],
    rows: Sequence[ConverseRow],
    texts: Sequence[str],
    guardrailed_texts: Sequence[str],
    changed_rows: frozenset[int],
    scope: _Scope,
) -> bool:
    """Whether the returned texts sit in the flat layout and only touch rows the vendor also changed."""
    if len(guardrailed_texts) != len(texts):
        return False
    view: Final = _texts_applied_rows(body, guardrailed_texts, scope)
    if len(view) != len(rows):
        return False
    rows_changed_by_texts: Final = frozenset(
        index for index, (row, viewed) in enumerate(zip(rows, view)) if row.message != viewed.message
    )
    return rows_changed_by_texts <= changed_rows


def _child(node: object, key: str | int) -> object:
    try:
        match key:
            case str():
                return _OBJECT_BY_REFERENCE.validate_python(node).get(key)
            case int():
                items: Final = _ARRAY_BY_REFERENCE.validate_python(node)
                return items[key] if 0 <= key < len(items) else None
    except ValidationError:
        return None


def _holder(body: object, path: JsonPath) -> _StringHolder | None:
    container: Final = reduce(_child, path[:-1], body)
    return (container, path[-1]) if isinstance(_child(container, path[-1]), str) else None


def _leaf_identity(holder: _StringHolder) -> _LeafIdentity:
    container, key = holder
    return id(container), key


def _changed_leaves(
    holders: Sequence[_StringHolder], texts: Sequence[str], guardrailed_texts: Sequence[str]
) -> frozenset[_LeafIdentity]:
    return frozenset(
        _leaf_identity(holder) for holder, before, after in zip(holders, texts, guardrailed_texts) if after != before
    )


def _tool_result_write(body: Mapping[str, object], edit: ToolResultText) -> _Write | None:
    container: Final = reduce(_child, edit.content[:-1], body)
    blocks: Final = as_json_array(_child(container, edit.content[-1]))
    if blocks is None:
        return None
    dropped: Final = frozenset(edit.indices[1:])
    return _Assign(
        (container, edit.content[-1]),
        [
            {"text": edit.text} if index == edit.indices[0] else block
            for index, block in enumerate(blocks)
            if index not in dropped
        ],
    )


def _is_cache_point(entry: object) -> bool:
    fields: Final = as_json_object(entry)
    return fields is not None and "cachePoint" in fields


def _tool_name(entry: object) -> object:
    return _child(_child(entry, "toolSpec"), "name")


def _entries_after(unsent: Sequence[tuple[int, object]], anchor: int) -> tuple[object, ...]:
    return tuple(entry for at, entry in unsent if at == anchor)


def _sent_anchor(sent: frozenset[int], anchor: int, index: int) -> int:
    return index if index in sent else anchor


def _replaced(entry: ReturnedTool) -> int | None:
    return entry if isinstance(entry, int) else entry.replaces


def _replaced_tools_write(body: Mapping[str, object], edit: ReplacedTools) -> _Write | None:
    """
    Rebuild ``toolConfig.tools`` in the order the guardrail returned. Entries it was never sent (cache points, system
    tools) stay right after the tool they followed, and a cache point goes away with its tool. Refuses when
    ``toolChoice`` names a tool that is no longer there, and creates ``toolConfig`` when the request had none. A
    ``toolConfig`` left with no tool stays until the request is sent, so later guardrails still see the tool history.
    """
    tool_config: Final = _child(body, "toolConfig")
    tools: Final = () if tool_config is None else as_json_array(_child(tool_config, "tools"))
    if tools is None:
        return None
    anchors: Final = tuple(accumulate(range(len(tools)), partial(_sent_anchor, edit.sent), initial=-1))[1:]
    unsent: Final = tuple(
        (anchor, tool) for index, (anchor, tool) in enumerate(zip(anchors, tools)) if index not in edit.sent
    )
    replaced: Final = frozenset(_replaced(entry) for entry in edit.tools)
    placed: Final = chain.from_iterable(
        (tools[entry], *_entries_after(unsent, entry))
        if isinstance(entry, int)
        else (*entry.blocks, *(() if entry.replaces is None else _entries_after(unsent, entry.replaces)))
        for entry in edit.tools
    )
    orphans: Final = tuple(tool for at, tool in unsent if at not in replaced and at >= 0 and not _is_cache_point(tool))
    rebuilt: Final = [*_entries_after(unsent, -1), *placed, *orphans]
    chosen: Final = _child(_child(_child(tool_config, "toolChoice"), "tool"), "name")
    if isinstance(chosen, str) and all(_tool_name(tool) != chosen for tool in rebuilt):
        return None
    if tool_config is None:
        return None if all(map(_is_cache_point, rebuilt)) else _Assign((body, "toolConfig"), {"tools": rebuilt})
    return _Assign((tool_config, "tools"), rebuilt)


def _resolved(body: Mapping[str, object], edit: Edit) -> _Write | None:
    match edit:
        case ToolResultText():
            return _tool_result_write(body, edit)
        case ReplacedTools():
            return _replaced_tools_write(body, edit)
        case (path, text):
            holder: Final = _holder(body, path)
            return None if holder is None else _Assign(holder, text)


def _kept_write(write: _Write, flat_changes: frozenset[_LeafIdentity]) -> bool:
    return _leaf_identity(write.holder) not in flat_changes


def _holder_edits(
    body: Mapping[str, object], edits: Sequence[Edit], flat_changes: frozenset[_LeafIdentity]
) -> tuple[_Write, ...] | None:
    resolved: Final = tuple(_resolved(body, edit) for edit in edits)
    if any(write is None for write in resolved):
        return None
    return tuple(write for write in resolved if write is not None and _kept_write(write, flat_changes))


def _apply(writes: Sequence[_Assign | _Remove]) -> None:
    for write in writes:
        match write:
            case _Assign(holder=(container, key), value=value):
                container[key] = value
            case _Remove(holder=(container, key)):
                del container[key]


def _block_text_holder(block: object) -> _ContentHolder | None:
    if isinstance(_child(block, "text"), str):
        return block, "text"
    guarded: Final = _child(_child(block, "guardContent"), "text")
    return (guarded, "text") if isinstance(_child(guarded, "text"), str) else None


def _final_texts(
    holders: Sequence[_StringHolder], applied_texts: Sequence[str], writes: Sequence[_Write]
) -> Mapping[_LeafIdentity, object]:
    flat: Final = {_leaf_identity(holder): text for holder, text in zip(holders, applied_texts)}
    structured: Final = {_leaf_identity(write.holder): write.value for write in writes}
    return MappingProxyType({**flat, **structured})


def _blanked(final: Mapping[_LeafIdentity, object], block: object) -> bool:
    holder: Final = _block_text_holder(block)
    if holder is None:
        return False
    container, key = holder
    value: Final = final.get((id(container), key))
    return isinstance(value, str) and value != _child(container, key) and not value.strip()


def _blank_blocks(final: Mapping[_LeafIdentity, object], holder: _ContentHolder) -> frozenset[int]:
    container, key = holder
    return frozenset(id(block) for block in as_json_array(_child(container, key)) or () if _blanked(final, block))


def _empties_turn(drop: _BlankDrop) -> bool:
    (container, key), dropped = drop
    blocks: Final = as_json_array(_child(container, key)) or ()
    return bool(dropped) and all(id(block) in dropped or _is_cache_point(block) for block in blocks)


def _turn_drop(final: Mapping[_LeafIdentity, object], message: object) -> _BlankDrop:
    return (message, "content"), _blank_blocks(final, (message, "content"))


def _blank_drops(body: Mapping[str, object], final: Mapping[_LeafIdentity, object]) -> tuple[_BlankDrop, ...]:
    """The system and message text blocks the guardrail left blank, except in a turn they would leave empty."""
    turns: Final = (_turn_drop(final, message) for message in as_json_array(body.get("messages")) or ())
    system: Final = ((body, "system"), _blank_blocks(final, (body, "system")))
    return tuple(drop for drop in (system, *(turn for turn in turns if not _empties_turn(turn))) if drop[1])


def _blank_drop_write(drop: _BlankDrop) -> _Assign | _Remove:
    holder, dropped = drop
    container, key = holder
    kept: Final = [block for block in as_json_array(_child(container, key)) or () if id(block) not in dropped]
    return _Assign(holder, kept) if not all(map(_is_cache_point, kept)) else _Remove(holder)


def _edits_drop_a_text_change(
    body: Mapping[str, object],
    edits: Sequence[Edit],
    texts: Sequence[str],
    guardrailed_texts: Sequence[str],
    scope: _Scope,
) -> bool:
    """
    Whether the row and tool edits alone leave out a change the guardrail made in its returned texts. Texts of the
    length that was sent are read position by position, so a changed string whose field the edits leave at its
    original value is dropped. Texts of another length cannot be placed, so what they mask cannot be checked either,
    while an empty list leaves every text as it was.
    """
    if not guardrailed_texts:
        return False
    if len(guardrailed_texts) != len(texts):
        return True
    rewritten: Final = _BODY_BY_REFERENCE.validate_python(json.loads(json.dumps(body)))
    _, holders = _extract_converse_texts(rewritten, scope.skip_system, scope.skip_tool, scope.scan_only_tool_results)
    writes: Final = _holder_edits(rewritten, edits, frozenset())
    if writes is None:
        return True
    _apply(writes)
    edited: Final = [container[key] for container, key in holders]
    return any(
        returned != before and after == before for before, returned, after in zip(texts, guardrailed_texts, edited)
    )


def _extract_converse_output_texts(
    content_blocks: Sequence[object],
) -> tuple[list[str], list[_StringHolder]]:
    """
    Collect user-visible text from Bedrock Converse output content blocks.

    Covers ``text`` blocks plus the other content-bearing fields a model can
    emit -- ``toolUse.input``, ``reasoningContent.reasoningText.text`` and
    ``citationsContent.content[].text`` -- while leaving structural values such
    as reasoning signatures and citation sources untouched.
    """
    holders: Final[list[_StringHolder]] = []
    for block in content_blocks:
        if not isinstance(block, dict):
            continue
        _collect_block_text(block, holders)
        tool_use = block.get("toolUse")
        if isinstance(tool_use, dict):
            _collect_strings(tool_use.get("input"), holders)
        reasoning = block.get("reasoningContent")
        if isinstance(reasoning, dict):
            reasoning_text = reasoning.get("reasoningText")
            if isinstance(reasoning_text, dict):
                _collect_block_text(reasoning_text, holders)
        citations = block.get("citationsContent")
        if isinstance(citations, dict):
            for cited in citations.get("content") or []:
                if isinstance(cited, dict):
                    _collect_block_text(cited, holders)
    texts: Final = [container[key] for container, key in holders]
    return texts, holders


def _write_back_texts(
    guardrailed_texts: Sequence[str],
    holders: Sequence[_StringHolder],
) -> None:
    if len(guardrailed_texts) < len(holders):
        verbose_proxy_logger.warning(
            "BedrockPassthroughGuardrailHandler: guardrail returned %d texts for %d "
            "extracted fields; the unreturned fields keep their original text",
            len(guardrailed_texts),
            len(holders),
        )
    for idx, (container, key) in enumerate(holders):
        if idx >= len(guardrailed_texts):
            break
        container[key] = guardrailed_texts[idx]


_GroupKey: TypeAlias = str | tuple[str, int]


class _TextContainer(Protocol):
    """JSON object whose ``key`` entry holds a guardrailable text string."""

    def __getitem__(self, key: str, /) -> str: ...

    def __setitem__(self, key: str, value: str, /) -> None: ...


_DeltaHolder = tuple[_GroupKey, _TextContainer, str]

_StreamGroupKey: TypeAlias = tuple[int | None, _GroupKey]


class _StreamFrame(TypedDict):
    """One raw event-stream frame plus the guardrailable texts it carries."""

    raw: ReadOnly[bytes]
    texts: ReadOnly[Sequence[tuple[_StreamGroupKey, str]]]


def _unpack_uint32(buffer: bytes) -> int:
    return struct.unpack("!I", buffer)[0]


def _collect_stream_delta_text_holders(delta: object) -> list[_DeltaHolder]:
    """
    Collect the user-visible text strings a Bedrock Converse ``contentBlockDelta``
    can carry, matching the coverage of the non-streaming output handler.

    Each holder is ``(group_key, container, key)`` where ``container[key]`` is the
    text. ``group_key`` ties together fragments that belong to the same logical
    stream (e.g. a single mask token split across frames) so they are
    concatenated before guardrailing and redistributed afterwards. Structural
    values such as reasoning signatures, redacted reasoning and citation sources
    are left out so they are never rewritten.
    """
    holders: Final[list[_DeltaHolder]] = []
    if not isinstance(delta, dict):
        return holders
    if isinstance(delta.get("text"), str):
        holders.append(("text", delta, "text"))
    tool_use: Final = delta.get("toolUse")
    if isinstance(tool_use, dict) and isinstance(tool_use.get("input"), str):
        holders.append(("tool", tool_use, "input"))
    reasoning: Final = delta.get("reasoningContent")
    if isinstance(reasoning, dict) and isinstance(reasoning.get("text"), str):
        holders.append(("reasoning", reasoning, "text"))
    citations: Final = delta.get("citationsContent")
    if isinstance(citations, dict):
        for index, cited in enumerate(citations.get("content") or []):
            if isinstance(cited, dict) and isinstance(cited.get("text"), str):
                holders.append((("citation", index), cited, "text"))
    return holders


def _delta_frame_texts(payload: Mapping[str, object]) -> tuple[tuple[_StreamGroupKey, str], ...]:
    """The guardrailable texts of one ``contentBlockDelta`` payload, each keyed by its content block and kind."""
    block_index: Final = payload.get("contentBlockIndex")
    # Group per content block: a guardrail edit that changes length must not move text between blocks.
    return tuple(
        ((block_index if isinstance(block_index, int) else None, group_key), container[key])
        for group_key, container, key in _collect_stream_delta_text_holders(payload.get("delta"))
    )


@dataclass(frozen=True, slots=True)
class _StreamTextGroups:
    """Stream texts joined per group, keeping where each fragment came from so a rewrite can be split back."""

    groups: tuple[_StreamGroupKey, ...]
    members: Mapping[_StreamGroupKey, Sequence[tuple[int, int]]]
    fragments: Mapping[_StreamGroupKey, Sequence[str]]

    def texts(self) -> tuple[str, ...]:
        return tuple("".join(self.fragments[group]) for group in self.groups)


def _group_stream_texts(frame_texts: Sequence[Sequence[tuple[_StreamGroupKey, str]]]) -> _StreamTextGroups:
    """Group the per-frame texts by key in first-seen order; ``groups`` keeps only the groups with any text."""
    order: Final[list[_StreamGroupKey]] = []  # mutable-ok: built in one pass over the frames
    members: Final[dict[_StreamGroupKey, list[tuple[int, int]]]] = {}  # mutable-ok: built in one pass over the frames
    fragments: Final[dict[_StreamGroupKey, list[str]]] = {}  # mutable-ok: built in one pass over the frames
    for frame_idx, texts in enumerate(frame_texts):
        for local_idx, (group_key, text) in enumerate(texts):
            if group_key not in members:
                members[group_key] = []
                fragments[group_key] = []
                order.append(group_key)
            members[group_key].append((frame_idx, local_idx))
            fragments[group_key].append(text)
    return _StreamTextGroups(
        groups=tuple(group for group in order if "".join(fragments[group])),
        members=members,
        fragments=fragments,
    )


def _split_group_texts(groups: _StreamTextGroups, rewritten_texts: Sequence[str]) -> Mapping[tuple[int, int], str]:
    """Cut each group's rewritten text back into its fragments, sized like the original fragments."""
    split: Final[dict[tuple[int, int], str]] = {}  # mutable-ok: built in one pass over the groups
    for group_key, rewritten in zip(groups.groups, rewritten_texts):
        members = groups.members[group_key]
        originals = groups.fragments[group_key]
        total_orig = sum(len(t) for t in originals) or 1
        pos = 0
        for k, member in enumerate(members):
            if k == len(members) - 1:
                split[member] = rewritten[pos:]
            else:
                end = pos + round(len(rewritten) * len(originals[k]) / total_orig)
                split[member] = rewritten[pos:end]
                pos = end
    return split


def _event_stream_message(headers_bytes: bytes, payload: bytes) -> bytes:
    """One AWS event-stream message: prelude, prelude CRC, headers, payload and message CRC."""
    prelude: Final = struct.pack("!II", 12 + len(headers_bytes) + len(payload) + 4, len(headers_bytes))
    prelude_crc: Final = crc32(prelude) & 0xFFFFFFFF
    message: Final = prelude + struct.pack("!I", prelude_crc) + headers_bytes + payload
    return message + struct.pack("!I", crc32(message) & 0xFFFFFFFF)


def _string_header(name: str, value: str) -> bytes:
    name_bytes: Final = name.encode()
    value_bytes: Final = value.encode()
    return bytes((len(name_bytes),)) + name_bytes + b"\x07" + struct.pack("!H", len(value_bytes)) + value_bytes


def _string_headers(headers: Mapping[str, str]) -> bytes:
    return b"".join(_string_header(name, value) for name, value in headers.items())


def _converse_event(event_type: str, payload: Mapping[str, object]) -> bytes:
    return _event_stream_message(
        _string_headers({":event-type": event_type, ":content-type": "application/json", ":message-type": "event"}),
        json.dumps(payload, separators=(",", ":")).encode(),
    )


def _converse_exception(message: str, status_code: int) -> bytes:
    """An in-stream exception frame, which boto3 raises as ``EventStreamError`` carrying ``message``."""
    return _event_stream_message(
        _string_headers(
            {
                ":exception-type": "validationException" if status_code < 500 else "internalServerException",
                ":content-type": "application/json",
                ":message-type": "exception",
            }
        ),
        json.dumps({"message": message}, separators=(",", ":")).encode(),
    )


def _base64_json(event: Mapping[str, object]) -> str:
    return base64.b64encode(json.dumps(event, separators=(",", ":")).encode()).decode()


def _invoke_chunk(event: Mapping[str, object]) -> bytes:
    return _event_stream_message(
        _string_headers({":event-type": "chunk", ":content-type": "application/json", ":message-type": "event"}),
        json.dumps({"bytes": _base64_json(event)}, separators=(",", ":")).encode(),
    )


@dataclass(slots=True)
class ConverseStreamFrame:
    """One frame of a Converse event stream as the proxy relays it.

    ``payload`` is the parsed JSON of an event frame; exception frames and bytes that do not decode keep it None and
    are relayed as received. ``ends_stream`` marks the empty item the framer adds once the upstream has ended, so an
    exception frame or a closed connection ends the stream as ``messageStop`` does. ``guardrailed_payload`` is set
    when a guardrail rewrote the frame's text."""

    raw: bytes
    event_type: str | None = None
    payload: Mapping[str, object] | None = None
    guardrailed_payload: Mapping[str, object] | None = None
    ends_stream: bool = False

    @property
    def block_index(self) -> int | None:
        index: Final = self.payload.get("contentBlockIndex") if self.payload is not None else None
        return index if isinstance(index, int) else None

    def scanned_texts(self) -> tuple[tuple[_StreamGroupKey, str], ...]:
        current: Final = self.guardrailed_payload if self.guardrailed_payload is not None else self.payload
        return _delta_frame_texts(current) if self.event_type == "contentBlockDelta" and current is not None else ()

    def received_texts(self) -> tuple[tuple[_StreamGroupKey, str], ...]:
        return (
            _delta_frame_texts(self.payload)
            if self.event_type == "contentBlockDelta" and self.payload is not None
            else ()
        )

    def encode(self) -> bytes:
        if self.guardrailed_payload is None:
            return self.raw
        headers_length: Final = _unpack_uint32(self.raw[4:8])
        return _event_stream_message(
            self.raw[12 : 12 + headers_length], json.dumps(self.guardrailed_payload, separators=(",", ":")).encode()
        )


def _anthropic_sse(event: Mapping[str, object] | None) -> bytes:
    if event is None:
        return b""
    return f"event: {event['type']}\ndata: {json.dumps(event, separators=(',', ':'))}\n\n".encode()


@dataclass(slots=True)
class InvokeStreamFrame:
    """One frame of an invoke-with-response-stream as the proxy relays it.

    ``event`` is the Anthropic Messages stream event a ``chunk`` frame carries base64 encoded in the ``bytes`` of its
    ``payload``. Exception frames and bytes that do not decode keep it None and are relayed as received.
    ``ends_stream`` marks the empty item the framer adds once the upstream has ended, and ``guardrailed_event`` is set
    when a guardrail rewrote the event."""

    raw: bytes
    payload: Mapping[str, object] | None = None
    event: Mapping[str, object] | None = None
    guardrailed_event: Mapping[str, object] | None = None
    ends_stream: bool = False

    def sse(self) -> bytes:
        """The event as a ``/v1/messages`` stream carries it, the shape its guardrail translation scans."""
        return _anthropic_sse(self.guardrailed_event if self.guardrailed_event is not None else self.event)

    def received_sse(self) -> bytes:
        """The event as received from the upstream, before any guardrail rewrote it."""
        return _anthropic_sse(self.event)

    def encode(self) -> bytes:
        if self.guardrailed_event is None or self.payload is None:
            return self.raw
        headers_length: Final = _unpack_uint32(self.raw[4:8])
        return _event_stream_message(
            self.raw[12 : 12 + headers_length],
            json.dumps({**self.payload, "bytes": _base64_json(self.guardrailed_event)}, separators=(",", ":")).encode(),
        )


_JSON_OBJECT: Final = TypeAdapter(Mapping[str, object])
_Frame: Final = TypeVar("_Frame", ConverseStreamFrame, InvokeStreamFrame)


def _event_message(raw: bytes) -> tuple[str, bytes] | None:
    """The event type and payload of one event-stream frame, or None for an exception frame or bytes that do not
    decode."""
    from botocore.eventstream import EventStreamBuffer, ParserError

    buffer: Final = EventStreamBuffer()
    buffer.add_data(raw)
    try:
        message: Final = next(iter(buffer))
    except (ParserError, StopIteration) as e:
        verbose_proxy_logger.debug(
            "BedrockPassthroughGuardrailHandler: could not decode event-stream frame, relaying it unmodified: %s", e
        )
        return None
    # botocore-stubs types EventStreamMessage.headers as int; at runtime it is the decoded header dict.
    headers: Final = _JSON_OBJECT.validate_python(message.headers)
    event_type: Final = headers.get(":event-type")
    if headers.get(":message-type") != "event" or not isinstance(event_type, str):
        return None
    return event_type, message.payload


def _decode_converse_frame(raw: bytes) -> ConverseStreamFrame:
    message: Final = _event_message(raw)
    if message is None:
        return ConverseStreamFrame(raw=raw)
    event_type, payload_bytes = message
    try:
        payload: Final = _JSON_OBJECT.validate_python(json.loads(payload_bytes))
    except (json.JSONDecodeError, UnicodeDecodeError, ValidationError):
        return ConverseStreamFrame(raw=raw, event_type=event_type)
    return ConverseStreamFrame(raw=raw, event_type=event_type, payload=payload)


def _decode_invoke_frame(raw: bytes) -> InvokeStreamFrame:
    message: Final = _event_message(raw)
    if message is None or message[0] != "chunk":
        return InvokeStreamFrame(raw=raw)
    try:
        payload: Final = _JSON_OBJECT.validate_python(json.loads(message[1]))
        encoded: Final = payload.get("bytes")
        event: Final = (
            _JSON_OBJECT.validate_python(json.loads(base64.b64decode(encoded))) if isinstance(encoded, str) else None
        )
    except (ValueError, ValidationError):
        return InvokeStreamFrame(raw=raw)
    if event is None or not isinstance(event.get("type"), str):
        return InvokeStreamFrame(raw=raw)
    return InvokeStreamFrame(raw=raw, payload=payload, event=event)


async def _event_stream_frames(
    chunks: AsyncIterable[bytes], frame_type: type[_Frame], decode: Callable[[bytes], _Frame]
) -> AsyncGenerator[_Frame, None]:
    """Cut the upstream bytes into frames as soon as each frame has fully arrived, then add an empty item that ends
    the stream once the upstream has ended.

    Bytes that cannot be framed (a length below the 16-byte minimum, or a stream that ends inside a frame) are relayed
    as received."""
    pending = b""  # rebind-ok: bytes received but not yet framed
    framing = True  # rebind-ok: cleared once a frame length is unusable
    async for chunk in chunks:
        if not framing:
            yield frame_type(raw=chunk)
            continue
        pending += chunk
        while len(pending) >= 4:
            total_length = _unpack_uint32(pending[:4])
            if total_length < 16:
                framing = False
                break
            if len(pending) < total_length:
                break
            yield decode(pending[:total_length])
            pending = pending[total_length:]
        if not framing and pending:
            yield frame_type(raw=pending)
            pending = b""
    if pending:
        yield frame_type(raw=pending)
    yield frame_type(raw=b"", ends_stream=True)


def _converse_frames(items: Sequence[object]) -> tuple[ConverseStreamFrame, ...]:
    """The frames among ``items``, with the bytes an earlier guardrail's block message added read back into frames."""
    return tuple(
        item if isinstance(item, ConverseStreamFrame) else _decode_converse_frame(item)
        for item in items
        if isinstance(item, (ConverseStreamFrame, bytes))
    )


def _reads_stream_marks(guardrail: "CustomGuardrail") -> bool:
    return getattr(guardrail, "reads_stream_marks", False) is True


def _json_object_member(value: object, key: str) -> Mapping[str, object] | None:
    """``value[key]`` when both are JSON objects, else None."""
    try:
        return _JSON_OBJECT.validate_python(_JSON_OBJECT.validate_python(value).get(key))
    except ValidationError:
        return None


def _reject_json_constant(name: str) -> object:
    raise ValueError(f"{name} is not JSON")


def _finished_tool_input(text: str) -> object:
    """A tool call's streamed input as the JSON document Converse returns, or the text as received when it is not
    JSON (cut short, rewritten by a guardrail, or holding NaN or nesting too deep to send on)."""
    try:
        return json.loads(text, parse_constant=_reject_json_constant) if text else {}
    except (ValueError, RecursionError):
        return text


def _finished_converse_content(
    frames: Sequence[ConverseStreamFrame], groups: _StreamTextGroups
) -> list[dict[str, object]]:  # mutable-ok: the content of the finished response guardrails receive
    """The finished reply's text, reasoning and tool-call blocks, shaped as a non-streaming Converse response
    holds them."""
    texts: Final = dict(zip(groups.groups, groups.texts()))
    tool_starts: Final[dict[int | None, Mapping[str, object]]] = {}  # mutable-ok: built in one pass over the frames
    keys: Final[dict[_StreamGroupKey, None]] = {}  # mutable-ok: blocks in stream order, built in one pass
    for frame in frames:
        tool_use = (
            _json_object_member(_json_object_member(frame.payload, "start"), "toolUse")
            if frame.event_type == "contentBlockStart"
            else None
        )
        if tool_use is not None:
            tool_starts[frame.block_index] = tool_use
            keys[(frame.block_index, "tool")] = None
        keys.update((key, None) for key, _ in frame.scanned_texts())
    content: Final[list[dict[str, object]]] = []  # mutable-ok: built in one pass over the blocks
    for block_index, kind in keys:
        text = texts.get((block_index, kind), "")
        if kind == "tool":
            tool_start = tool_starts.get(block_index, {})
            named = {name: tool_start[name] for name in ("toolUseId", "name", "type") if name in tool_start}
            content.append({"toolUse": {**named, "input": _finished_tool_input(text)}})
        elif text and kind == "reasoning":
            content.append({"reasoningContent": {"reasoningText": {"text": text}}})
        elif text:
            content.append({"text": text})
    return content


def _stream_marked_request_data(
    request_data: Mapping[str, object], frames: Sequence[ConverseStreamFrame], groups: _StreamTextGroups, ended: bool
) -> dict[str, object]:  # mutable-ok: the request dict guardrails receive
    """``request_data`` with the marks chat completions gives a streamed answer: ``stream`` on every scan and, once
    the stream has ended, the finished ``response``."""
    if not ended:
        return {**request_data, "stream": True}
    stop_reason: Final = next(
        (
            frame.payload.get("stopReason")
            for frame in frames
            if frame.event_type == "messageStop" and frame.payload is not None
        ),
        None,
    )
    message: Final = {"role": "assistant", "content": _finished_converse_content(frames, groups)}
    response: Final = {"output": {"message": message}, **({"stopReason": stop_reason} if stop_reason else {})}
    return {**request_data, "stream": True, "response": response}


def _invoke_frames(items: Sequence[object]) -> tuple[InvokeStreamFrame, ...]:
    return tuple(item for item in items if isinstance(item, InvokeStreamFrame))


def _sse_events(sse: bytes) -> tuple[Mapping[str, object], ...]:
    return tuple(
        _JSON_OBJECT.validate_python(json.loads(line[len("data:") :]))
        for line in sse.decode().splitlines()
        if line.startswith("data:")
    )


def _invoke_message_stopped(frames: Sequence[InvokeStreamFrame]) -> bool:
    """Whether a ``message_delta`` with a ``stop_reason`` came, as a ``/v1/messages`` stream ends."""
    return any(
        frame.event is not None
        and frame.event.get("type") == "message_delta"
        and isinstance(delta := frame.event.get("delta"), Mapping)
        and _JSON_OBJECT.validate_python(delta).get("stop_reason") is not None
        for frame in frames
    )


def _invoke_stream_has_ended(frames: Sequence[InvokeStreamFrame]) -> bool:
    """Ended on a ``message_delta`` with a ``stop_reason`` or once the upstream has ended."""
    return any(frame.ends_stream for frame in frames) or _invoke_message_stopped(frames)


_UPSTREAM_ENDED_SSE: Final = _anthropic_sse(
    {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
        "usage": {"output_tokens": 0},
    }
)


def _invoke_scan_sse(frames: Sequence[InvokeStreamFrame], events: Sequence[bytes]) -> tuple[bytes, ...]:
    """The ``/v1/messages`` events to scan for these frames, closed with a ``message_delta`` when the upstream ended
    without one so the translation scans them as a whole stream, tool calls included."""
    if any(frame.ends_stream for frame in frames) and not _invoke_message_stopped(frames):
        return (*events, _UPSTREAM_ENDED_SSE)
    return tuple(events)


async def _scan_invoke_stream(
    frames: Sequence[InvokeStreamFrame],
    guardrail_to_apply: "CustomGuardrail",
    litellm_logging_obj: Optional["LiteLLMLoggingObj"],
    user_api_key_dict: Optional["UserAPIKeyAuth"],
    request_data: dict[str, object] | None,  # mutable-ok: the request dict guardrails receive
) -> None:
    """Scan the Anthropic events relayed so far as ``/v1/messages`` scans its stream, with the request marked as
    streaming and those events as its ``responses``. Once the stream has ended, the rewrites are written back into the
    frames, which reach the caller only when the stream is held until moderated."""
    received: Final = tuple(frame.sse() for frame in frames)
    scan_events: Final = _invoke_scan_sse(frames, received)
    scanned: Final[list[object]] = list(scan_events)  # mutable-ok: the /v1/messages translation rewrites it in place
    if request_data is not None:
        request_data["stream"] = True  # rebind-ok: marks the request guardrails receive as a stream
        request_data["responses"] = scanned  # rebind-ok: each scan sees every event relayed so far
    await AnthropicMessagesHandler().process_output_streaming_response(  # pyright: ignore[reportUnknownMemberType]  # takes the untyped request dict
        responses_so_far=scanned,
        guardrail_to_apply=guardrail_to_apply,
        litellm_logging_obj=litellm_logging_obj,
        user_api_key_dict=user_api_key_dict,
        request_data=request_data,
        deliver_ended_stream_rewrites=_invoke_stream_has_ended(frames),
    )
    for frame, before, after in zip(frames, received, scanned):
        if isinstance(after, bytes) and after != before and (rewritten := _sse_events(after)):
            frame.guardrailed_event = rewritten[0]


def _invoke_block_frames(
    exc: "ModifyResponseException", stream_started: bool, frames: Sequence[InvokeStreamFrame]
) -> tuple[bytes, ...]:
    """The ``/v1/messages`` block events for the stream relayed so far, each in its own ``chunk`` frame."""
    sse_chunks: Final = AnthropicMessagesHandler().build_block_sse_chunks(
        exc, stream_started=stream_started, responses_so_far=[frame.sse() for frame in frames]
    )
    return tuple(_invoke_chunk(event) for event in _sse_events(b"".join(sse_chunks)))


def _write_back_stream_texts(
    frames: Sequence[ConverseStreamFrame], groups: _StreamTextGroups, rewritten_texts: Sequence[str]
) -> None:
    split: Final = _split_group_texts(groups, rewritten_texts)
    for frame_idx, frame in enumerate(frames):
        current = frame.guardrailed_payload if frame.guardrailed_payload is not None else frame.payload
        if frame.event_type != "contentBlockDelta" or current is None:
            continue
        rewritten = copy.deepcopy(current)
        changed = False
        for local_idx, (_, container, key) in enumerate(_collect_stream_delta_text_holders(rewritten.get("delta"))):
            new_text = split.get((frame_idx, local_idx))
            if new_text is not None and new_text != container[key]:
                container[key] = new_text
                changed = True
        if changed:
            frame.guardrailed_payload = rewritten


_INVOKE_DELTA_TEXT_FIELDS: Final = MappingProxyType(
    {
        "text_delta": ("text", "text"),
        "input_json_delta": ("tool", "partial_json"),
        "thinking_delta": ("reasoning", "thinking"),
    }
)


@dataclass(frozen=True, slots=True)
class _InvokeDeltaText:
    """The guardrailable text one Anthropic event's delta carries, keyed by its content block and kind like the
    Converse holders: text, tool input and thinking. Signatures and citations are never rewritten."""

    group: _StreamGroupKey
    delta: Mapping[str, object]
    field: str
    text: str


def _invoke_delta_text(event: Mapping[str, object] | None) -> _InvokeDeltaText | None:
    if event is None:
        return None
    try:
        delta: Final = _JSON_OBJECT.validate_python(event.get("delta"))
    except ValidationError:
        return None
    kind: Final = delta.get("type")
    field: Final = _INVOKE_DELTA_TEXT_FIELDS.get(kind) if isinstance(kind, str) else None
    text: Final = delta.get(field[1]) if field is not None else None
    if field is None or not isinstance(text, str):
        return None
    index: Final = event.get("index")
    return _InvokeDeltaText((index if isinstance(index, int) else None, field[0]), delta, field[1], text)


async def _post_call_rewritten_texts(
    groups: _StreamTextGroups,
    proxy_logging_obj: "ProxyLogging",
    user_api_key_dict: "UserAPIKeyAuth",
    data: dict[str, object],  # mutable-ok: the request dict post_call hooks receive
) -> Sequence[str] | None:
    """Run the post_call guardrails on a buffered stream's texts as one Converse response with a content block per
    group, and return each group's text as the guardrails left it, or None when the response came back in another
    shape."""
    synthetic_response: Final[dict] = {
        "output": {
            "message": {
                "role": "assistant",
                "content": [{"text": text} for text in groups.texts()],
            }
        },
        "stopReason": "end_turn",
    }

    processed: Final = await proxy_logging_obj.post_call_success_hook(
        data=data,
        user_api_key_dict=user_api_key_dict,
        response=synthetic_response,
    )

    if not isinstance(processed, dict):
        verbose_proxy_logger.debug(
            "BedrockPassthroughGuardrailHandler: post_call_success_hook returned %s, leaving event stream unmodified",
            type(processed).__name__,
        )
        return None

    try:
        processed_blocks: Final = processed["output"]["message"]["content"]
        return [processed_blocks[i]["text"] for i in range(len(groups.groups))]
    except (KeyError, IndexError, TypeError):
        return None


async def _guardrail_buffered_invoke_stream(
    body_bytes: bytes,
    proxy_logging_obj: "ProxyLogging",
    user_api_key_dict: "UserAPIKeyAuth",
    data: dict[str, object],  # mutable-ok: the request dict post_call hooks receive
) -> bytes:
    """Scan a whole Claude invoke stream the way the buffered Converse path scans its stream, and write the guardrail
    rewrites back into the ``chunk`` frames that carried each text."""

    async def received() -> AsyncGenerator[bytes, None]:
        yield body_bytes

    frames: Final = [frame async for frame in _event_stream_frames(received(), InvokeStreamFrame, _decode_invoke_frame)]
    texts: Final = [_invoke_delta_text(frame.event) for frame in frames]
    groups: Final = _group_stream_texts([[] if text is None else [(text.group, text.text)] for text in texts])
    if not groups.groups:
        return body_bytes
    rewritten_texts: Final = await _post_call_rewritten_texts(groups, proxy_logging_obj, user_api_key_dict, data)
    if rewritten_texts is None:
        return body_bytes
    split: Final = _split_group_texts(groups, rewritten_texts)
    for frame_idx, (frame, text) in enumerate(zip(frames, texts)):
        if (
            text is not None
            and frame.event is not None
            and (new_text := split.get((frame_idx, 0))) is not None
            and new_text != text.text
        ):
            frame.guardrailed_event = {**frame.event, "delta": {**text.delta, text.field: new_text}}
    return b"".join(frame.encode() for frame in frames)


class BedrockPassthroughGuardrailHandler(BaseTranslation):
    @staticmethod
    def is_event_stream_content_type(content_type: str) -> bool:
        return _EVENT_STREAM_CONTENT_TYPE in content_type

    @staticmethod
    def event_stream_media_type() -> str:
        return _EVENT_STREAM_MEDIA_TYPE

    @staticmethod
    def event_stream_endpoint_is_de_anonymizable(endpoint: str, model: str, body: object) -> bool:
        """The streams the buffered path can scan whole and rewrite, the same ones the relay scans."""
        return BedrockPassthroughGuardrailHandler.event_stream_endpoint_is_relayable(endpoint, model, body)

    @staticmethod
    def event_stream_endpoint_is_relayable(endpoint: str, model: str, body: object) -> bool:
        """Converse streams of every model, and invoke streams of Claude, whose chunks carry Anthropic Messages stream
        events. Claude is told by its model id or, for ids that do not name it such as application inference profile
        ARNs, by the ``anthropic_version`` Bedrock requires in every Claude Messages request body."""
        return _is_converse_endpoint(endpoint) or (
            _is_invoke_stream_endpoint(endpoint)
            and (
                BaseAWSLLM.get_bedrock_invoke_provider(model) == "anthropic"
                or (isinstance(body, Mapping) and "anthropic_version" in body)
            )
        )

    @staticmethod
    async def de_anonymize_event_stream(
        body_bytes: bytes,
        proxy_logging_obj: "ProxyLogging",
        user_api_key_dict: "UserAPIKeyAuth",
        data: dict[str, object],  # mutable-ok: the request dict post_call hooks receive
    ) -> bytes:
        if _is_invoke_stream_endpoint(str(data.get("endpoint") or "")):
            return await _guardrail_buffered_invoke_stream(body_bytes, proxy_logging_obj, user_api_key_dict, data)

        from botocore.eventstream import EventStreamBuffer

        frames: Final[list[_StreamFrame]] = []
        offset = 0

        while offset + 16 <= len(body_bytes):
            total_length = _unpack_uint32(body_bytes[offset : offset + 4])
            if total_length < 16 or offset + total_length > len(body_bytes):
                break
            frame_raw = body_bytes[offset : offset + total_length]
            offset += total_length

            try:
                buf = EventStreamBuffer()
                buf.add_data(frame_raw)
                msg = next(iter(buf))
                event_type = msg.headers.get(":event-type")
                payload_bytes = msg.payload
            except Exception as e:
                verbose_proxy_logger.debug(
                    "BedrockPassthroughGuardrailHandler: could not decode event-stream "
                    "frame, forwarding it unmodified: %s",
                    e,
                )
                frames.append({"raw": frame_raw, "texts": []})
                continue

            texts: Sequence[tuple[_StreamGroupKey, str]] = ()
            if event_type == "contentBlockDelta":
                try:
                    payload_dict: dict[str, object] = json.loads(payload_bytes)
                    texts = _delta_frame_texts(payload_dict)
                except Exception as e:
                    verbose_proxy_logger.debug(
                        "BedrockPassthroughGuardrailHandler: could not parse "
                        "contentBlockDelta payload, forwarding frame unmodified: %s",
                        e,
                    )

            frames.append({"raw": frame_raw, "texts": texts})

        trailing_bytes: Final = body_bytes[offset:]

        groups: Final = _group_stream_texts([frame["texts"] for frame in frames])
        if not groups.groups:
            return body_bytes

        de_anonymized_texts: Final = await _post_call_rewritten_texts(
            groups, proxy_logging_obj, user_api_key_dict, data
        )
        if de_anonymized_texts is None:
            return body_bytes

        new_text_map: Final = _split_group_texts(groups, de_anonymized_texts)

        result_parts: Final[list[bytes]] = []

        for frame_idx, frame in enumerate(frames):
            if not frame["texts"]:
                result_parts.append(frame["raw"])
                continue

            frame_raw = frame["raw"]
            orig_total = _unpack_uint32(frame_raw[0:4])
            orig_hdrs_len = _unpack_uint32(frame_raw[4:8])
            headers_bytes = frame_raw[12 : 12 + orig_hdrs_len]

            try:
                payload_dict = json.loads(frame_raw[12 + orig_hdrs_len : orig_total - 4])
                for local_idx, (_, container, key) in enumerate(
                    _collect_stream_delta_text_holders(payload_dict.get("delta"))
                ):
                    new_text = new_text_map.get((frame_idx, local_idx))
                    if new_text is not None:
                        container[key] = new_text
                new_payload = json.dumps(payload_dict, separators=(",", ":")).encode()
            except Exception:
                result_parts.append(frame_raw)
                continue

            result_parts.append(_event_stream_message(headers_bytes, new_payload))

        result_parts.append(trailing_bytes)
        return b"".join(result_parts)

    async def process_input_messages(
        self,
        data: dict,
        guardrail_to_apply: "CustomGuardrail",
        litellm_logging_obj: Optional["LiteLLMLoggingObj"] = None,
    ) -> Mapping[str, object]:
        endpoint: Final = data.get("endpoint", "")
        body: Final = data.get("data")

        if not _is_converse_endpoint(endpoint):
            return await _generic_passthrough_handler().process_input_messages(
                data=data,
                guardrail_to_apply=guardrail_to_apply,
                litellm_logging_obj=litellm_logging_obj,
            )

        if not isinstance(body, dict) or not isinstance(body.get("messages"), list):
            return data
        request_body: Final = _BODY_BY_REFERENCE.validate_python(body)

        reads_rows: Final = not _reads_flat_texts_only(guardrail_to_apply)
        scope: Final = _Scope(
            skip_system=effective_skip_system_message_for_guardrail(guardrail_to_apply),
            skip_tool=effective_skip_tool_message_for_guardrail(guardrail_to_apply),
            scan_only_tool_results=effective_scan_only_tool_results_for_guardrail(guardrail_to_apply),
        )
        texts, holders = _extract_converse_texts(
            request_body,
            scope.skip_system,
            scope.skip_tool,
            scope.scan_only_tool_results,
        )
        request: Final = parse_converse_request(request_body) if reads_rows else None
        rows: Final = converse_rows(request, include_tool_calls=not scope.skip_tool) if request else ()
        scoped_indices: Final = scoped_structured_message_indices(
            tuple(row.message for row in rows),
            scan_only_tool_results=scope.scan_only_tool_results,
            skip_system=scope.skip_system,
            skip_tool=scope.skip_tool,
        )

        if not texts and not scoped_indices:
            return data

        sent_messages: Final = [rows[index].message for index in scoped_indices]
        inputs: Final = GenericGuardrailAPIInputs(texts=list(texts))
        if sent_messages:
            inputs["structured_messages"] = json.loads(json.dumps(sent_messages))
        tools: Final = converse_tools(request) if request and not scope.scan_only_tool_results else ()
        if tools:
            inputs["tools"] = copy.deepcopy([tool.param for tool in tools])
        model: Final = data.get("model")
        if model:
            inputs["model"] = model

        guardrailed_inputs: Final = await guardrail_to_apply.apply_guardrail(
            inputs=inputs,
            request_data=data,
            input_type="request",
            logging_obj=litellm_logging_obj,
        )

        guardrailed_texts: Final = guardrailed_inputs.get("texts", [])
        guardrailed_messages: Final = guardrailed_inputs.get("structured_messages")
        rewritten_rows: Final = (
            guardrailed_messages
            if sent_messages and isinstance(guardrailed_messages, list) and guardrailed_messages != sent_messages
            else None
        )
        changed_rows: Final[frozenset[int]] = (
            frozenset()
            if rewritten_rows is None
            else frozenset(
                index
                for index, sent, returned in zip(scoped_indices, sent_messages, rewritten_rows)
                if sent != returned
            )
        )
        texts_follow_rows: Final = rewritten_rows is None or _texts_follow_rows(
            request_body, rows, texts, guardrailed_texts, changed_rows, scope
        )
        applied_texts: Final = guardrailed_texts if texts_follow_rows else texts
        row_edits: Final = (
            ()
            if rewritten_rows is None
            else structured_edits(
                rows, scoped_indices, rewritten_rows, partial(_texts_applied_rows, request_body, applied_texts, scope)
            )
        )
        returned_tools: Final = guardrailed_inputs.get("tools")
        tool_changes: Final = (
            ()
            if returned_tools is None or not reads_rows
            else tool_edits(tools, returned_tools, model if isinstance(model, str) else None)
        )
        holder_edits: Final = (
            None
            if row_edits is None
            or tool_changes is None
            or (
                not texts_follow_rows
                and _edits_drop_a_text_change(
                    request_body, (*row_edits, *tool_changes), texts, guardrailed_texts, scope
                )
            )
            else _holder_edits(
                request_body, (*row_edits, *tool_changes), _changed_leaves(holders, texts, applied_texts)
            )
        )
        if holder_edits is None:
            raise unappliable_request_rewrite(guardrail_to_apply.guardrail_name)
        if holder_edits or (applied_texts and list(applied_texts) != texts):
            blank_drops: Final = _blank_drops(request_body, _final_texts(holders, applied_texts, holder_edits))
            if applied_texts:
                _write_back_texts(applied_texts, holders)
            _apply(holder_edits)
            _apply(tuple(_blank_drop_write(drop) for drop in blank_drops))

        return data

    async def process_output_response(
        self,
        response: object,
        guardrail_to_apply: "CustomGuardrail",
        litellm_logging_obj: Optional["LiteLLMLoggingObj"] = None,
        user_api_key_dict: Optional["UserAPIKeyAuth"] = None,
        request_data: dict | None = None,
    ) -> object:
        endpoint: Final[object] = (request_data or {}).get("endpoint", "")
        if (
            isinstance(endpoint, str)
            and endpoint
            and not _is_converse_endpoint(endpoint)
            and not _is_invoke_stream_endpoint(endpoint)
        ):
            return await _generic_passthrough_handler().process_output_response(
                response=response,
                guardrail_to_apply=guardrail_to_apply,
                litellm_logging_obj=litellm_logging_obj,
                user_api_key_dict=user_api_key_dict,
                request_data=request_data,
            )

        if not isinstance(response, dict):
            return response

        output_message: Final = (
            response.get("output", {}).get("message", {}) if isinstance(response.get("output"), dict) else {}
        )
        content_blocks: Final = output_message.get("content") if isinstance(output_message, dict) else None

        if not isinstance(content_blocks, list):
            return response

        texts, holders = _extract_converse_output_texts(content_blocks)

        if not texts:
            return response

        await self._scan_output_texts(
            texts, holders, guardrail_to_apply, litellm_logging_obj, user_api_key_dict, request_data
        )
        return response

    async def _scan_output_texts(
        self,
        texts: list[str],
        holders: list[_StringHolder],
        guardrail_to_apply: "CustomGuardrail",
        litellm_logging_obj: Optional["LiteLLMLoggingObj"],
        user_api_key_dict: Optional["UserAPIKeyAuth"],
        request_data: dict[str, object] | None,  # mutable-ok: the request dict guardrails receive
    ) -> None:
        effective_request_data = request_data or {}
        if "litellm_metadata" not in effective_request_data and user_api_key_dict is not None:
            user_metadata: Final = self.transform_user_api_key_dict_to_metadata(user_api_key_dict)
            if user_metadata:
                effective_request_data = {
                    **effective_request_data,
                    "litellm_metadata": user_metadata,
                }

        inputs: Final = GenericGuardrailAPIInputs(texts=texts)
        model: Final = effective_request_data.get("model")
        if isinstance(model, str) and model:
            inputs["model"] = model

        guardrailed_inputs: Final = await guardrail_to_apply.apply_guardrail(
            inputs=inputs,
            request_data=effective_request_data,
            input_type="response",
            logging_obj=litellm_logging_obj,
        )

        guardrailed_texts: Final = guardrailed_inputs.get("texts", [])
        if guardrailed_texts:
            _write_back_texts(guardrailed_texts, holders)

    async def process_output_streaming_response(
        self,
        responses_so_far: Sequence[object],
        guardrail_to_apply: "CustomGuardrail",
        litellm_logging_obj: Optional["LiteLLMLoggingObj"] = None,
        user_api_key_dict: Optional["UserAPIKeyAuth"] = None,
        request_data: dict[str, object] | None = None,  # mutable-ok: the request dict guardrails receive
        stream_transform_sink: StreamTransformSink | None = None,
        deliver_ended_stream_rewrites: bool = False,
    ) -> Sequence[object]:
        """
        Scan the Converse frames relayed so far as the buffered path scans a whole response: one text per content
        block and kind. Once the stream has ended, guardrail rewrites are written back into the frames, which reach
        the caller only when the stream is held until moderated. A guardrail that sets ``reads_stream_marks`` gets
        a copy of the request marked as chat completions marks a streamed answer; the others get it unchanged.
        Such a guardrail also gets the ended scan of a reply with no text, such as a call to a tool that takes no
        input, with no texts to scan. Invoke frames are scanned as Anthropic Messages events.
        """
        invoke_frames: Final = _invoke_frames(responses_so_far)
        if invoke_frames:
            await _scan_invoke_stream(
                invoke_frames, guardrail_to_apply, litellm_logging_obj, user_api_key_dict, request_data
            )
            return responses_so_far
        frames: Final = _converse_frames(responses_so_far)
        groups: Final = _group_stream_texts([frame.scanned_texts() for frame in frames])
        ended: Final = self._check_streaming_has_ended(frames)
        reads_marks: Final = _reads_stream_marks(guardrail_to_apply)
        if not groups.groups:
            if request_data is not None and reads_marks and ended and _finished_converse_content(frames, groups):
                await self._scan_output_texts(
                    [],
                    [],
                    guardrail_to_apply,
                    litellm_logging_obj,
                    user_api_key_dict,
                    _stream_marked_request_data(request_data, frames, groups, ended),
                )
            return responses_so_far
        content: Final = [{"text": text} for text in groups.texts()]
        await self.process_output_response(  # pyright: ignore[reportUnknownMemberType]  # takes the untyped request dict
            response={"output": {"message": {"role": "assistant", "content": content}}},
            guardrail_to_apply=guardrail_to_apply,
            litellm_logging_obj=litellm_logging_obj,
            user_api_key_dict=user_api_key_dict,
            request_data=(
                _stream_marked_request_data(request_data, frames, groups, ended)
                if request_data is not None and reads_marks
                else request_data
            ),
        )
        if ended:
            _write_back_stream_texts(frames, groups, [block["text"] for block in content])
        return responses_so_far

    def needs_ended_stream_scan(self, guardrail: "CustomGuardrail", responses_so_far: Sequence[object]) -> bool:
        return _reads_stream_marks(guardrail)

    def get_streaming_scan_key(self, responses_so_far: Sequence[object]) -> StreamingScanKey | None:
        invoke_frames: Final = _invoke_frames(responses_so_far)
        if invoke_frames:
            return AnthropicMessagesHandler().get_streaming_scan_key(
                list(_invoke_scan_sse(invoke_frames, [frame.received_sse() for frame in invoke_frames]))
            )
        frames: Final = _converse_frames(responses_so_far)
        return StreamingScanKey(
            texts=_group_stream_texts([frame.received_texts() for frame in frames]).texts(),
            stream_ended=self._check_streaming_has_ended(frames),
        )

    def _check_streaming_has_ended(self, responses_so_far: Sequence[object]) -> bool:
        return _invoke_stream_has_ended(_invoke_frames(responses_so_far)) or any(
            frame.event_type == "messageStop" or frame.ends_stream for frame in _converse_frames(responses_so_far)
        )

    def build_block_sse_chunks(
        self,
        exc: "ModifyResponseException",
        stream_started: bool = False,
        responses_so_far: Sequence[object] | None = None,
    ) -> Sequence[bytes]:
        """
        End the message with the guardrail's message as a new text block and ``stopReason`` ``guardrail_intervened``,
        closing any content block left open and starting the message when nothing was relayed yet. An invoke stream
        gets the ``/v1/messages`` block events instead.
        """
        invoke_frames: Final = _invoke_frames(responses_so_far or ())
        if invoke_frames or _is_invoke_stream_endpoint(str(exc.request_data.get("endpoint") or "")):
            return _invoke_block_frames(exc, stream_started, invoke_frames)
        frames: Final = _converse_frames(responses_so_far or ())
        started: Final = frozenset(
            index
            for index in (
                frame.block_index for frame in frames if frame.event_type in ("contentBlockStart", "contentBlockDelta")
            )
            if index is not None
        )
        stopped: Final = frozenset(
            index
            for index in (frame.block_index for frame in frames if frame.event_type == "contentBlockStop")
            if index is not None
        )
        block_index: Final = max(started | stopped, default=-1) + 1
        return (
            *(
                ()
                if any(frame.event_type == "messageStart" for frame in frames)
                else (_converse_event("messageStart", {"role": "assistant"}),)
            ),
            *(_converse_event("contentBlockStop", {"contentBlockIndex": index}) for index in sorted(started - stopped)),
            _converse_event("contentBlockDelta", {"contentBlockIndex": block_index, "delta": {"text": exc.message}}),
            _converse_event("contentBlockStop", {"contentBlockIndex": block_index}),
            _converse_event("messageStop", {"stopReason": "guardrail_intervened"}),
        )

    @staticmethod
    def owns_event_stream_item(item: object) -> bool:
        return isinstance(item, (ConverseStreamFrame, InvokeStreamFrame))

    @staticmethod
    def event_stream_items(
        chunks: AsyncIterable[bytes], endpoint: str
    ) -> AsyncGenerator[ConverseStreamFrame | InvokeStreamFrame, None]:
        if _is_invoke_stream_endpoint(endpoint):
            return _event_stream_frames(chunks, InvokeStreamFrame, _decode_invoke_frame)
        return _event_stream_frames(chunks, ConverseStreamFrame, _decode_converse_frame)

    @staticmethod
    def encode_event_stream_item(item: object) -> bytes:
        if isinstance(item, (ConverseStreamFrame, InvokeStreamFrame)):
            return item.encode()
        if isinstance(item, bytes):
            return item
        raise TypeError(f"cannot relay {type(item).__name__} on a Bedrock event stream")

    @staticmethod
    def event_stream_error_item(message: str, status_code: int) -> bytes:
        return _converse_exception(message, status_code)
