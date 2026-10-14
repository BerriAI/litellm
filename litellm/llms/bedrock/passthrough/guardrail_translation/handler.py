import copy
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import partial, reduce
from itertools import accumulate, chain
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final, Optional, Protocol, TypeAlias

from pydantic import InstanceOf, TypeAdapter
from typing_extensions import ReadOnly, TypedDict

from litellm._logging import verbose_proxy_logger
from litellm.llms.base_llm.guardrail_translation.base_translation import BaseTranslation
from litellm.llms.base_llm.guardrail_translation.utils import (
    effective_scan_only_tool_results_for_guardrail,
    effective_skip_system_message_for_guardrail,
    effective_skip_tool_message_for_guardrail,
    scoped_structured_message_indices,
    unappliable_request_rewrite,
)
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
    from litellm.integrations.custom_guardrail import CustomGuardrail
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.llms.pass_through.guardrail_translation.handler import (
        PassThroughEndpointHandler,
    )
    from litellm.proxy.utils import ProxyLogging
    from litellm.types.proxy.auth.user_api_key_auth import UserAPIKeyAuth

_CONVERSE_ACTIONS: Final = frozenset({"converse", "converse-stream"})
_EVENT_STREAM_CONTENT_TYPE: Final = "vnd.amazon.eventstream"
_EVENT_STREAM_MEDIA_TYPE: Final = "application/vnd.amazon.eventstream"


def _is_converse_endpoint(endpoint: str) -> bool:
    parts: Final = endpoint.rstrip("/").split("/")
    return bool(parts) and parts[-1] in _CONVERSE_ACTIONS


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


@dataclass(frozen=True, slots=True)
class _DropToolConfig:
    pass


_Write: TypeAlias = _Assign | _DropToolConfig
_ContentHolder: TypeAlias = tuple[object, str]
_BlankDrop: TypeAlias = tuple[_ContentHolder, frozenset[int]]
_BODY_BY_REFERENCE: Final = TypeAdapter(InstanceOf[dict[str, object]])


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
    last, in the same order as in the row built from them.
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
    match key:
        case str():
            fields: Final = as_json_object(node)
            return None if fields is None else fields.get(key)
        case int():
            items: Final = as_json_array(node)
            return items[key] if items is not None and 0 <= key < len(items) else None


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
    ``toolChoice`` names a tool that is no longer there, and drops ``toolConfig`` when nothing is left.
    """
    tool_config: Final = _child(body, "toolConfig")
    tools: Final = as_json_array(_child(tool_config, "tools"))
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
    return _DropToolConfig() if all(map(_is_cache_point, rebuilt)) else _Assign((tool_config, "tools"), rebuilt)


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
    match write:
        case _Assign():
            return _leaf_identity(write.holder) not in flat_changes
        case _DropToolConfig():
            return True


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


def _tool_result_part(item: object) -> str | None:
    text: Final = _child(item, "text")
    if isinstance(text, str):
        return text
    fields: Final = as_json_object(item)
    return json.dumps(fields["json"]) if fields is not None and "json" in fields else None


def _tool_result_text(tool_result: object) -> str:
    parts: Final = as_json_array(_child(tool_result, "content")) or ()
    rendered: Final = "\n".join(part for item in parts if (part := _tool_result_part(item)) is not None).strip()
    return rendered or "<non-text tool result omitted>"


def _neutralized(block: object, guarded: bool) -> object:
    tool_use: Final = _child(block, "toolUse")
    if as_json_object(tool_use) is not None:
        arguments: Final = json.dumps(_child(tool_use, "input"))
        return {"text": f"[tool call {_child(tool_use, 'toolUseId')}: {_child(tool_use, 'name')}({arguments})]"}
    tool_result: Final = _child(block, "toolResult")
    if as_json_object(tool_result) is None:
        return block
    text: Final = f"[tool result for {_child(tool_result, 'toolUseId')}: {_tool_result_text(tool_result)}]"
    return {"guardContent": {"text": {"text": text}}} if guarded else {"text": text}


def _has_tool_block(content: Sequence[object]) -> bool:
    return any(_child(block, "toolUse") is not None or _child(block, "toolResult") is not None for block in content)


def _tool_config_drop(body: Mapping[str, object]) -> tuple[_Assign | _Remove, ...]:
    """Remove ``toolConfig`` and turn every ``toolUse`` and ``toolResult`` block into text."""
    guarded: Final = "guardrailConfig" in body
    turns: Final = as_json_array(body.get("messages")) or ()
    neutralized: Final = tuple(
        _Assign((message, "content"), [_neutralized(block, guarded) for block in content])
        for message in turns
        if (content := as_json_array(_child(message, "content"))) is not None and _has_tool_block(content)
    )
    return (*neutralized, _Remove((body, "toolConfig")))


def _write_back_structured_edits(body: Mapping[str, object], writes: Sequence[_Write]) -> None:
    _apply(tuple(write for write in writes if isinstance(write, _Assign)))
    if any(isinstance(write, _DropToolConfig) for write in writes):
        _apply(_tool_config_drop(body))


def _block_text_holder(block: object) -> _ContentHolder | None:
    if isinstance(_child(block, "text"), str):
        return block, "text"
    guarded: Final = _child(_child(block, "guardContent"), "text")
    return (guarded, "text") if isinstance(_child(guarded, "text"), str) else None


def _final_texts(
    holders: Sequence[_StringHolder], applied_texts: Sequence[str], writes: Sequence[_Write]
) -> Mapping[_LeafIdentity, object]:
    flat: Final = {_leaf_identity(holder): text for holder, text in zip(holders, applied_texts)}
    structured: Final = {_leaf_identity(write.holder): write.value for write in writes if isinstance(write, _Assign)}
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


def _blank_drops(body: Mapping[str, object], final: Mapping[_LeafIdentity, object]) -> tuple[_BlankDrop, ...] | None:
    """The system and message text blocks the guardrail left blank, or None when dropping them empties a turn."""
    turns: Final = tuple(
        ((message, "content"), _blank_blocks(final, (message, "content")))
        for message in as_json_array(body.get("messages")) or ()
    )
    if any(_empties_turn(turn) for turn in turns):
        return None
    system: Final = ((body, "system"), _blank_blocks(final, (body, "system")))
    return tuple(drop for drop in (system, *turns) if drop[1])


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
    _write_back_structured_edits(rewritten, writes)
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
    import struct

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


class BedrockPassthroughGuardrailHandler(BaseTranslation):
    @staticmethod
    def is_event_stream_content_type(content_type: str) -> bool:
        return _EVENT_STREAM_CONTENT_TYPE in content_type

    @staticmethod
    def event_stream_media_type() -> str:
        return _EVENT_STREAM_MEDIA_TYPE

    @staticmethod
    def event_stream_endpoint_is_de_anonymizable(endpoint: str) -> bool:
        return _is_converse_endpoint(endpoint)

    @staticmethod
    async def de_anonymize_event_stream(
        body_bytes: bytes,
        proxy_logging_obj: "ProxyLogging",
        user_api_key_dict: "UserAPIKeyAuth",
        data: dict,
    ) -> bytes:
        import json as _json
        import struct
        from binascii import crc32 as esm_crc32

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

            texts: list[tuple[_StreamGroupKey, str]] = []
            if event_type == "contentBlockDelta":
                try:
                    payload_dict: dict[str, object] = _json.loads(payload_bytes)
                    block_index = payload_dict.get("contentBlockIndex")
                    # Group per content block: a guardrail edit that changes length must not move text between blocks.
                    texts = [
                        ((block_index if isinstance(block_index, int) else None, group_key), container[key])
                        for group_key, container, key in _collect_stream_delta_text_holders(payload_dict.get("delta"))
                    ]
                except Exception as e:
                    verbose_proxy_logger.debug(
                        "BedrockPassthroughGuardrailHandler: could not parse "
                        "contentBlockDelta payload, forwarding frame unmodified: %s",
                        e,
                    )

            frames.append({"raw": frame_raw, "texts": texts})

        trailing_bytes: Final = body_bytes[offset:]

        group_order: Final[list[_StreamGroupKey]] = []
        group_members: Final[dict[_StreamGroupKey, list[tuple[int, int]]]] = {}
        group_texts: Final[dict[_StreamGroupKey, list[str]]] = {}
        for frame_idx, frame in enumerate(frames):
            for local_idx, (group_key, text) in enumerate(frame["texts"]):
                if group_key not in group_members:
                    group_members[group_key] = []
                    group_texts[group_key] = []
                    group_order.append(group_key)
                group_members[group_key].append((frame_idx, local_idx))
                group_texts[group_key].append(text)

        active_groups: Final = [gk for gk in group_order if "".join(group_texts[gk])]
        if not active_groups:
            return body_bytes

        synthetic_response: Final[dict] = {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [{"text": "".join(group_texts[gk])} for gk in active_groups],
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
                "BedrockPassthroughGuardrailHandler: post_call_success_hook returned %s, "
                "leaving event stream unmodified",
                type(processed).__name__,
            )
            return body_bytes

        try:
            processed_blocks: Final = processed["output"]["message"]["content"]
            de_anonymized_texts: Final = [processed_blocks[i]["text"] for i in range(len(active_groups))]
        except (KeyError, IndexError, TypeError):
            return body_bytes

        new_text_map: Final[dict[tuple[int, int], str]] = {}
        for group_key, de_anonymized_text in zip(active_groups, de_anonymized_texts):
            members = group_members[group_key]
            orig_texts = group_texts[group_key]
            total_orig = sum(len(t) for t in orig_texts) or 1
            de_anon_len = len(de_anonymized_text)
            pos = 0
            for k, member in enumerate(members):
                if k == len(members) - 1:
                    new_text_map[member] = de_anonymized_text[pos:]
                else:
                    end = pos + round(de_anon_len * len(orig_texts[k]) / total_orig)
                    new_text_map[member] = de_anonymized_text[pos:end]
                    pos = end

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
                payload_dict = _json.loads(frame_raw[12 + orig_hdrs_len : orig_total - 4])
                for local_idx, (_, container, key) in enumerate(
                    _collect_stream_delta_text_holders(payload_dict.get("delta"))
                ):
                    new_text = new_text_map.get((frame_idx, local_idx))
                    if new_text is not None:
                        container[key] = new_text
                new_payload = _json.dumps(payload_dict, separators=(",", ":")).encode()
            except Exception:
                result_parts.append(frame_raw)
                continue

            new_total = 12 + orig_hdrs_len + len(new_payload) + 4
            prelude = struct.pack("!II", new_total, orig_hdrs_len)
            prelude_crc_val = esm_crc32(prelude) & 0xFFFFFFFF
            prelude_crc_b = struct.pack("!I", prelude_crc_val)
            part_for_msg_crc = prelude_crc_b + headers_bytes + new_payload
            msg_crc_val = esm_crc32(part_for_msg_crc, prelude_crc_val) & 0xFFFFFFFF
            msg_crc_b = struct.pack("!I", msg_crc_val)

            result_parts.append(prelude + prelude_crc_b + headers_bytes + new_payload + msg_crc_b)

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
            inputs["structured_messages"] = copy.deepcopy(sent_messages)
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
            if returned_tools is None
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
        blank_drops: Final = (
            None
            if holder_edits is None
            else _blank_drops(request_body, _final_texts(holders, applied_texts, holder_edits))
        )
        if holder_edits is None or blank_drops is None:
            raise unappliable_request_rewrite(guardrail_to_apply.guardrail_name)
        if applied_texts:
            _write_back_texts(applied_texts, holders)
        _write_back_structured_edits(request_body, holder_edits)
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
        endpoint: Final = (request_data or {}).get("endpoint", "")
        if endpoint and not _is_converse_endpoint(endpoint):
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

        effective_request_data = request_data or {}
        if "litellm_metadata" not in effective_request_data and user_api_key_dict is not None:
            user_metadata: Final = self.transform_user_api_key_dict_to_metadata(user_api_key_dict)
            if user_metadata:
                effective_request_data = {
                    **effective_request_data,
                    "litellm_metadata": user_metadata,
                }

        inputs: Final = GenericGuardrailAPIInputs(texts=texts)
        model: Final = effective_request_data.get("model") if effective_request_data else None
        if model:
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

        return response
