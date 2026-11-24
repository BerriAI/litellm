"""
OpenAI Responses API Handler for Unified Guardrails

This module provides a class-based handler for OpenAI Responses API format.
The class methods can be overridden for custom behavior.

Pattern Overview:
-----------------
1. Extract text content from input/output (both string and list formats)
2. Create async tasks to apply guardrails to each text segment
3. Track mappings to know where each response belongs
4. Apply guardrail responses back to the original structure

Responses API Format:
---------------------
Input: Union[str, List[Dict]] where each dict has:
  - role: str
  - content: Union[str, List[Dict]] (can have text items)
  - type: str (e.g., "message")

Output: response.output is List[GenericResponseOutputItem] where each has:
  - type: str (e.g., "message")
  - id: str
  - status: str
  - role: str
  - content: List[OutputText] where OutputText has:
    - type: str (e.g., "output_text")
    - text: str
"""

import asyncio
import copy
import json
import time
import uuid
from collections.abc import (
    AsyncIterable,
    Awaitable,
    Callable,
    Collection,
    Iterable,
    Iterator,
    Mapping,
    MutableSequence,
    Sequence,
)
from dataclasses import dataclass
from functools import partial
from itertools import accumulate, chain, groupby, repeat
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final, NamedTuple, Protocol, TypeAlias, TypeVar, Union, cast, runtime_checkable

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError
from typing_extensions import (
    ReadOnly,
    TypedDict,
    TypeIs,  # noqa: TID251  # TypeIs reaches typing only on 3.13
)

from litellm._logging import verbose_proxy_logger
from litellm.completion_extras.litellm_responses_transformation.transformation import (
    LiteLLMResponsesTransformationHandler,
    OpenAiResponsesToChatCompletionStreamIterator,
    tool_call_dict_from_output_item,
)
from litellm.litellm_core_utils.core_helpers import as_object_list, has_attributes
from litellm.litellm_core_utils.tool_fields import qualified_member_name
from litellm.llms.base_llm.guardrail_translation.base_translation import (
    BaseTranslation,
    StreamingScanKey,
    StreamTransformSink,
)
from litellm.llms.base_llm.guardrail_translation.utils import (
    blocked_responses_stream_usage,
    effective_scan_only_tool_results_for_guardrail,
    effective_scan_tool_calls_as_text_for_guardrail,
    effective_skip_system_message_for_guardrail,
    effective_skip_tool_message_for_guardrail,
    list_field,
    merge_guardrailed_scoped_messages,
    role_out_of_guardrail_scope,
    scanned_hosted_web_tool_call,
    scoped_structured_message_indices,
    stream_item_field,
    stream_item_fingerprint,
    stream_item_items,
    unappliable_request_rewrite,
    with_web_search_call_arguments,
)
from litellm.llms.openai.responses.guardrail_translation.tool_merge import (
    append_returned_tools,
    merge_guardrailed_tools,
)
from litellm.responses.additional_tools import additional_tools_in_input
from litellm.responses.litellm_completion_transformation.compaction_items import (
    compaction_summary,
    output_compaction_summary,
    with_compaction_summary,
)
from litellm.responses.litellm_completion_transformation.custom_tools import without_malformed_tool_flags
from litellm.responses.litellm_completion_transformation.reasoning_items import (
    thinking_block_texts,
    with_thinking_text,
)
from litellm.responses.litellm_completion_transformation.tool_search import is_server_tool_search_tool
from litellm.responses.litellm_completion_transformation.transformation import (
    TOOL_CALL_OUTPUT_ITEM_TYPES,
    LiteLLMCompletionResponsesConfig,
    tool_output_to_chat_content,
)
from litellm.types.llms.base import LiteLLMBaseModel
from litellm.types.llms.openai import (
    AllMessageValues,
    BaseLiteLLMOpenAIResponseObject,
    ChatCompletionToolCallChunk,
    ChatCompletionToolCallFunctionChunk,
    ChatCompletionToolParam,
    ContentPartAddedEvent,
    ContentPartDoneEvent,
    ContentPartDonePartOutputText,
    ErrorEvent,
    ErrorEventError,
    OutputItemAddedEvent,
    OutputItemDoneEvent,
    OutputTextDeltaEvent,
    OutputTextDoneEvent,
    ResponseAPIUsage,
    ResponseCompactionItem,
    ResponseCompletedEvent,
    ResponsesAPIResponse,
    ResponsesAPIStreamEvents,
    ResponsesAPIStreamingResponse,
)
from litellm.types.responses.main import (
    GenericResponseOutputItem,
    OutputText,
)
from litellm.types.utils import GenericGuardrailAPIInputs

if TYPE_CHECKING:
    from fastapi import HTTPException

    from litellm.integrations.custom_guardrail import (
        CustomGuardrail,
        ModifyResponseException,
    )
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.types.llms.openai import ResponseInputParam
    from litellm.types.proxy.auth.user_api_key_auth import UserAPIKeyAuth


class _MessageContentPosition(NamedTuple):
    output_index: int
    content_index: int


@dataclass(frozen=True, slots=True)
class _CompactionSummaryPosition:
    output_index: int


@dataclass(frozen=True, slots=True)
class _ReasoningSummaryPosition:
    output_index: int
    summary_index: int


@dataclass(frozen=True, slots=True)
class _ReasoningContentPosition:
    output_index: int
    content_index: int
    streams_as_summary: bool


@dataclass(frozen=True, slots=True)
class _ThinkingBlockPosition:
    output_index: int
    block_index: int


@dataclass(frozen=True, slots=True)
class _ProviderThinkingBlockPosition:
    block_index: int


_AnswerPosition: TypeAlias = _MessageContentPosition | _CompactionSummaryPosition
_ReasoningPosition: TypeAlias = _ReasoningSummaryPosition | _ReasoningContentPosition | _ThinkingBlockPosition
_OutputItemPosition: TypeAlias = _AnswerPosition | _ReasoningPosition
_ReasoningTextPosition: TypeAlias = _ReasoningPosition | _ProviderThinkingBlockPosition
_OutputPosition: TypeAlias = _OutputItemPosition | _ProviderThinkingBlockPosition
_REASONING_POSITION_TYPES: Final = (_ReasoningSummaryPosition, _ReasoningContentPosition, _ThinkingBlockPosition)


def _reasoning_output_texts(item: object, output_index: int) -> tuple[tuple[_ReasoningPosition, str], ...]:
    if stream_item_field(item, "type") != "reasoning":
        return ()
    summary_parts: Final = stream_item_items(item, "summary")
    summary_texts: Final = tuple(
        (_ReasoningSummaryPosition(output_index, summary_index), summary_text)
        for summary_index, part in enumerate(summary_parts)
        if isinstance(summary_text := stream_item_field(part, "text"), str) and summary_text
    )
    content_texts: Final = tuple(
        (_ReasoningContentPosition(output_index, content_index, streams_as_summary=not summary_parts), content_text)
        for content_index, part in enumerate(stream_item_items(item, "content"))
        if isinstance(content_text := stream_item_field(part, "text"), str) and content_text
    )
    thinking_texts: Final = tuple(
        (_ThinkingBlockPosition(output_index, block_index), text)
        for block_index, text in thinking_block_texts(stream_item_field(item, "encrypted_content")).items()
    )
    return (*summary_texts, *content_texts, *thinking_texts)


def _provider_thinking_blocks(response: object) -> tuple[object, ...]:
    return stream_item_items(stream_item_field(response, "provider_specific_fields"), "thinking_blocks")


def _provider_thinking_texts(response: object) -> tuple[tuple[_ProviderThinkingBlockPosition, str], ...]:
    return tuple(
        (_ProviderThinkingBlockPosition(block_index), text)
        for block_index, block in enumerate(_provider_thinking_blocks(response))
        if isinstance(text := stream_item_field(block, "thinking"), str) and text
    )


def _answer_output_items(outputs: Sequence[object]) -> tuple[tuple[int, object], ...]:
    return tuple((index, item) for index, item in enumerate(outputs) if stream_item_field(item, "type") != "reasoning")


def _reasoning_text_inputs(texts: Sequence[str], model: object) -> GenericGuardrailAPIInputs:
    return (
        GenericGuardrailAPIInputs(texts=list(texts), model=model)
        if isinstance(model, str) and model
        else GenericGuardrailAPIInputs(texts=list(texts))
    )


async def _guardrailed_reasoning_texts(
    texts: tuple[str, ...],
    model: object,
    guardrail_to_apply: "CustomGuardrail",
    litellm_logging_obj: "LiteLLMLoggingObj | None",
    request_data: dict[str, object],
) -> Sequence[str]:
    guardrailed: Final = await guardrail_to_apply.apply_guardrail(
        inputs=_reasoning_text_inputs(texts, model),
        request_data=request_data,
        input_type="response",
        logging_obj=litellm_logging_obj,
    )
    return guardrailed.get("texts") or texts


async def _scan_reasoning_batch(
    batch: tuple[str, ...], scan: Callable[[tuple[str, ...]], Awaitable[Sequence[str]]]
) -> tuple[str, ...]:
    texts: Final = await scan(batch)
    if len(texts) == len(batch) or len(batch) == 1:
        return tuple(texts[: len(batch)])
    single_texts: Final = await asyncio.gather(*(scan((text,)) for text in batch))
    return tuple(guardrailed[0] for guardrailed in single_texts)


def _response_reasoning_texts(response: object) -> tuple[tuple[_ReasoningTextPosition, str], ...]:
    item_texts: Final = chain.from_iterable(
        _reasoning_output_texts(output_item, output_index)
        for output_index, output_item in enumerate(stream_item_items(response, "output"))
    )
    return (*item_texts, *_provider_thinking_texts(response))


class _ToolCallShape(NamedTuple):
    name: str | None
    arguments: str


class _ToolCallFunctionFields(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    name: str | None = None
    arguments: str = ""


class _ToolCallFields(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    function: _ToolCallFunctionFields


def _tool_call_shapes(tool_calls: Sequence[ChatCompletionToolCallChunk]) -> tuple[_ToolCallShape, ...]:
    return tuple(
        _ToolCallShape(name=tool_call["function"].get("name"), arguments=tool_call["function"].get("arguments", ""))
        for tool_call in tool_calls
    )


def _returned_tool_call_shape(tool_call: object) -> _ToolCallShape | None:
    payload: Final = tool_call.model_dump() if isinstance(tool_call, BaseModel) else tool_call
    try:
        fields: Final = _ToolCallFields.model_validate(payload)
    except ValidationError:
        return None
    return _ToolCallShape(name=fields.function.name, arguments=fields.function.arguments)


def _post_guardrail_tool_call_shapes(
    returned_tool_calls: Sequence[object] | None,
    pre_guardrail_tool_calls: tuple[_ToolCallShape, ...],
    guardrail_name: str | None,
) -> tuple[_ToolCallShape, ...]:
    if not pre_guardrail_tool_calls:
        return pre_guardrail_tool_calls
    if returned_tool_calls is None or len(returned_tool_calls) != len(pre_guardrail_tool_calls):
        verbose_proxy_logger.warning(
            "OpenAI Responses API: guardrail %s returned %s tool calls for the %d scanned, "
            "leaving the tool call output items unchanged",
            guardrail_name,
            "no" if returned_tool_calls is None else len(returned_tool_calls),
            len(pre_guardrail_tool_calls),
        )
        return pre_guardrail_tool_calls
    returned_shapes: Final = tuple(_returned_tool_call_shape(tool_call) for tool_call in returned_tool_calls)
    validated_shapes: Final = tuple(shape for shape in returned_shapes if shape is not None)
    if len(validated_shapes) != len(returned_shapes):
        verbose_proxy_logger.warning(
            "OpenAI Responses API: guardrail %s returned tool calls without a function name and arguments, "
            "leaving the tool call output items unchanged",
            guardrail_name,
        )
        return pre_guardrail_tool_calls
    return validated_shapes


def _tool_call_rewrite(before: _ToolCallShape, after: _ToolCallShape) -> _ToolCallShape:
    return _ToolCallShape(name=after.name if after.name != before.name else None, arguments=after.arguments)


def _undeliverable_tool_call_rewrite_reason(
    call_ids: Sequence[str],
    tool_call_item_count: int,
    post_guardrail_tool_call_count: int,
    unresolved_argument_event: bool,
    rewritten_call_ids: frozenset[str],
    event_call_ids: frozenset[str],
) -> str | None:
    if len(call_ids) != tool_call_item_count:
        return (
            f"{tool_call_item_count - len(call_ids)} of the stream's {tool_call_item_count} tool call items "
            "carry no call_id"
        )
    if len(frozenset(call_ids)) != len(call_ids):
        return "the stream's tool call items repeat a call_id"
    if len(call_ids) != post_guardrail_tool_call_count:
        return (
            f"the guardrail returned {post_guardrail_tool_call_count} tool calls for the stream's "
            f"{len(call_ids)} tool call items"
        )
    if unresolved_argument_event:
        return "a tool call argument event names an item_id that no output_item event introduced"
    missing_call_ids: Final = sorted(rewritten_call_ids - event_call_ids)
    if missing_call_ids:
        return f"no stream event carries the rewritten call_id {', '.join(missing_call_ids)}"
    return None


class ResponseOutputEnvelope(TypedDict, total=False):
    """Dict form of a Responses API response, as far as guardrail write-back reads it."""

    output: ReadOnly[Sequence[object]]
    model: ReadOnly[str | None]


class ResponsesStreamChunk(TypedDict, total=False):
    """Responses API streaming event, as far as the accumulated-stream helpers read it."""

    type: ReadOnly[str]
    text: ReadOnly[str]
    delta: ReadOnly[str]
    item_id: ReadOnly[str]
    output_index: ReadOnly[int]
    content_index: ReadOnly[int]
    summary_index: ReadOnly[int]


class _StreamEventFields(Protocol):
    def get(self, key: str, /) -> object: ...


def _stream_event_field(event: _StreamEventFields, field: str) -> object:
    value: Final = event.get(field)
    return stream_item_field(event, field) if value is None else value


_TERMINAL_ENVELOPE_EVENT_TYPES: Final = frozenset(
    {
        ResponsesAPIStreamEvents.RESPONSE_COMPLETED.value,
        ResponsesAPIStreamEvents.RESPONSE_FAILED.value,
        ResponsesAPIStreamEvents.RESPONSE_INCOMPLETE.value,
    }
)


_TOOL_CALL_ITEM_TYPES: Final = frozenset({"function_call", "custom_tool_call"})
# Every output item the client acts on, as the Responses stream iterator classes them. Wider than the items
# whose payload a guardrail scans.
_CLIENT_RUN_TOOL_CALL_ITEM_TYPES: Final = _TOOL_CALL_ITEM_TYPES | {
    "mcp_approval_request",
    "computer_call",
    "local_shell_call",
    "shell_call",
    "apply_patch_call",
}
_TOOL_SEARCH_CALL_TYPE: Final = "tool_search_call"
_WEB_SEARCH_CALL_TYPE: Final = "web_search_call"
_ITEM_ID_KEYED_TOOL_CALL_TYPES: Final = frozenset({_TOOL_SEARCH_CALL_TYPE, _WEB_SEARCH_CALL_TYPE})
_OUTPUT_TOOL_CALL_ITEM_TYPES: Final = _TOOL_CALL_ITEM_TYPES | {_TOOL_SEARCH_CALL_TYPE}
_SCANNED_TOOL_SEARCH_NAME: Final = "tool_search"
_TOOL_SEARCH_ARGUMENTS_ADAPTER: Final = TypeAdapter(dict[str, object])
_RAW_ITEMS_ADAPTER: Final = TypeAdapter(tuple[object, ...])
_STREAM_EVENT_TYPE_ADAPTER: Final = TypeAdapter(str)
_TOOL_CALL_PAYLOAD_FIELDS: Final[Mapping[str, str]] = MappingProxyType(
    {"function_call": "arguments", "custom_tool_call": "input"}
)
_TOOL_CALL_PAYLOAD_DELTA_EVENT_TYPES: Final = frozenset(
    {"response.function_call_arguments.delta", "response.custom_tool_call_input.delta"}
)
_TOOL_CALL_PAYLOAD_DONE_EVENT_FIELDS: Final[Mapping[str, str]] = MappingProxyType(
    {"response.function_call_arguments.done": "arguments", "response.custom_tool_call_input.done": "input"}
)
_TOOL_CALL_PAYLOAD_EVENT_TYPES: Final = _TOOL_CALL_PAYLOAD_DELTA_EVENT_TYPES | frozenset(
    _TOOL_CALL_PAYLOAD_DONE_EVENT_FIELDS
)
_OUTPUT_ITEM_EVENT_TYPES: Final = frozenset({"response.output_item.added", "response.output_item.done"})
_OUTPUT_TEXT_EVENT_TYPES: Final = frozenset(
    {"response.output_text.delta", "response.output_text.done", "response.refusal.delta", "response.refusal.done"}
)
_TEXT_DELTA_EVENT_TYPES: Final = frozenset({"response.output_text.delta", "response.refusal.delta"})
_REASONING_SUMMARY_TEXT_EVENT_TYPES: Final = frozenset(
    {"response.reasoning_summary_text.delta", "response.reasoning_summary_text.done"}
)
_REASONING_TEXT_EVENT_TYPES: Final = frozenset({"response.reasoning_text.delta", "response.reasoning_text.done"})
_STREAM_TEXT_EVENT_TYPES: Final = (
    _OUTPUT_TEXT_EVENT_TYPES | _REASONING_SUMMARY_TEXT_EVENT_TYPES | _REASONING_TEXT_EVENT_TYPES
)
_TOOL_OUTPUT_TEXT_PART_TYPES: Final = frozenset({"input_text", "output_text", "text"})
_PATCHABLE_ITEM_FIELDS: Final[Mapping[str, str]] = MappingProxyType(
    {"function_call_output": "output", "custom_tool_call_output": "output", "message": "content"}
)


def _item_rewrite_field(item: Mapping[str, object]) -> str | None:
    item_type: Final = item.get("type")
    if item_type is None:
        return "content" if "content" in item else None
    if not isinstance(item_type, str):
        return None
    return _PATCHABLE_ITEM_FIELDS.get(item_type)


def _rewritten_compaction_item(item: Mapping[str, object], rewritten: object) -> Mapping[str, object] | None:
    encrypted_content: Final = item.get("encrypted_content")
    summary: Final = rewritten.get("content") if isinstance(rewritten, Mapping) else None
    if not isinstance(encrypted_content, str) or not isinstance(summary, str) or not summary:
        return None
    return {**item, "encrypted_content": with_compaction_summary(encrypted_content, summary)}


def _rewritten_input_item(item: Mapping[str, object], rewritten: object) -> Mapping[str, object] | None:
    if compaction_summary(item) is not None:
        return _rewritten_compaction_item(item, rewritten)
    field: Final = _item_rewrite_field(item)
    if field is None or not isinstance(rewritten, Mapping):
        return None
    rewritten_content: Final = rewritten.get("content")
    if isinstance(item.get(field), str) and isinstance(rewritten_content, str):
        return {**item, field: rewritten_content}
    rewritten_row: Final = cast("AllMessageValues", rewritten)  # cast-ok: guardrails hand back chat-shaped rows
    converted_items, _ = LiteLLMResponsesTransformationHandler().convert_chat_completion_messages_to_responses_api(
        [rewritten_row]
    )
    if len(converted_items) != 1 or not isinstance(converted_items[0], Mapping):
        return None
    first_converted: Final = cast("Mapping[str, object]", converted_items[0])  # cast-ok: isinstance-checked above
    converted_value: Final = first_converted.get(field)
    if converted_value is None:
        return None
    return {**item, field: converted_value}


def _is_tool_call_item(item: object) -> bool:
    return isinstance(item, Mapping) and item.get("type") in _TOOL_CALL_ITEM_TYPES


def _is_tool_search_item(item: object) -> bool:
    match item:
        case {"type": "tool_search_call" | "tool_search_output"}:
            return True
        case _:
            return False


def _joins_assistant_turn(item: object) -> bool:
    return _is_tool_call_item(item) or _is_tool_search_item(item)


def _hosted_web_tool_call(item: object, index: int) -> ChatCompletionToolCallChunk | None:
    hosted_call: Final = scanned_hosted_web_tool_call(item)
    if hosted_call is None:
        return None
    tool_id, name, arguments = hosted_call
    return ChatCompletionToolCallChunk(
        id=tool_id,
        type="function",
        function=ChatCompletionToolCallFunctionChunk(name=name, arguments=arguments),
        index=index,
    )


def _is_output_tool_call_item(item: object) -> bool:
    return (
        stream_item_field(item, "type") in _OUTPUT_TOOL_CALL_ITEM_TYPES
        or scanned_hosted_web_tool_call(item) is not None
    )


def _tool_call_output_item_mapping(item: object) -> Mapping[str, object] | None:
    if not _is_output_tool_call_item(item):
        return None
    if isinstance(item, Mapping):
        return cast("Mapping[str, object]", item)  # cast-ok: output items are str-keyed JSON objects
    return item.model_dump() if isinstance(item, BaseModel) else None


def _tool_call_item_key(item: object) -> str | None:
    call_id: Final = stream_item_field(item, "call_id")
    if isinstance(call_id, str) and call_id:
        return call_id
    item_id: Final = stream_item_field(item, "id")
    if stream_item_field(item, "type") in _ITEM_ID_KEYED_TOOL_CALL_TYPES and isinstance(item_id, str) and item_id:
        return item_id
    return None


def _is_tool_call_output_item(item: object) -> bool:
    return _tool_call_output_item_mapping(item) is not None


def _indexed_tool_call_output_items(outputs: Sequence[object]) -> tuple[tuple[int, object], ...]:
    return tuple((index, item) for index, item in enumerate(outputs) if _is_tool_call_output_item(item))


def _scanned_tool_call_item(item: Mapping[str, object]) -> Mapping[str, object]:
    """A ``tool_search_call`` reaches guardrails as a ``tool_search`` call whose arguments are JSON text."""
    if item.get("type") != _TOOL_SEARCH_CALL_TYPE:
        return item
    return {**item, "name": _SCANNED_TOOL_SEARCH_NAME, "arguments": json.dumps(item.get("arguments") or {})}


def _scanned_tool_call(item: Mapping[str, object], index: int) -> ChatCompletionToolCallChunk:
    web_tool_call: Final = _hosted_web_tool_call(item, index)
    if web_tool_call is not None:
        return web_tool_call
    return tool_call_dict_from_output_item(_scanned_tool_call_item(item), index)


def _rewritten_web_search_call(item: object, payload: str | None) -> object:
    return item if payload is None else with_web_search_call_arguments(item, payload)


def _tool_search_rewrite(payload: str) -> Mapping[str, object]:
    try:
        return {"arguments": _TOOL_SEARCH_ARGUMENTS_ADAPTER.validate_json(payload)}
    except ValidationError:
        return {"arguments": {}, "status": "incomplete"}


def _rewritten_tool_search_call(item: object, payload: str | None) -> object:
    if payload is None:
        return item
    update: Final = _tool_search_rewrite(payload)
    if isinstance(item, BaseModel):
        return item.model_copy(update=update)
    if isinstance(item, Mapping):
        return {**item, **update}
    return item


_RewriteKey = TypeVar("_RewriteKey")


def _delta_replacements(rewrites: Mapping[_RewriteKey, str]) -> Mapping[_RewriteKey, Iterator[str]]:
    return MappingProxyType({key: chain((rewritten,), repeat("")) for key, rewritten in rewrites.items()})


@dataclass(frozen=True, slots=True)
class _ReasoningStreamRewrites:
    summaries: Mapping[tuple[int, object], str]
    contents: Mapping[tuple[int, object], str]
    summary_deltas: Mapping[tuple[int, object], Iterator[str]]
    content_deltas: Mapping[tuple[int, object], Iterator[str]]


def _streamed_summary_index(position: _OutputItemPosition) -> int | None:
    match position:
        case _ReasoningSummaryPosition():
            return position.summary_index
        case _ReasoningContentPosition(streams_as_summary=True):
            return position.content_index
        case _:
            return None


def _reasoning_stream_rewrites(rewrites_by_position: Mapping[_OutputItemPosition, str]) -> _ReasoningStreamRewrites:
    summaries: Final[Mapping[tuple[int, object], str]] = MappingProxyType(
        {
            (position.output_index, summary_index): rewritten
            for position, rewritten in rewrites_by_position.items()
            if (summary_index := _streamed_summary_index(position)) is not None
        }
    )
    contents: Final[Mapping[tuple[int, object], str]] = MappingProxyType(
        {
            (position.output_index, position.content_index): rewritten
            for position, rewritten in rewrites_by_position.items()
            if isinstance(position, _ReasoningContentPosition)
        }
    )
    return _ReasoningStreamRewrites(
        summaries=summaries,
        contents=contents,
        summary_deltas=_delta_replacements(summaries),
        content_deltas=_delta_replacements(contents),
    )


def _deliver_into_output(outputs: Sequence[object], index: int, item: object, delivered: object) -> None:
    if delivered is not item and isinstance(outputs, MutableSequence):
        outputs[index] = delivered  # rebind-ok: a frozen output item takes a rewrite by being swapped in the list


def _released_tool_call_payload(responses_so_far: Sequence[object], item_id: object) -> str | None:
    events: Final = tuple(event for event in responses_so_far if stream_item_field(event, "item_id") == item_id)
    finished: Final = tuple(
        payload
        for event in events
        if isinstance(event_type := stream_item_field(event, "type"), str)
        and event_type in _TOOL_CALL_PAYLOAD_DONE_EVENT_FIELDS
        and isinstance(payload := stream_item_field(event, _TOOL_CALL_PAYLOAD_DONE_EVENT_FIELDS[event_type]), str)
    )
    if finished:
        return finished[-1]
    deltas: Final = tuple(
        delta
        for event in events
        if stream_item_field(event, "type") in _TOOL_CALL_PAYLOAD_DELTA_EVENT_TYPES
        and isinstance(delta := stream_item_field(event, "delta"), str)
    )
    return "".join(deltas) if deltas else None


def _with_released_payload(item: Mapping[str, object], responses_so_far: Sequence[object]) -> Mapping[str, object]:
    field: Final = _TOOL_CALL_PAYLOAD_FIELDS.get(str(item.get("type")))
    if field is None:
        return item
    payload: Final = _released_tool_call_payload(responses_so_far, item.get("id"))
    return item if payload is None else {**item, field: payload}


class _StreamPartKey(NamedTuple):
    item_id: object
    output_index: object
    content_index: object
    summary_index: object


def _stream_part_key(event: object) -> _StreamPartKey:
    return _StreamPartKey(
        item_id=stream_item_field(event, "item_id"),
        output_index=stream_item_field(event, "output_index"),
        content_index=stream_item_field(event, "content_index"),
        summary_index=stream_item_field(event, "summary_index") or 0,
    )


def _stream_event_whole_text(event: object) -> object | None:
    text: Final = stream_item_field(event, "text")
    return text if isinstance(text, str) else stream_item_field(event, "refusal")


def _carries_stream_text(event: object) -> bool:
    return isinstance(_stream_event_whole_text(event), str) or isinstance(stream_item_field(event, "delta"), str)


def _stream_part_texts(events: Sequence[object]) -> Mapping[_StreamPartKey, str]:
    keyed_events: Final = tuple(
        (_stream_part_key(event), _stream_event_whole_text(event), stream_item_field(event, "delta"))
        for event in events
        if _carries_stream_text(event)
    )

    def part_text(part_key: _StreamPartKey) -> str:
        done_texts: Final = tuple(text for key, text, _ in keyed_events if key == part_key and isinstance(text, str))
        if done_texts:
            return done_texts[-1]
        return "".join(delta for key, _, delta in keyed_events if key == part_key and isinstance(delta, str))

    return MappingProxyType({key: part_text(key) for key in dict.fromkeys(key for key, _, _ in keyed_events)})


def _events_of_types(events: Sequence[object], event_types: frozenset[str]) -> tuple[object, ...]:
    return tuple(event for event in events if stream_item_field(event, "type") in event_types)


def _indexed_text_parts(texts_by_index: Mapping[object, str], part_type: str) -> tuple[Mapping[str, str], ...]:
    last_index: Final = max((index for index in texts_by_index if isinstance(index, int)), default=-1)
    return tuple({"type": part_type, "text": texts_by_index.get(index, "")} for index in range(last_index + 1))


def _released_message_item(text: str) -> Mapping[str, object]:
    content: Final = [{"type": "output_text", "text": text}]
    return {"type": "message", "role": "assistant", "content": content}


def _released_message_items(text_events: Sequence[object]) -> Mapping[int, Mapping[str, object]]:
    part_texts: Final = _stream_part_texts(text_events)
    output_indexes: Final = frozenset(key.output_index for key in part_texts if isinstance(key.output_index, int))
    return MappingProxyType(
        {
            output_index: {
                "type": "message",
                "role": "assistant",
                "content": _indexed_text_parts(
                    {key.content_index: text for key, text in part_texts.items() if key.output_index == output_index},
                    "output_text",
                ),
            }
            for output_index in output_indexes
        }
    )


def _is_reasoning_event(event: object) -> bool:
    event_type: Final = stream_item_field(event, "type")
    if event_type in _OUTPUT_ITEM_EVENT_TYPES:
        return stream_item_field(stream_item_field(event, "item"), "type") == "reasoning"
    return event_type in _REASONING_SUMMARY_TEXT_EVENT_TYPES | _REASONING_TEXT_EVENT_TYPES


def _carries_whole_output_item(event: _StreamEventFields) -> bool:
    event_type: Final = _stream_event_field(event, "type")
    if event_type == "response.output_item.added":
        return stream_item_field(_stream_event_field(event, "item"), "type") == "reasoning"
    return event_type == "response.output_item.done"


def _released_parts(
    item_parts: Sequence[object], streamed_texts: Mapping[object, str], part_type: str
) -> tuple[object, ...]:
    streamed_parts: Final = _indexed_text_parts(streamed_texts, part_type)
    return tuple(
        streamed_parts[index] if index in streamed_texts or index >= len(item_parts) else item_parts[index]
        for index in range(max(len(item_parts), len(streamed_parts)))
    )


def _released_reasoning_item(item_events: Sequence[object]) -> Mapping[str, object]:
    announced_items: Final = tuple(
        stream_item_field(event, "item")
        for event in _events_of_types(item_events, _OUTPUT_ITEM_EVENT_TYPES)
        if _is_reasoning_event(event)
    )
    fields: Final = dict(chain.from_iterable(_incomplete_item_fields(item).items() for item in announced_items))
    summary_texts: Final = _stream_part_texts(_events_of_types(item_events, _REASONING_SUMMARY_TEXT_EVENT_TYPES))
    content_texts: Final = _stream_part_texts(_events_of_types(item_events, _REASONING_TEXT_EVENT_TYPES))
    summary: Final = _released_parts(
        stream_item_items(fields, "summary"),
        {key.summary_index: text for key, text in summary_texts.items()},
        "summary_text",
    )
    content: Final = _released_parts(
        stream_item_items(fields, "content"),
        {key.content_index: text for key, text in content_texts.items()},
        "reasoning_text",
    )
    return {
        **fields,
        "type": "reasoning",
        "summary": summary if summary_texts else fields.get("summary"),
        "content": content if content_texts else fields.get("content"),
    }


def _released_reasoning_items(responses_so_far: Sequence[object]) -> Mapping[int, Mapping[str, object]]:
    output_indexes: Final = tuple(
        dict.fromkeys(
            output_index
            for event in responses_so_far
            if _is_reasoning_event(event) and isinstance(output_index := stream_item_field(event, "output_index"), int)
        )
    )
    items: Final = tuple(
        (
            output_index,
            _released_reasoning_item(
                tuple(event for event in responses_so_far if stream_item_field(event, "output_index") == output_index)
            ),
        )
        for output_index in output_indexes
    )
    return MappingProxyType({index: item for index, item in items if _reasoning_output_texts(item, index)})


def _has_answer_stream_text(responses_so_far: Sequence[object]) -> bool:
    return any(_carries_stream_text(event) and not _is_reasoning_event(event) for event in responses_so_far)


def _reads_like_a_dict(chunk: object) -> bool:
    return hasattr(chunk, "get")


def _envelope_has_reasoning_text(chunk: object) -> bool:
    outputs: Final = stream_item_items(stream_item_field(chunk, "response"), "output")
    return any(_reasoning_output_texts(item, output_index) for output_index, item in enumerate(outputs))


def _reasoning_stream_slots(position: _ReasoningPosition) -> frozenset[tuple[str, int, int]]:
    match position:
        case _ReasoningSummaryPosition(output_index=output_index, summary_index=summary_index):
            return frozenset({("summary", output_index, summary_index)})
        case _ReasoningContentPosition(output_index=output_index, content_index=content_index, streams_as_summary=True):
            return frozenset({("content", output_index, content_index), ("summary", output_index, content_index)})
        case _ReasoningContentPosition(output_index=output_index, content_index=content_index):
            return frozenset({("content", output_index, content_index)})
        case _ThinkingBlockPosition(output_index=output_index, block_index=block_index):
            return frozenset({("thinking", output_index, block_index)})


def _envelope_holds_streamed_text(
    envelope_texts: Sequence[tuple[_ReasoningPosition, str]], position: _ReasoningPosition, text: str
) -> bool:
    streamed_slots: Final = _reasoning_stream_slots(position)
    return any(
        held.startswith(text) and not streamed_slots.isdisjoint(_reasoning_stream_slots(held_position))
        for held_position, held in envelope_texts
    )


def _reasoning_streamed_outside_envelope(
    responses_so_far: Sequence[object], envelope_texts: Sequence[tuple[_ReasoningTextPosition, str]]
) -> tuple[tuple[_ReasoningPosition, str], ...]:
    envelope_item_texts: Final = tuple(
        (position, text) for position, text in envelope_texts if isinstance(position, _REASONING_POSITION_TYPES)
    )
    streamed_texts: Final = chain.from_iterable(
        _reasoning_output_texts(item, output_index)
        for output_index, item in _released_reasoning_items(responses_so_far).items()
    )
    return tuple(
        (position, text)
        for position, text in streamed_texts
        if not _envelope_holds_streamed_text(envelope_item_texts, position, text)
    )


def _is_placeable_stream_text(
    event: object, reasoning_indexes: Collection[int], tool_item_ids: Collection[object]
) -> bool:
    event_type: Final = stream_item_field(event, "type")
    output_index: Final = stream_item_field(event, "output_index")
    has_content_index: Final = isinstance(stream_item_field(event, "content_index"), int)
    if event_type in _TOOL_CALL_PAYLOAD_EVENT_TYPES:
        return stream_item_field(event, "item_id") in tool_item_ids
    if not isinstance(output_index, int):
        return False
    if event_type in _OUTPUT_TEXT_EVENT_TYPES:
        return has_content_index and output_index not in reasoning_indexes
    if event_type in _REASONING_TEXT_EVENT_TYPES:
        return has_content_index
    return event_type in _REASONING_SUMMARY_TEXT_EVENT_TYPES


def _has_unplaceable_stream_text(responses_so_far: Sequence[object]) -> bool:
    reasoning_indexes: Final = frozenset(_released_reasoning_items(responses_so_far))
    tool_item_ids: Final = frozenset(item.get("id") for item in _released_tool_call_items(responses_so_far))
    return any(
        not _is_placeable_stream_text(event, reasoning_indexes, tool_item_ids)
        for event in responses_so_far
        if _carries_stream_text(event)
    )


def _released_tool_call_items(responses_so_far: Sequence[object]) -> tuple[Mapping[str, object], ...]:
    announced: Final = tuple(
        item
        for item in (
            _tool_call_output_item_mapping(stream_item_field(event, "item"))
            for event in responses_so_far
            if stream_item_field(event, "type") in _OUTPUT_ITEM_EVENT_TYPES
        )
        if item is not None
    )
    latest_by_id: Final = MappingProxyType({item.get("id"): item for item in announced})
    return tuple(_with_released_payload(item, responses_so_far) for item in latest_by_id.values())


def _released_stream_output(responses_so_far: Sequence[object]) -> tuple[Mapping[str, object], ...]:
    text_events: Final = _events_of_types(responses_so_far, _OUTPUT_TEXT_EVENT_TYPES)
    tool_items: Final = _released_tool_call_items(responses_so_far)
    reasoning_items: Final = _released_reasoning_items(responses_so_far)
    if not reasoning_items:
        released_text: Final = "".join(_stream_part_texts(text_events).values())
        message_items: Final = (_released_message_item(released_text),) if released_text else ()
        return (*message_items, *tool_items)
    stream_items: Final = {**_released_message_items(text_events), **reasoning_items}
    aligned_items: Final = tuple(
        stream_items.get(output_index, _released_message_item("")) for output_index in range(max(stream_items) + 1)
    )
    return (*aligned_items, *tool_items)


def _released_envelope(responses_so_far: Sequence[object]) -> Mapping[str, object]:
    response: Final = {"status": "incomplete", "output": list(_released_stream_output(responses_so_far))}
    return {"type": ResponsesAPIStreamEvents.RESPONSE_INCOMPLETE.value, "response": response}


def _last_message_role(messages: Sequence[object]) -> str | None:
    if not messages:
        return None
    last: Final = messages[-1]
    role: Final = last.get("role") if isinstance(last, Mapping) else getattr(last, "role", None)
    return role if isinstance(role, str) else None


def _provenance_unit_bounds(
    raw_input: Sequence[object],
    solo_conversions: Sequence[Sequence[object]],
) -> tuple[tuple[int, int], ...]:
    trailing_roles: Final = tuple(
        accumulate(
            (
                "assistant" if _is_tool_search_item(item) else _last_message_role(messages)
                for item, messages in zip(raw_input, solo_conversions)
            ),
            lambda previous, current: current if current is not None else previous,
        )
    )
    start_indexes: Final = tuple(
        index
        for index in range(len(raw_input))
        if index == 0 or not (_joins_assistant_turn(raw_input[index]) and trailing_roles[index - 1] == "assistant")
    )
    return tuple(zip(start_indexes, (*start_indexes[1:], len(raw_input))))


def _input_item_provenance(
    raw_input: Sequence[object],
    expected_messages: Sequence[object],
    request_tools: object = None,
) -> tuple[Mapping[int, int], frozenset[int]] | None:
    if not all(isinstance(item, Mapping) for item in raw_input):
        return None
    request_params: Final = {"tools": request_tools}
    solo_conversions: Final = tuple(
        LiteLLMCompletionResponsesConfig.transform_responses_api_input_to_messages(
            input=cast("ResponseInputParam", [item]),  # cast-ok: items checked as Mappings above
            responses_api_request=request_params,
            replay_tool_search=True,
            member_name=qualified_member_name,
        )
        for item in raw_input
    )
    full_conversion: Final = tuple(
        LiteLLMCompletionResponsesConfig.transform_responses_api_input_to_messages(
            input=cast("ResponseInputParam", list(raw_input)),  # cast-ok: items checked as Mappings above
            responses_api_request=request_params,
            replay_tool_search=True,
            member_name=qualified_member_name,
        )
    )
    if full_conversion != tuple(expected_messages):
        return None
    units: Final = _provenance_unit_bounds(raw_input, solo_conversions)
    unit_messages: Final = tuple(
        tuple(solo_conversions[start])
        if end - start == 1
        else tuple(
            LiteLLMCompletionResponsesConfig.transform_responses_api_input_to_messages(
                input=cast("ResponseInputParam", list(raw_input[start:end])),  # cast-ok: checked as Mappings above
                responses_api_request=request_params,
                replay_tool_search=True,
                member_name=qualified_member_name,
            )
        )
        for start, end in units
    )
    if tuple(message for messages in unit_messages for message in messages) != full_conversion:
        return None
    boundaries: Final = tuple(accumulate((len(messages) for messages in unit_messages), initial=0))
    item_for_message: Final = MappingProxyType(
        {
            message_index: start
            for unit_index, (start, end) in enumerate(units)
            if end - start == 1
            for message_index in range(boundaries[unit_index], boundaries[unit_index + 1])
        }
    )
    tainted: Final = frozenset(
        message_index
        for unit_index, (start, end) in enumerate(units)
        if end - start > 1
        for message_index in range(boundaries[unit_index], boundaries[unit_index + 1])
    )
    return item_for_message, tainted


class _RequestFields(NamedTuple):
    input: tuple[object, ...]
    instructions: str | None


class _ExtractedInputs(NamedTuple):
    inputs: GenericGuardrailAPIInputs
    task_mappings: tuple[tuple[int, int | None], ...]
    instructions: str | None


def scannable_instructions(data: Mapping[str, object], *, skip_system: bool = False) -> str | None:
    instructions: Final = data.get("instructions")
    return instructions if isinstance(instructions, str) and instructions and not skip_system else None


def _input_item_role(item: object) -> str:
    if stream_item_field(item, "type") in TOOL_CALL_OUTPUT_ITEM_TYPES:
        return "tool"
    role: Final = item.get("role") if isinstance(item, Mapping) else None
    return role.lower() if isinstance(role, str) else ""


def _scanned_text_field(item: Mapping[str, object]) -> str:
    return "output" if item.get("type") in TOOL_CALL_OUTPUT_ITEM_TYPES else "content"


def _content_part_text_field(part: object) -> str:
    return "refusal" if stream_item_field(part, "type") == "refusal" else "text"


def _has_writable_field(part: object, field: str) -> bool:
    return isinstance(part, dict) or hasattr(part, field)


def _tool_output_part_text(part: object) -> str | None:
    if not isinstance(part, dict):
        return None
    fields: Final = _item_fields_adapter.validate_python(part)
    text: Final = fields.get("text")
    return text if fields.get("type") in _TOOL_OUTPUT_TEXT_PART_TYPES and isinstance(text, str) else None


def tool_output_texts(item: object) -> tuple[tuple[int | None, str], ...]:
    if not stream_item_field(item, "call_id") or stream_item_field(item, "type") == "computer_call_output":
        return ()
    parts: Final = stream_item_items(item, "output")
    part_texts: Final = tuple(
        (index, text) for index, part in enumerate(parts) if (text := _tool_output_part_text(part))
    )
    if part_texts:
        return part_texts
    tool_content: Final = tool_output_to_chat_content(stream_item_field(item, "output"))
    return ((None, tool_content),) if isinstance(tool_content, str) and tool_content else ()


_Grouped = TypeVar("_Grouped")


def _pair_call_id(pair: tuple[str, object]) -> str:
    return pair[0]


def _grouped_by_call_id(pairs: Iterable[tuple[str, _Grouped]]) -> Mapping[str, tuple[_Grouped, ...]]:
    ordered: Final = sorted(pairs, key=_pair_call_id)
    return MappingProxyType(
        {call_id: tuple(value for _, value in group) for call_id, group in groupby(ordered, key=_pair_call_id)}
    )


def _texts_by_tool_row(
    messages: Sequence[Mapping[str, object]], input_data: "str | ResponseInputParam"
) -> Mapping[int, tuple[str, ...]]:
    if isinstance(input_data, str):
        return MappingProxyType({})
    output_texts_by_call_id: Final = _grouped_by_call_id(
        (str(stream_item_field(item, "call_id")), tuple(text for _, text in tool_output_texts(item)))
        for item in input_data
        if stream_item_field(item, "type") in TOOL_CALL_OUTPUT_ITEM_TYPES
    )
    row_indices_by_call_id: Final = _grouped_by_call_id(
        (str(message.get("tool_call_id")), index)
        for index, message in enumerate(messages)
        if message.get("role") == "tool"
    )
    return MappingProxyType(
        dict(
            chain.from_iterable(
                zip(indices, output_texts_by_call_id[call_id], strict=True)
                for call_id, indices in row_indices_by_call_id.items()
                if len(indices) == len(output_texts_by_call_id.get(call_id, ()))
            )
        )
    )


def _is_string_tool_row(message: Mapping[str, object]) -> bool:
    return message.get("role") == "tool" and isinstance(message.get("content"), str)


def _split_into_text_parts(message: AllMessageValues, texts: tuple[str, ...]) -> AllMessageValues:
    if not _is_string_tool_row(message) or len(texts) < 2:
        return message
    return cast(  # cast-ok: a tool row whose content is its text parts as chat text blocks
        "AllMessageValues", {**message, "content": [{"type": "text", "text": text} for text in texts]}
    )


def _with_split_rows_restored(
    rows: Sequence[AllMessageValues], split_rows: Sequence[tuple[AllMessageValues, AllMessageValues]]
) -> Sequence[AllMessageValues]:
    if not isinstance(rows, list):
        return rows
    return [next((original for split, original in split_rows if row == split), row) for row in rows]


def _is_textless_tool_row(message: Mapping[str, object], texts: tuple[str, ...] | None) -> bool:
    return _is_string_tool_row(message) and (not message.get("content") or texts == ())


@runtime_checkable
class _SupportsSetItem(Protocol):
    def __setitem__(self, key: str, value: object, /) -> None: ...


def _assignable(item: object) -> _SupportsSetItem:
    if isinstance(item, _SupportsSetItem):
        return item
    raise TypeError(f"'{type(item).__name__}' object does not support item assignment")


def _summarized_encrypted_content(item: Mapping[str, object]) -> str | None:
    if compaction_summary(item) is None:
        return None
    encrypted_content: Final = item["encrypted_content"]
    return encrypted_content if isinstance(encrypted_content, str) else None


def _rewritten_whole_text(item: object, field: str, guardrail_response: str) -> object | None:
    if isinstance(stream_item_field(item, field), str):
        return guardrail_response
    if tool_output_texts(item) == ((None, guardrail_response),):
        return None
    return guardrail_response


def _patched_request_fields(
    raw_input: object,
    instructions: object,
    original_messages: Sequence[object],
    structured_messages: Sequence[object],
    request_tools: object = None,
) -> _RequestFields | None:
    if not isinstance(raw_input, list) or len(original_messages) != len(structured_messages):
        return None
    offset: Final = 1 if instructions else 0
    provenance: Final = _input_item_provenance(raw_input, tuple(original_messages)[offset:], request_tools)
    if provenance is None:
        return None
    item_for_message, tainted = provenance
    changed: Final = tuple(
        (index, rewritten)
        for index, (original, rewritten) in enumerate(zip(original_messages, structured_messages))
        if original != rewritten
    )
    instruction_rewrites: Final = tuple(rewritten for index, rewritten in changed if index < offset)
    rewritten_instructions: Final = (
        instruction_rewrites[0].get("content")
        if instruction_rewrites and isinstance(instruction_rewrites[0], Mapping)
        else instructions
    )
    instructions_value: Final = rewritten_instructions if isinstance(rewritten_instructions, str) else None
    if rewritten_instructions is not None and instructions_value is None:
        return None
    body_changes: Final = tuple((index - offset, rewritten) for index, rewritten in changed if index >= offset)
    if any(message_index in tainted or message_index not in item_for_message for message_index, _ in body_changes):
        return None
    replacements: Final = MappingProxyType(
        {
            item_for_message[message_index]: _rewritten_input_item(
                cast("Mapping[str, object]", raw_input[item_for_message[message_index]]),  # cast-ok: checked Mappings
                rewritten,
            )
            for message_index, rewritten in body_changes
        }
    )
    if len(replacements) != len(body_changes) or any(item is None for item in replacements.values()):
        return None
    return _RequestFields(
        input=tuple(replacements.get(index, item) for index, item in enumerate(raw_input)),
        instructions=instructions_value,
    )


def _patch_or_convert_request_fields(
    raw_input: object,
    instructions: object,
    original_messages: Sequence[object],
    structured_messages: Sequence[AllMessageValues],
    request_tools: object = None,
) -> _RequestFields | None:
    if not isinstance(structured_messages, list):
        return None
    patched: Final = _patched_request_fields(
        raw_input, instructions, original_messages, structured_messages, request_tools
    )
    if patched is not None:
        return patched
    input_items, converted_instructions = (
        LiteLLMResponsesTransformationHandler().convert_chat_completion_messages_to_responses_api(structured_messages)
    )
    return _RequestFields(
        input=_with_tool_searches_restored(input_items, raw_input), instructions=converted_instructions
    )


def _tool_search_items_by_call_id(raw_input: Sequence[object], item_type: str) -> Mapping[str, object]:
    return MappingProxyType(
        {
            call_id: item
            for item in raw_input
            if stream_item_field(item, "type") == item_type
            and isinstance(call_id := stream_item_field(item, "call_id"), str)
        }
    )


def _restored_tool_search(
    item: object, calls: Mapping[str, object], outputs: Mapping[str, object]
) -> tuple[object, ...]:
    call_id: Final = stream_item_field(item, "call_id")
    if stream_item_field(item, "type") != "function_call" or not isinstance(call_id, str) or call_id not in calls:
        return (item,)
    arguments: Final = stream_item_field(item, "arguments")
    restored_call: Final = _rewritten_tool_search_call(
        calls[call_id], arguments if isinstance(arguments, str) else None
    )
    return (restored_call, outputs[call_id]) if call_id in outputs else (restored_call,)


def _with_tool_searches_restored(input_items: Sequence[object], raw_input: object) -> tuple[object, ...]:
    raw_items: Final = _RAW_ITEMS_ADAPTER.validate_python(raw_input) if isinstance(raw_input, list) else ()
    if not any(map(_is_tool_search_item, raw_items)):
        return tuple(input_items)
    calls: Final = _tool_search_items_by_call_id(raw_items, _TOOL_SEARCH_CALL_TYPE)
    outputs: Final = _tool_search_items_by_call_id(raw_items, "tool_search_output")
    return tuple(chain.from_iterable(_restored_tool_search(item, calls, outputs) for item in input_items))


def _next_stream_sequence_number(responses_so_far: Sequence[object] | None) -> int:
    sequence_numbers: Final = (
        item.get("sequence_number") if isinstance(item, dict) else getattr(item, "sequence_number", None)
        for item in reversed(responses_so_far or ())
    )
    return next((n + 1 for n in sequence_numbers if isinstance(n, int)), 0)


def _is_responses_input(
    input_data: object,
) -> TypeIs["str | ResponseInputParam"]:  # guard-ok: str, or a list of Mapping items
    if isinstance(input_data, str):
        return True
    items: Final = cast(list[object], input_data)  # cast-ok: isinstance narrows to list[Unknown] only
    return isinstance(input_data, list) and all(isinstance(item, (Mapping, ResponseCompactionItem)) for item in items)


_StreamEventT = TypeVar("_StreamEventT")


async def rewrite_buffered_responses_stream(
    stream: AsyncIterable[_StreamEventT],
    guardrail: "CustomGuardrail",
    request_data: dict[str, object],  # mutable-ok: the per-request state dict every guardrail hook shares
) -> tuple[_StreamEventT, ...]:
    events: Final = [event async for event in stream]
    await OpenAIResponsesHandler().process_output_streaming_response(
        responses_so_far=events,
        guardrail_to_apply=guardrail,
        request_data=request_data,
        deliver_ended_stream_rewrites=True,
    )
    return tuple(events)


class OpenAIResponsesHandler(BaseTranslation):
    """
    Handler for processing OpenAI Responses API with guardrails.

    This class provides methods to:
    1. Process input (pre-call hook)
    2. Process output response (post-call hook)

    Methods can be overridden to customize behavior for different message formats.
    """

    delivers_ended_stream_rewrites = True
    assembles_streamed_response = True

    def get_structured_messages(
        self, data: dict[str, object], *, keep_tool_errors: bool = False
    ) -> list[AllMessageValues] | None:
        """
        Convert Responses API request data to OpenAI-spec structured messages.

        Transforms `input` (string or ResponseInputParam) and optional
        `instructions` into chat completion messages.
        """
        input_data: Final[object] = data.get("input")
        if not _is_responses_input(input_data):
            return None
        messages: Final = LiteLLMCompletionResponsesConfig.transform_responses_api_input_to_messages(
            input=input_data,
            responses_api_request=data,
            replay_tool_search=True,
            member_name=qualified_member_name,
        )
        return cast(list[AllMessageValues], messages) if messages else None

    async def process_input_messages(
        self,
        data: dict[str, object],
        guardrail_to_apply: "CustomGuardrail",
        litellm_logging_obj: "LiteLLMLoggingObj | None" = None,
    ) -> dict[str, object]:
        """
        Process input by applying guardrails to text content.

        Handles both string input and list of message objects.
        """
        raw_input: Final[object] = data.get("input")
        if not isinstance(raw_input, (str, list)):
            return data
        input_data: Final = cast("str | ResponseInputParam", raw_input)  # cast-ok: fields read via stream_item_field
        skip_system: Final = effective_skip_system_message_for_guardrail(guardrail_to_apply)
        skip_tool: Final = effective_skip_tool_message_for_guardrail(guardrail_to_apply)
        scan_only: Final = effective_scan_only_tool_results_for_guardrail(guardrail_to_apply)
        structured_messages: Final = self.get_structured_messages(data)
        all_messages: Final = structured_messages or []
        texts_by_tool_row: Final = _texts_by_tool_row(all_messages, input_data)
        scoped_indices: Final = tuple(
            index
            for index in scoped_structured_message_indices(
                all_messages, scan_only_tool_results=scan_only, skip_system=skip_system, skip_tool=skip_tool
            )
            if not _is_textless_tool_row(all_messages[index], texts_by_tool_row.get(index))
        )
        split_rows: Final = tuple(
            (_split_into_text_parts(all_messages[index], texts_by_tool_row.get(index, ())), all_messages[index])
            for index in scoped_indices
        )
        scoped_structured_messages: Final = [split for split, _ in split_rows] if structured_messages else None
        raw_tools: Final = data.get("tools")
        original_tools: Final[tuple[Mapping[str, object], ...]] = (
            tuple(raw_tools) if isinstance(raw_tools, list) else ()
        )
        scanned_tools: Final = tuple(without_malformed_tool_flags(tool) for tool in original_tools)
        flattened_tool_groups: Final = tuple(
            () if is_server_tool_search_tool(tool) else form.chat_tools
            for tool, form in zip(
                scanned_tools,
                LiteLLMCompletionResponsesConfig.responses_tools_to_chat_forms(scanned_tools, qualified_member_name),
                strict=True,
            )
        )
        extracted: Final = self._extract_guardrail_inputs(
            data,
            input_data,
            () if scan_only else flattened_tool_groups,
            skip_system=skip_system,
            skip_tool=skip_tool,
            scan_only=scan_only,
        )
        if (
            effective_scan_tool_calls_as_text_for_guardrail(guardrail_to_apply)
            and not scan_only
            and not isinstance(input_data, str)
        ):
            history_tool_calls: Final = [
                _scanned_tool_call(item, item_idx)
                for item_idx, item in enumerate(input_data)
                if item.get("type") in _TOOL_CALL_ITEM_TYPES
            ]
            if history_tool_calls:
                extracted.inputs["tool_calls"] = history_tool_calls
        if not extracted.inputs.get("texts") and not extracted.inputs.get("tool_calls"):
            return data
        if scoped_structured_messages:
            extracted.inputs["structured_messages"] = scoped_structured_messages
        guardrailed_inputs: Final = await guardrail_to_apply.apply_guardrail(
            inputs=extracted.inputs,
            request_data=data,
            input_type="request",
            logging_obj=litellm_logging_obj,
        )
        self._apply_guardrailed_tools_to_data(
            data,
            original_tools,
            flattened_tool_groups,
            guardrailed_inputs.get("tools"),
            scan_only=scan_only,
            guardrail_name=guardrail_to_apply.guardrail_name,
        )
        written_back: Final = self._written_back_request_fields(
            data,
            structured_messages or (),
            scoped_indices,
            scoped_structured_messages,
            tuple((split, original) for split, original in split_rows if split is not original),
            guardrail_to_apply,
            guardrailed_inputs,
        )
        if written_back is not None:
            data["input"] = list(written_back.input)
            if written_back.instructions is None:
                data.pop("instructions", None)
            else:
                data["instructions"] = written_back.instructions  # rebind-ok: data is an out-param
        else:
            await self._apply_guardrailed_texts(data, input_data, extracted, guardrail_to_apply, guardrailed_inputs)
        verbose_proxy_logger.debug("OpenAI Responses API: Processed input messages: %s", data.get("input"))
        return data

    async def _apply_guardrailed_texts(
        self,
        data: dict[str, object],
        input_data: "str | ResponseInputParam",
        extracted: _ExtractedInputs,
        guardrail_to_apply: "CustomGuardrail",
        guardrailed_inputs: GenericGuardrailAPIInputs,
    ) -> None:
        returned_texts: Final = guardrailed_inputs.get("texts")
        if not returned_texts:
            return
        rewritten_texts: Final = tuple(returned_texts)
        offset: Final = 0 if extracted.instructions is None else 1
        input_texts: Final = rewritten_texts[offset:]
        expected: Final = 1 if isinstance(input_data, str) else len(extracted.task_mappings)
        if len(rewritten_texts) != offset + expected:
            raise unappliable_request_rewrite(guardrail_to_apply.guardrail_name)
        if offset:
            data["instructions"] = rewritten_texts[0]  # rebind-ok: data is an out-param
        if isinstance(input_data, str):
            data["input"] = input_texts[0]  # rebind-ok: data is an out-param
            return
        await self._apply_guardrail_responses_to_input(
            messages=input_data,
            responses=input_texts,
            task_mappings=extracted.task_mappings,
        )

    def _extract_guardrail_inputs(
        self,
        data: Mapping[str, object],
        input_data: "str | ResponseInputParam",
        flattened_tool_groups: Sequence[Sequence[Mapping[str, object]]],
        *,
        skip_system: bool = False,
        skip_tool: bool = False,
        scan_only: bool = False,
    ) -> _ExtractedInputs:
        instructions: Final = scannable_instructions(data, skip_system=skip_system or scan_only)
        texts_to_check: Final[list[str]] = [] if instructions is None else [instructions]
        images_to_check: Final[list[str]] = []
        task_mappings: Final[list[tuple[int, int | None]]] = []
        tools_to_check: Final[list[ChatCompletionToolParam]] = list(  # mutable-ok: guardrail inputs want a list
            copy.deepcopy(
                tuple(
                    cast(ChatCompletionToolParam, tool)  # cast-ok: mcp tools ride along in the guardrail's tool list
                    for group in flattened_tool_groups
                    for tool in group
                )
            )
        )
        if isinstance(input_data, str):
            if not scan_only:
                texts_to_check.append(input_data)
        else:
            for msg_idx, message in enumerate(input_data):
                if role_out_of_guardrail_scope(
                    _input_item_role(message),
                    skip_system_message=skip_system,
                    skip_tool_message=skip_tool,
                    scan_only_tool_results=scan_only,
                ):
                    continue
                self._extract_input_text_and_images(
                    message=message,
                    msg_idx=msg_idx,
                    texts_to_check=texts_to_check,
                    images_to_check=images_to_check,
                    task_mappings=task_mappings,
                )
        inputs: Final = GenericGuardrailAPIInputs(texts=texts_to_check)
        if images_to_check:
            inputs["images"] = images_to_check
        if tools_to_check:
            inputs["tools"] = tools_to_check
        model: Final = data.get("model")
        if isinstance(model, str):
            inputs["model"] = model
        return _ExtractedInputs(inputs=inputs, task_mappings=tuple(task_mappings), instructions=instructions)

    @staticmethod
    def _written_back_request_fields(
        data: Mapping[str, object],
        structured_messages: Sequence[AllMessageValues],
        scoped_indices: Sequence[int],
        scoped_structured_messages: Sequence[AllMessageValues] | None,
        split_rows: Sequence[tuple[AllMessageValues, AllMessageValues]],
        guardrail_to_apply: "CustomGuardrail",
        guardrailed_inputs: GenericGuardrailAPIInputs,
    ) -> _RequestFields | None:
        returned: Final = guardrailed_inputs.get("structured_messages")
        if returned is None or returned is scoped_structured_messages:
            return None
        guardrailed: Final = _with_split_rows_restored(returned, split_rows)
        covers_full_request: Final = len(scoped_indices) == len(structured_messages) or (
            guardrail_to_apply.structured_messages_cover_full_request() and len(guardrailed) == len(structured_messages)
        )
        merged: Final = (
            guardrailed
            if covers_full_request
            else merge_guardrailed_scoped_messages(
                full_messages=structured_messages, scoped_indices=scoped_indices, guardrailed_scoped=guardrailed
            )
        )
        return _patch_or_convert_request_fields(
            data.get("input"), data.get("instructions"), structured_messages, merged, data.get("tools")
        )

    def extract_request_tool_names(self, data: Mapping[str, object]) -> list[str]:
        """Extract tool names from Responses API request (tools[].name for function
        and custom, tools[].server_label for mcp), including tools carried by
        ``additional_tools`` input items."""
        names: Final[list[str]] = []
        request_tools: Final[tuple[object, ...]] = (
            *list_field(data, "tools"),
            *additional_tools_in_input(data.get("input")),
        )
        for tool in request_tools:
            if not isinstance(tool, dict):
                continue
            if tool.get("type") in ("function", "custom") and tool.get("name"):
                names.append(str(tool["name"]))
            elif tool.get("type") == "mcp" and tool.get("server_label"):
                names.append(str(tool["server_label"]))
        return names

    def _apply_guardrailed_tools_to_data(
        self,
        data: dict[str, object],
        original_tools: Sequence[Mapping[str, object]],
        flattened_tool_groups: Sequence[Sequence[Mapping[str, object]]],
        guardrailed_tools: Sequence[ChatCompletionToolParam] | None,
        *,
        scan_only: bool,
        guardrail_name: str | None,
    ) -> None:
        if guardrailed_tools is None:
            return
        merge: Final = append_returned_tools if scan_only else merge_guardrailed_tools
        merged: Final = merge(original_tools, flattened_tool_groups, guardrailed_tools)
        if merged is None:
            raise unappliable_request_rewrite(guardrail_name)
        data["tools"] = list(merged)  # rebind-ok: in-place request rewrite

    def _extract_input_text_and_images(
        self,
        message: Mapping[str, object],
        msg_idx: int,
        texts_to_check: list[str],
        images_to_check: list[str],
        task_mappings: list[tuple[int, int | None]],
    ) -> None:
        """
        Extract text content and images from an input message.

        Override this method to customize text/image extraction logic.
        """
        summary: Final = compaction_summary(message)
        if summary is not None:
            texts_to_check.append(summary)
            task_mappings.append((msg_idx, None))
            return
        if message.get("type") in TOOL_CALL_OUTPUT_ITEM_TYPES:
            for content_idx, text in tool_output_texts(message):
                texts_to_check.append(text)
                task_mappings.append((msg_idx, content_idx))
            return
        content: Final = message.get("content", None)
        if content is None:
            return

        if isinstance(content, str):
            # Simple string content
            texts_to_check.append(content)
            task_mappings.append((msg_idx, None))

        elif isinstance(content, list):
            # List content (e.g., multimodal with text and images)
            for content_idx, content_item in enumerate(content):
                if isinstance(content_item, dict):
                    # Extract text
                    text_str = content_item.get(_content_part_text_field(content_item), None)
                    if text_str is not None:
                        texts_to_check.append(text_str)
                        task_mappings.append((msg_idx, int(content_idx)))

                    # Extract images
                    if content_item.get("type") == "image_url":
                        image_url = content_item.get("image_url", {})
                        if isinstance(image_url, dict):
                            url = image_url.get("url")
                            if url:
                                images_to_check.append(url)

    async def _apply_guardrail_responses_to_input(
        self,
        messages: Sequence[Mapping[str, object]],
        responses: Sequence[str],
        task_mappings: Sequence[tuple[int, int | None]],
    ) -> None:
        """
        Apply guardrail responses back to input messages.

        ``responses`` pairs positionally with ``task_mappings``; the caller rejects
        the request when the two disagree, so this never has to guess an alignment.

        Override this method to customize how responses are applied.
        """
        for guardrail_response, mapping in zip(responses, task_mappings):
            msg_idx = cast(int, mapping[0])
            content_idx_optional = cast(int | None, mapping[1])
            item = messages[msg_idx]

            encrypted_content = _summarized_encrypted_content(item)
            if encrypted_content is not None:
                _assignable(item)["encrypted_content"] = with_compaction_summary(encrypted_content, guardrail_response)
                continue

            field = _scanned_text_field(item)
            content = item.get(field, None)
            if content is None:
                continue

            content_parts = as_object_list(content)
            if content_idx_optional is None:
                rewritten = _rewritten_whole_text(item, field, guardrail_response)
                if rewritten is not None:
                    _assignable(item)[field] = rewritten

            elif content_parts is not None:
                # Replace specific text item in list content
                self._write_content_part_text(content_parts[content_idx_optional], guardrail_response)

    async def process_output_response(
        self,
        response: Union["ResponsesAPIResponse", ResponseOutputEnvelope],
        guardrail_to_apply: "CustomGuardrail",
        litellm_logging_obj: "LiteLLMLoggingObj | None" = None,
        user_api_key_dict: "UserAPIKeyAuth | None" = None,
        request_data: dict[str, object] | None = None,
    ) -> Union["ResponsesAPIResponse", ResponseOutputEnvelope]:
        """
        Process output response by applying guardrails to text content and tool calls.

        Args:
            response: LiteLLM ResponsesAPIResponse object
            guardrail_to_apply: The guardrail instance to apply
            litellm_logging_obj: Optional logging object
            user_api_key_dict: User API key metadata to pass to guardrails

        Returns:
            Modified response with guardrail applied to content

        Response Format Support:
            - response.output is a list of output items
            - Each output item can be:
              * GenericResponseOutputItem with a content list of OutputText objects
              * ResponseFunctionToolCall or CustomToolCallOutputItem with tool call data
            - Each OutputText object has a text field
        """

        texts_to_check: Final[list[str]] = []
        images_to_check: Final[list[str]] = []
        tool_calls_to_check: Final[list[ChatCompletionToolCallChunk]] = []
        task_mappings: Final[list[_OutputPosition]] = []

        # Handle both dict and Pydantic object responses
        response_output: Sequence[object]
        if isinstance(response, dict):
            response_output = response.get("output", [])
        elif hasattr(response, "output"):
            response_output = response.output or []
        else:
            verbose_proxy_logger.debug("OpenAI Responses API: No output found in response")
            return response

        # Step 1: Extract all text content and tool calls from response output
        for output_idx, output_item in _answer_output_items(response_output):
            self._extract_output_text_and_images(
                output_item=output_item,
                output_idx=output_idx,
                texts_to_check=texts_to_check,
                images_to_check=images_to_check,
                task_mappings=task_mappings,
                tool_calls_to_check=tool_calls_to_check,
            )

        reasoning_texts: Final = _response_reasoning_texts(response)
        reasoning_in_answer_call: Final = guardrail_to_apply.scans_all_response_texts_in_one_call()
        if reasoning_in_answer_call:
            texts_to_check.extend(text for _, text in reasoning_texts)
            task_mappings.extend(position for position, _ in reasoning_texts)

        # Step 2: Apply guardrail to all texts in batch
        if texts_to_check or tool_calls_to_check:
            inputs: Final = GenericGuardrailAPIInputs(texts=texts_to_check)
            if images_to_check:
                inputs["images"] = images_to_check
            if tool_calls_to_check:
                inputs["tool_calls"] = tool_calls_to_check
            # Include model information from the response if available
            response_model: str | None = None
            if isinstance(response, dict):
                response_model = response.get("model")
            elif hasattr(response, "model"):
                response_model = getattr(response, "model", None)
            if response_model:
                inputs["model"] = response_model

            pre_guardrail_tool_calls: Final = _tool_call_shapes(tool_calls_to_check)
            guardrailed_inputs: Final = await guardrail_to_apply.apply_guardrail(
                inputs=inputs,
                request_data=self._request_data_with_response(request_data, response, user_api_key_dict),
                input_type="response",
                logging_obj=litellm_logging_obj,
            )

            guardrailed_texts: Final = guardrailed_inputs.get("texts", [])
            post_guardrail_tool_calls: Final = _post_guardrail_tool_call_shapes(
                returned_tool_calls=guardrailed_inputs.get("tool_calls"),
                pre_guardrail_tool_calls=pre_guardrail_tool_calls,
                guardrail_name=guardrail_to_apply.guardrail_name,
            )

            # Step 3: Map guardrail responses back to original response structure
            await self._apply_guardrail_responses_to_output(
                response=response,
                responses=guardrailed_texts,
                task_mappings=task_mappings,
            )
            self._write_tool_call_rewrites_to_output(
                outputs=response_output,
                pre_guardrail_tool_calls=pre_guardrail_tool_calls,
                post_guardrail_tool_calls=post_guardrail_tool_calls,
            )

        if reasoning_texts and not reasoning_in_answer_call:
            await self._apply_guardrail_to_reasoning_texts(
                response=response,
                reasoning_texts=reasoning_texts,
                guardrail_to_apply=guardrail_to_apply,
                litellm_logging_obj=litellm_logging_obj,
                request_data=self._request_data_with_response(request_data, response, user_api_key_dict),
            )

        verbose_proxy_logger.debug("OpenAI Responses API: Processed output response: %s", response)

        return response

    async def process_output_streaming_response(
        self,
        responses_so_far: list[Any],
        guardrail_to_apply: "CustomGuardrail",
        litellm_logging_obj: "LiteLLMLoggingObj | None" = None,
        user_api_key_dict: "UserAPIKeyAuth | None" = None,
        request_data: dict[str, object] | None = None,
        stream_transform_sink: StreamTransformSink | None = None,
        deliver_ended_stream_rewrites: bool = False,
    ) -> Sequence[object]:
        """
        Process output streaming response by applying guardrails to text content.

        Mirrors the Chat Completions handler pattern: extract text from the final
        chunk, apply the guardrail, then write the result back in-place so the
        caller sees the modified content (e.g. PII tokens replaced).

        For terminal envelope events (``response.completed``, and equally
        ``response.incomplete`` / ``response.failed``, whose envelopes carry the
        partial output) we use the same per-item extraction + task-mapping
        approach as ``process_output_response`` so that unmasking / blocking
        works correctly for every output item. With
        ``deliver_ended_stream_rewrites`` the earlier text-carrying events
        (``response.output_text.delta`` / ``.done``,
        ``response.content_part.done``, ``response.output_item.done``, and the
        reasoning summary and reasoning text events) are synced to the
        rewritten envelope too, so a client reading deltas sees the rewrite
        instead of the raw model output; a stream with no envelope gets its
        rewrite spread over the buffered text events, unless it carries
        reasoning, in which case it is scanned through an envelope built from
        its events so each part keeps its own rewrite, and a rewrite
        observed where no write-back is possible is reported as undeliverable,
        so the pipeline executor discards it and releases the original events.
        """
        if not responses_so_far:
            return responses_so_far

        final_chunk: Final = responses_so_far[-1]
        # Accept both plain dicts and Pydantic models (BaseLiteLLMOpenAIResponseObject
        # exposes a .get() shim, so all the .get() calls below work for both).
        if not _reads_like_a_dict(final_chunk):
            return responses_so_far

        # ------------------------------------------------------------------ #
        # Case 1: terminal envelope events (completed/incomplete/failed).     #
        # the accumulated response is available in the final chunk; iterate   #
        # output items, apply guardrail, write back. Falls through to the     #
        # string fallback when the envelope yields nothing to check.          #
        # ------------------------------------------------------------------ #
        if final_chunk.get("type") in _TERMINAL_ENVELOPE_EVENT_TYPES:
            response_obj: Final[ResponseOutputEnvelope] = final_chunk.get("response") or {}
            outputs: Final[Sequence[object]] = (
                (response_obj.get("output") or []) if hasattr(response_obj, "get") else []
            )

            texts_to_check: Final[list[str]] = []
            tool_calls_to_check: Final[list[ChatCompletionToolCallChunk]] = []
            task_mappings: Final[list[_OutputPosition]] = []

            for output_idx, output_item in _answer_output_items(outputs):
                self._extract_output_text_and_images(
                    output_item=output_item,
                    output_idx=output_idx,
                    texts_to_check=texts_to_check,
                    images_to_check=[],
                    task_mappings=task_mappings,
                    tool_calls_to_check=tool_calls_to_check,
                )

            reasoning_texts: Final = _response_reasoning_texts(response_obj)
            reasoning_in_answer_call: Final = guardrail_to_apply.scans_all_response_texts_in_one_call()
            reasoning_outside_envelope: Final = (
                _reasoning_streamed_outside_envelope(responses_so_far, reasoning_texts)
                if not reasoning_in_answer_call
                else ()
            )
            answer_outside_envelope: Final = (
                not texts_to_check and not tool_calls_to_check and _has_answer_stream_text(responses_so_far)
            )
            if reasoning_in_answer_call and not answer_outside_envelope:
                texts_to_check.extend(text for _, text in reasoning_texts)
                task_mappings.extend(position for position, _ in reasoning_texts)

            if texts_to_check or tool_calls_to_check:
                inputs = GenericGuardrailAPIInputs(texts=texts_to_check)
                if tool_calls_to_check:
                    inputs["tool_calls"] = cast(list[ChatCompletionToolCallChunk], tool_calls_to_check)
                response_model = response_obj.get("model")
                if response_model:
                    inputs["model"] = response_model

                pre_guardrail_tool_calls: Final = _tool_call_shapes(tool_calls_to_check)
                guardrailed_inputs: Final = await guardrail_to_apply.apply_guardrail(
                    inputs=inputs,
                    request_data=self._request_data_with_response(request_data, response_obj, user_api_key_dict),
                    input_type="response",
                    logging_obj=litellm_logging_obj,
                )

                guardrailed_texts: Final = guardrailed_inputs.get("texts", [])
                post_guardrail_tool_calls: Final = _post_guardrail_tool_call_shapes(
                    returned_tool_calls=guardrailed_inputs.get("tool_calls"),
                    pre_guardrail_tool_calls=pre_guardrail_tool_calls,
                    guardrail_name=guardrail_to_apply.guardrail_name,
                )

                # Write guardrailed texts back into the output items in-place.
                # final_chunk is a reference into responses_so_far so this
                # mutates the list that the caller holds.
                await self._apply_guardrail_responses_to_output(
                    response=response_obj,
                    responses=guardrailed_texts,
                    task_mappings=task_mappings,
                )
                if deliver_ended_stream_rewrites:
                    rewrites_by_position: Final[Mapping[_OutputPosition, str]] = MappingProxyType(
                        {
                            task_mappings[task_idx]: rewritten
                            for task_idx, rewritten in enumerate(guardrailed_texts)
                            if task_idx < len(texts_to_check) and rewritten != texts_to_check[task_idx]
                        }
                    )
                    self._sync_stream_events_with_output_rewrites(responses_so_far, response_obj, rewrites_by_position)
                    self._deliver_ended_stream_tool_call_rewrites(
                        responses_so_far=responses_so_far,
                        outputs=outputs,
                        pre_guardrail_tool_calls=pre_guardrail_tool_calls,
                        post_guardrail_tool_calls=post_guardrail_tool_calls,
                        guardrail_name=guardrail_to_apply.guardrail_name or "unknown",
                    )

            if reasoning_texts and not reasoning_in_answer_call:
                reasoning_rewrites: Final = await self._apply_guardrail_to_reasoning_texts(
                    response=response_obj,
                    reasoning_texts=reasoning_texts,
                    guardrail_to_apply=guardrail_to_apply,
                    litellm_logging_obj=litellm_logging_obj,
                    request_data=self._request_data_with_response(request_data, response_obj, user_api_key_dict),
                )
                if deliver_ended_stream_rewrites:
                    self._sync_stream_events_with_output_rewrites(responses_so_far, response_obj, reasoning_rewrites)

            envelope_has_scannable_content: Final = bool(texts_to_check or tool_calls_to_check or reasoning_texts)
            if envelope_has_scannable_content and not answer_outside_envelope:
                outside_rewrites: Final = await self._apply_guardrail_to_reasoning_outside_envelope(
                    responses_so_far=responses_so_far,
                    envelope=response_obj,
                    outside_texts=reasoning_outside_envelope,
                    guardrail_to_apply=guardrail_to_apply,
                    litellm_logging_obj=litellm_logging_obj,
                    request_data=request_data,
                    user_api_key_dict=user_api_key_dict,
                )
                if deliver_ended_stream_rewrites:
                    self._sync_stream_events_with_output_rewrites(responses_so_far, response_obj, outside_rewrites)
                return responses_so_far

        # ------------------------------------------------------------------ #
        # Case 2: response.output_item.done — extract tool calls only, then  #
        # fall through to the text fallback when a caller expects rewrites   #
        # delivered, so a truncated buffer still reports text undeliverable. #
        # ------------------------------------------------------------------ #
        if final_chunk.get("type") == "response.output_item.done":
            finished_item: Final = stream_item_field(final_chunk, "item")
            finished_summary: Final = output_compaction_summary(finished_item)
            model_response_stream: Final = (
                OpenAiResponsesToChatCompletionStreamIterator.translate_responses_chunk_to_openai_stream(final_chunk)
            )
            finished_web_tool_call: Final = _hosted_web_tool_call(finished_item, 0)
            tool_calls: Final = [
                *cast(list[ChatCompletionToolCallChunk], model_response_stream.choices[0].delta.tool_calls or []),
                *(() if finished_web_tool_call is None else (finished_web_tool_call,)),
            ]
            if tool_calls or finished_summary is not None:
                inputs = GenericGuardrailAPIInputs()
                if finished_summary is not None:
                    inputs["texts"] = [finished_summary]
                if tool_calls:
                    inputs["tool_calls"] = tool_calls
                if hasattr(model_response_stream, "model") and model_response_stream.model:
                    inputs["model"] = model_response_stream.model
                await guardrail_to_apply.apply_guardrail(
                    inputs=inputs,
                    request_data=request_data if request_data is not None else {},
                    input_type="response",
                    logging_obj=litellm_logging_obj,
                )
            if not deliver_ended_stream_rewrites:
                return responses_so_far

        if (
            deliver_ended_stream_rewrites
            and _released_reasoning_items(responses_so_far)
            and not _has_unplaceable_stream_text(responses_so_far)
            and not _envelope_has_reasoning_text(final_chunk)
        ):
            await self.process_output_streaming_response(
                responses_so_far=[*responses_so_far, _released_envelope(responses_so_far)],
                guardrail_to_apply=guardrail_to_apply,
                litellm_logging_obj=litellm_logging_obj,
                user_api_key_dict=user_api_key_dict,
                request_data=request_data,
                stream_transform_sink=stream_transform_sink,
                deliver_ended_stream_rewrites=True,
            )
            return responses_so_far

        # ------------------------------------------------------------------ #
        # Fallback: apply guardrail to the accumulated text string. With no   #
        # envelope to rewrite, a rewrite a caller expects delivered is spread #
        # over the buffered text events instead.                              #
        # ------------------------------------------------------------------ #
        string_so_far: Final = self.get_streaming_string_so_far(responses_so_far)
        if string_so_far:
            fallback_inputs: Final = GenericGuardrailAPIInputs(texts=[string_so_far])
            response_model = (
                final_chunk.get("response", {}).get("model") if isinstance(final_chunk.get("response"), dict) else None
            )
            if response_model:
                fallback_inputs["model"] = response_model
            fallback_outputs: Final = await guardrail_to_apply.apply_guardrail(
                inputs=fallback_inputs,
                request_data=request_data if request_data is not None else {},
                input_type="response",
                logging_obj=litellm_logging_obj,
            )
            fallback_texts: Final = fallback_outputs.get("texts")
            if deliver_ended_stream_rewrites and fallback_texts and tuple(fallback_texts) != (string_so_far,):
                self._spread_text_rewrite_over_stream_events(
                    stream_events=responses_so_far,
                    rewritten_text=fallback_texts[0],
                    guardrail_name=guardrail_to_apply.guardrail_name or "unknown",
                )
        return responses_so_far

    def _request_data_with_response(
        self,
        request_data: dict[str, object] | None,
        response: Union["ResponsesAPIResponse", ResponseOutputEnvelope],
        user_api_key_dict: "UserAPIKeyAuth | None",
    ) -> dict[str, object]:
        scanned_request_data: Final = {} if request_data is None else request_data
        if "response" not in scanned_request_data:
            scanned_request_data["response"] = response
        if "litellm_metadata" not in scanned_request_data:
            user_metadata: Final = self.transform_user_api_key_dict_to_metadata(user_api_key_dict)
            if user_metadata:
                scanned_request_data["litellm_metadata"] = user_metadata
        return scanned_request_data

    async def _apply_guardrail_to_reasoning_texts(
        self,
        response: Union["ResponsesAPIResponse", ResponseOutputEnvelope],
        reasoning_texts: Sequence[tuple[_ReasoningTextPosition, str]],
        guardrail_to_apply: "CustomGuardrail",
        litellm_logging_obj: "LiteLLMLoggingObj | None",
        request_data: dict[str, object],
    ) -> Mapping[_OutputPosition, str]:
        distinct_texts: Final = tuple(dict.fromkeys(text for _, text in reasoning_texts))
        batches: Final = (
            (distinct_texts,)
            if guardrail_to_apply.scans_each_text_separately()
            else tuple((text,) for text in distinct_texts)
        )
        scan: Final = partial(
            _guardrailed_reasoning_texts,
            model=stream_item_field(response, "model"),
            guardrail_to_apply=guardrail_to_apply,
            litellm_logging_obj=litellm_logging_obj,
            request_data=request_data,
        )
        guardrailed_batches: Final = await asyncio.gather(*(_scan_reasoning_batch(batch, scan) for batch in batches))
        rewrites: Final = MappingProxyType(
            dict(zip(distinct_texts, chain.from_iterable(guardrailed_batches), strict=True))
        )
        await self._apply_guardrail_responses_to_output(
            response=response,
            responses=tuple(rewrites[text] for _, text in reasoning_texts),
            task_mappings=tuple(position for position, _ in reasoning_texts),
        )
        return MappingProxyType(
            {position: rewrites[text] for position, text in reasoning_texts if rewrites[text] != text}
        )

    async def _apply_guardrail_to_reasoning_outside_envelope(
        self,
        responses_so_far: Sequence[object],
        envelope: ResponseOutputEnvelope,
        outside_texts: Sequence[tuple[_ReasoningPosition, str]],
        guardrail_to_apply: "CustomGuardrail",
        litellm_logging_obj: "LiteLLMLoggingObj | None",
        request_data: dict[str, object] | None,
        user_api_key_dict: "UserAPIKeyAuth | None",
    ) -> Mapping[_OutputPosition, str]:
        if not outside_texts:
            return MappingProxyType({})
        return await self._apply_guardrail_to_reasoning_texts(
            response=ResponseOutputEnvelope(
                output=_released_stream_output(responses_so_far), model=envelope.get("model")
            ),
            reasoning_texts=outside_texts,
            guardrail_to_apply=guardrail_to_apply,
            litellm_logging_obj=litellm_logging_obj,
            request_data=self._request_data_with_response(request_data, envelope, user_api_key_dict),
        )

    def _sync_stream_events_with_output_rewrites(
        self,
        responses_so_far: Sequence[object],
        response: Union["ResponsesAPIResponse", ResponseOutputEnvelope],
        rewrites_by_position: Mapping[_OutputPosition, str],
    ) -> None:
        item_rewrites: Final[Mapping[_OutputItemPosition, str]] = MappingProxyType(
            {
                position: rewritten
                for position, rewritten in rewrites_by_position.items()
                if not isinstance(position, _ProviderThinkingBlockPosition)
            }
        )
        if item_rewrites:
            self._sync_stream_events_with_rewrites(
                stream_events=responses_so_far[:-1],
                rewrites_by_position=item_rewrites,
            )
        if len(item_rewrites) != len(rewrites_by_position):
            self._sync_provider_fields_to_stream_events(responses_so_far, response)

    def _spread_text_rewrite_over_stream_events(
        self,
        stream_events: Sequence[object],
        rewritten_text: str,
        guardrail_name: str,
    ) -> None:
        """Deliver a text rewrite on a stream with no completed envelope by
        spreading it over the text parts the guardrail scanned, in stream
        order: the whole rewrite on the first part and every later part
        blanked, through the same sync the envelope path uses. A scanned
        event the sync cannot place (one that is not an ``output_text`` or
        ``refusal`` delta or done, or lacks integer ``output_index`` / ``content_index``) makes
        the rewrite undeliverable, so the pipeline executor discards it and
        releases the original events."""
        scanned_events: Final = tuple(event for event in stream_events if _carries_stream_text(event))
        scanned_positions: Final = tuple(
            dict.fromkeys(
                (stream_item_field(event, "output_index"), stream_item_field(event, "content_index"))
                for event in scanned_events
            )
        )
        placeable_positions: Final = tuple(
            _MessageContentPosition(output_index, content_index)
            for output_index, content_index in scanned_positions
            if isinstance(output_index, int) and isinstance(content_index, int)
        )
        if len(placeable_positions) != len(scanned_positions) or any(
            stream_item_field(event, "type") not in _OUTPUT_TEXT_EVENT_TYPES | _TEXT_DELTA_EVENT_TYPES
            for event in scanned_events
        ):
            from litellm.proxy.policy_engine.pipeline_executor import UndeliverableStreamRewrite

            raise UndeliverableStreamRewrite(
                guardrail_name,
                "the scanned text events are not all output_text or refusal deltas with an integer output_index and "
                "content_index, so the text rewrite has nowhere to land",
            )
        self._sync_stream_events_with_rewrites(
            stream_events=stream_events,
            rewrites_by_position=MappingProxyType(dict(zip(placeable_positions, chain((rewritten_text,), repeat(""))))),
        )

    def message_part_texts(self, response: "ResponsesAPIResponse") -> Mapping[_MessageContentPosition, str]:
        """Text of every message content part, keyed by its position in ``response.output``."""
        texts: Final[list[str]] = []  # mutable-ok: the shared extractor below appends into caller-owned lists
        positions: Final[list[_OutputPosition]] = []  # mutable-ok: the same extractor's position sink
        for output_idx, output_item in _answer_output_items(stream_item_items(response, "output")):
            self._extract_output_text_and_images(
                output_item=output_item,
                output_idx=output_idx,
                texts_to_check=texts,
                images_to_check=[],
                task_mappings=positions,
            )
        return MappingProxyType(
            {
                position: text
                for position, text in zip(positions, texts)
                if isinstance(position, _MessageContentPosition)
            }
        )

    def stream_text_fits_message_parts(self, response: "ResponsesAPIResponse", stream_events: Sequence[object]) -> bool:
        message_positions: Final = frozenset(self.message_part_texts(response))
        return all(
            stream_item_field(event, "type") in _OUTPUT_TEXT_EVENT_TYPES
            and (stream_item_field(event, "output_index"), stream_item_field(event, "content_index"))
            in message_positions
            for event in stream_events
            if _carries_stream_text(event)
        )

    async def spread_text_rewrite(
        self,
        response: "ResponsesAPIResponse",
        rewritten_text: str,
        stream_events: Sequence[object] = (),
    ) -> None:
        """Replace all message text in ``response`` with one rewrite, for a guardrail that answers
        one string for everything it scanned: the whole rewrite goes on the first part and every
        later part is blanked, in the body and in any buffered events spelling out the same text."""
        rewrites_by_position: Final[Mapping[_OutputItemPosition, str]] = MappingProxyType(
            dict(zip(self.message_part_texts(response), chain((rewritten_text,), repeat(""))))
        )
        await self._apply_guardrail_responses_to_output(
            response=response,
            responses=list(rewrites_by_position.values()),
            task_mappings=list(rewrites_by_position),
        )
        self._sync_stream_events_with_rewrites(stream_events=stream_events, rewrites_by_position=rewrites_by_position)

    @staticmethod
    def _write_event_field(event: object, field: str, value: object) -> None:
        if isinstance(event, dict):
            event[field] = value  # rebind-ok: delivering the rewrite means editing the buffered event in place
        else:
            setattr(event, field, value)

    @staticmethod
    def _write_content_part_text(part: object, text: str) -> None:
        field: Final = _content_part_text_field(part)
        if _has_writable_field(part, field):
            OpenAIResponsesHandler._write_event_field(part, field, text)

    @staticmethod
    def _write_compaction_summary(item: object, summary: str) -> None:
        encrypted_content: Final = stream_item_field(item, "encrypted_content")
        if isinstance(encrypted_content, str):
            OpenAIResponsesHandler._write_event_field(
                item, "encrypted_content", with_compaction_summary(encrypted_content, summary)
            )

    @staticmethod
    def _write_part_text(parts: Sequence[object], index: int, text: str) -> None:
        if 0 <= index < len(parts):
            OpenAIResponsesHandler._write_content_part_text(parts[index], text)

    @staticmethod
    def _write_reasoning_rewrite(item: object, position: _ReasoningPosition, text: str) -> None:
        match position:
            case _ReasoningSummaryPosition():
                OpenAIResponsesHandler._write_part_text(
                    stream_item_items(item, "summary"), position.summary_index, text
                )
            case _ReasoningContentPosition():
                OpenAIResponsesHandler._write_part_text(
                    stream_item_items(item, "content"), position.content_index, text
                )
                if position.streams_as_summary:
                    OpenAIResponsesHandler._write_part_text(
                        stream_item_items(item, "summary"), position.content_index, text
                    )
            case _ThinkingBlockPosition():
                encrypted_content: Final = stream_item_field(item, "encrypted_content")
                if isinstance(encrypted_content, str):
                    OpenAIResponsesHandler._write_event_field(
                        item, "encrypted_content", with_thinking_text(encrypted_content, position.block_index, text)
                    )

    @staticmethod
    def _sync_reasoning_stream_event(
        event: _StreamEventFields,
        output_index: int,
        rewrites: _ReasoningStreamRewrites,
    ) -> None:
        summary_position: Final = (output_index, _stream_event_field(event, "summary_index") or 0)
        content_position: Final = (output_index, _stream_event_field(event, "content_index"))
        write_field: Final = OpenAIResponsesHandler._write_event_field
        part: Final = _stream_event_field(event, "part")
        part_field: Final = "reasoning" if isinstance(stream_item_field(part, "reasoning"), str) else "text"
        match _stream_event_field(event, "type"):
            case "response.reasoning_summary_text.delta" if summary_position in rewrites.summaries:
                write_field(event, "delta", next(rewrites.summary_deltas[summary_position]))
            case "response.reasoning_summary_text.done" if summary_position in rewrites.summaries:
                write_field(event, "text", rewrites.summaries[summary_position])
            case "response.reasoning_summary_part.added" | "response.reasoning_summary_part.done" if (
                summary_position in rewrites.summaries and stream_item_field(part, "text")
            ):
                write_field(part, "text", rewrites.summaries[summary_position])
            case "response.reasoning_text.delta" if content_position in rewrites.contents:
                write_field(event, "delta", next(rewrites.content_deltas[content_position]))
            case "response.reasoning_text.done" if content_position in rewrites.contents:
                write_field(event, "text", rewrites.contents[content_position])
            case "response.content_part.added" if content_position in rewrites.contents and stream_item_field(
                part, part_field
            ):
                write_field(part, part_field, rewrites.contents[content_position])
            case "response.content_part.done" if content_position in rewrites.contents and part is not None:
                write_field(part, part_field, rewrites.contents[content_position])
            case _:
                pass

    def _sync_stream_events_with_rewrites(
        self,
        stream_events: Sequence[object],
        rewrites_by_position: Mapping[_OutputItemPosition, str],
    ) -> None:
        """Sync pre-completion stream events with the rewritten completed
        response, keyed by the position each text holds in it: the first
        ``output_text.delta`` or ``refusal.delta`` for a rewritten item carries
        the full rewritten text and the rest are blanked, while
        ``output_text.done``, ``refusal.done``, ``content_part.done``, and
        ``output_item.done`` events carry the full rewritten text, so every
        event a client may read agrees with the rewritten ``response.completed``
        payload. Reasoning summary and reasoning text events follow the same
        rule."""
        delta_replacements: Final = _delta_replacements(rewrites_by_position)
        reasoning_rewrites: Final = _reasoning_stream_rewrites(rewrites_by_position)
        for event in stream_events:
            if not has_attributes(event, _StreamEventFields, "get"):
                continue
            event_type = _stream_event_field(event, "type")
            output_index = _stream_event_field(event, "output_index")
            content_index = _stream_event_field(event, "content_index")
            if _carries_whole_output_item(event) and isinstance(output_index, int):
                self._sync_output_item_done_event(
                    _stream_event_field(event, "item"), output_index, rewrites_by_position
                )
                continue
            if isinstance(output_index, int):
                self._sync_reasoning_stream_event(event, output_index, reasoning_rewrites)
            if not isinstance(output_index, int) or not isinstance(content_index, int):
                continue
            position = _MessageContentPosition(output_index, content_index)
            if event_type in _TEXT_DELTA_EVENT_TYPES and position in delta_replacements:
                self._write_event_field(event, "delta", next(delta_replacements[position]))
            elif event_type == "response.output_text.done" and position in rewrites_by_position:
                self._write_event_field(event, "text", rewrites_by_position[position])
            elif event_type == "response.refusal.done" and position in rewrites_by_position:
                self._write_event_field(event, "refusal", rewrites_by_position[position])
            elif event_type == "response.content_part.done" and position in rewrites_by_position:
                self._write_content_part_text(event.get("part"), rewrites_by_position[position])

    @staticmethod
    def _sync_provider_fields_to_stream_events(stream_events: Sequence[object], response: object) -> None:
        earlier_responses: Final = tuple(
            earlier
            for earlier in (stream_item_field(event, "response") for event in stream_events)
            if stream_item_field(earlier, "provider_specific_fields") is not None
        )
        for earlier in earlier_responses:
            OpenAIResponsesHandler._write_event_field(
                earlier, "provider_specific_fields", stream_item_field(response, "provider_specific_fields")
            )

    @staticmethod
    def _sync_output_item_done_event(
        item: object,
        output_index: int,
        rewrites_by_position: Mapping[_OutputItemPosition, str],
    ) -> None:
        summary_rewrite: Final = rewrites_by_position.get(_CompactionSummaryPosition(output_index))
        if summary_rewrite is not None:
            OpenAIResponsesHandler._write_compaction_summary(item, summary_rewrite)
            return
        item_rewrites: Final = tuple(
            (position, rewritten)
            for position, rewritten in rewrites_by_position.items()
            if position.output_index == output_index
        )
        if stream_item_field(item, "type") == "reasoning":
            for position, rewritten in item_rewrites:
                if isinstance(position, _REASONING_POSITION_TYPES):
                    OpenAIResponsesHandler._write_reasoning_rewrite(item, position, rewritten)
            return
        content: Final = stream_item_items(item, "content")
        for position, rewritten in item_rewrites:
            if isinstance(position, _MessageContentPosition):
                OpenAIResponsesHandler._write_part_text(content, position.content_index, rewritten)

    def _deliver_ended_stream_tool_call_rewrites(
        self,
        responses_so_far: Sequence[object],
        outputs: Sequence[object],
        pre_guardrail_tool_calls: tuple[_ToolCallShape, ...],
        post_guardrail_tool_calls: tuple[_ToolCallShape, ...],
        guardrail_name: str,
    ) -> None:
        """Write ended-stream guardrail tool-call rewrites into the completed
        envelope's ``function_call`` and ``custom_tool_call`` items and sync the
        earlier stream events, keyed by ``call_id`` (a ``tool_search_call`` without one, as
        OpenAI streams its server search, is keyed by its item ``id``). The guardrail sees the
        envelope's tool calls in output order, which is how a rewritten call
        finds its ``call_id``; the stream events find their call through the
        ``call_id`` on ``output_item`` events and the ``item_id`` on argument
        and custom-input events, since an
        event's ``output_index`` need not match the envelope's (the chat bridge
        numbers tool calls from 1 while the envelope lists them after the
        message). A rewrite whose calls do not line up with the envelope, or
        whose events cannot be found, is reported as undeliverable, so the
        pipeline executor discards it and releases the original events."""
        if post_guardrail_tool_calls == pre_guardrail_tool_calls:
            return
        indexed_tool_call_items: Final = _indexed_tool_call_output_items(outputs)
        tool_call_items: Final = tuple(output_item for _, output_item in indexed_tool_call_items)
        call_ids: Final = tuple(
            call_id for output_item in tool_call_items if (call_id := _tool_call_item_key(output_item)) is not None
        )
        stream_events: Final = responses_so_far[:-1]
        call_id_by_item_id: Final = self._tool_call_ids_by_item_id(stream_events)
        event_call_ids: Final = tuple(
            self._tool_call_event_call_id(event, call_id_by_item_id) for event in stream_events
        )
        rewrites_by_call_id: Final = MappingProxyType(
            {
                call_id: _tool_call_rewrite(before, after)
                for call_id, before, after in zip(call_ids, pre_guardrail_tool_calls, post_guardrail_tool_calls)
                if after != before
            }
        )
        unresolved_argument_event: Final = any(
            call_id is None and stream_item_field(event, "type") in _TOOL_CALL_PAYLOAD_EVENT_TYPES
            for event, call_id in zip(stream_events, event_call_ids)
        )
        undeliverable_reason: Final = _undeliverable_tool_call_rewrite_reason(
            call_ids=call_ids,
            tool_call_item_count=len(tool_call_items),
            post_guardrail_tool_call_count=len(post_guardrail_tool_calls),
            unresolved_argument_event=unresolved_argument_event,
            rewritten_call_ids=frozenset(rewrites_by_call_id),
            event_call_ids=frozenset(call_id for call_id in event_call_ids if call_id is not None),
        )
        if undeliverable_reason is not None:
            from litellm.proxy.policy_engine.pipeline_executor import UndeliverableStreamRewrite

            raise UndeliverableStreamRewrite(guardrail_name, undeliverable_reason)
        for (index, output_item), rewrite in (
            (indexed_item, rewrites_by_call_id[call_id])
            for indexed_item, call_id in zip(indexed_tool_call_items, call_ids)
            if call_id in rewrites_by_call_id
        ):
            _deliver_into_output(
                outputs, index, output_item, self._write_tool_call_item(output_item, rewrite.name, rewrite.arguments)
            )
        delta_replacements: Final = MappingProxyType(
            {call_id: chain((rewrite.arguments,), repeat("")) for call_id, rewrite in rewrites_by_call_id.items()}
        )
        for event, call_id in zip(stream_events, event_call_ids):
            if call_id not in rewrites_by_call_id:
                continue
            match stream_item_field(event, "type"):
                case str() as event_type if event_type in _TOOL_CALL_PAYLOAD_DELTA_EVENT_TYPES:
                    self._write_event_field(event, "delta", next(delta_replacements[call_id]))
                case str() as event_type if event_type in _TOOL_CALL_PAYLOAD_DONE_EVENT_FIELDS:
                    self._write_event_field(
                        event, _TOOL_CALL_PAYLOAD_DONE_EVENT_FIELDS[event_type], rewrites_by_call_id[call_id].arguments
                    )
                case "response.output_item.added":
                    self._write_tool_call_item(
                        stream_item_field(event, "item"), rewrites_by_call_id[call_id].name, None
                    )
                case "response.output_item.done":
                    done_item = stream_item_field(event, "item")
                    delivered_item = self._write_tool_call_item(
                        done_item, rewrites_by_call_id[call_id].name, rewrites_by_call_id[call_id].arguments
                    )
                    if delivered_item is not done_item:
                        self._write_event_field(event, "item", delivered_item)
                case _:
                    pass

    def _write_tool_call_rewrites_to_output(
        self,
        outputs: Sequence[object],
        pre_guardrail_tool_calls: tuple[_ToolCallShape, ...],
        post_guardrail_tool_calls: tuple[_ToolCallShape, ...],
    ) -> None:
        indexed_tool_call_items: Final = _indexed_tool_call_output_items(outputs)
        if len(indexed_tool_call_items) != len(post_guardrail_tool_calls):
            return
        for (index, output_item), rewrite in (
            (indexed_item, _tool_call_rewrite(before, after))
            for indexed_item, before, after in zip(
                indexed_tool_call_items, pre_guardrail_tool_calls, post_guardrail_tool_calls
            )
            if after != before
        ):
            _deliver_into_output(
                outputs, index, output_item, self._write_tool_call_item(output_item, rewrite.name, rewrite.arguments)
            )

    @staticmethod
    def _tool_call_ids_by_item_id(stream_events: Sequence[object]) -> Mapping[str, str]:
        items: Final = tuple(
            stream_item_field(event, "item")
            for event in stream_events
            if stream_item_field(event, "type") in _OUTPUT_ITEM_EVENT_TYPES
        )
        return MappingProxyType(
            {
                item_id: call_id
                for item in items
                if stream_item_field(item, "type") in _TOOL_CALL_ITEM_TYPES
                and isinstance(item_id := stream_item_field(item, "id"), str)
                and isinstance(call_id := stream_item_field(item, "call_id"), str)
            }
        )

    @staticmethod
    def _tool_call_event_call_id(event: object, call_id_by_item_id: Mapping[str, str]) -> str | None:
        event_type: Final = stream_item_field(event, "type")
        if event_type in _TOOL_CALL_PAYLOAD_EVENT_TYPES:
            item_id: Final = stream_item_field(event, "item_id")
            return call_id_by_item_id.get(item_id) if isinstance(item_id, str) else None
        if event_type not in _OUTPUT_ITEM_EVENT_TYPES:
            return None
        item: Final = stream_item_field(event, "item")
        return _tool_call_item_key(item) if _is_output_tool_call_item(item) else None

    @staticmethod
    def _write_tool_call_item(item: object, name: str | None, payload: str | None) -> object:
        """Apply a rewrite and return the item that carries it.

        A ``tool_search_call`` keeps its name and takes the rewritten arguments as a JSON
        object on a copy, since its item may be frozen. Other tool calls are edited in place.
        """
        item_type: Final = stream_item_field(item, "type")
        if item is None:
            return item
        if item_type == _TOOL_SEARCH_CALL_TYPE:
            return _rewritten_tool_search_call(item, payload)
        if item_type == _WEB_SEARCH_CALL_TYPE:
            return _rewritten_web_search_call(item, payload)
        if name is not None:
            OpenAIResponsesHandler._write_event_field(item, "name", name)
        if payload is not None and isinstance(item_type, str) and item_type in _TOOL_CALL_PAYLOAD_FIELDS:
            OpenAIResponsesHandler._write_event_field(item, _TOOL_CALL_PAYLOAD_FIELDS[item_type], payload)
        return item

    def _check_streaming_has_ended(self, responses_so_far: Sequence[object]) -> bool:
        """
        Check if the streaming has ended.
        """
        if not responses_so_far:
            return False
        return stream_item_field(responses_so_far[-1], "type") in _TERMINAL_ENVELOPE_EVENT_TYPES

    def get_streaming_scan_key(self, responses_so_far: Sequence[object]) -> StreamingScanKey | None:
        if not responses_so_far or not hasattr(responses_so_far[-1], "get"):
            return None
        last_event: Final = responses_so_far[-1]
        last_event_type: Final = stream_item_field(last_event, "type")
        if last_event_type == ResponsesAPIStreamEvents.OUTPUT_ITEM_DONE.value:
            return None
        if last_event_type in _TERMINAL_ENVELOPE_EVENT_TYPES:
            return self._completed_response_scan_key(stream_item_field(last_event, "response"))
        return StreamingScanKey(
            texts=(self.get_streaming_string_so_far(responses_so_far),),
            tool_calls_in_flight=self._has_streamed_tool_call_events(responses_so_far),
        )

    def released_stream_as_ended(self, responses_so_far: Sequence[object]) -> tuple[object, ...]:
        if self._check_streaming_has_ended(responses_so_far):
            return tuple(responses_so_far)
        ends_on_finished_item: Final = (
            bool(responses_so_far)
            and stream_item_field(responses_so_far[-1], "type") == ResponsesAPIStreamEvents.OUTPUT_ITEM_DONE.value
        )
        envelope_holds_released_reasoning: Final = bool(
            _released_reasoning_items(responses_so_far)
        ) and not _has_unplaceable_stream_text(responses_so_far)
        if (
            not ends_on_finished_item
            and not self._has_streamed_tool_call_events(responses_so_far)
            and not envelope_holds_released_reasoning
        ):
            return tuple(responses_so_far)
        return (*responses_so_far, _released_envelope(responses_so_far))

    def starts_client_tool_call(self, item: object) -> bool:
        return self._has_streamed_tool_call_events((item,)) or (
            stream_item_field(item, "type") in _OUTPUT_ITEM_EVENT_TYPES
            and stream_item_field(stream_item_field(item, "item"), "type") in _CLIENT_RUN_TOOL_CALL_ITEM_TYPES
        )

    @staticmethod
    def _has_streamed_tool_call_events(responses_so_far: Sequence[object]) -> bool:
        return any(
            stream_item_field(event, "type") in _TOOL_CALL_PAYLOAD_EVENT_TYPES
            or (
                stream_item_field(event, "type") in _OUTPUT_ITEM_EVENT_TYPES
                and _is_output_tool_call_item(stream_item_field(event, "item"))
            )
            for event in responses_so_far
        )

    @staticmethod
    def _completed_response_scan_key(response: object) -> StreamingScanKey:
        output_items: Final = stream_item_items(response, "output")
        message_items: Final = tuple(item for item in output_items if not _is_output_tool_call_item(item))
        message_parts: Final = chain.from_iterable(stream_item_items(item, "content") for item in message_items)
        part_texts: Final = tuple(
            text for part in message_parts if isinstance(text := stream_item_field(part, "text"), str) and text
        )
        summaries: Final = tuple(summary for item in output_items if (summary := output_compaction_summary(item)))
        reasoning_texts: Final = chain.from_iterable(
            _reasoning_output_texts(item, output_index) for output_index, item in enumerate(output_items)
        )
        return StreamingScanKey(
            texts=(
                *part_texts,
                *summaries,
                *(text for _, text in reasoning_texts),
                *(text for _, text in _provider_thinking_texts(response)),
            ),
            tool_calls=tuple(stream_item_fingerprint(item) for item in output_items if _is_output_tool_call_item(item)),
            stream_ended=True,
        )

    def build_stream_error_items(
        self,
        exc: "HTTPException",
        responses_so_far: Sequence[object] | None = None,
    ) -> Sequence[object] | None:
        from litellm.proxy.common_request_processing import (
            serialize_http_exception_detail,
        )

        message, _ = serialize_http_exception_detail(exc.detail)
        return (
            ErrorEvent(
                type=ResponsesAPIStreamEvents.ERROR,
                sequence_number=_next_stream_sequence_number(responses_so_far),
                error=ErrorEventError(
                    type="guardrail_error",
                    code=str(exc.status_code),
                    message=message,
                    param=None,
                ),
            ),
        )

    def get_streaming_string_so_far(self, responses_so_far: Sequence[object]) -> str:
        """
        Get the string so far from the responses so far.

        ``response.output_text.done`` events carry the whole part in ``text``, while
        ``response.output_text.delta`` events carry fragments in ``delta``. A stream
        that dies before its done event (``response.failed`` / ``response.incomplete``)
        has text only in deltas, so per content part the done text wins when present
        and the joined deltas fill in otherwise, never both.
        """
        return "".join(_stream_part_texts(responses_so_far).values())

    def _has_text_content(self, response: "ResponsesAPIResponse") -> bool:
        """
        Check if response has any text content to process.

        Override this method to customize text content detection.
        """
        if not hasattr(response, "output") or response.output is None:
            return False

        for output_item in response.output:
            if isinstance(output_item, BaseModel):
                try:
                    generic_response_output_item = GenericResponseOutputItem.model_validate(output_item.model_dump())
                    if generic_response_output_item.content:
                        output_item = generic_response_output_item
                except Exception:
                    continue
            if isinstance(output_item, (GenericResponseOutputItem, dict)):
                content = (
                    output_item.content
                    if isinstance(output_item, GenericResponseOutputItem)
                    else output_item.get("content", [])
                )
                if content:
                    for content_item in content:
                        # Check if it's an OutputText with text
                        if isinstance(content_item, OutputText):
                            if content_item.text:
                                return True
                        elif isinstance(content_item, dict):
                            if content_item.get("text"):
                                return True
        return False

    def _extract_output_text_and_images(
        self,
        output_item: object,
        output_idx: int,
        texts_to_check: list[str],
        images_to_check: list[str],
        task_mappings: list[_OutputPosition],
        tool_calls_to_check: list[ChatCompletionToolCallChunk] | None = None,
    ) -> None:
        """
        Extract text content, images, and tool calls from a response output item.

        Override this method to customize text/image/tool extraction logic.
        """

        tool_call_item: Final = _tool_call_output_item_mapping(output_item)
        if tool_call_item is not None:
            if tool_calls_to_check is not None:
                tool_calls_to_check.append(_scanned_tool_call(tool_call_item, output_idx))
            return

        summary: Final = output_compaction_summary(output_item)
        if summary is not None:
            texts_to_check.append(summary)
            task_mappings.append(_CompactionSummaryPosition(output_idx))
            return

        # Handle both GenericResponseOutputItem and dict
        content: Sequence[object] | None = None
        if isinstance(output_item, BaseModel):
            try:
                output_item_dump: Final = output_item.model_dump()
                generic_response_output_item: Final = GenericResponseOutputItem.model_validate(output_item_dump)
                if generic_response_output_item.content:
                    content = generic_response_output_item.content
            except Exception:
                # Try to extract content directly from output_item if validation fails
                if hasattr(output_item, "content") and output_item.content:
                    content = output_item.content
                else:
                    return
        elif isinstance(output_item, dict):
            content = output_item.get("content", [])
        else:
            return

        if not content:
            return

        verbose_proxy_logger.debug(
            "OpenAI Responses API: Processing output item %d with %d content parts", output_idx, len(content)
        )

        for content_idx, content_item in enumerate(content):
            text_content = stream_item_field(content_item, _content_part_text_field(content_item))
            if isinstance(text_content, str) and text_content:
                texts_to_check.append(text_content)
                task_mappings.append(_MessageContentPosition(output_idx, int(content_idx)))

    async def _apply_guardrail_responses_to_output(
        self,
        response: Union["ResponsesAPIResponse", ResponseOutputEnvelope],
        responses: Sequence[str],
        task_mappings: Sequence[_OutputPosition],
    ) -> None:
        """
        Apply guardrail responses back to output response.

        Override this method to customize how responses are applied.
        """
        response_output: Final = stream_item_items(response, "output")

        for task_idx, guardrail_response in enumerate(responses):
            position = task_mappings[task_idx]

            if isinstance(position, _ProviderThinkingBlockPosition):
                self._write_event_field(
                    _provider_thinking_blocks(response)[position.block_index], "thinking", guardrail_response
                )
                continue

            if position.output_index >= len(response_output):
                continue

            output_item = response_output[position.output_index]

            match position:
                case _CompactionSummaryPosition():
                    self._write_compaction_summary(output_item, guardrail_response)
                case _MessageContentPosition():
                    self._write_part_text(
                        stream_item_items(output_item, "content"), position.content_index, guardrail_response
                    )
                case _ReasoningSummaryPosition() | _ReasoningContentPosition() | _ThinkingBlockPosition():
                    self._write_reasoning_rewrite(output_item, position, guardrail_response)

    def build_block_sse_chunks(
        self,
        exc: "ModifyResponseException",
        stream_started: bool = False,
        responses_so_far: Sequence[object] | None = None,
    ) -> Sequence[bytes]:
        """
        Build Responses API SSE events that deliver the guardrail block message
        and terminate the stream cleanly, mirroring the non-streaming block
        response: a completed response whose only output is the violation text,
        with the real usage the upstream call consumed.

        - ``stream_started`` False (buffered / pre-stream): nothing has been
          sent, so emit the full synthetic sequence (``response.created``
          through ``response.completed``).
        - ``stream_started`` True (sampling / mid-stream): events already
          reached the client, so continue the in-progress response: close the
          output item still open on the wire, deliver the block message as a
          new output item under the same response id, and close with a
          ``response.completed`` carrying only the replacement item.

        """
        events: Final = (
            self._block_continuation_events(exc, responses_so_far or ())
            if stream_started
            else self._standalone_block_events(exc)
        )
        return tuple(
            f"event: {_STREAM_EVENT_TYPE_ADAPTER.validate_python(stream_item_field(event, 'type'))}\n"
            f"data: {event.model_dump_json(exclude_none=True, exclude_unset=True, serialize_as_any=True)}\n\n".encode()
            for event in events
        )

    @staticmethod
    def _standalone_block_events(exc: "ModifyResponseException") -> Sequence[ResponsesAPIStreamingResponse]:
        from litellm.responses.streaming_iterator import build_synthetic_response_events

        return build_synthetic_response_events(
            transformed=build_blocked_response(exc),
            logging_obj=None,
            chunk_size=max(len(exc.message), 1),
        )

    @staticmethod
    def _block_continuation_events(
        exc: "ModifyResponseException", responses_so_far: Sequence[object]
    ) -> Sequence[ResponsesAPIStreamingResponse]:
        from litellm.responses.streaming_iterator import opening_item_payload, opening_part_payload

        response_id, model, output_index = _continuation_identity(exc, responses_so_far)
        item: Final = _blocked_output_item(exc)
        item_id: Final = item.id
        part: Final[_BlockedContentPart] = {"type": "output_text", "text": exc.message, "annotations": ()}
        done_part: Final[_BlockedDoneContentPart] = {
            "type": "output_text",
            "text": exc.message,
            "annotations": (),
            "logprobs": None,
        }
        events: Final = (
            *_open_item_closing_events(responses_so_far),
            OutputItemAddedEvent(
                type=ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED,
                output_index=output_index,
                item=BaseLiteLLMOpenAIResponseObject.model_validate(opening_item_payload(item.model_dump())),
            ),
            ContentPartAddedEvent(
                type=ResponsesAPIStreamEvents.CONTENT_PART_ADDED,
                item_id=item_id,
                output_index=output_index,
                content_index=0,
                part=BaseLiteLLMOpenAIResponseObject.model_validate(opening_part_payload(part)),
            ),
            OutputTextDeltaEvent(
                type=ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA,
                item_id=item_id,
                output_index=output_index,
                content_index=0,
                delta=exc.message,
            ),
            OutputTextDoneEvent(
                type=ResponsesAPIStreamEvents.OUTPUT_TEXT_DONE,
                item_id=item_id,
                output_index=output_index,
                content_index=0,
                text=exc.message,
            ),
            ContentPartDoneEvent(
                type=ResponsesAPIStreamEvents.CONTENT_PART_DONE,
                item_id=item_id,
                output_index=output_index,
                content_index=0,
                part=ContentPartDonePartOutputText.model_validate(done_part),
            ),
            OutputItemDoneEvent(
                type=ResponsesAPIStreamEvents.OUTPUT_ITEM_DONE,
                output_index=output_index,
                item=item,
            ),
            ResponseCompletedEvent(
                type=ResponsesAPIStreamEvents.RESPONSE_COMPLETED,
                response=_blocked_response(exc, response_id=response_id, model=model, output_item=item),
            ),
        )
        return tuple(
            event.model_copy(update={"sequence_number": number})
            for number, event in enumerate(events, start=_next_stream_sequence_number(responses_so_far))
        )


class _BlockedContentPart(TypedDict):
    type: ReadOnly[str]
    text: ReadOnly[str]
    annotations: ReadOnly[tuple[object, ...]]


class _BlockedDoneContentPart(TypedDict):
    type: ReadOnly[str]
    text: ReadOnly[str]
    annotations: ReadOnly[tuple[object, ...]]
    logprobs: ReadOnly[None]


class _BlockedItemPayload(TypedDict):
    type: ReadOnly[str]
    id: ReadOnly[str]
    status: ReadOnly[str]
    role: ReadOnly[str]
    content: ReadOnly[tuple[_BlockedContentPart, ...]]


class _BlockedResponsePayload(TypedDict):
    id: ReadOnly[str]
    object: ReadOnly[str]
    created_at: ReadOnly[int]
    model: ReadOnly[str]
    output: ReadOnly[tuple[GenericResponseOutputItem, ...]]
    status: ReadOnly[str]
    usage: ReadOnly[ResponseAPIUsage]


def _blocked_output_item(exc: "ModifyResponseException") -> GenericResponseOutputItem:
    payload: Final[_BlockedItemPayload] = {
        "type": "message",
        "id": f"msg_{uuid.uuid4()}",
        "status": "completed",
        "role": "assistant",
        "content": ({"type": "output_text", "text": exc.message, "annotations": ()},),
    }
    return GenericResponseOutputItem.model_validate(payload)


def build_blocked_response(exc: "ModifyResponseException") -> ResponsesAPIResponse:
    return _blocked_response(exc, response_id=f"resp_{uuid.uuid4()}", model=exc.model)


def _blocked_response(
    exc: "ModifyResponseException",
    response_id: str,
    model: str,
    output_item: GenericResponseOutputItem | None = None,
) -> ResponsesAPIResponse:
    payload: Final[_BlockedResponsePayload] = {
        "id": response_id,
        "object": "response",
        "created_at": int(time.time()),
        "model": model,
        "output": (output_item if output_item is not None else _blocked_output_item(exc),),
        "status": "completed",
        "usage": blocked_responses_stream_usage(exc.original_response),
    }
    return ResponsesAPIResponse.model_validate(payload)


def _continuation_identity(exc: "ModifyResponseException", responses_so_far: Sequence[object]) -> tuple[str, str, int]:
    responses: Final = tuple(
        response for item in responses_so_far if (response := stream_item_field(item, "response")) is not None
    )
    response_id: Final = next(
        (rid for response in responses if isinstance(rid := stream_item_field(response, "id"), str) and rid),
        f"resp_{uuid.uuid4()}",
    )
    model: Final = next(
        (m for response in responses if isinstance(m := stream_item_field(response, "model"), str) and m),
        exc.model,
    )
    indices: Final = tuple(
        index for item in responses_so_far if isinstance(index := stream_item_field(item, "output_index"), int)
    )
    return response_id, model, max(indices) + 1 if indices else 0


@dataclass(frozen=True, slots=True)
class _OpenItemState:
    item_id: str
    item_type: str
    role: str
    output_index: int
    content_index: int
    text: str
    part_open: bool
    payload: object


def _open_item_state(responses_so_far: Sequence[object]) -> _OpenItemState | None:
    typed: Final = tuple((stream_item_field(event, "type"), event) for event in responses_so_far)
    added: Final = tuple(
        (added_index, stream_item_field(event, "item"))
        for event_type, event in typed
        if event_type == "response.output_item.added"
        and isinstance(added_index := stream_item_field(event, "output_index"), int)
    )
    done_indices: Final = frozenset(
        done_index
        for event_type, event in typed
        if event_type == "response.output_item.done"
        and isinstance(done_index := stream_item_field(event, "output_index"), int)
    )
    open_added: Final = tuple((index, payload) for index, payload in added if index not in done_indices)
    if not open_added:
        return None
    output_index, item_payload = open_added[-1]
    if item_payload is None:
        return None
    item_id: Final = stream_item_field(item_payload, "id")
    if not isinstance(item_id, str) or not item_id:
        return None
    raw_type: Final = stream_item_field(item_payload, "type")
    raw_role: Final = stream_item_field(item_payload, "role")
    part_added: Final = tuple(
        part_index
        for event_type, event in typed
        if event_type == "response.content_part.added"
        and stream_item_field(event, "item_id") == item_id
        and isinstance(part_index := stream_item_field(event, "content_index"), int)
    )
    part_done: Final = frozenset(
        part_done_index
        for event_type, event in typed
        if event_type == "response.content_part.done"
        and stream_item_field(event, "item_id") == item_id
        and isinstance(part_done_index := stream_item_field(event, "content_index"), int)
    )
    open_parts: Final = tuple(index for index in part_added if index not in part_done)
    text: Final = "".join(
        delta
        for event_type, event in typed
        if event_type == "response.output_text.delta"
        and stream_item_field(event, "item_id") == item_id
        and isinstance(delta := stream_item_field(event, "delta"), str)
    )
    return _OpenItemState(
        item_id=item_id,
        item_type=raw_type if isinstance(raw_type, str) and raw_type else "message",
        role=raw_role if isinstance(raw_role, str) and raw_role else "assistant",
        output_index=output_index,
        content_index=open_parts[-1] if open_parts else 0,
        text=text,
        part_open=bool(open_parts),
        payload=item_payload,
    )


_item_fields_adapter: Final = TypeAdapter(Mapping[str, object])
_no_item_fields: Final[Mapping[str, object]] = MappingProxyType({})


def _incomplete_item_fields(payload: object) -> Mapping[str, object]:
    raw: Final = payload.model_dump() if isinstance(payload, BaseModel) else payload
    if not isinstance(raw, dict):
        return _no_item_fields
    return _item_fields_adapter.validate_python(raw)


def _open_item_closing_events(responses_so_far: Sequence[object]) -> Sequence[ResponsesAPIStreamingResponse]:
    """Close the output item still in progress on the relayed stream before the
    block item is appended: strict Responses clients reject a
    ``response.completed`` that arrives while an earlier ``output_item.added``
    was never closed. A message item closes ``completed`` with exactly the text
    the client has received so far; any other item type (a function call the
    guardrail rejected, for instance) closes ``incomplete`` so the synthetic
    done event can never authorize acting on it."""
    open_item: Final = _open_item_state(responses_so_far)
    if open_item is None:
        return ()
    if open_item.item_type != "message":
        return (
            OutputItemDoneEvent(
                type=ResponsesAPIStreamEvents.OUTPUT_ITEM_DONE,
                output_index=open_item.output_index,
                item=BaseLiteLLMOpenAIResponseObject.model_validate(
                    MappingProxyType({**_incomplete_item_fields(open_item.payload), "status": "incomplete"})
                ),
            ),
        )
    partial_part: Final[_BlockedContentPart] = {
        "type": "output_text",
        "text": open_item.text,
        "annotations": (),
    }
    closed_payload: Final[_BlockedItemPayload] = {
        "type": open_item.item_type,
        "id": open_item.item_id,
        "status": "completed",
        "role": open_item.role,
        "content": (partial_part,),
    }
    item_done: Final = OutputItemDoneEvent(
        type=ResponsesAPIStreamEvents.OUTPUT_ITEM_DONE,
        output_index=open_item.output_index,
        item=GenericResponseOutputItem.model_validate(closed_payload),
    )
    if not open_item.part_open:
        return (item_done,)
    partial_done_part: Final[_BlockedDoneContentPart] = {
        "type": "output_text",
        "text": open_item.text,
        "annotations": (),
        "logprobs": None,
    }
    return (
        OutputTextDoneEvent(
            type=ResponsesAPIStreamEvents.OUTPUT_TEXT_DONE,
            item_id=open_item.item_id,
            output_index=open_item.output_index,
            content_index=open_item.content_index,
            text=open_item.text,
        ),
        ContentPartDoneEvent(
            type=ResponsesAPIStreamEvents.CONTENT_PART_DONE,
            item_id=open_item.item_id,
            output_index=open_item.output_index,
            content_index=open_item.content_index,
            part=ContentPartDonePartOutputText.model_validate(partial_done_part),
        ),
        item_done,
    )
