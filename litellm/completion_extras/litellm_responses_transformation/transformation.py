"""
Handler for transforming /chat/completions api requests to litellm.responses requests
"""

import json
import os
from collections.abc import AsyncIterator, Callable, Iterable, Iterator, Mapping, Sequence
from itertools import accumulate, chain
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final, Literal, TypedDict, TypeVar, Union, cast

from openai.types.chat import ChatCompletion
from openai.types.responses import Response
from openai.types.responses.custom_tool_param import CustomToolParam
from openai.types.responses.response_input_param import (
    FunctionCallOutput,
    ResponseCustomToolCallOutputParam,
    ResponseCustomToolCallParam,
)
from openai.types.responses.tool_choice_custom_param import ToolChoiceCustomParam
from openai.types.responses.tool_choice_function_param import ToolChoiceFunctionParam
from openai.types.responses.tool_param import FunctionToolParam
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

import litellm
from litellm import ModelResponse
from litellm._logging import verbose_logger
from litellm.integrations.anthropic_cache_control_hook import supports_openai_prompt_cache_breakpoint
from litellm.litellm_core_utils.hidden_params import get_hidden_params, get_or_create_hidden_params
from litellm.litellm_core_utils.prompt_templates.common_utils import (
    responses_reasoning_items_from_thinking_blocks,
    with_prompt_cache_breakpoint,
)
from litellm.llms.base_llm.base_model_iterator import BaseModelResponseIterator
from litellm.llms.base_llm.bridges.completion_transformation import (
    CompletionTransformationBridge,
)
from litellm.responses.litellm_completion_transformation.tool_search import is_anthropic_tool_search_chat_tool
from litellm.responses.sse_output_recovery import (
    parse_sse_json_chunk,
    record_output_item_chunk,
    record_output_text_chunk,
)
from litellm.responses.utils import ResponsesAPIRequestUtils, normalize_responses_api_stream_options
from litellm.types.llms.openai import (
    ChatCompletionAnnotation,
    ChatCompletionReasoningItem,
    ChatCompletionToolCallChunk,
    ChatCompletionToolCallFunctionChunk,
    ChatCompletionToolParamFunctionChunk,
    PromptCacheBreakpoint,
    Reasoning,
    ResponsesAPIOptionalRequestParams,
    ResponsesAPIResponse,
    ResponsesAPIStreamEvents,
)
from litellm.types.utils import GenericStreamingChunk, ModelResponseStream

if TYPE_CHECKING:
    from openai.types.responses import ResponseInputImageParam, ResponseOutputItem
    from openai.types.responses.response_text_config_param import (
        ResponseTextConfigParam as ResponseText,
    )
    from pydantic import BaseModel

    from litellm import LiteLLMLoggingObj
    from litellm.llms.base_llm.base_model_iterator import BaseModelResponseIterator
    from litellm.types.llms.openai import (
        ALL_RESPONSES_API_TOOL_PARAMS,
        AllMessageValues,
        ChatCompletionFileObject,
        ChatCompletionImageObject,
        ChatCompletionRedactedThinkingBlock,
        ChatCompletionThinkingBlock,
        ChatCompletionToolReferenceObject,
        OpenAIMessageContentListBlock,
    )
    from litellm.types.utils import Choices, Message


_CHAT_COMPLETION_FIELDS: Final = frozenset((*ModelResponse.model_fields, "usage"))
_RESPONSES_API_ONLY_FIELDS: Final = frozenset((*Response.model_fields, *ResponsesAPIResponse.model_fields)) - frozenset(
    ChatCompletion.model_fields
)


def _strip_prompt_cache_breakpoint_from_content_block(value: object) -> object:
    if not isinstance(value, dict):
        return value
    content_block: Final = cast(dict[str, object], value)  # cast-ok: isinstance confirms the content block is a mapping
    return {key: item for key, item in content_block.items() if key != "prompt_cache_breakpoint"}


def _strip_prompt_cache_breakpoints_from_content(value: object) -> object:
    if isinstance(value, list):
        list_content: Final = cast(list[object], value)  # cast-ok: isinstance confirms a list of content blocks
        return [_strip_prompt_cache_breakpoint_from_content_block(item) for item in list_content]
    if isinstance(value, tuple):
        tuple_content: Final = cast(tuple[object, ...], value)  # cast-ok: isinstance confirms a tuple of content blocks
        return tuple(_strip_prompt_cache_breakpoint_from_content_block(item) for item in tuple_content)
    return _strip_prompt_cache_breakpoint_from_content_block(value)


def _strip_prompt_cache_breakpoints_from_item(value: object) -> object:
    if not isinstance(value, dict):
        return value
    input_item: Final = cast(dict[str, object], value)  # cast-ok: isinstance confirms a Responses input item mapping
    return {
        key: _strip_prompt_cache_breakpoints_from_content(item) if key in ("content", "output") else item
        for key, item in input_item.items()
        if key != "prompt_cache_breakpoint"
    }


def _strip_prompt_cache_breakpoints(input_items: list[object]) -> list[object]:
    return [_strip_prompt_cache_breakpoints_from_item(item) for item in input_items]


def _is_audio_input_part(part: object) -> bool:
    if not isinstance(part, dict):
        return False
    content_part: Final = cast(dict[str, object], part)  # cast-ok: isinstance confirms a content block mapping
    return content_part.get("type") == "input_audio"


def _breakpoint_of(part: object) -> object:
    if not isinstance(part, dict):
        return None
    content_part: Final = cast(dict[str, object], part)  # cast-ok: isinstance confirms a content block mapping
    return content_part.get("prompt_cache_breakpoint")


def _pending_audio_breakpoint(pending: object, part: object) -> object:
    if not _is_audio_input_part(part):
        return None
    return pending if pending is not None else _breakpoint_of(part)


def _carried_breakpoints(content: Sequence[object]) -> tuple[object, ...]:
    seeded_from_the_end: Final = chain((None,), reversed(content))
    pending_after_each: Final = tuple(accumulate(seeded_from_the_end, _pending_audio_breakpoint))
    return tuple(reversed(pending_after_each[:-1]))


def _with_carried_breakpoint(part: object, marker: object) -> object:
    if marker is None or not isinstance(part, dict) or _breakpoint_of(part) is not None:
        return part
    kept_part: Final = cast(dict[str, object], part)  # cast-ok: isinstance confirms a content block mapping
    return {**kept_part, "prompt_cache_breakpoint": marker}


def _without_audio_input_parts_in_content(value: object) -> object:
    if not isinstance(value, list):
        return value
    content: Final = cast(list[object], value)  # cast-ok: isinstance confirms a list of content blocks
    if not any(_is_audio_input_part(part) for part in content):
        return value
    return [
        _with_carried_breakpoint(part, marker)
        for part, marker in zip(content, _carried_breakpoints(content))
        if not _is_audio_input_part(part)
    ]


def _without_audio_input_parts_in_item(value: object) -> object:
    if not isinstance(value, dict):
        return value
    input_item: Final = cast(dict[str, object], value)  # cast-ok: isinstance confirms a Responses input item mapping
    return {
        key: _without_audio_input_parts_in_content(item) if key in ("content", "output") else item
        for key, item in input_item.items()
    }


def _without_audio_input_parts(input_items: list[object]) -> list[object]:
    return [_without_audio_input_parts_in_item(item) for item in input_items]


def _served_provider(litellm_params: Mapping[str, object]) -> str | None:
    custom_llm_provider: Final = litellm_params.get("custom_llm_provider")
    return custom_llm_provider if isinstance(custom_llm_provider, str) else None


def _supports_audio_input(model: str, litellm_params: Mapping[str, object]) -> bool:
    provider: Final = _served_provider(litellm_params)
    base_model: Final = litellm_params.get("base_model")
    return litellm.supports_audio_input(model=model, custom_llm_provider=provider) or (
        isinstance(base_model, str)
        and bool(base_model)
        and litellm.supports_audio_input(model=base_model, custom_llm_provider=provider)
    )


def _drops_audio_input(model: str, litellm_params: Mapping[str, object]) -> bool:
    if not (litellm_params.get("drop_params") or litellm.drop_params):
        return False
    return not _supports_audio_input(model, litellm_params)


def _provider_metadata(response_fields: Mapping[str, object] | None) -> Mapping[str, object]:
    return MappingProxyType(
        {
            key: value
            for key, value in (response_fields.items() if response_fields else ())
            if value is not None and key not in _CHAT_COMPLETION_FIELDS and key not in _RESPONSES_API_ONLY_FIELDS
        }
    )


def _upstream_response_id(response_id: str | None) -> str | None:
    if response_id is None:
        return None
    return ResponsesAPIRequestUtils.decode_previous_response_id_to_original_previous_response_id(response_id)


class _ReasoningSummaryText(TypedDict):
    type: str
    text: str


class _BuiltReasoningItem(TypedDict):
    type: Literal["reasoning"]
    id: str
    encrypted_content: str | None
    summary: Sequence[_ReasoningSummaryText]


def _get_reasoning_items(
    msg: "AllMessageValues",
) -> list[ChatCompletionReasoningItem]:
    """Extract reasoning_items from a message dict with proper typing."""
    items: Final = msg.get("reasoning_items")
    if items:
        return items
    return []


def _reasoning_input_items(msg: "AllMessageValues") -> list[dict[str, object]]:  # mutable-ok: API message payload
    """Reasoning input items for an assistant message.

    Stored reasoning items win because they carry an id the Responses API minted; thinking
    blocks are the fallback for turns that arrived over another API surface.
    """
    items: Final = _get_reasoning_items(msg)
    stored: Final = [_reasoning_item_to_response_input(item) for item in items]
    if stored:
        return stored
    raw_blocks: Final = msg.get("thinking_blocks") or ()
    blocks: Final = cast("Iterable[ChatCompletionThinkingBlock]", raw_blocks)  # cast-ok: untyped client json
    replayed: Final = responses_reasoning_items_from_thinking_blocks(blocks)
    return [dict(item) for item in replayed]


def _build_reasoning_item(
    item_id: str,
    encrypted_content: str | None,
    summary_raw: Iterable[object] | None,
) -> _BuiltReasoningItem:
    """Build a ChatCompletionReasoningItem-shaped dict from raw response data.

    Handles both pydantic objects (attribute access) and plain dicts.
    """
    summary: Final[list[_ReasoningSummaryText]] = []
    for s in summary_raw or []:
        if isinstance(s, dict):
            summary.append({"type": s.get("type", "summary_text"), "text": s.get("text", "")})
        else:
            summary.append(
                {
                    "type": getattr(s, "type", "summary_text"),
                    "text": getattr(s, "text", ""),
                }
            )
    return {
        "id": item_id,
        "type": "reasoning",
        "encrypted_content": encrypted_content,
        "summary": summary,
    }


def _reasoning_item_from_output_item(item: object) -> _BuiltReasoningItem | None:
    from openai.types.responses import ResponseReasoningItem

    if isinstance(item, ResponseReasoningItem):
        return _build_reasoning_item(
            item_id=item.id,
            encrypted_content=getattr(item, "encrypted_content", None),
            summary_raw=item.summary,
        )
    if isinstance(item, dict) and item.get("type") == "reasoning":
        return _build_reasoning_item(
            item_id=item.get("id", ""),
            encrypted_content=item.get("encrypted_content"),
            summary_raw=item.get("summary"),
        )
    return None


def _reasoning_items_from_output_items(output_items: Sequence[object]) -> tuple[_BuiltReasoningItem, ...]:
    return tuple(
        reasoning_item
        for reasoning_item in (_reasoning_item_from_output_item(item) for item in output_items)
        if reasoning_item is not None
    )


_TOOL_SEARCH_ITEM_TYPES: Final = frozenset({"tool_search_call", "tool_search_output"})
_TOOL_SEARCH_REPLAY_ITEM_TYPES: Final = frozenset({"reasoning", *_TOOL_SEARCH_ITEM_TYPES})
_TOOL_SEARCH_ITEMS_KEY: Final = "tool_search_items"


def _json_field(value: object, key: str) -> object:
    return cast(Mapping[str, object], value).get(key) if isinstance(value, dict) else None  # cast-ok: untyped JSON


def _tool_search_replay_item(item: object) -> Mapping[str, object] | None:
    raw: Final[object] = item.model_dump(exclude_unset=True) if isinstance(item, BaseModel) else item
    if not isinstance(raw, dict):
        return None
    fields: Final = cast(dict[str, object], raw)  # cast-ok: provider JSON item, keys forwarded verbatim
    if fields.get("type") not in _TOOL_SEARCH_REPLAY_ITEM_TYPES:
        return None
    return {key: value for key, value in fields.items() if value is not None}


def _tool_search_items_from_output_items(output_items: object) -> tuple[Mapping[str, object], ...]:
    if not isinstance(output_items, list):
        return ()
    entries: Final = cast(Sequence[object], output_items)  # cast-ok: each entry is re-validated
    items: Final = tuple(item for item in (_tool_search_replay_item(entry) for entry in entries) if item is not None)
    last_search: Final = max(
        (index for index, item in enumerate(items) if item.get("type") in _TOOL_SEARCH_ITEM_TYPES), default=-1
    )
    return items[: last_search + 1]


def _tool_search_provider_fields(
    items: Sequence[Mapping[str, object]],
) -> dict[str, object] | None:  # mutable-ok: Message.provider_specific_fields is typed as a dict
    return {_TOOL_SEARCH_ITEMS_KEY: [dict(item) for item in items]} if items else None


def _tool_search_input_items(msg: object) -> tuple[Mapping[str, object], ...]:
    return _tool_search_items_from_output_items(
        _json_field(_json_field(msg, "provider_specific_fields"), _TOOL_SEARCH_ITEMS_KEY)
    )


def carries_tool_search_items(provider_specific_fields: object) -> bool:
    return bool(_json_field(provider_specific_fields, _TOOL_SEARCH_ITEMS_KEY))


def _tool_definition_name(tool: object) -> tuple[str, ...]:
    name: Final = _json_field(tool, "server_label" if _json_field(tool, "type") == "mcp" else "name")
    return (name,) if isinstance(name, str) else ()


def _namespace_members(tool: object) -> Sequence[object]:
    members: Final = _json_field(tool, "tools")
    if _json_field(tool, "type") != "namespace" or not isinstance(members, list):
        return (tool,)
    return cast(Sequence[object], members)  # cast-ok: untyped JSON, each member re-checked


def _loaded_tools(item: Mapping[str, object]) -> Sequence[object]:
    tools: Final = item.get("tools")
    if item.get("type") != "tool_search_output" or not isinstance(tools, list):
        return ()
    return cast(Sequence[object], tools)  # cast-ok: untyped JSON, each tool re-checked


def _json_list(value: object) -> Sequence[object] | None:
    return cast(Sequence[object], value) if isinstance(value, list) else None  # cast-ok: untyped JSON


def _message_loaded_tools(message: object) -> Iterator[object]:
    loaded: Final = chain.from_iterable(_loaded_tools(item) for item in _tool_search_input_items(message))
    return chain.from_iterable(_namespace_members(tool) for tool in loaded)


def replayed_tool_search_tools(request: object) -> tuple[object, ...]:
    messages: Final = _json_list(_json_field(request, "messages")) or ()
    return tuple(chain.from_iterable(_message_loaded_tools(message) for message in messages))


def replayed_tool_search_tool_names(request: object) -> tuple[str, ...]:
    return tuple(chain.from_iterable(_tool_definition_name(tool) for tool in replayed_tool_search_tools(request)))


def _loads_any_tool(message: object, names: frozenset[str]) -> bool:
    loaded_names: Final = chain.from_iterable(_tool_definition_name(tool) for tool in _message_loaded_tools(message))
    return any(name in names for name in loaded_names)


def _without_tool_search_items(message: object) -> object:
    fields: Final = _json_field(message, "provider_specific_fields")
    if not isinstance(fields, dict):
        return message
    kept: Final = {
        key: value
        for key, value in cast(Mapping[str, object], fields).items()  # cast-ok: untyped JSON
        if key != _TOOL_SEARCH_ITEMS_KEY
    }
    return {**cast(Mapping[str, object], message), "provider_specific_fields": kept}  # cast-ok: dict checked above


def without_tool_search_items_loading(
    request: object, names: frozenset[str]
) -> list[object] | None:  # mutable-ok: request messages stay a list for every downstream handler
    messages: Final = _json_list(_json_field(request, "messages"))
    if messages is None:
        return None
    return [_without_tool_search_items(message) if _loads_any_tool(message, names) else message for message in messages]


def _effective_tools(request: object) -> Sequence[object]:
    override: Final = _json_list(_json_field(_json_field(request, "extra_body"), "tools"))
    return override if override is not None else (_json_list(_json_field(request, "tools")) or ())


def _requests_tool_search(request: object) -> bool:
    return any(_json_field(tool, "type") == "tool_search" for tool in _effective_tools(request))


def _reasoning_ids(items: Sequence[Mapping[str, object]]) -> frozenset[str]:
    ids: Final = (item.get("id") for item in items if item.get("type") == "reasoning")
    return frozenset(item_id for item_id in ids if isinstance(item_id, str))


def _assistant_replay_items(msg: "AllMessageValues", replay_tool_search: bool) -> tuple[Mapping[str, object], ...]:
    searched: Final = _tool_search_input_items(msg) if replay_tool_search else ()
    searched_reasoning_ids: Final = _reasoning_ids(searched)
    reasoning: Final = (item for item in _reasoning_input_items(msg) if item.get("id") not in searched_reasoning_ids)
    return (*searched, *reasoning)


def _as_chat_reasoning_items(
    reasoning_items: Sequence[_BuiltReasoningItem | ChatCompletionReasoningItem],
) -> list[ChatCompletionReasoningItem] | None:
    if not reasoning_items:
        return None
    return cast(list[ChatCompletionReasoningItem], list(reasoning_items))


def _shift_annotation(annotation: ChatCompletionAnnotation, offset: int) -> ChatCompletionAnnotation:
    shifted: Final = {
        key: value + offset if key in ("start_index", "end_index") and isinstance(value, int) else value
        for key, value in annotation.items()
    }
    return cast(ChatCompletionAnnotation, shifted)


def _message_annotations(message: "Message") -> Sequence[ChatCompletionAnnotation]:
    annotations: Sequence[ChatCompletionAnnotation] | None = getattr(message, "annotations", None)
    return annotations or ()


def _message_reasoning_content(message: "Message") -> str | None:
    reasoning_content: str | None = getattr(message, "reasoning_content", None)
    return reasoning_content


def _message_reasoning_items(message: "Message") -> Sequence[ChatCompletionReasoningItem]:
    reasoning_items: Sequence[ChatCompletionReasoningItem] | None = getattr(message, "reasoning_items", None)
    return reasoning_items or ()


def _shifted_annotations(messages: Sequence["Message"], offsets: Sequence[int]) -> Iterator[ChatCompletionAnnotation]:
    for message, offset in zip(messages, offsets):
        for annotation in _message_annotations(message):
            yield _shift_annotation(annotation, offset)


def _fold_choices_into_one(
    choices: Sequence["Choices"],
    tool_calls: Sequence[Mapping[str, object]],
    trailing_reasoning_content: str | None,
    trailing_reasoning_item: _BuiltReasoningItem | None = None,
    tool_search_items: Sequence[Mapping[str, object]] = (),
    structured_output: bool = False,
    trailing_reasoning_items: Sequence[_BuiltReasoningItem] = (),
) -> "Choices":
    """Clients that only read choices[0] must still see every text and tool call.

    Under a structured ``response_format`` each message item is a complete document, so
    ``content`` is the first one, the same item the openai SDK returns as ``output_parsed``.
    """
    from litellm.types.utils import Choices, Message

    messages: Final = tuple(choice.message for choice in choices)
    content_messages: Final = messages[:1] if structured_output else messages
    texts: Final = tuple(str(message.content or "") for message in content_messages)
    offsets: Final = (0, *accumulate(len(text) for text in texts))
    annotations: Final = tuple(_shifted_annotations(content_messages, offsets))
    reasoning_contents: Final = (*(_message_reasoning_content(m) for m in messages), trailing_reasoning_content)
    accumulated_trailing_items: Final = (
        *trailing_reasoning_items,
        *(() if trailing_reasoning_item is None else (trailing_reasoning_item,)),
    )
    reasoning_items: Final = (
        *chain.from_iterable(_message_reasoning_items(m) for m in messages),
        *accumulated_trailing_items,
    )
    return Choices(
        message=Message(
            role=messages[0].role if messages else "assistant",
            content="".join(texts) if messages else None,
            annotations=list(annotations) or None,
            tool_calls=list(tool_calls) or None,
            reasoning_content=" ".join(value for value in reasoning_contents if value) or None,
            reasoning_items=_as_chat_reasoning_items(reasoning_items),
            provider_specific_fields=_tool_search_provider_fields(tool_search_items),
        ),
        finish_reason="tool_calls" if tool_calls else "stop",
        index=0,
    )


def _append_reasoning_if_present(
    item: object,
    pending: list[_BuiltReasoningItem],  # mutable-ok: accumulator
    current_content: str | None,
) -> tuple[bool, str | None]:
    """Extract reasoning item if present, append to pending, and update concatenated summary."""
    reasoning_item: Final = _reasoning_item_from_output_item(item)
    if reasoning_item is None:
        return False, current_content
    pending.append(reasoning_item)
    summary_texts: Final = [s["text"] for s in reasoning_item["summary"] if s.get("text")]
    if not summary_texts:
        return True, current_content
    step_text: Final = " ".join(summary_texts)
    merged_content: Final = f"{current_content} {step_text}".strip() if current_content else step_text
    return True, merged_content


def _tool_call_dict_from_output_item_candidate(
    item: object,
    index: int,
    apply_patch_type: type | None,
) -> Mapping[str, object] | None:
    """Convert typed or raw tool call items into chat completion tool call mapping."""
    from openai.types.responses import ResponseFunctionToolCall

    if isinstance(item, ResponseFunctionToolCall):
        from litellm.responses.litellm_completion_transformation.transformation import (
            LiteLLMCompletionResponsesConfig,
        )

        return LiteLLMCompletionResponsesConfig.convert_response_function_tool_call_to_chat_completion_tool_call(
            tool_call_item=item,
            index=index,
        )

    if apply_patch_type is not None and isinstance(item, apply_patch_type):
        from litellm.responses.litellm_completion_transformation.transformation import (
            LiteLLMCompletionResponsesConfig,
        )

        return LiteLLMCompletionResponsesConfig.convert_apply_patch_tool_call_to_chat_completion_tool_call(
            tool_call_item=item,
            index=index,
        )

    if isinstance(item, (dict, BaseModel)):
        raw_item: Final = item if isinstance(item, dict) else item.model_dump()
        if raw_item.get("type") in ("function_call", "custom_tool_call"):
            return tool_call_dict_from_output_item(raw_item, index, include_namespace=True)

    return None


def _responses_event_type(chunk: Mapping[str, object]) -> object:
    event_type: Final = chunk.get("type")
    return event_type.value if isinstance(event_type, ResponsesAPIStreamEvents) else event_type


class _TypedPayload(BaseModel):
    """The ``type`` discriminator of a response_format dict or a Responses output item."""

    model_config = ConfigDict(frozen=True, extra="ignore")
    type: str | None = None


_TYPED_PAYLOAD: Final = TypeAdapter(_TypedPayload)


def _payload_type(payload: object) -> str | None:
    try:
        return _TYPED_PAYLOAD.validate_python(payload).type
    except ValidationError:
        return None


def _is_message_item_done(chunk: Mapping[str, object]) -> bool:
    if _responses_event_type(chunk) != ResponsesAPIStreamEvents.OUTPUT_ITEM_DONE.value:
        return False
    return _payload_type(chunk.get("item")) == "message"


def is_structured_output_request(optional_params: Mapping[str, object]) -> bool:
    return _payload_type(optional_params.get("response_format")) in ("json_schema", "json_object")


_ToolChoiceT = TypeVar("_ToolChoiceT")


def _map_incomplete_reason_to_finish_reason(incomplete_reason: str | None) -> Literal["length", "content_filter"]:
    if incomplete_reason == "content_filter":
        return "content_filter"
    return "length"


_PROMPT_CACHE_BREAKPOINT: Final = TypeAdapter(PromptCacheBreakpoint)


def _prompt_cache_breakpoint_for_wire(marker: object, drop_params: bool) -> object:
    if marker is None or not drop_params:
        return marker
    try:
        return _PROMPT_CACHE_BREAKPOINT.validate_python(marker)
    except ValidationError:
        verbose_logger.debug("Chat provider: dropping malformed prompt_cache_breakpoint %r under drop_params", marker)
        return None


def _input_file_from_file_value(file_value: object) -> dict[str, object]:
    if not isinstance(file_value, dict):
        return {"type": "input_file"}
    file_dict: Final = cast("dict[str, object]", file_value)  # cast-ok: runtime dict checked
    return {
        "type": "input_file",
        **{key: file_dict[key] for key in ("file_id", "file_data", "filename") if key in file_dict},
    }


def _incomplete_reason_from_response_payload(response_payload: object) -> str | None:
    if not isinstance(response_payload, Mapping):
        return None
    incomplete_details: Final = response_payload.get("incomplete_details")
    if not isinstance(incomplete_details, Mapping):
        return None
    reason: Final = incomplete_details.get("reason")
    return reason if isinstance(reason, str) else None


class _ChatToolCallDict(ChatCompletionToolCallChunk, total=False):
    provider_specific_fields: Mapping[str, object]


def tool_call_dict_from_output_item(
    item: Mapping[str, Any], index: int, *, include_namespace: bool = False
) -> _ChatToolCallDict:
    """Convert a ``function_call`` or ``custom_tool_call`` output item dict to a chat
    completions tool_call dict. Custom (grammar/freeform) tool calls carry their raw
    string payload in ``input`` rather than ``arguments``; both map to
    ``function.arguments`` so chat clients (e.g. Cursor agent mode) receive them like
    any other tool call. The single conversion rule shared by the non-streaming
    accumulator and the streaming ``output_item.added`` branch."""
    from litellm.responses.litellm_completion_transformation.transformation import (
        LiteLLMCompletionResponsesConfig,
    )

    item_type: Final[object] = item.get("type")
    is_custom: Final = item_type == "custom_tool_call"
    arguments: Final = (item.get("input") if is_custom else item.get("arguments")) or ""
    name: Final = item.get("name") or ("custom_tool" if is_custom else "")
    function_chunk: Final = ChatCompletionToolCallFunctionChunk(name=name, arguments=arguments)
    tool_call_dict: Final = _ChatToolCallDict(
        id=LiteLLMCompletionResponsesConfig.tool_call_id_from_responses_item(item.get("id"), item.get("call_id")),
        type="function",
        function=function_chunk,
        index=index,
    )
    raw_provider_fields: Final[object] = item.get("provider_specific_fields")
    if isinstance(raw_provider_fields, dict):
        provider_specific_fields = raw_provider_fields
    elif raw_provider_fields and hasattr(raw_provider_fields, "__dict__"):
        provider_specific_fields = vars(raw_provider_fields)
    else:
        provider_specific_fields = None
    namespace: Final = item.get("namespace") if include_namespace else None
    merged_provider_fields: Final = {
        **(provider_specific_fields or {}),
        **({"namespace": namespace} if isinstance(namespace, str) else {}),
    }
    if merged_provider_fields:
        tool_call_dict["provider_specific_fields"] = merged_provider_fields
        function_chunk["provider_specific_fields"] = merged_provider_fields
    return tool_call_dict


def _tool_call_namespace(tool_call: object) -> str | None:
    top_level: Final = _json_field(_json_field(tool_call, "provider_specific_fields"), "namespace")
    function_fields: Final = _json_field(_json_field(tool_call, "function"), "provider_specific_fields")
    nested: Final = _json_field(function_fields, "namespace")
    return top_level if isinstance(top_level, str) else nested if isinstance(nested, str) else None


def _flat_responses_tool_choice(choice_type: str, name: str) -> ToolChoiceFunctionParam | ToolChoiceCustomParam:
    if choice_type == "custom":
        return ToolChoiceCustomParam(type="custom", name=name)
    return ToolChoiceFunctionParam(type="function", name=name)


def _reasoning_item_to_response_input(
    r_item: ChatCompletionReasoningItem,
) -> dict[str, object]:
    """Convert a stored ChatCompletionReasoningItem back to a Responses API input item.

    An item without an id is sent without one: the Responses API accepts that and
    verifies the encrypted content on its own, while it rejects any id it did not mint.
    """
    item_id: Final = r_item.get("id")
    encrypted_content: Final = r_item.get("encrypted_content")
    return {
        "type": "reasoning",
        **({"id": item_id} if item_id else {}),
        # summary is always required by the Responses API, even when empty
        "summary": r_item.get("summary") or [],
        **({"encrypted_content": encrypted_content} if encrypted_content else {}),
    }


def _requested_defer_loading(tool: Mapping[str, object], function_tool: Mapping[str, object]) -> object:
    tool_level: Final = tool.get("defer_loading")
    return tool_level if tool_level is not None else function_tool.get("defer_loading")


def _defer_loading_field(
    tool: Mapping[str, object],
    function_tool: Mapping[str, object],
    *,
    tool_search: Literal["hosted", "client", "absent"],
    drop_params: bool,
) -> Mapping[str, bool]:
    defer_loading: Final = _requested_defer_loading(tool, function_tool)
    if defer_loading is None or (defer_loading is False and tool_search == "absent"):
        return {}
    if tool_search == "client":
        verbose_logger.debug(
            "Chat provider: dropping defer_loading=%r beside a client-executed tool_search", defer_loading
        )
        return {}
    if isinstance(defer_loading, bool) and tool_search == "hosted":
        return {"defer_loading": defer_loading}
    if drop_params:
        verbose_logger.debug("Chat provider: dropping defer_loading=%r under drop_params", defer_loading)
        return {}
    problem: Final = (
        'defer_loading needs a {"type": "tool_search"} tool in the same request'
        if isinstance(defer_loading, bool)
        else f"defer_loading must be a boolean, got {defer_loading!r}"
    )
    raise litellm.UnsupportedParamsError(
        message=f"{problem}. To drop it, set `litellm.drop_params=True` or pass `drop_params=True`."
    )


def _without_dropped_defer_loading(
    tool: Mapping[str, object], *, tool_search: Literal["hosted", "client", "absent"], drop_params: bool
) -> Mapping[str, object]:
    if tool.get("type") != "function" or "defer_loading" not in tool:
        return tool
    kept: Final = _defer_loading_field(tool, {}, tool_search=tool_search, drop_params=drop_params)
    return {**{key: value for key, value in tool.items() if key != "defer_loading"}, **kept}


class LiteLLMResponsesTransformationHandler(CompletionTransformationBridge):
    """
    Handler for transforming /chat/completions api requests to litellm.responses requests
    """

    def __init__(self):
        pass

    def _normalize_tool_choice_for_responses_api(
        self, tool_choice: _ToolChoiceT
    ) -> _ToolChoiceT | ToolChoiceFunctionParam | ToolChoiceCustomParam | Literal["auto", "none", "required"]:
        """Chat tool_choice nests the name under function/custom; Responses API expects top-level name."""
        if not isinstance(tool_choice, dict):
            return tool_choice
        choice_type: Final = tool_choice.get("type")
        if isinstance(choice_type, str) and choice_type in ("auto", "none", "required"):
            return choice_type
        if choice_type not in ("function", "custom"):
            return tool_choice
        if isinstance(tool_choice.get("name"), str) and tool_choice.get("name"):
            # Return only Responses shape so stray chat ``function``/``custom`` keys are not sent upstream.
            return _flat_responses_tool_choice(choice_type, tool_choice["name"])
        nested: Final = tool_choice.get(choice_type)
        if isinstance(nested, dict):
            nested_name: Final = nested.get("name")
            if isinstance(nested_name, str) and nested_name:
                return _flat_responses_tool_choice(choice_type, nested_name)
        return tool_choice

    def _handle_raw_dict_response_item(self, item: dict[str, Any], index: int) -> tuple["Choices | None", int]:
        """
        Handle raw dict response items from Responses API (e.g., GPT-5 Codex format).

        Args:
            item: Raw dict response item with 'type' field
            index: Current choice index

        Returns:
            Tuple of (Choice object or None, updated index)
        """
        from litellm.types.utils import Choices, Message

        item_type: Final = item.get("type")

        # Ignore reasoning items for now
        if item_type == "reasoning":
            return None, index

        # Handle message items with output_text content
        if item_type == "message":
            content_list: Final = item.get("content", [])
            for content_item in content_list:
                if isinstance(content_item, dict):
                    content_type = content_item.get("type")
                    if content_type == "output_text":
                        response_text = content_item.get("text", "")
                        # Extract annotations from content if present
                        annotations = LiteLLMResponsesTransformationHandler._convert_annotations_to_chat_format(
                            content_item.get("annotations", None)
                        )
                        msg = Message(
                            role=item.get("role", "assistant"),
                            content=response_text if response_text else "",
                            annotations=annotations,
                        )
                        choice = Choices(message=msg, finish_reason="stop", index=index)
                        return choice, index + 1

        # function_call / custom_tool_call dicts are intercepted and accumulated by
        # _convert_response_output_to_choices before this callback is reached

        # Unknown or unsupported type
        return None, index

    def convert_chat_completion_messages_to_responses_api(
        self,
        messages: list["AllMessageValues"],
        *,
        drop_params: bool = False,
        keep_prompt_cache_breakpoints: bool = False,
        replay_tool_search: bool = False,
        include_namespace: bool = False,
    ) -> tuple[list[object], str | None]:
        converted_input_items, instructions = self._convert_chat_completion_messages_to_responses_input(
            messages,
            drop_params=drop_params,
            replay_tool_search=replay_tool_search,
            include_namespace=include_namespace,
        )
        return (
            converted_input_items
            if keep_prompt_cache_breakpoints
            else _strip_prompt_cache_breakpoints(converted_input_items),
            instructions,
        )

    def _convert_chat_completion_messages_to_responses_input(
        self,
        messages: list["AllMessageValues"],
        *,
        drop_params: bool = False,
        replay_tool_search: bool = False,
        include_namespace: bool = False,
    ) -> tuple[list[object], str | None]:
        input_items: Final[list[object]] = []
        instructions: str | None = None
        custom_tool_call_ids: Final = frozenset(
            tool_call["id"]
            for msg in messages
            if msg.get("role") == "assistant" and isinstance(msg.get("tool_calls"), list)
            for tool_call in msg.get("tool_calls") or ()
            if isinstance(tool_call, dict)
            and not tool_call.get("function")
            and isinstance(tool_call.get("custom"), dict)
        )

        leading_system_count: Final = next(
            (index for index, msg in enumerate(messages) if msg.get("role") != "system"),
            len(messages),
        )

        for index, msg in enumerate(messages):
            role = msg.get("role")
            content = msg.get("content", "")
            tool_calls = msg.get("tool_calls")
            tool_call_id = msg.get("tool_call_id")

            if role == "system":
                # Extract system message as instructions
                if isinstance(content, str) and index < leading_system_count:
                    if instructions:
                        # Concatenate multiple system prompts with a space
                        instructions = f"{instructions} {content}"
                    else:
                        instructions = content
                else:
                    input_items.append(
                        {
                            "type": "message",
                            "role": role,
                            "content": self._convert_content_to_responses_format(
                                content,
                                role,
                                drop_params=drop_params,
                            ),
                        }
                    )
            elif role == "tool":
                # Convert tool message to function call output format
                # The Responses API expects 'output' to be a list with input_text/input_image types
                # Using list format for consistency across text and multimodal content
                tool_output: list[dict[str, object]]
                if content is None:
                    tool_output = []
                elif isinstance(content, str):
                    # Convert string to list with input_text
                    tool_output = [{"type": "input_text", "text": content}]
                elif isinstance(content, list):
                    # Transform list content to Responses API format
                    tool_output = self._convert_content_to_responses_format(
                        content,
                        "user",  # Use "user" role to get input_* types
                        drop_params=drop_params,
                    )
                else:
                    # Fallback: convert unexpected types to input_text
                    tool_output = [{"type": "input_text", "text": str(content)}]
                if tool_call_id in custom_tool_call_ids:
                    input_items.append(
                        ResponseCustomToolCallOutputParam(
                            type="custom_tool_call_output",
                            call_id=tool_call_id,
                            output=content if isinstance(content, str) else tool_output,
                        )
                    )
                else:
                    input_items.append(
                        FunctionCallOutput(
                            type="function_call_output",
                            call_id=tool_call_id,
                            output=tool_output,
                        )
                    )
            elif role == "assistant" and tool_calls and isinstance(tool_calls, list):
                input_items.extend(_assistant_replay_items(msg, replay_tool_search))
                if content:
                    input_items.append(
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": self._convert_content_to_responses_format(
                                content, "assistant", drop_params=drop_params
                            ),
                        }
                    )
                for tool_call in tool_calls:
                    function = tool_call.get("function")
                    custom = tool_call.get("custom")
                    if function:
                        input_tool_call: dict[str, object] = {
                            "type": "function_call",
                            "call_id": tool_call["id"],
                        }
                        if "name" in function:
                            input_tool_call["name"] = function["name"]
                        if "arguments" in function:
                            input_tool_call["arguments"] = function["arguments"]
                        if include_namespace and (
                            namespace := _tool_call_namespace(cast(object, tool_call))  # cast-ok: untyped
                        ):
                            input_tool_call["namespace"] = namespace
                        input_items.append(input_tool_call)
                    elif isinstance(custom, dict):
                        input_items.append(
                            ResponseCustomToolCallParam(
                                type="custom_tool_call",
                                call_id=tool_call["id"],
                                name=custom.get("name", ""),
                                input=custom.get("input", ""),
                            )
                        )
                    else:
                        raise ValueError(f"tool call not supported: {tool_call}")
            elif content is not None:
                if role == "assistant":
                    input_items.extend(_assistant_replay_items(msg, replay_tool_search))
                input_items.append(
                    {
                        "type": "message",
                        "role": role,
                        "content": self._convert_content_to_responses_format(
                            content, cast(str, role), drop_params=drop_params
                        ),
                    }
                )
            elif role == "assistant":
                input_items.extend(_assistant_replay_items(msg, replay_tool_search))

        return input_items, instructions

    def _map_optional_params_to_responses_api_request(
        self,
        optional_params: dict,
        responses_api_request: "ResponsesAPIOptionalRequestParams",
        drop_params: bool = False,
    ) -> None:
        """Map optional_params into responses_api_request (mutates in place)."""
        for key, value in optional_params.items():
            if value is None:
                continue
            if key in ("max_tokens", "max_completion_tokens"):
                responses_api_request["max_output_tokens"] = value
            elif key == "tools" and value is not None:
                responses_api_request["tools"] = self._convert_tools_to_responses_format(
                    cast(list[dict[str, object]], value), drop_params=drop_params
                )
            elif key == "response_format":
                text_format = self._transform_response_format_to_text_format(value)
                if text_format:
                    responses_api_request["text"] = self._merge_text(responses_api_request, text_format)
            elif key == "verbosity":
                responses_api_request["text"] = self._merge_text(
                    responses_api_request,
                    MappingProxyType({"verbosity": value}),  # pyright: ignore[reportUnknownArgumentType]  # untyped value
                )
            elif key == "tool_choice":
                responses_api_request["tool_choice"] = self._normalize_tool_choice_for_responses_api(value)
            elif key == "stream_options":
                stream_options = normalize_responses_api_stream_options(value)
                if stream_options is not None:
                    responses_api_request["stream_options"] = stream_options
            elif key in ResponsesAPIOptionalRequestParams.__annotations__:
                responses_api_request[key] = value
            elif key == "previous_response_id":
                responses_api_request["previous_response_id"] = value
            elif key == "reasoning_effort":
                responses_api_request["reasoning"] = self.map_reasoning_effort(value)
            elif key == "web_search_options":
                self._add_web_search_tool(responses_api_request, value)

    @staticmethod
    def _merge_text(
        responses_api_request: "ResponsesAPIOptionalRequestParams", update: Mapping[str, object]
    ) -> "ResponseText":
        existing: Final = cast(  # cast-ok: text field is a ResponseText | dict[str, Any] | None union
            "dict[str, object]",
            dict(responses_api_request).get("text") or {},
        )
        return cast(  # cast-ok: merged mapping is a valid ResponseText shape
            "ResponseText",
            {**existing, **update},
        )

    def _build_sanitized_litellm_params(self, litellm_params: dict) -> dict[str, object]:
        """Build sanitized litellm_params with merged metadata."""
        responses_optional_param_keys: Final = set(ResponsesAPIOptionalRequestParams.__annotations__.keys())
        sanitized: Final[dict[str, object]] = {
            key: value for key, value in litellm_params.items() if key not in responses_optional_param_keys
        }
        legacy_metadata: Final = litellm_params.get("metadata")
        existing_litellm_metadata: Final = litellm_params.get("litellm_metadata")
        merged_litellm_metadata: Final[dict[str, object]] = {}
        if isinstance(legacy_metadata, dict):
            merged_litellm_metadata.update(legacy_metadata)
        if isinstance(existing_litellm_metadata, dict):
            merged_litellm_metadata.update(existing_litellm_metadata)
        if merged_litellm_metadata:
            sanitized["litellm_metadata"] = merged_litellm_metadata
        else:
            sanitized.pop("litellm_metadata", None)
        return sanitized

    def _merge_responses_api_request_into_request_data(
        self,
        request_data: dict[str, object],
        responses_api_request: "ResponsesAPIOptionalRequestParams",
        instructions: str | None,
    ) -> None:
        """Add non-None values from responses_api_request into request_data."""
        for key, value in responses_api_request.items():
            if value is None:
                continue
            if key == "instructions" and instructions:
                request_data["instructions"] = instructions
            elif key == "user" and isinstance(value, str):
                # OpenAI API requires user param to be max 64 chars - truncate if longer
                if len(value) <= 64:
                    request_data["user"] = value
                else:
                    request_data["user"] = value[:64]
            else:
                request_data[key] = value

    def transform_request(
        self,
        model: str,
        messages: list["AllMessageValues"],
        optional_params: dict,
        litellm_params: dict[str, object],
        headers: dict,
        litellm_logging_obj: "LiteLLMLoggingObj",
        client: object | None = None,
    ) -> dict:
        optional_params = self._extract_extra_body_params(optional_params)
        provider: Final = _served_provider(litellm_params)
        base_model: Final = litellm_params.get("base_model")
        supports_prompt_cache_breakpoint: Final = supports_openai_prompt_cache_breakpoint(model, provider) or (
            isinstance(base_model, str)
            and bool(base_model)
            and supports_openai_prompt_cache_breakpoint(base_model, provider)
        )
        converted_input_items, converted_instructions = self.convert_chat_completion_messages_to_responses_api(
            messages,
            drop_params=bool(litellm_params.get("drop_params") or litellm.drop_params),
            keep_prompt_cache_breakpoints=supports_prompt_cache_breakpoint,
            replay_tool_search=_requests_tool_search(cast(object, optional_params)),  # cast-ok: upcast
            include_namespace=True,
        )
        # OpenAI's Responses API rejects an empty input. For a system-only
        # request, carry the system message as a system-role input item instead
        # of instructions, mirroring how non-string system content is already
        # handled in convert_chat_completion_messages_to_responses_api.
        is_system_only_request: Final = not converted_input_items and converted_instructions is not None
        target_input_items: Final = (
            _without_audio_input_parts(converted_input_items)
            if _drops_audio_input(model, litellm_params)
            else converted_input_items
        )
        input_items: Final = (
            [
                {
                    "type": "message",
                    "role": "system",
                    "content": [{"type": "input_text", "text": converted_instructions}],
                }
            ]
            if is_system_only_request
            else target_input_items
        )
        instructions: Final = None if is_system_only_request else converted_instructions

        # Build responses API request using the reverse transformation logic
        responses_api_request: Final = ResponsesAPIOptionalRequestParams()

        # Set instructions if we found a system message
        if instructions:
            responses_api_request["instructions"] = instructions

        self._map_optional_params_to_responses_api_request(
            optional_params,
            responses_api_request,
            drop_params=bool(litellm_params.get("drop_params") or litellm.drop_params),
        )

        stream: Final = optional_params.get("stream") or litellm_params.get("stream", False)
        verbose_logger.debug("Chat provider: Stream parameter: %s", stream)

        # Ensure stream is properly set in the request
        if stream:
            responses_api_request["stream"] = True

        # Handle session management if previous_response_id is provided
        previous_response_id: Final = optional_params.get("previous_response_id")
        if previous_response_id:
            # Use the existing session handler for responses API
            verbose_logger.debug("Chat provider: Warning ignoring previous response ID: %s", previous_response_id)

        # Convert back to responses API format for the actual request

        api_model: Final = model

        from litellm.types.utils import CallTypes

        setattr(litellm_logging_obj, "call_type", CallTypes.responses.value)

        sanitized_litellm_params: Final = self._build_sanitized_litellm_params(litellm_params)

        request_data: Final = {
            "model": api_model,
            "input": input_items,
            "litellm_logging_obj": litellm_logging_obj,
            **sanitized_litellm_params,
            "client": client,
        }

        verbose_logger.debug("Chat provider: Final request model=%s, input_items=%s", api_model, len(input_items))

        self._merge_responses_api_request_into_request_data(request_data, responses_api_request, instructions)

        if headers:
            request_data["extra_headers"] = headers

        return request_data

    @staticmethod
    def _convert_response_output_to_choices(
        output_items: Sequence[object],
        handle_raw_dict_callback: Callable[..., tuple["Choices | None", int]] | None = None,
        structured_output: bool = False,
    ) -> list["Choices"]:
        """
        Convert Responses API output items to chat completion choices.

        Args:
            output_items: List of items from ResponsesAPIResponse.output
            handle_raw_dict_callback: Optional callback for handling raw dict items
            structured_output: The chat request carried a json response_format, so only the
                first message item becomes content

        Returns:
            List of Choices objects
        """
        from openai.types.responses import ResponseOutputMessage

        try:
            from openai.types.responses.response_output_item import (
                ResponseApplyPatchToolCall,
            )
        except ImportError:
            ResponseApplyPatchToolCall = None

        from litellm.types.utils import Choices, Message

        choices: Final[list[Choices]] = []
        index = 0
        reasoning_content: str | None = None
        pending_reasoning_items: Final[list[_BuiltReasoningItem]] = []  # mutable-ok: accumulator

        # Collect all tool calls to put them in a single choice
        # (Chat Completions API expects all tool calls in one message)
        accumulated_tool_calls: Final[list[Mapping[str, object]]] = []
        tool_call_index = 0

        for item in output_items:
            is_reasoning, reasoning_content = _append_reasoning_if_present(
                item, pending_reasoning_items, reasoning_content
            )
            if is_reasoning:
                continue

            tool_call_dict = _tool_call_dict_from_output_item_candidate(
                item, tool_call_index, ResponseApplyPatchToolCall
            )
            if tool_call_dict is not None:
                accumulated_tool_calls.append(tool_call_dict)
                tool_call_index += 1
                continue

            if isinstance(item, ResponseOutputMessage):
                for content in item.content:
                    response_text = getattr(content, "text", "")
                    # Extract annotations from content if present
                    raw_annotations = getattr(content, "annotations", None)
                    annotations = LiteLLMResponsesTransformationHandler._convert_annotations_to_chat_format(
                        raw_annotations
                    )
                    msg = Message(
                        role=item.role,
                        content=response_text if response_text else "",
                        reasoning_content=reasoning_content,
                        annotations=annotations,
                        reasoning_items=_as_chat_reasoning_items(pending_reasoning_items),
                    )

                    choices.append(
                        Choices(
                            message=msg,
                            finish_reason="stop",
                            index=index,
                        )
                    )

                    reasoning_content = None  # flush
                    pending_reasoning_items.clear()  # flush
                    index += 1

            elif isinstance(item, (dict, BaseModel)) and handle_raw_dict_callback is not None:
                raw_item: Final = item if isinstance(item, dict) else item.model_dump()
                choice, index = handle_raw_dict_callback(item=raw_item, index=index)
                if choice is not None:
                    choices.append(choice)

        tool_search_items: Final = _tool_search_items_from_output_items(output_items)
        if not choices and not accumulated_tool_calls:
            if not tool_search_items:
                return choices
            return [
                LiteLLMResponsesTransformationHandler._build_empty_choice(
                    output_items, "stop", tool_search_items=tool_search_items
                )
            ]
        return [
            _fold_choices_into_one(
                choices=choices,
                tool_calls=accumulated_tool_calls,
                trailing_reasoning_content=reasoning_content,
                trailing_reasoning_items=pending_reasoning_items,
                tool_search_items=tool_search_items,
                structured_output=structured_output,
            )
        ]

    @staticmethod
    def _build_empty_choice(
        output_items: Sequence[object],
        finish_reason: Literal["stop", "length", "content_filter"],
        tool_search_items: Sequence[Mapping[str, object]] = (),
    ) -> "Choices":
        from litellm.types.utils import Choices, Message

        reasoning_items: Final = _reasoning_items_from_output_items(output_items)
        reasoning_content: Final = " ".join(
            summary_block["text"]
            for reasoning_item in reasoning_items
            for summary_block in reasoning_item["summary"]
            if summary_block.get("text")
        )
        message: Final = Message(
            content=None if tool_search_items else "",
            reasoning_content=reasoning_content if reasoning_content else None,
            reasoning_items=_as_chat_reasoning_items(reasoning_items),
            provider_specific_fields=_tool_search_provider_fields(tool_search_items),
        )
        return Choices(message=message, finish_reason=finish_reason, index=0)

    @classmethod
    def _extract_output_from_completed_event(cls, parsed_chunk: Mapping[str, object]) -> list[dict[str, object]] | None:
        response_payload: Final = parsed_chunk.get("response")
        if not isinstance(response_payload, dict):
            return None
        response_output: Final = response_payload.get("output")
        if not isinstance(response_output, list) or len(response_output) == 0:
            return None
        return cast(list[dict[str, object]], response_output)

    @classmethod
    def _recover_output_items_from_raw_sse(cls, raw_sse: str | None) -> list[dict[str, object]]:
        if not raw_sse or not isinstance(raw_sse, str):
            return []

        recovered_output_items: Final[dict[int, dict[str, object]]] = {}
        recovered_text_only_items: Final[dict[int, dict[str, object]]] = {}

        for chunk in raw_sse.splitlines():
            parsed_chunk = parse_sse_json_chunk(chunk)
            if parsed_chunk is None:
                continue

            event_type = parsed_chunk.get("type")

            if event_type == ResponsesAPIStreamEvents.RESPONSE_COMPLETED:
                recovered_output = cls._extract_output_from_completed_event(parsed_chunk)
                if recovered_output is not None:
                    return recovered_output
                continue

            if event_type == ResponsesAPIStreamEvents.OUTPUT_ITEM_DONE:
                record_output_item_chunk(
                    parsed_chunk=parsed_chunk,
                    output_items=recovered_output_items,
                )
                continue

            if event_type == ResponsesAPIStreamEvents.OUTPUT_TEXT_DONE:
                record_output_text_chunk(
                    parsed_chunk=parsed_chunk,
                    output_items=recovered_output_items,
                    text_only_items=recovered_text_only_items,
                )
                continue

        # Merge text-only items into the recovered output items. Real
        # OUTPUT_ITEM_DONE events take precedence at any given output_index,
        # but text-only items at indices without a matching OUTPUT_ITEM_DONE
        # must still be preserved (e.g. multi-output responses where some
        # indices only emitted OUTPUT_TEXT_DONE).
        merged_items: Final[dict[int, dict[str, object]]] = {**recovered_text_only_items}
        merged_items.update(recovered_output_items)

        if merged_items:
            return [item for _, item in sorted(merged_items.items())]

        return []

    @classmethod
    def _recover_output_items_from_logging(cls, logging_obj: "LiteLLMLoggingObj") -> list[dict[str, object]]:
        model_call_details: Final = getattr(logging_obj, "model_call_details", {}) or {}
        original_response: Final = model_call_details.get("original_response")
        return cls._recover_output_items_from_raw_sse(original_response)

    def transform_response(
        self,
        model: str,
        raw_response: "BaseModel",
        model_response: "ModelResponse",
        logging_obj: "LiteLLMLoggingObj",
        request_data: dict,
        messages: list["AllMessageValues"],
        optional_params: dict,
        litellm_params: dict,
        encoding: object,
        api_key: str | None = None,
        json_mode: bool | None = None,
    ) -> "ModelResponse":
        """Transform Responses API response to chat completion response"""
        from litellm.responses.utils import ResponseAPILoggingUtils
        from litellm.types.llms.openai import ResponsesAPIResponse

        if not isinstance(raw_response, ResponsesAPIResponse):
            raise ValueError(f"Unexpected response type: {type(raw_response)}")

        if raw_response.error is not None:
            raise ValueError(f"Error in response: {raw_response.error}")

        output_items = raw_response.output
        if len(output_items) == 0:
            recovered_output_items: Final[list[ResponseOutputItem | dict[str, object]]] = [
                *self._recover_output_items_from_logging(logging_obj)
            ]
            if recovered_output_items:
                output_items = recovered_output_items
                raw_response.output = recovered_output_items
                verbose_logger.warning(
                    "Recovered empty Responses API output from raw SSE for model=%s",
                    model,
                )

        # Convert response output to choices using the static helper
        choices: Final = self._convert_response_output_to_choices(
            output_items=output_items,
            handle_raw_dict_callback=self._handle_raw_dict_response_item,
            structured_output=is_structured_output_request(
                cast("Mapping[str, object]", optional_params)  # cast-ok: optional_params is an untyped dict
            ),
        )

        response_is_incomplete: Final = raw_response.status == "incomplete" or (
            raw_response.incomplete_details is not None and raw_response.incomplete_details.reason is not None
        )

        if len(choices) == 0 and not response_is_incomplete:
            raise ValueError(f"Unknown items in responses API response: {output_items}")

        if response_is_incomplete:
            incomplete_finish_reason: Final = _map_incomplete_reason_to_finish_reason(
                raw_response.incomplete_details.reason if raw_response.incomplete_details is not None else None
            )
            if len(choices) == 0:
                choices.append(self._build_empty_choice(output_items, incomplete_finish_reason))
            else:
                for choice in choices:
                    choice.finish_reason = incomplete_finish_reason

        setattr(model_response, "choices", choices)

        model_response.model = model

        setattr(
            model_response,
            "usage",
            ResponseAPILoggingUtils.transform_response_api_usage_to_chat_usage(raw_response.usage),
        )

        model_response.id = _upstream_response_id(raw_response.id) or raw_response.id
        for key, value in _provider_metadata(raw_response.model_extra).items():
            setattr(model_response, key, value)

        # Preserve hidden params from the ResponsesAPIResponse, especially the headers
        # which contain important provider information like x-request-id
        raw_response_hidden_params: Final = get_hidden_params(raw_response) or {}
        if raw_response_hidden_params:
            model_response_hidden_params: Final = get_or_create_hidden_params(model_response)
            # Merge the raw_response hidden params with model_response hidden params
            # Preserve existing keys in model_response but add/override with raw_response params
            for key, value in raw_response_hidden_params.items():
                if key == "additional_headers" and key in model_response_hidden_params:
                    existing_additional_headers = model_response_hidden_params.get("additional_headers", {})
                    merged_headers = {
                        **cast(  # cast-ok: preserve mapping operations on dynamic response metadata
                            "dict[str, object]", value
                        ),
                        **cast(  # cast-ok: preserve mapping operations on dynamic response metadata
                            "dict[str, object]", existing_additional_headers
                        ),
                    }
                    model_response_hidden_params[key] = merged_headers
                else:
                    model_response_hidden_params[key] = value

        return model_response

    def get_model_response_iterator(
        self,
        streaming_response: Union[Iterator[str], AsyncIterator[str], "ModelResponse", "BaseModel"],
        sync_stream: bool,
        json_mode: bool | None = False,
        structured_output: bool = False,
    ) -> BaseModelResponseIterator:
        return OpenAiResponsesToChatCompletionStreamIterator(
            streaming_response, sync_stream, json_mode, structured_output=structured_output
        )

    def _convert_content_str_to_input_text(self, content: str, role: str) -> dict[str, object]:
        if role in ("user", "system", "developer", "tool"):
            return {"type": "input_text", "text": content}
        else:
            return {"type": "output_text", "text": content}

    def _convert_content_to_responses_format_image(
        self, content: "ChatCompletionImageObject", role: str
    ) -> "ResponseInputImageParam":
        from openai.types.responses import ResponseInputImageParam

        content_image_url: Final = content.get("image_url")
        actual_image_url: str | None = None
        detail: Literal["low", "high", "auto"] | None = None

        if isinstance(content_image_url, str):
            actual_image_url = content_image_url
        elif isinstance(content_image_url, dict):
            actual_image_url = content_image_url.get("url")
            detail = cast(
                Literal["low", "high", "auto"] | None,
                content_image_url.get("detail"),
            )

        if actual_image_url is None:
            raise ValueError(f"Invalid image URL: {content_image_url}")

        image_param: Final = ResponseInputImageParam(image_url=actual_image_url, detail="auto", type="input_image")

        if detail:
            image_param["detail"] = detail

        return image_param

    def _convert_content_to_responses_format(
        self,
        content: str
        | list[object]
        | Iterable[
            Union[
                "OpenAIMessageContentListBlock",
                "ChatCompletionThinkingBlock",
                "ChatCompletionRedactedThinkingBlock",
                "ChatCompletionToolReferenceObject",
            ]
        ]
        | None,
        role: str,
        drop_params: bool = False,
    ) -> list[dict[str, object]]:
        """Convert chat completion content to responses API format"""
        from litellm.types.llms.openai import ChatCompletionImageObject

        verbose_logger.debug("Chat provider: Converting content to responses format - input type: %s", type(content))

        if content is None:
            return [self._convert_content_str_to_input_text("", role)]
        elif isinstance(content, str):
            result = [self._convert_content_str_to_input_text(content, role)]
            verbose_logger.debug("Chat provider: String content -> %s", result)
            return result
        elif isinstance(content, list):
            result = []
            for i, item in enumerate(content):
                verbose_logger.debug("Chat provider: Processing content item %s: %s = %s", i, type(item), item)
                if isinstance(item, str):
                    converted = self._convert_content_str_to_input_text(item, role)
                    result.append(converted)
                    verbose_logger.debug("Chat provider:   -> %s", converted)
                elif isinstance(item, dict):
                    # Handle multimodal content
                    original_type = item.get("type")
                    if original_type == "text":
                        converted = with_prompt_cache_breakpoint(
                            self._convert_content_str_to_input_text(item.get("text", ""), role),
                            _prompt_cache_breakpoint_for_wire(item.get("prompt_cache_breakpoint"), drop_params),
                        )
                        result.append(converted)
                        verbose_logger.debug("Chat provider:   text -> %s", converted)
                    elif original_type == "image_url":
                        # Map to responses API image format
                        converted = with_prompt_cache_breakpoint(
                            cast(
                                dict,
                                self._convert_content_to_responses_format_image(
                                    cast(ChatCompletionImageObject, item), role
                                ),
                            ),
                            _prompt_cache_breakpoint_for_wire(item.get("prompt_cache_breakpoint"), drop_params),
                        )
                        result.append(converted)
                        verbose_logger.debug("Chat provider:   image_url -> %s", converted)
                    elif original_type == "input_audio":
                        converted = with_prompt_cache_breakpoint(
                            {"type": "input_audio", "input_audio": item.get("input_audio")},
                            _prompt_cache_breakpoint_for_wire(item.get("prompt_cache_breakpoint"), drop_params),
                        )
                        result.append(converted)
                        verbose_logger.debug("Chat provider:   input_audio -> %s", converted)
                    else:
                        # Try to map other types to responses API format
                        item_type = original_type or "input_text"
                        if item_type == "image":
                            converted = {"type": "input_image", **item}
                            result.append(converted)
                            verbose_logger.debug("Chat provider:   image -> %s", converted)
                        elif item_type == "file":
                            converted = with_prompt_cache_breakpoint(
                                _input_file_from_file_value(
                                    cast("ChatCompletionFileObject", item).get("file"),  # cast-ok: type tag checked
                                ),
                                _prompt_cache_breakpoint_for_wire(item.get("prompt_cache_breakpoint"), drop_params),
                            )
                            result.append(converted)
                            verbose_logger.debug("Chat provider:   file -> %s", converted)
                        elif item_type == "tool_reference":
                            verbose_logger.debug(
                                "Chat provider:   tool_reference has no responses API equivalent; skipped"
                            )
                        elif item_type in [
                            "input_text",
                            "input_image",
                            "output_text",
                            "refusal",
                            "input_file",
                            "computer_screenshot",
                            "summary_text",
                        ]:
                            # Already in responses API format
                            result.append(item)
                            verbose_logger.debug("Chat provider:   passthrough -> %s", item)
                        else:
                            # Default to input_text for unknown types
                            converted = with_prompt_cache_breakpoint(
                                self._convert_content_str_to_input_text(str(item.get("text", item)), role),
                                _prompt_cache_breakpoint_for_wire(item.get("prompt_cache_breakpoint"), drop_params),
                            )
                            result.append(converted)
                            verbose_logger.debug("Chat provider:   unknown(%s) -> %s", original_type, converted)
            verbose_logger.debug("Chat provider: Final converted content: %s", result)
            return result
        else:
            result = [self._convert_content_str_to_input_text(str(content), role)]
            verbose_logger.debug("Chat provider: Other content type -> %s", result)
            return result

    def _convert_tools_to_responses_format(
        self, tools: list[dict[str, object]], drop_params: bool = False
    ) -> list["ALL_RESPONSES_API_TOOL_PARAMS"]:
        """Convert chat completion tools to responses API tools format"""
        tool_search: Final[Literal["hosted", "client", "absent"]] = (
            "client"
            if any(tool.get("type") == "tool_search" and tool.get("execution") == "client" for tool in tools)
            else "hosted"
            if any(tool.get("type") == "tool_search" or is_anthropic_tool_search_chat_tool(tool) for tool in tools)
            else "absent"
        )
        responses_tools: Final[list[ALL_RESPONSES_API_TOOL_PARAMS]] = []
        for tool in tools:
            # convert function tool from chat completion to responses API format
            if tool.get("type") == "function" and isinstance(function_payload := tool.get("function"), dict):
                function_tool = cast(ChatCompletionToolParamFunctionChunk, function_payload)  # cast-ok: client tool
                responses_tools.append(
                    FunctionToolParam(
                        name=function_tool["name"],
                        parameters=function_tool.get("parameters"),
                        strict=function_tool.get("strict"),
                        type="function",
                        description=function_tool.get("description"),
                        **_defer_loading_field(tool, function_tool, tool_search=tool_search, drop_params=drop_params),
                    )
                )
            elif tool.get("type") == "custom" and isinstance(custom_payload := tool.get("custom"), dict):
                from litellm.litellm_core_utils.prompt_templates.common_utils import (
                    convert_custom_tool_format_to_responses_shape,
                )

                flat_custom = CustomToolParam(type="custom", name=custom_payload.get("name", ""))
                if custom_payload.get("description") is not None:
                    flat_custom["description"] = custom_payload["description"]
                if isinstance(custom_payload.get("format"), dict):
                    flat_custom["format"] = convert_custom_tool_format_to_responses_shape(custom_payload["format"])
                responses_tools.append(flat_custom)
            else:
                responses_tools.append(
                    _without_dropped_defer_loading(tool, tool_search=tool_search, drop_params=drop_params)
                )

        return cast(list["ALL_RESPONSES_API_TOOL_PARAMS"], responses_tools)

    def _extract_extra_body_params(self, optional_params: dict):
        """
        Extract extra_body from optional_params and separate supported Responses API params
        from unsupported ones. Supported params are moved to top-level optional_params,
        unsupported params remain in extra_body.
        """
        # Extract extra_body and separate supported params from unsupported ones
        extra_body: Final = optional_params.pop("extra_body", None) or {}
        if not extra_body:
            return optional_params

        supported_responses_api_params: Final = set(ResponsesAPIOptionalRequestParams.__annotations__.keys())
        # Also include params we handle specially
        supported_responses_api_params.update(
            {
                "previous_response_id",
                "reasoning_effort",  # We map this to "reasoning"
            }
        )

        # Extract supported params from extra_body and merge into optional_params
        extra_body_copy: Final = extra_body.copy()
        for key, value in extra_body_copy.items():
            if key in supported_responses_api_params:
                # Prefer extra_body value if it exists (may have more complete info like summary in reasoning_effort)
                optional_params[key] = extra_body.pop(key)

        return optional_params

    def map_reasoning_effort(self, reasoning_effort: object) -> Reasoning:
        # If dict is passed, convert it directly to Reasoning object
        if isinstance(reasoning_effort, dict):
            return Reasoning(
                **cast(Reasoning, reasoning_effort)  # cast-ok: dict is forwarded verbatim to the provider
            )

        # Check if auto-summary is enabled via flag or environment variable
        # Priority: litellm.reasoning_auto_summary flag > LITELLM_REASONING_AUTO_SUMMARY env var
        auto_summary_enabled: Final = (
            litellm.reasoning_auto_summary or os.getenv("LITELLM_REASONING_AUTO_SUMMARY", "false").lower() == "true"
        )

        return (
            Reasoning(effort=reasoning_effort, summary="detailed")
            if auto_summary_enabled
            else Reasoning(effort=reasoning_effort)
        )

    _map_reasoning_effort = map_reasoning_effort

    def _add_web_search_tool(
        self,
        responses_api_request: ResponsesAPIOptionalRequestParams,
        web_search_options: object,
    ) -> None:
        """
        Add web search tool to responses API request.

        Args:
            responses_api_request: The responses API request dict to modify
            web_search_options: Web search configuration (dict or other value)
        """
        if "tools" not in responses_api_request or responses_api_request["tools"] is None:
            responses_api_request["tools"] = []

        # Get the tools list with proper type narrowing
        tools = responses_api_request["tools"]
        if tools is None:
            tools = []
            responses_api_request["tools"] = tools

        web_search_tool: Final[dict[str, object]] = {"type": "web_search"}
        if isinstance(web_search_options, dict):
            web_search_tool.update(web_search_options)

        # Cast to Any to match the expected union type for tools list items
        tools.append(cast(Any, web_search_tool))

    def transform_response_format_to_text_format(self, response_format: object) -> "ResponseText | None":
        return self._transform_response_format_to_text_format(response_format)

    def _transform_response_format_to_text_format(self, response_format: object) -> "ResponseText | None":
        """
        Transform Chat Completion response_format parameter to Responses API text.format parameter.

        Chat Completion response_format structure:
        {
            "type": "json_schema",
            "json_schema": {
                "name": "schema_name",
                "schema": {...},
                "strict": True
            }
        }

        Responses API text parameter structure:
        {
            "format": {
                "type": "json_schema",
                "name": "schema_name",
                "schema": {...},
                "strict": True
            }
        }
        """
        if not response_format:
            return None

        if isinstance(response_format, dict):
            format_type: Final = response_format.get("type")

            if format_type == "json_schema":
                json_schema: Final = response_format.get("json_schema", {})
                return {
                    "format": {
                        "type": "json_schema",
                        "name": json_schema.get("name", "response_schema"),
                        "schema": json_schema.get("schema", {}),
                        "strict": json_schema.get("strict", False),
                    }
                }
            elif format_type == "json_object":
                return {"format": {"type": "json_object"}}
            elif format_type == "text":
                return {"format": {"type": "text"}}

        return None

    @staticmethod
    def _convert_annotations_to_chat_format(
        annotations: list[Any] | None,
    ) -> list[ChatCompletionAnnotation] | None:
        """
        Convert annotations from Responses API to Chat Completions format.

        Annotations are already in compatible format between both APIs,
        so we just need to convert Pydantic models to dicts.
        """
        if not annotations:
            return None

        result: Final[list[ChatCompletionAnnotation]] = []
        for annotation in annotations:
            try:
                # Convert Pydantic models to dicts (handles both v1 and v2)
                if hasattr(annotation, "model_dump"):
                    annotation_dict = annotation.model_dump()
                elif hasattr(annotation, "dict"):
                    annotation_dict = annotation.dict()
                elif isinstance(annotation, dict):
                    annotation_dict = annotation
                else:
                    # Skip unsupported annotation types
                    verbose_logger.debug("Skipping unsupported annotation type: %s", type(annotation))
                    continue

                result.append(annotation_dict)
            except Exception as e:
                # Skip malformed annotations
                verbose_logger.debug("Skipping malformed annotation: %s, error: %s", annotation, e)
                continue

        return result if result else None

    def _map_responses_status_to_finish_reason(self, status: str | None) -> str:
        """Map responses API status to chat completion finish_reason"""
        if not status:
            return "stop"

        status_mapping: Final = {
            "completed": "stop",
            "incomplete": "length",
            "failed": "stop",
            "cancelled": "stop",
        }

        return status_mapping.get(status, "stop")


class OpenAiResponsesToChatCompletionStreamIterator(BaseModelResponseIterator):
    def __init__(
        self,
        streaming_response: Union[Iterator[str], AsyncIterator[str], "ModelResponse", "BaseModel"],
        sync_stream: bool,
        json_mode: bool | None = False,
        structured_output: bool = False,
    ):
        super().__init__(streaming_response, sync_stream, json_mode)
        self._chat_completion_id: str | None = None
        self._served_service_tier: str | None = None
        self._tool_call_index_map: dict[int, int] = {}  # mutable-ok: per-stream accumulator state
        self._structured_output: Final = structured_output
        self._first_message_item_done: bool = False

    def _handle_string_chunk(
        self, str_line: Union[str, "BaseModel"]
    ) -> Union["GenericStreamingChunk", "ModelResponseStream"]:
        from pydantic import BaseModel

        if isinstance(str_line, BaseModel):
            return self.chunk_parser(str_line.model_dump())

        if not str_line or str_line.startswith("event:"):
            # ignore.
            return GenericStreamingChunk(text="", tool_use=None, is_finished=False, finish_reason="", usage=None)
        index: Final = str_line.find("data:")
        if index != -1:
            str_line = str_line[index + 5 :]

        return self.chunk_parser(json.loads(str_line))

    @staticmethod
    def _sequential_tool_call_index(
        tool_call_index_map: dict[int, int] | None,  # mutable-ok: per-stream state, remapped in place
        output_index: int,
    ) -> int:
        """Chat-completions tool_call indices must be 0-based and sequential, but
        Responses API ``output_index`` counts every output item (reasoning,
        message, ...), so the first tool call of a reasoning model arrives at
        output_index >= 1 and strict SSE accumulators (e.g. Cursor agent mode)
        misplace it. When a per-stream map is provided, remap each distinct
        output_index to the next sequential slot; without a map (stateless
        callers), fall back to the raw output_index."""
        if tool_call_index_map is None:
            return output_index
        if output_index not in tool_call_index_map:
            tool_call_index_map[output_index] = len(tool_call_index_map)
        return tool_call_index_map[output_index]

    @staticmethod
    def translate_responses_chunk_to_openai_stream(
        parsed_chunk: dict | BaseModel,
        tool_call_index_map: dict[int, int] | None = None,  # mutable-ok: per-stream state, remapped in place
    ) -> "ModelResponseStream":
        """
        Translate a Responses API streaming chunk to OpenAI chat completion streaming format.

        Args:
            parsed_chunk: Dict containing the Responses API event chunk
            tool_call_index_map: Per-stream output_index -> sequential tool_call index map

        Returns:
            ModelResponseStream: OpenAI-formatted streaming chunk

        Raises:
            ValueError: If chunk is invalid or missing required fields
        """
        from litellm.types.llms.openai import ChatCompletionToolCallFunctionChunk
        from litellm.types.utils import (
            ChatCompletionToolCallChunk,
            Delta,
            ModelResponseStream,
            StreamingChoices,
        )

        if not parsed_chunk:
            raise ValueError("Chat provider: Empty parsed_chunk")

        if isinstance(parsed_chunk, BaseModel):
            parsed_chunk = parsed_chunk.model_dump()
        if not isinstance(parsed_chunk, dict):
            raise ValueError(f"Chat provider: Invalid chunk type {type(parsed_chunk)}")

        # Handle different event types from responses API
        event_type = parsed_chunk.get("type")
        if isinstance(event_type, ResponsesAPIStreamEvents):
            event_type = event_type.value

        if parsed_chunk.get("object") == "chat.completion.chunk" or (
            event_type is None and isinstance(parsed_chunk.get("choices"), list) and parsed_chunk.get("choices")
        ):
            return ModelResponseStream(**parsed_chunk)

        verbose_logger.debug("Chat provider: Processing event type: %s", event_type)

        if event_type == "response.created":
            # Initial response creation event
            verbose_logger.debug("Chat provider: response.created -> %s", parsed_chunk)
            created_response: Final = parsed_chunk.get("response")
            return ModelResponseStream(
                id=_upstream_response_id(created_response.get("id")) if created_response else None,
                choices=[
                    StreamingChoices(
                        index=0,
                        delta=Delta(content=""),
                        finish_reason=None,
                    )
                ],
            )
        elif event_type == "response.output_item.added":
            # New output item added
            output_item = parsed_chunk.get("item", {})
            if output_item.get("type") in ("function_call", "custom_tool_call"):
                converted: Final = tool_call_dict_from_output_item(
                    output_item, parsed_chunk.get("output_index", 0), include_namespace=True
                )
                provider_specific_fields: Final = converted.get("provider_specific_fields")

                function_chunk: Final = ChatCompletionToolCallFunctionChunk(
                    name=converted["function"]["name"] or None,
                    arguments=converted["function"]["arguments"] or parsed_chunk.get("arguments") or "",
                )
                if provider_specific_fields:
                    function_chunk["provider_specific_fields"] = provider_specific_fields

                tool_call_index = OpenAiResponsesToChatCompletionStreamIterator._sequential_tool_call_index(
                    tool_call_index_map, parsed_chunk.get("output_index", 0)
                )
                tool_call_chunk: Final = _ChatToolCallDict(
                    id=converted["id"],
                    index=tool_call_index,
                    type="function",
                    function=function_chunk,
                    **({"provider_specific_fields": provider_specific_fields} if provider_specific_fields else {}),
                )

                return ModelResponseStream(
                    choices=[
                        StreamingChoices(
                            index=0,
                            delta=Delta(tool_calls=[tool_call_chunk]),
                            finish_reason=None,
                        )
                    ]
                )
        elif event_type in (
            ResponsesAPIStreamEvents.FUNCTION_CALL_ARGUMENTS_DELTA,
            ResponsesAPIStreamEvents.CUSTOM_TOOL_CALL_INPUT_DELTA,
        ):
            content_part: str | None = parsed_chunk.get("delta", None)
            if content_part:
                tool_call_index = OpenAiResponsesToChatCompletionStreamIterator._sequential_tool_call_index(
                    tool_call_index_map, parsed_chunk.get("output_index", 0)
                )
                return ModelResponseStream(
                    choices=[
                        StreamingChoices(
                            index=0,
                            delta=Delta(
                                tool_calls=[
                                    ChatCompletionToolCallChunk(
                                        id=None,
                                        index=tool_call_index,
                                        type="function",
                                        function=ChatCompletionToolCallFunctionChunk(name=None, arguments=content_part),
                                    )
                                ]
                            ),
                            finish_reason=None,
                        )
                    ]
                )
            else:
                raise ValueError(f"Chat provider: Invalid function argument delta {parsed_chunk}")
        elif event_type == ResponsesAPIStreamEvents.OUTPUT_ITEM_DONE:
            # New output item added
            output_item = parsed_chunk.get("item", {})
            if output_item.get("type") in ("function_call", "custom_tool_call"):
                if tool_call_index_map is None:
                    # Stateless callers (the responses guardrail handler extracting
                    # tool calls from a buffered output_item.done) get the complete
                    # tool call; per-stream callers already received it via
                    # output_item.added and the argument delta events
                    return ModelResponseStream(
                        choices=[
                            StreamingChoices(
                                index=0,
                                delta=Delta(
                                    tool_calls=(
                                        tool_call_dict_from_output_item(
                                            output_item, parsed_chunk.get("output_index", 0), include_namespace=True
                                        ),
                                    )
                                ),
                                finish_reason=None,
                            )
                        ]
                    )
                # Do NOT emit finish_reason here — response.completed handles the terminal
                # finish_reason. Emitting "tool_calls" here would prematurely terminate
                # the stream before subsequent tool calls arrive (same fix as #17246 for
                # the message-type branch). The item's fields were already streamed via
                # output_item.added and the argument delta events.
                return ModelResponseStream(
                    choices=[
                        StreamingChoices(
                            index=0,
                            delta=Delta(),
                            finish_reason=None,
                        )
                    ]
                )
            elif output_item.get("type") == "message":
                # Message completion should NOT emit finish_reason
                # This is the fix for issue #17246 - don't end stream prematurely
                return ModelResponseStream(
                    choices=[
                        StreamingChoices(
                            index=0,
                            delta=Delta(content=""),
                            finish_reason=None,
                        )
                    ]
                )

        elif event_type == "response.output_text.delta":
            # Content part added to output
            content_part = parsed_chunk.get("delta", None)
            if content_part is not None:
                return ModelResponseStream(
                    choices=[
                        StreamingChoices(
                            index=0,
                            delta=Delta(content=content_part),
                            finish_reason=None,
                        )
                    ]
                )
            else:
                raise ValueError(f"Chat provider: Invalid text delta {parsed_chunk}")
        elif event_type == "response.reasoning_summary_text.delta":
            content_part = parsed_chunk.get("delta", None)
            if content_part:
                return ModelResponseStream(
                    choices=[
                        StreamingChoices(
                            index=cast(int, parsed_chunk.get("summary_index")),
                            delta=Delta(reasoning_content=content_part),
                        )
                    ]
                )
        elif event_type in ("response.completed", "response.incomplete"):
            response_data: Final = parsed_chunk.get("response", {})
            output_items: Final = response_data.get("output", []) if response_data else []

            has_function_calls: Final = any(
                item.get("type") in ("function_call", "custom_tool_call")
                for item in output_items
                if isinstance(item, dict)
            )

            finish_reason: Final = (
                _map_incomplete_reason_to_finish_reason(_incomplete_reason_from_response_payload(response_data))
                if event_type == "response.incomplete"
                else ("tool_calls" if has_function_calls else "stop")
            )

            terminal_reasoning_items_typed: Final = _as_chat_reasoning_items(
                _reasoning_items_from_output_items(output_items)
            )

            usage = None
            if response_data.get("usage"):
                from litellm.responses.utils import ResponseAPILoggingUtils

                usage = ResponseAPILoggingUtils.transform_response_api_usage_to_chat_usage(response_data.get("usage"))
            provider_metadata: Final = _provider_metadata(response_data)
            served_service_tier: Final = response_data.get("service_tier")
            return ModelResponseStream(
                choices=[
                    StreamingChoices(
                        index=0,
                        delta=Delta(
                            content="",
                            reasoning_items=terminal_reasoning_items_typed,
                            provider_specific_fields=_tool_search_provider_fields(
                                _tool_search_items_from_output_items(cast(object, output_items))  # cast-ok: untyped
                            ),
                        ),
                        finish_reason=finish_reason,
                    )
                ],
                usage=usage,
                provider_specific_fields=dict(provider_metadata) or None,
                **(
                    MappingProxyType({"service_tier": served_service_tier})
                    if isinstance(served_service_tier, str)
                    else MappingProxyType({})
                ),
            )
        else:
            pass
        # For any unhandled event types, create a minimal valid chunk or skip
        verbose_logger.debug("Chat provider: Unhandled event type '%s', creating empty chunk", event_type)

        # Return a minimal valid chunk for unknown events
        return ModelResponseStream(
            choices=[
                StreamingChoices(
                    index=0,
                    delta=Delta(content=""),
                    finish_reason=None,
                )
            ]
        )

    def chunk_parser(self, chunk: dict[str, object]) -> "ModelResponseStream":
        """
        Parse a Responses API streaming chunk and convert to OpenAI format.

        Args:
            chunk: Dict containing the Responses API event chunk

        Returns:
            ModelResponseStream: OpenAI-formatted streaming chunk
        """
        verbose_logger.debug("Chat provider: transform_streaming_response called with chunk: %s", chunk)
        self._remember_served_service_tier(chunk)
        translated: Final = (
            self._empty_content_chunk()
            if self._is_text_delta_after_structured_answer(chunk)
            else OpenAiResponsesToChatCompletionStreamIterator.translate_responses_chunk_to_openai_stream(
                chunk, tool_call_index_map=self._tool_call_index_map
            )
        )
        if self._structured_output and _is_message_item_done(chunk):
            self._first_message_item_done = True
        return self._with_served_service_tier(self._with_stream_scoped_id(translated))

    def _is_text_delta_after_structured_answer(self, chunk: Mapping[str, object]) -> bool:
        """Under a structured response_format the first message item is the whole answer, so
        text from any later message item is dropped, matching the non-stream fold."""
        return (
            self._structured_output
            and self._first_message_item_done
            and _responses_event_type(chunk) == ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA.value
        )

    @staticmethod
    def _empty_content_chunk() -> "ModelResponseStream":
        from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices

        return ModelResponseStream(choices=[StreamingChoices(index=0, delta=Delta(content=""), finish_reason=None)])

    def _remember_served_service_tier(self, chunk: dict[str, object]) -> None:
        response_payload: Final = chunk.get("response")
        if not isinstance(response_payload, dict):
            return
        served_tier: Final = response_payload.get("service_tier")
        if isinstance(served_tier, str) and served_tier:
            self._served_service_tier = served_tier

    def _with_served_service_tier(self, chunk: "ModelResponseStream") -> "ModelResponseStream":
        if self._served_service_tier is not None and chunk.model_dump().get("service_tier") is None:
            setattr(chunk, "service_tier", self._served_service_tier)  # noqa: B010  # pydantic extra, not a declared field
        return chunk

    def _with_stream_scoped_id(self, chunk: "ModelResponseStream") -> "ModelResponseStream":
        if self._chat_completion_id is None:
            self._chat_completion_id = chunk.id
        else:
            chunk.id = self._chat_completion_id
        return chunk
