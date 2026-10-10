import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import chain, groupby
from typing import TYPE_CHECKING, Any, Final, Optional, Protocol, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError
from typing_extensions import ReadOnly, TypedDict

from litellm._logging import verbose_proxy_logger
from litellm.llms.base_llm.guardrail_translation.base_translation import BaseTranslation
from litellm.llms.base_llm.guardrail_translation.utils import (
    effective_skip_system_message_for_guardrail,
    effective_skip_tool_message_for_guardrail,
    scoped_structured_message_indices,
    unappliable_request_rewrite,
)
from litellm.types.llms.openai import (
    AllMessageValues,
    ChatCompletionAssistantMessage,
    ChatCompletionAssistantToolCall,
    ChatCompletionSystemMessage,
    ChatCompletionTextObject,
    ChatCompletionToolCallFunctionChunk,
    ChatCompletionToolMessage,
    ChatCompletionToolParam,
    ChatCompletionToolParamFunctionChunk,
    ChatCompletionUserMessage,
)
from litellm.types.utils import GenericGuardrailAPIInputs

if TYPE_CHECKING:
    from litellm.integrations.custom_guardrail import CustomGuardrail
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.llms.pass_through.guardrail_translation.handler import (
        PassThroughEndpointHandler,
    )
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.utils import ProxyLogging

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


_StringHolder = tuple[Any, str | int]


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


def _extract_converse_texts(
    body: dict,
    skip_system: bool,
    skip_tool: bool,
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
    that model. Scanning it stays the fail-closed default.
    """
    holders: Final[list[_StringHolder]] = []

    if not skip_system:
        for block in body.get("system") or []:
            if isinstance(block, dict):
                _collect_block_text(block, holders)

    for message in body.get("messages") or []:
        if not isinstance(message, dict):
            continue
        for block in message.get("content") or []:
            if not isinstance(block, dict):
                continue
            if skip_tool and ("toolUse" in block or "toolResult" in block):
                continue
            _collect_block_text(block, holders)
            tool_use = block.get("toolUse")
            if isinstance(tool_use, dict):
                _collect_strings(tool_use.get("input"), holders)
            tool_result = block.get("toolResult")
            if isinstance(tool_result, dict):
                for inner in tool_result.get("content") or []:
                    if isinstance(inner, dict):
                        _collect_block_text(inner, holders)
                        _collect_strings(inner.get("json"), holders)

    _collect_strings(body.get("additionalModelRequestFields"), holders)

    texts: Final = [container[key] for container, key in holders]
    return texts, holders


class _ConverseModel(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, populate_by_name=True)


class _ConverseGuardText(_ConverseModel):
    text: str = ""


class _ConverseGuardContent(_ConverseModel):
    text: _ConverseGuardText | None = None


class _ConverseToolUse(_ConverseModel):
    tool_use_id: str = Field(default="", alias="toolUseId")
    name: str = ""
    tool_input: object = Field(default=None, alias="input")


class _ConverseToolResultContent(_ConverseModel):
    text: str = ""
    json_value: object = Field(default=None, alias="json")


class _ConverseToolResult(_ConverseModel):
    tool_use_id: str = Field(default="", alias="toolUseId")
    content: tuple[_ConverseToolResultContent, ...] = ()


class _ConverseContentBlock(_ConverseModel):
    text: str = ""
    guard_content: _ConverseGuardContent | None = Field(default=None, alias="guardContent")
    tool_use: _ConverseToolUse | None = Field(default=None, alias="toolUse")
    tool_result: _ConverseToolResult | None = Field(default=None, alias="toolResult")


class _ConverseMessage(_ConverseModel):
    role: str = ""
    content: tuple[_ConverseContentBlock, ...] = ()


class _ConverseToolInputSchema(_ConverseModel):
    json_schema: Mapping[str, object] | None = Field(default=None, alias="json")


class _ConverseToolSpec(_ConverseModel):
    name: str = ""
    description: str = ""
    input_schema: _ConverseToolInputSchema | None = Field(default=None, alias="inputSchema")


class _ConverseTool(_ConverseModel):
    tool_spec: _ConverseToolSpec | None = Field(default=None, alias="toolSpec")


class _ConverseToolConfig(_ConverseModel):
    tools: tuple[_ConverseTool, ...] = ()


class _ConverseRequest(_ConverseModel):
    system: tuple[_ConverseContentBlock, ...] = ()
    messages: tuple[_ConverseMessage, ...] = ()
    tool_config: _ConverseToolConfig | None = Field(default=None, alias="toolConfig")


def _parse_converse_request(validate: Callable[[], _ConverseRequest]) -> _ConverseRequest | None:
    try:
        return validate()
    except ValidationError:
        verbose_proxy_logger.debug("BedrockPassthroughGuardrailHandler: Converse body not convertible to messages")
        return None


def _converse_block_text(block: _ConverseContentBlock) -> str:
    if block.text:
        return block.text
    guard_text: Final = block.guard_content.text if block.guard_content else None
    return guard_text.text if guard_text else ""


@dataclass(frozen=True, slots=True)
class _BlockRef:
    message_index: int | None
    block_index: int


@dataclass(frozen=True, slots=True)
class _ConvertedMessage:
    message: AllMessageValues
    text_blocks: tuple[_BlockRef, ...]


_IndexedBlock: TypeAlias = tuple[_BlockRef, _ConverseContentBlock]
_TEXT_PARTS: Final = TypeAdapter(tuple[ChatCompletionTextObject, ...])
_JSON_OBJECT: Final = TypeAdapter(dict[str, Any])
_TextEdit: TypeAlias = tuple[_BlockRef, str]


def _indexed_blocks(message_index: int | None, blocks: Sequence[_ConverseContentBlock]) -> tuple[_IndexedBlock, ...]:
    return tuple((_BlockRef(message_index, block_index), block) for block_index, block in enumerate(blocks))


def _text_blocks(blocks: Sequence[_IndexedBlock]) -> tuple[tuple[_BlockRef, str], ...]:
    return tuple((ref, text) for ref, block in blocks if (text := _converse_block_text(block)))


def _text_content(text_blocks: Sequence[tuple[_BlockRef, str]]) -> tuple[ChatCompletionTextObject, ...]:
    return tuple(ChatCompletionTextObject(type="text", text=text) for _, text in text_blocks)


def _refs(text_blocks: Sequence[tuple[_BlockRef, str]]) -> tuple[_BlockRef, ...]:
    return tuple(ref for ref, _ in text_blocks)


def _converse_tool_message(tool_result: _ConverseToolResult) -> ChatCompletionToolMessage:
    return ChatCompletionToolMessage(
        role="tool",
        tool_call_id=tool_result.tool_use_id,
        content="\n".join(
            part.text or json.dumps(part.json_value)
            for part in tool_result.content
            if part.text or part.json_value is not None
        ),
    )


def _converse_tool_call(tool_use: _ConverseToolUse) -> ChatCompletionAssistantToolCall:
    return ChatCompletionAssistantToolCall(
        id=tool_use.tool_use_id,
        type="function",
        function=ChatCompletionToolCallFunctionChunk(
            name=tool_use.name,
            arguments="{}" if tool_use.tool_input is None else json.dumps(tool_use.tool_input),
        ),
    )


def _converse_user_group(is_tool_result: bool, group: Sequence[_IndexedBlock]) -> tuple[_ConvertedMessage, ...]:
    if is_tool_result:
        return tuple(
            _ConvertedMessage(_converse_tool_message(block.tool_result), ()) for _, block in group if block.tool_result
        )
    text_blocks: Final = _text_blocks(group)
    if not text_blocks:
        return ()
    message: Final = ChatCompletionUserMessage(role="user", content=list(_text_content(text_blocks)))
    return (_ConvertedMessage(message, _refs(text_blocks)),)


def _converse_user_messages(blocks: Sequence[_IndexedBlock]) -> tuple[_ConvertedMessage, ...]:
    return tuple(
        chain.from_iterable(
            _converse_user_group(is_tool_result, tuple(group))
            for is_tool_result, group in groupby(blocks, key=lambda item: item[1].tool_result is not None)
        )
    )


def _converse_assistant_messages(blocks: Sequence[_IndexedBlock]) -> tuple[_ConvertedMessage, ...]:
    text_blocks: Final = _text_blocks(blocks)
    content: Final = list(_text_content(text_blocks))
    tool_calls: Final = list(_converse_tool_call(block.tool_use) for _, block in blocks if block.tool_use)
    if tool_calls:
        message: Final = ChatCompletionAssistantMessage(
            role="assistant", content=content or None, tool_calls=tool_calls
        )
        return (_ConvertedMessage(message, _refs(text_blocks)),)
    if not content:
        return ()
    return (_ConvertedMessage(ChatCompletionAssistantMessage(role="assistant", content=content), _refs(text_blocks)),)


def _converse_message(message_index: int, message: _ConverseMessage) -> tuple[_ConvertedMessage, ...]:
    blocks: Final = _indexed_blocks(message_index, message.content)
    if message.role == "assistant":
        return _converse_assistant_messages(blocks)
    return _converse_user_messages(blocks)


def _converse_structured_messages(request: _ConverseRequest) -> tuple[_ConvertedMessage, ...]:
    converted: Final = tuple(
        chain.from_iterable(_converse_message(index, message) for index, message in enumerate(request.messages))
    )
    system_blocks: Final = _text_blocks(_indexed_blocks(None, request.system))
    if not system_blocks:
        return converted
    system: Final = ChatCompletionSystemMessage(role="system", content=list(_text_content(system_blocks)))
    return (_ConvertedMessage(system, _refs(system_blocks)), *converted)


def _without_content(message: Mapping[str, object]) -> tuple[tuple[str, object], ...]:
    return tuple(sorted(((key, value) for key, value in message.items() if key != "content"), key=lambda item: item[0]))


def _returned_text_parts(message: Mapping[str, object]) -> tuple[str, ...] | None:
    try:
        parts: Final = _TEXT_PARTS.validate_python(message.get("content"))
    except ValidationError:
        return None
    return tuple(part["text"] for part in parts)


def _message_text_edits(converted: _ConvertedMessage, returned: Mapping[str, object]) -> tuple[_TextEdit, ...] | None:
    if returned == converted.message:
        return ()
    if not converted.text_blocks or _without_content(returned) != _without_content(converted.message):
        return None
    texts: Final = _returned_text_parts(returned)
    if texts is None or len(texts) != len(converted.text_blocks):
        return None
    return tuple(zip(converted.text_blocks, texts))


def _structured_text_edits(
    converted: Sequence[_ConvertedMessage],
    scoped_indices: Sequence[int],
    returned: Sequence[Mapping[str, object]],
) -> tuple[_TextEdit, ...] | None:
    if len(returned) != len(scoped_indices):
        return None
    per_message: Final = tuple(
        _message_text_edits(converted[index], message) for index, message in zip(scoped_indices, returned)
    )
    if any(edits is None for edits in per_message):
        return None
    return tuple(chain.from_iterable(edits for edits in per_message if edits is not None))


def _write_back_structured_texts(body: Mapping[str, Any], edits: Sequence[_TextEdit]) -> None:
    for ref, text in edits:
        blocks = body["system"] if ref.message_index is None else body["messages"][ref.message_index]["content"]
        block = blocks[ref.block_index]
        if block.get("text"):
            block["text"] = text
        else:
            block["guardContent"]["text"]["text"] = text


def _converse_tools(request: _ConverseRequest) -> tuple[ChatCompletionToolParam, ...]:
    tools: Final = request.tool_config.tools if request.tool_config else ()
    return tuple(
        ChatCompletionToolParam(
            type="function",
            function=ChatCompletionToolParamFunctionChunk(
                name=spec.name,
                description=spec.description,
                parameters=dict(spec.input_schema.json_schema),
            ),
        )
        for tool in tools
        if (spec := tool.tool_spec) and spec.name and spec.input_schema and spec.input_schema.json_schema is not None
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
    guardrailed_texts: list[str],
    holders: list[_StringHolder],
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


class _StreamFrame(TypedDict):
    """One raw event-stream frame plus the guardrailable texts it carries."""

    raw: ReadOnly[bytes]
    texts: ReadOnly[Sequence[tuple[_GroupKey, str]]]


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

            texts: list[tuple[_GroupKey, str]] = []
            if event_type == "contentBlockDelta":
                try:
                    payload_dict: dict[str, object] = _json.loads(payload_bytes)
                    texts = [
                        (group_key, container[key])
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

        group_order: Final[list[_GroupKey]] = []
        group_members: Final[dict[_GroupKey, list[tuple[int, int]]]] = {}
        group_texts: Final[dict[_GroupKey, list[str]]] = {}
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

        skip_system: Final = effective_skip_system_message_for_guardrail(guardrail_to_apply)
        skip_tool: Final = effective_skip_tool_message_for_guardrail(guardrail_to_apply)

        texts, holders = _extract_converse_texts(body, skip_system, skip_tool)
        converse_request: Final = _parse_converse_request(lambda: _ConverseRequest.model_validate(body))
        converted: Final = _converse_structured_messages(converse_request) if converse_request else ()
        structured_messages: Final = tuple(item.message for item in converted)
        scoped_message_indices: Final = scoped_structured_message_indices(
            structured_messages,
            scan_only_tool_results=False,
            skip_system=skip_system,
            skip_tool=skip_tool,
        )

        if not texts and not scoped_message_indices:
            return data

        inputs: Final = GenericGuardrailAPIInputs(texts=texts)
        if scoped_message_indices:
            inputs["structured_messages"] = list(structured_messages[index] for index in scoped_message_indices)
        tools: Final = _converse_tools(converse_request) if converse_request else ()
        if tools:
            inputs["tools"] = list(tools)
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
        if guardrailed_texts:
            _write_back_texts(guardrailed_texts, holders)

        guardrailed_messages: Final = guardrailed_inputs.get("structured_messages")
        if guardrailed_messages is not None and guardrailed_messages is not inputs.get("structured_messages"):
            edits: Final = _structured_text_edits(converted, scoped_message_indices, guardrailed_messages)
            if edits is None:
                raise unappliable_request_rewrite(guardrail_to_apply.guardrail_name)
            _write_back_structured_texts(_JSON_OBJECT.validate_python(body), edits)

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
