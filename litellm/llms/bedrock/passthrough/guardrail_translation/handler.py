import functools
import itertools
from collections.abc import AsyncGenerator, AsyncIterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Final, Optional, Protocol, TypeAlias

from pydantic import TypeAdapter, ValidationError
from typing_extensions import ReadOnly, TypedDict

from litellm._logging import verbose_proxy_logger
from litellm.llms.base_llm.guardrail_translation.base_translation import BaseTranslation, StreamingScanKey
from litellm.llms.base_llm.guardrail_translation.utils import (
    effective_skip_system_message_for_guardrail,
    effective_skip_tool_message_for_guardrail,
)
from litellm.types.utils import GenericGuardrailAPIInputs

if TYPE_CHECKING:
    from fastapi import HTTPException

    from litellm.exceptions import ModifyResponseException
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
_STREAM_SCAN_ACTIONS: Final = frozenset({"converse-stream", "invoke-with-response-stream"})
_INVOKE_STREAM_ACTION: Final = "invoke-with-response-stream"
_FRAME_PRELUDE_LENGTH: Final = 12
_FRAME_MIN_LENGTH: Final = 16
_BLOCK_EXCEPTION_TYPE: Final = "validationException"
_INVOKE_STREAM_END_MARKER: Final = "amazon-bedrock-invocationMetrics"


def _is_converse_endpoint(endpoint: str) -> bool:
    parts: Final = endpoint.rstrip("/").split("/")
    return bool(parts) and parts[-1] in _CONVERSE_ACTIONS


_STR_MAPPING_ADAPTER: Final = TypeAdapter(dict[str, object])
_STR_MAPPING_LIST_ADAPTER: Final = TypeAdapter(list[dict[str, object]])


def _str_mapping(value: object) -> Mapping[str, object] | None:
    try:
        return _STR_MAPPING_ADAPTER.validate_python(value)
    except ValidationError:
        return None


def _str_mappings(value: object) -> tuple[Mapping[str, object], ...]:
    try:
        return tuple(_STR_MAPPING_LIST_ADAPTER.validate_python(value))
    except ValidationError:
        return ()


def _is_stream_scannable_endpoint(endpoint: str) -> bool:
    parts: Final = endpoint.rstrip("/").split("/")
    return bool(parts) and parts[-1] in _STREAM_SCAN_ACTIONS


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
    ``tools`` input instead, which this handler does not populate because a
    Bedrock ``toolSpec`` is not the OpenAI tool shape those consumers expect.

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


def _encode_string_header(name: str, value: str) -> bytes:
    import struct

    name_bytes: Final = name.encode()
    value_bytes: Final = value.encode()
    return (
        struct.pack("!B", len(name_bytes))
        + name_bytes
        + struct.pack("!B", 7)
        + struct.pack("!H", len(value_bytes))
        + value_bytes
    )


def _encode_frame(headers_bytes: bytes, payload: bytes) -> bytes:
    import struct
    from binascii import crc32 as esm_crc32

    prelude: Final = struct.pack(
        "!II", _FRAME_PRELUDE_LENGTH + len(headers_bytes) + len(payload) + 4, len(headers_bytes)
    )
    prelude_crc: Final = esm_crc32(prelude) & 0xFFFFFFFF
    body: Final = struct.pack("!I", prelude_crc) + headers_bytes + payload
    return prelude + body + struct.pack("!I", esm_crc32(body, prelude_crc) & 0xFFFFFFFF)


def _exception_frame(message: str) -> bytes:
    """A Bedrock event-stream exception message. The AWS SDKs raise it as an error that carries ``message``,
    which is the only way to end a stream with a readable error once the 200 and the first frames are out."""
    import json as _json

    headers_bytes: Final = (
        _encode_string_header(":message-type", "exception")
        + _encode_string_header(":exception-type", _BLOCK_EXCEPTION_TYPE)
        + _encode_string_header(":content-type", "application/json")
    )
    return _encode_frame(headers_bytes, _json.dumps({"message": message}).encode())


def _complete_frame_length(buffer: bytes) -> int | None:
    """Byte length of the frame at the start of ``buffer`` when the whole frame has arrived, else None."""
    if len(buffer) < _FRAME_MIN_LENGTH:
        return None
    total_length: Final = _unpack_uint32(buffer[:4])
    if total_length < _FRAME_MIN_LENGTH or total_length > len(buffer):
        return None
    return total_length


def _looks_like_frame(item: object) -> bool:
    if not isinstance(item, bytes) or len(item) < _FRAME_MIN_LENGTH:
        return False
    from binascii import crc32 as esm_crc32

    return _unpack_uint32(item[:4]) == len(item) and _unpack_uint32(item[8:12]) == (esm_crc32(item[:8]) & 0xFFFFFFFF)


class _DecodedFrame(TypedDict):
    event_type: ReadOnly[str | None]
    message_type: ReadOnly[str | None]
    payload: ReadOnly[object]


def _parse_json_payload(payload: bytes) -> object:
    import json as _json

    try:
        return _json.loads(payload) if payload else None
    except (ValueError, UnicodeDecodeError):
        return None


def _header_str(headers: Mapping[str, object], name: str) -> str | None:
    value: Final = headers.get(name)
    return value if isinstance(value, str) else None


def _decode_frame_bytes(frame: bytes) -> _DecodedFrame | None:
    from botocore.eventstream import EventStreamBuffer

    try:
        buf: Final = EventStreamBuffer()
        buf.add_data(frame)
        msg: Final = next(iter(buf))
    except Exception as e:  # noqa: BLE001  # an undecodable frame is relayed, never scanned
        verbose_proxy_logger.debug("BedrockPassthroughGuardrailHandler: could not decode event-stream frame: %s", e)
        return None
    headers: Final = _str_mapping(msg.headers) or {}
    return {
        "event_type": _header_str(headers, ":event-type"),
        "message_type": _header_str(headers, ":message-type"),
        "payload": _parse_json_payload(msg.payload),
    }


class EventStreamFrame(bytes):
    """One whole event-stream frame: the raw bytes the client receives, which also remembers its decoded form
    so repeated scans of a growing stream do not reparse the earlier frames. The memory lives and dies with
    the stream's ``responses_so_far``."""

    @functools.cached_property
    def decoded(self) -> _DecodedFrame | None:
        return _decode_frame_bytes(bytes(self))


def _decode_frame(frame: bytes) -> _DecodedFrame | None:
    return frame.decoded if isinstance(frame, EventStreamFrame) else _decode_frame_bytes(frame)


def _invoke_chunk_texts(chunk: Mapping[str, object]) -> tuple[str, ...]:
    """Text a decoded ``chunk.bytes`` payload carries, across the model families Bedrock Invoke streams
    (Anthropic deltas, Nova ``contentBlockDelta``, Llama ``generation``, Titan ``outputText``, Mistral
    ``outputs[].text``, Cohere ``text``, AI21 ``choices[].delta.content`` and ``completions[].data.text``)."""
    content_block_delta: Final = _str_mapping(chunk.get("contentBlockDelta"))
    delta: Final = _str_mapping(content_block_delta.get("delta") if content_block_delta else chunk.get("delta"))
    if delta is not None:
        return tuple(str(v) for k in ("text", "partial_json") if isinstance(v := delta.get(k), str) and v)
    direct: Final = tuple(
        str(value)
        for key in ("generation", "outputText", "text", "completion")
        if isinstance(value := chunk.get(key), str) and value
    )
    if direct:
        return direct
    outputs: Final = tuple(
        str(text) for o in _str_mappings(chunk.get("outputs")) if isinstance(text := o.get("text"), str)
    )
    if outputs:
        return outputs
    choices: Final = tuple(
        str(content)
        for choice in _str_mappings(chunk.get("choices"))
        if isinstance(content := (_str_mapping(choice.get("delta")) or {}).get("content"), str)
    )
    if choices:
        return choices
    return tuple(
        str(completion_text)
        for completion in _str_mappings(chunk.get("completions"))
        if isinstance(completion_text := (_str_mapping(completion.get("data")) or {}).get("text"), str)
    )


def _invoke_chunk_index(chunk: Mapping[str, object]) -> int | None:
    content_block_delta: Final = _str_mapping(chunk.get("contentBlockDelta"))
    index: Final = content_block_delta.get("contentBlockIndex") if content_block_delta else chunk.get("index")
    return index if isinstance(index, int) else None


def _decode_invoke_chunk(payload: object) -> Mapping[str, object] | None:
    import base64
    import json as _json

    encoded: Final = (_str_mapping(payload) or {}).get("bytes")
    if not isinstance(encoded, str):
        return None
    try:
        return _str_mapping(_json.loads(base64.b64decode(encoded)))
    except (ValueError, TypeError):
        return None


def _frame_texts(decoded: _DecodedFrame) -> tuple[tuple[_StreamGroupKey, str], ...]:
    payload: Final = _str_mapping(decoded["payload"])
    if decoded["event_type"] == "contentBlockDelta" and payload is not None:
        block_index: Final = payload.get("contentBlockIndex")
        return tuple(
            ((block_index if isinstance(block_index, int) else None, group_key), container[key])
            for group_key, container, key in _collect_stream_delta_text_holders(payload.get("delta"))
        )
    if decoded["event_type"] == "chunk":
        chunk: Final = _decode_invoke_chunk(decoded["payload"])
        if chunk is None:
            return ()
        return tuple(((_invoke_chunk_index(chunk), "invoke"), text) for text in _invoke_chunk_texts(chunk))
    return ()


def _frame_ends_stream(decoded: _DecodedFrame) -> bool:
    if decoded["event_type"] == "messageStop":
        return True
    if decoded["event_type"] != "chunk":
        return False
    chunk: Final = _decode_invoke_chunk(decoded["payload"])
    return chunk is not None and (
        chunk.get("type") == "message_stop" or "messageStop" in chunk or _INVOKE_STREAM_END_MARKER in chunk
    )


def _decoded_frames(responses_so_far: Sequence[object]) -> tuple[_DecodedFrame, ...]:
    return tuple(
        decoded for item in responses_so_far if isinstance(item, bytes) and (decoded := _decode_frame(item)) is not None
    )


def _streamed_text_pairs(frames: Sequence[_DecodedFrame]) -> tuple[tuple[_StreamGroupKey, str], ...]:
    return tuple(pair for decoded in frames for pair in _frame_texts(decoded))  # comprehension-ok: flattens frames


def _streamed_texts_by_group(frames: Sequence[_DecodedFrame]) -> tuple[str, ...]:
    """The text streamed so far, joined per content block (and per text / tool / reasoning group inside it) in
    first-seen order, so one block's text never bleeds into another's scan."""
    pairs: Final = _streamed_text_pairs(frames)
    first_seen: Final = {
        group_key: rank for rank, group_key in enumerate(dict.fromkeys(group_key for group_key, _ in pairs))
    }
    ordered: Final = sorted(pairs, key=lambda pair: first_seen[pair[0]])
    joined: Final = tuple(
        "".join(text for _, text in group) for _, group in itertools.groupby(ordered, key=lambda pair: pair[0])
    )
    return tuple(text for text in joined if text)


async def _split_event_stream_frames(chunks: AsyncIterable[bytes]) -> AsyncGenerator[bytes, None]:
    """Re-chunk a raw Bedrock event-stream body so each item is one whole frame, which is what the
    streaming guardrail loop scans and what a block can be inserted between."""
    buffer = b""  # rebind-ok: carries the partial frame left over from the previous chunk
    async for chunk in chunks:
        buffer += chunk
        while (frame_length := _complete_frame_length(buffer)) is not None:
            yield EventStreamFrame(buffer[:frame_length])
            buffer = buffer[frame_length:]
    if buffer:
        yield buffer


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
    def event_stream_endpoint_is_scannable(endpoint: str) -> bool:
        return _is_stream_scannable_endpoint(endpoint)

    @staticmethod
    def is_event_stream_item(item: object) -> bool:
        return _looks_like_frame(item)

    @staticmethod
    def iter_event_stream_frames(chunks: AsyncIterable[bytes]) -> AsyncGenerator[bytes, None]:
        return _split_event_stream_frames(chunks)

    def get_streaming_scan_key(self, responses_so_far: Sequence[object]) -> StreamingScanKey | None:
        frames: Final = _decoded_frames(responses_so_far)
        return StreamingScanKey(
            texts=_streamed_texts_by_group(frames),
            stream_ended=any(_frame_ends_stream(decoded) for decoded in frames),
        )

    def _check_streaming_has_ended(self, responses_so_far: Sequence[object]) -> bool:
        return any(_frame_ends_stream(decoded) for decoded in _decoded_frames(responses_so_far))

    async def process_output_streaming_response(
        self,
        responses_so_far: Sequence[object],
        guardrail_to_apply: "CustomGuardrail",
        litellm_logging_obj: Optional["LiteLLMLoggingObj"] = None,
        user_api_key_dict: Optional["UserAPIKeyAuth"] = None,
        request_data: Mapping[str, object] | None = None,
        stream_transform_sink: object | None = None,
        deliver_ended_stream_rewrites: bool = False,
    ) -> Sequence[object]:
        """Scan the text streamed so far, one entry per content block, through ``apply_guardrail``. A block
        propagates as the guardrail's exception and the streaming loop ends the stream with an exception frame.
        Text rewrites are not delivered here (frames already reached the client), masking guardrails keep the
        buffered ``de_anonymize_event_stream`` path."""
        from litellm.exceptions import ModifyResponseException

        texts: Final = _streamed_texts_by_group(_decoded_frames(responses_so_far))
        if not texts:
            return responses_so_far
        user_metadata: Final = (
            self.transform_user_api_key_dict_to_metadata(user_api_key_dict)
            if user_api_key_dict is not None and "litellm_metadata" not in (request_data or {})
            else None
        )
        effective_request_data: Final[Mapping[str, object]] = {
            **(request_data or {}),
            **({"litellm_metadata": user_metadata} if user_metadata else {}),
        }
        model: Final = effective_request_data.get("model")
        inputs: Final = (
            GenericGuardrailAPIInputs(texts=list(texts), model=str(model))
            if model
            else GenericGuardrailAPIInputs(texts=list(texts))
        )
        try:
            await guardrail_to_apply.apply_guardrail(  # pyright: ignore[reportUnknownMemberType]  # CustomGuardrail.apply_guardrail takes an untyped dict
                inputs=inputs,
                request_data=dict(effective_request_data),
                input_type="response",
                logging_obj=litellm_logging_obj,
            )
        except ModifyResponseException as e:
            if e.original_response is None:
                e.original_response = {
                    "output": {"message": {"role": "assistant", "content": [{"text": text} for text in texts]}},
                    "stopReason": "end_turn",
                }
            raise
        return responses_so_far

    def build_block_sse_chunks(
        self,
        exc: "ModifyResponseException",
        stream_started: bool = False,
        responses_so_far: Sequence[object] | None = None,
    ) -> Sequence[bytes]:
        return (_exception_frame(exc.message),)

    def build_stream_error_items(
        self,
        exc: "HTTPException",
        responses_so_far: Sequence[object] | None = None,
    ) -> Sequence[bytes]:
        from litellm.proxy.common_request_processing import (
            serialize_http_exception_detail,  # pyright: ignore[reportUnknownVariableType]  # returns an untyped dict tail
        )

        message: Final[str] = serialize_http_exception_detail(exc.detail)[0]
        return (_exception_frame(message),)

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

        skip_system: Final = effective_skip_system_message_for_guardrail(guardrail_to_apply)
        skip_tool: Final = effective_skip_tool_message_for_guardrail(guardrail_to_apply)

        texts, holders = _extract_converse_texts(body, skip_system, skip_tool)

        if not texts:
            return data

        inputs: Final = GenericGuardrailAPIInputs(texts=texts)
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
