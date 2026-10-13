import copy
import json
import struct
from binascii import crc32
from collections.abc import AsyncGenerator, AsyncIterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, Optional, Protocol, TypeAlias

from pydantic import TypeAdapter, ValidationError
from typing_extensions import ReadOnly, TypedDict

from litellm._logging import verbose_proxy_logger
from litellm.llms.base_llm.guardrail_translation.base_translation import (
    BaseTranslation,
    StreamingScanKey,
    StreamTransformSink,
)
from litellm.llms.base_llm.guardrail_translation.utils import (
    effective_skip_system_message_for_guardrail,
    effective_skip_tool_message_for_guardrail,
)
from litellm.types.utils import GenericGuardrailAPIInputs

if TYPE_CHECKING:
    from fastapi import HTTPException

    from litellm.integrations.custom_guardrail import CustomGuardrail, ModifyResponseException
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


@dataclass(slots=True)
class ConverseStreamFrame:
    """One frame of a Converse event stream as the proxy relays it.

    ``payload`` is the parsed JSON of an event frame; exception frames and bytes that do not decode keep it None and
    are relayed as received. ``guardrailed_payload`` is set when a guardrail rewrote the frame's text."""

    raw: bytes
    event_type: str | None = None
    payload: Mapping[str, object] | None = None
    guardrailed_payload: Mapping[str, object] | None = None

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


_JSON_OBJECT: Final = TypeAdapter(Mapping[str, object])


def _decode_converse_frame(raw: bytes) -> ConverseStreamFrame:
    from botocore.eventstream import EventStreamBuffer, ParserError

    buffer: Final = EventStreamBuffer()
    buffer.add_data(raw)
    try:
        message: Final = next(iter(buffer))
    except (ParserError, StopIteration) as e:
        verbose_proxy_logger.debug(
            "BedrockPassthroughGuardrailHandler: could not decode event-stream frame, relaying it unmodified: %s", e
        )
        return ConverseStreamFrame(raw=raw)
    # botocore-stubs types EventStreamMessage.headers as int; at runtime it is the decoded header dict.
    headers: Final = _JSON_OBJECT.validate_python(message.headers)
    event_type: Final = headers.get(":event-type")
    if headers.get(":message-type") != "event" or not isinstance(event_type, str):
        return ConverseStreamFrame(raw=raw)
    try:
        payload: Final = _JSON_OBJECT.validate_json(message.payload)
    except ValidationError:
        return ConverseStreamFrame(raw=raw, event_type=event_type)
    return ConverseStreamFrame(raw=raw, event_type=event_type, payload=payload)


async def _converse_stream_frames(chunks: AsyncIterable[bytes]) -> AsyncGenerator[ConverseStreamFrame, None]:
    """Cut the upstream bytes into frames as soon as each frame has fully arrived.

    Bytes that cannot be framed (a length below the 16-byte minimum, or a stream that ends inside a frame) are relayed
    as received."""
    pending = b""  # rebind-ok: bytes received but not yet framed
    framing = True  # rebind-ok: cleared once a frame length is unusable
    async for chunk in chunks:
        if not framing:
            yield ConverseStreamFrame(raw=chunk)
            continue
        pending += chunk
        while len(pending) >= 4:
            total_length = _unpack_uint32(pending[:4])
            if total_length < 16:
                framing = False
                break
            if len(pending) < total_length:
                break
            yield _decode_converse_frame(pending[:total_length])
            pending = pending[total_length:]
        if not framing and pending:
            yield ConverseStreamFrame(raw=pending)
            pending = b""
    if pending:
        yield ConverseStreamFrame(raw=pending)


def _converse_frames(items: Sequence[object]) -> tuple[ConverseStreamFrame, ...]:
    return tuple(item for item in items if isinstance(item, ConverseStreamFrame))


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
                "BedrockPassthroughGuardrailHandler: post_call_success_hook returned %s, "
                "leaving event stream unmodified",
                type(processed).__name__,
            )
            return body_bytes

        try:
            processed_blocks: Final = processed["output"]["message"]["content"]
            de_anonymized_texts: Final = [processed_blocks[i]["text"] for i in range(len(groups.groups))]
        except (KeyError, IndexError, TypeError):
            return body_bytes

        new_text_map: Final = _split_group_texts(groups, de_anonymized_texts)  # pyright: ignore[reportUnknownArgumentType]  # the hook returns an untyped response

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
        the caller only when the stream is held until moderated.
        """
        frames: Final = _converse_frames(responses_so_far)
        groups: Final = _group_stream_texts([frame.scanned_texts() for frame in frames])
        if not groups.groups:
            return responses_so_far
        content: Final = [{"text": text} for text in groups.texts()]
        await self.process_output_response(  # pyright: ignore[reportUnknownMemberType]  # takes the untyped request dict
            response={"output": {"message": {"role": "assistant", "content": content}}},
            guardrail_to_apply=guardrail_to_apply,
            litellm_logging_obj=litellm_logging_obj,
            user_api_key_dict=user_api_key_dict,
            request_data=request_data,
        )
        if self._check_streaming_has_ended(frames):
            _write_back_stream_texts(frames, groups, [block["text"] for block in content])
        return responses_so_far

    def get_streaming_scan_key(self, responses_so_far: Sequence[object]) -> StreamingScanKey | None:
        frames: Final = _converse_frames(responses_so_far)
        return StreamingScanKey(
            texts=_group_stream_texts([frame.received_texts() for frame in frames]).texts(),
            stream_ended=self._check_streaming_has_ended(frames),
        )

    def _check_streaming_has_ended(self, responses_so_far: Sequence[object]) -> bool:
        return any(frame.event_type == "messageStop" for frame in _converse_frames(responses_so_far))

    def build_block_sse_chunks(
        self,
        exc: "ModifyResponseException",
        stream_started: bool = False,
        responses_so_far: Sequence[object] | None = None,
    ) -> Sequence[bytes]:
        """
        End the message with the guardrail's message as a new text block and ``stopReason`` ``guardrail_intervened``,
        closing any content block left open and starting the message when nothing was relayed yet.
        """
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

    def build_stream_error_items(
        self,
        exc: "HTTPException",
        responses_so_far: Sequence[object] | None = None,
    ) -> Sequence[object] | None:
        from litellm.proxy.common_request_processing import sse_error_payload

        status_code, error = sse_error_payload(exc)
        return (_converse_exception(str(error.get("message")), status_code),)

    @staticmethod
    def owns_event_stream_item(item: object) -> bool:
        return isinstance(item, ConverseStreamFrame)

    @staticmethod
    def event_stream_items(chunks: AsyncIterable[bytes]) -> AsyncGenerator[ConverseStreamFrame, None]:
        return _converse_stream_frames(chunks)

    @staticmethod
    def encode_event_stream_item(item: object) -> bytes:
        if isinstance(item, ConverseStreamFrame):
            return item.encode()
        if isinstance(item, bytes):
            return item
        raise TypeError(f"cannot relay {type(item).__name__} on a Converse event stream")

    @staticmethod
    def event_stream_error_item(message: str, status_code: int) -> bytes:
        return _converse_exception(message, status_code)
