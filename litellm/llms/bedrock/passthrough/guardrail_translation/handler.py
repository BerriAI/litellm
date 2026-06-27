from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union

from litellm._logging import verbose_proxy_logger
from litellm.llms.base_llm.guardrail_translation.base_translation import BaseTranslation
from litellm.llms.base_llm.guardrail_translation.utils import (
    effective_skip_system_message_for_guardrail,
    effective_skip_tool_message_for_guardrail,
)
from litellm.types.utils import GenericGuardrailAPIInputs

if TYPE_CHECKING:
    from litellm.integrations.custom_guardrail import CustomGuardrail
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.utils import ProxyLogging

_CONVERSE_ACTIONS = frozenset({"converse", "converse-stream"})
_EVENT_STREAM_CONTENT_TYPE = "vnd.amazon.eventstream"
_EVENT_STREAM_MEDIA_TYPE = "application/vnd.amazon.eventstream"


def _is_converse_endpoint(endpoint: str) -> bool:
    parts = endpoint.rstrip("/").split("/")
    return bool(parts) and parts[-1] in _CONVERSE_ACTIONS


def _generic_passthrough_handler() -> BaseTranslation:
    """
    Fallback for non-Converse Bedrock routes (e.g. invoke). The generic
    handler scans the full request/response payload so blocking guardrails
    still run, matching how other passthrough providers are guarded.
    """
    from litellm.llms.pass_through.guardrail_translation.handler import (
        PassThroughEndpointHandler,
    )

    return PassThroughEndpointHandler()


_StringHolder = Tuple[Any, Union[str, int]]


def _collect_strings(node: Any, holders: List[_StringHolder]) -> None:
    """
    Record a (container, key) holder for every non-empty string value nested
    under an arbitrary JSON node, so prompt content a caller hides in fields
    like ``toolUse.input`` or ``toolResult.content[].json`` is still scanned
    and can be written back in place. Iterative to avoid unbounded recursion
    on deeply nested payloads.
    """
    stack: List[Any] = [node]
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


def _collect_block_text(block: dict, holders: List[_StringHolder]) -> None:
    text = block.get("text")
    if isinstance(text, str) and text:
        holders.append((block, "text"))


def _extract_converse_texts(
    body: dict,
    skip_system: bool,
    skip_tool: bool,
) -> Tuple[List[str], List[_StringHolder]]:
    """
    Walk a Bedrock Converse request body and collect text content.

    Returns (texts, holders) where each holder is the (container, key) pair
    that owns the extracted string, so write-back mutates it in place. Besides
    top-level ``text`` blocks this scans the arbitrary-JSON fields a caller can
    hide prompt content in -- ``toolUse.input`` and
    ``toolResult.content[].json`` (alongside ``toolResult.content[].text``) --
    as well as the request-level fields still forwarded to Bedrock that a caller
    can route blocked content through: ``toolConfig.tools`` (tool names,
    descriptions and input schemas) and ``additionalModelRequestFields``. Tool
    message blocks are skipped when tool messages are excluded, but tool
    definitions are always scanned to match the chat-completions guardrail path.
    """
    holders: List[_StringHolder] = []

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

    tool_config = body.get("toolConfig")
    if isinstance(tool_config, dict):
        _collect_strings(tool_config.get("tools"), holders)

    _collect_strings(body.get("additionalModelRequestFields"), holders)

    texts = [container[key] for container, key in holders]
    return texts, holders


def _extract_converse_output_texts(
    content_blocks: List[Any],
) -> Tuple[List[str], List[_StringHolder]]:
    """
    Collect user-visible text from Bedrock Converse output content blocks.

    Covers ``text`` blocks plus the other content-bearing fields a model can
    emit -- ``toolUse.input``, ``reasoningContent.reasoningText.text`` and
    ``citationsContent.content[].text`` -- while leaving structural values such
    as reasoning signatures and citation sources untouched.
    """
    holders: List[_StringHolder] = []
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
    texts = [container[key] for container, key in holders]
    return texts, holders


def _write_back_texts(
    guardrailed_texts: List[str],
    holders: List[_StringHolder],
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


_DeltaHolder = Tuple[Any, Any, Union[str, int]]


def _collect_stream_delta_text_holders(delta: Any) -> List[_DeltaHolder]:
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
    holders: List[_DeltaHolder] = []
    if not isinstance(delta, dict):
        return holders
    if isinstance(delta.get("text"), str):
        holders.append(("text", delta, "text"))
    tool_use = delta.get("toolUse")
    if isinstance(tool_use, dict) and isinstance(tool_use.get("input"), str):
        holders.append(("tool", tool_use, "input"))
    reasoning = delta.get("reasoningContent")
    if isinstance(reasoning, dict) and isinstance(reasoning.get("text"), str):
        holders.append(("reasoning", reasoning, "text"))
    citations = delta.get("citationsContent")
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

        frames: list[dict] = []
        offset = 0

        while offset + 16 <= len(body_bytes):
            total_length = struct.unpack("!I", body_bytes[offset : offset + 4])[0]
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

            texts: List[Tuple[Any, str]] = []
            if event_type == "contentBlockDelta":
                try:
                    payload_dict = _json.loads(payload_bytes)
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

        trailing_bytes = body_bytes[offset:]

        group_order: List[Any] = []
        group_members: dict[Any, list[Tuple[int, int]]] = {}
        group_texts: dict[Any, list[str]] = {}
        for frame_idx, frame in enumerate(frames):
            for local_idx, (group_key, text) in enumerate(frame["texts"]):
                if group_key not in group_members:
                    group_members[group_key] = []
                    group_texts[group_key] = []
                    group_order.append(group_key)
                group_members[group_key].append((frame_idx, local_idx))
                group_texts[group_key].append(text)

        active_groups = [gk for gk in group_order if "".join(group_texts[gk])]
        if not active_groups:
            return body_bytes

        synthetic_response: dict = {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [{"text": "".join(group_texts[gk])} for gk in active_groups],
                }
            },
            "stopReason": "end_turn",
        }

        processed = await proxy_logging_obj.post_call_success_hook(
            data=data,
            user_api_key_dict=user_api_key_dict,
            response=synthetic_response,  # type: ignore[arg-type]
        )

        if not isinstance(processed, dict):
            verbose_proxy_logger.debug(
                "BedrockPassthroughGuardrailHandler: post_call_success_hook returned %s, "
                "leaving event stream unmodified",
                type(processed).__name__,
            )
            return body_bytes

        try:
            processed_blocks = processed["output"]["message"]["content"]  # type: ignore[index]
            de_anonymized_texts = [processed_blocks[i]["text"] for i in range(len(active_groups))]
        except (KeyError, IndexError, TypeError):
            return body_bytes

        new_text_map: dict[Tuple[int, int], str] = {}
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

        result_parts: list[bytes] = []

        for frame_idx, frame in enumerate(frames):
            if not frame["texts"]:
                result_parts.append(frame["raw"])
                continue

            frame_raw = frame["raw"]
            orig_total = struct.unpack("!I", frame_raw[0:4])[0]
            orig_hdrs_len = struct.unpack("!I", frame_raw[4:8])[0]
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
    ) -> Any:
        endpoint = data.get("endpoint", "")
        body = data.get("data")

        if not _is_converse_endpoint(endpoint):
            return await _generic_passthrough_handler().process_input_messages(
                data=data,
                guardrail_to_apply=guardrail_to_apply,
                litellm_logging_obj=litellm_logging_obj,
            )

        if not isinstance(body, dict) or not isinstance(body.get("messages"), list):
            return data

        skip_system = effective_skip_system_message_for_guardrail(guardrail_to_apply)
        skip_tool = effective_skip_tool_message_for_guardrail(guardrail_to_apply)

        texts, holders = _extract_converse_texts(body, skip_system, skip_tool)

        if not texts:
            return data

        inputs = GenericGuardrailAPIInputs(texts=texts)
        model = data.get("model")
        if model:
            inputs["model"] = model

        guardrailed_inputs = await guardrail_to_apply.apply_guardrail(
            inputs=inputs,
            request_data=data,
            input_type="request",
            logging_obj=litellm_logging_obj,
        )

        guardrailed_texts = guardrailed_inputs.get("texts", [])
        if guardrailed_texts:
            _write_back_texts(guardrailed_texts, holders)

        return data

    async def process_output_response(
        self,
        response: Any,
        guardrail_to_apply: "CustomGuardrail",
        litellm_logging_obj: Optional["LiteLLMLoggingObj"] = None,
        user_api_key_dict: Optional[Any] = None,
        request_data: Optional[dict] = None,
    ) -> Any:
        endpoint = (request_data or {}).get("endpoint", "")
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

        output_message = (
            response.get("output", {}).get("message", {}) if isinstance(response.get("output"), dict) else {}
        )
        content_blocks = output_message.get("content") if isinstance(output_message, dict) else None

        if not isinstance(content_blocks, list):
            return response

        texts, holders = _extract_converse_output_texts(content_blocks)

        if not texts:
            return response

        effective_request_data = request_data or {}
        if "litellm_metadata" not in effective_request_data and user_api_key_dict is not None:
            user_metadata = self.transform_user_api_key_dict_to_metadata(user_api_key_dict)
            if user_metadata:
                effective_request_data = {
                    **effective_request_data,
                    "litellm_metadata": user_metadata,
                }

        inputs = GenericGuardrailAPIInputs(texts=texts)
        model = effective_request_data.get("model") if effective_request_data else None
        if model:
            inputs["model"] = model

        guardrailed_inputs = await guardrail_to_apply.apply_guardrail(
            inputs=inputs,
            request_data=effective_request_data,
            input_type="response",
            logging_obj=litellm_logging_obj,
        )

        guardrailed_texts = guardrailed_inputs.get("texts", [])
        if guardrailed_texts:
            _write_back_texts(guardrailed_texts, holders)

        return response
