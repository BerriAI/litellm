from typing import TYPE_CHECKING, Any, List, Optional, Tuple

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


def _is_converse_endpoint(endpoint: str) -> bool:
    parts = endpoint.rstrip("/").split("/")
    return bool(parts) and parts[-1] in _CONVERSE_ACTIONS


def _extract_converse_texts(
    body: dict,
    skip_system: bool,
    skip_tool: bool,
) -> Tuple[List[str], List[Tuple[str, int, int]]]:
    """
    Walk a Bedrock Converse request body and collect text content.

    Returns (texts, task_mappings) where each task_mapping is
    ("system", block_idx, -1) or ("message", msg_idx, content_idx).
    """
    texts: List[str] = []
    task_mappings: List[Tuple[str, int, int]] = []

    if not skip_system:
        for i, block in enumerate(body.get("system") or []):
            text = block.get("text") if isinstance(block, dict) else None
            if text:
                texts.append(text)
                task_mappings.append(("system", i, -1))

    for msg_idx, message in enumerate(body.get("messages") or []):
        if not isinstance(message, dict):
            continue
        for content_idx, block in enumerate(message.get("content") or []):
            if not isinstance(block, dict):
                continue
            if skip_tool and ("toolUse" in block or "toolResult" in block):
                continue
            text = block.get("text")
            if text:
                texts.append(text)
                task_mappings.append(("message", msg_idx, content_idx))

    return texts, task_mappings


def _write_back_texts(
    body: dict,
    guardrailed_texts: List[str],
    task_mappings: List[Tuple[str, int, int]],
) -> None:
    for idx, mapping in enumerate(task_mappings):
        if idx >= len(guardrailed_texts):
            break
        location, outer_idx, inner_idx = mapping
        if location == "system":
            system = body.get("system")
            if system and isinstance(system, list) and outer_idx < len(system):
                system[outer_idx]["text"] = guardrailed_texts[idx]
        else:
            messages = body.get("messages")
            if not (
                messages and isinstance(messages, list) and outer_idx < len(messages)
            ):
                continue
            content = messages[outer_idx].get("content")
            if content and isinstance(content, list) and inner_idx < len(content):
                content[inner_idx]["text"] = guardrailed_texts[idx]


class BedrockPassthroughGuardrailHandler(BaseTranslation):
    @staticmethod
    async def de_anonymize_converse_stream(  # noqa: PLR0915
        body_bytes: bytes,
        proxy_logging_obj: "ProxyLogging",
        user_api_key_dict: "UserAPIKeyAuth",
        data: dict,
    ) -> bytes:
        import json as _json
        import struct

        from botocore.eventstream import EventStreamBuffer
        from botocore.eventstream import crc32 as esm_crc32

        frames: list[dict] = []
        text_delta_indices: list[int] = []
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
            except Exception:
                frames.append({"raw": frame_raw, "is_text_delta": False, "text": None})
                continue

            text: Optional[str] = None
            is_text_delta = False
            if event_type == "contentBlockDelta":
                try:
                    payload_dict = _json.loads(payload_bytes)
                    delta = payload_dict.get("delta", {})
                    if isinstance(delta, dict) and "text" in delta:
                        text = delta["text"]
                        is_text_delta = True
                except Exception:
                    pass

            if is_text_delta:
                text_delta_indices.append(len(frames))
            frames.append(
                {"raw": frame_raw, "is_text_delta": is_text_delta, "text": text}
            )

        if not text_delta_indices:
            return body_bytes

        full_text = "".join(frames[i]["text"] for i in text_delta_indices)  # type: ignore[misc]
        synthetic_response: dict = {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [{"text": full_text}],
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
            return body_bytes

        try:
            de_anonymized_text: str = processed["output"]["message"]["content"][0]["text"]  # type: ignore[index]
        except (KeyError, IndexError, TypeError):
            return body_bytes

        orig_lengths = [len(frames[i]["text"] or "") for i in text_delta_indices]  # type: ignore[misc]
        total_orig = sum(orig_lengths) or 1
        de_anon_len = len(de_anonymized_text)
        chunk_texts: list[str] = []
        pos = 0
        for k, orig_len in enumerate(orig_lengths):
            if k == len(orig_lengths) - 1:
                chunk_texts.append(de_anonymized_text[pos:])
            else:
                end = pos + round(de_anon_len * orig_len / total_orig)
                chunk_texts.append(de_anonymized_text[pos:end])
                pos = end
        text_chunk_map: dict[int, str] = dict(zip(text_delta_indices, chunk_texts))

        result_parts: list[bytes] = []

        for frame_idx, frame in enumerate(frames):
            if not frame["is_text_delta"]:
                result_parts.append(frame["raw"])
                continue

            new_text = text_chunk_map[frame_idx]
            frame_raw = frame["raw"]
            orig_total = struct.unpack("!I", frame_raw[0:4])[0]
            orig_hdrs_len = struct.unpack("!I", frame_raw[4:8])[0]
            headers_bytes = frame_raw[12 : 12 + orig_hdrs_len]

            try:
                payload_dict = _json.loads(
                    frame_raw[12 + orig_hdrs_len : orig_total - 4]
                )
                payload_dict["delta"]["text"] = new_text
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

            result_parts.append(
                prelude + prelude_crc_b + headers_bytes + new_payload + msg_crc_b
            )

        return b"".join(result_parts)

    async def process_input_messages(
        self,
        data: dict,
        guardrail_to_apply: "CustomGuardrail",
        litellm_logging_obj: Optional["LiteLLMLoggingObj"] = None,
    ) -> Any:
        endpoint = data.get("endpoint", "")
        body = data.get("data")

        if not _is_converse_endpoint(endpoint) or not isinstance(body, dict):
            verbose_proxy_logger.debug(
                "BedrockPassthroughGuardrailHandler: skipping non-converse endpoint %s",
                endpoint,
            )
            return data

        if not isinstance(body.get("messages"), list):
            return data

        skip_system = effective_skip_system_message_for_guardrail(guardrail_to_apply)
        skip_tool = effective_skip_tool_message_for_guardrail(guardrail_to_apply)

        texts, task_mappings = _extract_converse_texts(body, skip_system, skip_tool)

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
            _write_back_texts(body, guardrailed_texts, task_mappings)

        return data

    async def process_output_response(
        self,
        response: Any,
        guardrail_to_apply: "CustomGuardrail",
        litellm_logging_obj: Optional["LiteLLMLoggingObj"] = None,
        user_api_key_dict: Optional[Any] = None,
        request_data: Optional[dict] = None,
    ) -> Any:
        if not isinstance(response, dict):
            return response

        output_message = (
            response.get("output", {}).get("message", {})
            if isinstance(response.get("output"), dict)
            else {}
        )
        content_blocks = (
            output_message.get("content") if isinstance(output_message, dict) else None
        )

        if not isinstance(content_blocks, list):
            return response

        texts: List[str] = []
        text_indices: List[int] = []
        for i, block in enumerate(content_blocks):
            if isinstance(block, dict) and "text" in block:
                texts.append(block["text"])
                text_indices.append(i)

        if not texts:
            return response

        effective_request_data = request_data or {}
        if (
            "litellm_metadata" not in effective_request_data
            and user_api_key_dict is not None
        ):
            user_metadata = self.transform_user_api_key_dict_to_metadata(
                user_api_key_dict
            )
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
        for list_pos, block_idx in enumerate(text_indices):
            if list_pos < len(guardrailed_texts):
                content_blocks[block_idx]["text"] = guardrailed_texts[list_pos]

        return response
