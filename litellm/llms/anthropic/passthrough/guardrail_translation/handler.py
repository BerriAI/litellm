"""Anthropic /v1/messages passthrough guardrail translation (SSE event stream)."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Final, Optional

from litellm._logging import verbose_proxy_logger
from litellm.llms.base_llm.guardrail_translation.base_translation import BaseTranslation

if TYPE_CHECKING:
    from litellm.integrations.custom_guardrail import CustomGuardrail
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.utils import ProxyLogging

_EVENT_STREAM_MEDIA_TYPE: Final = "text/event-stream"
_MESSAGES_SUFFIXES: Final = frozenset({"messages", "v1/messages"})


def _is_messages_endpoint(endpoint: str) -> bool:
    normalized = endpoint.rstrip("/").split("?")[0]
    return any(normalized.endswith(suffix) for suffix in _MESSAGES_SUFFIXES)


def _parse_sse_blocks(body_bytes: bytes) -> list[bytes]:
    """Split an SSE body into event blocks (including trailing separators)."""
    if not body_bytes:
        return []
    # Keep separators so we can rebuild the stream byte-for-byte aside from rewrites.
    parts = body_bytes.split(b"\n\n")
    blocks: list[bytes] = []
    for i, part in enumerate(parts):
        if i < len(parts) - 1:
            blocks.append(part + b"\n\n")
        elif part:
            blocks.append(part)
    return blocks


def _event_payload(block: bytes) -> tuple[str | None, dict[str, Any] | None]:
    try:
        text = block.decode("utf-8")
    except UnicodeDecodeError:
        return None, None
    event_type: str | None = None
    data_line: str | None = None
    for line in text.splitlines():
        if line.startswith("event:"):
            event_type = line[6:].strip()
        elif line.startswith("data:"):
            data_line = line[5:].strip()
    if not data_line:
        return event_type, None
    try:
        payload = json.loads(data_line)
    except json.JSONDecodeError:
        return event_type, None
    if not isinstance(payload, dict):
        return event_type, None
    return event_type, payload


class AnthropicPassthroughGuardrailHandler(BaseTranslation):
    @staticmethod
    def is_event_stream_content_type(content_type: str) -> bool:
        return "text/event-stream" in content_type

    @staticmethod
    def event_stream_media_type() -> str:
        return _EVENT_STREAM_MEDIA_TYPE

    @staticmethod
    def event_stream_endpoint_is_de_anonymizable(endpoint: str) -> bool:
        return _is_messages_endpoint(endpoint)

    @staticmethod
    async def de_anonymize_event_stream(
        body_bytes: bytes,
        proxy_logging_obj: "ProxyLogging",
        user_api_key_dict: "UserAPIKeyAuth",
        data: dict,
    ) -> bytes:
        """
        Buffer Anthropic SSE frames, run post-call guardrails on concatenated
        text_delta content, and rewrite text_delta payloads in place.

        Placeholders from output_parse_pii are routinely split across multiple
        text_delta events, so per-frame replacement cannot work; we concatenate
        first, then redistribute the de-anonymized text across the original
        frames (full rewrite on the first text_delta, empty on the rest).
        """
        blocks = _parse_sse_blocks(body_bytes)
        text_block_indices: list[int] = []
        texts: list[str] = []

        for idx, block in enumerate(blocks):
            event_type, payload = _event_payload(block)
            if event_type != "content_block_delta" or not payload:
                continue
            delta = payload.get("delta")
            if not isinstance(delta, dict) or delta.get("type") != "text_delta":
                continue
            text = delta.get("text")
            if not isinstance(text, str):
                continue
            text_block_indices.append(idx)
            texts.append(text)

        if not texts:
            return body_bytes

        combined = "".join(texts)
        synthetic_response: Final[dict] = {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": combined}],
            "stop_reason": "end_turn",
        }

        processed = await proxy_logging_obj.post_call_success_hook(
            data=data,
            user_api_key_dict=user_api_key_dict,
            response=synthetic_response,
        )
        if not isinstance(processed, dict):
            verbose_proxy_logger.debug(
                "AnthropicPassthroughGuardrailHandler: post_call_success_hook returned %s, "
                "leaving event stream unmodified",
                type(processed).__name__,
            )
            return body_bytes

        try:
            content = processed["content"]
            de_anonymized = content[0]["text"]
            if not isinstance(de_anonymized, str):
                return body_bytes
        except (KeyError, IndexError, TypeError):
            return body_bytes

        # Put the full rewrite on the first text_delta; blank the rest so
        # split placeholders cannot survive across frames.
        replacements = [de_anonymized] + [""] * (len(text_block_indices) - 1)
        out: list[bytes] = list(blocks)
        for block_idx, new_text in zip(text_block_indices, replacements):
            event_type, payload = _event_payload(out[block_idx])
            if event_type != "content_block_delta" or not payload:
                continue
            delta = payload.get("delta")
            if not isinstance(delta, dict):
                continue
            delta["text"] = new_text
            payload["delta"] = delta
            # Rebuild a minimal SSE block; preserve event name.
            new_block = (
                f"event: content_block_delta\ndata: {json.dumps(payload, separators=(',', ':'))}\n\n"
            ).encode("utf-8")
            out[block_idx] = new_block

        return b"".join(out)

    async def process_input_messages(
        self,
        data: dict,
        guardrail_to_apply: "CustomGuardrail",
        litellm_logging_obj: Optional["LiteLLMLoggingObj"] = None,
    ) -> Mapping[str, object]:
        from litellm.llms.pass_through.guardrail_translation.handler import (
            PassThroughEndpointHandler,
        )

        return await PassThroughEndpointHandler().process_input_messages(
            data=data,
            guardrail_to_apply=guardrail_to_apply,
            litellm_logging_obj=litellm_logging_obj,
        )

    async def process_output_response(
        self,
        response: object,
        guardrail_to_apply: "CustomGuardrail",
        litellm_logging_obj: Optional["LiteLLMLoggingObj"] = None,
        user_api_key_dict: Optional["UserAPIKeyAuth"] = None,
        request_data: dict | None = None,
    ) -> object:
        from litellm.llms.pass_through.guardrail_translation.handler import (
            PassThroughEndpointHandler,
        )

        return await PassThroughEndpointHandler().process_output_response(
            response=response,
            guardrail_to_apply=guardrail_to_apply,
            litellm_logging_obj=litellm_logging_obj,
            user_api_key_dict=user_api_key_dict,
            request_data=request_data,
        )
