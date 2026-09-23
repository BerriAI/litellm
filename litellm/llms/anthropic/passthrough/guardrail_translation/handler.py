"""Anthropic /v1/messages passthrough guardrail translation (SSE event stream)."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Final

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


def _parse_sse_blocks(body_bytes: bytes) -> tuple[bytes, ...]:
    """Split an SSE body into event blocks (including trailing separators)."""
    if not body_bytes:
        return ()
    # Keep separators so we can rebuild the stream byte-for-byte aside from rewrites.
    parts: Final = body_bytes.split(b"\n\n")
    last: Final = len(parts) - 1
    return tuple(part + b"\n\n" if i < last else part for i, part in enumerate(parts) if i < last or part)


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


def _text_delta(block: bytes) -> str | None:
    """The text of a content_block_delta/text_delta event, or None for any other block."""
    event_type, payload = _event_payload(block)
    if event_type != "content_block_delta" or not payload:
        return None
    delta = payload.get("delta")
    if not isinstance(delta, dict) or delta.get("type") != "text_delta":
        return None
    text = delta.get("text")
    return text if isinstance(text, str) else None


def _with_text(block: bytes, new_text: str) -> bytes:
    """Rebuild a content_block_delta block carrying ``new_text``; other blocks pass through."""
    event_type, payload = _event_payload(block)
    if event_type != "content_block_delta" or not payload:
        return block
    delta = payload.get("delta")
    if not isinstance(delta, dict):
        return block
    # ``payload`` was just parsed from ``block`` and is not shared, so editing it is local.
    delta["text"] = new_text
    return f"event: content_block_delta\ndata: {json.dumps(payload, separators=(',', ':'))}\n\n".encode()


def _first_text(processed: Mapping[str, object]) -> str | None:
    """The text of the first content block in a guardrail-processed Messages response."""
    content: Final = processed.get("content")
    if not isinstance(content, list) or not content:
        return None
    first: Final[object] = content[0]
    if not isinstance(first, Mapping):
        return None
    text: Final = first.get("text")
    return text if isinstance(text, str) else None


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
        proxy_logging_obj: ProxyLogging,
        user_api_key_dict: UserAPIKeyAuth,
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
        blocks: Final = _parse_sse_blocks(body_bytes)
        deltas: Final = tuple(
            (idx, text) for idx, text in ((i, _text_delta(block)) for i, block in enumerate(blocks)) if text is not None
        )
        if not deltas:
            return body_bytes

        combined: Final = "".join(text for _, text in deltas)
        synthetic_response: Final[dict] = {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": combined}],
            "stop_reason": "end_turn",
        }

        processed: Final = await proxy_logging_obj.post_call_success_hook(
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

        de_anonymized: Final = _first_text(processed)
        if de_anonymized is None:
            return body_bytes

        # Put the full rewrite on the first text_delta; blank the rest so
        # split placeholders cannot survive across frames.
        delta_indices: Final = frozenset(idx for idx, _ in deltas)
        first_idx: Final = deltas[0][0]
        return b"".join(
            _with_text(block, de_anonymized if idx == first_idx else "") if idx in delta_indices else block
            for idx, block in enumerate(blocks)
        )

    async def process_input_messages(
        self,
        data: dict,
        guardrail_to_apply: CustomGuardrail,
        litellm_logging_obj: LiteLLMLoggingObj | None = None,
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
        guardrail_to_apply: CustomGuardrail,
        litellm_logging_obj: LiteLLMLoggingObj | None = None,
        user_api_key_dict: UserAPIKeyAuth | None = None,
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
