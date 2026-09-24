"""Anthropic /v1/messages passthrough guardrail translation (SSE event stream)."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from types import MappingProxyType
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
# SSE allows CRLF, LF or CR line endings, so an event ends at a blank line in any of them.
_SSE_EVENT_END: Final = re.compile(rb"\r\n\r\n|\n\n|\r\r")
_SSE_TRAILING_END: Final = re.compile(rb"(?:\r\n\r\n|\n\n|\r\r)\Z")


def _is_messages_endpoint(endpoint: str) -> bool:
    normalized = endpoint.rstrip("/").split("?")[0]
    return any(normalized.endswith(suffix) for suffix in _MESSAGES_SUFFIXES)


def _parse_sse_blocks(body_bytes: bytes) -> tuple[bytes, ...]:
    """Split an SSE body into event blocks, each keeping its own trailing separator."""
    if not body_bytes:
        return ()
    ends: Final = tuple(match.end() for match in _SSE_EVENT_END.finditer(body_bytes))
    starts: Final = (0, *ends)
    stops: Final = (*ends, len(body_bytes))
    return tuple(body_bytes[start:stop] for start, stop in zip(starts, stops) if stop > start)


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


def _text_delta(block: bytes) -> tuple[int, str] | None:
    """The content block index and text of a text_delta event, or None for any other block."""
    event_type, payload = _event_payload(block)
    if event_type != "content_block_delta" or not payload:
        return None
    index = payload.get("index")
    delta = payload.get("delta")
    if not isinstance(index, int) or not isinstance(delta, dict) or delta.get("type") != "text_delta":
        return None
    text = delta.get("text")
    return (index, text) if isinstance(text, str) else None


def _with_text(block: bytes, new_text: str) -> bytes:
    """Rebuild a content_block_delta block carrying ``new_text``, keeping its framing."""
    event_type, payload = _event_payload(block)
    if event_type != "content_block_delta" or not payload:
        return block
    delta = payload.get("delta")
    if not isinstance(delta, dict):
        return block
    # ``payload`` was just parsed from ``block`` and is not shared, so editing it is local.
    delta["text"] = new_text
    trailing: Final = _SSE_TRAILING_END.search(block)
    separator: Final = trailing.group(0) if trailing else b""
    line_end: Final = separator[: len(separator) // 2].decode() or "\n"
    data: Final = json.dumps(payload, separators=(",", ":"))
    return f"event: content_block_delta{line_end}data: {data}".encode() + separator


def _processed_texts(processed: Mapping[str, object], count: int) -> tuple[str, ...] | None:
    """The text of the first ``count`` content blocks in a guardrail-processed response."""
    content: Final = processed.get("content")
    if not isinstance(content, list):
        return None
    first: Final[list[object]] = content[:count]
    texts: Final = tuple(
        text
        for text in (block.get("text") if isinstance(block, Mapping) else None for block in first)
        if isinstance(text, str)
    )
    return texts if len(texts) == count else None


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
        each content block's text first, then redistribute the de-anonymized
        text across that block's frames (full rewrite on its first text_delta,
        empty on the rest).
        """
        blocks: Final = _parse_sse_blocks(body_bytes)
        deltas: Final = tuple(
            (position, found)
            for position, found in ((position, _text_delta(block)) for position, block in enumerate(blocks))
            if found is not None
        )
        if not deltas:
            return body_bytes

        # One synthetic text block per Anthropic content block, so text from separate
        # blocks is never merged or moved across the tool/thinking blocks between them.
        indices: Final = tuple(sorted(frozenset(index for _, (index, _) in deltas)))
        synthetic_response: Final[dict] = {
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "text", "text": "".join(text for _, (i, text) in deltas if i == index)} for index in indices
            ],
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

        rewritten: Final = _processed_texts(processed, len(indices))
        if rewritten is None:
            return body_bytes

        # Put each block's full rewrite on its first text_delta and blank the rest, so
        # placeholders split across frames cannot survive.
        text_for_index: Final = MappingProxyType({index: text for index, text in zip(indices, rewritten)})
        index_at: Final = MappingProxyType({position: index for position, (index, _) in deltas})
        first_positions: Final = frozenset(
            min(position for position, (i, _) in deltas if i == index) for index in indices
        )
        return b"".join(
            (_with_text(block, text_for_index[index_at[position]] if position in first_positions else ""))
            if position in index_at
            else block
            for position, block in enumerate(blocks)
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
