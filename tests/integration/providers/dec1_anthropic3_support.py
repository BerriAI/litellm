"""Shared rig for the `test_dec1_anthropic3_*` files: round 3 of the direct Anthropic contracts on chat.

Round 3 reuses the round 1 peer (`dec1_anthropic_support.peer`), which records every request the proxy sends to
an owned loopback HTTP server playing api.anthropic.com, and the round 2 canned bodies. What it adds is the
reading side of a chat stream for the newer fields: Anthropic's chat stream iterator
(`ModelResponseIterator.chunk_parser`, `litellm/llms/anthropic/chat/handler.py:1006`) puts container, web fetch
results, citations, server tool calls, code execution results, compaction blocks and stop details on
`choices[0].delta.provider_specific_fields` of the chunk that carried them, so the helpers here fold those per
key, last value wins, the way `stream_chunk_builder` does.

No request may reach a real provider host: every deployment points `api_base` at the loopback peer. Every
prompt carries a fresh `marker()` because the shared proxy has a Redis response cache on.
"""

from __future__ import annotations

import base64
from collections.abc import Mapping, Sequence
from typing import Final

from integration._support.client import Gateway, eventually, list_value, object_value
from integration.providers.dec1_anthropic_support import (
    LANDING_SECONDS,
    MESSAGE_STOP,
    PRICES,
    block_delta,
    block_start,
    block_stop,
    deltas,
    message_delta,
    message_start,
    user,
)
from pydantic import JsonValue

ONE_HOUR_WRITE_RATE: Final = 0.006
ONE_HOUR_PRICES: Final[Mapping[str, JsonValue]] = {
    **PRICES,
    "cache_creation_input_token_cost_above_1hr": ONE_HOUR_WRITE_RATE,
}
CONTAINER: Final[Mapping[str, JsonValue]] = {"id": "container_dec1run", "expires_at": "2026-11-17T09:30:00Z"}
FETCH_URL: Final = "https://docs.dec1.invalid/page"
MCP_TOOL_USE: Final[Mapping[str, JsonValue]] = {
    "type": "mcp_tool_use",
    "id": "mcptoolu_dec1",
    "name": "search_docs",
    "server_name": "dec1-docs",
    "input": {"query": "widgets"},
}
MCP_TOOL_RESULT: Final[Mapping[str, JsonValue]] = {
    "type": "mcp_tool_result",
    "tool_use_id": "mcptoolu_dec1",
    "is_error": False,
    "content": [{"type": "text", "text": "widgets ship in blue"}],
}
WEB_FETCH_RESULT: Final[Mapping[str, JsonValue]] = {
    "type": "web_fetch_tool_result",
    "tool_use_id": "srvtoolu_dec1fetch",
    "content": {
        "type": "web_fetch_result",
        "url": FETCH_URL,
        "content": {
            "type": "document",
            "source": {"type": "text", "media_type": "text/plain", "data": "synthetic page body"},
            "title": "Synthetic page",
        },
        "retrieved_at": "2026-11-17T08:30:00Z",
    },
}
BASH_RESULT: Final[Mapping[str, JsonValue]] = {
    "type": "bash_code_execution_tool_result",
    "tool_use_id": "srvtoolu_dec1bash",
    "content": {
        "type": "bash_code_execution_result",
        "stdout": "4\n",
        "stderr": "",
        "return_code": 0,
        "content": [],
    },
}
CHAR_CITATION: Final[Mapping[str, JsonValue]] = {
    "type": "char_location",
    "cited_text": "The synthetic widget ships in blue.",
    "document_index": 0,
    "document_title": "Widget sheet",
    "start_char_index": 0,
    "end_char_index": 35,
}
REDACTED_DATA: Final = "EmwKAhgBEgy3dec1redactedpayload"


def stream_body(model: str, **extra: JsonValue) -> dict[str, JsonValue]:
    """A streaming chat request for `model` with one fresh user turn and usage on the final chunk."""
    return {
        "model": model,
        "messages": [user()],
        "stream": True,
        "stream_options": {"include_usage": True},
        **extra,
    }


def text_block(index: int, text: str) -> tuple[bytes, ...]:
    """Start, one text delta and stop of a text block at `index`."""
    return (
        block_start(index, {"type": "text", "text": ""}),
        block_delta(index, {"type": "text_delta", "text": text}),
        block_stop(index),
    )


def whole_block(index: int, block: Mapping[str, JsonValue]) -> tuple[bytes, ...]:
    """A block whose full content arrives in `content_block_start`, as tool results and compaction do."""
    return (block_start(index, block), block_stop(index))


def server_tool_use(index: int, tool_id: str, name: str, partial_json: str) -> tuple[bytes, ...]:
    """A `server_tool_use` block whose input arrives as one `input_json_delta`."""
    return (
        block_start(index, {"type": "server_tool_use", "id": tool_id, "name": name, "input": {}}),
        block_delta(index, {"type": "input_json_delta", "partial_json": partial_json}),
        block_stop(index),
    )


def framed(
    *blocks: tuple[bytes, ...],
    stop_reason: str = "end_turn",
    delta_usage: Mapping[str, JsonValue] | None = None,
    start_usage: Mapping[str, JsonValue] | None = None,
) -> tuple[bytes, ...]:
    """A whole Anthropic stream: `message_start`, the given blocks in order, `message_delta`, `message_stop`."""
    return (
        message_start(usage=start_usage),
        *(chunk for block in blocks for chunk in block),
        message_delta(stop_reason, usage=delta_usage),
        MESSAGE_STOP,
    )


def provider_field_values(events: Sequence[Mapping[str, JsonValue]], key: str) -> tuple[JsonValue, ...]:
    """Every value of `delta.provider_specific_fields[key]` across the stream, in arrival order."""
    return tuple(
        object_value(delta["provider_specific_fields"])[key]
        for delta in deltas(events)
        if isinstance(delta.get("provider_specific_fields"), dict)
        and key in object_value(delta["provider_specific_fields"])
    )


def last_provider_field(events: Sequence[Mapping[str, JsonValue]], key: str) -> JsonValue:
    """The last streamed value of `delta.provider_specific_fields[key]`, which is what an accumulated message keeps."""
    values: Final = provider_field_values(events, key)
    assert values, f"no chunk carried provider_specific_fields.{key}"
    return values[-1]


def streamed_thinking_blocks(events: Sequence[Mapping[str, JsonValue]]) -> tuple[dict[str, JsonValue], ...]:
    """Every `delta.thinking_blocks` entry across the stream, in arrival order."""
    return tuple(
        object_value(block) for delta in deltas(events) for block in list_value(delta.get("thinking_blocks") or [])
    )


def streamed_tool_calls(events: Sequence[Mapping[str, JsonValue]]) -> tuple[dict[str, JsonValue], ...]:
    return tuple(object_value(call) for delta in deltas(events) for call in list_value(delta.get("tool_calls") or []))


def final_usage(events: Sequence[Mapping[str, JsonValue]]) -> dict[str, JsonValue]:
    usages: Final = [object_value(event["usage"]) for event in events if isinstance(event.get("usage"), dict)]
    assert usages, "the stream carried no usage chunk"
    return usages[-1]


def encoded_file_id(file_id: str, model: str) -> str:
    """The proxy's model-scoped file id, spelled out from `encode_file_id_with_model`
    (`litellm/proxy/openai_files_endpoints/common_utils.py:327`)."""
    return "file-" + base64.urlsafe_b64encode(f"litellm:{file_id};model,{model}".encode()).decode().rstrip("=")


def spend_of(gateway: Gateway, model_group: str) -> float:
    """The `spend` of the one SpendLogs row of a model group created for one test."""
    rows: Final = eventually(
        lambda: gateway.rows('SELECT spend FROM "LiteLLM_SpendLogs" WHERE model_group = %s', (model_group,)),
        lambda found: len(found) == 1,
        seconds=LANDING_SECONDS,
    )
    value: Final = rows[0]["spend"]
    assert isinstance(value, (int, float, str)), rows
    return float(value)
