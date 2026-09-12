import json
from collections.abc import Mapping
from typing import Final

import pytest
from openai._streaming import ServerSentEvent

from litellm.litellm_core_utils.prompt_templates.server_tool_responses import (
    combined_usage,
    object_items,
    object_value,
    response_has_client_tools,
)
from litellm.litellm_core_utils.prompt_templates.server_tool_stream import ServerToolStream

_MEMORY: Final = frozenset(("litellm_memory_search",))


def test_usage_sums_nested_token_counts_and_preserves_provider_metadata() -> None:
    assert combined_usage(
        (
            {
                "input_tokens": 100,
                "input_tokens_details": {"cached_tokens": 80},
                "service_tier": "standard",
                "flag": True,
            },
            {
                "input_tokens": 140,
                "input_tokens_details": {"cached_tokens": 120},
                "service_tier": "standard",
                "flag": True,
            },
        )
    ) == {"input_tokens": 240, "input_tokens_details": {"cached_tokens": 200}, "service_tier": "standard", "flag": True}


def _event(stream: ServerToolStream, data: Mapping[str, object]) -> bytes:
    return b"".join(stream.feed(ServerSentEvent(data=json.dumps(data))))


def test_anthropic_stream_hides_memory_keeps_client_tool_ids_and_streams_text_before_completion() -> None:
    stream: Final = ServerToolStream("anthropic_messages", _MEMORY)
    start: Final = _event(
        stream,
        {
            "type": "message_start",
            "message": {
                "id": "msg_1",
                "role": "assistant",
                "model": "test",
                "content": [],
                "usage": {"input_tokens": 100, "output_tokens": 0},
            },
        },
    )
    assert b"msg_1" in start
    text: Final = _event(
        stream, {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": "Working"}}
    )
    assert b"Working" in text
    assert _event(stream, {"type": "content_block_stop", "index": 0})
    assert not _event(
        stream,
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {
                "type": "tool_use",
                "id": "memory_1",
                "name": "litellm_memory_search",
                "input": {"query": "routing"},
            },
        },
    )
    assert not _event(stream, {"type": "content_block_stop", "index": 1})
    client: Final = _event(
        stream,
        {
            "type": "content_block_start",
            "index": 2,
            "content_block": {
                "type": "tool_use",
                "id": "client_1",
                "name": "Read",
                "input": {"path": "README.md"},
            },
        },
    )
    assert b'"index": 1' in client and b"client_1" in client and b"README.md" in client
    assert _event(stream, {"type": "content_block_stop", "index": 2})
    assert not _event(
        stream, {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 20}}
    )
    assert not _event(stream, {"type": "message_stop"})
    native, delayed = stream.finish_round()
    assert delayed == ()
    assert len(object_items(native["content"])) == 3
    public: Final = stream.response()
    assert object_items(public["content"]) == (
        {"type": "text", "text": "Working"},
        {"type": "tool_use", "id": "client_1", "name": "Read", "input": {"path": "README.md"}},
    )
    terminal: Final = b"".join(stream.finish())
    assert terminal.count(b"event: message_stop") == 1
    assert b"memory_1" not in terminal


def test_chat_stream_buffers_fragmented_memory_names_without_delaying_visible_text() -> None:
    stream: Final = ServerToolStream("acompletion", _MEMORY)

    def chunk(delta: Mapping[str, object], finish: str | None = None) -> bytes:
        return _event(
            stream,
            {
                "id": "chat_1",
                "model": "test",
                "created": 1,
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            },
        )

    assert b"Thinking aloud" in chunk({"role": "assistant", "content": "Thinking aloud"})
    assert not chunk(
        {
            "tool_calls": [
                {
                    "index": 0,
                    "id": "mem_1",
                    "type": "function",
                    "function": {"name": "litellm_memory_", "arguments": ""},
                }
            ]
        }
    )
    assert not chunk({"tool_calls": [{"index": 0, "function": {"name": "search", "arguments": '{"query":"routing"}'}}]})
    assert not chunk(
        {
            "tool_calls": [
                {
                    "index": 1,
                    "id": "read_1",
                    "type": "function",
                    "function": {"name": "Read", "arguments": '{"path":"README.md"}'},
                }
            ]
        }
    )
    assert not chunk({}, "tool_calls")
    native, client = stream.finish_round()
    assert len(object_items(object_value(object_items(native["choices"])[0]["message"])["tool_calls"])) == 2
    wire: Final = b"".join(client)
    assert b"read_1" in wire and b'"index": 0' in wire
    assert b"litellm_memory" not in wire and b"mem_1" not in wire
    assert b"[DONE]" in b"".join(stream.finish())


def test_incomplete_stream_never_yields_an_executable_memory_call() -> None:
    stream: Final = ServerToolStream("anthropic_messages", _MEMORY)
    assert not _event(
        stream,
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "tool_use",
                "id": "memory_1",
                "name": "litellm_memory_search",
                "input": {},
            },
        },
    )
    with pytest.raises(ValueError, match="terminal"):
        stream.finish_round()


def test_chat_custom_tool_keeps_its_id_and_fragmented_input() -> None:
    stream: Final = ServerToolStream("acompletion", _MEMORY)
    pieces: Final = (
        {
            "tool_calls": [
                {"index": 0, "id": "patch_1", "type": "custom", "custom": {"name": "apply_patch", "input": "*** Begin"}}
            ]
        },
        {"tool_calls": [{"index": 0, "custom": {"input": " Patch\n*** End Patch"}}]},
    )
    for delta in pieces:
        assert not _event(
            stream,
            {
                "id": "chat_custom",
                "model": "test",
                "created": 1,
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
            },
        )
    assert not _event(
        stream,
        {
            "id": "chat_custom",
            "model": "test",
            "created": 1,
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
        },
    )
    native, delayed = stream.finish_round()
    assert response_has_client_tools(native, "acompletion", _MEMORY)
    public: Final = b"".join(delayed)
    assert b"patch_1" in public and b"apply_patch" in public and b"End Patch" in public
    message: Final = object_value(object_items(stream.response()["choices"])[0]["message"])
    call: Final = object_items(message["tool_calls"])[0]
    assert call["id"] == "patch_1"
    assert call["custom"] == {"name": "apply_patch", "input": "*** Begin Patch\n*** End Patch"}


def test_responses_rounds_keep_custom_tool_identity_and_emit_one_aggregate_completion() -> None:
    requested: Final = {
        "instructions": "Client instructions",
        "tools": [{"type": "custom", "name": "apply_patch"}],
        "previous_response_id": "resp_previous",
    }
    stream: Final = ServerToolStream("aresponses", _MEMORY, requested)
    stream.response_id = "resp_public"
    first: Final = _event(
        stream,
        {
            "type": "response.created",
            "response": {
                "id": "resp_native_1",
                "output": [],
                "instructions": "Injected memory instructions",
                "tools": [{"name": "litellm_memory_search"}],
            },
        },
    )
    assert b"resp_public" in first and b"resp_native_1" not in first
    assert (
        b"Client instructions" in first
        and b"Injected memory instructions" not in first
        and b"litellm_memory_search" not in first
    )
    memory: Final = {
        "type": "function_call",
        "id": "fc_memory",
        "call_id": "memory_1",
        "name": "litellm_memory_search",
        "arguments": "{}",
    }
    assert not _event(stream, {"type": "response.output_item.added", "output_index": 0, "item": memory})
    assert not _event(
        stream,
        {"type": "response.function_call_arguments.delta", "output_index": 0, "item_id": "fc_memory", "delta": "{}"},
    )
    assert not _event(
        stream,
        {
            "type": "response.completed",
            "response": {
                "id": "resp_native_1",
                "status": "completed",
                "output": [memory],
                "usage": {"input_tokens": 100, "output_tokens": 10},
            },
        },
    )
    stream.finish_round()
    stream.begin_round()
    assert not _event(stream, {"type": "response.created", "response": {"id": "resp_native_2", "output": []}})
    custom: Final = {
        "type": "custom_tool_call",
        "id": "ctc_apply",
        "call_id": "apply_1",
        "name": "apply_patch",
        "input": "*** Begin Patch\n*** End Patch",
    }
    shown: Final = _event(stream, {"type": "response.output_item.added", "output_index": 0, "item": custom})
    assert b"apply_1" in shown and b'"output_index": 0' in shown
    delta: Final = _event(
        stream,
        {
            "type": "response.custom_tool_call_input.delta",
            "output_index": 0,
            "item_id": "ctc_apply",
            "delta": custom["input"],
        },
    )
    assert b"ctc_apply" in delta and b"Begin Patch" in delta
    assert not _event(
        stream,
        {
            "type": "response.completed",
            "response": {
                "id": "resp_native_2",
                "status": "completed",
                "output": [custom],
                "usage": {"input_tokens": 120, "output_tokens": 20},
            },
        },
    )
    native, delayed = stream.finish_round()
    assert not delayed and response_has_client_tools(native, "aresponses", _MEMORY)
    final: Final = stream.response()
    assert final["id"] == "resp_public" and final["output"] == [custom]
    assert {name: final[name] for name in requested} == requested
    assert final["usage"] == {"input_tokens": 220, "output_tokens": 30}
    terminal: Final = b"".join(stream.finish())
    assert terminal.count(b"event: response.completed") == 1 and b"litellm_memory" not in terminal
    wire: Final = b"".join((first, shown, delta, terminal))
    sequences: Final = tuple(
        json.loads(line[6:])["sequence_number"] for line in wire.decode().splitlines() if line.startswith("data: ")
    )
    assert sequences == tuple(range(len(sequences)))
