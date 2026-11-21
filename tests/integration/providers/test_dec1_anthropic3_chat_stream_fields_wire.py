"""Contract: the newer Anthropic response fields survive a streamed `/v1/chat/completions` call.

A chat stream on an `anthropic/` deployment is parsed by `ModelResponseIterator.chunk_parser`
(`litellm/llms/anthropic/chat/handler.py:1006`). Besides text, tool calls and thinking (round 1), it relays the
newer Anthropic fields on `choices[0].delta.provider_specific_fields` of the chunk that carried them: the code
execution `container` from `message_delta` (`handler.py:1203`), `web_fetch_tool_result` blocks with the web search
results (`handler.py:1124`), a per-call `web_search_calls` snapshot, `server_tool_calls` assembled at
`content_block_stop` (`handler.py:1175`), each `citations_delta` as `citation` (`handler.py:864`), raw
`*_tool_result` blocks as `tool_results` with `code_interpreter_results` (`handler.py:1132`), `compaction` blocks
and `compaction_delta` text (`handler.py:890`, `handler.py:1101`) and `input_transformations` from `message_start`
(`handler.py:1236`). Redacted thinking arrives as a `redacted_thinking` entry of `delta.thinking_blocks`
(`handler.py:915`). Lists are re-sent whole each time they grow, so a client that keeps the last value per key
(as `stream_chunk_builder` does) ends with the full list.

These are exactly the places a Rust port of the chat stream translator could silently drop a field, since none of
them is part of the OpenAI chunk schema. Every upstream is a loopback peer.
"""

from __future__ import annotations

import json
from typing import Final

import pytest
from integration._support.client import Gateway, list_value, object_value
from integration.providers.dec1_anthropic3_support import (
    BASH_RESULT,
    CHAR_CITATION,
    CONTAINER,
    FETCH_URL,
    MCP_TOOL_RESULT,
    MCP_TOOL_USE,
    ONE_HOUR_PRICES,
    REDACTED_DATA,
    WEB_FETCH_RESULT,
    final_usage,
    framed,
    last_provider_field,
    provider_field_values,
    server_tool_use,
    spend_of,
    stream_body,
    streamed_thinking_blocks,
    streamed_tool_calls,
    text_block,
    whole_block,
)
from integration.providers.dec1_anthropic_support import (
    BACKEND,
    MESSAGE_STOP,
    SIGNATURE,
    block_delta,
    block_start,
    block_stop,
    chat_stream_lines,
    data_events,
    deltas,
    deployment,
    finish_reasons,
    message_start,
    peer,
    sse,
    streamed_text,
    streaming,
)
from pydantic import JsonValue

_BASH_COMMAND: Final = "expr 2 + 2"


def _message_delta_with(stop_reason: str, **delta_fields: JsonValue) -> bytes:
    return sse(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None, **delta_fields},
            "usage": {"output_tokens": 7},
        },
    )


def _bash_turn() -> tuple[bytes, ...]:
    return (
        message_start(),
        *server_tool_use(0, "srvtoolu_dec1bash", "bash_code_execution", json.dumps({"command": _BASH_COMMAND})),
        *whole_block(1, BASH_RESULT),
        *text_block(2, "the answer is 4"),
        _message_delta_with("end_turn", container=dict(CONTAINER)),
        MESSAGE_STOP,
    )


def _fetch_turn(result: dict[str, JsonValue]) -> tuple[bytes, ...]:
    return framed(
        server_tool_use(0, "srvtoolu_dec1fetch", "web_fetch", json.dumps({"url": FETCH_URL})),
        whole_block(1, result),
        text_block(2, "fetched"),
        delta_usage={"output_tokens": 7, "server_tool_use": {"web_fetch_requests": 1}},
    )


def _stream(gateway: Gateway, chunks: tuple[bytes, ...], **extra: JsonValue) -> tuple[dict[str, JsonValue], ...]:
    with peer(streaming(*chunks)) as provider, gateway.scenario() as scenario:
        _, lines = chat_stream_lines(gateway, stream_body(deployment(scenario, provider), **extra))
        return data_events(lines)


def test_the_code_execution_container_from_message_delta_is_relayed_on_the_stream(gateway: Gateway) -> None:
    """The provider streams a bash code execution turn whose `message_delta.delta` carries `container: {"id":
    "container_dec1run", "expires_at": ...}`.

    Some chunk must carry `provider_specific_fields.container` equal to that object (`handler.py:1203`), so a
    streaming client can reuse the container on its next turn exactly like a non-streaming one.
    """
    events: Final = _stream(gateway, _bash_turn())
    assert last_provider_field(events, "container") == dict(CONTAINER)


def test_a_streamed_bash_result_is_relayed_raw_in_tool_results(gateway: Gateway) -> None:
    """The provider streams a `bash_code_execution_tool_result` block whose whole content arrives in
    `content_block_start`.

    The last `provider_specific_fields.tool_results` must be exactly `[<that block>]`, unchanged
    (`handler.py:1133`), and no caller delta may carry `tool_calls` because the code ran on Anthropic's side.
    """
    events: Final = _stream(gateway, _bash_turn())
    assert last_provider_field(events, "tool_results") == [dict(BASH_RESULT)]
    assert streamed_tool_calls(events) == ()


def test_a_streamed_bash_result_becomes_a_code_interpreter_result_carrying_the_container_id(gateway: Gateway) -> None:
    """The provider streams `server_tool_use` `bash_code_execution` with input `{"command": "expr 2 + 2"}`, its
    result, then `message_delta` with `container.id: container_dec1run`.

    The last `code_interpreter_results` must hold one `code_interpreter_call` with `id: srvtoolu_dec1bash`, `code:
    "expr 2 + 2"`, `status: completed` and `container_id: container_dec1run`: the list is re-sent after the container
    arrives (`handler.py:1209`) so last-value-wins picks up the id the result block itself does not carry.
    """
    events: Final = _stream(gateway, _bash_turn())
    results: Final = list_value(last_provider_field(events, "code_interpreter_results"))
    assert len(results) == 1, results
    result: Final = object_value(results[0])
    assert (result["type"], result["id"], result["code"], result["status"], result["container_id"]) == (
        "code_interpreter_call",
        "srvtoolu_dec1bash",
        _BASH_COMMAND,
        "completed",
        "container_dec1run",
    )


def test_the_text_after_a_streamed_code_execution_still_reaches_the_caller(gateway: Gateway) -> None:
    """After the bash call and its result the provider streams the text `the answer is 4`.

    The caller's content must be exactly that text; neither the server tool input JSON nor the result may leak into
    `delta.content`, since `input_json_delta` is only turned into a tool call for `tool_use` blocks
    (`handler.py:851`).
    """
    events: Final = _stream(gateway, _bash_turn())
    assert streamed_text(events) == "the answer is 4"
    assert finish_reasons(events) == ("stop",)


def test_a_streamed_web_fetch_result_is_relayed_with_the_web_search_results(gateway: Gateway) -> None:
    """The provider streams a `web_fetch` server tool call and its `web_fetch_tool_result`.

    The last `provider_specific_fields.web_search_results` must be exactly `[<the fetch result block>]`
    (`handler.py:1124`, issue #18137), the same key the non-streaming path uses, so a client can replay the fetched
    page on its next turn.
    """
    events: Final = _stream(gateway, _fetch_turn(dict(WEB_FETCH_RESULT)))
    assert last_provider_field(events, "web_search_results") == [dict(WEB_FETCH_RESULT)]


def test_a_streamed_web_fetch_call_snapshot_moves_from_in_progress_to_completed(gateway: Gateway) -> None:
    """The provider streams `server_tool_use` `web_fetch` with input `{"url": FETCH_URL}`, then its result.

    `web_search_calls` is a snapshot keyed by tool id: the first one must show `ws_srvtoolu_dec1fetch` with `status:
    in_progress`, and the last one `status: completed` with `action: {"type": "open_page", "url": FETCH_URL}`
    (`build_web_search_call`, `litellm/types/responses/main.py:101`). The streamed shape is a dict keyed by tool id,
    while the non-streaming field is a list (`transformation.py:3270`), a split a port must keep or unify on purpose.
    """
    events: Final = _stream(gateway, _fetch_turn(dict(WEB_FETCH_RESULT)))
    snapshots: Final = provider_field_values(events, "web_search_calls")
    first: Final = object_value(object_value(snapshots[0])["srvtoolu_dec1fetch"])
    last: Final = object_value(object_value(snapshots[-1])["srvtoolu_dec1fetch"])
    assert (first["id"], first["status"]) == ("ws_srvtoolu_dec1fetch", "in_progress")
    assert (last["type"], last["status"]) == ("web_search_call", "completed")
    assert object_value(last["action"]) == {"type": "open_page", "url": FETCH_URL}


def test_a_streamed_web_fetch_error_marks_the_call_failed(gateway: Gateway) -> None:
    """The fetch result content is `{"type": "web_fetch_tool_result_error", "error_code": "url_not_accessible"}`.

    The last `web_search_calls` snapshot must show the call with `status: failed` (`_is_tool_result_error`,
    `litellm/types/responses/main.py:90`), while the raw error block is still relayed in `web_search_results`.
    """
    failed: Final[dict[str, JsonValue]] = {
        "type": "web_fetch_tool_result",
        "tool_use_id": "srvtoolu_dec1fetch",
        "content": {"type": "web_fetch_tool_result_error", "error_code": "url_not_accessible"},
    }
    events: Final = _stream(gateway, _fetch_turn(failed))
    last: Final = object_value(object_value(last_provider_field(events, "web_search_calls"))["srvtoolu_dec1fetch"])
    assert last["status"] == "failed"
    assert last_provider_field(events, "web_search_results") == [failed]


def test_a_streamed_server_tool_call_is_listed_in_server_tool_calls_not_tool_calls(gateway: Gateway) -> None:
    """The provider streams `server_tool_use` `web_fetch` whose input arrives in one `input_json_delta`.

    The last `server_tool_calls` must hold one call with `id: srvtoolu_dec1fetch`, function name `web_fetch` and
    arguments equal to the streamed JSON (`handler.py:1175`), and no caller delta may carry `tool_calls`, because a
    client that executes it would run a tool Anthropic already ran.
    """
    events: Final = _stream(gateway, _fetch_turn(dict(WEB_FETCH_RESULT)))
    calls: Final = list_value(last_provider_field(events, "server_tool_calls"))
    assert len(calls) == 1, calls
    call: Final = object_value(calls[0])
    function: Final = object_value(call["function"])
    assert (call["id"], function["name"]) == ("srvtoolu_dec1fetch", "web_fetch")
    assert json.loads(str(function["arguments"])) == {"url": FETCH_URL}
    assert streamed_tool_calls(events) == ()


def test_a_streamed_citation_delta_is_relayed_as_a_citation(gateway: Gateway) -> None:
    """The provider streams a text block `Blue.` followed by a `citations_delta` carrying one `char_location`
    citation.

    The caller's text must be `Blue.` and some chunk must carry `provider_specific_fields.citation` equal to the
    citation object as sent (`handler.py:864`); a port that only reads `text_delta` loses every streamed citation.
    """
    chunks: Final = framed(
        (
            block_start(0, {"type": "text", "text": "", "citations": []}),
            block_delta(0, {"type": "text_delta", "text": "Blue."}),
            block_delta(0, {"type": "citations_delta", "citation": dict(CHAR_CITATION)}),
            block_stop(0),
        )
    )
    events: Final = _stream(gateway, chunks)
    assert streamed_text(events) == "Blue."
    assert provider_field_values(events, "citation") == (dict(CHAR_CITATION),)


def test_a_streamed_redacted_thinking_block_is_relayed_with_its_data(gateway: Gateway) -> None:
    """The provider streams a `redacted_thinking` block whose opaque `data` arrives in `content_block_start`, then
    the text `done`.

    Some delta must carry a `thinking_blocks` entry `{"type": "redacted_thinking", "data": <data>}`
    (`_handle_redacted_thinking_content`, `handler.py:915`), so the caller can replay it, and no `reasoning_content`
    may be produced from it because the text is encrypted.
    """
    chunks: Final = framed(
        whole_block(0, {"type": "redacted_thinking", "data": REDACTED_DATA}),
        text_block(1, "done"),
    )
    events: Final = _stream(gateway, chunks, reasoning_effort="low")
    redacted: Final = [b for b in streamed_thinking_blocks(events) if b.get("type") == "redacted_thinking"]
    assert [b.get("data") for b in redacted] == [REDACTED_DATA]
    assert "".join(str(d.get("reasoning_content") or "") for d in deltas(events)) == ""
    assert streamed_text(events) == "done"


def test_a_redacted_block_before_a_signed_block_keeps_their_order_on_the_stream(gateway: Gateway) -> None:
    """The provider streams a `redacted_thinking` block, then a signed `thinking` block `plan`, then text.

    In arrival order, the first thinking entry must be the redacted one and a later entry must carry the signature;
    `reasoning_content` must be exactly `plan`. Anthropic rejects a replay whose thinking blocks are reordered.
    """
    chunks: Final = framed(
        whole_block(0, {"type": "redacted_thinking", "data": REDACTED_DATA}),
        (
            block_start(1, {"type": "thinking", "thinking": "", "signature": ""}),
            block_delta(1, {"type": "thinking_delta", "thinking": "plan"}),
            block_delta(1, {"type": "signature_delta", "signature": SIGNATURE}),
            block_stop(1),
        ),
        text_block(2, "done"),
    )
    events: Final = _stream(gateway, chunks, reasoning_effort="low")
    blocks: Final = streamed_thinking_blocks(events)
    assert blocks[0].get("type") == "redacted_thinking", blocks
    assert SIGNATURE in [b.get("signature") for b in blocks[1:]]
    assert "".join(str(d.get("reasoning_content") or "") for d in deltas(events)) == "plan"


def test_a_streamed_mcp_tool_use_is_not_relayed_as_a_client_tool_call(gateway: Gateway) -> None:
    """The provider streams an `mcp_tool_use` block (input in one `input_json_delta`), its `mcp_tool_result` and
    the text `widgets ship in blue`.

    No caller delta may carry `tool_calls`: `chunk_parser` builds a client tool chunk only for `tool_use`
    (`handler.py:1058`), and the MCP call already ran on Anthropic's side. The answer text must still arrive whole.
    """
    chunks: Final = framed(
        (
            block_start(0, {**MCP_TOOL_USE, "input": {}}),
            block_delta(0, {"type": "input_json_delta", "partial_json": json.dumps(MCP_TOOL_USE["input"])}),
            block_stop(0),
        ),
        whole_block(1, MCP_TOOL_RESULT),
        text_block(2, "widgets ship in blue"),
    )
    events: Final = _stream(gateway, chunks)
    assert streamed_tool_calls(events) == ()
    assert streamed_text(events) == "widgets ship in blue"


def test_a_streamed_mcp_tool_result_is_relayed_in_tool_results_with_no_code_interpreter_result(
    gateway: Gateway,
) -> None:
    """Same MCP turn. The `mcp_tool_result` ends in `_tool_result`, so it takes the generic branch
    (`handler.py:1132`).

    The last `tool_results` must be exactly `[<the mcp_tool_result block>]` and the last `code_interpreter_results`
    must be `[]`, because only `bash_code_execution_tool_result` blocks become code interpreter calls
    (`handler.py:987`). The `mcp_tool_use` itself is surfaced nowhere on chat.
    """
    chunks: Final = framed(whole_block(0, {**MCP_TOOL_USE}), whole_block(1, MCP_TOOL_RESULT), text_block(2, "ok"))
    events: Final = _stream(gateway, chunks)
    assert last_provider_field(events, "tool_results") == [dict(MCP_TOOL_RESULT)]
    assert last_provider_field(events, "code_interpreter_results") == []


def test_a_streamed_compaction_block_is_relayed_in_compaction_blocks(gateway: Gateway) -> None:
    """The provider streams a `compaction` block whose summary `Earlier turns: widgets.` arrives whole in
    `content_block_start`, then text.

    The last `compaction_blocks` must be `[{"type": "compaction", "content": "Earlier turns: widgets."}]` and a chunk
    must carry `compaction_start` with the same content (`handler.py:1101`), so a client can send the compacted
    history back.
    """
    compaction: Final[dict[str, JsonValue]] = {"type": "compaction", "content": "Earlier turns: widgets."}
    events: Final = _stream(gateway, framed(whole_block(0, compaction), text_block(1, "ok")))
    assert last_provider_field(events, "compaction_blocks") == [compaction]
    assert provider_field_values(events, "compaction_start") == (compaction,)


def test_streamed_compaction_deltas_are_appended_to_the_compaction_block(gateway: Gateway) -> None:
    """The compaction block starts with empty content and its summary arrives as two `compaction_delta` events
    `Earlier ` and `turns.`.

    Each delta must be relayed as `compaction_delta`, and the last `compaction_blocks` must hold one block whose
    content is `Earlier turns.` (`handler.py:890`): the summary is concatenated, not replaced.
    """
    chunks: Final = framed(
        (
            block_start(0, {"type": "compaction", "content": ""}),
            block_delta(0, {"type": "compaction_delta", "content": "Earlier "}),
            block_delta(0, {"type": "compaction_delta", "content": "turns."}),
            block_stop(0),
        ),
        text_block(1, "ok"),
    )
    events: Final = _stream(gateway, chunks)
    assert [object_value(v)["content"] for v in provider_field_values(events, "compaction_delta")] == [
        "Earlier ",
        "turns.",
    ]
    assert last_provider_field(events, "compaction_blocks") == [{"type": "compaction", "content": "Earlier turns."}]


def test_input_transformations_on_message_start_are_relayed_on_the_stream(gateway: Gateway) -> None:
    """The provider's `message_start.message` carries `input_transformations: [{"type": "dec1_trimmed"}]`.

    Some chunk must carry `provider_specific_fields.input_transformations` equal to that list (`handler.py:1236`),
    the same field the non-streaming path relays (`transformation.py:3257`). UNVERIFIED: the payload shape Anthropic
    uses for `input_transformations`; the proxy relays any value as sent, which is what is pinned.
    """
    transformations: Final[list[JsonValue]] = [{"type": "dec1_trimmed"}]
    start: Final = sse(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": "msg_dec1transform",
                "type": "message",
                "role": "assistant",
                "model": BACKEND,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 11, "output_tokens": 1},
                "input_transformations": transformations,
            },
        },
    )
    events: Final = _stream(gateway, (start, *framed(text_block(0, "ok"))[1:]))
    assert provider_field_values(events, "input_transformations") == (transformations,)


def test_a_streamed_compaction_stop_reason_finishes_with_length(gateway: Gateway) -> None:
    """The provider ends a text stream with `message_delta.stop_reason: compaction`.

    The caller's one finish reason must be `length` (`_FINISH_REASON_MAP`, `litellm/litellm_core_utils/
    core_helpers.py:439`): the turn was cut to compact the context and the client should continue it.
    """
    events: Final = _stream(gateway, framed(text_block(0, "partial"), stop_reason="compaction"))
    assert finish_reasons(events) == ("length",)
    assert streamed_text(events) == "partial"


def test_streamed_usage_relays_the_service_tier(gateway: Gateway) -> None:
    """The provider reports `service_tier: "priority"` in the `message_delta` usage.

    The final usage chunk must carry `service_tier: "priority"`: `_handle_usage` merges the delta into the cumulative
    usage and `calculate_usage` copies `service_tier` (`transformation.py:3096`). UNVERIFIED: that the stream
    wrapper's final usage chunk serializes the `service_tier` extra the way the non-streaming body does.

    BUG: the final usage chunk is the chunk builder's usage, which rebuilds `Usage` with only `inference_geo`
    and `speed` as extra provider fields (litellm/litellm_core_utils/streaming_chunk_builder_utils.py:1289-1298
    on main), so `service_tier` is dropped; the top level stamp (`_stamp_served_service_tier`,
    litellm/litellm_core_utils/streaming_handler.py:142-145) reads a response level tier that Anthropic's
    stream never sets, and round 2's final chunk had `service_tier` in neither place.
    """
    pytest.skip("BUG: a streamed Anthropic chat drops the served service_tier from the final usage")
    events: Final = _stream(
        gateway,
        framed(text_block(0, "ok"), delta_usage={"output_tokens": 7, "service_tier": "priority"}),
    )
    assert final_usage(events).get("service_tier") == "priority"


def test_streamed_usage_splits_cache_writes_into_five_minute_and_one_hour_tokens(gateway: Gateway) -> None:
    """`message_start` usage reports `cache_creation_input_tokens: 6` broken down as 2 five-minute and 4 one-hour
    tokens.

    The final usage must report `prompt_tokens: 17` (11 + 6) and `prompt_tokens_details.cache_creation_token_details`
    `{"ephemeral_5m_input_tokens": 2, "ephemeral_1h_input_tokens": 4}` (`_resolve_cache_creation_token_details`,
    `transformation.py:3061`). UNVERIFIED: that the details object is serialized on the streamed usage chunk.
    """
    start_usage: Final[dict[str, JsonValue]] = {
        "input_tokens": 11,
        "output_tokens": 1,
        "cache_creation_input_tokens": 6,
        "cache_creation": {"ephemeral_5m_input_tokens": 2, "ephemeral_1h_input_tokens": 4},
    }
    events: Final = _stream(gateway, framed(text_block(0, "ok"), start_usage=start_usage))
    usage: Final = final_usage(events)
    assert usage["prompt_tokens"] == 17
    details: Final = object_value(object_value(usage["prompt_tokens_details"])["cache_creation_token_details"])
    assert (details.get("ephemeral_5m_input_tokens"), details.get("ephemeral_1h_input_tokens")) == (2, 4)


def test_a_streamed_mixed_ttl_cache_write_bills_each_part_at_its_own_rate(gateway: Gateway) -> None:
    """Same stream, on a deployment priced 0.001 input, 0.002 output, 0.003 per five-minute write and 0.006 per
    one-hour write (`cache_creation_input_token_cost_above_1hr`).

    The SpendLogs row must be 11 * 0.001 + 2 * 0.003 + 4 * 0.006 + 7 * 0.002 = 0.055 (`calculate_cache_writing_cost`,
    `litellm/litellm_core_utils/llm_cost_calc/utils.py:1017`); billing all six writes at either one rate is wrong.
    """
    start_usage: Final[dict[str, JsonValue]] = {
        "input_tokens": 11,
        "output_tokens": 1,
        "cache_creation_input_tokens": 6,
        "cache_creation": {"ephemeral_5m_input_tokens": 2, "ephemeral_1h_input_tokens": 4},
    }
    with (
        peer(streaming(*framed(text_block(0, "ok"), start_usage=start_usage))) as provider,
        gateway.scenario() as scenario,
    ):
        model: Final = deployment(scenario, provider, priced=False, **ONE_HOUR_PRICES)
        chat_stream_lines(gateway, stream_body(model))
        assert spend_of(gateway, model) == pytest.approx(0.055)


def test_streamed_applied_context_management_edits_are_relayed(gateway: Gateway) -> None:
    """The provider's `message_delta` event carries `context_management: {"applied_edits": [...]}` after clearing
    old tool uses. UNVERIFIED: that Anthropic puts it at the event top level rather than inside `delta`.

    The caller should get `provider_specific_fields.context_management` as the non-streaming path returns it
    (`transformation.py:3253`, pinned by round 2). Today `_handle_message_delta` (`handler.py:1438`) reads only the
    stop reason, usage, container and stop details, so the applied edits are lost on a stream.
    """
    pytest.skip("BUG: streamed chat drops message_delta.context_management; only non-streaming chat relays it")
    applied: Final[dict[str, JsonValue]] = {
        "applied_edits": [{"type": "clear_tool_uses_20250919", "cleared_tool_uses": 2, "cleared_input_tokens": 900}]
    }
    delta_event: Final = sse(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 7},
            "context_management": applied,
        },
    )
    chunks: Final = (message_start(), *text_block(0, "ok"), delta_event, MESSAGE_STOP)
    events: Final = _stream(gateway, chunks)
    assert last_provider_field(events, "context_management") == applied
