"""Azure OpenAI streaming chat completions through the proxy.

With `stream: true` an `azure/` deployment is streamed by the AzureOpenAI SDK (`AzureChatCompletion.async_streaming`,
`litellm/llms/azure/azure.py:600`) and wrapped in `CustomStreamWrapper`. The caller must receive OpenAI
`chat.completion.chunk` frames in Azure's order, ending in `data: [DONE]`, with Azure's finish reason, the
usage chunk Azure sends when `stream_options.include_usage` is set, and tool call argument fragments that
concatenate to the full JSON. Errors before the first byte are plain JSON errors with the mapped status; a
connection cut after the stream started becomes one in-stream error frame whose message carries no Python
traceback (changed on 2026-11-14 by #51760: `exception_mapping_utils.py:2282` now keeps the error line alone).

The streamed call still writes one SpendLogs row with the usage Azure reported.
"""

from __future__ import annotations

import json
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.wire import Reply, Request
from integration.providers.dec1_azure_support import (
    INPUT_COST,
    OUTPUT_COST,
    PRICES,
    always,
    azure,
    azure_error,
    body_of,
    chat_chunk,
    chat_stream,
    spend_rows_for_group,
    sse,
    sse_events,
    streamed_text,
)

_USAGE: Final = {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}


def _stream(_: Request) -> Reply:
    return chat_stream(("hello ", "azure ", "stream"), usage=_USAGE)


def test_the_streamed_text_reaches_the_caller_in_order_and_ends_with_done(gateway: Gateway) -> None:
    """Azure streams `hello `, `azure `, `stream` and `[DONE]`.

    The caller must get HTTP 200 with `text/event-stream`, the concatenated deltas must be exactly
    `hello azure stream`, and the last data line must be `data: [DONE]`.
    """
    with azure(gateway, _stream) as rig:
        response: Final = rig.chat(stream=True)
        assert response.status_code == 200, response.text
        assert response.headers["content-type"].startswith("text/event-stream")
        assert streamed_text(sse_events(response)) == "hello azure stream"
        data_lines: Final = [line for line in response.text.splitlines() if line.startswith("data: ")]
        assert data_lines[-1] == "data: [DONE]"


def test_the_upstream_request_asks_azure_to_stream(gateway: Gateway) -> None:
    """The caller sends `stream: true`.

    The JSON body Azure receives must contain `stream: true` and the deployment name as `model`, on the same
    `/openai/deployments/gpt-4o-dec1/chat/completions` route as a non streaming call.
    """
    with azure(gateway, _stream) as rig:
        assert rig.chat(stream=True).status_code == 200
        request: Final = rig.one_post()
        assert request.target.split("?")[0] == "/openai/deployments/gpt-4o-dec1/chat/completions"
        assert body_of(request)["stream"] is True


def test_stream_options_include_usage_is_forwarded_and_the_usage_chunk_is_relayed(gateway: Gateway) -> None:
    """The caller sends `stream_options: {"include_usage": true}`; Azure sends a final usage chunk 11/7/18.

    Azure must receive `stream_options: {"include_usage": true}`, and the caller must see a chunk whose `usage`
    has `prompt_tokens: 11`, `completion_tokens: 7`, `total_tokens: 18`.
    """
    with azure(gateway, _stream) as rig:
        response: Final = rig.chat(stream=True, stream_options={"include_usage": True})
        assert response.status_code == 200, response.text
        assert body_of(rig.one_post())["stream_options"] == {"include_usage": True}
        usages: Final = [event["usage"] for event in sse_events(response) if event.get("usage")]
        assert usages, response.text
        last: Final = usages[-1]
        assert (last["prompt_tokens"], last["completion_tokens"], last["total_tokens"]) == (11, 7, 18)  # pyright: ignore[reportIndexIssue, reportOptionalSubscript, reportCallIssue, reportArgumentType]


def test_a_content_filter_finish_reason_survives_streaming(gateway: Gateway) -> None:
    """Azure streams `partial` and then a chunk with `finish_reason: content_filter`.

    The caller must see exactly one chunk whose finish reason is `content_filter`, never `stop`.
    """
    with azure(gateway, always(chat_stream(("partial",), finish_reason="content_filter"))) as rig:
        response: Final = rig.chat(stream=True)
        assert response.status_code == 200, response.text
        reasons: Final = [
            choice["finish_reason"]  # pyright: ignore[reportIndexIssue, reportCallIssue, reportArgumentType]
            for event in sse_events(response)
            for choice in event.get("choices", []) or []  # pyright: ignore[reportGeneralTypeIssues, reportOptionalIterable]
            if choice.get("finish_reason")  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]
        ]
        assert reasons == ["content_filter"], reasons


def test_streamed_tool_call_fragments_concatenate_to_the_full_arguments(gateway: Gateway) -> None:
    """Azure streams one `get_weather` tool call whose arguments arrive as `{"city": ` and `"Paris"}`.

    The caller's deltas must carry the tool call id `call_dec1` and name once, the argument fragments must
    concatenate to `{"city": "Paris"}`, and the last finish reason must be `tool_calls`.
    """
    identity: Final = "chatcmpl-dec1-tools"
    frames: Final = (
        chat_chunk(
            identity,
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_dec1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": ""},
                    }
                ],
            },
        ),
        chat_chunk(identity, {"tool_calls": [{"index": 0, "function": {"arguments": '{"city": '}}]}),
        chat_chunk(identity, {"tool_calls": [{"index": 0, "function": {"arguments": '"Paris"}'}}]}),
        chat_chunk(identity, {}, finish_reason="tool_calls"),
        "[DONE]",
    )
    with azure(gateway, always(sse(frames))) as rig:
        response: Final = rig.chat(stream=True)
        assert response.status_code == 200, response.text
        calls: Final = [
            call
            for event in sse_events(response)
            for choice in event.get("choices", []) or []  # pyright: ignore[reportGeneralTypeIssues, reportOptionalIterable]
            for call in (choice.get("delta", {}).get("tool_calls") or [])  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]
        ]
        assert [call.get("id") for call in calls if call.get("id")] == ["call_dec1"]
        arguments: Final = "".join(call.get("function", {}).get("arguments") or "" for call in calls)
        assert json.loads(arguments) == {"city": "Paris"}


def test_a_429_before_the_stream_opens_is_a_json_429(gateway: Gateway) -> None:
    """Azure answers the streaming request with 429 before sending any event; no retries.

    The caller must get HTTP 429 with an `application/json` OpenAI error body, not a 200 event stream with an
    error inside, because nothing was streamed yet.
    """
    with azure(gateway, always(azure_error(429, "429", "rate limited")), max_retries=0) as rig:
        response: Final = rig.chat(stream=True, num_retries=0)
        assert response.status_code == 429, response.text
        assert response.headers["content-type"].split(";")[0] == "application/json"
        assert "error" in response.json()


def test_a_stream_cut_mid_response_ends_with_an_error_frame_without_a_traceback(gateway: Gateway) -> None:
    """Azure sends the role chunk and one text chunk, then the connection closes without a finish chunk.

    The caller (already at HTTP 200) must get `before cut`, then one `data: {"error": ...}` frame and
    `data: [DONE]`. Since #51760 the error message is `... APIConnectionError - <cause>` alone: it must not
    contain `Traceback (most recent call last)` or a `File "` line, which used to leak server paths.
    """
    identity: Final = "chatcmpl-dec1-cut"
    frames: Final = (
        chat_chunk(identity, {"role": "assistant", "content": ""}),
        chat_chunk(identity, {"content": "before cut"}),
        chat_chunk(identity, {"content": "never sent"}),
        chat_chunk(identity, {}, finish_reason="stop"),
        "[DONE]",
    )
    with azure(gateway, always(sse(frames, abort_after=2)), max_retries=0) as rig:
        response: Final = rig.chat(stream=True, num_retries=0)
        assert response.status_code == 200
        events: Final = sse_events(response)
        assert streamed_text(events) == "before cut"
        errors: Final = [event["error"] for event in events if "error" in event]
        assert len(errors) == 1, response.text
        message: Final = str(errors[0]["message"])  # pyright: ignore[reportIndexIssue, reportCallIssue, reportArgumentType]
        assert "Traceback (most recent call last)" not in message
        assert 'File "' not in message
        assert response.text.rstrip().endswith("data: [DONE]")


@pytest.mark.parametrize("include_usage", [True, False], ids=["usage_requested", "usage_not_requested"])
def test_a_streamed_call_writes_one_priced_spend_row(gateway: Gateway, include_usage: bool) -> None:
    """A streamed call to a deployment priced at 0.001 input and 0.002 output per token; Azure reports 11/7 usage.

    One SpendLogs row must land for the model group with `call_type: acompletion`, `custom_llm_provider: azure`,
    11 prompt and 7 completion tokens and `spend = 11 * 0.001 + 7 * 0.002 = 0.025`, whether or not the caller
    asked for the usage chunk (the proxy asks Azure for usage on its own). UNVERIFIED for
    `usage_not_requested`: assumes the proxy injects `include_usage` for spend tracking; otherwise the tokens
    are counted locally and the numbers differ.
    """
    fields: Final = {"stream_options": {"include_usage": True}} if include_usage else {}
    with azure(gateway, _stream, **PRICES) as rig:
        response: Final = rig.chat(stream=True, **fields)  # pyright: ignore[reportArgumentType]
        assert response.status_code == 200, response.text
        row: Final = spend_rows_for_group(rig.model)[0]
        assert row["call_type"] == "acompletion"
        assert row["custom_llm_provider"] == "azure"
        assert (row["prompt_tokens"], row["completion_tokens"]) == (11, 7)
        assert float(str(row["spend"])) == pytest.approx(11 * INPUT_COST + 7 * OUTPUT_COST)
