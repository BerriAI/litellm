import json
import time
from collections.abc import Iterator, Mapping, Sequence
from typing import Final, cast

import httpx
import pytest
import respx
from pydantic import BaseModel

import litellm
from litellm.exceptions import MidStreamFallbackError
from litellm.responses.litellm_completion_transformation.compaction_items import summary_only_compaction_content
from litellm.responses.litellm_completion_transformation.compaction_trigger import (
    SUMMARY_PROMPT,
    CompactionSummaryStream,
    as_compaction_reply,
    ends_in_compaction_trigger,
    summary_from_reply,
    with_trailing_compaction_trigger_as_summary_prompt,
)
from litellm.types.llms.openai import ResponseInputParam, ResponsesAPIStreamEvents
from litellm.types.utils import Delta, ModelResponse, ModelResponseStream, StreamingChoices, Usage
from tests.unit.proxy.conftest import httpx_transport

_TRIGGER: Final = {"type": "compaction_trigger"}
_ASK: Final = {"role": "user", "content": "Read secret.txt and remember it."}
_CALL: Final = {"type": "function_call", "call_id": "toolu_read", "name": "shell", "arguments": '{"cmd":"cat"}'}
_CALL_OUTPUT: Final = {"type": "function_call_output", "call_id": "toolu_read", "output": "MARKER-42"}
_HISTORY: Final = (_ASK, _CALL, _CALL_OUTPUT)
_TOOLS: Final = ({"type": "function", "name": "shell", "parameters": {"type": "object", "properties": {}}},)
_SUMMARY: Final = "The user asked to remember secret.txt, which holds MARKER-42."
_SUMMARY_REPLY: Final = f"<analysis>Read the file.</analysis>\n<summary>\n{_SUMMARY}\n</summary>"
_MODEL: Final = "anthropic/claude-sonnet-5-5"
_FORCED_TOOL_MODEL: Final = "anthropic/claude-haiku-5-5"
_ANTHROPIC_MESSAGES_URL: Final = "https://api.anthropic.com/v1/messages"
_BEDROCK_MODEL: Final = "bedrock/us.anthropic.claude-sonnet-5-5"
_BEDROCK_CONVERSE_URL: Final = (
    "https://bedrock-runtime.us-east-1.amazonaws.com/model/us.anthropic.claude-sonnet-5-5/converse"
)
_THINKING: Final = {"type": "thinking", "thinking": "I should summarize.", "signature": "sig_1"}
_TOOL_USE: Final = {"type": "tool_use", "id": "toolu_more", "name": "shell", "input": {}}


def _trigger_request() -> list[object]:
    return [*_HISTORY, _TRIGGER]


@pytest.mark.parametrize(
    ("input", "holds"),
    [
        pytest.param([*_HISTORY, _TRIGGER], True, id="trigger_last"),
        pytest.param("Summarize this.", False, id="string_input"),
        pytest.param([], False, id="empty"),
        pytest.param(list(_HISTORY), False, id="no_trigger"),
        pytest.param([_ASK, _TRIGGER, _CALL, _CALL_OUTPUT], False, id="trigger_mid_history"),
    ],
)
def test_ends_in_compaction_trigger(input: object, holds: bool) -> None:
    assert ends_in_compaction_trigger(input) is holds


def test_the_trigger_becomes_the_summary_prompt_in_its_own_place() -> None:
    rewritten: Final = with_trailing_compaction_trigger_as_summary_prompt(cast(ResponseInputParam, _trigger_request()))

    assert isinstance(rewritten, list)
    assert rewritten[:-1] == list(_HISTORY)
    assert rewritten[-1] == {"role": "user", "type": "message", "content": SUMMARY_PROMPT}


@pytest.mark.parametrize(
    "input",
    [
        pytest.param("Summarize this.", id="string_input"),
        pytest.param(list(_HISTORY), id="no_trigger"),
        pytest.param([_ASK, _TRIGGER, _CALL, _CALL_OUTPUT], id="trigger_mid_history"),
    ],
)
def test_input_without_a_trigger_is_returned_as_is(input: str | list[object]) -> None:
    assert with_trailing_compaction_trigger_as_summary_prompt(cast(ResponseInputParam, input)) is input


@pytest.mark.parametrize(
    ("text", "summary"),
    [
        pytest.param(_SUMMARY_REPLY, _SUMMARY, id="summary_tags"),
        pytest.param(f"  {_SUMMARY}\n", _SUMMARY, id="no_tags"),
        pytest.param(None, None, id="none"),
        pytest.param("", None, id="empty"),
        pytest.param(" \n ", None, id="whitespace"),
        pytest.param("<summary> </summary>", None, id="empty_tags"),
        pytest.param("<SUMMARY>first</Summary> <summary>second</summary>", "first", id="first_pair_any_case"),
        pytest.param("<summary>open <summary>inner</summary>", "open <summary>inner", id="nested_opening"),
    ],
)
def test_summary_from_reply(text: str | None, summary: str | None) -> None:
    assert summary_from_reply(text) == summary


def test_a_reply_full_of_unclosed_summary_tags_is_read_in_linear_time() -> None:
    reply: Final = "<summary>" * 20_000
    started: Final = time.perf_counter()

    summary: Final = summary_from_reply(reply)

    assert summary == reply
    assert time.perf_counter() - started < 1.0


def _chat_reply(message: Mapping[str, object]) -> ModelResponse:
    return ModelResponse(
        id="chatcmpl-summary",
        model="claude-sonnet-5-5",
        choices=[{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", **message}}],
        usage={"prompt_tokens": 30, "completion_tokens": 12, "total_tokens": 42},
    )


def test_as_compaction_reply_carries_only_the_summary() -> None:
    reply: Final = as_compaction_reply(
        _chat_reply(
            {
                "content": _SUMMARY_REPLY,
                "tool_calls": [
                    {"id": "toolu_more", "type": "function", "function": {"name": "shell", "arguments": "{}"}}
                ],
                "reasoning_content": "I should summarize.",
            }
        ),
        llm_provider="anthropic",
    )
    message: Final = reply.choices[0].message

    assert message.content is None
    assert message.tool_calls is None
    assert message.reasoning_content == "I should summarize."
    assert message.provider_specific_fields == {"compaction_blocks": [{"type": "compaction", "content": _SUMMARY}]}
    assert reply.usage == Usage(prompt_tokens=30, completion_tokens=12, total_tokens=42)


def test_as_compaction_reply_is_a_502_when_the_model_wrote_no_summary() -> None:
    tool_call_only: Final = _chat_reply(
        {
            "content": None,
            "tool_calls": [{"id": "toolu_more", "type": "function", "function": {"name": "shell", "arguments": "{}"}}],
        }
    )

    with pytest.raises(litellm.APIError) as raised:
        as_compaction_reply(tool_call_only, llm_provider="anthropic")
    assert raised.value.status_code == 502
    assert "compaction_trigger" in str(raised.value)


def _stream_chunk(delta: Delta, **fields: object) -> ModelResponseStream:
    return ModelResponseStream(id="chatcmpl-summary", choices=[StreamingChoices(index=0, delta=delta)], **fields)


def test_compaction_summary_stream_keeps_the_answer_out_of_the_stream() -> None:
    summary: Final = CompactionSummaryStream(llm_provider="anthropic")
    taken: Final = tuple(
        summary.take(chunk)
        for chunk in (
            _stream_chunk(Delta(reasoning_content="Thinking.")),
            _stream_chunk(Delta(content="<summary>The user asked ")),
            _stream_chunk(
                Delta(
                    content="to remember MARKER-42.</summary>",
                    tool_calls=[{"id": "toolu_more", "type": "function", "function": {"name": "shell"}}],
                )
            ),
            _stream_chunk(Delta(), usage=Usage(prompt_tokens=30, completion_tokens=12, total_tokens=42)),
        )
    )

    assert [chunk.choices[0].delta.content for chunk in taken] == [None, None, None, None]
    assert [chunk.choices[0].delta.tool_calls for chunk in taken] == [None, None, None, None]
    assert taken[0].choices[0].delta.reasoning_content == "Thinking."
    assert getattr(taken[3], "usage", None) == Usage(prompt_tokens=30, completion_tokens=12, total_tokens=42)
    assert summary.provider_specific_fields("claude-sonnet-5-5") == {
        "compaction_blocks": [{"type": "compaction", "content": "The user asked to remember MARKER-42."}]
    }


def test_an_empty_compaction_summary_stream_is_a_502() -> None:
    summary: Final = CompactionSummaryStream(llm_provider="anthropic")
    summary.take(_stream_chunk(Delta(reasoning_content="Thinking.")))

    with pytest.raises(litellm.APIError) as raised:
        summary.provider_specific_fields("claude-sonnet-5-5")
    assert raised.value.status_code == 502


def _anthropic_message(*content: Mapping[str, object]) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "msg_summary",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-5-5",
            "content": list(content),
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 30, "output_tokens": 12},
        },
    )


def _anthropic_sse(*events: Mapping[str, object]) -> bytes:
    return b"".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode() for event in events)


def _anthropic_block_events(index: int, block: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    match block:
        case {"type": "text", "text": str() as text}:
            start, delta = {"type": "text", "text": ""}, {"type": "text_delta", "text": text}
        case {"type": "thinking", "thinking": str() as thinking}:
            start, delta = {"type": "thinking", "thinking": ""}, {"type": "thinking_delta", "thinking": thinking}
        case _:
            start, delta = {**block, "input": {}}, {"type": "input_json_delta", "partial_json": "{}"}
    return (
        {"type": "content_block_start", "index": index, "content_block": start},
        {"type": "content_block_delta", "index": index, "delta": delta},
        {"type": "content_block_stop", "index": index},
    )


def _anthropic_stream(*content: Mapping[str, object]) -> httpx.Response:
    blocks: Final = tuple(
        event for index, block in enumerate(content) for event in _anthropic_block_events(index, block)
    )
    return httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        content=_anthropic_sse(
            {
                "type": "message_start",
                "message": {
                    "id": "msg_summary_stream",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-sonnet-5-5",
                    "content": [],
                    "usage": {"input_tokens": 30, "output_tokens": 1},
                },
            },
            *blocks,
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 12}},
            {"type": "message_stop"},
        ),
    )


def _sent_body(route: respx.Route) -> Mapping[str, object]:
    return json.loads(route.calls.last.request.content)


def _last_user_text(body: Mapping[str, object]) -> str:
    messages: Final = cast(Sequence[Mapping[str, object]], body["messages"])
    content: Final = messages[-1]["content"]
    if isinstance(content, str):
        return content
    return cast(Sequence[Mapping[str, str]], content)[-1]["text"]


def _types(output: Sequence[object]) -> list[object]:
    return [getattr(item, "type", None) for item in output]


@pytest.mark.usefixtures(httpx_transport.__name__)
def test_a_compaction_trigger_over_anthropic_returns_one_compaction_item(respx_mock: respx.MockRouter) -> None:
    route: Final = respx_mock.post(_ANTHROPIC_MESSAGES_URL).mock(
        return_value=_anthropic_message({"type": "text", "text": _SUMMARY_REPLY})
    )

    response: Final = litellm.responses(
        model=_MODEL, api_key="mock_api_key", input=_trigger_request(), tools=list(_TOOLS)
    )
    body: Final = _sent_body(route)

    (item,) = response.output
    assert item.type == "compaction"
    assert item.id.startswith("cmp_")
    assert item.encrypted_content == summary_only_compaction_content(_SUMMARY)
    assert response.usage is not None
    assert (response.usage.input_tokens, response.usage.output_tokens, response.usage.total_tokens) == (30, 12, 42)
    assert _last_user_text(body).endswith(SUMMARY_PROMPT)
    assert "compaction_trigger" not in json.dumps(body)
    assert [tool["name"] for tool in cast(Sequence[Mapping[str, object]], body["tools"])] == ["shell"]


@pytest.mark.parametrize(
    "tool_choice",
    [
        pytest.param("required", id="required"),
        pytest.param({"type": "function", "name": "shell"}, id="named_tool"),
    ],
)
@pytest.mark.usefixtures(httpx_transport.__name__)
def test_a_compaction_trigger_with_a_forced_tool_lets_the_model_write_the_summary(
    respx_mock: respx.MockRouter, tool_choice: object
) -> None:
    route: Final = respx_mock.post(_ANTHROPIC_MESSAGES_URL).mock(
        return_value=_anthropic_message({"type": "text", "text": _SUMMARY_REPLY})
    )

    response: Final = litellm.responses(
        model=_FORCED_TOOL_MODEL,
        api_key="mock_api_key",
        input=_trigger_request(),
        tools=list(_TOOLS),
        tool_choice=tool_choice,
    )

    assert _sent_body(route)["tool_choice"] == {"type": "auto"}
    assert [(item.type, item.encrypted_content) for item in response.output] == [
        ("compaction", summary_only_compaction_content(_SUMMARY))
    ]


@pytest.mark.usefixtures(httpx_transport.__name__)
def test_a_forced_tool_without_a_trigger_is_still_forced(respx_mock: respx.MockRouter) -> None:
    route: Final = respx_mock.post(_ANTHROPIC_MESSAGES_URL).mock(
        return_value=_anthropic_message({"type": "text", "text": "Done."})
    )

    _ = litellm.responses(
        model=_FORCED_TOOL_MODEL,
        api_key="mock_api_key",
        input=list(_HISTORY),
        tools=list(_TOOLS),
        tool_choice="required",
    )

    assert _sent_body(route)["tool_choice"] == {"type": "any"}


def _event_summary(events: Sequence[BaseModel]) -> list[tuple[object, object]]:
    return [(event.type, getattr(getattr(event, "item", None), "type", None)) for event in events]


@pytest.mark.usefixtures(httpx_transport.__name__)
def test_a_streamed_compaction_trigger_over_anthropic_streams_one_compaction_item(
    respx_mock: respx.MockRouter,
) -> None:
    respx_mock.post(_ANTHROPIC_MESSAGES_URL).mock(
        return_value=_anthropic_stream({"type": "text", "text": _SUMMARY_REPLY})
    )

    events: Final = tuple(
        litellm.responses(
            model=_MODEL, api_key="mock_api_key", input=_trigger_request(), tools=list(_TOOLS), stream=True
        )
    )
    done: Final = [event for event in events if event.type == ResponsesAPIStreamEvents.OUTPUT_ITEM_DONE]
    completed: Final = events[-1]

    assert _event_summary(events) == [
        (ResponsesAPIStreamEvents.RESPONSE_CREATED, None),
        (ResponsesAPIStreamEvents.RESPONSE_IN_PROGRESS, None),
        (ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED, "compaction"),
        (ResponsesAPIStreamEvents.OUTPUT_ITEM_DONE, "compaction"),
        (ResponsesAPIStreamEvents.RESPONSE_COMPLETED, None),
    ]
    (compaction_done,) = done
    assert compaction_done.item.encrypted_content == summary_only_compaction_content(_SUMMARY)
    assert [item.id for item in completed.response.output] == [compaction_done.item.id]
    assert completed.response.usage is not None
    assert (
        completed.response.usage.input_tokens,
        completed.response.usage.output_tokens,
        completed.response.usage.total_tokens,
    ) == (30, 12, 42)


class _StreamThatDropsAfter(httpx.SyncByteStream):
    def __init__(self, text: str) -> None:
        self._text: Final = text

    def __iter__(self) -> Iterator[bytes]:
        yield _anthropic_sse(
            {
                "type": "message_start",
                "message": {
                    "id": "msg_dropped",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-sonnet-5-5",
                    "content": [],
                    "usage": {"input_tokens": 30, "output_tokens": 1},
                },
            },
            *_anthropic_block_events(0, {"type": "text", "text": self._text})[:2],
        )
        raise httpx.ReadError("connection dropped")


def _output_tokens_billed_on_disconnect(respx_mock: respx.MockRouter, input: list[object]) -> int:
    respx_mock.post(_ANTHROPIC_MESSAGES_URL).mock(
        return_value=httpx.Response(
            200, headers={"content-type": "text/event-stream"}, stream=_StreamThatDropsAfter(_SUMMARY_REPLY)
        )
    )
    stream: Final = litellm.responses(
        model=_MODEL, api_key="mock_api_key", input=input, tools=list(_TOOLS), stream=True
    )
    with pytest.raises(MidStreamFallbackError, match="connection dropped"):
        _ = tuple(stream)
    partial: Final = stream.partial_response_on_disconnect()
    assert partial is not None and partial.response.usage is not None
    return partial.response.usage.output_tokens


@pytest.mark.usefixtures(httpx_transport.__name__)
def test_a_summary_turn_cut_off_mid_stream_bills_the_summary_written_so_far(respx_mock: respx.MockRouter) -> None:
    summary_turn: Final = _output_tokens_billed_on_disconnect(respx_mock, _trigger_request())
    normal_turn: Final = _output_tokens_billed_on_disconnect(respx_mock, list(_HISTORY))

    assert normal_turn > 1
    assert summary_turn == normal_turn


@pytest.mark.usefixtures(httpx_transport.__name__)
def test_a_summary_turn_with_thinking_and_a_tool_call_still_returns_one_compaction_item(
    respx_mock: respx.MockRouter,
) -> None:
    content: Final = (_THINKING, _TOOL_USE, {"type": "text", "text": _SUMMARY_REPLY})
    respx_mock.post(_ANTHROPIC_MESSAGES_URL).mock(
        side_effect=[_anthropic_message(*content), _anthropic_stream(*content)]
    )

    response: Final = litellm.responses(
        model=_MODEL, api_key="mock_api_key", input=_trigger_request(), tools=list(_TOOLS)
    )
    events: Final = tuple(
        litellm.responses(
            model=_MODEL, api_key="mock_api_key", input=_trigger_request(), tools=list(_TOOLS), stream=True
        )
    )
    done_types: Final = [event.item.type for event in events if event.type == ResponsesAPIStreamEvents.OUTPUT_ITEM_DONE]

    assert sorted(_types(response.output)) == ["compaction", "reasoning"]
    assert done_types == ["reasoning", "compaction"]
    assert _types(events[-1].response.output) == ["reasoning", "compaction"]


@pytest.mark.usefixtures(httpx_transport.__name__)
def test_the_returned_compaction_item_is_sent_back_as_the_summary(respx_mock: respx.MockRouter) -> None:
    route: Final = respx_mock.post(_ANTHROPIC_MESSAGES_URL).mock(
        side_effect=[
            _anthropic_message({"type": "text", "text": _SUMMARY_REPLY}),
            _anthropic_message({"type": "text", "text": "MARKER-42"}),
        ]
    )
    compacted: Final = litellm.responses(
        model=_MODEL, api_key="mock_api_key", input=_trigger_request(), tools=list(_TOOLS)
    )

    litellm.responses(
        model=_MODEL,
        api_key="mock_api_key",
        input=[
            *_HISTORY,
            *(item.model_dump(exclude_none=True) for item in compacted.output),
            {"role": "user", "content": "What was in secret.txt?"},
        ],
        tools=list(_TOOLS),
    )

    assert _sent_body(route)["messages"] == [
        {
            "role": "user",
            "content": [{"type": "text", "text": _SUMMARY}, {"type": "text", "text": "What was in secret.txt?"}],
        }
    ]


@pytest.mark.usefixtures(httpx_transport.__name__)
def test_a_summary_turn_without_text_is_a_502(respx_mock: respx.MockRouter) -> None:
    respx_mock.post(_ANTHROPIC_MESSAGES_URL).mock(
        side_effect=[_anthropic_message(_TOOL_USE), _anthropic_stream(_TOOL_USE)]
    )

    with pytest.raises(litellm.APIError) as raised:
        litellm.responses(model=_MODEL, api_key="mock_api_key", input=_trigger_request(), tools=list(_TOOLS))
    stream: Final = litellm.responses(
        model=_MODEL, api_key="mock_api_key", input=_trigger_request(), tools=list(_TOOLS), stream=True
    )
    with pytest.raises(litellm.APIError) as raised_mid_stream:
        tuple(stream)

    assert raised.value.status_code == 502
    assert raised_mid_stream.value.status_code == 502


@pytest.mark.usefixtures(httpx_transport.__name__)
def test_a_request_without_a_trigger_still_gets_a_normal_turn(respx_mock: respx.MockRouter) -> None:
    route: Final = respx_mock.post(_ANTHROPIC_MESSAGES_URL).mock(
        return_value=_anthropic_message({"type": "text", "text": "MARKER-42"})
    )

    response: Final = litellm.responses(model=_MODEL, api_key="mock_api_key", input=list(_HISTORY), tools=list(_TOOLS))

    assert _types(response.output) == ["message"]
    assert response.output_text == "MARKER-42"
    assert SUMMARY_PROMPT not in json.dumps(_sent_body(route))


@pytest.mark.usefixtures(httpx_transport.__name__)
def test_a_compaction_trigger_for_claude_on_bedrock_returns_one_compaction_item(
    respx_mock: respx.MockRouter,
) -> None:
    route: Final = respx_mock.post(_BEDROCK_CONVERSE_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "output": {"message": {"role": "assistant", "content": [{"text": _SUMMARY_REPLY}]}},
                "stopReason": "end_turn",
                "usage": {"inputTokens": 30, "outputTokens": 12, "totalTokens": 42},
            },
        )
    )

    response: Final = litellm.responses(
        model=_BEDROCK_MODEL,
        input=_trigger_request(),
        tools=list(_TOOLS),
        aws_access_key_id="AKIACOMPACTION",
        aws_secret_access_key="compaction-secret",
        aws_region_name="us-east-1",
    )
    body: Final = _sent_body(route)
    tool_config: Final = cast(Mapping[str, object], body["toolConfig"])

    assert [(item.type, item.encrypted_content) for item in response.output] == [
        ("compaction", summary_only_compaction_content(_SUMMARY))
    ]
    assert "toolChoice" not in tool_config or tool_config["toolChoice"] != {"none": {}}
    messages: Final = cast(Sequence[Mapping[str, Sequence[Mapping[str, str]]]], body["messages"])
    assert messages[-1]["content"][-1]["text"].endswith(SUMMARY_PROMPT)
