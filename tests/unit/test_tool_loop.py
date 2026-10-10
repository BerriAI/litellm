import json
from typing import Final

import httpx
import pytest
import respx

import litellm
from litellm.tool_loop import ToolLoopMaxRoundsExceeded
from litellm.types.llms.openai import ChatCompletionToolMessage
from litellm.types.utils import ChatCompletionMessageToolCall

OPENAI_CHAT_COMPLETIONS_URL: Final = "https://api.openai.com/v1/chat/completions"
WEATHER_TOOLS: Final = (
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    },
)


def _openai_response(content: str | None, tool_calls: list | None = None) -> dict:
    return {
        "id": "chatcmpl-tool-loop",
        "object": "chat.completion",
        "created": 1739462947,
        "model": "gpt-5-mini",
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls" if tool_calls else "stop",
                "message": {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": tool_calls,
                },
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def _tool_call(call_id: str, name: str, arguments: dict) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def _tool_result(tool_call: ChatCompletionMessageToolCall) -> ChatCompletionToolMessage:
    return ChatCompletionToolMessage(role="tool", content='{"temp": "72F"}', tool_call_id=tool_call.id or "")


def _request_bodies(respx_mock: respx.MockRouter) -> list[dict]:
    return [json.loads(call.request.content) for call in respx_mock.calls]


def test_final_answer_without_tool_calls_returns_content(respx_mock: respx.MockRouter) -> None:
    route: Final = respx_mock.post(OPENAI_CHAT_COMPLETIONS_URL).mock(
        return_value=httpx.Response(200, json=_openai_response("done"))
    )
    executor_called: Final = []

    def executor(tc: ChatCompletionMessageToolCall) -> ChatCompletionToolMessage:
        executor_called.append(tc)
        return _tool_result(tc)

    answer: Final = litellm.run_tool_loop(
        model="openai/gpt-5-mini",
        messages=[{"role": "user", "content": "hi"}],
        tools=WEATHER_TOOLS,
        execute_tool=executor,
        api_key="sk-test",
    )

    assert answer == "done"
    assert executor_called == []
    assert route.call_count == 1


def test_two_rounds_appends_assistant_and_tool_messages_in_order(respx_mock: respx.MockRouter) -> None:
    tool_calls: Final = [
        _tool_call("call_1", "get_weather", {"city": "Paris"}),
        _tool_call("call_2", "get_weather", {"city": "Tokyo"}),
    ]
    route: Final = respx_mock.post(OPENAI_CHAT_COMPLETIONS_URL).mock(
        side_effect=[
            httpx.Response(200, json=_openai_response(None, tool_calls)),
            httpx.Response(200, json=_openai_response("Paris 72F, Tokyo 60F")),
        ]
    )
    executed: Final = []

    def executor(tc: ChatCompletionMessageToolCall) -> ChatCompletionToolMessage:
        executed.append(tc)
        return _tool_result(tc)

    messages: Final = [{"role": "user", "content": "weather in Paris and Tokyo?"}]
    messages_snapshot: Final = [dict(message) for message in messages]

    answer: Final = litellm.run_tool_loop(
        model="openai/gpt-5-mini",
        messages=messages,
        tools=WEATHER_TOOLS,
        execute_tool=executor,
        api_key="sk-test",
    )

    assert answer == "Paris 72F, Tokyo 60F"
    assert route.call_count == 2
    assert [tc.id for tc in executed] == ["call_1", "call_2"]
    assert [tc.function.name for tc in executed] == ["get_weather", "get_weather"]
    assert [tc.function.arguments for tc in executed] == [
        '{"city": "Paris"}',
        '{"city": "Tokyo"}',
    ]

    second_body: Final = _request_bodies(respx_mock)[1]
    assert second_body["messages"] == [
        {"role": "user", "content": "weather in Paris and Tokyo?"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                },
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "Tokyo"}'},
                },
            ],
        },
        {"role": "tool", "content": '{"temp": "72F"}', "tool_call_id": "call_1"},
        {"role": "tool", "content": '{"temp": "72F"}', "tool_call_id": "call_2"},
    ]

    assert len(messages) == len(messages_snapshot)
    assert messages == messages_snapshot


def test_response_format_and_tools_forwarded_every_round(respx_mock: respx.MockRouter) -> None:
    respx_mock.post(OPENAI_CHAT_COMPLETIONS_URL).mock(
        side_effect=[
            httpx.Response(200, json=_openai_response(None, [_tool_call("call_1", "get_weather", {"city": "Paris"})])),
            httpx.Response(200, json=_openai_response('{"summary": "sunny"}')),
        ]
    )
    response_format: Final = {
        "type": "json_schema",
        "json_schema": {
            "name": "weather_report",
            "schema": {
                "type": "object",
                "properties": {"summary": {"type": "string"}},
                "required": ["summary"],
            },
        },
    }

    litellm.run_tool_loop(
        model="openai/gpt-5-mini",
        messages=[{"role": "user", "content": "weather?"}],
        tools=WEATHER_TOOLS,
        execute_tool=_tool_result,
        response_format=response_format,
        api_key="sk-test",
    )

    bodies: Final = _request_bodies(respx_mock)
    assert len(bodies) == 2
    for body in bodies:
        assert body["response_format"] == response_format
        assert body["tools"] == list(WEATHER_TOOLS)


def test_max_rounds_exceeded_raises_without_executing_last_round(respx_mock: respx.MockRouter) -> None:
    tool_call: Final = _tool_call("call_1", "get_weather", {"city": "Paris"})
    route: Final = respx_mock.post(OPENAI_CHAT_COMPLETIONS_URL).mock(
        return_value=httpx.Response(200, json=_openai_response(None, [tool_call]))
    )
    executed: Final = []

    def executor(tc: ChatCompletionMessageToolCall) -> ChatCompletionToolMessage:
        executed.append(tc)
        return _tool_result(tc)

    with pytest.raises(ToolLoopMaxRoundsExceeded) as exc_info:
        litellm.run_tool_loop(
            model="openai/gpt-5-mini",
            messages=[{"role": "user", "content": "weather?"}],
            tools=WEATHER_TOOLS,
            execute_tool=executor,
            max_rounds=2,
            api_key="sk-test",
        )

    assert exc_info.value.max_rounds == 2
    assert route.call_count == 2
    assert [tc.id for tc in executed] == ["call_1"]


def test_max_rounds_below_one_rejected_before_any_request(respx_mock: respx.MockRouter) -> None:
    route: Final = respx_mock.post(OPENAI_CHAT_COMPLETIONS_URL).mock(
        return_value=httpx.Response(200, json=_openai_response("done"))
    )

    with pytest.raises(ValueError, match="max_rounds must be >= 1"):
        litellm.run_tool_loop(
            model="openai/gpt-5-mini",
            messages=[{"role": "user", "content": "hi"}],
            tools=WEATHER_TOOLS,
            execute_tool=_tool_result,
            max_rounds=0,
            api_key="sk-test",
        )

    assert route.call_count == 0


def test_stream_rejected_before_any_request(respx_mock: respx.MockRouter) -> None:
    route: Final = respx_mock.post(OPENAI_CHAT_COMPLETIONS_URL).mock(
        return_value=httpx.Response(200, json=_openai_response("done"))
    )

    with pytest.raises(ValueError, match="stream=True is not supported"):
        litellm.run_tool_loop(
            model="openai/gpt-5-mini",
            messages=[{"role": "user", "content": "hi"}],
            tools=WEATHER_TOOLS,
            execute_tool=_tool_result,
            stream=True,
            api_key="sk-test",
        )

    assert route.call_count == 0


def test_custom_tool_call_raises_type_error_without_executing(respx_mock: respx.MockRouter) -> None:
    custom_response: Final = _openai_response(
        None,
        [{"id": "call_custom", "type": "custom", "custom": {"name": "apply_patch", "input": "*** patch"}}],
    )
    route: Final = respx_mock.post(OPENAI_CHAT_COMPLETIONS_URL).mock(
        return_value=httpx.Response(200, json=custom_response)
    )
    executor_called: Final = []

    def executor(tc: ChatCompletionMessageToolCall) -> ChatCompletionToolMessage:
        executor_called.append(tc)
        return _tool_result(tc)

    with pytest.raises(TypeError, match="custom tool call call_custom"):
        litellm.run_tool_loop(
            model="openai/gpt-5-mini",
            messages=[{"role": "user", "content": "hi"}],
            tools=WEATHER_TOOLS,
            execute_tool=executor,
            api_key="sk-test",
        )

    assert executor_called == []
    assert route.call_count == 1


def _responses_payload(response_id: str, output: list) -> dict:
    return {
        "id": response_id,
        "object": "response",
        "created_at": 1734366691,
        "status": "completed",
        "model": "gpt-5.5",
        "output": output,
        "parallel_tool_calls": True,
        "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "metadata": None,
        "temperature": None,
        "tool_choice": "auto",
        "tools": [],
        "top_p": None,
        "max_output_tokens": None,
        "previous_response_id": None,
        "reasoning": None,
        "truncation": None,
        "user": None,
    }


def test_responses_bridge_replays_reasoning_items_across_rounds(respx_mock: respx.MockRouter) -> None:
    round_one: Final = _responses_payload(
        "resp_1",
        [
            {"type": "reasoning", "id": "rs_abc123", "summary": [], "encrypted_content": "enc_xyz"},
            {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_1",
                "name": "get_weather",
                "arguments": '{"city": "Paris"}',
                "status": "completed",
            },
        ],
    )
    round_two: Final = _responses_payload(
        "resp_2",
        [
            {
                "type": "message",
                "id": "msg_1",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Paris is 72F", "annotations": []}],
            }
        ],
    )
    route: Final = respx_mock.post("https://api.openai.com/v1/responses").mock(
        side_effect=[httpx.Response(200, json=round_one), httpx.Response(200, json=round_two)]
    )
    executed: Final = []

    def executor(tc: ChatCompletionMessageToolCall) -> ChatCompletionToolMessage:
        executed.append(tc)
        return _tool_result(tc)

    answer: Final = litellm.run_tool_loop(
        model="openai/responses/gpt-5.5",
        messages=[{"role": "user", "content": "weather in Paris?"}],
        tools=WEATHER_TOOLS,
        execute_tool=executor,
        api_key="sk-test",
    )

    assert answer == "Paris is 72F"
    assert route.call_count == 2
    assert [tc.id for tc in executed] == ["fc_1"]

    second_input: Final = _request_bodies(respx_mock)[1]["input"]
    item_types: Final = [item.get("type") for item in second_input]
    reasoning_index: Final = next(i for i, item in enumerate(second_input) if item.get("type") == "reasoning")
    function_call_index: Final = next(
        i for i, item in enumerate(second_input) if item.get("type") == "function_call"
    )
    reasoning_item: Final = second_input[reasoning_index]
    assert reasoning_item["id"] == "rs_abc123"
    assert reasoning_item["encrypted_content"] == "enc_xyz"
    assert reasoning_index < function_call_index, f"reasoning item must precede function_call: {item_types}"


async def test_arun_tool_loop_two_rounds(respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "module_level_aclient", AsyncHTTPHandler())
    tool_calls: Final = [
        _tool_call("call_1", "get_weather", {"city": "Paris"}),
        _tool_call("call_2", "get_weather", {"city": "Tokyo"}),
    ]
    route: Final = respx_mock.post(OPENAI_CHAT_COMPLETIONS_URL).mock(
        side_effect=[
            httpx.Response(200, json=_openai_response(None, tool_calls)),
            httpx.Response(200, json=_openai_response("Paris 72F, Tokyo 60F")),
        ]
    )
    executed: Final = []

    async def executor(tc: ChatCompletionMessageToolCall) -> ChatCompletionToolMessage:
        executed.append(tc)
        return _tool_result(tc)

    messages: Final = [{"role": "user", "content": "weather in Paris and Tokyo?"}]
    messages_snapshot: Final = [dict(message) for message in messages]

    answer: Final = await litellm.arun_tool_loop(
        model="openai/gpt-5-mini",
        messages=messages,
        tools=WEATHER_TOOLS,
        execute_tool=executor,
        api_key="sk-test",
    )

    assert answer == "Paris 72F, Tokyo 60F"
    assert route.call_count == 2
    assert [tc.id for tc in executed] == ["call_1", "call_2"]

    second_body: Final = _request_bodies(respx_mock)[1]
    assert second_body["messages"] == [
        {"role": "user", "content": "weather in Paris and Tokyo?"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                },
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "Tokyo"}'},
                },
            ],
        },
        {"role": "tool", "content": '{"temp": "72F"}', "tool_call_id": "call_1"},
        {"role": "tool", "content": '{"temp": "72F"}', "tool_call_id": "call_2"},
    ]
    assert messages == messages_snapshot
