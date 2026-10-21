import json
from typing import Final

from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue, TypeAdapter

_MODEL: Final = "zai-org/GLM-5.3"
_API_KEY: Final = "synthetic-friendliai-key"
_PROMPT: Final = "What is the weather in Paris? Use the tool."
_REASONING: Final = "The user wants the weather in Paris, so call get_weather."
_TOOL_RESULT: Final = "18C and sunny"
_ANSWER: Final = "It is 18C and sunny in Paris."
_LITELLM_ONLY_FIELDS: Final = frozenset({"thinking_blocks", "reasoning_items", "provider_specific_fields"})
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_MESSAGES: Final = TypeAdapter(tuple[dict[str, JsonValue], ...])
_CALL: Final[JsonValue] = {
    "id": "call_paris",
    "type": "function",
    "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
}
_PARAMETERS: Final[JsonValue] = {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"],
}


def _respond(request: Request) -> Reply:
    assert (request.method, request.target) == ("POST", "/chat/completions")
    assert request.headers["authorization"] == f"Bearer {_API_KEY}"
    return Reply(
        body=json.dumps(
            {
                "id": "friendliai-replay",
                "object": "chat.completion",
                "created": 1,
                "model": _MODEL,
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": _ANSWER}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 40, "completion_tokens": 9, "total_tokens": 49},
            }
        ).encode()
    )


def _sent_messages(requests: tuple[Request, ...]) -> tuple[dict[str, JsonValue], ...]:
    assert len(requests) == 1
    body: Final = _JSON_OBJECT.validate_json(requests[0].body)
    assert body["model"] == _MODEL
    return _MESSAGES.validate_python(body["messages"])


def test_chat_replay_of_returned_assistant_message_reaches_friendliai_without_litellm_fields(gateway: Gateway) -> None:
    with wire_server(_respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"friendliai/{_MODEL}", api_base=wire.url, api_key=_API_KEY)
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [
                    {"role": "user", "content": _PROMPT},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{**_CALL, "provider_specific_fields": {"thought_signature": "sig"}}],
                        "reasoning_content": _REASONING,
                        "thinking_blocks": [{"type": "thinking", "thinking": _REASONING, "signature": "sig"}],
                        "provider_specific_fields": {"reasoning": _REASONING},
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_paris",
                        "content": _TOOL_RESULT,
                        "provider_specific_fields": {},
                    },
                ],
                "tools": [{"type": "function", "function": {"name": "get_weather", "parameters": _PARAMETERS}}],
            },
        )
        assert response.status_code == 200, response.text
        assert _sent_messages(wire.drain()) == (
            {"role": "user", "content": _PROMPT},
            {"role": "assistant", "tool_calls": [_CALL], "reasoning_content": _REASONING},
            {"role": "tool", "tool_call_id": "call_paris", "content": _TOOL_RESULT},
        )


def test_messages_replay_of_thinking_and_tool_use_reaches_friendliai_without_litellm_fields(gateway: Gateway) -> None:
    with wire_server(_respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"friendliai/{_MODEL}", api_base=wire.url, api_key=_API_KEY)
        response: Final = gateway.request(
            "POST",
            "/v1/messages",
            {
                "model": model,
                "max_tokens": 256,
                "messages": [
                    {"role": "user", "content": _PROMPT},
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "thinking", "thinking": _REASONING, "signature": "sig"},
                            {"type": "tool_use", "id": "call_paris", "name": "get_weather", "input": {"city": "Paris"}},
                        ],
                    },
                    {
                        "role": "user",
                        "content": [{"type": "tool_result", "tool_use_id": "call_paris", "content": _TOOL_RESULT}],
                    },
                ],
                "tools": [{"name": "get_weather", "input_schema": _PARAMETERS}],
            },
        )
        assert response.status_code == 200, response.text
        assert _JSON_OBJECT.validate_json(response.content)["content"] == [{"type": "text", "text": _ANSWER}]
        messages: Final = _sent_messages(wire.drain())
        assert [sorted(_LITELLM_ONLY_FIELDS & message.keys()) for message in messages] == [[]] * len(messages)
        assistant: Final = next(message for message in messages if message["role"] == "assistant")
        assert assistant["reasoning_content"] == _REASONING
