import json
from typing import Final

import pytest
from pydantic import JsonValue, TypeAdapter

import litellm
from litellm import Router
from tests.integration._support.wire import Reply, Request, wire_server

_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_TOOL: Final = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
}
_ARGUMENTS: Final = '{"city":"Paris"}'


def _tool_reply(request: Request) -> Reply:
    body: Final = _OBJECT.validate_json(request.body)
    if "/responses" in request.target:
        return Reply(
            body=json.dumps(
                {
                    "id": "resp_weather",
                    "object": "response",
                    "created_at": 1789788253,
                    "model": "gpt-5.6",
                    "status": "completed",
                    "output": [
                        {
                            "type": "function_call",
                            "id": "fc_weather",
                            "call_id": "call_weather",
                            "name": "get_weather",
                            "arguments": _ARGUMENTS,
                            "status": "completed",
                        }
                    ],
                    "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                }
            ).encode()
        )
    return Reply(
        body=json.dumps(
            {
                "id": "chatcmpl-weather",
                "object": "chat.completion",
                "created": 1789788253,
                "model": body.get("model"),
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_weather",
                                    "type": "function",
                                    "function": {"name": "get_weather", "arguments": _ARGUMENTS},
                                }
                            ],
                        },
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            }
        ).encode()
    )


@pytest.mark.parametrize(
    ("model", "placement", "endpoint"),
    (
        ("gpt-5.4-mini", "unset", "chat/completions"),
        ("gpt-5.5", "unset", "chat/completions"),
        ("gpt-5.6", "unset", "responses"),
        ("gpt-5.4-mini", "extra-body", "responses"),
        ("gpt-5.4-mini", "none", "chat/completions"),
        ("gpt-5.4-mini", "deployment", "responses"),
    ),
)
def test_function_tools_reach_the_chat_servable_endpoint_and_keep_tool_call_shape(
    model: str, placement: str, endpoint: str
) -> None:
    with wire_server(_tool_reply) as wire:
        parameters: Final = {
            "model": f"azure/{model}",
            "api_base": wire.url,
            "api_key": "synthetic-azure-key",
            "api_version": "2025-04-01-preview",
        }
        fields: Final = (
            {"extra_body": {"reasoning_effort": "medium"}}
            if placement == "extra-body"
            else {"reasoning_effort": "none"}
            if placement == "none"
            else {}
        )
        messages: Final = [{"role": "user", "content": "What's the weather in Paris? Use the tool."}]
        if placement == "deployment":
            router: Final = Router(
                model_list=[
                    {
                        "model_name": "weather",
                        "litellm_params": {**parameters, "reasoning_effort": "medium"},
                    }
                ],
                num_retries=0,
            )
            response: Final = router.completion(model="weather", messages=messages, tools=[_TOOL])
        else:
            response_direct: Final = litellm.completion(
                **parameters,
                **fields,
                messages=messages,
                tools=[_TOOL],
                num_retries=0,
            )
        answer: Final = response if placement == "deployment" else response_direct
        (sent,) = tuple(request for request in wire.drain() if request.method == "POST")
        expected_path: Final = (
            "/openai/responses" if endpoint == "responses" else f"/openai/deployments/{model}/chat/completions"
        )
        assert sent.target.split("?")[0] == expected_path, sent.target
        assert answer.choices[0].finish_reason == "tool_calls"
        assert answer.choices[0].message.tool_calls[0].function.name == "get_weather"
        assert json.loads(answer.choices[0].message.tool_calls[0].function.arguments) == {"city": "Paris"}
        outbound: Final = _OBJECT.validate_json(sent.body)
        if endpoint == "responses":
            assert outbound["tools"] == [
                {
                    "type": "function",
                    "name": "get_weather",
                    "parameters": _TOOL["function"]["parameters"],
                    "description": None,
                    "strict": None,
                }
            ]
        else:
            assert outbound["tools"] == [_TOOL]
