import json
from typing import Final

import respx
from httpx import Response

import litellm

_COMPLETION: Final = {
    "id": "cmpl-1",
    "object": "text_completion",
    "created": 1,
    "model": "glm",
    "choices": [{"index": 0, "text": '{"capital": "Paris"}', "finish_reason": "stop", "logprobs": None}],
    "usage": {"prompt_tokens": 5, "completion_tokens": 6, "total_tokens": 11},
}


@respx.mock
def test_text_completion_sends_chat_only_params_in_the_request_body() -> None:
    route: Final = respx.post("http://vllm:8000/v1/completions").mock(return_value=Response(200, json=_COMPLETION))
    tools: Final = [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}]

    response: Final = litellm.text_completion(
        model="hosted_vllm/glm",
        api_base="http://vllm:8000/v1",
        prompt="The capital of France is",
        response_format={"type": "json_object"},
        tools=tools,
        tool_choice="auto",
    )

    body: Final = json.loads(route.calls.last.request.content)
    assert body["response_format"] == {"type": "json_object"}
    assert body["tools"] == tools
    assert body["tool_choice"] == "auto"
    assert response.choices[0].text == '{"capital": "Paris"}'
