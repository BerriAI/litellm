import json
from typing import Final

import httpx
import respx

from litellm import text_completion

FIREWORKS_COMPLETIONS_URL: Final = "https://api.fireworks.ai/inference/v1/completions"
FIREWORKS_MODEL: Final = "accounts/fireworks/models/llama-v3p1-8b-instruct"


def _fireworks_completion_response() -> dict[str, object]:
    return {
        "id": "cmpl-fireworks-migration",
        "object": "text_completion",
        "created": 1,
        "model": FIREWORKS_MODEL,
        "choices": [
            {
                "text": "hello world",
                "index": 0,
                "logprobs": {"tokens": ["hello", " world"], "token_logprobs": [-0.1, -0.2]},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
    }


def test_fireworks_text_completion_sends_prompt_array(respx_mock: respx.MockRouter) -> None:
    route: Final = respx_mock.post(FIREWORKS_COMPLETIONS_URL).mock(
        return_value=httpx.Response(200, json=_fireworks_completion_response())
    )

    response: Final = text_completion(
        model=f"fireworks_ai/{FIREWORKS_MODEL}",
        prompt=["What's the weather in SF?", "How is Manchester?"],
        max_tokens=5,
        api_key="fireworks-test-key",
    )

    request_body: Final = json.loads(route.calls.last.request.read())
    assert request_body["prompt"] == ["What's the weather in SF?", "How is Manchester?"]
    assert [choice.text for choice in response.choices] == ["hello world"]
    assert route.calls.last.request.headers["Authorization"] == "Bearer fireworks-test-key"


def test_fireworks_text_completion_echo_returns_prompt_and_logprobs(respx_mock: respx.MockRouter) -> None:
    route: Final = respx_mock.post(FIREWORKS_COMPLETIONS_URL).mock(
        return_value=httpx.Response(200, json=_fireworks_completion_response())
    )

    response: Final = text_completion(
        model=f"fireworks_ai/{FIREWORKS_MODEL}",
        prompt="hello",
        max_tokens=1,
        stop="\n",
        logprobs=1,
        echo=True,
        api_key="fireworks-test-key",
    )

    request_body: Final = json.loads(route.calls.last.request.read())
    assert request_body["echo"] is True
    assert request_body["stop"] == "\n"
    assert response.choices[0].text == "hello world"
    assert response.choices[0].logprobs.token_logprobs == [-0.1, -0.2]
