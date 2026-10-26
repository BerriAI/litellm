import json
from typing import Final

import httpx
import respx

from litellm import text_completion


@respx.mock
def test_ollama_text_completion_sends_prompt() -> None:
    endpoint: Final = "http://localhost:11434/api/generate"
    route: Final = respx.post(endpoint).mock(
        return_value=httpx.Response(
            200,
            json={
                "model": "llama3.1",
                "created_at": "2024-01-01T00:00:00Z",
                "response": "Hello!",
                "done": True,
                "context": [1],
                "total_duration": 1,
                "load_duration": 1,
                "prompt_eval_count": 2,
                "eval_count": 1,
            },
        )
    )

    response: Final = text_completion(
        model="ollama/llama3.1",
        prompt="hello",
        api_base="http://localhost:11434",
    )

    request_body: Final = json.loads(route.calls.last.request.read())
    assert request_body["model"] == "llama3.1"
    assert request_body["prompt"] == "hello"
    assert response.choices[0].text == "Hello!"
