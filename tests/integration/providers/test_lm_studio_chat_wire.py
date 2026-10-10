from __future__ import annotations

import json
from typing import Final

from integration._support.client import JSON_OBJECT, Gateway, list_value, object_value
from integration._support.wire import Reply, Request, wire_server

_BACKEND: Final = "typhoon2-quen2.5-7b-instruct"
_API_KEY: Final = "synthetic-lm-studio-key"
_PROMPT: Final = "What's the weather like in San Francisco?"
_ANSWER: Final = "Foggy and 14 degrees."


def test_lm_studio_completion(gateway: Gateway) -> None:
    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/v1/chat/completions"
        assert request.headers["authorization"] == f"Bearer {_API_KEY}"
        body: Final = JSON_OBJECT.validate_json(request.body)
        assert body["model"] == _BACKEND
        assert body["messages"] == [{"role": "user", "content": _PROMPT}]
        return Reply(
            body=json.dumps(
                {
                    "id": "chatcmpl-lm-studio",
                    "object": "chat.completion",
                    "created": 1,
                    "model": _BACKEND,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": _ANSWER},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 12,
                        "completion_tokens": 6,
                        "total_tokens": 18,
                    },
                }
            ).encode()
        )

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"lm_studio/{_BACKEND}", api_base=f"{wire.url}/v1", api_key=_API_KEY)
        response: Final = gateway.chat(model, text=_PROMPT)
        received: Final = wire.drain()

    assert len(received) == 1
    message: Final = object_value(object_value(list_value(response["choices"])[0])["message"])
    assert message["content"] == _ANSWER
    assert response["usage"] == {
        "prompt_tokens": 12,
        "completion_tokens": 6,
        "total_tokens": 18,
    }
