import json
import uuid
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue, TypeAdapter

_BACKEND: Final = "grok-4.6-web-search-unmapped"
_API_KEY: Final = "synthetic-xai-key"
_SYSTEM_PROMPT: Final = "Answer in one short sentence and cite the source."
_ALLOWED_DOMAINS: Final = ("weather.example.com", "news.example.org")
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])


def _responses_reply(identity: str, text: str) -> bytes:
    return json.dumps(
        {
            "id": identity,
            "object": "response",
            "created_at": 1,
            "status": "completed",
            "model": _BACKEND,
            "output": [
                {
                    "type": "message",
                    "id": f"msg-{identity}",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": text, "annotations": []}],
                }
            ],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [{"type": "web_search"}],
            "usage": {"input_tokens": 23, "output_tokens": 41, "total_tokens": 64},
        }
    ).encode()


@pytest.mark.covers("other.provider_wire.xai.chat_web_search_reaches_responses_with_instructions_and_filters")
def test_xai_chat_web_search_is_sent_to_responses_with_instructions_and_nested_filters(gateway: Gateway) -> None:
    identity: Final = f"xai-web-search-{uuid.uuid4().hex}"
    user_prompt: Final = f"What is the weather in Paris today? Request {identity}."

    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/v1/responses", request.target
        assert request.headers["authorization"] == f"Bearer {_API_KEY}"
        body: Final = _JSON_OBJECT.validate_json(request.body)
        assert body["model"] == _BACKEND
        assert body["instructions"] == _SYSTEM_PROMPT
        assert body["input"] == [
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": user_prompt}]}
        ]
        assert body["tools"] == [{"type": "web_search", "filters": {"allowed_domains": list(_ALLOWED_DOMAINS)}}]
        assert "web_search_options" not in body
        return Reply(body=_responses_reply(identity, "Sunny, 21C."))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"xai/{_BACKEND}", api_base=f"{wire.url}/v1", api_key=_API_KEY)
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "web_search_options": {"filters": {"allowed_domains": list(_ALLOWED_DOMAINS)}},
            },
        )
        assert response.status_code == 200, response.text
        payload: Final = _JSON_OBJECT.validate_json(response.content)
        choices: Final = payload["choices"]
        assert isinstance(choices, list) and len(choices) == 1, response.text
        choice: Final = choices[0]
        assert isinstance(choice, dict), response.text
        message: Final = choice["message"]
        assert isinstance(message, dict), response.text
        assert (message["role"], message["content"]) == ("assistant", "Sunny, 21C."), response.text
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", "/v1/responses")]
