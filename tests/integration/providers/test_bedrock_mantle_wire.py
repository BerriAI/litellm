import json
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue, TypeAdapter

_BACKEND: Final = "openai.gpt-5.6-sol"
_API_KEY: Final = "synthetic-mantle-bearer"
_PROMPT: Final = "synthetic long conversation control"
_PROMPT_TOKENS: Final = 1055489
_MODEL_MAXIMUM: Final = 1050000
_RESPONSES_PATH: Final = "/openai/v1/responses"
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_OVERFLOW_BODY: Final = json.dumps(
    {
        "error": {
            "code": "validation_error",
            "message": f"prompt tokens ({_PROMPT_TOKENS}) exceed model maximum ({_MODEL_MAXIMUM}) for {_BACKEND}",
            "type": "invalid_request_error",
        }
    }
).encode()


def _overflow_peer(request: Request) -> Reply:
    assert request.method == "POST"
    assert request.target == _RESPONSES_PATH
    assert request.headers["authorization"] == f"Bearer {_API_KEY}"
    body: Final = _JSON_OBJECT.validate_json(request.body)
    assert body["model"] == _BACKEND
    assert _PROMPT in json.dumps(body["input"]), body
    return Reply(status=400, body=_OVERFLOW_BODY)


@pytest.mark.covers("other.provider_wire.bedrock_mantle.context_overflow_is_reported_as_prompt_too_long")
def test_bedrock_mantle_context_overflow_returns_400_saying_prompt_is_too_long(gateway: Gateway) -> None:
    with wire_server(_overflow_peer) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"bedrock_mantle/{_BACKEND}", api_base=wire.url, api_key=_API_KEY)
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": _PROMPT}]},
        )
        assert response.status_code == 400, response.text
        error: Final = _JSON_OBJECT.validate_json(response.content)["error"]
        assert isinstance(error, dict), response.text
        assert error["code"] == "400", response.text
        message: Final = error["message"]
        assert isinstance(message, str), response.text
        assert f"prompt is too long: {_PROMPT_TOKENS} tokens > {_MODEL_MAXIMUM} maximum" in message, response.text
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", _RESPONSES_PATH)]
