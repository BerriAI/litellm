import json
import uuid
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue, TypeAdapter

_BACKEND: Final = "deepseek-v4-pro"
_API_KEY: Final = "synthetic-tencent-key"
_PROMPT: Final = "What is 17 + 26? Answer with just the number."
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_REASONING_REQUESTS: Final[tuple[tuple[str, dict[str, JsonValue], dict[str, JsonValue]], ...]] = (
    ("thinking_enabled", {"thinking": {"type": "enabled"}}, {"type": "enabled"}),
    ("reasoning_effort_none", {"reasoning_effort": "none"}, {"type": "disabled"}),
)


def _completion(identity: str) -> bytes:
    return json.dumps(
        {
            "id": identity,
            "object": "chat.completion",
            "created": 1,
            "model": _BACKEND,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "43", "reasoning_content": "17 plus 26 is 43."},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 23, "completion_tokens": 41, "total_tokens": 64},
        }
    ).encode()


@pytest.mark.covers("other.provider_wire.tencent.thinking_reaches_provider_in_request_body")
@pytest.mark.parametrize(
    ("reasoning_params", "expected_thinking"),
    tuple(case[1:] for case in _REASONING_REQUESTS),
    ids=tuple(case[0] for case in _REASONING_REQUESTS),
)
def test_tencent_thinking_is_sent_in_provider_body_instead_of_failing_the_request(
    gateway: Gateway, reasoning_params: dict[str, JsonValue], expected_thinking: dict[str, JsonValue]
) -> None:
    identity: Final = f"tencent-thinking-{uuid.uuid4().hex}"

    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/chat/completions"
        assert request.headers["authorization"] == f"Bearer {_API_KEY}"
        assert request.headers["content-type"] == "application/json"
        assert _JSON_OBJECT.validate_json(request.body) == {
            "model": _BACKEND,
            "messages": [{"role": "user", "content": _PROMPT}],
            "thinking": expected_thinking,
        }
        return Reply(body=_completion(identity))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"tencent/{_BACKEND}", api_base=wire.url, api_key=_API_KEY)
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": _PROMPT}], **reasoning_params},
        )
        assert response.status_code == 200, response.text
        payload: Final = _JSON_OBJECT.validate_json(response.content)
        assert payload["id"] == identity
        assert payload["choices"] == [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "43",
                    "reasoning_content": "17 plus 26 is 43.",
                    "provider_specific_fields": {"refusal": None},
                },
                "provider_specific_fields": {},
            }
        ]
        assert payload["usage"] == {"prompt_tokens": 23, "completion_tokens": 41, "total_tokens": 64}
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", "/chat/completions")]
