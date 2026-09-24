import json
import uuid
from typing import Final

import pytest
from integration._support.client import Gateway, eventually, object_value
from integration._support.database import read_rows
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue, TypeAdapter

_MODEL: Final = "accounts/fireworks/models/kimi-k3"
_API_KEY: Final = "synthetic-fireworks-key"
_PROMPT: Final = "keep this conversation on one replica"
_SESSION_ID: Final = "conversation-affinity-6220"
_CACHED_TOKENS: Final = 7
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])


def _cached_reply(request: Request, identity: str) -> Reply:
    assert request.method == "POST"
    assert request.target == "/chat/completions"
    assert request.headers["authorization"] == f"Bearer {_API_KEY}"
    body: Final = _JSON_OBJECT.validate_json(request.body)
    assert body["model"] == _MODEL, body
    return Reply(
        body=json.dumps(
            {
                "id": identity,
                "object": "chat.completion",
                "created": 1,
                "model": _MODEL,
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "pinned"}, "finish_reason": "stop"}
                ],
                "usage": {
                    "prompt_tokens": 12,
                    "completion_tokens": 1,
                    "total_tokens": 13,
                    "prompt_tokens_details": {"cached_tokens": _CACHED_TOKENS},
                },
            }
        ).encode()
    )


@pytest.mark.covers("other.provider_wire.fireworks_ai.session_id_sent_as_affinity_header_and_cached_tokens_logged")
def test_fireworks_session_id_sends_affinity_header_and_logs_cache_read_tokens(gateway: Gateway) -> None:
    identity: Final = f"fw-session-affinity-{uuid.uuid4().hex}"
    with wire_server(lambda request: _cached_reply(request, identity)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"fireworks_ai/{_MODEL}", api_base=wire.url, api_key=_API_KEY)
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": _PROMPT}]},
            headers={"x-litellm-session-id": _SESSION_ID},
        )
        assert response.status_code == 200, response.text
        payload: Final = _JSON_OBJECT.validate_json(response.content)
        assert payload["id"] == identity, response.text
        requests: Final = wire.drain()
        assert [(request.method, request.target) for request in requests] == [("POST", "/chat/completions")]
        assert requests[0].headers.get("x-session-affinity") == _SESSION_ID, requests[0].headers
        rows: Final = eventually(
            lambda: read_rows('SELECT metadata FROM "LiteLLM_SpendLogs" WHERE request_id=%s', (identity,)),
            lambda values: len(values) == 1,
            seconds=70,
        )
        usage_values: Final = object_value(object_value(rows[0]["metadata"])["additional_usage_values"])
        assert usage_values.get("cache_read_input_tokens") == _CACHED_TOKENS, rows[0]["metadata"]
