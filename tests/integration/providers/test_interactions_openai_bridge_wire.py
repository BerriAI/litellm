import json
from datetime import UTC, datetime, timedelta
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.openai_wire import answering_model_discovery
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue, TypeAdapter

_BACKEND: Final = "gpt-interactions-bridge"
_API_KEY: Final = "synthetic-openai-key"
_REPLY_TEXT: Final = "interactions bridge control"
_CREATED_AT: Final = 86400
_WIDEST_UTC_OFFSET: Final = timedelta(hours=14)
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])


def _reply(include_usage: bool) -> bytes:
    usage: Final = {"usage": {"input_tokens": 9, "output_tokens": 4, "total_tokens": 13}} if include_usage else {}
    return json.dumps(
        {
            "id": "resp_interactions_bridge",
            "object": "response",
            "created_at": _CREATED_AT,
            "status": "completed",
            "model": _BACKEND,
            "output": [
                {
                    "type": "message",
                    "id": "msg_interactions_bridge",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": _REPLY_TEXT, "annotations": []}],
                }
            ],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            **usage,
        }
    ).encode()


def _peer(include_usage: bool):
    def respond(request: Request) -> Reply:
        assert request.method == "POST" and request.target == "/v1/responses", request.target
        assert request.headers["authorization"] == f"Bearer {_API_KEY}"
        return Reply(body=_reply(include_usage))

    return respond


@pytest.mark.parametrize("route", ("/v1beta/interactions", "/interactions"))
@pytest.mark.parametrize(
    ("include_usage", "expected_usage"),
    (
        pytest.param(True, {"total_input_tokens": 9, "total_output_tokens": 4}, id="usage-reported"),
        pytest.param(False, None, id="usage-absent"),
    ),
)
def test_interactions_with_an_openai_model_answers_in_the_interactions_shape(
    gateway: Gateway,
    route: str,
    include_usage: bool,
    expected_usage: dict[str, JsonValue] | None,
) -> None:
    with wire_server(answering_model_discovery(_peer(include_usage))) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"openai/{_BACKEND}", api_base=f"{wire.url}/v1", api_key=_API_KEY)
        response: Final = gateway.request(
            "POST",
            route,
            {"model": model, "input": "synthetic interaction", "system_instruction": "synthetic instruction"},
        )
        assert response.status_code == 200, response.text
        payload: Final = _JSON_OBJECT.validate_json(response.content)
        assert payload["object"] == "interaction" and payload["status"] == "completed", response.text
        assert str(payload["id"]).startswith("resp_"), response.text
        assert payload["model"] == model, response.text
        assert payload["outputs"] == [{"type": "text", "text": _REPLY_TEXT}], response.text
        assert payload["steps"] == [
            {"type": "model_output", "content": [{"type": "text", "text": _REPLY_TEXT}]}
        ], response.text
        assert payload["usage"] == expected_usage, response.text
        created: Final = payload["created"]
        assert isinstance(created, str) and created == payload["updated"], response.text
        local_created: Final = datetime.fromisoformat(created).replace(tzinfo=UTC)
        assert abs(local_created - datetime.fromtimestamp(_CREATED_AT, UTC)) <= _WIDEST_UTC_OFFSET, response.text
        requests: Final = tuple(request for request in wire.drain() if request.method == "POST")
        assert len(requests) == 1, requests
        outbound: Final = _JSON_OBJECT.validate_json(requests[0].body)
        assert outbound["model"] == _BACKEND and outbound["input"] == "synthetic interaction", requests[0].body
        assert outbound["instructions"] == "synthetic instruction", requests[0].body
