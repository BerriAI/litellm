import json
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.openai_wire import answering_model_discovery
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue, TypeAdapter

_BACKEND: Final = "gpt-reasoning-item"
_API_KEY: Final = "synthetic-openai-key"
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_SUMMARY: Final[list[JsonValue]] = [{"type": "summary_text", "text": "thought"}]
_FIRST_TURN: Final[dict[str, JsonValue]] = {"role": "user", "content": "first synthetic turn"}
_LAST_TURN: Final[dict[str, JsonValue]] = {"role": "user", "content": "second synthetic turn"}
_REPLY: Final = json.dumps(
    {
        "id": "resp_reasoning_item",
        "object": "response",
        "created_at": 1,
        "status": "completed",
        "model": _BACKEND,
        "output": [
            {
                "type": "message",
                "id": "msg_reasoning_item",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "reasoning item control", "annotations": []}],
            }
        ],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "usage": {"input_tokens": 9, "output_tokens": 4, "total_tokens": 13},
    }
).encode()


def _peer(request: Request) -> Reply:
    assert request.method == "POST" and request.target == "/v1/responses", request.target
    assert request.headers["authorization"] == f"Bearer {_API_KEY}"
    return Reply(body=_REPLY)


@pytest.mark.parametrize(
    ("item", "expected"),
    (
        pytest.param(
            {"type": "reasoning", "id": "rs_1", "summary": _SUMMARY},
            {"type": "reasoning", "id": "rs_1", "summary": _SUMMARY},
            id="well-formed",
        ),
        pytest.param(
            {
                "type": "reasoning",
                "id": "rs_1",
                "summary": [],
                "status": None,
                "content": None,
                "encrypted_content": None,
            },
            {"type": "reasoning", "id": "rs_1", "summary": []},
            id="null-optional-fields-dropped",
        ),
        pytest.param(
            {"type": "reasoning", "id": "rs_1", "summary": _SUMMARY, "status": "completed"},
            {"type": "reasoning", "id": "rs_1", "summary": _SUMMARY, "status": "completed"},
            id="status-kept",
        ),
        pytest.param(
            {"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "opaque-blob"},
            {"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "opaque-blob"},
            id="encrypted-content-kept",
        ),
        pytest.param(
            {"type": "reasoning", "id": "rs_1", "summary": [], "vendor_note": "kept"},
            {"type": "reasoning", "id": "rs_1", "summary": [], "vendor_note": "kept"},
            id="unknown-field-kept",
        ),
        pytest.param(
            {"type": "reasoning", "id": 7, "summary": []},
            {"type": "reasoning", "id": 7, "summary": []},
            id="id-is-a-number",
        ),
        pytest.param(
            {"type": "reasoning", "id": "rs_1", "summary": "not a list"},
            {"type": "reasoning", "id": "rs_1", "summary": "not a list"},
            id="summary-is-text",
        ),
        pytest.param(
            {"type": "reasoning", "id": "rs_1", "summary": None},
            {"type": "reasoning", "id": "rs_1", "summary": None},
            id="summary-is-null",
        ),
        pytest.param(
            {"type": "reasoning", "summary": []},
            {"type": "reasoning", "summary": []},
            id="id-absent",
        ),
        pytest.param(
            {"type": "reasoning", "id": "rs_1", "summary": [], "status": "sideways"},
            {"type": "reasoning", "id": "rs_1", "summary": [], "status": "sideways"},
            id="status-outside-the-enum",
        ),
    ),
)
def test_responses_forwards_a_reasoning_input_item_to_openai(
    gateway: Gateway,
    item: dict[str, JsonValue],
    expected: dict[str, JsonValue],
) -> None:
    with wire_server(answering_model_discovery(_peer)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"openai/{_BACKEND}", api_base=f"{wire.url}/v1", api_key=_API_KEY)
        response: Final = gateway.request(
            "POST", "/v1/responses", {"model": model, "input": [_FIRST_TURN, item, _LAST_TURN]}
        )
        assert response.status_code == 200, response.text
        payload: Final = _JSON_OBJECT.validate_json(response.content)
        assert payload["status"] == "completed", response.text
        assert payload["output"][0]["content"][0]["text"] == "reasoning item control", response.text
        requests: Final = tuple(request for request in wire.drain() if request.method == "POST")
        assert len(requests) == 1, requests
        outbound: Final = _JSON_OBJECT.validate_json(requests[0].body)["input"]
        assert isinstance(outbound, list) and len(outbound) == 3, requests[0].body
        assert outbound[1] == expected, requests[0].body
