import json
from typing import Final

import pytest
from integration._support.client import Gateway, eventually  # pyright: ignore[reportUnknownVariableType]  # integration fixture module
from integration._support.database import read_rows
from integration._support.vertex import service_account_json
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue, TypeAdapter

_GEMINI_MODEL: Final = "gemini/gemini-3-flash-preview"
_GEMINI_KEY: Final = "synthetic-gemini-interactions-bridge-key"
_VERTEX_PROJECT: Final = "vertex-bridge-proj"
_JSON_OBJECT: Final[TypeAdapter[dict[str, JsonValue]]] = TypeAdapter(dict[str, JsonValue])


def _spend_rows_for_request(request_id: str) -> list[dict[str, JsonValue]]:  # mutable-ok: read_rows return contract
    return read_rows('SELECT spend FROM "LiteLLM_SpendLogs" WHERE request_id=%s', (request_id,))


def _single_row(rows: list[dict[str, JsonValue]]) -> bool:  # mutable-ok: predicate over read_rows list
    return len(rows) == 1


def test_gemini_chat_completions_pinned_to_interactions_bills_thought_tokens_on_spend_row(gateway: Gateway) -> None:
    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/v1beta/interactions"
        assert request.headers["x-goog-api-key"] == _GEMINI_KEY
        assert request.headers["api-revision"] == "2026-05-20"
        assert json.loads(request.body) == {
            "model": "gemini-3-flash-preview",
            "input": [{"type": "user_input", "content": [{"type": "text", "text": "solve puzzle"}]}],
            "store": False,
        }
        return Reply(
            body=json.dumps(
                {
                    "id": "v1_bridge_wire_spend_1",
                    "object": "interaction",
                    "status": "completed",
                    "steps": [
                        {
                            "type": "thought",
                            "signature": "sig_wire_1",
                            "summary": [{"type": "text", "text": "working it out"}],
                        },
                        {
                            "type": "model_output",
                            "content": [{"type": "text", "text": "42"}],
                        },
                    ],
                    "usage": {
                        "total_input_tokens": 10,
                        "total_output_tokens": 5,
                        "total_thought_tokens": 15,
                        "total_tokens": 30,
                    },
                }
            ).encode()
        )

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=_GEMINI_MODEL,
            api_base=wire.url,
            api_key=_GEMINI_KEY,
            input_cost_per_token=0.001,
            output_cost_per_token=0.002,
            model_info={"mode": "interactions"},
        )
        key: Final = scenario.key(models=[model])
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": "solve puzzle"}],
                "store": False,
            },
            key=key,
        )
        assert response.status_code == 200, response.text
        payload: Final = _JSON_OBJECT.validate_json(response.content)
        req_id: Final = str(payload["id"])
        assert req_id == "v1_bridge_wire_spend_1", response.text
        expected_cost: Final = 10 * 0.001 + (5 + 15) * 0.002
        assert float(response.headers["x-litellm-response-cost"]) == pytest.approx(expected_cost), response.headers  # pyright: ignore[reportUnknownMemberType]  # pytest.approx untyped overload
        rows: Final[list[dict[str, JsonValue]]] = eventually(  # mutable-ok: read_rows return contract
            lambda: _spend_rows_for_request(req_id),
            _single_row,
            seconds=70,
        )
        spend_val: Final = rows[0]["spend"]
        assert isinstance(spend_val, (int, float, str))
        assert float(spend_val) == pytest.approx(expected_cost), rows  # pyright: ignore[reportUnknownMemberType]  # pytest.approx untyped overload


def test_vertex_chat_completions_pinned_to_interactions_authenticates_and_hits_global_v1beta1(gateway: Gateway) -> None:
    def respond(request: Request) -> Reply:
        if request.target == "/_oauth/token":
            return Reply(body=json.dumps({"access_token": "scripted-vertex-token", "expires_in": 3600}).encode())
        assert request.method == "POST"
        assert request.target == f"/v1beta1/projects/{_VERTEX_PROJECT}/locations/global/interactions"
        assert request.headers["authorization"] == "Bearer scripted-vertex-token"
        assert json.loads(request.body) == {
            "model": "gemini-3-flash-preview",
            "input": [{"type": "user_input", "content": [{"type": "text", "text": "hello vertex"}]}],
        }
        return Reply(
            body=json.dumps(
                {
                    "id": "v1_vertex_bridge_wire_1",
                    "object": "interaction",
                    "status": "completed",
                    "steps": [{"type": "model_output", "content": [{"type": "text", "text": "VERTEX_OK"}]}],
                    "usage": {"total_input_tokens": 8, "total_output_tokens": 4, "total_tokens": 12},
                }
            ).encode()
        )

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="vertex_ai/gemini-3-flash-preview",
            api_base=wire.url,
            vertex_project=_VERTEX_PROJECT,
            vertex_credentials=service_account_json(_VERTEX_PROJECT, wire.url),
            model_info={"mode": "interactions"},
        )
        key: Final = scenario.key(models=[model])
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": "hello vertex"}]},
            key=key,
        )
        assert response.status_code == 200, response.text
        payload: Final = _JSON_OBJECT.validate_json(response.content)
        assert payload["id"] == "v1_vertex_bridge_wire_1", response.text
