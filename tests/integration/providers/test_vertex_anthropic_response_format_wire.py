import json
from typing import Final

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from integration._support.client import Gateway, Scenario
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue, TypeAdapter

_PROJECT: Final = "scripted-project"
_LOCATION: Final = "global"
_SONNET_5_5: Final = "claude-sonnet-5-5"
_SONNET_4_6: Final = "claude-sonnet-4-6"
_FORCED_TOOL_CHOICE_ERROR: Final = 'tool_choice: type "tool" and "any" are not supported for this model.'
_JSON_TOOL_INPUT: Final = {"name": "Paris", "country": "France"}
_RESPONSE_FORMAT: Final = {
    "type": "json_schema",
    "json_schema": {
        "name": "city",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"name": {"type": "string"}, "country": {"type": "string"}},
            "required": ["name", "country"],
            "additionalProperties": False,
        },
    },
}
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])


def _service_account_json(token_url: str) -> str:
    private_key: Final = (
        rsa.generate_private_key(public_exponent=65537, key_size=2048)
        .private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
        .decode()
    )
    return json.dumps(
        {
            "type": "service_account",
            "project_id": _PROJECT,
            "private_key_id": "scripted",
            "private_key": private_key,
            "client_email": f"scripted@{_PROJECT}.iam.gserviceaccount.com",
            "client_id": "0",
            "auth_uri": f"{token_url}/_oauth/authorize",
            "token_uri": f"{token_url}/_oauth/token",
        }
    )


def _model_path(model: str) -> str:
    return (
        f"/v1/projects/{_PROJECT}/locations/{_LOCATION}"
        f"/publishers/anthropic/models/{model}:rawPredict"
    )


def _anthropic_tool_use_message(model: str) -> dict[str, JsonValue]:
    return {
        "id": "msg_scripted",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [
            {"type": "tool_use", "id": "toolu_1", "name": "json_tool_call", "input": _JSON_TOOL_INPUT}
        ],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 11, "output_tokens": 7},
    }


def _model(gateway: Gateway, scenario: Scenario, wire_url: str, backend: str) -> str:
    return scenario.model(
        model=f"vertex_ai/{backend}",
        api_base=wire_url,
        api_key=None,
        vertex_project=_PROJECT,
        vertex_location=_LOCATION,
        vertex_credentials=_service_account_json(gateway.upstream_url.rstrip("/")),
    )


def _request(gateway: Gateway, model: str) -> tuple[int, dict[str, JsonValue], str]:
    response: Final = gateway.client.post(
        "/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": "Give me Paris as JSON"}],
            "response_format": _RESPONSE_FORMAT,
        },
        headers={"Authorization": f"Bearer {gateway.key}"},
        timeout=30,
    )
    return response.status_code, response.json(), response.text


def test_vertex_sonnet_5_5_response_format_sends_json_tool_without_forced_tool_choice(
    gateway: Gateway,
) -> None:
    def respond(request: Request) -> Reply:
        assert request.target == _model_path(_SONNET_5_5)
        body: Final = _JSON_OBJECT.validate_json(request.body)
        tool_choice: Final = body.get("tool_choice")
        if isinstance(tool_choice, dict) and tool_choice.get("type") in ("tool", "any"):
            # Observed live from Vertex claude-sonnet-5-5 on 2026-09-29, LIT-8983
            return Reply(
                status=400,
                body=json.dumps(
                    {
                        "type": "error",
                        "error": {"type": "invalid_request_error", "message": _FORCED_TOOL_CHOICE_ERROR},
                    }
                ).encode(),
            )
        return Reply(body=json.dumps(_anthropic_tool_use_message(_SONNET_5_5)).encode())

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _model(gateway, scenario, wire.url, _SONNET_5_5)
        status, payload, text = _request(gateway, model)
        assert status == 200, text
        assert json.loads(str(payload["choices"][0]["message"]["content"])) == _JSON_TOOL_INPUT
        received: Final = wire.drain()
        assert len(received) == 1
        upstream_body: Final = _JSON_OBJECT.validate_json(received[0].body)
        assert [tool["name"] for tool in upstream_body["tools"]] == ["json_tool_call"]
        assert "tool_choice" not in upstream_body


def test_vertex_sonnet_4_6_response_format_still_forces_json_tool_call(gateway: Gateway) -> None:
    def respond(request: Request) -> Reply:
        assert request.target == _model_path(_SONNET_4_6)
        return Reply(body=json.dumps(_anthropic_tool_use_message(_SONNET_4_6)).encode())

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _model(gateway, scenario, wire.url, _SONNET_4_6)
        status, payload, text = _request(gateway, model)
        assert status == 200, text
        assert json.loads(str(payload["choices"][0]["message"]["content"])) == _JSON_TOOL_INPUT
        received: Final = wire.drain()
        assert len(received) == 1
        upstream_body: Final = _JSON_OBJECT.validate_json(received[0].body)
        assert upstream_body["tool_choice"] == {"type": "tool", "name": "json_tool_call"}
        assert [tool["name"] for tool in upstream_body["tools"]] == ["json_tool_call"]
