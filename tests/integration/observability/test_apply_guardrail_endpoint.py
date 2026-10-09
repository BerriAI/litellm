import json
import uuid
from contextlib import ExitStack
from pathlib import Path
from typing import Final

import httpx
import yaml
from integration._support.client import Gateway, string_value
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, wire_server
from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter

_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_BLOCK_REASON: Final = "synthetic blocked by apply endpoint"
_BLOCKED_TEXT: Final = "forbidden integration prompt"
_METADATA_TEXT: Final = "metadata forwarding prompt"
_PROVIDER_RESPONSE: Final = {
    "id": "chatcmpl-apply-guardrail",
    "object": "chat.completion",
    "created": 1,
    "model": "gpt-4o-mini",
    "choices": [
        {"index": 0, "message": {"role": "assistant", "content": "provider response"}, "finish_reason": "stop"}
    ],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}


class _CreatedGuardrail(BaseModel):
    guardrail_id: str
    guardrail_name: str


class _ApplyResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    response_text: str


class _ApplyErrorBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str
    type: str
    param: str | None
    code: int


class _ApplyError(BaseModel):
    model_config = ConfigDict(extra="forbid")

    error: _ApplyErrorBody


def _delete_guardrail(gateway: Gateway, guardrail_id: str) -> None:
    deleted: Final = gateway.request("DELETE", f"/guardrails/{guardrail_id}")
    assert deleted.status_code == 200, deleted.text


def _create_guardrail(
    gateway: Gateway,
    cleanups: ExitStack,
    name: str,
    params: dict[str, JsonValue],
) -> str:
    response: Final = gateway.request(
        "POST",
        "/guardrails",
        {"guardrail": {"guardrail_name": name, "litellm_params": params}},
    )
    assert response.status_code == 200, response.text
    created: Final = _CreatedGuardrail.model_validate_json(response.content)
    assert created.guardrail_name == name, response.text
    cleanups.callback(_delete_guardrail, gateway, created.guardrail_id)
    return created.guardrail_name


def _assert_guardrail_request(request: Request, text: str, response: httpx.Response) -> None:
    assert (request.method, request.target) == ("POST", "/beta/litellm_basic_guardrail_api")
    payload: Final = _JSON_OBJECT.validate_json(request.body)
    assert set(payload) == {
        "input_type",
        "litellm_call_id",
        "litellm_trace_id",
        "structured_messages",
        "images",
        "tools",
        "texts",
        "request_data",
        "request_headers",
        "litellm_version",
        "additional_provider_specific_params",
        "tool_calls",
        "model",
    }, response.text
    assert {
        field: payload[field]
        for field in (
            "input_type",
            "structured_messages",
            "images",
            "tools",
            "texts",
            "additional_provider_specific_params",
            "tool_calls",
            "model",
        )
    } == {
        "input_type": "request",
        "structured_messages": None,
        "images": None,
        "tools": None,
        "texts": [text],
        "additional_provider_specific_params": {},
        "tool_calls": None,
        "model": None,
    }, response.text
    assert payload["request_data"] == {}, response.text
    request_headers: Final = payload["request_headers"]
    assert request_headers is None or isinstance(request_headers, dict), response.text
    if isinstance(request_headers, dict):
        assert "authorization" not in request_headers, response.text
    assert payload["litellm_call_id"] is None or isinstance(payload["litellm_call_id"], str), response.text
    assert payload["litellm_trace_id"] is None or isinstance(payload["litellm_trace_id"], str), response.text
    assert payload["litellm_version"] is None or isinstance(payload["litellm_version"], str), response.text
    if response.headers.get("x-litellm-call-id") is not None:
        assert payload["litellm_call_id"] == response.headers["x-litellm-call-id"], response.text
    assert request.headers["x-api-key"] == "synthetic-apply-guardrail-key", response.text


def test_apply_guardrail_returns_a_block_error_for_generic_guardrails(gateway: Gateway) -> None:
    def block(_request: Request) -> Reply:
        return Reply(body=json.dumps({"action": "BLOCKED", "blocked_reason": _BLOCK_REASON}).encode())

    with wire_server(block) as guardrail, gateway.scenario() as scenario:
        name: Final = _create_guardrail(
            gateway,
            scenario.cleanups,
            f"blocked-apply-{uuid.uuid4().hex}",
            {
                "guardrail": "generic_guardrail_api",
                "mode": "pre_call",
                "default_on": False,
                "api_base": f"{guardrail.url}/beta/litellm_basic_guardrail_api",
                "api_key": "synthetic-apply-guardrail-key",
            },
        )
        response: Final = gateway.request(
            "POST",
            "/guardrails/apply_guardrail",
            {"guardrail_name": name, "text": _BLOCKED_TEXT},
        )
        assert response.status_code == 400, response.text
        error: Final = _ApplyError.model_validate_json(response.content)
        assert error.model_dump() == {
            "error": {
                "message": _BLOCK_REASON,
                "type": "internal_server_error",
                "param": None,
                "code": 400,
            }
        }, response.text
        requests: Final = guardrail.drain()
        assert [(request.method, request.target) for request in requests] == [
            ("POST", "/beta/litellm_basic_guardrail_api")
        ]
        request: Final = requests[0]
        _assert_guardrail_request(request, _BLOCKED_TEXT, response)


def test_apply_guardrail_forwards_messages_to_the_guardrail(gateway: Gateway) -> None:
    def allow(_request: Request) -> Reply:
        return Reply(body=json.dumps({"data": {"guardrailsResult": {"Allowed": True, "Reason": ""}}}).encode())

    text: Final = f"akto apply text {uuid.uuid4().hex}"
    messages: Final = [
        {"role": "system", "content": "akto system prompt"},
        {"role": "user", "content": "akto user content different from the text field"},
    ]
    with wire_server(allow) as akto, gateway.scenario() as scenario:
        name: Final = _create_guardrail(
            gateway,
            scenario.cleanups,
            f"akto-apply-{uuid.uuid4().hex}",
            {
                "guardrail": "akto",
                "mode": "pre_call",
                "default_on": False,
                "akto_base_url": akto.url,
                "akto_api_key": "synthetic-akto-apply-key",
            },
        )
        with_messages: Final = gateway.request(
            "POST",
            "/guardrails/apply_guardrail",
            {"guardrail_name": name, "text": text, "messages": messages},
        )
        assert with_messages.status_code == 200, with_messages.text
        assert _ApplyResponse.model_validate_json(with_messages.content).model_dump() == {"response_text": text}, (
            with_messages.text
        )
        without_messages: Final = gateway.request(
            "POST",
            "/guardrails/apply_guardrail",
            {"guardrail_name": name, "text": text},
        )
        assert without_messages.status_code == 200, without_messages.text
        assert _ApplyResponse.model_validate_json(without_messages.content).model_dump() == {"response_text": text}, (
            without_messages.text
        )

        requests: Final = akto.drain()
        assert [(request.method, request.target) for request in requests] == [
            ("POST", "/api/http-proxy?akto_connector=litellm&guardrails=true"),
            ("POST", "/api/http-proxy?akto_connector=litellm&guardrails=true"),
        ]
        assert [request.headers["authorization"] for request in requests] == ["synthetic-akto-apply-key"] * 2
        bodies: Final = tuple(
            _JSON_OBJECT.validate_json(
                string_value(
                    json.loads(string_value(_JSON_OBJECT.validate_json(request.body)["requestPayload"]))["body"]
                )
            )
            for request in requests
        )
        assert bodies[0]["messages"] == messages, bodies[0]
        assert bodies[1]["messages"] == [{"role": "user", "content": text}], bodies[1]


def test_apply_guardrail_forwards_metadata_to_custom_code(
    gateway: Gateway,
    tmp_path: Path,
) -> None:
    def record(request: Request) -> Reply:
        assert request.target == "/metadata"
        return Reply(body=b'{"received":true}')

    with wire_server(record) as recorder:
        config: Final = _JSON_OBJECT.validate_python(
            yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
        )
        settings: Final = _JSON_OBJECT.validate_python(config["litellm_settings"])
        config_path: Final = tmp_path / "apply-guardrail-owned-proxy.yaml"
        config_path.write_text(
            yaml.safe_dump(
                {
                    **config,
                    "litellm_settings": {**settings, "user_url_allowed_hosts": ["127.0.0.1"]},
                }
            )
        )
        custom_code: Final = f"""
async def apply_guardrail(inputs, request_data, input_type):
    await http_post(
        "{recorder.url}/metadata",
        body={{
            "texts": inputs.get("texts", []),
            "metadata": request_data.get("metadata", {{}}),
            "input_type": input_type,
        }},
    )
    return allow()
"""
        with owned_proxy(gateway, tmp_path, {}, config=config_path) as candidate:
            with candidate.scenario() as scenario:
                name: Final = _create_guardrail(
                    candidate,
                    scenario.cleanups,
                    f"metadata-apply-{uuid.uuid4().hex}",
                    {
                        "guardrail": "custom_code",
                        "mode": "pre_call",
                        "default_on": False,
                        "custom_code": custom_code,
                    },
                )
                with_metadata: Final = candidate.request(
                    "POST",
                    "/guardrails/apply_guardrail",
                    {
                        "guardrail_name": name,
                        "text": _METADATA_TEXT,
                        "metadata": {"forbidden_topics": ["tax"]},
                    },
                )
                assert with_metadata.status_code == 200, with_metadata.text
                assert _ApplyResponse.model_validate_json(with_metadata.content).model_dump() == {
                    "response_text": _METADATA_TEXT
                }, with_metadata.text
                without_metadata: Final = candidate.request(
                    "POST",
                    "/guardrails/apply_guardrail",
                    {"guardrail_name": name, "text": _METADATA_TEXT},
                )
                assert without_metadata.status_code == 200, without_metadata.text
                assert _ApplyResponse.model_validate_json(without_metadata.content).model_dump() == {
                    "response_text": _METADATA_TEXT
                }, without_metadata.text

        requests: Final = recorder.drain()
        assert [(request.method, request.target) for request in requests] == [
            ("POST", "/metadata"),
            ("POST", "/metadata"),
        ]
        assert tuple(_JSON_OBJECT.validate_json(request.body) for request in requests) == (
            {
                "texts": [_METADATA_TEXT],
                "metadata": {"forbidden_topics": ["tax"]},
                "input_type": "request",
            },
            {"texts": [_METADATA_TEXT], "metadata": {}, "input_type": "request"},
        )
