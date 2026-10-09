import hashlib
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Final

import pytest
import yaml
from anthropic.types import Message
from integration._support import claude_code as cc
from integration._support.client import Gateway, eventually, gateway_from_environment
from integration._support.database import read_rows
from integration._support.process import owned_proxy_process
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue, TypeAdapter

_BACKEND: Final = "claude-sonnet-4-6"
_API_KEY: Final = "synthetic-anthropic-key"
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_MESSAGE: Final = TypeAdapter(Message)


def _expected_stream_events(
    stream_reply: tuple[bytes, ...], model: str
) -> tuple[tuple[str, dict[str, JsonValue]], ...]:
    return tuple(
        (
            event,
            {
                **data,
                **(
                    {
                        "message": {
                            **_JSON_OBJECT.validate_python(data["message"]),
                            "model": model,
                        }
                    }
                    if event == "message_start"
                    else {}
                ),
            },
        )
        for event, data in cc.sse_events(b"".join(stream_reply).decode())
    )


def _owned_config(path: Path) -> Path:
    config: Final = _JSON_OBJECT.validate_python(
        yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    )
    general_settings: Final = _JSON_OBJECT.validate_python(config.get("general_settings", {}))
    litellm_settings: Final = _JSON_OBJECT.validate_python(config.get("litellm_settings", {}))
    path.write_text(
        yaml.safe_dump(
            {
                **config,
                "general_settings": {**general_settings, "enable_claude_code_gateway": True},
                "litellm_settings": {**litellm_settings, "cache": False},
            }
        )
    )
    return path


def _cli_bearer_headers(key: str) -> dict[str, str]:
    return {
        **{name: value for name, value in cc.cli_headers(key).items() if name != "x-api-key"},
        "authorization": f"Bearer {key}",
    }


@pytest.fixture(scope="module")
def enabled_gateway(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Gateway]:
    with gateway_from_environment() as base_gateway:
        directory: Final = tmp_path_factory.mktemp("anthropic-gateway")
        config: Final = _owned_config(directory / "gateway.yaml")
        with owned_proxy_process(base_gateway, directory, {}, config=config, workers=2) as owned:
            yield owned.gateway


@pytest.mark.parametrize(
    "path",
    ["/claude_code_gateway/v1/messages?beta=true", "/claude_code_gateway/v1/messages/count_tokens?beta=true"],
    ids=["messages", "count_tokens"],
)
def test_shared_gateway_route_is_disabled_without_reaching_provider(gateway: Gateway, path: str) -> None:
    def respond(request: Request) -> Reply:
        raise AssertionError(f"disabled gateway must not reach the provider: {request.target}")

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"anthropic/{_BACKEND}", api_base=wire.url, api_key=_API_KEY)
        response: Final = gateway.client.post(
            path,
            json={"model": model, "max_tokens": 16, "messages": [{"role": "user", "content": "disabled control"}]},
            headers=_cli_bearer_headers(gateway.key),
        )
        assert response.status_code == 404, response.text
        assert _JSON_OBJECT.validate_json(response.content) == {"detail": "Claude Code gateway is not enabled"}, (
            response.text
        )
        assert wire.drain() == ()


def _restricted_model_responses(enabled_gateway: Gateway) -> tuple[int, str, int, str, str, tuple[Request, ...]]:
    def respond(request: Request) -> Reply:
        raise AssertionError(f"restricted model must not reach the provider: {request.target}")

    with enabled_gateway.scenario() as scenario, wire_server(respond) as wire:
        model_a: Final = scenario.model(model=f"anthropic/{_BACKEND}", api_base=wire.url, api_key=_API_KEY)
        model_b: Final = scenario.model(model=f"anthropic/{_BACKEND}", api_base=wire.url, api_key=_API_KEY)
        key: Final = scenario.key(models=[model_a])
        request_body: Final = {**cc.claude_code_request("restricted-model"), "model": model_b}
        gateway_response: Final = enabled_gateway.client.post(
            "/claude_code_gateway/v1/messages?beta=true",
            json=request_body,
            headers=_cli_bearer_headers(key),
        )
        messages_response: Final = enabled_gateway.client.post(
            "/v1/messages?beta=true",
            json=request_body,
            headers=cc.cli_headers(key),
        )
        return (
            gateway_response.status_code,
            gateway_response.text,
            messages_response.status_code,
            messages_response.text,
            model_b,
            wire.drain(),
        )


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
@pytest.mark.timeout(300)
def test_enabled_gateway_matches_native_messages_route(enabled_gateway: Gateway, stream: bool) -> None:
    identity: Final = "msg_gateway_" + ("stream" if stream else "json")
    usage: Final = {"input_tokens": 6, "output_tokens": 1}
    content: Final = ({"type": "text", "text": "PONG"},)
    request_body: Final = {
        **cc.claude_code_request(f"cache-bust-{identity}"),
        "model": "",
        "stream": stream,
    }
    expected_stream: Final = cc.message_stream(identity, _BACKEND, content, usage)
    expected_provider_body: Final = {**request_body, "model": _BACKEND}
    client_betas: Final = sorted(cc.CLI_BETA.split(","))

    def respond(request: Request) -> Reply:
        assert request.method == "POST" and request.target == "/v1/messages", request.target
        assert request.headers["x-api-key"] == _API_KEY, request.headers
        assert request.headers["anthropic-version"] == "2023-06-01", request.headers
        assert sorted(request.headers["anthropic-beta"].split(",")) == client_betas, request.headers
        assert "authorization" not in request.headers, request.headers
        assert _JSON_OBJECT.validate_json(request.body) == expected_provider_body, request.body
        if stream:
            return Reply(chunks=expected_stream, content_type="text/event-stream")
        return Reply(body=cc.message_reply(identity, _BACKEND, content, usage))

    with enabled_gateway.scenario() as scenario, wire_server(respond) as wire:
        model_a: Final = scenario.model(model=f"anthropic/{_BACKEND}", api_base=wire.url, api_key=_API_KEY)
        model_b: Final = scenario.model(model=f"anthropic/{_BACKEND}", api_base=wire.url, api_key=_API_KEY)
        key: Final = scenario.key(models=[model_a])
        expected_message: Final = _MESSAGE.validate_python(
            {**_JSON_OBJECT.validate_json(cc.message_reply(identity, _BACKEND, content, usage)), "model": model_a}
        )
        expected_stream_events: Final = _expected_stream_events(expected_stream, model_a)
        body: Final = {**request_body, "model": model_a}
        headers: Final = _cli_bearer_headers(key)
        gateway_response: Final = enabled_gateway.client.post(
            "/claude_code_gateway/v1/messages?beta=true",
            json=body,
            headers=headers,
        )
        assert gateway_response.status_code == 200, gateway_response.text
        gateway_requests: Final = wire.drain()
        assert len(gateway_requests) == 1, gateway_requests

        messages_response: Final = enabled_gateway.client.post(
            "/v1/messages?beta=true",
            json=body,
            headers=headers,
        )
        assert messages_response.status_code == 200, messages_response.text
        messages_requests: Final = wire.drain()
        assert len(messages_requests) == 1, messages_requests
        if stream:
            gateway_events: Final = cc.sse_events(gateway_response.text)
            messages_events: Final = cc.sse_events(messages_response.text)
            assert gateway_events == messages_events, (gateway_response.text, messages_response.text)
            assert gateway_events == expected_stream_events, (gateway_response.text, messages_response.text)
        else:
            gateway_message: Final = _MESSAGE.validate_json(gateway_response.content)
            messages_message: Final = _MESSAGE.validate_json(messages_response.content)
            assert gateway_message.model_dump(mode="json") == expected_message.model_dump(mode="json"), (
                gateway_response.text,
            )
            assert messages_message.model_dump(mode="json") == expected_message.model_dump(mode="json"), (
                messages_response.text,
            )

        (gateway_request,) = gateway_requests
        (messages_request,) = messages_requests
        assert gateway_request.method == messages_request.method == "POST"
        assert gateway_request.target == messages_request.target == "/v1/messages"
        assert gateway_request.headers["x-api-key"] == messages_request.headers["x-api-key"] == _API_KEY
        assert (
            gateway_request.headers["anthropic-version"]
            == messages_request.headers["anthropic-version"]
            == "2023-06-01"
        )
        assert gateway_request.headers["anthropic-beta"] == messages_request.headers["anthropic-beta"]
        assert sorted(gateway_request.headers["anthropic-beta"].split(",")) == client_betas
        assert _JSON_OBJECT.validate_json(gateway_request.body) == expected_provider_body, gateway_request.body
        assert _JSON_OBJECT.validate_json(messages_request.body) == expected_provider_body, messages_request.body


@pytest.mark.timeout(300)
def test_enabled_gateway_count_tokens_alias_matches_native_count_tokens_route(enabled_gateway: Gateway) -> None:
    messages: Final = [{"role": "user", "content": "Count this gateway request."}]
    system: Final = [{"type": "text", "text": "Count the gateway message and tool."}]
    tools: Final = [
        {
            "name": "lookup",
            "description": "Look up a synthetic record.",
            "input_schema": {"type": "object", "properties": {"record_id": {"type": "string"}}},
        }
    ]
    expected_provider_body: Final = {"model": _BACKEND, "messages": messages, "system": system, "tools": tools}

    def respond(request: Request) -> Reply:
        assert request.method == "POST" and request.target == "/v1/messages/count_tokens", request.target
        assert request.headers["x-api-key"] == _API_KEY, request.headers
        assert request.headers["anthropic-version"] == "2023-06-01", request.headers
        assert request.headers["anthropic-beta"] == "token-counting-2024-11-01", request.headers
        assert "authorization" not in request.headers, request.headers
        assert _JSON_OBJECT.validate_json(request.body) == expected_provider_body, request.body
        return Reply(body=b'{"input_tokens": 41}')

    with enabled_gateway.scenario() as scenario, wire_server(respond) as wire:
        model: Final = scenario.model(model=f"anthropic/{_BACKEND}", api_base=wire.url, api_key=_API_KEY)
        restricted_model: Final = scenario.model(model=f"anthropic/{_BACKEND}", api_base=wire.url, api_key=_API_KEY)
        key: Final = scenario.key(models=[model])
        body: Final = {"model": model, "messages": messages, "system": system, "tools": tools}
        headers: Final = {**_cli_bearer_headers(key), "anthropic-beta": "token-counting-2024-11-01"}
        gateway_response: Final = enabled_gateway.client.post(
            "/claude_code_gateway/v1/messages/count_tokens?beta=true", json=body, headers=headers
        )
        assert gateway_response.status_code == 200, gateway_response.text
        assert _JSON_OBJECT.validate_json(gateway_response.content) == {"input_tokens": 41}, gateway_response.text
        native_response: Final = enabled_gateway.client.post(
            "/v1/messages/count_tokens?beta=true", json=body, headers=headers
        )
        assert native_response.status_code == 200, native_response.text
        assert native_response.content == gateway_response.content, (native_response.text, gateway_response.text)
        assert tuple((request.method, request.target) for request in wire.drain()) == (
            ("POST", "/v1/messages/count_tokens"),
            ("POST", "/v1/messages/count_tokens"),
        )

        restricted_response: Final = enabled_gateway.client.post(
            "/claude_code_gateway/v1/messages/count_tokens?beta=true",
            json={**body, "model": restricted_model},
            headers=headers,
        )
        assert restricted_response.status_code == 403, restricted_response.text
        assert _JSON_OBJECT.validate_json(restricted_response.content) == {
            "error": {
                "message": (
                    f"The requested model '{restricted_model}' is not available for this API key, or the model name "
                    "is invalid. Check the models available to you and try again."
                ),
                "type": "key_model_access_denied",
                "param": "model",
                "code": "403",
            }
        }, restricted_response.text
        assert wire.drain() == ()


@pytest.mark.timeout(300)
def test_enabled_gateway_refuses_model_outside_virtual_key(enabled_gateway: Gateway) -> None:
    result: Final = _restricted_model_responses(enabled_gateway)
    expected_body: Final = {
        "error": {
            "message": (
                f"The requested model '{result[4]}' is not available for this API key, or the model name is invalid. "
                "Check the models available to you and try again."
            ),
            "type": "key_model_access_denied",
            "param": "model",
            "code": "403",
        }
    }
    assert result[0] == result[2] == 403, (result[1], result[3])
    assert _JSON_OBJECT.validate_json(result[1]) == expected_body, result[1]
    assert _JSON_OBJECT.validate_json(result[3]) == expected_body, result[3]
    assert result[1] == result[3], (result[1], result[3])
    assert result[5] == ()


@pytest.mark.timeout(300)
def test_enabled_gateway_attributes_spend_to_virtual_key(enabled_gateway: Gateway) -> None:
    gateway_identity: Final = f"msg_gateway_spend_gateway_{uuid.uuid4().hex}"
    native_identity: Final = f"msg_gateway_spend_native_{uuid.uuid4().hex}"
    response_ids: Final = iter((gateway_identity, native_identity))
    content: Final = ({"type": "text", "text": "PONG"},)
    usage: Final = {"input_tokens": 6, "output_tokens": 1}
    request_body: Final = {**cc.claude_code_request("cache-bust-msg-gateway-spend"), "model": "", "stream": False}
    expected_provider_body: Final = {**request_body, "model": _BACKEND}
    client_betas: Final = sorted(cc.CLI_BETA.split(","))

    def respond(request: Request) -> Reply:
        assert request.method == "POST" and request.target == "/v1/messages", request.target
        assert request.headers["x-api-key"] == _API_KEY, request.headers
        assert request.headers["anthropic-version"] == "2023-06-01", request.headers
        assert sorted(request.headers["anthropic-beta"].split(",")) == client_betas, request.headers
        assert "authorization" not in request.headers, request.headers
        assert _JSON_OBJECT.validate_json(request.body) == expected_provider_body, request.body
        return Reply(body=cc.message_reply(next(response_ids), _BACKEND, content, usage))

    with enabled_gateway.scenario() as scenario, wire_server(respond) as wire:
        model: Final = scenario.model(model=f"anthropic/{_BACKEND}", api_base=wire.url, api_key=_API_KEY)
        key: Final = scenario.key(models=[model])
        native_key: Final = scenario.key(models=[model])
        body: Final = {**request_body, "model": model}
        response: Final = enabled_gateway.client.post(
            "/claude_code_gateway/v1/messages?beta=true",
            json=body,
            headers=_cli_bearer_headers(key),
        )
        assert response.status_code == 200, response.text
        native_response: Final = enabled_gateway.client.post(
            "/v1/messages?beta=true",
            json=body,
            headers=cc.cli_headers(native_key),
        )
        assert native_response.status_code == 200, native_response.text
        requests: Final = wire.drain()
        assert tuple(request.target for request in requests) == ("/v1/messages", "/v1/messages"), requests
        gateway_key_hash: Final = hashlib.sha256(key.encode()).hexdigest()
        native_key_hash: Final = hashlib.sha256(native_key.encode()).hexdigest()
        spend_rows: Final = eventually(
            lambda: (
                read_rows(
                    'SELECT api_key, model_group FROM "LiteLLM_SpendLogs" WHERE api_key=%s',
                    (gateway_key_hash,),
                ),
                read_rows(
                    'SELECT api_key, model_group FROM "LiteLLM_SpendLogs" WHERE api_key=%s',
                    (native_key_hash,),
                ),
            ),
            lambda rows: len(rows[0]) == 1 and len(rows[1]) == 1,
            seconds=70,
            return_last_on_timeout=True,
        )
        assert spend_rows == (
            [{"api_key": gateway_key_hash, "model_group": model}],
            [{"api_key": native_key_hash, "model_group": model}],
        ), spend_rows
