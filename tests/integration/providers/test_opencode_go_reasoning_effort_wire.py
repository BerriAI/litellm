from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server
from integration.pricing.test_opencode_go_reported_cost_wire import _respond
from pydantic import JsonValue, TypeAdapter

_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_CHAT: Final = "/v1/chat/completions"


def _input(path: str) -> dict[str, JsonValue]:
    if path == "/v1/responses":
        return {"input": "hello"}
    return {"messages": [{"role": "user", "content": "hello"}], "max_tokens": 32}


def _assert_chat(response: str, stream: bool) -> None:
    if stream:
        assert response.rstrip().endswith("data: [DONE]"), response
        events: Final = tuple(
            _OBJECT.validate_json(line.removeprefix("data: "))
            for line in response.splitlines()
            if line.startswith("data: {")
        )
        groups: Final = tuple(
            TypeAdapter(tuple[dict[str, JsonValue], ...]).validate_python(e["choices"]) for e in events
        )
        choices: Final = tuple(group[0] for group in groups if group)
        deltas: Final = tuple(_OBJECT.validate_python(choice["delta"]) for choice in choices)
        assert "".join(str(delta.get("content") or "") for delta in deltas) == "ok", response
        assert tuple(c["finish_reason"] for c in choices if c.get("finish_reason")) == ("stop",), response
        return
    body: Final = _OBJECT.validate_json(response)
    choices_body: Final = TypeAdapter(tuple[dict[str, JsonValue], ...]).validate_python(body["choices"])
    assert _OBJECT.validate_python(choices_body[0]["message"])["content"] == "ok", response
    assert choices_body[0]["finish_reason"] == "stop", response


@pytest.mark.parametrize("effort", ("none", "low", "medium", "high"))
@pytest.mark.parametrize("stream", (False, True), ids=("non-stream", "stream"))
def test_chat_request_reasoning_forwarded_regression(gateway: Gateway, effort: str, stream: bool) -> None:
    def respond(request: Request) -> Reply:
        assert request.target.endswith("/chat/completions"), request.target
        assert request.headers["authorization"] == "Bearer synthetic-go-key"
        assert _OBJECT.validate_json(request.body).get("reasoning_effort") == effort
        return _respond(request)

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="opencode_go/glm-5.3", api_base=wire.url, api_key="synthetic-go-key", drop_params=True
        )
        response: Final = gateway.request(
            "POST", _CHAT, {"model": model, **_input(_CHAT), "reasoning_effort": effort, "stream": stream}
        )
        assert response.status_code == 200, response.text
        _assert_chat(response.text, stream)


def test_chat_strict_reasoning_forwarded_regression(gateway: Gateway) -> None:
    def respond(request: Request) -> Reply:
        assert request.target.endswith("/chat/completions"), request.target
        assert _OBJECT.validate_json(request.body).get("reasoning_effort") == "high"
        return _respond(request)

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="opencode_go/glm-5.3", api_base=wire.url, api_key="synthetic-go-key", drop_params=False
        )
        response: Final = gateway.request("POST", _CHAT, {"model": model, **_input(_CHAT), "reasoning_effort": "high"})
        assert response.status_code == 200, response.text
        _assert_chat(response.text, False)


@pytest.mark.parametrize("session_field", (None, "metadata", "litellm_session_id"))
def test_deployment_reasoning_and_session_forwarded_regression(gateway: Gateway, session_field: str | None) -> None:
    def respond(request: Request) -> Reply:
        assert request.target.endswith("/chat/completions"), request.target
        if session_field is not None:
            assert request.headers["x-opencode-session"] == "reasoning-session"
        assert _OBJECT.validate_json(request.body).get("reasoning_effort") == "none"
        return _respond(request)

    session: Final[dict[str, JsonValue]] = (
        {}
        if session_field is None
        else {"metadata": {"session_id": "reasoning-session"}}
        if session_field == "metadata"
        else {"litellm_session_id": "reasoning-session"}
    )
    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="opencode_go/glm-5.3",
            api_base=wire.url,
            api_key="synthetic-go-key",
            drop_params=True,
            reasoning_effort="none",
        )
        response: Final = gateway.request("POST", _CHAT, {"model": model, **_input(_CHAT), **session})
        assert response.status_code == 200, response.text
        _assert_chat(response.text, False)


@pytest.mark.parametrize("path", ("/v1/responses", "/v1/messages"))
def test_translated_reasoning_forwarded_regression(gateway: Gateway, path: str) -> None:
    def respond(request: Request) -> Reply:
        assert request.target.endswith("/chat/completions"), request.target
        effort: Final = _OBJECT.validate_json(request.body).get("reasoning_effort")
        assert effort == "high" if path == "/v1/responses" else isinstance(effort, str) and bool(effort), effort
        return _respond(request)

    reasoning: Final[dict[str, JsonValue]] = (
        {"reasoning": {"effort": "high"}}
        if path == "/v1/responses"
        else {"thinking": {"type": "enabled", "budget_tokens": 2048}}
    )
    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="opencode_go/glm-5.3", api_base=wire.url, api_key="synthetic-go-key", drop_params=True
        )
        response: Final = gateway.request("POST", path, {"model": model, **_input(path), **reasoning})
        assert response.status_code == 200, response.text
        assert "ok" in response.text, response.text


@pytest.mark.parametrize("path", (_CHAT, "/v1/responses", "/v1/messages"))
def test_absent_reasoning_not_injected_guard(gateway: Gateway, path: str) -> None:
    def respond(request: Request) -> Reply:
        assert request.target.endswith("/chat/completions"), request.target
        assert "reasoning_effort" not in _OBJECT.validate_json(request.body)
        return _respond(request)

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model="opencode_go/glm-5.3", api_base=wire.url, api_key="synthetic-go-key")
        response: Final = gateway.request("POST", path, {"model": model, **_input(path)})
        assert response.status_code == 200, response.text
        assert "ok" in response.text, response.text


@pytest.mark.parametrize("drop_params", (True, False))
def test_unknown_model_reasoning_guard(gateway: Gateway, drop_params: bool) -> None:
    def respond(request: Request) -> Reply:
        assert request.target.endswith("/chat/completions"), request.target
        assert "reasoning_effort" not in _OBJECT.validate_json(request.body)
        return _respond(request)

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="opencode_go/not-in-catalog-x", api_base=wire.url, api_key="synthetic-go-key", drop_params=drop_params
        )
        response: Final = gateway.request("POST", _CHAT, {"model": model, **_input(_CHAT), "reasoning_effort": "high"})
        assert response.status_code == (200 if drop_params else 400), response.text
        if drop_params:
            _assert_chat(response.text, False)
            return
        assert "UnsupportedParamsError" in response.text, response.text
        assert wire.drain() == ()


@pytest.mark.parametrize("model,protocol", (("minimax-m3", "messages"), ("gpt-6-luna", "responses")))
def test_native_protocol_reasoning_guard(gateway: Gateway, model: str, protocol: str) -> None:
    def respond(request: Request) -> Reply:
        assert request.target.endswith(f"/v1/{protocol}"), request.target
        body: Final = _OBJECT.validate_json(request.body)
        if protocol == "messages":
            assert _OBJECT.validate_python(body["thinking"])["type"] == "enabled", body
        else:
            assert _OBJECT.validate_python(body["reasoning"])["effort"] == "high", body
        return _respond(request)

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        deployment: Final = scenario.model(model=f"opencode_go/{model}", api_base=wire.url, api_key="synthetic-go-key")
        response: Final = gateway.request(
            "POST", _CHAT, {"model": deployment, **_input(_CHAT), "reasoning_effort": "high"}
        )
        assert response.status_code == 200, response.text
        _assert_chat(response.text, False)
