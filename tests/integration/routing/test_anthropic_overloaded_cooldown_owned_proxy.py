import json
import os
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Final
from urllib.parse import unquote

import httpx
import pytest
import yaml
from integration._support.anthropic_sse import (
    LIFECYCLE,
    error_body,
    event_types,
    message_id,
    message_stream,
    parse_sse,
    status_reply,
    stream_reply,
    user_prompt,
)
from integration._support.client import (
    Gateway,
    JsonValue,
    eventually,
    gateway_from_environment,
    object_value,
    string_value,
)
from integration._support.database import read_rows
from integration._support.process import graceful_stop_seconds, owned_proxy
from integration._support.wire import Reply, Request, Wire, wire_server

_OVERLOADED: Final = "claude-overloaded"
_HEALTHY: Final = "claude-healthy"
_COUNTED: Final = "claude-counted"
_COUNTED_OVERLOADED: Final = "claude-counted-overloaded"
_COUNTED_RATE_LIMITED: Final = "claude-counted-rate-limited"
_CHAT_ONLY_BACKEND: Final = "audit-chat-only"
_API_KEY: Final = "synthetic-anthropic-key"
_TEXT: Final = "Hello"
_COOLDOWN_GROUP: Final = "audit-overloaded-cooldown"
_COUNT_DISABLED_GROUP: Final = "audit-count-tokens-disabled"
_COUNT_UPSTREAM_GROUP: Final = "audit-count-tokens-upstream-503"
_COUNT_OVERLOADED_GROUP: Final = "audit-count-tokens-upstream-529"
_COUNT_RATE_LIMITED_GROUP: Final = "audit-count-tokens-upstream-429"
_GATEWAY: Final = "/claude_code_gateway"
_GATEWAY_GROUPS: Final = {
    "non-stream": "audit-gateway-overloaded-non-stream",
    "stream": "audit-gateway-overloaded-stream",
}
_COUNT_STATUSES: Final = {_COUNTED: 503, _COUNTED_OVERLOADED: 529, _COUNTED_RATE_LIMITED: 429}
_ADAPTIVE_UPSTREAMS: Final = {503: "claude-adaptive-unavailable"}
_ADAPTIVE_FLUSH_SECONDS: Final = 40
_COUNT_TOKENS_TARGET: Final = "/v1/messages/count_tokens"
_UPSTREAM_URL_PLACEHOLDER: Final = "upstream-url"

pytestmark: Final = pytest.mark.timeout(2 * graceful_stop_seconds() + 120)


def _served_id(marker: str) -> str:
    return f"msg_{_HEALTHY}_{marker}"


def _respond(request: Request) -> Reply:
    body: Final = object_value(json.loads(request.body))
    backend: Final = str(body["model"])
    if unquote(request.target) == _COUNT_TOKENS_TARGET:
        status: Final = _COUNT_STATUSES[backend]
        return Reply(status=status, body=error_body(status, f"scripted {status}"))
    if backend == _OVERLOADED:
        return status_reply(529)
    if backend in _ADAPTIVE_UPSTREAMS.values():
        return status_reply(next(status for status, name in _ADAPTIVE_UPSTREAMS.items() if name == backend))
    if backend == _HEALTHY:
        return stream_reply(message_stream(_served_id(user_prompt(body)), backend, _TEXT))
    return Reply(status=404, body=b'{"error": "no scripted backend"}')


def _deployment(name: str, model: str, api_base: str, api_key: str, **params: JsonValue) -> dict[str, JsonValue]:
    return {"model_name": name, "litellm_params": {"model": model, "api_base": api_base, "api_key": api_key, **params}}


def _adaptive_router(status: int) -> str:
    return f"audit-adaptive-router-{status}"


def _adaptive_backend(status: int) -> str:
    return f"audit-adaptive-backend-{status}"


def _adaptive_deployments(status: int) -> tuple[dict[str, JsonValue], ...]:
    return (
        {
            "model_name": _adaptive_router(status),
            "litellm_params": {
                "model": "auto_router/adaptive_router",
                "adaptive_router_config": {"available_models": [_adaptive_backend(status)]},
            },
        },
        {
            **_deployment(
                _adaptive_backend(status),
                f"anthropic/{_ADAPTIVE_UPSTREAMS[status]}",
                _UPSTREAM_URL_PLACEHOLDER,
                _API_KEY,
                input_cost_per_token=0.000001,
            ),
            "model_info": {"adaptive_router_preferences": {"quality_tier": 2, "strengths": []}},
        },
    )


def _config(wire: Wire, directory: Path) -> Path:
    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config["model_list"] = [
        _deployment(_COOLDOWN_GROUP, f"anthropic/{_OVERLOADED}", _UPSTREAM_URL_PLACEHOLDER, _API_KEY, order=1),
        _deployment(_COOLDOWN_GROUP, f"anthropic/{_HEALTHY}", _UPSTREAM_URL_PLACEHOLDER, _API_KEY, order=2),
        _deployment(
            _COUNT_DISABLED_GROUP,
            f"hosted_vllm/{_CHAT_ONLY_BACKEND}",
            _UPSTREAM_URL_PLACEHOLDER + "/v1",
            "synthetic-vllm-key",
        ),
        _deployment(_COUNT_UPSTREAM_GROUP, f"anthropic/{_COUNTED}", _UPSTREAM_URL_PLACEHOLDER, _API_KEY),
        _deployment(_COUNT_OVERLOADED_GROUP, f"anthropic/{_COUNTED_OVERLOADED}", _UPSTREAM_URL_PLACEHOLDER, _API_KEY),
        _deployment(
            _COUNT_RATE_LIMITED_GROUP, f"anthropic/{_COUNTED_RATE_LIMITED}", _UPSTREAM_URL_PLACEHOLDER, _API_KEY
        ),
        *(
            _deployment(group, f"anthropic/{_OVERLOADED}", _UPSTREAM_URL_PLACEHOLDER, _API_KEY)
            for group in _GATEWAY_GROUPS.values()
        ),
        *chain.from_iterable(_adaptive_deployments(status) for status in _ADAPTIVE_UPSTREAMS),
    ]
    config["router_settings"] = {
        "num_retries": 0,
        "allowed_fails_policy": {"InternalServerErrorAllowedFails": 0},
        "cooldown_time": 120,
        "redis_host": os.environ["REDIS_HOST"],
        "redis_port": int(os.environ["REDIS_PORT"]),
    }
    config["litellm_settings"]["disable_token_counter"] = True
    config["general_settings"]["enable_claude_code_gateway"] = True
    path: Final = directory / "anthropic-overloaded-cooldown.yaml"
    path.write_text(yaml.safe_dump(config).replace(_UPSTREAM_URL_PLACEHOLDER, wire.url))
    return path


@dataclass(frozen=True, slots=True)
class _Rig:
    proxy: Gateway
    wire: Wire

    def messages(self, path: str, model: str, marker: str, **extra: JsonValue) -> httpx.Response:
        return self.proxy.request(
            "POST",
            path,
            {
                "model": model,
                "max_tokens": 16,
                "stream": True,
                "messages": [{"role": "user", "content": marker}],
                **extra,
            },
            headers={"anthropic-version": "2023-06-01"},
        )

    def count_tokens(self, model: str, prefix: str = "") -> httpx.Response:
        return self.proxy.request(
            "POST",
            prefix + _COUNT_TOKENS_TARGET,
            {"model": model, "messages": [{"role": "user", "content": "count me"}]},
            headers={"anthropic-version": "2023-06-01"},
        )

    def targets_called(self) -> tuple[str, ...]:
        return tuple(unquote(request.target) for request in self.wire.drain())

    def backends_called(self) -> tuple[str, ...]:
        return tuple(str(object_value(json.loads(request.body))["model"]) for request in self.wire.drain())


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Rig]:
    directory: Final = tmp_path_factory.mktemp("anthropic-overloaded-cooldown")
    with gateway_from_environment() as gateway, wire_server(_respond) as wire:
        overrides: Final = {"ANTHROPIC_API_BASE": wire.url, "ANTHROPIC_API_KEY": _API_KEY}
        with owned_proxy(gateway, directory, overrides, config=_config(wire, directory), workers=2) as proxy:
            yield _Rig(proxy, wire)


@pytest.fixture(autouse=True)
def _fresh_wire(rig: _Rig) -> None:
    rig.wire.drain()


def _assert_overloaded_json(response: httpx.Response) -> None:
    assert response.status_code == 529, response.text
    payload: Final = object_value(json.loads(response.content))
    assert object_value(payload["error"])["type"] == "overloaded_error", response.text


def _count_tokens_error(response: httpx.Response, status: int) -> dict[str, JsonValue]:
    assert response.status_code == status, response.text
    detail: Final = object_value(object_value(json.loads(response.content))["detail"])
    assert detail["type"] == "error", response.text
    return object_value(detail["error"])


def test_an_overloaded_deployment_is_cooled_down_by_the_internal_server_error_allowed_fails_policy(rig: _Rig) -> None:
    marker: Final = uuid.uuid4().hex
    first: Final = rig.messages("/v1/messages", _COOLDOWN_GROUP, marker, num_retries=0, disable_fallbacks=True)
    _assert_overloaded_json(first)
    assert rig.backends_called() == (_OVERLOADED,)
    second: Final = rig.messages("/v1/messages", _COOLDOWN_GROUP, marker, num_retries=0, disable_fallbacks=True)
    assert second.status_code == 200, second.text
    events: Final = parse_sse(second.text)
    assert event_types(events) == LIFECYCLE, events
    assert message_id(events) == _served_id(marker), events
    assert rig.backends_called() == (_HEALTHY,)


def test_count_tokens_disabled_without_a_provider_counter_answers_529_overloaded_error(rig: _Rig) -> None:
    response: Final = rig.count_tokens(_COUNT_DISABLED_GROUP)
    error: Final = _count_tokens_error(response, 529)
    assert error["type"] == "overloaded_error", response.text
    assert "Token counting is disabled" in string_value(error["message"]), response.text
    assert rig.targets_called() == ()


def test_count_tokens_whose_upstream_answers_503_answers_529_overloaded_error(rig: _Rig) -> None:
    response: Final = rig.count_tokens(_COUNT_UPSTREAM_GROUP)
    error: Final = _count_tokens_error(response, 529)
    assert error["type"] == "overloaded_error", response.text
    assert "scripted 503" in string_value(error["message"]), response.text
    assert rig.targets_called() == (_COUNT_TOKENS_TARGET,)


@pytest.mark.parametrize(
    ("group", "status", "error_type"),
    (
        pytest.param(_COUNT_OVERLOADED_GROUP, 529, "overloaded_error", id="529"),
        pytest.param(_COUNT_RATE_LIMITED_GROUP, 429, "rate_limit_error", id="429"),
    ),
)
def test_count_tokens_keeps_other_upstream_statuses_and_their_detail_body(
    rig: _Rig, group: str, status: int, error_type: str
) -> None:
    response: Final = rig.count_tokens(group)
    error: Final = _count_tokens_error(response, status)
    assert error["type"] == error_type, response.text
    assert f"scripted {status}" in string_value(error["message"]), response.text
    assert rig.targets_called() == (_COUNT_TOKENS_TARGET,)


def test_pass_through_messages_stream_forwards_an_upstream_529_verbatim(rig: _Rig) -> None:
    response: Final = rig.messages("/anthropic/v1/messages", _OVERLOADED, uuid.uuid4().hex)
    assert response.status_code == 529, response.text
    assert response.content == error_body(529, "scripted 529"), response.text
    assert rig.backends_called() == (_OVERLOADED,)


@pytest.mark.parametrize("shape", _GATEWAY_GROUPS.keys())
def test_the_claude_code_gateway_messages_route_answers_an_upstream_529_as_overloaded_error(
    rig: _Rig, shape: str
) -> None:
    response: Final = rig.messages(
        f"{_GATEWAY}/v1/messages",
        _GATEWAY_GROUPS[shape],
        uuid.uuid4().hex,
        stream=shape == "stream",
        num_retries=0,
    )
    _assert_overloaded_json(response)
    assert rig.backends_called() == (_OVERLOADED,)


def test_the_claude_code_gateway_count_tokens_route_answers_an_upstream_503_as_529(rig: _Rig) -> None:
    response: Final = rig.count_tokens(_COUNT_UPSTREAM_GROUP, prefix=_GATEWAY)
    error: Final = _count_tokens_error(response, 529)
    assert error["type"] == "overloaded_error", response.text
    assert "scripted 503" in string_value(error["message"]), response.text
    assert rig.targets_called() == (_COUNT_TOKENS_TARGET,)


def _adaptive_session_rows(session: str) -> list[dict[str, JsonValue]]:
    return read_rows(
        'SELECT model_name, exhaustion_count FROM "LiteLLM_AdaptiveRouterSession" WHERE session_id = %s', (session,)
    )


@pytest.mark.parametrize(
    ("status", "answered"),
    (
        pytest.param(503, 503, id="upstream-503"),
    ),
)
def test_an_adaptive_router_session_counts_an_upstream_overload_as_exhaustion(
    rig: _Rig, status: int, answered: int
) -> None:
    session: Final = f"audit-adaptive-{uuid.uuid4().hex}"
    response: Final = rig.proxy.request(
        "POST",
        "/v1/chat/completions",
        {
            "model": _adaptive_router(status),
            "max_tokens": 16,
            "messages": [{"role": "user", "content": session}],
        },
        headers={"x-litellm-session-id": session},
    )
    assert response.status_code == answered, response.text
    assert rig.backends_called() == (_ADAPTIVE_UPSTREAMS[status],)
    rows: Final = eventually(
        lambda: _adaptive_session_rows(session),
        lambda rows: len(rows) > 0,
        seconds=_ADAPTIVE_FLUSH_SECONDS,
        return_last_on_timeout=True,
    )
    assert rows == [{"model_name": _adaptive_backend(status), "exhaustion_count": 1}], response.text
