import json
import os
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
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
from integration._support.client import Gateway, JsonValue, gateway_from_environment, object_value
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, Wire, wire_server

_OVERLOADED: Final = "claude-overloaded"
_HEALTHY: Final = "claude-healthy"
_COUNTED: Final = "claude-counted"
_CHAT_ONLY_BACKEND: Final = "audit-chat-only"
_API_KEY: Final = "synthetic-anthropic-key"
_TEXT: Final = "Hello"
_COOLDOWN_GROUP: Final = "audit-overloaded-cooldown"
_COUNT_DISABLED_GROUP: Final = "audit-count-tokens-disabled"
_COUNT_UPSTREAM_GROUP: Final = "audit-count-tokens-upstream-503"
_COUNT_TOKENS_TARGET: Final = "/v1/messages/count_tokens"
_UPSTREAM_URL_PLACEHOLDER: Final = "upstream-url"

pytestmark = pytest.mark.timeout(240)


def _served_id(marker: str) -> str:
    return f"msg_{_HEALTHY}_{marker}"


def _respond(request: Request) -> Reply:
    if unquote(request.target) == _COUNT_TOKENS_TARGET:
        return Reply(status=503, body=error_body(503, "scripted 503"))
    body: Final = object_value(json.loads(request.body))
    backend: Final = str(body["model"])
    if backend == _OVERLOADED:
        return status_reply(529)
    if backend == _HEALTHY:
        return stream_reply(message_stream(_served_id(user_prompt(body)), backend, _TEXT))
    return Reply(status=404, body=b'{"error": "no scripted backend"}')


def _deployment(name: str, model: str, api_base: str, api_key: str, **params: JsonValue) -> dict[str, JsonValue]:
    return {"model_name": name, "litellm_params": {"model": model, "api_base": api_base, "api_key": api_key, **params}}


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
    ]
    config["router_settings"] = {
        "num_retries": 0,
        "allowed_fails_policy": {"ServiceUnavailableErrorAllowedFails": 0},
        "cooldown_time": 120,
        "redis_host": os.environ["REDIS_HOST"],
        "redis_port": int(os.environ["REDIS_PORT"]),
    }
    config["litellm_settings"]["disable_token_counter"] = True
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

    def count_tokens(self, model: str) -> httpx.Response:
        return self.proxy.request(
            "POST",
            _COUNT_TOKENS_TARGET,
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


def test_an_overloaded_deployment_is_cooled_down_by_the_service_unavailable_allowed_fails_policy(rig: _Rig) -> None:
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
    _assert_overloaded_json(response)
    assert "Token counting is disabled" in response.text, response.text
    assert rig.targets_called() == ()


def test_count_tokens_whose_upstream_answers_503_answers_529_overloaded_error(rig: _Rig) -> None:
    response: Final = rig.count_tokens(_COUNT_UPSTREAM_GROUP)
    _assert_overloaded_json(response)
    assert "scripted 503" in response.text, response.text
    assert rig.targets_called() == (_COUNT_TOKENS_TARGET,)


def test_pass_through_messages_stream_forwards_an_upstream_529_verbatim(rig: _Rig) -> None:
    response: Final = rig.messages("/anthropic/v1/messages", _OVERLOADED, uuid.uuid4().hex)
    assert response.status_code == 529, response.text
    assert response.content == error_body(529, "scripted 529"), response.text
    assert rig.backends_called() == (_OVERLOADED,)
