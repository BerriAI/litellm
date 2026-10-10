import json
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

import httpx
import pytest
import yaml
from integration._support.anthropic_sse import (
    LIFECYCLE,
    PING,
    Attempts,
    delta_text,
    dropping_reply,
    error_body,
    error_frame,
    event_types,
    message_id,
    message_stream,
    parse_sse,
    status_reply,
    stream_reply,
    user_prompt,
)
from integration._support.client import Gateway, gateway_from_environment, object_value
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue

_PRIMARY: Final = "claude-under-test"
_FALLBACK: Final = "claude-fallback"
_CONTEXT_WINDOW: Final = "claude-context-window"
_API_KEY: Final = "synthetic-anthropic-key"
_TEXT: Final = "Hello"

_ROUTER_BUDGET: Final = "audit-router-budget"
_STRING_BUDGET: Final = "audit-string-budget"
_WITH_FALLBACKS: Final = "audit-primary"
_FALLBACK_GROUP: Final = "audit-fallback"
_CW_FALLBACK_GROUP: Final = "audit-cw-fallback"
_LONELY: Final = "audit-lonely"
_POLICY: Final = "audit-policy"
_UPSTREAM_URL_PLACEHOLDER: Final = "upstream-url"

Behavior = Literal[
    "drop-once", "drop-always", "hold-ping", "drop-then-too-long", "drop-then-401", "overloaded-frames-fallback-503"
]

pytestmark = pytest.mark.timeout(240)


def _served_id(backend: str, marker: str, attempt: int) -> str:
    return f"msg_{backend}_{marker}_a{attempt}"


def _primary_reply(behavior: str, attempt: int, full: tuple[bytes, bytes, bytes, bytes]) -> Reply:
    dropped: Final = dropping_reply(full, abort_after=1)
    match behavior:
        case "drop-once":
            return dropped if attempt == 1 else stream_reply(full)
        case "drop-always":
            return dropped
        case "hold-ping":
            return stream_reply((full[0], PING, full[1] + full[2] + full[3]), pause=0.25)
        case "drop-then-too-long":
            if attempt == 1:
                return dropped
            return Reply(status=400, body=error_body(400, "prompt is too long: 250000 tokens > 200000 maximum"))
        case "drop-then-401":
            return dropped if attempt == 1 else status_reply(401)
        case "overloaded-frames-fallback-503":
            return stream_reply((error_frame(529, "scripted overloaded"),))
    raise AssertionError(behavior)


def _fallback_reply(behavior: str, full: tuple[bytes, bytes, bytes, bytes]) -> Reply:
    if behavior == "overloaded-frames-fallback-503":
        return status_reply(503)
    return stream_reply(full)


def _respond(attempts: Attempts) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        assert (request.method, request.target) == ("POST", "/v1/messages"), request
        assert request.headers["x-api-key"] == _API_KEY, request.headers
        body: Final = object_value(json.loads(request.body))
        assert "num_retries" not in body, body
        backend: Final = str(body["model"])
        behavior, marker = user_prompt(body).split(":", 1)
        attempt: Final = attempts.record(f"{backend}:{marker}")
        full: Final = message_stream(_served_id(backend, marker, attempt), backend, _TEXT)
        if backend != _PRIMARY:
            return _fallback_reply(behavior, full)
        return _primary_reply(behavior, attempt, full)

    return respond


def _deployment(name: str, backend: str, **extra: JsonValue) -> dict[str, JsonValue]:
    return {
        "model_name": name,
        "litellm_params": {
            "model": f"anthropic/{backend}",
            "api_base": _UPSTREAM_URL_PLACEHOLDER,
            "api_key": _API_KEY,
            **extra,
        },
    }


def _config(wire: Wire, directory: Path) -> Path:
    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config["model_list"] = [
        _deployment(_ROUTER_BUDGET, _PRIMARY),
        _deployment(_STRING_BUDGET, _PRIMARY, num_retries="2"),
        _deployment(_WITH_FALLBACKS, _PRIMARY, num_retries=1),
        _deployment(_FALLBACK_GROUP, _FALLBACK),
        _deployment(_CW_FALLBACK_GROUP, _CONTEXT_WINDOW),
        _deployment(_LONELY, _PRIMARY, num_retries=1),
        _deployment(_POLICY, _PRIMARY),
    ]
    config["router_settings"] = {
        "num_retries": 1,
        "disable_cooldowns": True,
        "fallbacks": [{_WITH_FALLBACKS: [_FALLBACK_GROUP]}],
        "context_window_fallbacks": [{_WITH_FALLBACKS: [_CW_FALLBACK_GROUP]}],
        "model_group_retry_policy": {_POLICY: {"DefaultRetries": 2}},
    }
    path: Final = directory / "messages-retry-budget-sources.yaml"
    path.write_text(yaml.safe_dump(config).replace(_UPSTREAM_URL_PLACEHOLDER, wire.url))
    return path


@dataclass(frozen=True, slots=True)
class _Rig:
    proxy: Gateway
    attempts: Attempts

    def stream(self, model: str, behavior: Behavior, marker: str, **extra: JsonValue) -> httpx.Response:
        return self.proxy.request(
            "POST",
            "/v1/messages",
            {
                "model": model,
                "max_tokens": 16,
                "stream": True,
                "messages": [{"role": "user", "content": f"{behavior}:{marker}"}],
                **extra,
            },
        )

    def attempts_on(self, backend: str, marker: str) -> int:
        return self.attempts.count(f"{backend}:{marker}")


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Rig]:
    attempts: Final = Attempts()
    directory: Final = tmp_path_factory.mktemp("messages-retry-budget-sources")
    with gateway_from_environment() as gateway, wire_server(_respond(attempts)) as wire:
        with owned_proxy(gateway, directory, {}, config=_config(wire, directory), workers=2) as proxy:
            yield _Rig(proxy, attempts)


def _assert_completed(response: httpx.Response, served_id: str) -> None:
    assert response.status_code == 200, response.text
    events: Final = parse_sse(response.text)
    assert event_types(events) == LIFECYCLE, events
    assert message_id(events) == served_id, events
    assert delta_text(events) == _TEXT, events


def _assert_failed_after_message_start(response: httpx.Response, served_id: str) -> None:
    assert response.status_code == 200, response.text
    events: Final = parse_sse(response.text)
    types: Final = event_types(events)
    assert types[0] == "message_start", events
    assert types[-1] == "error", events
    assert "content_block_delta" not in types, events
    assert message_id(events) == served_id, events


def test_router_num_retries_governs_a_group_without_its_own_budget(rig: _Rig) -> None:
    marker: Final = uuid.uuid4().hex
    _assert_completed(rig.stream(_ROUTER_BUDGET, "drop-once", marker), _served_id(_PRIMARY, marker, 2))
    assert rig.attempts_on(_PRIMARY, marker) == 2


def test_a_digit_string_deployment_budget_is_honored_as_a_number(rig: _Rig) -> None:
    marker: Final = uuid.uuid4().hex
    response: Final = rig.stream(_STRING_BUDGET, "drop-always", marker)
    _assert_failed_after_message_start(response, _served_id(_PRIMARY, marker, 3))
    assert rig.attempts_on(_PRIMARY, marker) == 3


def test_request_num_retries_zero_turns_the_router_budget_off(rig: _Rig) -> None:
    marker: Final = uuid.uuid4().hex
    response: Final = rig.stream(_ROUTER_BUDGET, "drop-once", marker, num_retries=0)
    _assert_failed_after_message_start(response, _served_id(_PRIMARY, marker, 1))
    assert rig.attempts_on(_PRIMARY, marker) == 1


def test_lifecycle_frames_are_held_until_content_while_pings_go_out_live(rig: _Rig) -> None:
    marker: Final = uuid.uuid4().hex
    response: Final = rig.stream(_ROUTER_BUDGET, "hold-ping", marker)
    assert response.status_code == 200, response.text
    events: Final = parse_sse(response.text)
    assert event_types(events) == ("ping", *LIFECYCLE), events
    assert message_id(events) == _served_id(_PRIMARY, marker, 1), events
    assert delta_text(events) == _TEXT, events
    assert rig.attempts_on(_PRIMARY, marker) == 1


def test_a_retry_policy_default_retries_sets_the_budget_for_a_pre_content_drop(rig: _Rig) -> None:
    marker: Final = uuid.uuid4().hex
    response: Final = rig.stream(_POLICY, "drop-always", marker)
    _assert_failed_after_message_start(response, _served_id(_PRIMARY, marker, 3))
    assert rig.attempts_on(_PRIMARY, marker) == 3


def test_request_num_retries_zero_turns_a_retry_policy_off(rig: _Rig) -> None:
    marker: Final = uuid.uuid4().hex
    response: Final = rig.stream(_POLICY, "drop-once", marker, num_retries=0)
    _assert_failed_after_message_start(response, _served_id(_PRIMARY, marker, 1))
    assert rig.attempts_on(_PRIMARY, marker) == 1


def test_fallbacks_run_only_after_the_same_group_budget_is_spent(rig: _Rig) -> None:
    marker: Final = uuid.uuid4().hex
    response: Final = rig.stream(_WITH_FALLBACKS, "drop-always", marker)
    _assert_completed(response, _served_id(_FALLBACK, marker, 1))
    assert response.headers.get("x-litellm-attempted-fallbacks") == "1", dict(response.headers)
    assert response.headers.get("x-litellm-model-group") == _FALLBACK_GROUP, dict(response.headers)
    assert rig.attempts_on(_PRIMARY, marker) == 2
    assert rig.attempts_on(_FALLBACK, marker) == 1


def test_a_retry_raising_a_context_window_error_reaches_the_context_window_fallback(rig: _Rig) -> None:
    marker: Final = uuid.uuid4().hex
    _assert_completed(rig.stream(_WITH_FALLBACKS, "drop-then-too-long", marker), _served_id(_CONTEXT_WINDOW, marker, 1))
    assert rig.attempts_on(_PRIMARY, marker) == 2
    assert rig.attempts_on(_FALLBACK, marker) == 0
    assert rig.attempts_on(_CONTEXT_WINDOW, marker) == 1


def test_a_retry_rejected_with_401_ends_the_retries_and_reaches_the_client_unchanged(rig: _Rig) -> None:
    marker: Final = uuid.uuid4().hex
    response: Final = rig.stream(_LONELY, "drop-then-401", marker)
    assert response.status_code == 401, response.text
    assert "authentication_error" in response.text, response.text
    assert "content_block_delta" not in response.text, response.text
    assert rig.attempts_on(_PRIMARY, marker) == 2


def test_overloaded_frames_whose_fallback_fails_answer_the_mapped_internal_server_error(rig: _Rig) -> None:
    marker: Final = uuid.uuid4().hex
    response: Final = rig.stream(_WITH_FALLBACKS, "overloaded-frames-fallback-503", marker)
    assert response.status_code == 500, response.text
    assert "content_block_delta" not in response.text, response.text
    assert rig.attempts_on(_PRIMARY, marker) == 2
    assert rig.attempts_on(_FALLBACK, marker) == 2
