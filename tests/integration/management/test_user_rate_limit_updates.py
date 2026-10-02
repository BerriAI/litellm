import json
from collections.abc import Mapping
from contextlib import ExitStack
from typing import Final
from uuid import uuid4

import httpx
import pytest
from pydantic import JsonValue, TypeAdapter

from tests.integration._support.client import (
    JSON_OBJECT,
    Gateway,
    eventually,
    object_value,
    string_value,
)
from tests.integration._support.database import read_rows
from tests.integration._support.wire import Reply, Request, wire_server

_HEADER_VALUE_ADAPTER: Final[TypeAdapter[str | None]] = TypeAdapter(str | None)
_UPSTREAM_REPLIES: Final[Mapping[str, Mapping[str, JsonValue]]] = {
    "/v1/chat/completions": {
        "id": "chatcmpl_hook_isolation",
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-5.6",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    },
    "/v1/responses": {
        "id": "resp_hook_isolation",
        "object": "response",
        "created_at": 1,
        "status": "completed",
        "model": "gpt-5.6",
        "output": [
            {
                "type": "message",
                "id": "msg_hook_isolation",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "hi", "annotations": []}],
            }
        ],
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
        "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
    },
}


def _chat(proxy: Gateway, model: str, key: str) -> httpx.Response:
    return proxy.request(
        "POST",
        "/v1/chat/completions",
        {"model": model, "messages": [{"role": "user", "content": f"user rate limit probe {uuid4().hex}"}]},
        key=key,
    )


def _assert_user_rate_limit_error(response: httpx.Response, user: str, limit_type: str) -> None:
    request_limit: Final[str | None] = _HEADER_VALUE_ADAPTER.validate_python(
        response.headers.get("x-ratelimit-user-limit-requests")
    )
    token_limit: Final[str | None] = _HEADER_VALUE_ADAPTER.validate_python(
        response.headers.get("x-ratelimit-user-limit-tokens")
    )
    context: Final = (
        f"Expected a user {limit_type} limit error for {user}, received HTTP {response.status_code} with "
        f"user limits requests={request_limit}, tokens={token_limit}: {response.text}"
    )
    assert response.status_code == 429, context
    body: Final = JSON_OBJECT.validate_json(response.content)
    error: Final = object_value(body["error"])
    message: Final = string_value(error["message"])
    assert error.get("type") == "throttling_error", context
    assert message.startswith(f"Rate limit exceeded for user: {user}. Limit type: {limit_type}. Current limit: 1,"), (
        context
    )


def _route_request(proxy: Gateway, route: str, model: str, key: str, stream: bool) -> httpx.Response:
    marker: Final = f"user rpm route probe {uuid4().hex}"
    if route == "/v1/messages":
        return proxy.request(
            "POST",
            route,
            {"model": model, "max_tokens": 16, "messages": [{"role": "user", "content": marker}]},
            key=key,
            headers={"anthropic-version": "2023-06-01"},
        )
    if route == "/v1/responses":
        return proxy.request(
            "POST",
            route,
            {"model": model, "input": marker, "max_output_tokens": 16, "store": False},
            key=key,
        )
    return proxy.request(
        "POST",
        route,
        {"model": model, "messages": [{"role": "user", "content": marker}], "stream": stream},
        key=key,
    )


def _assert_route_user_requests_limit_error(response: httpx.Response, user: str, route: str) -> None:
    request_limit: Final[str | None] = _HEADER_VALUE_ADAPTER.validate_python(
        response.headers.get("x-ratelimit-user-limit-requests")
    )
    token_limit: Final[str | None] = _HEADER_VALUE_ADAPTER.validate_python(
        response.headers.get("x-ratelimit-user-limit-tokens")
    )
    context: Final = (
        f"Expected a user requests limit error for {user} on {route}, received HTTP {response.status_code} with "
        f"user limits requests={request_limit}, tokens={token_limit}: {response.text}"
    )
    assert response.status_code == 429, context
    body: Final = JSON_OBJECT.validate_json(response.content)
    error: Final = object_value(body["error"])
    assert route != "/v1/messages" or body.get("type") == "error", context
    expected_error_type: Final = "rate_limit_error" if route == "/v1/messages" else "throttling_error"
    assert error.get("type") == expected_error_type, context
    message: Final = string_value(error["message"])
    assert message.startswith(f"Rate limit exceeded for user: {user}. Limit type: requests. Current limit: 1,"), context


def _assert_user_rate_limit_on_every_proxy(
    gateway: Gateway,
    peer: Gateway,
    model: str,
    user: str,
    key: str,
) -> None:
    responses: Final = eventually(
        lambda: (_chat(gateway, model, key), _chat(peer, model, key)),
        lambda observed: all(response.status_code == 429 for response in observed),
        seconds=10,
        return_last_on_timeout=True,
    )
    context: Final = tuple(
        (
            response.status_code,
            response.headers.get("x-ratelimit-user-limit-requests"),
            response.headers.get("x-ratelimit-user-limit-tokens"),
            response.text,
        )
        for response in responses
    )
    assert tuple(response.status_code for response in responses) == (429, 429), (
        f"Expected the user RPM limit on gateway and peer for {user}, received {context!r}"
    )
    _assert_user_rate_limit_error(responses[0], user, "requests")
    _assert_user_rate_limit_error(responses[1], user, "requests")


@pytest.mark.parametrize("field", ("tpm_limit", "rpm_limit"))
def test_user_rate_limit_lowered_on_gateway_is_enforced_by_peer(gateway: Gateway, peer: Gateway, field: str) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        user: Final = scenario.user(tpm_limit=100000, rpm_limit=1000)
        key: Final = scenario.key(user_id=user, models=[model])

        gateway_warm: Final = _chat(gateway, model, key)
        peer_warm: Final = _chat(peer, model, key)
        assert gateway_warm.status_code == 200, (
            f"Gateway rejected the initial user-limited request: {gateway_warm.text}"
        )
        assert peer_warm.status_code == 200, f"Peer rejected the initial user-limited request: {peer_warm.text}"
        assert peer_warm.headers.get("x-ratelimit-user-limit-requests") == "1000", peer_warm.headers
        assert peer_warm.headers.get("x-ratelimit-user-limit-tokens") == "100000", peer_warm.headers

        gateway.post("/user/update", {"user_id": user, field: 1})

        expected_tpm: Final = 1 if field == "tpm_limit" else 100000
        expected_rpm: Final = 1 if field == "rpm_limit" else 1000
        rows: Final = read_rows(
            'SELECT tpm_limit, rpm_limit FROM "LiteLLM_UserTable" WHERE user_id = %s',
            (user,),
        )
        assert rows == [{"tpm_limit": expected_tpm, "rpm_limit": expected_rpm}], (
            f"User {field} update did not persist without changing the other limit: {rows!r}"
        )

        limit_type: Final = "tokens" if field == "tpm_limit" else "requests"
        peer_limited: Final = eventually(
            lambda: _chat(peer, model, key),
            lambda response: response.status_code == 429,
            seconds=10,
            return_last_on_timeout=True,
        )
        _assert_user_rate_limit_error(peer_limited, user, limit_type)


def test_user_rate_limit_explicit_null_clears_and_omitted_limit_is_untouched(gateway: Gateway, peer: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        user: Final = scenario.user(tpm_limit=100000, rpm_limit=1)
        key: Final = scenario.key(user_id=user, models=[model])

        gateway_warm: Final = _chat(gateway, model, key)
        assert gateway_warm.status_code == 200, f"Gateway rejected the initial request under RPM 1: {gateway_warm.text}"
        peer_limited: Final = _chat(peer, model, key)
        _assert_user_rate_limit_error(peer_limited, user, "requests")

        gateway.post("/user/update", {"user_id": user, "rpm_limit": None})

        cleared_rows: Final = read_rows(
            'SELECT tpm_limit, rpm_limit FROM "LiteLLM_UserTable" WHERE user_id = %s',
            (user,),
        )
        assert cleared_rows == [{"tpm_limit": 100000, "rpm_limit": None}], (
            f"Clearing RPM changed the wrong user limits: {cleared_rows!r}"
        )
        info: Final = gateway.get("/v2/user/info", {"user_id": user})
        assert info["tpm_limit"] == 100000, f"User info omitted or changed TPM after RPM clear: {info!r}"
        assert info["rpm_limit"] is None, f"User info did not report the cleared RPM limit: {info!r}"

        gateway_after_clear: Final = _chat(gateway, model, key)
        assert gateway_after_clear.status_code == 200, (
            f"Gateway still enforced RPM after it was cleared: {gateway_after_clear.text}"
        )
        peer_after_clear: Final = eventually(
            lambda: _chat(peer, model, key),
            lambda response: response.status_code == 200,
            seconds=10,
        )
        assert peer_after_clear.status_code == 200, (
            f"Peer did not stop enforcing RPM after it was cleared: {peer_after_clear.text}"
        )

        gateway.post("/user/update", {"user_id": user, "tpm_limit": 50000})
        omitted_rows: Final = read_rows(
            'SELECT tpm_limit, rpm_limit FROM "LiteLLM_UserTable" WHERE user_id = %s',
            (user,),
        )
        assert omitted_rows == [{"tpm_limit": 50000, "rpm_limit": None}], (
            f"Omitting RPM during the TPM update changed it: {omitted_rows!r}"
        )


@pytest.mark.parametrize(
    ("route", "stream", "upstream_target", "expected_rpm_header"),
    (
        pytest.param("/v1/messages", False, "/v1/responses", "1000", id="messages"),
        pytest.param("/v1/responses", False, "/v1/responses", "1000", id="responses"),
        pytest.param("/v1/chat/completions", True, None, None, id="streaming-chat-completions"),
    ),
)
def test_user_rpm_lowered_on_gateway_is_enforced_by_peer_on_llm_route(
    gateway: Gateway,
    peer: Gateway,
    route: str,
    stream: bool,
    upstream_target: str | None,
    expected_rpm_header: str | None,
) -> None:
    with gateway.scenario() as scenario, ExitStack() as resources:

        def upstream(request: Request) -> Reply:
            assert request.target == upstream_target, request.target
            reply: Final = _UPSTREAM_REPLIES[request.target]
            return Reply(body=json.dumps(reply).encode())

        provider: Final = resources.enter_context(wire_server(upstream)) if upstream_target is not None else None
        model: Final = (
            scenario.model()
            if provider is None
            else scenario.model(model="openai/gpt-5.6", api_base=provider.url + "/v1")
        )
        user: Final = scenario.user(tpm_limit=100000, rpm_limit=1000)
        key: Final = scenario.key(user_id=user, models=[model])

        gateway_warm: Final = _route_request(gateway, route, model, key, stream)
        peer_warm: Final = _route_request(peer, route, model, key, stream)
        assert gateway_warm.status_code == 200, f"Gateway rejected {route}: {gateway_warm.text}"
        assert peer_warm.status_code == 200, f"Peer rejected {route}: {peer_warm.text}"
        assert expected_rpm_header is None or (
            peer_warm.headers.get("x-ratelimit-user-limit-requests") == expected_rpm_header
        ), f"Peer returned unexpected user RPM headers for {route}: {dict(peer_warm.headers)!r}"
        targets: Final = tuple(request.target for request in provider.drain()) if provider is not None else ()
        expected_targets: Final = (upstream_target, upstream_target) if upstream_target is not None else ()
        assert targets == expected_targets, targets

        gateway.post("/user/update", {"user_id": user, "rpm_limit": 1})
        rows: Final = read_rows(
            'SELECT tpm_limit, rpm_limit FROM "LiteLLM_UserTable" WHERE user_id = %s',
            (user,),
        )
        assert rows == [{"tpm_limit": 100000, "rpm_limit": 1}], f"User RPM update changed the wrong limits: {rows!r}"

        peer_limited: Final = eventually(
            lambda: _route_request(peer, route, model, key, stream),
            lambda response: response.status_code == 429,
            seconds=10,
            return_last_on_timeout=True,
        )
        _assert_route_user_requests_limit_error(peer_limited, user, route)


def test_internal_user_cannot_clear_own_rpm_limit(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        user: Final = scenario.user(tpm_limit=100000, rpm_limit=1, user_role="internal_user")
        key: Final = scenario.key(user_id=user, models=[model])
        denied: Final = gateway.request(
            "POST",
            "/user/update",
            {"user_id": user, "rpm_limit": None},
            key=key,
        )
        context: Final = f"Expected internal-user route denial, received HTTP {denied.status_code}: {denied.text}"
        assert denied.status_code == 401, context
        assert "Only proxy admin can be used to generate" in denied.text, context
        assert "Route=/user/update" in denied.text, context

        rows: Final = read_rows(
            'SELECT tpm_limit, rpm_limit FROM "LiteLLM_UserTable" WHERE user_id = %s',
            (user,),
        )
        assert rows == [{"tpm_limit": 100000, "rpm_limit": 1}], f"Denied self-update changed user limits: {rows!r}"

        first_chat: Final = _chat(gateway, model, key)
        assert first_chat.status_code == 200, f"Internal-user first chat was rejected: {first_chat.text}"
        second_chat: Final = _chat(gateway, model, key)
        _assert_user_rate_limit_error(second_chat, user, "requests")


def test_bulk_update_lowered_rpm_is_enforced_on_every_proxy(gateway: Gateway, peer: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        first_user: Final = scenario.user(tpm_limit=100000, rpm_limit=1000)
        first_key: Final = scenario.key(user_id=first_user, models=[model])
        second_user: Final = scenario.user(tpm_limit=100000, rpm_limit=1000)
        second_key: Final = scenario.key(user_id=second_user, models=[model])

        warm_responses: Final = (
            _chat(gateway, model, first_key),
            _chat(peer, model, first_key),
            _chat(gateway, model, second_key),
            _chat(peer, model, second_key),
        )
        assert tuple(response.status_code for response in warm_responses) == (200, 200, 200, 200), (
            f"Expected both users to warm on gateway and peer: {tuple(response.text for response in warm_responses)!r}"
        )

        bulk_update: Final = gateway.post(
            "/user/bulk_update",
            {
                "users": [
                    {"user_id": first_user, "rpm_limit": 1},
                    {"user_id": second_user, "rpm_limit": 1},
                ]
            },
        )
        assert (
            bulk_update["total_requested"],
            bulk_update["successful_updates"],
            bulk_update["failed_updates"],
        ) == (2, 2, 0), bulk_update
        results_json: Final = bulk_update.get("results")
        assert isinstance(results_json, list), bulk_update
        results: Final = tuple(object_value(result) for result in results_json)
        assert tuple((string_value(result["user_id"]), result["success"]) for result in results) == (
            (first_user, True),
            (second_user, True),
        ), results

        rows: Final = read_rows(
            'SELECT user_id, tpm_limit, rpm_limit FROM "LiteLLM_UserTable" WHERE user_id IN (%s, %s) ORDER BY user_id',
            (first_user, second_user),
        )
        expected_rows: Final = tuple(
            {"user_id": user, "tpm_limit": 100000, "rpm_limit": 1} for user in sorted((first_user, second_user))
        )
        assert tuple(rows) == expected_rows, f"Bulk RPM update changed unexpected limits: {rows!r}"

        _assert_user_rate_limit_on_every_proxy(gateway, peer, model, first_user, first_key)
        _assert_user_rate_limit_on_every_proxy(gateway, peer, model, second_user, second_key)
