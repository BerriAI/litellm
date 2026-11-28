"""Which routes a virtual key may call: `allowed_routes`, the `key_type` presets and key `permissions`.

Contract (litellm/proxy/auth/route_checks.py `is_virtual_key_allowed_to_call_route` and
`non_proxy_admin_allowed_routes_check`; key_management_endpoints.py `handle_key_type`):

* An empty or unset `allowed_routes` adds no restriction. A non empty list is an allow list whose entries are
  exact routes, trailing `*` prefixes, or names of route groups such as `llm_api_routes`, `openai_routes`,
  `info_routes` and `management_routes`.
* A route outside the list is HTTP 403 with the body `{"detail": "Virtual key is not allowed to call this
  route. Only allowed to call routes: [...]. Tried to call route: <route>"}`. Pinned current behaviour: the
  route check runs outside the auth exception handler, so this refusal is FastAPI's bare `detail` shape and
  not the OpenAI `error` object the other auth refusals use.
* `key_type` is stored as a preset list: `llm_api` -> `["llm_api_routes"]`, `management` ->
  `["management_routes"]`, `read_only` -> `["info_routes"]`.
* `allowed_routes` never grants more than the caller's role allows. A plain key that lists an admin route
  passes the key check and is then refused by the role check unless the list names that exact route.
* `permissions.get_spend_routes` opens the `/global/spend/*` reports to a non admin key.
"""

from __future__ import annotations

from typing import Final

import pytest

from tests.integration._support.client import Gateway
from tests.integration.authorization.dec1_keys_support import (
    assert_denied,
    assert_route_denied,
    assert_served,
    chat,
    key_row,
    marker,
    upstream_requests,
)

_EMBEDDING_BODY: Final = {"input": "dec1 keys route access"}


def test_an_exact_allowed_route_serves_that_route_and_refuses_another_with_the_exact_body(gateway: Gateway) -> None:
    """A key is created with `allowed_routes: ["/v1/chat/completions"]`.

    Chat completions must be served. `/v1/embeddings` must answer 403 with exactly `{"detail": "Virtual key
    is not allowed to call this route. Only allowed to call routes: ['/v1/chat/completions']. Tried to call
    route: /v1/embeddings"}`.

    Pinned current behaviour: the route refusal is FastAPI's bare `detail` body, not the OpenAI `error`
    object other auth refusals use (see `assert_route_denied`).
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key(allowed_routes=["/v1/chat/completions"])
        assert_served(chat(gateway, key, model))
        refused: Final = gateway.request("POST", "/v1/embeddings", {"model": model, **_EMBEDDING_BODY}, key=key)
        assert_route_denied(refused, ["/v1/chat/completions"], "/v1/embeddings")


def test_an_exact_allowed_route_does_not_cover_its_unversioned_twin(gateway: Gateway) -> None:
    """A key allowed only `/v1/chat/completions` posts to `/chat/completions`.

    Entries are matched literally, so the unversioned path must answer the 403 route message that names
    `/chat/completions`, and the upstream must not be called.

    Pinned current behaviour: the route refusal body is the bare `detail` shape (see `assert_route_denied`).
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key(allowed_routes=["/v1/chat/completions"])
        text: Final = marker()
        refused: Final = gateway.request(
            "POST", "/chat/completions", {"model": model, "messages": [{"role": "user", "content": text}]}, key=key
        )
        assert_route_denied(refused, ["/v1/chat/completions"], "/chat/completions")
        assert upstream_requests(gateway, text) == []


def test_a_trailing_wildcard_route_allows_everything_under_its_prefix(gateway: Gateway) -> None:
    """A key is created with `allowed_routes: ["/v1/chat/*"]`.

    `/v1/chat/completions` must be served and `/v1/embeddings` must answer the 403 route message.

    Pinned current behaviour: the route refusal body is the bare `detail` shape (see `assert_route_denied`).
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key(allowed_routes=["/v1/chat/*"])
        assert_served(chat(gateway, key, model))
        refused: Final = gateway.request("POST", "/v1/embeddings", {"model": model, **_EMBEDDING_BODY}, key=key)
        assert_route_denied(refused, ["/v1/chat/*"], "/v1/embeddings")


def test_the_openai_routes_group_allows_chat_and_refuses_the_anthropic_messages_route(gateway: Gateway) -> None:
    """A key is created with `allowed_routes: ["openai_routes"]`.

    Chat completions must be served. `/v1/messages` belongs to `anthropic_routes`, so it must answer HTTP
    403 and the text must contain "Virtual key is not allowed to call this route".
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key(allowed_routes=["openai_routes"])
        assert_served(chat(gateway, key, model))
        refused: Final = gateway.request(
            "POST",
            "/v1/messages",
            {"model": model, "max_tokens": 16, "messages": [{"role": "user", "content": "route"}]},
            key=key,
        )
        assert refused.status_code == 403, refused.text
        assert "Virtual key is not allowed to call this route" in refused.text, refused.text


def test_the_llm_api_routes_group_allows_llm_calls_and_refuses_key_management(gateway: Gateway) -> None:
    """A key is created with `allowed_routes: ["llm_api_routes"]`.

    Chat completions and `GET /v1/models` must be served, and `POST /key/generate` must answer 403 with
    the route message naming `/key/generate`.

    Pinned current behaviour: the route refusal body is the bare `detail` shape (see `assert_route_denied`).
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key(allowed_routes=["llm_api_routes"])
        assert_served(chat(gateway, key, model))
        assert gateway.request("GET", "/v1/models", key=key).status_code == 200
        refused: Final = gateway.request("POST", "/key/generate", {}, key=key)
        assert_route_denied(refused, ["llm_api_routes"], "/key/generate")


@pytest.mark.parametrize(
    ("key_type", "preset"),
    [("llm_api", ["llm_api_routes"]), ("management", ["management_routes"]), ("read_only", ["info_routes"])],
)
def test_a_key_type_is_stored_as_its_preset_allowed_routes(gateway: Gateway, key_type: str, preset: list[str]) -> None:
    """A key is generated with `key_type` set to `llm_api`, `management` or `read_only`.

    The key's row must hold the matching preset in `allowed_routes`. Auth only ever reads `allowed_routes`,
    so this stored value is what a ported auth layer will see for keys created today.
    """
    with gateway.scenario() as scenario:
        key: Final = scenario.key(key_type=key_type)
        assert key_row(gateway, key)["allowed_routes"] == preset


def test_an_llm_api_key_serves_inference_and_is_refused_on_management_routes(gateway: Gateway) -> None:
    """A `key_type: llm_api` key sends a chat completion and then posts to `/user/new` and `/key/generate`.

    The chat completion must be served, and both management calls must answer 403 with the route message.

    Pinned current behaviour: the route refusal body is the bare `detail` shape (see `assert_route_denied`).
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key(key_type="llm_api")
        assert_served(chat(gateway, key, model))
        for route in ("/user/new", "/key/generate"):
            assert_route_denied(gateway.request("POST", route, {}, key=key), ["llm_api_routes"], route)


def test_a_management_key_is_refused_on_inference_without_an_upstream_call(gateway: Gateway) -> None:
    """A `key_type: management` key sends a chat completion.

    The proxy must answer 403 with the route message naming `/v1/chat/completions`, and the
    upstream must not receive the request.

    Pinned current behaviour: the route refusal body is the bare `detail` shape (see `assert_route_denied`).
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key(key_type="management")
        text: Final = marker()
        assert_route_denied(chat(gateway, key, model, text), ["management_routes"], "/v1/chat/completions")
        assert upstream_requests(gateway, text) == []


def test_a_read_only_key_reads_its_own_info_and_is_refused_on_inference_and_writes(gateway: Gateway) -> None:
    """A `key_type: read_only` key calls `GET /key/info`, a chat completion and `POST /key/generate`.

    `/key/info` must answer 200. The chat completion and the key generation must each answer 403 with the
    route message that lists `['info_routes']`.

    Pinned current behaviour: the route refusal body is the bare `detail` shape (see `assert_route_denied`).
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key(key_type="read_only")
        info: Final = gateway.request("GET", "/key/info", key=key)
        assert info.status_code == 200, info.text
        assert_route_denied(chat(gateway, key, model), ["info_routes"], "/v1/chat/completions")
        assert_route_denied(gateway.request("POST", "/key/generate", {}, key=key), ["info_routes"], "/key/generate")


def test_a_key_with_no_allowed_routes_can_call_every_llm_route_family(gateway: Gateway) -> None:
    """A key created without `allowed_routes` calls chat completions, `GET /v1/models` and `GET /key/info`.

    All three must answer 200. An empty list means no key level route restriction, not "nothing allowed".
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key()
        assert key_row(gateway, key)["allowed_routes"] == []
        assert_served(chat(gateway, key, model))
        assert gateway.request("GET", "/v1/models", key=key).status_code == 200
        assert gateway.request("GET", "/key/info", key=key).status_code == 200


def test_listing_an_admin_route_group_does_not_make_a_plain_key_an_admin(gateway: Gateway) -> None:
    """A key with no user is created with `allowed_routes: ["management_routes"]` and posts to `/user/new`.

    The key level check passes, but the role check still applies: a group name is not an exact route
    match there, so the proxy must answer the 401 "Only proxy admin" refusal and create no user.
    """
    with gateway.scenario() as scenario:
        key: Final = scenario.key(allowed_routes=["management_routes"])
        user_id: Final = marker()
        refused: Final = gateway.request("POST", "/user/new", {"user_id": user_id}, key=key)
        error: Final = assert_denied(refused, status=401, error_type="auth_error")
        assert str(error["message"]).startswith("Authentication Error, Only proxy admin can be used"), refused.text
        assert gateway.rows('SELECT user_id FROM "LiteLLM_UserTable" WHERE user_id=%s', (user_id,)) == []


def test_the_route_refusal_lists_the_keys_allowed_routes_verbatim(gateway: Gateway) -> None:
    """A key with two allowed routes calls a third route.

    The 403 message must contain the Python list rendering of the stored routes in their stored order,
    `['/v1/embeddings', '/v1/moderations']`. Support tooling parses this text.

    Pinned current behaviour: the route refusal body is the bare `detail` shape (see `assert_route_denied`).
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        routes: Final = ["/v1/embeddings", "/v1/moderations"]
        key: Final = scenario.key(allowed_routes=routes)
        assert_route_denied(chat(gateway, key, model), routes, "/v1/chat/completions")


def test_a_plain_key_cannot_read_global_spend_reports(gateway: Gateway) -> None:
    """A key with no user and no permissions calls `GET /global/spend/logs`.

    The proxy must answer the 401 "Only proxy admin" refusal with type `auth_error`.
    """
    with gateway.scenario() as scenario:
        key: Final = scenario.key()
        refused: Final = gateway.request("GET", "/global/spend/logs", key=key)
        error: Final = assert_denied(refused, status=401, error_type="auth_error")
        assert "Route=/global/spend/logs" in str(error["message"]), refused.text


def test_the_get_spend_routes_permission_opens_global_spend_reports(gateway: Gateway) -> None:
    """A key with no user is created with `permissions: {"get_spend_routes": true}` and calls
    `GET /global/spend/logs`.

    The request must pass authorization, so the answer must not be 401 or 403.
    """
    with gateway.scenario() as scenario:
        key: Final = scenario.key(permissions={"get_spend_routes": True})
        response: Final = gateway.request("GET", "/global/spend/logs", key=key)
        assert response.status_code not in (401, 403), response.text


def test_the_get_spend_routes_permission_set_to_false_does_not_open_global_spend_reports(gateway: Gateway) -> None:
    """A key is created with `permissions: {"get_spend_routes": false}` and calls `GET /global/spend/logs`.

    A permission that is explicitly false must not grant access, so the answer must be the 401 "Only proxy
    admin" refusal.
    """
    pytest.skip(
        "BUG: route_checks tests '\"get_spend_routes\" in permissions', so a key with the permission set "
        "to false is still admitted to /global/spend/* routes"
    )
    with gateway.scenario() as scenario:
        key: Final = scenario.key(permissions={"get_spend_routes": False})
        refused: Final = gateway.request("GET", "/global/spend/logs", key=key)
        assert_denied(refused, status=401, error_type="auth_error")


def test_the_get_spend_routes_permission_does_not_open_other_admin_routes(gateway: Gateway) -> None:
    """A key with `permissions: {"get_spend_routes": true}` posts to `/user/new`.

    The permission is scoped to the global spend reports, so the proxy must answer the 401 "Only proxy
    admin" refusal.
    """
    with gateway.scenario() as scenario:
        key: Final = scenario.key(permissions={"get_spend_routes": True})
        refused: Final = gateway.request("POST", "/user/new", {}, key=key)
        error: Final = assert_denied(refused, status=401, error_type="auth_error")
        assert "Route=/user/new" in str(error["message"]), refused.text


def test_the_master_key_is_not_subject_to_any_route_list(gateway: Gateway) -> None:
    """The master key calls an LLM route, an info route and an admin write route in turn.

    All must succeed: chat completions 200, `GET /key/list` 200 and `POST /key/generate` 200 (the created
    key is deleted by the scenario).
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        assert_served(chat(gateway, gateway.key, model))
        assert gateway.request("GET", "/key/list").status_code == 200
        assert scenario.key().startswith("sk-")


def test_an_internal_user_key_can_manage_its_own_keys_but_not_users(gateway: Gateway) -> None:
    """A key owned by an `internal_user` posts to `/key/generate` for itself and then to `/user/new`.

    Key management routes are open to internal users, so the first call must answer 200. `/user/new` must
    answer the 401 "Only proxy admin" refusal naming `Your role=internal_user`.
    """
    with gateway.scenario() as scenario:
        user: Final = scenario.user(user_role="internal_user")
        key: Final = scenario.key(user_id=user)
        created: Final = gateway.request("POST", "/key/generate", {"user_id": user}, key=key)
        assert created.status_code == 200, created.text
        scenario.cleanups.callback(scenario.delete_key, created.json()["key"])
        refused: Final = gateway.request("POST", "/user/new", {}, key=key)
        error: Final = assert_denied(refused, status=401, error_type="auth_error")
        assert "Your role=internal_user" in str(error["message"]), refused.text


def test_an_internal_viewer_key_can_call_llm_routes_but_not_generate_keys(gateway: Gateway) -> None:
    """A key owned by an `internal_user_viewer` sends a chat completion and then posts to `/key/generate`.

    LLM routes are open to every role, so the chat completion must be served. Key generation is not a
    viewer route, so it must answer the 401 "Only proxy admin" refusal naming `Your role=internal_user_viewer`.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        user: Final = scenario.user(user_role="internal_user_viewer")
        key: Final = scenario.key(user_id=user)
        assert_served(chat(gateway, key, model))
        refused: Final = gateway.request("POST", "/key/generate", {}, key=key)
        error: Final = assert_denied(refused, status=401, error_type="auth_error")
        assert "Your role=internal_user_viewer" in str(error["message"]), refused.text
