"""Provider pass-through routes under a key's `allowed_routes` and its model allowlist.

Every mapped provider prefix (`/anthropic`, `/cohere`, `/mistral`, `/gemini`, `/openai`, `/azure`, ...) is an LLM
API route (`RouteChecks.is_llm_api_route`, `_types.py:411`), so a key limited to `llm_api_routes` reaches them and
a key limited to a list of literal routes reaches only what that list names: an exact route, a route prefix, a
trailing `*` wildcard, or a `LiteLLMRoutes` group name. A refusal is 403 and never reaches the provider.

Model allowlists apply to the built-in provider routes, unlike operator-configured pass-through endpoints
(`get_model_from_request`, `auth_utils.py:2810`). The model is read from the JSON body (`model`) on Anthropic,
Cohere, Mistral and OpenAI, and from the path on Gemini (`/v1beta/models/<model>:<action>`) and on Azure when the
deployment segment names a router model group. A refused call answers 403 with
`The requested model '<model>' is not available for this API key, or the model name is invalid. Check the
models available to you and try again.` (`model_access_denied.py:5`), in the Anthropic envelope on `/anthropic/*`
with `litellm_error_type` naming the denial, and in the OpenAI envelope with `type` naming it everywhere else.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from typing import Final

import pytest
from integration._support.client import Gateway, gateway_from_environment
from integration._support.wire import Wire, wire_server
from integration.providers.dec1_passthru_support import (
    ANTHROPIC_MODEL,
    COHERE_MODEL,
    GEMINI_MODEL,
    OPENAI_MODEL,
    anthropic_body,
    anthropic_error,
    bearer,
    cohere_body,
    gemini_body,
    gemini_path,
    marker,
    message_of,
    openai_body,
    openai_error,
    posts,
    provider_proxy,
    scripted_provider,
    send,
)

pytestmark = pytest.mark.timeout(300)


def _denied(model: str) -> str:
    return (
        f"The requested model '{model}' is not available for this API key, or the model name is invalid. "
        "Check the models available to you and try again."
    )


@pytest.fixture(scope="module")
def wire() -> Iterator[Wire]:
    with wire_server(scripted_provider) as served:
        yield served


@pytest.fixture(scope="module")
def proxy(wire: Wire, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Gateway]:
    with (
        gateway_from_environment() as shared,
        provider_proxy(shared, tmp_path_factory.mktemp("dec1-passthru-access"), wire) as candidate,
    ):
        yield candidate


@pytest.fixture(autouse=True)
def _drained(wire: Wire) -> None:
    wire.drain()


def test_a_key_limited_to_llm_api_routes_reaches_every_provider_route(proxy: Gateway, wire: Wire) -> None:
    """A key with `allowed_routes: ["llm_api_routes"]` calls `/anthropic/v1/messages`, `/cohere/v2/chat` and
    `/mistral/v1/chat/completions`. All three must answer 200 and reach the provider: the mapped pass-through
    prefixes and their `/*` wildcards are members of `llm_api_routes`."""
    with proxy.scenario() as scenario:
        key: Final = scenario.key(allowed_routes=["llm_api_routes"])
        statuses: Final = [
            send(proxy, "/anthropic/v1/messages", anthropic_body(marker()), headers=bearer(key)).status_code,
            send(proxy, "/cohere/v2/chat", cohere_body(marker()), headers=bearer(key)).status_code,
            send(proxy, "/mistral/v1/chat/completions", openai_body(marker()), headers=bearer(key)).status_code,
        ]
        assert statuses == [200, 200, 200]
        assert len(posts(wire)) == 3


def test_a_key_limited_to_chat_completions_is_refused_on_a_provider_route_naming_the_route(
    proxy: Gateway, wire: Wire
) -> None:
    """A key with `allowed_routes: ["/v1/chat/completions"]` calls `/cohere/v2/chat`. It must be refused with 403
    whose message is exactly `Virtual key is not allowed to call this route. Only allowed to call routes:
    ['/v1/chat/completions']. Tried to call route: /cohere/v2/chat` (`route_checks.py:269`), and the provider must
    receive nothing."""
    with proxy.scenario() as scenario:
        key: Final = scenario.key(allowed_routes=["/v1/chat/completions"])
        response: Final = send(proxy, "/cohere/v2/chat", cohere_body(marker()), headers=bearer(key))
        error: Final = openai_error(response, 403)
        assert message_of(error) == (
            "Virtual key is not allowed to call this route. Only allowed to call routes: ['/v1/chat/completions']. "
            "Tried to call route: /cohere/v2/chat"
        ), response.text
        assert posts(wire) == ()


def test_a_route_refusal_on_the_anthropic_route_is_a_permission_error_in_the_anthropic_envelope(
    proxy: Gateway, wire: Wire
) -> None:
    """The same chat-only key calls `/anthropic/v1/messages`. The 403 must be `{"type": "error", "error":
    {"type": "permission_error", "message": "Virtual key is not allowed to call this route. ..."}}`: 403 maps to
    `permission_error` (`ANTHROPIC_ERROR_TYPE_MAP`). The provider must receive nothing."""
    with proxy.scenario() as scenario:
        key: Final = scenario.key(allowed_routes=["/v1/chat/completions"])
        response: Final = send(proxy, "/anthropic/v1/messages", anthropic_body(marker()), headers=bearer(key))
        error: Final = anthropic_error(response, 403)
        assert error["type"] == "permission_error", response.text
        assert message_of(error).startswith("Virtual key is not allowed to call this route."), response.text
        assert posts(wire) == ()


def test_a_route_prefix_entry_opens_that_provider_and_no_other(proxy: Gateway, wire: Wire) -> None:
    """A key with `allowed_routes: ["/mistral"]` must be served on `/mistral/v1/chat/completions` (an entry covers
    the route itself and everything under it) and refused with 403 on `/cohere/v2/chat`."""
    with proxy.scenario() as scenario:
        key: Final = scenario.key(allowed_routes=["/mistral"])
        served: Final = send(proxy, "/mistral/v1/chat/completions", openai_body(marker()), headers=bearer(key))
        refused: Final = send(proxy, "/cohere/v2/chat", cohere_body(marker()), headers=bearer(key))
        assert (served.status_code, refused.status_code) == (200, 403), (served.text, refused.text)
        assert len(posts(wire)) == 1


def test_a_route_prefix_entry_does_not_open_a_sibling_that_only_shares_its_letters(proxy: Gateway, wire: Wire) -> None:
    """A key with `allowed_routes: ["/openai"]` calls `/openai_passthrough/v1/chat/completions`. It must be
    refused with 403: a prefix entry matches whole path segments, never a sibling such as `/openai_passthrough`
    (`auth_route_matches_prefix`)."""
    with proxy.scenario() as scenario:
        key: Final = scenario.key(allowed_routes=["/openai"])
        response: Final = send(
            proxy, "/openai_passthrough/v1/chat/completions", openai_body(marker()), headers=bearer(key)
        )
        assert response.status_code == 403, response.text
        assert posts(wire) == ()


def test_a_wildcard_entry_opens_its_provider_and_no_other(proxy: Gateway, wire: Wire) -> None:
    """A key with `allowed_routes: ["/anthropic/*"]` must be served on `/anthropic/v1/messages` and refused with
    403 on `/mistral/v1/chat/completions`."""
    with proxy.scenario() as scenario:
        key: Final = scenario.key(allowed_routes=["/anthropic/*"])
        served: Final = send(proxy, "/anthropic/v1/messages", anthropic_body(marker()), headers=bearer(key))
        refused: Final = send(proxy, "/mistral/v1/chat/completions", openai_body(marker()), headers=bearer(key))
        assert (served.status_code, refused.status_code) == (200, 403), (served.text, refused.text)
        assert len(posts(wire)) == 1


def test_the_passthrough_routes_wildcard_group_opens_every_provider_route(proxy: Gateway, wire: Wire) -> None:
    """A key with `allowed_routes: ["passthrough_routes_wildcard"]` (a `LiteLLMRoutes` group of `<prefix>/*`
    entries) must be served on both `/cohere/v2/chat` and a Gemini `generateContent` call, and refused with 403 on
    `/v1/chat/completions`, which is not a pass-through route."""
    with proxy.scenario() as scenario:
        key: Final = scenario.key(allowed_routes=["passthrough_routes_wildcard"])
        cohere: Final = send(proxy, "/cohere/v2/chat", cohere_body(marker()), headers=bearer(key))
        gemini: Final = send(proxy, gemini_path(), gemini_body(marker()), headers={"x-goog-api-key": key})
        chat: Final = send(proxy, "/v1/chat/completions", openai_body(marker()), headers=bearer(key))
        assert (cohere.status_code, gemini.status_code, chat.status_code) == (200, 200, 403), chat.text


def test_the_mapped_pass_through_routes_group_opens_a_provider_route(proxy: Gateway, wire: Wire) -> None:
    """A key with `allowed_routes: ["mapped_pass_through_routes"]` calls `/cohere/v2/chat`. The intended answer
    is 200, as with `passthrough_routes_wildcard`. Today the group holds bare prefixes such as `/cohere`, and a
    group is matched only exactly, by `*` wildcard or by `{param}` pattern (`RouteChecks.check_route_access`), so
    no route under a prefix matches and the key is refused with 403."""
    pytest.skip("BUG: allowed_routes ['mapped_pass_through_routes'] matches no route under the listed prefixes")
    with proxy.scenario() as scenario:
        key: Final = scenario.key(allowed_routes=["mapped_pass_through_routes"])
        response: Final = send(proxy, "/cohere/v2/chat", cohere_body(marker()), headers=bearer(key))
        assert response.status_code == 200, response.text
        assert len(posts(wire)) == 1


def test_a_key_without_the_body_model_is_refused_on_the_anthropic_route(proxy: Gateway, wire: Wire) -> None:
    """A key with `models: ["dec1-only-this-model"]` posts `model: claude-sonnet-4-5` to `/anthropic/v1/messages`.
    The answer must be 403 in the Anthropic envelope with `error.type` `permission_error`, `litellm_error_type`
    `key_model_access_denied` and the client message naming `claude-sonnet-4-5`, and the provider must receive
    nothing."""
    with proxy.scenario() as scenario:
        key: Final = scenario.key(models=["dec1-only-this-model"])
        response: Final = send(proxy, "/anthropic/v1/messages", anthropic_body(marker()), headers=bearer(key))
        error: Final = anthropic_error(response, 403)
        assert error["type"] == "permission_error", response.text
        assert error.get("litellm_error_type") == "key_model_access_denied", response.text
        assert message_of(error) == _denied(ANTHROPIC_MODEL), response.text
        assert posts(wire) == ()


def test_a_key_listing_the_body_model_is_served_on_the_anthropic_route(proxy: Gateway, wire: Wire) -> None:
    """A key with `models: ["claude-sonnet-4-5"]` posts that model to `/anthropic/v1/messages`. It must be served
    with 200, which shows the model check compares the raw provider model name in the body with the key's list."""
    with proxy.scenario() as scenario:
        key: Final = scenario.key(models=[ANTHROPIC_MODEL])
        response: Final = send(proxy, "/anthropic/v1/messages", anthropic_body(marker()), headers=bearer(key))
        assert response.status_code == 200, response.text
        assert len(posts(wire)) == 1


@pytest.mark.parametrize(
    ("route", "body", "model"),
    [
        pytest.param("/cohere/v2/chat", cohere_body("dec1"), COHERE_MODEL, id="cohere"),
        pytest.param(
            "/mistral/v1/chat/completions",
            openai_body("dec1", model="mistral-large-latest"),
            "mistral-large-latest",
            id="mistral",
        ),
        pytest.param("/openai/v1/chat/completions", openai_body("dec1"), OPENAI_MODEL, id="openai"),
    ],
)
def test_a_key_without_the_body_model_is_refused_on_an_openai_envelope_route(
    proxy: Gateway, wire: Wire, route: str, body: dict[str, object], model: str
) -> None:
    """A key with `models: ["dec1-only-this-model"]` posts a body naming another model to a Cohere, Mistral or
    OpenAI pass-through route. Each must answer 403 with `error.type` `key_model_access_denied`, `error.param`
    `model` and the client message naming the body model, and the provider must receive nothing."""
    with proxy.scenario() as scenario:
        key: Final = scenario.key(models=["dec1-only-this-model"])
        response: Final = proxy.client.post(
            route, json={**body, "messages": [{"role": "user", "content": marker()}]}, headers=bearer(key)
        )
        error: Final = openai_error(response, 403)
        assert (error["type"], error["param"]) == ("key_model_access_denied", "model"), response.text
        assert message_of(error) == _denied(model), response.text
        assert posts(wire) == ()


def test_the_gemini_route_checks_the_model_named_in_the_path(proxy: Gateway, wire: Wire) -> None:
    """A key with `models: ["gemini-2.5-flash"]` calls `:generateContent` on `gemini-2.5-pro` and then on
    `gemini-2.5-flash`. The first must be refused with 403 `key_model_access_denied` naming `gemini-2.5-pro` and
    reach no provider; the second must be served. The Gemini body has no `model`, so the path is the only source
    (`_model_from_route_path`)."""
    with proxy.scenario() as scenario:
        key: Final = scenario.key(models=[GEMINI_MODEL])
        refused: Final = send(
            proxy, gemini_path("gemini-2.5-pro"), gemini_body(marker()), headers={"x-goog-api-key": key}
        )
        error: Final = openai_error(refused, 403)
        assert message_of(error) == _denied("gemini-2.5-pro"), refused.text
        assert posts(wire) == ()
        served: Final = send(proxy, gemini_path(), gemini_body(marker()), headers={"x-goog-api-key": key})
        assert served.status_code == 200, served.text
        assert len(posts(wire)) == 1


def test_a_team_without_the_body_model_refuses_its_key_on_the_anthropic_route(proxy: Gateway, wire: Wire) -> None:
    """A team has `models: ["dec1-team-only-model"]` and its key has no model list of its own. The key posts
    `claude-sonnet-4-5` to `/anthropic/v1/messages`. The answer must be 403 in the Anthropic envelope with
    `litellm_error_type` `team_model_access_denied`, and the provider must receive nothing."""
    with proxy.scenario() as scenario:
        team: Final = scenario.team(models=["dec1-team-only-model"])
        key: Final = scenario.key(team_id=team)
        response: Final = send(proxy, "/anthropic/v1/messages", anthropic_body(marker()), headers=bearer(key))
        error: Final = anthropic_error(response, 403)
        assert error.get("litellm_error_type") == "team_model_access_denied", response.text
        assert posts(wire) == ()


def test_a_model_restricted_key_may_still_list_models_through_a_provider_route(proxy: Gateway, wire: Wire) -> None:
    """A key with `models: ["dec1-only-this-model"]` sends `GET /anthropic/v1/models`, which names no model. It must
    be served with 200: with no model in the body or path there is nothing to check. UNVERIFIED: assumes
    `can_key_call_model` is skipped when `get_model_from_request` returns None."""
    with proxy.scenario() as scenario:
        key: Final = scenario.key(models=["dec1-only-this-model"])
        response: Final = send(proxy, "/anthropic/v1/models", None, headers=bearer(key), method="GET")
        assert response.status_code == 200, response.text
        wire.drain()


def test_a_router_model_group_in_the_azure_deployment_segment_is_checked_against_the_key(
    proxy: Gateway, wire: Wire
) -> None:
    """A router model group `dec1-azure-<hex>` exists and a key has `models: ["dec1-only-this-model"]`. The key
    calls `/azure/openai/deployments/dec1-azure-<hex>/chat/completions`. It must be refused with 403
    `key_model_access_denied` naming the group (`_router_model_from_azure_route`), and the provider must receive
    nothing."""
    group: Final = f"dec1-azure-{uuid.uuid4().hex[:10]}"
    with proxy.scenario() as scenario:
        scenario.model(name=group, model="azure/dec1-deployment", api_version="2024-10-21")
        key: Final = scenario.key(models=["dec1-only-this-model"])
        response: Final = send(
            proxy,
            f"/azure/openai/deployments/{group}/chat/completions",
            {"messages": [{"role": "user", "content": marker()}]},
            headers=bearer(key),
            params={"api-version": "2024-10-21"},
        )
        error: Final = openai_error(response, 403)
        assert message_of(error) == _denied(group), response.text
        assert posts(wire) == ()


def test_a_model_restricted_key_cannot_reach_an_unrouted_azure_deployment(proxy: Gateway, wire: Wire) -> None:
    """A key with `models: ["dec1-only-this-model"]` calls `/azure/openai/deployments/gpt-unlisted/chat/completions`
    where `gpt-unlisted` is no router model. Intended: 403, since the call spends the operator's Azure key on a
    model the key was never granted. Today `_model_from_route_path` anchors its `/openai/deployments/` regex at the
    start of the route, so on `/azure/...` it finds nothing, no model is checked, and the call is forwarded."""
    pytest.skip("BUG: a model-restricted key reaches any Azure deployment that is not a router model group")
    with proxy.scenario() as scenario:
        key: Final = scenario.key(models=["dec1-only-this-model"])
        response: Final = send(
            proxy,
            "/azure/openai/deployments/gpt-unlisted/chat/completions",
            {"messages": [{"role": "user", "content": marker()}]},
            headers=bearer(key),
            params={"api-version": "2024-10-21"},
        )
        assert response.status_code == 403, response.text
        assert posts(wire) == ()
