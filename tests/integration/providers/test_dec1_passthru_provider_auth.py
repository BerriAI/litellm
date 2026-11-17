"""Provider pass-through routes: which header carries the caller's key and what the provider receives instead.

The built-in routes `/anthropic/*`, `/cohere/*`, `/mistral/*`, `/gemini/*`, `/openai/*` and `/azure/*` all take a
LiteLLM virtual key from the caller and send the operator's provider credential upstream. The header that
carries the key differs by route: `/anthropic` accepts `x-api-key` or `Authorization`, `/gemini` only reads the
`key` query parameter or `x-goog-api-key` (`llm_passthrough_endpoints.py:474`), and every mapped route also
accepts the legacy `litellm_user_api_key` header (`user_api_key_auth.py:1600`). The provider must never see
the virtual key. It sees the operator key from `<PROVIDER>_API_KEY` (or from a router deployment flagged
`use_in_pass_through: true`) in the provider's own header shape, on the base URL from `<PROVIDER>_API_BASE`
with the base's path kept in front of the caller's endpoint.

Header forwarding differs by route too, and a rewrite of the front door has to keep the difference: the
Anthropic route forwards the caller's own headers minus LiteLLM secrets (`_upstream_headers_for_anthropic_route`),
while Cohere, Mistral, Gemini, OpenAI and Azure forward nothing but `x-pass-*` headers with the prefix removed
(`litellm/passthrough/utils.py:61`), and an `x-pass-` header may never set a credential.

Errors on `/anthropic/*` use the Anthropic envelope (`{"type": "error", "error": {...}}`,
`anthropic_error_payload.py:50`); every other provider route keeps the OpenAI envelope.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Iterator
from typing import Final

import pytest
from integration._support.client import Gateway, gateway_from_environment
from integration._support.wire import Wire, wire_server
from integration.providers.dec1_passthru_support import (
    ANTHROPIC_OPERATOR_KEY,
    ANTHROPIC_PREFIX,
    AZURE_OPERATOR_KEY,
    AZURE_PREFIX,
    COHERE_OPERATOR_KEY,
    COHERE_PREFIX,
    GEMINI_MODEL,
    GEMINI_OPERATOR_KEY,
    GEMINI_PREFIX,
    MISTRAL_OPERATOR_KEY,
    MISTRAL_PREFIX,
    OPENAI_OPERATOR_KEY,
    OPENAI_PREFIX,
    anthropic_body,
    anthropic_error,
    assert_key_absent,
    bearer,
    body_of,
    cohere_body,
    gemini_body,
    gemini_path,
    marker,
    message_of,
    only_post,
    openai_body,
    openai_error,
    path_of,
    posts,
    provider_proxy,
    query_of,
    scripted_provider,
    send,
)
from pydantic import JsonValue

pytestmark = pytest.mark.timeout(300)

_CREDENTIALLESS_DETAIL: Final = (
    "No Anthropic credential is configured on this proxy and the request carried no upstream "
    "Anthropic credential. The LiteLLM virtual key is not forwarded to Anthropic. Configure an "
    "Anthropic credential (ANTHROPIC_API_KEY or ANTHROPIC_AUTH_TOKEN, or a model with "
    "use_in_pass_through: true), or send your own Anthropic API key in the x-api-key header or "
    "your own Anthropic OAuth token in the Authorization header."
)


@pytest.fixture(scope="module")
def wire() -> Iterator[Wire]:
    with wire_server(scripted_provider) as served:
        yield served


@pytest.fixture(scope="module")
def proxy(wire: Wire, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Gateway]:
    with (
        gateway_from_environment() as shared,
        provider_proxy(shared, tmp_path_factory.mktemp("dec1-passthru-auth"), wire) as candidate,
    ):
        yield candidate


@pytest.fixture(scope="module")
def credentialless(wire: Wire, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Gateway]:
    """A second proxy whose Anthropic route has no operator credential at all."""
    with (
        gateway_from_environment() as shared,
        provider_proxy(
            shared,
            tmp_path_factory.mktemp("dec1-passthru-credentialless"),
            wire,
            environment={"ANTHROPIC_API_KEY": "", "ANTHROPIC_AUTH_TOKEN": ""},
        ) as candidate,
    ):
        yield candidate


@pytest.fixture(autouse=True)
def _drained(wire: Wire) -> None:
    wire.drain()


def test_an_anthropic_call_with_the_key_in_x_api_key_reaches_the_env_base_with_the_operator_key(
    proxy: Gateway, wire: Wire
) -> None:
    """A virtual key is sent in `x-api-key` to `POST /anthropic/v1/messages`. The call must answer 200 and reach
    the peer once as `POST /an/v1/messages` (the env base's `/an` path kept in front of the endpoint) with
    `x-api-key` set to the operator's `ANTHROPIC_API_KEY`, and the virtual key nowhere in the request."""
    with proxy.scenario() as scenario:
        key: Final = scenario.key()
        response: Final = send(proxy, "/anthropic/v1/messages", anthropic_body(marker()), headers={"x-api-key": key})
        assert response.status_code == 200, response.text
        upstream: Final = only_post(wire)
        assert path_of(upstream) == f"{ANTHROPIC_PREFIX}/v1/messages", upstream.target
        assert upstream.headers["x-api-key"] == ANTHROPIC_OPERATOR_KEY, upstream.headers
        assert_key_absent(upstream, key)


def test_an_anthropic_call_with_the_key_as_a_bearer_token_sends_only_the_operator_x_api_key(
    proxy: Gateway, wire: Wire
) -> None:
    """The virtual key is sent as `Authorization: Bearer <key>`. The provider request must carry the operator key
    in `x-api-key` and no `authorization` header at all: the caller's credential headers are dropped before the
    proxy's own are added (`_ANTHROPIC_UPSTREAM_CREDENTIAL_HEADERS`)."""
    with proxy.scenario() as scenario:
        key: Final = scenario.key()
        response: Final = send(proxy, "/anthropic/v1/messages", anthropic_body(marker()), headers=bearer(key))
        assert response.status_code == 200, response.text
        upstream: Final = only_post(wire)
        assert upstream.headers["x-api-key"] == ANTHROPIC_OPERATOR_KEY, upstream.headers
        assert "authorization" not in upstream.headers, upstream.headers
        assert_key_absent(upstream, key)


@pytest.mark.parametrize(
    ("route", "body_factory"),
    [
        pytest.param("/anthropic/v1/messages", anthropic_body, id="anthropic"),
        pytest.param("/cohere/v2/chat", cohere_body, id="cohere"),
        pytest.param("/mistral/v1/chat/completions", openai_body, id="mistral"),
    ],
)
def test_the_litellm_user_api_key_header_authenticates_a_provider_route_and_is_not_forwarded(
    proxy: Gateway, wire: Wire, route: str, body_factory: Callable[[str], dict[str, JsonValue]]
) -> None:
    """On a mapped pass-through route a key sent only in the `litellm_user_api_key` header is the caller's
    credential (`check_api_key_for_custom_headers_or_pass_through_endpoints`). The call must answer 200, and the
    provider request must contain neither that header nor the key's value."""
    with proxy.scenario() as scenario:
        key: Final = scenario.key()
        response: Final = send(proxy, route, body_factory(marker()), headers={"litellm_user_api_key": key})
        assert response.status_code == 200, response.text
        upstream: Final = only_post(wire)
        assert "litellm_user_api_key" not in upstream.headers, upstream.headers
        assert_key_absent(upstream, key)


def test_the_anthropic_route_forwards_the_callers_own_headers(proxy: Gateway, wire: Wire) -> None:
    """A call to `/anthropic/v1/messages` carries `anthropic-version: 2023-06-01` and a custom
    `x-dec1-trace: <marker>`. Both must reach the provider unchanged: unlike the other provider routes, the
    Anthropic route relays the caller's headers minus LiteLLM secrets and framing headers."""
    trace: Final = marker("trace")
    with proxy.scenario() as scenario:
        key: Final = scenario.key()
        headers: Final = {**bearer(key), "anthropic-version": "2023-06-01", "x-dec1-trace": trace}
        assert send(proxy, "/anthropic/v1/messages", anthropic_body(marker()), headers=headers).status_code == 200
        upstream: Final = only_post(wire)
        assert upstream.headers.get("anthropic-version") == "2023-06-01", upstream.headers
        assert upstream.headers.get("x-dec1-trace") == trace, upstream.headers


def test_an_unknown_key_on_the_anthropic_route_is_a_401_in_the_anthropic_envelope(proxy: Gateway, wire: Wire) -> None:
    """A well-formed key the proxy never issued is sent to `/anthropic/v1/messages`. The answer must be 401 with
    `{"type": "error", "error": {"type": "authentication_error", ...}}`, the message must not echo the key, and the
    provider must receive nothing."""
    unknown: Final = f"sk-dec1-passthru-{uuid.uuid4().hex}"
    response: Final = send(proxy, "/anthropic/v1/messages", anthropic_body(marker()), headers={"x-api-key": unknown})
    error: Final = anthropic_error(response, 401)
    assert error["type"] == "authentication_error", response.text
    assert unknown not in response.text
    assert posts(wire) == ()


def test_a_call_without_any_key_on_the_anthropic_route_is_a_401_in_the_anthropic_envelope(
    proxy: Gateway, wire: Wire
) -> None:
    """`POST /anthropic/v1/messages` with no credential header at all must answer 401 in the Anthropic envelope
    with `error.type` `authentication_error` and must not reach the provider."""
    response: Final = send(proxy, "/anthropic/v1/messages", anthropic_body(marker()), headers={})
    assert anthropic_error(response, 401)["type"] == "authentication_error", response.text
    assert posts(wire) == ()


def test_an_anthropic_route_error_echoes_the_callers_x_request_id(proxy: Gateway) -> None:
    """A refused `/anthropic/*` call that carried `x-request-id: <id>` must put that id at the top level of the
    error body as `request_id`, as Anthropic's own API does (`route_error_content`)."""
    request_id: Final = f"req_{uuid.uuid4().hex}"
    response: Final = send(
        proxy,
        "/anthropic/v1/messages",
        anthropic_body(marker()),
        headers={"x-api-key": f"sk-dec1-unknown-{uuid.uuid4().hex}", "x-request-id": request_id},
    )
    anthropic_error(response, 401)
    assert response.json()["request_id"] == request_id, response.text


def test_an_unknown_key_on_the_cohere_route_is_a_401_in_the_openai_envelope(proxy: Gateway, wire: Wire) -> None:
    """The same unknown key sent to `/cohere/v2/chat` must be refused with 401 in the OpenAI envelope
    (`{"error": {"type": "auth_error", ...}}`), because only `/anthropic/*` switches envelopes, and must not reach
    the provider."""
    unknown: Final = f"sk-dec1-passthru-{uuid.uuid4().hex}"
    response: Final = send(proxy, "/cohere/v2/chat", cohere_body(marker()), headers=bearer(unknown))
    error: Final = openai_error(response, 401)
    assert error["type"] == "auth_error", response.text
    assert unknown not in response.text
    assert posts(wire) == ()


def test_the_credentialless_anthropic_route_refuses_a_call_that_brings_no_anthropic_credential(
    credentialless: Gateway, wire: Wire
) -> None:
    """On a proxy with neither `ANTHROPIC_API_KEY` nor `ANTHROPIC_AUTH_TOKEN` nor a `use_in_pass_through` Anthropic
    deployment, a caller authenticating with only its virtual key must be refused with 401 in the Anthropic
    envelope and the exact `_CREDENTIALLESS_ANTHROPIC_MISSING_CREDENTIAL_DETAIL` text, and the provider must
    receive nothing. The virtual key must never be sent to Anthropic as if it were an Anthropic key."""
    with credentialless.scenario() as scenario:
        key: Final = scenario.key()
        response: Final = send(credentialless, "/anthropic/v1/messages", anthropic_body(marker()), headers=bearer(key))
        error: Final = anthropic_error(response, 401)
        assert message_of(error) == _CREDENTIALLESS_DETAIL, response.text
        assert posts(wire) == ()


def test_the_credentialless_anthropic_route_forwards_a_callers_own_x_api_key(
    credentialless: Gateway, wire: Wire
) -> None:
    """On the same credentialless proxy a caller authenticates with `x-litellm-api-key: <virtual key>` and sends
    its own Anthropic key in `x-api-key`. The provider must receive that caller key in `x-api-key` and the virtual
    key nowhere: bring-your-own-key is the only way this route can reach Anthropic."""
    caller_key: Final = f"sk-ant-dec1-caller-{uuid.uuid4().hex[:8]}"
    with credentialless.scenario() as scenario:
        key: Final = scenario.key()
        response: Final = send(
            credentialless,
            "/anthropic/v1/messages",
            anthropic_body(marker()),
            headers={"x-litellm-api-key": key, "x-api-key": caller_key},
        )
        assert response.status_code == 200, response.text
        upstream: Final = only_post(wire)
        assert upstream.headers["x-api-key"] == caller_key, upstream.headers
        assert_key_absent(upstream, key)


def test_a_cohere_call_reaches_the_env_base_under_its_path_with_the_operator_bearer(proxy: Gateway, wire: Wire) -> None:
    """`POST /cohere/v2/chat` with a virtual key must reach the peer as `POST /co/v2/chat` (the base path of
    `COHERE_API_BASE` kept) with `authorization: Bearer <COHERE_API_KEY>` and the caller's JSON body unchanged."""
    with proxy.scenario() as scenario:
        key: Final = scenario.key()
        sent: Final = cohere_body(marker())
        assert send(proxy, "/cohere/v2/chat", sent, headers=bearer(key)).status_code == 200
        upstream: Final = only_post(wire)
        assert path_of(upstream) == f"{COHERE_PREFIX}/v2/chat", upstream.target
        assert upstream.headers["authorization"] == f"Bearer {COHERE_OPERATOR_KEY}", upstream.headers
        assert body_of(upstream) == sent, upstream.body
        assert_key_absent(upstream, key)


def test_a_mistral_call_reaches_the_env_base_with_the_operator_bearer(proxy: Gateway, wire: Wire) -> None:
    """`POST /mistral/v1/chat/completions` must reach the peer as `POST /mi/v1/chat/completions` with
    `authorization: Bearer <MISTRAL_API_KEY>` (`mistral_proxy_route`), and the virtual key must not be sent."""
    with proxy.scenario() as scenario:
        key: Final = scenario.key()
        assert (
            send(proxy, "/mistral/v1/chat/completions", openai_body(marker()), headers=bearer(key)).status_code == 200
        )
        upstream: Final = only_post(wire)
        assert path_of(upstream) == f"{MISTRAL_PREFIX}/v1/chat/completions", upstream.target
        assert upstream.headers["authorization"] == f"Bearer {MISTRAL_OPERATOR_KEY}", upstream.headers
        assert_key_absent(upstream, key)


def test_an_openai_call_carries_the_operator_key_as_bearer_and_as_api_key(proxy: Gateway, wire: Wire) -> None:
    """`POST /openai/v1/chat/completions` must reach `OPENAI_API_BASE` as `/oa/v1/chat/completions` with both
    `authorization: Bearer <OPENAI_API_KEY>` and `api-key: <OPENAI_API_KEY>`: the OpenAI-shaped handler always sets
    both (`BaseOpenAIPassThroughHandler._assemble_headers`)."""
    with proxy.scenario() as scenario:
        key: Final = scenario.key()
        assert send(proxy, "/openai/v1/chat/completions", openai_body(marker()), headers=bearer(key)).status_code == 200
        upstream: Final = only_post(wire)
        assert path_of(upstream) == f"{OPENAI_PREFIX}/v1/chat/completions", upstream.target
        assert upstream.headers["authorization"] == f"Bearer {OPENAI_OPERATOR_KEY}", upstream.headers
        assert upstream.headers["api-key"] == OPENAI_OPERATOR_KEY, upstream.headers
        assert_key_absent(upstream, key)


def test_the_openai_passthrough_alias_reaches_the_same_upstream_path(proxy: Gateway, wire: Wire) -> None:
    """`/openai_passthrough/{endpoint}` is the dedicated alias of `/openai/{endpoint}`. A call to
    `/openai_passthrough/v1/chat/completions` must reach the same `/oa/v1/chat/completions` with the same operator
    bearer. UNVERIFIED: the alias is read from the route docstring; confirm the decorator in
    `openai_passthrough_endpoints.py`."""
    with proxy.scenario() as scenario:
        key: Final = scenario.key()
        response: Final = send(
            proxy, "/openai_passthrough/v1/chat/completions", openai_body(marker()), headers=bearer(key)
        )
        assert response.status_code == 200, response.text
        upstream: Final = only_post(wire)
        assert path_of(upstream) == f"{OPENAI_PREFIX}/v1/chat/completions", upstream.target
        assert upstream.headers["authorization"] == f"Bearer {OPENAI_OPERATOR_KEY}", upstream.headers


def test_an_azure_call_for_a_deployment_the_router_does_not_serve_uses_the_env_base_and_key(
    proxy: Gateway, wire: Wire
) -> None:
    """`POST /azure/openai/deployments/dec1-unrouted/chat/completions?api-version=2024-10-21` names no router model,
    so it must go to `AZURE_API_BASE` as `/az/openai/deployments/dec1-unrouted/chat/completions` with the
    `api-version` query kept and `api-key: <AZURE_API_KEY>`."""
    with proxy.scenario() as scenario:
        key: Final = scenario.key()
        response: Final = send(
            proxy,
            "/azure/openai/deployments/dec1-unrouted/chat/completions",
            {"messages": [{"role": "user", "content": marker()}]},
            headers=bearer(key),
            params={"api-version": "2024-10-21"},
        )
        assert response.status_code == 200, response.text
        upstream: Final = only_post(wire)
        assert path_of(upstream) == f"{AZURE_PREFIX}/openai/deployments/dec1-unrouted/chat/completions"
        assert query_of(upstream).get("api-version") == "2024-10-21", upstream.target
        assert upstream.headers["api-key"] == AZURE_OPERATOR_KEY, upstream.headers
        assert_key_absent(upstream, key)


def test_cohere_does_not_forward_ordinary_client_headers(proxy: Gateway, wire: Wire) -> None:
    """A Cohere call carrying `x-dec1-trace` and `x-client-name` must reach the provider without either header:
    the non-Anthropic provider routes forward no caller header unless it is `x-pass-` prefixed."""
    with proxy.scenario() as scenario:
        key: Final = scenario.key()
        headers: Final = {**bearer(key), "x-dec1-trace": marker("trace"), "x-client-name": "dec1"}
        assert send(proxy, "/cohere/v2/chat", cohere_body(marker()), headers=headers).status_code == 200
        upstream: Final = only_post(wire)
        assert "x-dec1-trace" not in upstream.headers, upstream.headers
        assert "x-client-name" not in upstream.headers, upstream.headers


def test_an_x_pass_header_is_forwarded_without_its_prefix(proxy: Gateway, wire: Wire) -> None:
    """A Mistral call carrying `x-pass-x-dec1-route: blue` must reach the provider with `x-dec1-route: blue` and
    no `x-pass-x-dec1-route` (`BasePassthroughUtils.forward_headers_from_request`)."""
    with proxy.scenario() as scenario:
        key: Final = scenario.key()
        headers: Final = {**bearer(key), "x-pass-x-dec1-route": "blue"}
        assert send(proxy, "/mistral/v1/chat/completions", openai_body(marker()), headers=headers).status_code == 200
        upstream: Final = only_post(wire)
        assert upstream.headers.get("x-dec1-route") == "blue", upstream.headers
        assert "x-pass-x-dec1-route" not in upstream.headers, upstream.headers


@pytest.mark.parametrize("protected", ["authorization", "api-key", "x-api-key", "x-goog-api-key", "x-amz-date"])
def test_an_x_pass_header_cannot_set_a_credential_header(proxy: Gateway, wire: Wire, protected: str) -> None:
    """A Cohere call sends `x-pass-<credential header>: caller-forged`. The provider must not see the forged value
    in that header, and `authorization` must still be the operator bearer (`_PASS_THROUGH_PROTECTED_HEADERS` and the
    `x-amz-` prefix block)."""
    with proxy.scenario() as scenario:
        key: Final = scenario.key()
        headers: Final = {**bearer(key), f"x-pass-{protected}": "caller-forged"}
        assert send(proxy, "/cohere/v2/chat", cohere_body(marker()), headers=headers).status_code == 200
        upstream: Final = only_post(wire)
        assert upstream.headers.get(protected) != "caller-forged", upstream.headers
        assert upstream.headers["authorization"] == f"Bearer {COHERE_OPERATOR_KEY}", upstream.headers


def test_the_gemini_key_query_parameter_authenticates_and_is_replaced_by_the_operator_key(
    proxy: Gateway, wire: Wire
) -> None:
    """`POST /gemini/v1beta/models/gemini-2.5-flash:generateContent?key=<virtual key>` must answer 200 and reach
    `/ge/v1beta/models/gemini-2.5-flash:generateContent` with `key=<GEMINI_API_KEY>` in the query: the caller's
    `key` is the LiteLLM credential and is overwritten before the call (`merged_params.update({"key": ...})`)."""
    with proxy.scenario() as scenario:
        key: Final = scenario.key()
        response: Final = send(proxy, gemini_path(), gemini_body(marker()), headers={}, params={"key": key})
        assert response.status_code == 200, response.text
        upstream: Final = only_post(wire)
        assert path_of(upstream) == f"{GEMINI_PREFIX}/v1beta/models/{GEMINI_MODEL}:generateContent"
        assert query_of(upstream)["key"] == GEMINI_OPERATOR_KEY, upstream.target
        assert_key_absent(upstream, key)


def test_the_gemini_x_goog_api_key_header_authenticates_and_is_not_forwarded(proxy: Gateway, wire: Wire) -> None:
    """The virtual key is sent as `x-goog-api-key`. The call must answer 200, the provider query must carry the
    operator `key`, and the provider must not receive an `x-goog-api-key` header: Gemini forwards no caller
    header."""
    with proxy.scenario() as scenario:
        key: Final = scenario.key()
        response: Final = send(proxy, gemini_path(), gemini_body(marker()), headers={"x-goog-api-key": key})
        assert response.status_code == 200, response.text
        upstream: Final = only_post(wire)
        assert "x-goog-api-key" not in upstream.headers, upstream.headers
        assert query_of(upstream)["key"] == GEMINI_OPERATOR_KEY, upstream.target
        assert_key_absent(upstream, key)


def test_a_gemini_call_with_only_an_authorization_header_is_refused(proxy: Gateway, wire: Wire) -> None:
    """The Gemini route reads the key only from `?key=` or `x-goog-api-key` and authenticates the literal
    `Bearer None` when both are missing. A call with a valid key in `Authorization: Bearer` alone must therefore be
    refused with 401 and reach no provider. UNVERIFIED: depends on `user_api_key_auth(api_key=...)` not falling
    back to the request's own `Authorization` header; if it does, the route accepts it and this test should be
    flipped."""
    with proxy.scenario() as scenario:
        key: Final = scenario.key()
        response: Final = send(proxy, gemini_path(), gemini_body(marker()), headers=bearer(key))
        assert response.status_code == 401, response.text
        assert posts(wire) == ()


def test_the_callers_query_string_reaches_the_provider(proxy: Gateway, wire: Wire) -> None:
    """`GET /cohere/v1/models?page_size=5&endpoint=chat` must reach the provider as a `GET` on `/co/v1/models`
    with both query parameters unchanged."""
    with proxy.scenario() as scenario:
        key: Final = scenario.key()
        response: Final = send(
            proxy,
            "/cohere/v1/models",
            None,
            headers=bearer(key),
            method="GET",
            params={"page_size": "5", "endpoint": "chat"},
        )
        assert response.status_code == 200, response.text
        received: Final = [request for request in wire.drain() if path_of(request) == f"{COHERE_PREFIX}/v1/models"]
        assert len(received) == 1, received
        assert received[0].method == "GET"
        assert query_of(received[0]) == {"page_size": "5", "endpoint": "chat"}, received[0].target


def test_a_get_is_answered_with_the_providers_body_unchanged(proxy: Gateway, wire: Wire) -> None:
    """`GET /anthropic/v1/models` must answer 200 with exactly the JSON the provider returned, here
    `{"object": "list", "data": [], "path": "/an/v1/models"}`."""
    with proxy.scenario() as scenario:
        key: Final = scenario.key()
        response: Final = send(proxy, "/anthropic/v1/models", None, headers=bearer(key), method="GET")
        assert response.status_code == 200, response.text
        expected: Final[dict[str, JsonValue]] = {"object": "list", "data": [], "path": f"{ANTHROPIC_PREFIX}/v1/models"}
        assert response.json() == expected
        wire.drain()


def test_a_blocked_key_is_refused_on_a_provider_route(proxy: Gateway, wire: Wire) -> None:
    """A key that was blocked with `POST /key/block` calls `/mistral/v1/chat/completions`. The answer must be 401
    and the provider must receive nothing; blocking applies to pass-through exactly as to `/chat/completions`."""
    with proxy.scenario() as scenario:
        key: Final = scenario.key()
        proxy.post("/key/block", {"key": key})
        response: Final = send(proxy, "/mistral/v1/chat/completions", openai_body(marker()), headers=bearer(key))
        assert response.status_code == 401, response.text
        assert posts(wire) == ()


def test_a_use_in_pass_through_deployment_key_replaces_the_env_key_but_not_the_env_base(
    proxy: Gateway, wire: Wire
) -> None:
    """A router deployment `anthropic/claude-sonnet-4-5` with `api_key: dec1-deployment-anthropic-key`,
    `api_base: http://127.0.0.1:9/unused` and `use_in_pass_through: true` is added. An `/anthropic/v1/messages`
    call must then send `x-api-key: dec1-deployment-anthropic-key` (deployment credentials win over
    `ANTHROPIC_API_KEY`, `PassthroughEndpointRouter.get_credentials`) yet still go to `ANTHROPIC_API_BASE`: the
    Anthropic route takes its base only from the environment (`anthropic_pass_through_api_base`)."""
    with proxy.scenario() as scenario:
        scenario.model(
            model="anthropic/claude-sonnet-4-5",
            api_key="dec1-deployment-anthropic-key",
            api_base="http://127.0.0.1:9/unused",
            use_in_pass_through=True,
        )
        key: Final = scenario.key()
        assert send(proxy, "/anthropic/v1/messages", anthropic_body(marker()), headers=bearer(key)).status_code == 200
        upstream: Final = only_post(wire)
        assert upstream.headers["x-api-key"] == "dec1-deployment-anthropic-key", upstream.headers
        assert path_of(upstream) == f"{ANTHROPIC_PREFIX}/v1/messages", upstream.target


def test_a_deployment_without_use_in_pass_through_does_not_lend_its_key(proxy: Gateway, wire: Wire) -> None:
    """A Cohere router deployment with `api_key: dec1-not-for-passthrough` but no `use_in_pass_through` flag is
    added. A `/cohere/v2/chat` call must still send the env `COHERE_API_KEY`: only flagged deployments lend their
    credentials to pass-through (`_resolve_matching_deployment_credentials`)."""
    with proxy.scenario() as scenario:
        scenario.model(model="cohere_chat/command-a-03-2025", api_key="dec1-not-for-passthrough")
        key: Final = scenario.key()
        assert send(proxy, "/cohere/v2/chat", cohere_body(marker()), headers=bearer(key)).status_code == 200
        upstream: Final = only_post(wire)
        assert upstream.headers["authorization"] == f"Bearer {COHERE_OPERATOR_KEY}", upstream.headers
