"""Vertex Claude credentials on `/v1/messages`, and Vertex credentials supplied per request.

`/v1/messages` to a `vertex_ai/claude-*` deployment mints its Google token in
`VertexAIPartnerModelsAnthropicMessagesConfig.(a)validate_anthropic_messages_environment`
(`litellm/llms/vertex_ai/vertex_ai_partner_models/anthropic/messages/transformation.py:103-164`) through the
same `VertexBase` cache chat uses (`vertex_llm_base.py`). Round 1 pinned chat's credential forms and token
lifecycle and a few messages cases (object credentials, aliases, reuse, quota project, non-default
universe). This module pins the rest of the lifecycle on the messages route, which #51294 moves to the Rust
`auth-gcp` crate: key file paths, refresh on expiry, failed grants, outages, refresh-token grants and
per-deployment isolation.

Per-request credentials. `vertex_credentials`, `vertex_project` and `vertex_location` are on the request
body blocklist (`litellm/proxy/auth/auth_utils.py:500-520`): without an admin opt-in the proxy answers 401
"<param> is not allowed in request body" before any provider call (`_banned_param_error`, `:770`). A
deployment opts in with `configurable_clientside_auth_params: ["vertex_credentials"]` (`_is_param_allowed`,
`:211-232`), but a value starting `os.environ/` is refused even then, because only an admin may hold a secret
reference (`_reject_secret_reference_in`, `:606-620`).

Every deployment's `api_base` and `token_uri` point at loopback peers.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.wire import Reply, Request
from integration.providers.dec1_vertex4_support import bearer, vertex_echo
from integration.providers.dec1_vertex_support import (
    CLAUDE,
    JWT_BEARER,
    SA_PROJECT,
    SCOPE,
    Peer,
    TokenPlan,
    authorized_user,
    claude_path,
    form_of,
    jwt_claims,
    marker,
    path_of,
    peer,
    service_account,
    service_account_text,
    vertex,
)
from pydantic import JsonValue


def _messages(**extra: JsonValue) -> dict[str, JsonValue]:
    return {"max_tokens": 32, "messages": [{"role": "user", "content": f"creds {marker()}"}], **extra}


def _token_only(_request: Request) -> Reply:
    return Reply(status=404, body=b'{"error": "token peer only"}')


@pytest.fixture
def caller_peer() -> Iterator[Peer]:
    """A second token endpoint, standing for the caller's own service account."""
    with peer(_token_only) as owned:
        yield owned


# ---------------------------------------------------------------- credential forms on /v1/messages


def test_messages_reads_a_key_file_path_from_disk(gateway: Gateway, tmp_path: Path) -> None:
    """`vertex_credentials` given as a path to a service account file is read from disk on `/v1/messages`
    (`load_vertex_credentials_source`, `vertex_llm_base.py:203-206`; `credential-gaps.md` row 3): the
    rawPredict call carries a token minted at the file's `token_uri` and, with `vertex_project` unset, names
    the file's `project_id`."""
    key: Final = tmp_path / "claude-key.json"
    with vertex(
        gateway,
        vertex_echo,
        backend=CLAUDE,
        vertex_project=None,
        credentials=lambda base: (key.write_text(service_account_text(base)), str(key))[1],
    ) as rig:
        assert rig.post("/v1/messages", _messages()).status_code == 200
        call: Final = rig.peer.one_call()
        assert path_of(call) == claude_path(project=SA_PROJECT) + ":rawPredict"
        assert bearer(call) in rig.peer.plan.issued


def test_messages_mints_with_a_jwt_bearer_grant_for_the_cloud_platform_scope(gateway: Gateway) -> None:
    """The first `/v1/messages` call posts a JWT bearer grant to the key's `token_uri`; the assertion's
    issuer is the key's `client_email`, its audience `https://oauth2.googleapis.com/token` and its scope
    `cloud-platform`. google-auth signs the canonical Google token endpoint as the audience whatever the key's
    `token_uri` (google/oauth2/service_account.py, google-auth 2.52 in uv.lock); only the POST goes to
    `token_uri`. The Rust port mints through gcp_auth, so this is the request shape Google must keep seeing."""
    with vertex(gateway, vertex_echo, backend=CLAUDE) as rig:
        assert rig.post("/v1/messages", _messages()).status_code == 200
        grant: Final = form_of(rig.peer.token_calls()[0])
        assert grant["grant_type"] == JWT_BEARER
        claims: Final = jwt_claims(grant["assertion"])
        assert (claims["iss"], claims["aud"], claims["scope"]) == (
            f"dec1-vertex@{SA_PROJECT}.iam.gserviceaccount.com",
            "https://oauth2.googleapis.com/token",
            SCOPE,
        )


def test_messages_refreshes_an_expired_token_before_the_next_call(gateway: Gateway) -> None:
    """With tokens that expire immediately (`expires_in: 0`), the second `/v1/messages` call carries a
    newer token than the first: an INVALID cached token is refreshed before sending (`_resolve_access_token`,
    `vertex_llm_base.py:1131-1145`)."""
    with vertex(gateway, vertex_echo, backend=CLAUDE, plan=TokenPlan(expires_in=0)) as rig:
        assert rig.post("/v1/messages", _messages()).status_code == 200
        assert rig.post("/v1/messages", _messages()).status_code == 200
        first, second = (bearer(call) for call in rig.peer.model_calls())
        assert first != second
        assert {first, second} <= set(rig.peer.plan.issued)


def test_messages_with_a_rejected_grant_makes_no_vertex_call(gateway: Gateway) -> None:
    """When the token endpoint answers 400 `invalid_grant`, `/v1/messages` fails with an error status and
    rawPredict is never called. UNVERIFIED: the exact status (chat answers 500 for the same failure in
    round 1); only `>= 400` is asserted."""
    rejected: Final = Reply(status=400, body=json.dumps({"error": "invalid_grant"}).encode())
    with vertex(gateway, vertex_echo, backend=CLAUDE, plan=TokenPlan(error=rejected)) as rig:
        response: Final = rig.post("/v1/messages", _messages())
        assert response.status_code >= 400, response.text
        assert rig.peer.model_calls() == ()
        assert len(rig.peer.token_calls()) >= 1


def test_messages_recovers_after_a_token_endpoint_outage(gateway: Gateway) -> None:
    """While the token endpoint answers 503 a `/v1/messages` call fails without reaching rawPredict; once it
    answers again the next call mints and succeeds, because a failed load is never cached."""
    plan: Final = TokenPlan(error=Reply(status=503, body=b'{"error": "backendError"}'))
    with vertex(gateway, vertex_echo, backend=CLAUDE, plan=plan) as rig:
        assert rig.post("/v1/messages", _messages()).status_code >= 500
        assert rig.peer.model_calls() == ()
        plan.error = None
        assert rig.post("/v1/messages", _messages()).status_code == 200
        assert bearer(rig.peer.one_call()) in plan.issued


def test_messages_authorized_user_refreshes_with_its_refresh_token(gateway: Gateway) -> None:
    """`authorized_user` credentials on `/v1/messages` mint with `grant_type=refresh_token` carrying the
    file's `refresh_token`, `client_id` and `client_secret` (`credential-gaps.md` row 11), and the minted
    token is the rawPredict bearer.

    HARNESS (run 44176682961): not observable here. `Credentials.from_authorized_user_info` always replaces the
    file's `token_uri` with `https://oauth2.googleapis.com/token` (google/oauth2/credentials.py), and litellm
    loads authorized users through it (litellm/llms/vertex_ai/vertex_llm_base.py:319-325), so the refresh never
    reaches the peer and the CI egress guard refuses the real endpoint (500)."""
    pytest.skip("HARNESS: google-auth pins authorized_user token_uri to oauth2.googleapis.com, which CI cannot reach")
    made: Final[list[dict[str, JsonValue]]] = []

    def build(base: str) -> str:
        made.append(authorized_user(base))
        return json.dumps(made[0])

    with vertex(gateway, vertex_echo, backend=CLAUDE, credentials=build) as rig:
        assert rig.post("/v1/messages", _messages()).status_code == 200
        grant: Final = form_of(rig.peer.token_calls()[0])
        assert grant["grant_type"] == "refresh_token"
        assert grant["refresh_token"] == made[0]["refresh_token"]
        assert grant["client_id"] == made[0]["client_id"]
        assert bearer(rig.peer.one_call()) in rig.peer.plan.issued


def test_two_claude_deployments_with_different_keys_never_share_a_token(gateway: Gateway) -> None:
    """Two Vertex Claude deployments with different service accounts each send `/v1/messages` with a token
    from their own endpoint; the provider cache is keyed by credentials, not by model. Each peer numbers its
    tokens from `token-1`, so the two bearers are equal as strings; what shows they are not shared is that each
    token endpoint minted exactly once and each call carries a token its own endpoint issued."""
    with (
        vertex(gateway, vertex_echo, backend=CLAUDE) as first,
        vertex(gateway, vertex_echo, backend=CLAUDE) as second,
    ):
        assert first.post("/v1/messages", _messages()).status_code == 200
        assert second.post("/v1/messages", _messages()).status_code == 200
        assert bearer(first.peer.one_call()) in first.peer.plan.issued
        assert bearer(second.peer.one_call()) in second.peer.plan.issued
        assert (len(first.peer.token_calls()), len(second.peer.token_calls())) == (1, 1)


def test_messages_and_chat_on_one_claude_deployment_both_carry_minted_tokens(gateway: Gateway) -> None:
    """One Vertex Claude deployment serves `/v1/chat/completions` and `/v1/messages`: both rawPredict calls
    carry tokens from the deployment's key, and at most one token per route is minted for two calls each."""
    with vertex(gateway, vertex_echo, backend=CLAUDE) as rig:
        for _ in range(2):
            assert rig.chat().status_code == 200
            assert rig.post("/v1/messages", _messages()).status_code == 200
        calls: Final = rig.peer.model_calls()
        assert len(calls) == 4
        assert {bearer(call) for call in calls} <= set(rig.peer.plan.issued)
        assert len(rig.peer.token_calls()) <= 2 * max(gateway.workers, 1)


# ---------------------------------------------------------------- per-request credentials


@pytest.mark.parametrize("route", ["/v1/chat/completions", "/v1/messages"])
@pytest.mark.parametrize("param", ["vertex_credentials", "vertex_project", "vertex_location"])
def test_a_request_body_vertex_param_is_refused_without_an_opt_in(gateway: Gateway, route: str, param: str) -> None:
    """A caller that sends `vertex_credentials`, `vertex_project` or `vertex_location` in the body of a
    chat or messages request to a deployment without an opt-in gets 401 with "<param> is not allowed in
    request body" (`auth_utils.py:770`), and neither the model host nor the token endpoint is called."""
    backend: Final = CLAUDE if route == "/v1/messages" else "gemini-2.5-flash"
    with vertex(gateway, vertex_echo, backend=backend) as rig:
        body: Final = _messages() if route == "/v1/messages" else {"messages": _messages()["messages"]}
        response: Final = rig.post(route, {**body, param: "dec1-caller-value"})
        assert response.status_code == 401, response.text
        assert f"{param} is not allowed in request body" in response.text
        assert rig.peer.model_calls() == ()
        assert rig.peer.token_calls() == ()


def test_an_os_environ_reference_in_request_credentials_is_refused_even_with_the_opt_in(gateway: Gateway) -> None:
    """A deployment that allows `vertex_credentials` from the caller still refuses `vertex_credentials:
    "os.environ/VERTEXAI_CREDENTIALS"` with 401 "vertex_credentials holds a secret reference"
    (`_reject_secret_reference_in`, `auth_utils.py:606-620`), so a caller cannot make the proxy read its own
    environment. No provider call is made."""
    with vertex(gateway, vertex_echo, configurable_clientside_auth_params=["vertex_credentials"]) as rig:
        response: Final = rig.post(
            "/v1/chat/completions",
            {"messages": _messages()["messages"], "vertex_credentials": "os.environ/VERTEXAI_CREDENTIALS"},
        )
        assert response.status_code == 401, response.text
        assert "vertex_credentials holds a secret reference" in response.text
        assert rig.peer.model_calls() == ()


def test_opted_in_request_credentials_mint_at_the_callers_token_endpoint(gateway: Gateway, caller_peer: Peer) -> None:
    """With `configurable_clientside_auth_params: ["vertex_credentials"]`, a caller's service account JSON in
    the body is used instead of the deployment's: the token comes from the caller key's `token_uri` and the
    deployment's endpoint mints nothing. UNVERIFIED: assumes request kwargs override deployment
    `litellm_params` in the router merge (the merge now lives in the Rust router core)."""
    caller_key: Final = service_account_text(caller_peer.url, project="dec1-caller-project")
    with vertex(gateway, vertex_echo, configurable_clientside_auth_params=["vertex_credentials"]) as rig:
        response: Final = rig.post(
            "/v1/chat/completions", {"messages": _messages()["messages"], "vertex_credentials": caller_key}
        )
        assert response.status_code == 200, response.text
        assert bearer(rig.peer.one_call()) in caller_peer.plan.issued
        assert rig.peer.token_calls() == ()


def test_opted_in_request_credentials_on_messages_mint_at_the_callers_token_endpoint(
    gateway: Gateway, caller_peer: Peer
) -> None:
    """The same opt-in on a Vertex Claude deployment: `/v1/messages` with the caller's `vertex_credentials`
    sends a token from the caller key's endpoint. `credential-gaps.md` row 23 notes the Rust messages route
    treats every value as deployment sourced, so its `token_uri` check for request credentials never runs.
    UNVERIFIED: assumes `vertex_credentials` from the body reaches `litellm_params` on this route."""
    caller_key: Final = json.dumps(service_account(caller_peer.url, project="dec1-caller-project"))
    with vertex(
        gateway, vertex_echo, backend=CLAUDE, configurable_clientside_auth_params=["vertex_credentials"]
    ) as rig:
        response: Final = rig.post("/v1/messages", _messages(vertex_credentials=caller_key))
        assert response.status_code == 200, response.text
        assert bearer(rig.peer.one_call()) in caller_peer.plan.issued
