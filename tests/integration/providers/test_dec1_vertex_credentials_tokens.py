"""How the proxy turns a deployment's Vertex credentials into Google access tokens.

Every deployment here names a service account (or an authorized user) whose `token_uri` is the owned
peer's `/_oauth/token`, so google-auth's token exchange is observable and scriptable and never reaches
`oauth2.googleapis.com`. The Python contract (`litellm/llms/vertex_ai/vertex_llm_base.py`):

The first request loads the credentials and mints a token with a JWT bearer grant (`grant_type=
urn:ietf:params:oauth:grant-type:jwt-bearer`, an assertion signed by the key, scope `cloud-platform`).
Tokens are cached per credential identity and project. A FRESH token is reused, a STALE one (inside
google-auth's 3m45s refresh threshold) is served while a background refresh runs, and an INVALID
(expired) one is refreshed before the request is sent. A key file is identified by path plus inode,
mtime and size, so rotating the file mints with the new key, while a replacement that does not load
keeps the earlier credentials serving. Project resolution falls back from `vertex_project` to the
service account's `project_id`, or an authorized user's `quota_project_id`.

The Rust `auth-gcp` crate (draft PR #51294 and `credential-gaps.md`) reimplements this with gcp_auth
and google-cloud-auth caches, so these tests are the parity contract for token reuse, refresh timing
and project resolution.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Final

import pytest
from integration._support.client import Gateway, eventually
from integration._support.wire import Reply
from integration.providers.dec1_vertex_support import (
    CLAUDE,
    JWT_BEARER,
    PROJECT,
    QUOTA_PROJECT,
    SA_PROJECT,
    SCOPE,
    TokenPlan,
    authorized_user,
    claude_echo,
    claude_path,
    form_of,
    gemini_echo,
    jwt_claims,
    marker,
    models_path,
    path_of,
    service_account,
    service_account_text,
    vertex,
)

# google-auth signs every service account JWT grant for this audience, whatever the key's `token_uri`.
GOOGLE_TOKEN_AUDIENCE: Final = "https://oauth2.googleapis.com/token"


def _bearer_tokens(calls: tuple[object, ...]) -> list[str]:
    return [call.headers["authorization"].removeprefix("Bearer ") for call in calls]


# ---------------------------------------------------------------- minting


def test_the_first_request_mints_a_token_with_a_signed_jwt_bearer_grant(gateway: Gateway) -> None:
    """The first Gemini call makes the proxy POST the service account's `token_uri` with form fields
    `grant_type=urn:ietf:params:oauth:grant-type:jwt-bearer` and an `assertion` whose claims name the
    service account as `iss`, `https://oauth2.googleapis.com/token` as `aud` and
    `https://www.googleapis.com/auth/cloud-platform` as `scope`. google-auth signs the canonical Google token
    endpoint as the audience whatever the key's `token_uri` (`_make_authorization_grant_assertion`,
    google/oauth2/service_account.py, google-auth 2.52 in uv.lock); only the POST goes to `token_uri`."""
    with vertex(gateway, gemini_echo) as rig:
        assert rig.chat().status_code == 200
        token_call: Final = rig.peer.token_calls()[0]
        form: Final = form_of(token_call)
        assert form["grant_type"] == JWT_BEARER
        claims: Final = jwt_claims(form["assertion"])
        assert claims["iss"] == f"dec1-vertex@{SA_PROJECT}.iam.gserviceaccount.com"
        assert claims["aud"] == GOOGLE_TOKEN_AUDIENCE
        assert claims["scope"] == SCOPE


def test_the_minted_token_is_the_bearer_on_the_model_call(gateway: Gateway) -> None:
    """Whatever the token endpoint returns as `access_token` is exactly what the model call carries as
    `Authorization: Bearer <token>`; nothing else (no `x-goog-api-key`, no caller key) is added."""
    with vertex(gateway, gemini_echo) as rig:
        assert rig.chat().status_code == 200
        tokens: Final = _bearer_tokens(rig.peer.model_calls())
        assert len(tokens) == 1 and tokens[0] in rig.peer.plan.issued
        assert gateway.key not in tokens[0]


def test_a_fresh_token_is_reused_across_requests(gateway: Gateway) -> None:
    """With `expires_in: 3600`, four requests with distinct prompts mint at most one token per proxy
    worker, and every model call carries an issued token. A rewrite that mints per request would
    multiply Google token traffic by the request rate."""
    with vertex(gateway, gemini_echo, plan=TokenPlan(expires_in=3600)) as rig:
        for _ in range(4):
            assert rig.chat().status_code == 200
        assert len(rig.peer.model_calls()) == 4
        assert 1 <= len(rig.peer.token_calls()) <= max(gateway.workers, 1)
        assert set(_bearer_tokens(rig.peer.model_calls())) <= set(rig.peer.plan.issued)


def test_an_expired_token_is_refreshed_before_the_next_request(gateway: Gateway) -> None:
    """With `expires_in: 0` every cached token is INVALID by the next request, so the proxy blocks on a
    refresh and the second model call carries a newer token than the first."""
    with vertex(gateway, gemini_echo, plan=TokenPlan(expires_in=0)) as rig:
        assert rig.chat().status_code == 200
        assert rig.chat().status_code == 200
        first, second = _bearer_tokens(rig.peer.model_calls())
        assert first != second
        assert {first, second} <= set(rig.peer.plan.issued)


def test_a_token_inside_the_refresh_threshold_is_served_then_refreshed_in_the_background(gateway: Gateway) -> None:
    """A token with `expires_in: 120` is STALE (inside google-auth's 3m45s threshold) but usable: the
    request is sent with it, a background refresh then mints another token without any further model
    call, and the next request carries a token issued after the first one."""
    with vertex(gateway, gemini_echo, plan=TokenPlan(expires_in=120)) as rig:
        assert rig.chat().status_code == 200
        first: Final = _bearer_tokens(rig.peer.model_calls())[0]
        eventually(lambda: len(rig.peer.token_calls()), lambda count: count >= 2, seconds=20)
        assert len(rig.peer.model_calls()) == 1
        assert rig.chat().status_code == 200
        later: Final = _bearer_tokens(rig.peer.model_calls())[1]
        assert rig.peer.plan.issued.index(later) > rig.peer.plan.issued.index(first)


def test_two_deployments_with_different_keys_mint_their_own_tokens(gateway: Gateway) -> None:
    """Two deployments on two peers with two different service accounts each get a token from their own
    token endpoint; neither model call carries the other deployment's token."""
    with vertex(gateway, gemini_echo) as first, vertex(gateway, gemini_echo) as second:
        assert first.chat().status_code == 200
        assert second.chat().status_code == 200
        assert len(first.peer.token_calls()) >= 1 and len(second.peer.token_calls()) >= 1
        assert _bearer_tokens(first.peer.model_calls())[0] in first.peer.plan.issued
        assert _bearer_tokens(second.peer.model_calls())[0] in second.peer.plan.issued


def test_a_rejected_grant_fails_the_request_without_a_model_call(gateway: Gateway) -> None:
    """When the token endpoint answers 400 `{"error": "invalid_grant"}` the caller gets an error status
    and the model endpoint is never called. UNVERIFIED: the exact status (expected 500, since the
    google-auth RefreshError has no HTTP status of its own)."""
    rejected: Final = Reply(
        status=400, body=json.dumps({"error": "invalid_grant", "error_description": "Invalid JWT Signature."}).encode()
    )
    with vertex(gateway, gemini_echo, plan=TokenPlan(error=rejected)) as rig:
        response: Final = rig.chat()
        assert response.status_code >= 400, response.text
        assert response.status_code == 500, response.text
        assert rig.peer.model_calls() == ()
        assert len(rig.peer.token_calls()) >= 1


def test_a_token_endpoint_outage_fails_the_request_and_the_next_one_recovers(gateway: Gateway) -> None:
    """While the token endpoint answers 503 the caller gets an error and no model call is made; once it
    answers again the next request mints a token and succeeds, because failed loads are not cached."""
    plan: Final = TokenPlan(error=Reply(status=503, body=b'{"error": "backendError"}'))
    with vertex(gateway, gemini_echo, plan=plan) as rig:
        assert rig.chat().status_code >= 500
        assert rig.peer.model_calls() == ()
        plan.error = None
        assert rig.chat().status_code == 200
        assert len(rig.peer.model_calls()) == 1


# ---------------------------------------------------------------- credential forms and project resolution


def test_credentials_given_as_a_mapping_are_used_like_json_text(gateway: Gateway) -> None:
    """`vertex_credentials` registered as a JSON object (a YAML mapping in config) rather than a string
    is loaded by the dict branch of `load_auth`, mints a token and serves the call."""
    with vertex(gateway, gemini_echo, credentials=lambda base: service_account(base)) as rig:
        assert rig.chat().status_code == 200
        assert len(rig.peer.token_calls()) >= 1
        assert len(rig.peer.model_calls()) == 1


def test_credentials_given_as_a_file_path_are_read_from_disk(gateway: Gateway, tmp_path: Path) -> None:
    """`vertex_credentials` naming a readable JSON file on the proxy host is loaded from that file;
    the minted token comes from the file's `token_uri`."""
    key_file: Final = tmp_path / "vertex-key.json"

    def write(base: str) -> str:
        key_file.write_text(service_account_text(base))
        return str(key_file)

    with vertex(gateway, gemini_echo, credentials=write) as rig:
        assert rig.chat().status_code == 200
        assert len(rig.peer.token_calls()) >= 1


def test_rotating_the_key_file_mints_with_the_new_key(gateway: Gateway, tmp_path: Path) -> None:
    """A key file is cached by path plus inode, mtime and size, so after the file is replaced by a
    different service account the next request signs a new JWT whose header `kid` is the new key's
    `private_key_id`."""
    key_file: Final = tmp_path / "rotating.json"

    def write(base: str) -> str:
        key_file.write_text(service_account_text(base, private_key_id="key-before"))
        return str(key_file)

    with vertex(gateway, gemini_echo, credentials=write) as rig:
        assert rig.chat().status_code == 200
        replacement: Final = tmp_path / "rotating.json.new"
        replacement.write_text(service_account_text(rig.peer.url, private_key_id="key-after-rotation"))
        replacement.replace(key_file)
        assert rig.chat().status_code == 200
        headers: Final = [
            json.loads(base64.urlsafe_b64decode(form_of(call)["assertion"].split(".")[0] + "=="))
            for call in rig.peer.token_calls()
        ]
        assert headers[0]["kid"] == "key-before"
        assert headers[-1]["kid"] == "key-after-rotation"


def test_a_replaced_key_file_that_does_not_load_keeps_the_earlier_credentials(gateway: Gateway, tmp_path: Path) -> None:
    """When the key file is overwritten with text that is not JSON, the next request is still served
    with the credentials loaded before (`_credentials_from_an_earlier_key_file`), instead of failing
    every call until the file is fixed."""
    key_file: Final = tmp_path / "half-written.json"

    def write(base: str) -> str:
        key_file.write_text(service_account_text(base))
        return str(key_file)

    with vertex(gateway, gemini_echo, credentials=write) as rig:
        assert rig.chat().status_code == 200
        key_file.write_text('{"type": "service_account", "private_key": "-----BEGIN')
        response: Final = rig.chat()
        assert response.status_code == 200, response.text
        assert len(rig.peer.model_calls()) == 2


def test_authorized_user_credentials_refresh_with_the_refresh_token_grant(gateway: Gateway) -> None:
    """`authorized_user` credentials (gcloud application default login) mint by POSTing their
    `token_uri` with `grant_type=refresh_token` plus the file's `client_id`, `client_secret` and
    `refresh_token`, and the model call carries the returned token.

    HARNESS (run 44176682961): not observable here. `Credentials.from_authorized_user_info` always replaces the
    file's `token_uri` with `https://oauth2.googleapis.com/token` (google/oauth2/credentials.py, "always
    overrides", google-auth 2.52 pinned in uv.lock), and litellm loads authorized users through it
    (litellm/llms/vertex_ai/vertex_llm_base.py:319-325), so the refresh never reaches the peer and the CI egress
    guard refuses the real endpoint (500 "Network is unreachable").
    """
    pytest.skip("HARNESS: google-auth pins authorized_user token_uri to oauth2.googleapis.com, which CI cannot reach")
    user: Final[dict[str, object]] = {}

    def build(base: str) -> str:
        user.update(authorized_user(base))
        return json.dumps(user)

    with vertex(gateway, gemini_echo, credentials=build) as rig:
        assert rig.chat().status_code == 200
        form: Final = form_of(rig.peer.token_calls()[0])
        assert form["grant_type"] == "refresh_token"
        assert (form["client_id"], form["client_secret"], form["refresh_token"]) == (
            user["client_id"],
            user["client_secret"],
            user["refresh_token"],
        )


def test_an_authorized_user_without_vertex_project_uses_its_quota_project(gateway: Gateway) -> None:
    """With `authorized_user` credentials and no `vertex_project`, the project in the Vertex path is the
    credentials' `quota_project_id` (`load_auth`); Rust fixed the same case in `credential-gaps.md`.

    HARNESS (run 44176682961): not observable here. `Credentials.from_authorized_user_info` always replaces the
    file's `token_uri` with `https://oauth2.googleapis.com/token` (google/oauth2/credentials.py, "always
    overrides", google-auth 2.52 pinned in uv.lock), and litellm loads authorized users through it
    (litellm/llms/vertex_ai/vertex_llm_base.py:319-325), so the refresh never reaches the peer and the CI egress
    guard refuses the real endpoint (500 "Network is unreachable").
    """
    pytest.skip("HARNESS: google-auth pins authorized_user token_uri to oauth2.googleapis.com, which CI cannot reach")
    with vertex(
        gateway, gemini_echo, vertex_project=None, credentials=lambda base: json.dumps(authorized_user(base))
    ) as rig:
        assert rig.chat().status_code == 200
        assert path_of(rig.peer.one_call()) == models_path(project=QUOTA_PROJECT) + ":generateContent"


def test_an_authorized_user_without_any_project_is_refused_without_a_model_call(gateway: Gateway) -> None:
    """An `authorized_user` with no `quota_project_id` and no `vertex_project` cannot resolve a project,
    so the request fails with a 500 and the model is never called. The credentials are refreshed before the
    project is resolved, and google-auth sends that refresh to `oauth2.googleapis.com` whatever the
    `token_uri` (google/oauth2/credentials.py `from_authorized_user_info`, google-auth 2.52), so in CI the
    500 names the unreachable token endpoint rather than `Could not resolve project_id`; the message is not
    pinned because it depends on the network."""
    with vertex(
        gateway,
        gemini_echo,
        vertex_project=None,
        credentials=lambda base: json.dumps(authorized_user(base, quota_project=None)),
    ) as rig:
        response: Final = rig.chat()
        assert response.status_code == 500, response.text
        assert rig.peer.model_calls() == ()


def test_vertex_project_wins_over_the_service_account_project(gateway: Gateway) -> None:
    """An explicit `vertex_project` is the one in the path even though the service account names
    another `project_id`; the key's project is only a fallback."""
    with vertex(gateway, gemini_echo) as rig:
        assert rig.chat().status_code == 200
        path: Final = path_of(rig.peer.one_call())
        assert f"/projects/{PROJECT}/" in path and SA_PROJECT not in path


def test_aliased_credentials_project_and_location_reach_the_path(gateway: Gateway) -> None:
    """With only the `vertex_ai_*` spellings set, the call goes to `projects/dec1-alias-project/
    locations/asia-northeast1` and carries a token minted from `vertex_ai_credentials`. UNVERIFIED:
    assumes `/model/new` keeps the aliased keys in litellm_params."""
    from integration.providers.dec1_vertex_support import peer

    with peer(gemini_echo) as owned, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="vertex_ai/gemini-2.5-flash",
            api_key=None,
            api_base=owned.url,
            vertex_ai_project="dec1-alias-project",
            vertex_ai_location="asia-northeast1",
            vertex_ai_credentials=service_account_text(owned.url),
        )
        response: Final = gateway.request(
            "POST", "/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": f"a {marker()}"}]}
        )
        assert response.status_code == 200, response.text
        assert (
            path_of(owned.one_call())
            == models_path(project="dec1-alias-project", location="asia-northeast1") + ":generateContent"
        )
        assert len(owned.token_calls()) >= 1


# ---------------------------------------------------------------- Claude on Vertex, /v1/messages credential parity (PR #51294 gaps)


def _messages(text: str | None = None) -> dict[str, object]:
    return {"max_tokens": 32, "messages": [{"role": "user", "content": text or f"m {marker()}"}]}


def test_messages_overwrites_a_deployment_authorization_header_with_the_minted_token(gateway: Gateway) -> None:
    """Python always sets `Authorization: Bearer <minted token>` on `/v1/messages` to Vertex Claude, even
    when the deployment's `extra_headers` carry their own `Authorization`. `credential-gaps.md` row 26:
    the Rust draft forwards the existing header and mints nothing, which would send a stale or foreign
    credential to Google."""
    with vertex(
        gateway, claude_echo, backend=CLAUDE, extra_headers={"Authorization": "Bearer deployment-own-token"}
    ) as rig:
        response: Final = rig.post("/v1/messages", _messages())
        assert response.status_code == 200, response.text
        call: Final = rig.peer.one_call()
        assert call.headers["authorization"] != "Bearer deployment-own-token"
        assert call.headers["authorization"].removeprefix("Bearer ") in rig.peer.plan.issued


def test_messages_ignores_a_deployment_api_key(gateway: Gateway) -> None:
    """A Vertex Claude deployment that also carries `api_key` still authenticates with the minted token
    on `/v1/messages`; the key is sent neither as `x-api-key` nor as the bearer (`credential-gaps.md`
    row 25)."""
    with vertex(gateway, claude_echo, backend=CLAUDE, api_key="sk-should-not-be-sent") as rig:
        assert rig.post("/v1/messages", _messages()).status_code == 200
        call: Final = rig.peer.one_call()
        assert "x-api-key" not in call.headers
        assert "sk-should-not-be-sent" not in json.dumps(dict(call.headers))
        assert call.headers["authorization"].removeprefix("Bearer ") in rig.peer.plan.issued


def test_messages_with_authorized_user_credentials_uses_the_quota_project(gateway: Gateway) -> None:
    """On `/v1/messages`, `authorized_user` credentials without `vertex_project` resolve the project
    from `quota_project_id` and the rawPredict path names it; the Rust draft only learned this in
    `471ba58f` (`credential-gaps.md`, project table).

    HARNESS (run 44176682961): not observable here. `Credentials.from_authorized_user_info` always replaces the
    file's `token_uri` with `https://oauth2.googleapis.com/token` (google/oauth2/credentials.py, "always
    overrides", google-auth 2.52 pinned in uv.lock), and litellm loads authorized users through it
    (litellm/llms/vertex_ai/vertex_llm_base.py:319-325), so the refresh never reaches the peer and the CI egress
    guard refuses the real endpoint (500 "Network is unreachable").
    """
    pytest.skip("HARNESS: google-auth pins authorized_user token_uri to oauth2.googleapis.com, which CI cannot reach")
    with vertex(
        gateway,
        claude_echo,
        backend=CLAUDE,
        vertex_project=None,
        credentials=lambda base: json.dumps(authorized_user(base)),
    ) as rig:
        assert rig.post("/v1/messages", _messages()).status_code == 200
        assert path_of(rig.peer.one_call()) == claude_path(project=QUOTA_PROJECT) + ":rawPredict"


def test_messages_reuses_one_token_across_requests(gateway: Gateway) -> None:
    """Three `/v1/messages` calls to the same Vertex Claude deployment mint at most one token per proxy
    worker, the same caching chat has; the Rust provider cache must not mint per request."""
    with vertex(gateway, claude_echo, backend=CLAUDE) as rig:
        for _ in range(3):
            assert rig.post("/v1/messages", _messages()).status_code == 200
        assert len(rig.peer.model_calls()) == 3
        assert 1 <= len(rig.peer.token_calls()) <= max(gateway.workers, 1)


def test_messages_service_account_in_a_non_default_universe_signs_its_own_token(gateway: Gateway) -> None:
    """A service account with `universe_domain` other than `googleapis.com` gets a self-signed JWT from
    google-auth instead of an OAuth exchange, so the token endpoint is never called and the bearer is a
    three-part JWT. `credential-gaps.md` row 22: gcp_auth always calls `token_uri`, a Rust gap.
    UNVERIFIED: relies on google-auth's `always_use_jwt_access` for non-default universes."""
    with vertex(
        gateway,
        claude_echo,
        backend=CLAUDE,
        credentials=lambda base: service_account_text(base, universe_domain="dec1-universe.example"),
    ) as rig:
        assert rig.post("/v1/messages", _messages()).status_code == 200
        bearer: Final = rig.peer.one_call().headers["authorization"].removeprefix("Bearer ")
        assert bearer.count(".") == 2
        assert rig.peer.token_calls() == ()
