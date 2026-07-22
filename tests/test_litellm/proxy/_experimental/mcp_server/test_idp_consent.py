"""Tests for the delegated-OBO consent-capture flow logic (Path B, item 2.2).

Covers the pure core: PKCE generation, sealed-state round-trip and tamper rejection, authorize-URL
building (offline_access + S256), and the server-side code->grant exchange with its fail-closed paths.
The token POST and the crypto are injected, so no live IdP is needed.
"""

import base64
import hashlib

import pytest
from pydantic import SecretStr

from litellm.proxy._experimental.mcp_server.idp_consent import (
    CaptureState,
    build_authorize_url,
    exchange_code_for_grant,
    generate_pkce,
    seal_capture_state,
    unseal_capture_state,
)
from litellm.proxy._experimental.mcp_server.outbound_credentials.idp_oauth_config import (
    IdpOAuthProvider,
    IdpOAuthProviderRegistry,
)
from litellm.proxy._experimental.mcp_server.outbound_credentials.idp_subject_source import (
    idp_grant_key,
)

_PROVIDER = IdpOAuthProvider(
    token_url="https://idp.example.com/oauth2/v1/token",
    authorize_url="https://idp.example.com/oauth2/v1/authorize",
    client_id="gateway-client",
    client_secret=SecretStr("gateway-secret"),
    scopes=("openid", "offline_access"),
)


def test_generate_pkce_is_s256_of_the_verifier():
    verifier, challenge = generate_pkce()
    expected = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip("=")
    assert challenge == expected
    assert "=" not in challenge  # base64url, unpadded
    # Two calls produce different verifiers (not a constant).
    assert generate_pkce()[0] != verifier


def test_sealed_state_round_trips():
    store: dict[str, str] = {}

    def encrypt(plain):
        token = f"sealed::{plain}"
        store[token] = plain
        return token

    def decrypt(blob):
        return store.get(blob)

    state = CaptureState(user_id="alice", token_url=_PROVIDER.token_url, code_verifier="verifier-xyz", issued_at=1000.0)
    sealed = seal_capture_state(state, encrypt)
    assert unseal_capture_state(sealed, decrypt) == state


def test_state_is_fresh_bounds_replay():
    from litellm.proxy._experimental.mcp_server.idp_consent import state_is_fresh

    state = CaptureState(user_id="a", token_url=_PROVIDER.token_url, code_verifier="v", issued_at=1000.0)
    assert state_is_fresh(state, now=1000.0, max_age_seconds=600) is True
    assert state_is_fresh(state, now=1599.0, max_age_seconds=600) is True
    assert state_is_fresh(state, now=1601.0, max_age_seconds=600) is False  # expired
    assert state_is_fresh(state, now=900.0, max_age_seconds=600) is False  # issued in the future (clock skew / forged)


def test_unseal_rejects_tampered_or_undecryptable_state():
    # decrypt returning None (bad/forged blob) -> None, never a partial/forged CaptureState.
    assert unseal_capture_state("garbage", lambda _b: None) is None
    # decrypts to non-CaptureState JSON -> None.
    assert unseal_capture_state("x", lambda _b: '{"not":"a state"}') is None


def test_build_authorize_url_carries_offline_access_and_s256():
    url = build_authorize_url(
        _PROVIDER,
        redirect_uri="https://gw.example.com/v1/mcp/idp/callback",
        state="sealed-state",
        code_challenge="chal",
    )
    assert url.startswith("https://idp.example.com/oauth2/v1/authorize?")
    assert "response_type=code" in url
    assert "client_id=gateway-client" in url
    assert "code_challenge=chal" in url
    assert "code_challenge_method=S256" in url
    assert "offline_access" in url
    assert "redirect_uri=https%3A%2F%2Fgw.example.com%2Fv1%2Fmcp%2Fidp%2Fcallback" in url


def test_build_authorize_url_preserves_existing_query():
    provider = _PROVIDER.model_copy(update={"authorize_url": "https://idp.example.com/authorize?tenant=acme"})
    url = build_authorize_url(provider, redirect_uri="https://gw/cb", state="s", code_challenge="c")
    assert "tenant=acme" in url
    assert "response_type=code" in url


@pytest.mark.asyncio
async def test_exchange_code_for_grant_parses_the_grant():
    captured_form: dict[str, str] = {}

    async def post(url, form, headers):
        captured_form.update(form)
        return {"access_token": "user-at", "refresh_token": "user-rt", "expires_in": 3600, "scope": "openid offline_access"}

    grant = await exchange_code_for_grant(
        _PROVIDER, code="auth-code", redirect_uri="https://gw/cb", code_verifier="ver", post=post
    )
    assert grant is not None
    assert grant.access_token == "user-at"
    assert grant.refresh_token == "user-rt"
    assert grant.expires_in == 3600
    assert grant.scopes == ("openid", "offline_access")
    # The authorization_code grant is posted with the PKCE verifier and the client id.
    assert captured_form["grant_type"] == "authorization_code"
    assert captured_form["code"] == "auth-code"
    assert captured_form["code_verifier"] == "ver"
    assert captured_form["client_id"] == "gateway-client"


@pytest.mark.asyncio
async def test_exchange_code_falls_closed_on_missing_access_token_or_failure():
    async def no_access_token(url, form, headers):
        return {"refresh_token": "rt"}  # no access_token

    async def transport_failure(url, form, headers):
        return None

    assert await exchange_code_for_grant(_PROVIDER, code="c", redirect_uri="r", code_verifier="v", post=no_access_token) is None
    assert await exchange_code_for_grant(_PROVIDER, code="c", redirect_uri="r", code_verifier="v", post=transport_failure) is None


@pytest.mark.asyncio
async def test_exchange_code_carries_forward_scopes_when_response_omits_scope():
    async def post(url, form, headers):
        return {"access_token": "at"}  # no scope, no refresh_token, no expires_in

    grant = await exchange_code_for_grant(_PROVIDER, code="c", redirect_uri="r", code_verifier="v", post=post)
    assert grant is not None
    assert grant.scopes == ("openid", "offline_access")
    assert grant.refresh_token is None
    assert grant.expires_in is None


def test_provider_grant_key_matches_the_mint_arm_key():
    # The captured grant must land under the exact key the mint arm reads.
    assert _PROVIDER.grant_key == idp_grant_key("https://idp.example.com/oauth2/v1/token")


def test_registry_resolves_by_grant_key_and_token_url():
    registry = IdpOAuthProviderRegistry((_PROVIDER,))
    assert registry.get(_PROVIDER.grant_key) is _PROVIDER
    assert registry.get_by_token_url("https://idp.example.com/oauth2/v1/token/") is _PROVIDER  # normalized
    assert registry.get("idp::https://other/token") is None
