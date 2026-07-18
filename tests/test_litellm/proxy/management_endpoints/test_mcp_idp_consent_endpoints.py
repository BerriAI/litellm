"""Tests for the delegated-OBO consent-capture endpoints (Path B, item 2.2).

These pin the endpoint wiring on top of the (separately tested) flow logic: the authorize route seals
the signed-in user into the redirect and points at the configured IdP, and the terminal callback
unseals the user, exchanges the code, and stores the grant under the mint arm's key.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from pydantic import SecretStr

from litellm.proxy._experimental.mcp_server import idp_consent
from litellm.proxy._experimental.mcp_server.idp_consent import CaptureState, seal_capture_state
from litellm.proxy._experimental.mcp_server.outbound_credentials import idp_subject_provider
from litellm.proxy._experimental.mcp_server.outbound_credentials.idp_oauth_config import (
    IdpOAuthProvider,
    IdpOAuthProviderRegistry,
    set_idp_oauth_registry,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_utils.encrypt_decrypt_utils import encrypt_value_helper
from litellm.proxy.management_endpoints.mcp_management_endpoints import (
    mcp_idp_authorize,
    mcp_idp_callback,
)

_TOKEN_URL = "https://idp.example.com/oauth2/v1/token"
_PROVIDER = IdpOAuthProvider(
    token_url=_TOKEN_URL,
    authorize_url="https://idp.example.com/oauth2/v1/authorize",
    client_id="gateway-client",
    client_secret=SecretStr("gateway-secret"),
)


@pytest.fixture(autouse=True)
def _salt(monkeypatch):
    monkeypatch.setenv("LITELLM_SALT_KEY", "test-salt-key-for-idp-consent-endpoint-tests")


@pytest.fixture(autouse=True)
def _registry():
    set_idp_oauth_registry(IdpOAuthProviderRegistry((_PROVIDER,)))
    yield
    set_idp_oauth_registry(IdpOAuthProviderRegistry(()))


@pytest.fixture(autouse=True)
def _base_url(monkeypatch):
    monkeypatch.setenv("PROXY_BASE_URL", "https://gw.example.com")


@pytest.mark.asyncio
async def test_authorize_redirects_to_the_idp_sealing_the_user():
    resp = await mcp_idp_authorize(
        MagicMock(), user_api_key_dict=UserAPIKeyAuth(user_id="alice"), token_url=_TOKEN_URL
    )
    location = resp.headers["location"]
    assert location.startswith("https://idp.example.com/oauth2/v1/authorize?")
    assert "code_challenge_method=S256" in location
    assert "offline_access" in location
    assert "redirect_uri=https%3A%2F%2Fgw.example.com%2Fv1%2Fmcp%2Fidp%2Fcallback" in location


@pytest.mark.asyncio
async def test_authorize_404_for_an_unconfigured_idp():
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        await mcp_idp_authorize(MagicMock(), user_api_key_dict=UserAPIKeyAuth(user_id="alice"), token_url="https://other/token")
    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_authorize_400_without_a_user():
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        await mcp_idp_authorize(MagicMock(), user_api_key_dict=UserAPIKeyAuth(user_id=None), token_url=_TOKEN_URL)
    assert exc.value.status_code == 400


def _seal(user_id: str, *, issued_at: float | None = None) -> str:
    import time

    return seal_capture_state(
        CaptureState(user_id=user_id, token_url=_TOKEN_URL, code_verifier="the-verifier", issued_at=issued_at or time.time()),
        encrypt_value_helper,
    )


@pytest.mark.asyncio
async def test_callback_exchanges_and_stores_the_grant_for_the_sealed_user(monkeypatch):
    exchanges: list[dict[str, str]] = []
    captures: list[tuple] = []

    async def fake_post(url, form, headers):
        exchanges.append(form)
        return {"access_token": "alice-idp-at", "refresh_token": "alice-idp-rt", "expires_in": 3600}

    async def fake_capture(user_id, token_exchange_endpoint, access_token, *, refresh_token=None, expires_in=None, scopes=None):
        captures.append((user_id, token_exchange_endpoint, access_token, refresh_token))
        return True

    monkeypatch.setattr(idp_consent, "default_token_post", fake_post)
    monkeypatch.setattr(idp_subject_provider, "capture_user_idp_grant", fake_capture)

    resp = await mcp_idp_callback(MagicMock(), code="auth-code", state=_seal("alice"))

    # The code + PKCE verifier from the sealed state are exchanged, and the grant is stored under the
    # sealed user + the IdP token endpoint (the mint arm's key), never a caller-supplied identity.
    assert exchanges[0]["code"] == "auth-code"
    assert exchanges[0]["code_verifier"] == "the-verifier"
    assert captures == [("alice", _TOKEN_URL, "alice-idp-at", "alice-idp-rt")]
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_callback_503_when_the_grant_could_not_be_stored(monkeypatch):
    from fastapi import HTTPException

    async def fake_post(url, form, headers):
        return {"access_token": "alice-idp-at", "refresh_token": "alice-idp-rt", "expires_in": 3600}

    async def fake_capture(user_id, token_exchange_endpoint, access_token, *, refresh_token=None, expires_in=None, scopes=None):
        return False  # the store was skipped (e.g. database not connected)

    monkeypatch.setattr(idp_consent, "default_token_post", fake_post)
    monkeypatch.setattr(idp_subject_provider, "capture_user_idp_grant", fake_capture)

    # A grant that did not persist must not be reported as connected; the user would otherwise think
    # they linked their IdP while every later delegated call fails closed with no stored grant.
    with pytest.raises(HTTPException) as exc:
        await mcp_idp_callback(MagicMock(), code="auth-code", state=_seal("alice"))
    assert exc.value.status_code == 503


@pytest.mark.asyncio
async def test_callback_rejects_a_tampered_state(monkeypatch):
    from fastapi import HTTPException

    captured = []
    monkeypatch.setattr(idp_subject_provider, "capture_user_idp_grant", lambda *a, **k: captured.append(a))
    with pytest.raises(HTTPException) as exc:
        await mcp_idp_callback(MagicMock(), code="auth-code", state="not-a-valid-sealed-state")
    assert exc.value.status_code == 400
    assert captured == []  # nothing stored on a bad state


@pytest.mark.asyncio
async def test_callback_rejects_an_expired_state(monkeypatch):
    from fastapi import HTTPException

    captured = []
    monkeypatch.setattr(idp_subject_provider, "capture_user_idp_grant", lambda *a, **k: captured.append(a))
    # A state sealed well beyond the 600s lifetime must be rejected, bounding replay.
    with pytest.raises(HTTPException) as exc:
        await mcp_idp_callback(MagicMock(), code="auth-code", state=_seal("alice", issued_at=1.0))
    assert exc.value.status_code == 400
    assert captured == []


@pytest.mark.asyncio
async def test_callback_400_on_idp_error():
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        await mcp_idp_callback(MagicMock(), error="access_denied")
    assert exc.value.status_code == 400
