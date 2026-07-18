"""Tests for the delegated-OBO IdP subject-token source (Path B).

``StoredIdpGrantSource`` reads a stored IdP grant and refreshes it to a live subject token, failing
closed (None) whenever the user has no usable grant. ``IdpGrantRefresher`` runs the refresh_token
grant against the config's IdP endpoint and persists the rotation. The DB read, refresh POST, persist,
and clock are injected, so every case is exercised without a live IdP or DB.
"""

import asyncio

import pytest
from pydantic import SecretStr

from litellm.proxy._experimental.mcp_server.outbound_credentials.idp_subject_source import (
    IdpGrantRefresher,
    StoredIdpGrantSource,
    idp_grant_key,
)
from litellm.proxy._experimental.mcp_server.outbound_credentials.oauth_token_store import (
    OAuthToken,
)
from litellm.proxy._experimental.mcp_server.outbound_credentials.types import (
    TokenExchangeConfig,
)

_CONFIG = TokenExchangeConfig(
    token_exchange_endpoint="https://idp.example.com/oauth2/v1/token",
    client_id="gateway-client",
    client_secret=SecretStr("gateway-secret"),
)
_IDP_KEY = idp_grant_key("https://idp.example.com/oauth2/v1/token")


def _reader(grants):
    async def read(user_id, idp_key):
        return grants.get((user_id, idp_key))

    return read


def _never_refresh():
    async def refresh(user_id, idp_key, config, token):
        raise AssertionError("refresh must not run for a fresh grant")

    return refresh


def test_idp_grant_key_is_namespaced_and_normalizes_trailing_slash():
    assert idp_grant_key("https://idp.example.com/token/") == "idp::https://idp.example.com/token"
    assert idp_grant_key("https://idp.example.com/token") == "idp::https://idp.example.com/token"
    assert idp_grant_key("https://a/token").startswith("idp::")


@pytest.mark.asyncio
async def test_fresh_grant_is_returned_without_refreshing():
    grants = {("alice", _IDP_KEY): OAuthToken(access_token="live-at", expires_at=10_000.0)}
    source = StoredIdpGrantSource(_reader(grants), _never_refresh(), clock=lambda: 100.0)
    assert await source.subject_token("alice", _CONFIG) == "live-at"


@pytest.mark.asyncio
async def test_no_grant_returns_none():
    source = StoredIdpGrantSource(_reader({}), _never_refresh(), clock=lambda: 100.0)
    assert await source.subject_token("alice", _CONFIG) is None


@pytest.mark.asyncio
async def test_no_endpoint_returns_none():
    grants = {("alice", _IDP_KEY): OAuthToken(access_token="live-at")}
    source = StoredIdpGrantSource(_reader(grants), _never_refresh(), clock=lambda: 100.0)
    no_endpoint = TokenExchangeConfig(client_id="c", client_secret=SecretStr("s"))
    assert await source.subject_token("alice", no_endpoint) is None


@pytest.mark.asyncio
async def test_expired_grant_with_no_refresh_token_returns_none():
    grants = {("alice", _IDP_KEY): OAuthToken(access_token="stale", expires_at=50.0, refresh_token=None)}
    source = StoredIdpGrantSource(_reader(grants), _never_refresh(), clock=lambda: 100.0)
    assert await source.subject_token("alice", _CONFIG) is None


@pytest.mark.asyncio
async def test_expired_grant_is_refreshed_to_a_live_token():
    grants = {("alice", _IDP_KEY): OAuthToken(access_token="stale", expires_at=50.0, refresh_token="rt")}
    refreshes = []

    async def refresh(user_id, idp_key, config, token):
        refreshes.append((user_id, idp_key, token.refresh_token))
        return OAuthToken(access_token="fresh-at", expires_at=10_000.0, refresh_token="rt2")

    source = StoredIdpGrantSource(_reader(grants), refresh, clock=lambda: 100.0)
    assert await source.subject_token("alice", _CONFIG) == "fresh-at"
    assert refreshes == [("alice", _IDP_KEY, "rt")]


@pytest.mark.asyncio
async def test_a_refresh_that_still_yields_an_expired_token_returns_none():
    grants = {("alice", _IDP_KEY): OAuthToken(access_token="stale", expires_at=50.0, refresh_token="rt")}

    async def refresh(user_id, idp_key, config, token):
        return OAuthToken(access_token="still-stale", expires_at=60.0, refresh_token="rt")

    source = StoredIdpGrantSource(_reader(grants), refresh, clock=lambda: 100.0)
    assert await source.subject_token("alice", _CONFIG) is None


@pytest.mark.asyncio
async def test_concurrent_expired_reads_refresh_once():
    """Refresh is single-flighted per (user, idp) so IdP refresh-token rotation isn't raced: two
    concurrent callers for the same expired grant trigger exactly one refresh, not two (a second
    would spend an already-rotated refresh_token)."""
    grants = {("alice", _IDP_KEY): OAuthToken(access_token="stale", expires_at=50.0, refresh_token="rt")}
    refresh_count = 0
    gate = asyncio.Event()

    async def refresh(user_id, idp_key, config, token):
        nonlocal refresh_count
        refresh_count += 1
        await gate.wait()
        return OAuthToken(access_token="fresh-at", expires_at=10_000.0, refresh_token="rt2")

    source = StoredIdpGrantSource(_reader(grants), refresh, clock=lambda: 100.0)
    task_a = asyncio.ensure_future(source.subject_token("alice", _CONFIG))
    task_b = asyncio.ensure_future(source.subject_token("alice", _CONFIG))
    await asyncio.sleep(0)
    gate.set()
    results = await asyncio.gather(task_a, task_b)
    assert results == ["fresh-at", "fresh-at"]
    assert refresh_count == 1


class _RecordingPost:
    def __init__(self, body):
        self._body = body
        self.calls = []

    async def __call__(self, url, form, headers):
        self.calls.append((url, form, headers))
        return self._body


class _RecordingPersist:
    def __init__(self):
        self.calls = []

    async def __call__(self, user_id, idp_key, access_token, refresh_token, expires_in, scopes):
        self.calls.append((user_id, idp_key, access_token, refresh_token, expires_in, scopes))


@pytest.mark.asyncio
async def test_refresher_runs_refresh_token_grant_persists_and_returns_the_new_token():
    post = _RecordingPost({"access_token": "new-at", "refresh_token": "rt2", "expires_in": 3600})
    persist = _RecordingPersist()
    refresher = IdpGrantRefresher(post, persist, clock=lambda: 1000.0)
    stale = OAuthToken(access_token="old", expires_at=1.0, refresh_token="rt1", scopes=("read",))

    result = await refresher.refresh("alice", _IDP_KEY, _CONFIG, stale)

    assert result is not None
    assert result.access_token == "new-at"
    assert result.refresh_token == "rt2"
    assert result.expires_at == 1000.0 + 3600
    url, form, _headers = post.calls[0]
    assert url == "https://idp.example.com/oauth2/v1/token"
    assert form["grant_type"] == "refresh_token"
    assert form["refresh_token"] == "rt1"
    # The rotated triple is persisted so later reads see the new token without refreshing again.
    assert persist.calls == [("alice", _IDP_KEY, "new-at", "rt2", 3600, ("read",))]


@pytest.mark.asyncio
async def test_refresher_carries_forward_an_omitted_refresh_token():
    post = _RecordingPost({"access_token": "new-at", "expires_in": 3600})  # no rotated refresh_token
    persist = _RecordingPersist()
    refresher = IdpGrantRefresher(post, persist, clock=lambda: 0.0)
    stale = OAuthToken(access_token="old", expires_at=1.0, refresh_token="rt1")
    result = await refresher.refresh("alice", _IDP_KEY, _CONFIG, stale)
    assert result is not None and result.refresh_token == "rt1"


@pytest.mark.asyncio
async def test_refresher_returns_none_when_the_idp_call_fails():
    post = _RecordingPost(None)  # transport / HTTP failure surfaces as a miss
    persist = _RecordingPersist()
    refresher = IdpGrantRefresher(post, persist)
    stale = OAuthToken(access_token="old", expires_at=1.0, refresh_token="rt1")
    assert await refresher.refresh("alice", _IDP_KEY, _CONFIG, stale) is None
    assert persist.calls == []


@pytest.mark.asyncio
async def test_refresher_returns_none_without_client_credentials():
    # A token_exchange config missing client credentials cannot authenticate to the endpoint, so the
    # refresh cannot run and the caller fails closed rather than posting an unauthenticated grant.
    post = _RecordingPost({"access_token": "new-at"})
    refresher = IdpGrantRefresher(post, _RecordingPersist())
    no_creds = TokenExchangeConfig(token_exchange_endpoint="https://idp.example.com/token")
    stale = OAuthToken(access_token="old", expires_at=1.0, refresh_token="rt1")
    assert await refresher.refresh("alice", _IDP_KEY, no_creds, stale) is None
    assert post.calls == []
