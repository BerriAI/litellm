"""Tests for the authorization_code refresher: the refresh_token grant, parsing, and persist."""

import pytest

from litellm.proxy._experimental.mcp_server.outbound_credentials.authz_code_refresher import (
    AuthorizationCodeRefresher,
)
from litellm.proxy._experimental.mcp_server.outbound_credentials.oauth_token_store import (
    OAuthToken,
)


class _Server:
    def __init__(
        self,
        token_url="https://idp.example.com/token",
        client_id="cid",
        client_secret="sec",
        token_endpoint_auth_method=None,
    ):
        self.token_url = token_url
        self.client_id = client_id
        self.client_secret = client_secret
        self.token_endpoint_auth_method = token_endpoint_auth_method


def _lookup(server):
    return lambda server_id: server


def _endpoint(body, sink=None):
    async def post(url, form, headers):
        if sink is not None:
            sink.append((url, form, headers))
        return body

    return post


def _recording_persist(sink):
    async def persist(
        user_id, server_id, access_token, refresh_token, expires_in, scopes
    ):
        sink.append(
            (user_id, server_id, access_token, refresh_token, expires_in, scopes)
        )

    return persist


def _refresher(
    server=None, body=None, *, post_sink=None, persist_sink=None, clock=lambda: 1000.0
):
    return AuthorizationCodeRefresher(
        _lookup(server if server is not None else _Server()),
        _endpoint(body, post_sink),
        _recording_persist(persist_sink if persist_sink is not None else []),
        clock=clock,
    )


@pytest.mark.asyncio
async def test_refreshes_persists_and_returns_typed_token():
    persisted = []
    posted = []
    refresher = _refresher(
        body={
            "access_token": "new-at",
            "expires_in": 3600,
            "refresh_token": "new-rt",
            "scope": "a b",
        },
        post_sink=posted,
        persist_sink=persisted,
    )
    token = await refresher.refresh(
        "alice", "srv", OAuthToken(access_token="old", refresh_token="old-rt")
    )

    assert token is not None
    assert token.access_token == "new-at"
    assert token.refresh_token == "new-rt"
    assert token.expires_at == 1000.0 + 3600  # clock + expires_in -> epoch
    # the rotated triple is persisted for (user, server) with parsed scopes
    assert persisted == [("alice", "srv", "new-at", "new-rt", 3600, ("a", "b"))]
    # the grant carried the refresh_token + client credentials in the body (client_secret_post default)
    url, form, headers = posted[0]
    assert url == "https://idp.example.com/token"
    assert form == {
        "grant_type": "refresh_token",
        "refresh_token": "old-rt",
        "client_id": "cid",
        "client_secret": "sec",
    }
    assert "Authorization" not in headers


@pytest.mark.asyncio
async def test_client_secret_basic_sends_authorization_header_not_body():
    """A server with token_endpoint_auth_method=client_secret_basic authenticates via HTTP Basic;
    the secret must not also leak into the form body."""
    import base64

    posted = []
    server = _Server(token_endpoint_auth_method="client_secret_basic")
    refresher = _refresher(
        server=server,
        body={"access_token": "new-at"},
        post_sink=posted,
    )
    token = await refresher.refresh(
        "alice", "srv", OAuthToken(access_token="old", refresh_token="old-rt")
    )

    assert token is not None
    _url, form, headers = posted[0]
    expected = "Basic " + base64.b64encode(b"cid:sec").decode()
    assert headers["Authorization"] == expected
    assert "client_secret" not in form
    assert "client_id" not in form
    assert form == {"grant_type": "refresh_token", "refresh_token": "old-rt"}


@pytest.mark.asyncio
async def test_client_secret_basic_without_secret_is_a_failed_refresh():
    """A server set to client_secret_basic but missing its secret cannot authenticate; the refresh
    returns None (failed refresh -> needs reauth) and never posts a downgraded request to the IdP."""
    posted = []
    server = _Server(client_secret=None, token_endpoint_auth_method="client_secret_basic")
    refresher = _refresher(server=server, body={"access_token": "x"}, post_sink=posted)
    assert await refresher.refresh("a", "s", OAuthToken("old", refresh_token="rt")) is None
    assert posted == []  # never hit the IdP


@pytest.mark.asyncio
async def test_no_refresh_token_is_not_refreshable():
    posted = []
    refresher = _refresher(body={"access_token": "x"}, post_sink=posted)
    assert (
        await refresher.refresh("alice", "srv", OAuthToken(access_token="old")) is None
    )
    assert posted == []  # never hit the IdP


@pytest.mark.asyncio
async def test_unknown_server_or_no_token_url_yields_none():
    assert (
        await _refresher(server=None).refresh(
            "a", "s", OAuthToken("old", refresh_token="rt")
        )
        is None
    )
    no_url = _Server(token_url=None)
    assert (
        await _refresher(server=no_url).refresh(
            "a", "s", OAuthToken("old", refresh_token="rt")
        )
        is None
    )


@pytest.mark.asyncio
async def test_grant_failure_does_not_persist():
    persisted = []
    refresher = _refresher(
        body=None, persist_sink=persisted
    )  # token_endpoint signals failure
    assert (
        await refresher.refresh("a", "s", OAuthToken("old", refresh_token="rt")) is None
    )
    assert persisted == []


@pytest.mark.asyncio
async def test_response_without_access_token_does_not_persist():
    persisted = []
    refresher = _refresher(body={"expires_in": 60}, persist_sink=persisted)
    assert (
        await refresher.refresh("a", "s", OAuthToken("old", refresh_token="rt")) is None
    )
    assert persisted == []


@pytest.mark.asyncio
async def test_unrotated_refresh_token_is_carried_forward():
    persisted = []
    refresher = _refresher(body={"access_token": "new-at"}, persist_sink=persisted)
    token = await refresher.refresh(
        "a", "s", OAuthToken("old", refresh_token="keep-rt")
    )
    assert token is not None
    assert (
        token.refresh_token == "keep-rt"
    )  # response omitted refresh_token -> reuse the old one
    assert token.expires_at is None  # no expires_in -> no known expiry
    assert persisted[0][3] == "keep-rt"


@pytest.mark.asyncio
async def test_unrecorded_scope_is_carried_forward():
    persisted = []
    refresher = _refresher(body={"access_token": "new-at"}, persist_sink=persisted)
    token = await refresher.refresh(
        "a", "s", OAuthToken("old", refresh_token="rt", scopes=("read", "write"))
    )
    assert token is not None
    # response omitted "scope" -> the user's recorded grant is preserved, not dropped
    assert token.scopes == ("read", "write")
    assert persisted[0][5] == ("read", "write")


@pytest.mark.asyncio
async def test_returned_scope_overrides_prior_when_present():
    persisted = []
    refresher = _refresher(
        body={"access_token": "new-at", "scope": "read"},
        persist_sink=persisted,
    )
    token = await refresher.refresh(
        "a", "s", OAuthToken("old", refresh_token="rt", scopes=("read", "write"))
    )
    assert token is not None
    assert token.scopes == ("read",)  # a present scope replaces the prior grant
    assert persisted[0][5] == ("read",)
