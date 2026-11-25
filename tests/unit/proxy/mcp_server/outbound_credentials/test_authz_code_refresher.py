"""Tests for the authorization_code refresher: the refresh_token grant, parsing, and persist."""

import logging
from collections.abc import Mapping
from typing import Final
from unittest.mock import AsyncMock, MagicMock, call

import httpx
import pytest

from litellm._logging import verbose_logger
from litellm.proxy.mcp_server.outbound_credentials.authz_code_refresher import (
    AuthorizationCodeRefresher,
    refusal_body,
    refused_refresh_error,
)
from litellm.proxy.mcp_server.outbound_credentials.oauth_token_store import (
    OAuthToken,
)


class _Server:
    def __init__(
        self,
        token_url="https://idp.example.com/token",
        client_id="cid",
        client_secret="sec",
        token_endpoint_auth_method=None,
        upstream_resource=None,
        url=None,
        server_id="srv",
        configured_token_url=None,
    ):
        self.oauth_identity_binding = None
        self.token_url = token_url
        self.configured_token_url = configured_token_url
        self.client_id = client_id
        self.client_secret = client_secret
        self.token_endpoint_auth_method = token_endpoint_auth_method
        self.upstream_resource = upstream_resource
        self.url = url
        self.server_id = server_id

    @property
    def effective_token_url(self):
        return self.token_url or self.configured_token_url


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
        user_id: str,
        server_id: str,
        access_token: str,
        refresh_token: str | None,
        expires_in: int | None,
        scopes: tuple[str, ...] | None,
        cimd_client_id: str | None = None,
        refreshed_from: str | None = None,
    ) -> None:
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
@pytest.mark.parametrize("prior_scopes", [(), ("read", "write")])
async def test_unrecorded_scope_is_carried_forward(prior_scopes):
    persisted = []
    refresher = _refresher(body={"access_token": "new-at"}, persist_sink=persisted)
    token = await refresher.refresh(
        "a", "s", OAuthToken("old", refresh_token="rt", scopes=prior_scopes)
    )
    assert token is not None
    # response omitted "scope" -> the user's recorded grant is preserved, not dropped
    assert token.scopes == prior_scopes
    assert persisted[0][5] == prior_scopes


@pytest.mark.asyncio
async def test_refresh_sends_upstream_resource_when_set_explicitly():
    """A silent refresh must carry the same RFC 8707 resource its authorize/initial-token legs sent,
    or a strict authorization server rejects the refresh with invalid_target."""
    posted = []
    server = _Server(upstream_resource="https://api.example.com/mcp")
    refresher = _refresher(server=server, body={"access_token": "new-at"}, post_sink=posted)
    token = await refresher.refresh("a", "s", OAuthToken("old", refresh_token="rt"))
    assert token is not None
    _url, form, _headers = posted[0]
    assert form["resource"] == "https://api.example.com/mcp"


@pytest.mark.asyncio
async def test_refresh_sends_upstream_resource_auto_derived_from_url():
    posted = []
    server = _Server(upstream_resource="auto", url="https://mcp.example.com/mcp")
    refresher = _refresher(server=server, body={"access_token": "new-at"}, post_sink=posted)
    token = await refresher.refresh("a", "s", OAuthToken("old", refresh_token="rt"))
    assert token is not None
    _url, form, _headers = posted[0]
    assert form["resource"] == "https://mcp.example.com/mcp"


@pytest.mark.asyncio
async def test_refresh_omits_resource_when_unset():
    posted = []
    refresher = _refresher(body={"access_token": "new-at"}, post_sink=posted)
    token = await refresher.refresh("a", "s", OAuthToken("old", refresh_token="rt"))
    assert token is not None
    _url, form, _headers = posted[0]
    assert "resource" not in form


@pytest.mark.asyncio
@pytest.mark.parametrize("scope, expected", [("read", ("read",)), ("", ()), ("   ", ())])
async def test_returned_scope_overrides_prior_when_present(scope, expected):
    persisted = []
    refresher = _refresher(
        body={"access_token": "new-at", "scope": scope},
        persist_sink=persisted,
    )
    token = await refresher.refresh(
        "a", "s", OAuthToken("old", refresh_token="rt", scopes=("read", "write"))
    )
    assert token is not None
    assert token.scopes == expected
    assert persisted[0][5] == expected


@pytest.mark.asyncio
async def test_refresh_uses_admin_entered_token_url_when_issuer_yield_empties_resolved():
    """A pinned issuer empties the resolved token_url while configured_token_url keeps the
    admin-entered value; the refresh grant must POST there instead of silently failing."""
    posted = []
    refresher = _refresher(
        server=_Server(token_url=None, configured_token_url="https://idp.example.com/token"),
        body={"access_token": "new-at", "expires_in": 3600},
        post_sink=posted,
    )
    token = await refresher.refresh(
        "alice", "srv", OAuthToken(access_token="old", refresh_token="old-rt")
    )

    assert token is not None
    assert token.access_token == "new-at"
    assert posted[0][0] == "https://idp.example.com/token"


@pytest.mark.asyncio
async def test_identity_rejection_never_persists_or_returns_refreshed_token():
    from unittest.mock import AsyncMock

    from fastapi import HTTPException

    validator = AsyncMock(side_effect=HTTPException(status_code=403, detail="oauth_principal_mismatch"))
    persist = AsyncMock()
    refresher = AuthorizationCodeRefresher(
        _lookup(_Server()),
        _endpoint({"access_token": "foreign-token", "id_token": "foreign-identity"}),
        persist,
        identity_validator=validator,
    )
    assert await refresher.refresh("alice", "srv", OAuthToken(access_token="old", refresh_token="old-rt")) is None
    persist.assert_not_awaited()


@pytest.mark.asyncio
async def test_verified_refresh_preserves_binding_proof_in_storage():
    from unittest.mock import AsyncMock

    validator = AsyncMock(return_value="verified-binding")
    persist = AsyncMock()
    refresher = AuthorizationCodeRefresher(
        _lookup(_Server()),
        _endpoint({"access_token": "new", "refresh_token": "rotated"}),
        persist,
        identity_validator=validator,
    )
    token = await refresher.refresh("alice", "srv", OAuthToken(access_token="old", refresh_token="old-rt"))
    assert token.access_token == "new"
    assert token.refresh_token == "rotated"
    assert token.identity_binding_proof == "verified-binding"
    assert persist.await_args.kwargs["identity_binding_proof"] == "verified-binding"


@pytest.mark.asyncio
async def test_cimd_refresh_on_fresh_replica_preserves_user_and_client_identity(monkeypatch):
    from litellm.types.mcp_server.mcp_server_manager import MCPServer

    monkeypatch.setenv("PROXY_BASE_URL", "https://gateway.example.com")
    server = MCPServer.model_validate(
        {
            "server_id": "srv",
            "name": "srv",
            "server_name": "srv",
            "transport": "http",
            "url": "https://mcp.example.com/mcp",
            "auth_type": "oauth2",
            "oauth2_flow": "authorization_code",
            "token_url": "https://idp.example.com/token",
            "client_id_metadata_document_supported": True,
        }
    )
    posted = []
    persisted = []
    refreshed = await _refresher(
        server=server,
        body={"access_token": "new-at", "expires_in": 3600},
        post_sink=posted,
        persist_sink=persisted,
    ).refresh("alice", "srv", OAuthToken(access_token="old-at", refresh_token="old-rt"))
    assert refreshed is not None
    assert refreshed.access_token == "new-at"
    assert posted == [
        (
            "https://idp.example.com/token",
            {
                "grant_type": "refresh_token",
                "refresh_token": "old-rt",
                "client_id": "https://gateway.example.com/oauth/client-metadata.json",
            },
            {},
        )
    ]
    assert persisted == [("alice", "srv", "new-at", "old-rt", 3600, ())]
    assert server.client_id is None


@pytest.mark.asyncio
async def test_saved_cimd_identity_refreshes_without_discovery(monkeypatch):
    from litellm.proxy.mcp_server.outbound_credentials.v2_token_store import V2PerUserTokenStore
    from litellm.types.mcp_server.mcp_server_manager import MCPServer

    monkeypatch.setenv("PROXY_BASE_URL", "https://gateway.example.com")
    identity = "https://gateway.example.com/oauth/client-metadata.json"
    stored = {"access_token": "old", "refresh_token": "old-rt", "cimd_client_id": identity}

    async def read_credential(user_id, server_id):
        assert (user_id, server_id) == ("alice", "srv")
        return stored

    async def persist(user_id, server_id, access_token, refresh_token, expires_in, scopes, **metadata):
        assert (user_id, server_id, access_token) == ("alice", "srv", "new")
        assert metadata["cimd_client_id"] == identity

    async def post(url, form, headers):
        assert url == "https://idp.example.com/token"
        assert form["client_id"] == identity
        assert "client_secret" not in form
        assert "Authorization" not in headers
        return {"access_token": "new", "refresh_token": "rotated"}

    server = MCPServer(
        server_id="srv", name="srv", transport="http", auth_type="oauth2", oauth2_flow="authorization_code",
        authorization_url="https://idp.example.com/authorize", token_url="https://idp.example.com/token",
    )
    token = await V2PerUserTokenStore(read_credential).fetch("alice", "srv")
    assert token is not None
    refreshed = await AuthorizationCodeRefresher(lambda _: server, post, persist).refresh("alice", "srv", token)
    assert refreshed is not None
    assert refreshed.access_token == "new"
    assert refreshed.refresh_token == "rotated"
    assert refreshed.cimd_client_id == identity


@pytest.mark.parametrize(
    "body, expected",
    [
        ({"error": "invalid_grant", "error_description": "refresh token revoked"}, "invalid_grant"),
        pytest.param(
            {"error": "bad_refresh_token"},
            "bad_refresh_token",
            id="github-http-200-per-refreshing-user-access-tokens-docs-2026-11",
        ),
        pytest.param(
            {"ok": False, "error": "invalid_refresh_token"},
            "invalid_refresh_token",
            id="slack-http-200-per-oauth-v2-access-errors-docs-2026-11",
        ),
        ({"error": "invalid_token"}, "invalid_token"),
        ({"error": "invalid_client"}, None),
        ({"error": "temporarily_unavailable"}, None),
        ({"error": "server_error"}, None),
        ({"error": "HTTP 401"}, "HTTP 401"),
        ({"access_token": "new", "error": "invalid_grant"}, None),
        ({"access_token": "", "error": "invalid_grant"}, "invalid_grant"),
        ({"error": ""}, None),
        ({"error": ["invalid_grant"]}, None),
        ({}, None),
    ],
)
def test_any_named_refusal_except_a_client_or_provider_fault_is_classified(
    body: Mapping[str, object], expected: str | None
) -> None:
    assert refused_refresh_error(body) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body, refused",
    [
        ({"error": "invalid_grant"}, True),
        ({"error": "bad_refresh_token"}, True),
        ({"error": "invalid_client"}, False),
        ({"error": "unauthorized_client"}, False),
        ({"error": "unsupported_grant_type"}, False),
        ({"error": "invalid_scope"}, True),
        (None, False),
    ],
)
async def test_a_refusal_records_the_presented_refresh_token(
    body: Mapping[str, object] | None, refused: bool
) -> None:
    persist: Final = AsyncMock()
    record_refusal: Final = AsyncMock()
    refresher: Final = AuthorizationCodeRefresher(
        _lookup(_Server()), _endpoint(body), persist, record_refusal=record_refusal
    )
    token: Final = await refresher.refresh("alice", "srv", OAuthToken(access_token="old", refresh_token="dead-rt"))

    assert token is None
    persist.assert_not_awaited()
    assert record_refusal.await_args_list == ([call("alice", "srv", "dead-rt")] if refused else []), (
        "a refusal marks the grant; a failed request does not"
    )


@pytest.mark.asyncio
async def test_a_refusal_is_logged_on_one_line_whatever_the_provider_sends() -> None:
    capture: Final = MagicMock(spec=logging.Handler, level=logging.WARNING)
    verbose_logger.addHandler(capture)
    try:
        refusing: Final = _endpoint({"error": "invalid_grant\nforged log line"})
        refresher: Final = AuthorizationCodeRefresher(_lookup(_Server()), refusing, AsyncMock())
        await refresher.refresh("alice", "srv", OAuthToken(access_token="old", refresh_token="dead-rt"))
    finally:
        verbose_logger.removeHandler(capture)

    messages: Final = [handled.args[0].getMessage() for handled in capture.handle.call_args_list]
    assert messages and all("\n" not in message for message in messages), "a provider cannot add log lines"


@pytest.mark.asyncio
async def test_a_refresher_without_a_recorder_still_challenges_on_a_refusal() -> None:
    refused: Final = _endpoint({"error": "invalid_grant"})
    refresher: Final = AuthorizationCodeRefresher(_lookup(_Server()), refused, AsyncMock())
    assert await refresher.refresh("alice", "srv", OAuthToken(access_token="old", refresh_token="dead-rt")) is None


@pytest.mark.parametrize(
    "status, content, expected",
    [
        (400, b'{"error": "invalid_grant"}', {"error": "invalid_grant"}),
        (401, b'{"error": "invalid_grant"}', {"error": "invalid_grant"}),
        (400, b"not json", {"error": "HTTP 400"}),
        (401, b"", {"error": "HTTP 401"}),
        (400, b'["invalid_grant"]', {"error": "HTTP 400"}),
        (400, b'{"error_description": "expired"}', {"error_description": "expired", "error": "HTTP 400"}),
        (429, b'{"error": "invalid_grant"}', None),
        (503, b'{"error": "invalid_grant"}', None),
    ],
)
def test_every_400_or_401_from_the_token_endpoint_is_read_as_a_refusal(
    status: int, content: bytes, expected: Mapping[str, object] | None
) -> None:
    assert refusal_body(httpx.Response(status, content=content)) == expected
