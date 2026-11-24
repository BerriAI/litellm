"""The login-time exchange of a captured identity assertion for an IdP refresh token."""

import asyncio
import logging
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import SecretStr

from litellm.proxy._types import LiteLLM_MCPServerTable
from litellm.proxy.mcp_server.outbound_credentials.login_assertion_exchange import (
    exchange_login_assertion_for_refresh_token,
    login_exchange_servers,
    refresh_token_exchange_targets,
)
from litellm.proxy.mcp_server.outbound_credentials.result import Error, Ok, Result
from litellm.proxy.mcp_server.outbound_credentials.sso_assertion_refresher import SSOClientConfig
from litellm.proxy.mcp_server.outbound_credentials.sso_assertion_store import (
    SAML2_SUBJECT_TOKEN_TYPE,
    SSOIdentityAssertion,
    assertion_from_saml_login,
    login_refresh_token_expired,
    with_login_refresh_token,
)
from litellm.proxy.mcp_server.outbound_credentials.token_endpoint import ExchangedToken
from litellm.proxy.mcp_server.outbound_credentials.types import (
    ClientAuth,
    ClientSecretAuth,
    CredError,
    IdJagConfig,
    IdJagResourceClient,
    IdJagSubjectSource,
    NoneConfig,
    ServerSpec,
)
from litellm.types.mcp import MCPAuth, MCPTransport
from litellm.types.mcp_server.mcp_server_manager import MCPServer

_NOW = datetime(2026, 10, 16, 12, 0, tzinfo=timezone.utc)


def _config(
    *, source: IdJagSubjectSource = IdJagSubjectSource.refresh_token, client_id: str = "agent", endpoint: str = ""
) -> IdJagConfig:
    return IdJagConfig(
        org_token_endpoint=endpoint or "https://idp.example.com/token",
        resource_token_endpoint="https://mcp-as.example.com/token",
        client_id=client_id,
        client_auth=ClientSecretAuth(client_secret=SecretStr("s")),
        resource_client=IdJagResourceClient(
            client_id=client_id, client_auth=ClientSecretAuth(client_secret=SecretStr("s"))
        ),
        subject_source=source,
    )


def _spec(config: IdJagConfig | NoneConfig, server_id: str = "srv") -> ServerSpec:
    return ServerSpec(server_id=server_id, resource="https://mcp.example.com", config=config)


def _saml() -> SSOIdentityAssertion:
    return assertion_from_saml_login(b"<saml:Assertion/>", int(_NOW.timestamp()) + 300, "https://idp.example.com")


class _Endpoint:
    def __init__(self, result: Result[ExchangedToken, CredError]) -> None:
        self._result = result
        self.calls: list[tuple[str, str, dict[str, str], ClientAuth]] = []

    async def fetch(
        self, endpoint: str, client_id: str, grant_params: dict[str, str], client_auth: ClientAuth
    ) -> Result[ExchangedToken, CredError]:
        self.calls.append((endpoint, client_id, dict(grant_params), client_auth))
        return self._result


def test_targets_keep_only_refresh_token_sourced_id_jag_servers_one_per_idp_client():
    direct = _config(source=IdJagSubjectSource.login_assertion)
    first = _config(client_id="agent")
    twin = _config(client_id="agent")
    other = _config(client_id="other-agent")

    targets = refresh_token_exchange_targets(
        [_spec(direct, "a"), None, _spec(NoneConfig(), "b"), _spec(first, "c"), _spec(other, "d"), _spec(twin, "e")]
    )

    assert targets == (first, other)


@pytest.mark.asyncio
async def test_exchange_presents_the_captured_assertion_and_keeps_the_refresh_token():
    endpoint = _Endpoint(Ok(ExchangedToken(access_token="rt-1", expires_in=3600)))
    config = _config()
    saml = _saml()

    exchanged = await exchange_login_assertion_for_refresh_token(saml, (config,), endpoint, _NOW)

    ((url, client_id, params, auth),) = endpoint.calls
    assert (url, client_id, auth) == (config.org_token_endpoint, config.client_id, config.client_auth)
    assert params == {
        "grant_type": "urn:ietf:params:oauth:grant-type:token-exchange",
        "requested_token_type": "urn:ietf:params:oauth:token-type:refresh_token",
        "subject_token": saml.id_token.get_secret_value(),
        "subject_token_type": SAML2_SUBJECT_TOKEN_TYPE,
        "scope": "openid offline_access",
    }
    assert exchanged.exchanged_refresh_token is not None
    assert exchanged.exchanged_refresh_token.get_secret_value() == "rt-1"
    assert exchanged.exchanged_refresh_token_endpoint == "https://idp.example.com/token"
    assert exchanged.exchanged_refresh_token_client_id == "agent"
    assert exchanged.exchanged_refresh_token_expires_at == _NOW + timedelta(seconds=3600)
    assert exchanged.id_token == saml.id_token
    assert exchanged.expires_at == saml.expires_at
    assert exchanged.subject_token_type == SAML2_SUBJECT_TOKEN_TYPE
    assert not login_refresh_token_expired(exchanged, _NOW + timedelta(seconds=3599))
    assert login_refresh_token_expired(exchanged, _NOW + timedelta(seconds=3600))


@pytest.mark.asyncio
async def test_exchange_without_a_lifetime_leaves_expiry_to_the_idp():
    endpoint = _Endpoint(Ok(ExchangedToken(access_token="rt-1", expires_in=None)))

    exchanged = await exchange_login_assertion_for_refresh_token(_saml(), (_config(),), endpoint, _NOW)

    assert exchanged.exchanged_refresh_token_expires_at is None
    assert not login_refresh_token_expired(exchanged, _NOW + timedelta(days=365))


@pytest.mark.asyncio
async def test_a_refused_exchange_keeps_the_plain_assertion(caplog: pytest.LogCaptureFixture):
    endpoint = _Endpoint(Error(CredError.of_upstream_unavailable("token exchange failed with status 400")))
    saml = _saml()

    with caplog.at_level("WARNING"):
        exchanged = await exchange_login_assertion_for_refresh_token(saml, (_config(),), endpoint, _NOW)

    assert exchanged == saml
    assert exchanged.exchanged_refresh_token is None
    assert any("refused to exchange the login assertion" in message for message in caplog.messages)


@pytest.mark.asyncio
async def test_no_exchange_without_a_refresh_token_sourced_server():
    endpoint = _Endpoint(Ok(ExchangedToken(access_token="rt-1", expires_in=3600)))
    saml = _saml()

    assert await exchange_login_assertion_for_refresh_token(saml, (), endpoint, _NOW) == saml
    assert endpoint.calls == []


@pytest.mark.asyncio
async def test_an_oidc_login_is_exchanged_at_the_id_jag_client_and_keeps_its_sso_refresh_token():
    """The SSO refresh token belongs to the SSO client and cannot be redeemed by the ID-JAG client, so
    an OIDC login still exchanges its id_token and both tokens live side by side."""
    endpoint = _Endpoint(Ok(ExchangedToken(access_token="rt-1", expires_in=3600)))
    oidc = SSOIdentityAssertion(id_token=SecretStr("id-token"), refresh_token=SecretStr("sso-rt"))

    exchanged = await exchange_login_assertion_for_refresh_token(oidc, (_config(),), endpoint, _NOW)

    assert endpoint.calls[0][2]["subject_token"] == "id-token"
    assert endpoint.calls[0][2]["subject_token_type"] == "urn:ietf:params:oauth:token-type:id_token"
    assert exchanged.refresh_token is not None and exchanged.refresh_token.get_secret_value() == "sso-rt"
    assert exchanged.exchanged_refresh_token is not None
    assert exchanged.exchanged_refresh_token.get_secret_value() == "rt-1"


def _sso_client(client_id: str = "agent", endpoint: str = "https://idp.example.com/token") -> SSOClientConfig:
    return SSOClientConfig(
        token_endpoint=endpoint, client_id=client_id, client_secret=SecretStr("s"), auth_method="client_secret_basic"
    )


@pytest.mark.asyncio
async def test_an_oidc_login_through_the_id_jag_client_keeps_its_own_refresh_token_without_an_exchange():
    """Okta refuses requested_token_type=refresh_token for an id_token subject. When the SSO app is the
    ID-JAG client itself, the refresh token the login returned is already that client's, so it is bound
    to the target and nothing is posted to the IdP."""
    endpoint = _Endpoint(
        Error(
            CredError.of_upstream_unavailable(
                "token exchange failed with status 400: 'requested_token_type' is invalid or not supported."
            )
        )
    )
    oidc = SSOIdentityAssertion(id_token=SecretStr("id-token"), refresh_token=SecretStr("sso-rt"))

    kept = await exchange_login_assertion_for_refresh_token(
        oidc, (_config(),), endpoint, _NOW, sso_client=_sso_client()
    )

    assert endpoint.calls == []
    assert kept.exchanged_refresh_token is not None and kept.exchanged_refresh_token.get_secret_value() == "sso-rt"
    assert kept.exchanged_refresh_token_expires_at is None
    assert kept.exchanged_refresh_token_endpoint == "https://idp.example.com/token"
    assert kept.exchanged_refresh_token_client_id == "agent"
    assert kept.id_token == oidc.id_token and kept.refresh_token == oidc.refresh_token


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "sso_client",
    [
        pytest.param(_sso_client(client_id="sso-app"), id="other-client"),
        pytest.param(_sso_client(endpoint="https://other-idp.example.com/token"), id="other-endpoint"),
        pytest.param(None, id="unknown-client"),
    ],
)
async def test_a_login_refresh_token_from_another_client_or_issuer_is_never_presented(
    sso_client: SSOClientConfig | None,
):
    endpoint = _Endpoint(Ok(ExchangedToken(access_token="rt-1", expires_in=3600)))
    oidc = SSOIdentityAssertion(id_token=SecretStr("id-token"), refresh_token=SecretStr("sso-rt"))

    exchanged = await exchange_login_assertion_for_refresh_token(
        oidc, (_config(),), endpoint, _NOW, sso_client=sso_client
    )

    assert [(call[0], call[1], call[2]["subject_token"]) for call in endpoint.calls] == [
        ("https://idp.example.com/token", "agent", "id-token")
    ]
    assert exchanged.exchanged_refresh_token is not None
    assert exchanged.exchanged_refresh_token.get_secret_value() == "rt-1"


@pytest.mark.asyncio
async def test_the_login_refresh_token_binds_to_the_target_whose_client_issued_it_not_the_first():
    endpoint = _Endpoint(Ok(ExchangedToken(access_token="rt-1", expires_in=3600)))
    oidc = SSOIdentityAssertion(id_token=SecretStr("id-token"), refresh_token=SecretStr("sso-rt"))
    targets = (_config(client_id="other-agent"), _config(client_id="agent"))

    kept = await exchange_login_assertion_for_refresh_token(oidc, targets, endpoint, _NOW, sso_client=_sso_client())

    assert endpoint.calls == []
    assert kept.exchanged_refresh_token_client_id == "agent"
    assert kept.exchanged_refresh_token is not None and kept.exchanged_refresh_token.get_secret_value() == "sso-rt"


@pytest.mark.asyncio
async def test_a_saml_login_through_the_id_jag_client_still_exchanges_its_assertion():
    endpoint = _Endpoint(Ok(ExchangedToken(access_token="rt-1", expires_in=3600)))

    exchanged = await exchange_login_assertion_for_refresh_token(
        _saml(), (_config(),), endpoint, _NOW, sso_client=_sso_client()
    )

    assert endpoint.calls[0][2]["subject_token_type"] == "urn:ietf:params:oauth:token-type:saml2"
    assert exchanged.exchanged_refresh_token is not None
    assert exchanged.exchanged_refresh_token.get_secret_value() == "rt-1"


@pytest.mark.asyncio
async def test_a_login_already_exchanged_is_not_exchanged_again():
    endpoint = _Endpoint(Ok(ExchangedToken(access_token="rt-2", expires_in=3600)))
    done = with_login_refresh_token(_saml(), "rt-1", 3600, _NOW)

    assert await exchange_login_assertion_for_refresh_token(done, (_config(),), endpoint, _NOW) == done
    assert endpoint.calls == []


@pytest.mark.asyncio
async def test_several_idp_clients_mint_for_the_first_and_warn(caplog: pytest.LogCaptureFixture):
    endpoint = _Endpoint(Ok(ExchangedToken(access_token="rt-1", expires_in=3600)))
    first, second = _config(client_id="agent"), _config(client_id="other-agent")

    with caplog.at_level("WARNING"):
        exchanged = await exchange_login_assertion_for_refresh_token(_saml(), (first, second), endpoint, _NOW)

    assert [call[1] for call in endpoint.calls] == ["agent"]
    assert exchanged.exchanged_refresh_token is not None
    assert any("2 IdP clients" in message and "client_id=agent" in message for message in caplog.messages)


class _FakeCatalog:
    def __init__(self, config_servers: dict[str, MCPServer]) -> None:
        self.config_mcp_servers = config_servers
        self.built: list[tuple[str, bool]] = []

    async def build_mcp_server_from_table(
        self,
        mcp_server: LiteLLM_MCPServerTable,
        *,
        credentials_are_encrypted: bool = True,
        env_vars_are_encrypted: bool | None = None,
        register_oauth_discovery: bool = True,
    ) -> MCPServer:
        self.built.append((mcp_server.server_id, register_oauth_discovery))
        return MCPServer(
            server_id=mcp_server.server_id, name=mcp_server.server_id, url=mcp_server.url, transport=MCPTransport.http
        )


def _row(server_id: str, auth_type: MCPAuth, approval_status: str | None = None) -> LiteLLM_MCPServerTable:
    return LiteLLM_MCPServerTable(
        server_id=server_id,
        alias=server_id,
        url=f"https://{server_id}.example.com/mcp",
        transport=MCPTransport.http,
        auth_type=auth_type,
        approval_status=approval_status,
        created_at=_NOW,
        updated_at=_NOW,
    )


@pytest.mark.asyncio
async def test_login_exchange_servers_reads_config_and_db_id_jag_rows_not_the_registry():
    """A server added on another pod is in the DB before this pod's registry reloads, so the login
    exchange has to judge against the DB rows, like the retention gate, and only the ID-JAG ones."""
    config_server = MCPServer(
        server_id="from-config", name="from-config", url="https://cfg.example.com/mcp", transport=MCPTransport.http
    )
    catalog = _FakeCatalog({"from-config": config_server})

    servers = await login_exchange_servers(
        catalog,
        [
            _row("db-id-jag", MCPAuth.oauth2_id_jag),
            _row("db-id-jag-approved", MCPAuth.oauth2_id_jag, "approved"),
            _row("db-oauth", MCPAuth.oauth2),
        ],
    )

    assert [server.server_id for server in servers] == ["from-config", "db-id-jag", "db-id-jag-approved"]
    assert catalog.built == [("db-id-jag", False), ("db-id-jag-approved", False)]


@pytest.mark.asyncio
async def test_a_db_row_hides_the_config_server_with_the_same_id_like_the_live_catalog():
    """The catalog serves ``config_mcp_servers | registry``, so a DB row wins its server_id. The login
    exchange must mint for that live server, not for the hidden config entry's IdP client."""
    hidden = MCPServer(
        server_id="shared", name="shared", url="https://cfg.example.com/mcp", transport=MCPTransport.http
    )
    catalog = _FakeCatalog({"shared": hidden})

    servers = await login_exchange_servers(catalog, [_row("shared", MCPAuth.oauth2_id_jag)])

    assert [(server.server_id, server.url) for server in servers] == [("shared", "https://shared.example.com/mcp")]


@pytest.mark.asyncio
@pytest.mark.parametrize("approval_status", ["pending_review", "rejected", "draft"])
async def test_login_exchange_servers_skip_unapproved_submissions(approval_status: str):
    """A team key can submit an ID-JAG server naming any token endpoint. Until an admin approves it,
    no login assertion is posted there."""
    catalog = _FakeCatalog({})

    servers = await login_exchange_servers(catalog, [_row("submitted", MCPAuth.oauth2_id_jag, approval_status)])

    assert servers == ()
    assert catalog.built == []


class _StalledEndpoint(_Endpoint):
    """An IdP that accepts the connection and never answers."""

    async def fetch(
        self, endpoint: str, client_id: str, grant_params: dict[str, str], client_auth: ClientAuth
    ) -> Result[ExchangedToken, CredError]:
        await asyncio.sleep(0.5)
        return await super().fetch(endpoint, client_id, grant_params, client_auth)


@pytest.mark.asyncio
async def test_a_stalled_idp_does_not_hold_up_the_login(caplog: pytest.LogCaptureFixture):
    """The exchange is optional. A user who only wants the dashboard must not wait on an MCP IdP that
    stopped answering, so the exchange has its own short deadline and the login keeps the plain assertion."""
    saml = _saml()

    with caplog.at_level(logging.WARNING, logger="litellm.proxy"):
        exchanged = await exchange_login_assertion_for_refresh_token(
            saml,
            (_config(),),
            _StalledEndpoint(Ok(ExchangedToken(access_token="rt-1", expires_in=3600))),
            _NOW,
            deadline_seconds=0.05,
        )

    assert exchanged == saml
    assert any("did not answer the login-time refresh token exchange" in r.getMessage() for r in caplog.records)


class _OneBadRowCatalog(_FakeCatalog):
    async def build_mcp_server_from_table(
        self,
        mcp_server: LiteLLM_MCPServerTable,
        *,
        credentials_are_encrypted: bool = True,
        env_vars_are_encrypted: bool | None = None,
        register_oauth_discovery: bool = True,
    ) -> MCPServer:
        if mcp_server.server_id == "broken":
            raise ValueError("credentials cannot be decrypted")
        return await super().build_mcp_server_from_table(
            mcp_server,
            credentials_are_encrypted=credentials_are_encrypted,
            env_vars_are_encrypted=env_vars_are_encrypted,
            register_oauth_discovery=register_oauth_discovery,
        )


@pytest.mark.asyncio
async def test_one_unbuildable_db_row_does_not_cost_the_other_servers_their_exchange(caplog: pytest.LogCaptureFixture):
    catalog = _OneBadRowCatalog({})

    with caplog.at_level(logging.WARNING, logger="litellm.proxy"):
        servers = await login_exchange_servers(
            catalog, [_row("broken", MCPAuth.oauth2_id_jag), _row("healthy", MCPAuth.oauth2_id_jag)]
        )

    assert [server.server_id for server in servers] == ["healthy"]
    assert any("server_id=broken" in r.getMessage() for r in caplog.records)
