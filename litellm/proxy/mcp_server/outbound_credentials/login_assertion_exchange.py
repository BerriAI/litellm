"""Exchange the identity assertion captured at SSO login for a refresh token, at login.

An ``oauth2_id_jag`` server configured with ``id_jag_subject_source: refresh_token`` does not present the
login assertion on ID-JAG leg 1. It presents a refresh token that the IdP issued for that assertion through
one RFC 8693 exchange (``requested_token_type`` refresh_token), which is how a SAML requesting app works at
an IdP whose ID-JAG exchange does not take a ``saml2`` subject directly. That exchange has to happen at
login: a SAML assertion is valid for minutes, the first MCP call that needs it can come hours later, and
the refresh token outlives the assertion by as long as the IdP decides. An OIDC login signed in through the very
IdP client the ID-JAG server uses already holds such a refresh token: it is bound to that client, so it is
presented as is and no further exchange is posted, which Okta refuses for an id_token subject anyway.

The servers consulted are the two configuration authorities, the pod-local config declaration and the DB
rows, like the retention gate, not the in-memory registry, which is a per-process snapshot that can lag a
server added on another pod. The exchange authenticates as the ID-JAG server's own IdP client, since the
refresh token is bound to the client that later spends it. One row holds one refresh token per user, so every server in this mode has
to share one IdP client; a second client is logged and skipped rather than silently handed a token it
cannot redeem. A refused or unreachable IdP never fails the login, the row keeps the plain assertion and
the ID-JAG call reports the missing refresh token as the precondition it is.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from typing import Final, Protocol

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import LiteLLM_MCPServerTable
from litellm.proxy.mcp_server.outbound_credentials.result import Error, Ok, Result
from litellm.proxy.mcp_server.outbound_credentials.sso_assertion_refresher import (
    SSOClientConfig,
    sso_client_config,
)
from litellm.proxy.mcp_server.outbound_credentials.sso_assertion_store import (
    SSOIdentityAssertion,
    with_login_refresh_token,
)
from litellm.proxy.mcp_server.outbound_credentials.token_endpoint import ExchangedToken
from litellm.proxy.mcp_server.outbound_credentials.types import (
    ClientAuth,
    CredError,
    IdJagConfig,
    IdJagSubjectSource,
    ServerSpec,
)
from litellm.types.mcp import MCPAuth
from litellm.types.mcp_server.mcp_server_manager import MCPServer

_TOKEN_EXCHANGE_GRANT_TYPE: Final = "urn:ietf:params:oauth:grant-type:token-exchange"
REFRESH_TOKEN_TOKEN_TYPE: Final = "urn:ietf:params:oauth:token-type:refresh_token"
_LOGIN_EXCHANGE_DEADLINE_SECONDS: Final = 15.0

_REFRESH_TOKEN_SCOPE: Final = "openid offline_access"
_RUNTIME_APPROVAL_STATES: Final = (None, "active", "approved")  # the states get_runtime_mcp_server_rows serves


class LoginExchangeCatalog(Protocol):
    """The slice of the MCP server manager the login-time exchange reads."""

    @property
    def config_mcp_servers(self) -> Mapping[str, MCPServer]: ...

    async def build_mcp_server_from_table(
        self,
        mcp_server: LiteLLM_MCPServerTable,
        *,
        credentials_are_encrypted: bool = True,
        env_vars_are_encrypted: bool | None = None,
        register_oauth_discovery: bool = True,
    ) -> MCPServer: ...


class TokenExchangeEndpoint(Protocol):
    async def fetch(
        self, endpoint: str, client_id: str, grant_params: Mapping[str, str], client_auth: ClientAuth
    ) -> Result[ExchangedToken, CredError]: ...


def _is_runtime_id_jag_row(row: LiteLLM_MCPServerTable) -> bool:
    """An ID-JAG row the runtime registry would serve. A pending or rejected submission never gets a
    login assertion posted to the token endpoint it names."""
    return row.auth_type == MCPAuth.oauth2_id_jag and row.approval_status in _RUNTIME_APPROVAL_STATES


async def login_exchange_servers(
    catalog: LoginExchangeCatalog, db_rows: Sequence[LiteLLM_MCPServerTable]
) -> tuple[MCPServer, ...]:
    """Every server the exchange has to consider: the config-declared ones plus the approved ID-JAG rows
    in the DB, built the same way the catalog builds them so their credentials are decrypted. A DB row
    hides a config server with the same ``server_id``, exactly as the live catalog serves them."""
    id_jag_rows: Final = tuple(row for row in db_rows if _is_runtime_id_jag_row(row))
    db_servers: Final = tuple([await _built_or_skipped(catalog, row) for row in id_jag_rows])
    db_by_id: Final = {server.server_id: server for server in db_servers if server is not None}
    return tuple({**catalog.config_mcp_servers, **db_by_id}.values())


async def _built_or_skipped(catalog: LoginExchangeCatalog, row: LiteLLM_MCPServerTable) -> MCPServer | None:
    """One row that cannot be built (undecryptable credentials, a half-configured submission) is skipped
    with a warning. It must not cost every other refresh-token server its login exchange."""
    try:
        return await catalog.build_mcp_server_from_table(row, register_oauth_discovery=False)
    except Exception as exc:  # noqa: BLE001  # a per-row build failure is logged and isolated, not propagated
        verbose_proxy_logger.warning(
            "ID-JAG: MCP server row server_id=%s could not be built for the login-time refresh token exchange "
            "and is skipped: %s",
            row.server_id,
            exc,
        )
        return None


def refresh_token_exchange_targets(specs: Iterable[ServerSpec | None]) -> tuple[IdJagConfig, ...]:
    """The ID-JAG configs that present a login refresh token, one per IdP client, in registration order."""
    configs: Final = tuple(
        spec.config
        for spec in specs
        if spec is not None
        and isinstance(spec.config, IdJagConfig)
        and spec.config.subject_source is IdJagSubjectSource.refresh_token
    )
    clients: Final = tuple((config.org_token_endpoint, config.client_id) for config in configs)
    return tuple(config for index, config in enumerate(configs) if clients.index(clients[index]) == index)


async def exchange_login_assertion_for_refresh_token(
    assertion: SSOIdentityAssertion,
    targets: tuple[IdJagConfig, ...],
    token_endpoint: TokenExchangeEndpoint,
    now: datetime,
    deadline_seconds: float = _LOGIN_EXCHANGE_DEADLINE_SECONDS,
    sso_client: SSOClientConfig | None = None,
) -> SSOIdentityAssertion:
    """The assertion as captured when no server wants a refresh token, otherwise the assertion plus the
    refresh token one target's IdP client can redeem. An OIDC login's own refresh token is that token
    when the SSO client that received it is the target's client at the same token endpoint; otherwise it
    is bound to a client the ID-JAG client is not, and the assertion is exchanged for one at the first
    target's client."""
    if not targets or assertion.exchanged_refresh_token is not None:
        return assertion
    reusable: Final = _target_redeeming_the_login_refresh_token(assertion, targets, sso_client)
    if reusable is not None and assertion.refresh_token is not None:
        verbose_proxy_logger.info(
            "ID-JAG: the refresh token issued at OIDC login belongs to client_id=%s at %s, the IdP client of a server "
            "with id_jag_subject_source: refresh_token, so it is kept as that subject and no further exchange is posted",
            reusable.client_id,
            reusable.org_token_endpoint,
        )
        return with_login_refresh_token(
            assertion,
            assertion.refresh_token.get_secret_value(),
            None,
            now,
            endpoint=reusable.org_token_endpoint,
            client_id=reusable.client_id,
        )
    config: Final = targets[0]
    if len(targets) > 1:
        verbose_proxy_logger.warning(
            "ID-JAG: %d IdP clients are configured with id_jag_subject_source refresh_token; the login refresh "
            "token is minted for client_id=%s at %s and servers using another client cannot redeem it.",
            len(targets),
            config.client_id,
            config.org_token_endpoint,
        )
    params: Final = {
        "grant_type": _TOKEN_EXCHANGE_GRANT_TYPE,
        "requested_token_type": REFRESH_TOKEN_TOKEN_TYPE,
        "subject_token": assertion.id_token.get_secret_value(),
        "subject_token_type": assertion.subject_token_type or config.subject_token_type,
        "scope": _REFRESH_TOKEN_SCOPE,
    }
    try:
        outcome: Final = await asyncio.wait_for(
            token_endpoint.fetch(config.org_token_endpoint, config.client_id, params, config.client_auth),
            timeout=deadline_seconds,
        )
    except (TimeoutError, asyncio.TimeoutError):
        verbose_proxy_logger.warning(
            "ID-JAG: the IdP at %s did not answer the login-time refresh token exchange within %.0fs, so the login "
            "finishes with the captured assertion only. Servers with id_jag_subject_source: refresh_token will ask "
            "this user to sign in again.",
            config.org_token_endpoint,
            deadline_seconds,
        )
        return assertion
    match outcome:
        case Error(err):
            verbose_proxy_logger.warning(
                "ID-JAG: the IdP at %s refused to exchange the login assertion for a refresh token (%s); the "
                "login goes on with the assertion alone and servers presenting a refresh token will ask the "
                "user to sign in again.",
                config.org_token_endpoint,
                err.summary,
            )
            return assertion
        case Ok(token):
            verbose_proxy_logger.info(
                "ID-JAG: exchanged the login assertion for a refresh token at %s (expires_in=%s)",
                config.org_token_endpoint,
                token.expires_in,
            )
            return with_login_refresh_token(
                assertion,
                token.access_token,
                token.expires_in,
                now,
                endpoint=config.org_token_endpoint,
                client_id=config.client_id,
            )


def _target_redeeming_the_login_refresh_token(
    assertion: SSOIdentityAssertion, targets: tuple[IdJagConfig, ...], sso_client: SSOClientConfig | None
) -> IdJagConfig | None:
    """The target whose IdP client is the SSO client the login's refresh token was issued to. A refresh
    token is redeemable only by the client that received it at the endpoint that issued it, so both have
    to match exactly before the login token is presented anywhere."""
    if assertion.refresh_token is None or sso_client is None:
        return None
    return next(
        (
            config
            for config in targets
            if config.org_token_endpoint == sso_client.token_endpoint and config.client_id == sso_client.client_id
        ),
        None,
    )


async def exchange_login_assertion_if_configured(
    assertion: SSOIdentityAssertion | None,
) -> SSOIdentityAssertion | None:
    """The SSO-callback hook: consults the configured and DB-declared MCP servers and performs the
    exchange when one of them presents a login refresh token. Nothing here can fail the login."""
    if assertion is None:
        return None
    from litellm.proxy.mcp_server.db import get_all_mcp_servers  # noqa: PLC0415  # import cycle
    from litellm.proxy.mcp_server.mcp_server_manager import (  # noqa: PLC0415  # avoids import cycle
        global_mcp_server_manager,
    )
    from litellm.proxy.mcp_server.outbound_credentials.adapter import (  # noqa: PLC0415  # import cycle
        to_server_spec,
    )
    from litellm.proxy.mcp_server.outbound_credentials.token_endpoint import (  # noqa: PLC0415  # cycle
        TokenEndpointClient,
    )
    from litellm.proxy.proxy_server import prisma_client  # noqa: PLC0415  # runtime global

    try:
        db_rows: Final = () if prisma_client is None else await get_all_mcp_servers(prisma_client)
        servers: Final = await login_exchange_servers(global_mcp_server_manager, db_rows)
        targets: Final = refresh_token_exchange_targets(to_server_spec(server) for server in servers)
        return await exchange_login_assertion_for_refresh_token(
            assertion,
            targets,
            TokenEndpointClient(),
            datetime.now(timezone.utc),
            sso_client=sso_client_config(os.environ),
        )
    except Exception as exc:  # noqa: BLE001  # the login itself must not fail on an egress-side exchange
        verbose_proxy_logger.warning("ID-JAG: the login-time refresh token exchange failed unexpectedly: %s", exc)
        return assertion
