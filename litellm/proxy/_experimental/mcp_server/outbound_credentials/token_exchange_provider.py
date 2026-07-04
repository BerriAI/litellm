"""Composition root for the v2-native token_exchange (OBO) exchanger.

Wires the pure ``OboTokenExchanger`` to its runtime edges: the real httpx POST against the IdP and
the configured cache sizing/TTL constants. ``build_token_exchanger`` is built once at egress
construction and reused, so the in-process exchanged-token cache survives across requests. Unlike the
per-user store, nothing here reads a runtime global at build time (the httpx client is acquired per
call), so it needs no lazy wrapper.
"""

from __future__ import annotations

import httpx

from litellm._logging import verbose_logger
from litellm.constants import (
    MCP_OAUTH2_TOKEN_CACHE_DEFAULT_TTL,
    MCP_OAUTH2_TOKEN_CACHE_MIN_TTL,
    MCP_OAUTH2_TOKEN_EXPIRY_BUFFER_SECONDS,
    MCP_TOKEN_EXCHANGE_CACHE_MAX_SIZE,
)
from litellm.proxy._experimental.mcp_server.outbound_credentials.oauth_token_store import (
    InMemoryTokenCacheBackend,
)
from litellm.proxy._experimental.mcp_server.outbound_credentials.token_exchanger import (
    OboTokenExchanger,
    SubjectTokenRejected,
    TokenExchangeClientError,
)

# RFC 6749 5.2 error codes that mean the gateway's own request/credentials are wrong (not the
# caller's subject token), so they surface as a 500 the caller can't fix by re-authenticating.
_GATEWAY_FAULT_OAUTH_ERRORS = frozenset(
    {"invalid_client", "unauthorized_client", "unsupported_grant_type", "invalid_target", "invalid_scope"}
)


def _oauth_error_fields(response: httpx.Response) -> tuple[str | None, str | None]:
    """Read the RFC 6749 5.2 ``error`` code and the IdP's step-up ``claims`` blob from a
    token-endpoint error body, as ``(error, claims)`` with None for whatever is absent.

    ``claims`` is the Entra Conditional Access / CAE challenge (a JSON string the client must
    replay to the IdP to satisfy the step-up); it is the caller's own requirement, not an IdP
    internal, so it may travel to the caller. The ``error_description`` is deliberately not read:
    it can carry IdP internals and must never reach the caller.
    """
    try:
        body: object = response.json()
    except Exception:  # noqa: BLE001
        return None, None
    if not isinstance(body, dict):
        return None, None
    code = body.get("error")
    claims = body.get("claims")
    return (
        code if isinstance(code, str) else None,
        claims if isinstance(claims, str) and claims else None,
    )


async def _post_exchange_endpoint(
    url: str, form: dict[str, str], client_auth_headers: dict[str, str]
) -> dict[str, object] | None:
    from litellm.llms.custom_httpx.http_handler import (  # noqa: PLC0415
        get_async_httpx_client,  # pyright: ignore
    )
    from litellm.types.llms.custom_http import httpxSpecialProvider  # noqa: PLC0415

    # litellm's httpx handler and httpx.Response are only partially typed; the IdP returns a JSON
    # object and the exchanger validates each field, so the untyped boundary is contained here.
    # A 4xx is the IdP rejecting the subject (non-retryable -> 401 via SubjectTokenRejected); any
    # other failure is a miss (-> None -> upstream_unavailable -> 503), matching v1's fail-closed.
    headers = {"Accept": "application/json", **client_auth_headers}
    try:
        client = get_async_httpx_client(llm_provider=httpxSpecialProvider.MCP)  # pyright: ignore
        response = await client.post(url, headers=headers, data=form)  # pyright: ignore
        response.raise_for_status()  # pyright: ignore
        parsed: object = response.json()  # pyright: ignore
    except httpx.HTTPStatusError as status_err:
        status_code = status_err.response.status_code
        if 400 <= status_code < 500:
            oauth_error, claims = _oauth_error_fields(status_err.response)
            if oauth_error in _GATEWAY_FAULT_OAUTH_ERRORS:
                verbose_logger.warning(
                    "MCP token exchange rejected as %s (HTTP %d); check the gateway client credentials, "
                    "audience, and scope for this server",
                    oauth_error,
                    status_code,
                )
                raise TokenExchangeClientError(oauth_error) from status_err
            raise SubjectTokenRejected(
                f"IdP rejected the subject token (HTTP {status_code})",
                claims=claims,
            ) from status_err
        verbose_logger.warning("MCP token exchange request failed: %s", status_err)
        return None
    except Exception as exc:  # noqa: BLE001
        verbose_logger.warning("MCP token exchange request failed: %s", exc)
        return None
    if not isinstance(parsed, dict):
        # A valid-but-non-object JSON body (list/string/number) would crash the field parsing; map it
        # to a miss so it surfaces as a typed upstream_unavailable, not a 500.
        verbose_logger.warning("MCP token exchange returned non-object JSON (%s)", type(parsed).__name__)
        return None
    return parsed  # pyright: ignore


def build_token_exchanger() -> OboTokenExchanger:
    return OboTokenExchanger(
        _post_exchange_endpoint,
        cache=InMemoryTokenCacheBackend(max_size=MCP_TOKEN_EXCHANGE_CACHE_MAX_SIZE),
        default_ttl_seconds=MCP_OAUTH2_TOKEN_CACHE_DEFAULT_TTL,
        min_ttl_seconds=MCP_OAUTH2_TOKEN_CACHE_MIN_TTL,
        expiry_buffer_seconds=MCP_OAUTH2_TOKEN_EXPIRY_BUFFER_SECONDS,
    )
