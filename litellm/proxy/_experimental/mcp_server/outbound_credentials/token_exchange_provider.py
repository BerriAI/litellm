"""Composition root for the v2-native token_exchange (OBO) exchanger.

Wires the pure ``OboTokenExchanger`` to its runtime edges: the real httpx POST against the IdP and
the configured cache sizing/TTL constants. ``build_token_exchanger`` is built once at egress
construction and reused, so the in-process exchanged-token cache survives across requests. Unlike the
per-user store, nothing here reads a runtime global at build time (the httpx client is acquired per
call), so it needs no lazy wrapper.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

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
    ExchangeHttpPost,
    OboTokenExchanger,
    SubjectTokenRejected,
    TokenExchangeClientError,
)

# RFC 6749 5.2 error codes that mean the gateway's own request/credentials are wrong (not the
# caller's subject token), so they surface as a 500 the caller can't fix by re-authenticating.
_GATEWAY_FAULT_OAUTH_ERRORS: Final = frozenset(
    {"invalid_client", "unauthorized_client", "unsupported_grant_type", "invalid_target", "invalid_scope"}
)
_INVALID_ASSERTION_AADSTS_PREFIX: Final = "50027"


@dataclass(frozen=True, slots=True)
class OAuthErrorBody:
    error: str | None
    claims: str | None
    error_codes: tuple[str, ...]

    @property
    def gateway_fault(self) -> str | None:
        if self.error is None or self.error not in _GATEWAY_FAULT_OAUTH_ERRORS:
            return None
        if any(code.startswith(_INVALID_ASSERTION_AADSTS_PREFIX) for code in self.error_codes):
            return None
        return self.error


def oauth_error_fields(response: httpx.Response) -> OAuthErrorBody:
    """Read the RFC 6749 5.2 ``error`` code, the IdP's step-up ``claims`` blob and Entra's
    ``error_codes`` sub-codes from a token-endpoint error body, None or empty for whatever is absent.

    ``claims`` is the Entra Conditional Access / CAE challenge (a JSON string the client must
    replay to the IdP to satisfy the step-up); it is the caller's own requirement, not an IdP
    internal, so it may travel to the caller. The ``error_description`` is deliberately not read:
    it can carry IdP internals and must never reach the caller.
    """
    try:
        body: Final[object] = response.json()
    except Exception:  # noqa: BLE001
        return OAuthErrorBody(error=None, claims=None, error_codes=())
    if not isinstance(body, dict):
        return OAuthErrorBody(error=None, claims=None, error_codes=())
    code: Final = body.get("error")
    claims: Final = body.get("claims")
    raw_codes: Final = body.get("error_codes")
    return OAuthErrorBody(
        error=code if isinstance(code, str) else None,
        claims=claims if isinstance(claims, str) and claims else None,
        error_codes=tuple(str(c) for c in raw_codes if isinstance(c, (int, str)))
        if isinstance(raw_codes, list)
        else (),
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
    headers: Final = {"Accept": "application/json", **client_auth_headers}
    try:
        client: Final = get_async_httpx_client(llm_provider=httpxSpecialProvider.MCP)  # pyright: ignore
        response: Final = await client.post(  # pyright: ignore[reportUnknownMemberType]  # untyped handler
            url, headers=headers, data=form
        )
        response.raise_for_status()  # pyright: ignore
        parsed: Final[object] = response.json()  # pyright: ignore
    except httpx.HTTPStatusError as status_err:
        status_code: Final = status_err.response.status_code
        if status_code in (408, 429):
            # Retry hints, not subject rejections: the IdP is shedding load, so a 401 would tell the
            # caller to sign in again for nothing; surface it like a transport failure.
            verbose_logger.warning("MCP token exchange throttled or timed out (HTTP %d)", status_code)
            return None
        if 400 <= status_code < 500:
            oauth_error: Final = oauth_error_fields(status_err.response)
            gateway_fault: Final = oauth_error.gateway_fault
            if gateway_fault is not None:
                verbose_logger.warning(
                    "MCP token exchange rejected as %s (HTTP %d); check the gateway client credentials, "
                    "audience, and scope for this server",
                    gateway_fault,
                    status_code,
                )
                raise TokenExchangeClientError(gateway_fault) from status_err
            raise SubjectTokenRejected(
                f"IdP rejected the subject token (HTTP {status_code})",
                claims=oauth_error.claims,
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


def build_token_exchanger(*, post: ExchangeHttpPost = _post_exchange_endpoint) -> OboTokenExchanger:
    return OboTokenExchanger(
        post,
        cache=InMemoryTokenCacheBackend(max_size=MCP_TOKEN_EXCHANGE_CACHE_MAX_SIZE),
        default_ttl_seconds=MCP_OAUTH2_TOKEN_CACHE_DEFAULT_TTL,
        min_ttl_seconds=MCP_OAUTH2_TOKEN_CACHE_MIN_TTL,
        expiry_buffer_seconds=MCP_OAUTH2_TOKEN_EXPIRY_BUFFER_SECONDS,
    )
