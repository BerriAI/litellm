"""Verify signed approval references for high-risk MCP tools.

A server admin marks tools as requiring approval via ``MCPServer.approval_policy``.
Calls to those tools must carry a compact JWS in the
``x-litellm-mcp-approval-reference`` header, signed by the approval service pinned
in the policy and bound to this server and this tool. The gateway only verifies:
it never waits on, notifies, or mints anything for the external approval workflow.
"""

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Literal, TypeAlias, cast

import jwt
from jwt.types import Options

from litellm._logging import verbose_logger
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.constants import (
    MCP_APPROVAL_JWKS_CACHE_TTL_SECONDS,
    MCP_APPROVAL_JWT_ALGORITHMS,
    MCP_APPROVAL_REFERENCE_HEADER,
)
from litellm.llms.custom_httpx.http_handler import get_async_httpx_client
from litellm.types.llms.custom_http import httpxSpecialProvider
from litellm.types.mcp_server.mcp_server_manager import MCPApprovalPolicy
from litellm.types.utils import StandardLoggingMCPApprovalReference

_jwks_cache: Final = InMemoryCache(default_ttl=MCP_APPROVAL_JWKS_CACHE_TTL_SECONDS)

JwksFetcher: TypeAlias = Callable[[str], Awaitable[Sequence[Mapping[str, object]]]]

_REQUIRED_CLAIMS: Final = ("iss", "exp", "jti", "mcp_server", "mcp_tool")


@dataclass(frozen=True, slots=True)
class ApprovalVerified:
    record: StandardLoggingMCPApprovalReference


@dataclass(frozen=True, slots=True)
class ApprovalRejected:
    reason: Literal["missing", "invalid", "expired"]
    message: str


@dataclass(frozen=True, slots=True)
class ApprovalVerifierUnavailable:
    message: str


ApprovalCheck: TypeAlias = ApprovalVerified | ApprovalRejected | ApprovalVerifierUnavailable


async def _fetch_jwks(jwks_url: str) -> Sequence[Mapping[str, object]]:
    cached: Final = await _jwks_cache.async_get_cache(jwks_url)
    if isinstance(cached, list):
        return cached
    client: Final = get_async_httpx_client(llm_provider=httpxSpecialProvider.Oauth2Check)
    response: Final = await client.get(jwks_url)
    response.raise_for_status()
    document: Final = cast(object, response.json())  # cast-ok: httpx Response.json() returns untyped Any
    keys: Final = document.get("keys") if isinstance(document, dict) else None
    if not isinstance(keys, list):
        raise TypeError(f"JWKS document at {jwks_url} has no 'keys' array")
    await _jwks_cache.async_set_cache(jwks_url, keys, ttl=MCP_APPROVAL_JWKS_CACHE_TTL_SECONDS)
    return keys


def _approval_header_value(raw_headers: Mapping[str, str] | None) -> str | None:
    return (
        next(
            (value for name, value in raw_headers.items() if name.lower() == MCP_APPROVAL_REFERENCE_HEADER),
            None,
        )
        if raw_headers
        else None
    )


def _select_signing_key(
    kid: object,
    keys: Sequence[Mapping[str, object]],
) -> "jwt.PyJWK | ApprovalRejected | ApprovalVerifierUnavailable":
    for key in keys:
        if kid is None or key.get("kid") == kid:
            try:
                return jwt.PyJWK(dict(key))
            except (jwt.PyJWTError, ValueError, TypeError):
                return ApprovalVerifierUnavailable(
                    message="the approval service's JWKS returned a signing key that could not be used"
                )
    return ApprovalRejected(
        reason="invalid",
        message="the approval reference's signing key is not in the approval service's JWKS",
    )


def _decode_claims(
    token: str,
    policy: MCPApprovalPolicy,
    signing_key: "jwt.PyJWK",
) -> "Mapping[str, object] | ApprovalRejected":
    decode_options: Final[Options] = {
        "require": list(_REQUIRED_CLAIMS),
        "verify_aud": policy.audience is not None,
    }
    try:
        return jwt.decode(
            token,
            signing_key,
            algorithms=list(MCP_APPROVAL_JWT_ALGORITHMS),
            issuer=policy.issuer,
            audience=policy.audience,
            options=decode_options,
        )
    except jwt.ExpiredSignatureError:
        return ApprovalRejected(reason="expired", message="the approval reference has expired")
    except Exception as exc:
        return ApprovalRejected(
            reason="invalid",
            message=f"the approval reference failed verification: {type(exc).__name__}",
        )


def _bind_claims(
    claims: Mapping[str, object],
    policy: MCPApprovalPolicy,
    policy_tool: str,
    server_id: str,
) -> "StandardLoggingMCPApprovalReference | ApprovalRejected":
    if claims.get("mcp_tool") != policy_tool:
        return ApprovalRejected(
            reason="invalid",
            message="the approval reference is not issued for this tool",
        )
    if not server_id or claims.get("mcp_server") != server_id:
        return ApprovalRejected(
            reason="invalid",
            message="the approval reference is not issued for this MCP server",
        )
    subject: Final = claims.get("sub")
    expires_at: Final = claims.get("exp")
    if not isinstance(expires_at, int | float):
        return ApprovalRejected(
            reason="invalid",
            message="the approval reference carries an unusable expiry claim",
        )
    return StandardLoggingMCPApprovalReference(
        jti=str(claims["jti"]),
        issuer=str(claims["iss"]),
        subject=subject if isinstance(subject, str) else None,
        expires_at=int(expires_at),
    )


async def verify_approval_reference(
    *,
    policy: MCPApprovalPolicy,
    policy_tool: str,
    server_id: str,
    raw_headers: Mapping[str, str] | None,
    fetch_jwks: JwksFetcher = _fetch_jwks,
) -> ApprovalCheck:
    """Verify the request's approval reference against ``policy`` for ``policy_tool``.

    ``policy_tool`` is the configured ``policy.tools`` entry that matched the called
    tool, so the ``mcp_tool`` claim binds to the spelling the admin configured.
    ``server_id`` is this server's unique identifier, the only value the ``mcp_server``
    claim may equal. Never raises: every failure is a value the caller maps onto its
    public error contract.
    """
    token: Final = _approval_header_value(raw_headers)
    if token is None:
        return ApprovalRejected(
            reason="missing",
            message="this tool requires an approval reference issued by the configured approval service",
        )
    try:
        header: Final = jwt.get_unverified_header(token)
    except jwt.PyJWTError:
        return ApprovalRejected(reason="invalid", message="the approval reference is not a decodable JWT")
    if header.get("alg") not in MCP_APPROVAL_JWT_ALGORITHMS:
        return ApprovalRejected(
            reason="invalid",
            message="the approval reference uses a disallowed signature algorithm",
        )
    try:
        keys: Final = await fetch_jwks(policy.jwks_url)
    except Exception as exc:  # noqa: BLE001  # a JWKS fetch failure must fail closed as unavailable, not allow
        verbose_logger.warning("could not fetch the approval service's JWKS: %s", exc)
        return ApprovalVerifierUnavailable(message=f"could not fetch the approval service's JWKS: {type(exc).__name__}")
    signing_key: Final = _select_signing_key(header.get("kid"), keys)
    if not isinstance(signing_key, jwt.PyJWK):
        return signing_key
    claims: Final = _decode_claims(token, policy, signing_key)
    if isinstance(claims, ApprovalRejected):
        return claims
    bound: Final = _bind_claims(claims, policy, policy_tool, server_id)
    if isinstance(bound, ApprovalRejected):
        return bound
    return ApprovalVerified(record=bound)
