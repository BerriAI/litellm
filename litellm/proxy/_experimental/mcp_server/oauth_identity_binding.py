"""Per-user OAuth identity binding: verify the upstream OIDC principal matches the LiteLLM caller.

Closes the confused-deputy gap where a browser authenticated upstream as one principal produces a
token that the relay stores under a different, LiteLLM-authenticated principal: before the token
endpoint returns, stores, or caches an exchanged token for an identity-bound server, the id_token
is validated (signature via the pinned issuer's JWKS, issuer, audience, expiry) and its principal
claim is compared to the caller's trusted LiteLLM identity. Mismatches fail closed in enforce mode
and are logged in audit mode.
"""

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Final, Literal

import jwt
from fastapi import HTTPException

from litellm._logging import verbose_logger
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.llms.custom_httpx.http_handler import get_async_httpx_client
from litellm.types.llms.custom_http import httpxSpecialProvider
from litellm.types.mcp_server.mcp_server_manager import MCPOAuthIdentityBinding, MCPServer

_ALLOWED_ID_TOKEN_ALGORITHMS: Final = (
    "RS256",
    "RS384",
    "RS512",
    "ES256",
    "ES384",
    "ES512",
    "PS256",
    "PS384",
    "PS512",
)
_JWKS_CACHE_TTL_SECONDS: Final = 3600
_jwks_cache: Final = InMemoryCache(default_ttl=_JWKS_CACHE_TTL_SECONDS)

JwksFetcher = Callable[[MCPOAuthIdentityBinding], Awaitable[list[Mapping[str, object]]]]
CallerPrincipalLoader = Callable[[str, MCPOAuthIdentityBinding], Awaitable[str | None]]

_RejectionCode = Literal["oauth_principal_mismatch", "oauth_identity_binding_failed"]


@dataclass(frozen=True, slots=True)
class _BindingRejection:
    code: _RejectionCode
    description: str


async def _fetch_issuer_jwks(binding: MCPOAuthIdentityBinding) -> list[Mapping[str, object]]:
    jwks_url: Final[str] = binding.jwks_url or await _discover_jwks_url(binding.issuer)
    cached: Final = await _jwks_cache.async_get_cache(jwks_url)
    if isinstance(cached, list):
        return cached
    client: Final = get_async_httpx_client(llm_provider=httpxSpecialProvider.Oauth2Check)
    response: Final = await client.get(jwks_url)
    response.raise_for_status()
    document: Final = response.json()
    keys: Final = document.get("keys") if isinstance(document, dict) else None
    if not isinstance(keys, list):
        raise TypeError(f"JWKS document at {jwks_url} has no 'keys' array")
    await _jwks_cache.async_set_cache(jwks_url, keys, ttl=_JWKS_CACHE_TTL_SECONDS)
    return keys


async def _discover_jwks_url(issuer: str) -> str:
    discovery_url: Final = f"{issuer.rstrip('/')}/.well-known/openid-configuration"
    client: Final = get_async_httpx_client(llm_provider=httpxSpecialProvider.Oauth2Check)
    response: Final = await client.get(discovery_url)
    response.raise_for_status()
    metadata: Final = response.json()
    jwks_uri: Final = metadata.get("jwks_uri") if isinstance(metadata, dict) else None
    if not isinstance(jwks_uri, str) or not jwks_uri:
        raise ValueError(f"OIDC discovery at {discovery_url} returned no jwks_uri")
    return jwks_uri


def _select_signing_key(id_token: str, keys: list[Mapping[str, object]]) -> "jwt.PyJWK | _BindingRejection":
    header: Final = jwt.get_unverified_header(id_token)
    kid: Final = header.get("kid")
    for key in keys:
        if kid is None or key.get("kid") == kid:
            return jwt.PyJWK(dict(key))
    return _BindingRejection(
        code="oauth_identity_binding_failed",
        description=f"id_token signing key (kid={kid!r}) not found in the issuer's JWKS",
    )


def _decode_id_token(
    id_token: str,
    binding: MCPOAuthIdentityBinding,
    signing_key: "jwt.PyJWK",
) -> "Mapping[str, object] | _BindingRejection":
    try:
        return jwt.decode(
            id_token,
            signing_key.key,
            algorithms=list(_ALLOWED_ID_TOKEN_ALGORITHMS),
            issuer=binding.issuer,
            audience=binding.audiences if binding.audiences else None,
            options={
                "require": ["iss", "exp"],
                "verify_aud": bool(binding.audiences),
            },
        )
    except jwt.InvalidTokenError as exc:
        return _BindingRejection(
            code="oauth_identity_binding_failed",
            description=f"id_token validation failed: {exc}",
        )


def _upstream_principal(
    claims: Mapping[str, object],
    binding: MCPOAuthIdentityBinding,
) -> "str | _BindingRejection":
    principal: Final = claims.get(binding.principal_claim)
    if not isinstance(principal, str) or not principal:
        return _BindingRejection(
            code="oauth_identity_binding_failed",
            description=f"id_token has no usable '{binding.principal_claim}' claim",
        )
    if binding.principal_claim == "email" and binding.require_email_verified and claims.get("email_verified") is not True:
        return _BindingRejection(
            code="oauth_identity_binding_failed",
            description="id_token email is not verified (email_verified is not true)",
        )
    return principal


async def _load_caller_principal(litellm_user_id: str, binding: MCPOAuthIdentityBinding) -> str | None:
    if binding.caller_field == "user_id":
        return litellm_user_id
    from litellm.proxy._experimental.mcp_server.bridge_token_flow import (  # noqa: PLC0415  # inline import avoids a module-load circular import
        load_active_user_by_id,
    )

    loaded: Final = await load_active_user_by_id(litellm_user_id)
    if isinstance(loaded, str):
        return None
    return loaded.user_email


def _principals_match(upstream: str, caller: str, binding: MCPOAuthIdentityBinding) -> bool:
    if binding.principal_claim == "email" or binding.caller_field == "user_email":
        return upstream.strip().casefold() == caller.strip().casefold()
    return upstream == caller


async def _evaluate_binding(
    binding: MCPOAuthIdentityBinding,
    token_response: Mapping[str, object],
    litellm_user_id: str | None,
    grant_type: str,
    jwks_fetcher: JwksFetcher,
    caller_principal_loader: CallerPrincipalLoader,
) -> _BindingRejection | None:
    id_token: Final = token_response.get("id_token")
    if not isinstance(id_token, str) or not id_token:
        if grant_type == "refresh_token":
            return None
        return _BindingRejection(
            code="oauth_identity_binding_failed",
            description="the upstream token response carries no id_token to bind the credential to a principal",
        )
    if not litellm_user_id:
        return _BindingRejection(
            code="oauth_identity_binding_failed",
            description="the request carries no resolvable LiteLLM user identity to bind the credential to",
        )
    try:
        keys: Final = await jwks_fetcher(binding)
    except Exception as exc:  # noqa: BLE001  # a JWKS fetch failure must fail closed, not surface as a 500
        return _BindingRejection(
            code="oauth_identity_binding_failed",
            description=f"could not fetch the issuer's JWKS: {exc}",
        )
    signing_key: Final = _select_signing_key(id_token, keys)
    if isinstance(signing_key, _BindingRejection):
        return signing_key
    claims: Final = _decode_id_token(id_token, binding, signing_key)
    if isinstance(claims, _BindingRejection):
        return claims
    upstream: Final = _upstream_principal(claims, binding)
    if isinstance(upstream, _BindingRejection):
        return upstream
    caller: Final = await caller_principal_loader(litellm_user_id, binding)
    if not caller:
        return _BindingRejection(
            code="oauth_identity_binding_failed",
            description=f"the LiteLLM user has no '{binding.caller_field}' to compare the upstream principal against",
        )
    if not _principals_match(upstream, caller, binding):
        return _BindingRejection(
            code="oauth_principal_mismatch",
            description="The browser account does not match the selected credential owner.",
        )
    return None


async def enforce_oauth_identity_binding(
    server: MCPServer,
    token_response: Mapping[str, object],
    litellm_user_id: str | None,
    grant_type: str,
    jwks_fetcher: JwksFetcher = _fetch_issuer_jwks,
    caller_principal_loader: CallerPrincipalLoader = _load_caller_principal,
) -> None:
    """Validate the exchanged token's upstream principal against the LiteLLM caller.

    No-op when the server has no binding or it is disabled. In enforce mode a failure raises 403
    before the caller returns, stores, or caches the token; in audit mode failures are logged only.
    A refresh_token grant without an id_token is allowed in both modes: the stored credential keeps
    the binding established at the original authorization_code exchange.
    """
    binding: Final = server.oauth_identity_binding
    if binding is None or binding.mode == "disabled":
        return
    rejection: Final = await _evaluate_binding(
        binding=binding,
        token_response=token_response,
        litellm_user_id=litellm_user_id,
        grant_type=grant_type,
        jwks_fetcher=jwks_fetcher,
        caller_principal_loader=caller_principal_loader,
    )
    if rejection is None:
        return
    if binding.mode == "audit":
        verbose_logger.warning(
            "oauth_identity_binding audit: server=%s user=%s grant=%s rejected=%s (%s)",
            server.server_id,
            litellm_user_id,
            grant_type,
            rejection.code,
            rejection.description,
        )
        return
    raise HTTPException(
        status_code=403,
        detail={
            "error": rejection.code,
            "error_description": rejection.description,
            "server_id": server.server_id,
            "credential_owner": "caller",
            "credential_stored": False,
        },
    )
