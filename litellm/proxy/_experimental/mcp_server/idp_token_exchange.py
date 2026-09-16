"""The identity-provider side of the RFC 8693 token exchange on ``POST /token``: a native
client that already holds a JWT from the customer's IdP trades it for the same proxy-API
credential ``lite login`` stores, proven by the proxy's own JWT auth (signature, claims,
and the user and team sync it performs), so no browser round trip is needed."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Final, Protocol

from fastapi import HTTPException, Request

from litellm._logging import verbose_proxy_logger
from litellm.proxy._experimental.mcp_server.gateway_dcr_flow import SubjectIdentity, SubjectTokenRefusal
from litellm.proxy._types import JWTAuthBuilderResult, ProxyException
from litellm.proxy.auth.handle_jwt import JWTAuthManager

EXCHANGE_ROUTE: Final = "/token"
REJECTED_SUBJECT_TOKEN: Final = "subject_token was rejected by the gateway's JWT auth"


@dataclass(frozen=True, slots=True)
class TokenExchangePrerequisites:
    """The deployment-level gates ``user_api_key_auth`` applies before it verifies any JWT
    bearer. Discovery and registration advertise the exchange grant only when every one of
    them holds, and an exchange attempt is refused naming the first one that does not."""

    jwt_auth_enabled: bool
    has_database: bool
    licensed: bool

    @property
    def available(self) -> bool:
        return self.jwt_auth_enabled and self.has_database and self.licensed

    def refusal(self) -> SubjectTokenRefusal | None:
        if not self.jwt_auth_enabled:
            return SubjectTokenRefusal(
                error="unsupported_grant_type",
                description="JWT auth is not enabled on this gateway, so it cannot exchange IdP tokens",
            )
        if not self.has_database:
            return SubjectTokenRefusal(
                error="unsupported_grant_type",
                description="this gateway has no database, so it cannot exchange IdP tokens",
            )
        if not self.licensed:
            return SubjectTokenRefusal(
                error="unsupported_grant_type",
                description="JWT auth is an enterprise only feature; no license is set",
            )
        return None


def read_token_exchange_prerequisites() -> TokenExchangePrerequisites:
    from litellm.proxy.proxy_server import (  # noqa: PLC0415  # rebound after startup, so read them per call
        general_settings,
        premium_user,
        prisma_client,
    )

    return TokenExchangePrerequisites(
        jwt_auth_enabled=general_settings.get("enable_jwt_auth", False) is True,
        has_database=prisma_client is not None,
        licensed=premium_user is True,
    )


def token_exchange_available() -> bool:
    return read_token_exchange_prerequisites().available


class AuthorizeSubjectToken(Protocol):
    """Injected JWT authorization ``(subject_token, request_headers)``: the proxy's
    ``JWTAuthManager.auth_builder`` in production, which raises when the token is not
    acceptable and otherwise names the user and team it resolved."""

    def __call__(
        self, subject_token: str, request_headers: Mapping[str, str], /
    ) -> Awaitable[JWTAuthBuilderResult]: ...


async def exchange_idp_subject_token(subject_token: str, request: Request) -> SubjectIdentity | SubjectTokenRefusal:
    from litellm.proxy.proxy_server import (  # noqa: PLC0415  # rebound after startup, so read them per call
        general_settings,
        jwt_handler,
        prisma_client,
        proxy_logging_obj,
        user_api_key_cache,
    )

    async def authorize(token: str, request_headers: Mapping[str, str]) -> JWTAuthBuilderResult:
        return await JWTAuthManager.auth_builder(
            api_key=token,
            jwt_handler=jwt_handler,
            request_data={},
            general_settings=general_settings,
            route=EXCHANGE_ROUTE,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            parent_otel_span=None,
            proxy_logging_obj=proxy_logging_obj,
            request_headers=request_headers,
            request_method="POST",
        )

    return await identity_from_subject_token(
        subject_token,
        request_headers=request.headers,
        prerequisites=read_token_exchange_prerequisites(),
        is_jwt=jwt_handler.is_jwt,
        authorize=authorize,
    )


async def identity_from_subject_token(
    subject_token: str,
    request_headers: Mapping[str, str],
    prerequisites: TokenExchangePrerequisites,
    is_jwt: Callable[[str], bool],
    authorize: AuthorizeSubjectToken,
) -> SubjectIdentity | SubjectTokenRefusal:
    """Apply the same gates ``user_api_key_auth`` applies to a JWT bearer, then let the
    proxy's JWT auth prove the token. A rejection comes back as ``invalid_request``, which
    RFC 8693 section 2.2.2 prescribes for an invalid or unacceptable subject token. The
    reason stays in the proxy log: this endpoint is public and JWT auth's own wording can
    name the JWKS URL it fetched or quote the IdP's response."""
    unmet: Final = prerequisites.refusal()
    if unmet is not None:
        return unmet
    if not is_jwt(subject_token):
        return SubjectTokenRefusal(error="invalid_request", description="subject_token is not a JWT")
    try:
        result: Final = await authorize(subject_token, request_headers)
    except HTTPException as denied:
        return _rejected_by_jwt_auth(denied.detail)
    except ProxyException as denied:
        return _rejected_by_jwt_auth(denied.message)
    except Exception as denied:  # noqa: BLE001  # auth_jwt raises a plain Exception on signature and claim failures
        return _rejected_by_jwt_auth(denied)
    user_id: Final = result["user_id"]
    if user_id is None:
        return SubjectTokenRefusal(error="invalid_request", description="subject_token names no user the gateway knows")
    return SubjectIdentity(user_id=user_id, team_id=result["team_id"])


def _rejected_by_jwt_auth(reason: object) -> SubjectTokenRefusal:
    verbose_proxy_logger.warning("token exchange refused a subject_token: %s", reason)
    return SubjectTokenRefusal(error="invalid_request", description=REJECTED_SUBJECT_TOKEN)
