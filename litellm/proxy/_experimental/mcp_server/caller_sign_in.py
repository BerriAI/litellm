"""Caller-side sign-in requirements for MCP connects.

A guardrail that evaluates tool calls in the caller's own identity (an On-Behalf-Of exchange of the caller's
bearer) needs the caller signed in with its issuer before the first tool call, and a tool call's JSON-RPC
error cannot carry ``WWW-Authenticate``. ``token_exchange`` (OBO) servers have the same need: the caller must
present a subject token the gateway can exchange. Both cases share one contract: a connect that carries no
usable subject answers 401 with the RFC 9728 challenge, and the protected-resource metadata advertises the
issuers and scopes the caller signs in for.

Guardrails implement :class:`CallerSignInProvider`; :func:`caller_sign_in_for` merges the OBO server's own
requirement with every registered provider's so the challenge and the metadata always agree. The MCP package
never imports a concrete provider.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Protocol, cast, runtime_checkable

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

import litellm
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.types.mcp import MCPAuth

if TYPE_CHECKING:
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.types.mcp_server.mcp_server_manager import MCPServer


@dataclass(frozen=True, slots=True)
class CallerSignIn:
    """The issuers a caller signs in with and the scopes it requests before calling a server."""

    issuers: tuple[str, ...]
    scopes: tuple[str, ...]


@runtime_checkable
class CallerSignInProvider(Protocol):
    def caller_sign_in(self, server: MCPServer, user_api_key_auth: UserAPIKeyAuth | None) -> CallerSignIn | None:
        """The sign-in this provider requires of callers hitting ``server``; ``None`` when it does not gate
        the server for this caller (``user_api_key_auth=None`` is the anonymous metadata fetch that follows
        a challenge)."""
        ...


class _JwtIssuerEntry(BaseModel):
    model_config = ConfigDict(extra="ignore")

    issuer: str | None = None


class _JwtAuthConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    issuers: list[_JwtIssuerEntry] = []  # mutable-ok: pydantic copies the default per instance


_JWT_AUTH_ADAPTER: Final = TypeAdapter(_JwtAuthConfig)


def _jwt_auth_issuer_entries(jwtauth: object) -> tuple[_JwtIssuerEntry, ...]:
    try:
        return tuple(_JWT_AUTH_ADAPTER.validate_python(jwtauth, from_attributes=True).issuers)
    except ValidationError:
        return ()


def _providers() -> tuple[CallerSignInProvider, ...]:
    return tuple(
        callback
        for callback in litellm.logging_callback_manager.get_custom_loggers_for_type(CustomGuardrail)
        if isinstance(callback, CallerSignInProvider)
    )


def jwt_auth_issuers() -> tuple[str, ...]:
    """The OAuth issuer identifier(s) LiteLLM's JWT auth trusts, for the OBO PRM authorization_servers.

    In token_exchange the IdP that issues the subject JWT is the same one LiteLLM validates it
    against, so OBO discovery points clients at the JWT-auth issuer to obtain a subject token.
    Sourced from ``JWT_ISSUER`` and any configured ``litellm_jwtauth.issuers``.
    """
    import os  # noqa: PLC0415

    from litellm.proxy.proxy_server import (  # noqa: PLC0415  # lazy: proxy_server pulls the whole proxy graph
        general_settings,  # pyright: ignore[reportUnknownVariableType]  # proxy_server.general_settings is a raw untyped dict
    )

    env_issuer: Final = os.getenv("JWT_ISSUER")
    env: Final[tuple[str, ...]] = (env_issuer,) if env_issuer else ()

    settings: Final[Mapping[str, object]] = cast(Mapping[str, object], general_settings)
    configured: Final = tuple(
        entry.issuer for entry in _jwt_auth_issuer_entries(settings.get("litellm_jwtauth")) if entry.issuer
    )
    return tuple(dict.fromkeys((*env, *configured)))


def caller_sign_in_for(server: MCPServer, user_api_key_auth: UserAPIKeyAuth | None) -> CallerSignIn | None:
    """The merged sign-in requirement for ``server``: the OBO server's own issuer/scopes plus every
    registered provider's contribution. ``None`` when nothing requires sign-in, which is also the gate the
    connect-time challenge branches on."""
    contributions: Final = [
        contribution
        for contribution in (
            *(
                (CallerSignIn(issuers=jwt_auth_issuers(), scopes=tuple(server.scopes or ())),)
                if server.auth_type == MCPAuth.oauth2_token_exchange
                else ()
            ),
            *(provider.caller_sign_in(server, user_api_key_auth) for provider in _providers()),
        )
        if contribution is not None
    ]
    if not contributions:
        return None
    issuers: Final = tuple(dict.fromkeys(issuer for contribution in contributions for issuer in contribution.issuers))
    scopes: Final = tuple(dict.fromkeys(scope for contribution in contributions for scope in contribution.scopes))
    return CallerSignIn(issuers=issuers, scopes=scopes)
