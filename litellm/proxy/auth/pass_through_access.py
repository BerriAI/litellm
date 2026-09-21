"""Decides whether a caller may reach an ``auth: true`` pass-through route.

Build a ``PassThroughGrants`` for the caller and ask ``authorize_pass_through``. Every gate asks here, so a new grant
source is added in one place.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, Protocol, TypeAlias

from litellm.proxy._types import LiteLLM_JWTAuth, LiteLLM_TeamTable, UserAPIKeyAuth

_ALLOWLIST_KEY: Final = "allowed_passthrough_routes"


class AuthEnforcedLookup(Protocol):
    def __call__(self, *, route: str, method: str | None) -> bool: ...


def covered_by_prefix(route: str, prefix: str) -> bool:
    """Exact match, or ``route`` sits under ``prefix``: ``/a`` covers ``/a/b`` and not ``/ab``."""
    return route == prefix or route.startswith(prefix + "/")


def covered_by_pattern(route: str, pattern: str) -> bool:
    """Exact match, or a prefix match when ``pattern`` ends in ``*``: ``/a/*`` covers ``/a/b`` and not ``/a``."""
    if pattern.endswith("*"):
        return route.startswith(pattern[:-1])
    return route == pattern


def _metadata_allowlist(
    metadata: Mapping[str, object] | None,
    team_metadata: Mapping[str, object] | None = None,
) -> tuple[str, ...]:
    """``allowed_passthrough_routes`` from metadata. A non-empty key allowlist shadows the team's."""
    raw: Final = (metadata and metadata.get(_ALLOWLIST_KEY)) or (team_metadata and team_metadata.get(_ALLOWLIST_KEY))
    if not isinstance(raw, (list, tuple)):
        return ()
    return tuple(entry for entry in raw if isinstance(entry, str))


def _jwt_team_routes(jwt_auth: LiteLLM_JWTAuth | None) -> tuple[str, ...]:
    """``team_allowed_routes`` entries that name a path. Blanket entries (``*``, ``/*``) never grant."""
    if jwt_auth is None:
        return ()
    return tuple(entry for entry in jwt_auth.team_allowed_routes if entry.rstrip("*").strip("/"))


def _is_jwt_team_caller(token: UserAPIKeyAuth) -> bool:
    """
    A keyless JWT token with a team. ``team_allowed_routes`` is a team grant, and JWT-mapped virtual keys carry
    ``jwt_claims`` too but stay scoped to their key.
    """
    return token.jwt_claims is not None and token.token is None and token.team_id is not None


@dataclass(frozen=True, slots=True)
class PassThroughGrants:
    """``prefix_routes`` cover a path and everything under it; ``pattern_routes`` are exact or end in ``*``."""

    prefix_routes: tuple[str, ...] = ()
    pattern_routes: tuple[str, ...] = ()

    @staticmethod
    def for_token(token: UserAPIKeyAuth, jwt_auth: LiteLLM_JWTAuth | None = None) -> "PassThroughGrants":
        return PassThroughGrants(
            prefix_routes=_metadata_allowlist(token.metadata, token.team_metadata),
            pattern_routes=_jwt_team_routes(jwt_auth) if _is_jwt_team_caller(token) else (),
        )

    @staticmethod
    def for_jwt_team(team: LiteLLM_TeamTable | None, jwt_auth: LiteLLM_JWTAuth | None) -> "PassThroughGrants":
        return PassThroughGrants(
            prefix_routes=_metadata_allowlist(team.metadata if team is not None else None),
            pattern_routes=_jwt_team_routes(jwt_auth),
        )

    def covers(self, route: str) -> bool:
        return any(covered_by_prefix(route=route, prefix=grant) for grant in self.prefix_routes) or any(
            covered_by_pattern(route=route, pattern=grant) for grant in self.pattern_routes
        )


@dataclass(frozen=True, slots=True)
class NotAuthEnforced:
    pass


@dataclass(frozen=True, slots=True)
class Granted:
    pass


@dataclass(frozen=True, slots=True)
class Denied:
    pass


PassThroughAccess: TypeAlias = NotAuthEnforced | Granted | Denied


def authorize_pass_through(
    *,
    route: str,
    method: str | None,
    grants: PassThroughGrants,
    is_auth_enforced: AuthEnforcedLookup,
) -> PassThroughAccess:
    if not is_auth_enforced(route=route, method=method):
        return NotAuthEnforced()
    return Granted() if grants.covers(route) else Denied()
