"""Caller-identity primitives.

A ``Principal`` answers "who is making this request" using only the fields
that uniquely identify the caller. Per-row enrichment (budgets, team rows,
object permissions) is intentionally not modeled here; that data continues
to ride on ``UserAPIKeyAuth``.

Each subtype is a frozen dataclass with a ``kind`` discriminator suitable
for ``match``-style dispatch.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Union


@dataclass(frozen=True)
class ApiKeyPrincipal:
    kind: Literal["api_key"] = field(default="api_key", init=False)
    token_hash: str
    key_alias: Optional[str] = None
    user_id: Optional[str] = None
    team_id: Optional[str] = None
    org_id: Optional[str] = None
    project_id: Optional[str] = None
    agent_id: Optional[str] = None


@dataclass(frozen=True)
class JWTPrincipal:
    kind: Literal["jwt"] = field(default="jwt", init=False)
    sub: Optional[str] = None
    iss: Optional[str] = None
    aud: Optional[Union[str, List[str]]] = None
    scopes: List[str] = field(default_factory=list)
    claims: Dict[str, Any] = field(default_factory=dict)
    mapped_user_id: Optional[str] = None
    mapped_team_id: Optional[str] = None
    mapped_org_id: Optional[str] = None


@dataclass(frozen=True)
class SSOPrincipal:
    kind: Literal["sso"] = field(default="sso", init=False)
    sso_user_id: str
    email: Optional[str] = None
    provider: Optional[str] = None


@dataclass(frozen=True)
class ServiceAccountPrincipal:
    kind: Literal["service_account"] = field(default="service_account", init=False)
    name: str


@dataclass(frozen=True)
class AnonymousPrincipal:
    kind: Literal["anonymous"] = field(default="anonymous", init=False)


Principal = Union[
    ApiKeyPrincipal,
    JWTPrincipal,
    SSOPrincipal,
    ServiceAccountPrincipal,
    AnonymousPrincipal,
]
