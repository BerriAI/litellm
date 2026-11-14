"""Role and scope policy for the control plane. Roles grant scopes, scopes gate routes, the dataplane binding
on the token decides where the scopes apply (``*`` or no binding is fleet-wide)."""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Final, Literal, TypeAlias


class Scope(str, Enum):
    DATAPLANES_READ = "dataplanes:read"
    DATAPLANES_WRITE = "dataplanes:write"
    USAGE_READ = "usage:read"
    FLEET_READ = "fleet:read"
    PROXY_READ = "proxy:read"
    PROXY_WRITE = "proxy:write"
    AUDIT_READ = "audit:read"
    ROLES_WRITE = "roles:write"


ALL_SCOPES: Final = frozenset(Scope)
READ_SCOPES: Final = frozenset({Scope.DATAPLANES_READ, Scope.USAGE_READ, Scope.FLEET_READ, Scope.PROXY_READ})

ADMIN_ROLE: Final = "admin"
EDITOR_ROLE: Final = "editor"
VIEWER_ROLE: Final = "viewer"
USAGE_VIEWER_ROLE: Final = "usage_viewer"

DEFAULT_ROLES: Final[Mapping[str, frozenset[Scope]]] = MappingProxyType(
    {
        ADMIN_ROLE: ALL_SCOPES,
        EDITOR_ROLE: READ_SCOPES | {Scope.PROXY_WRITE},
        VIEWER_ROLE: READ_SCOPES,
        USAGE_VIEWER_ROLE: frozenset({Scope.USAGE_READ}),
    }
)

FLEET_WIDE_BINDING: Final = "*"

TokenKind: TypeAlias = Literal["session", "oidc", "master_key"]


@dataclass(frozen=True, slots=True)
class RoleDefinition:
    scopes: frozenset[Scope]
    inherits: frozenset[str]


@dataclass(frozen=True, slots=True)
class UnknownScope:
    role: str
    scope: str


@dataclass(frozen=True, slots=True)
class UnknownParent:
    role: str
    parent: str


RolePolicyError: TypeAlias = UnknownScope | UnknownParent


class RolePolicy:
    """Resolved role -> scopes table. Built once from defaults plus operator-defined roles."""

    def __init__(self, roles: Mapping[str, frozenset[Scope]]) -> None:
        self._roles: Final = MappingProxyType(dict(roles))

    @property
    def names(self) -> frozenset[str]:
        return frozenset(self._roles)

    def scopes_for(self, role: str) -> frozenset[Scope]:
        return self._roles.get(role, frozenset())

    def known(self, raw: tuple[str, ...]) -> frozenset[str]:
        return frozenset(name for name in raw if name in self._roles)

    def grant(self, roles: frozenset[str]) -> frozenset[Scope]:
        return frozenset[Scope]().union(*(self.scopes_for(role) for role in roles))


def default_policy() -> RolePolicy:
    return RolePolicy(DEFAULT_ROLES)


def _resolve(name: str, definitions: Mapping[str, RoleDefinition], seen: frozenset[str]) -> frozenset[Scope]:
    definition: Final = definitions[name]
    parents: Final = definition.inherits - seen
    return definition.scopes.union(*(_resolve(parent, definitions, seen | {name}) for parent in parents))


def build_policy(custom: Mapping[str, RoleDefinition], include_defaults: bool = True) -> RolePolicy | RolePolicyError:
    """Operator roles override same-named defaults. Inheritance may reference defaults or other custom roles."""
    base: Final[Mapping[str, RoleDefinition]] = (
        {name: RoleDefinition(scopes=scopes, inherits=frozenset()) for name, scopes in DEFAULT_ROLES.items()}
        if include_defaults
        else {}
    )
    definitions: Final = {**base, **custom}
    unknown_parents: Final = tuple(
        UnknownParent(role=name, parent=parent)
        for name, definition in custom.items()
        for parent in sorted(definition.inherits - frozenset(definitions))
    )  # comprehension-ok: flattening (role, parent) pairs
    if unknown_parents:
        return unknown_parents[0]
    return RolePolicy({name: _resolve(name, definitions, frozenset()) for name in definitions})


def parse_role_definition(
    role: str, scopes: tuple[str, ...], inherits: tuple[str, ...]
) -> RoleDefinition | UnknownScope:
    known: Final = frozenset(scope.value for scope in Scope)
    unknown: Final = tuple(scope for scope in scopes if scope not in known)
    if unknown:
        return UnknownScope(role=role, scope=unknown[0])
    return RoleDefinition(scopes=frozenset(Scope(scope) for scope in scopes), inherits=frozenset(inherits))


@dataclass(frozen=True, slots=True)
class DataplaneBinding:
    """Which dataplanes a principal may touch. ``all`` is fleet-wide, ``ids`` is an explicit set."""

    all: bool
    ids: frozenset[str]

    def allows(self, dataplane_id: str) -> bool:
        return self.all or dataplane_id in self.ids

    def restrict(self, requested: frozenset[str] | None) -> frozenset[str] | None:
        """Intersect a caller's requested dataplane filter with the binding. ``None`` means every bound dataplane."""
        if requested is None:
            return None if self.all else self.ids
        return requested if self.all else requested & self.ids


@dataclass(frozen=True, slots=True)
class Principal:
    subject: str
    email: str | None
    roles: frozenset[str]
    scopes: frozenset[Scope]
    dataplanes: DataplaneBinding
    token_kind: TokenKind

    def has_scope(self, scope: Scope) -> bool:
        return scope in self.scopes

    def missing_scopes(self, required: frozenset[Scope]) -> frozenset[Scope]:
        return required - self.scopes


def clamp_scopes(granted: frozenset[Scope], claimed: frozenset[Scope] | None) -> frozenset[Scope]:
    """A token may narrow what its roles grant, never widen it. ``None`` is a token without a scope claim and
    keeps the full role grant. A claim that names only unknown scopes narrows the grant to nothing."""
    return granted if claimed is None else granted & claimed


def binding_for(dataplane_ids: frozenset[str]) -> DataplaneBinding:
    fleet_wide: Final = not dataplane_ids or FLEET_WIDE_BINDING in dataplane_ids
    return DataplaneBinding(all=fleet_wide, ids=frozenset() if fleet_wide else dataplane_ids)


def parse_scopes(raw: tuple[str, ...] | None) -> frozenset[Scope] | None:
    """Unknown scope names (``openid``, ``profile``) are dropped. Only an absent claim is ``None``."""
    if raw is None:
        return None
    known: Final = frozenset(scope.value for scope in Scope)
    return frozenset(Scope(value) for value in raw if value in known)


def build_principal(
    policy: RolePolicy,
    subject: str,
    email: str | None,
    roles: frozenset[str],
    claimed_scopes: frozenset[Scope] | None,
    dataplane_ids: frozenset[str],
    token_kind: TokenKind,
) -> Principal:
    return Principal(
        subject=subject,
        email=email,
        roles=roles,
        scopes=clamp_scopes(policy.grant(roles), claimed_scopes),
        dataplanes=binding_for(dataplane_ids),
        token_kind=token_kind,
    )
