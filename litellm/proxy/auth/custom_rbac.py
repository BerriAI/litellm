"""
Custom RBAC roles.

Proxy admins define named roles whose permissions are an allow-list of routes, either in
``general_settings.custom_rbac_roles`` or through the ``/custom_role`` endpoints. Users
assigned such a role are governed entirely by it: any route the role does not grant is
denied, so the built-in role permissions never widen a custom role.
"""

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Protocol

from fastapi import HTTPException
from pydantic import TypeAdapter, ValidationError

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import LiteLLMRoutes, LitellmUserRoles
from litellm.types.custom_rbac import CustomRBACRole, CustomRBACRoleResponse

CUSTOM_RBAC_ROLES_CONFIG_KEY: Final = "custom_rbac_roles"
ALL_ROUTES_WILDCARD: Final = "*"
_ROUTE_GROUP_NAMES: Final = frozenset(LiteLLMRoutes.__members__)
_BUILTIN_ROLE_NAMES: Final = frozenset(role.value for role in LitellmUserRoles)
_ENGINE_CACHE_TTL_SECONDS: Final = 30.0
_CONFIG_ROLES_ADAPTER: Final = TypeAdapter(tuple[CustomRBACRole, ...])
_ORDER_BY_ROLE_NAME: Final[Mapping[str, object]] = MappingProxyType({"role_name": "asc"})


class _RoleRecord(Protocol):
    def dict(self) -> Mapping[str, object]: ...


class _CustomRoleTable(Protocol):
    async def find_many(self, order: Mapping[str, object]) -> Sequence[_RoleRecord]: ...


@dataclass(frozen=True, slots=True)
class CustomRBACEngine:
    """Resolved route permissions per custom role, with inheritance already flattened."""

    effective_routes: Mapping[str, frozenset[str]]

    def is_governed_role(self, role_name: str | None) -> bool:
        return role_name is not None and role_name in self.effective_routes

    def is_route_allowed(self, role_name: str, route: str) -> bool:
        return any(
            _permission_grants_route(permission=permission, route=route)
            for permission in self.effective_routes.get(role_name, frozenset())
        )


def _permission_grants_route(permission: str, route: str) -> bool:
    from litellm.proxy.auth.route_checks import RouteChecks

    if permission == ALL_ROUTES_WILDCARD:
        return True
    if permission in _ROUTE_GROUP_NAMES:
        return RouteChecks.check_route_access(route=route, allowed_routes=LiteLLMRoutes[permission].value)
    return RouteChecks.check_route_access(route=route, allowed_routes=(permission,))


def _resolve_routes(
    role_name: str,
    roles_by_name: Mapping[str, CustomRBACRole],
    visited: frozenset[str],
) -> frozenset[str]:
    role: Final = roles_by_name.get(role_name)
    if role is None or role_name in visited:
        return frozenset()
    return frozenset(role.allowed_routes).union(
        *(
            _resolve_routes(role_name=parent, roles_by_name=roles_by_name, visited=visited | frozenset((role_name,)))
            for parent in role.inherits
        ),
        frozenset(),
    )


def build_custom_rbac_engine(roles: Sequence[CustomRBACRole]) -> CustomRBACEngine:
    roles_by_name: Final = MappingProxyType({role.role_name: role for role in roles})
    return CustomRBACEngine(
        effective_routes=MappingProxyType(
            {
                role_name: _resolve_routes(role_name=role_name, roles_by_name=roles_by_name, visited=frozenset())
                for role_name in roles_by_name
            }
        )
    )


def is_reserved_role_name(role_name: str) -> bool:
    return role_name in _BUILTIN_ROLE_NAMES


def validate_role_permissions(allowed_routes: Sequence[str]) -> tuple[str, ...]:
    """The entries that are neither ``*``, a route group name, nor a route path."""
    return tuple(
        permission
        for permission in allowed_routes
        if permission != ALL_ROUTES_WILDCARD and permission not in _ROUTE_GROUP_NAMES and not permission.startswith("/")
    )


def get_config_custom_rbac_roles() -> tuple[CustomRBACRole, ...]:
    from litellm.proxy.proxy_server import general_settings

    configured: Final = general_settings.get(CUSTOM_RBAC_ROLES_CONFIG_KEY)
    if not configured:
        return ()
    try:
        return _CONFIG_ROLES_ADAPTER.validate_python(configured)
    except ValidationError as exc:
        verbose_proxy_logger.error("Invalid general_settings.%s: %s", CUSTOM_RBAC_ROLES_CONFIG_KEY, exc)
        return ()


async def get_db_custom_rbac_roles(table: _CustomRoleTable) -> tuple[CustomRBACRoleResponse, ...]:
    records: Final = await table.find_many(order=_ORDER_BY_ROLE_NAME)
    return tuple(CustomRBACRoleResponse.model_validate(record.dict()) for record in records)


class _EngineCache:
    def __init__(self, ttl_seconds: float) -> None:
        self._ttl_seconds = ttl_seconds
        self._engine: CustomRBACEngine | None = None
        self._expires_at: float = 0.0

    def get_fresh(self) -> CustomRBACEngine | None:
        if self._engine is None or time.monotonic() >= self._expires_at:
            return None
        return self._engine

    def get_stale(self) -> CustomRBACEngine | None:
        return self._engine

    def set(self, engine: CustomRBACEngine) -> None:
        self._engine = engine
        self._expires_at = time.monotonic() + self._ttl_seconds

    def clear(self) -> None:
        self._engine = None
        self._expires_at = 0.0


_ENGINE_CACHE: Final = _EngineCache(ttl_seconds=_ENGINE_CACHE_TTL_SECONDS)


def invalidate_custom_rbac_engine_cache() -> None:
    _ENGINE_CACHE.clear()


def _custom_role_table() -> _CustomRoleTable | None:
    from litellm.proxy.proxy_server import prisma_client
    from litellm.repositories.table_repositories import CustomRBACRoleRepository

    if prisma_client is None:
        return None
    return CustomRBACRoleRepository(prisma_client=prisma_client).table


async def validate_assigned_user_role(user_role: LitellmUserRoles | str | None) -> None:
    """Reject a user_role that is neither a built-in role nor a currently defined custom role."""
    if user_role is None or isinstance(user_role, LitellmUserRoles):
        return

    engine: Final = await get_active_custom_rbac_engine()
    if engine is not None and engine.is_governed_role(user_role):
        return

    raise HTTPException(
        status_code=400,
        detail=f"user_role={user_role} is not a built-in role and no custom RBAC role with that name is defined",
    )


async def get_active_custom_rbac_engine() -> CustomRBACEngine | None:
    """The engine for the currently configured roles, or None when no custom role exists.

    A DB read failure reuses the last known policy so a transient outage cannot silently
    downgrade a governed role to the built-in role permissions.
    """
    cached: Final = _ENGINE_CACHE.get_fresh()
    if cached is not None:
        return cached

    table: Final = _custom_role_table()
    try:
        db_roles: Final = () if table is None else await get_db_custom_rbac_roles(table=table)
    except Exception as exc:  # noqa: BLE001  # any DB failure must keep the last known policy, not drop it
        verbose_proxy_logger.exception("Failed to load custom RBAC roles from the DB: %s", exc)
        return _ENGINE_CACHE.get_stale()

    roles: Final = get_config_custom_rbac_roles() + tuple(
        CustomRBACRole(
            role_name=role.role_name,
            description=role.description,
            allowed_routes=role.allowed_routes,
            inherits=role.inherits,
        )
        for role in db_roles
    )
    if not roles:
        return None

    engine: Final = build_custom_rbac_engine(roles=roles)
    _ENGINE_CACHE.set(engine)
    return engine
