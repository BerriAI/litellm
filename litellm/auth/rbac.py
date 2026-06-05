"""Role-based access control for the LiteLLM proxy.

Built-in roles (:class:`~litellm.proxy._types.LitellmUserRoles`) keep their
hard-coded route handling. RBAC adds *custom* roles whose route permissions are
declared in the proxy config and enforced centrally in
``RouteChecks.non_proxy_admin_allowed_routes_check``.

The model follows casbin's allow-list RBAC: roles grant permissions, a role may
inherit other roles (transitively), and anything not granted is denied. A
permission is the right to call a route, expressed either as a
:class:`~litellm.proxy._types.LiteLLMRoutes` group name (e.g. ``llm_api_routes``,
``info_routes``), an exact/prefix path, a wildcard pattern, or ``"*"`` for all
routes.
"""

import json
import threading
from typing import Dict, List, Optional, Set, Union

from pydantic import BaseModel, Field

from litellm.proxy._types import LiteLLMRoutes


class RBACRole(BaseModel):
    name: str
    description: Optional[str] = None
    allowed_routes: List[str] = Field(default_factory=list)
    inherits: List[str] = Field(default_factory=list)


class RBACPolicy(BaseModel):
    roles: List[RBACRole] = Field(default_factory=list)


class RBACEngine:
    def __init__(self, policy: RBACPolicy):
        self._roles: Dict[str, RBACRole] = {role.name: role for role in policy.roles}
        self._effective_routes: Dict[str, List[str]] = {
            name: self._resolve_routes(name) for name in self._roles
        }

    def _resolve_routes(self, name: str) -> List[str]:
        resolved: List[str] = []
        visited: Set[str] = set()
        stack = [name]
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            role = self._roles.get(current)
            if role is None:
                continue
            resolved.extend(role.allowed_routes)
            stack.extend(role.inherits)
        return resolved

    def is_governed_role(self, role_name: Optional[str]) -> bool:
        return bool(role_name) and role_name in self._roles

    def is_route_allowed(self, role_name: str, route: str) -> bool:
        from litellm.auth.route_checks import RouteChecks

        for entry in self._effective_routes.get(role_name, []):
            if entry == "*":
                return True
            if entry in LiteLLMRoutes._member_names_:
                if RouteChecks.check_route_access(
                    route=route,
                    allowed_routes=LiteLLMRoutes._member_map_[entry].value,
                ):
                    return True
            elif RouteChecks._route_matches_allowed_route(
                route=route, allowed_route=entry
            ):
                return True
            elif RouteChecks._route_matches_wildcard_pattern(
                route=route, pattern=entry
            ):
                return True
        return False

    @property
    def role_names(self) -> List[str]:
        return list(self._roles)

    @classmethod
    def from_config(cls, config: Union["RBACPolicy", dict]) -> "RBACEngine":
        if isinstance(config, RBACPolicy):
            return cls(config)
        return cls(RBACPolicy(**config))


_engine_lock = threading.Lock()
_cached_key: Optional[str] = None
_cached_engine: Optional[RBACEngine] = None


def get_rbac_engine(
    rbac_config: Optional[Union[RBACPolicy, dict]],
) -> Optional[RBACEngine]:
    """Build (and cache) an engine for the given config.

    The engine is rebuilt only when the config changes, so the per-request hot
    path is a dict lookup plus a string compare.
    """
    global _cached_key, _cached_engine
    if not rbac_config:
        return None
    if isinstance(rbac_config, RBACPolicy):
        key = rbac_config.model_dump_json()
    else:
        key = json.dumps(rbac_config, sort_keys=True, default=str)
    if key == _cached_key:
        return _cached_engine
    with _engine_lock:
        if key != _cached_key:
            _cached_engine = RBACEngine.from_config(rbac_config)
            _cached_key = key
    return _cached_engine


def get_configured_rbac_engine() -> Optional[RBACEngine]:
    from litellm.proxy.proxy_server import general_settings

    if not general_settings:
        return None
    return get_rbac_engine(general_settings.get("rbac"))
