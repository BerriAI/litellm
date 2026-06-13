from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple

import casbin

from litellm.proxy.auth_v2.authorization.base import Authorizer
from litellm.proxy.auth_v2.authorization.roles import Role

if TYPE_CHECKING:
    from litellm.proxy.auth_v2.models import Principal

_MODEL_TEXT = """
[request_definition]
r = sub, obj, act

[policy_definition]
p = sub, obj, act

[role_definition]
g = _, _

[policy_effect]
e = some(where (p.eft == allow))

[matchers]
m = g(r.sub, p.sub) && keyMatch(r.obj, p.obj) && regexMatch(r.act, "^(" + p.act + ")$")
"""

_DEFAULT_GROUPING: List[Tuple[str, str]] = [
    (Role.PLATFORM_ADMIN.value, Role.PLATFORM_VIEWER.value),
    (Role.PLATFORM_ADMIN.value, Role.ORG_ADMIN.value),
    (Role.PLATFORM_ADMIN.value, Role.TEAM_ADMIN.value),
    (Role.ORG_ADMIN.value, Role.ORG_VIEWER.value),
    (Role.TEAM_ADMIN.value, Role.TEAM_MEMBER.value),
]

_DEFAULT_POLICY: List[Tuple[str, str, str]] = [
    (Role.PLATFORM_ADMIN.value, "/*", ".*"),
    (Role.PLATFORM_ADMIN.value, "/scim/v2/*", ".*"),
    (Role.PLATFORM_VIEWER.value, "/*", "GET"),
]


class RBACEngine(Authorizer):
    def __init__(self, policy_path: Optional[str] = None) -> None:
        model = casbin.Model()
        model.load_model_from_text(_MODEL_TEXT)
        if policy_path:
            self._enforcer = casbin.Enforcer(model, casbin.FileAdapter(policy_path))
            return
        self._enforcer = casbin.Enforcer(model)
        for sub, inherits in _DEFAULT_GROUPING:
            self._enforcer.add_grouping_policy(sub, inherits)
        for rule in _DEFAULT_POLICY:
            self._enforcer.add_policy(*rule)

    def enforce(self, principal: "Principal", obj: str, act: str) -> bool:
        return any(
            self._enforcer.enforce(role.value, obj, act) for role in principal.roles
        )

    def has_any_role(self, principal: "Principal", allowed: Tuple[Role, ...]) -> bool:
        allowed_values = {role.value for role in allowed}
        for role in principal.roles:
            if role.value in allowed_values:
                return True
            implicit = set(self._enforcer.get_implicit_roles_for_user(role.value))
            if allowed_values & implicit:
                return True
        return False
