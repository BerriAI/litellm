from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import casbin
import yaml
from pydantic import BaseModel

from litellm.proxy.auth_v2.authorization.base import Authorizer
from litellm.proxy.auth_v2.authorization.roles import Role

if TYPE_CHECKING:
    from litellm.proxy.auth_v2.models import Principal

_MODEL_TEXT = """
[request_definition]
r = sub, obj, act

[policy_definition]
p = sub_rule, obj_rule, act

[policy_effect]
e = some(where (p.eft == allow))

[matchers]
m = eval(p_sub_rule) && eval(p_obj_rule) && regexMatch(r_act, "^(" + p_act + ")$")
"""


class ProtectedResource(BaseModel):
    """The resource a principal is acting on, exposed to policy as ``r_obj``."""

    endpoint: Optional[str] = None
    method: Optional[str] = None
    model: Optional[str] = None
    mcp_server: Optional[str] = None
    mcp_tool: Optional[str] = None


class _SafeClaims(dict):
    """Claims map that yields None for absent keys.

    Casbin evaluates every policy row in a single matcher, so a row referencing
    ``r_sub.claims['x']`` would raise KeyError for any principal lacking that
    claim and abort the whole decision. Returning None keeps such a row simply
    non-matching instead of poisoning unrelated rows.
    """

    def __missing__(self, key: str) -> None:
        return None


class _Subject:
    """Principal attributes exposed to policy as ``r_sub``."""

    def __init__(
        self,
        roles: List[str],
        teams: List[str],
        org: Optional[str],
        scopes: List[str],
        claims: Dict[str, Any],
        user_id: Optional[str],
        email: Optional[str],
    ) -> None:
        self.roles = roles
        self.teams = teams
        self.org = org
        self.scopes = scopes
        self.claims = _SafeClaims(claims)
        self.user_id = user_id
        self.email = email


def _subject_view(principal: "Principal") -> _Subject:
    roles = [role.value for role in principal.roles]
    claim_roles = principal.claims.get("roles", [])
    if isinstance(claim_roles, list):
        roles += [r for r in claim_roles if isinstance(r, str) and r not in roles]
    return _Subject(
        roles=roles,
        teams=[team.name for team in principal.teams if team.name],
        org=principal.organization.id if principal.organization else None,
        scopes=list(principal.scopes),
        claims=principal.claims,
        user_id=principal.user.id if principal.user else None,
        email=principal.user.email if principal.user else None,
    )


def _load_policies(policy_path: str) -> List[Tuple[str, str, str]]:
    with open(policy_path, "r") as handle:
        document = yaml.safe_load(handle) or {}
    entries = document.get("policies") or []
    rules: List[Tuple[str, str, str]] = []
    for index, entry in enumerate(entries):
        try:
            rules.append((entry["sub_rule"], entry["obj_rule"], entry["act"]))
        except (TypeError, KeyError) as exc:
            raise ValueError(
                f"abac policy entry {index} must define sub_rule, obj_rule and act"
            ) from exc
    return rules


class ABACEngine(Authorizer):
    """Attribute-based authorizer: subject x resource policies via Casbin.

    Policies are operator-supplied YAML (``{policies: [{sub_rule, obj_rule,
    act}]}``) loaded into an in-memory enforcer. The CSV FileAdapter is avoided
    on purpose: it retains the quotes around comma-bearing expressions, turning
    an eval'd condition into a truthy string literal and silently allowing.
    """

    def __init__(self, policy_path: Optional[str] = None) -> None:
        model = casbin.Model()
        model.load_model_from_text(_MODEL_TEXT)
        self._enforcer = casbin.Enforcer(model)
        if policy_path:
            for rule in _load_policies(policy_path):
                self._enforcer.add_policy(*rule)

    def decide(self, principal: "Principal", resource: ProtectedResource) -> bool:
        try:
            return self._enforcer.enforce(
                _subject_view(principal), resource, resource.method or ""
            )
        except Exception:
            return False

    def enforce(self, principal: "Principal", obj: str, act: str) -> bool:
        return self.decide(principal, ProtectedResource(endpoint=obj, method=act))

    def has_any_role(self, principal: "Principal", allowed: Tuple[Role, ...]) -> bool:
        allowed_values = {role.value for role in allowed}
        return any(role.value in allowed_values for role in principal.roles)
