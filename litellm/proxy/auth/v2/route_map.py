from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class GovernedRoute:
    resource: str
    action: str
    # Request-data keys searched, in order, for the concrete resource id. The
    # first present, non-empty value becomes the casbin object ``<resource>:<id>``.
    # When none are found the object falls back to ``<resource>:*``.
    id_fields: List[str] = field(default_factory=list)


# Governs the model and team management planes. Every other route is
# intentionally left ungoverned (loud-open) until later slices wire it in. Team
# membership/permission routes (member_add, etc.) are deferred with the recursive
# `manage` action.
_MODEL_ID_FIELDS = ["model_id", "id"]
_TEAM_ID_FIELDS = ["team_id", "id"]

_GOVERNED: Dict[str, GovernedRoute] = {
    "/model/new": GovernedRoute("model", "write"),
    "/model/update": GovernedRoute("model", "write", _MODEL_ID_FIELDS),
    "/model/delete": GovernedRoute("model", "delete", _MODEL_ID_FIELDS),
    "/model/info": GovernedRoute("model", "read", _MODEL_ID_FIELDS),
    "/team/new": GovernedRoute("team", "write"),
    "/team/update": GovernedRoute("team", "write", _TEAM_ID_FIELDS),
    "/team/delete": GovernedRoute("team", "delete", _TEAM_ID_FIELDS),
    "/team/info": GovernedRoute("team", "read", _TEAM_ID_FIELDS),
    # The policy-admin surface governs itself: only a role permitted to manage the
    # "policy" resource (the bootstrap proxy_admin role does) may edit policies.
    "/auth/v2/policy/permission/add": GovernedRoute("policy", "write"),
    "/auth/v2/policy/permission/remove": GovernedRoute("policy", "delete"),
    "/auth/v2/policy/assignment/add": GovernedRoute("policy", "write"),
    "/auth/v2/policy/assignment/remove": GovernedRoute("policy", "delete"),
    "/auth/v2/policy/list": GovernedRoute("policy", "read"),
}


# Inference routes carry a `model` in the body and are authorized on the data
# plane (casbin ABAC over the principal's allowed-model attribute), not the
# control-plane RBAC map.
_INFERENCE_ROUTES = {
    "/chat/completions",
    "/v1/chat/completions",
    "/completions",
    "/v1/completions",
    "/embeddings",
    "/v1/embeddings",
    "/responses",
    "/v1/responses",
}


def match_route(route: str) -> Optional[GovernedRoute]:
    """Return the governance rule for ``route``, or None if v2 doesn't yet own it."""
    normalized = route.rstrip("/") or "/"
    return _GOVERNED.get(normalized)


def is_inference_route(route: str) -> bool:
    return (route.rstrip("/") or "/") in _INFERENCE_ROUTES
