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


# Control-plane management resources governed by RBAC. Every route not listed
# here (and not an inference route) is loud-open until a later slice wires it in;
# non-uniform surfaces (guardrails, mcp servers, credentials) are deferred
# because their verbs don't map cleanly onto read/write/delete/manage.
_MODEL_ID_FIELDS = ["model_id", "id"]
_TEAM_ID_FIELDS = ["team_id", "id"]
_KEY_ID_FIELDS = ["key", "token", "key_name"]
_USER_ID_FIELDS = ["user_id"]
_ORG_ID_FIELDS = ["organization_id"]
_VECTOR_STORE_ID_FIELDS = ["vector_store_id", "id"]
_BUDGET_ID_FIELDS = ["budget_id", "id"]
_CUSTOMER_ID_FIELDS = ["user_id"]

_GOVERNED: Dict[str, GovernedRoute] = {
    "/model/new": GovernedRoute("model", "write"),
    "/model/update": GovernedRoute("model", "write", _MODEL_ID_FIELDS),
    "/model/delete": GovernedRoute("model", "delete", _MODEL_ID_FIELDS),
    "/model/info": GovernedRoute("model", "read", _MODEL_ID_FIELDS),
    "/team/new": GovernedRoute("team", "write"),
    "/team/update": GovernedRoute("team", "write", _TEAM_ID_FIELDS),
    "/team/delete": GovernedRoute("team", "delete", _TEAM_ID_FIELDS),
    "/team/info": GovernedRoute("team", "read", _TEAM_ID_FIELDS),
    # Membership changes are the `manage` action on the team resource.
    "/team/member_add": GovernedRoute("team", "manage", _TEAM_ID_FIELDS),
    "/team/member_update": GovernedRoute("team", "manage", _TEAM_ID_FIELDS),
    "/team/member_delete": GovernedRoute("team", "manage", _TEAM_ID_FIELDS),
    "/key/generate": GovernedRoute("key", "write"),
    "/key/update": GovernedRoute("key", "write", _KEY_ID_FIELDS),
    "/key/delete": GovernedRoute("key", "delete", _KEY_ID_FIELDS),
    "/key/info": GovernedRoute("key", "read", _KEY_ID_FIELDS),
    "/key/list": GovernedRoute("key", "read"),
    "/key/block": GovernedRoute("key", "write", _KEY_ID_FIELDS),
    "/key/unblock": GovernedRoute("key", "write", _KEY_ID_FIELDS),
    "/user/new": GovernedRoute("user", "write"),
    "/user/update": GovernedRoute("user", "write", _USER_ID_FIELDS),
    "/user/delete": GovernedRoute("user", "delete", _USER_ID_FIELDS),
    "/user/info": GovernedRoute("user", "read", _USER_ID_FIELDS),
    "/user/list": GovernedRoute("user", "read"),
    "/organization/new": GovernedRoute("organization", "write"),
    "/organization/update": GovernedRoute("organization", "write", _ORG_ID_FIELDS),
    "/organization/delete": GovernedRoute("organization", "delete", _ORG_ID_FIELDS),
    "/organization/info": GovernedRoute("organization", "read", _ORG_ID_FIELDS),
    "/organization/list": GovernedRoute("organization", "read"),
    "/organization/member_add": GovernedRoute("organization", "manage", _ORG_ID_FIELDS),
    # The policy-admin surface governs itself: only a role permitted to manage the
    # "policy" resource (the bootstrap proxy_admin role does) may edit policies.
    "/auth/v2/policy/permission/add": GovernedRoute("policy", "write"),
    "/auth/v2/policy/permission/remove": GovernedRoute("policy", "delete"),
    "/auth/v2/policy/assignment/add": GovernedRoute("policy", "write"),
    "/auth/v2/policy/assignment/remove": GovernedRoute("policy", "delete"),
    "/auth/v2/policy/list": GovernedRoute("policy", "read"),
    "/vector_store/new": GovernedRoute("vector_store", "write"),
    "/vector_store/update": GovernedRoute(
        "vector_store", "write", _VECTOR_STORE_ID_FIELDS
    ),
    "/vector_store/delete": GovernedRoute(
        "vector_store", "delete", _VECTOR_STORE_ID_FIELDS
    ),
    "/vector_store/info": GovernedRoute(
        "vector_store", "read", _VECTOR_STORE_ID_FIELDS
    ),
    "/vector_store/list": GovernedRoute("vector_store", "read"),
    "/budget/new": GovernedRoute("budget", "write"),
    "/budget/update": GovernedRoute("budget", "write", _BUDGET_ID_FIELDS),
    "/budget/delete": GovernedRoute("budget", "delete", _BUDGET_ID_FIELDS),
    "/budget/info": GovernedRoute("budget", "read", _BUDGET_ID_FIELDS),
    "/budget/list": GovernedRoute("budget", "read"),
    "/budget/settings": GovernedRoute("budget", "read"),
    "/customer/new": GovernedRoute("customer", "write"),
    "/customer/update": GovernedRoute("customer", "write", _CUSTOMER_ID_FIELDS),
    "/customer/delete": GovernedRoute("customer", "delete", _CUSTOMER_ID_FIELDS),
    "/customer/info": GovernedRoute("customer", "read", _CUSTOMER_ID_FIELDS),
    "/customer/list": GovernedRoute("customer", "read"),
    "/customer/block": GovernedRoute("customer", "write", _CUSTOMER_ID_FIELDS),
    "/customer/unblock": GovernedRoute("customer", "write", _CUSTOMER_ID_FIELDS),
    # MCP server and guardrail admin surfaces expose collection-level operations
    # (register/list/health/submissions), so objects stay at "<resource>:*"; the
    # runtime verbs (apply_guardrail, test_custom_code) are not management and are
    # intentionally left to the data/runtime path.
    "/v1/mcp/server/register": GovernedRoute("mcp_server", "write"),
    "/v1/mcp/server/health": GovernedRoute("mcp_server", "read"),
    "/v1/mcp/server/submissions": GovernedRoute("mcp_server", "read"),
    "/guardrails/register": GovernedRoute("guardrail", "write"),
    "/guardrails/list": GovernedRoute("guardrail", "read"),
    "/guardrails/submissions": GovernedRoute("guardrail", "read"),
}


# Inference routes carry a `model` in the body and are authorized by the same
# role system: the `call` action on the `model:<id>` object. They are matched
# here (by body, not path) rather than in the RPC route map above.
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


@dataclass(frozen=True)
class _RestRule:
    method: str
    path: str
    is_prefix: bool
    route: GovernedRoute


# REST-style resources encode the verb in the HTTP method (POST /credentials =
# create, GET /credentials = list) and the id in the path, so they need
# method-aware, prefix-capable matching rather than the verb-in-path lookup the
# RPC routes above use. Path-param ids are not extracted yet, so objects stay at
# "<resource>:*" (per-id credential policies are a follow-up).
_REST_RULES: List[_RestRule] = [
    _RestRule("POST", "/credentials", False, GovernedRoute("credential", "write")),
    _RestRule("GET", "/credentials", False, GovernedRoute("credential", "read")),
    _RestRule("GET", "/credentials/", True, GovernedRoute("credential", "read")),
    _RestRule("DELETE", "/credentials/", True, GovernedRoute("credential", "delete")),
]


def match_route(route: str, method: Optional[str] = None) -> Optional[GovernedRoute]:
    """Return the governance rule for ``route``, or None if v2 doesn't yet own it.

    ``method`` is required to resolve REST resources whose verb is the HTTP
    method; the verb-in-path RPC routes are method-agnostic and match without it.
    """
    normalized = route.rstrip("/") or "/"
    rule = _GOVERNED.get(normalized)
    if rule is not None:
        return rule
    if method is not None:
        verb = method.upper()
        for rest in _REST_RULES:
            if rest.method != verb:
                continue
            if rest.is_prefix:
                if route.startswith(rest.path):
                    return rest.route
            elif normalized == rest.path:
                return rest.route
    return None


def is_inference_route(route: str) -> bool:
    return (route.rstrip("/") or "/") in _INFERENCE_ROUTES
