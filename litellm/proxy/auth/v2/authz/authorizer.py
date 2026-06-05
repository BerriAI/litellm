import logging
from typing import Any, Dict, Optional

from ..audit import AuthzDecision, Decision, record
from ..principal import Principal
from ..protocols import SupportsEnforce
from .route_map import GovernedRoute, match_route

logger = logging.getLogger("litellm.proxy.auth.v2")


class AuthorizationDenied(Exception):
    """Raised when a governed route is denied by policy. Translated to a 403 at the edge."""

    def __init__(self, subject: str, obj: str, action: str):
        self.subject = subject
        self.obj = obj
        self.action = action
        super().__init__(
            f"auth_v2: {subject} is not permitted to '{action}' on '{obj}'"
        )


def _build_object(rule: GovernedRoute, request_data: Optional[Dict[str, Any]]) -> str:
    data = request_data or {}
    for key in rule.id_fields:
        value = data.get(key)
        if value:
            return f"{rule.resource}:{value}"
    return f"{rule.resource}:*"


def authorize(
    principal: Principal,
    route: str,
    request_data: Optional[Dict[str, Any]],
    enforcer: SupportsEnforce,
    method: Optional[str] = None,
    auth_method: Optional[str] = None,
) -> None:
    """Enforce policy for ``route``. No-op (loud) for routes v2 doesn't yet govern.

    Raises :class:`AuthorizationDenied` when a governed route is denied. Every
    outcome (allow, deny, loud-open) is recorded to the audit trail.
    ``enforcer`` is anything exposing ``enforce(subject, domain, obj, action)``.
    ``method`` is required to resolve REST resources (e.g. credentials) whose verb
    is the HTTP method; without it those routes would be treated as loud-open.
    """
    rule = match_route(route, method)
    if rule is None:
        logger.warning(
            "auth_v2: route '%s' is not yet protected by auth_v2; allowing. "
            "This must not reach production with auth_v2 enabled.",
            route,
        )
        record(
            AuthzDecision(
                decision=Decision.LOUD_OPEN,
                subject=principal.subject,
                domain=principal.domain,
                obj=f"route:{route}",
                action="*",
                route=route,
                reason="route not yet governed by auth_v2",
                auth_method=auth_method,
            )
        )
        return

    obj = _build_object(rule, request_data)
    allowed = enforcer.enforce(principal.subject, principal.domain, obj, rule.action)
    record(
        AuthzDecision(
            decision=Decision.ALLOW if allowed else Decision.DENY,
            subject=principal.subject,
            domain=principal.domain,
            obj=obj,
            action=rule.action,
            route=route,
            reason="control-plane policy",
            auth_method=auth_method,
        )
    )
    if not allowed:
        raise AuthorizationDenied(
            subject=principal.subject, obj=obj, action=rule.action
        )
