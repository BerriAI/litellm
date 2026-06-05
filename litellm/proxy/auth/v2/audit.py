from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Callable, List, Optional

logger = logging.getLogger("litellm.proxy.auth.v2.audit")


class Decision(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    # A route auth_v2 does not yet govern, allowed (loudly) during build-out.
    LOUD_OPEN = "loud_open"


@dataclass(frozen=True)
class AuthzDecision:
    """One authorization decision, the unit of the compliance audit trail."""

    decision: Decision
    subject: str
    domain: str
    obj: str
    action: str
    route: str
    reason: str
    auth_method: Optional[str] = None


AuditSink = Callable[[AuthzDecision], None]

_sinks: List[AuditSink] = []


def register_sink(sink: AuditSink) -> None:
    """Register a sink (DB / SIEM writer) that receives every authz decision."""
    _sinks.append(sink)


def reset_sinks() -> None:
    _sinks.clear()


def record(decision: AuthzDecision) -> None:
    """Emit an authorization decision to the audit log and registered sinks.

    Every governed allow/deny and every loud-open passes through here, so the
    trail is complete and centralized. Sinks are isolated: a failing sink is
    logged but never affects the request's outcome.
    """
    logger.info(
        "auth_v2 authz %s: subject=%s action=%s obj=%s route=%s reason=%s method=%s",
        decision.decision.value,
        decision.subject,
        decision.action,
        decision.obj,
        decision.route,
        decision.reason,
        decision.auth_method or "-",
    )
    for sink in _sinks:
        try:
            sink(decision)
        except Exception:
            logger.exception("auth_v2 audit sink raised; decision still enforced")
