"""The per-request identity bundle.

``IdentityContext`` is what downstream consumers (auth, spend, guardrails,
logging, audit) should read identity from. Today it travels alongside the
legacy ``UserAPIKeyAuth`` via the adapter functions in
``litellm.identity.adapter``.

The bundle is mutable on purpose: identity fields like ``end_user_id`` are
sometimes resolved or overridden after initial extraction, and the
existing ``UserAPIKeyAuth`` mutation patterns must keep working.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from litellm.identity.principal import AnonymousPrincipal, Principal


@dataclass
class RequestIds:
    request_id: Optional[str] = None
    trace_id: Optional[str] = None
    session_id: Optional[str] = None
    mcp_session_id: Optional[str] = None


@dataclass
class ClientInfo:
    ip: Optional[str] = None
    user_agent: Optional[str] = None
    forwarded_chain: List[str] = field(default_factory=list)


@dataclass
class AuditInfo:
    changed_by: Optional[str] = None


@dataclass
class IdentityContext:
    principal: Principal = field(default_factory=AnonymousPrincipal)
    end_user_id: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    access_group_ids: List[str] = field(default_factory=list)
    request: RequestIds = field(default_factory=RequestIds)
    client: ClientInfo = field(default_factory=ClientInfo)
    audit: AuditInfo = field(default_factory=AuditInfo)
