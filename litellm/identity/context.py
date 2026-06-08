"""The per-request identity bundle consumed downstream of auth.

Mutable on purpose: fields like ``end_user_id`` are resolved or overridden
after initial extraction.
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
