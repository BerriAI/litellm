"""
Serialization helpers — Prisma row -> response Pydantic model.

Prisma returns ``datetime`` objects and Json columns as Python native types
already, so we just convert datetimes to ISO-8601 strings and pass JSON
columns through. Centralized here so every endpoint serializes the same
way (consistent timestamps in the SDK).
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from litellm.proxy.agent_session_endpoints.schemas import (
    AgentResponse,
    ConversationMessage,
    RunResponse,
    SessionResponse,
)


def _iso(value: Any) -> Optional[str]:
    """Format a Prisma datetime as ISO-8601 (UTC); pass through strings."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _as_dict(value: Any) -> Optional[Dict[str, Any]]:
    """Coerce Prisma JSON column into a dict (or ``None`` if missing)."""
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    return None


def _as_list_of_dict(value: Any) -> List[Dict[str, Any]]:
    if not value:
        return []
    if isinstance(value, list):
        return [v for v in value if isinstance(v, dict)]
    return []


def agent_row_to_response(row: Any) -> AgentResponse:
    return AgentResponse(
        id=row.id,
        name=row.name,
        model=row.model,
        system_prompt=row.system_prompt,
        default_repos=_as_list_of_dict(row.default_repos) or None,
        default_env_vars=_as_dict(row.default_env_vars),
        tools_config=_as_dict(row.tools_config),
        metadata=_as_dict(row.metadata),
        team_id=row.team_id,
        created_at=_iso(row.created_at),
        updated_at=_iso(row.updated_at),
    )


def session_row_to_response(
    row: Any,
    daemon_token: Optional[str] = None,
) -> SessionResponse:
    return SessionResponse(
        id=row.id,
        agent_id=row.agent_id,
        status=row.status,
        vm_id=row.vm_id,
        vm_provider=row.vm_provider,
        repos=_as_list_of_dict(row.repos),
        expires_at=_iso(row.expires_at),
        last_heartbeat_at=_iso(row.last_heartbeat_at),
        created_at=_iso(row.created_at),
        updated_at=_iso(row.updated_at),
        terminated_at=_iso(row.terminated_at),
        daemon_token=daemon_token,
    )


def run_row_to_response(row: Any) -> RunResponse:
    prompt = _as_dict(row.prompt) or {}
    return RunResponse(
        id=row.id,
        session_id=row.session_id,
        status=row.status,
        prompt=prompt,
        parent_run_id=row.parent_run_id,
        result=row.result,
        git_branches=_as_list_of_dict(row.git_branches) or None,
        created_at=_iso(row.created_at),
        updated_at=_iso(row.updated_at),
        started_at=_iso(row.started_at),
        terminated_at=_iso(row.terminated_at),
    )


def event_row_to_message(row: Any) -> ConversationMessage:
    # NOTE: the DB column is ``event_type``/``payload`` but the wire
    # shape we surface to SDK consumers is ``type``/``data`` (matches
    # Cursor's API + the SSE event frame in run_endpoints._sse_event_data).
    # Rename here so /conversation and the SSE stream agree.
    return ConversationMessage(
        run_id=row.run_id,
        seq=row.seq,
        type=row.event_type,
        data=_as_dict(row.payload) or {},
        created_at=_iso(row.created_at),
    )
