"""
Event emission helpers (S6-06).

Each public function wraps ``emit_event`` with a fixed event_type + a typed
payload shape so call sites don't have to memorize event-string conventions.

Every function is best-effort: failures are swallowed so a broken webhook
subscriber CANNOT regress the request path.

Event schemas (versioned by ``schema_version`` field):

  capability.invoked  — every spend-log write
    { schema_version: "1", app_id, entity_type, entity_id, spend, request_id,
      user_id, team_id, ts }

  budget.exhausted  — emitted from spend tracking when over budget
    { schema_version: "1", scope, scope_id, budget, spend, ts }

  agent.healthcheck.failed  — emitted from /v1/agents?health_check=true sweep
    { schema_version: "1", agent_id, agent_name, error, ts }

  mcp.tool.called  — emitted post-MCP-tool execution
    { schema_version: "1", server_id, tool_name, namespaced_tool_name, app_id, ts }
"""

from datetime import datetime
from typing import Optional

from litellm._logging import verbose_proxy_logger
from litellm.proxy.webhook_endpoints.dispatcher import emit_event


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


async def emit_capability_invoked(
    *,
    app_id: Optional[str],
    entity_type: Optional[str],
    entity_id: Optional[str],
    spend: float,
    request_id: Optional[str] = None,
    user_id: Optional[str] = None,
    team_id: Optional[str] = None,
) -> None:
    if entity_type is None:
        return  # non-capability rows: no subscriber expects this
    try:
        await emit_event(
            "capability.invoked",
            {
                "schema_version": "1",
                "app_id": app_id,
                "entity_type": entity_type,
                "entity_id": entity_id,
                "spend": float(spend or 0.0),
                "request_id": request_id,
                "user_id": user_id,
                "team_id": team_id,
                "ts": _now_iso(),
            },
            app_id=app_id,
        )
    except Exception as e:  # pragma: no cover — best-effort fire-and-forget
        verbose_proxy_logger.debug("emit_capability_invoked failed: %s", e)


async def emit_budget_exhausted(
    *,
    scope: str,
    scope_id: str,
    budget: float,
    spend: float,
    app_id: Optional[str] = None,
) -> None:
    """scope ∈ {'key', 'user', 'team', 'org', 'app'}."""
    try:
        await emit_event(
            "budget.exhausted",
            {
                "schema_version": "1",
                "scope": scope,
                "scope_id": scope_id,
                "budget": float(budget),
                "spend": float(spend),
                "ts": _now_iso(),
            },
            app_id=app_id,
        )
    except Exception as e:
        verbose_proxy_logger.debug("emit_budget_exhausted failed: %s", e)


async def emit_agent_healthcheck_failed(
    *,
    agent_id: str,
    agent_name: Optional[str],
    error: Optional[str],
) -> None:
    try:
        await emit_event(
            "agent.healthcheck.failed",
            {
                "schema_version": "1",
                "agent_id": agent_id,
                "agent_name": agent_name,
                "error": error,
                "ts": _now_iso(),
            },
        )
    except Exception as e:
        verbose_proxy_logger.debug("emit_agent_healthcheck_failed failed: %s", e)


async def emit_mcp_tool_called(
    *,
    server_id: Optional[str],
    tool_name: Optional[str],
    namespaced_tool_name: Optional[str],
    app_id: Optional[str] = None,
) -> None:
    try:
        await emit_event(
            "mcp.tool.called",
            {
                "schema_version": "1",
                "server_id": server_id,
                "tool_name": tool_name,
                "namespaced_tool_name": namespaced_tool_name,
                "app_id": app_id,
                "ts": _now_iso(),
            },
            app_id=app_id,
        )
    except Exception as e:
        verbose_proxy_logger.debug("emit_mcp_tool_called failed: %s", e)
