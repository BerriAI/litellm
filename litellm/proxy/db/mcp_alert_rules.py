"""
MCP alert rules: check if a tool call matches any configured alert rule
and fire webhooks when it does.
"""

import fnmatch
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import httpx

from litellm._logging import verbose_proxy_logger
from litellm.proxy.utils import PrismaClient


async def check_and_fire_mcp_alerts(
    prisma_client: PrismaClient,
    mcp_server_name: str,
    tool_name: Optional[str],
    request_id: str,
    user_id: Optional[str],
    api_key_hash: Optional[str],
    team_id: Optional[str],
) -> None:
    """
    Check if a tool call matches any alert rules and fire webhooks.
    Called after inserting into SpendLogMCPServerIndex.
    """
    if not tool_name:
        return

    try:
        where: Dict[str, Any] = {"enabled": True}
        rules = await prisma_client.db.litellm_mcpalertrule.find_many(
            where=where
        )

        for rule in rules:
            if rule.mcp_server_name and rule.mcp_server_name != mcp_server_name:
                continue
            if not fnmatch.fnmatch(tool_name.lower(), rule.tool_name_pattern.lower()):
                continue

            payload = {
                "alert_name": rule.alert_name,
                "alert_rule_id": rule.id,
                "mcp_server_name": mcp_server_name,
                "tool_name": tool_name,
                "request_id": request_id,
                "user_id": user_id,
                "api_key_hash": api_key_hash,
                "team_id": team_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "description": rule.description,
                "message": (
                    f"MCP Alert: Tool '{tool_name}' was called on server "
                    f"'{mcp_server_name}' (rule: {rule.alert_name})"
                ),
            }

            try:
                async with httpx.AsyncClient(timeout=10) as client:
                    await client.post(
                        rule.webhook_url,
                        json=payload,
                        headers={"Content-Type": "application/json"},
                    )
                verbose_proxy_logger.info(
                    "MCP alert fired: rule=%s, tool=%s, server=%s",
                    rule.alert_name,
                    tool_name,
                    mcp_server_name,
                )
            except Exception as webhook_err:
                verbose_proxy_logger.warning(
                    "MCP alert webhook failed for rule %s: %s",
                    rule.alert_name,
                    webhook_err,
                )
    except Exception as e:
        verbose_proxy_logger.warning(
            "MCP alert rule check failed (non-fatal): %s", e
        )
