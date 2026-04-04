"""
Track MCP server usage for operational visibility: insert into
SpendLogMCPServerIndex when spend logs are written, so "last N requests
for MCP server X" and "which users/keys called which tools" queries are fast.
"""

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from litellm._logging import verbose_proxy_logger
from litellm.proxy.utils import PrismaClient


def _parse_mcp_info_from_payload(
    payload: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """
    Extract MCP server info from spend log payload metadata.
    Returns dict with mcp_server_name, tool_name if present.
    """
    meta = payload.get("metadata")
    if not meta:
        return None
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except (json.JSONDecodeError, TypeError):
            return None
    if not isinstance(meta, dict):
        return None
    mcp_tool_call = meta.get("mcp_tool_call_metadata")
    if not mcp_tool_call or not isinstance(mcp_tool_call, dict):
        return None

    mcp_server_name = mcp_tool_call.get("mcp_server_name")
    if not mcp_server_name:
        namespaced = payload.get("mcp_namespaced_tool_name") or mcp_tool_call.get(
            "namespaced_tool_name"
        )
        if namespaced and isinstance(namespaced, str) and "/" in namespaced:
            mcp_server_name = namespaced.split("/", 1)[0]
    if not mcp_server_name:
        return None

    tool_name = mcp_tool_call.get("name")
    return {
        "mcp_server_name": mcp_server_name,
        "tool_name": tool_name,
    }


async def process_spend_logs_mcp_server_usage(
    prisma_client: PrismaClient,
    logs_to_process: List[Dict[str, Any]],
) -> None:
    """
    After spend logs are written: insert SpendLogMCPServerIndex rows
    from mcp_tool_call_metadata in each payload.
    """
    if not logs_to_process:
        return

    index_rows: List[Dict[str, Any]] = []

    for payload in logs_to_process:
        request_id = payload.get("request_id")
        start_time = payload.get("startTime")
        if not request_id or not start_time:
            continue
        if isinstance(start_time, str):
            try:
                start_time = datetime.fromisoformat(
                    start_time.replace("Z", "+00:00")
                )
            except (ValueError, TypeError):
                continue
        if start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=timezone.utc)

        mcp_info = _parse_mcp_info_from_payload(payload)
        if not mcp_info:
            continue

        meta = payload.get("metadata")
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except (json.JSONDecodeError, TypeError):
                meta = {}
        if not isinstance(meta, dict):
            meta = {}

        api_key_hash = payload.get("api_key") or meta.get("user_api_key_hash")
        user_id = payload.get("user") or meta.get("user_api_key_user_id")
        team_id = payload.get("team_id") or meta.get("user_api_key_team_id")

        index_rows.append(
            {
                "request_id": request_id,
                "mcp_server_name": mcp_info["mcp_server_name"],
                "tool_name": mcp_info.get("tool_name"),
                "api_key_hash": str(api_key_hash) if api_key_hash else None,
                "user_id": str(user_id) if user_id else None,
                "team_id": str(team_id) if team_id else None,
                "start_time": start_time,
            }
        )

    if not index_rows:
        return

    try:
        await prisma_client.db.litellm_spendlogmcpserverindex.create_many(
            data=index_rows,
            skip_duplicates=True,
        )
    except Exception as e:
        verbose_proxy_logger.warning(
            "MCP server usage tracking (SpendLogMCPServerIndex) failed (non-fatal): %s",
            e,
        )

    for row in index_rows:
        try:
            from litellm.proxy.db.mcp_alert_rules import check_and_fire_mcp_alerts

            await check_and_fire_mcp_alerts(
                prisma_client=prisma_client,
                mcp_server_name=row["mcp_server_name"],
                tool_name=row.get("tool_name"),
                request_id=row["request_id"],
                user_id=row.get("user_id"),
                api_key_hash=row.get("api_key_hash"),
                team_id=row.get("team_id"),
            )
        except Exception as alert_err:
            verbose_proxy_logger.debug(
                "MCP alert check failed (non-fatal): %s", alert_err
            )
