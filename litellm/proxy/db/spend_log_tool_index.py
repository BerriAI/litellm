"""
Track tool usage for the dashboard: insert into SpendLogToolIndex when spend logs
are written, so "last N requests for tool X" and "how is this tool called in production"
queries are fast.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Set

from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.safe_json_loads import safe_json_loads
from litellm.proxy.utils import PrismaClient


def _add_tool_calls_to_set(tool_calls: Any, out: Set[str]) -> None:
    """Extract tool names from OpenAI-style tool_calls list into out."""
    if not isinstance(tool_calls, list):
        return
    for tc in tool_calls:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function")
        if isinstance(fn, dict):
            name = fn.get("name")
            if name and isinstance(name, str) and name.strip():
                out.add(name.strip())


def _parse_tool_names_from_payload(payload: Dict[str, Any]) -> Set[str]:
    """
    Extract deduplicated tool names from a spend log payload.
    Sources: mcp_namespaced_tool_name, response (tool_calls), proxy_server_request (tools).
    """
    tool_names: Set[str] = set()

    # Top-level MCP tool name (single tool per request for that flow)
    mcp_name = payload.get("mcp_namespaced_tool_name")
    if mcp_name and isinstance(mcp_name, str) and mcp_name.strip():
        tool_names.add(mcp_name.strip())

    # Response: OpenAI-style tool_calls[].function.name or choices[0].message.tool_calls
    response_raw = payload.get("response")
    if response_raw:
        response_obj = (
            safe_json_loads(response_raw, default=None)
            if isinstance(response_raw, str)
            else response_raw
        )
        if isinstance(response_obj, dict):
            _add_tool_calls_to_set(response_obj.get("tool_calls"), tool_names)
            choices = response_obj.get("choices")
            if isinstance(choices, list) and choices:
                msg = choices[0].get("message") if isinstance(choices[0], dict) else None
                if isinstance(msg, dict):
                    _add_tool_calls_to_set(msg.get("tool_calls"), tool_names)

    # Request body: tools[].function.name
    request_raw = payload.get("proxy_server_request")
    if request_raw:
        request_obj = (
            safe_json_loads(request_raw, default=None)
            if isinstance(request_raw, str)
            else request_raw
        )
        if isinstance(request_obj, dict):
            body = request_obj.get("body", request_obj)
            if isinstance(body, dict):
                request_obj = body
        if isinstance(request_obj, dict):
            tools = request_obj.get("tools")
            if isinstance(tools, list):
                for t in tools:
                    if isinstance(t, dict):
                        fn = t.get("function")
                        if isinstance(fn, dict):
                            name = fn.get("name")
                            if name and isinstance(name, str) and name.strip():
                                tool_names.add(name.strip())

    return tool_names


async def process_spend_logs_tool_usage(
    prisma_client: PrismaClient,
    logs_to_process: List[Dict[str, Any]],
) -> None:
    """
    After spend logs are written: insert SpendLogToolIndex rows from each payload.
    Extracts tool names from mcp_namespaced_tool_name, response tool_calls, and
    proxy_server_request tools.
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

        tool_names = _parse_tool_names_from_payload(payload)
        for tool_name in tool_names:
            index_rows.append({
                "request_id": request_id,
                "tool_name": tool_name,
                "start_time": start_time,
            })

    if not index_rows:
        return

    try:
        index_data = []
        for r in index_rows:
            st = r["start_time"]
            if isinstance(st, str):
                try:
                    st = datetime.fromisoformat(st.replace("Z", "+00:00"))
                except (ValueError, TypeError):
                    continue
            if st.tzinfo is None:
                st = st.replace(tzinfo=timezone.utc)
            index_data.append({
                "request_id": r["request_id"],
                "tool_name": r["tool_name"],
                "start_time": st,
            })
        if index_data:
            await prisma_client.db.litellm_spendlogtoolindex.create_many(
                data=index_data,
                skip_duplicates=True,
            )
    except Exception as e:
        verbose_proxy_logger.warning(
            "Tool usage tracking (SpendLogToolIndex) failed (non-fatal): %s", e
        )
