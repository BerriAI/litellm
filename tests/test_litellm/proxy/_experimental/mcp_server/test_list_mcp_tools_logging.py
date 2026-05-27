"""LIT-1774 regression: list_mcp_tools must populate mcp_tool_call_metadata on
the StandardLoggingPayload so external callbacks can attribute the event to
the MCP server(s) and tool counts.
"""
import asyncio
import os
import uuid
from datetime import datetime
from typing import List

import pytest


def _build_logging_obj_for_list_tools():
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    import litellm  # noqa: F401
    from litellm.utils import function_setup, Rules
    from litellm.types.utils import CallTypes

    start = datetime.now()
    data = {
        "model": "MCP: list_tools",
        "call_type": CallTypes.list_mcp_tools.value,
        "litellm_call_id": str(uuid.uuid4()),
        "litellm_trace_id": None,
        "metadata": {
            "spend_logs_metadata": {
                "mcp_operation": "list_tools",
                "requested_mcp_servers": ["zapier-mcp", "deepwiki-mcp"],
            },
        },
        "input": [{"role": "system", "content": {"mcp_operation": "list_tools"}}],
    }
    logging_obj, _ = function_setup(
        original_function="list_mcp_tools",
        rules_obj=Rules(),
        start_time=start,
        **data,
    )
    logging_obj.call_type = CallTypes.list_mcp_tools.value
    logging_obj.model = "MCP: list_tools"
    return logging_obj, start


def _run_success_handler(logging_obj, start):
    end = datetime.now()
    asyncio.run(
        logging_obj.async_success_handler(
            result=[{"name": "zapier-mcp/send_email"}],
            start_time=start,
            end_time=end,
        )
    )
    return logging_obj.model_call_details.get("standard_logging_object")


def test_list_mcp_tools_baseline_without_metadata_is_null():
    """Sanity: if model_call_details["mcp_tool_call_metadata"] is never set,
    StandardLoggingPayload.metadata.mcp_tool_call_metadata is None. This is the
    pre-fix shape — kept as a regression anchor.
    """
    logging_obj, start = _build_logging_obj_for_list_tools()
    slp = _run_success_handler(logging_obj, start)
    assert slp is not None
    metadata = slp.get("metadata") or {}
    assert metadata.get("mcp_tool_call_metadata") is None


def test_list_mcp_tools_propagates_mcp_tool_call_metadata_to_payload():
    """When _list_mcp_tools populates model_call_details["mcp_tool_call_metadata"]
    (the LIT-1774 fix), the resulting StandardLoggingPayload exposes it so
    external callbacks can read it directly.
    """
    from litellm.types.utils import CallTypes, StandardLoggingMCPToolCall

    logging_obj, start = _build_logging_obj_for_list_tools()
    requested = ["zapier-mcp", "deepwiki-mcp"]
    per_server_tool_counts = {"zapier-mcp": 3, "deepwiki-mcp": 1}
    standard_logging_mcp_tool_call = StandardLoggingMCPToolCall(
        name=CallTypes.list_mcp_tools.value,
        arguments={"requested_mcp_servers": requested},
        mcp_server_name=",".join(per_server_tool_counts.keys()),
        namespaced_tool_name=None,
        result={
            "allowed_server_count": 2,
            "tool_count_total": 4,
            "per_server_tool_counts": per_server_tool_counts,
        },
    )
    logging_obj.model_call_details["mcp_tool_call_metadata"] = (
        standard_logging_mcp_tool_call
    )

    slp = _run_success_handler(logging_obj, start)
    assert slp is not None
    metadata = slp.get("metadata") or {}
    mcp_md = metadata.get("mcp_tool_call_metadata")
    assert mcp_md is not None, "mcp_tool_call_metadata must be populated"
    assert mcp_md.get("name") == CallTypes.list_mcp_tools.value
    assert mcp_md.get("mcp_server_name") == "zapier-mcp,deepwiki-mcp"
    assert mcp_md.get("arguments") == {"requested_mcp_servers": requested}
    result = mcp_md.get("result") or {}
    assert result.get("allowed_server_count") == 2
    assert result.get("tool_count_total") == 4
    assert result.get("per_server_tool_counts") == per_server_tool_counts
    assert mcp_md.get("namespaced_tool_name") is None


def test_list_mcp_tools_with_no_servers_yields_null_server_name():
    """Edge case: when no servers resolved (empty list), mcp_server_name should
    be None instead of an empty string, so downstream filters do not match it.
    """
    from litellm.types.utils import CallTypes, StandardLoggingMCPToolCall

    logging_obj, start = _build_logging_obj_for_list_tools()
    standard_logging_mcp_tool_call = StandardLoggingMCPToolCall(
        name=CallTypes.list_mcp_tools.value,
        arguments={"requested_mcp_servers": None},
        mcp_server_name=None,
        namespaced_tool_name=None,
        result={
            "allowed_server_count": 0,
            "tool_count_total": 0,
            "per_server_tool_counts": {},
        },
    )
    logging_obj.model_call_details["mcp_tool_call_metadata"] = (
        standard_logging_mcp_tool_call
    )

    slp = _run_success_handler(logging_obj, start)
    metadata = slp.get("metadata") or {}
    mcp_md = metadata.get("mcp_tool_call_metadata")
    assert mcp_md is not None
    assert mcp_md.get("mcp_server_name") is None
    assert mcp_md.get("result", {}).get("allowed_server_count") == 0
    assert mcp_md.get("result", {}).get("tool_count_total") == 0
