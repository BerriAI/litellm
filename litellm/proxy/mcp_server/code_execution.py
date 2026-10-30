from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import datetime
from typing import Final
from uuid import uuid4

from mcp.shared.exceptions import MCPError
from mcp.types import CallToolResult, TextContent, Tool
from pydantic import ConfigDict, JsonValue, TypeAdapter

import litellm
from litellm.proxy.mcp_server.contracts import OperationContext
from litellm.types.llms.base import LiteLLMBaseModel
from litellm.types.mcp import MCPCodeModeSettings

CODE_TOOL_NAME: Final = "litellm_execute_code"
CODE_MODE_META: Final = "litellm.ai/code_mode"
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])


class CodeInput(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    code: str


def settings() -> MCPCodeModeSettings:
    configured: Final = litellm.mcp_code_mode
    return (
        MCPCodeModeSettings(enabled=configured)
        if isinstance(configured, bool)
        else MCPCodeModeSettings.model_validate(configured)
    )


def validate_settings(configured: object) -> MCPCodeModeSettings:
    validated: Final = (
        MCPCodeModeSettings(enabled=configured)
        if isinstance(configured, bool)
        else MCPCodeModeSettings.model_validate(configured)
    )
    if validated.enabled:
        from litellm.proxy.mcp_server.code_runtime import check_runtime

        check_runtime()
    return validated


def tool_names(tool: Tool) -> frozenset[str]:
    """Recognize canonical, bare and configured display names for one identity."""
    from litellm.proxy.mcp_server.mcp_server_manager import global_mcp_server_manager
    from litellm.proxy.mcp_server.tool_search import (
        _mcp_proxy_identity,  # pyright: ignore[reportPrivateUsage]  # shared identity metadata contract
    )
    from litellm.proxy.mcp_server.utils import strip_known_server_prefix

    try:
        identity: Final = _mcp_proxy_identity(tool)
    except TypeError:
        return frozenset((tool.name,))
    canonical: Final = identity["tool_name"]
    server: Final = next(
        (item for item in global_mcp_server_manager.get_registry().values() if item.server_id == identity["server_id"]),
        None,
    )
    if server is None:
        return frozenset((tool.name, canonical))
    bare: Final = strip_known_server_prefix(canonical, server)
    display: Final = (server.tool_name_to_display_name or {}).get(bare, canonical)
    return frozenset((tool.name, canonical, bare, display))


def compatibility(tool: Tool) -> Mapping[str, JsonValue]:
    from litellm.proxy.mcp_server.tool_search import MCP_PROXY_TOOL_NAMES, VIRTUAL_TOOL_NAMES

    names: Final = tool_names(tool)
    if names.intersection((*MCP_PROXY_TOOL_NAMES, *VIRTUAL_TOOL_NAMES, CODE_TOOL_NAME)):
        return {"status": "unsupported", "reason": "Nested orchestration is unsupported"}
    if tool.execution is not None and tool.execution.task_support == "required":
        return {"status": "unsupported", "reason": "Task-based execution is unsupported in code mode"}
    excluded: Final = next((reason for name, reason in settings().excluded_tools.items() if name in names), None)
    return (
        {"status": "unsupported", "reason": excluded}
        if excluded is not None
        else {
            "status": "available",
            "reason": "JSON tool calls only. Interactive continuations are unsupported; upstream behavior has not been verified.",
        }
    )


def code_tool() -> Tool:
    from litellm.proxy.mcp_server.code_runtime import CodeExecutionLimits

    limits: Final = CodeExecutionLimits()
    return Tool(
        name=CODE_TOOL_NAME,
        description=(
            "Run JavaScript to compose MCP tool calls. Use await tools.call_tool({tool_id, arguments}); "
            "tool_id is an ID from search_tools or an exact tool name from the catalog. "
            "Each call returns the full MCP envelope: structuredContent, content and isError. "
            "Inspect isError before using results. Emit only needed results with text(value). "
            "No filesystem, network, imports, persistent state or interactive continuations. "
            "A failed script may have completed some calls; never blindly replay writes. "
            f"Limits: {limits.timeout_seconds:g} seconds, {limits.max_calls} calls, "
            f"{limits.max_concurrent_calls} concurrent calls, {limits.output_bytes // 1024} KiB output."
        ),
        input_schema=CodeInput.model_json_schema(),
    )


def with_code_tool(
    tools: Sequence[Tool], *, first_page: bool = True
) -> list[Tool]:  # mutable-ok: MCP SDK requires a list
    if not settings().enabled:
        return list(tools)
    if any(tool.name == CODE_TOOL_NAME for tool in tools):
        raise MCPError(code=-32602, message="MCP tool name conflicts with the reserved code execution tool")
    annotated: Final = [
        tool.model_copy(update={"meta": {**(tool.meta or {}), CODE_MODE_META: compatibility(tool)}}) for tool in tools
    ]
    return [*annotated, *([code_tool()] if first_page else [])]


def error_result(message: str) -> CallToolResult:
    return CallToolResult(content=[TextContent(type="text", text=message)], is_error=True)


def serialize_tool_result(
    result: CallToolResult,
) -> dict[str, JsonValue]:  # mutable-ok: runtime JSON callback contract
    return _JSON_OBJECT.validate_python(result.model_dump(mode="json", by_alias=True, exclude_none=True))


def with_code_trace(context: OperationContext) -> OperationContext:
    from litellm.proxy.litellm_pre_call_utils import get_chain_id_from_headers

    headers: Final = dict(context.raw_headers or {})
    if get_chain_id_from_headers(headers):
        return context
    return replace(context, raw_headers={**headers, "x-litellm-trace-id": str(uuid4())})


async def run_code_tool(arguments: Mapping[str, object] | None, context: OperationContext) -> CallToolResult:
    from litellm.proxy.mcp_server.operations import (
        _fire_mcp_tool_call_logging,
        fire_mcp_tool_call_failure_logging,
        prepare_virtual_mcp_call,
    )

    if not settings().enabled:
        return error_result("MCP code mode is disabled")
    caller: Final = context.user_api_key_auth
    if caller is None:
        return error_result("MCP code execution requires an authenticated caller")
    traced: Final = with_code_trace(context)
    started: Final = datetime.now()  # noqa: DTZ005  # MCP logging uses naive timestamps
    data, logging_obj = await prepare_virtual_mcp_call(
        CODE_TOOL_NAME,
        dict(arguments or {}),
        caller,
        raw_headers=traced.raw_headers,
        client_ip=traced.client_ip,
        guardrails_only=True,
    )
    try:
        processed: Final = TypeAdapter(dict[str, object]).validate_python(
            data.get("modified_arguments", data.get("mcp_arguments", data.get("arguments", {})))
        )
        result: Final = await _execute_script(processed, traced)
        if logging_obj is None:
            return result
        return await _fire_mcp_tool_call_logging(
            logging_obj,
            result,
            started,
            datetime.now(),  # noqa: DTZ005  # MCP logging uses naive timestamps
            caller,
            data,
        )
    except Exception as exc:
        await fire_mcp_tool_call_failure_logging(logging_obj, exc, started, caller, data)
        raise


async def _execute_script(arguments: Mapping[str, object], context: OperationContext) -> CallToolResult:
    from litellm.proxy.mcp_server.operations import _dispatch_virtual_mcp_tool, _list_mcp_tools, find_mcp_proxy_tool
    from litellm.proxy.mcp_server.tool_search import (
        MCP_PROXY_TOOL_NAMES,
        VIRTUAL_TOOL_NAMES,
        mcp_proxy_tool_id,
    )

    if not settings().enabled:
        return error_result("MCP code mode is disabled")
    caller: Final = context.user_api_key_auth
    if caller is None:
        return error_result("MCP code execution requires an authenticated caller")
    from litellm.proxy.mcp_server.code_runtime import execute_code

    request: Final = CodeInput.model_validate(arguments or {})
    auth, header, servers, server_headers, oauth_headers, raw_headers, client_ip = context.legacy_auth()
    forbidden: Final = {*MCP_PROXY_TOOL_NAMES, *VIRTUAL_TOOL_NAMES, CODE_TOOL_NAME}

    async def find_tool(tool_id: str) -> Tool | None:
        if re.fullmatch(r"[0-9a-f]{32}", tool_id):
            return await find_mcp_proxy_tool(
                tool_id,
                user_api_key_auth=caller,
                mcp_servers=servers,
                client_ip=client_ip,
                mcp_auth_header=header,
                mcp_server_auth_headers=server_headers,
                oauth2_headers=oauth_headers,
                raw_headers=raw_headers,
            )
        listing: Final = await _list_mcp_tools(
            user_api_key_auth=auth,
            mcp_servers=servers,
            client_ip=client_ip,
            mcp_auth_header=header,
            mcp_server_auth_headers=server_headers,
            oauth2_headers=oauth_headers,
            raw_headers=raw_headers,
            mcp_proxy_mode=True,
        )
        matches: Final = tuple(tool for tool in listing.tools if tool_id in tool_names(tool))
        return matches[0] if len(matches) == 1 else None

    async def call_tool(
        tool_id: str,
        tool_arguments: dict[str, JsonValue],  # mutable-ok: runtime JSON callback contract
    ) -> dict[str, JsonValue]:  # mutable-ok: runtime JSON callback contract
        if tool_id in forbidden:
            return serialize_tool_result(error_result("Nested orchestration tools are unsupported"))
        selected: Final = await find_tool(tool_id)
        if selected is None:
            return serialize_tool_result(error_result("Unknown, ambiguous or unauthorized tool"))
        support: Final = compatibility(selected)
        if support["status"] == "unsupported":
            return serialize_tool_result(error_result(f"Tool is unsupported in code mode: {support['reason']}"))
        result: Final = await _dispatch_virtual_mcp_tool(
            name="call_tool",
            arguments={"tool_id": mcp_proxy_tool_id(selected), "arguments": tool_arguments},
            user_api_key_auth=auth,
            client_ip=client_ip,
            mcp_servers=servers,
            mcp_auth_header=header,
            mcp_server_auth_headers=server_headers,
            oauth2_headers=oauth_headers,
            raw_headers=raw_headers,
            mcp_proxy_mode=True,
            incoming_bearer_token=context.incoming_bearer_token,
        )
        assert result is not None
        from litellm.proxy.mcp_server.result_conversion import INPUT_REQUIRED_UNSUPPORTED_MESSAGE

        if any(
            isinstance(block, TextContent) and block.text == INPUT_REQUIRED_UNSUPPORTED_MESSAGE
            for block in result.content
        ):
            return serialize_tool_result(
                error_result(
                    "This tool requires an interactive continuation, which is unsupported in code mode. "
                    "Call it directly instead. Earlier calls in this script may already have completed."
                )
            )
        return serialize_tool_result(result)

    result: Final = await execute_code(request.code, call_tool)
    return CallToolResult(
        content=[
            TextContent(
                type="text",
                text=json.dumps(
                    {"output": result.output, "error": result.error, "calls_started": result.calls_started}
                ),
            )
        ],
        is_error=result.error is not None,
    )
