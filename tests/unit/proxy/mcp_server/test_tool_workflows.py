import json
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Final, cast
from unittest.mock import AsyncMock, patch

import pytest
from jsonschema import Draft202012Validator
from mcp.types import CallToolResult, ListToolsResult, TextContent, Tool

import litellm
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.mcp_server import operations
from litellm.proxy.mcp_server.tool_search import (
    get_mcp_proxy_tool_definitions,
    handle_mcp_proxy_tool,
    mcp_proxy_tool_id,
    with_mcp_proxy_identity,
)
from litellm.proxy.mcp_server.tool_workflows import check_tool_workflow, find_workflow_sources
from litellm.types.mcp import MCPTransport
from litellm.types.mcp_server.mcp_server_manager import MCPServer

FETCH: Final = Tool.model_validate(
    {
        "name": "fetch",
        "inputSchema": {"type": "object", "properties": {"repository": {"type": "string"}}, "required": ["repository"]},
        "outputSchema": {
            "type": "object",
            "properties": {"issue_id": {"type": "string"}},
            "required": ["issue_id"],
        },
    }
)
CREATE: Final = Tool.model_validate(
    {
        "name": "create",
        "inputSchema": {
            "type": "object",
            "properties": {"issue_id": {"type": "string"}, "title": {"type": "string"}},
            "required": ["issue_id", "title"],
        },
    }
)


def _result_text(result: CallToolResult) -> str:
    content: Final = result.content[0]
    assert isinstance(content, TextContent)
    return content.text


def test_workflow_check_reports_declared_binding_and_remaining_inputs() -> None:
    result: Final = check_tool_workflow(FETCH, CREATE)

    assert result.binding_fields == ("issue_id",)
    assert result.first_step_inputs == ("repository",)
    assert result.second_step_inputs == ("title",)


def test_workflow_source_discovery_orders_verified_candidates() -> None:
    no_output: Final = FETCH.model_copy(update={"name": "no-output", "output_schema": None})
    extra_input: Final = FETCH.model_copy(
        update={
            "name": "extra-input",
            "input_schema": {
                "type": "object",
                "properties": {"repository": {"type": "string"}, "token": {"type": "string"}},
                "required": ["repository", "token"],
            },
        }
    )
    sources: Final = find_workflow_sources((no_output, extra_input, FETCH, CREATE), CREATE)

    assert tuple(source.name for source, _ in sources) == ("fetch", "extra-input")
    assert all(check.binding_fields == ("issue_id",) for _, check in sources)


@pytest.mark.parametrize(
    "output_schema,input_field",
    [
        ({"type": "object", "properties": {"issue_id": {"type": "string"}}}, {"type": "string"}),
        (
            {"type": "object", "properties": {"issue_id": {"type": "integer"}}, "required": ["issue_id"]},
            {"type": "string"},
        ),
        (
            {"type": "object", "properties": {"issue_id": {"type": "string"}}, "required": ["issue_id"]},
            {"type": "string", "pattern": "^ISSUE-"},
        ),
        (
            {
                "type": "object",
                "properties": {"issue_id": {"type": "string", "nullable": True}},
                "required": ["issue_id"],
            },
            {"type": "string"},
        ),
        (
            {"type": "object", "properties": {"issue_id": {"type": "string"}}, "required": ["issue_id", 1]},
            {"type": "string"},
        ),
    ],
)
def test_workflow_check_does_not_claim_an_unproven_binding(
    output_schema: Mapping[str, object], input_field: Mapping[str, object]
) -> None:
    first: Final = FETCH.model_copy(update={"output_schema": output_schema})
    second: Final = CREATE.model_copy(
        update={
            "input_schema": {
                "type": "object",
                "properties": {"issue_id": input_field},
                "required": ["issue_id"],
            }
        }
    )

    result: Final = check_tool_workflow(first, second)

    assert result.binding_fields == ()
    assert result.second_step_inputs == ("issue_id",)


def test_workflow_tool_is_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "mcp_tool_search", {})
    assert "check_workflow" not in {tool["name"] for tool in get_mcp_proxy_tool_definitions()}
    assert "find_workflow_sources" not in {tool["name"] for tool in get_mcp_proxy_tool_definitions()}

    monkeypatch.setattr(litellm, "mcp_tool_search", {"enable_workflow_discovery": True})
    definitions: Final = get_mcp_proxy_tool_definitions()
    for tool in definitions:
        Draft202012Validator.check_schema(tool["inputSchema"])
    assert {"check_workflow", "find_workflow_sources"}.issubset({tool["name"] for tool in definitions})


@pytest.mark.asyncio
async def test_workflow_check_rejects_disabled_feature_and_invalid_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auth: Final = UserAPIKeyAuth(api_key="test-key")
    monkeypatch.setattr(litellm, "mcp_tool_search", {})
    disabled: Final = await handle_mcp_proxy_tool(
        name="check_workflow", arguments={"tool_ids": ["first", "second"]}, user_api_key_dict=auth
    )

    monkeypatch.setattr(litellm, "mcp_tool_search", {"enable_workflow_discovery": True})
    malformed: Final = await handle_mcp_proxy_tool(
        name="check_workflow", arguments={"tool_ids": ["first"]}, user_api_key_dict=auth
    )
    bad_page: Final = await handle_mcp_proxy_tool(
        name="find_workflow_sources", arguments={"target_tool_id": "id", "limit": True}, user_api_key_dict=auth
    )

    assert disabled.is_error is True
    assert malformed.is_error is True
    assert _result_text(malformed) == "tool_ids must contain exactly two string IDs"
    assert bad_page.is_error is True


@pytest.mark.asyncio
async def test_workflow_check_only_uses_the_callers_accessible_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "mcp_tool_search", {"enable_workflow_discovery": True})
    manager: Final = operations.global_mcp_server_manager
    server: Final = MCPServer(server_id="workflow", name="workflow", transport=MCPTransport.http, url="http://workflow")
    denied_server: Final = MCPServer(
        server_id="another-server", name="another-server", transport=MCPTransport.http, url="http://denied"
    )
    auth: Final = UserAPIKeyAuth(api_key="test-key")
    fetch_id: Final = mcp_proxy_tool_id(
        with_mcp_proxy_identity(FETCH.model_copy(update={"name": "workflow-fetch"}), server.server_id)
    )
    create_id: Final = mcp_proxy_tool_id(
        with_mcp_proxy_identity(CREATE.model_copy(update={"name": "workflow-create"}), server.server_id)
    )
    denied_id: Final = mcp_proxy_tool_id(
        with_mcp_proxy_identity(CREATE.model_copy(update={"name": "another-server-create"}), denied_server.server_id)
    )
    client: Final = SimpleNamespace(discovery_auth_fingerprint=AsyncMock(return_value="credential-hash"))

    with (
        patch.dict(manager.registry, {server.server_id: server, denied_server.server_id: denied_server}),
        patch.dict(manager.tool_name_to_mcp_server_name_mapping),
        patch.object(operations, "_get_allowed_mcp_servers", AsyncMock(return_value=[server])),
        patch.object(manager, "create_mcp_client", AsyncMock(return_value=client)),
        patch.object(
            manager, "_fetch_tools_with_timeout", AsyncMock(return_value=ListToolsResult(tools=[FETCH, CREATE]))
        ),
    ):
        try:
            allowed: Final = await handle_mcp_proxy_tool(
                name="check_workflow", arguments={"tool_ids": [fetch_id, create_id]}, user_api_key_dict=auth
            )
            denied: Final = await handle_mcp_proxy_tool(
                name="check_workflow", arguments={"tool_ids": [fetch_id, denied_id]}, user_api_key_dict=auth
            )
            discovery: Final = await handle_mcp_proxy_tool(
                name="find_workflow_sources",
                arguments={"target_tool_id": create_id, "offset": 0, "limit": 1},
                user_api_key_dict=auth,
            )
            denied_target: Final = await handle_mcp_proxy_tool(
                name="find_workflow_sources", arguments={"target_tool_id": denied_id}, user_api_key_dict=auth
            )
        finally:
            manager.invalidate_server_definition_caches(server.server_id)

    assert allowed.is_error is False
    assert json.loads(_result_text(allowed))["binding_fields"] == ["issue_id"]
    assert denied.is_error is True
    assert _result_text(denied) == "Unknown or unauthorized tool_id"
    assert discovery.is_error is False
    discovered: Final = cast(dict[str, object], json.loads(_result_text(discovery)))
    assert discovered["total_candidates"] == 1
    sources: Final = cast(list[dict[str, object]], discovered["sources"])
    assert sources[0]["source_tool_id"] == fetch_id
    assert denied_target.is_error is True
