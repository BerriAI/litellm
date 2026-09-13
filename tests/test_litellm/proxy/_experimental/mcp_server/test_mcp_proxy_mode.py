import json
from datetime import datetime

import pytest
from fastapi import HTTPException
from mcp.shared.exceptions import McpError
from pydantic import AnyUrl

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._experimental.mcp_server import server
from litellm.proxy._experimental.mcp_server.mcp_context import _mcp_proxy_mode
from litellm.proxy._types import LiteLLM_ObjectPermissionTable, UserAPIKeyAuth

AUTH = UserAPIKeyAuth(api_key="key")


@pytest.fixture
def proxy_mode():
    token = _mcp_proxy_mode.set(True)
    try:
        yield
    finally:
        _mcp_proxy_mode.reset(token)


@pytest.mark.asyncio
@pytest.mark.usefixtures("proxy_mode")
async def test_proxy_call_rejects_non_proxy_tool_names() -> None:
    result = await server._dispatch_virtual_mcp_tool(
        name="math_stdio-add", arguments={"a": 1, "b": 2}, user_api_key_auth=AUTH, client_ip=None
    )

    assert result is not None
    assert result.isError is True
    assert "unavailable on /mcp/proxy" in result.content[0].text


@pytest.mark.asyncio
@pytest.mark.usefixtures("proxy_mode")
async def test_proxy_rejects_non_tool_protocol_operations() -> None:
    options = server.server.create_initialization_options()
    assert options.capabilities.prompts is None
    assert options.capabilities.resources is None
    assert options.capabilities.tools is not None

    with pytest.raises(McpError):
        await server.list_prompts()
    with pytest.raises(McpError):
        await server.get_prompt("prompt", {})
    with pytest.raises(McpError):
        await server.list_resources()
    with pytest.raises(McpError):
        await server.list_resource_templates()
    with pytest.raises(McpError):
        await server.read_resource(AnyUrl("https://example.com/resource"))


class FailureRecorder(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.events: list[tuple[str, str]] = []

    async def async_log_failure_event(
        self, kwargs: dict[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        self.events.append(("failure", json.dumps(kwargs.get("standard_logging_object"), default=str)))

    async def async_log_success_event(
        self, kwargs: dict[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        self.events.append(("success", json.dumps(kwargs.get("standard_logging_object"), default=str)))

    async def async_post_call_failure_hook(
        self,
        request_data: dict[str, object],
        original_exception: Exception,
        user_api_key_dict: UserAPIKeyAuth,
        traceback_str: str | None = None,
    ) -> None:
        self.events.append(("post_failure", json.dumps(request_data, default=str)))


@pytest.mark.asyncio
@pytest.mark.usefixtures("proxy_mode")
async def test_proxy_scope_exception_emits_failure_log(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = FailureRecorder()
    monkeypatch.setattr(litellm, "callbacks", [recorder])
    auth = UserAPIKeyAuth(
        api_key="scope-denial-key-hash",
        object_permission=LiteLLM_ObjectPermissionTable(object_permission_id="denied", mcp_servers=["no-mcp-servers"]),
    )
    arguments = {"tool_id": "denied-scope", "arguments": {}}

    with pytest.raises(HTTPException) as denied:
        await server._dispatch_virtual_mcp_tool(
            name="call_tool",
            arguments=arguments,
            user_api_key_auth=auth,
            client_ip=None,
            mcp_servers=["ungranted"],
            raw_headers={"authorization": "Bearer raw-scope-secret", "x-litellm-call-id": "scope-denial"},
        )

    assert denied.value.status_code == 403
    assert denied.value.detail == {"error": "The key is not allowed to access the requested MCP servers: ungranted"}
    assert [kind for kind, _ in recorder.events] == ["failure", "post_failure"]
    payload = json.loads(recorder.events[0][1])
    assert payload["id"] == "scope-denial"
    assert payload["call_type"] == "call_mcp_tool"
    assert payload["status"] == "failure"
    assert payload["response_cost"] == 0
    assert "ungranted" in payload["error_str"]
    hook_payload = json.loads(recorder.events[1][1])
    assert hook_payload["standard_logging_object"] == payload
    assert hook_payload["arguments"] == arguments
    assert "raw_headers" not in hook_payload
    assert "raw-scope-secret" not in recorder.events[1][1]
