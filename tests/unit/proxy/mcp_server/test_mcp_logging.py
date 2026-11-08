import asyncio
import json
import os
import subprocess
import sys
from collections.abc import Generator
from datetime import datetime
from pathlib import Path
from queue import SimpleQueue
from typing import Final
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from mcp import Client
from mcp.server.auth.middleware.auth_context import auth_context_var
from mcp.server.context import ServerRequestContext
from mcp.server.session import ServerSession
from mcp.shared.exceptions import MCPError
from mcp.types import CallToolResult, EmptyResult, InitializeRequestParams, Request, RequestParams, TextContent
from mcp.types import Tool as MCPTool
from pydantic import ConfigDict, ValidationError

import litellm
from litellm.caching.caching import DualCache
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import AlertType, LiteLLM_ObjectPermissionTable, UserAPIKeyAuth
from litellm.proxy.mcp_server.mcp_server_manager import (
    MCPServerManager,
)
from litellm.proxy.mcp_server.server import (
    _log_sdk_rejection,
    mcp_server_tool_call,
    server,
    set_auth_context,
)
from litellm.types.mcp import MCPPostCallResponseObject


class SDKRejectionParams(RequestParams):
    model_config = ConfigDict(frozen=True)
    name: int | None = None
    arguments: dict[str, str] | None = None
    level: int | str | None = None


class SDKRejectionLogger(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.failures: list[dict[str, object]] = []
        self.pre_calls = 0
        self.post_failures = 0
        self.fail_post_hook = False

    async def async_pre_call_hook(
        self, user_api_key_dict: UserAPIKeyAuth, cache: DualCache, data: dict[str, object], call_type: str
    ) -> dict[str, object]:
        self.pre_calls += 1
        return data

    async def async_log_failure_event(
        self, kwargs: dict[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        self.failures.append(kwargs)

    async def async_post_call_failure_hook(
        self,
        request_data: dict[str, object],
        original_exception: Exception,
        user_api_key_dict: UserAPIKeyAuth,
        traceback_str: str | None = None,
    ) -> None:
        self.post_failures += 1
        if self.fail_post_hook:
            raise RuntimeError("callback unavailable")


@pytest.fixture
def sdk_rejection_logger() -> Generator[SDKRejectionLogger, None, None]:
    logger: Final = SDKRejectionLogger()
    token: Final = auth_context_var.set(None)
    set_auth_context(
        user_api_key_auth=UserAPIKeyAuth(
            api_key="sk-sdk-rejection-unit-test", user_id="sdk-rejection-user", team_id="sdk-rejection-team"
        )
    )
    try:
        with patch("litellm.callbacks", [logger]):
            yield logger
    finally:
        auth_context_var.reset(token)
        litellm.logging_callback_manager._reset_all_callbacks()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "method,params,code,message",
    [
        ("tools/call-2", {"arguments": {"secret": "DO_NOT_LOG_THIS"}}, -32601, "Method not found"),
        ("tools/call", {"name": 123, "arguments": {"secret": "DO_NOT_LOG_THIS"}}, -32602, "Invalid request parameters"),
        ("tools/call", {"arguments": {"secret": "DO_NOT_LOG_THIS"}}, -32602, "Invalid request parameters"),
        ("logging/setLevel", {"level": 123}, -32602, "Invalid request parameters"),
        ("logging/setLevel", {"level": "info"}, -32601, "Method not found"),
        ("completion/complete", {}, -32602, "Invalid request parameters"),
        ("server/discover", {}, -32601, "Method not found"),
        ("subscriptions/listen", {}, -32601, "Method not found"),
    ],
)
async def test_sdk_rejection_emits_one_redacted_zero_cost_failure(
    sdk_rejection_logger: SDKRejectionLogger, method: str, params: dict[str, object], code: int, message: str
) -> None:
    async with Client(server, mode="legacy") as client:
        with pytest.raises(MCPError) as rejected:
            await client.session.send_request(
                Request(method=method, params=SDKRejectionParams.model_validate(params)), EmptyResult
            )
        assert rejected.value.code == code
        assert rejected.value.message == message
        await client.session.send_ping()

    assert sdk_rejection_logger.pre_calls == 0
    assert sdk_rejection_logger.post_failures == 1
    assert len(sdk_rejection_logger.failures) == 1
    payload: Final = sdk_rejection_logger.failures[0]["standard_logging_object"]
    assert payload["status"] == "failure"
    assert payload["response_cost"] == 0
    assert payload["prompt_tokens"] == 0
    assert payload["completion_tokens"] == 0
    assert payload["model"] == "MCP: request_rejected"
    assert payload["metadata"]["user_api_key_user_id"] == "sdk-rejection-user"
    assert payload["metadata"]["user_api_key_team_id"] == "sdk-rejection-team"
    rejection: Final = payload["metadata"]["spend_logs_metadata"]["mcp_sdk_rejection"]
    assert rejection["method"] == method
    assert rejection["request_id"]
    assert rejection["error_code"] == code
    assert rejection["error_message"] == message
    assert rejection["stage"] == "sdk_dispatch"
    assert "DO_NOT_LOG_THIS" not in json.dumps(payload, default=str)
    assert "sk-sdk-rejection-unit-test" not in json.dumps(payload, default=str)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "method,params,code",
    [
        ("tools/call-2", {}, -32601),
        ("tools/call", {"name": 123}, -32602),
    ],
)
async def test_sdk_caller_errors_do_not_trigger_llm_alerts(
    sdk_rejection_logger: SDKRejectionLogger, method: str, params: dict[str, object], code: int
) -> None:
    from litellm.proxy.proxy_server import proxy_logging_obj

    with (
        patch.object(proxy_logging_obj, "alert_types", [AlertType.llm_exceptions]),
        patch.object(proxy_logging_obj, "alerting_handler", new_callable=AsyncMock) as alerts,
    ):
        async with Client(server, mode="legacy") as client:
            with pytest.raises(MCPError) as rejected:
                await client.session.send_request(
                    Request(method=method, params=SDKRejectionParams.model_validate(params)), EmptyResult
                )
            assert rejected.value.code == code
        await asyncio.sleep(0)
        alerts.assert_not_awaited()
        assert len(sdk_rejection_logger.failures) == 1
        assert sdk_rejection_logger.post_failures == 1
        assert sdk_rejection_logger.pre_calls == 0

        await proxy_logging_obj.post_call_failure_hook(
            request_data={},
            original_exception=HTTPException(status_code=500, detail="upstream unavailable"),
            user_api_key_dict=UserAPIKeyAuth(),
        )
        await asyncio.sleep(0)
        alerts.assert_awaited_once()
        assert alerts.await_args.kwargs["level"] == "High"
        assert alerts.await_args.kwargs["alert_type"] == AlertType.llm_exceptions


@pytest.mark.asyncio
async def test_sdk_rejection_sync_logging_allows_concurrent_ping(
    sdk_rejection_logger: SDKRejectionLogger,
) -> None:
    loop: Final = asyncio.get_running_loop()
    ping_responses: Final = SimpleQueue[EmptyResult]()
    async with Client(server, mode="legacy") as client:
        def sync_failure(
            kwargs: dict[str, object], response_obj: object, start_time: datetime, end_time: datetime
        ) -> None:
            ping_responses.put(asyncio.run_coroutine_threadsafe(client.session.send_ping(), loop).result(timeout=10))

        with patch("litellm.failure_callback", [sync_failure]):
            with pytest.raises(MCPError) as rejected:
                await client.session.send_request(Request(method="unknown/method", params=RequestParams()), EmptyResult)
            assert rejected.value.code == -32601

    assert ping_responses.qsize() == 1
    assert isinstance(ping_responses.get_nowait(), EmptyResult)
    assert len(sdk_rejection_logger.failures) == 1
    assert sdk_rejection_logger.post_failures == 1
    assert sdk_rejection_logger.pre_calls == 0


@pytest.mark.asyncio
async def test_handler_method_not_found_does_not_emit_sdk_rejection(
    sdk_rejection_logger: SDKRejectionLogger,
) -> None:
    context: Final = ServerRequestContext(
        session=MagicMock(spec=ServerSession),
        lifespan_context=None,
        protocol_version="2025-11-25",
        method="tools/list",
        params={},
        request_id="handler-error",
    )
    handler_error: Final = MCPError(code=-32601, message="Handler rejected the request")
    with pytest.raises(MCPError) as rejected:
        await _log_sdk_rejection(context, AsyncMock(side_effect=handler_error))
    assert rejected.value is handler_error
    assert sdk_rejection_logger.failures == []
    assert sdk_rejection_logger.post_failures == 0


@pytest.mark.asyncio
async def test_sdk_initialization_errors_are_not_logged_as_tool_rejections(
    sdk_rejection_logger: SDKRejectionLogger,
) -> None:
    context: Final = ServerRequestContext(
        session=MagicMock(spec=ServerSession),
        lifespan_context=None,
        protocol_version="2025-11-25",
        method="initialize",
        params={},
        request_id="bad-init",
    )
    with pytest.raises(ValidationError) as sdk_error:
        InitializeRequestParams.model_validate({})
    with pytest.raises(ValidationError) as rejected:
        await _log_sdk_rejection(context, AsyncMock(side_effect=sdk_error.value))
    assert rejected.value is sdk_error.value
    assert sdk_rejection_logger.pre_calls == 0
    assert sdk_rejection_logger.failures == []
    assert sdk_rejection_logger.post_failures == 0


@pytest.mark.asyncio
async def test_sdk_response_survives_failure_callback_error(sdk_rejection_logger: SDKRejectionLogger) -> None:
    sdk_rejection_logger.fail_post_hook = True
    async with Client(server, mode="legacy") as client:
        with pytest.raises(MCPError) as rejected:
            await client.session.send_request(Request(method="tools/call-2", params=RequestParams()), EmptyResult)
        assert rejected.value.code == -32601
        assert rejected.value.message == "Method not found"
    assert sdk_rejection_logger.post_failures == 1
    assert len(sdk_rejection_logger.failures) == 1


@pytest.mark.asyncio
async def test_sdk_rejection_without_admitted_identity_is_not_attributed(
    sdk_rejection_logger: SDKRejectionLogger,
) -> None:
    token: Final = auth_context_var.set(None)
    try:
        async with Client(server, mode="legacy") as client:
            with pytest.raises(MCPError) as rejected:
                await client.session.send_request(Request(method="tools/call-2", params=RequestParams()), EmptyResult)
            assert rejected.value.code == -32601
    finally:
        auth_context_var.reset(token)
    assert sdk_rejection_logger.failures == []
    assert sdk_rejection_logger.post_failures == 0


@pytest.mark.asyncio
async def test_sdk_rejection_keeps_concurrent_caller_identity(sdk_rejection_logger: SDKRejectionLogger) -> None:
    async def reject_as(user: str) -> None:
        token: Final = auth_context_var.set(None)
        set_auth_context(user_api_key_auth=UserAPIKeyAuth(user_id=user, team_id=f"{user}-team"))
        try:
            async with Client(server, mode="legacy") as client:
                with pytest.raises(MCPError) as rejected:
                    await client.session.send_request(
                        Request(method=f"unknown/{user}", params=RequestParams()), EmptyResult
                    )
                assert rejected.value.code == -32601
        finally:
            auth_context_var.reset(token)

    await asyncio.gather(reject_as("caller-a"), reject_as("caller-b"))
    assert len(sdk_rejection_logger.failures) == 2
    assert sdk_rejection_logger.post_failures == 2
    assert {
        failure["standard_logging_object"]["metadata"]["user_api_key_user_id"]
        for failure in sdk_rejection_logger.failures
    } == {"caller-a", "caller-b"}
    for failure in sdk_rejection_logger.failures:
        metadata: Final = failure["standard_logging_object"]["metadata"]
        identity: Final = metadata["user_api_key_user_id"]
        assert metadata["user_api_key_team_id"] == f"{identity}-team"
        assert metadata["spend_logs_metadata"]["mcp_sdk_rejection"]["method"] == f"unknown/{identity}"


def _mcp_request_ctx(**overrides):
    from types import SimpleNamespace

    from mcp.server.context import ServerRequestContext

    kwargs = {
        "session": SimpleNamespace(),
        "lifespan_context": {},
        "protocol_version": "2025-06-18",
        "method": "",
        "params": None,
        "request_id": 1,
        "meta": None,
        "request": None,
    }
    kwargs.update(overrides)
    return ServerRequestContext(**kwargs)


def _call_tool_params(name, arguments=None):
    from mcp.types import CallToolRequestParams

    return CallToolRequestParams(name=name, arguments=arguments)

class TestMCPLogger(CustomLogger):
    def __init__(self):
        self.standard_logging_payload = None
        self.mcp_tool_call_payloads = []
        super().__init__()

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        print("success event")
        payload = kwargs.get("standard_logging_object", None)
        self.standard_logging_payload = payload
        # Async success events from other calls (e.g. a mocked acompletion whose
        # log task is delivered late) race with the MCP event for the single
        # last-writer slot; keep MCP tool calls in their own list so assertions
        # are order-independent.
        if payload is not None and payload.get("call_type") == "call_mcp_tool":
            self.mcp_tool_call_payloads.append(payload)
        print(f"Captured standard_logging_payload: {payload}")


def _set_authorized_user(server_ids):
    """Configure auth context with permission to call the specified servers."""
    server_list = list(server_ids)
    user_auth = UserAPIKeyAuth(
        api_key="test",
        user_id="test_user",
        object_permission=LiteLLM_ObjectPermissionTable(
            object_permission_id="mcp-test-permissions",
            mcp_servers=server_list,
        ),
    )
    set_auth_context(user_api_key_auth=user_auth, mcp_servers=server_list)


@pytest.mark.asyncio
async def test_mcp_cost_tracking():
    # Create a mock tool call result
    litellm.logging_callback_manager._reset_all_callbacks()
    mock_result = CallToolResult(
        content=[TextContent(type="text", text="Test response")], isError=False
    )

    # Create a mock MCPClient
    mock_client = AsyncMock()
    mock_client.call_tool = AsyncMock(return_value=mock_result)
    mock_client.list_tools = AsyncMock(
        return_value=[
            MCPTool(
                name="add_tools",
                description="Test tool",
                inputSchema={
                    "type": "object",
                    "properties": {"test": {"type": "string"}},
                },
            )
        ]
    )

    # Mock the MCPClient constructor
    def mock_client_constructor(*args, **kwargs):
        return mock_client

    # Initialize the server manager
    local_mcp_server_manager = MCPServerManager()

    with patch(
        "litellm.proxy.mcp_server.upstream.MCPClient",
        mock_client_constructor,
    ):
        # Load the server config
        await local_mcp_server_manager.load_servers_from_config(
            mcp_servers_config={
                "zapier_gmail_server": {
                    "url": os.getenv("ZAPIER_MCP_HTTPS_SERVER_URL"),
                    "mcp_info": {
                        "mcp_server_cost_info": {
                            "default_cost_per_query": 1.2,
                        }
                    },
                }
            }
        )

        # Set up the test logger
        test_logger = TestMCPLogger()
        litellm.callbacks = [test_logger]

        # Initialize the tool mapping
        await local_mcp_server_manager._initialize_tool_name_to_mcp_server_name_mapping()

        # Patch the global manager in both modules where it's used
        with (
            patch(
                "litellm.proxy.mcp_server.mcp_server_manager.global_mcp_server_manager",
                local_mcp_server_manager,
            ),
            patch(
                "litellm.proxy.mcp_server.operations.global_mcp_server_manager",
                local_mcp_server_manager,
            ),
        ):

            _set_authorized_user(local_mcp_server_manager.get_all_mcp_server_ids())

            print(
                "tool_name_to_mcp_server_name_mapping",
                local_mcp_server_manager.tool_name_to_mcp_server_name_mapping,
            )

            # Manually add the tool mapping to ensure it's available (since mocking might not capture it properly)
            local_mcp_server_manager.tool_name_to_mcp_server_name_mapping[
                "add_tools"
            ] = "zapier_gmail_server"
            local_mcp_server_manager.tool_name_to_mcp_server_name_mapping[
                "zapier_gmail_server-add_tools"
            ] = "zapier_gmail_server"

            # Call mcp tool
            response = await mcp_server_tool_call(
                _mcp_request_ctx(),
                _call_tool_params("zapier_gmail_server-add_tools", {"test": "test"}),
            )

            # wait 1-2 seconds for logging to be processed
            await asyncio.sleep(2)

            logged_standard_logging_payload = (
                test_logger.mcp_tool_call_payloads[-1]
                if test_logger.mcp_tool_call_payloads
                else None
            )
            print("logged_standard_logging_payload", logged_standard_logging_payload)

            # Add assertions
            assert response is not None
            # Handle CallToolResult - access .content for the list of content items
            if isinstance(response, CallToolResult):
                response_list = response.content
            else:
                response_list = list(
                    response
                )  # Convert iterable to list for backward compatibility
            assert len(response_list) == 1
            assert isinstance(response_list[0], TextContent)
            assert response_list[0].text == "Test response"

            # Verify client methods were called
            mock_client.call_tool.assert_called_once()

            ######
            # verify response cost is 1.2 as set on default_cost_per_query
            # Critical - the cost is tracked as $1.2
            assert (
                logged_standard_logging_payload is not None
            ), "Standard logging payload should not be None"
            assert logged_standard_logging_payload["response_cost"] == 1.2


@pytest.mark.asyncio
async def test_mcp_cost_tracking_per_tool():
    """Test that individual tool costs are tracked correctly when tool_name_to_cost_per_query is configured"""
    # Create a mock tool call result
    litellm.logging_callback_manager._reset_all_callbacks()
    mock_result = CallToolResult(
        content=[TextContent(type="text", text="Test response")], isError=False
    )

    # Create a mock MCPClient
    mock_client = AsyncMock()
    mock_client.call_tool = AsyncMock(return_value=mock_result)
    mock_client.list_tools = AsyncMock(
        return_value=[
            MCPTool(
                name="expensive_tool",
                description="Expensive tool",
                inputSchema={
                    "type": "object",
                    "properties": {"data": {"type": "string"}},
                },
            ),
            MCPTool(
                name="cheap_tool",
                description="Cheap tool",
                inputSchema={
                    "type": "object",
                    "properties": {"data": {"type": "string"}},
                },
            ),
        ]
    )

    # Mock the MCPClient constructor
    def mock_client_constructor(*args, **kwargs):
        return mock_client

    # Initialize the server manager
    local_mcp_server_manager = MCPServerManager()

    with patch(
        "litellm.proxy.mcp_server.upstream.MCPClient",
        mock_client_constructor,
    ):
        # Load the server config with per-tool costs
        await local_mcp_server_manager.load_servers_from_config(
            mcp_servers_config={
                "test_server": {
                    "url": os.getenv("ZAPIER_MCP_HTTPS_SERVER_URL"),
                    "mcp_info": {
                        "mcp_server_cost_info": {
                            "default_cost_per_query": 0.5,  # Default cost
                            "tool_name_to_cost_per_query": {
                                "expensive_tool": 5.0,  # High cost tool
                                "cheap_tool": 0.1,  # Low cost tool
                            },
                        }
                    },
                }
            }
        )

        # Set up the test logger
        test_logger = TestMCPLogger()
        litellm.callbacks = [test_logger]

        # Initialize the tool mapping
        await local_mcp_server_manager._initialize_tool_name_to_mcp_server_name_mapping()

        # Manually add the tool mapping to ensure it's available (since mocking might not capture it properly)
        local_mcp_server_manager.tool_name_to_mcp_server_name_mapping[
            "expensive_tool"
        ] = "test_server"
        local_mcp_server_manager.tool_name_to_mcp_server_name_mapping[
            "test_server-expensive_tool"
        ] = "test_server"
        local_mcp_server_manager.tool_name_to_mcp_server_name_mapping["cheap_tool"] = (
            "test_server"
        )
        local_mcp_server_manager.tool_name_to_mcp_server_name_mapping[
            "test_server-cheap_tool"
        ] = "test_server"

        # Patch the global manager in both modules where it's used
        with (
            patch(
                "litellm.proxy.mcp_server.mcp_server_manager.global_mcp_server_manager",
                local_mcp_server_manager,
            ),
            patch(
                "litellm.proxy.mcp_server.operations.global_mcp_server_manager",
                local_mcp_server_manager,
            ),
        ):

            _set_authorized_user(local_mcp_server_manager.get_all_mcp_server_ids())

            print(
                "tool_name_to_mcp_server_name_mapping",
                local_mcp_server_manager.tool_name_to_mcp_server_name_mapping,
            )

            # Test 1: Call expensive_tool - should cost 5.0
            response1 = await mcp_server_tool_call(
                _mcp_request_ctx(),
                _call_tool_params("test_server-expensive_tool", {"data": "test_expensive"}),
            )

            # wait for logging to be processed
            await asyncio.sleep(2)

            logged_standard_logging_payload_1 = (
                test_logger.mcp_tool_call_payloads[-1]
                if test_logger.mcp_tool_call_payloads
                else None
            )
            print(
                "logged_standard_logging_payload_1", logged_standard_logging_payload_1
            )

            # Verify expensive tool cost
            assert (
                logged_standard_logging_payload_1 is not None
            ), "Standard logging payload 1 should not be None"
            assert logged_standard_logging_payload_1["response_cost"] == 5.0

            # Reset logger for second test
            test_logger.standard_logging_payload = None
            test_logger.mcp_tool_call_payloads.clear()

            # Test 2: Call cheap_tool - should cost 0.1
            response2 = await mcp_server_tool_call(
                _mcp_request_ctx(),
                _call_tool_params("test_server-cheap_tool", {"data": "test_cheap"}),
            )

            # wait for logging to be processed
            await asyncio.sleep(2)

            logged_standard_logging_payload_2 = (
                test_logger.mcp_tool_call_payloads[-1]
                if test_logger.mcp_tool_call_payloads
                else None
            )
            print(
                "logged_standard_logging_payload_2", logged_standard_logging_payload_2
            )

            # Verify cheap tool cost
            assert (
                logged_standard_logging_payload_2 is not None
            ), "Standard logging payload 2 should not be None"
            assert logged_standard_logging_payload_2["response_cost"] == 0.1

            # Add basic response assertions
            assert response1 is not None
            assert response2 is not None

            response_list_1 = list(response1.content)
            response_list_2 = list(response2.content)

            assert len(response_list_1) == 1
            assert len(response_list_2) == 1
            assert isinstance(response_list_1[0], TextContent)
            assert isinstance(response_list_2[0], TextContent)
            assert response_list_1[0].text == "Test response"
            assert response_list_2[0].text == "Test response"

            # Verify client methods were called twice
            assert mock_client.call_tool.call_count == 2


class MCPLoggerHook(TestMCPLogger):
    async def async_post_mcp_tool_call_hook(
        self, kwargs, response_obj: MCPPostCallResponseObject, start_time, end_time
    ) -> MCPPostCallResponseObject | None:
        print("post mcp tool call response_obj", response_obj)
        # update the MCPPostCallResponseObject with the response_cost
        response_obj.hidden_params.response_cost = 1.42
        return response_obj


@pytest.mark.asyncio
async def test_mcp_tool_call_hook():
    # Create a mock tool call result
    litellm.logging_callback_manager._reset_all_callbacks()
    mock_result = CallToolResult(
        content=[TextContent(type="text", text="Test response")], isError=False
    )

    # Create a mock MCPClient
    mock_client = AsyncMock()
    mock_client.call_tool = AsyncMock(return_value=mock_result)
    mock_client.list_tools = AsyncMock(
        return_value=[
            MCPTool(
                name="add_tools",
                description="Test tool",
                inputSchema={
                    "type": "object",
                    "properties": {"test": {"type": "string"}},
                },
            )
        ]
    )

    # Mock the MCPClient constructor
    def mock_client_constructor(*args, **kwargs):
        return mock_client

    # Initialize the server manager
    local_mcp_server_manager = MCPServerManager()

    with patch(
        "litellm.proxy.mcp_server.upstream.MCPClient",
        mock_client_constructor,
    ):
        # Load the server config
        await local_mcp_server_manager.load_servers_from_config(
            mcp_servers_config={
                "zapier_gmail_server": {
                    "url": os.getenv("ZAPIER_MCP_HTTPS_SERVER_URL"),
                }
            }
        )

        # Set up the test logger
        test_logger = MCPLoggerHook()
        litellm.callbacks = [test_logger]

        # Initialize the tool mapping
        await local_mcp_server_manager._initialize_tool_name_to_mcp_server_name_mapping()

        # Manually add the tool mapping to ensure it's available (since mocking might not capture it properly)
        local_mcp_server_manager.tool_name_to_mcp_server_name_mapping["add_tools"] = (
            "zapier_gmail_server"
        )
        local_mcp_server_manager.tool_name_to_mcp_server_name_mapping[
            "zapier_gmail_server-add_tools"
        ] = "zapier_gmail_server"

        # Patch the global manager in both modules where it's used
        with (
            patch(
                "litellm.proxy.mcp_server.mcp_server_manager.global_mcp_server_manager",
                local_mcp_server_manager,
            ),
            patch(
                "litellm.proxy.mcp_server.operations.global_mcp_server_manager",
                local_mcp_server_manager,
            ),
        ):

            _set_authorized_user(local_mcp_server_manager.get_all_mcp_server_ids())

            print(
                "tool_name_to_mcp_server_name_mapping",
                local_mcp_server_manager.tool_name_to_mcp_server_name_mapping,
            )

            # Call mcp tool using the correct separator format (- not /)
            response = await mcp_server_tool_call(
                _mcp_request_ctx(),
                _call_tool_params("zapier_gmail_server-add_tools", {"test": "test"}),
            )

            # wait 1-2 seconds for logging to be processed
            await asyncio.sleep(2)

            # check logged standard logging payload
            logged_standard_logging_payload = (
                test_logger.mcp_tool_call_payloads[-1]
                if test_logger.mcp_tool_call_payloads
                else None
            )
            print("logged_standard_logging_payload", logged_standard_logging_payload)
            assert (
                logged_standard_logging_payload is not None
            ), "Standard logging payload should not be None"
            assert logged_standard_logging_payload["response_cost"] == 1.42


_QUEUED_LOGGING_OUTLIVES_TEST = '''
import time

from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER

ran_at = []


async def _record_run():
    ran_at.append(time.monotonic())


async def test_1_leaves_logging_queued_behind_a_stopped_worker():
    GLOBAL_LOGGING_WORKER.ensure_initialized_and_enqueue(_record_run())
    await GLOBAL_LOGGING_WORKER.stop()
    assert ran_at == []


async def test_2_starts_after_the_previous_tests_logging_ran():
    started_at = time.monotonic()
    GLOBAL_LOGGING_WORKER.ensure_initialized_and_enqueue(_record_run())
    await GLOBAL_LOGGING_WORKER.flush()
    assert [t < started_at for t in ran_at] == [True, False]
'''


def test_logging_queued_by_one_test_is_drained_before_the_next(tmp_path: Path):
    """Regression: a logging coroutine queued by one test must not run inside a later test (it would log into that
    test's callbacks, which is how test_mcp_tool_call_hook captured a gpt-4o-mini payload under xdist)."""
    (tmp_path / "conftest.py").write_text((Path(__file__).parent / "conftest.py").read_text())
    (tmp_path / "pyproject.toml").write_text('[tool.pytest.ini_options]\nasyncio_mode = "auto"\n')
    (tmp_path / "test_queued_logging.py").write_text(_QUEUED_LOGGING_OUTLIVES_TEST)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "test_queued_logging.py"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
