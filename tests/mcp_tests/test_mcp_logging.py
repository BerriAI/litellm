import os
import sys
import pytest
import asyncio
from typing import Optional
from unittest.mock import AsyncMock, patch


sys.path.insert(0, os.path.abspath("../../.."))  # Adds the parent directory to the system path
import litellm
from litellm.types.utils import StandardLoggingPayload
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._experimental.mcp_server.server import (
    mcp_server_tool_call,
    set_auth_context,
)
from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
    MCPServerManager,
)
from litellm.proxy._types import LiteLLM_ObjectPermissionTable, UserAPIKeyAuth
from litellm.types.mcp import MCPPostCallResponseObject
from litellm.types.utils import HiddenParams
from mcp.types import Tool as MCPTool, CallToolResult, TextContent


class TestMCPLogger(CustomLogger):
    def __init__(self):
        self.standard_logging_payload = None
        super().__init__()

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        print("success event")
        self.standard_logging_payload = kwargs.get("standard_logging_object", None)
        print(f"Captured standard_logging_payload: {self.standard_logging_payload}")


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
    mock_result = CallToolResult(content=[TextContent(type="text", text="Test response")], isError=False)

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
        "litellm.proxy._experimental.mcp_server.mcp_server_manager.MCPClient",
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
                "litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager",
                local_mcp_server_manager,
            ),
            patch(
                "litellm.proxy._experimental.mcp_server.server.global_mcp_server_manager",
                local_mcp_server_manager,
            ),
        ):
            _set_authorized_user(local_mcp_server_manager.get_all_mcp_server_ids())

            print(
                "tool_name_to_mcp_server_name_mapping",
                local_mcp_server_manager.tool_name_to_mcp_server_name_mapping,
            )

            # Manually add the tool mapping to ensure it's available (since mocking might not capture it properly)
            local_mcp_server_manager.tool_name_to_mcp_server_name_mapping["add_tools"] = "zapier_gmail_server"
            local_mcp_server_manager.tool_name_to_mcp_server_name_mapping["zapier_gmail_server-add_tools"] = (
                "zapier_gmail_server"
            )

            # Call mcp tool
            response = await mcp_server_tool_call(
                name="zapier_gmail_server-add_tools",  # Use correct prefixed name with - separator
                arguments={"test": "test"},
            )

            # wait 1-2 seconds for logging to be processed
            await asyncio.sleep(2)

            logged_standard_logging_payload = test_logger.standard_logging_payload
            print("logged_standard_logging_payload", logged_standard_logging_payload)

            # Add assertions
            assert response is not None
            # Handle CallToolResult - access .content for the list of content items
            if isinstance(response, CallToolResult):
                response_list = response.content
            else:
                response_list = list(response)  # Convert iterable to list for backward compatibility
            assert len(response_list) == 1
            assert isinstance(response_list[0], TextContent)
            assert response_list[0].text == "Test response"

            # Verify client methods were called
            mock_client.call_tool.assert_called_once()

            ######
            # verify response cost is 1.2 as set on default_cost_per_query
            # Critical - the cost is tracked as $1.2
            assert logged_standard_logging_payload is not None, "Standard logging payload should not be None"
            assert logged_standard_logging_payload["response_cost"] == 1.2


@pytest.mark.asyncio
async def test_mcp_cost_tracking_per_tool():
    """Test that individual tool costs are tracked correctly when tool_name_to_cost_per_query is configured"""
    # Create a mock tool call result
    litellm.logging_callback_manager._reset_all_callbacks()
    mock_result = CallToolResult(content=[TextContent(type="text", text="Test response")], isError=False)

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
        "litellm.proxy._experimental.mcp_server.mcp_server_manager.MCPClient",
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
        local_mcp_server_manager.tool_name_to_mcp_server_name_mapping["expensive_tool"] = "test_server"
        local_mcp_server_manager.tool_name_to_mcp_server_name_mapping["test_server-expensive_tool"] = "test_server"
        local_mcp_server_manager.tool_name_to_mcp_server_name_mapping["cheap_tool"] = "test_server"
        local_mcp_server_manager.tool_name_to_mcp_server_name_mapping["test_server-cheap_tool"] = "test_server"

        # Patch the global manager in both modules where it's used
        with (
            patch(
                "litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager",
                local_mcp_server_manager,
            ),
            patch(
                "litellm.proxy._experimental.mcp_server.server.global_mcp_server_manager",
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
                name="test_server-expensive_tool",  # Use correct prefixed name with - separator
                arguments={"data": "test_expensive"},
            )

            # wait for logging to be processed
            await asyncio.sleep(2)

            logged_standard_logging_payload_1 = test_logger.standard_logging_payload
            print("logged_standard_logging_payload_1", logged_standard_logging_payload_1)

            # Verify expensive tool cost
            assert logged_standard_logging_payload_1 is not None, "Standard logging payload 1 should not be None"
            assert logged_standard_logging_payload_1["response_cost"] == 5.0

            # Reset logger for second test
            test_logger.standard_logging_payload = None

            # Test 2: Call cheap_tool - should cost 0.1
            response2 = await mcp_server_tool_call(
                name="test_server-cheap_tool",  # Use correct prefixed name with - separator
                arguments={"data": "test_cheap"},
            )

            # wait for logging to be processed
            await asyncio.sleep(2)

            logged_standard_logging_payload_2 = test_logger.standard_logging_payload
            print("logged_standard_logging_payload_2", logged_standard_logging_payload_2)

            # Verify cheap tool cost
            assert logged_standard_logging_payload_2 is not None, "Standard logging payload 2 should not be None"
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


class MCPLoggerHook(CustomLogger):
    def __init__(self):
        self.standard_logging_payload = None
        super().__init__()

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        print("success event")
        self.standard_logging_payload = kwargs.get("standard_logging_object", None)
        print(f"Captured standard_logging_payload: {self.standard_logging_payload}")

    async def async_post_mcp_tool_call_hook(
        self, kwargs, response_obj: MCPPostCallResponseObject, start_time, end_time
    ) -> Optional[MCPPostCallResponseObject]:
        print("post mcp tool call response_obj", response_obj)
        # update the MCPPostCallResponseObject with the response_cost
        response_obj.hidden_params.response_cost = 1.42
        return response_obj


@pytest.mark.asyncio
async def test_mcp_tool_call_hook():
    # Create a mock tool call result
    litellm.logging_callback_manager._reset_all_callbacks()
    mock_result = CallToolResult(content=[TextContent(type="text", text="Test response")], isError=False)

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
        "litellm.proxy._experimental.mcp_server.mcp_server_manager.MCPClient",
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
        local_mcp_server_manager.tool_name_to_mcp_server_name_mapping["add_tools"] = "zapier_gmail_server"
        local_mcp_server_manager.tool_name_to_mcp_server_name_mapping["zapier_gmail_server-add_tools"] = (
            "zapier_gmail_server"
        )

        # Patch the global manager in both modules where it's used
        with (
            patch(
                "litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager",
                local_mcp_server_manager,
            ),
            patch(
                "litellm.proxy._experimental.mcp_server.server.global_mcp_server_manager",
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
                name="zapier_gmail_server-add_tools",  # Use correct prefixed name with - separator
                arguments={"test": "test"},
            )

            # wait 1-2 seconds for logging to be processed
            await asyncio.sleep(2)

            # check logged standard logging payload
            logged_standard_logging_payload = test_logger.standard_logging_payload
            print("logged_standard_logging_payload", logged_standard_logging_payload)
            assert logged_standard_logging_payload is not None, "Standard logging payload should not be None"
            assert logged_standard_logging_payload["response_cost"] == 1.42


# ---------------------------------------------------------------------------
# Tests for #28929: standard_logging_object on /mcp/ JSON-RPC protocol-level
# rejections (unknown method / malformed params). These exercise the additive
# send-wrapper + helper added to handle_streamable_http_mcp's dispatch path.
#
# The load-bearing assertion is that a REAL standard_logging_object is emitted
# to a registered failure callback (not merely that an internal hook was
# called) — the rejection path builds a genuine LiteLLMLoggingObj and drives
# its async_failure_handler, mirroring list_mcp_tools / call_mcp_tool.
# ---------------------------------------------------------------------------
import json
from datetime import datetime

from litellm.proxy._experimental.mcp_server.server import (
    _jsonrpc_rejection_reason,
    _log_mcp_protocol_rejection,
    _parse_jsonrpc_request_for_logging,
    _wrap_send_for_protocol_error_logging,
)
from litellm.types.utils import CallTypes


class _ProtocolRejectionLogger(CustomLogger):
    """Captures the standard_logging_object dispatched on the failure path."""

    def __init__(self):
        self.failure_payloads = []
        super().__init__()

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        slo = kwargs.get("standard_logging_object", None)
        self.failure_payloads.append(slo)

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        # Should never fire for a protocol rejection.
        self.failure_payloads.append(("UNEXPECTED_SUCCESS", kwargs.get("standard_logging_object")))


def _jsonrpc_error_body(code, message="boom", _id=1):
    return json.dumps({"jsonrpc": "2.0", "id": _id, "error": {"code": code, "message": message}}).encode("utf-8")


def _jsonrpc_result_body(_id=1):
    return json.dumps({"jsonrpc": "2.0", "id": _id, "result": {"ok": True}}).encode("utf-8")


def test_mcp_jsonrpc_rejection_reason_mapping():
    assert _jsonrpc_rejection_reason(-32601) == "unknown_method"
    assert _jsonrpc_rejection_reason(-32602) == "malformed_params"
    assert _jsonrpc_rejection_reason(-32700) == "parse_error"
    assert _jsonrpc_rejection_reason(-32600) == "invalid_request"
    # Unknown / None codes degrade to a generic, low-cardinality reason.
    assert _jsonrpc_rejection_reason(12345) == "protocol_error"
    assert _jsonrpc_rejection_reason(None) == "protocol_error"


@pytest.mark.asyncio
async def test_mcp_protocol_rejection_emits_standard_logging_object():
    """A JSON-RPC error response (no result) must cause a REAL
    standard_logging_object to be dispatched to a registered failure callback,
    carrying the rejection metadata (#28929)."""
    litellm.logging_callback_manager._reset_all_callbacks()
    capture = _ProtocolRejectionLogger()
    litellm.callbacks = [capture]

    user_auth = UserAPIKeyAuth(api_key="test", user_id="u1")
    sent_messages = []

    async def fake_send(message):
        sent_messages.append(message)

    try:
        wrapped = _wrap_send_for_protocol_error_logging(
            fake_send,
            request_method="tools/call",
            params={"name": "totally_made_up", "arguments": {}},
            request_id=7,
            user_api_key_auth=user_auth,
            raw_headers={},
            start_time=datetime.now(),
        )
        body = _jsonrpc_error_body(-32602, "Invalid request parameters", _id=7)
        await wrapped({"type": "http.response.start", "status": 200, "headers": []})
        await wrapped({"type": "http.response.body", "body": body})
        # async_failure_handler may dispatch on the loop; give it a tick.
        await asyncio.sleep(1)
    finally:
        litellm.callbacks = []

    # Response forwarded byte-for-byte and unchanged.
    assert any(m.get("type") == "http.response.body" and m.get("body") == body for m in sent_messages), (
        "original response bytes must be forwarded unchanged"
    )

    # A genuine, non-None standard_logging_object reached the callback.
    non_none = [p for p in capture.failure_payloads if isinstance(p, dict)]
    assert non_none, (
        f"expected a non-None standard_logging_object on the failure callback, got: {capture.failure_payloads!r}"
    )
    slo = non_none[0]
    assert slo.get("call_type") == CallTypes.call_mcp_tool.value
    # status is a failure-bearing value (StandardLoggingPayloadStatus).
    assert slo.get("status") == "failure"
    # Rejection metadata threaded through spend_logs_metadata.
    spend_meta = (slo.get("metadata") or {}).get("spend_logs_metadata") or {}
    assert spend_meta.get("mcp_operation") == "tools/call"
    assert spend_meta.get("rejection_reason") == "malformed_params"
    assert spend_meta.get("jsonrpc_error_code") == -32602
    assert spend_meta.get("mcp_tool_name") == "totally_made_up"


@pytest.mark.asyncio
async def test_mcp_protocol_unknown_method_emits_slo():
    """An unknown-method (-32601) rejection emits an SLO with the right
    reason and no tool name (empty params)."""
    litellm.logging_callback_manager._reset_all_callbacks()
    capture = _ProtocolRejectionLogger()
    litellm.callbacks = [capture]

    user_auth = UserAPIKeyAuth(api_key="test", user_id="u1")

    async def fake_send(message):
        pass

    try:
        wrapped = _wrap_send_for_protocol_error_logging(
            fake_send,
            request_method="tools/totally_made_up",
            params={},
            request_id=1,
            user_api_key_auth=user_auth,
            raw_headers={},
            start_time=datetime.now(),
        )
        await wrapped(
            {
                "type": "http.response.body",
                "body": _jsonrpc_error_body(-32601, "Method not found"),
            }
        )
        await asyncio.sleep(1)
    finally:
        litellm.callbacks = []

    non_none = [p for p in capture.failure_payloads if isinstance(p, dict)]
    assert non_none, f"expected an SLO, got: {capture.failure_payloads!r}"
    spend_meta = (non_none[0].get("metadata") or {}).get("spend_logs_metadata") or {}
    assert spend_meta.get("rejection_reason") == "unknown_method"
    assert "mcp_tool_name" not in spend_meta


@pytest.mark.asyncio
async def test_mcp_protocol_batch_rejection_emits_slo():
    """A JSON-RPC batch (array body) the MCP SDK rejects as one error must
    still produce one SLO, attributed to mcp_operation='batch' (#28929,
    Codex batch finding)."""
    litellm.logging_callback_manager._reset_all_callbacks()
    capture = _ProtocolRejectionLogger()
    litellm.callbacks = [capture]

    user_auth = UserAPIKeyAuth(api_key="test", user_id="u1")

    async def fake_send(message):
        pass

    try:
        # Mirrors how handle_streamable_http_mcp maps a top-level array to a
        # synthetic 'batch' operation before wrapping send.
        wrapped = _wrap_send_for_protocol_error_logging(
            fake_send,
            request_method="batch",
            params={},
            request_id=None,
            user_api_key_auth=user_auth,
            raw_headers={},
            start_time=datetime.now(),
        )
        await wrapped(
            {
                "type": "http.response.body",
                "body": _jsonrpc_error_body(-32602, "Validation error", _id="server-error"),
            }
        )
        await asyncio.sleep(1)
    finally:
        litellm.callbacks = []

    non_none = [p for p in capture.failure_payloads if isinstance(p, dict)]
    assert non_none, f"expected a batch SLO, got: {capture.failure_payloads!r}"
    spend_meta = (non_none[0].get("metadata") or {}).get("spend_logs_metadata") or {}
    assert spend_meta.get("mcp_operation") == "batch"
    assert spend_meta.get("rejection_reason") == "malformed_params"


@pytest.mark.asyncio
async def test_mcp_protocol_success_path_emits_no_failure_record():
    """A JSON-RPC success response (has result) must NOT emit a failure
    record, and the response must still be forwarded unchanged."""
    litellm.logging_callback_manager._reset_all_callbacks()
    capture = _ProtocolRejectionLogger()
    litellm.callbacks = [capture]

    user_auth = UserAPIKeyAuth(api_key="test", user_id="u1")
    sent_messages = []

    async def fake_send(message):
        sent_messages.append(message)

    try:
        wrapped = _wrap_send_for_protocol_error_logging(
            fake_send,
            request_method="tools/call",
            params={"name": "add", "arguments": {}},
            request_id=1,
            user_api_key_auth=user_auth,
            raw_headers={},
            start_time=datetime.now(),
        )
        body = _jsonrpc_result_body(_id=1)
        await wrapped({"type": "http.response.body", "body": body})
        await asyncio.sleep(1)
    finally:
        litellm.callbacks = []

    assert capture.failure_payloads == [], (
        f"success path must not dispatch any failure record, got: {capture.failure_payloads!r}"
    )
    assert any(m.get("body") == body for m in sent_messages)


@pytest.mark.asyncio
async def test_mcp_protocol_wrapper_noop_when_not_a_request():
    """When the POST carries no JSON-RPC method (request_method is None) the
    wrapper returns the original send untouched (no logging seam at all)."""

    async def fake_send(message):
        pass

    returned = _wrap_send_for_protocol_error_logging(
        fake_send,
        request_method=None,
        params={},
        request_id=None,
        user_api_key_auth=UserAPIKeyAuth(api_key="test", user_id="u1"),
        raw_headers={},
        start_time=datetime.now(),
    )
    assert returned is fake_send, "non-request POSTs must not be wrapped"


@pytest.mark.asyncio
async def test_mcp_protocol_logging_failure_does_not_break_response():
    """If logging-object construction raises, the wrapper must swallow it and
    still forward the client response (observability must never drop a
    response)."""
    litellm.logging_callback_manager._reset_all_callbacks()
    litellm.callbacks = []

    user_auth = UserAPIKeyAuth(api_key="test", user_id="u1")
    sent_messages = []

    async def fake_send(message):
        sent_messages.append(message)

    try:
        with patch(
            "litellm.proxy._experimental.mcp_server.server.function_setup",
            side_effect=RuntimeError("logging init down"),
        ):
            wrapped = _wrap_send_for_protocol_error_logging(
                fake_send,
                request_method="tools/call",
                params={"name": "add"},
                request_id=1,
                user_api_key_auth=user_auth,
                raw_headers={},
                start_time=datetime.now(),
            )
            body = _jsonrpc_error_body(-32602)
            # Must not raise.
            await wrapped({"type": "http.response.body", "body": body})
    finally:
        litellm.callbacks = []

    assert any(m.get("body") == body for m in sent_messages), "response must be forwarded even when logging init raises"


@pytest.mark.asyncio
async def test_mcp_log_protocol_rejection_no_auth_is_noop():
    """The helper is a no-op (no exception, no SLO) when there is no user
    auth."""
    litellm.logging_callback_manager._reset_all_callbacks()
    capture = _ProtocolRejectionLogger()
    litellm.callbacks = [capture]
    try:
        await _log_mcp_protocol_rejection(
            request_method="tools/call",
            params={"name": "x"},
            jsonrpc_error={"code": -32602, "message": "bad"},
            request_id=1,
            user_api_key_auth=None,
            raw_headers={},
            start_time=datetime.now(),
        )
        await asyncio.sleep(0.2)
    finally:
        litellm.callbacks = []
    assert capture.failure_payloads == []


@pytest.mark.asyncio
async def test_mcp_protocol_wrapper_skips_oversized_body():
    """V2 guard: a response body larger than the parse cap is never
    deserialized (a permitted tool result), so no SLO is emitted even if it
    happens to contain an error-shaped substring."""
    litellm.logging_callback_manager._reset_all_callbacks()
    capture = _ProtocolRejectionLogger()
    litellm.callbacks = [capture]

    sent = []

    async def fake_send(message):
        sent.append(message)

    # Build a body > 64 KiB that still contains the error marker bytes.
    big = b'{"result": {"x": "' + (b"a" * (70 * 1024)) + b'"}, "error": 1}'
    try:
        wrapped = _wrap_send_for_protocol_error_logging(
            fake_send,
            request_method="tools/call",
            params={"name": "add"},
            request_id=1,
            user_api_key_auth=UserAPIKeyAuth(api_key="test", user_id="u1"),
            raw_headers={},
            start_time=datetime.now(),
        )
        await wrapped({"type": "http.response.body", "body": big})
        await asyncio.sleep(0.3)
    finally:
        litellm.callbacks = []

    assert capture.failure_payloads == [], "oversized body must not be parsed/logged"
    assert any(m.get("body") == big for m in sent)


@pytest.mark.asyncio
async def test_mcp_protocol_wrapper_skips_non_error_body():
    """V2 guard: the cheap byte pre-check skips bodies that don't look like an
    error envelope (no '"error"', or a '"result"' present) without json.loads."""
    litellm.logging_callback_manager._reset_all_callbacks()
    capture = _ProtocolRejectionLogger()
    litellm.callbacks = [capture]

    async def fake_send(message):
        pass

    try:
        wrapped = _wrap_send_for_protocol_error_logging(
            fake_send,
            request_method="tools/call",
            params={"name": "add"},
            request_id=1,
            user_api_key_auth=UserAPIKeyAuth(api_key="test", user_id="u1"),
            raw_headers={},
            start_time=datetime.now(),
        )
        # No "error" marker at all.
        await wrapped({"type": "http.response.body", "body": b'{"jsonrpc":"2.0","id":1}'})
        # Both "error" and "result" present -> not a pure rejection.
        await wrapped(
            {
                "type": "http.response.body",
                "body": b'{"jsonrpc":"2.0","id":1,"result":{},"error":null}',
            }
        )
        await asyncio.sleep(0.3)
    finally:
        litellm.callbacks = []

    assert capture.failure_payloads == [], "non-error bodies must not be logged"


@pytest.mark.asyncio
async def test_mcp_protocol_wrapper_ignores_non_body_messages_and_empty():
    """The wrapper forwards http.response.start and empty bodies without
    attempting to log."""
    litellm.logging_callback_manager._reset_all_callbacks()
    capture = _ProtocolRejectionLogger()
    litellm.callbacks = [capture]

    sent = []

    async def fake_send(message):
        sent.append(message)

    try:
        wrapped = _wrap_send_for_protocol_error_logging(
            fake_send,
            request_method="tools/call",
            params={"name": "add"},
            request_id=1,
            user_api_key_auth=UserAPIKeyAuth(api_key="test", user_id="u1"),
            raw_headers={},
            start_time=datetime.now(),
        )
        await wrapped({"type": "http.response.start", "status": 200, "headers": []})
        await wrapped({"type": "http.response.body", "body": b""})
        await asyncio.sleep(0.2)
    finally:
        litellm.callbacks = []

    assert capture.failure_payloads == []
    assert len(sent) == 2  # both forwarded


@pytest.mark.asyncio
async def test_mcp_protocol_wrapper_handles_malformed_error_body():
    """An error-looking but invalid-JSON body trips the JSONDecodeError branch
    and is swallowed (no SLO, response still forwarded)."""
    litellm.logging_callback_manager._reset_all_callbacks()
    capture = _ProtocolRejectionLogger()
    litellm.callbacks = [capture]

    sent = []

    async def fake_send(message):
        sent.append(message)

    try:
        wrapped = _wrap_send_for_protocol_error_logging(
            fake_send,
            request_method="tools/call",
            params={"name": "add"},
            request_id=1,
            user_api_key_auth=UserAPIKeyAuth(api_key="test", user_id="u1"),
            raw_headers={},
            start_time=datetime.now(),
        )
        # passes the byte pre-check ('"error"' present, no '"result"') but is
        # not valid JSON.
        body = b'{"error": this-is-not-json'
        await wrapped({"type": "http.response.body", "body": body})
        await asyncio.sleep(0.2)
    finally:
        litellm.callbacks = []

    assert capture.failure_payloads == []
    assert any(m.get("body") == body for m in sent)


@pytest.mark.asyncio
async def test_mcp_log_protocol_rejection_logging_obj_none_is_noop(monkeypatch):
    """If function_setup yields no logging object, the helper returns without
    raising (covers the None-guard branch)."""
    litellm.logging_callback_manager._reset_all_callbacks()
    capture = _ProtocolRejectionLogger()
    litellm.callbacks = [capture]

    import litellm.proxy._experimental.mcp_server.server as mcp_server_mod

    def _fake_function_setup(*args, **kwargs):
        return None, {}

    monkeypatch.setattr(mcp_server_mod, "function_setup", _fake_function_setup)
    try:
        await _log_mcp_protocol_rejection(
            request_method="tools/call",
            params={"name": "x"},
            jsonrpc_error={"code": -32602, "message": "bad"},
            request_id=1,
            user_api_key_auth=UserAPIKeyAuth(api_key="test", user_id="u1"),
            raw_headers={},
            start_time=datetime.now(),
        )
        await asyncio.sleep(0.2)
    finally:
        litellm.callbacks = []

    assert capture.failure_payloads == []


@pytest.mark.asyncio
async def test_mcp_protocol_wrapper_logs_only_once_per_request():
    """already_logged short-circuit: a second error body on the same request
    does not produce a second SLO (covers the early-return guard)."""
    litellm.logging_callback_manager._reset_all_callbacks()
    capture = _ProtocolRejectionLogger()
    litellm.callbacks = [capture]

    async def fake_send(message):
        pass

    try:
        wrapped = _wrap_send_for_protocol_error_logging(
            fake_send,
            request_method="tools/call",
            params={"name": "add"},
            request_id=1,
            user_api_key_auth=UserAPIKeyAuth(api_key="test", user_id="u1"),
            raw_headers={},
            start_time=datetime.now(),
        )
        await wrapped({"type": "http.response.body", "body": _jsonrpc_error_body(-32602)})
        await wrapped({"type": "http.response.body", "body": _jsonrpc_error_body(-32601)})
        await asyncio.sleep(1)
    finally:
        litellm.callbacks = []

    non_none = [p for p in capture.failure_payloads if isinstance(p, dict)]
    assert len(non_none) == 1, f"exactly one SLO expected, got {len(non_none)}"


@pytest.mark.asyncio
async def test_mcp_protocol_wrapper_swallows_unexpected_error(monkeypatch):
    """A non-JSON unexpected error raised inside the detection block is caught
    by the broad guard and never breaks the response (covers the BLE001
    branch)."""
    litellm.logging_callback_manager._reset_all_callbacks()
    litellm.callbacks = []

    import litellm.proxy._experimental.mcp_server.server as mcp_server_mod

    async def _boom(**kwargs):
        raise RuntimeError("unexpected")

    monkeypatch.setattr(mcp_server_mod, "_log_mcp_protocol_rejection", _boom)

    sent = []

    async def fake_send(message):
        sent.append(message)

    wrapped = _wrap_send_for_protocol_error_logging(
        fake_send,
        request_method="tools/call",
        params={"name": "add"},
        request_id=1,
        user_api_key_auth=UserAPIKeyAuth(api_key="test", user_id="u1"),
        raw_headers={},
        start_time=datetime.now(),
    )
    body = _jsonrpc_error_body(-32602)
    # Must not raise despite _log_mcp_protocol_rejection blowing up.
    await wrapped({"type": "http.response.body", "body": body})
    assert any(m.get("body") == body for m in sent)


def test_parse_jsonrpc_request_for_logging_single():
    method, params, rid = _parse_jsonrpc_request_for_logging(
        b'{"jsonrpc":"2.0","method":"tools/call","id":7,"params":{"name":"x"}}'
    )
    assert method == "tools/call"
    assert params == {"name": "x"}
    assert rid == 7


def test_parse_jsonrpc_request_for_logging_non_string_method_skipped():
    # method present but not a string -> not loggable as a request.
    method, params, rid = _parse_jsonrpc_request_for_logging(b'{"jsonrpc":"2.0","method":123,"id":1}')
    assert method is None
    assert params == {}
    assert rid is None


def test_parse_jsonrpc_request_for_logging_non_dict_params():
    # params present but not a dict -> params stays empty, method still parsed.
    method, params, rid = _parse_jsonrpc_request_for_logging(
        b'{"jsonrpc":"2.0","method":"tools/call","id":1,"params":[1,2]}'
    )
    assert method == "tools/call"
    assert params == {}


def test_parse_jsonrpc_request_for_logging_missing_jsonrpc_version():
    method, params, rid = _parse_jsonrpc_request_for_logging(b'{"method":"tools/call","id":1}')
    assert method is None


def test_parse_jsonrpc_request_for_logging_batch():
    method, params, rid = _parse_jsonrpc_request_for_logging(b'[{"jsonrpc":"2.0","method":"tools/call","id":1}]')
    assert method == "batch"
    assert params == {}
    assert rid is None


def test_parse_jsonrpc_request_for_logging_truncated_or_non_json():
    # Truncated/oversized peek or plain non-JSON -> no attribution.
    assert _parse_jsonrpc_request_for_logging(b'{"jsonrpc":"2.0","meth') == (
        None,
        {},
        None,
    )
    assert _parse_jsonrpc_request_for_logging(b"not json at all") == (None, {}, None)
    # A bare JSON scalar (neither dict nor list).
    assert _parse_jsonrpc_request_for_logging(b'"a string"') == (None, {}, None)
