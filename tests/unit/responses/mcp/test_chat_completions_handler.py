import json
import sys
import types
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from types import MappingProxyType, SimpleNamespace
from typing import Final

import pytest
import respx
from httpx import Response
from mcp.types import CallToolResult, TextContent
from mcp.types import Tool as MCPTool
from unittest.mock import AsyncMock, MagicMock, patch

import litellm
from litellm.proxy._experimental.mcp_server.faults.list_outcomes import AggregateToolListing
from litellm.types.mcp_server.mcp_server_manager import MCPServer
from litellm.types.utils import ModelResponse, ModelResponseStream
from litellm.utils import CustomStreamWrapper

from litellm.responses.mcp import chat_completions_handler
from litellm.responses.mcp.chat_completions_handler import (
    acompletion_with_mcp,
)
from litellm.responses.mcp.litellm_proxy_mcp_handler import (
    LiteLLM_Proxy_MCP_Handler,
)
from litellm.responses.utils import ResponsesAPIRequestUtils


@pytest.mark.asyncio
async def test_acompletion_with_mcp_returns_normal_completion_without_tools(
    monkeypatch,
):
    mock_acompletion = AsyncMock(return_value="normal_response")

    with patch("litellm.acompletion", mock_acompletion):
        result = await acompletion_with_mcp(
            model="test-model",
            messages=[],
            tools=None,
        )

    assert result == "normal_response"
    mock_acompletion.assert_awaited_once()


@pytest.mark.asyncio
async def test_acompletion_with_mcp_without_auto_execution_calls_model(monkeypatch):
    tools = [{"type": "function", "function": {"name": "tool"}}]
    mock_acompletion = AsyncMock(return_value="ok")

    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "should_use_litellm_mcp_gateway",
        staticmethod(lambda tools: True),
    )
    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "_parse_mcp_tools",
        staticmethod(lambda tools: (tools, [])),
    )

    async def mock_process(**_):
        return ([], {})

    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "process_mcp_tools_without_openai_transform",
        mock_process,
    )
    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "transform_mcp_tools_to_openai",
        staticmethod(lambda *_, **__: ["openai-tool"]),
    )
    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "should_auto_execute_tools",
        staticmethod(lambda **_: False),
    )
    captured_secret_fields = {}

    def mock_extract(**kwargs):
        captured_secret_fields["value"] = kwargs.get("secret_fields")
        return (None, None, None, None)

    monkeypatch.setattr(
        ResponsesAPIRequestUtils,
        "extract_mcp_headers_from_request",
        staticmethod(mock_extract),
    )

    with patch("litellm.acompletion", mock_acompletion):
        result = await acompletion_with_mcp(
            model="test-model",
            messages=[],
            tools=tools,
            secret_fields={"api_key": "value"},
        )

    assert result == "ok"
    mock_acompletion.assert_awaited_once()
    assert mock_acompletion.await_args is not None
    kwargs = mock_acompletion.await_args.kwargs
    assert kwargs.get("_skip_mcp_handler") is True
    assert kwargs.get("tools") == ["openai-tool"]
    assert captured_secret_fields["value"] == {"api_key": "value"}


@pytest.mark.asyncio
async def test_acompletion_with_mcp_passes_mcp_server_auth_headers_to_process_tools(
    monkeypatch,
):
    """
    Test that MCP auth headers extracted from secret_fields (e.g. x-mcp-linear_config-authorization)
    are passed to _process_mcp_tools_without_openai_transform for dynamic auth when fetching tools.
    """
    tools = [{"type": "mcp", "server_url": "litellm_proxy"}]
    mock_acompletion = AsyncMock(return_value="ok")

    captured_process_kwargs = {}

    async def mock_process(**kwargs):
        captured_process_kwargs.update(kwargs)
        return ([], {})

    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "should_use_litellm_mcp_gateway",
        staticmethod(lambda t: True),
    )
    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "_parse_mcp_tools",
        staticmethod(lambda t: (t, [])),
    )
    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "process_mcp_tools_without_openai_transform",
        mock_process,
    )
    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "transform_mcp_tools_to_openai",
        staticmethod(lambda *_, **__: ["openai-tool"]),
    )
    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "should_auto_execute_tools",
        staticmethod(lambda **_: False),
    )

    # secret_fields with raw_headers containing MCP auth - extract_mcp_headers_from_request
    # will parse these and pass to _process_mcp_tools_without_openai_transform
    secret_fields = {
        "raw_headers": {
            "x-mcp-linear_config-authorization": "Bearer linear-token",
        },
    }

    with patch("litellm.acompletion", mock_acompletion):
        await acompletion_with_mcp(
            model="test-model",
            messages=[],
            tools=tools,
            secret_fields=secret_fields,
        )

    assert captured_process_kwargs["raw_headers"] == secret_fields["raw_headers"]
    assert "mcp_server_auth_headers" in captured_process_kwargs
    mcp_server_auth_headers = captured_process_kwargs["mcp_server_auth_headers"]
    assert mcp_server_auth_headers is not None
    assert "linear_config" in mcp_server_auth_headers
    assert (
        mcp_server_auth_headers["linear_config"]["Authorization"]
        == "Bearer linear-token"
    )


@pytest.mark.asyncio
async def test_acompletion_with_mcp_auto_exec_performs_follow_up(monkeypatch):
    from litellm.utils import CustomStreamWrapper
    from litellm.types.utils import (
        ModelResponseStream,
        StreamingChoices,
        Delta,
        ChatCompletionDeltaToolCall,
        Function,
    )
    from unittest.mock import MagicMock

    tools = [{"type": "function", "function": {"name": "tool"}}]

    # Create mock streaming chunks for initial response
    def create_chunk(content, finish_reason=None, tool_calls=None):
        return ModelResponseStream(
            id="test-stream",
            model="test",
            created=1234567890,
            object="chat.completion.chunk",
            choices=[
                StreamingChoices(
                    index=0,
                    delta=Delta(
                        content=content,
                        role="assistant",
                        tool_calls=tool_calls,
                    ),
                    finish_reason=finish_reason,
                )
            ],
        )

    initial_chunks = [
        create_chunk(
            "",
            finish_reason="tool_calls",
            tool_calls=[
                ChatCompletionDeltaToolCall(
                    id="call-1",
                    type="function",
                    function=Function(name="tool", arguments="{}"),
                    index=0,
                )
            ],
        ),
    ]

    follow_up_chunks = [
        create_chunk("Hello"),
        create_chunk(" world", finish_reason="stop"),
    ]

    logging_obj = MagicMock()
    logging_obj.model_call_details = {}

    class InitialStreamingResponse(CustomStreamWrapper):
        def __init__(self):
            super().__init__(
                completion_stream=None,
                model="test",
                logging_obj=logging_obj,
            )
            self.chunks = initial_chunks
            self._index = 0

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self._index < len(self.chunks):
                chunk = self.chunks[self._index]
                self._index += 1
                return chunk
            raise StopAsyncIteration

    class FollowUpStreamingResponse(CustomStreamWrapper):
        def __init__(self):
            super().__init__(
                completion_stream=None,
                model="test",
                logging_obj=logging_obj,
            )
            self.chunks = follow_up_chunks
            self._index = 0

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self._index < len(self.chunks):
                chunk = self.chunks[self._index]
                self._index += 1
                return chunk
            raise StopAsyncIteration

    async def mock_acompletion(**kwargs):
        if kwargs.get("stream", False):
            messages = kwargs.get("messages", [])
            is_follow_up = any(
                msg.get("role") == "tool"
                or (isinstance(msg, dict) and "tool_call_id" in str(msg))
                for msg in messages
            )
            if is_follow_up:
                return FollowUpStreamingResponse()
            else:
                return InitialStreamingResponse()
        # Non-streaming should not happen
        return ModelResponse(
            id="1",
            model="test",
            choices=[],
            created=0,
            object="chat.completion",
        )

    mock_acompletion_func = AsyncMock(side_effect=mock_acompletion)

    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "should_use_litellm_mcp_gateway",
        staticmethod(lambda tools: True),
    )
    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "_parse_mcp_tools",
        staticmethod(lambda tools: (tools, [])),
    )

    async def mock_process(**_):
        return (tools, {"tool": "server"})

    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "process_mcp_tools_without_openai_transform",
        mock_process,
    )
    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "transform_mcp_tools_to_openai",
        staticmethod(lambda *_, **__: tools),
    )
    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "should_auto_execute_tools",
        staticmethod(lambda **_: True),
    )
    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "extract_tool_calls_from_chat_response",
        staticmethod(
            lambda **_: [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "tool", "arguments": "{}"},
                }
            ]
        ),
    )

    async def mock_execute(**_):
        return [{"tool_call_id": "call-1", "result": "executed"}]

    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "execute_tool_calls",
        mock_execute,
    )
    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "create_follow_up_messages_for_chat",
        staticmethod(
            lambda **_: [
                {"role": "user", "content": "hello"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "tool", "arguments": "{}"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call-1",
                    "name": "tool",
                    "content": "executed",
                },
            ]
        ),
    )
    monkeypatch.setattr(
        ResponsesAPIRequestUtils,
        "extract_mcp_headers_from_request",
        staticmethod(lambda **_: (None, None, None, None)),
    )

    # Patch litellm.acompletion at module level to catch function-level imports
    with (
        patch("litellm.acompletion", mock_acompletion_func),
        patch.object(
            chat_completions_handler,
            "litellm_acompletion",
            mock_acompletion_func,
            create=True,
        ),
    ):
        result = await acompletion_with_mcp(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hello"}],
            tools=tools,
            stream=True,
        )

        # Consume the stream to trigger the iterator and follow-up call
        # The initial stream has one chunk with finish_reason="tool_calls"
        # which will trigger tool execution and follow-up call
        chunks = []
        async for chunk in result:
            chunks.append(chunk)
            # After consuming the initial chunk, the follow-up call should be made
            # Break after first chunk since that's when follow-up is triggered
            break

    # With new implementation, first call should be streaming
    assert mock_acompletion_func.await_count >= 2
    first_call = mock_acompletion_func.await_args_list[0].kwargs
    # First call should be streaming in new implementation
    assert first_call["stream"] is True
    # Find the follow-up call (should have tool role messages)
    follow_up_call = None
    for call in mock_acompletion_func.await_args_list:
        messages = call.kwargs.get("messages", [])
        if messages and any(
            msg.get("role") == "tool" for msg in messages if isinstance(msg, dict)
        ):
            follow_up_call = call.kwargs
            break
    assert follow_up_call is not None, "Should have a follow-up call"
    assert follow_up_call["stream"] is True


@pytest.mark.asyncio
async def test_acompletion_with_mcp_adds_metadata_to_streaming(monkeypatch):
    """
    Test that acompletion_with_mcp adds MCP metadata to CustomStreamWrapper
    and it appears in the final chunk's delta.provider_specific_fields.
    """
    from litellm.utils import CustomStreamWrapper
    from litellm.types.utils import ModelResponseStream, StreamingChoices, Delta
    from litellm.litellm_core_utils.litellm_logging import Logging

    tools = [{"type": "mcp", "server_url": "litellm_proxy/mcp/local"}]
    openai_tools = [{"type": "function", "function": {"name": "local_search"}}]
    tool_calls = [
        {
            "id": "call-1",
            "type": "function",
            "function": {"name": "local_search", "arguments": "{}"},
        }
    ]
    tool_results = [{"tool_call_id": "call-1", "result": "executed"}]

    # Create mock streaming chunks
    def create_chunk(content, finish_reason=None):
        return ModelResponseStream(
            id="test-stream",
            model="test-model",
            created=1234567890,
            object="chat.completion.chunk",
            choices=[
                StreamingChoices(
                    index=0,
                    delta=Delta(
                        content=content,
                        role="assistant",
                    ),
                    finish_reason=finish_reason,
                )
            ],
        )

    chunks = [
        create_chunk("Hello"),
        create_chunk(" world", finish_reason="stop"),  # Final chunk
    ]

    # Create a proper CustomStreamWrapper
    from unittest.mock import MagicMock

    logging_obj = MagicMock()
    logging_obj.model_call_details = {}

    class MockStreamingResponse(CustomStreamWrapper):
        def __init__(self):
            super().__init__(
                completion_stream=None,
                model="test-model",
                logging_obj=logging_obj,
            )
            self.chunks = chunks
            self._index = 0
            self.sent_last_chunk = False

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self._index < len(self.chunks):
                chunk = self.chunks[self._index]
                self._index += 1
                if self._index == len(self.chunks):
                    self.sent_last_chunk = True
                # Add mcp_list_tools to first chunk if present
                if not self.sent_first_chunk:
                    chunk = self._add_mcp_list_tools_to_first_chunk(chunk)
                    self.sent_first_chunk = True
                return chunk
            raise StopAsyncIteration

    mock_acompletion = AsyncMock(return_value=MockStreamingResponse())

    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "should_use_litellm_mcp_gateway",
        staticmethod(lambda tools: True),
    )
    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "_parse_mcp_tools",
        staticmethod(lambda tools: (tools, [])),
    )

    async def mock_process(**_):
        return (tools, {"local_search": "local"})

    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "process_mcp_tools_without_openai_transform",
        mock_process,
    )
    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "transform_mcp_tools_to_openai",
        staticmethod(lambda *_, **__: openai_tools),
    )
    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "should_auto_execute_tools",
        staticmethod(lambda **_: False),
    )
    monkeypatch.setattr(
        ResponsesAPIRequestUtils,
        "extract_mcp_headers_from_request",
        staticmethod(lambda **_: (None, None, None, None)),
    )

    with patch("litellm.acompletion", mock_acompletion):
        result = await acompletion_with_mcp(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hello"}],
            tools=tools,
            stream=True,
        )

    # Verify result is CustomStreamWrapper
    assert isinstance(result, CustomStreamWrapper)

    # Verify _hidden_params contains mcp_metadata
    assert hasattr(result, "_hidden_params")
    assert "mcp_metadata" in result._hidden_params
    mcp_metadata = result._hidden_params["mcp_metadata"]
    assert "mcp_list_tools" in mcp_metadata
    assert mcp_metadata["mcp_list_tools"] == openai_tools

    # Consume the stream and check chunks
    all_chunks = []
    async for chunk in result:
        all_chunks.append(chunk)
    assert len(all_chunks) > 0

    # Verify mcp_list_tools is in the first chunk
    first_chunk = all_chunks[0]
    assert (
        hasattr(first_chunk, "choices") and first_chunk.choices
    ), "First chunk must have choices"
    choice = first_chunk.choices[0]
    assert hasattr(choice, "delta") and choice.delta, "First choice must have delta"
    provider_fields = getattr(choice.delta, "provider_specific_fields", None)
    assert (
        provider_fields is not None
    ), f"First chunk should have provider_specific_fields. Delta: {choice.delta}"
    assert (
        "mcp_list_tools" in provider_fields
    ), f"First chunk should have mcp_list_tools. Fields: {provider_fields}"
    assert provider_fields["mcp_list_tools"] == openai_tools


@pytest.mark.asyncio
async def test_acompletion_with_mcp_streaming_initial_call_is_streaming(monkeypatch):
    """
    Test that acompletion_with_mcp makes the initial LLM call with streaming=True
    when stream=True is requested, instead of making a non-streaming call first.
    """
    from litellm.utils import CustomStreamWrapper
    from litellm.types.utils import ModelResponseStream, StreamingChoices, Delta

    tools = [{"type": "mcp", "server_url": "litellm_proxy/mcp/local"}]
    openai_tools = [{"type": "function", "function": {"name": "local_search"}}]

    # Create mock streaming chunks
    def create_chunk(content, finish_reason=None):
        return ModelResponseStream(
            id="test-stream",
            model="test-model",
            created=1234567890,
            object="chat.completion.chunk",
            choices=[
                StreamingChoices(
                    index=0,
                    delta=Delta(
                        content=content,
                        role="assistant",
                    ),
                    finish_reason=finish_reason,
                )
            ],
        )

    chunks = [
        create_chunk("", finish_reason="tool_calls"),  # Final chunk with tool_calls
    ]

    # Create a proper CustomStreamWrapper
    from unittest.mock import MagicMock

    logging_obj = MagicMock()
    logging_obj.model_call_details = {}

    class MockStreamingResponse(CustomStreamWrapper):
        def __init__(self):
            super().__init__(
                completion_stream=None,
                model="test-model",
                logging_obj=logging_obj,
            )
            self.chunks = chunks
            self._index = 0

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self._index < len(self.chunks):
                chunk = self.chunks[self._index]
                self._index += 1
                return chunk
            raise StopAsyncIteration

    mock_acompletion = AsyncMock(return_value=MockStreamingResponse())

    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "should_use_litellm_mcp_gateway",
        staticmethod(lambda tools: True),
    )
    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "_parse_mcp_tools",
        staticmethod(lambda tools: (tools, [])),
    )

    async def mock_process(**_):
        return (tools, {"local_search": "local"})

    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "process_mcp_tools_without_openai_transform",
        mock_process,
    )
    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "transform_mcp_tools_to_openai",
        staticmethod(lambda *_, **__: openai_tools),
    )
    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "should_auto_execute_tools",
        staticmethod(lambda **_: True),
    )
    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "extract_tool_calls_from_chat_response",
        staticmethod(
            lambda **_: [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "local_search", "arguments": "{}"},
                }
            ]
        ),
    )

    async def mock_execute(**_):
        return [{"tool_call_id": "call-1", "result": "executed"}]

    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "execute_tool_calls",
        mock_execute,
    )
    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "create_follow_up_messages_for_chat",
        staticmethod(
            lambda **_: [
                {"role": "user", "content": "hello"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "local_search", "arguments": "{}"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call-1",
                    "name": "local_search",
                    "content": "executed",
                },
            ]
        ),
    )
    monkeypatch.setattr(
        ResponsesAPIRequestUtils,
        "extract_mcp_headers_from_request",
        staticmethod(lambda **_: (None, None, None, None)),
    )

    # Patch litellm.acompletion at module level to catch function-level imports
    with (
        patch("litellm.acompletion", mock_acompletion),
        patch.object(
            chat_completions_handler,
            "litellm_acompletion",
            mock_acompletion,
            create=True,
        ),
    ):
        result = await acompletion_with_mcp(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hello"}],
            tools=tools,
            stream=True,
        )

    # Verify result is CustomStreamWrapper
    assert isinstance(result, CustomStreamWrapper)

    # Verify that the first call was made with stream=True
    assert mock_acompletion.await_count >= 1
    first_call = mock_acompletion.await_args_list[0].kwargs
    assert (
        first_call["stream"] is True
    ), "First call should be streaming with new implementation"


@pytest.mark.asyncio
async def test_acompletion_with_mcp_streaming_metadata_in_correct_chunks(monkeypatch):
    """
    Test that MCP metadata is added to the correct chunks:
    - mcp_list_tools should be in the first chunk
    - mcp_tool_calls and mcp_call_results should be in the final chunk of initial response
    """
    from litellm.utils import CustomStreamWrapper
    from litellm.types.utils import (
        ModelResponseStream,
        StreamingChoices,
        Delta,
        ChatCompletionDeltaToolCall,
        Function,
    )

    tools = [{"type": "mcp", "server_url": "litellm_proxy/mcp/local"}]
    openai_tools = [{"type": "function", "function": {"name": "local_search"}}]
    tool_calls = [
        {
            "id": "call-1",
            "type": "function",
            "function": {"name": "local_search", "arguments": "{}"},
        }
    ]
    tool_results = [{"tool_call_id": "call-1", "result": "executed"}]

    # Create mock streaming chunks
    def create_chunk(content, finish_reason=None, tool_calls=None):
        return ModelResponseStream(
            id="test-stream",
            model="test-model",
            created=1234567890,
            object="chat.completion.chunk",
            choices=[
                StreamingChoices(
                    index=0,
                    delta=Delta(
                        content=content,
                        role="assistant",
                        tool_calls=tool_calls,
                    ),
                    finish_reason=finish_reason,
                )
            ],
        )

    initial_chunks = [
        create_chunk(
            "",
            finish_reason="tool_calls",
            tool_calls=[
                ChatCompletionDeltaToolCall(
                    id="call-1",
                    type="function",
                    function=Function(name="local_search", arguments="{}"),
                    index=0,
                )
            ],
        ),  # Final chunk with tool_calls
    ]

    follow_up_chunks = [
        create_chunk("Hello"),
        create_chunk(" world", finish_reason="stop"),
    ]

    # Create a proper CustomStreamWrapper
    from unittest.mock import MagicMock

    logging_obj = MagicMock()
    logging_obj.model_call_details = {}

    class InitialStreamingResponse(CustomStreamWrapper):
        def __init__(self):
            super().__init__(
                completion_stream=None,
                model="test-model",
                logging_obj=logging_obj,
            )
            self.chunks = initial_chunks
            self._index = 0

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self._index < len(self.chunks):
                chunk = self.chunks[self._index]
                self._index += 1
                return chunk
            raise StopAsyncIteration

    class FollowUpStreamingResponse(CustomStreamWrapper):
        def __init__(self):
            super().__init__(
                completion_stream=None,
                model="test-model",
                logging_obj=logging_obj,
            )
            self.chunks = follow_up_chunks
            self._index = 0

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self._index < len(self.chunks):
                chunk = self.chunks[self._index]
                self._index += 1
                return chunk
            raise StopAsyncIteration

    acompletion_calls = []

    async def mock_acompletion(**kwargs):
        acompletion_calls.append(kwargs)
        if kwargs.get("stream", False):
            messages = kwargs.get("messages", [])
            is_follow_up = any(
                msg.get("role") == "tool"
                or (isinstance(msg, dict) and "tool_call_id" in str(msg))
                for msg in messages
            )
            if is_follow_up:
                return FollowUpStreamingResponse()
            else:
                return InitialStreamingResponse()
        pytest.fail("Non-streaming call should not happen with new implementation")

    mock_acompletion_func = AsyncMock(side_effect=mock_acompletion)

    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "should_use_litellm_mcp_gateway",
        staticmethod(lambda tools: True),
    )
    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "_parse_mcp_tools",
        staticmethod(lambda tools: (tools, [])),
    )

    async def mock_process(**_):
        return (tools, {"local_search": "local"})

    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "process_mcp_tools_without_openai_transform",
        mock_process,
    )
    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "transform_mcp_tools_to_openai",
        staticmethod(lambda *_, **__: openai_tools),
    )
    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "should_auto_execute_tools",
        staticmethod(lambda **_: True),
    )
    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "extract_tool_calls_from_chat_response",
        staticmethod(lambda **_: tool_calls),
    )

    async def mock_execute(**_):
        return tool_results

    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "execute_tool_calls",
        mock_execute,
    )
    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "create_follow_up_messages_for_chat",
        staticmethod(
            lambda **_: [
                {"role": "user", "content": "hello"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "local_search", "arguments": "{}"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call-1",
                    "name": "local_search",
                    "content": "executed",
                },
            ]
        ),
    )
    monkeypatch.setattr(
        ResponsesAPIRequestUtils,
        "extract_mcp_headers_from_request",
        staticmethod(lambda **_: (None, None, None, None)),
    )

    # Patch litellm.acompletion at module level to catch function-level imports
    with (
        patch("litellm.acompletion", mock_acompletion_func),
        patch.object(
            chat_completions_handler,
            "litellm_acompletion",
            side_effect=mock_acompletion,
            create=True,
        ),
    ):
        result = await acompletion_with_mcp(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hello"}],
            tools=tools,
            stream=True,
        )

        # Verify result is CustomStreamWrapper
        assert isinstance(result, CustomStreamWrapper)

        # Consume the stream and verify metadata placement
        # NOTE: Stream consumption must be inside the patch context to avoid real API calls
        all_chunks = []
        async for chunk in result:
            all_chunks.append(chunk)
        assert len(all_chunks) > 0

        # Find first chunk and final chunk from initial response
        # mcp_list_tools is added to the first chunk (all_chunks[0])
        first_chunk = all_chunks[0] if all_chunks else None
        initial_final_chunk = None

        for chunk in all_chunks:
            if hasattr(chunk, "choices") and chunk.choices:
                choice = chunk.choices[0]
                if (
                    hasattr(choice, "finish_reason")
                    and choice.finish_reason == "tool_calls"
                ):
                    initial_final_chunk = chunk

        assert first_chunk is not None, "Should have a first chunk"
        assert (
            initial_final_chunk is not None
        ), "Should have a final chunk from initial response"

        # Verify mcp_list_tools is in the first chunk
        assert (
            hasattr(first_chunk, "choices") and first_chunk.choices
        ), "First chunk must have choices"
        first_choice = first_chunk.choices[0]
        assert (
            hasattr(first_choice, "delta") and first_choice.delta
        ), "First choice must have delta"
        first_provider_fields = getattr(
            first_choice.delta, "provider_specific_fields", None
        )
        assert (
            first_provider_fields is not None
        ), "First chunk should have provider_specific_fields"
        assert (
            "mcp_list_tools" in first_provider_fields
        ), "First chunk should have mcp_list_tools"

        # Verify mcp_tool_calls and mcp_call_results are in the final chunk of initial response
        assert (
            hasattr(initial_final_chunk, "choices") and initial_final_chunk.choices
        ), "Final chunk must have choices"
        final_choice = initial_final_chunk.choices[0]
        assert (
            hasattr(final_choice, "delta") and final_choice.delta
        ), "Final choice must have delta"
        final_provider_fields = getattr(
            final_choice.delta, "provider_specific_fields", None
        )
        assert (
            final_provider_fields is not None
        ), "Final chunk should have provider_specific_fields"
        assert "mcp_tool_calls" in final_provider_fields, "Should have mcp_tool_calls"
        assert (
            "mcp_call_results" in final_provider_fields
        ), "Should have mcp_call_results"


@pytest.mark.asyncio
async def test_execute_tool_calls_sets_proxy_server_request_arguments(monkeypatch):
    """
    Test that _execute_tool_calls sets proxy_server_request with arguments in logging_request_data
    so that arguments are available in callbacks.
    """
    import importlib
    from unittest.mock import MagicMock

    # Capture the kwargs passed to function_setup
    captured_kwargs = {}

    def mock_function_setup(original_function, rules_obj, start_time, **kwargs):
        captured_kwargs.update(kwargs)
        # Return a mock logging object
        logging_obj = MagicMock()
        logging_obj.model_call_details = {}
        logging_obj.pre_call = MagicMock()
        logging_obj.post_call = MagicMock()
        logging_obj.async_post_mcp_tool_call_hook = AsyncMock()
        logging_obj.async_success_handler = AsyncMock()
        return logging_obj, kwargs

    # Mock the MCP server manager
    mock_result = MagicMock()
    mock_result.content = [MagicMock(text="test result")]

    async def mock_call_tool(**kwargs):
        return mock_result

    # NOTE: avoid monkeypatch string path here because `litellm.responses` is also
    # exported as a function on the top-level `litellm` package, which can confuse
    # pytest's dotted-path resolver.
    mcp_handler_module = importlib.import_module(
        "litellm.responses.mcp.litellm_proxy_mcp_handler"
    )
    monkeypatch.setattr(mcp_handler_module, "function_setup", mock_function_setup)
    monkeypatch.setattr(
        "litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager.call_tool",
        mock_call_tool,
    )

    # Create test data
    tool_calls = [
        {
            "id": "call-1",
            "type": "function",
            "function": {
                "name": "test_tool",
                "arguments": '{"param1": "value1", "param2": 123}',
            },
        }
    ]
    tool_server_map = {"test_tool": "test_server"}
    user_api_key_auth = MagicMock()
    user_api_key_auth.api_key = "test_key"

    # Call _execute_tool_calls
    result = await LiteLLM_Proxy_MCP_Handler.execute_tool_calls(
        tool_server_map=tool_server_map,
        tool_calls=tool_calls,
        user_api_key_auth=user_api_key_auth,
    )

    # Verify that proxy_server_request was set with arguments
    assert (
        "proxy_server_request" in captured_kwargs
    ), "proxy_server_request should be in logging_request_data"
    proxy_server_request = captured_kwargs["proxy_server_request"]
    assert "body" in proxy_server_request, "proxy_server_request should have body"
    assert "name" in proxy_server_request["body"], "body should have name"
    assert "arguments" in proxy_server_request["body"], "body should have arguments"
    assert proxy_server_request["body"]["name"] == "test_tool", "name should match"
    assert proxy_server_request["body"]["arguments"] == {
        "param1": "value1",
        "param2": 123,
    }, "arguments should be parsed correctly"


@pytest.mark.asyncio
async def test_acompletion_with_mcp_streaming_drain_error_does_not_drop_final_chunk(monkeypatch):
    """
    Regression test: after yielding the final chunk, MCPStreamingIterator drains
    the inner CustomStreamWrapper to fire end-of-stream spend logging. If the
    inner stream raises a non-StopAsyncIteration error during that drain (e.g.
    a transient APIError on the trailing usage chunk), the error must not
    escape __anext__ and drop the already-assembled final chunk.
    """
    from unittest.mock import MagicMock

    from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices
    from litellm.utils import CustomStreamWrapper

    tools = [{"type": "mcp", "server_url": "litellm_proxy/mcp/local"}]
    openai_tools = [{"type": "function", "function": {"name": "local_search"}}]

    def create_chunk(content, finish_reason=None):
        return ModelResponseStream(
            id="test-stream",
            model="test-model",
            created=1234567890,
            object="chat.completion.chunk",
            choices=[
                StreamingChoices(
                    index=0,
                    delta=Delta(content=content, role="assistant"),
                    finish_reason=finish_reason,
                )
            ],
        )

    chunks = [
        create_chunk("Hello"),
        create_chunk(" world", finish_reason="stop"),
    ]

    logging_obj = MagicMock()
    logging_obj.model_call_details = {}

    class DrainErrorStreamingResponse(CustomStreamWrapper):
        def __init__(self):
            super().__init__(
                completion_stream=None,
                model="test-model",
                logging_obj=logging_obj,
            )
            self.chunks = chunks
            self._index = 0

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self._index < len(self.chunks):
                chunk = self.chunks[self._index]
                self._index += 1
                return chunk
            if self._index == len(self.chunks):
                self._index += 1
                raise RuntimeError("connection dropped on trailing usage chunk")
            raise StopAsyncIteration

    mock_acompletion = AsyncMock(return_value=DrainErrorStreamingResponse())

    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "should_use_litellm_mcp_gateway",
        staticmethod(lambda tools: True),
    )
    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "_parse_mcp_tools",
        staticmethod(lambda tools: (tools, [])),
    )

    async def mock_process(**_):
        return (tools, {"local_search": "local"})

    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "process_mcp_tools_without_openai_transform",
        mock_process,
    )
    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "transform_mcp_tools_to_openai",
        staticmethod(lambda *_, **__: openai_tools),
    )
    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "should_auto_execute_tools",
        staticmethod(lambda **_: True),
    )
    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "extract_tool_calls_from_chat_response",
        staticmethod(lambda **_: []),
    )
    monkeypatch.setattr(
        ResponsesAPIRequestUtils,
        "extract_mcp_headers_from_request",
        staticmethod(lambda **_: (None, None, None, None)),
    )

    with patch("litellm.acompletion", mock_acompletion):
        result = await acompletion_with_mcp(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hello"}],
            tools=tools,
            stream=True,
        )

        all_chunks = []
        async for chunk in result:
            all_chunks.append(chunk)

    final_chunks = [
        chunk
        for chunk in all_chunks
        if chunk.choices and chunk.choices[0].finish_reason == "stop"
    ]
    assert len(final_chunks) == 1, f"Final chunk must survive a drain error. Got chunks: {all_chunks}"
    assert all_chunks[-1].choices[0].finish_reason == "stop"


@pytest.mark.asyncio
async def test_acompletion_with_mcp_streaming_drains_inner_stream_after_exhaustion(monkeypatch):
    from unittest.mock import MagicMock

    from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices
    from litellm.utils import CustomStreamWrapper

    tools = [{"type": "mcp", "server_url": "litellm_proxy/mcp/local"}]
    openai_tools = [{"type": "function", "function": {"name": "local_search"}}]

    def create_chunk(content):
        return ModelResponseStream(
            id="test-stream",
            model="test-model",
            created=1234567890,
            object="chat.completion.chunk",
            choices=[
                StreamingChoices(
                    index=0,
                    delta=Delta(content=content, role="assistant"),
                    finish_reason=None,
                )
            ],
        )

    chunks = [create_chunk("Hello"), create_chunk(" world")]
    logging_obj = MagicMock()
    logging_obj.model_call_details = {}

    class ExhaustingStreamingResponse(CustomStreamWrapper):
        def __init__(self):
            super().__init__(
                completion_stream=None,
                model="test-model",
                logging_obj=logging_obj,
            )
            self.chunks = chunks
            self._index = 0
            self.drained_after_exhaustion = False

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self._index < len(self.chunks):
                chunk = self.chunks[self._index]
                self._index += 1
                return chunk
            if self._index == len(self.chunks):
                self._index += 1
                raise StopAsyncIteration
            self.drained_after_exhaustion = True
            raise StopAsyncIteration

    initial_stream = ExhaustingStreamingResponse()
    mock_acompletion = AsyncMock(return_value=initial_stream)

    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "should_use_litellm_mcp_gateway",
        staticmethod(lambda tools: True),
    )
    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "_parse_mcp_tools",
        staticmethod(lambda tools: (tools, [])),
    )

    async def mock_process(**_):
        return (tools, {"local_search": "local"})

    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "process_mcp_tools_without_openai_transform",
        mock_process,
    )
    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "transform_mcp_tools_to_openai",
        staticmethod(lambda *_, **__: openai_tools),
    )
    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "should_auto_execute_tools",
        staticmethod(lambda **_: True),
    )
    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "extract_tool_calls_from_chat_response",
        staticmethod(lambda **_: []),
    )
    monkeypatch.setattr(
        ResponsesAPIRequestUtils,
        "extract_mcp_headers_from_request",
        staticmethod(lambda **_: (None, None, None, None)),
    )

    with patch("litellm.acompletion", mock_acompletion):
        result = await acompletion_with_mcp(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hello"}],
            tools=tools,
            stream=True,
        )

        all_chunks = []
        async for chunk in result:
            all_chunks.append(chunk)

    assert len(all_chunks) == 3
    assert initial_stream.drained_after_exhaustion is True


@pytest.mark.asyncio
@respx.mock
async def test_acompletion_with_mcp_forwards_unserved_external_mcp_tool_to_the_provider(monkeypatch):
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import global_mcp_server_manager

    zapier_tool = {"type": "mcp", "server_label": "zapier", "server_url": "https://mcp.zapier.com/api/mcp/mcp"}
    monkeypatch.setitem(sys.modules, "litellm.proxy.proxy_server", types.SimpleNamespace(prisma_client=None))
    monkeypatch.setattr(global_mcp_server_manager, "get_registry", lambda: {})
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    provider = respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=Response(
            200,
            json={
                "id": "chatcmpl-zapier",
                "object": "chat.completion",
                "created": 0,
                "model": "gpt-4.1",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )
    )

    result = await acompletion_with_mcp(
        model="openai/gpt-4.1",
        messages=[{"role": "user", "content": "hello"}],
        tools=[zapier_tool],
        api_key="sk-test",
        acompletion=True,
    )

    assert isinstance(result, ModelResponse)
    assert result.id == "chatcmpl-zapier"
    assert json.loads(provider.calls.last.request.content)["tools"] == [zapier_tool]


@pytest.mark.asyncio
@pytest.mark.parametrize("selected", [False, True])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("selection_source", ["metadata", "litellm_metadata", "body"])
@pytest.mark.parametrize("logging_failure", [False, True])
async def test_request_selected_mcp_guardrail_blocks_before_upstream(monkeypatch, selected, stream, selection_source, logging_failure):
    from litellm.exceptions import GuardrailRaisedException
    from mcp.types import Tool
    from litellm.caching.caching import DualCache
    from litellm.integrations.custom_guardrail import CustomGuardrail
    from litellm.proxy import proxy_server
    from litellm.proxy._types import UserAPIKeyAuth, LiteLLM_ObjectPermissionTable
    from litellm.proxy._experimental.mcp_server import mcp_server_manager, server, tool_registry
    from litellm.proxy._experimental.mcp_server.faults.list_outcomes import AggregateToolListing
    from litellm.proxy.utils import ProxyLogging
    from litellm.types.guardrails import GuardrailEventHooks
    from litellm.types.mcp_server.mcp_server_manager import MCPServer

    class BlockSelected(CustomGuardrail):
        async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
            if self.should_run_guardrail(data, GuardrailEventHooks.pre_mcp_call):
                raise GuardrailRaisedException(message="request-selected MCP block", blocked_content=True)
            return data

    guardrail = BlockSelected(guardrail_name="block-all", event_hook="pre_mcp_call", default_on=False)
    monkeypatch.setattr(litellm, "callbacks", [guardrail])
    manager = mcp_server_manager.MCPServerManager()
    manager.registry = {"observer": MCPServer(
        server_id="observer", name="observer", server_name="observer", transport="http",
        url="https://observer.example/mcp", spec_path="observer.json", auth_type="none",
    )}
    manager.tool_name_to_mcp_server_name_mapping = {"observer-execute": "observer"}
    upstream = AsyncMock(return_value={"executed": True})
    registry = tool_registry.MCPToolRegistry()
    registry.register_tool("observer-execute", "Execute", {"type": "object"}, upstream)
    monkeypatch.setattr(tool_registry, "global_mcp_tool_registry", registry)
    monkeypatch.setattr(mcp_server_manager, "global_mcp_server_manager", manager)
    monkeypatch.setattr(proxy_server, "proxy_logging_obj", ProxyLogging(user_api_key_cache=DualCache()))
    monkeypatch.setattr(server, "_get_tools_from_mcp_servers", AsyncMock(return_value=AggregateToolListing(
        tools=[Tool(name="observer-execute", inputSchema={"type": "object"})], outcomes={}
    )))
    responses = [
        ModelResponse(choices=[{"message": {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call-1", "type": "function", "function": {"name": "observer-execute", "arguments": "{}"}}
        ]}, "finish_reason": "tool_calls"}]),
        ModelResponse(choices=[{"message": {"role": "assistant", "content": "done"}}]),
    ]
    if stream:
        from litellm.types.utils import ModelResponseStream
        responses = [
            await litellm.acompletion(
                model="openai/gpt-5", messages=[{"role": "user", "content": "execute"}], stream=True,
                mock_response=ModelResponseStream(choices=[{"index": 0, "delta": {
                    "role": "assistant", "content": None, "tool_calls": [{
                        "index": 0, "id": "call-1", "type": "function",
                        "function": {"name": "observer-execute", "arguments": "{}"},
                    }],
                }, "finish_reason": "tool_calls"}]),
            ),
            await litellm.acompletion(
                model="openai/gpt-5", messages=[{"role": "user", "content": "done"}],
                stream=True, mock_response="done",
            ),
        ]
    if logging_failure:
        from litellm.responses.mcp import litellm_proxy_mcp_handler
        def fail_logging(*args, **kwargs):
            raise RuntimeError("logging initialization failed")
        monkeypatch.setattr(litellm_proxy_mcp_handler, "function_setup", fail_logging)
    model_call = AsyncMock(side_effect=responses)
    monkeypatch.setattr(litellm, "acompletion", model_call)
    result = await acompletion_with_mcp(
        model="test-model", messages=[{"role": "user", "content": "execute"}],
        tools=[{"type": "mcp", "server_url": "litellm_proxy/observer", "require_approval": "never"}],
        stream=stream,
        user_api_key_auth=UserAPIKeyAuth(
            object_permission=LiteLLM_ObjectPermissionTable(object_permission_id="test", mcp_servers=["observer"])
        ),
        **({"guardrails": ["block-all"] if selected else []} if selection_source == "body" else {
            selection_source: {"guardrails": ["block-all"] if selected else []}
        }),
    )
    if stream:
        chunks = [chunk async for chunk in result]
        assert chunks
    assert model_call.await_count == 2
    assert upstream.await_count == (0 if selected else 1)
    tool_message = model_call.await_args.kwargs["messages"][-1]
    assert ("request-selected MCP block" in tool_message["content"]) is selected


_ECHO_ARGUMENTS: Final = '{"value": "PING"}'
_CHAT_ROUND_ONE: Final = MappingProxyType(
    {
        "id": "chatcmpl-round-one",
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-5.6",
        "choices": (
            MappingProxyType(
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": MappingProxyType(
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": (
                                MappingProxyType(
                                    {
                                        "id": "call_echo",
                                        "type": "function",
                                        "function": MappingProxyType({"name": "echo", "arguments": _ECHO_ARGUMENTS}),
                                    }
                                ),
                            ),
                        }
                    ),
                }
            ),
        ),
        "usage": MappingProxyType(
            {
                "prompt_tokens": 71,
                "completion_tokens": 19,
                "total_tokens": 90,
                "prompt_tokens_details": MappingProxyType({"cached_tokens": 32}),
            }
        ),
    }
)
_CHAT_ROUND_TWO: Final = MappingProxyType(
    {
        "id": "chatcmpl-round-two",
        "object": "chat.completion",
        "created": 2,
        "model": "gpt-5.6",
        "choices": (
            MappingProxyType(
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": MappingProxyType({"role": "assistant", "content": "PING"}),
                }
            ),
        ),
        "usage": MappingProxyType(
            {
                "prompt_tokens": 149,
                "completion_tokens": 5,
                "total_tokens": 154,
                "prompt_tokens_details": MappingProxyType({"cached_tokens": 64}),
            }
        ),
    }
)


def _serve_chat_rounds(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch, rounds: Sequence[Response]
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setenv("DISABLE_AIOHTTP_TRANSPORT", "True")
    litellm.in_memory_llm_clients_cache.flush_cache()
    respx_mock.post("https://mcp-chat-rounds.example.com/v1/chat/completions").mock(side_effect=list(rounds))


def _ok(body: Mapping[str, object]) -> Response:
    return Response(200, content=json.dumps(body, default=dict))


def _chat_stream_events(body: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    completed: Final = json.loads(json.dumps(body, default=dict))
    choice: Final = completed["choices"][0]
    message: Final = choice["message"]
    calls: Final = message.get("tool_calls") or []
    delta: Final = (
        {**message, "tool_calls": [{**call, "index": index} for index, call in enumerate(calls)]} if calls else message
    )
    head: Final = {"id": completed["id"], "object": "chat.completion.chunk", "created": 1, "model": completed["model"]}
    return (
        {**head, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
        {**head, "choices": [{"index": 0, "delta": {}, "finish_reason": choice["finish_reason"]}]},
        {**head, "choices": [], "usage": completed["usage"]},
    )


def _sse(events: Sequence[Mapping[str, object]]) -> bytes:
    return b"".join(f"data: {json.dumps(event)}\n\n".encode() for event in events)


def _streamed(body: Mapping[str, object]) -> Response:
    return Response(
        200,
        content=_sse(_chat_stream_events(body)) + b"data: [DONE]\n\n",
        headers={"content-type": "text/event-stream"},
    )


async def _read_all(stream: CustomStreamWrapper) -> tuple[ModelResponseStream, ...]:
    return tuple([chunk async for chunk in stream])


_MCP_ECHO_TOOL: Final = MCPTool.model_validate(
    {"name": "echo", "inputSchema": {"type": "object", "properties": {"value": {"type": "string"}}}},
    by_name=False,
)
_FLAT_ALLOWED_ECHO: Final = {"type": "allowed_tools", "mode": "auto", "tools": [{"type": "function", "name": "echo"}]}


_ALLOWED_ONLY_ANOTHER_TOOL: Final = {
    "type": "allowed_tools",
    "mode": "auto",
    "tools": [{"type": "function", "name": "another_tool"}],
}


_DEMO_MCP_SERVER: Final = MCPServer(
    server_id="demo",
    name="demo",
    server_name="demo",
    transport="http",
    url="https://demo.example/mcp",
    auth_type="none",
)


def _demo_server_named(name: str) -> MCPServer | None:
    return _DEMO_MCP_SERVER if name.lower() == "demo" else None


def _served_by_a_recording_mcp_server(
    monkeypatch: pytest.MonkeyPatch,
    call_tool: AsyncMock,
    listed: MCPTool = _MCP_ECHO_TOOL,
    catalog: SimpleNamespace = SimpleNamespace(operation=nullcontext),
) -> None:
    monkeypatch.setattr(
        "litellm.proxy._experimental.mcp_server.server._get_tools_from_mcp_servers",
        AsyncMock(return_value=AggregateToolListing(tools=[listed], outcomes={})),
    )
    monkeypatch.setitem(
        sys.modules, "litellm.proxy.proxy_server", SimpleNamespace(proxy_logging_obj=None, prisma_client=None)
    )
    monkeypatch.setattr(
        "litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager",
        SimpleNamespace(
            catalog=catalog,
            get_registry=MagicMock(return_value={}),
            call_tool=call_tool,
            get_mcp_server_from_tool_name=MagicMock(return_value=None),
            get_mcp_server_by_name=MagicMock(side_effect=_demo_server_named),
            get_allowed_mcp_servers=AsyncMock(return_value=[_DEMO_MCP_SERVER.server_id]),
            get_mcp_servers_from_ids=MagicMock(return_value=[_DEMO_MCP_SERVER]),
        ),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_an_mcp_tool_the_choice_excludes_never_runs_when_the_model_ignored_the_choice(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch, stream: bool
):
    _serve_chat_rounds(respx_mock, monkeypatch, (_streamed(_CHAT_ROUND_ONE) if stream else _ok(_CHAT_ROUND_ONE),))
    call_tool: Final = AsyncMock()
    _served_by_a_recording_mcp_server(monkeypatch, call_tool)

    response: Final = await litellm.acompletion(
        model="openai/gpt-5.6",
        messages=[{"role": "user", "content": "Call echo with PING"}],
        tools=[{"type": "mcp", "server_url": "litellm_proxy/mcp/demo", "require_approval": "never"}],
        tool_choice=_ALLOWED_ONLY_ANOTHER_TOOL,
        api_key="sk-mcp-chat-rounds",
        api_base="https://mcp-chat-rounds.example.com/v1",
        stream=stream,
    )
    chunks: Final = await _read_all(response) if stream else ()

    handed_back: Final = (
        tuple(call.function.name for chunk in chunks for call in chunk.choices[0].delta.tool_calls or ())
        if stream
        else tuple(call.function.name for call in response.choices[0].message.tool_calls)
    )
    assert handed_back == ("echo",)
    call_tool.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_a_required_choice_asks_the_follow_up_in_auto_mode_for_the_same_tools_once_its_mcp_call_ran(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch, stream: bool
):
    rounds: Final = (_CHAT_ROUND_ONE, _CHAT_ROUND_TWO)
    _serve_chat_rounds(respx_mock, monkeypatch, tuple(_streamed(body) if stream else _ok(body) for body in rounds))
    call_tool: Final = AsyncMock(return_value=CallToolResult(content=[TextContent(type="text", text="PING")]))
    _served_by_a_recording_mcp_server(monkeypatch, call_tool)

    response: Final = await litellm.acompletion(
        model="openai/gpt-5.6",
        messages=[{"role": "user", "content": "Call echo with PING"}],
        tools=[{"type": "mcp", "server_url": "litellm_proxy/mcp/demo", "require_approval": "never"}],
        tool_choice={"type": "allowed_tools", "mode": "required", "tools": [{"type": "function", "name": "echo"}]},
        api_key="sk-mcp-chat-rounds",
        api_base="https://mcp-chat-rounds.example.com/v1",
        stream=stream,
    )
    if stream:
        await _read_all(response)

    first, follow_up = (json.loads(call.request.content)["tool_choice"] for call in respx_mock.calls)
    assert call_tool.await_count == 1
    assert first["allowed_tools"]["mode"] == "required"
    assert follow_up == {**first, "allowed_tools": {**first["allowed_tools"], "mode": "auto"}}


def _anthropic_call_to_echo(name: str = "echo") -> Response:
    return Response(
        200,
        json={
            "id": "msg_echo",
            "type": "message",
            "role": "assistant",
            "model": "claude-haiku-4-5",
            "content": [{"type": "tool_use", "id": "toolu_echo", "name": name, "input": {"value": "PING"}}],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        },
    )


def _anthropic_answer() -> Response:
    return Response(
        200,
        json={
            "id": "msg_answer",
            "type": "message",
            "role": "assistant",
            "model": "claude-haiku-4-5",
            "content": [{"type": "text", "text": "PING back"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        },
    )


_WHOLE_DEMO_SERVER: Final = {"type": "allowed_tools", "mode": "auto", "tools": [{"type": "mcp", "server_label": "demo"}]}
_WHOLE_OTHER_SERVER: Final = {"type": "allowed_tools", "mode": "auto", "tools": [{"type": "mcp", "server_label": "other"}]}


async def _call_echo_on_anthropic_under_drop_params(
    respx_mock: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
    rounds: Sequence[Response],
    tool_choice: object,
    server_url: str = "litellm_proxy/mcp/demo",
) -> tuple[ModelResponse, respx.Route]:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setenv("DISABLE_AIOHTTP_TRANSPORT", "True")
    litellm.in_memory_llm_clients_cache.flush_cache()
    route: Final = respx_mock.post("https://api.anthropic.com/v1/messages").mock(side_effect=list(rounds))
    response: Final = await litellm.acompletion(
        model="anthropic/claude-haiku-4-5",
        messages=[{"role": "user", "content": "Call echo with PING"}],
        tools=[{"type": "mcp", "server_label": "demo", "server_url": server_url, "require_approval": "never"}],
        tool_choice=tool_choice,
        drop_params=True,
        api_key="sk-ant-fake",
    )
    return response, route


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_choice", [_ALLOWED_ONLY_ANOTHER_TOOL, _WHOLE_OTHER_SERVER])
async def test_a_dropped_choice_hands_back_an_mcp_call_it_excludes(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch, tool_choice: object
):
    call_tool: Final = AsyncMock()
    _served_by_a_recording_mcp_server(monkeypatch, call_tool)

    response, route = await _call_echo_on_anthropic_under_drop_params(
        respx_mock, monkeypatch, (_anthropic_call_to_echo(),), tool_choice
    )

    call_tool.assert_not_awaited()
    assert "tool_choice" not in json.loads(route.calls[0].request.content)
    assert [call.function.name for call in response.choices[0].message.tool_calls] == ["echo"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool_choice", "server_url"),
    [
        (_FLAT_ALLOWED_ECHO, "litellm_proxy/mcp/demo"),
        (_WHOLE_DEMO_SERVER, "litellm_proxy/mcp/demo"),
        (_WHOLE_DEMO_SERVER, "litellm_proxy"),
    ],
)
async def test_a_dropped_choice_still_runs_an_mcp_call_it_lists(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch, tool_choice: object, server_url: str
):
    call_tool: Final = AsyncMock(return_value=CallToolResult(content=[TextContent(type="text", text="PING")]))
    _served_by_a_recording_mcp_server(monkeypatch, call_tool)

    response, _route = await _call_echo_on_anthropic_under_drop_params(
        respx_mock, monkeypatch, (_anthropic_call_to_echo(), _anthropic_answer()), tool_choice, server_url
    )

    assert call_tool.await_count == 1
    assert response.choices[0].message.content == "PING back"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tool_choice",
    [
        {"type": "allowed_tools", "mode": "auto", "tools": []},
        {"type": "allowed_tools", "mode": "auto", "tools": [{"type": "mcp", "name": ["echo"]}]},
    ],
)
async def test_a_malformed_choice_under_drop_params_runs_an_mcp_call_as_if_never_sent(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch, tool_choice: object
):
    call_tool: Final = AsyncMock(return_value=CallToolResult(content=[TextContent(type="text", text="PING")]))
    _served_by_a_recording_mcp_server(monkeypatch, call_tool)

    response, route = await _call_echo_on_anthropic_under_drop_params(
        respx_mock, monkeypatch, (_anthropic_call_to_echo(), _anthropic_answer()), tool_choice
    )

    assert call_tool.await_count == 1
    assert response.choices[0].message.content == "PING back"
    assert all("tool_choice" not in json.loads(call.request.content) for call in route.calls)


_DEMO_ECHO_TOOL: Final = MCPTool.model_validate(
    {"name": "demo-echo", "inputSchema": {"type": "object", "properties": {"value": {"type": "string"}}}},
    by_name=False,
)
_DEMO_IN_AN_ACCESS_GROUP: Final = _DEMO_MCP_SERVER.model_copy(update={"access_groups": ["demo_group"]})


def _choice_of(*tools: Mapping[str, object]) -> Mapping[str, object]:
    return {"type": "allowed_tools", "mode": "auto", "tools": list(tools)}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool_choice", "server_url", "listed"),
    [
        (_choice_of({"type": "mcp", "server_label": "demo", "name": "echo"}), "litellm_proxy/mcp/demo", _DEMO_ECHO_TOOL),
        (
            _choice_of({"type": "mcp", "server_label": "demo", "name": "demo-echo"}),
            "litellm_proxy/mcp/demo",
            _DEMO_ECHO_TOOL,
        ),
        (_WHOLE_DEMO_SERVER, "litellm_proxy/mcp/DEMO", _MCP_ECHO_TOOL),
        (_WHOLE_DEMO_SERVER, "litellm_proxy/mcp/demo_group", _MCP_ECHO_TOOL),
    ],
)
async def test_an_mcp_entry_allows_the_tool_the_gateway_resolves_for_its_label_and_name(
    respx_mock: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
    tool_choice: object,
    server_url: str,
    listed: MCPTool,
):
    call_tool: Final = AsyncMock(return_value=CallToolResult(content=[TextContent(type="text", text="PING")]))
    _served_by_a_recording_mcp_server(
        monkeypatch,
        call_tool,
        listed,
        SimpleNamespace(
            operation=nullcontext,
            current=lambda: SimpleNamespace(servers={_DEMO_MCP_SERVER.server_id: _DEMO_IN_AN_ACCESS_GROUP}),
        ),
    )

    response, _route = await _call_echo_on_anthropic_under_drop_params(
        respx_mock, monkeypatch, (_anthropic_call_to_echo(listed.name), _anthropic_answer()), tool_choice, server_url
    )

    assert call_tool.await_count == 1
    assert response.choices[0].message.content == "PING back"


@pytest.mark.asyncio
async def test_a_named_mcp_entry_allows_no_other_tool_on_its_server(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
):
    call_tool: Final = AsyncMock()
    _served_by_a_recording_mcp_server(monkeypatch, call_tool, _DEMO_ECHO_TOOL)

    response, _route = await _call_echo_on_anthropic_under_drop_params(
        respx_mock,
        monkeypatch,
        (_anthropic_call_to_echo("demo-echo"),),
        _choice_of({"type": "mcp", "server_label": "demo", "name": "reverse"}),
    )

    call_tool.assert_not_awaited()
    assert [call.function.name for call in response.choices[0].message.tool_calls] == ["demo-echo"]


async def _call_echo_on_openai_with_the_demo_server(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch, rounds: Sequence[Response], tool_choice: object
) -> tuple[object, ...]:
    _serve_chat_rounds(respx_mock, monkeypatch, rounds)
    await litellm.acompletion(
        model="openai/gpt-5.6",
        messages=[{"role": "user", "content": "Call echo with PING"}],
        tools=[
            {
                "type": "mcp",
                "server_label": "demo",
                "server_url": "litellm_proxy/mcp/demo",
                "require_approval": "never",
            },
            {"type": "function", "function": {"name": "get_time", "parameters": {"type": "object"}}},
        ],
        tool_choice=tool_choice,
        api_key="sk-mcp-chat-rounds",
        api_base="https://mcp-chat-rounds.example.com/v1",
    )
    return tuple(json.loads(call.request.content).get("tool_choice") for call in respx_mock.calls)


@pytest.mark.asyncio
async def test_the_provider_gets_the_converted_function_in_place_of_a_gateway_mcp_entry(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
):
    call_tool: Final = AsyncMock(return_value=CallToolResult(content=[TextContent(type="text", text="PING")]))
    _served_by_a_recording_mcp_server(monkeypatch, call_tool)

    sent: Final = await _call_echo_on_openai_with_the_demo_server(
        respx_mock,
        monkeypatch,
        (_ok(_CHAT_ROUND_ONE), _ok(_CHAT_ROUND_TWO)),
        _choice_of({"type": "mcp", "server_label": "demo"}, {"type": "function", "name": "get_time"}),
    )

    assert sent[0] == {
        "type": "allowed_tools",
        "allowed_tools": {
            "mode": "auto",
            "tools": [
                {"type": "function", "function": {"name": "get_time"}},
                {"type": "function", "function": {"name": "echo"}},
            ],
        },
    }
    assert call_tool.await_count == 1


@pytest.mark.asyncio
async def test_a_gateway_mcp_entry_that_allows_no_converted_tool_sends_none_and_runs_nothing(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
):
    call_tool: Final = AsyncMock()
    _served_by_a_recording_mcp_server(monkeypatch, call_tool)

    sent: Final = await _call_echo_on_openai_with_the_demo_server(
        respx_mock,
        monkeypatch,
        (_ok(_CHAT_ROUND_ONE),),
        _choice_of({"type": "mcp", "server_label": "demo", "name": "reverse"}),
    )

    assert sent == ("none",)
    call_tool.assert_not_awaited()
