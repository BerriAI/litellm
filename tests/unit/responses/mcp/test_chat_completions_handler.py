import asyncio
import json
import sys
import types
from collections.abc import Mapping, Sequence
from datetime import datetime
from types import MappingProxyType
from typing import Final, cast
from unittest.mock import AsyncMock, patch

import pytest
import respx
from httpx import Response
from mcp.types import Tool as MCPTool

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from litellm.responses.mcp import chat_completions_handler
from litellm.responses.mcp.chat_completions_handler import (
    acompletion_with_mcp,
)
from litellm.responses.mcp.litellm_proxy_mcp_handler import (
    LiteLLM_Proxy_MCP_Handler,
)
from litellm.responses.utils import ResponsesAPIRequestUtils
from litellm.types.utils import ModelResponse, StandardLoggingPayload


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
    from unittest.mock import MagicMock

    from litellm.types.utils import (
        ChatCompletionDeltaToolCall,
        Delta,
        Function,
        ModelResponseStream,
        StreamingChoices,
    )
    from litellm.utils import CustomStreamWrapper

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
    from litellm.litellm_core_utils.litellm_logging import Logging
    from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices
    from litellm.utils import CustomStreamWrapper

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
    from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices
    from litellm.utils import CustomStreamWrapper

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
    from litellm.types.utils import (
        ChatCompletionDeltaToolCall,
        Delta,
        Function,
        ModelResponseStream,
        StreamingChoices,
    )
    from litellm.utils import CustomStreamWrapper

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
        "litellm.proxy.mcp_server.mcp_server_manager.global_mcp_server_manager.call_tool",
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
    from litellm.proxy.mcp_server.mcp_server_manager import global_mcp_server_manager

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
    from mcp.types import Tool

    from litellm.caching.caching import DualCache
    from litellm.exceptions import GuardrailRaisedException
    from litellm.integrations.custom_guardrail import CustomGuardrail
    from litellm.proxy import proxy_server
    from litellm.proxy._types import LiteLLM_ObjectPermissionTable, UserAPIKeyAuth
    from litellm.proxy.mcp_server import mcp_server_manager, server, tool_registry
    from litellm.proxy.mcp_server.faults.list_outcomes import AggregateToolListing
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
_CONTEXT_WINDOW_ERROR: Final = MappingProxyType(
    {
        "error": MappingProxyType(
            {
                "message": "Your input exceeds the context window of this model.",
                "type": "invalid_request_error",
                "param": "messages",
                "code": "context_length_exceeded",
            }
        )
    }
)


class _ChatRoundRecorder(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.logged: tuple[StandardLoggingPayload, ...] = ()
        self.failed: tuple[StandardLoggingPayload, ...] = ()

    async def async_log_success_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        self.logged = (*self.logged, cast(StandardLoggingPayload, kwargs["standard_logging_object"]))

    async def async_log_failure_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        self.failed = (*self.failed, cast(StandardLoggingPayload, kwargs["standard_logging_object"]))


def _chat_round_cost(body: Mapping[str, object]) -> float:
    return litellm.completion_cost(
        completion_response=ModelResponse(**json.loads(json.dumps(body, default=dict))),
        model="gpt-5.6",
        custom_llm_provider="openai",
    )


async def _settled() -> None:
    await asyncio.sleep(0)
    await asyncio.wait_for(GLOBAL_LOGGING_WORKER.flush(), timeout=10)


async def _auto_execute_chat_echo(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch, rounds: Sequence[Response], call_id: str
) -> tuple[_ChatRoundRecorder, ModelResponse | Exception]:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setenv("DISABLE_AIOHTTP_TRANSPORT", "True")
    litellm.in_memory_llm_clients_cache.flush_cache()
    respx_mock.post("https://mcp-chat-rounds.example.com/v1/chat/completions").mock(side_effect=list(rounds))
    recorder: Final = _ChatRoundRecorder()
    logging_obj: Final = Logging(
        model="openai/gpt-5.6",
        messages=[{"role": "user", "content": "Call echo with PING"}],
        stream=False,
        call_type="acompletion",
        start_time=datetime(2026, 1, 1),
        litellm_call_id=call_id,
        function_id=call_id,
        dynamic_async_success_callbacks=[recorder],
        dynamic_async_failure_callbacks=[recorder],
    )
    echo_tool: Final = MCPTool.model_validate(
        {"name": "echo", "inputSchema": {"type": "object", "properties": {"value": {"type": "string"}}}},
        by_name=False,
    )
    with (
        patch.object(
            LiteLLM_Proxy_MCP_Handler,
            "process_mcp_tools_without_openai_transform",
            AsyncMock(return_value=([echo_tool], {"echo": "demo"})),
        ),
        patch.object(
            LiteLLM_Proxy_MCP_Handler,
            "execute_tool_calls",
            AsyncMock(return_value=[{"tool_call_id": "call_echo", "result": "PING", "name": "echo"}]),
        ),
    ):
        try:
            response: Final = await litellm.acompletion(
                model="openai/gpt-5.6",
                messages=[{"role": "user", "content": "Call echo with PING"}],
                tools=[{"type": "mcp", "server_url": "litellm_proxy/mcp/demo", "require_approval": "never"}],
                api_key="sk-mcp-chat-rounds",
                api_base="https://mcp-chat-rounds.example.com/v1",
                litellm_logging_obj=logging_obj,
            )
        except Exception as error:
            await _settled()
            return recorder, error
    await _settled()
    assert isinstance(response, ModelResponse)
    return recorder, response


def _ok(body: Mapping[str, object]) -> Response:
    return Response(200, content=json.dumps(body, default=dict))


@pytest.mark.asyncio
async def test_non_stream_auto_execute_logs_one_row_billing_both_rounds_and_matching_the_client_cost(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
):
    recorder, response = await _auto_execute_chat_echo(
        respx_mock, monkeypatch, (_ok(_CHAT_ROUND_ONE), _ok(_CHAT_ROUND_TWO)), "mcp-chat-rounds-both"
    )

    assert isinstance(response, ModelResponse)
    assert response.choices[0].message.content == "PING"
    both_rounds: Final = _chat_round_cost(_CHAT_ROUND_ONE) + _chat_round_cost(_CHAT_ROUND_TWO)
    assert len(recorder.logged) == 1
    row: Final = recorder.logged[0]
    assert row["id"] == response.id
    assert (row["prompt_tokens"], row["completion_tokens"]) == (71 + 149, 19 + 5)
    assert row["metadata"]["usage_object"]["prompt_tokens_details"]["cached_tokens"] == 32 + 64
    assert row["response_cost"] == pytest.approx(both_rounds)
    assert row["cost_breakdown"]["total_cost"] == pytest.approx(both_rounds)
    assert response.hidden_params["response_cost"] == pytest.approx(both_rounds)
    assert (response.usage.prompt_tokens, response.usage.completion_tokens) == (71 + 149, 19 + 5)
    assert recorder.failed == ()


@pytest.mark.asyncio
async def test_non_stream_auto_execute_without_a_tool_call_logs_the_one_round_it_made(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
):
    recorder, response = await _auto_execute_chat_echo(
        respx_mock, monkeypatch, (_ok(_CHAT_ROUND_TWO),), "mcp-chat-rounds-no-tool-call"
    )

    assert isinstance(response, ModelResponse)
    assert len(recorder.logged) == 1
    assert recorder.logged[0]["id"] == response.id
    assert (recorder.logged[0]["prompt_tokens"], recorder.logged[0]["completion_tokens"]) == (149, 5)
    assert recorder.logged[0]["response_cost"] == pytest.approx(_chat_round_cost(_CHAT_ROUND_TWO))
    assert response.hidden_params["response_cost"] == pytest.approx(recorder.logged[0]["response_cost"])


@pytest.mark.asyncio
async def test_non_stream_auto_execute_bills_the_paid_first_round_when_the_follow_up_fails(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
):
    recorder, outcome = await _auto_execute_chat_echo(
        respx_mock,
        monkeypatch,
        (_ok(_CHAT_ROUND_ONE), Response(400, content=json.dumps(_CONTEXT_WINDOW_ERROR, default=dict))),
        "mcp-chat-rounds-follow-up-fails",
    )

    assert isinstance(outcome, litellm.BadRequestError)
    assert recorder.logged == ()
    assert len(recorder.failed) == 1
    assert (recorder.failed[0]["prompt_tokens"], recorder.failed[0]["completion_tokens"]) == (71, 19)
    assert recorder.failed[0]["response_cost"] == pytest.approx(_chat_round_cost(_CHAT_ROUND_ONE))
    assert recorder.failed[0]["response_cost"] > 0
