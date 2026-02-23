import pytest
from unittest.mock import AsyncMock, patch

from litellm.types.utils import ModelResponse

from litellm.responses.mcp import chat_completions_handler
from litellm.responses.mcp.chat_completions_handler import (
    acompletion_with_mcp,
)
from litellm.responses.mcp.litellm_proxy_mcp_handler import (
    LiteLLM_Proxy_MCP_Handler,
)
from litellm.responses.utils import ResponsesAPIRequestUtils


@pytest.mark.asyncio
async def test_acompletion_with_mcp_returns_normal_completion_without_tools(monkeypatch):
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
        "_should_use_litellm_mcp_gateway",
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
        "_process_mcp_tools_without_openai_transform",
        mock_process,
    )
    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "_transform_mcp_tools_to_openai",
        staticmethod(lambda *_, **__: ["openai-tool"]),
    )
    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "_should_auto_execute_tools",
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
async def test_acompletion_with_mcp_auto_exec_performs_follow_up(monkeypatch):
    from litellm.utils import CustomStreamWrapper
    from litellm.types.utils import ModelResponseStream, StreamingChoices, Delta, ChatCompletionDeltaToolCall, Function
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
                msg.get("role") == "tool" or (isinstance(msg, dict) and "tool_call_id" in str(msg))
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
        "_should_use_litellm_mcp_gateway",
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
        "_process_mcp_tools_without_openai_transform",
        mock_process,
    )
    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "_transform_mcp_tools_to_openai",
        staticmethod(lambda *_, **__: tools),
    )
    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "_should_auto_execute_tools",
        staticmethod(lambda **_: True),
    )
    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "_extract_tool_calls_from_chat_response",
        staticmethod(lambda **_: [{"id": "call-1", "type": "function", "function": {"name": "tool", "arguments": "{}"}}]),
    )
    async def mock_execute(**_):
        return [{"tool_call_id": "call-1", "result": "executed"}]

    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "_execute_tool_calls",
        mock_execute,
    )
    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "_create_follow_up_messages_for_chat",
        staticmethod(lambda **_: [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "tool", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "call-1", "name": "tool", "content": "executed"}
        ]),
    )
    monkeypatch.setattr(
        ResponsesAPIRequestUtils,
        "extract_mcp_headers_from_request",
        staticmethod(lambda **_: (None, None, None, None)),
    )

    # Patch litellm.acompletion at module level to catch function-level imports
    with patch("litellm.acompletion", mock_acompletion_func), \
         patch.object(chat_completions_handler, "litellm_acompletion", mock_acompletion_func, create=True):
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
        if messages and any(msg.get("role") == "tool" for msg in messages if isinstance(msg, dict)):
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
    tool_calls = [{"id": "call-1", "type": "function", "function": {"name": "local_search", "arguments": "{}"}}]
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
        "_should_use_litellm_mcp_gateway",
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
        "_process_mcp_tools_without_openai_transform",
        mock_process,
    )
    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "_transform_mcp_tools_to_openai",
        staticmethod(lambda *_, **__: openai_tools),
    )
    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "_should_auto_execute_tools",
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
    assert hasattr(first_chunk, "choices") and first_chunk.choices, "First chunk must have choices"
    choice = first_chunk.choices[0]
    assert hasattr(choice, "delta") and choice.delta, "First choice must have delta"
    provider_fields = getattr(choice.delta, "provider_specific_fields", None)
    assert provider_fields is not None, f"First chunk should have provider_specific_fields. Delta: {choice.delta}"
    assert "mcp_list_tools" in provider_fields, f"First chunk should have mcp_list_tools. Fields: {provider_fields}"
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
        "_should_use_litellm_mcp_gateway",
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
        "_process_mcp_tools_without_openai_transform",
        mock_process,
    )
    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "_transform_mcp_tools_to_openai",
        staticmethod(lambda *_, **__: openai_tools),
    )
    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "_should_auto_execute_tools",
        staticmethod(lambda **_: True),
    )
    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "_extract_tool_calls_from_chat_response",
        staticmethod(lambda **_: [{"id": "call-1", "type": "function", "function": {"name": "local_search", "arguments": "{}"}}]),
    )
    async def mock_execute(**_):
        return [{"tool_call_id": "call-1", "result": "executed"}]

    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "_execute_tool_calls",
        mock_execute,
    )
    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "_create_follow_up_messages_for_chat",
        staticmethod(lambda **_: [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "local_search", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "call-1", "name": "local_search", "content": "executed"}
        ]),
    )
    monkeypatch.setattr(
        ResponsesAPIRequestUtils,
        "extract_mcp_headers_from_request",
        staticmethod(lambda **_: (None, None, None, None)),
    )

    # Patch litellm.acompletion at module level to catch function-level imports
    with patch("litellm.acompletion", mock_acompletion), \
         patch.object(chat_completions_handler, "litellm_acompletion", mock_acompletion, create=True):
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
    assert first_call["stream"] is True, "First call should be streaming with new implementation"


@pytest.mark.asyncio
async def test_acompletion_with_mcp_streaming_metadata_in_correct_chunks(monkeypatch):
    """
    Test that MCP metadata is added to the correct chunks:
    - mcp_list_tools should be in the first chunk
    - mcp_tool_calls and mcp_call_results should be in the final chunk of initial response
    """
    from litellm.utils import CustomStreamWrapper
    from litellm.types.utils import ModelResponseStream, StreamingChoices, Delta, ChatCompletionDeltaToolCall, Function

    tools = [{"type": "mcp", "server_url": "litellm_proxy/mcp/local"}]
    openai_tools = [{"type": "function", "function": {"name": "local_search"}}]
    tool_calls = [{"id": "call-1", "type": "function", "function": {"name": "local_search", "arguments": "{}"}}]
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
                msg.get("role") == "tool" or (isinstance(msg, dict) and "tool_call_id" in str(msg))
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
        "_should_use_litellm_mcp_gateway",
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
        "_process_mcp_tools_without_openai_transform",
        mock_process,
    )
    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "_transform_mcp_tools_to_openai",
        staticmethod(lambda *_, **__: openai_tools),
    )
    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "_should_auto_execute_tools",
        staticmethod(lambda **_: True),
    )
    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "_extract_tool_calls_from_chat_response",
        staticmethod(lambda **_: tool_calls),
    )
    async def mock_execute(**_):
        return tool_results

    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "_execute_tool_calls",
        mock_execute,
    )
    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "_create_follow_up_messages_for_chat",
        staticmethod(lambda **_: [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "local_search", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "call-1", "name": "local_search", "content": "executed"}
        ]),
    )
    monkeypatch.setattr(
        ResponsesAPIRequestUtils,
        "extract_mcp_headers_from_request",
        staticmethod(lambda **_: (None, None, None, None)),
    )

    # Patch litellm.acompletion at module level to catch function-level imports
    with patch("litellm.acompletion", mock_acompletion_func), \
         patch.object(chat_completions_handler, "litellm_acompletion", side_effect=mock_acompletion, create=True):
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
                if hasattr(choice, "finish_reason") and choice.finish_reason == "tool_calls":
                    initial_final_chunk = chunk

        assert first_chunk is not None, "Should have a first chunk"
        assert initial_final_chunk is not None, "Should have a final chunk from initial response"

        # Verify mcp_list_tools is in the first chunk
        assert hasattr(first_chunk, "choices") and first_chunk.choices, "First chunk must have choices"
        first_choice = first_chunk.choices[0]
        assert hasattr(first_choice, "delta") and first_choice.delta, "First choice must have delta"
        first_provider_fields = getattr(first_choice.delta, "provider_specific_fields", None)
        assert first_provider_fields is not None, "First chunk should have provider_specific_fields"
        assert "mcp_list_tools" in first_provider_fields, "First chunk should have mcp_list_tools"

        # Verify mcp_tool_calls and mcp_call_results are in the final chunk of initial response
        assert hasattr(initial_final_chunk, "choices") and initial_final_chunk.choices, "Final chunk must have choices"
        final_choice = initial_final_chunk.choices[0]
        assert hasattr(final_choice, "delta") and final_choice.delta, "Final choice must have delta"
        final_provider_fields = getattr(final_choice.delta, "provider_specific_fields", None)
        assert final_provider_fields is not None, "Final chunk should have provider_specific_fields"
        assert "mcp_tool_calls" in final_provider_fields, "Should have mcp_tool_calls"
        assert "mcp_call_results" in final_provider_fields, "Should have mcp_call_results"


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
    result = await LiteLLM_Proxy_MCP_Handler._execute_tool_calls(
        tool_server_map=tool_server_map,
        tool_calls=tool_calls,
        user_api_key_auth=user_api_key_auth,
    )
    
    # Verify that proxy_server_request was set with arguments
    assert "proxy_server_request" in captured_kwargs, "proxy_server_request should be in logging_request_data"
    proxy_server_request = captured_kwargs["proxy_server_request"]
    assert "body" in proxy_server_request, "proxy_server_request should have body"
    assert "name" in proxy_server_request["body"], "body should have name"
    assert "arguments" in proxy_server_request["body"], "body should have arguments"
    assert proxy_server_request["body"]["name"] == "test_tool", "name should match"
    assert proxy_server_request["body"]["arguments"] == {"param1": "value1", "param2": 123}, "arguments should be parsed correctly"
