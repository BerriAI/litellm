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
        "_should_use_litellm_mcp_gateway",
        staticmethod(lambda t: True),
    )
    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "_parse_mcp_tools",
        staticmethod(lambda t: (t, [])),
    )
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
        "_execute_tool_calls",
        mock_execute,
    )
    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "_create_follow_up_messages_for_chat",
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
        "_execute_tool_calls",
        mock_execute,
    )
    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "_create_follow_up_messages_for_chat",
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
    - mcp_list_tools should be in the first chunk yielded to the client
    - mcp_tool_calls and mcp_call_results should be in the final chunk of the follow-up response
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

        finish_reasons = [
            chunk.choices[0].finish_reason
            for chunk in all_chunks
            if chunk.choices and chunk.choices[0].finish_reason is not None
        ]
        assert finish_reasons == ["stop"], f"Client stream must end with a single stop. Got: {finish_reasons}"

        first_chunk = all_chunks[0]
        first_provider_fields = getattr(
            first_chunk.choices[0].delta, "provider_specific_fields", None
        )
        assert (
            first_provider_fields is not None
        ), "First chunk should have provider_specific_fields"
        assert (
            "mcp_list_tools" in first_provider_fields
        ), "First chunk should have mcp_list_tools"

        final_chunk = all_chunks[-1]
        assert final_chunk.choices[0].finish_reason == "stop"
        final_provider_fields = getattr(
            final_chunk.choices[0].delta, "provider_specific_fields", None
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
    result = await LiteLLM_Proxy_MCP_Handler._execute_tool_calls(
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


def _create_mcp_stream_chunk(content, finish_reason=None, tool_calls=None):
    from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices

    return ModelResponseStream(
        id="test-stream",
        model="test-model",
        created=1234567890,
        object="chat.completion.chunk",
        choices=[
            StreamingChoices(
                index=0,
                delta=Delta(content=content, role="assistant", tool_calls=tool_calls),
                finish_reason=finish_reason,
            )
        ],
    )


def _make_mock_stream_class(stream_chunks):
    from unittest.mock import MagicMock

    from litellm.utils import CustomStreamWrapper

    logging_obj = MagicMock()
    logging_obj.model_call_details = {}

    class MockStream(CustomStreamWrapper):
        def __init__(self):
            super().__init__(
                completion_stream=None,
                model="test-model",
                logging_obj=logging_obj,
            )
            self.chunks = stream_chunks
            self._index = 0

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self._index < len(self.chunks):
                chunk = self.chunks[self._index]
                self._index += 1
                return chunk
            raise StopAsyncIteration

    return MockStream


def _patch_mcp_auto_exec_scaffolding(monkeypatch, tools, openai_tools, tool_calls, tool_results):
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


@pytest.mark.asyncio
async def test_acompletion_with_mcp_streaming_suppresses_intermediate_tool_call_turn(monkeypatch):
    """
    Regression test for https://github.com/BerriAI/litellm/issues/31910:
    when the proxy auto-executes MCP tools with stream=True, the intermediate
    tool-call turn (raw tool_call deltas + finish_reason "tool_calls") must not
    leak into the client stream. The client should see a single assistant
    message ending with exactly one terminal finish_reason.
    """
    from litellm.types.utils import ChatCompletionDeltaToolCall, Function

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

    initial_chunks = [
        _create_mcp_stream_chunk("Let me check. "),
        _create_mcp_stream_chunk(
            None,
            tool_calls=[
                ChatCompletionDeltaToolCall(
                    id="call-1",
                    type="function",
                    function=Function(name="local_search", arguments="{}"),
                    index=0,
                )
            ],
        ),
        _create_mcp_stream_chunk("", finish_reason="tool_calls"),
    ]
    follow_up_chunks = [
        _create_mcp_stream_chunk("Hello"),
        _create_mcp_stream_chunk(" world", finish_reason="stop"),
    ]

    InitialStream = _make_mock_stream_class(initial_chunks)
    FollowUpStream = _make_mock_stream_class(follow_up_chunks)

    async def mock_acompletion(**kwargs):
        messages = kwargs.get("messages", [])
        is_follow_up = any(
            isinstance(msg, dict) and msg.get("role") == "tool" for msg in messages
        )
        return FollowUpStream() if is_follow_up else InitialStream()

    mock_acompletion_func = AsyncMock(side_effect=mock_acompletion)
    _patch_mcp_auto_exec_scaffolding(monkeypatch, tools, openai_tools, tool_calls, tool_results)

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

        all_chunks = []
        async for chunk in result:
            all_chunks.append(chunk)

    assert all(
        not getattr(chunk.choices[0].delta, "tool_calls", None) for chunk in all_chunks if chunk.choices
    ), f"tool_call deltas must not leak into the client stream. Got: {all_chunks}"

    finish_reasons = [
        chunk.choices[0].finish_reason
        for chunk in all_chunks
        if chunk.choices and chunk.choices[0].finish_reason is not None
    ]
    assert finish_reasons == ["stop"], f"Expected a single terminal stop. Got: {finish_reasons}"
    assert all_chunks[-1].choices[0].finish_reason == "stop"

    content = "".join(
        chunk.choices[0].delta.content or "" for chunk in all_chunks if chunk.choices and chunk.choices[0].delta
    )
    assert content == "Let me check. Hello world"


@pytest.mark.asyncio
async def test_acompletion_with_mcp_streaming_flushes_tool_call_turn_when_no_follow_up(monkeypatch):
    """
    When tool execution produces no results (so no follow-up stream is created),
    the held tool-call chunks and the finish_reason "tool_calls" chunk must be
    flushed so the client still receives a complete, terminated stream.
    """
    from litellm.types.utils import ChatCompletionDeltaToolCall, Function

    tools = [{"type": "mcp", "server_url": "litellm_proxy/mcp/local"}]
    openai_tools = [{"type": "function", "function": {"name": "local_search"}}]
    tool_calls = [
        {
            "id": "call-1",
            "type": "function",
            "function": {"name": "local_search", "arguments": "{}"},
        }
    ]

    initial_chunks = [
        _create_mcp_stream_chunk(
            None,
            tool_calls=[
                ChatCompletionDeltaToolCall(
                    id="call-1",
                    type="function",
                    function=Function(name="local_search", arguments="{}"),
                    index=0,
                )
            ],
        ),
        _create_mcp_stream_chunk("", finish_reason="tool_calls"),
    ]

    InitialStream = _make_mock_stream_class(initial_chunks)

    async def mock_acompletion(**kwargs):
        return InitialStream()

    mock_acompletion_func = AsyncMock(side_effect=mock_acompletion)
    _patch_mcp_auto_exec_scaffolding(monkeypatch, tools, openai_tools, tool_calls, tool_results=[])

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

        all_chunks = []
        async for chunk in result:
            all_chunks.append(chunk)

    assert any(
        getattr(chunk.choices[0].delta, "tool_calls", None) for chunk in all_chunks if chunk.choices
    ), "Held tool-call chunks must be flushed when no follow-up stream is created"
    assert all_chunks[-1].choices[0].finish_reason == "tool_calls"


@pytest.mark.asyncio
async def test_acompletion_with_mcp_streaming_no_duplicate_chunk_on_abrupt_termination(monkeypatch):
    """
    Regression test for a duplicate-chunk bug in the abrupt-termination fallback:
    when the initial stream ends via StopAsyncIteration without a finish_reason
    chunk and the last collected chunk is a held tool-call delta, that chunk is
    both in held_tool_call_chunks and collected_chunks[-1]. It must be yielded
    to the client exactly once.
    """
    from litellm.types.utils import ChatCompletionDeltaToolCall, Function

    tools = [{"type": "mcp", "server_url": "litellm_proxy/mcp/local"}]
    openai_tools = [{"type": "function", "function": {"name": "local_search"}}]
    tool_calls = [
        {
            "id": "call-1",
            "type": "function",
            "function": {"name": "local_search", "arguments": "{}"},
        }
    ]

    initial_chunks = [
        _create_mcp_stream_chunk("partial answer "),
        _create_mcp_stream_chunk(
            None,
            tool_calls=[
                ChatCompletionDeltaToolCall(
                    id="call-1",
                    type="function",
                    function=Function(name="local_search", arguments="{}"),
                    index=0,
                )
            ],
        ),
    ]

    InitialStream = _make_mock_stream_class(initial_chunks)

    async def mock_acompletion(**kwargs):
        return InitialStream()

    mock_acompletion_func = AsyncMock(side_effect=mock_acompletion)
    _patch_mcp_auto_exec_scaffolding(monkeypatch, tools, openai_tools, tool_calls, tool_results=[])

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

        all_chunks = []
        async for chunk in result:
            all_chunks.append(chunk)

    tool_call_chunk_count = sum(
        1 for chunk in all_chunks if chunk.choices and getattr(chunk.choices[0].delta, "tool_calls", None)
    )
    assert tool_call_chunk_count == 1, (
        f"The held tool-call chunk must be yielded exactly once on abrupt termination. Got chunks: {all_chunks}"
    )
    assert len(all_chunks) == len(set(id(chunk) for chunk in all_chunks)), "No chunk object may be yielded twice"


@pytest.mark.asyncio
async def test_acompletion_with_mcp_streaming_abrupt_termination_with_follow_up_suppresses_tool_turn(monkeypatch):
    """
    When the initial stream ends via StopAsyncIteration without ever emitting a
    finish_reason chunk but tool execution still succeeds, the held tool-call
    chunks must be suppressed and the follow-up answer streamed, same as the
    clean-termination path.
    """
    from litellm.types.utils import ChatCompletionDeltaToolCall, Function

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

    initial_chunks = [
        _create_mcp_stream_chunk(
            None,
            tool_calls=[
                ChatCompletionDeltaToolCall(
                    id="call-1",
                    type="function",
                    function=Function(name="local_search", arguments="{}"),
                    index=0,
                )
            ],
        ),
    ]
    follow_up_chunks = [
        _create_mcp_stream_chunk("Hello"),
        _create_mcp_stream_chunk(" world", finish_reason="stop"),
    ]

    InitialStream = _make_mock_stream_class(initial_chunks)
    FollowUpStream = _make_mock_stream_class(follow_up_chunks)

    async def mock_acompletion(**kwargs):
        messages = kwargs.get("messages", [])
        is_follow_up = any(
            isinstance(msg, dict) and msg.get("role") == "tool" for msg in messages
        )
        return FollowUpStream() if is_follow_up else InitialStream()

    mock_acompletion_func = AsyncMock(side_effect=mock_acompletion)
    _patch_mcp_auto_exec_scaffolding(monkeypatch, tools, openai_tools, tool_calls, tool_results)

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

        all_chunks = []
        async for chunk in result:
            all_chunks.append(chunk)

    assert all(
        not getattr(chunk.choices[0].delta, "tool_calls", None) for chunk in all_chunks if chunk.choices
    ), f"tool_call deltas must not leak even when the initial stream ends abruptly. Got: {all_chunks}"
    finish_reasons = [
        chunk.choices[0].finish_reason
        for chunk in all_chunks
        if chunk.choices and chunk.choices[0].finish_reason is not None
    ]
    assert finish_reasons == ["stop"], f"Expected a single terminal stop. Got: {finish_reasons}"
    content = "".join(
        chunk.choices[0].delta.content or "" for chunk in all_chunks if chunk.choices and chunk.choices[0].delta
    )
    assert content == "Hello world"


@pytest.mark.asyncio
async def test_acompletion_with_mcp_streaming_empty_initial_stream_terminates_cleanly(monkeypatch):
    tools = [{"type": "mcp", "server_url": "litellm_proxy/mcp/local"}]
    openai_tools = [{"type": "function", "function": {"name": "local_search"}}]

    InitialStream = _make_mock_stream_class([])

    async def mock_acompletion(**kwargs):
        return InitialStream()

    mock_acompletion_func = AsyncMock(side_effect=mock_acompletion)
    _patch_mcp_auto_exec_scaffolding(monkeypatch, tools, openai_tools, tool_calls=[], tool_results=[])

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

        all_chunks = []
        async for chunk in result:
            all_chunks.append(chunk)

    assert all_chunks == [], "An empty initial stream must terminate cleanly with no chunks"
