import pytest

import litellm
from litellm.types.utils import ModelResponse


@pytest.mark.asyncio
async def test_acompletion_mcp_auto_exec(monkeypatch):
    from types import SimpleNamespace

    from litellm.responses.mcp.litellm_proxy_mcp_handler import (
        LiteLLM_Proxy_MCP_Handler,
    )
    from litellm.responses.utils import ResponsesAPIRequestUtils

    dummy_tool = SimpleNamespace(
        name="local_search",
        description="search",
        inputSchema={"type": "object", "properties": {}},
    )

    async def fake_process(user_api_key_auth, mcp_tools_with_litellm_proxy, **kwargs):
        return [dummy_tool], {"local_search": "local"}

    async def fake_execute(**kwargs):
        fake_execute.called = True  # type: ignore[attr-defined]
        tool_calls = kwargs.get("tool_calls") or []
        assert tool_calls, "tool calls should be present during auto execution"
        call_entry = tool_calls[0]
        call_id = call_entry.get("id") or call_entry.get("call_id") or "call"
        return [
            {
                "tool_call_id": call_id,
                "result": "executed",
                "name": call_entry.get("name", "local_search"),
            }
        ]

    fake_execute.called = False  # type: ignore[attr-defined]

    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "_process_mcp_tools_without_openai_transform",
        fake_process,
    )
    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "_execute_tool_calls",
        fake_execute,
    )
    monkeypatch.setattr(
        ResponsesAPIRequestUtils,
        "extract_mcp_headers_from_request",
        staticmethod(lambda secret_fields, tools: (None, None, None, None)),
    )

    response = await litellm.acompletion(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "hello"}],
        tools=[
            {
                "type": "mcp",
                "server_url": "litellm_proxy/mcp/local",
                "server_label": "local",
                "require_approval": "never",
            }
        ],
        mock_response="Final answer",
        mock_tool_calls=[
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": "local_search", "arguments": "{}"},
            }
        ],
    )

    assert isinstance(response, ModelResponse)
    assert response.choices[0].message.content == "Final answer"
    assert fake_execute.called is True  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_acompletion_mcp_respects_manual_approval(monkeypatch):
    from types import SimpleNamespace

    from litellm.responses.mcp.litellm_proxy_mcp_handler import (
        LiteLLM_Proxy_MCP_Handler,
    )
    from litellm.responses.utils import ResponsesAPIRequestUtils

    dummy_tool = SimpleNamespace(
        name="local_search",
        description="search",
        inputSchema={"type": "object", "properties": {}},
    )

    async def fake_process(user_api_key_auth, mcp_tools_with_litellm_proxy, **kwargs):
        return [dummy_tool], {"local_search": "local"}

    async def fake_execute(**kwargs):
        pytest.fail("auto execution should not run when approval is required")

    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "_process_mcp_tools_without_openai_transform",
        fake_process,
    )
    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "_execute_tool_calls",
        fake_execute,
    )
    monkeypatch.setattr(
        ResponsesAPIRequestUtils,
        "extract_mcp_headers_from_request",
        staticmethod(lambda secret_fields, tools: (None, None, None, None)),
    )

    response = await litellm.acompletion(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "hello"}],
        tools=[
            {
                "type": "mcp",
                "server_url": "litellm_proxy/mcp/local",
                "server_label": "local",
                "require_approval": "manual",
            }
        ],
        mock_response="Pending tool",
        mock_tool_calls=[
            {
                "id": "call-2",
                "type": "function",
                "function": {"name": "local_search", "arguments": "{}"},
            }
        ],
    )

    assert isinstance(response, ModelResponse)
    tool_calls = response.choices[0].message.tool_calls
    assert tool_calls is not None and len(tool_calls) == 1


@pytest.mark.asyncio
async def test_completion_mcp_with_streaming_no_timeout_error(monkeypatch):
    """
    Test that litellm.completion with stream=True and MCP tools does not raise
    RuntimeError: Timeout context manager should be used inside a task.
    
    This test ensures that the fix in ba43f742ab86d51b7da63077b85b39d0ac808d30
    prevents event loop nesting issues when using MCP tools with streaming.
    
    The fix changes completion() to return a coroutine from acompletion_with_mcp,
    which acompletion() then awaits, avoiding event loop nesting.
    """
    from types import SimpleNamespace
    from unittest.mock import patch

    from litellm.responses.mcp.litellm_proxy_mcp_handler import (
        LiteLLM_Proxy_MCP_Handler,
    )
    from litellm.responses.utils import ResponsesAPIRequestUtils
    from litellm.utils import CustomStreamWrapper

    dummy_tool = SimpleNamespace(
        name="local_search",
        description="search",
        inputSchema={"type": "object", "properties": {}},
    )

    async def fake_process(user_api_key_auth, mcp_tools_with_litellm_proxy, **kwargs):
        return [dummy_tool], {"local_search": "local"}

    async def fake_execute(**kwargs):
        fake_execute.called = True  # type: ignore[attr-defined]
        tool_calls = kwargs.get("tool_calls") or []
        assert tool_calls, "tool calls should be present during auto execution"
        call_entry = tool_calls[0]
        call_id = call_entry.get("id") or call_entry.get("call_id") or "call"
        return [
            {
                "tool_call_id": call_id,
                "result": "executed",
                "name": call_entry.get("name", "local_search"),
            }
        ]

    fake_execute.called = False  # type: ignore[attr-defined]

    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "_process_mcp_tools_without_openai_transform",
        fake_process,
    )
    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "_execute_tool_calls",
        fake_execute,
    )
    monkeypatch.setattr(
        ResponsesAPIRequestUtils,
        "extract_mcp_headers_from_request",
        staticmethod(lambda secret_fields, tools: (None, None, None, None)),
    )

    # Create a mock streaming response
    from unittest.mock import MagicMock, AsyncMock
    logging_obj = MagicMock()
    logging_obj.model_call_details = {}
    logging_obj.async_failure_handler = AsyncMock()
    
    class MockStreamingResponse(CustomStreamWrapper):
        def __init__(self):
            super().__init__(
                completion_stream=None,
                model="gpt-4o-mini",
                logging_obj=logging_obj,
            )
            self.chunks = [
                type('Chunk', (), {
                    'choices': [type('Choice', (), {
                        'delta': type('Delta', (), {
                            'content': 'Final'
                        })()
                    })()]
                })(),
                type('Chunk', (), {
                    'choices': [type('Choice', (), {
                        'delta': type('Delta', (), {
                            'content': ' answer'
                        })()
                    })()]
                })(),
            ]
            self._index = 0

        def __iter__(self):
            return self

        def __next__(self):
            if self._index < len(self.chunks):
                chunk = self.chunks[self._index]
                self._index += 1
                # Add mcp_list_tools to first chunk if present
                if not self.sent_first_chunk:
                    chunk = self._add_mcp_list_tools_to_first_chunk(chunk)
                    self.sent_first_chunk = True
                return chunk
            raise StopIteration

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self._index < len(self.chunks):
                chunk = self.chunks[self._index]
                self._index += 1
                # Add mcp_list_tools to first chunk if present
                if not self.sent_first_chunk:
                    chunk = self._add_mcp_list_tools_to_first_chunk(chunk)
                    self.sent_first_chunk = True
                return chunk
            raise StopAsyncIteration

    # Track calls to acompletion
    acompletion_calls = []

    # Create mock streaming response for initial call
    from unittest.mock import MagicMock, AsyncMock
    logging_obj = MagicMock()
    logging_obj.model_call_details = {}
    logging_obj.async_failure_handler = AsyncMock()
    
    from litellm.types.utils import (
        ModelResponseStream,
        StreamingChoices,
        Delta,
        ChatCompletionDeltaToolCall,
        Function,
    )
    
    # Create initial streaming chunks with tool_calls
    # Add tool_calls to the final chunk so stream_chunk_builder can extract them
    tool_calls = [
        ChatCompletionDeltaToolCall(
            id="call-1",
            type="function",
            function=Function(name="local_search", arguments="{}"),
            index=0,
        )
    ]
    
    initial_chunks = [
        ModelResponseStream(
            id="test-1",
            model="gpt-4o-mini",
            created=1234567890,
            object="chat.completion.chunk",
            choices=[
                StreamingChoices(
                    index=0,
                    delta=Delta(
                        content="",
                        role="assistant",
                        tool_calls=tool_calls,
                    ),
                    finish_reason="tool_calls",
                )
            ],
        )
    ]
    
    class InitialStreamingResponse(CustomStreamWrapper):
        def __init__(self):
            super().__init__(
                completion_stream=None,
                model="gpt-4o-mini",
                logging_obj=logging_obj,
            )
            self.chunks = initial_chunks
            self._index = 0

        def __iter__(self):
            return self

        def __next__(self):
            if self._index < len(self.chunks):
                chunk = self.chunks[self._index]
                self._index += 1
                # Add mcp_list_tools to first chunk if present
                if not self.sent_first_chunk:
                    chunk = self._add_mcp_list_tools_to_first_chunk(chunk)
                    self.sent_first_chunk = True
                return chunk
            raise StopIteration

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self._index < len(self.chunks):
                chunk = self.chunks[self._index]
                self._index += 1
                # Add mcp_list_tools to first chunk if present
                if not self.sent_first_chunk:
                    chunk = self._add_mcp_list_tools_to_first_chunk(chunk)
                    self.sent_first_chunk = True
                return chunk
            raise StopAsyncIteration

    async def mock_acompletion(**kwargs):
        acompletion_calls.append(kwargs)
        # With new implementation, first call should be streaming
        if kwargs.get("stream", False):
            # Check if this is the follow-up call
            messages = kwargs.get("messages", [])
            is_follow_up = any(
                msg.get("role") == "tool" or (isinstance(msg, dict) and "tool_call_id" in str(msg))
                for msg in messages
            )
            if is_follow_up:
                # Follow-up call (streaming)
                return MockStreamingResponse()
            else:
                # Initial call (streaming)
                return InitialStreamingResponse()
        # Non-streaming call should not happen with new implementation, but handle it
        return ModelResponse(
            id="test-1",
            model="gpt-4o-mini",
            choices=[{
                "message": {
                    "role": "assistant",
                    "tool_calls": [{
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "local_search",
                            "arguments": "{}"
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }],
            created=0,
            object="chat.completion",
        )

    with patch("litellm.acompletion", side_effect=mock_acompletion):
        # This should not raise RuntimeError: Timeout context manager should be used inside a task
        # completion() returns a coroutine when MCP tools are present, which acompletion() awaits
        response = litellm.completion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hello"}],
            tools=[
                {
                    "type": "mcp",
                    "server_url": "litellm_proxy/mcp/local",
                    "server_label": "local",
                    "require_approval": "never",
                }
            ],
            stream=True,
            mock_response="Final answer",
            mock_tool_calls=[
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "local_search", "arguments": "{}"},
                }
            ],
        )

        # completion() returns a coroutine when MCP tools are present
        import asyncio
        assert asyncio.iscoroutine(response), "completion() should return a coroutine when MCP tools are present"
        
        # Await the coroutine (this is what acompletion() does internally)
        # This should not raise RuntimeError: Timeout context manager should be used inside a task
        result = await response
        
        # Verify response is a streaming response
        assert isinstance(result, CustomStreamWrapper) or hasattr(result, '__iter__')
        
        # Consume the stream to ensure it works (run in separate thread to avoid event loop conflict)
        from concurrent.futures import ThreadPoolExecutor
        def consume_stream():
            return list(result)
        with ThreadPoolExecutor(max_workers=1) as executor:
            chunks = executor.submit(consume_stream).result()
        assert len(chunks) > 0, "Should have received streaming chunks"
        
        # Verify tool execution was called
        assert fake_execute.called is True  # type: ignore[attr-defined]
        
        # Verify acompletion was called (should be called by acompletion_with_mcp)
        assert len(acompletion_calls) >= 1, "acompletion should be called"


@pytest.mark.asyncio
async def test_mcp_metadata_in_streaming_final_chunk(monkeypatch):
    """
    Test that MCP metadata is added correctly to streaming chunks:
    - mcp_list_tools should be in the first chunk
    - mcp_tool_calls and mcp_call_results should be in the final chunk of initial response
    - Follow-up response should be streamed after initial response
    """
    from types import SimpleNamespace
    from unittest.mock import patch

    from litellm.responses.mcp.litellm_proxy_mcp_handler import (
        LiteLLM_Proxy_MCP_Handler,
    )
    from litellm.responses.utils import ResponsesAPIRequestUtils
    from litellm.utils import CustomStreamWrapper
    from litellm.types.utils import (
        ModelResponseStream,
        StreamingChoices,
        Delta,
        ChatCompletionDeltaToolCall,
        Function,
    )
    from litellm.litellm_core_utils.litellm_logging import Logging

    dummy_tool = SimpleNamespace(
        name="local_search",
        description="search",
        inputSchema={"type": "object", "properties": {}},
    )

    async def fake_process(user_api_key_auth, mcp_tools_with_litellm_proxy, **kwargs):
        return [dummy_tool], {"local_search": "local"}

    async def fake_execute(**kwargs):
        tool_calls = kwargs.get("tool_calls") or []
        call_entry = tool_calls[0]
        call_id = call_entry.get("id") or call_entry.get("call_id") or "call"
        return [
            {
                "tool_call_id": call_id,
                "result": "executed",
                "name": call_entry.get("name", "local_search"),
            }
        ]

    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "_process_mcp_tools_without_openai_transform",
        fake_process,
    )
    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "_execute_tool_calls",
        fake_execute,
    )
    monkeypatch.setattr(
        ResponsesAPIRequestUtils,
        "extract_mcp_headers_from_request",
        staticmethod(lambda secret_fields, tools: (None, None, None, None)),
    )

    # Create mock streaming chunks
    def create_chunk(content, finish_reason=None, tool_calls=None):
        delta = Delta(
            content=content,
            role="assistant",
        )
        if tool_calls:
            delta.tool_calls = tool_calls
        return ModelResponseStream(
            id="test-stream",
            model="gpt-4o-mini",
            created=1234567890,
            object="chat.completion.chunk",
            choices=[
                StreamingChoices(
                    index=0,
                    delta=delta,
                    finish_reason=finish_reason,
                )
            ],
        )

    # Create initial streaming chunks with tool_calls
    # Add tool_calls to the final chunk so stream_chunk_builder can extract them
    tool_calls = [
        ChatCompletionDeltaToolCall(
            id="call-1",
            type="function",
            function=Function(name="local_search", arguments="{}"),
            index=0,
        )
    ]
    initial_chunks = [
        create_chunk("", finish_reason="tool_calls", tool_calls=tool_calls),  # Final chunk with tool_calls
    ]
    
    # Create follow-up streaming chunks
    follow_up_chunks = [
        create_chunk("Hello"),
        create_chunk(" world"),
        create_chunk("!", finish_reason="stop"),  # Final chunk
    ]

    # Create a proper CustomStreamWrapper with logging_obj
    from unittest.mock import MagicMock, AsyncMock
    logging_obj = MagicMock()
    logging_obj.model_call_details = {}
    logging_obj.async_failure_handler = AsyncMock()

    class InitialStreamingResponse(CustomStreamWrapper):
        def __init__(self):
            super().__init__(
                completion_stream=None,
                model="gpt-4o-mini",
                logging_obj=logging_obj,
            )
            self.chunks = initial_chunks
            self._index = 0

        def __iter__(self):
            return self

        def __next__(self):
            if self._index < len(self.chunks):
                chunk = self.chunks[self._index]
                self._index += 1
                # Add mcp_list_tools to first chunk if present
                if not self.sent_first_chunk:
                    chunk = self._add_mcp_list_tools_to_first_chunk(chunk)
                    self.sent_first_chunk = True
                return chunk
            raise StopIteration

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self._index < len(self.chunks):
                chunk = self.chunks[self._index]
                self._index += 1
                # Add mcp_list_tools to first chunk if present
                if not self.sent_first_chunk:
                    chunk = self._add_mcp_list_tools_to_first_chunk(chunk)
                    self.sent_first_chunk = True
                return chunk
            raise StopAsyncIteration

    class FollowUpStreamingResponse(CustomStreamWrapper):
        def __init__(self):
            super().__init__(
                completion_stream=None,
                model="gpt-4o-mini",
                logging_obj=logging_obj,
            )
            self.chunks = follow_up_chunks
            self._index = 0

        def __iter__(self):
            return self

        def __next__(self):
            if self._index < len(self.chunks):
                chunk = self.chunks[self._index]
                self._index += 1
                # Add mcp_list_tools to first chunk if present
                if not self.sent_first_chunk:
                    chunk = self._add_mcp_list_tools_to_first_chunk(chunk)
                    self.sent_first_chunk = True
                return chunk
            raise StopIteration

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self._index < len(self.chunks):
                chunk = self.chunks[self._index]
                self._index += 1
                # Add mcp_list_tools to first chunk if present
                if not self.sent_first_chunk:
                    chunk = self._add_mcp_list_tools_to_first_chunk(chunk)
                    self.sent_first_chunk = True
                return chunk
            raise StopAsyncIteration

    # Track calls to acompletion
    acompletion_calls = []

    async def mock_acompletion(**kwargs):
        acompletion_calls.append(kwargs)
        # With new implementation, first call should be streaming
        if kwargs.get("stream", False):
            # Check if this is the follow-up call (has tool results in messages)
            messages = kwargs.get("messages", [])
            is_follow_up = any(
                msg.get("role") == "tool" or (isinstance(msg, dict) and "tool_call_id" in str(msg))
                for msg in messages
            )
            
            if is_follow_up:
                # Follow-up call - return follow-up chunks
                return FollowUpStreamingResponse()
            else:
                # Initial streaming call - return chunks with tool_calls
                return InitialStreamingResponse()
        # Non-streaming call should not happen with new implementation, but handle it
        return ModelResponse(
            id="test-1",
            model="gpt-4o-mini",
            choices=[{
                "message": {
                    "role": "assistant",
                    "tool_calls": [{
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "local_search",
                            "arguments": "{}"
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }],
            created=0,
            object="chat.completion",
        )

    with patch("litellm.acompletion", side_effect=mock_acompletion):
        response = litellm.completion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hello"}],
            tools=[
                {
                    "type": "mcp",
                    "server_url": "litellm_proxy/mcp/local",
                    "server_label": "local",
                    "require_approval": "never",
                }
            ],
            stream=True,
            mock_response="Final answer",
            mock_tool_calls=[
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "local_search", "arguments": "{}"},
                }
            ],
        )

        import asyncio
        assert asyncio.iscoroutine(response)
        result = await response

        assert isinstance(result, CustomStreamWrapper)

        # Consume the stream and check chunks (run in separate thread to avoid event loop conflict)
        from concurrent.futures import ThreadPoolExecutor
        def consume_stream():
            return list(result)
        with ThreadPoolExecutor(max_workers=1) as executor:
            all_chunks = executor.submit(consume_stream).result()
        assert len(all_chunks) > 0, "Should have received streaming chunks"

        # Find chunks from initial response (with tool_calls finish_reason)
        initial_chunks_list = []
        follow_up_chunks_list = []
        for chunk in all_chunks:
            if hasattr(chunk, "choices") and chunk.choices:
                choice = chunk.choices[0]
                if hasattr(choice, "finish_reason") and choice.finish_reason == "tool_calls":
                    initial_chunks_list.append(chunk)
                elif hasattr(choice, "finish_reason") and choice.finish_reason == "stop":
                    follow_up_chunks_list.append(chunk)
                elif not hasattr(choice, "finish_reason") or choice.finish_reason is None:
                    # Chunks without finish_reason could be from either stream
                    # Check if we've seen tool_calls yet
                    if initial_chunks_list:
                        follow_up_chunks_list.append(chunk)
                    else:
                        initial_chunks_list.append(chunk)

        # Verify initial response chunks
        assert len(initial_chunks_list) > 0, "Should have initial response chunks"
        
        # Find the final chunk from initial response (with tool_calls finish_reason)
        initial_final_chunk = None
        for chunk in initial_chunks_list:
            if hasattr(chunk, "choices") and chunk.choices:
                choice = chunk.choices[0]
                if hasattr(choice, "finish_reason") and choice.finish_reason == "tool_calls":
                    initial_final_chunk = chunk
                    break
        
        if initial_final_chunk is None and initial_chunks_list:
            initial_final_chunk = initial_chunks_list[-1]

        assert initial_final_chunk is not None, "Should have a final chunk from initial response"

        # Verify mcp_list_tools is in the first chunk of initial response
        first_chunk = initial_chunks_list[0] if initial_chunks_list else None
        assert first_chunk is not None, "Should have a first chunk"
        if hasattr(first_chunk, "choices") and first_chunk.choices:
            choice = first_chunk.choices[0]
            if hasattr(choice, "delta") and choice.delta:
                provider_fields = getattr(choice.delta, "provider_specific_fields", None)
                assert provider_fields is not None, "First chunk should have provider_specific_fields"
                assert "mcp_list_tools" in provider_fields, "First chunk should have mcp_list_tools"

        # Verify mcp_tool_calls and mcp_call_results are in the final chunk of initial response
        if hasattr(initial_final_chunk, "choices") and initial_final_chunk.choices:
            choice = initial_final_chunk.choices[0]
            if hasattr(choice, "delta") and choice.delta:
                provider_fields = getattr(choice.delta, "provider_specific_fields", None)
                assert provider_fields is not None, "Final chunk should have provider_specific_fields"
                assert "mcp_tool_calls" in provider_fields, "Should have mcp_tool_calls"
                assert "mcp_call_results" in provider_fields, "Should have mcp_call_results"
        
        # Verify follow-up response chunks are present
        assert len(follow_up_chunks_list) > 0, "Should have follow-up response chunks"


@pytest.mark.asyncio
async def test_mcp_streaming_metadata_ordering(monkeypatch):
    """
    Test that MCP metadata appears in the correct order:
    - mcp_list_tools should appear in the first chunk (before tool_calls)
    - mcp_tool_calls and mcp_call_results should appear in the final chunk of initial response
    - Follow-up response should be streamed after initial response completes
    """
    from types import SimpleNamespace
    from unittest.mock import patch

    from litellm.responses.mcp.litellm_proxy_mcp_handler import (
        LiteLLM_Proxy_MCP_Handler,
    )
    from litellm.responses.utils import ResponsesAPIRequestUtils
    from litellm.utils import CustomStreamWrapper
    from litellm.types.utils import (
        ModelResponseStream,
        StreamingChoices,
        Delta,
        ChatCompletionDeltaToolCall,
        Function,
    )

    dummy_tool = SimpleNamespace(
        name="local_search",
        description="search",
        inputSchema={"type": "object", "properties": {}},
    )

    async def fake_process(user_api_key_auth, mcp_tools_with_litellm_proxy, **kwargs):
        return [dummy_tool], {"local_search": "local"}

    async def fake_execute(**kwargs):
        tool_calls = kwargs.get("tool_calls") or []
        call_entry = tool_calls[0]
        call_id = call_entry.get("id") or call_entry.get("call_id") or "call"
        return [
            {
                "tool_call_id": call_id,
                "result": "executed",
                "name": call_entry.get("name", "local_search"),
            }
        ]

    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "_process_mcp_tools_without_openai_transform",
        fake_process,
    )
    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "_execute_tool_calls",
        fake_execute,
    )
    monkeypatch.setattr(
        ResponsesAPIRequestUtils,
        "extract_mcp_headers_from_request",
        staticmethod(lambda secret_fields, tools: (None, None, None, None)),
    )

    # Create mock streaming chunks
    def create_chunk(content, finish_reason=None, tool_calls=None):
        delta = Delta(
            content=content,
            role="assistant",
        )
        if tool_calls:
            delta.tool_calls = tool_calls
        return ModelResponseStream(
            id="test-stream",
            model="gpt-4o-mini",
            created=1234567890,
            object="chat.completion.chunk",
            choices=[
                StreamingChoices(
                    index=0,
                    delta=delta,
                    finish_reason=finish_reason,
                )
            ],
        )

    # Create initial streaming chunks with tool_calls
    # Add tool_calls to the final chunk so stream_chunk_builder can extract them
    tool_calls = [
        ChatCompletionDeltaToolCall(
            id="call-1",
            type="function",
            function=Function(name="local_search", arguments="{}"),
            index=0,
        )
    ]
    initial_chunks = [
        create_chunk("", finish_reason="tool_calls", tool_calls=tool_calls),  # Final chunk with tool_calls
    ]
    
    # Create follow-up streaming chunks
    follow_up_chunks = [
        create_chunk("Hello"),
        create_chunk(" world"),
        create_chunk("!", finish_reason="stop"),  # Final chunk
    ]

    # Create a proper CustomStreamWrapper with logging_obj
    from unittest.mock import MagicMock, AsyncMock
    logging_obj = MagicMock()
    logging_obj.model_call_details = {}
    logging_obj.async_failure_handler = AsyncMock()

    class InitialStreamingResponse(CustomStreamWrapper):
        def __init__(self):
            super().__init__(
                completion_stream=None,
                model="gpt-4o-mini",
                logging_obj=logging_obj,
            )
            self.chunks = initial_chunks
            self._index = 0

        def __iter__(self):
            return self

        def __next__(self):
            if self._index < len(self.chunks):
                chunk = self.chunks[self._index]
                self._index += 1
                # Add mcp_list_tools to first chunk if present
                if not self.sent_first_chunk:
                    chunk = self._add_mcp_list_tools_to_first_chunk(chunk)
                    self.sent_first_chunk = True
                return chunk
            raise StopIteration

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self._index < len(self.chunks):
                chunk = self.chunks[self._index]
                self._index += 1
                # Add mcp_list_tools to first chunk if present
                if not self.sent_first_chunk:
                    chunk = self._add_mcp_list_tools_to_first_chunk(chunk)
                    self.sent_first_chunk = True
                return chunk
            raise StopAsyncIteration

    class FollowUpStreamingResponse(CustomStreamWrapper):
        def __init__(self):
            super().__init__(
                completion_stream=None,
                model="gpt-4o-mini",
                logging_obj=logging_obj,
            )
            self.chunks = follow_up_chunks
            self._index = 0

        def __iter__(self):
            return self

        def __next__(self):
            if self._index < len(self.chunks):
                chunk = self.chunks[self._index]
                self._index += 1
                # Add mcp_list_tools to first chunk if present
                if not self.sent_first_chunk:
                    chunk = self._add_mcp_list_tools_to_first_chunk(chunk)
                    self.sent_first_chunk = True
                return chunk
            raise StopIteration

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self._index < len(self.chunks):
                chunk = self.chunks[self._index]
                self._index += 1
                # Add mcp_list_tools to first chunk if present
                if not self.sent_first_chunk:
                    chunk = self._add_mcp_list_tools_to_first_chunk(chunk)
                    self.sent_first_chunk = True
                return chunk
            raise StopAsyncIteration

    # Track calls to acompletion
    acompletion_calls = []

    async def mock_acompletion(**kwargs):
        acompletion_calls.append(kwargs)
        # With new implementation, first call should be streaming
        if kwargs.get("stream", False):
            # Check if this is the follow-up call (has tool results in messages)
            messages = kwargs.get("messages", [])
            is_follow_up = any(
                msg.get("role") == "tool" or (isinstance(msg, dict) and "tool_call_id" in str(msg))
                for msg in messages
            )
            
            if is_follow_up:
                # Follow-up call - return follow-up chunks
                return FollowUpStreamingResponse()
            else:
                # Initial streaming call - return chunks with tool_calls
                return InitialStreamingResponse()
        # Non-streaming call should not happen with new implementation
        pytest.fail("Non-streaming call should not happen with new implementation")

    with patch("litellm.acompletion", side_effect=mock_acompletion):
        response = litellm.completion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hello"}],
            tools=[
                {
                    "type": "mcp",
                    "server_url": "litellm_proxy/mcp/local",
                    "server_label": "local",
                    "require_approval": "never",
                }
            ],
            stream=True,
            mock_response="Final answer",
            mock_tool_calls=[
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "local_search", "arguments": "{}"},
                }
            ],
        )

        import asyncio
        assert asyncio.iscoroutine(response)
        result = await response

        assert isinstance(result, CustomStreamWrapper)

        # Consume the stream and verify order (run in separate thread to avoid event loop conflict)
        from concurrent.futures import ThreadPoolExecutor
        def consume_stream():
            return list(result)
        with ThreadPoolExecutor(max_workers=1) as executor:
            all_chunks = executor.submit(consume_stream).result()
        assert len(all_chunks) > 0, "Should have received streaming chunks"

        # Track when we see each type of metadata
        mcp_list_tools_seen = False
        mcp_tool_calls_seen = False
        mcp_call_results_seen = False
        tool_calls_finish_reason_seen = False
        follow_up_content_seen = False

        for i, chunk in enumerate(all_chunks):
            if hasattr(chunk, "choices") and chunk.choices:
                choice = chunk.choices[0]
                if hasattr(choice, "delta") and choice.delta:
                    provider_fields = getattr(choice.delta, "provider_specific_fields", None)
                    if provider_fields:
                        if "mcp_list_tools" in provider_fields:
                            mcp_list_tools_seen = True
                            # mcp_list_tools should appear before tool_calls finish_reason
                            assert not tool_calls_finish_reason_seen, \
                                "mcp_list_tools should appear before tool_calls finish_reason"
                        if "mcp_tool_calls" in provider_fields:
                            mcp_tool_calls_seen = True
                        if "mcp_call_results" in provider_fields:
                            mcp_call_results_seen = True
                
                if hasattr(choice, "finish_reason") and choice.finish_reason == "tool_calls":
                    tool_calls_finish_reason_seen = True
                    # mcp_tool_calls and mcp_call_results should be in the same chunk as tool_calls finish_reason
                    if hasattr(choice, "delta") and choice.delta:
                        provider_fields = getattr(choice.delta, "provider_specific_fields", None)
                        assert provider_fields is not None
                        assert "mcp_tool_calls" in provider_fields, \
                            "mcp_tool_calls should be in the chunk with tool_calls finish_reason"
                        assert "mcp_call_results" in provider_fields, \
                            "mcp_call_results should be in the chunk with tool_calls finish_reason"
                
                if hasattr(choice, "delta") and choice.delta and choice.delta.content:
                    content = choice.delta.content
                    if content and ("Hello" in content or "world" in content or "!" in content):
                        follow_up_content_seen = True
                        # Follow-up content should appear after tool_calls finish_reason
                        assert tool_calls_finish_reason_seen, \
                            "Follow-up content should appear after tool_calls finish_reason"

        # Verify all metadata was seen
        assert mcp_list_tools_seen, "Should have seen mcp_list_tools"
        assert mcp_tool_calls_seen, "Should have seen mcp_tool_calls"
        assert mcp_call_results_seen, "Should have seen mcp_call_results"
        assert tool_calls_finish_reason_seen, "Should have seen tool_calls finish_reason"
        assert follow_up_content_seen, "Should have seen follow-up content"
