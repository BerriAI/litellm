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

    async def fake_process(user_api_key_auth, mcp_tools_with_litellm_proxy):
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

    async def fake_process(user_api_key_auth, mcp_tools_with_litellm_proxy):
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

    async def fake_process(user_api_key_auth, mcp_tools_with_litellm_proxy):
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
    class MockStreamingResponse(CustomStreamWrapper):
        def __init__(self):
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
                return chunk
            raise StopIteration

    # Track calls to acompletion
    acompletion_calls = []

    async def mock_acompletion(**kwargs):
        acompletion_calls.append(kwargs)
        # First call (non-streaming for tool extraction)
        if not kwargs.get("stream", False):
            # Return a ModelResponse with tool_calls using dict format
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
        # Second call (streaming follow-up)
        return MockStreamingResponse()

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
        
        # Consume the stream to ensure it works
        chunks = list(result)
        assert len(chunks) > 0, "Should have received streaming chunks"
        
        # Verify tool execution was called
        assert fake_execute.called is True  # type: ignore[attr-defined]
        
        # Verify acompletion was called (should be called by acompletion_with_mcp)
        assert len(acompletion_calls) >= 1, "acompletion should be called"
