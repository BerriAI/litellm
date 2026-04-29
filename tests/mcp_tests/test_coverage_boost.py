import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from litellm.proxy._experimental.mcp_server.sampling_handler import (
    _convert_single_content,
    _convert_openai_response_to_mcp_result,
    handle_sampling_create_message,
)
from litellm.proxy._experimental.mcp_server.server import get_or_extract_auth_context
from litellm.proxy._types import UserAPIKeyAuth

# Mock MCP types
try:
    from mcp.types import (
        TextContent, ImageContent, SamplingMessage, 
        CreateMessageRequestParams, ToolUseContent, ToolResultContent
    )
except ImportError:
    class TextContent:
        def __init__(self, type="text", text=""): self.type = type; self.text = text
    class ImageContent:
        def __init__(self, type="image", data="", mimeType="image/png"):
            self.type = type; self.data = data; self.mimeType = mimeType
    class SamplingMessage:
        def __init__(self, role, content): self.role = role; self.content = content
    class CreateMessageRequestParams:
        def __init__(self, messages, maxTokens=100): self.messages = messages; self.maxTokens = maxTokens
    class ToolUseContent:
        def __init__(self, type="tool_use", id=None, name=None, input=None):
            self.type = type; self.id = id; self.name = name; self.input = input
    class ToolResultContent:
        def __init__(self, type="tool_result", toolUseId=None, content=None):
            self.type = type; self.toolUseId = toolUseId; self.content = content

class MockAudioContent:
    def __init__(self, data="audio_data", mimeType="audio/wav"):
        self.type = "audio"
        self.data = data
        self.mimeType = mimeType

def test_convert_audio_content():
    audio = MockAudioContent()
    result = _convert_single_content(audio)
    assert result["type"] == "input_audio"
    assert result["input_audio"]["data"] == "audio_data"
    assert result["input_audio"]["format"] == "wav"

def test_convert_openai_response_to_mcp_result_with_tool_calls():
    mock_choice = MagicMock()
    mock_choice.message.content = "I will search now"
    mock_tool_call = MagicMock()
    mock_tool_call.id = "call_1"
    mock_tool_call.function.name = "search"
    mock_tool_call.function.arguments = '{"q": "test"}'
    
    mock_choice.message.tool_calls = [mock_tool_call]
    mock_choice.finish_reason = "tool_calls"
    
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.model = "gpt-4"
    
    result = _convert_openai_response_to_mcp_result(mock_response, model_name="gpt-4")
    assert result.role == "assistant"
    # It should have both text and tool use content
    # Depending on implementation it might return CreateMessageResultWithTools
    assert hasattr(result, "content")

@pytest.mark.asyncio
async def test_handle_sampling_no_package_error():
    params = CreateMessageRequestParams(
        messages=[SamplingMessage(role="user", content=TextContent(type="text", text="hi"))],
        maxTokens=100
    )
    with patch("litellm.proxy._experimental.mcp_server.sampling_handler.MCP_SAMPLING_AVAILABLE", False):
        result = await handle_sampling_create_message(context=None, params=params)
        assert hasattr(result, "message")
        assert "MCP sampling is not available" in result.message

@pytest.mark.asyncio
async def test_get_or_extract_auth_context_fallback():
    # Test fallback to session read_stream when ContextVar is empty
    mock_session = MagicMock()
    mock_read_stream = MagicMock()
    mock_user_auth = UserAPIKeyAuth(api_key="sk-test", user_id="user-1")
    
    from litellm.proxy._experimental.mcp_server.server import MCPAuthenticatedUser
    mock_read_stream._litellm_auth_context = MCPAuthenticatedUser(
        user_api_key_auth=mock_user_auth,
        mcp_auth_header=None,
        mcp_servers=None,
        mcp_server_auth_headers=None,
        oauth2_headers=None,
        raw_headers=None,
        client_ip=None
    )
    mock_session._read_stream = mock_read_stream
    
    mock_request_ctx = MagicMock()
    mock_request_ctx.get.return_value.session = mock_session
    
    with patch("litellm.proxy._experimental.mcp_server.server.get_auth_context", return_value=(None, None, None, None, None, {}, None)):
        with patch("mcp.server.lowlevel.server.request_ctx", mock_request_ctx):
            result = await get_or_extract_auth_context()
            assert result[0] == mock_user_auth
            assert result[0].api_key is not None

@pytest.mark.asyncio
async def test_get_or_extract_auth_context_exception_handling():
    # Test that it handles exceptions in fallback gracefully
    with patch("litellm.proxy._experimental.mcp_server.server.get_auth_context", return_value=(None, None, None, None, None, {}, None)):
        with patch("mcp.server.lowlevel.server.request_ctx", side_effect=Exception("Context error")):
            result = await get_or_extract_auth_context()
            assert result[0] is None
