import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from litellm.proxy._experimental.mcp_server.sampling_handler import (
    _resolve_model_from_preferences,
    _convert_mcp_content_to_openai,
    _convert_mcp_messages_to_openai,
    _convert_mcp_tools_to_openai,
    _convert_mcp_tool_choice_to_openai,
    _convert_openai_response_to_mcp_result,
    handle_sampling_create_message,
)

# Mock MCP types if not available
try:
    from mcp.types import (
        ModelPreferences,
        ModelHint,
        SamplingMessage,
        TextContent,
        ImageContent,
        Tool,
        ToolChoice,
        CreateMessageRequestParams,
        ToolUseContent,
        ToolResultContent,
    )
except ImportError:
    # Minimal mocks for testing when mcp package is not installed
    class ModelHint:
        def __init__(self, name=None):
            self.name = name

    class ModelPreferences:
        def __init__(self, hints=None):
            self.hints = hints

    class SamplingMessage:
        def __init__(self, role, content):
            self.role = role
            self.content = content

    class TextContent:
        def __init__(self, type="text", text=""):
            self.type = type
            self.text = text

    class ImageContent:
        def __init__(self, type="image", data="", mimeType="image/png"):
            self.type = type
            self.data = data
            self.mimeType = mimeType

    class Tool:
        def __init__(self, name, description=None, inputSchema=None):
            self.name = name
            self.description = description
            self.inputSchema = inputSchema

    class ToolChoice:
        def __init__(self, mode="auto"):
            self.mode = mode

    class CreateMessageRequestParams:
        def __init__(
            self,
            messages,
            modelPreferences=None,
            systemPrompt=None,
            maxTokens=None,
            temperature=None,
            stopSequences=None,
            tools=None,
            toolChoice=None,
            metadata=None,
        ):
            self.messages = messages
            self.modelPreferences = modelPreferences
            self.systemPrompt = systemPrompt
            self.maxTokens = maxTokens
            self.temperature = temperature
            self.stopSequences = stopSequences
            self.tools = tools
            self.toolChoice = toolChoice
            self.metadata = metadata

    class ToolUseContent:
        def __init__(self, type="tool_use", id=None, name=None, input=None):
            self.type = type
            self.id = id
            self.name = name
            self.input = input

    class ToolResultContent:
        def __init__(self, type="tool_result", toolUseId=None, content=None):
            self.type = type
            self.toolUseId = toolUseId
            self.content = content


def test_resolve_model_from_preferences():
    # Test 1: Direct match
    prefs = ModelPreferences(hints=[ModelHint(name="gpt-4")])
    with patch("litellm.proxy.proxy_server.llm_router") as mock_router:
        mock_router.get_model_names.return_value = ["gpt-4", "gpt-3.5-turbo"]
        assert _resolve_model_from_preferences(prefs) == "gpt-4"

    # Test 2: Substring match
    prefs = ModelPreferences(hints=[ModelHint(name="claude")])
    with patch("litellm.proxy.proxy_server.llm_router") as mock_router:
        mock_router.get_model_names.return_value = ["anthropic/claude-3"]
        assert _resolve_model_from_preferences(prefs) == "anthropic/claude-3"

    # Test 3: Default fallback
    assert _resolve_model_from_preferences(None, default_model="fallback") == "fallback"


def test_convert_mcp_content_to_openai():
    # Text content
    text = TextContent(type="text", text="hello")
    assert _convert_mcp_content_to_openai(text) == {"type": "text", "text": "hello"}

    # Image content
    img = ImageContent(type="image", data="base64data", mimeType="image/jpeg")
    assert _convert_mcp_content_to_openai(img) == {
        "type": "image_url",
        "image_url": {"url": "data:image/jpeg;base64,base64data"},
    }

    # List of content
    content_list = [text, img]
    result = _convert_mcp_content_to_openai(content_list)
    assert len(result) == 2
    assert result[0]["type"] == "text"
    assert result[1]["type"] == "image_url"


def test_convert_mcp_messages_to_openai():
    msg1 = SamplingMessage(role="user", content=TextContent(type="text", text="hi"))
    msg2 = SamplingMessage(
        role="assistant", content=TextContent(type="text", text="hello")
    )

    # Standard messages
    openai_msgs = _convert_mcp_messages_to_openai([msg1, msg2], system_prompt="system")
    assert len(openai_msgs) == 3
    assert openai_msgs[0] == {"role": "system", "content": "system"}
    assert openai_msgs[1]["role"] == "user"
    assert openai_msgs[2]["role"] == "assistant"

    # Tool use/result conversion
    tool_use = ToolUseContent(
        type="tool_use", id="call_1", name="search", input={"q": "test"}
    )
    msg_tool_use = SamplingMessage(
        role="assistant",
        content=[TextContent(type="text", text="searching..."), tool_use],
    )

    openai_msgs = _convert_mcp_messages_to_openai([msg_tool_use])
    assert len(openai_msgs) == 1
    assert openai_msgs[0]["role"] == "assistant"
    assert "tool_calls" in openai_msgs[0]
    assert openai_msgs[0]["tool_calls"][0]["function"]["name"] == "search"
    assert openai_msgs[0]["content"] == "searching..."

    tool_result = ToolResultContent(
        type="tool_result",
        toolUseId="call_1",
        content=[TextContent(type="text", text="found it")],
    )
    msg_tool_result = SamplingMessage(role="user", content=[tool_result])
    openai_msgs = _convert_mcp_messages_to_openai([msg_tool_result])
    assert len(openai_msgs) == 1
    assert openai_msgs[0]["role"] == "tool"
    assert openai_msgs[0]["tool_call_id"] == "call_1"
    assert openai_msgs[0]["content"] == "found it"


def test_convert_mcp_tools_to_openai():
    mcp_tool = Tool(name="my_tool", description="desc", inputSchema={"type": "object"})
    openai_tools = _convert_mcp_tools_to_openai([mcp_tool])
    assert len(openai_tools) == 1
    assert openai_tools[0]["type"] == "function"
    assert openai_tools[0]["function"]["name"] == "my_tool"


def test_convert_mcp_tool_choice_to_openai():
    assert _convert_mcp_tool_choice_to_openai(ToolChoice(mode="auto")) == "auto"
    assert _convert_mcp_tool_choice_to_openai(ToolChoice(mode="required")) == "required"
    assert _convert_mcp_tool_choice_to_openai(ToolChoice(mode="none")) == "none"


@pytest.mark.asyncio
async def test_handle_sampling_create_message_success():
    params = CreateMessageRequestParams(
        messages=[
            SamplingMessage(role="user", content=TextContent(type="text", text="hi"))
        ],
        maxTokens=100,
    )

    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(
            message=MagicMock(content="hello response", tool_calls=None),
            finish_reason="stop",
        )
    ]
    mock_response.model = "gpt-4o-mini"

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_completion:
        mock_completion.return_value = mock_response
        result = await handle_sampling_create_message(context=None, params=params)

        assert result.role == "assistant"
        assert result.content.text == "hello response"
        assert result.model == "gpt-4o-mini"


@pytest.mark.asyncio
async def test_handle_sampling_with_auth_cost_tracking():
    from litellm.proxy._types import UserAPIKeyAuth

    params = CreateMessageRequestParams(
        messages=[
            SamplingMessage(role="user", content=TextContent(type="text", text="hi"))
        ],
        maxTokens=100,
    )
    user_auth = UserAPIKeyAuth(api_key="sk-123", user_id="user-456", team_id="team-789")

    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(
            message=MagicMock(content="ok", tool_calls=None), finish_reason="stop"
        )
    ]
    mock_response.model = "gpt-4o-mini"

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_completion:
        mock_completion.return_value = mock_response
        await handle_sampling_create_message(
            context=None, params=params, user_api_key_auth=user_auth
        )

        # Verify auth was injected into metadata
        kwargs = mock_completion.call_args.kwargs
        assert kwargs["user"] == "user-456"
        assert kwargs["metadata"]["user_api_key"] is not None
        assert kwargs["metadata"]["user_api_key_team_id"] == "team-789"
