"""
Unit tests for the MCP Sampling Handler.
Tests the sampling/createMessage handler that routes MCP sampling
requests through litellm.acompletion().
"""

from unittest.mock import AsyncMock, MagicMock, patch
import pytest


# ─────────────────────────────────────────────────────────────
# Helper factories
# ─────────────────────────────────────────────────────────────
def _make_text_content(text: str):
    """Create a mock TextContent."""
    tc = MagicMock()
    tc.type = "text"
    tc.text = text
    return tc


def _make_image_content(data: str = "base64data", mime_type: str = "image/png"):
    """Create a mock ImageContent."""
    ic = MagicMock()
    ic.type = "image"
    ic.data = data
    ic.mimeType = mime_type
    return ic


def _make_sampling_message(role: str, content):
    """Create a mock SamplingMessage."""
    msg = MagicMock()
    msg.role = role
    msg.content = content
    return msg


def _make_model_preferences(hints=None, cost=None, speed=None, intelligence=None):
    """Create a mock ModelPreferences."""
    prefs = MagicMock()
    prefs.hints = hints or []
    prefs.costPriority = cost
    prefs.speedPriority = speed
    prefs.intelligencePriority = intelligence
    return prefs


def _make_hint(name: str):
    """Create a mock model hint."""
    hint = MagicMock()
    hint.name = name
    return hint


def _make_params(
    messages=None,
    model_preferences=None,
    system_prompt=None,
    max_tokens=100,
    temperature=None,
    stop_sequences=None,
    tools=None,
    tool_choice=None,
    metadata=None,
):
    """Create a mock CreateMessageRequestParams."""
    params = MagicMock()
    params.messages = messages or []
    params.modelPreferences = model_preferences
    params.systemPrompt = system_prompt
    params.maxTokens = max_tokens
    params.temperature = temperature
    params.stopSequences = stop_sequences
    params.tools = tools
    params.toolChoice = tool_choice
    params.metadata = metadata
    return params


def _make_completion_response(
    content="Hello!", model="gpt-4o-mini", finish_reason="stop", tool_calls=None
):
    """Create a mock litellm completion response."""
    response = MagicMock()
    choice = MagicMock()
    choice.finish_reason = finish_reason
    message = MagicMock()
    message.content = content
    message.tool_calls = tool_calls
    choice.message = message
    response.choices = [choice]
    response.model = model
    return response


# ─────────────────────────────────────────────────────────────
# Tests: Message conversion
# ─────────────────────────────────────────────────────────────
class TestConvertMCPMessagesToOpenAI:
    """Tests for _convert_mcp_messages_to_openai."""

    def test_should_convert_simple_text_message(self):
        from litellm.proxy._experimental.mcp_server.sampling_handler import (
            _convert_mcp_messages_to_openai,
        )

        tc = _make_text_content("Hello")
        msg = _make_sampling_message("user", tc)
        result = _convert_mcp_messages_to_openai([msg])
        assert len(result) == 1
        assert result[0]["role"] == "user"

    def test_should_add_system_prompt(self):
        from litellm.proxy._experimental.mcp_server.sampling_handler import (
            _convert_mcp_messages_to_openai,
        )

        tc = _make_text_content("Hello")
        msg = _make_sampling_message("user", tc)
        result = _convert_mcp_messages_to_openai([msg], system_prompt="Be helpful")
        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "Be helpful"

    def test_should_convert_image_content(self):
        from litellm.proxy._experimental.mcp_server.sampling_handler import (
            _convert_mcp_messages_to_openai,
        )

        ic = _make_image_content("base64imgdata", "image/jpeg")
        msg = _make_sampling_message("user", ic)
        result = _convert_mcp_messages_to_openai([msg])
        assert len(result) == 1
        content = result[0]["content"]
        assert isinstance(content, list)
        assert content[0]["type"] == "image_url"
        assert "base64imgdata" in content[0]["image_url"]["url"]

    def test_should_convert_list_of_mixed_content(self):
        from litellm.proxy._experimental.mcp_server.sampling_handler import (
            _convert_mcp_messages_to_openai,
        )

        tc = _make_text_content("Describe this image")
        ic = _make_image_content("imgdata")
        msg = _make_sampling_message("user", [tc, ic])
        result = _convert_mcp_messages_to_openai([msg])
        assert len(result) == 1
        content = result[0]["content"]
        assert isinstance(content, list)
        assert len(content) == 2

    def test_should_convert_multiple_messages(self):
        from litellm.proxy._experimental.mcp_server.sampling_handler import (
            _convert_mcp_messages_to_openai,
        )

        user_msg = _make_sampling_message("user", _make_text_content("Hi"))
        assistant_msg = _make_sampling_message(
            "assistant", _make_text_content("Hello!")
        )
        result = _convert_mcp_messages_to_openai([user_msg, assistant_msg])
        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"


# ─────────────────────────────────────────────────────────────
# Tests: Model resolution
# ─────────────────────────────────────────────────────────────
class TestResolveModel:
    """Tests for _resolve_model_from_preferences."""

    def test_should_use_default_model_when_no_preferences(self):
        from litellm.proxy._experimental.mcp_server.sampling_handler import (
            _resolve_model_from_preferences,
        )

        result = _resolve_model_from_preferences(
            None, default_model="claude-3.5-sonnet"
        )
        assert result == "claude-3.5-sonnet"

    @patch("litellm.proxy.proxy_server.llm_router", None)
    def test_should_raise_error_when_no_model_available(self):
        from litellm.proxy._experimental.mcp_server.sampling_handler import (
            _resolve_model_from_preferences,
        )

        with patch("litellm.model_list", []):
            with pytest.raises(ValueError, match="No model could be resolved"):
                _resolve_model_from_preferences(None)

    @patch("litellm.proxy.proxy_server.llm_router", None)
    def test_should_fallback_to_configured_default_model(self):
        from litellm.proxy._experimental.mcp_server.sampling_handler import (
            _resolve_model_from_preferences,
        )
        import litellm

        with patch("litellm.model_list", []):
            # Simulate configured default model
            with patch.object(
                litellm, "default_mcp_sampling_model", "claude-3-haiku", create=True
            ):
                result = _resolve_model_from_preferences(None)
                assert result == "claude-3-haiku"

    @patch("litellm.model_list", ["gpt-4o", "claude-3.5-sonnet", "gemini-pro"])
    @patch("litellm.proxy.proxy_server.llm_router", None)
    def test_should_match_hint_by_substring(self):
        from litellm.proxy._experimental.mcp_server.sampling_handler import (
            _resolve_model_from_preferences,
        )

        hint = _make_hint("claude")
        prefs = _make_model_preferences(hints=[hint])
        result = _resolve_model_from_preferences(prefs)
        assert "claude" in result.lower()

    @patch("litellm.model_list", ["gpt-4o"])
    @patch("litellm.proxy.proxy_server.llm_router", None)
    def test_should_use_default_when_no_hint_matches(self):
        from litellm.proxy._experimental.mcp_server.sampling_handler import (
            _resolve_model_from_preferences,
        )

        hint = _make_hint("nonexistent-model")
        prefs = _make_model_preferences(hints=[hint])
        result = _resolve_model_from_preferences(prefs, default_model="gpt-4o")
        assert result == "gpt-4o"


# ─────────────────────────────────────────────────────────────
# Tests: Tool conversion
# ─────────────────────────────────────────────────────────────
class TestConvertMCPToolsToOpenAI:
    """Tests for _convert_mcp_tools_to_openai."""

    def test_should_return_none_for_no_tools(self):
        from litellm.proxy._experimental.mcp_server.sampling_handler import (
            _convert_mcp_tools_to_openai,
        )

        assert _convert_mcp_tools_to_openai(None) is None
        assert _convert_mcp_tools_to_openai([]) is None

    def test_should_convert_mcp_tool_to_openai_format(self):
        from litellm.proxy._experimental.mcp_server.sampling_handler import (
            _convert_mcp_tools_to_openai,
        )

        tool = MagicMock()
        tool.name = "get_weather"
        tool.description = "Get weather for a city"
        tool.inputSchema = {
            "type": "object",
            "properties": {"city": {"type": "string"}},
        }
        result = _convert_mcp_tools_to_openai([tool])
        assert result is not None
        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "get_weather"
        assert result[0]["function"]["description"] == "Get weather for a city"


# ─────────────────────────────────────────────────────────────
# Tests: Tool choice conversion
# ─────────────────────────────────────────────────────────────
class TestConvertMCPToolChoiceToOpenAI:
    """Tests for _convert_mcp_tool_choice_to_openai."""

    def test_should_return_none_for_no_tool_choice(self):
        from litellm.proxy._experimental.mcp_server.sampling_handler import (
            _convert_mcp_tool_choice_to_openai,
        )

        assert _convert_mcp_tool_choice_to_openai(None) is None

    def test_should_convert_auto_mode(self):
        from litellm.proxy._experimental.mcp_server.sampling_handler import (
            _convert_mcp_tool_choice_to_openai,
        )

        tc = MagicMock()
        tc.mode = "auto"
        assert _convert_mcp_tool_choice_to_openai(tc) == "auto"

    def test_should_convert_required_mode(self):
        from litellm.proxy._experimental.mcp_server.sampling_handler import (
            _convert_mcp_tool_choice_to_openai,
        )

        tc = MagicMock()
        tc.mode = "required"
        assert _convert_mcp_tool_choice_to_openai(tc) == "required"

    def test_should_convert_none_mode(self):
        from litellm.proxy._experimental.mcp_server.sampling_handler import (
            _convert_mcp_tool_choice_to_openai,
        )

        tc = MagicMock()
        tc.mode = "none"
        assert _convert_mcp_tool_choice_to_openai(tc) == "none"


# ─────────────────────────────────────────────────────────────
# Tests: Response conversion
# ─────────────────────────────────────────────────────────────
class TestConvertOpenAIResponseToMCPResult:
    """Tests for _convert_openai_response_to_mcp_result."""

    def test_should_convert_text_response(self):
        from litellm.proxy._experimental.mcp_server.sampling_handler import (
            _convert_openai_response_to_mcp_result,
        )

        response = _make_completion_response(content="Hello!", model="gpt-4o")
        result = _convert_openai_response_to_mcp_result(response, "gpt-4o")
        assert result.role == "assistant"
        assert result.model == "gpt-4o"
        assert result.stopReason == "endTurn"
        assert result.content.text == "Hello!"

    def test_should_set_max_tokens_stop_reason(self):
        from litellm.proxy._experimental.mcp_server.sampling_handler import (
            _convert_openai_response_to_mcp_result,
        )

        response = _make_completion_response(
            content="Partial...", finish_reason="length"
        )
        result = _convert_openai_response_to_mcp_result(response, "gpt-4o")
        assert result.stopReason == "maxTokens"

    def test_should_convert_tool_calls_response(self):
        from litellm.proxy._experimental.mcp_server.sampling_handler import (
            _convert_openai_response_to_mcp_result,
        )

        tc = MagicMock()
        tc.id = "call_123"
        tc.function.name = "get_weather"
        tc.function.arguments = '{"city": "NYC"}'
        response = _make_completion_response(
            content=None, finish_reason="tool_calls", tool_calls=[tc]
        )
        result = _convert_openai_response_to_mcp_result(response, "gpt-4o")
        assert result.stopReason == "toolUse"
        assert isinstance(result.content, list)
        # Should contain ToolUseContent
        tool_use = result.content[0]
        assert tool_use.type == "tool_use"
        assert tool_use.name == "get_weather"


# ─────────────────────────────────────────────────────────────
# Tests: Full handler
# ─────────────────────────────────────────────────────────────
@patch("litellm.proxy.proxy_server.llm_router", None)
class TestHandleSamplingCreateMessage:
    """Tests for the main handle_sampling_create_message function."""

    @pytest.mark.asyncio
    async def test_should_call_litellm_acompletion(self):
        from litellm.proxy._experimental.mcp_server.sampling_handler import (
            handle_sampling_create_message,
        )

        mock_response = _make_completion_response(content="Test response")
        params = _make_params(
            messages=[_make_sampling_message("user", _make_text_content("Hello"))],
            max_tokens=100,
        )
        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_completion:
            mock_completion.return_value = mock_response
            result = await handle_sampling_create_message(
                context=MagicMock(),
                params=params,
                default_model="gpt-4o-mini",
            )
            mock_completion.assert_called_once()
            assert result.role == "assistant"
            assert result.content.text == "Test response"

    @pytest.mark.asyncio
    async def test_should_include_temperature(self):
        from litellm.proxy._experimental.mcp_server.sampling_handler import (
            handle_sampling_create_message,
        )

        mock_response = _make_completion_response()
        params = _make_params(
            messages=[_make_sampling_message("user", _make_text_content("Hi"))],
            temperature=0.7,
        )
        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_completion:
            mock_completion.return_value = mock_response
            await handle_sampling_create_message(
                context=MagicMock(),
                params=params,
                default_model="gpt-4o-mini",
            )
            call_kwargs = mock_completion.call_args[1]
            assert call_kwargs["temperature"] == 0.7

    @pytest.mark.asyncio
    async def test_should_include_stop_sequences(self):
        from litellm.proxy._experimental.mcp_server.sampling_handler import (
            handle_sampling_create_message,
        )

        mock_response = _make_completion_response()
        params = _make_params(
            messages=[_make_sampling_message("user", _make_text_content("Hi"))],
            stop_sequences=["STOP", "END"],
        )
        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_completion:
            mock_completion.return_value = mock_response
            await handle_sampling_create_message(
                context=MagicMock(),
                params=params,
                default_model="gpt-4o-mini",
            )
            call_kwargs = mock_completion.call_args[1]
            assert call_kwargs["stop"] == ["STOP", "END"]

    @pytest.mark.asyncio
    async def test_should_include_tools_and_tool_choice(self):
        from litellm.proxy._experimental.mcp_server.sampling_handler import (
            handle_sampling_create_message,
        )

        mock_response = _make_completion_response()
        tool = MagicMock()
        tool.name = "search"
        tool.description = "Search the web"
        tool.inputSchema = {"type": "object", "properties": {"q": {"type": "string"}}}
        tc = MagicMock()
        tc.mode = "auto"
        params = _make_params(
            messages=[_make_sampling_message("user", _make_text_content("Search"))],
            tools=[tool],
            tool_choice=tc,
        )
        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_completion:
            mock_completion.return_value = mock_response
            await handle_sampling_create_message(
                context=MagicMock(),
                params=params,
                default_model="gpt-4o-mini",
            )
            call_kwargs = mock_completion.call_args[1]
            assert "tools" in call_kwargs
            assert call_kwargs["tool_choice"] == "auto"

    @pytest.mark.asyncio
    async def test_should_return_error_on_exception(self):
        from litellm.proxy._experimental.mcp_server.sampling_handler import (
            handle_sampling_create_message,
        )

        params = _make_params(
            messages=[_make_sampling_message("user", _make_text_content("Hi"))],
        )
        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_completion:
            mock_completion.side_effect = Exception("API error")
            result = await handle_sampling_create_message(
                context=MagicMock(),
                params=params,
                default_model="gpt-4o-mini",
            )
            assert hasattr(result, "code")
            assert result.code == -1
            assert "API error" in result.message
