"""
WebFetch interception tests.
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch

from litellm.constants import LITELLM_WEB_FETCH_TOOL_NAME
from litellm.integrations.webfetch_interception.handler import (
    WebFetchInterceptionLogger,
)
from litellm.integrations.webfetch_interception.tools import (
    get_litellm_web_fetch_tool,
    is_web_fetch_tool,
)
from litellm.integrations.webfetch_interception.transformation import (
    WebFetchTransformation,
)
from litellm.types.utils import LlmProviders


class TestWebFetchInterceptionLogger:
    """Test WebFetchInterceptionLogger."""

    @pytest.fixture
    def logger(self):
        return WebFetchInterceptionLogger(
            enabled_providers=[LlmProviders.BEDROCK],
            fetch_tool_name="my-firecrawl-fetch",
        )

    @pytest.fixture
    def mock_fetch_response(self):
        from litellm.llms.base_llm.fetch.transformation import WebFetchResponse

        return WebFetchResponse(
            url="https://example.com",
            title="Test Title",
            content="# Test Content\n\nThis is test content.",
        )

    def test_init(self, logger):
        """Test logger initialization."""
        assert logger.enabled_providers == ["bedrock"]
        assert logger.fetch_tool_name == "my-firecrawl-fetch"

    def test_init_all_providers(self):
        """Test logger initialization with all providers."""
        logger = WebFetchInterceptionLogger(enabled_providers=None)
        assert logger.enabled_providers == ["bedrock"]  # Default

    @pytest.mark.asyncio
    async def test_async_pre_call_deployment_hook_no_tools(self, logger):
        """Test deployment hook with no tools."""
        kwargs = {"model": "bedrock/claude-3", "custom_llm_provider": "bedrock"}
        result = await logger.async_pre_call_deployment_hook(kwargs, None)
        assert result is None

    @pytest.mark.asyncio
    async def test_async_pre_call_deployment_hook_with_fetch_tool(self, logger):
        """Test deployment hook converts native fetch tools."""
        native_tool = {
            "type": "web_fetch_20250305",
            "name": "web_fetch",
        }
        kwargs = {
            "model": "bedrock/claude-3",
            "custom_llm_provider": "bedrock",
            "tools": [native_tool],
        }
        result = await logger.async_pre_call_deployment_hook(kwargs, None)

        assert result is not None
        assert "tools" in result
        assert len(result["tools"]) == 1
        assert result["tools"][0]["function"]["name"] == LITELLM_WEB_FETCH_TOOL_NAME

    @pytest.mark.asyncio
    async def test_async_pre_call_deployment_hook_disallowed_provider(self, logger):
        """Test deployment hook skips non-enabled providers."""
        kwargs = {
            "model": "openai/gpt-4",
            "custom_llm_provider": "openai",
            "tools": [{"type": "web_fetch_20250305", "name": "web_fetch"}],
        }
        result = await logger.async_pre_call_deployment_hook(kwargs, None)
        assert result is None

    @pytest.mark.asyncio
    async def test_async_should_run_agentic_loop_no_fetch(self, logger):
        """Test agentic loop detection with no fetch tool."""
        response = {"content": [{"type": "text", "text": "Hello"}]}
        should_run, info = await logger.async_should_run_agentic_loop(
            response=response,
            model="bedrock/claude-3",
            messages=[],
            tools=[],
            stream=False,
            custom_llm_provider="bedrock",
            kwargs={},
        )
        assert should_run is False
        assert info == {}

    @pytest.mark.asyncio
    async def test_async_should_run_agentic_loop_with_fetch(self, logger):
        """Test agentic loop detection with fetch tool_use."""
        response = {
            "content": [
                {
                    "type": "tool_use",
                    "id": "fetch_1",
                    "name": LITELLM_WEB_FETCH_TOOL_NAME,
                    "input": {"url": "https://example.com"},
                }
            ]
        }
        should_run, info = await logger.async_should_run_agentic_loop(
            response=response,
            model="bedrock/claude-3",
            messages=[],
            tools=[{"name": LITELLM_WEB_FETCH_TOOL_NAME}],
            stream=False,
            custom_llm_provider="bedrock",
            kwargs={},
        )
        assert should_run is True
        assert "tool_calls" in info
        assert len(info["tool_calls"]) == 1
        assert info["tool_type"] == "webfetch"

    @pytest.mark.asyncio
    async def test_async_should_run_chat_completion_agentic_loop_with_fetch(
        self, logger
    ):
        """Test chat completion agentic loop detection."""
        # Create explicit mock structure for OpenAI-style response
        response = Mock()

        # Mock choice with message containing tool_calls
        choice_mock = Mock()
        message_mock = Mock()

        # Mock function inside tool_call
        function_mock = Mock()
        function_mock.name = LITELLM_WEB_FETCH_TOOL_NAME
        function_mock.arguments = '{"url": "https://example.com"}'

        # Mock tool_call
        tool_call_mock = Mock()
        tool_call_mock.id = "call_1"
        tool_call_mock.function = function_mock
        tool_call_mock.type = "function"

        message_mock.tool_calls = [tool_call_mock]
        choice_mock.message = message_mock
        response.choices = [choice_mock]

        should_run, info = await logger.async_should_run_chat_completion_agentic_loop(
            response=response,
            model="bedrock/claude-3",
            messages=[],
            tools=[
                {"type": "function", "function": {"name": LITELLM_WEB_FETCH_TOOL_NAME}}
            ],
            stream=False,
            custom_llm_provider="bedrock",
            kwargs={},
        )
        assert should_run is True
        assert "tool_calls" in info

    @pytest.mark.asyncio
    async def test_async_run_agentic_loop(self, logger, mock_fetch_response):
        """Test full agentic loop execution."""
        tool_calls = [
            {
                "id": "fetch_1",
                "name": LITELLM_WEB_FETCH_TOOL_NAME,
                "input": {"url": "https://example.com"},
            }
        ]
        tools_dict = {
            "tool_calls": tool_calls,
            "tool_type": "webfetch",
            "provider": "bedrock",
            "response_format": "anthropic",
            "thinking_blocks": [],
        }

        # Mock the fetch execution
        with patch.object(
            logger, "_get_fetch_config", new_callable=AsyncMock
        ) as mock_get_config:
            mock_config = AsyncMock()
            mock_config.afetch_url = AsyncMock(return_value=mock_fetch_response)
            mock_config.ui_friendly_name.return_value = "Firecrawl"
            mock_get_config.return_value = mock_config

            with patch(
                "litellm.integrations.webfetch_interception.handler.anthropic_messages.acreate",
                new_callable=AsyncMock,
            ) as mock_acreate:
                mock_acreate.return_value = {
                    "content": [{"type": "text", "text": "Result"}]
                }

                result = await logger.async_run_agentic_loop(
                    tools=tools_dict,
                    model="bedrock/claude-3",
                    messages=[{"role": "user", "content": "Fetch this"}],
                    response={},
                    anthropic_messages_provider_config=None,
                    anthropic_messages_optional_request_params={},
                    logging_obj=None,
                    stream=False,
                    kwargs={},
                )

                assert result is not None

    @pytest.mark.asyncio
    async def test_async_run_chat_completion_agentic_loop(
        self, logger, mock_fetch_response
    ):
        """Test chat completion agentic loop execution."""
        tool_calls = [
            {
                "id": "call_1",
                "function": {
                    "name": LITELLM_WEB_FETCH_TOOL_NAME,
                    "arguments": '{"url": "https://example.com"}',
                },
            }
        ]
        tools_dict = {
            "tool_calls": tool_calls,
            "tool_type": "webfetch",
            "provider": "bedrock",
            "response_format": "openai",
        }

        # Mock the fetch execution
        with patch.object(
            logger, "_get_fetch_config", new_callable=AsyncMock
        ) as mock_get_config:
            mock_config = AsyncMock()
            mock_config.afetch_url = AsyncMock(return_value=mock_fetch_response)
            mock_config.ui_friendly_name.return_value = "Firecrawl"
            mock_get_config.return_value = mock_config

            with patch(
                "litellm.acompletion",
                new_callable=AsyncMock,
            ) as mock_acompletion:
                mock_acompletion.return_value = {
                    "choices": [{"message": {"content": "Result"}}]
                }

                result = await logger.async_run_chat_completion_agentic_loop(
                    tools=tools_dict,
                    model="bedrock/claude-3",
                    messages=[{"role": "user", "content": "Fetch this"}],
                    response={},
                    optional_params={},
                    logging_obj=None,
                    stream=False,
                    kwargs={},
                )

                assert result is not None


class TestWebFetchTransformation:
    """Test WebFetchTransformation."""

    def test_transform_request_anthropic_no_fetch(self):
        """Test no fetch detected in Anthropic response."""
        response = {"content": [{"type": "text", "text": "Hello"}]}
        has_fetch, tool_calls = WebFetchTransformation.transform_request(
            response, stream=False, response_format="anthropic"
        )
        assert has_fetch is False
        assert tool_calls == []

    def test_transform_request_anthropic_with_fetch(self):
        """Test fetch detected in Anthropic response."""
        response = {
            "content": [
                {
                    "type": "tool_use",
                    "id": "fetch_1",
                    "name": LITELLM_WEB_FETCH_TOOL_NAME,
                    "input": {"url": "https://example.com"},
                }
            ]
        }
        has_fetch, tool_calls = WebFetchTransformation.transform_request(
            response, stream=False, response_format="anthropic"
        )
        assert has_fetch is True
        assert len(tool_calls) == 1
        assert tool_calls[0]["input"]["url"] == "https://example.com"

    def test_transform_response_anthropic(self):
        """Test Anthropic response transformation."""
        tool_calls = [
            {
                "id": "fetch_1",
                "name": LITELLM_WEB_FETCH_TOOL_NAME,
                "input": {"url": "https://example.com"},
            }
        ]
        fetch_results = ["Title: Test\n\nContent: Hello"]

        assistant_msg, user_msg = WebFetchTransformation.transform_response(
            tool_calls, fetch_results, response_format="anthropic"
        )

        assert assistant_msg["role"] == "assistant"
        assert assistant_msg["content"][0]["type"] == "tool_use"

        assert user_msg["role"] == "user"
        assert user_msg["content"][0]["type"] == "tool_result"

    def test_transform_response_openai(self):
        """Test OpenAI response transformation."""
        tool_calls = [
            {
                "id": "call_1",
                "function": {
                    "name": LITELLM_WEB_FETCH_TOOL_NAME,
                    "arguments": '{"url": "https://example.com"}',
                },
            }
        ]
        fetch_results = ["Title: Test\n\nContent: Hello"]

        assistant_msg, tool_msgs = WebFetchTransformation.transform_response(
            tool_calls, fetch_results, response_format="openai"
        )

        assert assistant_msg["role"] == "assistant"
        assert "tool_calls" in assistant_msg

        assert isinstance(tool_msgs, list)
        assert tool_msgs[0]["role"] == "tool"

    def test_format_fetch_response(self):
        """Test fetch response formatting."""
        from litellm.llms.base_llm.fetch.transformation import WebFetchResponse

        response = WebFetchResponse(
            url="https://example.com",
            title="Test Title",
            content="Test content",
        )

        formatted = WebFetchTransformation.format_fetch_response(response)
        assert "Test Title" in formatted
        assert "https://example.com" in formatted
        assert "Test content" in formatted


class TestWebFetchTools:
    """Test web fetch tools."""

    def test_get_litellm_web_fetch_tool(self):
        """Test standard fetch tool definition."""
        tool = get_litellm_web_fetch_tool()
        assert tool["name"] == LITELLM_WEB_FETCH_TOOL_NAME
        assert "url" in tool["input_schema"]["properties"]
        assert "url" in tool["input_schema"]["required"]

    def test_is_web_fetch_tool_standard(self):
        """Test detection of standard fetch tool."""
        assert is_web_fetch_tool({"name": LITELLM_WEB_FETCH_TOOL_NAME}) is True

    def test_is_web_fetch_tool_native_anthropic(self):
        """Test detection of native Anthropic fetch tool."""
        assert (
            is_web_fetch_tool({"type": "web_fetch_20250305", "name": "web_fetch"})
            is True
        )

    def test_is_web_fetch_tool_not_fetch(self):
        """Test that non-fetch tools are not detected."""
        assert is_web_fetch_tool({"name": "calculator"}) is False
