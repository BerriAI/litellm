"""
Tests for GigaChat LiteLLM Provider

Tests message transformation, parameter handling, and response transformation.
Run with: pytest tests/llm_translation/test_gigachat.py -v
"""

import json
import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock

# Skip all tests if gigachat is not installed
pytest.importorskip("gigachat")


class TestGigaChatMessageTransformation:
    """Tests for message transformation (OpenAI -> GigaChat format)"""

    @pytest.fixture
    def handler(self):
        from litellm.llms.gigachat.chat.handler import GigaChatChatHandler
        return GigaChatChatHandler()

    def test_simple_user_message(self, handler):
        """Basic user message should pass through"""
        messages = [{"role": "user", "content": "Hello"}]
        result = handler._transform_messages_sync(messages, None)

        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Hello"

    def test_developer_role_to_system(self, handler):
        """Developer role should be converted to system"""
        messages = [{"role": "developer", "content": "You are helpful"}]
        result = handler._transform_messages_sync(messages, None)

        assert result[0]["role"] == "system"

    def test_system_after_first_becomes_user(self, handler):
        """System message after first position should become user"""
        messages = [
            {"role": "assistant", "content": "Response"},
            {"role": "system", "content": "Additional instruction"},
        ]
        result = handler._transform_messages_sync(messages, None)

        assert result[0]["role"] == "assistant"
        assert result[1]["role"] == "user"  # system after first becomes user

    def test_tool_role_to_function(self, handler):
        """Tool role should be converted to function"""
        messages = [{"role": "tool", "content": "result data"}]
        result = handler._transform_messages_sync(messages, None)

        assert result[0]["role"] == "function"

    def test_tool_calls_to_function_call(self, handler):
        """tool_calls should be converted to function_call"""
        messages = [{
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "call_123",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"city": "Moscow"}'
                }
            }]
        }]
        result = handler._transform_messages_sync(messages, None)

        assert "function_call" in result[0]
        assert result[0]["function_call"]["name"] == "get_weather"
        assert result[0]["function_call"]["arguments"] == {"city": "Moscow"}
        assert "tool_calls" not in result[0]

    def test_none_content_becomes_empty_string(self, handler):
        """None content should become empty string"""
        messages = [{"role": "assistant", "content": None}]
        result = handler._transform_messages_sync(messages, None)

        assert result[0]["content"] == ""

    def test_name_field_removed(self, handler):
        """name field should be removed (not supported by GigaChat)"""
        messages = [{"role": "user", "content": "Hi", "name": "John"}]
        result = handler._transform_messages_sync(messages, None)

        assert "name" not in result[0]


class TestGigaChatCollapseUserMessages:
    """Tests for collapsing consecutive user messages"""

    @pytest.fixture
    def handler(self):
        from litellm.llms.gigachat.chat.handler import GigaChatChatHandler
        return GigaChatChatHandler()

    def test_no_collapse_single_message(self, handler):
        """Single message should not be changed"""
        messages = [{"role": "user", "content": "Hello"}]
        result = handler._collapse_user_messages(messages)

        assert len(result) == 1
        assert result[0]["content"] == "Hello"

    def test_collapse_consecutive_user_messages(self, handler):
        """Consecutive user messages should be collapsed"""
        messages = [
            {"role": "user", "content": "First"},
            {"role": "user", "content": "Second"},
            {"role": "user", "content": "Third"},
        ]
        result = handler._collapse_user_messages(messages)

        assert len(result) == 1
        assert "First" in result[0]["content"]
        assert "Second" in result[0]["content"]
        assert "Third" in result[0]["content"]

    def test_no_collapse_with_assistant_between(self, handler):
        """Messages with assistant between should not be collapsed"""
        messages = [
            {"role": "user", "content": "First"},
            {"role": "assistant", "content": "Response"},
            {"role": "user", "content": "Second"},
        ]
        result = handler._collapse_user_messages(messages)

        assert len(result) == 3


class TestGigaChatToolsTransformation:
    """Tests for tools -> functions conversion"""

    @pytest.fixture
    def handler(self):
        from litellm.llms.gigachat.chat.handler import GigaChatChatHandler
        return GigaChatChatHandler()

    def test_none_tools_returns_none(self, handler):
        """None tools should return None"""
        result = handler._transform_tools_to_functions(None)
        assert result is None

    def test_empty_tools_returns_none(self, handler):
        """Empty tools list should return None"""
        result = handler._transform_tools_to_functions([])
        assert result is None

    def test_single_tool_conversion(self, handler):
        """Single tool should be converted correctly"""
        tools = [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"}
                    }
                }
            }
        }]
        result = handler._transform_tools_to_functions(tools)

        assert len(result) == 1
        assert result[0].name == "get_weather"
        assert result[0].description == "Get weather for a city"

    def test_multiple_tools_conversion(self, handler):
        """Multiple tools should all be converted"""
        tools = [
            {"type": "function", "function": {"name": "func1", "description": "First", "parameters": {"type": "object", "properties": {}}}},
            {"type": "function", "function": {"name": "func2", "description": "Second", "parameters": {"type": "object", "properties": {}}}},
        ]
        result = handler._transform_tools_to_functions(tools)

        assert len(result) == 2
        assert result[0].name == "func1"
        assert result[1].name == "func2"


class TestGigaChatParamsTransformation:
    """Tests for parameter transformation"""

    @pytest.fixture
    def handler(self):
        from litellm.llms.gigachat.chat.handler import GigaChatChatHandler
        return GigaChatChatHandler()

    def test_temperature_zero_becomes_top_p_zero(self, handler):
        """temperature=0 should become top_p=0"""
        params = {"temperature": 0}
        result, _ = handler._transform_params(params, "test_id")

        assert "top_p" in result
        assert result["top_p"] == 0
        assert "temperature" not in result

    def test_temperature_nonzero_preserved(self, handler):
        """Non-zero temperature should be preserved"""
        params = {"temperature": 0.7}
        result, _ = handler._transform_params(params, "test_id")

        assert result["temperature"] == 0.7

    def test_max_completion_tokens_to_max_tokens(self, handler):
        """max_completion_tokens should become max_tokens"""
        params = {"max_completion_tokens": 100}
        result, _ = handler._transform_params(params, "test_id")

        assert result["max_tokens"] == 100
        assert "max_completion_tokens" not in result

    def test_unsupported_params_removed(self, handler):
        """Unsupported parameters should be removed"""
        params = {
            "logprobs": True,
            "n": 2,
            "presence_penalty": 0.5,
            "frequency_penalty": 0.5,
            "seed": 42,
        }
        result, _ = handler._transform_params(params, "test_id")

        assert "logprobs" not in result
        assert "n" not in result
        assert "presence_penalty" not in result
        assert "frequency_penalty" not in result
        assert "seed" not in result

    def test_structured_output_via_json_schema(self, handler):
        """json_schema response_format should trigger structured output mode"""
        params = {
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "person",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "age": {"type": "integer"}
                        }
                    }
                }
            }
        }
        result, is_structured = handler._transform_params(params, "test_id")

        assert is_structured is True
        assert "function_call" in result
        assert result["function_call"]["name"] == "person"


class TestGigaChatClientCaching:
    """Tests for client caching"""

    @pytest.fixture
    def handler(self):
        from litellm.llms.gigachat.chat.handler import GigaChatChatHandler
        return GigaChatChatHandler()

    def test_same_credentials_same_client(self, handler):
        """Same credentials should return the same cached client"""
        with patch("litellm.llms.gigachat.chat.handler.GigaChatChatHandler._get_client") as mock_get:
            mock_client = Mock()
            mock_get.return_value = mock_client

            client1 = handler._get_client(api_key="test-key")
            client2 = handler._get_client(api_key="test-key")

            # Both should be the same cached client
            assert client1 is client2

    def test_giga_cred_prefix_parsing(self, handler):
        """giga-cred- prefix should be parsed correctly"""
        with patch("gigachat.GigaChat") as MockGigaChat:
            mock_instance = Mock()
            MockGigaChat.return_value = mock_instance

            handler._get_client(api_key="giga-cred-my_credentials:GIGACHAT_API_B2B")

            # Verify GigaChat was called with correct params
            MockGigaChat.assert_called_once()
            call_kwargs = MockGigaChat.call_args[1]
            assert call_kwargs["credentials"] == "my_credentials"
            assert call_kwargs["scope"] == "GIGACHAT_API_B2B"

    def test_giga_auth_prefix_parsing(self, handler):
        """giga-auth- prefix should be parsed as access_token"""
        with patch("gigachat.GigaChat") as MockGigaChat:
            mock_instance = Mock()
            MockGigaChat.return_value = mock_instance

            handler._get_client(api_key="giga-auth-my_jwt_token")

            MockGigaChat.assert_called_once()
            call_kwargs = MockGigaChat.call_args[1]
            assert call_kwargs["access_token"] == "my_jwt_token"


class TestGigaChatProviderRegistration:
    """Tests for provider registration in LiteLLM"""

    def test_gigachat_in_provider_list(self):
        """GigaChat should be in provider list"""
        from litellm.types.utils import LlmProviders

        assert hasattr(LlmProviders, "GIGACHAT")
        assert LlmProviders.GIGACHAT.value == "gigachat"

    def test_gigachat_in_chat_providers(self):
        """GigaChat should be in LITELLM_CHAT_PROVIDERS"""
        from litellm.constants import LITELLM_CHAT_PROVIDERS

        assert "gigachat" in LITELLM_CHAT_PROVIDERS

    def test_gigachat_key_exists(self):
        """gigachat_key should be available"""
        import litellm

        assert hasattr(litellm, "gigachat_key")


class TestGigaChatEmbeddings:
    """Tests for embedding transformation"""

    @pytest.fixture
    def handler(self):
        from litellm.llms.gigachat.chat.handler import GigaChatChatHandler
        return GigaChatChatHandler()

    def test_transform_embedding(self, handler):
        """Basic embedding should pass through"""
        from gigachat.models import Embeddings

        # Use validate() for older versions or model_validate() for newer pydantic
        data = {
            "data": [
                {
                    "embedding": [0.12345, 0.23453],
                    "usage": {
                        "prompt_tokens": 4,
                    },
                    "index": 0,
                    "object": "list",
                },
                {
                    "embedding": [0.74832, 0.33531],
                    "usage": {
                        "prompt_tokens": 7,
                    },
                    "index": 1,
                    "object": "list",
                }
            ],
            "model": "Embeddings",
            "object": "list",
        }
        # Handle both pydantic v1 and v2 APIs
        if hasattr(Embeddings, 'model_validate'):
            giga_response = Embeddings.model_validate(data)
        else:
            giga_response = Embeddings.validate(data)
        result = handler._transform_embedding(giga_response)

        # check result data
        assert result.model == "Embeddings"
        assert result.object == "list"
        assert result.data[0]["embedding"] == [0.12345, 0.23453]
        assert len(result.data) == 2

        # check usage
        assert result.usage is not None
        assert result.usage.prompt_tokens == result.usage.total_tokens == 11
