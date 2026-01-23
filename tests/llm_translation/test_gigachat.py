"""
Tests for GigaChat LiteLLM Provider

Tests message transformation, parameter handling, and response transformation.
Run with: pytest tests/llm_translation/test_gigachat.py -v
"""

import pytest
from unittest.mock import MagicMock, patch


class TestGigaChatMessageTransformation:
    """Tests for message transformation (OpenAI -> GigaChat format)"""

    @pytest.fixture
    def config(self):
        from litellm.llms.gigachat.chat.transformation import GigaChatConfig
        return GigaChatConfig()

    def test_simple_user_message(self, config):
        """Basic user message should pass through"""
        messages = [{"role": "user", "content": "Hello"}]
        result = config._transform_messages(messages)

        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Hello"

    def test_developer_role_to_system(self, config):
        """Developer role should be converted to system"""
        messages = [{"role": "developer", "content": "You are helpful"}]
        result = config._transform_messages(messages)

        assert result[0]["role"] == "system"

    def test_system_after_first_becomes_user(self, config):
        """System message after first position should become user"""
        messages = [
            {"role": "assistant", "content": "Response"},
            {"role": "system", "content": "Additional instruction"},
        ]
        result = config._transform_messages(messages)

        assert result[0]["role"] == "assistant"
        assert result[1]["role"] == "user"  # system after first becomes user

    def test_tool_role_to_function(self, config):
        """Tool role should be converted to function"""
        messages = [{"role": "tool", "content": "result data"}]
        result = config._transform_messages(messages)

        assert result[0]["role"] == "function"

    def test_tool_calls_to_function_call(self, config):
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
        result = config._transform_messages(messages)

        assert "function_call" in result[0]
        assert result[0]["function_call"]["name"] == "get_weather"
        assert result[0]["function_call"]["arguments"] == {"city": "Moscow"}
        assert "tool_calls" not in result[0]

    def test_none_content_becomes_empty_string(self, config):
        """None content should become empty string"""
        messages = [{"role": "assistant", "content": None}]
        result = config._transform_messages(messages)

        assert result[0]["content"] == ""

    def test_name_field_removed(self, config):
        """name field should be removed (not supported by GigaChat)"""
        messages = [{"role": "user", "content": "Hi", "name": "John"}]
        result = config._transform_messages(messages)

        assert "name" not in result[0]


class TestGigaChatCollapseUserMessages:
    """Tests for collapsing consecutive user messages"""

    @pytest.fixture
    def config(self):
        from litellm.llms.gigachat.chat.transformation import GigaChatConfig
        return GigaChatConfig()

    def test_no_collapse_single_message(self, config):
        """Single message should not be changed"""
        messages = [{"role": "user", "content": "Hello"}]
        result = config._collapse_user_messages(messages)

        assert len(result) == 1
        assert result[0]["content"] == "Hello"

    def test_collapse_consecutive_user_messages(self, config):
        """Consecutive user messages should be collapsed"""
        messages = [
            {"role": "user", "content": "First"},
            {"role": "user", "content": "Second"},
            {"role": "user", "content": "Third"},
        ]
        result = config._collapse_user_messages(messages)

        assert len(result) == 1
        assert "First" in result[0]["content"]
        assert "Second" in result[0]["content"]
        assert "Third" in result[0]["content"]

    def test_no_collapse_with_assistant_between(self, config):
        """Messages with assistant between should not be collapsed"""
        messages = [
            {"role": "user", "content": "First"},
            {"role": "assistant", "content": "Response"},
            {"role": "user", "content": "Second"},
        ]
        result = config._collapse_user_messages(messages)

        assert len(result) == 3


class TestGigaChatToolsTransformation:
    """Tests for tools -> functions conversion"""

    @pytest.fixture
    def config(self):
        from litellm.llms.gigachat.chat.transformation import GigaChatConfig
        return GigaChatConfig()

    def test_single_tool_conversion(self, config):
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
        result = config._convert_tools_to_functions(tools)

        assert len(result) == 1
        assert result[0]["name"] == "get_weather"
        assert result[0]["description"] == "Get weather for a city"

    def test_multiple_tools_conversion(self, config):
        """Multiple tools should all be converted"""
        tools = [
            {"type": "function", "function": {"name": "func1", "description": "First", "parameters": {"type": "object", "properties": {}}}},
            {"type": "function", "function": {"name": "func2", "description": "Second", "parameters": {"type": "object", "properties": {}}}},
        ]
        result = config._convert_tools_to_functions(tools)

        assert len(result) == 2
        assert result[0]["name"] == "func1"
        assert result[1]["name"] == "func2"


class TestGigaChatParamsTransformation:
    """Tests for parameter transformation"""

    @pytest.fixture
    def config(self):
        from litellm.llms.gigachat.chat.transformation import GigaChatConfig
        return GigaChatConfig()

    def test_temperature_zero_becomes_top_p_zero(self, config):
        """temperature=0 should become top_p=0"""
        params = {"temperature": 0}
        result = config.map_openai_params(
            non_default_params=params,
            optional_params={},
            model="GigaChat",
            drop_params=False,
        )

        assert "top_p" in result
        assert result["top_p"] == 0
        assert "temperature" not in result

    def test_temperature_nonzero_preserved(self, config):
        """Non-zero temperature should be preserved"""
        params = {"temperature": 0.7}
        result = config.map_openai_params(
            non_default_params=params,
            optional_params={},
            model="GigaChat",
            drop_params=False,
        )

        assert result["temperature"] == 0.7

    def test_max_completion_tokens_to_max_tokens(self, config):
        """max_completion_tokens should become max_tokens"""
        params = {"max_completion_tokens": 100}
        result = config.map_openai_params(
            non_default_params=params,
            optional_params={},
            model="GigaChat",
            drop_params=False,
        )

        assert result["max_tokens"] == 100

    def test_structured_output_via_json_schema(self, config):
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
        result = config.map_openai_params(
            non_default_params=params,
            optional_params={},
            model="GigaChat",
            drop_params=False,
        )

        assert "_structured_output" in result
        assert result["_structured_output"] is True
        assert "function_call" in result
        assert result["function_call"]["name"] == "person"


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

    def test_gigachat_config_exists(self):
        """GigaChatConfig should be available"""
        import litellm

        assert hasattr(litellm, "GigaChatConfig")


class TestGigaChatTransformRequest:
    """Tests for request transformation"""

    @pytest.fixture
    def config(self):
        from litellm.llms.gigachat.chat.transformation import GigaChatConfig
        return GigaChatConfig()

    def test_basic_request(self, config):
        """Basic request should be transformed correctly"""
        messages = [{"role": "user", "content": "Hello"}]
        result = config.transform_request(
            model="gigachat/GigaChat",
            messages=messages,
            optional_params={},
            litellm_params={},
            headers={},
        )

        assert result["model"] == "GigaChat"
        assert len(result["messages"]) == 1
        assert result["messages"][0]["role"] == "user"

    def test_request_with_temperature(self, config):
        """Request with temperature should include it"""
        messages = [{"role": "user", "content": "Hello"}]
        result = config.transform_request(
            model="gigachat/GigaChat",
            messages=messages,
            optional_params={"temperature": 0.7},
            litellm_params={},
            headers={},
        )

        assert result["temperature"] == 0.7

    def test_request_with_functions(self, config):
        """Request with functions should include them"""
        messages = [{"role": "user", "content": "Hello"}]
        functions = [{"name": "test", "description": "Test", "parameters": {}}]
        result = config.transform_request(
            model="gigachat/GigaChat",
            messages=messages,
            optional_params={"functions": functions},
            litellm_params={},
            headers={},
        )

        assert "functions" in result
        assert len(result["functions"]) == 1


class TestGigaChatSupportedParams:
    """Tests for supported parameters"""

    @pytest.fixture
    def config(self):
        from litellm.llms.gigachat.chat.transformation import GigaChatConfig
        return GigaChatConfig()

    def test_supported_params(self, config):
        """Check supported parameters list"""
        supported = config.get_supported_openai_params("GigaChat")

        assert "temperature" in supported
        assert "max_tokens" in supported
        assert "max_completion_tokens" in supported
        assert "tools" in supported
        assert "response_format" in supported
        assert "stream" in supported


class TestGigaChatBasicAuth:
    """Tests for basic auth (user/password) token flow"""

    def test_request_basic_token_sync_uses_httpx_basic_auth_tuple(self):
        """
        Ensure basic auth token request uses httpx 'auth=(user, password)' tuple and
        hits the /token endpoint.
        """
        from litellm.llms.gigachat.authenticator import _request_basic_token_sync
        from litellm.llms.gigachat.common_utils import USER_AGENT, build_url

        api_base = "https://example.com/api/v1"
        user = "alice"
        password = "secret"

        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"access_token": "token-123", "expires_at": 1234567890}

        mock_httpx_client = MagicMock()
        mock_httpx_client.post.return_value = mock_response

        mock_httpx_client_ctx = MagicMock()
        mock_httpx_client_ctx.__enter__.return_value = mock_httpx_client
        mock_httpx_client_ctx.__exit__.return_value = None

        with patch(
            "litellm.llms.gigachat.authenticator.httpx.Client",
            return_value=mock_httpx_client_ctx,
        ) as mock_client_ctor:
            token, expires_at = _request_basic_token_sync(
                user=user,
                password=password,
                api_base=api_base,
            )

        assert token == "token-123"
        assert expires_at == 1234567890

        mock_client_ctor.assert_called_once_with(verify=False)
        mock_httpx_client.post.assert_called_once()

        called_args, called_kwargs = mock_httpx_client.post.call_args
        assert called_args[0] == build_url(api_base, "token")
        assert called_kwargs["auth"] == (user, password)
        assert called_kwargs["headers"] == {"User-Agent": USER_AGENT}

    def test_get_access_token_falls_back_to_basic_auth(self):
        """If no OAuth credentials exist, get_access_token should use basic auth path."""
        from litellm.llms.gigachat import authenticator

        with patch.object(authenticator, "_get_credentials", return_value=None), patch.object(
            authenticator, "_token_cache"
        ) as mock_cache, patch.object(
            authenticator, "_request_basic_token_sync", return_value=("token-abc", 9999999999999)
        ) as mock_request:
            mock_cache.get_cache.return_value = None
            token = authenticator.get_access_token(
                credentials=None,
                user="bob",
                password="pw",
            )

        assert token == "token-abc"
        mock_request.assert_called_once_with("bob", "pw", ssl_verify=None)

    def test_get_access_token_raises_without_credentials_or_user_password(self):
        """No credentials and no user/password should raise a GigaChatAuthError."""
        from litellm.llms.gigachat import authenticator
        from litellm.llms.gigachat.common_utils import GigaChatAuthError

        with patch.object(authenticator, "_get_credentials", return_value=None), patch.object(
            authenticator, "_get_user_password", return_value=(None, None)
        ):
            with pytest.raises(GigaChatAuthError) as e:
                authenticator.get_access_token(credentials=None, user=None, password=None)

        assert e.value.status_code == 401
