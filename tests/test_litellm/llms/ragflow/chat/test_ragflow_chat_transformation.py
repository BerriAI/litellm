"""
Test file for RAGFlow chat transformation functionality.

Tests the model name parsing, URL construction, and request transformation
for RAGFlow's OpenAI-compatible API with custom path structures.
"""

import os
import sys
from unittest.mock import Mock, patch

import pytest

# Add the project root to Python path
sys.path.insert(0, os.path.abspath("../../../../.."))

import litellm
from litellm.llms.ragflow.chat.transformation import RAGFlowConfig
from litellm.types.llms.openai import AllMessageValues


class TestRAGFlowChatTransformation:
    """Test suite for RAGFlow chat transformation functionality."""

    def test_parse_ragflow_model_chat(self):
        """Test parsing of chat model format."""
        config = RAGFlowConfig()
        
        model = "ragflow/chat/my-chat-id/gpt-4o-mini"
        endpoint_type, entity_id, model_name = config._parse_ragflow_model(model)
        
        assert endpoint_type == "chat"
        assert entity_id == "my-chat-id"
        assert model_name == "gpt-4o-mini"

    def test_parse_ragflow_model_agent(self):
        """Test parsing of agent model format."""
        config = RAGFlowConfig()
        
        model = "ragflow/agent/my-agent-id/gpt-4o-mini"
        endpoint_type, entity_id, model_name = config._parse_ragflow_model(model)
        
        assert endpoint_type == "agent"
        assert entity_id == "my-agent-id"
        assert model_name == "gpt-4o-mini"

    def test_parse_ragflow_model_with_slashes_in_model_name(self):
        """Test parsing when model name contains slashes."""
        config = RAGFlowConfig()
        
        model = "ragflow/chat/my-chat-id/openai/gpt-4o-mini"
        endpoint_type, entity_id, model_name = config._parse_ragflow_model(model)
        
        assert endpoint_type == "chat"
        assert entity_id == "my-chat-id"
        assert model_name == "openai/gpt-4o-mini"

    def test_parse_ragflow_model_invalid_format(self):
        """Test parsing with invalid model format."""
        config = RAGFlowConfig()
        
        with pytest.raises(ValueError, match="Invalid RAGFlow model format"):
            config._parse_ragflow_model("ragflow/chat/model-name")
        
        with pytest.raises(ValueError, match="Invalid RAGFlow model format"):
            config._parse_ragflow_model("invalid/chat/id/model")
        
        with pytest.raises(ValueError, match="Must start with 'ragflow/'"):
            config._parse_ragflow_model("not-ragflow/chat/id/model")

    def test_parse_ragflow_model_invalid_endpoint_type(self):
        """Test parsing with invalid endpoint type."""
        config = RAGFlowConfig()
        
        with pytest.raises(ValueError, match="Invalid RAGFlow endpoint type"):
            config._parse_ragflow_model("ragflow/invalid/my-id/model")

    def test_get_complete_url_chat(self):
        """Test URL construction for chat endpoint."""
        config = RAGFlowConfig()
        
        model = "ragflow/chat/my-chat-id/gpt-4o-mini"
        api_base = "http://localhost:9380"
        
        url = config.get_complete_url(
            api_base=api_base,
            api_key=None,
            model=model,
            optional_params={},
            litellm_params={},
            stream=False,
        )
        
        assert url == "http://localhost:9380/api/v1/chats_openai/my-chat-id/chat/completions"

    def test_get_complete_url_agent(self):
        """Test URL construction for agent endpoint."""
        config = RAGFlowConfig()
        
        model = "ragflow/agent/my-agent-id/gpt-4o-mini"
        api_base = "http://localhost:9380"
        
        url = config.get_complete_url(
            api_base=api_base,
            api_key=None,
            model=model,
            optional_params={},
            litellm_params={},
            stream=False,
        )
        
        assert url == "http://localhost:9380/api/v1/agents_openai/my-agent-id/chat/completions"

    def test_get_complete_url_strips_v1(self):
        """Test URL construction when api_base ends with /v1."""
        config = RAGFlowConfig()
        
        model = "ragflow/chat/my-chat-id/gpt-4o-mini"
        api_base = "http://localhost:9380/v1"
        
        url = config.get_complete_url(
            api_base=api_base,
            api_key=None,
            model=model,
            optional_params={},
            litellm_params={},
            stream=False,
        )
        
        assert url == "http://localhost:9380/api/v1/chats_openai/my-chat-id/chat/completions"

    def test_get_complete_url_strips_api_v1(self):
        """Test URL construction when api_base ends with /api/v1."""
        config = RAGFlowConfig()
        
        model = "ragflow/agent/my-agent-id/gpt-4o-mini"
        api_base = "http://localhost:9380/api/v1"
        
        url = config.get_complete_url(
            api_base=api_base,
            api_key=None,
            model=model,
            optional_params={},
            litellm_params={},
            stream=False,
        )
        
        assert url == "http://localhost:9380/api/v1/agents_openai/my-agent-id/chat/completions"

    def test_get_complete_url_from_litellm_params(self):
        """Test URL construction with api_base from litellm_params."""
        config = RAGFlowConfig()
        
        model = "ragflow/chat/my-chat-id/gpt-4o-mini"
        # Create a simple dict-like object for litellm_params
        class LiteLLMParams:
            def __init__(self):
                self.api_base = "http://ragflow-server:9380"
        
        litellm_params = LiteLLMParams()
        
        url = config.get_complete_url(
            api_base=None,
            api_key=None,
            model=model,
            optional_params={},
            litellm_params=litellm_params,
            stream=False,
        )
        
        assert url == "http://ragflow-server:9380/api/v1/chats_openai/my-chat-id/chat/completions"

    def test_get_complete_url_missing_api_base(self):
        """Test URL construction when api_base is missing."""
        config = RAGFlowConfig()
        
        model = "ragflow/chat/my-chat-id/gpt-4o-mini"
        
        with pytest.raises(ValueError, match="api_base is required"):
            config.get_complete_url(
                api_base=None,
                api_key=None,
                model=model,
                optional_params={},
                litellm_params={},
                stream=False,
            )

    @patch.dict(os.environ, {"RAGFLOW_API_BASE": "http://env-ragflow:9380"})
    def test_get_complete_url_from_environment(self):
        """Test URL construction with api_base from environment variable."""
        config = RAGFlowConfig()
        
        model = "ragflow/agent/my-agent-id/gpt-4o-mini"
        
        url = config.get_complete_url(
            api_base=None,
            api_key=None,
            model=model,
            optional_params={},
            litellm_params={},
            stream=False,
        )
        
        assert url == "http://env-ragflow:9380/api/v1/agents_openai/my-agent-id/chat/completions"

    def test_validate_environment_sets_headers(self):
        """Test that validate_environment sets proper headers."""
        config = RAGFlowConfig()
        
        headers = {}
        model = "ragflow/chat/my-chat-id/gpt-4o-mini"
        messages = [{"role": "user", "content": "Hello"}]
        api_key = "test-api-key"
        
        result_headers = config.validate_environment(
            headers=headers,
            model=model,
            messages=messages,
            optional_params={},
            litellm_params={},
            api_key=api_key,
            api_base="http://localhost:9380",
        )
        
        assert result_headers["Authorization"] == "Bearer test-api-key"
        assert result_headers["Content-Type"] == "application/json"

    def test_validate_environment_stores_actual_model(self):
        """Test that validate_environment stores actual model name."""
        config = RAGFlowConfig()
        
        headers = {}
        model = "ragflow/chat/my-chat-id/gpt-4o-mini"
        messages = [{"role": "user", "content": "Hello"}]
        litellm_params = {}
        
        config.validate_environment(
            headers=headers,
            model=model,
            messages=messages,
            optional_params={},
            litellm_params=litellm_params,
            api_key="test-key",
            api_base="http://localhost:9380",
        )
        
        assert litellm_params["_ragflow_actual_model"] == "gpt-4o-mini"

    @patch.dict(os.environ, {"RAGFLOW_API_KEY": "env-api-key"})
    def test_validate_environment_from_environment(self):
        """Test that validate_environment gets api_key from environment."""
        config = RAGFlowConfig()
        
        headers = {}
        model = "ragflow/agent/my-agent-id/gpt-4o-mini"
        messages = [{"role": "user", "content": "Hello"}]
        
        result_headers = config.validate_environment(
            headers=headers,
            model=model,
            messages=messages,
            optional_params={},
            litellm_params={},
            api_key=None,
            api_base="http://localhost:9380",
        )
        
        assert result_headers["Authorization"] == "Bearer env-api-key"

    def test_validate_environment_from_litellm_params(self):
        """Test that validate_environment gets api_key from litellm_params."""
        config = RAGFlowConfig()
        
        headers = {}
        model = "ragflow/chat/my-chat-id/gpt-4o-mini"
        messages = [{"role": "user", "content": "Hello"}]
        # Create a simple object for litellm_params with api_key attribute
        class LiteLLMParams:
            def __init__(self):
                self.api_key = "litellm-params-key"
            def __setitem__(self, key, value):
                setattr(self, key, value)
        
        litellm_params = LiteLLMParams()
        
        result_headers = config.validate_environment(
            headers=headers,
            model=model,
            messages=messages,
            optional_params={},
            litellm_params=litellm_params,
            api_key=None,
            api_base="http://localhost:9380",
        )
        
        assert result_headers["Authorization"] == "Bearer litellm-params-key"

    def test_transform_request_uses_actual_model(self):
        """Test that transform_request uses the actual model name."""
        config = RAGFlowConfig()
        
        model = "ragflow/chat/my-chat-id/gpt-4o-mini"
        messages = [{"role": "user", "content": "Hello"}]
        litellm_params = {"_ragflow_actual_model": "gpt-4o-mini"}
        
        # Test the actual behavior by checking the model in the result
        result = config.transform_request(
            model=model,
            messages=messages,
            optional_params={},
            litellm_params=litellm_params,
            headers={},
        )
        
        # The result should contain the actual model name, not the full ragflow path
        assert result["model"] == "gpt-4o-mini"
        assert result["messages"] == messages

    def test_transform_request_fallback_parsing(self):
        """Test that transform_request falls back to parsing if _ragflow_actual_model is missing."""
        config = RAGFlowConfig()
        
        model = "ragflow/agent/my-agent-id/gpt-4o-mini"
        messages = [{"role": "user", "content": "Hello"}]
        litellm_params = {}  # Missing _ragflow_actual_model
        
        result = config.transform_request(
            model=model,
            messages=messages,
            optional_params={},
            litellm_params=litellm_params,
            headers={},
        )
        
        # Should parse and use the actual model name
        assert result["model"] == "gpt-4o-mini"
        assert result["messages"] == messages

    def test_get_openai_compatible_provider_info(self):
        """Test _get_openai_compatible_provider_info returns correct values."""
        config = RAGFlowConfig()
        
        model = "ragflow/chat/my-chat-id/gpt-4o-mini"
        api_base = "http://localhost:9380"
        api_key = "test-key"
        
        result_api_base, result_api_key, result_provider = config._get_openai_compatible_provider_info(
            model=model,
            api_base=api_base,
            api_key=api_key,
            custom_llm_provider="ragflow",
        )
        
        assert result_api_base == api_base
        assert result_api_key == api_key
        assert result_provider == "ragflow"

    @patch.dict(os.environ, {"RAGFLOW_API_BASE": "http://env-base:9380", "RAGFLOW_API_KEY": "env-key"})
    def test_get_openai_compatible_provider_info_from_env(self):
        """Test _get_openai_compatible_provider_info gets values from environment."""
        config = RAGFlowConfig()
        
        model = "ragflow/agent/my-agent-id/gpt-4o-mini"
        
        result_api_base, result_api_key, result_provider = config._get_openai_compatible_provider_info(
            model=model,
            api_base=None,
            api_key=None,
            custom_llm_provider="ragflow",
        )
        
        assert result_api_base == "http://env-base:9380"
        assert result_api_key == "env-key"
        assert result_provider == "ragflow"

