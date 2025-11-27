"""
Unit tests for RAGFlow configuration.

These tests validate the RAGFlowChatConfig class which extends OpenAIGPTConfig.
RAGFlow is an OpenAI-compatible RAG engine provider.
"""

import os
import sys

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path

import pytest

import litellm
import litellm.utils
from litellm import completion
from litellm.llms.ragflow.chat.transformation import RAGFlowChatConfig


class TestRAGFlowConfig:
    """Test class for RAGFlow functionality"""

    def test_default_api_base(self):
        """Test that default API base is used when none is provided"""
        config = RAGFlowChatConfig()
        headers = {}
        api_key = "fake-ragflow-key"

        # Call validate_environment without specifying api_base
        result = config.validate_environment(
            headers=headers,
            model="ragflow-agent-123",
            messages=[{"role": "user", "content": "Hey"}],
            optional_params={},
            litellm_params={},
            api_key=api_key,
            api_base=None,  # Not providing api_base
        )

        # Verify headers are still set correctly
        assert result["Authorization"] == f"Bearer {api_key}"
        assert result["Content-Type"] == "application/json"

    def test_custom_api_base(self):
        """Test that custom API base is used when provided"""
        config = RAGFlowChatConfig()
        custom_base = "http://custom-ragflow.example.com:9380/v1"
        
        api_base, api_key = config._get_openai_compatible_provider_info(
            api_base=custom_base,
            api_key="test-key"
        )
        
        assert api_base == custom_base
        assert api_key == "test-key"

    def test_get_complete_url_appends_chat_completions(self):
        """Test that /chat/completions is appended to api_base"""
        config = RAGFlowChatConfig()
        
        result = config.get_complete_url(
            api_base="http://localhost:9380/v1",
            api_key="test-key",
            model="ragflow-agent-123",
            optional_params={},
            litellm_params={},
            stream=False
        )
        
        assert result == "http://localhost:9380/v1/chat/completions"

    def test_get_complete_url_no_duplicate_chat_completions(self):
        """Test that /chat/completions is not duplicated if already present"""
        config = RAGFlowChatConfig()
        
        result = config.get_complete_url(
            api_base="http://localhost:9380/v1/chat/completions",
            api_key="test-key",
            model="ragflow-agent-123",
            optional_params={},
            litellm_params={},
            stream=False
        )
        
        assert result == "http://localhost:9380/v1/chat/completions"

    def test_get_supported_openai_params(self):
        """Test that get_supported_openai_params returns correct params"""
        config = RAGFlowChatConfig()
        
        supported_params = config.get_supported_openai_params("ragflow-agent-123")
        
        # Should include standard OpenAI params
        assert "temperature" in supported_params
        assert "max_tokens" in supported_params
        assert "stream" in supported_params
        assert "tools" in supported_params

    def test_map_openai_params(self):
        """Test that OpenAI parameters are mapped correctly"""
        config = RAGFlowChatConfig()
        
        non_default_params = {
            "temperature": 0.7,
            "max_tokens": 1000,
            "stream": True
        }
        
        result = config.map_openai_params(
            non_default_params=non_default_params,
            optional_params={},
            model="ragflow-agent-123",
            drop_params=False
        )
        
        # All supported params should be included
        assert result.get("temperature") == 0.7
        assert result.get("max_tokens") == 1000
        assert result.get("stream") is True

    def test_map_openai_params_max_completion_tokens_mapping(self):
        """Test that max_completion_tokens is mapped to max_tokens"""
        config = RAGFlowChatConfig()
        
        non_default_params = {
            "max_completion_tokens": 1000,
            "temperature": 0.7
        }
        
        result = config.map_openai_params(
            non_default_params=non_default_params,
            optional_params={},
            model="ragflow-agent-123",
            drop_params=False
        )
        
        # max_completion_tokens should be mapped to max_tokens
        assert result.get("max_tokens") == 1000
        assert "max_completion_tokens" not in result
        assert result.get("temperature") == 0.7

    def test_transform_request_basic(self):
        """Test basic transform_request functionality"""
        config = RAGFlowChatConfig()
        
        messages = [
            {"role": "user", "content": "What is RAG?"}
        ]
        
        optional_params = {
            "temperature": 0.7,
            "max_tokens": 500
        }
        
        result = config.transform_request(
            model="ragflow-agent-123",
            messages=messages,
            optional_params=optional_params,
            litellm_params={},
            headers={}
        )
        
        # Check that messages and params are preserved
        assert result["model"] == "ragflow-agent-123"
        assert len(result["messages"]) == 1
        assert result["messages"][0]["role"] == "user"
        assert result["messages"][0]["content"] == "What is RAG?"
        assert result.get("temperature") == 0.7
        assert result.get("max_tokens") == 500

    def test_transform_request_with_tools(self):
        """Test transform_request with tools parameter"""
        config = RAGFlowChatConfig()
        
        messages = [
            {"role": "user", "content": "Search for documents"}
        ]
        
        optional_params = {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "search_documents",
                        "description": "Search the knowledge base"
                    }
                }
            ],
            "tool_choice": "auto"
        }
        
        result = config.transform_request(
            model="ragflow-agent-123",
            messages=messages,
            optional_params=optional_params,
            litellm_params={},
            headers={}
        )
        
        # Check that tools are preserved
        assert "tools" in result
        assert len(result["tools"]) == 1
        assert result["tools"][0]["type"] == "function"
        assert result.get("tool_choice") == "auto"

    def test_transform_messages_handles_content_list(self):
        """Test that messages with content list are converted to string"""
        config = RAGFlowChatConfig()
        
        # Message with content as list (multimodal format)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello"}
                ]
            }
        ]
        
        result = config._transform_messages(
            messages=messages,
            model="ragflow-agent-123",
            is_async=False
        )
        
        # Content should be converted to string
        assert isinstance(result[0]["content"], str)
