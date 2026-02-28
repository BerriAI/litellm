"""
Tests for get_llm_provider_logic.py

Focuses on None model handling to prevent 'argument of type NoneType is not iterable' errors.
"""

import pytest

import litellm
from litellm.litellm_core_utils.get_llm_provider_logic import (
    get_llm_provider,
    handle_anthropic_text_model_custom_llm_provider,
    handle_cohere_chat_model_custom_llm_provider,
)


class TestHandleNoneModel:
    """Tests for handling None model parameter"""

    def test_handle_cohere_chat_model_with_none_model(self):
        """Test that handle_cohere_chat_model_custom_llm_provider handles None model gracefully"""
        # Should not raise TypeError: argument of type 'NoneType' is not iterable
        result = handle_cohere_chat_model_custom_llm_provider(
            model=None, custom_llm_provider="cohere"
        )
        assert result == (None, "cohere")

    def test_handle_anthropic_text_model_with_none_model(self):
        """Test that handle_anthropic_text_model_custom_llm_provider handles None model gracefully"""
        # Should not raise TypeError: argument of type 'NoneType' is not iterable
        result = handle_anthropic_text_model_custom_llm_provider(
            model=None, custom_llm_provider="anthropic"
        )
        assert result == (None, "anthropic")

    def test_get_llm_provider_with_none_model_raises_clear_error(self):
        """Test that get_llm_provider raises a clear error when model is None"""
        # The ValueError is caught by the function's exception handler and wrapped in BadRequestError
        with pytest.raises(litellm.exceptions.BadRequestError) as exc_info:
            get_llm_provider(model=None)

        assert "model parameter is required but was None" in str(exc_info.value)


class TestValidModelHandling:
    """Tests to ensure valid models still work correctly after the None checks"""

    def test_handle_cohere_chat_model_with_valid_model(self):
        """Test that valid cohere models still work"""
        result = handle_cohere_chat_model_custom_llm_provider(
            model="cohere/command-r-plus", custom_llm_provider=None
        )
        # Should parse the model correctly
        assert result[0] == "command-r-plus" or result[0] == "cohere/command-r-plus"

    def test_handle_cohere_chat_model_with_model_without_slash(self):
        """Test that models without slash work"""
        result = handle_cohere_chat_model_custom_llm_provider(
            model="command-r-plus", custom_llm_provider="cohere"
        )
        assert result[0] == "command-r-plus"

    def test_handle_anthropic_text_model_with_valid_model(self):
        """Test that valid anthropic models still work"""
        result = handle_anthropic_text_model_custom_llm_provider(
            model="anthropic/claude-2", custom_llm_provider=None
        )
        # Should parse the model correctly
        assert "claude-2" in result[0]

    def test_handle_anthropic_text_model_with_model_without_slash(self):
        """Test that models without slash work"""
        result = handle_anthropic_text_model_custom_llm_provider(
            model="claude-2", custom_llm_provider="anthropic"
        )
        assert result[0] == "claude-2"
