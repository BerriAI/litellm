import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

from litellm.llms.meta_llama.chat.transformation import LlamaAPIConfig


def test_get_supported_openai_params():
    """Test that LlamaAPIConfig correctly filters unsupported parameters"""
    config = LlamaAPIConfig()

    # Test error handling
    with patch("litellm.get_model_info", side_effect=Exception("Test error")):
        params = config.get_supported_openai_params("llama-3.3-8B-instruct")
        assert "function_call" not in params
        assert "tools" not in params
        assert "tool_choice" not in params


def test_map_openai_params():
    """Test that LlamaAPIConfig correctly maps OpenAI parameters"""
    config = LlamaAPIConfig()

    # Test response_format handling - json_schema is allowed
    non_default_params = {"response_format": {"type": "json_schema"}}
    optional_params = {"response_format": True}
    result = config.map_openai_params(
        non_default_params, optional_params, "llama-3.3-8B-instruct", False
    )
    assert "response_format" in result
    assert result["response_format"]["type"] == "json_schema"

    # Test response_format handling - other types are removed
    non_default_params = {"response_format": {"type": "text"}}
    optional_params = {"response_format": True}
    result = config.map_openai_params(
        non_default_params, optional_params, "llama-3.3-8B-instruct", False
    )
    assert "response_format" not in result

    # Test that other parameters are passed through
    non_default_params = {
        "temperature": 0.7,
        "response_format": {"type": "json_schema"},
    }
    optional_params = {"temperature": True, "response_format": True}
    result = config.map_openai_params(
        non_default_params, optional_params, "llama-3.3-8B-instruct", False
    )
    assert "temperature" in result
    assert result["temperature"] == 0.7
    assert "response_format" in result
