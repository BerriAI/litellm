import os
import sys

import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

import litellm
from litellm.llms.meta_llama.chat.transformation import LlamaAPIConfig


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


def test_llama_api_streaming_no_307_error():
    """
    Test that the OpenAI-compatible httpx clients use follow_redirects=True.

    meta_llama routes through the OpenAI SDK path (BaseOpenAILLM), so the
    follow_redirects setting on that SDK's underlying httpx client is what
    actually prevents 307 redirect errors for LLaMA API streaming.
    """
    from litellm.llms.openai.common_utils import BaseOpenAILLM

    # Verify the async httpx client has follow_redirects enabled
    async_client = BaseOpenAILLM._get_async_http_client()
    assert async_client is not None
    assert (
        async_client.follow_redirects is True
    ), "Async httpx client should set follow_redirects=True to prevent 307 errors"

    # Verify the sync httpx client has follow_redirects enabled
    sync_client = BaseOpenAILLM._get_sync_http_client()
    assert sync_client is not None
    assert (
        sync_client.follow_redirects is True
    ), "Sync httpx client should set follow_redirects=True to prevent 307 errors"
