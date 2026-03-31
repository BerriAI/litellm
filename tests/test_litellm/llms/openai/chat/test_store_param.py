"""
Tests for the `store` parameter being correctly forwarded to OpenAI.

Related issue: https://github.com/BerriAI/litellm/issues/23087

When users pass `store=True` in a completion request, it must be forwarded
to OpenAI so that `metadata` (which the proxy may attach) is accepted by
the API.  Previously `store` was listed as a supported param but was never
threaded through `completion()` -> `get_optional_params()` -> `transform_request()`.
"""

import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.utils import get_optional_params


class TestStoreParamForwarding:
    """Tests that `store` flows through the parameter processing pipeline."""

    def test_store_true_forwarded_for_openai(self):
        """store=True should appear in optional_params for OpenAI models."""
        result = get_optional_params(
            model="gpt-5.1",
            custom_llm_provider="openai",
            store=True,
        )
        assert result.get("store") is True

    def test_store_false_forwarded_for_openai(self):
        """store=False should appear in optional_params for OpenAI models."""
        result = get_optional_params(
            model="gpt-4o",
            custom_llm_provider="openai",
            store=False,
        )
        assert result.get("store") is False

    def test_store_none_not_forwarded(self):
        """When store is None (default), it should not appear in optional_params."""
        result = get_optional_params(
            model="gpt-4o",
            custom_llm_provider="openai",
        )
        assert "store" not in result

    def test_store_with_gpt5_model(self):
        """store=True should work with GPT-5 models (the exact models from the bug report)."""
        for model in ["gpt-5.1", "gpt-5.2"]:
            result = get_optional_params(
                model=model,
                custom_llm_provider="openai",
                store=True,
            )
            assert result.get("store") is True, f"store not forwarded for {model}"

    def test_store_in_supported_params(self):
        """Verify store is listed as a supported OpenAI param for relevant models."""
        from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig

        config = OpenAIGPTConfig()
        for model in ["gpt-4o", "gpt-5.1", "gpt-5.2"]:
            supported = config.get_supported_openai_params(model)
            assert "store" in supported, f"store not in supported params for {model}"

    def test_store_in_transform_request(self):
        """Verify store appears in the final transformed request dict."""
        from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig

        config = OpenAIGPTConfig()
        messages = [{"role": "user", "content": "Hello"}]
        optional_params = {"store": True}
        result = config.transform_request(
            model="gpt-5.1",
            messages=messages,
            optional_params=optional_params,
            litellm_params={},
            headers={},
        )
        assert result.get("store") is True

    def test_store_true_with_metadata(self):
        """When store=True and metadata is also set, both should be in the request."""
        from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig

        config = OpenAIGPTConfig()
        messages = [{"role": "user", "content": "Hello"}]
        optional_params = {"store": True, "metadata": {"key": "value"}}
        result = config.transform_request(
            model="gpt-5.1",
            messages=messages,
            optional_params=optional_params,
            litellm_params={},
            headers={},
        )
        assert result.get("store") is True
        assert result.get("metadata") == {"key": "value"}

    def test_store_param_in_gpt5_search_model(self):
        """store should be supported for GPT-5 search models."""
        from litellm.llms.openai.chat.gpt_5_transformation import OpenAIGPT5Config

        config = OpenAIGPT5Config()
        supported = config.get_supported_openai_params("gpt-5-search-api")
        assert "store" in supported
