"""
Tests for the `store` parameter being correctly forwarded to OpenAI.

Related issue: https://github.com/BerriAI/litellm/issues/23087

The `store` parameter was listed in OPENAI_CHAT_COMPLETION_PARAMS and
DEFAULT_CHAT_COMPLETION_PARAM_VALUES but was silently dropped because
get_non_default_completion_params() excluded it (as a "known" param)
while optional_param_args didn't include it (not a named param of
completion()). The fix adds a safety net in completion() that forwards
any kwargs present in DEFAULT_CHAT_COMPLETION_PARAM_VALUES that aren't
already in optional_param_args.
"""

import os
import sys

sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.constants import DEFAULT_CHAT_COMPLETION_PARAM_VALUES
from litellm.utils import get_non_default_completion_params, get_optional_params


class TestStoreParamForwarding:
    """Tests that `store` flows through the parameter processing pipeline."""

    def test_store_true_forwarded_for_openai(self):
        """should forward store=True for OpenAI models via kwargs"""
        result = get_optional_params(
            model="gpt-5.1",
            custom_llm_provider="openai",
            store=True,
        )
        assert result.get("store") is True

    def test_store_false_forwarded_for_openai(self):
        """should forward store=False for OpenAI models"""
        result = get_optional_params(
            model="gpt-4o",
            custom_llm_provider="openai",
            store=False,
        )
        assert result.get("store") is False

    def test_store_none_not_forwarded(self):
        """should not include store when it is None (default)"""
        result = get_optional_params(
            model="gpt-4o",
            custom_llm_provider="openai",
        )
        assert "store" not in result

    def test_store_with_gpt5_models(self):
        """should forward store=True for GPT-5 family models"""
        for model in ["gpt-5.1", "gpt-5.2"]:
            result = get_optional_params(
                model=model,
                custom_llm_provider="openai",
                store=True,
            )
            assert result.get("store") is True, f"store not forwarded for {model}"

    def test_store_in_supported_params(self):
        """should list store as a supported OpenAI param"""
        from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig

        config = OpenAIGPTConfig()
        for model in ["gpt-4o", "gpt-5.1", "gpt-5.2"]:
            supported = config.get_supported_openai_params(model)
            assert "store" in supported, f"store not in supported params for {model}"

    def test_store_in_transform_request(self):
        """should include store in the final transformed request body"""
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
        """should forward both store and metadata when both are set"""
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


class TestDefaultParamValuesSafetyNet:
    """Tests that any param in DEFAULT_CHAT_COMPLETION_PARAM_VALUES flows
    through completion() even without being a named parameter."""

    def test_known_openai_param_excluded_from_non_default(self):
        """should confirm get_non_default_completion_params excludes known OpenAI params"""
        kwargs = {"store": True, "temperature": 0.5}
        non_default = get_non_default_completion_params(kwargs=kwargs)
        assert "store" not in non_default
        assert "temperature" not in non_default

    def test_unknown_param_included_in_non_default(self):
        """should pass through unknown provider-specific params"""
        kwargs = {"my_custom_provider_param": "foo"}
        non_default = get_non_default_completion_params(kwargs=kwargs)
        assert non_default.get("my_custom_provider_param") == "foo"

    def test_safety_net_forwards_recognized_kwargs(self):
        """should forward kwargs in DEFAULT_CHAT_COMPLETION_PARAM_VALUES
        that are not already in optional_param_args"""
        optional_param_args = {
            "model": "gpt-5.1",
            "custom_llm_provider": "openai",
            "temperature": 0.7,
        }
        kwargs = {"store": True, "metadata": {"key": "value"}}
        for k, v in kwargs.items():
            if (
                k in DEFAULT_CHAT_COMPLETION_PARAM_VALUES
                and k not in optional_param_args
                and v is not None
            ):
                optional_param_args[k] = v

        assert optional_param_args["store"] is True
        assert optional_param_args["metadata"] == {"key": "value"}
        assert optional_param_args["temperature"] == 0.7

    def test_safety_net_does_not_override_existing(self):
        """should not override a param that's already in optional_param_args"""
        optional_param_args = {
            "model": "gpt-5.1",
            "custom_llm_provider": "openai",
            "temperature": 0.7,
        }
        kwargs = {"temperature": 0.9}
        for k, v in kwargs.items():
            if (
                k in DEFAULT_CHAT_COMPLETION_PARAM_VALUES
                and k not in optional_param_args
                and v is not None
            ):
                optional_param_args[k] = v

        assert optional_param_args["temperature"] == 0.7

    def test_safety_net_skips_none_values(self):
        """should not forward params with None value (the default)"""
        optional_param_args = {"model": "gpt-5.1", "custom_llm_provider": "openai"}
        kwargs = {"store": None}
        for k, v in kwargs.items():
            if (
                k in DEFAULT_CHAT_COMPLETION_PARAM_VALUES
                and k not in optional_param_args
                and v is not None
            ):
                optional_param_args[k] = v

        assert "store" not in optional_param_args

    def test_safety_net_forwards_falsy_non_none(self):
        """should forward store=False (falsy but not None)"""
        optional_param_args = {"model": "gpt-5.1", "custom_llm_provider": "openai"}
        kwargs = {"store": False}
        for k, v in kwargs.items():
            if (
                k in DEFAULT_CHAT_COMPLETION_PARAM_VALUES
                and k not in optional_param_args
                and v is not None
            ):
                optional_param_args[k] = v

        assert optional_param_args.get("store") is False
