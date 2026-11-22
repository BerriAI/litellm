"""
Test that LiteLLM_Params and GenericLiteLLMParams handle reserved keys gracefully.

This test verifies the fix for the bug where passing a dict containing 'self',
'params', or '__class__' keys to LiteLLM_Params() would cause:
    TypeError: LiteLLM_Params.__init__() got multiple values for argument 'self'
"""

import pytest

from litellm.types.router import GenericLiteLLMParams, LiteLLM_Params


class TestLiteLLMParamsReservedKeys:
    """Test that reserved keys in input data are filtered out gracefully."""

    def test_litellm_params_with_self_key(self):
        """Test LiteLLM_Params handles 'self' key in input dict."""
        params_dict = {"model": "gpt-4", "self": "some_value", "api_key": "test-key"}
        params = LiteLLM_Params(**params_dict)
        assert params.model == "gpt-4"
        assert params.api_key == "test-key"
        assert not hasattr(params, "self") or params.get("self") is None

    def test_litellm_params_with_params_key(self):
        """Test LiteLLM_Params handles 'params' key in input dict."""
        params_dict = {"model": "gpt-4", "params": "bad_value"}
        params = LiteLLM_Params(**params_dict)
        assert params.model == "gpt-4"

    def test_litellm_params_with_class_key(self):
        """Test LiteLLM_Params handles '__class__' key in input dict."""
        params_dict = {"model": "gpt-4", "__class__": "bad_value"}
        params = LiteLLM_Params(**params_dict)
        assert params.model == "gpt-4"

    def test_generic_litellm_params_with_self_key(self):
        """Test GenericLiteLLMParams handles 'self' key in input dict."""
        params_dict = {"self": "some_value", "api_key": "test-key"}
        params = GenericLiteLLMParams(**params_dict)
        assert params.api_key == "test-key"

    def test_generic_litellm_params_with_params_key(self):
        """Test GenericLiteLLMParams handles 'params' key in input dict."""
        params_dict = {"params": "bad_value", "api_key": "test-key"}
        params = GenericLiteLLMParams(**params_dict)
        assert params.api_key == "test-key"

    def test_generic_litellm_params_with_class_key(self):
        """Test GenericLiteLLMParams handles '__class__' key in input dict."""
        params_dict = {"__class__": "bad_value", "api_key": "test-key"}
        params = GenericLiteLLMParams(**params_dict)
        assert params.api_key == "test-key"

    def test_max_retries_string_conversion(self):
        """Test that max_retries is converted from string to int."""
        params = LiteLLM_Params(model="gpt-4", max_retries="5")
        assert params.max_retries == 5
        assert isinstance(params.max_retries, int)

    def test_extra_fields_preserved(self):
        """Test that extra fields are preserved when reserved keys are filtered."""
        params_dict = {
            "model": "gpt-4",
            "self": "ignored",
            "custom_field": "custom_value",
        }
        params = LiteLLM_Params(**params_dict)
        assert params.model == "gpt-4"
        assert params.custom_field == "custom_value"

    def test_normal_instantiation_still_works(self):
        """Test that normal instantiation without reserved keys works."""
        params = LiteLLM_Params(
            model="gpt-4", api_key="test-key", custom_llm_provider="openai"
        )
        assert params.model == "gpt-4"
        assert params.api_key == "test-key"
        assert params.custom_llm_provider == "openai"

    def test_multiple_reserved_keys(self):
        """Test filtering multiple reserved keys at once."""
        params_dict = {
            "model": "gpt-4",
            "self": "value1",
            "params": "value2",
            "__class__": "value3",
            "api_key": "test-key",
        }
        params = LiteLLM_Params(**params_dict)
        assert params.model == "gpt-4"
        assert params.api_key == "test-key"
