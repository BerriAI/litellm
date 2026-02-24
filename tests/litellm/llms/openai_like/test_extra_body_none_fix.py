"""
Unit tests for fix: normalize extra_body=None to {} to prevent TypeError
Fixes: https://github.com/BerriAI/litellm/issues/21891

When extra_body is explicitly passed as None (e.g. via litellm.responses(..., extra_body=None)),
two code paths would crash with TypeError: 'NoneType' object is not a mapping
when trying to unpack **extra_body or **optional_params["extra_body"].
"""

import sys
import os

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
)

import pytest


class TestOpenAILikeChatHandlerExtraBodyNone:
    """
    Tests for fix in litellm/llms/openai_like/chat/handler.py:
    extra_body = optional_params.pop("extra_body", {}) or {}
    """

    def test_transform_request_extra_body_none_does_not_raise(self):
        """extra_body=None must not raise TypeError when building request dict."""
        from litellm.llms.openai_like.chat.transformation import OpenAIGPTConfig

        config = OpenAIGPTConfig()
        optional_params = {
            "stream": False,
            "extra_body": None,  # explicitly None
        }
        # Should not raise TypeError: 'NoneType' object is not a mapping
        result = config.transform_request(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
            optional_params=optional_params,
            litellm_params={},
            headers={},
        )
        assert isinstance(result, dict)
        # extra_body=None should be treated as {} - no extra keys added
        assert "extra_body" not in result

    def test_transform_request_extra_body_empty_dict_works(self):
        """extra_body={} (normal case) must continue to work."""
        from litellm.llms.openai_like.chat.transformation import OpenAIGPTConfig

        config = OpenAIGPTConfig()
        optional_params = {
            "stream": False,
            "extra_body": {},
        }
        result = config.transform_request(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
            optional_params=optional_params,
            litellm_params={},
            headers={},
        )
        assert isinstance(result, dict)

    def test_transform_request_extra_body_with_dict_merges(self):
        """extra_body with real keys should still be merged into request."""
        from litellm.llms.openai_like.chat.transformation import OpenAIGPTConfig

        config = OpenAIGPTConfig()
        optional_params = {
            "stream": False,
            "extra_body": {"custom_param": "custom_value"},
        }
        result = config.transform_request(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
            optional_params=optional_params,
            litellm_params={},
            headers={},
        )
        assert isinstance(result, dict)
        assert result.get("custom_param") == "custom_value"


class TestAddProviderSpecificParamsExtraBodyNone:
    """
    Tests for fix in litellm/utils.py: add_provider_specific_params_to_optional_params()
    """

    def test_get_optional_params_extra_body_none_does_not_raise(self):
        """get_optional_params with extra_body=None must not raise TypeError."""
        import litellm

        # Should not raise TypeError: 'NoneType' object is not a mapping
        result = litellm.utils.get_optional_params(
            model="gpt-4o",
            custom_llm_provider="openai",
            extra_body=None,  # explicitly None
        )
        assert isinstance(result, dict)
        # extra_body=None should produce empty extra_body or no extra_body key
        extra_body = result.get("extra_body", {})
        assert extra_body == {} or extra_body is None

    def test_get_optional_params_extra_body_empty_dict_works(self):
        """get_optional_params with extra_body={} must continue to work."""
        import litellm

        result = litellm.utils.get_optional_params(
            model="gpt-4o",
            custom_llm_provider="openai",
            extra_body={},
        )
        assert isinstance(result, dict)

    def test_get_optional_params_extra_body_none_for_hosted_vllm(self):
        """get_optional_params with extra_body=None for hosted_vllm (openai_compatible_providers) must not raise."""
        import litellm

        # Should not raise TypeError
        result = litellm.utils.get_optional_params(
            model="gpt-4o",
            custom_llm_provider="hosted_vllm",
            extra_body=None,
        )
        assert isinstance(result, dict)

    def test_get_optional_params_extra_body_none_for_openrouter(self):
        """get_optional_params with extra_body=None for openrouter must not raise."""
        import litellm

        result = litellm.utils.get_optional_params(
            model="gpt-4o",
            custom_llm_provider="openrouter",
            extra_body=None,
        )
        assert isinstance(result, dict)
