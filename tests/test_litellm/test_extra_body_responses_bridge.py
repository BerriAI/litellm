"""
Regression tests for #20982 — ``extra_body`` is not passed through the
responses-to-completion bridge.

When calling ``litellm.responses()`` with ``extra_body`` for a model
that uses the completion bridge, the ``extra_body`` parameter was silently
dropped because:
1. It's an explicit parameter, so it's not in ``**kwargs``
2. The handler and transformation functions didn't accept/forward it
"""

import os
import sys

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

from litellm.responses.litellm_completion_transformation.transformation import (
    LiteLLMCompletionResponsesConfig,
)


class TestExtraBodyResponsesBridge:
    """Verify ``extra_body`` flows through the responses-to-completion
    transformation."""

    def test_extra_body_included_in_completion_request(self):
        """extra_body should appear in the transformed completion request."""
        extra_body = {"provider": {"order": ["Together"], "allow_fallbacks": False}}
        result = LiteLLMCompletionResponsesConfig.transform_responses_api_request_to_chat_completion_request(
            model="openrouter/test-model",
            input="Hello",
            responses_api_request={},
            custom_llm_provider="openrouter",
            stream=False,
            extra_headers=None,
            extra_body=extra_body,
        )
        assert result.get("extra_body") == extra_body

    def test_extra_body_none_by_default(self):
        """When not provided, extra_body should be None in the request."""
        result = LiteLLMCompletionResponsesConfig.transform_responses_api_request_to_chat_completion_request(
            model="openrouter/test-model",
            input="Hello",
            responses_api_request={},
            custom_llm_provider="openrouter",
            stream=False,
            extra_headers=None,
        )
        assert result.get("extra_body") is None

    def test_extra_body_preserved_with_other_params(self):
        """extra_body should coexist with other parameters like
        extra_headers, temperature, etc."""
        extra_body = {"custom_key": "custom_value"}
        extra_headers = {"X-Custom": "header"}
        result = LiteLLMCompletionResponsesConfig.transform_responses_api_request_to_chat_completion_request(
            model="test-model",
            input="Hello",
            responses_api_request={"temperature": 0.7, "max_output_tokens": 100},
            custom_llm_provider=None,
            stream=True,
            extra_headers=extra_headers,
            extra_body=extra_body,
        )
        assert result["extra_body"] == extra_body
        assert result["extra_headers"] == extra_headers
        assert result["temperature"] == 0.7
        assert result["max_tokens"] == 100

    def test_extra_body_empty_dict(self):
        """An empty extra_body dict should still be passed through."""
        result = LiteLLMCompletionResponsesConfig.transform_responses_api_request_to_chat_completion_request(
            model="test-model",
            input="Hello",
            responses_api_request={},
            extra_body={},
        )
        assert result.get("extra_body") == {}

    def test_extra_body_nested_dict(self):
        """Deeply nested extra_body should be preserved as-is."""
        extra_body = {
            "provider": {
                "order": ["Together", "Fireworks"],
                "allow_fallbacks": False,
                "quantizations": ["fp16"],
            },
            "transforms": ["middle-out"],
        }
        result = LiteLLMCompletionResponsesConfig.transform_responses_api_request_to_chat_completion_request(
            model="openrouter/model",
            input="Test",
            responses_api_request={},
            extra_body=extra_body,
        )
        assert result["extra_body"] == extra_body
        assert result["extra_body"]["provider"]["order"] == ["Together", "Fireworks"]
