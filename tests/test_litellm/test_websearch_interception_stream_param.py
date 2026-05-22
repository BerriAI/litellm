"""
Unit tests for fix #28458: Internal _websearch_interception_converted_stream flag
must NOT leak into extra_body when building provider-specific params.

The websearch interception deployment hook sets
kwargs["_websearch_interception_converted_stream"] = True to track that stream
was converted from True to False. This internal flag was leaking through
add_provider_specific_params_to_optional_params into extra_body, causing
OpenAI to reject the request with "Unknown parameter:
'_websearch_interception_converted_stream'".

The fix filters out any key starting with '_' from being added to extra_body,
since underscore-prefixed keys are internal flags, not provider params.
"""

import pytest

from litellm.utils import add_provider_specific_params_to_optional_params


class TestInternalFlagsNotInExtraBody:
    """Verify internal _-prefixed flags are excluded from extra_body."""

    def test_websearch_flag_excluded_from_extra_body(self):
        """_websearch_interception_converted_stream must not appear in extra_body."""
        passed_params = {
            "temperature": 0.7,
            "_websearch_interception_converted_stream": True,
        }
        optional_params = {}
        openai_params = ["temperature", "model", "messages"]

        result = add_provider_specific_params_to_optional_params(
            optional_params=optional_params,
            passed_params=passed_params,
            custom_llm_provider="openai",
            openai_params=openai_params,
        )

        extra_body = result.get("extra_body", {})
        assert "_websearch_interception_converted_stream" not in extra_body

    def test_regular_non_openai_params_still_in_extra_body(self):
        """Non-underscore, non-OpenAI params should still go to extra_body."""
        passed_params = {
            "temperature": 0.7,
            "custom_provider_param": "value",
        }
        optional_params = {}
        openai_params = ["temperature", "model", "messages"]

        result = add_provider_specific_params_to_optional_params(
            optional_params=optional_params,
            passed_params=passed_params,
            custom_llm_provider="openai",
            openai_params=openai_params,
        )

        extra_body = result.get("extra_body", {})
        assert extra_body.get("custom_provider_param") == "value"

    def test_all_underscore_prefixed_flags_excluded(self):
        """Any key starting with _ should be excluded from extra_body."""
        passed_params = {
            "_websearch_interception_converted_stream": True,
            "_websearch_interception_other_flag": "test",
            "_some_other_internal_flag": 42,
            "valid_param": "keep",
        }
        optional_params = {}
        openai_params = ["model", "messages"]

        result = add_provider_specific_params_to_optional_params(
            optional_params=optional_params,
            passed_params=passed_params,
            custom_llm_provider="openai",
            openai_params=openai_params,
        )

        extra_body = result.get("extra_body", {})
        assert "_websearch_interception_converted_stream" not in extra_body
        assert "_websearch_interception_other_flag" not in extra_body
        assert "_some_other_internal_flag" not in extra_body
        assert extra_body.get("valid_param") == "keep"

    def test_openai_compatible_providers_also_filtered(self):
        """Internal flags should also be filtered for openai-compatible providers."""
        passed_params = {
            "_websearch_interception_converted_stream": True,
            "custom_param": "value",
        }
        optional_params = {}
        openai_params = ["model", "messages"]

        result = add_provider_specific_params_to_optional_params(
            optional_params=optional_params,
            passed_params=passed_params,
            custom_llm_provider="azure",
            openai_params=openai_params,
        )

        extra_body = result.get("extra_body", {})
        assert "_websearch_interception_converted_stream" not in extra_body
        assert extra_body.get("custom_param") == "value"

    def test_non_openai_provider_flags_excluded(self):
        """Internal flags should also be filtered for non-OpenAI providers (else branch)."""
        passed_params = {
            "_websearch_interception_converted_stream": True,
            "_other_internal": "skip",
            "custom_param": "keep",
        }
        optional_params = {}
        openai_params = ["model", "messages"]

        result = add_provider_specific_params_to_optional_params(
            optional_params=optional_params,
            passed_params=passed_params,
            custom_llm_provider="anthropic",
            openai_params=openai_params,
        )

        assert "_websearch_interception_converted_stream" not in result
        assert "_other_internal" not in result
        assert result.get("custom_param") == "keep"

    def test_no_extra_body_when_only_internal_flags(self):
        """If only internal flags exist (besides openai params), extra_body should be empty."""
        passed_params = {
            "temperature": 0.7,
            "_websearch_interception_converted_stream": True,
        }
        optional_params = {}
        openai_params = ["temperature", "model", "messages"]

        result = add_provider_specific_params_to_optional_params(
            optional_params=optional_params,
            passed_params=passed_params,
            custom_llm_provider="openai",
            openai_params=openai_params,
        )

        extra_body = result.get("extra_body", {})
        assert len(extra_body) == 0
