"""
Tests for OpenAIBatchesAPI._get_openai_compatible_batch_metadata.

The OpenAI Batch API requires metadata: Dict[str, str].
Proxy hooks (policies, guardrails) inject non-string values such as
    applied_policies: ["my-policy"]
    _model_armor_response: {...}
    queue_time_seconds: 1.23
which cause:
    "Invalid type for 'metadata.applied_policies':
     expected a string, but got an array instead."

This suite verifies the sanitization layer converts every non-string
value to a JSON string and drops internal LiteLLM-only keys.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.llms.openai.openai import OpenAIBatchesAPI


class TestGetOpenaiCompatibleBatchMetadata:
    """Tests for OpenAIBatchesAPI._get_openai_compatible_batch_metadata."""

    def test_string_values_pass_through_unchanged(self):
        metadata = {"user_key": "user_value", "run_id": "abc123"}
        result = OpenAIBatchesAPI._get_openai_compatible_batch_metadata(metadata)
        assert result == {"user_key": "user_value", "run_id": "abc123"}

    def test_applied_policies_list_serialized_to_json_string(self):
        """Regression: applied_policies injected as a list caused a 400 from OpenAI."""
        metadata = {"applied_policies": ["my-policy", "rate-limit-policy"]}
        result = OpenAIBatchesAPI._get_openai_compatible_batch_metadata(metadata)
        assert "applied_policies" in result
        assert isinstance(result["applied_policies"], str)
        assert "my-policy" in result["applied_policies"]

    def test_applied_guardrails_list_serialized_to_json_string(self):
        metadata = {"applied_guardrails": ["guardrail-a", "guardrail-b"]}
        result = OpenAIBatchesAPI._get_openai_compatible_batch_metadata(metadata)
        assert isinstance(result["applied_guardrails"], str)
        assert "guardrail-a" in result["applied_guardrails"]

    def test_dict_values_serialized_to_json_string(self):
        metadata = {
            "_model_armor_response": {
                "sanitizationResult": {"filterMatchState": "MATCH_FOUND"}
            }
        }
        result = OpenAIBatchesAPI._get_openai_compatible_batch_metadata(metadata)
        assert "_model_armor_response" in result
        assert isinstance(result["_model_armor_response"], str)
        assert "MATCH_FOUND" in result["_model_armor_response"]

    def test_float_values_serialized_to_string(self):
        metadata = {"queue_time_seconds": 0.5}
        result = OpenAIBatchesAPI._get_openai_compatible_batch_metadata(metadata)
        assert result == {"queue_time_seconds": "0.5"}

    def test_none_values_excluded(self):
        metadata = {"key": "value", "empty": None}
        result = OpenAIBatchesAPI._get_openai_compatible_batch_metadata(metadata)
        assert "empty" not in result
        assert result == {"key": "value"}

    def test_standard_logging_guardrail_information_excluded(self):
        metadata = {
            "standard_logging_guardrail_information": {"some": "logging_data"},
            "user_key": "keep_me",
        }
        result = OpenAIBatchesAPI._get_openai_compatible_batch_metadata(metadata)
        assert "standard_logging_guardrail_information" not in result
        assert result == {"user_key": "keep_me"}

    def test_non_dict_input_returns_empty_dict(self):
        assert OpenAIBatchesAPI._get_openai_compatible_batch_metadata(None) == {}
        assert OpenAIBatchesAPI._get_openai_compatible_batch_metadata("string") == {}
        assert OpenAIBatchesAPI._get_openai_compatible_batch_metadata(123) == {}

    def test_empty_dict_returns_empty_dict(self):
        assert OpenAIBatchesAPI._get_openai_compatible_batch_metadata({}) == {}

    def test_all_values_in_result_are_strings(self):
        """Every value in the sanitized output must be a plain str."""
        metadata = {
            "_model_armor_response": {"blocked": True},
            "_model_armor_status": "success",
            "queue_time_seconds": 1.23,
            "applied_policies": ["policy-a"],
            "applied_guardrails": ["guardrail-x"],
            "standard_logging_guardrail_information": {"internal": True},
            "user_metadata_key": "user_value",
            "none_field": None,
        }
        result = OpenAIBatchesAPI._get_openai_compatible_batch_metadata(metadata)

        for key, value in result.items():
            assert isinstance(
                value, str
            ), f"metadata[{key!r}] is {type(value)}, not str"

        assert "standard_logging_guardrail_information" not in result
        assert "none_field" not in result
        assert result["_model_armor_status"] == "success"
        assert result["user_metadata_key"] == "user_value"

    def test_result_compatible_with_litellm_batch(self):
        """Verify sanitized metadata can construct a LiteLLMBatch without error."""
        import time

        from litellm.types.utils import LiteLLMBatch

        metadata = {
            "applied_policies": ["my-policy"],
            "applied_guardrails": ["guardrail-a"],
            "_model_armor_response": {"blocked": False},
            "queue_time_seconds": 0.05,
            "user_key": "value",
        }
        sanitized = OpenAIBatchesAPI._get_openai_compatible_batch_metadata(metadata)

        # This would raise an error before the fix
        batch = LiteLLMBatch(
            id="batch_abc123",
            object="batch",
            endpoint="/v1/chat/completions",
            input_file_id="file-123",
            completion_window="24h",
            status="validating",
            created_at=int(time.time()),
            metadata=sanitized,
        )
        assert batch.metadata == sanitized
