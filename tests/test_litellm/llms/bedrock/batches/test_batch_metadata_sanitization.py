"""
Test that BedrockBatchesConfig._get_openai_compatible_batch_metadata
sanitizes non-string metadata values injected by proxy guardrail hooks.

The OpenAI Batch Pydantic model requires metadata: Dict[str, str].
Proxy hooks (Model Armor, OpenAI Moderations, queue time tracking) inject
dicts, floats, and other non-string values that cause a ValidationError
when constructing LiteLLMBatch. This test suite verifies the sanitization
layer prevents that.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.llms.bedrock.batches.transformation import BedrockBatchesConfig


class TestGetOpenaiCompatibleBatchMetadata:
    """Tests for _get_openai_compatible_batch_metadata."""

    def test_string_values_pass_through_unchanged(self):
        metadata = {"user_key": "user_value", "run_id": "abc123"}
        result = BedrockBatchesConfig._get_openai_compatible_batch_metadata(metadata)
        assert result == {"user_key": "user_value", "run_id": "abc123"}

    def test_dict_values_serialized_to_json_string(self):
        metadata = {
            "_model_armor_response": {
                "sanitizationResult": {"filterMatchState": "MATCH_FOUND"}
            }
        }
        result = BedrockBatchesConfig._get_openai_compatible_batch_metadata(metadata)
        assert "_model_armor_response" in result
        assert isinstance(result["_model_armor_response"], str)
        assert "MATCH_FOUND" in result["_model_armor_response"]

    def test_float_values_serialized_to_string(self):
        metadata = {"queue_time_seconds": 0.5}
        result = BedrockBatchesConfig._get_openai_compatible_batch_metadata(metadata)
        assert result == {"queue_time_seconds": "0.5"}

    def test_none_values_excluded(self):
        metadata = {"key": "value", "empty": None}
        result = BedrockBatchesConfig._get_openai_compatible_batch_metadata(metadata)
        assert "empty" not in result
        assert result == {"key": "value"}

    def test_standard_logging_guardrail_information_excluded(self):
        metadata = {
            "standard_logging_guardrail_information": {"some": "logging_data"},
            "user_key": "keep_me",
        }
        result = BedrockBatchesConfig._get_openai_compatible_batch_metadata(metadata)
        assert "standard_logging_guardrail_information" not in result
        assert result == {"user_key": "keep_me"}

    def test_non_dict_input_returns_empty_dict(self):
        assert BedrockBatchesConfig._get_openai_compatible_batch_metadata(None) == {}
        assert BedrockBatchesConfig._get_openai_compatible_batch_metadata("string") == {}
        assert BedrockBatchesConfig._get_openai_compatible_batch_metadata(123) == {}

    def test_empty_dict_returns_empty_dict(self):
        assert BedrockBatchesConfig._get_openai_compatible_batch_metadata({}) == {}

    def test_mixed_metadata_from_guardrails(self):
        """Simulate real metadata contaminated by proxy guardrails."""
        metadata = {
            "_model_armor_response": {"sanitizationResult": {"key": "val"}},
            "_model_armor_status": "success",
            "_openai_moderation_response": {"id": "mod-123", "flagged": False},
            "queue_time_seconds": 1.23,
            "headers": {"Authorization": "Bearer sk-xxx"},
            "standard_logging_guardrail_information": {"internal": True},
            "user_metadata_key": "user_value",
            "none_field": None,
        }
        result = BedrockBatchesConfig._get_openai_compatible_batch_metadata(metadata)

        # All values must be strings
        for key, value in result.items():
            assert isinstance(value, str), f"metadata[{key!r}] is {type(value)}, not str"

        # Excluded keys
        assert "standard_logging_guardrail_information" not in result
        assert "none_field" not in result

        # Preserved keys
        assert result["_model_armor_status"] == "success"
        assert result["user_metadata_key"] == "user_value"

    def test_result_compatible_with_litellm_batch(self):
        """Verify sanitized metadata can construct a LiteLLMBatch without error."""
        import time

        from litellm.types.utils import LiteLLMBatch

        metadata = {
            "_model_armor_response": {"blocked": True},
            "queue_time_seconds": 0.05,
            "user_key": "value",
        }
        sanitized = BedrockBatchesConfig._get_openai_compatible_batch_metadata(metadata)

        # This would raise ValidationError before the fix
        batch = LiteLLMBatch(
            id="arn:aws:bedrock:us-east-1:123:model-invocation-job/test",
            object="batch",
            endpoint="/v1/chat/completions",
            input_file_id="file-123",
            completion_window="24h",
            status="validating",
            created_at=int(time.time()),
            metadata=sanitized,
        )
        assert batch.metadata == sanitized
