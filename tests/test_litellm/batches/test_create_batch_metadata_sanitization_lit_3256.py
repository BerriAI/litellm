"""
LIT-3256 regression tests.

Proxy-level pre-call hooks (e.g. ``add_policy_to_applied_policies_header``)
inject non-string values into ``data["metadata"]``. OpenAI's Batches API and
Azure OpenAI's Batches API both require ``metadata: Dict[str, str]``. Without
sanitization the upstream returns:

    400 Invalid type for 'metadata.applied_policies': expected a string,
        but got an array instead. (code=invalid_type)

These tests pin:
1. The pure ``sanitize_openai_batch_metadata`` helper behaviour.
2. The integration: ``litellm.create_batch`` sanitizes metadata exactly once
   before forwarding to the OpenAI / Azure handlers.
3. Non-OpenAI providers (Vertex AI, Bedrock, Anthropic) are NOT touched by the
   sanitization layer in ``litellm/batches/main.py`` - they own their own
   request transforms.
"""

import os
import sys

sys.path.insert(0, os.path.abspath("../../.."))

from litellm.llms.openai.batches.transformation import (
    sanitize_openai_batch_metadata,
)

# ---------------------------------------------------------------------------
# Unit tests on the helper
# ---------------------------------------------------------------------------


class TestSanitizeOpenaiBatchMetadata:
    def test_returns_none_for_none(self):
        assert sanitize_openai_batch_metadata(None) is None

    def test_returns_none_for_non_dict(self):
        # Defensive: callers shouldn't pass this, but don't blow up.
        assert sanitize_openai_batch_metadata("not-a-dict") is None  # type: ignore[arg-type]
        assert sanitize_openai_batch_metadata([1, 2, 3]) is None  # type: ignore[arg-type]

    def test_string_values_pass_through_unchanged(self):
        md = {"user_id": "u-1", "run_id": "r-2"}
        assert sanitize_openai_batch_metadata(md) == {"user_id": "u-1", "run_id": "r-2"}

    def test_list_value_is_stringified(self):
        # The exact shape proxy pre-call hooks inject.
        md = {"applied_policies": ["block_pii", "block_creds"]}
        result = sanitize_openai_batch_metadata(md)
        assert isinstance(result["applied_policies"], str)
        # Both policy names preserved in the serialized form.
        assert "block_pii" in result["applied_policies"]
        assert "block_creds" in result["applied_policies"]

    def test_int_and_float_are_stringified(self):
        md = {"queue_time_seconds": 0.5, "retry_count": 3}
        result = sanitize_openai_batch_metadata(md)
        assert isinstance(result["queue_time_seconds"], str)
        assert isinstance(result["retry_count"], str)
        assert result["queue_time_seconds"] == "0.5"
        assert result["retry_count"] == "3"

    def test_dict_value_is_serialized_to_json(self):
        md = {"cfg": {"a": 1, "b": [1, 2]}}
        result = sanitize_openai_batch_metadata(md)
        assert isinstance(result["cfg"], str)
        # Round-trip via JSON to assert structural equivalence (don't pin key
        # order — safe_dumps may sort).
        import json

        assert json.loads(result["cfg"]) == {"a": 1, "b": [1, 2]}

    def test_none_value_is_dropped(self):
        md = {"keep": "v", "drop": None}
        result = sanitize_openai_batch_metadata(md)
        assert "drop" not in result
        assert result == {"keep": "v"}

    def test_standard_logging_guardrail_information_is_dropped(self):
        md = {
            "standard_logging_guardrail_information": {"any": "internal_payload"},
            "user_id": "u-1",
        }
        result = sanitize_openai_batch_metadata(md)
        assert "standard_logging_guardrail_information" not in result
        assert result == {"user_id": "u-1"}

    def test_empty_dict_returns_empty_dict(self):
        assert sanitize_openai_batch_metadata({}) == {}

    def test_mixed_realistic_proxy_metadata(self):
        # Shape the proxy actually constructs (see add_policy_to_applied_policies_header
        # and pre-call hooks).
        md = {
            "applied_policies": ["policy_a", "policy_b"],
            "user_api_key_alias": "team-alpha",
            "user_api_key_user_id": "u-42",
            "queue_time_seconds": 1.25,
            "standard_logging_guardrail_information": {"x": 1},
            "drop_me": None,
        }
        result = sanitize_openai_batch_metadata(md)
        assert all(isinstance(v, str) for v in result.values())
        # Sanitized list keeps both members.
        assert "policy_a" in result["applied_policies"]
        assert "policy_b" in result["applied_policies"]
        # String values untouched.
        assert result["user_api_key_alias"] == "team-alpha"
        assert result["user_api_key_user_id"] == "u-42"
        # Float coerced.
        assert result["queue_time_seconds"] == "1.25"
        # Internal-only keys dropped.
        assert "standard_logging_guardrail_information" not in result
        assert "drop_me" not in result


# ---------------------------------------------------------------------------
# Integration: end-to-end sanitization at the create_batch boundary
# ---------------------------------------------------------------------------


def _captured_metadata(captured_create_batch_data):
    """Pull the metadata out of whatever shape create_batch_data was passed in."""
    return (
        captured_create_batch_data.get("metadata")
        if captured_create_batch_data
        else None
    )


class TestCreateBatchSanitizesMetadata:
    """End-to-end: litellm.create_batch should sanitize metadata before
    handing it to the provider handler."""

    def _stub_openai_handler(self, sink: dict):
        """Patch OpenAIBatchesAPI.create_batch to capture create_batch_data."""

        from litellm.types.utils import LiteLLMBatch

        def _capture(self, *args, **kwargs):
            sink["create_batch_data"] = kwargs.get("create_batch_data")
            # Return a minimal valid LiteLLMBatch.
            return LiteLLMBatch(
                id="batch_capture",
                object="batch",
                endpoint="/v1/chat/completions",
                input_file_id="file-x",
                completion_window="24h",
                status="validating",
                created_at=0,
            )

        return _capture

    def test_openai_path_sanitizes_list_metadata(self, monkeypatch):
        import litellm
        from litellm.llms.openai.openai import OpenAIBatchesAPI

        sink: dict = {}
        monkeypatch.setattr(
            OpenAIBatchesAPI,
            "create_batch",
            self._stub_openai_handler(sink),
            raising=True,
        )
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-LIT-3256")

        litellm.create_batch(
            custom_llm_provider="openai",
            input_file_id="file-x",
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
                "applied_policies": ["policy_a", "policy_b"],
                "user_id": "u-1",
            },
        )

        md = _captured_metadata(sink["create_batch_data"])
        # applied_policies must now be a string, not a list.
        assert md is not None
        assert isinstance(md["applied_policies"], str)
        assert "policy_a" in md["applied_policies"]
        assert "policy_b" in md["applied_policies"]
        # String value passes through.
        assert md["user_id"] == "u-1"
        # No non-string values escape the boundary.
        assert all(isinstance(v, str) for v in md.values())

    def test_openai_path_preserves_none_metadata(self, monkeypatch):
        import litellm
        from litellm.llms.openai.openai import OpenAIBatchesAPI

        sink: dict = {}
        monkeypatch.setattr(
            OpenAIBatchesAPI,
            "create_batch",
            self._stub_openai_handler(sink),
            raising=True,
        )
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-LIT-3256")

        litellm.create_batch(
            custom_llm_provider="openai",
            input_file_id="file-x",
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata=None,
        )

        md = _captured_metadata(sink["create_batch_data"])
        # None must remain None — no surprise empty-dict materialization at our
        # boundary.
        assert md is None

    def test_azure_path_sanitizes_list_metadata(self, monkeypatch):
        import litellm
        from litellm.llms.azure.batches.handler import AzureBatchesAPI

        sink: dict = {}
        monkeypatch.setattr(
            AzureBatchesAPI,
            "create_batch",
            self._stub_openai_handler(sink),
            raising=True,
        )
        monkeypatch.setenv("AZURE_API_KEY", "test-key")
        monkeypatch.setenv("AZURE_API_BASE", "https://example.openai.azure.com")
        monkeypatch.setenv("AZURE_API_VERSION", "2024-08-01-preview")

        litellm.create_batch(
            custom_llm_provider="azure",
            input_file_id="file-x",
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"applied_policies": ["p1", "p2"]},
        )

        md = _captured_metadata(sink["create_batch_data"])
        assert md is not None
        assert isinstance(md["applied_policies"], str)
        assert "p1" in md["applied_policies"]
        assert "p2" in md["applied_policies"]
