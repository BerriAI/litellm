"""
Tests for LIT-3256: v1/batches fails when policies are applied to the request.

When a global policy (guardrail) is attached to a request, LiteLLM's pre-call
hooks add ``applied_policies`` (a list) and ``guardrails`` (a list) to
``data["metadata"]``.  The OpenAI batch API only accepts ``Dict[str, str]``
for the ``metadata`` field, so those list values cause a 400:

    Invalid type for 'metadata.applied_policies':
        expected a string, but got an array instead.

The fix: ``_sanitize_openai_batch_metadata`` converts list values to
comma-separated strings and drops complex objects before the metadata is
forwarded to the provider.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from litellm.proxy.batches_endpoints.endpoints import _sanitize_openai_batch_metadata


# ---------------------------------------------------------------------------
# Unit tests for the sanitization helper
# ---------------------------------------------------------------------------


class TestSanitizeOpenAIBatchMetadata:
    def test_none_input_returns_none(self):
        assert _sanitize_openai_batch_metadata(None) is None

    def test_empty_dict_returns_none(self):
        assert _sanitize_openai_batch_metadata({}) is None

    def test_string_values_preserved(self):
        result = _sanitize_openai_batch_metadata({"key": "value", "another": "thing"})
        assert result == {"key": "value", "another": "thing"}

    def test_list_applied_policies_converted_to_csv(self):
        """The main regression: applied_policies list → comma-separated string."""
        result = _sanitize_openai_batch_metadata(
            {"applied_policies": ["policy_a", "policy_b", "policy_c"]}
        )
        assert result == {"applied_policies": "policy_a,policy_b,policy_c"}

    def test_list_guardrails_converted_to_csv(self):
        result = _sanitize_openai_batch_metadata(
            {"guardrails": ["guardrail_1", "guardrail_2"]}
        )
        assert result == {"guardrails": "guardrail_1,guardrail_2"}

    def test_single_item_list_converted_to_string(self):
        result = _sanitize_openai_batch_metadata({"applied_policies": ["only_one"]})
        assert result == {"applied_policies": "only_one"}

    def test_empty_list_converted_to_empty_string(self):
        result = _sanitize_openai_batch_metadata({"applied_policies": []})
        assert result == {"applied_policies": ""}

    def test_complex_objects_dropped(self):
        """user_api_key_auth and similar Pydantic objects must not leak to OpenAI."""
        mock_auth = MagicMock()
        result = _sanitize_openai_batch_metadata(
            {
                "user_key": "some-string-value",
                "user_api_key_auth": mock_auth,
            }
        )
        assert result == {"user_key": "some-string-value"}
        assert "user_api_key_auth" not in result

    def test_dict_values_dropped(self):
        result = _sanitize_openai_batch_metadata({"nested": {"a": 1}, "flat": "ok"})
        assert result == {"flat": "ok"}
        assert "nested" not in result

    def test_mixed_metadata_realistic_scenario(self):
        """
        Simulate the full metadata dict that LiteLLM pre-call hooks inject
        when a global policy is attached to a request.
        """
        mock_auth = MagicMock()  # UserAPIKeyAuth
        raw_metadata = {
            # User-provided metadata (should survive)
            "job_name": "nightly_embeddings",
            "environment": "production",
            # Added by policy engine (list → csv)
            "applied_policies": ["content_filter_v2", "pii_redact"],
            # Added by guardrail hooks (list → csv)
            "guardrails": ["bedrock_guardrail", "custom_guard"],
            # Internal LiteLLM tracking (complex object → dropped)
            "user_api_key_auth": mock_auth,
        }

        result = _sanitize_openai_batch_metadata(raw_metadata)

        assert result is not None
        assert result["job_name"] == "nightly_embeddings"
        assert result["environment"] == "production"
        assert result["applied_policies"] == "content_filter_v2,pii_redact"
        assert result["guardrails"] == "bedrock_guardrail,custom_guard"
        assert "user_api_key_auth" not in result

        # All retained values must be strings (OpenAI contract)
        for v in result.values():
            assert isinstance(v, str), f"Expected str, got {type(v)} for value {v!r}"

    def test_returns_none_when_only_complex_objects(self):
        mock_auth = MagicMock()
        result = _sanitize_openai_batch_metadata({"user_api_key_auth": mock_auth})
        assert result is None

    def test_integer_values_in_list_converted_to_str(self):
        result = _sanitize_openai_batch_metadata({"tags": [1, 2, 3]})
        assert result == {"tags": "1,2,3"}


# ---------------------------------------------------------------------------
# End-to-end scenario: simulate pre-call hooks injecting metadata then
# verify the sanitisation step that runs just before LiteLLMBatchCreateRequest
# ---------------------------------------------------------------------------


def test_full_pre_call_metadata_scenario():
    """
    Simulate the exact data dict that LiteLLM proxy builds after pre-call hooks
    run (policy engine + guardrail hooks + auth tracking) and verify that
    _sanitize_openai_batch_metadata produces a Dict[str, str] safe for OpenAI.
    """
    from litellm.proxy.batches_endpoints.endpoints import (
        _sanitize_openai_batch_metadata,
    )

    mock_auth = MagicMock()  # would be UserAPIKeyAuth

    # This is what data["metadata"] looks like after pre-call processing when
    # a global policy is attached and applied_policies gets injected as a list.
    data = {
        "input_file_id": "file-abc",
        "endpoint": "/v1/responses",
        "completion_window": "24h",
        "metadata": {
            # user-supplied
            "job_name": "embed_nightly",
            # injected by policy engine as list (the bug)
            "applied_policies": ["content_filter_v2", "pii_redact"],
            # injected by guardrail hooks as list
            "guardrails": ["bedrock_guardrail"],
            # injected by auth layer as a Pydantic object
            "user_api_key_auth": mock_auth,
        },
    }

    # Mimic the endpoint code path
    data["metadata"] = _sanitize_openai_batch_metadata(data["metadata"])

    meta = data["metadata"]
    assert meta is not None

    # User-supplied strings are preserved
    assert meta["job_name"] == "embed_nightly"

    # Lists converted to CSV — no longer arrays
    assert meta["applied_policies"] == "content_filter_v2,pii_redact"
    assert meta["guardrails"] == "bedrock_guardrail"

    # Complex objects dropped — not leaked to OpenAI
    assert "user_api_key_auth" not in meta

    # Every value is a plain string (OpenAI API contract)
    for k, v in meta.items():
        assert isinstance(v, str), f"metadata[{k!r}] must be str, got {type(v)}"
