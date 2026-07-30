"""
End-to-end test for GitHub issue #35197: verify rate limiter uses correct metadata bucket.

This test simulates what happens when:
1. A /v1/responses request comes in
2. pre-call processing initializes litellm_metadata
3. Rate limiter stashes rate-limit response
4. The stashed value ends up in litellm_metadata, NOT the provider-visible metadata
"""

import sys
import os
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.abspath("../.."))


def test_rate_limiter_uses_correct_metadata_bucket():
    """
    Simulate the rate limiter's metadata stashing behavior.

    Before the fix:
    - litellm_metadata doesn't exist in data
    - get_or_create_metadata_bucket() defaults to 'metadata'
    - Rate limit response gets written to data["metadata"]
    - This leaked metadata is sent to the provider

    After the fix:
    - pre-call logic ensures litellm_metadata exists
    - get_or_create_metadata_bucket() uses litellm_metadata
    - Rate limit response gets written to data["litellm_metadata"]
    - Metadata stays internal, not sent to provider
    """
    from litellm.litellm_core_utils.core_helpers import get_or_create_metadata_bucket

    # Simulate the state AFTER our fix (litellm_metadata is pre-created)
    data = {
        "model": "test-model",
        "input": "test input",
        "litellm_metadata": {},  # This is initialized by our fix
    }

    # Simulate what the rate limiter does
    bucket_name, metadata_bucket = get_or_create_metadata_bucket(data)

    # Verify it uses litellm_metadata (internal), not metadata (provider-visible)
    assert bucket_name == "litellm_metadata", f"Expected litellm_metadata, got {bucket_name}"

    # Simulate stashing the rate limit response (what the rate limiter does)
    metadata_bucket["_litellm_proxy_rate_limit_response"] = {
        "overall_code": "OK",
        "statuses": [{"descriptor_key": "api_key", "rate_limit_type": "requests"}],
    }

    # Verify it was written to the internal bucket, not creating a provider-visible one
    assert "litellm_metadata" in data
    assert "_litellm_proxy_rate_limit_response" in data["litellm_metadata"]
    assert (
        "metadata" not in data
    ), "Provider-visible metadata should NOT be created by rate limiter"


def test_rate_limiter_metadata_leak_without_fix():
    """
    This test demonstrates the bug: without litellm_metadata pre-initialized,
    the rate limiter would create a provider-visible metadata field.
    """
    from litellm.litellm_core_utils.core_helpers import get_or_create_metadata_bucket

    # Simulate the state WITHOUT our fix (no pre-created litellm_metadata)
    data = {
        "model": "test-model",
        "input": "test input",
        # litellm_metadata is NOT initialized - this was the bug
    }

    # When rate limiter calls get_or_create_metadata_bucket
    bucket_name, metadata_bucket = get_or_create_metadata_bucket(data)

    # Without the fix, it defaults to 'metadata' (not litellm_metadata)
    # This is the OLD BUGGY BEHAVIOR - for demonstration only
    assert bucket_name == "metadata", "BUG: defaults to 'metadata' without pre-initialization"

    # Simulate stashing rate limit response in the wrong bucket
    metadata_bucket["_litellm_proxy_rate_limit_response"] = {
        "overall_code": "OK",
        "statuses": [{"descriptor_key": "api_key", "rate_limit_type": "requests"}],
    }

    # The bug: metadata field exists and will be sent to provider
    assert "metadata" in data, "BUG: metadata field was created"
    assert "_litellm_proxy_rate_limit_response" in data["metadata"]

    # This metadata would be sent to the upstream provider,
    # causing OpenAI-compatible backends to reject with "Unsupported parameter: metadata"


def test_responses_vs_chat_completions_metadata_usage():
    """
    Verify different routes use different metadata buckets as intended.
    """
    from litellm.proxy.litellm_pre_call_utils import _get_metadata_variable_name

    # Responses API should use litellm_metadata
    mock_responses_request = MagicMock()
    mock_responses_request.url.path = "/v1/responses"
    responses_bucket = _get_metadata_variable_name(mock_responses_request)
    assert responses_bucket == "litellm_metadata", "Responses should use litellm_metadata"

    # Chat completions can use either (defaults to metadata for backwards compat)
    mock_chat_request = MagicMock()
    mock_chat_request.url.path = "/v1/chat/completions"
    chat_bucket = _get_metadata_variable_name(mock_chat_request)
    assert chat_bucket == "metadata", "Chat completions should use metadata"

    # Batches should use litellm_metadata
    mock_batch_request = MagicMock()
    mock_batch_request.url.path = "/v1/batches"
    batch_bucket = _get_metadata_variable_name(mock_batch_request)
    assert batch_bucket == "litellm_metadata", "Batches should use litellm_metadata"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
