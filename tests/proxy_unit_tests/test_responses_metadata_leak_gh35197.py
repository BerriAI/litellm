"""
Regression test for GitHub issue #35197: /v1/responses leaks rate-limiter metadata to upstream.

When a request to /v1/responses has rate limits enabled, the rate limiter should stash
internal state in litellm_metadata (proxy-internal), not metadata (provider-visible).
"""

import os
import sys
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.abspath("../.."))


class TestResponsesMetadataLeak:
    """Test that /v1/responses does not leak rate-limiter metadata to upstream."""

    def test_responses_route_uses_litellm_metadata(self):
        """
        Verify that _get_metadata_variable_name correctly identifies
        /v1/responses as a route that should use litellm_metadata.
        """
        from litellm.proxy.litellm_pre_call_utils import _get_metadata_variable_name

        # Mock request for /v1/responses
        mock_request = MagicMock()
        mock_request.url.path = "/v1/responses"

        # This is what determines which metadata bucket to use
        metadata_var_name = _get_metadata_variable_name(mock_request)

        assert (
            metadata_var_name == "litellm_metadata"
        ), f"Expected 'litellm_metadata' for /v1/responses, got '{metadata_var_name}'"

    def test_batch_routes_use_litellm_metadata(self):
        """
        Verify that /v1/batches also uses litellm_metadata.
        """
        from litellm.proxy.litellm_pre_call_utils import _get_metadata_variable_name

        mock_request = MagicMock()
        mock_request.url.path = "/v1/batches"

        metadata_var_name = _get_metadata_variable_name(mock_request)

        assert (
            metadata_var_name == "litellm_metadata"
        ), f"Expected 'litellm_metadata' for /v1/batches, got '{metadata_var_name}'"

    def test_chat_completions_uses_metadata(self):
        """
        Verify that /v1/chat/completions uses 'metadata' (backwards compatibility).
        """
        from litellm.proxy.litellm_pre_call_utils import _get_metadata_variable_name

        mock_request = MagicMock()
        mock_request.url.path = "/v1/chat/completions"

        metadata_var_name = _get_metadata_variable_name(mock_request)

        assert (
            metadata_var_name == "metadata"
        ), f"Expected 'metadata' for /v1/chat/completions, got '{metadata_var_name}'"

    def test_litellm_metadata_bucket_selection(self):
        """
        Test the core logic: get_or_create_metadata_bucket should use
        the correct bucket name based on whether litellm_metadata exists.
        """
        from litellm.litellm_core_utils.core_helpers import get_or_create_metadata_bucket

        # Test 1: When litellm_metadata is already in the data, use it
        data_with_litellm_metadata = {"litellm_metadata": {"existing": "value"}}
        bucket_name, bucket = get_or_create_metadata_bucket(data_with_litellm_metadata)

        assert bucket_name == "litellm_metadata"
        assert bucket["existing"] == "value"

        # Test 2: When litellm_metadata doesn't exist, defaults to metadata
        # (This is the old behavior that caused the bug)
        data_without_litellm_metadata = {}
        bucket_name, bucket = get_or_create_metadata_bucket(data_without_litellm_metadata)

        assert bucket_name == "metadata"

        # Test 3: The fix: if we pre-create litellm_metadata (as our fix does),
        # then get_or_create_metadata_bucket will use it instead of metadata
        data_prefilled = {"litellm_metadata": {}}
        bucket_name, bucket = get_or_create_metadata_bucket(data_prefilled)

        assert bucket_name == "litellm_metadata"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
