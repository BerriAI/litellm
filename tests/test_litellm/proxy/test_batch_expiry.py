"""
Tests for batch output_expires_after passthrough and team-level expiry enforcement.
"""

import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

import litellm
from litellm.types.llms.openai import CreateBatchRequest


class TestCreateBatchOutputExpiresAfterPassthrough:
    """Verify output_expires_after flows through create_batch to the provider."""

    def test_output_expires_after_included_in_request(self):
        """When output_expires_after is provided, it reaches the openai batches instance."""
        captured = {}

        original_create = None

        def capturing_create(**kwargs):
            captured.update(kwargs)
            mock_response = MagicMock()
            mock_response.id = "batch_123"
            return mock_response

        with patch(
            "litellm.batches.main.openai_batches_instance"
        ) as mock_instance:
            mock_instance.create_batch.side_effect = capturing_create
            litellm.create_batch(
                completion_window="24h",
                endpoint="/v1/chat/completions",
                input_file_id="file-abc123",
                output_expires_after={"anchor": "created_at", "seconds": 86400},
                custom_llm_provider="openai",
            )

        create_batch_data = captured["create_batch_data"]
        assert create_batch_data["output_expires_after"] == {
            "anchor": "created_at",
            "seconds": 86400,
        }

    def test_output_expires_after_absent_when_not_provided(self):
        """Backward compat: output_expires_after not in request when omitted."""
        captured = {}

        def capturing_create(**kwargs):
            captured.update(kwargs)
            mock_response = MagicMock()
            mock_response.id = "batch_123"
            return mock_response

        with patch(
            "litellm.batches.main.openai_batches_instance"
        ) as mock_instance:
            mock_instance.create_batch.side_effect = capturing_create
            litellm.create_batch(
                completion_window="24h",
                endpoint="/v1/chat/completions",
                input_file_id="file-abc123",
                custom_llm_provider="openai",
            )

        create_batch_data = captured["create_batch_data"]
        assert "output_expires_after" not in create_batch_data
