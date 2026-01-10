"""
Unit tests for add_deployment_guardrails_to_metadata function.
Tests for GitHub issue #18363 fix.
"""
import pytest
from unittest.mock import MagicMock

import sys
import os

sys.path.insert(0, os.path.abspath("../.."))

from litellm.proxy.litellm_pre_call_utils import add_deployment_guardrails_to_metadata


class TestAddDeploymentGuardrailsToMetadata:
    """Tests for the add_deployment_guardrails_to_metadata function."""

    def test_adds_guardrails_from_deployment(self):
        """Test that guardrails from deployment's litellm_params are added to metadata."""
        # Setup mock router
        mock_router = MagicMock()
        mock_router.get_model_list.return_value = [
            {
                "model_name": "gpt-4o-mini",
                "litellm_params": {
                    "model": "gpt-4o-mini",
                    "guardrails": ["test_guardrail", "another_guardrail"],
                },
            }
        ]

        # Setup request data with empty metadata
        data = {"model": "gpt-4o-mini", "metadata": {}}

        # Call the function
        add_deployment_guardrails_to_metadata(
            data=data,
            llm_router=mock_router,
            model_name="gpt-4o-mini",
        )

        # Assert guardrails were added
        assert "guardrails" in data["metadata"]
        assert data["metadata"]["guardrails"] == ["test_guardrail", "another_guardrail"]

    def test_merges_with_existing_guardrails(self):
        """Test that deployment guardrails are merged with existing request guardrails."""
        # Setup mock router
        mock_router = MagicMock()
        mock_router.get_model_list.return_value = [
            {
                "model_name": "gpt-4o-mini",
                "litellm_params": {
                    "model": "gpt-4o-mini",
                    "guardrails": ["deployment_guardrail"],
                },
            }
        ]

        # Setup request data with existing guardrails
        data = {
            "model": "gpt-4o-mini",
            "metadata": {"guardrails": ["request_guardrail"]},
        }

        # Call the function
        add_deployment_guardrails_to_metadata(
            data=data,
            llm_router=mock_router,
            model_name="gpt-4o-mini",
        )

        # Assert both guardrails are present (request first, then deployment)
        assert data["metadata"]["guardrails"] == [
            "request_guardrail",
            "deployment_guardrail",
        ]

    def test_no_duplicates_when_merging(self):
        """Test that duplicate guardrails are not added."""
        # Setup mock router
        mock_router = MagicMock()
        mock_router.get_model_list.return_value = [
            {
                "model_name": "gpt-4o-mini",
                "litellm_params": {
                    "model": "gpt-4o-mini",
                    "guardrails": ["same_guardrail", "deployment_only"],
                },
            }
        ]

        # Setup request data with overlapping guardrail
        data = {
            "model": "gpt-4o-mini",
            "metadata": {"guardrails": ["same_guardrail"]},
        }

        # Call the function
        add_deployment_guardrails_to_metadata(
            data=data,
            llm_router=mock_router,
            model_name="gpt-4o-mini",
        )

        # Assert no duplicates
        assert data["metadata"]["guardrails"] == ["same_guardrail", "deployment_only"]

    def test_handles_no_router(self):
        """Test that function handles None router gracefully."""
        data = {"model": "gpt-4o-mini", "metadata": {}}

        # Should not raise
        add_deployment_guardrails_to_metadata(
            data=data,
            llm_router=None,
            model_name="gpt-4o-mini",
        )

        # No guardrails should be added
        assert "guardrails" not in data["metadata"]

    def test_handles_no_model(self):
        """Test that function handles None model gracefully."""
        mock_router = MagicMock()
        data = {"metadata": {}}

        # Should not raise
        add_deployment_guardrails_to_metadata(
            data=data,
            llm_router=mock_router,
            model_name=None,
        )

        # No guardrails should be added
        assert "guardrails" not in data["metadata"]

    def test_handles_no_deployments(self):
        """Test that function handles no deployments found gracefully."""
        mock_router = MagicMock()
        mock_router.get_model_list.return_value = []

        data = {"model": "unknown-model", "metadata": {}}

        # Should not raise
        add_deployment_guardrails_to_metadata(
            data=data,
            llm_router=mock_router,
            model_name="unknown-model",
        )

        # No guardrails should be added
        assert "guardrails" not in data["metadata"]

    def test_handles_deployment_without_guardrails(self):
        """Test that function handles deployments without guardrails gracefully."""
        mock_router = MagicMock()
        mock_router.get_model_list.return_value = [
            {
                "model_name": "gpt-4o-mini",
                "litellm_params": {"model": "gpt-4o-mini"},  # No guardrails
            }
        ]

        data = {"model": "gpt-4o-mini", "metadata": {}}

        # Should not raise
        add_deployment_guardrails_to_metadata(
            data=data,
            llm_router=mock_router,
            model_name="gpt-4o-mini",
        )

        # No guardrails should be added
        assert "guardrails" not in data["metadata"]

    def test_uses_litellm_metadata_when_present(self):
        """Test that function uses litellm_metadata when present (for assistant endpoints)."""
        mock_router = MagicMock()
        mock_router.get_model_list.return_value = [
            {
                "model_name": "gpt-4o-mini",
                "litellm_params": {
                    "model": "gpt-4o-mini",
                    "guardrails": ["test_guardrail"],
                },
            }
        ]

        # Setup request data with litellm_metadata (assistant endpoints)
        data = {"model": "gpt-4o-mini", "litellm_metadata": {}}

        # Call the function
        add_deployment_guardrails_to_metadata(
            data=data,
            llm_router=mock_router,
            model_name="gpt-4o-mini",
        )

        # Assert guardrails were added to litellm_metadata
        assert "guardrails" in data["litellm_metadata"]
        assert data["litellm_metadata"]["guardrails"] == ["test_guardrail"]
