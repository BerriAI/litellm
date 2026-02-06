"""
Tests for litellm/proxy/management_endpoints/common_utils.py

Covers the fix for GitHub issue #20304:
Empty guardrails/policies arrays sent by the UI should NOT trigger the
enterprise (premium) license check.
"""

import os
import sys
from unittest.mock import patch

import pytest
from fastapi import HTTPException

sys.path.insert(0, os.path.abspath("../../../"))

from litellm.proxy._types import (
    LiteLLM_ManagementEndpoint_MetadataFields_Premium,
)
from litellm.proxy.management_endpoints.common_utils import (
    _update_metadata_fields,
)


class TestUpdateMetadataFieldsEmptyCollections:
    """
    Regression tests for issue #20304.

    The UI sends empty arrays (`[]`) for enterprise-only fields like
    guardrails, policies, and logging even when the user hasn't configured
    these features.  The backend must not treat empty collections as an
    intent to use the feature, and therefore must not trigger the premium
    license check.
    """

    @patch("litellm.proxy.management_endpoints.common_utils._premium_user_check")
    def test_empty_list_does_not_trigger_premium_check(self, mock_premium_check):
        """Empty lists for premium fields should be silently ignored."""
        updated_kv = {
            "team_id": "test-team",
            "guardrails": [],
            "policies": [],
            "logging": [],
        }
        _update_metadata_fields(updated_kv=updated_kv)
        mock_premium_check.assert_not_called()

    @patch("litellm.proxy.management_endpoints.common_utils._premium_user_check")
    def test_empty_dict_does_not_trigger_premium_check(self, mock_premium_check):
        """Empty dicts for premium fields should be silently ignored."""
        updated_kv = {
            "team_id": "test-team",
            "secret_manager_settings": {},
        }
        _update_metadata_fields(updated_kv=updated_kv)
        mock_premium_check.assert_not_called()

    @patch("litellm.proxy.management_endpoints.common_utils._premium_user_check")
    def test_none_value_does_not_trigger_premium_check(self, mock_premium_check):
        """None values for premium fields should be silently ignored."""
        updated_kv = {
            "team_id": "test-team",
            "guardrails": None,
            "policies": None,
        }
        _update_metadata_fields(updated_kv=updated_kv)
        mock_premium_check.assert_not_called()

    @patch("litellm.proxy.management_endpoints.common_utils._premium_user_check")
    def test_absent_fields_do_not_trigger_premium_check(self, mock_premium_check):
        """Fields not present in the dict should not trigger premium check."""
        updated_kv = {
            "team_id": "test-team",
            "team_alias": "example-team",
        }
        _update_metadata_fields(updated_kv=updated_kv)
        mock_premium_check.assert_not_called()

    @patch("litellm.proxy.management_endpoints.common_utils._premium_user_check")
    def test_non_empty_list_triggers_premium_check(self, mock_premium_check):
        """Non-empty lists for premium fields should trigger the premium check."""
        updated_kv = {
            "team_id": "test-team",
            "guardrails": ["my-guardrail"],
        }
        _update_metadata_fields(updated_kv=updated_kv)
        mock_premium_check.assert_called()

    @patch("litellm.proxy.management_endpoints.common_utils._premium_user_check")
    def test_non_empty_value_triggers_premium_check(self, mock_premium_check):
        """Non-empty string values for premium fields should trigger the premium check."""
        updated_kv = {
            "team_id": "test-team",
            "tags": ["production"],
        }
        _update_metadata_fields(updated_kv=updated_kv)
        mock_premium_check.assert_called()

    @patch("litellm.proxy.management_endpoints.common_utils._premium_user_check")
    def test_ui_typical_payload_does_not_trigger_premium_check(self, mock_premium_check):
        """
        Simulate the exact payload the UI sends when no enterprise features
        are configured.  This must NOT trigger the premium check.
        """
        # This is the payload structure the UI sends (from issue #20304)
        updated_kv = {
            "team_id": "67848772-1a8b-4343-938c-17e60f1db860",
            "team_alias": "example-team",
            "models": ["gpt-4"],
            "metadata": {
                "guardrails": [],
                "logging": [],
            },
            "policies": [],
        }
        _update_metadata_fields(updated_kv=updated_kv)
        mock_premium_check.assert_not_called()
