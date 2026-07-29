"""
Tests for litellm/proxy/management_endpoints/common_utils.py

Specifically tests that _update_metadata_fields does not trigger premium
user checks when premium fields are present but empty.

Related: https://github.com/BerriAI/litellm/issues/20534
"""

from unittest.mock import patch

import pytest

from litellm.proxy.management_endpoints.common_utils import (
    _has_non_empty_value,
    _update_metadata_fields,
)


class TestHasNonEmptyValue:
    """Tests for the _has_non_empty_value helper."""

    def test_none_is_empty(self):
        assert _has_non_empty_value(None) is False

    def test_empty_list_is_empty(self):
        assert _has_non_empty_value([]) is False

    def test_empty_string_is_empty(self):
        assert _has_non_empty_value("") is False

    def test_blank_string_is_empty(self):
        assert _has_non_empty_value("   ") is False

    def test_non_empty_list_has_value(self):
        assert _has_non_empty_value(["policy-a"]) is True

    def test_non_empty_string_has_value(self):
        assert _has_non_empty_value("30d") is True

    def test_dict_has_value(self):
        assert _has_non_empty_value({"key": "val"}) is True

    def test_empty_dict_has_value(self):
        # empty dict is not None/list/str, so it counts as non-empty
        assert _has_non_empty_value({}) is True


class TestUpdateMetadataFieldsPremiumCheck:
    """
    Tests that _update_metadata_fields skips premium user checks for empty
    values but still enforces them for real values.

    Issue: The UI sends the full form on every team update, including premium
    fields like `policies: []`. The backend was treating these empty values
    as premium feature usage and returning 403.
    """

    @patch(
        "litellm.proxy.management_endpoints.common_utils._premium_user_check",
        side_effect=Exception("Should not be called"),
    )
    def test_empty_policies_skips_premium_check(self, mock_check):
        """policies: [] should NOT trigger premium user check."""
        updated_kv = {
            "team_id": "team-123",
            "team_alias": "my-team",
            "policies": [],
        }
        _update_metadata_fields(updated_kv)
        mock_check.assert_not_called()

    @patch(
        "litellm.proxy.management_endpoints.common_utils._premium_user_check",
        side_effect=Exception("Should not be called"),
    )
    def test_empty_guardrails_skips_premium_check(self, mock_check):
        """guardrails: [] should NOT trigger premium user check."""
        updated_kv = {
            "team_id": "team-123",
            "guardrails": [],
        }
        _update_metadata_fields(updated_kv)
        mock_check.assert_not_called()

    @patch(
        "litellm.proxy.management_endpoints.common_utils._premium_user_check",
        side_effect=Exception("Should not be called"),
    )
    def test_empty_string_team_member_key_duration_skips_premium_check(
        self, mock_check
    ):
        """team_member_key_duration: '' should NOT trigger premium user check."""
        updated_kv = {
            "team_id": "team-123",
            "team_member_key_duration": "",
        }
        _update_metadata_fields(updated_kv)
        mock_check.assert_not_called()

    @patch(
        "litellm.proxy.management_endpoints.common_utils._premium_user_check",
        side_effect=Exception("Should not be called"),
    )
    def test_full_ui_payload_with_empty_premium_fields_skips_premium_check(
        self, mock_check
    ):
        """A realistic UI payload with all empty premium fields should not 403."""
        updated_kv = {
            "team_id": "team-123",
            "team_alias": "renamed-team",
            "models": ["gpt-4o"],
            "max_budget": 200,
            "policies": [],
            "guardrails": [],
            "logging": [],
            "team_member_key_duration": "",
            "prompts": [],
        }
        _update_metadata_fields(updated_kv)
        mock_check.assert_not_called()

    @patch(
        "litellm.proxy.management_endpoints.common_utils._premium_user_check",
    )
    def test_non_empty_policies_triggers_premium_check(self, mock_check):
        """policies: ['real-policy'] SHOULD trigger premium user check."""
        updated_kv = {
            "team_id": "team-123",
            "policies": ["real-policy"],
        }
        _update_metadata_fields(updated_kv)
        mock_check.assert_called()

    @patch(
        "litellm.proxy.management_endpoints.common_utils._premium_user_check",
    )
    def test_non_empty_guardrails_triggers_premium_check(self, mock_check):
        """guardrails: ['my-guardrail'] SHOULD trigger premium user check."""
        updated_kv = {
            "team_id": "team-123",
            "guardrails": ["my-guardrail"],
        }
        _update_metadata_fields(updated_kv)
        mock_check.assert_called()

    @patch(
        "litellm.proxy.management_endpoints.common_utils._premium_user_check",
    )
    def test_non_empty_team_member_key_duration_triggers_premium_check(
        self, mock_check
    ):
        """team_member_key_duration: '30d' SHOULD trigger premium user check."""
        updated_kv = {
            "team_id": "team-123",
            "team_member_key_duration": "30d",
        }
        _update_metadata_fields(updated_kv)
        mock_check.assert_called()
