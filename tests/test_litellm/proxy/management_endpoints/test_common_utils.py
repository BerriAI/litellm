"""
Tests for litellm/proxy/management_endpoints/common_utils.py

Covers the fix for GitHub issue #20304:
Empty guardrails/policies arrays sent by the UI should NOT trigger the
enterprise (premium) license check, but should still be applied so that
users can intentionally clear previously-set fields.
"""

from unittest.mock import patch

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

    However, empty collections must still be written into metadata so that
    users can intentionally clear a previously-set field (e.g. removing all
    guardrails by sending `guardrails: []`).
    """

    @patch("litellm.proxy.management_endpoints.common_utils._premium_user_check")
    def test_empty_list_does_not_trigger_premium_check(self, mock_premium_check):
        """Empty lists for premium fields must not trigger the premium check."""
        updated_kv = {
            "team_id": "test-team",
            "guardrails": [],
            "policies": [],
            "logging": [],
        }
        _update_metadata_fields(updated_kv=updated_kv)
        mock_premium_check.assert_not_called()

    @patch("litellm.proxy.management_endpoints.common_utils._premium_user_check")
    def test_empty_list_still_updates_metadata(self, mock_premium_check):
        """
        Empty lists must still be moved into metadata so users can clear
        previously-set fields (e.g. remove all guardrails).
        """
        updated_kv = {
            "team_id": "test-team",
            "guardrails": [],
            "policies": [],
        }
        _update_metadata_fields(updated_kv=updated_kv)
        # The fields should have been moved into metadata
        assert "guardrails" not in updated_kv, (
            "guardrails should be popped from top-level"
        )
        assert "policies" not in updated_kv, (
            "policies should be popped from top-level"
        )
        assert updated_kv["metadata"]["guardrails"] == []
        assert updated_kv["metadata"]["policies"] == []

    @patch("litellm.proxy.management_endpoints.common_utils._premium_user_check")
    def test_empty_dict_does_not_trigger_premium_check(self, mock_premium_check):
        """Empty dicts for premium fields must not trigger the premium check."""
        updated_kv = {
            "team_id": "test-team",
            "secret_manager_settings": {},
        }
        _update_metadata_fields(updated_kv=updated_kv)
        mock_premium_check.assert_not_called()

    @patch("litellm.proxy.management_endpoints.common_utils._premium_user_check")
    def test_empty_dict_still_updates_metadata(self, mock_premium_check):
        """
        Empty dicts must still be moved into metadata so users can clear
        previously-set fields.
        """
        updated_kv = {
            "team_id": "test-team",
            "secret_manager_settings": {},
        }
        _update_metadata_fields(updated_kv=updated_kv)
        assert "secret_manager_settings" not in updated_kv, (
            "secret_manager_settings should be popped from top-level"
        )
        assert updated_kv["metadata"]["secret_manager_settings"] == {}

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
    def test_non_empty_list_updates_metadata(self, mock_premium_check):
        """Non-empty lists should be moved into metadata."""
        updated_kv = {
            "team_id": "test-team",
            "guardrails": ["my-guardrail"],
        }
        _update_metadata_fields(updated_kv=updated_kv)
        assert "guardrails" not in updated_kv
        assert updated_kv["metadata"]["guardrails"] == ["my-guardrail"]

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
