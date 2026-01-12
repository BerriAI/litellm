"""
Unit tests for auth_utils functions related to rate limiting.
"""

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.auth_utils import (
    get_key_model_rpm_limit,
    get_key_model_tpm_limit,
)


class TestGetKeyModelRpmLimit:
    """Tests for get_key_model_rpm_limit function."""

    def test_returns_key_metadata_when_present(self):
        """Key metadata takes priority over team metadata."""
        user_api_key_dict = UserAPIKeyAuth(
            api_key="sk-123",
            metadata={"model_rpm_limit": {"gpt-4": 100}},
            team_metadata={"model_rpm_limit": {"gpt-4": 50}},
        )
        result = get_key_model_rpm_limit(user_api_key_dict)
        assert result == {"gpt-4": 100}

    def test_falls_back_to_team_metadata_when_key_has_other_metadata(self):
        """Should fall back to team metadata when key metadata exists but has no model_rpm_limit."""
        user_api_key_dict = UserAPIKeyAuth(
            api_key="sk-123",
            metadata={
                "some_other_key": "value"
            },  # Has metadata, but not model_rpm_limit
            team_metadata={"model_rpm_limit": {"gpt-4": 50}},
        )
        result = get_key_model_rpm_limit(user_api_key_dict)
        assert result == {"gpt-4": 50}

    def test_extracts_from_model_max_budget(self):
        """Should extract rpm_limit from model_max_budget when metadata is empty."""
        user_api_key_dict = UserAPIKeyAuth(
            api_key="sk-123",
            model_max_budget={
                "gpt-4": {"rpm_limit": 100, "tpm_limit": 1000},
                "gpt-3.5-turbo": {"rpm_limit": 200},
            },
        )
        result = get_key_model_rpm_limit(user_api_key_dict)
        assert result == {"gpt-4": 100, "gpt-3.5-turbo": 200}

    def test_skips_models_without_rpm_limit(self):
        """Should skip models that don't have rpm_limit in model_max_budget."""
        user_api_key_dict = UserAPIKeyAuth(
            api_key="sk-123",
            model_max_budget={
                "gpt-4": {"rpm_limit": 100},
                "gpt-3.5-turbo": {"tpm_limit": 1000},  # No rpm_limit
            },
        )
        result = get_key_model_rpm_limit(user_api_key_dict)
        assert result == {"gpt-4": 100}

    def test_returns_none_when_no_limits_configured(self):
        """Should return None when no rate limits are configured."""
        user_api_key_dict = UserAPIKeyAuth(api_key="sk-123")
        result = get_key_model_rpm_limit(user_api_key_dict)
        assert result is None


class TestGetKeyModelTpmLimit:
    """Tests for get_key_model_tpm_limit function."""

    def test_returns_key_metadata_when_present(self):
        """Key metadata takes priority over team metadata."""
        user_api_key_dict = UserAPIKeyAuth(
            api_key="sk-123",
            metadata={"model_tpm_limit": {"gpt-4": 10000}},
            team_metadata={"model_tpm_limit": {"gpt-4": 5000}},
        )
        result = get_key_model_tpm_limit(user_api_key_dict)
        assert result == {"gpt-4": 10000}

    def test_falls_back_to_team_metadata_when_key_has_other_metadata(self):
        """Should fall back to team metadata when key metadata exists but has no model_tpm_limit."""
        user_api_key_dict = UserAPIKeyAuth(
            api_key="sk-123",
            metadata={
                "some_other_key": "value"
            },  # Has metadata, but not model_tpm_limit
            team_metadata={"model_tpm_limit": {"gpt-4": 5000}},
        )
        result = get_key_model_tpm_limit(user_api_key_dict)
        assert result == {"gpt-4": 5000}

    def test_extracts_from_model_max_budget(self):
        """Should extract tpm_limit from model_max_budget when metadata is empty."""
        user_api_key_dict = UserAPIKeyAuth(
            api_key="sk-123",
            model_max_budget={
                "gpt-4": {"tpm_limit": 10000, "rpm_limit": 100},
                "gpt-3.5-turbo": {"tpm_limit": 20000},
            },
        )
        result = get_key_model_tpm_limit(user_api_key_dict)
        assert result == {"gpt-4": 10000, "gpt-3.5-turbo": 20000}

    def test_skips_models_without_tpm_limit(self):
        """Should skip models that don't have tpm_limit in model_max_budget."""
        user_api_key_dict = UserAPIKeyAuth(
            api_key="sk-123",
            model_max_budget={
                "gpt-4": {"tpm_limit": 10000},
                "gpt-3.5-turbo": {"rpm_limit": 100},  # No tpm_limit
            },
        )
        result = get_key_model_tpm_limit(user_api_key_dict)
        assert result == {"gpt-4": 10000}

    def test_returns_none_when_no_limits_configured(self):
        """Should return None when no rate limits are configured."""
        user_api_key_dict = UserAPIKeyAuth(api_key="sk-123")
        result = get_key_model_tpm_limit(user_api_key_dict)
        assert result is None

    def test_model_max_budget_priority_over_team(self):
        """model_max_budget should take priority over team_metadata."""
        user_api_key_dict = UserAPIKeyAuth(
            api_key="sk-123",
            model_max_budget={"gpt-4": {"tpm_limit": 10000}},
            team_metadata={"model_tpm_limit": {"gpt-4": 5000}},
        )
        result = get_key_model_tpm_limit(user_api_key_dict)
        assert result == {"gpt-4": 10000}
