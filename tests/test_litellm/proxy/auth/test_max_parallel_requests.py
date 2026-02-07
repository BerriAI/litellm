"""
Tests for max_parallel_requests per model/user/team support.

Tests the helper function and rate limiter integration.
"""

import pytest
from typing import Dict, Optional

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.auth_utils import get_key_model_max_parallel_requests


class TestGetKeyModelMaxParallelRequests:
    """Tests for get_key_model_max_parallel_requests helper function."""

    def test_returns_none_when_no_limits_set(self):
        """Should return None when no max_parallel_requests limits are configured."""
        user_api_key_dict = UserAPIKeyAuth(
            api_key="sk-test",
            metadata={},
        )
        result = get_key_model_max_parallel_requests(user_api_key_dict)
        assert result is None

    def test_extracts_from_metadata(self):
        """Should extract limits from key metadata."""
        user_api_key_dict = UserAPIKeyAuth(
            api_key="sk-test",
            metadata={
                "model_max_parallel_requests": {
                    "gpt-4": 5,
                    "gpt-3.5-turbo": 10,
                }
            },
        )
        result = get_key_model_max_parallel_requests(user_api_key_dict)
        assert result == {"gpt-4": 5, "gpt-3.5-turbo": 10}

    def test_extracts_from_model_max_budget(self):
        """Should extract limits from model_max_budget when metadata not set."""
        user_api_key_dict = UserAPIKeyAuth(
            api_key="sk-test",
            metadata={},  # Empty dict, not None
            model_max_budget={
                "gpt-4": {"max_parallel_requests": 3, "budget": 100.0},
                "gpt-3.5-turbo": {"budget": 50.0},  # No max_parallel_requests
            },
        )
        result = get_key_model_max_parallel_requests(user_api_key_dict)
        assert result == {"gpt-4": 3}

    def test_extracts_from_team_metadata(self):
        """Should extract limits from team_metadata as fallback."""
        user_api_key_dict = UserAPIKeyAuth(
            api_key="sk-test",
            metadata={},  # Empty dict, not None
            team_metadata={
                "model_max_parallel_requests": {
                    "gpt-4": 8,
                }
            },
        )
        result = get_key_model_max_parallel_requests(user_api_key_dict)
        assert result == {"gpt-4": 8}

    def test_metadata_takes_precedence(self):
        """Key metadata should take precedence over team_metadata."""
        user_api_key_dict = UserAPIKeyAuth(
            api_key="sk-test",
            metadata={"model_max_parallel_requests": {"gpt-4": 5}},
            team_metadata={"model_max_parallel_requests": {"gpt-4": 10}},
        )
        result = get_key_model_max_parallel_requests(user_api_key_dict)
        assert result == {"gpt-4": 5}

    def test_skips_none_values_in_model_max_budget(self):
        """Should skip models with None max_parallel_requests in budget."""
        user_api_key_dict = UserAPIKeyAuth(
            api_key="sk-test",
            metadata={},  # Empty dict, not None
            model_max_budget={
                "gpt-4": {"max_parallel_requests": None},
                "gpt-3.5-turbo": {"max_parallel_requests": 5},
            },
        )
        result = get_key_model_max_parallel_requests(user_api_key_dict)
        assert result == {"gpt-3.5-turbo": 5}


class TestUserMaxParallelRequests:
    """Tests for user-level max_parallel_requests field."""

    def test_field_exists_on_user_api_key_auth(self):
        """UserAPIKeyAuth should have user_max_parallel_requests field."""
        user_api_key_dict = UserAPIKeyAuth(
            api_key="sk-test",
            user_id="user-123",
            user_max_parallel_requests=10,
        )
        assert user_api_key_dict.user_max_parallel_requests == 10

    def test_field_defaults_to_none(self):
        """user_max_parallel_requests should default to None."""
        user_api_key_dict = UserAPIKeyAuth(
            api_key="sk-test",
            user_id="user-123",
        )
        assert user_api_key_dict.user_max_parallel_requests is None


class TestTeamMaxParallelRequests:
    """Tests for team-level max_parallel_requests field."""

    def test_field_exists_on_user_api_key_auth(self):
        """UserAPIKeyAuth should have team_max_parallel_requests field."""
        user_api_key_dict = UserAPIKeyAuth(
            api_key="sk-test",
            team_id="team-456",
            team_max_parallel_requests=50,
        )
        assert user_api_key_dict.team_max_parallel_requests == 50

    def test_field_defaults_to_none(self):
        """team_max_parallel_requests should default to None."""
        user_api_key_dict = UserAPIKeyAuth(
            api_key="sk-test",
            team_id="team-456",
        )
        assert user_api_key_dict.team_max_parallel_requests is None
