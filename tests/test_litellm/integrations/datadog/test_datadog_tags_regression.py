import os
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.abspath("../../../"))

from litellm.integrations.datadog.datadog_handler import get_datadog_tags
from litellm.integrations.datadog.datadog_cost_management import (
    DatadogCostManagementLogger,
)
from litellm.types.utils import StandardLoggingPayload, StandardLoggingMetadata


class TestDatadogTagsRegression:
    @pytest.fixture
    def mock_env_vars(self):
        """Mock environment variables to isolate environment."""
        with patch.dict(
            os.environ,
            {
                "DD_ENV": "test-env",
                "DD_SERVICE": "test-service",
                "DD_VERSION": "1.0.0",
                "HOSTNAME": "test-host",
                "POD_NAME": "test-pod",
                "DD_API_KEY": "mock-api-key",
                "DD_APP_KEY": "mock-app-key",
            },
        ):
            yield

    def test_get_datadog_tags_regression(self, mock_env_vars):
        """
        Regression Test: Ensure that get_datadog_tags still produces basic tags correctly
        AND now includes the new team tag when provided.
        """
        # Case 1: Legacy behavior (no team info)
        payload_legacy = StandardLoggingPayload(metadata={})
        tags_legacy = get_datadog_tags(payload_legacy)

        # Verify base tags exist (legacy requirement)
        assert "env:test-env" in tags_legacy
        assert "service:test-service" in tags_legacy
        # Verify NO team tag (should not invent one)
        assert "team:" not in tags_legacy

        # Case 2: New feature (team info provided)
        payload_with_team = StandardLoggingPayload(
            metadata=StandardLoggingMetadata(user_api_key_team_alias="regression-team")
        )
        tags_with_team = get_datadog_tags(payload_with_team)

        # Verify base tags STILL exist
        assert "env:test-env" in tags_with_team
        assert "service:test-service" in tags_with_team
        # Verify NEW team tag is added
        assert "team:regression-team" in tags_with_team

    @pytest.mark.asyncio
    async def test_datadog_cost_management_tags_regression(self, mock_env_vars):
        """
        Regression Test: Ensure DatadogCostManagementLogger extracts tags correctly,
        preserving existing behavior while adding the team tag capability.
        """
        logger = DatadogCostManagementLogger()

        # Case 1: Legacy metadata (user alias only)
        payload_legacy = StandardLoggingPayload(
            metadata=StandardLoggingMetadata(user_api_key_alias="legacy-user")
        )

        tags_legacy = logger._extract_tags(payload_legacy)

        assert tags_legacy["env"] == "test-env"
        assert tags_legacy["user"] == "legacy-user"
        assert "team" not in tags_legacy  # Should not exist

        # Case 2: New metadata (team alias)
        payload_new = StandardLoggingPayload(
            metadata=StandardLoggingMetadata(
                user_api_key_alias="new-user", user_api_key_team_alias="new-team-alias"
            )
        )

        tags_new = logger._extract_tags(payload_new)

        assert tags_new["env"] == "test-env"
        assert tags_new["user"] == "new-user"
        assert tags_new["team"] == "new-team-alias"  # New feature verified
