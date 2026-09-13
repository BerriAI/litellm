import datetime
import os
from unittest.mock import patch

import pytest


from litellm.integrations.datadog.datadog import DataDogLogger
from litellm.integrations.datadog.datadog_handler import get_datadog_tags, normalize_datadog_tag_value
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
                "DD_SITE": "test.datadoghq.com",
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
        assert not any(t.startswith("team:") for t in tags_legacy)

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

    @pytest.mark.parametrize(
        ("value", "expected"),
        (
            ("P&T", "p_t"),
            ("CTO-B2B", "cto-b2b"),
            ("  Team  &  Key!!  ", "team_key"),
            ("regression-team", "regression-team"),
        ),
    )
    def test_normalize_datadog_tag_value(self, value, expected):
        assert normalize_datadog_tag_value(value) == expected

    def test_get_datadog_tags_normalizes_alias_and_request_tag_values(self, mock_env_vars):
        payload = StandardLoggingPayload(
            request_tags=["capability:P&T"],
            metadata=StandardLoggingMetadata(user_api_key_team_alias="CTO-B2B"),
        )

        tags = get_datadog_tags(payload)

        assert "request_tag:capability:p_t" in tags
        assert "team:cto-b2b" in tags

    def test_get_datadog_tags_keeps_non_string_tag_values(self, mock_env_vars):
        payload = StandardLoggingPayload(
            request_tags=[12345, "capability:P&T"],
            metadata=StandardLoggingMetadata(user_api_key_team_id=67890),
        )

        tags = get_datadog_tags(payload)

        assert "request_tag:12345" in tags
        assert "request_tag:capability:p_t" in tags
        assert "team:67890" in tags

    @pytest.mark.asyncio
    async def test_non_string_request_tag_still_emits_the_datadog_payload(self, mock_env_vars):
        with patch("asyncio.create_task"):
            logger = DataDogLogger()
        payload = StandardLoggingPayload(request_tags=[12345], metadata=StandardLoggingMetadata())

        await logger.async_log_success_event(
            kwargs={"standard_logging_object": payload},
            response_obj=None,
            start_time=datetime.datetime(2026, 1, 1),
            end_time=datetime.datetime(2026, 1, 1),
        )

        assert len(logger.log_queue) == 1
        assert "request_tag:12345" in logger.log_queue[0]["ddtags"].split(",")

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

    @pytest.mark.asyncio
    async def test_datadog_cost_management_normalizes_alias_and_custom_tag_values(self, mock_env_vars):
        logger = DatadogCostManagementLogger(cost_tag_keys=["capability"])
        payload = StandardLoggingPayload(
            request_tags=["capability:Space & Punctuation!"],
            metadata=StandardLoggingMetadata(
                user_api_key_alias="P&T",
                user_api_key_team_alias="CTO-B2B",
            ),
        )

        tags = logger._extract_tags(payload)

        assert tags["user"] == "p_t"
        assert tags["team"] == "cto-b2b"
        assert tags["capability"] == "space_punctuation"

    @pytest.mark.asyncio
    async def test_datadog_cost_management_keeps_non_string_alias_values(self, mock_env_vars):
        logger = DatadogCostManagementLogger()
        payload = StandardLoggingPayload(
            metadata=StandardLoggingMetadata(user_api_key_alias=12345, user_api_key_team_id=67890),
        )

        tags = logger._extract_tags(payload)

        assert tags["user"] == "12345"
        assert tags["team"] == "67890"
