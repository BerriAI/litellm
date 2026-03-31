"""
Unit Tests for Prometheus rate limit allowed/used metrics.
Tests both inline emission (from RateLimitResponse) and cron-based initialization.
"""

from typing import Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class FakeGauge:
    """Fake Prometheus Gauge that records .labels().set() calls."""

    def __init__(self):
        self._values: Dict[tuple, float] = {}

    def labels(self, *args, **kwargs):
        key = args if args else tuple(sorted(kwargs.items()))
        self._key = key
        return self

    def set(self, value):
        self._values[self._key] = value


@pytest.fixture
def prometheus_logger():
    """Create a PrometheusLogger with fake gauge metrics for testing."""
    from litellm.integrations.prometheus import PrometheusLogger

    logger = PrometheusLogger.__new__(PrometheusLogger)
    logger.litellm_api_key_rate_limit_allowed_metric = FakeGauge()
    logger.litellm_api_key_rate_limit_used_metric = FakeGauge()
    logger.litellm_team_rate_limit_allowed_metric = FakeGauge()
    logger.litellm_team_rate_limit_used_metric = FakeGauge()
    return logger


def _make_rate_limit_response(statuses):
    return {"overall_code": "OK", "statuses": statuses}


def _make_status(descriptor_key, rate_limit_type, current_limit, limit_remaining):
    return {
        "code": "OK",
        "descriptor_key": descriptor_key,
        "rate_limit_type": rate_limit_type,
        "current_limit": current_limit,
        "limit_remaining": limit_remaining,
    }


class TestSetRateLimitMetricsFromResponse:
    def test_api_key_rpm_and_tpm(self, prometheus_logger):
        """Test that api_key RPM and TPM metrics are set correctly."""
        response = _make_rate_limit_response(
            [
                _make_status("api_key", "requests", 100, 95),
                _make_status("api_key", "tokens", 50000, 48000),
            ]
        )

        prometheus_logger._set_rate_limit_metrics_from_response(
            rate_limit_response=response,
            hashed_api_key="sk-hash-123",
            api_key_alias="my-key",
            team_id=None,
            team_alias=None,
        )

        allowed = prometheus_logger.litellm_api_key_rate_limit_allowed_metric._values
        used = prometheus_logger.litellm_api_key_rate_limit_used_metric._values

        assert ("sk-hash-123", "my-key", "rpm") in allowed
        assert allowed[("sk-hash-123", "my-key", "rpm")] == 100
        assert used[("sk-hash-123", "my-key", "rpm")] == 5  # 100 - 95

        assert ("sk-hash-123", "my-key", "tpm") in allowed
        assert allowed[("sk-hash-123", "my-key", "tpm")] == 50000
        assert used[("sk-hash-123", "my-key", "tpm")] == 2000  # 50000 - 48000

    def test_team_rpm_and_tpm(self, prometheus_logger):
        """Test that team RPM and TPM metrics are set correctly."""
        response = _make_rate_limit_response(
            [
                _make_status("team", "requests", 200, 150),
                _make_status("team", "tokens", 100000, 80000),
            ]
        )

        prometheus_logger._set_rate_limit_metrics_from_response(
            rate_limit_response=response,
            hashed_api_key=None,
            api_key_alias=None,
            team_id="team-abc",
            team_alias="My Team",
        )

        allowed = prometheus_logger.litellm_team_rate_limit_allowed_metric._values
        used = prometheus_logger.litellm_team_rate_limit_used_metric._values

        assert ("team-abc", "My Team", "rpm") in allowed
        assert allowed[("team-abc", "My Team", "rpm")] == 200
        assert used[("team-abc", "My Team", "rpm")] == 50

        assert ("team-abc", "My Team", "tpm") in allowed
        assert allowed[("team-abc", "My Team", "tpm")] == 100000
        assert used[("team-abc", "My Team", "tpm")] == 20000

    def test_mixed_key_and_team(self, prometheus_logger):
        """Test response with both api_key and team statuses."""
        response = _make_rate_limit_response(
            [
                _make_status("api_key", "requests", 100, 90),
                _make_status("team", "requests", 500, 400),
            ]
        )

        prometheus_logger._set_rate_limit_metrics_from_response(
            rate_limit_response=response,
            hashed_api_key="sk-hash",
            api_key_alias="alias",
            team_id="team-1",
            team_alias="Team One",
        )

        key_allowed = (
            prometheus_logger.litellm_api_key_rate_limit_allowed_metric._values
        )
        team_allowed = prometheus_logger.litellm_team_rate_limit_allowed_metric._values

        assert key_allowed[("sk-hash", "alias", "rpm")] == 100
        assert team_allowed[("team-1", "Team One", "rpm")] == 500

    def test_skips_max_parallel_requests(self, prometheus_logger):
        """max_parallel_requests statuses should be ignored."""
        response = _make_rate_limit_response(
            [
                _make_status("api_key", "max_parallel_requests", 10, 8),
            ]
        )

        prometheus_logger._set_rate_limit_metrics_from_response(
            rate_limit_response=response,
            hashed_api_key="sk-hash",
            api_key_alias="alias",
            team_id=None,
            team_alias=None,
        )

        assert (
            len(prometheus_logger.litellm_api_key_rate_limit_allowed_metric._values)
            == 0
        )

    def test_skips_non_key_team_descriptors(self, prometheus_logger):
        """Descriptors like 'user', 'end_user' should not emit metrics."""
        response = _make_rate_limit_response(
            [
                _make_status("user", "requests", 100, 90),
                _make_status("end_user", "tokens", 50000, 40000),
            ]
        )

        prometheus_logger._set_rate_limit_metrics_from_response(
            rate_limit_response=response,
            hashed_api_key="sk-hash",
            api_key_alias="alias",
            team_id="team-1",
            team_alias="Team",
        )

        assert (
            len(prometheus_logger.litellm_api_key_rate_limit_allowed_metric._values)
            == 0
        )
        assert (
            len(prometheus_logger.litellm_team_rate_limit_allowed_metric._values) == 0
        )

    def test_handles_none_response_gracefully(self, prometheus_logger):
        """Should not crash when rate_limit_response is None."""
        # The caller checks for None before calling, but the method should handle empty statuses
        response = _make_rate_limit_response([])

        prometheus_logger._set_rate_limit_metrics_from_response(
            rate_limit_response=response,
            hashed_api_key=None,
            api_key_alias=None,
            team_id=None,
            team_alias=None,
        )

        assert (
            len(prometheus_logger.litellm_api_key_rate_limit_allowed_metric._values)
            == 0
        )

    def test_used_never_negative(self, prometheus_logger):
        """If limit_remaining > current_limit (unlikely but defensive), used should be 0."""
        response = _make_rate_limit_response(
            [
                _make_status("api_key", "requests", 100, 200),
            ]
        )

        prometheus_logger._set_rate_limit_metrics_from_response(
            rate_limit_response=response,
            hashed_api_key="sk-hash",
            api_key_alias="alias",
            team_id=None,
            team_alias=None,
        )

        used = prometheus_logger.litellm_api_key_rate_limit_used_metric._values
        assert used[("sk-hash", "alias", "rpm")] == 0

    def test_none_limit_remaining_treated_as_zero(self, prometheus_logger):
        """If limit_remaining is None, used should equal current_limit."""
        response = _make_rate_limit_response(
            [
                _make_status("api_key", "requests", 100, None),
            ]
        )

        prometheus_logger._set_rate_limit_metrics_from_response(
            rate_limit_response=response,
            hashed_api_key="sk-hash",
            api_key_alias="alias",
            team_id=None,
            team_alias=None,
        )

        used = prometheus_logger.litellm_api_key_rate_limit_used_metric._values
        assert used[("sk-hash", "alias", "rpm")] == 100

    def test_skips_when_hashed_api_key_is_none(self, prometheus_logger):
        """api_key metrics should not be set if hashed_api_key is None."""
        response = _make_rate_limit_response(
            [
                _make_status("api_key", "requests", 100, 90),
            ]
        )

        prometheus_logger._set_rate_limit_metrics_from_response(
            rate_limit_response=response,
            hashed_api_key=None,
            api_key_alias=None,
            team_id=None,
            team_alias=None,
        )

        assert (
            len(prometheus_logger.litellm_api_key_rate_limit_allowed_metric._values)
            == 0
        )

    def test_skips_when_team_id_is_none(self, prometheus_logger):
        """team metrics should not be set if team_id is None."""
        response = _make_rate_limit_response(
            [
                _make_status("team", "requests", 100, 90),
            ]
        )

        prometheus_logger._set_rate_limit_metrics_from_response(
            rate_limit_response=response,
            hashed_api_key=None,
            api_key_alias=None,
            team_id=None,
            team_alias=None,
        )

        assert (
            len(prometheus_logger.litellm_team_rate_limit_allowed_metric._values) == 0
        )


class TestCronInitializeKeyRateLimitMetrics:
    @pytest.mark.asyncio
    async def test_initialize_key_rate_limit_metrics(self, prometheus_logger):
        """Test that cron job sets allowed gauges for keys with rate limits."""
        from litellm.proxy._types import UserAPIKeyAuth

        mock_keys = [
            UserAPIKeyAuth(
                token="sk-hashed-1",
                key_alias="key-one",
                rpm_limit=100,
                tpm_limit=50000,
            ),
            UserAPIKeyAuth(
                token="sk-hashed-2",
                key_alias="key-two",
                rpm_limit=200,
                tpm_limit=None,
            ),
            UserAPIKeyAuth(
                token="sk-hashed-3",
                key_alias="key-three",
                rpm_limit=None,
                tpm_limit=None,
            ),
        ]

        mock_response = {
            "keys": mock_keys,
            "total_count": 3,
        }

        with patch(
            "litellm.proxy.proxy_server.prisma_client",
            new=MagicMock(),
        ):
            with patch(
                "litellm.proxy.management_endpoints.key_management_endpoints._list_key_helper",
                new=AsyncMock(return_value=mock_response),
            ):
                await prometheus_logger._initialize_key_rate_limit_metrics()

        allowed = prometheus_logger.litellm_api_key_rate_limit_allowed_metric._values
        used = prometheus_logger.litellm_api_key_rate_limit_used_metric._values

        # key-one: both RPM and TPM
        assert allowed[("sk-hashed-1", "key-one", "rpm")] == 100
        assert allowed[("sk-hashed-1", "key-one", "tpm")] == 50000
        assert used[("sk-hashed-1", "key-one", "rpm")] == 0
        assert used[("sk-hashed-1", "key-one", "tpm")] == 0

        # key-two: only RPM
        assert allowed[("sk-hashed-2", "key-two", "rpm")] == 200
        assert ("sk-hashed-2", "key-two", "tpm") not in allowed

        # key-three: no limits, should not appear
        assert ("sk-hashed-3", "key-three", "rpm") not in allowed
        assert ("sk-hashed-3", "key-three", "tpm") not in allowed


class TestCronInitializeTeamRateLimitMetrics:
    @pytest.mark.asyncio
    async def test_initialize_team_rate_limit_metrics(self, prometheus_logger):
        """Test that cron job sets allowed gauges for teams with rate limits."""
        mock_team_1 = MagicMock()
        mock_team_1.team_id = "team-1"
        mock_team_1.team_alias = "Alpha Team"
        mock_team_1.rpm_limit = 500
        mock_team_1.tpm_limit = 200000

        mock_team_2 = MagicMock()
        mock_team_2.team_id = "team-2"
        mock_team_2.team_alias = "Beta Team"
        mock_team_2.rpm_limit = None
        mock_team_2.tpm_limit = 100000

        mock_team_3 = MagicMock()
        mock_team_3.team_id = "team-3"
        mock_team_3.team_alias = "Gamma Team"
        mock_team_3.rpm_limit = None
        mock_team_3.tpm_limit = None

        with patch(
            "litellm.proxy.proxy_server.prisma_client",
            new=MagicMock(),
        ):
            with patch(
                "litellm.proxy.management_endpoints.team_endpoints.get_paginated_teams",
                new=AsyncMock(
                    return_value=([mock_team_1, mock_team_2, mock_team_3], 3)
                ),
            ):
                await prometheus_logger._initialize_team_rate_limit_metrics()

        allowed = prometheus_logger.litellm_team_rate_limit_allowed_metric._values
        used = prometheus_logger.litellm_team_rate_limit_used_metric._values

        # team-1: both RPM and TPM
        assert allowed[("team-1", "Alpha Team", "rpm")] == 500
        assert allowed[("team-1", "Alpha Team", "tpm")] == 200000
        assert used[("team-1", "Alpha Team", "rpm")] == 0
        assert used[("team-1", "Alpha Team", "tpm")] == 0

        # team-2: only TPM
        assert ("team-2", "Beta Team", "rpm") not in allowed
        assert allowed[("team-2", "Beta Team", "tpm")] == 100000

        # team-3: no limits, should not appear
        assert ("team-3", "Gamma Team", "rpm") not in allowed
        assert ("team-3", "Gamma Team", "tpm") not in allowed
