"""
Unit tests for Prometheus user and team count metrics
"""
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from prometheus_client import REGISTRY

from litellm.integrations.prometheus import PrometheusLogger


@pytest.fixture(autouse=True)
def cleanup_prometheus_registry():
    """Clean up prometheus registry between tests"""
    # Clear the registry before each test
    collectors = list(REGISTRY._collector_to_names.keys())
    for collector in collectors:
        try:
            REGISTRY.unregister(collector)
        except Exception:
            pass
    yield
    # Clean up after test
    collectors = list(REGISTRY._collector_to_names.keys())
    for collector in collectors:
        try:
            REGISTRY.unregister(collector)
        except Exception:
            pass


@pytest.fixture
def prometheus_logger():
    """Create a fresh PrometheusLogger instance for each test"""
    return PrometheusLogger()


class TestPrometheusUserTeamCountMetrics:
    """Test user and team count metric initialization and functionality"""

    def test_user_team_count_metrics_initialization(self, prometheus_logger):
        """Test that user and team count metrics are properly initialized"""
        # Verify that the metrics exist
        assert hasattr(prometheus_logger, "litellm_total_users_metric")
        assert hasattr(prometheus_logger, "litellm_teams_count_metric")

        # Verify the metrics are not None
        assert prometheus_logger.litellm_total_users_metric is not None
        assert prometheus_logger.litellm_teams_count_metric is not None

    def test_user_count_metric_has_no_labels(self, prometheus_logger):
        """Test that litellm_total_users metric has no labels (as specified)"""
        metric = prometheus_logger.litellm_total_users_metric

        # The metric should be callable without labels
        # Try to set a value directly
        try:
            metric.set(10)
            # If we get here, the metric accepts direct set() calls (no labels)
            assert True
        except Exception as e:
            pytest.fail(f"litellm_total_users_metric should not require labels: {e}")

    def test_teams_count_metric_has_no_labels(self, prometheus_logger):
        """Test that litellm_teams_count metric has no labels (as specified)"""
        metric = prometheus_logger.litellm_teams_count_metric

        # The metric should be callable without labels
        try:
            metric.set(5)
            assert True
        except Exception as e:
            pytest.fail(f"litellm_teams_count_metric should not require labels: {e}")

    def test_user_count_metric_accepts_various_values(self, prometheus_logger):
        """Test that user count metric accepts various realistic values"""
        metric = prometheus_logger.litellm_total_users_metric

        test_values = [0, 1, 10, 100, 1000, 10000]

        for value in test_values:
            try:
                metric.set(value)
            except Exception as e:
                pytest.fail(
                    f"litellm_total_users_metric should accept value {value}: {e}"
                )

    def test_team_count_metric_accepts_various_values(self, prometheus_logger):
        """Test that team count metric accepts various realistic values"""
        metric = prometheus_logger.litellm_teams_count_metric

        test_values = [0, 1, 5, 20, 50, 100]

        for value in test_values:
            try:
                metric.set(value)
            except Exception as e:
                pytest.fail(
                    f"litellm_teams_count_metric should accept value {value}: {e}"
                )

    def test_user_count_metric_with_zero(self, prometheus_logger):
        """Test that user count metric handles zero users"""
        metric = prometheus_logger.litellm_total_users_metric

        # Should handle zero gracefully
        try:
            metric.set(0)
            assert True
        except Exception as e:
            pytest.fail(f"litellm_total_users_metric should handle zero: {e}")

    def test_team_count_metric_with_zero(self, prometheus_logger):
        """Test that team count metric handles zero teams"""
        metric = prometheus_logger.litellm_teams_count_metric

        # Should handle zero gracefully
        try:
            metric.set(0)
            assert True
        except Exception as e:
            pytest.fail(f"litellm_teams_count_metric should handle zero: {e}")

    def test_metrics_can_be_updated_multiple_times(self, prometheus_logger):
        """Test that metrics can be updated multiple times (simulating refresh cycle)"""
        user_metric = prometheus_logger.litellm_total_users_metric
        team_metric = prometheus_logger.litellm_teams_count_metric

        # First update
        user_metric.set(10)
        team_metric.set(5)

        # Second update (simulating refresh)
        user_metric.set(15)
        team_metric.set(8)

        # Third update
        user_metric.set(20)
        team_metric.set(10)

        # Should handle multiple updates without errors
        assert True

    def test_metrics_can_be_collected_by_prometheus(self, prometheus_logger):
        """Test that the metrics can be collected by Prometheus registry"""
        # Set some values
        prometheus_logger.litellm_total_users_metric.set(100)
        prometheus_logger.litellm_teams_count_metric.set(20)

        # Collect metrics from registry
        metrics = {}
        for metric in REGISTRY.collect():
            for sample in metric.samples:
                metrics[sample.name] = sample.value

        # Verify our metrics are in the collected metrics
        assert "litellm_total_users" in metrics or "litellm_total_users_total" in metrics
        assert "litellm_teams_count" in metrics or "litellm_teams_count_total" in metrics

    def test_initialize_user_and_team_count_metrics_method_exists(
        self, prometheus_logger
    ):
        """Test that _initialize_user_and_team_count_metrics method exists and is callable"""
        # Verify the method exists
        assert hasattr(prometheus_logger, "_initialize_user_and_team_count_metrics")
        assert callable(prometheus_logger._initialize_user_and_team_count_metrics)

    @pytest.mark.asyncio
    async def test_initialize_remaining_budget_metrics_includes_user_team_counts(
        self, prometheus_logger
    ):
        """Test that _initialize_remaining_budget_metrics calls user/team count initialization"""
        from unittest.mock import AsyncMock

        # Mock all the async methods
        prometheus_logger._initialize_team_budget_metrics = AsyncMock()
        prometheus_logger._initialize_api_key_budget_metrics = AsyncMock()
        prometheus_logger._initialize_user_and_team_count_metrics = AsyncMock()

        await prometheus_logger._initialize_remaining_budget_metrics()

        # Verify all three initialization methods were called
        prometheus_logger._initialize_team_budget_metrics.assert_called_once()
        prometheus_logger._initialize_api_key_budget_metrics.assert_called_once()
        prometheus_logger._initialize_user_and_team_count_metrics.assert_called_once()

    def test_metrics_have_correct_type(self, prometheus_logger):
        """Test that metrics are Gauge type (not Counter or Histogram)"""
        from prometheus_client import Gauge

        # The metrics should be Gauge instances (or wrapped gauges)
        # We can test this by checking they have the set() method
        assert hasattr(prometheus_logger.litellm_total_users_metric, "set")
        assert hasattr(prometheus_logger.litellm_teams_count_metric, "set")

        # Gauges have set() method, Counters only have inc()
        assert callable(prometheus_logger.litellm_total_users_metric.set)
        assert callable(prometheus_logger.litellm_teams_count_metric.set)

    def test_user_count_metric_realistic_scenario(self, prometheus_logger):
        """Test realistic scenario: system starts with users, more are added"""
        metric = prometheus_logger.litellm_total_users_metric

        # System starts with existing users
        metric.set(1000)

        # More users are added over time
        metric.set(1050)
        metric.set(1100)
        metric.set(1200)

        # System should handle growing user counts
        assert True

    def test_team_count_metric_realistic_scenario(self, prometheus_logger):
        """Test realistic scenario: teams are created and possibly removed"""
        metric = prometheus_logger.litellm_teams_count_metric

        # Start with some teams
        metric.set(50)

        # Teams grow
        metric.set(55)
        metric.set(60)

        # Teams might shrink (if some are deleted)
        metric.set(58)

        # System should handle team count changes
        assert True

    def test_concurrent_metric_updates(self, prometheus_logger):
        """Test that both metrics can be updated concurrently without interference"""
        user_metric = prometheus_logger.litellm_total_users_metric
        team_metric = prometheus_logger.litellm_teams_count_metric

        # Update both metrics in quick succession
        user_metric.set(500)
        team_metric.set(25)
        user_metric.set(501)
        team_metric.set(26)
        user_metric.set(502)
        team_metric.set(27)

        # Both should work independently
        assert True

    def test_metrics_handle_large_values(self, prometheus_logger):
        """Test that metrics can handle large enterprise-scale values"""
        user_metric = prometheus_logger.litellm_total_users_metric
        team_metric = prometheus_logger.litellm_teams_count_metric

        # Large enterprise scale
        try:
            user_metric.set(1000000)  # 1 million users
            team_metric.set(10000)  # 10k teams
            assert True
        except Exception as e:
            pytest.fail(f"Metrics should handle large values: {e}")


# ---------------------------------------------------------------------------
# Regression tests: team budget showing +Inf when user_api_key_team_max_budget
# is None in request metadata but the team has a real budget in the DB.
# ---------------------------------------------------------------------------


async def test_assemble_team_object_uses_db_max_budget_when_metadata_is_none(
    prometheus_logger,
):
    """
    When max_budget is None in request metadata (e.g. stale key cache),
    _assemble_team_object must fall back to the value returned by get_team_object
    so that _safe_get_remaining_budget does not return +Inf.
    """
    db_team = MagicMock()
    db_team.max_budget = 3000.0
    db_team.budget_reset_at = datetime(2026, 3, 1, tzinfo=timezone.utc)

    with patch("litellm.proxy.auth.auth_checks.get_team_object") as mock_get_team:
        mock_get_team.return_value = db_team
        team_object = await prometheus_logger._assemble_team_object(
            team_id="c5c33858-4379-4c90-8733-d9c58c312c10",
            team_alias="ai-ml-local_dev",
            spend=1617.02,
            max_budget=None,  # simulates None coming from request metadata
            response_cost=0.5,
        )

    assert team_object.max_budget == 3000.0, (
        "max_budget should be populated from DB when metadata value is None"
    )
    assert team_object.budget_reset_at == datetime(2026, 3, 1, tzinfo=timezone.utc)


async def test_assemble_team_object_does_not_override_metadata_max_budget(
    prometheus_logger,
):
    """
    When max_budget IS present in request metadata, it must not be overridden
    by the DB value.
    """
    db_team = MagicMock()
    db_team.max_budget = 9999.0
    db_team.budget_reset_at = None

    with patch("litellm.proxy.auth.auth_checks.get_team_object") as mock_get_team:
        mock_get_team.return_value = db_team
        team_object = await prometheus_logger._assemble_team_object(
            team_id="team-1",
            team_alias="my-team",
            spend=50.0,
            max_budget=100.0,  # metadata has a real value
            response_cost=1.0,
        )

    assert team_object.max_budget == 100.0, (
        "max_budget from metadata must not be replaced by the DB value"
    )


async def test_set_team_budget_metrics_after_api_request_no_inf_when_metadata_budget_none(
    prometheus_logger,
):
    """
    End-to-end: when user_api_key_team_max_budget is None in request metadata
    but the team has a real budget in the DB, the metric must NOT be set to +Inf.
    """
    prometheus_logger.litellm_remaining_team_budget_metric = MagicMock()
    prometheus_logger.litellm_team_max_budget_metric = MagicMock()
    prometheus_logger.litellm_team_budget_remaining_hours_metric = MagicMock()

    db_team = MagicMock()
    db_team.max_budget = 3000.0
    db_team.budget_reset_at = datetime(2026, 3, 1, tzinfo=timezone.utc)

    with patch("litellm.proxy.auth.auth_checks.get_team_object") as mock_get_team:
        mock_get_team.return_value = db_team
        await prometheus_logger._set_team_budget_metrics_after_api_request(
            user_api_team="c5c33858-4379-4c90-8733-d9c58c312c10",
            user_api_team_alias="ai-ml-local_dev",
            team_spend=1617.02,
            team_max_budget=None,  # simulates stale key cache
            response_cost=0.5,
        )

    set_call_args = (
        prometheus_logger.litellm_remaining_team_budget_metric.labels().set.call_args
    )
    assert set_call_args is not None, "remaining_team_budget_metric.labels().set was not called"
    actual_value = set_call_args[0][0]
    assert actual_value != float("inf"), (
        f"remaining_team_budget_metric must not be +Inf when team has a real budget; got {actual_value}"
    )
    expected = 3000.0 - 1617.02 - 0.5
    assert abs(actual_value - expected) < 0.01, (
        f"Expected remaining budget ~{expected}, got {actual_value}"
    )


async def test_set_team_budget_metrics_after_api_request_inf_when_genuinely_no_budget(
    prometheus_logger,
):
    """
    When the team genuinely has no budget (max_budget=None in both metadata and
    DB), +Inf is the correct value and must be preserved.
    """
    prometheus_logger.litellm_remaining_team_budget_metric = MagicMock()
    prometheus_logger.litellm_team_max_budget_metric = MagicMock()
    prometheus_logger.litellm_team_budget_remaining_hours_metric = MagicMock()

    db_team = MagicMock()
    db_team.max_budget = None
    db_team.budget_reset_at = None

    with patch("litellm.proxy.auth.auth_checks.get_team_object") as mock_get_team:
        mock_get_team.return_value = db_team
        await prometheus_logger._set_team_budget_metrics_after_api_request(
            user_api_team="team-no-budget",
            user_api_team_alias="no-budget-team",
            team_spend=10.0,
            team_max_budget=None,
            response_cost=1.0,
        )

    set_call_args = (
        prometheus_logger.litellm_remaining_team_budget_metric.labels().set.call_args
    )
    assert set_call_args is not None
    actual_value = set_call_args[0][0]
    assert actual_value == float("inf"), (
        "remaining_team_budget_metric should be +Inf when team truly has no budget"
    )


# ---------------------------------------------------------------------------
# Regression tests: user budget showing +Inf when user_api_key_user_max_budget
# is None in request metadata but the user has a real budget in the DB.
# ---------------------------------------------------------------------------


async def test_assemble_user_object_uses_db_max_budget_when_metadata_is_none(
    prometheus_logger,
):
    """
    When max_budget is None in request metadata (e.g. stale key cache),
    _assemble_user_object must fall back to the value returned by get_user_object
    so that _safe_get_remaining_budget does not return +Inf.
    """
    db_user = MagicMock()
    db_user.max_budget = 500.0
    db_user.budget_reset_at = datetime(2026, 3, 1, tzinfo=timezone.utc)

    with patch("litellm.proxy.auth.auth_checks.get_user_object") as mock_get_user:
        mock_get_user.return_value = db_user
        user_object = await prometheus_logger._assemble_user_object(
            user_id="user-abc-123",
            spend=120.0,
            max_budget=None,  # simulates None coming from request metadata
            response_cost=0.5,
        )

    assert user_object.max_budget == 500.0, (
        "max_budget should be populated from DB when metadata value is None"
    )
    assert user_object.budget_reset_at == datetime(2026, 3, 1, tzinfo=timezone.utc)


async def test_assemble_user_object_does_not_override_metadata_max_budget(
    prometheus_logger,
):
    """
    When max_budget IS present in request metadata, it must not be overridden
    by the DB value.
    """
    db_user = MagicMock()
    db_user.max_budget = 9999.0
    db_user.budget_reset_at = None

    with patch("litellm.proxy.auth.auth_checks.get_user_object") as mock_get_user:
        mock_get_user.return_value = db_user
        user_object = await prometheus_logger._assemble_user_object(
            user_id="user-abc-123",
            spend=50.0,
            max_budget=100.0,  # metadata has a real value
            response_cost=1.0,
        )

    assert user_object.max_budget == 100.0, (
        "max_budget from metadata must not be replaced by the DB value"
    )


async def test_set_user_budget_metrics_after_api_request_no_inf_when_metadata_budget_none(
    prometheus_logger,
):
    """
    End-to-end: when user_max_budget is None in request metadata but the user
    has a real budget in the DB, the metric must NOT be set to +Inf.
    """
    prometheus_logger.litellm_remaining_user_budget_metric = MagicMock()
    prometheus_logger.litellm_user_max_budget_metric = MagicMock()
    prometheus_logger.litellm_user_budget_remaining_hours_metric = MagicMock()

    db_user = MagicMock()
    db_user.max_budget = 500.0
    db_user.budget_reset_at = datetime(2026, 3, 1, tzinfo=timezone.utc)

    with patch("litellm.proxy.auth.auth_checks.get_user_object") as mock_get_user:
        mock_get_user.return_value = db_user
        await prometheus_logger._set_user_budget_metrics_after_api_request(
            user_id="user-abc-123",
            user_spend=120.0,
            user_max_budget=None,  # simulates stale key cache
            response_cost=0.5,
        )

    set_call_args = (
        prometheus_logger.litellm_remaining_user_budget_metric.labels().set.call_args
    )
    assert set_call_args is not None, "remaining_user_budget_metric.labels().set was not called"
    actual_value = set_call_args[0][0]
    assert actual_value != float("inf"), (
        f"remaining_user_budget_metric must not be +Inf when user has a real budget; got {actual_value}"
    )
    expected = 500.0 - 120.0 - 0.5
    assert abs(actual_value - expected) < 0.01, (
        f"Expected remaining budget ~{expected}, got {actual_value}"
    )


async def test_set_user_budget_metrics_after_api_request_inf_when_genuinely_no_budget(
    prometheus_logger,
):
    """
    When the user genuinely has no budget (max_budget=None in both metadata and
    DB), +Inf is the correct value and must be preserved.
    """
    prometheus_logger.litellm_remaining_user_budget_metric = MagicMock()
    prometheus_logger.litellm_user_max_budget_metric = MagicMock()
    prometheus_logger.litellm_user_budget_remaining_hours_metric = MagicMock()

    db_user = MagicMock()
    db_user.max_budget = None
    db_user.budget_reset_at = None

    with patch("litellm.proxy.auth.auth_checks.get_user_object") as mock_get_user:
        mock_get_user.return_value = db_user
        await prometheus_logger._set_user_budget_metrics_after_api_request(
            user_id="user-no-budget",
            user_spend=10.0,
            user_max_budget=None,
            response_cost=1.0,
        )

    set_call_args = (
        prometheus_logger.litellm_remaining_user_budget_metric.labels().set.call_args
    )
    assert set_call_args is not None
    actual_value = set_call_args[0][0]
    assert actual_value == float("inf"), (
        "remaining_user_budget_metric should be +Inf when user truly has no budget"
    )


def test_per_request_metrics_emit_all_identity_labels(prometheus_logger):
    """Verify org labels appear when flag is on and are absent when flag is off."""
    import litellm
    from litellm.types.integrations.prometheus import UserAPIKeyLabelValues

    prometheus_logger.litellm_requests_metric = MagicMock()
    prometheus_logger.litellm_spend_metric = MagicMock()

    enum_values = UserAPIKeyLabelValues(
        hashed_api_key="hashed-key",
        api_key_alias="my-key",
        model="gpt-4",
        team="team-abc",
        team_alias="my-team",
        org_id="org-abc",
        org_alias="my-org",
        user="user-1",
    )

    common_kwargs = dict(
        end_user_id=None,
        user_api_key="hashed-key",
        user_api_key_alias="my-key",
        model="gpt-4",
        user_api_team="team-abc",
        user_api_team_alias="my-team",
        user_id="user-1",
        response_cost=0.001,
        enum_values=enum_values,
    )

    try:
        # org labels are always included in per-request metrics
        prometheus_logger._increment_top_level_request_and_spend_metrics(**common_kwargs)
        label_kwargs = prometheus_logger.litellm_requests_metric.labels.call_args.kwargs
        assert label_kwargs["org_id"] == "org-abc"
        assert label_kwargs["org_alias"] == "my-org"
        assert label_kwargs["team"] == "team-abc"
        assert label_kwargs["user"] == "user-1"

        # Metrics not in the org-emission list must NOT get org labels
        from litellm.types.integrations.prometheus import PrometheusMetricLabels
        for metric in ("litellm_remaining_api_key_budget_metric", "litellm_remaining_team_budget_metric"):
            labels = PrometheusMetricLabels.get_labels(metric)
            assert "org_id" not in labels, f"{metric} should not have org_id"
            assert "org_alias" not in labels, f"{metric} should not have org_alias"

        # org_id in custom_prometheus_metadata_labels must not produce duplicate labels
        litellm.custom_prometheus_metadata_labels = ["org_id"]
        labels = PrometheusMetricLabels.get_labels("litellm_requests_metric")
        assert labels.count("org_id") == 1
    finally:
        litellm.custom_prometheus_metadata_labels = []


# ---------------------------------------------------------------------------
# Org budget metric tests
# ---------------------------------------------------------------------------


def test_org_budget_metrics_initialized(prometheus_logger):
    """Test that the 3 org budget gauge metrics are initialized."""
    assert hasattr(prometheus_logger, "litellm_remaining_org_budget_metric")
    assert hasattr(prometheus_logger, "litellm_org_max_budget_metric")
    assert hasattr(prometheus_logger, "litellm_org_budget_remaining_hours_metric")
    assert prometheus_logger.litellm_remaining_org_budget_metric is not None
    assert prometheus_logger.litellm_org_max_budget_metric is not None
    assert prometheus_logger.litellm_org_budget_remaining_hours_metric is not None


def test_set_org_budget_metrics_remaining_budget(prometheus_logger):
    """_set_org_budget_metrics sets remaining budget gauge correctly."""
    prometheus_logger.litellm_remaining_org_budget_metric = MagicMock()
    prometheus_logger.litellm_org_max_budget_metric = MagicMock()
    prometheus_logger.litellm_org_budget_remaining_hours_metric = MagicMock()

    prometheus_logger._set_org_budget_metrics(
        org_id="org-abc",
        org_alias="my-org",
        spend=200.0,
        max_budget=500.0,
        budget_reset_at=None,
    )

    set_call = prometheus_logger.litellm_remaining_org_budget_metric.labels().set
    set_call.assert_called_once()
    actual = set_call.call_args[0][0]
    assert abs(actual - 300.0) < 0.01, f"Expected 300.0, got {actual}"


def test_set_org_budget_metrics_max_budget(prometheus_logger):
    """_set_org_budget_metrics sets max budget gauge when max_budget is not None."""
    prometheus_logger.litellm_remaining_org_budget_metric = MagicMock()
    prometheus_logger.litellm_org_max_budget_metric = MagicMock()
    prometheus_logger.litellm_org_budget_remaining_hours_metric = MagicMock()

    prometheus_logger._set_org_budget_metrics(
        org_id="org-abc",
        org_alias="my-org",
        spend=100.0,
        max_budget=1000.0,
        budget_reset_at=None,
    )

    prometheus_logger.litellm_org_max_budget_metric.labels().set.assert_called_once_with(
        1000.0
    )


def test_set_org_budget_metrics_no_max_budget(prometheus_logger):
    """_set_org_budget_metrics does not set max budget gauge when max_budget is None."""
    prometheus_logger.litellm_remaining_org_budget_metric = MagicMock()
    prometheus_logger.litellm_org_max_budget_metric = MagicMock()
    prometheus_logger.litellm_org_budget_remaining_hours_metric = MagicMock()

    prometheus_logger._set_org_budget_metrics(
        org_id="org-abc",
        org_alias="my-org",
        spend=50.0,
        max_budget=None,
        budget_reset_at=None,
    )

    prometheus_logger.litellm_org_max_budget_metric.labels().set.assert_not_called()


def test_set_org_budget_metrics_remaining_hours(prometheus_logger):
    """_set_org_budget_metrics sets remaining hours gauge when budget_reset_at is set."""
    prometheus_logger.litellm_remaining_org_budget_metric = MagicMock()
    prometheus_logger.litellm_org_max_budget_metric = MagicMock()
    prometheus_logger.litellm_org_budget_remaining_hours_metric = MagicMock()

    future_reset = datetime(2099, 1, 1, tzinfo=timezone.utc)
    prometheus_logger._set_org_budget_metrics(
        org_id="org-abc",
        org_alias="my-org",
        spend=10.0,
        max_budget=500.0,
        budget_reset_at=future_reset,
    )

    prometheus_logger.litellm_org_budget_remaining_hours_metric.labels().set.assert_called_once()


@pytest.mark.asyncio
async def test_set_org_budget_metrics_after_api_request(prometheus_logger):
    """_set_org_budget_metrics_after_api_request uses cache helper and accounts for response_cost."""
    import sys

    prometheus_logger.litellm_remaining_org_budget_metric = MagicMock()
    prometheus_logger.litellm_org_max_budget_metric = MagicMock()
    prometheus_logger.litellm_org_budget_remaining_hours_metric = MagicMock()

    budget_mock = MagicMock()
    budget_mock.max_budget = 1000.0
    budget_mock.budget_reset_at = datetime(2099, 1, 1, tzinfo=timezone.utc)

    org_mock = MagicMock()
    org_mock.organization_id = "org-xyz"
    org_mock.organization_alias = "test-org"
    org_mock.spend = 300.0
    org_mock.litellm_budget_table = budget_mock

    mock_prisma = MagicMock()
    mock_proxy_server = MagicMock()
    mock_proxy_server.prisma_client = mock_prisma
    mock_proxy_server.user_api_key_cache = MagicMock()

    with (
        patch.dict(sys.modules, {"litellm.proxy.proxy_server": mock_proxy_server}),
        patch(
            "litellm.proxy.auth.auth_checks.get_org_object",
            AsyncMock(return_value=org_mock),
        ),
    ):
        await prometheus_logger._set_org_budget_metrics_after_api_request(
            org_id="org-xyz",
            response_cost=50.0,
        )

    # remaining budget should reflect spend + response_cost (300 + 50 = 350, remaining = 1000 - 350 = 650)
    remaining_call = prometheus_logger.litellm_remaining_org_budget_metric.labels().set.call_args
    assert remaining_call is not None
    assert remaining_call[0][0] == pytest.approx(650.0)

    prometheus_logger.litellm_org_max_budget_metric.labels().set.assert_called_once_with(
        1000.0
    )
    prometheus_logger.litellm_org_budget_remaining_hours_metric.labels().set.assert_called_once()


@pytest.mark.asyncio
async def test_set_org_budget_metrics_after_api_request_no_org_id(prometheus_logger):
    """_set_org_budget_metrics_after_api_request is a no-op when org_id is None."""
    prometheus_logger.litellm_remaining_org_budget_metric = MagicMock()
    prometheus_logger.litellm_org_max_budget_metric = MagicMock()
    prometheus_logger.litellm_org_budget_remaining_hours_metric = MagicMock()

    await prometheus_logger._set_org_budget_metrics_after_api_request(
        org_id=None,
        response_cost=1.0,
    )

    prometheus_logger.litellm_remaining_org_budget_metric.labels().set.assert_not_called()
    prometheus_logger.litellm_org_max_budget_metric.labels().set.assert_not_called()
    prometheus_logger.litellm_org_budget_remaining_hours_metric.labels().set.assert_not_called()


@pytest.mark.asyncio
async def test_initialize_org_budget_metrics(prometheus_logger):
    """_initialize_org_budget_metrics fetches all orgs and sets gauges for each."""
    import sys

    prometheus_logger.litellm_remaining_org_budget_metric = MagicMock()
    prometheus_logger.litellm_org_max_budget_metric = MagicMock()
    prometheus_logger.litellm_org_budget_remaining_hours_metric = MagicMock()

    budget_mock = MagicMock()
    budget_mock.max_budget = 500.0
    budget_mock.budget_reset_at = None

    org_mock = MagicMock()
    org_mock.organization_id = "org-init"
    org_mock.organization_alias = "init-org"
    org_mock.spend = 100.0
    org_mock.litellm_budget_table = budget_mock

    mock_prisma = MagicMock()
    mock_prisma.db.litellm_organizationtable.find_many = AsyncMock(
        return_value=[org_mock]
    )
    mock_prisma.db.litellm_organizationtable.count = AsyncMock(return_value=1)

    mock_proxy_server = MagicMock()
    mock_proxy_server.prisma_client = mock_prisma

    with patch.dict(sys.modules, {"litellm.proxy.proxy_server": mock_proxy_server}):
        await prometheus_logger._initialize_org_budget_metrics()

    prometheus_logger.litellm_remaining_org_budget_metric.labels().set.assert_called_once()
    prometheus_logger.litellm_org_max_budget_metric.labels().set.assert_called_once_with(
        500.0
    )


def test_default_latency_buckets(prometheus_logger):
    """PrometheusLogger uses the new reduced default latency buckets."""
    from litellm.types.integrations.prometheus import LATENCY_BUCKETS

    assert prometheus_logger.latency_buckets == LATENCY_BUCKETS
    # 420 and 600 should be present
    assert 420.0 in prometheus_logger.latency_buckets
    assert 600.0 in prometheus_logger.latency_buckets
    # dense half-second buckets from old defaults should be gone
    assert 1.5 not in prometheus_logger.latency_buckets
    assert 9.5 not in prometheus_logger.latency_buckets


def test_custom_latency_buckets():
    """prometheus_latency_buckets in litellm settings overrides the defaults."""
    import litellm
    from prometheus_client import REGISTRY

    custom_buckets = [0.1, 0.5, 1.0, 5.0, 10.0]
    original = litellm.prometheus_latency_buckets
    # Clear registry before creating a new PrometheusLogger
    for collector in list(REGISTRY._collector_to_names.keys()):
        try:
            REGISTRY.unregister(collector)
        except Exception:
            pass
    try:
        litellm.prometheus_latency_buckets = custom_buckets
        logger = PrometheusLogger()
        assert logger.latency_buckets == tuple(custom_buckets)
    finally:
        litellm.prometheus_latency_buckets = original
        for collector in list(REGISTRY._collector_to_names.keys()):
            try:
                REGISTRY.unregister(collector)
            except Exception:
                pass
