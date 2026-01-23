"""
Unit tests for Prometheus user and team count metrics
"""
from unittest.mock import MagicMock

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
