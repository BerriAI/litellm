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
        assert (
            "litellm_total_users" in metrics or "litellm_total_users_total" in metrics
        )
        assert (
            "litellm_teams_count" in metrics or "litellm_teams_count_total" in metrics
        )

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

    def test_active_users_metric_initialized(self, prometheus_logger):
        """litellm_active_users gauge must exist alongside litellm_total_users."""
        assert hasattr(prometheus_logger, "litellm_active_users_metric")
        assert prometheus_logger.litellm_active_users_metric is not None

    @pytest.mark.asyncio
    async def test_initialize_counts_total_and_active_users(self, prometheus_logger):
        """litellm_total_users counts every row; litellm_active_users counts only
        billable (non SCIM-deactivated) users."""
        import sys

        prometheus_logger.litellm_total_users_metric = MagicMock()
        prometheus_logger.litellm_active_users_metric = MagicMock()
        prometheus_logger.litellm_teams_count_metric = MagicMock()

        async def _user_count(*args, where=None, **kwargs):
            # 10 rows, 2 of them SCIM-deactivated -> 8 billable
            return 2 if where is not None else 10

        mock_prisma = MagicMock()
        mock_prisma.db.litellm_usertable.count = _user_count
        mock_prisma.db.litellm_teamtable.count = AsyncMock(return_value=4)

        mock_proxy_server = MagicMock()
        mock_proxy_server.prisma_client = mock_prisma

        with patch.dict(sys.modules, {"litellm.proxy.proxy_server": mock_proxy_server}):
            await prometheus_logger._initialize_user_and_team_count_metrics()

        prometheus_logger.litellm_total_users_metric.set.assert_called_once_with(10)
        prometheus_logger.litellm_active_users_metric.set.assert_called_once_with(8)
        prometheus_logger.litellm_teams_count_metric.set.assert_called_once_with(4)

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

    assert (
        team_object.max_budget == 3000.0
    ), "max_budget should be populated from DB when metadata value is None"
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

    assert (
        team_object.max_budget == 100.0
    ), "max_budget from metadata must not be replaced by the DB value"


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
    assert (
        set_call_args is not None
    ), "remaining_team_budget_metric.labels().set was not called"
    actual_value = set_call_args[0][0]
    assert actual_value != float(
        "inf"
    ), f"remaining_team_budget_metric must not be +Inf when team has a real budget; got {actual_value}"
    expected = 3000.0 - 1617.02 - 0.5
    assert (
        abs(actual_value - expected) < 0.01
    ), f"Expected remaining budget ~{expected}, got {actual_value}"


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
    assert actual_value == float(
        "inf"
    ), "remaining_team_budget_metric should be +Inf when team truly has no budget"


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

    assert (
        user_object.max_budget == 500.0
    ), "max_budget should be populated from DB when metadata value is None"
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

    assert (
        user_object.max_budget == 100.0
    ), "max_budget from metadata must not be replaced by the DB value"


async def test_assemble_user_object_populates_user_email_and_alias_from_db(
    prometheus_logger,
):
    db_user = MagicMock()
    db_user.max_budget = None
    db_user.budget_reset_at = None
    db_user.user_email = "alice@example.com"
    db_user.user_alias = "Alice"

    with patch("litellm.proxy.auth.auth_checks.get_user_object") as mock_get_user:
        mock_get_user.return_value = db_user
        user_object = await prometheus_logger._assemble_user_object(
            user_id="user-abc-123",
            spend=10.0,
            max_budget=None,
            response_cost=0.5,
        )

    assert user_object.user_email == "alice@example.com"
    assert user_object.user_alias == "Alice"


def test_set_user_budget_metrics_default_no_email_alias_labels(
    prometheus_logger,
):
    """By default (flag off), only user label is emitted."""
    import litellm
    from litellm.proxy._types import LiteLLM_UserTable

    litellm.prometheus_user_budget_label_include_email_alias = False

    user = LiteLLM_UserTable(
        user_id="user-abc-123",
        user_email="alice@example.com",
        user_alias="Alice",
        spend=25.0,
        max_budget=100.0,
        budget_reset_at=datetime(2026, 3, 1, tzinfo=timezone.utc),
    )

    prometheus_logger.litellm_remaining_user_budget_metric = MagicMock()
    prometheus_logger.litellm_user_max_budget_metric = MagicMock()
    prometheus_logger.litellm_user_budget_remaining_hours_metric = MagicMock()

    prometheus_logger._set_user_budget_metrics(user)

    prometheus_logger.litellm_remaining_user_budget_metric.labels.assert_called_once_with(
        user="user-abc-123",
    )


def test_set_user_budget_metrics_includes_user_email_and_alias_labels_when_opted_in():
    """When prometheus_user_budget_label_include_email_alias=True, email+alias labels appear.

    The flag is read once per metric at logger construction time and snapshotted,
    so it must be enabled before the PrometheusLogger is built (mirroring how the
    proxy applies config at startup before instantiating callbacks).
    """
    import litellm
    from litellm.proxy._types import LiteLLM_UserTable

    litellm.prometheus_user_budget_label_include_email_alias = True

    try:
        prometheus_logger = PrometheusLogger()

        user = LiteLLM_UserTable(
            user_id="user-abc-123",
            user_email="alice@example.com",
            user_alias="Alice",
            spend=25.0,
            max_budget=100.0,
            budget_reset_at=datetime(2026, 3, 1, tzinfo=timezone.utc),
        )

        prometheus_logger.litellm_remaining_user_budget_metric = MagicMock()
        prometheus_logger.litellm_user_max_budget_metric = MagicMock()
        prometheus_logger.litellm_user_budget_remaining_hours_metric = MagicMock()

        prometheus_logger._set_user_budget_metrics(user)

        prometheus_logger.litellm_remaining_user_budget_metric.labels.assert_called_once_with(
            user="user-abc-123",
            user_email="alice@example.com",
            user_alias="Alice",
        )
        prometheus_logger.litellm_remaining_user_budget_metric.labels().set.assert_called_once_with(
            75.0
        )
        prometheus_logger.litellm_user_max_budget_metric.labels.assert_called_once_with(
            user="user-abc-123",
            user_email="alice@example.com",
            user_alias="Alice",
        )
        prometheus_logger.litellm_user_budget_remaining_hours_metric.labels.assert_called_once_with(
            user="user-abc-123",
            user_email="alice@example.com",
            user_alias="Alice",
        )
    finally:
        litellm.prometheus_user_budget_label_include_email_alias = False


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
    assert (
        set_call_args is not None
    ), "remaining_user_budget_metric.labels().set was not called"
    actual_value = set_call_args[0][0]
    assert actual_value != float(
        "inf"
    ), f"remaining_user_budget_metric must not be +Inf when user has a real budget; got {actual_value}"
    expected = 500.0 - 120.0 - 0.5
    assert (
        abs(actual_value - expected) < 0.01
    ), f"Expected remaining budget ~{expected}, got {actual_value}"


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
    assert actual_value == float(
        "inf"
    ), "remaining_user_budget_metric should be +Inf when user truly has no budget"


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
        prometheus_logger._increment_top_level_request_and_spend_metrics(
            **common_kwargs
        )
        label_kwargs = prometheus_logger.litellm_requests_metric.labels.call_args.kwargs
        assert label_kwargs["org_id"] == "org-abc"
        assert label_kwargs["org_alias"] == "my-org"
        assert label_kwargs["team"] == "team-abc"
        assert label_kwargs["user"] == "user-1"

        # Metrics not in the org-emission list must NOT get org labels
        from litellm.types.integrations.prometheus import PrometheusMetricLabels

        for metric in (
            "litellm_remaining_api_key_budget_metric",
            "litellm_remaining_team_budget_metric",
        ):
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
    remaining_call = (
        prometheus_logger.litellm_remaining_org_budget_metric.labels().set.call_args
    )
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


@pytest.fixture
def customer_metrics_enabled(monkeypatch):
    import litellm

    monkeypatch.setattr(litellm, "enable_end_user_cost_tracking_prometheus_only", True)
    monkeypatch.setattr(litellm, "disable_end_user_cost_tracking", False)
    monkeypatch.setattr(litellm, "max_end_user_budget_id", None)


def _customer_sample(metric_name: str, end_user_id: str):
    return REGISTRY.get_sample_value(metric_name, {"end_user": end_user_id})


def _mock_customer_row(user_id: str, spend: float, max_budget: float | None, budget_reset_at):
    budget_mock = MagicMock()
    budget_mock.max_budget = max_budget
    budget_mock.budget_reset_at = budget_reset_at
    row = MagicMock()
    row.user_id = user_id
    row.spend = spend
    row.litellm_budget_table = budget_mock
    return row


@pytest.mark.parametrize(
    "spend, max_budget, expected_remaining",
    [(125.0, 500.0, 375.0), (500.0, 500.0, 0.0), (0.0, 500.0, 500.0)],
)
def test_set_customer_budget_metrics_emits_remaining_and_max_budget(
    prometheus_logger, customer_metrics_enabled, spend, max_budget, expected_remaining
):
    prometheus_logger._set_customer_budget_metrics(
        end_user_id="cust-1",
        spend=spend,
        max_budget=max_budget,
        budget_reset_at=None,
    )

    assert _customer_sample("litellm_remaining_customer_budget_metric", "cust-1") == pytest.approx(
        expected_remaining
    )
    assert _customer_sample("litellm_customer_max_budget_metric", "cust-1") == pytest.approx(max_budget)
    assert _customer_sample("litellm_customer_budget_remaining_hours_metric", "cust-1") is None


def test_set_customer_budget_metrics_remaining_hours(prometheus_logger, customer_metrics_enabled):
    reset_at = datetime(2099, 1, 1, tzinfo=timezone.utc)
    prometheus_logger._set_customer_budget_metrics(
        end_user_id="cust-1",
        spend=1.0,
        max_budget=10.0,
        budget_reset_at=reset_at,
    )

    expected_hours = (reset_at - datetime.now(timezone.utc)).total_seconds() / 3600
    assert _customer_sample("litellm_customer_budget_remaining_hours_metric", "cust-1") == pytest.approx(
        expected_hours, abs=0.1
    )


def test_set_customer_budget_metrics_not_emitted_when_end_user_tracking_off(prometheus_logger, monkeypatch):
    import litellm

    monkeypatch.setattr(litellm, "enable_end_user_cost_tracking_prometheus_only", False)
    monkeypatch.setattr(litellm, "disable_end_user_cost_tracking", False)

    prometheus_logger._set_customer_budget_metrics(
        end_user_id="cust-off",
        spend=1.0,
        max_budget=10.0,
        budget_reset_at=datetime(2099, 1, 1, tzinfo=timezone.utc),
    )

    assert prometheus_logger.litellm_remaining_customer_budget_metric._metrics == {}
    assert prometheus_logger.litellm_customer_max_budget_metric._metrics == {}
    assert prometheus_logger.litellm_customer_budget_remaining_hours_metric._metrics == {}


def test_set_customer_budget_metrics_without_budget_only_emits_remaining(prometheus_logger, customer_metrics_enabled):
    prometheus_logger._set_customer_budget_metrics(
        end_user_id="cust-free",
        spend=3.0,
        max_budget=None,
        budget_reset_at=None,
    )

    assert _customer_sample("litellm_remaining_customer_budget_metric", "cust-free") == float("inf")
    assert _customer_sample("litellm_customer_max_budget_metric", "cust-free") is None
    assert _customer_sample("litellm_customer_budget_remaining_hours_metric", "cust-free") is None


@pytest.mark.asyncio
async def test_increment_remaining_budget_metrics_emits_customer_gauges_from_cached_end_user(
    prometheus_logger, customer_metrics_enabled
):
    import sys

    from litellm.models.budget import LiteLLM_BudgetTable
    from litellm.models.end_user import LiteLLM_EndUserTable

    end_user = LiteLLM_EndUserTable(
        user_id="cust-req",
        blocked=False,
        spend=300.0,
        budget_id="budget-1",
        litellm_budget_table=LiteLLM_BudgetTable(budget_id="budget-1", max_budget=1000.0),
    )
    get_end_user_object = AsyncMock()
    mock_proxy_server = MagicMock()
    mock_proxy_server.prisma_client = None
    mock_proxy_server.user_api_key_cache.async_get_cache = AsyncMock(return_value=end_user)

    with (
        patch.dict(sys.modules, {"litellm.proxy.proxy_server": mock_proxy_server}),
        patch("litellm.proxy.auth.auth_checks.get_end_user_object", get_end_user_object),  # test-quality-ok: [TQ008] assert the request path never reaches the DB-backed auth lookup
    ):
        await prometheus_logger._increment_remaining_budget_metrics(
            user_api_team=None,
            user_api_team_alias=None,
            user_api_key=None,
            user_api_key_alias=None,
            litellm_params={"metadata": {}},
            response_cost=50.0,
            end_user_id="cust-req",
        )

    get_end_user_object.assert_not_awaited()
    cache_read = mock_proxy_server.user_api_key_cache.async_get_cache
    cache_read.assert_awaited_once()
    assert cache_read.await_args.kwargs["key"] == "end_user_id:cust-req"
    assert _customer_sample("litellm_remaining_customer_budget_metric", "cust-req") == pytest.approx(650.0)
    assert _customer_sample("litellm_customer_max_budget_metric", "cust-req") == pytest.approx(1000.0)


@pytest.mark.asyncio
async def test_set_customer_budget_metrics_after_api_request_uses_cached_default_budget(
    prometheus_logger, customer_metrics_enabled
):
    import sys

    from litellm.models.budget import LiteLLM_BudgetTable
    from litellm.models.end_user import LiteLLM_EndUserTable

    end_user = LiteLLM_EndUserTable(
        user_id="cust-default",
        blocked=False,
        spend=0.5,
        budget_id=None,
        litellm_budget_table=LiteLLM_BudgetTable(budget_id="default-budget", max_budget=3.0),
    )
    mock_proxy_server = MagicMock()
    mock_proxy_server.user_api_key_cache.async_get_cache = AsyncMock(return_value=end_user)

    with patch.dict(sys.modules, {"litellm.proxy.proxy_server": mock_proxy_server}):
        await prometheus_logger._set_customer_budget_metrics_after_api_request(
            end_user_id="cust-default",
            response_cost=0.5,
        )

    assert _customer_sample("litellm_remaining_customer_budget_metric", "cust-default") == pytest.approx(2.0)
    assert _customer_sample("litellm_customer_max_budget_metric", "cust-default") == pytest.approx(3.0)


@pytest.mark.asyncio
async def test_set_customer_budget_metrics_after_api_request_without_budget_only_emits_remaining(
    prometheus_logger, customer_metrics_enabled
):
    import sys

    from litellm.models.end_user import LiteLLM_EndUserTable

    end_user = LiteLLM_EndUserTable(user_id="cust-no-budget", blocked=False, spend=2.0, budget_id=None)
    mock_proxy_server = MagicMock()
    mock_proxy_server.user_api_key_cache.async_get_cache = AsyncMock(return_value=end_user)

    with patch.dict(sys.modules, {"litellm.proxy.proxy_server": mock_proxy_server}):
        await prometheus_logger._set_customer_budget_metrics_after_api_request(
            end_user_id="cust-no-budget",
            response_cost=1.0,
        )

    assert _customer_sample("litellm_remaining_customer_budget_metric", "cust-no-budget") == float("inf")
    assert _customer_sample("litellm_customer_max_budget_metric", "cust-no-budget") is None


@pytest.mark.asyncio
async def test_set_customer_budget_metrics_after_api_request_skips_uncached_customer(
    prometheus_logger, customer_metrics_enabled
):
    import sys

    get_end_user_object = AsyncMock()
    mock_proxy_server = MagicMock()
    mock_proxy_server.prisma_client = MagicMock()
    mock_proxy_server.user_api_key_cache.async_get_cache = AsyncMock(return_value=None)

    with (
        patch.dict(sys.modules, {"litellm.proxy.proxy_server": mock_proxy_server}),
        patch("litellm.proxy.auth.auth_checks.get_end_user_object", get_end_user_object),  # test-quality-ok: [TQ008] assert a cache miss does not fall back to the DB-backed auth lookup
    ):
        await prometheus_logger._set_customer_budget_metrics_after_api_request(
            end_user_id="cust-uncached",
            response_cost=1.0,
        )

    get_end_user_object.assert_not_awaited()
    mock_proxy_server.prisma_client.assert_not_called()
    assert prometheus_logger.litellm_remaining_customer_budget_metric._metrics == {}


@pytest.mark.asyncio
async def test_set_customer_budget_metrics_after_api_request_without_end_user_is_noop(prometheus_logger):
    import sys

    mock_proxy_server = MagicMock()
    mock_proxy_server.user_api_key_cache.async_get_cache = AsyncMock()

    with patch.dict(sys.modules, {"litellm.proxy.proxy_server": mock_proxy_server}):
        await prometheus_logger._set_customer_budget_metrics_after_api_request(
            end_user_id=None,
            response_cost=1.0,
        )

    mock_proxy_server.user_api_key_cache.async_get_cache.assert_not_awaited()
    assert prometheus_logger.litellm_remaining_customer_budget_metric._metrics == {}


@pytest.mark.asyncio
async def test_initialize_customer_budget_metrics_emits_gauges_for_budgeted_customers(
    prometheus_logger, customer_metrics_enabled
):
    import sys

    reset_at = datetime(2099, 1, 1, tzinfo=timezone.utc)
    rows = [
        _mock_customer_row("cust-a", 100.0, 500.0, None),
        _mock_customer_row("cust-b", 20.0, 50.0, reset_at),
    ]
    find_many = AsyncMock(return_value=rows)
    mock_prisma = MagicMock()
    mock_prisma.db.litellm_endusertable.find_many = find_many
    mock_prisma.db.litellm_endusertable.count = AsyncMock(return_value=len(rows))
    mock_proxy_server = MagicMock()
    mock_proxy_server.prisma_client = mock_prisma

    with patch.dict(sys.modules, {"litellm.proxy.proxy_server": mock_proxy_server}):
        await prometheus_logger._initialize_customer_budget_metrics()

    assert find_many.await_args.kwargs["where"] == {"budget_id": {"not": None}}
    assert _customer_sample("litellm_remaining_customer_budget_metric", "cust-a") == pytest.approx(400.0)
    assert _customer_sample("litellm_customer_max_budget_metric", "cust-a") == pytest.approx(500.0)
    assert _customer_sample("litellm_customer_budget_remaining_hours_metric", "cust-a") is None
    assert _customer_sample("litellm_remaining_customer_budget_metric", "cust-b") == pytest.approx(30.0)
    assert _customer_sample("litellm_customer_max_budget_metric", "cust-b") == pytest.approx(50.0)
    assert _customer_sample("litellm_customer_budget_remaining_hours_metric", "cust-b") > 0


@pytest.mark.parametrize(
    "enable_prometheus_only, disable_end_user",
    [(False, False), (True, True)],
)
@pytest.mark.asyncio
async def test_initialize_customer_budget_metrics_skips_when_end_user_tracking_off(
    prometheus_logger, monkeypatch, enable_prometheus_only, disable_end_user
):
    import sys

    import litellm

    monkeypatch.setattr(litellm, "enable_end_user_cost_tracking_prometheus_only", enable_prometheus_only)
    monkeypatch.setattr(litellm, "disable_end_user_cost_tracking", disable_end_user)

    find_many = AsyncMock(return_value=[_mock_customer_row("cust-a", 100.0, 500.0, None)])
    mock_prisma = MagicMock()
    mock_prisma.db.litellm_endusertable.find_many = find_many
    mock_prisma.db.litellm_endusertable.count = AsyncMock(return_value=1)
    mock_proxy_server = MagicMock()
    mock_proxy_server.prisma_client = mock_prisma

    with patch.dict(sys.modules, {"litellm.proxy.proxy_server": mock_proxy_server}):
        await prometheus_logger._initialize_customer_budget_metrics()

    find_many.assert_not_awaited()
    assert _customer_sample("litellm_remaining_customer_budget_metric", "cust-a") is None


@pytest.mark.asyncio
async def test_initialize_remaining_budget_metrics_includes_customers(prometheus_logger, customer_metrics_enabled):
    import sys

    mock_prisma = MagicMock()
    mock_prisma.db.litellm_endusertable.find_many = AsyncMock(
        return_value=[_mock_customer_row("cust-startup", 5.0, 25.0, None)]
    )
    mock_prisma.db.litellm_endusertable.count = AsyncMock(return_value=1)
    mock_proxy_server = MagicMock()
    mock_proxy_server.prisma_client = mock_prisma

    with patch.dict(sys.modules, {"litellm.proxy.proxy_server": mock_proxy_server}):
        await prometheus_logger._initialize_remaining_budget_metrics()

    assert _customer_sample("litellm_remaining_customer_budget_metric", "cust-startup") == pytest.approx(20.0)


@pytest.mark.asyncio
async def test_initialize_customer_budget_metrics_counts_once_across_pages(prometheus_logger, customer_metrics_enabled):
    import sys

    pages = [
        [_mock_customer_row(f"cust-{i}", 1.0, 10.0, None) for i in range(50)],
        [_mock_customer_row(f"cust-{i}", 1.0, 10.0, None) for i in range(50, 100)],
        [_mock_customer_row("cust-100", 1.0, 10.0, None)],
    ]
    find_many = AsyncMock(side_effect=pages)
    count = AsyncMock(return_value=101)
    mock_prisma = MagicMock()
    mock_prisma.db.litellm_endusertable.find_many = find_many
    mock_prisma.db.litellm_endusertable.count = count
    mock_proxy_server = MagicMock()
    mock_proxy_server.prisma_client = mock_prisma

    with patch.dict(sys.modules, {"litellm.proxy.proxy_server": mock_proxy_server}):
        await prometheus_logger._initialize_customer_budget_metrics()

    assert find_many.await_count == 3
    count.assert_awaited_once()
    assert _customer_sample("litellm_remaining_customer_budget_metric", "cust-100") == pytest.approx(9.0)


@pytest.mark.asyncio
async def test_initialize_customer_budget_metrics_applies_default_budget_to_unbudgeted_customers(
    prometheus_logger, customer_metrics_enabled, monkeypatch
):
    import sys

    import litellm

    monkeypatch.setattr(litellm, "max_end_user_budget_id", "default-customer-budget")
    reset_at = datetime(2099, 1, 1, tzinfo=timezone.utc)
    default_budget = MagicMock()
    default_budget.max_budget = 10.0
    default_budget.budget_reset_at = reset_at
    explicit_row = _mock_customer_row("cust-explicit", 5.0, 100.0, None)
    default_row = _mock_customer_row("cust-default", 2.0, None, None)
    default_row.litellm_budget_table = None
    find_many = AsyncMock(return_value=[explicit_row, default_row])
    find_unique = AsyncMock(return_value=default_budget)
    mock_prisma = MagicMock()
    mock_prisma.db.litellm_endusertable.find_many = find_many
    mock_prisma.db.litellm_endusertable.count = AsyncMock(return_value=2)
    mock_prisma.db.litellm_budgettable.find_unique = find_unique
    mock_proxy_server = MagicMock()
    mock_proxy_server.prisma_client = mock_prisma

    with patch.dict(sys.modules, {"litellm.proxy.proxy_server": mock_proxy_server}):
        await prometheus_logger._initialize_customer_budget_metrics()

    assert find_unique.await_args.kwargs["where"] == {"budget_id": "default-customer-budget"}
    assert find_many.await_args.kwargs["where"] is None
    assert _customer_sample("litellm_remaining_customer_budget_metric", "cust-explicit") == pytest.approx(95.0)
    assert _customer_sample("litellm_customer_max_budget_metric", "cust-explicit") == pytest.approx(100.0)
    assert _customer_sample("litellm_remaining_customer_budget_metric", "cust-default") == pytest.approx(8.0)
    assert _customer_sample("litellm_customer_max_budget_metric", "cust-default") == pytest.approx(10.0)
    assert _customer_sample("litellm_customer_budget_remaining_hours_metric", "cust-default") > 0


@pytest.mark.asyncio
async def test_customer_max_budget_gauge_emitted_when_only_it_is_configured(customer_metrics_enabled, monkeypatch):
    import sys

    import litellm
    from litellm.models.budget import LiteLLM_BudgetTable
    from litellm.models.end_user import LiteLLM_EndUserTable
    from litellm.types.integrations.prometheus import NoOpMetric

    monkeypatch.setattr(
        litellm,
        "prometheus_metrics_config",
        [{"group": "customer-max-only", "metrics": ["litellm_customer_max_budget_metric"]}],
    )
    logger = PrometheusLogger()
    assert isinstance(logger.litellm_remaining_customer_budget_metric, NoOpMetric)
    assert not isinstance(logger.litellm_customer_max_budget_metric, NoOpMetric)

    end_user = LiteLLM_EndUserTable(
        user_id="cust-max-only",
        blocked=False,
        spend=1.0,
        budget_id="budget-1",
        litellm_budget_table=LiteLLM_BudgetTable(budget_id="budget-1", max_budget=40.0),
    )
    mock_proxy_server = MagicMock()
    mock_proxy_server.prisma_client = None
    mock_proxy_server.user_api_key_cache.async_get_cache = AsyncMock(return_value=end_user)

    with patch.dict(sys.modules, {"litellm.proxy.proxy_server": mock_proxy_server}):
        await logger._increment_remaining_budget_metrics(
            user_api_team=None,
            user_api_team_alias=None,
            user_api_key=None,
            user_api_key_alias=None,
            litellm_params={"metadata": {}},
            response_cost=1.0,
            end_user_id="cust-max-only",
        )

    assert _customer_sample("litellm_customer_max_budget_metric", "cust-max-only") == pytest.approx(40.0)


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


class TestSetTeamMembersMetric:
    """litellm_team_members_metric tracks the current member count per team."""

    def _gauge_value(self, team_id, team_alias):
        return REGISTRY.get_sample_value(
            "litellm_team_members_metric",
            {"team": team_id, "team_alias": team_alias},
        )

    def test_metric_initialized(self, prometheus_logger):
        assert hasattr(prometheus_logger, "litellm_team_members_metric")
        assert prometheus_logger.litellm_team_members_metric is not None

    @pytest.mark.parametrize("count", [0, 1, 3, 7])
    def test_sets_gauge_to_member_count(self, prometheus_logger, count):
        from litellm.proxy._types import LiteLLM_TeamTable, Member

        team = LiteLLM_TeamTable(
            team_id="team-a",
            team_alias="Acme",
            members_with_roles=[
                Member(user_id=f"u{i}", role="user") for i in range(count)
            ],
        )
        prometheus_logger.set_team_members_metric(team)
        assert self._gauge_value("team-a", "Acme") == float(count)

    def test_gauge_reflects_latest_count_not_delta(self, prometheus_logger):
        """Re-emitting overwrites with the authoritative count (set, not inc/dec)."""
        from litellm.proxy._types import LiteLLM_TeamTable, Member

        members = [Member(user_id=f"u{i}", role="user") for i in range(4)]
        team = LiteLLM_TeamTable(
            team_id="team-b", team_alias="Beta", members_with_roles=members
        )
        prometheus_logger.set_team_members_metric(team)
        assert self._gauge_value("team-b", "Beta") == 4.0

        # Drop two members and re-emit: gauge must read 2, not 4 and not -2.
        team.members_with_roles = members[:2]
        prometheus_logger.set_team_members_metric(team)
        assert self._gauge_value("team-b", "Beta") == 2.0

    def test_none_alias_falls_back_to_empty_string(self, prometheus_logger):
        from litellm.proxy._types import LiteLLM_TeamTable, Member

        team = LiteLLM_TeamTable(
            team_id="team-c",
            team_alias=None,
            members_with_roles=[Member(user_id="solo", role="admin")],
        )
        prometheus_logger.set_team_members_metric(team)
        assert self._gauge_value("team-c", "") == 1.0

    def test_teams_isolated_by_label(self, prometheus_logger):
        from litellm.proxy._types import LiteLLM_TeamTable, Member

        team_one = LiteLLM_TeamTable(
            team_id="team-1",
            team_alias="One",
            members_with_roles=[Member(user_id="a", role="user")],
        )
        team_two = LiteLLM_TeamTable(
            team_id="team-2",
            team_alias="Two",
            members_with_roles=[
                Member(user_id="b", role="user"),
                Member(user_id="c", role="user"),
                Member(user_id="d", role="user"),
            ],
        )
        prometheus_logger.set_team_members_metric(team_one)
        prometheus_logger.set_team_members_metric(team_two)
        assert self._gauge_value("team-1", "One") == 1.0
        assert self._gauge_value("team-2", "Two") == 3.0
