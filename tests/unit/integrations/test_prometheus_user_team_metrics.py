"""
Unit tests for Prometheus user and team count metrics
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

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












# ---------------------------------------------------------------------------
# Org budget metric tests
# ---------------------------------------------------------------------------


















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










































