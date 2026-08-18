"""
Unit tests for the NoOpMetric guard in _increment_remaining_budget_metrics
and the per-entity guards in _set_*_budget_metrics_after_api_request.

Regression tests that the specific bug can never happen again:
when budget gauges are excluded from prometheus_metrics_config (and therefore
created as NoOpMetric instances), the DB/cache lookup helpers must not be called.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from prometheus_client import REGISTRY

import litellm
from litellm.integrations.prometheus import PrometheusLogger
from litellm.types.integrations.prometheus import NoOpMetric

_BUDGET_EXCLUDED_CONFIG = [
    {
        "group": "core-only",
        "metrics": [
            "litellm_requests_metric",
            "litellm_total_tokens_metric",
        ],
    }
]


@pytest.fixture(autouse=True)
def cleanup_prometheus_registry():
    old_config = litellm.prometheus_metrics_config
    for collector in list(REGISTRY._collector_to_names.keys()):
        try:
            REGISTRY.unregister(collector)
        except Exception:
            pass
    yield
    litellm.prometheus_metrics_config = old_config
    for collector in list(REGISTRY._collector_to_names.keys()):
        try:
            REGISTRY.unregister(collector)
        except Exception:
            pass


def make_logger_with_budget_metrics_disabled() -> PrometheusLogger:
    litellm.prometheus_metrics_config = _BUDGET_EXCLUDED_CONFIG
    return PrometheusLogger()


def make_logger_with_all_metrics_enabled() -> PrometheusLogger:
    litellm.prometheus_metrics_config = None
    return PrometheusLogger()


COMMON_KWARGS = dict(
    user_api_team="team-123",
    user_api_team_alias="my-team",
    user_api_key="hashed-key",
    user_api_key_alias="my-key",
    litellm_params={"metadata": {}},
    response_cost=0.001,
    user_id="user-1",
    user_api_key_org_id="org-1",
)


class TestBudgetGaugesAreNoopWhenExcluded:
    def test_team_gauge_is_noop(self):
        logger = make_logger_with_budget_metrics_disabled()
        assert isinstance(logger.litellm_remaining_team_budget_metric, NoOpMetric)

    def test_api_key_gauge_is_noop(self):
        logger = make_logger_with_budget_metrics_disabled()
        assert isinstance(logger.litellm_remaining_api_key_budget_metric, NoOpMetric)

    def test_user_gauge_is_noop(self):
        logger = make_logger_with_budget_metrics_disabled()
        assert isinstance(logger.litellm_remaining_user_budget_metric, NoOpMetric)

    def test_org_gauge_is_noop(self):
        logger = make_logger_with_budget_metrics_disabled()
        assert isinstance(logger.litellm_remaining_org_budget_metric, NoOpMetric)

    def test_gauges_are_real_when_all_metrics_enabled(self):
        logger = make_logger_with_all_metrics_enabled()
        assert not isinstance(logger.litellm_remaining_team_budget_metric, NoOpMetric)
        assert not isinstance(logger.litellm_remaining_api_key_budget_metric, NoOpMetric)
        assert not isinstance(logger.litellm_remaining_user_budget_metric, NoOpMetric)
        assert not isinstance(logger.litellm_remaining_org_budget_metric, NoOpMetric)


class TestTopLevelGuard:
    @pytest.mark.asyncio
    async def test_no_db_lookups_when_all_budget_gauges_are_noop(self):
        """Regression: _increment_remaining_budget_metrics must return early
        without any I/O when all four budget gauges are NoOpMetric."""
        logger = make_logger_with_budget_metrics_disabled()

        assemble_team = AsyncMock(return_value=MagicMock())
        assemble_key = AsyncMock(return_value=MagicMock())
        assemble_user = AsyncMock(return_value=MagicMock())

        with (
            patch.object(logger, "_assemble_team_object", assemble_team),
            patch.object(logger, "_assemble_key_object", assemble_key),
            patch.object(logger, "_assemble_user_object", assemble_user),
        ):
            await logger._increment_remaining_budget_metrics(**COMMON_KWARGS)

        assemble_team.assert_not_called()
        assemble_key.assert_not_called()
        assemble_user.assert_not_called()

    @pytest.mark.asyncio
    async def test_db_lookups_run_when_budget_gauges_are_real(self):
        """When budget gauges are real Prometheus metrics, the assemble helpers
        must be called so I/O proceeds normally."""
        logger = make_logger_with_all_metrics_enabled()

        assemble_team = AsyncMock(
            return_value=MagicMock(
                team_id="team-123",
                team_alias="my-team",
                spend=0.001,
                max_budget=None,
                budget_reset_at=None,
            )
        )
        assemble_key = AsyncMock(
            return_value=MagicMock(
                token="hashed-key",
                key_alias="my-key",
                spend=0.001,
                max_budget=None,
                budget_reset_at=None,
            )
        )
        assemble_user = AsyncMock(
            return_value=MagicMock(
                user_id="user-1",
                spend=0.001,
                max_budget=None,
                budget_reset_at=None,
                user_email=None,
                user_alias=None,
            )
        )

        with (
            patch.object(logger, "_assemble_team_object", assemble_team),
            patch.object(logger, "_assemble_key_object", assemble_key),
            patch.object(logger, "_assemble_user_object", assemble_user),
            patch.object(logger, "_set_team_budget_metrics", MagicMock()),
            patch.object(logger, "_set_key_budget_metrics", MagicMock()),
            patch.object(logger, "_set_user_budget_metrics", MagicMock()),
            patch.object(logger, "_set_org_budget_metrics_after_api_request", AsyncMock()),
        ):
            await logger._increment_remaining_budget_metrics(**COMMON_KWARGS)

        assemble_team.assert_called_once()
        assemble_key.assert_called_once()
        assemble_user.assert_called_once()


class TestPerEntityGuards:
    @pytest.mark.asyncio
    async def test_team_guard_skips_lookup_when_team_gauge_is_noop(self):
        """Per-entity guard: team assemble helper is not called when team gauge is NoOp,
        even when key and user gauges are real."""
        logger = make_logger_with_all_metrics_enabled()
        logger.litellm_remaining_team_budget_metric = NoOpMetric()

        assemble_team = AsyncMock(return_value=MagicMock())
        assemble_key = AsyncMock(
            return_value=MagicMock(
                token="hashed-key",
                key_alias="my-key",
                spend=0.001,
                max_budget=None,
                budget_reset_at=None,
            )
        )
        assemble_user = AsyncMock(
            return_value=MagicMock(
                user_id="user-1",
                spend=0.001,
                max_budget=None,
                budget_reset_at=None,
                user_email=None,
                user_alias=None,
            )
        )

        with (
            patch.object(logger, "_assemble_team_object", assemble_team),
            patch.object(logger, "_assemble_key_object", assemble_key),
            patch.object(logger, "_assemble_user_object", assemble_user),
            patch.object(logger, "_set_team_budget_metrics", MagicMock()),
            patch.object(logger, "_set_key_budget_metrics", MagicMock()),
            patch.object(logger, "_set_user_budget_metrics", MagicMock()),
            patch.object(logger, "_set_org_budget_metrics_after_api_request", AsyncMock()),
        ):
            await logger._increment_remaining_budget_metrics(**COMMON_KWARGS)

        assemble_team.assert_not_called()
        assemble_key.assert_called_once()
        assemble_user.assert_called_once()

    @pytest.mark.asyncio
    async def test_key_guard_skips_lookup_when_key_gauge_is_noop(self):
        """Per-entity guard: key assemble helper is not called when key gauge is NoOp,
        even when team and user gauges are real."""
        logger = make_logger_with_all_metrics_enabled()
        logger.litellm_remaining_api_key_budget_metric = NoOpMetric()

        assemble_team = AsyncMock(
            return_value=MagicMock(
                team_id="team-123",
                team_alias="my-team",
                spend=0.001,
                max_budget=None,
                budget_reset_at=None,
            )
        )
        assemble_key = AsyncMock(return_value=MagicMock())
        assemble_user = AsyncMock(
            return_value=MagicMock(
                user_id="user-1",
                spend=0.001,
                max_budget=None,
                budget_reset_at=None,
                user_email=None,
                user_alias=None,
            )
        )

        with (
            patch.object(logger, "_assemble_team_object", assemble_team),
            patch.object(logger, "_assemble_key_object", assemble_key),
            patch.object(logger, "_assemble_user_object", assemble_user),
            patch.object(logger, "_set_team_budget_metrics", MagicMock()),
            patch.object(logger, "_set_key_budget_metrics", MagicMock()),
            patch.object(logger, "_set_user_budget_metrics", MagicMock()),
            patch.object(logger, "_set_org_budget_metrics_after_api_request", AsyncMock()),
        ):
            await logger._increment_remaining_budget_metrics(**COMMON_KWARGS)

        assemble_key.assert_not_called()
        assemble_team.assert_called_once()
        assemble_user.assert_called_once()

    @pytest.mark.asyncio
    async def test_user_guard_skips_lookup_when_user_gauge_is_noop(self):
        """Per-entity guard: user assemble helper is not called when user gauge is NoOp,
        even when team and key gauges are real."""
        logger = make_logger_with_all_metrics_enabled()
        logger.litellm_remaining_user_budget_metric = NoOpMetric()

        assemble_team = AsyncMock(
            return_value=MagicMock(
                team_id="team-123",
                team_alias="my-team",
                spend=0.001,
                max_budget=None,
                budget_reset_at=None,
            )
        )
        assemble_key = AsyncMock(
            return_value=MagicMock(
                token="hashed-key",
                key_alias="my-key",
                spend=0.001,
                max_budget=None,
                budget_reset_at=None,
            )
        )
        assemble_user = AsyncMock(return_value=MagicMock())

        with (
            patch.object(logger, "_assemble_team_object", assemble_team),
            patch.object(logger, "_assemble_key_object", assemble_key),
            patch.object(logger, "_assemble_user_object", assemble_user),
            patch.object(logger, "_set_team_budget_metrics", MagicMock()),
            patch.object(logger, "_set_key_budget_metrics", MagicMock()),
            patch.object(logger, "_set_user_budget_metrics", MagicMock()),
            patch.object(logger, "_set_org_budget_metrics_after_api_request", AsyncMock()),
        ):
            await logger._increment_remaining_budget_metrics(**COMMON_KWARGS)

        assemble_user.assert_not_called()
        assemble_team.assert_called_once()
        assemble_key.assert_called_once()

    @pytest.mark.asyncio
    async def test_set_team_budget_metrics_directly_skips_when_gauge_is_noop(self):
        """_set_team_budget_metrics_after_api_request returns early when team gauge is NoOp."""
        logger = make_logger_with_budget_metrics_disabled()
        assemble_team = AsyncMock(return_value=MagicMock())

        with patch.object(logger, "_assemble_team_object", assemble_team):
            await logger._set_team_budget_metrics_after_api_request(
                user_api_team="team-123",
                user_api_team_alias="my-team",
                team_spend=0.5,
                team_max_budget=10.0,
                response_cost=0.001,
            )

        assemble_team.assert_not_called()

    @pytest.mark.asyncio
    async def test_set_api_key_budget_metrics_directly_skips_when_gauge_is_noop(self):
        """_set_api_key_budget_metrics_after_api_request returns early when key gauge is NoOp."""
        logger = make_logger_with_budget_metrics_disabled()
        assemble_key = AsyncMock(return_value=MagicMock())

        with patch.object(logger, "_assemble_key_object", assemble_key):
            await logger._set_api_key_budget_metrics_after_api_request(
                user_api_key="hashed-key",
                user_api_key_alias="my-key",
                response_cost=0.001,
                key_max_budget=10.0,
                key_spend=0.5,
            )

        assemble_key.assert_not_called()

    @pytest.mark.asyncio
    async def test_set_user_budget_metrics_directly_skips_when_gauge_is_noop(self):
        """_set_user_budget_metrics_after_api_request returns early when user gauge is NoOp."""
        logger = make_logger_with_budget_metrics_disabled()
        assemble_user = AsyncMock(return_value=MagicMock())

        with patch.object(logger, "_assemble_user_object", assemble_user):
            await logger._set_user_budget_metrics_after_api_request(
                user_id="user-1",
                user_spend=0.5,
                user_max_budget=10.0,
                response_cost=0.001,
            )

        assemble_user.assert_not_called()

    @pytest.mark.asyncio
    async def test_set_org_budget_metrics_directly_skips_when_gauge_is_noop(self):
        """_set_org_budget_metrics_after_api_request returns early when org gauge is NoOp.
        The guard fires before any import of auth_checks, so prisma_client is never touched."""
        logger = make_logger_with_budget_metrics_disabled()

        set_org_metrics = MagicMock()
        with patch.object(logger, "_set_org_budget_metrics", set_org_metrics):
            await logger._set_org_budget_metrics_after_api_request(
                org_id="org-1",
                response_cost=0.001,
            )

        set_org_metrics.assert_not_called()
