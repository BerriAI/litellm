"""
Unit tests for the litellm_team_member_count_metric Prometheus gauge.

The gauge tracks the number of members currently in each team:
- Incremented when a member is added via /team/member_add
- Decremented when a member is removed via /team/member_delete
- Labels: ``team`` (team_id) and ``team_alias``
"""

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from prometheus_client import REGISTRY

import litellm
from litellm.integrations.prometheus import PrometheusLogger
from litellm.types.integrations.prometheus import PrometheusMetricLabels


@pytest.fixture(autouse=True)
def cleanup_prometheus_registry():
    """Clear the registry between tests so PrometheusLogger() can be re-created."""
    for collector in list(REGISTRY._collector_to_names.keys()):
        try:
            REGISTRY.unregister(collector)
        except Exception:
            pass
    # Reset callbacks so emit helpers don't pick up loggers from prior tests
    original_callbacks = list(litellm.callbacks)
    litellm.callbacks = []
    yield
    litellm.callbacks = original_callbacks
    for collector in list(REGISTRY._collector_to_names.keys()):
        try:
            REGISTRY.unregister(collector)
        except Exception:
            pass


@pytest.fixture
def prometheus_logger():
    """Fresh PrometheusLogger registered as a litellm callback."""
    logger = PrometheusLogger()
    litellm.callbacks = [logger]
    return logger


def _get_gauge_value(team_id: str, team_alias: str) -> float:
    """Read the current value of litellm_team_member_count_metric for a label set."""
    for metric in REGISTRY.collect():
        for sample in metric.samples:
            if (
                sample.name == "litellm_team_member_count_metric"
                and sample.labels.get("team") == team_id
                and sample.labels.get("team_alias") == team_alias
            ):
                return sample.value
    return 0.0


class TestTeamMemberCountMetricDefinition:
    """Verify the gauge is declared with the expected labels and shape."""

    def test_metric_attribute_exists(self, prometheus_logger):
        assert hasattr(prometheus_logger, "litellm_team_member_count_metric")
        assert prometheus_logger.litellm_team_member_count_metric is not None

    def test_metric_labels_are_team_and_team_alias(self):
        labels = PrometheusMetricLabels.litellm_team_member_count_metric
        assert "team" in labels
        assert "team_alias" in labels
        assert set(labels) == {"team", "team_alias"}

    def test_metric_is_a_gauge(self, prometheus_logger):
        """Gauge supports both .inc and .dec — Counter would not."""
        metric = prometheus_logger.litellm_team_member_count_metric
        labelled = metric.labels(team="t", team_alias="a")
        assert hasattr(labelled, "inc")
        assert hasattr(labelled, "dec")

    def test_metric_in_defined_prometheus_metrics_literal(self):
        """The metric name must be in DEFINED_PROMETHEUS_METRICS so
        get_labels_for_metric resolves it."""
        from typing import get_args

        from litellm.types.integrations.prometheus import DEFINED_PROMETHEUS_METRICS

        assert "litellm_team_member_count_metric" in get_args(
            DEFINED_PROMETHEUS_METRICS
        )


class TestEmitHelpers:
    """The static emit_* helpers are the entry point used from endpoint code."""

    def test_emit_added_increments_gauge(self, prometheus_logger):
        PrometheusLogger.emit_team_member_added_metric(
            team_id="team-abc", team_alias="my-team", count=1
        )
        assert _get_gauge_value("team-abc", "my-team") == 1.0

    def test_emit_added_supports_count_greater_than_one(self, prometheus_logger):
        PrometheusLogger.emit_team_member_added_metric(
            team_id="team-abc", team_alias="my-team", count=3
        )
        assert _get_gauge_value("team-abc", "my-team") == 3.0

    def test_emit_removed_decrements_gauge(self, prometheus_logger):
        PrometheusLogger.emit_team_member_added_metric(
            team_id="team-abc", team_alias="my-team", count=5
        )
        PrometheusLogger.emit_team_member_removed_metric(
            team_id="team-abc", team_alias="my-team", count=2
        )
        assert _get_gauge_value("team-abc", "my-team") == 3.0

    def test_emit_add_then_remove_to_zero(self, prometheus_logger):
        PrometheusLogger.emit_team_member_added_metric(
            team_id="team-abc", team_alias="my-team", count=2
        )
        PrometheusLogger.emit_team_member_removed_metric(
            team_id="team-abc", team_alias="my-team", count=2
        )
        assert _get_gauge_value("team-abc", "my-team") == 0.0

    def test_emit_zero_count_is_noop(self, prometheus_logger):
        PrometheusLogger.emit_team_member_added_metric(
            team_id="team-abc", team_alias="my-team", count=0
        )
        PrometheusLogger.emit_team_member_removed_metric(
            team_id="team-abc", team_alias="my-team", count=0
        )
        assert _get_gauge_value("team-abc", "my-team") == 0.0

    def test_emit_negative_count_is_noop(self, prometheus_logger):
        """Guards against accidental sign flips at the call site."""
        PrometheusLogger.emit_team_member_added_metric(
            team_id="team-abc", team_alias="my-team", count=-3
        )
        assert _get_gauge_value("team-abc", "my-team") == 0.0

    def test_emit_missing_team_id_is_noop(self, prometheus_logger):
        PrometheusLogger.emit_team_member_added_metric(
            team_id=None, team_alias="my-team", count=1
        )
        PrometheusLogger.emit_team_member_added_metric(
            team_id="", team_alias="my-team", count=1
        )
        # Nothing should have been recorded under any team label
        for metric in REGISTRY.collect():
            for sample in metric.samples:
                assert sample.name != "litellm_team_member_count_metric" or (
                    sample.value == 0.0
                )

    def test_emit_none_team_alias_is_handled(self, prometheus_logger):
        """team_alias is optional; we should fall back to an empty string label."""
        PrometheusLogger.emit_team_member_added_metric(
            team_id="team-xyz", team_alias=None, count=1
        )
        assert _get_gauge_value("team-xyz", "") == 1.0

    def test_emit_is_noop_when_logger_not_registered(self):
        """If no PrometheusLogger is in litellm.callbacks the helpers must not raise."""
        litellm.callbacks = []
        # Must not raise and must not need a logger to exist
        PrometheusLogger.emit_team_member_added_metric(
            team_id="team-abc", team_alias="my-team", count=1
        )
        PrometheusLogger.emit_team_member_removed_metric(
            team_id="team-abc", team_alias="my-team", count=1
        )

    def test_per_team_labels_are_independent(self, prometheus_logger):
        PrometheusLogger.emit_team_member_added_metric(
            team_id="team-a", team_alias="alias-a", count=4
        )
        PrometheusLogger.emit_team_member_added_metric(
            team_id="team-b", team_alias="alias-b", count=1
        )
        PrometheusLogger.emit_team_member_removed_metric(
            team_id="team-a", team_alias="alias-a", count=1
        )
        assert _get_gauge_value("team-a", "alias-a") == 3.0
        assert _get_gauge_value("team-b", "alias-b") == 1.0


# ---------------------------------------------------------------------------
# Endpoint integration: verify the helpers are invoked from the team endpoints.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_add_team_members_emits_metric_for_actual_delta(prometheus_logger):
    """
    _add_team_members_to_team must emit the actual delta (post-dedup), not the
    raw request size. We seed a team with 1 existing member, attempt to add 2
    members where 1 is a duplicate, and expect the gauge to increase by 1.
    """
    from litellm.proxy._types import (
        LiteLLM_TeamTable,
        Member,
        TeamMemberAddRequest,
        UserAPIKeyAuth,
        LitellmUserRoles,
    )
    from litellm.proxy.management_endpoints.team_endpoints import (
        _add_team_members_to_team,
    )

    team = LiteLLM_TeamTable(
        team_id="team-add-1",
        team_alias="add-team-alias",
        members_with_roles=[Member(role="user", user_id="existing-user")],
    )

    data = TeamMemberAddRequest(
        team_id="team-add-1",
        member=[
            Member(role="user", user_id="existing-user"),  # duplicate
            Member(role="user", user_id="new-user"),  # actually new
        ],
    )

    prisma_client = MagicMock()
    prisma_client.db = MagicMock()
    prisma_client.db.litellm_teamtable = MagicMock()
    prisma_client.db.litellm_teamtable.update = AsyncMock(return_value=team)

    fake_user = MagicMock()
    fake_user.user_id = "new-user"
    fake_user.user_email = None

    with (
        patch(
            "litellm.proxy.management_endpoints.team_endpoints._process_team_members",
            new=AsyncMock(return_value=([fake_user], [])),
        ),
    ):
        await _add_team_members_to_team(
            data=data,
            complete_team_data=team,
            prisma_client=prisma_client,
            user_api_key_dict=UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN),
            litellm_proxy_admin_name="admin",
        )

    # Only the new user counts toward the metric — the duplicate is filtered.
    assert _get_gauge_value("team-add-1", "add-team-alias") == 1.0


@pytest.mark.asyncio
async def test_team_member_delete_decrements_metric(prometheus_logger):
    """
    team_member_delete must decrement the gauge by 1 for the removed member.
    Pre-seed the gauge so we can verify the decrement against a non-zero start.
    """
    from litellm.proxy._types import (
        LitellmUserRoles,
        TeamMemberDeleteRequest,
        UserAPIKeyAuth,
    )
    from litellm.proxy.management_endpoints.team_endpoints import team_member_delete

    team_id = "team-del-metric-1"
    team_alias = "del-team-alias"
    user_id = "user-to-remove@example.com"

    # Seed gauge at 3 — simulates a team with 3 members tracked.
    PrometheusLogger.emit_team_member_added_metric(
        team_id=team_id, team_alias=team_alias, count=3
    )

    mock_team_row = MagicMock()
    mock_team_row.model_dump.return_value = {
        "team_id": team_id,
        "team_alias": team_alias,
        "members_with_roles": [
            {"user_id": user_id, "user_email": None, "role": "user"},
            {"user_id": "kept-user-1", "user_email": None, "role": "user"},
            {"user_id": "kept-user-2", "user_email": None, "role": "user"},
        ],
        "team_member_permissions": [],
        "metadata": {},
        "models": [],
        "spend": 0.0,
    }

    mock_prisma = MagicMock()
    mock_prisma.db = MagicMock()
    mock_prisma.db.litellm_teamtable = MagicMock()
    mock_prisma.db.litellm_teamtable.find_unique = AsyncMock(return_value=mock_team_row)
    mock_prisma.db.litellm_teamtable.update = AsyncMock(return_value=mock_team_row)
    mock_prisma.db.litellm_usertable = MagicMock()
    mock_prisma.db.litellm_usertable.find_many = AsyncMock(return_value=[])
    mock_prisma.db.litellm_teammembership = MagicMock()
    mock_prisma.db.litellm_teammembership.delete_many = AsyncMock(
        return_value=MagicMock()
    )
    mock_prisma.db.litellm_verificationtoken = MagicMock()
    mock_prisma.db.litellm_verificationtoken.find_many = AsyncMock(return_value=[])
    mock_prisma.db.litellm_verificationtoken.delete_many = AsyncMock(
        return_value=MagicMock()
    )

    # team_member_delete imports prisma_client from litellm.proxy.proxy_server.
    # Stub the module so we don't need to load the full proxy server.
    fake_proxy_server = MagicMock()
    fake_proxy_server.prisma_client = mock_prisma
    with patch.dict(sys.modules, {"litellm.proxy.proxy_server": fake_proxy_server}):
        await team_member_delete(
            data=TeamMemberDeleteRequest(team_id=team_id, user_id=user_id),
            user_api_key_dict=UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN),
        )

    # Started at 3, removed 1 → expect 2
    assert _get_gauge_value(team_id, team_alias) == 2.0
