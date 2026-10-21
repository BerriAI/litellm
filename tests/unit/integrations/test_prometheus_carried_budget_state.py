"""
Post-request budget gauges read the key/team/user/org state auth already resolved
from request metadata. get_*_object only runs when that state is missing (custom
auth, SDK callers, failure paths)
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from prometheus_client import REGISTRY

from litellm.integrations.prometheus import PrometheusLogger
from litellm.proxy._types import (
    LiteLLM_BudgetTable,
    LiteLLM_OrganizationTable,
    LiteLLM_TeamTable,
    LiteLLM_UserTable,
    UserAPIKeyAuth,
)
from litellm.proxy.litellm_pre_call_utils import LiteLLMProxyRequestSetup
from litellm.proxy.spend_tracking.carried_budget_state import (
    carried_budget_metadata,
    carry_organization_budget_state,
    carry_team_and_user_budget_state,
)
from litellm.types.proxy.carried_budget_state import (
    KeyBudgetSnapshot,
    TeamBudgetSnapshot,
    UserBudgetSnapshot,
)

TEAM_RESET_AT = datetime(2026, 10, 1, tzinfo=timezone.utc)
USER_RESET_AT = datetime(2026, 11, 1, tzinfo=timezone.utc)
KEY_RESET_AT = datetime(2026, 12, 1, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def cleanup_prometheus_registry():
    for collector in list(REGISTRY._collector_to_names.keys()):
        try:
            REGISTRY.unregister(collector)
        except Exception:
            pass
    yield
    for collector in list(REGISTRY._collector_to_names.keys()):
        try:
            REGISTRY.unregister(collector)
        except Exception:
            pass


@pytest.fixture
def prometheus_logger():
    return PrometheusLogger()


@pytest.fixture
def getters():
    """Every response-path object getter, patched where prometheus imports them from."""
    mocks = {
        "get_key_object": AsyncMock(return_value=UserAPIKeyAuth(token="hashed", budget_reset_at=KEY_RESET_AT)),
        "get_team_object": AsyncMock(
            return_value=LiteLLM_TeamTable(team_id="t1", budget_reset_at=TEAM_RESET_AT, max_budget=300.0)
        ),
        "get_user_object": AsyncMock(
            return_value=LiteLLM_UserTable(
                user_id="u1",
                budget_reset_at=USER_RESET_AT,
                user_email="alice@example.com",
                user_alias="Alice",
                max_budget=50.0,
            )
        ),
        "get_org_object": AsyncMock(
            return_value=LiteLLM_OrganizationTable(
                organization_id="o1",
                organization_alias="platform-org",
                budget_id="b1",
                created_by="admin",
                updated_by="admin",
                spend=40.0,
                litellm_budget_table=LiteLLM_BudgetTable(max_budget=500.0),
            )
        ),
    }
    with (
        patch.multiple(  # test-quality-ok: prometheus reads these proxy_server globals at call time, no injection seam
            "litellm.proxy.proxy_server", prisma_client=MagicMock(), user_api_key_cache=MagicMock()
        ),
        patch.multiple(  # test-quality-ok: the getters are the DB boundary this test counts calls to
            "litellm.proxy.auth.auth_checks", **mocks
        ),
    ):
        yield mocks


def _authed_token() -> UserAPIKeyAuth:
    token = UserAPIKeyAuth(
        token="hashed",
        key_alias="key-alias",
        team_id="t1",
        team_alias="team-alias",
        user_id="u1",
        user_email="alice@example.com",
        org_id="o1",
        spend=1.0,
        max_budget=10.0,
        team_spend=20.0,
        team_max_budget=300.0,
        user_spend=5.0,
        user_max_budget=50.0,
        budget_reset_at=KEY_RESET_AT,
    )
    carry_team_and_user_budget_state(
        valid_token=token,
        team_object=LiteLLM_TeamTable(team_id="t1", budget_reset_at=TEAM_RESET_AT, max_budget=300.0),
        user_object=LiteLLM_UserTable(user_id="u1", budget_reset_at=USER_RESET_AT, user_alias="Alice", max_budget=50.0),
    )
    carry_organization_budget_state(
        valid_token=token,
        org_table=LiteLLM_OrganizationTable(
            organization_id="o1",
            organization_alias="platform-org",
            budget_id="b1",
            created_by="admin",
            updated_by="admin",
            spend=40.0,
            litellm_budget_table=LiteLLM_BudgetTable(max_budget=500.0),
        ),
    )
    return token


def _request_metadata(token: UserAPIKeyAuth) -> dict:
    """What add_user_api_key_auth_to_request_metadata leaves in litellm_params["metadata"]."""
    return {
        **LiteLLMProxyRequestSetup.get_sanitized_user_information_from_key(token),
        **carried_budget_metadata(token),
    }


def _stub_gauges(prometheus_logger: PrometheusLogger) -> None:
    for name in (
        "litellm_remaining_api_key_budget_metric",
        "litellm_api_key_max_budget_metric",
        "litellm_api_key_budget_remaining_hours_metric",
        "litellm_remaining_team_budget_metric",
        "litellm_team_max_budget_metric",
        "litellm_team_budget_remaining_hours_metric",
        "litellm_remaining_user_budget_metric",
        "litellm_user_max_budget_metric",
        "litellm_user_budget_remaining_hours_metric",
        "litellm_remaining_org_budget_metric",
        "litellm_org_max_budget_metric",
        "litellm_org_budget_remaining_hours_metric",
    ):
        setattr(prometheus_logger, name, MagicMock())


async def _emit(prometheus_logger: PrometheusLogger, metadata: dict) -> None:
    await prometheus_logger._increment_remaining_budget_metrics(
        user_api_team="t1",
        user_api_team_alias="team-alias",
        user_api_key="hashed",
        user_api_key_alias="key-alias",
        litellm_params={"metadata": metadata},
        response_cost=2.0,
        user_id="u1",
        user_api_key_org_id="o1",
    )


@pytest.mark.asyncio
async def test_authed_request_sets_every_gauge_without_any_object_getter(prometheus_logger, getters):
    _stub_gauges(prometheus_logger)

    await _emit(prometheus_logger, _request_metadata(_authed_token()))

    assert all(getter.await_count == 0 for getter in getters.values()), {
        name: getter.await_count for name, getter in getters.items()
    }
    remaining = {
        "key": prometheus_logger.litellm_remaining_api_key_budget_metric.labels().set.call_args[0][0],
        "team": prometheus_logger.litellm_remaining_team_budget_metric.labels().set.call_args[0][0],
        "user": prometheus_logger.litellm_remaining_user_budget_metric.labels().set.call_args[0][0],
        "org": prometheus_logger.litellm_remaining_org_budget_metric.labels().set.call_args[0][0],
    }
    assert remaining == {
        "key": pytest.approx(7.0),
        "team": pytest.approx(278.0),
        "user": pytest.approx(43.0),
        "org": 458.0,
    }
    prometheus_logger.litellm_org_max_budget_metric.labels().set.assert_called_once_with(500.0)
    prometheus_logger.litellm_api_key_budget_remaining_hours_metric.labels().set.assert_called_once()
    prometheus_logger.litellm_team_budget_remaining_hours_metric.labels().set.assert_called_once()
    prometheus_logger.litellm_user_budget_remaining_hours_metric.labels().set.assert_called_once()


@pytest.mark.asyncio
async def test_metadata_without_carried_state_still_fetches_each_object_once(prometheus_logger, getters):
    _stub_gauges(prometheus_logger)

    await _emit(prometheus_logger, {"user_api_key_team_spend": 20.0, "user_api_key_team_max_budget": 300.0})

    assert {name: getter.await_count for name, getter in getters.items()} == {
        "get_key_object": 1,
        "get_team_object": 1,
        "get_user_object": 1,
        "get_org_object": 1,
    }
    prometheus_logger.litellm_remaining_org_budget_metric.labels().set.assert_called_once_with(458.0)
    prometheus_logger.litellm_team_budget_remaining_hours_metric.labels().set.assert_called_once()


@pytest.mark.asyncio
async def test_partial_carried_state_only_skips_the_carried_objects(prometheus_logger, getters):
    _stub_gauges(prometheus_logger)
    token = UserAPIKeyAuth(token="hashed", team_id="t1", user_id="u1", org_id="o1")
    carry_team_and_user_budget_state(
        valid_token=token,
        team_object=LiteLLM_TeamTable(team_id="t1", budget_reset_at=TEAM_RESET_AT),
        user_object=None,
    )

    await _emit(prometheus_logger, dict(carried_budget_metadata(token)))

    assert {name: getter.await_count for name, getter in getters.items()} == {
        "get_key_object": 1,
        "get_team_object": 0,
        "get_user_object": 1,
        "get_org_object": 1,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("metadata_max_budget", [300.0, None], ids=["metadata has max_budget", "filled from object"])
async def test_carried_objects_match_what_the_getters_would_have_produced(
    prometheus_logger, getters, metadata_max_budget
):
    metadata = _request_metadata(_authed_token())
    user_max_budget = 50.0 if metadata_max_budget is not None else None

    carried_team = await prometheus_logger._assemble_team_object(
        team_id="t1",
        team_alias="team-alias",
        spend=20.0,
        max_budget=metadata_max_budget,
        response_cost=2.0,
        carried=TeamBudgetSnapshot.from_metadata(metadata),
    )
    fetched_team = await prometheus_logger._assemble_team_object(
        team_id="t1", team_alias="team-alias", spend=20.0, max_budget=metadata_max_budget, response_cost=2.0
    )
    carried_user = await prometheus_logger._assemble_user_object(
        user_id="u1",
        spend=5.0,
        max_budget=user_max_budget,
        response_cost=2.0,
        carried=UserBudgetSnapshot.from_metadata(metadata),
        user_email="alice@example.com",
    )
    fetched_user = await prometheus_logger._assemble_user_object(
        user_id="u1", spend=5.0, max_budget=user_max_budget, response_cost=2.0
    )
    carried_key = await prometheus_logger._assemble_key_object(
        user_api_key="hashed",
        user_api_key_alias="key-alias",
        key_max_budget=10.0,
        key_spend=1.0,
        response_cost=2.0,
        carried=KeyBudgetSnapshot.from_metadata(metadata),
    )
    fetched_key = await prometheus_logger._assemble_key_object(
        user_api_key="hashed", user_api_key_alias="key-alias", key_max_budget=10.0, key_spend=1.0, response_cost=2.0
    )

    assert carried_team == fetched_team
    assert carried_team.max_budget == 300.0
    assert carried_user == fetched_user
    assert carried_user.max_budget == 50.0
    assert carried_key == fetched_key
    assert {name: getter.await_count for name, getter in getters.items()} == {
        "get_key_object": 1,
        "get_team_object": 1,
        "get_user_object": 1,
        "get_org_object": 0,
    }
