"""
Post-request budget gauges read the key/team/user/org state auth already resolved
from request metadata. get_*_object only runs when that state is missing (custom
auth, SDK callers, failure paths)
"""

from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Final
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from prometheus_client import REGISTRY

from litellm.caching.redis_batch import request_redis_batch_scope
from litellm.integrations.prometheus import PrometheusLogger
from litellm.proxy import proxy_server
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
from litellm.proxy.spend_tracking.spend_counter_batch import PendingSpendIncrement
from litellm.types.proxy.carried_budget_state import (
    CommittedBudgetSpend,
    EntitySpendCounterKeys,
    KeyBudgetSnapshot,
    TeamBudgetSnapshot,
    UserBudgetSnapshot,
)
from tests.unit.caching.redis_batch_double import FakeNatives, fake_send
from tests.unit.redis.stores.spend_counters_double import FakeSpendCounters

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


@pytest.fixture(autouse=True)
def spend_counters():
    spend: Final = FakeSpendCounters(redis=None)
    with patch(  # test-quality-ok: prometheus reads the proxy_server spend counters at emit time, no injection seam
        "litellm.proxy.proxy_server.spend_counter_store", spend.store()
    ):
        yield spend


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
        "user_api_key_auth": token,
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




_COUNTER_KEYS: Final = ("spend:key:hashed", "spend:team:t1", "spend:user:u1", "spend:org:o1")
_TWIN_KEYS: Final = (
    "spend:committed:key:hashed",
    "spend:committed:team:t1",
    "spend:committed:user:u1",
    "spend:committed:org:o1",
)


def _store(spend: FakeSpendCounters, counter_keys: Sequence[str], values: Sequence[float]) -> None:
    tier: Final = spend.memory if spend.redis is None else spend.redis
    tier.update(zip(counter_keys, values, strict=True))


def _store_counters(spend: FakeSpendCounters, key: float, team: float, user: float, org: float) -> None:
    _store(spend, _COUNTER_KEYS, (key, team, user, org))


def _store_committed(spend: FakeSpendCounters, key: float, team: float, user: float, org: float) -> None:
    _store(spend, _TWIN_KEYS, (key, team, user, org))


def _remaining(prometheus_logger: PrometheusLogger) -> dict[str, float]:
    return {
        "key": prometheus_logger.litellm_remaining_api_key_budget_metric.labels().set.call_args[0][0],
        "team": prometheus_logger.litellm_remaining_team_budget_metric.labels().set.call_args[0][0],
        "user": prometheus_logger.litellm_remaining_user_budget_metric.labels().set.call_args[0][0],
        "org": prometheus_logger.litellm_remaining_org_budget_metric.labels().set.call_args[0][0],
    }


def _token_committed_at_auth() -> UserAPIKeyAuth:
    token = _authed_token()
    token.committed_budget_spend = CommittedBudgetSpend(
        counter_keys=EntitySpendCounterKeys.of(
            token="hashed", team_id="t1", user_id="u1", org_id="o1", end_user_id=None
        ),
        key=3.0,
        team=30.0,
        user=8.0,
        org=60.0,
    )
    return token








def _billed_row_metadata() -> dict:
    return {key: value for key, value in _request_metadata(_authed_token()).items() if key != "user_api_key_auth"}










@pytest.fixture
def redis_spend_counters():
    spend: Final = FakeSpendCounters(redis={})
    with patch(  # test-quality-ok: prometheus reads the proxy_server spend counters at emit time, no injection seam
        "litellm.proxy.proxy_server.spend_counter_store", spend.store()
    ):
        yield spend


def _request_scope() -> request_redis_batch_scope:
    return request_redis_batch_scope(
        native_factory=FakeNatives(),  # pyright: ignore[reportArgumentType]  # test double
        native_send_factory=fake_send,  # pyright: ignore[reportArgumentType]  # test double
    )


async def _declare_the_own_two_dollar_charge() -> None:
    await proxy_server._apply_spend_counter_increments(
        tuple(PendingSpendIncrement(counter_key, 2.0) for counter_key in (*_COUNTER_KEYS, *_TWIN_KEYS))
    )


_BOTH_TWO_DOLLAR_CHARGES_COUNTED_ONCE: Final = {
    "key": pytest.approx(3.0),
    "team": pytest.approx(266.0),
    "user": pytest.approx(38.0),
    "org": pytest.approx(436.0),
}


















def _failed_request(metadata: dict, response_cost: float = 0.0) -> dict:
    return {
        "model": "openai/gpt-5.4-nano",
        "litellm_params": {"metadata": metadata},
        "standard_logging_object": {
            "model": "openai/gpt-5.4-nano",
            "model_id": "dep-1",
            "model_group": "chat",
            "api_base": "https://api.openai.com",
            "custom_llm_provider": "openai",
            "status": "failure",
            "response_cost": response_cost,
            "request_tags": [],
            "hidden_params": {"litellm_overhead_time_ms": None, "additional_headers": None},
            "metadata": {
                "user_api_key_hash": "hashed",
                "user_api_key_alias": "key-alias",
                "user_api_key_team_id": "t1",
                "user_api_key_team_alias": "team-alias",
                "user_api_key_user_id": "u1",
                "user_api_key_org_id": "o1",
                "requester_metadata": None,
                "user_api_key_end_user_id": None,
            },
        },
        "exception": Exception("upstream failed"),
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
