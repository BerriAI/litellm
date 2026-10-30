"""
Post-request budget gauges read the key/team/user/org state auth already resolved
from request metadata. get_*_object only runs when that state is missing (custom
auth, SDK callers, failure paths)
"""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from prometheus_client import REGISTRY

from litellm.caching.dual_cache import DualCache
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.caching.redis_cache import RedisCache
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
from litellm.proxy.spend_tracking.committed_spend import committed_spend_key
from litellm.types.proxy.carried_budget_state import (
    CommittedBudgetSpend,
    EntitySpendCounterKeys,
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


@pytest.fixture(autouse=True)
def spend_counter_cache():
    cache = DualCache()
    with patch(  # test-quality-ok: prometheus reads the proxy_server spend counters at emit time, no injection seam
        "litellm.proxy.proxy_server.spend_counter_cache", cache
    ):
        yield cache


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


def _store_counters(spend_counter_cache: DualCache, key: float, team: float, user: float, org: float) -> None:
    for counter_key, value in (
        ("spend:key:hashed", key),
        ("spend:team:t1", team),
        ("spend:user:u1", user),
        ("spend:org:o1", org),
    ):
        spend_counter_cache.in_memory_cache.set_cache(key=counter_key, value=value)


def _store_committed(spend_counter_cache: DualCache, key: float, team: float, user: float, org: float) -> None:
    for counter_key, value in (
        ("spend:key:hashed", key),
        ("spend:team:t1", team),
        ("spend:user:u1", user),
        ("spend:org:o1", org),
    ):
        spend_counter_cache.in_memory_cache.set_cache(key=committed_spend_key(counter_key), value=value)


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


@pytest.mark.asyncio
async def test_gauges_use_the_committed_spend_at_emit_when_other_requests_finished_after_auth(
    prometheus_logger, getters, spend_counter_cache
):
    """This request's 2.0 plus 1.0 from another request that finished first, so the gauge is never set back."""
    _stub_gauges(prometheus_logger)
    _store_committed(spend_counter_cache, key=6.0, team=33.0, user=11.0, org=63.0)

    await _emit(prometheus_logger, _request_metadata(_token_committed_at_auth()))

    assert _remaining(prometheus_logger) == {
        "key": pytest.approx(4.0),
        "team": pytest.approx(267.0),
        "user": pytest.approx(39.0),
        "org": pytest.approx(437.0),
    }


@pytest.mark.asyncio
async def test_gauges_add_the_cost_to_another_requests_charge_while_the_own_charge_is_not_in_yet(
    prometheus_logger, getters, spend_counter_cache
):
    """1.0 from another request reached the counters during this long request, this request's 2.0 has not."""
    _stub_gauges(prometheus_logger)
    _store_committed(spend_counter_cache, key=4.0, team=31.0, user=9.0, org=61.0)

    await _emit(prometheus_logger, _request_metadata(_token_committed_at_auth()))

    assert _remaining(prometheus_logger) == {
        "key": pytest.approx(4.0),
        "team": pytest.approx(267.0),
        "user": pytest.approx(39.0),
        "org": pytest.approx(437.0),
    }


@pytest.mark.asyncio
async def test_gauges_read_no_committed_counter_auth_did_not_name_as_budgeted(
    prometheus_logger, getters, spend_counter_cache
):
    _stub_gauges(prometheus_logger)
    _store_committed(spend_counter_cache, key=9.0, team=99.0, user=19.0, org=99.0)

    await _emit(prometheus_logger, _request_metadata(_authed_token()))

    assert _remaining(prometheus_logger) == {
        "key": pytest.approx(7.0),
        "team": pytest.approx(278.0),
        "user": pytest.approx(43.0),
        "org": 458.0,
    }


@pytest.mark.asyncio
async def test_slow_committed_read_on_a_failed_request_is_bounded_and_keeps_the_auth_time_spend(
    prometheus_logger, getters, monkeypatch
):
    async def never_answers(key_list: list[str]) -> dict[str, float]:
        await asyncio.sleep(30)
        return {}

    redis_cache = AsyncMock(spec=RedisCache)
    redis_cache.async_batch_get_cache = never_answers
    monkeypatch.setattr(
        "litellm.proxy.proxy_server.spend_counter_cache",
        DualCache(in_memory_cache=InMemoryCache(), redis_cache=redis_cache),
    )
    monkeypatch.setenv("PROMETHEUS_BUDGET_METRICS_PER_REQUEST_TIMEOUT", "0.01")
    _stub_gauges(prometheus_logger)

    await asyncio.wait_for(
        prometheus_logger.async_log_failure_event(
            _failed_request(_request_metadata(_token_committed_at_auth())), None, KEY_RESET_AT, KEY_RESET_AT
        ),
        timeout=5,
    )

    prometheus_logger.litellm_remaining_org_budget_metric.labels().set.assert_called_once_with(pytest.approx(440.0))


@pytest.mark.asyncio
async def test_gauges_add_the_cost_to_the_auth_time_spend_before_the_charge_reaches_the_counters(
    prometheus_logger, getters, spend_counter_cache
):
    _stub_gauges(prometheus_logger)
    _store_committed(spend_counter_cache, key=3.0, team=30.0, user=8.0, org=60.0)

    await _emit(prometheus_logger, _request_metadata(_token_committed_at_auth()))

    assert _remaining(prometheus_logger) == {
        "key": pytest.approx(5.0),
        "team": pytest.approx(268.0),
        "user": pytest.approx(40.0),
        "org": pytest.approx(438.0),
    }


@pytest.mark.asyncio
async def test_gauges_keep_the_auth_time_spend_when_the_proxy_spend_counters_cannot_be_read(
    prometheus_logger, getters, monkeypatch
):
    _stub_gauges(prometheus_logger)
    monkeypatch.delattr("litellm.proxy.proxy_server.spend_counter_cache")

    await _emit(prometheus_logger, _request_metadata(_token_committed_at_auth()))

    assert _remaining(prometheus_logger) == {
        "key": pytest.approx(5.0),
        "team": pytest.approx(268.0),
        "user": pytest.approx(40.0),
        "org": pytest.approx(438.0),
    }


@pytest.mark.asyncio
async def test_a_budget_hold_on_the_enforcement_counters_never_shows_in_the_gauges(
    prometheus_logger, getters, spend_counter_cache
):
    _stub_gauges(prometheus_logger)
    _store_committed(spend_counter_cache, key=5.0, team=32.0, user=10.0, org=62.0)
    _store_counters(spend_counter_cache, key=9.0, team=36.0, user=14.0, org=66.0)

    await _emit(prometheus_logger, _request_metadata(_token_committed_at_auth()))

    assert _remaining(prometheus_logger) == {
        "key": pytest.approx(5.0),
        "team": pytest.approx(268.0),
        "user": pytest.approx(40.0),
        "org": pytest.approx(438.0),
    }


def _failed_request(metadata: dict) -> dict:
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
async def test_failed_request_sets_the_org_gauge_from_the_carried_org_and_committed_spend(
    prometheus_logger, getters, spend_counter_cache
):
    _stub_gauges(prometheus_logger)
    _store_committed(spend_counter_cache, key=3.0, team=30.0, user=8.0, org=65.0)
    _store_counters(spend_counter_cache, key=9.0, team=36.0, user=14.0, org=90.0)

    await prometheus_logger.async_log_failure_event(
        _failed_request(_request_metadata(_token_committed_at_auth())), None, KEY_RESET_AT, KEY_RESET_AT
    )

    assert getters["get_org_object"].await_count == 0
    prometheus_logger.litellm_remaining_org_budget_metric.labels().set.assert_called_once_with(pytest.approx(435.0))


@pytest.mark.asyncio
async def test_failed_request_without_the_auth_object_still_reads_the_committed_org_spend(
    prometheus_logger, getters, spend_counter_cache
):
    _stub_gauges(prometheus_logger)
    _store_committed(spend_counter_cache, key=3.0, team=30.0, user=8.0, org=65.0)
    metadata = _request_metadata(_token_committed_at_auth())
    del metadata["user_api_key_auth"]

    await prometheus_logger.async_log_failure_event(_failed_request(metadata), None, KEY_RESET_AT, KEY_RESET_AT)

    prometheus_logger.litellm_remaining_org_budget_metric.labels().set.assert_called_once_with(pytest.approx(435.0))


@pytest.mark.asyncio
async def test_failed_request_without_a_carried_org_reads_no_committed_org_spend(
    prometheus_logger, getters, spend_counter_cache
):
    _stub_gauges(prometheus_logger)
    _store_committed(spend_counter_cache, key=3.0, team=30.0, user=8.0, org=65.0)
    token = _token_committed_at_auth().model_copy(update={"org_budget_snapshot": None})
    metadata = {key: value for key, value in _request_metadata(token).items() if key != "user_api_key_auth"}

    await prometheus_logger.async_log_failure_event(_failed_request(metadata), None, KEY_RESET_AT, KEY_RESET_AT)

    prometheus_logger.litellm_remaining_org_budget_metric.labels().set.assert_called_once_with(pytest.approx(460.0))


@pytest.mark.asyncio
async def test_ordinary_metadata_keys_named_like_snapshot_fields_change_no_gauge(prometheus_logger, getters):
    _stub_gauges(prometheus_logger)
    metadata = {
        "user_api_key_team_spend": 20.0,
        "user_api_key_team_max_budget": 300.0,
        "spend": 1000.0,
        "max_budget": 1.0,
        "budget_reset_at": None,
        "user_alias": "not-the-user",
    }

    await _emit(prometheus_logger, metadata)

    assert {name: getter.await_count for name, getter in getters.items()} == {
        "get_key_object": 1,
        "get_team_object": 1,
        "get_user_object": 1,
        "get_org_object": 1,
    }
    prometheus_logger.litellm_remaining_org_budget_metric.labels().set.assert_called_once_with(458.0)


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
