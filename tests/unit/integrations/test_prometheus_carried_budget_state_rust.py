import asyncio
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Final
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from prometheus_client import REGISTRY

from litellm.caching.redis_batch import flush_post_call_redis_batches, request_redis_batch_scope
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
)
from tests.unit.caching.redis_batch_double import FakeNatives, fake_send
from tests.unit.integrations.prometheus_native_support import (
    enable_native_prometheus,
    native_failure,
    series,
    standard_payload,
)
from tests.unit.redis.stores.spend_counters_double import FakeSpendCounters

TEAM_RESET_AT = datetime(2026, 10, 1, tzinfo=timezone.utc)
USER_RESET_AT = datetime(2026, 11, 1, tzinfo=timezone.utc)
KEY_RESET_AT = datetime(2026, 12, 1, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def native_prometheus(monkeypatch):
    enable_native_prometheus(monkeypatch)


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


def _gauge(name: str) -> float:
    [(_, value)] = series(name)
    return value


def _set_once_with(name: str, expected: object) -> None:
    assert len(series(name)) == 1
    assert _gauge(name) == expected


def _set_once(name: str) -> None:
    assert len(series(name)) == 1


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

    await _emit(prometheus_logger, _request_metadata(_authed_token()))

    assert all(getter.await_count == 0 for getter in getters.values()), {
        name: getter.await_count for name, getter in getters.items()
    }
    remaining = _remaining(prometheus_logger)
    assert remaining == {
        "key": pytest.approx(7.0),
        "team": pytest.approx(278.0),
        "user": pytest.approx(43.0),
        "org": 458.0,
    }
    _set_once_with("litellm_org_max_budget_metric", 500.0)
    _set_once("litellm_api_key_budget_remaining_hours_metric")
    _set_once("litellm_team_budget_remaining_hours_metric")
    _set_once("litellm_user_budget_remaining_hours_metric")


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
        "key": _gauge("litellm_remaining_api_key_budget_metric"),
        "team": _gauge("litellm_remaining_team_budget_metric"),
        "user": _gauge("litellm_remaining_user_budget_metric"),
        "org": _gauge("litellm_remaining_org_budget_metric"),
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
    prometheus_logger, getters, spend_counters
):
    """This request's 2.0 plus 1.0 from another request that finished first, so the gauge is never set back."""
    _store_committed(spend_counters, key=6.0, team=33.0, user=11.0, org=63.0)

    await _emit(prometheus_logger, _request_metadata(_token_committed_at_auth()))

    assert _remaining(prometheus_logger) == {
        "key": pytest.approx(4.0),
        "team": pytest.approx(267.0),
        "user": pytest.approx(39.0),
        "org": pytest.approx(437.0),
    }


@pytest.mark.asyncio
async def test_gauges_add_the_cost_to_another_requests_charge_while_the_own_charge_is_not_in_yet(
    prometheus_logger, getters, spend_counters
):
    """1.0 from another request reached the counters during this long request, this request's 2.0 has not."""
    _store_committed(spend_counters, key=4.0, team=31.0, user=9.0, org=61.0)

    await _emit(prometheus_logger, _request_metadata(_token_committed_at_auth()))

    assert _remaining(prometheus_logger) == {
        "key": pytest.approx(4.0),
        "team": pytest.approx(267.0),
        "user": pytest.approx(39.0),
        "org": pytest.approx(437.0),
    }


@pytest.mark.asyncio
async def test_gauges_read_no_committed_counter_auth_did_not_name_as_budgeted(
    prometheus_logger, getters, spend_counters
):
    _store_committed(spend_counters, key=9.0, team=99.0, user=19.0, org=99.0)

    await _emit(prometheus_logger, _request_metadata(_authed_token()))

    assert _remaining(prometheus_logger) == {
        "key": pytest.approx(7.0),
        "team": pytest.approx(278.0),
        "user": pytest.approx(43.0),
        "org": 458.0,
    }


def _billed_row_metadata() -> dict:
    return {key: value for key, value in _request_metadata(_authed_token()).items() if key != "user_api_key_auth"}


@pytest.mark.asyncio
async def test_row_with_no_auth_adds_its_cost_to_the_committed_spend_of_the_entities_it_is_charged_to(
    prometheus_logger, getters, spend_counters
):
    """Counter increments flush after the success callbacks, so the row's own charge is not in the counters yet."""
    _store_committed(spend_counters, key=6.0, team=33.0, user=11.0, org=63.0)

    await _emit(prometheus_logger, _billed_row_metadata())

    assert _remaining(prometheus_logger) == {
        "key": pytest.approx(2.0),
        "team": pytest.approx(265.0),
        "user": pytest.approx(37.0),
        "org": 435.0,
    }


@pytest.mark.asyncio
async def test_row_with_no_auth_and_no_committed_counters_keeps_the_stored_spend(
    prometheus_logger, getters, spend_counters
):

    await _emit(prometheus_logger, _billed_row_metadata())

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
    async def never_answers(counter: str) -> None:
        await asyncio.sleep(30)

    monkeypatch.setattr(
        "litellm.proxy.proxy_server.spend_counter_store",
        FakeSpendCounters(redis={}, batch_reads=False, read_hook=never_answers).store(),
    )
    monkeypatch.setenv("PROMETHEUS_BUDGET_METRICS_PER_REQUEST_TIMEOUT", "0.01")

    await asyncio.wait_for(
        native_failure(
            prometheus_logger,
            _failed_request(_request_metadata(_token_committed_at_auth())),
            None,
            KEY_RESET_AT,
            KEY_RESET_AT,
        ),
        timeout=5,
    )

    _set_once_with("litellm_remaining_org_budget_metric", pytest.approx(440.0))


@pytest.mark.asyncio
async def test_slow_committed_read_on_a_successful_request_still_sets_the_gauges_from_the_auth_time_spend(
    prometheus_logger, getters, monkeypatch
):
    async def never_answers(counter: str) -> None:
        await asyncio.sleep(30)

    monkeypatch.setattr(
        "litellm.proxy.proxy_server.spend_counter_store",
        FakeSpendCounters(redis={}, batch_reads=False, read_hook=never_answers).store(),
    )
    monkeypatch.setenv("PROMETHEUS_BUDGET_METRICS_PER_REQUEST_TIMEOUT", "0.5")

    await asyncio.wait_for(_emit(prometheus_logger, _request_metadata(_token_committed_at_auth())), timeout=5)

    assert _remaining(prometheus_logger) == {
        "key": pytest.approx(5.0),
        "team": pytest.approx(268.0),
        "user": pytest.approx(40.0),
        "org": pytest.approx(438.0),
    }


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


@pytest.mark.asyncio
async def test_gauges_add_the_own_charge_to_a_same_size_charge_of_another_request_while_the_own_waits_in_the_pipeline(
    prometheus_logger, getters, redis_spend_counters
):
    """Another request's 2.0 reached the counters and this request's 2.0 is queued, so both count."""
    _store_committed(redis_spend_counters, key=5.0, team=32.0, user=10.0, org=62.0)

    with _request_scope():
        await _declare_the_own_two_dollar_charge()
        await _emit(prometheus_logger, _request_metadata(_token_committed_at_auth()))
        await flush_post_call_redis_batches()

    assert _remaining(prometheus_logger) == _BOTH_TWO_DOLLAR_CHARGES_COUNTED_ONCE


@pytest.mark.asyncio
async def test_gauges_do_not_add_the_own_charge_twice_once_the_pipeline_went_out(
    prometheus_logger, getters, redis_spend_counters
):
    """The 2.0 over the auth time spend is this request's own charge, settled before the gauges read it."""
    _store_committed(redis_spend_counters, key=3.0, team=30.0, user=8.0, org=60.0)

    with _request_scope():
        await _declare_the_own_two_dollar_charge()
        redis_spend_counters.settle_post_call()
        await flush_post_call_redis_batches()
        await _emit(prometheus_logger, _request_metadata(_token_committed_at_auth()))

    assert _remaining(prometheus_logger) == {
        "key": pytest.approx(5.0),
        "team": pytest.approx(268.0),
        "user": pytest.approx(40.0),
        "org": pytest.approx(438.0),
    }


@pytest.mark.asyncio
async def test_an_unrelated_post_call_flush_does_not_make_the_gauge_drop_its_own_unsent_charge(
    prometheus_logger, getters, redis_spend_counters
):
    """A post-call flush went out first, then another request's 2.0 reached the counters. This request's 2.0 counts
    once whether it is undeclared, declared but unsent, or settled."""
    _store_committed(redis_spend_counters, key=5.0, team=32.0, user=10.0, org=62.0)
    metadata: Final = _request_metadata(_token_committed_at_auth())

    with _request_scope():
        await flush_post_call_redis_batches()
        await _emit(prometheus_logger, metadata)
        undeclared: Final = _remaining(prometheus_logger)
        await _declare_the_own_two_dollar_charge()
        await _emit(prometheus_logger, metadata)
        declared_unsent: Final = _remaining(prometheus_logger)
        redis_spend_counters.settle_post_call()
        await _emit(prometheus_logger, metadata)
        settled: Final = _remaining(prometheus_logger)
        await flush_post_call_redis_batches()

    assert {"undeclared": undeclared, "declared_unsent": declared_unsent, "settled": settled} == {
        "undeclared": _BOTH_TWO_DOLLAR_CHARGES_COUNTED_ONCE,
        "declared_unsent": _BOTH_TWO_DOLLAR_CHARGES_COUNTED_ONCE,
        "settled": _BOTH_TWO_DOLLAR_CHARGES_COUNTED_ONCE,
    }


@pytest.mark.asyncio
async def test_gauges_count_the_own_charge_once_when_it_settles_while_the_gauges_read_the_counters(
    prometheus_logger, getters, redis_spend_counters
):
    """Another request's 2.0 reached the counters, and this request's 2.0 goes out during the gauges' read."""
    _store_committed(redis_spend_counters, key=5.0, team=32.0, user=10.0, org=62.0)

    async def settle_during_read(_counter: str) -> None:
        redis_spend_counters.settle_post_call()

    redis_spend_counters.batch_reads = False
    redis_spend_counters.read_hook = settle_during_read

    with _request_scope():
        await _declare_the_own_two_dollar_charge()
        await _emit(prometheus_logger, _request_metadata(_token_committed_at_auth()))
        await flush_post_call_redis_batches()

    assert _remaining(prometheus_logger) == _BOTH_TWO_DOLLAR_CHARGES_COUNTED_ONCE


@pytest.mark.asyncio
async def test_gauges_without_redis_treat_a_charge_equal_to_the_own_as_the_own(
    prometheus_logger, getters, spend_counters
):
    """Without Redis the charge increments in memory at once, so a request scope says nothing about it."""
    _store_committed(spend_counters, key=5.0, team=32.0, user=10.0, org=62.0)

    with request_redis_batch_scope():
        await _emit(prometheus_logger, _request_metadata(_token_committed_at_auth()))

    assert _remaining(prometheus_logger) == {
        "key": pytest.approx(5.0),
        "team": pytest.approx(268.0),
        "user": pytest.approx(40.0),
        "org": pytest.approx(438.0),
    }


@pytest.mark.asyncio
async def test_gauges_add_the_cost_to_the_auth_time_spend_before_the_charge_reaches_the_counters(
    prometheus_logger, getters, spend_counters
):
    _store_committed(spend_counters, key=3.0, team=30.0, user=8.0, org=60.0)

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
    monkeypatch.delattr("litellm.proxy.proxy_server.spend_counter_store")

    await _emit(prometheus_logger, _request_metadata(_token_committed_at_auth()))

    assert _remaining(prometheus_logger) == {
        "key": pytest.approx(5.0),
        "team": pytest.approx(268.0),
        "user": pytest.approx(40.0),
        "org": pytest.approx(438.0),
    }


@pytest.mark.asyncio
async def test_a_budget_hold_on_the_enforcement_counters_never_shows_in_the_gauges(
    prometheus_logger, getters, spend_counters
):
    _store_committed(spend_counters, key=5.0, team=32.0, user=10.0, org=62.0)
    _store_counters(spend_counters, key=9.0, team=36.0, user=14.0, org=66.0)

    await _emit(prometheus_logger, _request_metadata(_token_committed_at_auth()))

    assert _remaining(prometheus_logger) == {
        "key": pytest.approx(5.0),
        "team": pytest.approx(268.0),
        "user": pytest.approx(40.0),
        "org": pytest.approx(438.0),
    }


def _failed_request(metadata: dict, response_cost: float = 0.0) -> dict:
    return {
        "model": "openai/gpt-5.4-nano",
        "litellm_params": {"metadata": metadata},
        "standard_logging_object": standard_payload(
            metadata={
                "user_api_key_hash": "hashed",
                "user_api_key_alias": "key-alias",
                "user_api_key_team_id": "t1",
                "user_api_key_team_alias": "team-alias",
                "user_api_key_user_id": "u1",
                "user_api_key_org_id": "o1",
                "requester_metadata": None,
                "user_api_key_end_user_id": None,
            },
            model="openai/gpt-5.4-nano",
            model_id="dep-1",
            model_group="chat",
            api_base="https://api.openai.com",
            custom_llm_provider="openai",
            status="failure",
            response_cost=response_cost,
            request_tags=[],
            hidden_params={"litellm_overhead_time_ms": None, "additional_headers": None},
        ),
        "exception": Exception("upstream failed"),
    }


@pytest.mark.asyncio
async def test_failed_request_sets_the_org_gauge_from_the_carried_org_and_committed_spend(
    prometheus_logger, getters, spend_counters
):
    _store_committed(spend_counters, key=3.0, team=30.0, user=8.0, org=65.0)
    _store_counters(spend_counters, key=9.0, team=36.0, user=14.0, org=90.0)

    await native_failure(
        prometheus_logger,
        _failed_request(_request_metadata(_token_committed_at_auth())),
        None,
        KEY_RESET_AT,
        KEY_RESET_AT,
    )

    assert getters["get_org_object"].await_count == 0
    _set_once_with("litellm_remaining_org_budget_metric", pytest.approx(435.0))


@pytest.mark.asyncio
async def test_failed_request_without_the_auth_object_still_reads_the_committed_org_spend(
    prometheus_logger, getters, spend_counters
):
    _store_committed(spend_counters, key=3.0, team=30.0, user=8.0, org=65.0)
    metadata = _request_metadata(_token_committed_at_auth())
    del metadata["user_api_key_auth"]

    await native_failure(prometheus_logger, _failed_request(metadata), None, KEY_RESET_AT, KEY_RESET_AT)

    _set_once_with("litellm_remaining_org_budget_metric", pytest.approx(435.0))


@pytest.mark.asyncio
async def test_failed_request_without_a_carried_org_reads_no_committed_org_spend(
    prometheus_logger, getters, spend_counters
):
    _store_committed(spend_counters, key=3.0, team=30.0, user=8.0, org=65.0)
    token = _token_committed_at_auth().model_copy(update={"org_budget_snapshot": None})
    metadata = {key: value for key, value in _request_metadata(token).items() if key != "user_api_key_auth"}

    await native_failure(prometheus_logger, _failed_request(metadata), None, KEY_RESET_AT, KEY_RESET_AT)

    _set_once_with("litellm_remaining_org_budget_metric", pytest.approx(460.0))


@pytest.mark.asyncio
async def test_failed_request_that_was_charged_counts_its_own_charge_while_it_waits_in_the_pipeline(
    prometheus_logger, getters, redis_spend_counters
):
    """A stream cut after 40 events still costs 2.0, and that charge is queued when the failure logs."""
    _store_committed(redis_spend_counters, key=5.0, team=32.0, user=10.0, org=62.0)

    with _request_scope():
        await _declare_the_own_two_dollar_charge()
        await native_failure(
            prometheus_logger,
            _failed_request(_request_metadata(_token_committed_at_auth()), response_cost=2.0),
            None,
            KEY_RESET_AT,
            KEY_RESET_AT,
        )
        await flush_post_call_redis_batches()

    _set_once_with("litellm_remaining_org_budget_metric", pytest.approx(436.0))


@pytest.mark.asyncio
async def test_ordinary_metadata_keys_named_like_snapshot_fields_change_no_gauge(prometheus_logger, getters):
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
    _set_once_with("litellm_remaining_org_budget_metric", 458.0)


@pytest.mark.asyncio
async def test_metadata_without_carried_state_still_fetches_each_object_once(prometheus_logger, getters):

    await _emit(prometheus_logger, {"user_api_key_team_spend": 20.0, "user_api_key_team_max_budget": 300.0})

    assert {name: getter.await_count for name, getter in getters.items()} == {
        "get_key_object": 1,
        "get_team_object": 1,
        "get_user_object": 1,
        "get_org_object": 1,
    }
    _set_once_with("litellm_remaining_org_budget_metric", 458.0)
    _set_once("litellm_team_budget_remaining_hours_metric")


@pytest.mark.asyncio
async def test_partial_carried_state_only_skips_the_carried_objects(prometheus_logger, getters):
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
