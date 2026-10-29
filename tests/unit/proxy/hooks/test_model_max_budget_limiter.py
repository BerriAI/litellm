import asyncio
import functools
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from types import MappingProxyType, SimpleNamespace
from typing import Final

import pytest

import litellm
from litellm.caching.caching import DualCache
from litellm.proxy._types import Litellm_EntityType, UserAPIKeyAuth
from litellm.proxy.db.db_transaction_queue.window_spend_update_queue import WindowSpendTransaction
from litellm.proxy.hooks.model_budget_shared_windows import DatabaseSharedWindows, SharedWindows
from litellm.proxy.hooks.model_max_budget_limiter import (
    PROXY_VirtualKeyModelMaxBudgetLimiter,
    build_model_max_budget_usage,
    model_access_groups_of,
    model_budget_spend_cache_key,
    resolve_model_budget,
    routed_models_of,
    team_member_budget_entity_id,
)
from litellm.router import Router
from litellm.types.caching import RedisPipelineIncrementOperation
from litellm.types.utils import LiteLLMBatch, Usage

KEY_HASH: Final = "key-hash-batch"
USER_ID: Final = "user-batch"
TEAM_ID: Final = "team-batch"
MODEL_GROUP: Final = "batch-qa-primary"
BATCH_COST: Final = 2.925e-05
CHAT_COST: Final = 0.001
KEY_SPEND_KEY: Final = f"virtual_key_spend:{KEY_HASH}:{MODEL_GROUP}:1d"
USER_SPEND_KEY: Final = f"user_model_spend:{USER_ID}:{MODEL_GROUP}:1d"


def _batch(batch_id: str, status: str) -> LiteLLMBatch:
    return LiteLLMBatch(
        id=batch_id,
        completion_window="24h",
        created_at=1,
        endpoint="/v1/chat/completions",
        input_file_id="file-batch",
        object="batch",
        status=status,
        usage=Usage(prompt_tokens=20, completion_tokens=18, total_tokens=38),
    )


def _event(
    call_type: str,
    response_cost: float,
    team_member_model_max_budget: Mapping[str, object] | None = None,
    user_id: str = USER_ID,
    team_id: str = TEAM_ID,
) -> dict[str, object]:
    return {
        "call_type": call_type,
        "standard_logging_object": {
            "call_type": call_type,
            "response_cost": response_cost,
            "model": "openai/gpt-5.4-mini",
            "model_group": MODEL_GROUP,
            "metadata": {
                "user_api_key_hash": KEY_HASH,
                "user_api_key_user_id": user_id,
                "user_api_key_team_id": team_id,
            },
        },
        "litellm_params": {
            "metadata": {
                "user_api_key_model_max_budget": {MODEL_GROUP: {"budget_limit": 0.0001, "time_period": "1d"}},
                "user_api_key_user_model_max_budget": {MODEL_GROUP: {"budget_limit": 0.0001, "time_period": "1d"}},
                **(
                    {"user_api_key_team_member_model_max_budget": team_member_model_max_budget}
                    if team_member_model_max_budget is not None
                    else {}
                ),
            }
        },
    }


async def _log_success(
    limiter: PROXY_VirtualKeyModelMaxBudgetLimiter, event: dict[str, object], response_obj: object = None
) -> None:
    await limiter.async_log_success_event(event, response_obj=response_obj, start_time=None, end_time=None)


async def _poll(limiter: PROXY_VirtualKeyModelMaxBudgetLimiter, batch: LiteLLMBatch, response_cost: float) -> None:
    await _log_success(limiter, _event("aretrieve_batch", response_cost), response_obj=batch)


async def _chat(
    limiter: PROXY_VirtualKeyModelMaxBudgetLimiter,
    team_member_model_max_budget: Mapping[str, object] | None = None,
    user_id: str = USER_ID,
    team_id: str = TEAM_ID,
) -> None:
    await _log_success(
        limiter,
        _event(
            "acompletion",
            CHAT_COST,
            team_member_model_max_budget=team_member_model_max_budget,
            user_id=user_id,
            team_id=team_id,
        ),
    )


async def _spend(limiter: PROXY_VirtualKeyModelMaxBudgetLimiter, spend_key: str) -> float:
    return await limiter.dual_cache.async_get_cache(key=spend_key) or 0.0


def _local_spend(limiter: PROXY_VirtualKeyModelMaxBudgetLimiter, spend_key: str) -> float:
    return limiter.dual_cache.in_memory_cache.get_cache(key=spend_key) or 0.0


@pytest.mark.asyncio
async def test_team_member_budget_is_charged_and_enforced_after_success():
    budget: Final = {MODEL_GROUP: {"max_budget": 0.0005, "budget_duration": "1d"}}
    limiter: Final = PROXY_VirtualKeyModelMaxBudgetLimiter(dual_cache=DualCache())

    await _chat(limiter, team_member_model_max_budget=budget)

    entity_id: Final = team_member_budget_entity_id(user_id=USER_ID, team_id=TEAM_ID)
    spend_key: Final = model_budget_spend_cache_key(
        entity_type=Litellm_EntityType.TEAM_MEMBER,
        entity_id=entity_id,
        budget_model=MODEL_GROUP,
        budget_duration="1d",
    )
    assert await _spend(limiter, spend_key) == CHAT_COST

    with pytest.raises(litellm.BudgetExceededError, match="LiteLLM Team Member"):
        await limiter.is_team_member_within_model_budget(
            user_id=USER_ID,
            team_id=TEAM_ID,
            team_member_model_max_budget=budget,
            model=MODEL_GROUP,
        )


def test_team_member_budget_entity_id_is_unambiguous():
    first_entity_id: Final = team_member_budget_entity_id(user_id="a:b", team_id="c")
    second_entity_id: Final = team_member_budget_entity_id(user_id="a", team_id="b:c")

    assert first_entity_id == "3:a:b:c"
    assert second_entity_id == "1:a:b:c"

    first_spend_key: Final = model_budget_spend_cache_key(
        entity_type=Litellm_EntityType.TEAM_MEMBER,
        entity_id=first_entity_id,
        budget_model=MODEL_GROUP,
        budget_duration="1d",
    )
    second_spend_key: Final = model_budget_spend_cache_key(
        entity_type=Litellm_EntityType.TEAM_MEMBER,
        entity_id=second_entity_id,
        budget_model=MODEL_GROUP,
        budget_duration="1d",
    )
    assert first_spend_key != second_spend_key


@pytest.mark.asyncio
async def test_team_member_budget_spend_isolated_by_user_and_team():
    budget: Final = {MODEL_GROUP: {"max_budget": CHAT_COST * 2, "budget_duration": "1d"}}
    limiter: Final = PROXY_VirtualKeyModelMaxBudgetLimiter(dual_cache=DualCache())

    await _chat(limiter, team_member_model_max_budget=budget)

    assert await limiter.is_team_member_within_model_budget(
        user_id="different-user",
        team_id=TEAM_ID,
        team_member_model_max_budget=budget,
        model=MODEL_GROUP,
    )
    assert await limiter.is_team_member_within_model_budget(
        user_id=USER_ID,
        team_id="different-team",
        team_member_model_max_budget=budget,
        model=MODEL_GROUP,
    )


@pytest.mark.asyncio
async def test_zero_team_member_model_budget_rejects_first_request():
    limiter: Final = PROXY_VirtualKeyModelMaxBudgetLimiter(dual_cache=DualCache())

    with pytest.raises(litellm.BudgetExceededError, match="LiteLLM Team Member"):
        await limiter.is_team_member_within_model_budget(
            user_id=USER_ID,
            team_id=TEAM_ID,
            team_member_model_max_budget={MODEL_GROUP: {"max_budget": 0, "budget_duration": "1d"}},
            model=MODEL_GROUP,
        )


class _Clock:
    def __init__(self) -> None:
        self.seconds = 0.0

    def now(self) -> float:
        return self.seconds

    def advance(self, seconds: float) -> None:
        self.seconds = self.seconds + seconds


class _SharedRedisDouble:
    def __init__(self, now: Callable[[], float] = time.time) -> None:
        self.now = now
        self.entries: Mapping[str, tuple[float, float | None]] = MappingProxyType({})

    def _live(self, key: str) -> tuple[float, float | None] | None:
        entry: Final = self.entries.get(key)
        if entry is None:
            return None
        expires_at: Final = entry[1]
        if expires_at is not None and expires_at <= self.now():
            return None
        return entry

    def _store(self, key: str, value: float, expires_at: float | None) -> None:
        self.entries = MappingProxyType({**self.entries, key: (value, expires_at)})

    async def async_get_cache(self, key: str, **kwargs: object) -> float | None:
        await asyncio.sleep(0)
        entry: Final = self._live(key)
        return None if entry is None else entry[0]

    async def async_set_cache(self, key: str, value: float, ttl: int | None = None, **kwargs: object) -> None:
        await asyncio.sleep(0)
        self._store(key, value, None if ttl is None else self.now() + ttl)

    async def async_increment(
        self,
        key: str,
        value: float,
        ttl: int | None = None,
        parent_otel_span: object = None,
        refresh_ttl: bool = False,
    ) -> float:
        await asyncio.sleep(0)
        live: Final = self._live(key)
        total: Final = value if live is None else live[0] + value
        kept_expiry: Final = None if live is None else live[1]
        expires_at: Final = (
            kept_expiry if ttl is None or (kept_expiry is not None and not refresh_ttl) else self.now() + ttl
        )
        self._store(key, total, expires_at)
        return total

    async def async_increment_pipeline(self, increment_list: list[RedisPipelineIncrementOperation]) -> list[float]:
        return [await self.async_increment(op["key"], op["increment_value"], ttl=op["ttl"]) for op in increment_list]


def _worker(redis: _SharedRedisDouble) -> PROXY_VirtualKeyModelMaxBudgetLimiter:
    return PROXY_VirtualKeyModelMaxBudgetLimiter(
        dual_cache=DualCache(redis_cache=redis)  # pyright: ignore[reportArgumentType]  # duck-typed Redis double
    )


async def _drain_redis_pushes() -> None:
    await asyncio.gather(*(task for task in asyncio.all_tasks() if task is not asyncio.current_task()))


@pytest.mark.asyncio
async def test_polls_of_a_finished_batch_charge_each_per_model_budget_once():
    limiter: Final = PROXY_VirtualKeyModelMaxBudgetLimiter(dual_cache=DualCache())
    first: Final = _batch("batch_first", "completed")

    await _poll(limiter, _batch("batch_first", "in_progress"), response_cost=0)
    for _ in range(3):
        await _poll(limiter, first, response_cost=BATCH_COST)

    assert await _spend(limiter, KEY_SPEND_KEY) == pytest.approx(BATCH_COST)
    assert await _spend(limiter, USER_SPEND_KEY) == pytest.approx(BATCH_COST)


@pytest.mark.asyncio
async def test_a_second_batch_and_chat_requests_still_charge_the_budget():
    limiter: Final = PROXY_VirtualKeyModelMaxBudgetLimiter(dual_cache=DualCache())

    await _poll(limiter, _batch("batch_first", "completed"), response_cost=BATCH_COST)
    await _poll(limiter, _batch("batch_first", "completed"), response_cost=BATCH_COST)
    await _poll(limiter, _batch("batch_second", "completed"), response_cost=BATCH_COST)
    await _chat(limiter)
    await _chat(limiter)

    assert await _spend(limiter, KEY_SPEND_KEY) == pytest.approx(2 * BATCH_COST + 2 * CHAT_COST)


@pytest.mark.asyncio
async def test_two_workers_polling_the_same_finished_batch_at_once_charge_it_once():
    redis: Final = _SharedRedisDouble()
    worker_a: Final = _worker(redis)
    worker_b: Final = _worker(redis)
    finished: Final = _batch("batch_first", "completed")

    await asyncio.gather(_poll(worker_a, finished, BATCH_COST), _poll(worker_b, finished, BATCH_COST))
    await _drain_redis_pushes()

    assert _local_spend(worker_a, KEY_SPEND_KEY) + _local_spend(worker_b, KEY_SPEND_KEY) == pytest.approx(BATCH_COST)
    assert await redis.async_get_cache(KEY_SPEND_KEY) == pytest.approx(BATCH_COST)


@pytest.mark.asyncio
async def test_a_batch_polled_within_every_budget_window_is_never_charged_again():
    clock: Final = _Clock()
    limiter: Final = _worker(_SharedRedisDouble(now=clock.now))
    finished: Final = _batch("batch_first", "completed")

    await _poll(limiter, finished, BATCH_COST)
    clock.advance(12 * 3600)
    await _poll(limiter, finished, BATCH_COST)
    clock.advance(18 * 3600)
    await _poll(limiter, finished, BATCH_COST)

    assert _local_spend(limiter, KEY_SPEND_KEY) == pytest.approx(BATCH_COST)


OPUS_GROUP: Final = "claude-opus"
OPUS_GROUP_MEMBERS: Final = MappingProxyType(
    {"claude-opus-4-8": frozenset({OPUS_GROUP}), "claude-opus-5-5": frozenset({OPUS_GROUP})}
)
FAMILY_CAP: Final = {"budget_limit": CHAT_COST, "time_period": "1d"}  # mutable-ok: BudgetConfig validates dicts only
ROOMY_CAP: Final = {
    "budget_limit": CHAT_COST * 100,
    "time_period": "1d",
}  # mutable-ok: BudgetConfig validates dicts only
WINDOW_EPOCH: Final = datetime(2026, 10, 20, 9, 30, tzinfo=timezone.utc).timestamp()


def _model_access_groups(
    groups_by_model: Mapping[str, frozenset[str]],
) -> Callable[[str, str | None], frozenset[str]]:
    return lambda model, _team_id: groups_by_model.get(model, frozenset())


@dataclass(frozen=True, slots=True)
class _WindowRow:
    window_start: datetime
    spend: float


class _WindowSpendDatabase:
    def __init__(self) -> None:
        self.rows: Mapping[str, _WindowRow] = MappingProxyType({})
        self.queued: tuple[WindowSpendTransaction, ...] = ()
        self.reachable = True

    @property
    def litellm_budgetwindowspend(self) -> "_WindowSpendDatabase":
        return self

    async def find_unique(self, where: Mapping[str, Mapping[str, str]]) -> _WindowRow | None:
        await asyncio.sleep(0)
        self._answer()
        return self.rows.get(where["entity_type_entity_id_window_duration"]["entity_id"])

    async def execute_raw(
        self,
        query: str,
        entity_type: str,
        entity_id: str,
        window_duration: str,
        window_start: datetime,
        replaces_window_start: datetime | None,
    ) -> int:
        await asyncio.sleep(0)
        self._answer()
        stored: Final = self.rows.get(entity_id)
        if stored is not None and (replaces_window_start is None or stored.window_start > replaces_window_start):
            return 0
        self.rows = MappingProxyType({**self.rows, entity_id: _WindowRow(window_start=window_start, spend=0.0)})
        return 1

    async def queue(self, transaction: WindowSpendTransaction) -> None:
        self.queued = (*self.queued, transaction)

    def flush(self) -> None:
        self.rows = functools.reduce(_upsert_window_spend, self.queued, self.rows)
        self.queued = ()

    def replica(self) -> DatabaseSharedWindows:
        return DatabaseSharedWindows(
            prisma_client_source=lambda: SimpleNamespace(db=self, writer_db=self),  # pyright: ignore[reportArgumentType]  # duck-typed prisma double
            sink=self.queue,
        )

    def _answer(self) -> None:
        if not self.reachable:
            raise ConnectionError("database is unreachable")


def _upsert_window_spend(
    rows: Mapping[str, _WindowRow], transaction: WindowSpendTransaction
) -> Mapping[str, _WindowRow]:
    window_start: Final = datetime.fromisoformat(transaction["window_start"])
    stored: Final = rows.get(transaction["entity_id"])
    row: Final = (
        _WindowRow(window_start=stored.window_start, spend=stored.spend + transaction["spend"])
        if stored is not None and stored.window_start >= window_start
        else _WindowRow(window_start=window_start, spend=transaction["spend"])
    )
    return MappingProxyType({**rows, transaction["entity_id"]: row})


def _family_limiter(
    groups_by_model: Mapping[str, frozenset[str]] = OPUS_GROUP_MEMBERS,
    dual_cache: DualCache | None = None,
    shared_windows: SharedWindows | None = None,
    clock: Callable[[], float] = time.time,
) -> PROXY_VirtualKeyModelMaxBudgetLimiter:
    return PROXY_VirtualKeyModelMaxBudgetLimiter(
        dual_cache=dual_cache or DualCache(),
        clock=clock,
        model_access_groups=_model_access_groups(groups_by_model),
        shared_windows=shared_windows or _WindowSpendDatabase().replica(),
    )


def _key(model_max_budget: Mapping[str, object]) -> UserAPIKeyAuth:
    return UserAPIKeyAuth(token=KEY_HASH, key_alias="opus-family", model_max_budget=dict(model_max_budget))


async def _spend_on(
    limiter: PROXY_VirtualKeyModelMaxBudgetLimiter, model: str, model_max_budget: Mapping[str, object]
) -> None:
    await _log_success(
        limiter,
        {
            "call_type": "acompletion",
            "standard_logging_object": {
                "call_type": "acompletion",
                "response_cost": CHAT_COST,
                "model": f"anthropic/{model}",
                "model_group": model,
                "metadata": {"user_api_key_hash": KEY_HASH},
            },
            "litellm_params": {"metadata": {"user_api_key_model_max_budget": dict(model_max_budget)}},
        },
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("family_entry", ["claude-opus*", OPUS_GROUP])
async def test_every_opus_version_draws_from_one_family_entry(family_entry: str):
    budget: Final = {family_entry: FAMILY_CAP}
    limiter: Final = _family_limiter()
    assert await limiter.is_key_within_model_budget(user_api_key_dict=_key(budget), model="claude-opus-5-5")

    await _spend_on(limiter, model="claude-opus-4-8", model_max_budget=budget)

    with pytest.raises(
        litellm.BudgetExceededError, match=rf"model=claude-opus-5-5, budget_model={re.escape(family_entry)}$"
    ):
        await limiter.is_key_within_model_budget(user_api_key_dict=_key(budget), model="claude-opus-5-5")
    assert await limiter.is_key_within_model_budget(user_api_key_dict=_key(budget), model="claude-sonnet-5")


@pytest.mark.asyncio
async def test_a_model_added_to_the_access_group_later_is_covered_with_no_key_change():
    budget: Final = {OPUS_GROUP: FAMILY_CAP}
    shared_cache: Final = DualCache()
    before_group_change: Final = _family_limiter(dual_cache=shared_cache)
    after_group_change: Final = _family_limiter(
        groups_by_model={**OPUS_GROUP_MEMBERS, "claude-opus-6": frozenset({OPUS_GROUP})}, dual_cache=shared_cache
    )

    await _spend_on(before_group_change, model="claude-opus-4-8", model_max_budget=budget)

    assert await before_group_change.is_key_within_model_budget(user_api_key_dict=_key(budget), model="claude-opus-6")
    with pytest.raises(litellm.BudgetExceededError):
        await after_group_change.is_key_within_model_budget(user_api_key_dict=_key(budget), model="claude-opus-6")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("budget", "checked_model", "blocked"),
    [
        ({"claude-opus-4-8": ROOMY_CAP, "claude-opus*": FAMILY_CAP}, "claude-opus-4-8", False),
        ({"claude-opus-4-8": ROOMY_CAP, "claude-opus*": FAMILY_CAP}, "claude-opus-5-5", False),
        ({OPUS_GROUP: FAMILY_CAP, "claude-opus*": ROOMY_CAP}, "claude-opus-5-5", True),
        ({"claude-opus-4*": FAMILY_CAP, "claude-opus*": ROOMY_CAP}, "claude-opus-4-7", True),
        ({"claude-opus-4*": FAMILY_CAP, "claude-opus*": ROOMY_CAP}, "claude-opus-5-5", False),
        ({"claude-opus-4*": {"budget_limit": CHAT_COST}, "claude-opus*": FAMILY_CAP}, "claude-opus-5-5", True),
        ({"claude-opus-4*": FAMILY_CAP, "c" + "*" * 15: ROOMY_CAP}, "claude-opus-4-7", True),
    ],
)
async def test_the_most_specific_entry_takes_the_spend_and_gates_the_request(
    budget: Mapping[str, object], checked_model: str, blocked: bool
):
    limiter: Final = _family_limiter()

    await _spend_on(limiter, model="claude-opus-4-8", model_max_budget=budget)

    if blocked:
        with pytest.raises(litellm.BudgetExceededError):
            await limiter.is_key_within_model_budget(user_api_key_dict=_key(budget), model=checked_model)
    else:
        assert await limiter.is_key_within_model_budget(user_api_key_dict=_key(budget), model=checked_model)


@pytest.mark.asyncio
async def test_an_exact_entry_refusal_reads_as_before_for_a_provider_prefixed_model():
    budget: Final = {"claude-opus-4-8": FAMILY_CAP}
    limiter: Final = _family_limiter()

    await _spend_on(limiter, model="claude-opus-4-8", model_max_budget=budget)

    with pytest.raises(litellm.BudgetExceededError, match=r"model=anthropic/claude-opus-4-8$"):
        await limiter.is_key_within_model_budget(user_api_key_dict=_key(budget), model="anthropic/claude-opus-4-8")


async def _spend_on_team_key(
    limiter: PROXY_VirtualKeyModelMaxBudgetLimiter,
    model: str,
    key_model_max_budget: Mapping[str, object],
    team_model_max_budget: Mapping[str, object],
) -> None:
    await _log_success(
        limiter,
        {
            "call_type": "acompletion",
            "standard_logging_object": {
                "call_type": "acompletion",
                "response_cost": CHAT_COST,
                "model": f"anthropic/{model}",
                "model_group": model,
                "metadata": {"user_api_key_hash": KEY_HASH, "user_api_key_team_id": TEAM_ID},
            },
            "litellm_params": {
                "metadata": {
                    "user_api_key_model_max_budget": dict(key_model_max_budget),
                    "user_api_key_team_model_max_budget": dict(team_model_max_budget),
                }
            },
        },
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("team_model_max_budget", "team_blocks"),
    [({"claude-opus*": FAMILY_CAP}, True), ({OPUS_GROUP: FAMILY_CAP}, True), ({"claude-opus-4-8": FAMILY_CAP}, False)],
)
async def test_a_key_entry_named_after_the_model_replaces_only_the_team_entry_of_that_name(
    team_model_max_budget: Mapping[str, object], team_blocks: bool
):
    key_model_max_budget: Final = {"claude-opus-4-8": ROOMY_CAP}
    limiter: Final = _family_limiter()

    await _spend_on_team_key(
        limiter,
        model="claude-opus-4-8",
        key_model_max_budget=key_model_max_budget,
        team_model_max_budget=team_model_max_budget,
    )
    team_check: Final = limiter.is_team_within_model_budget(
        team_id=TEAM_ID,
        team_model_max_budget=team_model_max_budget,
        key_model_max_budget=key_model_max_budget,
        model="claude-opus-4-8",
    )

    if team_blocks:
        with pytest.raises(litellm.BudgetExceededError, match=r"budget_model="):
            await team_check
    else:
        assert await team_check


TEAM_OPUS_GROUP: Final = "team-opus"


@pytest.mark.asyncio
@pytest.mark.parametrize("budget_entry", [TEAM_OPUS_GROUP, "claude-opus*"])
async def test_a_team_model_is_gated_and_charged_under_its_public_name(budget_entry: str):
    router: Final = Router(
        model_list=[
            {
                "model_name": f"model_name_{TEAM_ID}_opus",
                "litellm_params": {"model": "anthropic/claude-opus-4-8", "api_key": "sk-test"},
                "model_info": {
                    "id": "team-opus-deployment",
                    "team_id": TEAM_ID,
                    "team_public_model_name": "claude-opus-4-8",
                    "access_groups": [TEAM_OPUS_GROUP],
                },
            }
        ]
    )
    limiter: Final = PROXY_VirtualKeyModelMaxBudgetLimiter(
        dual_cache=DualCache(),
        model_access_groups=lambda model, team_id: model_access_groups_of(router=router, model=model, team_id=team_id),
        shared_windows=_WindowSpendDatabase().replica(),
    )
    budget: Final = {budget_entry: FAMILY_CAP}
    team_key: Final = UserAPIKeyAuth(token=KEY_HASH, team_id=TEAM_ID, model_max_budget=budget)
    assert await limiter.is_key_within_model_budget(user_api_key_dict=team_key, model="claude-opus-4-8")

    await _log_success(
        limiter,
        {
            "call_type": "acompletion",
            "standard_logging_object": {
                "call_type": "acompletion",
                "response_cost": CHAT_COST,
                "model": "anthropic/claude-opus-4-8",
                "model_group": f"model_name_{TEAM_ID}_opus",
                "metadata": {"user_api_key_hash": KEY_HASH, "user_api_key_team_id": TEAM_ID},
            },
            "litellm_params": {
                "metadata": {
                    "user_api_key_model_max_budget": budget,
                    "model_info": router.get_deployment(model_id="team-opus-deployment").model_info.model_dump(),
                }
            },
        },
    )

    with pytest.raises(litellm.BudgetExceededError, match=rf"budget_model={re.escape(budget_entry)}$"):
        await limiter.is_key_within_model_budget(user_api_key_dict=team_key, model="claude-opus-4-8")


@pytest.mark.asyncio
async def test_spend_charged_while_the_database_is_unreachable_reaches_the_shared_window_without_wiping_it():
    budget: Final = {"claude-opus*": ROOMY_CAP}
    database: Final = _WindowSpendDatabase()
    clock: Final = _Clock()
    clock.advance(WINDOW_EPOCH)
    replica_a: Final = _family_limiter(shared_windows=database.replica(), clock=clock.now)
    replica_b: Final = _family_limiter(shared_windows=database.replica(), clock=clock.now)

    await _spend_on(replica_a, model="claude-opus-4-8", model_max_budget=budget)
    database.flush()
    clock.advance(60)
    database.reachable = False
    await _spend_on(replica_b, model="claude-opus-4-8", model_max_budget=budget)
    database.reachable = True
    database.flush()
    clock.advance(60)
    await _spend_on(replica_b, model="claude-opus-5-5", model_max_budget=budget)
    database.flush()

    usage: Final = await build_model_max_budget_usage(
        entity_type=Litellm_EntityType.KEY,
        entity_id=KEY_HASH,
        model_max_budget=budget,
        cache=DualCache(),
        now=datetime.fromtimestamp(clock.now(), tz=timezone.utc),
        shared_windows=database.replica(),
    )
    assert usage["claude-opus*"]["current_spend"] == round(CHAT_COST * 3, 4)
    assert usage["claude-opus*"]["budget_reset_at"] == datetime.fromtimestamp(
        WINDOW_EPOCH + 24 * 3600, tz=timezone.utc
    ).isoformat()


@pytest.mark.asyncio
async def test_end_user_windows_stay_on_the_replica_without_redis():
    budget: Final = {"claude-opus*": FAMILY_CAP}
    table: Final = _WindowSpendDatabase()
    limiter: Final = _family_limiter(shared_windows=table.replica())

    await _log_success(
        limiter,
        {
            "call_type": "acompletion",
            "standard_logging_object": {
                "call_type": "acompletion",
                "response_cost": CHAT_COST,
                "model": "anthropic/claude-opus-4-8",
                "model_group": "claude-opus-4-8",
                "end_user": "end-user-1",
                "metadata": {},
            },
            "litellm_params": {"metadata": {"user_api_key_end_user_model_max_budget": budget}},
        },
    )
    table.flush()

    assert dict(table.rows) == {}
    with pytest.raises(litellm.BudgetExceededError, match="model=claude-opus-5-5"):
        await limiter.is_end_user_within_model_budget(
            end_user_id="end-user-1", end_user_model_max_budget=budget, model="claude-opus-5-5"
        )


def test_an_access_group_named_after_a_served_model_never_widens_that_models_entry():
    router: Final = Router(
        model_list=[
            {
                "model_name": "gpt-4o",
                "litellm_params": {"model": "openai/gpt-4o", "api_key": "sk-test"},
                "model_info": {"access_groups": ["gpt-4o", "openai-models"]},
            },
            {
                "model_name": "gpt-4o-mini",
                "litellm_params": {"model": "openai/gpt-4o-mini", "api_key": "sk-test"},
                "model_info": {"access_groups": ["gpt-4o", "openai-models"]},
            },
        ]
    )

    def governing_entry(model_max_budget: Mapping[str, object]) -> str | None:
        resolved: Final = resolve_model_budget(
            model="gpt-4o-mini",
            model_max_budget=model_max_budget,
            model_access_groups=lambda model, team_id: model_access_groups_of(router=router, model=model, team_id=team_id),
        )
        return None if resolved is None else resolved.budget_model

    assert governing_entry({"gpt-4o": FAMILY_CAP}) is None
    assert governing_entry({"openai-models": FAMILY_CAP}) == "openai-models"
    assert routed_models_of(router).members("gpt-4o") == frozenset()
    assert routed_models_of(router).members("openai-models") == frozenset({"gpt-4o", "gpt-4o-mini"})


@pytest.mark.asyncio
async def test_key_info_usage_reports_the_family_total():
    budget: Final = {"claude-opus*": ROOMY_CAP}
    limiter: Final = _family_limiter()

    await _spend_on(limiter, model="claude-opus-4-8", model_max_budget=budget)
    await _spend_on(limiter, model="claude-opus-5-5", model_max_budget=budget)

    usage: Final = await build_model_max_budget_usage(
        entity_type=Litellm_EntityType.KEY,
        entity_id=KEY_HASH,
        model_max_budget=budget,
        cache=limiter.dual_cache,
    )
    assert usage["claude-opus*"]["current_spend"] == round(CHAT_COST * 2, 4)


@pytest.mark.parametrize(
    ("pattern", "model", "governs"),
    [
        ("claude-opus*", "claude-opus-4-8", True),
        ("claude-opus*", "anthropic/claude-opus-4-8", True),
        ("anthropic/claude-opus*", "anthropic/claude-opus-4-8", True),
        ("anthropic/claude-opus*", "claude-opus-4-8", False),
        ("claude-opus*", "my-claude-opus-4-8", False),
        ("claude-opus*", "claude-sonnet-5", False),
        ("gpt-4.1*", "gpt-401-mini", False),
        ("*", "claude-sonnet-5", True),
    ],
)
def test_glob_entries_match_the_whole_model_name_with_only_star_as_a_wildcard(pattern: str, model: str, governs: bool):
    resolved: Final = resolve_model_budget(
        model=model, model_max_budget={pattern: FAMILY_CAP}, model_access_groups=_model_access_groups({})
    )

    assert (resolved is not None and resolved.budget_model == pattern) is governs


@pytest.mark.asyncio
async def test_replicas_without_redis_enforce_one_family_cap_together():
    budget: Final = {"claude-opus*": FAMILY_CAP}
    table: Final = _WindowSpendDatabase()
    replica_a: Final = _family_limiter(shared_windows=table.replica())
    replica_b: Final = _family_limiter(shared_windows=table.replica())

    await _spend_on(replica_a, model="claude-opus-4-8", model_max_budget=budget)
    table.flush()

    with pytest.raises(litellm.BudgetExceededError, match="model=claude-opus-5-5"):
        await replica_b.is_key_within_model_budget(user_api_key_dict=_key(budget), model="claude-opus-5-5")


@pytest.mark.asyncio
async def test_replicas_sharing_redis_enforce_the_family_cap_without_the_database():
    budget: Final = {"claude-opus*": FAMILY_CAP}
    redis: Final = _SharedRedisDouble()
    table: Final = _WindowSpendDatabase()
    replica_a: Final = _family_limiter(
        dual_cache=DualCache(redis_cache=redis),  # pyright: ignore[reportArgumentType]  # duck-typed Redis double
        shared_windows=table.replica(),
    )
    replica_b: Final = _family_limiter(
        dual_cache=DualCache(redis_cache=redis),  # pyright: ignore[reportArgumentType]  # duck-typed Redis double
        shared_windows=table.replica(),
    )

    await _spend_on(replica_a, model="claude-opus-4-8", model_max_budget=budget)
    await _drain_redis_pushes()
    table.flush()

    with pytest.raises(litellm.BudgetExceededError, match="model=claude-opus-5-5"):
        await replica_b.is_key_within_model_budget(user_api_key_dict=_key(budget), model="claude-opus-5-5")
    assert dict(table.rows) == {}


@pytest.mark.asyncio
async def test_a_replica_charges_the_open_shared_window_instead_of_restarting_it():
    budget: Final = {"claude-opus*": {"budget_limit": CHAT_COST * 1.5, "time_period": "1d"}}
    table: Final = _WindowSpendDatabase()
    clock: Final = _Clock()
    clock.advance(WINDOW_EPOCH)
    replica_a: Final = _family_limiter(shared_windows=table.replica(), clock=clock.now)
    replica_b: Final = _family_limiter(shared_windows=table.replica(), clock=clock.now)

    await _spend_on(replica_a, model="claude-opus-4-8", model_max_budget=budget)
    table.flush()
    clock.advance(3600)
    await _spend_on(replica_b, model="claude-opus-5-5", model_max_budget=budget)
    table.flush()

    for replica in (replica_a, replica_b):
        with pytest.raises(litellm.BudgetExceededError):
            await replica.is_key_within_model_budget(user_api_key_dict=_key(budget), model="claude-opus-5-5")


@pytest.mark.asyncio
async def test_key_info_on_one_replica_reports_spend_from_every_replica():
    budget: Final = {"claude-opus*": ROOMY_CAP}
    table: Final = _WindowSpendDatabase()
    replica_a: Final = _family_limiter(shared_windows=table.replica())
    replica_b: Final = _family_limiter(shared_windows=table.replica())

    await _spend_on(replica_a, model="claude-opus-4-8", model_max_budget=budget)
    await _spend_on(replica_b, model="claude-opus-5-5", model_max_budget=budget)
    table.flush()

    usage: Final = await build_model_max_budget_usage(
        entity_type=Litellm_EntityType.KEY,
        entity_id=KEY_HASH,
        model_max_budget=budget,
        cache=replica_a.dual_cache,
        shared_windows=table.replica(),
    )
    assert usage["claude-opus*"]["current_spend"] == round(CHAT_COST * 2, 4)
    assert usage["claude-opus*"]["budget_reset_at"] is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("budget_model", ["claude-opus*", "claude-opus-4-8"])
async def test_replicas_starting_a_window_moments_apart_keep_each_others_spend(budget_model: str):
    budget: Final = {budget_model: ROOMY_CAP}
    table: Final = _WindowSpendDatabase()
    clock: Final = _Clock()
    clock.advance(WINDOW_EPOCH)
    replica_a: Final = _family_limiter(shared_windows=table.replica(), clock=clock.now)
    replica_b: Final = _family_limiter(shared_windows=table.replica(), clock=clock.now)

    await _spend_on(replica_a, model="claude-opus-4-8", model_max_budget=budget)
    clock.advance(1)
    await _spend_on(replica_b, model="claude-opus-4-8", model_max_budget=budget)
    table.flush()

    usage: Final = await build_model_max_budget_usage(
        entity_type=Litellm_EntityType.KEY,
        entity_id=KEY_HASH,
        model_max_budget=budget,
        cache=DualCache(),
        now=datetime.fromtimestamp(clock.now(), tz=timezone.utc),
        shared_windows=table.replica(),
    )
    assert usage[budget_model]["current_spend"] == round(CHAT_COST * 2, 4)


@pytest.mark.asyncio
async def test_concurrent_first_charges_on_one_replica_both_count():
    budget: Final = {"claude-opus*": ROOMY_CAP}
    clock: Final = _Clock()
    clock.advance(WINDOW_EPOCH)
    limiter: Final = _family_limiter(shared_windows=_WindowSpendDatabase().replica(), clock=clock.now)

    await asyncio.gather(
        _spend_on(limiter, model="claude-opus-4-8", model_max_budget=budget),
        _spend_on(limiter, model="claude-opus-5-5", model_max_budget=budget),
    )

    usage: Final = await build_model_max_budget_usage(
        entity_type=Litellm_EntityType.KEY,
        entity_id=KEY_HASH,
        model_max_budget=budget,
        cache=limiter.dual_cache,
        now=datetime.fromtimestamp(clock.now(), tz=timezone.utc),
        shared_windows=_WindowSpendDatabase().replica(),
    )
    assert usage["claude-opus*"]["current_spend"] == round(CHAT_COST * 2, 4)
