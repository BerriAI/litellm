from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Final

import pytest

import litellm
from litellm.caching.redis_batch import request_redis_batch_scope
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.customer_hierarchy_auth import check_customer_hierarchy_budget
from litellm.proxy.auth.customer_spend import current_customer_spend as read_customer_spend
from litellm.proxy.auth.customer_spend import customer_spend_counter, settle_customer_spend
from litellm.proxy.auth.customer_spend_batch import cached_customer_spend_keys
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.spend_tracking.spend_counter_batch import (
    bind_admission_counter_keys,
    read_batched_spend_counter,
    release_spend_counter_batch,
    spend_counter_batch_scope,
)
from litellm.types.proxy.auth.customer_hierarchy import CustomerHierarchyAuth, CustomerScopeAuth
from tests.unit.caching.redis_batch_double import FakeNatives, fake_send
from tests.unit.redis.stores.spend_counters_double import FakeSpendCounters


def _hierarchy() -> CustomerHierarchyAuth:
    team: Final = CustomerScopeAuth(key="customer_team", entity_id="team", spend=2, max_budget=5)
    return CustomerHierarchyAuth(
        customer_org_id="tenant",
        customer_team_id="team",
        customer_member_id="member",
        scopes=(
            CustomerScopeAuth(key="customer_org", entity_id="tenant", spend=1, max_budget=5),
            team,
            CustomerScopeAuth(key="customer_member", entity_id="member", spend=3, max_budget=5),
        ),
        team_scopes=(team,),
    )


@pytest.mark.asyncio
async def test_warm_customer_headers_and_key_share_one_admission_mget() -> None:
    hierarchy: Final = _hierarchy()
    cache: Final = UserApiKeyCache()
    await cache.async_set_frozen_model("customer_member_id:member", hierarchy, 60)
    handle: Final = FakeSpendCounters(
        redis={"spend:key:hashed": 1, **{customer_spend_counter(scope): scope.spend for scope in hierarchy.scopes}}
    )
    store: Final = handle.store()

    async def read(counter: str, spend: float, maximum: float | None) -> float:
        reading: Final = await read_batched_spend_counter(counter)
        assert reading is not None and reading[0] is not None
        return reading[0]

    def customer_keys() -> frozenset[str]:
        return cached_customer_spend_keys(
            {
                "x-litellm-customer-org-alias": "tenant",
                "x-litellm-customer-team-id": "team",
                "x-litellm-customer-member-id": "member",
            },
            cache,
        )

    with spend_counter_batch_scope(store, counter_keys_provider=customer_keys):
        bind_admission_counter_keys(UserAPIKeyAuth(token="hashed"), end_user_id=None)
        assert await read_batched_spend_counter("spend:key:hashed") == (1, True)
        current: Final = await read_customer_spend(hierarchy, read)
    assert current is hierarchy
    assert len(handle.commands) == 1
    assert handle.commands[0].split()[0] == "MGET"
    assert frozenset(handle.commands[0].split()[1:]) == frozenset(
        {"spend:key:hashed", *(customer_spend_counter(scope) for scope in hierarchy.scopes)}
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("missing", (False, True))
async def test_customer_settlement_shares_seed_read_and_native_pipeline_with_key(missing: bool) -> None:
    hierarchy: Final = _hierarchy()
    handle: Final = FakeSpendCounters(
        redis={
            "spend:key:hashed": 1,
            **{
                customer_spend_counter(scope): scope.spend
                for scope in hierarchy.scopes
                if not missing or scope.key != "customer_member"
            },
        }
    )
    store: Final = handle.store()
    keys: Final = frozenset({"spend:key:hashed", *(customer_spend_counter(scope) for scope in hierarchy.scopes)})
    with request_redis_batch_scope(native_factory=FakeNatives(), native_send_factory=fake_send):
        with spend_counter_batch_scope(store, counter_keys=keys):
            await settle_customer_spend(
                {"user_api_key_customer_hierarchy": hierarchy.model_dump(mode="json")}, 0.5, store
            )
            assert await read_batched_spend_counter("spend:key:hashed") == (1, True)
            _ = store.increment_many_post_call((("spend:key:hashed", 0.5),))
        assert len(handle.post_call) == 2
        assert handle.post_call[0][0] is handle.post_call[1][0]
        assert handle.post_call[0][1] == tuple((customer_spend_counter(scope), 0.5) for scope in hierarchy.scopes)
        handle.settle_post_call()
    assert sum(command.startswith("MGET ") for command in handle.commands) == 1
    assert not any(command.startswith(("GET ", "PIPELINE ")) for command in handle.commands)
    assert sum(command.startswith("SETNX ") for command in handle.commands) == int(missing)
    assert handle.redis == {
        "spend:key:hashed": 1.5,
        **{customer_spend_counter(scope): scope.spend + 0.5 for scope in hierarchy.scopes},
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("closed", (False, True))
async def test_customer_settlement_reads_directly_when_the_batch_is_unavailable_or_closed(closed: bool) -> None:
    hierarchy: Final = _hierarchy()
    handle: Final = FakeSpendCounters(
        redis={customer_spend_counter(scope): scope.spend for scope in hierarchy.scopes[:-1]}, batch_reads=False
    )
    with spend_counter_batch_scope(handle.store()):
        if closed:
            release_spend_counter_batch()
        await settle_customer_spend(
            {"user_api_key_customer_hierarchy": hierarchy.model_dump(mode="json")}, 0.5, handle.store()
        )
    assert frozenset(command for command in handle.commands if command.startswith("GET ")) == frozenset(
        f"GET {customer_spend_counter(scope)}" for scope in hierarchy.scopes
    )
    assert sum(command.startswith("SETNX ") for command in handle.commands) == 1
    assert handle.redis == {customer_spend_counter(scope): scope.spend + 0.5 for scope in hierarchy.scopes}


@pytest.mark.asyncio
@pytest.mark.parametrize("blocked", ("customer_org", "customer_team", "customer_member"))
async def test_shared_admission_read_preserves_the_customer_budget_error(blocked: str) -> None:
    hierarchy: Final = _hierarchy()
    handle: Final = FakeSpendCounters(
        redis={customer_spend_counter(scope): 6 if scope.key == blocked else scope.spend for scope in hierarchy.scopes}
    )

    async def read(counter: str, spend: float, maximum: float | None) -> float:
        reading: Final = await read_batched_spend_counter(counter)
        assert reading is not None and reading[0] is not None
        return reading[0]

    with spend_counter_batch_scope(handle.store()):
        current: Final = await read_customer_spend(hierarchy, read)
    with pytest.raises(litellm.BudgetExceededError) as error:
        check_customer_hierarchy_budget(current)
    assert error.value.entity_type == blocked
    assert (
        error.value.entity_id
        == {"customer_org": "tenant", "customer_team": "team", "customer_member": "member"}[blocked]
    )
    assert error.value.current_cost == 6
    assert error.value.max_budget == 5


@pytest.mark.parametrize("headers", (None, {}, {"x-litellm-customer-member-id": "unknown"}))
def test_unresolved_customer_headers_do_not_speculatively_load_policies(headers: dict[str, str] | None) -> None:
    assert cached_customer_spend_keys(headers, UserApiKeyCache()) == frozenset()


@pytest.mark.asyncio
@pytest.mark.parametrize("reading", (0.0, 1.0, float("nan")))
async def test_unusable_or_lower_live_readings_do_not_discard_the_policy_spend_floor(reading: float) -> None:
    hierarchy: Final = _hierarchy()

    async def read(counter: str, spend: float, maximum: float | None) -> float:
        return reading

    assert await read_customer_spend(hierarchy, read) is hierarchy


async def current_customer_spend(hierarchy, store, prisma=None):
    from litellm.proxy import proxy_server

    with pytest.MonkeyPatch.context() as context:
        context.setattr(proxy_server, "spend_counter_store", store)
        context.setattr(proxy_server, "prisma_client", prisma)
        return await read_customer_spend(hierarchy, proxy_server.get_current_spend)


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", ("customer_org", "customer_team", "customer_member"))
async def test_a_lost_counter_reseeds_committed_spend_without_refreshing_the_customer_snapshot(scope: str) -> None:
    from tests.unit.proxy.db.test_spend_counter_reseed import _FakeFindUniqueTable

    row: Final = SimpleNamespace(spend=2.5, budget_reset_at=None)
    table: Final = _FakeFindUniqueTable(row)
    table_name: Final = {
        "customer_org": "litellm_customerorganizationtable",
        "customer_team": "litellm_customerteamtable",
        "customer_member": "litellm_customermembertable",
    }[scope]
    prisma: Final = SimpleNamespace(db=SimpleNamespace(**{table_name: table}))
    snapshot: Final = CustomerHierarchyAuth(
        customer_org_id="tenant", scopes=(CustomerScopeAuth(key=scope, entity_id="id:with:colons", max_budget=2),)
    )
    shared: Final[dict[str, object]] = {}
    first: Final = FakeSpendCounters(redis=shared).store()
    second: Final = FakeSpendCounters(redis=shared).store()
    for store in (first, second):
        loaded: Final = await current_customer_spend(snapshot, store, prisma)
        with pytest.raises(litellm.BudgetExceededError):
            check_customer_hierarchy_budget(loaded)
    assert snapshot.scopes[0].spend == 0
    assert len(table.where_clauses) == 1
    await first.delete((customer_spend_counter(snapshot.scopes[0]),))
    reseeded: Final = await current_customer_spend(snapshot, second, prisma)
    with pytest.raises(litellm.BudgetExceededError):
        check_customer_hierarchy_budget(reseeded)
    assert reseeded.scopes[0].spend == row.spend
    assert len(table.where_clauses) == 2


@pytest.mark.asyncio
async def test_a_database_snapshot_ahead_of_settlement_does_not_count_the_pending_charge_twice() -> None:
    store: Final = FakeSpendCounters().store()
    initial: Final = CustomerHierarchyAuth(
        customer_org_id="tenant", scopes=(CustomerScopeAuth(key="customer_org", entity_id="tenant"),)
    )
    _ = await current_customer_spend(initial, store)
    persisted: Final = initial.model_copy(update={"scopes": (initial.scopes[0].model_copy(update={"spend": 0.5}),)})
    _ = await current_customer_spend(persisted, store)
    await settle_customer_spend({"user_api_key_customer_hierarchy": initial.model_dump(mode="json")}, 0.5, store)
    assert (await current_customer_spend(persisted, store)).scopes[0].spend == 0.5


@pytest.mark.asyncio
async def test_member_b_settlement_blocks_cached_member_a_and_an_alias_only_request_on_another_worker() -> None:
    shared: Final[dict[str, object]] = {}
    settling: Final = FakeSpendCounters(redis=shared).store()
    reading: Final = FakeSpendCounters(redis=shared).store()
    organization: Final = CustomerScopeAuth(key="customer_org", entity_id="tenant", max_budget=1)
    member_a: Final = CustomerHierarchyAuth(
        customer_org_id="tenant",
        customer_member_id="a",
        scopes=(organization, CustomerScopeAuth(key="customer_member", entity_id="a")),
    )
    member_b: Final = CustomerHierarchyAuth(
        customer_org_id="tenant",
        customer_member_id="b",
        scopes=(organization, CustomerScopeAuth(key="customer_member", entity_id="b")),
    )
    alias_only: Final = CustomerHierarchyAuth(customer_org_id="tenant", scopes=(organization,))
    _ = await current_customer_spend(member_a, reading)
    _ = await current_customer_spend(alias_only, reading)
    await settle_customer_spend({"user_api_key_customer_hierarchy": member_b.model_dump(mode="json")}, 1.25, settling)
    for cached in (member_a, alias_only):
        current: Final = await current_customer_spend(cached, reading)
        with pytest.raises(litellm.BudgetExceededError) as rejected:
            check_customer_hierarchy_budget(current)
        assert rejected.value.entity_type == "customer_org"
        assert rejected.value.entity_id == "tenant"
    assert (await current_customer_spend(member_a, reading)).scopes[1].spend == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("redis", (True, False))
async def test_settlement_counts_once_on_each_scope_without_modifying_the_cached_snapshot(redis: bool) -> None:
    handle: Final = FakeSpendCounters(redis={} if redis else None)
    store: Final = handle.store()
    hierarchy: Final = CustomerHierarchyAuth(
        customer_org_id="tenant",
        scopes=(
            CustomerScopeAuth(key="customer_org", entity_id="tenant", spend=1, max_budget=5),
            CustomerScopeAuth(key="customer_team", entity_id="team", spend=2),
            CustomerScopeAuth(key="customer_member", entity_id="member", spend=3),
        ),
    )
    await settle_customer_spend({"user_api_key_customer_hierarchy": hierarchy.model_dump(mode="json")}, 0.5, store)
    current: Final = await current_customer_spend(hierarchy, store)
    assert tuple(scope.spend for scope in current.scopes) == (1.5, 2.5, 3.5)
    assert tuple(scope.spend for scope in hierarchy.scopes) == (1.0, 2.0, 3.0)


@pytest.mark.asyncio
async def test_budget_reset_uses_a_new_counter_even_when_an_old_request_settles_late() -> None:
    store: Final = FakeSpendCounters().store()
    old: Final = CustomerHierarchyAuth(
        customer_org_id="tenant",
        scopes=(CustomerScopeAuth(key="customer_org", entity_id="tenant", spend=2, max_budget=1),),
    )
    reset: Final = old.model_copy(
        update={
            "scopes": (
                old.scopes[0].model_copy(
                    update={"spend": 0.0, "budget_reset_at": datetime(2026, 1, 1, tzinfo=timezone.utc)}
                ),
            )
        }
    )
    await settle_customer_spend({"user_api_key_customer_hierarchy": old.model_dump(mode="json")}, 0.5, store)
    current: Final = await current_customer_spend(reset, store)
    check_customer_hierarchy_budget(current)
    assert current.scopes[0].spend == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "metadata", ({}, {"user_api_key_customer_hierarchy": {}}, {"user_api_key_customer_hierarchy": "bad"})
)
async def test_missing_or_invalid_customer_metadata_does_not_charge_a_counter(metadata: dict[str, object]) -> None:
    handle: Final = FakeSpendCounters()
    await settle_customer_spend(metadata, 1, handle.store())
    assert handle.redis == {}


@pytest.mark.asyncio
async def test_zero_cost_does_not_initialize_customer_counters() -> None:
    handle: Final = FakeSpendCounters()
    hierarchy: Final = CustomerHierarchyAuth(customer_org_id="tenant", scopes=())
    await settle_customer_spend(
        {"user_api_key_customer_hierarchy": hierarchy.model_dump(mode="json")}, 0, handle.store()
    )
    assert handle.redis == {}
