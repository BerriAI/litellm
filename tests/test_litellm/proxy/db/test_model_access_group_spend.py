"""Spend accumulation for model access group budgets."""

import asyncio
from collections.abc import Mapping, Sequence

import pytest

from litellm.litellm_core_utils.internal_call_metadata import MODEL_ACCESS_GROUP_METADATA_KEY
from litellm.proxy._types import DBSpendUpdateTransactions, Litellm_EntityType, SpendUpdateQueueItem
from litellm.proxy.db.db_spend_update_writer import DBSpendUpdateWriter, debitable_model_access_groups
from litellm.proxy.db.db_transaction_queue.daily_spend_update_queue import DailySpendUpdateQueue
from litellm.proxy.db.db_transaction_queue.redis_update_buffer import RedisUpdateBuffer
from litellm.proxy.db.db_transaction_queue.spend_update_queue import SpendUpdateQueue
from litellm.proxy.spend_tracking.spend_tracking_utils import get_request_model_access_groups


class _FakeRouter:
    """Deployment lookup returning the access groups each deployment declares."""

    def __init__(self, deployments: Mapping[str, Sequence[str] | None]) -> None:
        self._deployments = deployments

    def get_model_info(self, id: str) -> dict | None:
        if id not in self._deployments:
            return None
        declared = self._deployments[id]
        model_info: dict = {"id": id}
        if declared is not None:
            model_info["access_groups"] = list(declared)
        return {"model_name": "some-model", "model_info": model_info}


class _FakeBatchTable:
    def __init__(self) -> None:
        self.calls: list[tuple[dict, dict]] = []

    def update_many(self, where: dict, data: dict) -> None:
        self.calls.append((where, data))


class _FakeBatcher:
    def __init__(self) -> None:
        self.tables: dict[str, _FakeBatchTable] = {}

    def __getattr__(self, name: str) -> _FakeBatchTable:
        return self.tables.setdefault(name, _FakeBatchTable())


class _FakeBatchManager:
    def __init__(self, batcher: _FakeBatcher) -> None:
        self._batcher = batcher

    async def __aenter__(self) -> _FakeBatcher:
        return self._batcher

    async def __aexit__(self, *exc_info: object) -> bool:
        return False


class _FakeTransaction:
    def __init__(self, batcher: _FakeBatcher) -> None:
        self._batcher = batcher

    def batch_(self) -> _FakeBatchManager:
        return _FakeBatchManager(self._batcher)

    async def __aenter__(self) -> "_FakeTransaction":
        return self

    async def __aexit__(self, *exc_info: object) -> bool:
        return False


class _FakeDb:
    def __init__(self, batcher: _FakeBatcher) -> None:
        self._batcher = batcher

    def tx(self, timeout: object = None) -> _FakeTransaction:
        return _FakeTransaction(self._batcher)


class _FakePrismaClient:
    def __init__(self) -> None:
        self.batcher = _FakeBatcher()
        self.db = _FakeDb(self.batcher)


def _empty_transactions(**overrides: dict[str, float]) -> DBSpendUpdateTransactions:
    return DBSpendUpdateTransactions(
        user_list_transactions=overrides.get("user_list_transactions", {}),
        end_user_list_transactions=overrides.get("end_user_list_transactions", {}),
        key_list_transactions=overrides.get("key_list_transactions", {}),
        team_list_transactions=overrides.get("team_list_transactions", {}),
        team_member_list_transactions=overrides.get("team_member_list_transactions", {}),
        org_list_transactions=overrides.get("org_list_transactions", {}),
        tag_list_transactions=overrides.get("tag_list_transactions", {}),
        agent_list_transactions=overrides.get("agent_list_transactions", {}),
        model_access_group_list_transactions=overrides.get("model_access_group_list_transactions", {}),
    )


async def _drain(queue: SpendUpdateQueue) -> list[SpendUpdateQueueItem]:
    return await queue.flush_all_updates_from_in_memory_queue()


# --- enqueue ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_single_matched_group_enqueues_one_item_with_full_cost():
    writer = DBSpendUpdateWriter()

    await writer._update_model_access_group_db(
        response_cost=0.42,
        request_model_access_groups=["premium-pool"],
        served_model_id="deployment-1",
        prisma_client=object(),
        router=_FakeRouter({"deployment-1": ["premium-pool"]}),
    )

    updates = await _drain(writer.spend_update_queue)
    assert updates == [
        SpendUpdateQueueItem(
            entity_type=Litellm_EntityType.MODEL_ACCESS_GROUP,
            entity_id="premium-pool",
            response_cost=0.42,
        )
    ]


@pytest.mark.asyncio
async def test_every_matched_group_is_charged_the_full_cost_not_a_split():
    writer = DBSpendUpdateWriter()

    await writer._update_model_access_group_db(
        response_cost=0.30,
        request_model_access_groups=["pool-a", "pool-b", "pool-c"],
        served_model_id="deployment-1",
        prisma_client=object(),
        router=_FakeRouter({"deployment-1": ["pool-a", "pool-b", "pool-c"]}),
    )

    updates = await _drain(writer.spend_update_queue)
    assert [update["entity_id"] for update in updates] == ["pool-a", "pool-b", "pool-c"]
    assert [update["response_cost"] for update in updates] == [0.30, 0.30, 0.30]
    assert {update["entity_type"] for update in updates} == {Litellm_EntityType.MODEL_ACCESS_GROUP}


@pytest.mark.parametrize("attributed", [None, [], ()])
@pytest.mark.asyncio
async def test_no_attributed_groups_enqueues_nothing(attributed):
    writer = DBSpendUpdateWriter()

    await writer._update_model_access_group_db(
        response_cost=1.0,
        request_model_access_groups=attributed,
        served_model_id="deployment-1",
        prisma_client=object(),
        router=_FakeRouter({"deployment-1": ["premium-pool"]}),
    )

    assert await _drain(writer.spend_update_queue) == []


@pytest.mark.asyncio
async def test_no_prisma_client_enqueues_nothing():
    writer = DBSpendUpdateWriter()

    await writer._update_model_access_group_db(
        response_cost=1.0,
        request_model_access_groups=["premium-pool"],
        served_model_id="deployment-1",
        prisma_client=None,
        router=_FakeRouter({"deployment-1": ["premium-pool"]}),
    )

    assert await _drain(writer.spend_update_queue) == []


@pytest.mark.asyncio
async def test_group_outside_the_attributed_set_is_never_debited():
    """The served deployment also sits in a pool auth never attributed; that pool stays untouched."""
    writer = DBSpendUpdateWriter()

    await writer._update_model_access_group_db(
        response_cost=0.10,
        request_model_access_groups=["premium-pool"],
        served_model_id="deployment-1",
        prisma_client=object(),
        router=_FakeRouter({"deployment-1": ["premium-pool", "unattributed-pool"]}),
    )

    updates = await _drain(writer.spend_update_queue)
    assert [update["entity_id"] for update in updates] == ["premium-pool"]


# --- fallback guard --------------------------------------------------------


def test_fallback_to_a_model_in_another_pool_debits_nothing():
    assert (
        debitable_model_access_groups(
            attributed=["premium-pool"],
            served_model_id="fallback-deployment",
            router=_FakeRouter({"fallback-deployment": ["cheap-pool"]}),
        )
        == ()
    )


def test_fallback_to_a_model_in_no_pool_debits_nothing():
    assert (
        debitable_model_access_groups(
            attributed=["premium-pool"],
            served_model_id="fallback-deployment",
            router=_FakeRouter({"fallback-deployment": None}),
        )
        == ()
    )


def test_attributed_set_stands_when_the_served_deployment_is_unknown():
    assert debitable_model_access_groups(
        attributed=["premium-pool"],
        served_model_id="not-in-router",
        router=_FakeRouter({"deployment-1": ["premium-pool"]}),
    ) == ("premium-pool",)


def test_attributed_set_stands_without_a_router():
    assert debitable_model_access_groups(
        attributed=["premium-pool", "premium-pool"],
        served_model_id="deployment-1",
        router=None,
    ) == ("premium-pool",)


def test_partial_overlap_keeps_only_the_intersection():
    assert debitable_model_access_groups(
        attributed=["pool-a", "pool-b"],
        served_model_id="deployment-1",
        router=_FakeRouter({"deployment-1": ["pool-b", "pool-c"]}),
    ) == ("pool-b",)


def test_only_real_group_names_ever_become_entity_ids():
    """Whatever shape the attributed set arrives in, an empty or non-string name never reaches the queue."""
    assert debitable_model_access_groups(
        attributed=["pool-a", "", "pool-a", None, 7],
        served_model_id=None,
        router=None,
    ) == ("pool-a",)


# --- metadata extraction ---------------------------------------------------


def test_access_groups_read_from_request_metadata():
    kwargs = {"litellm_params": {"metadata": {MODEL_ACCESS_GROUP_METADATA_KEY: ["pool-a", "pool-b", "pool-a"]}}}
    assert get_request_model_access_groups(kwargs) == ("pool-a", "pool-b")


def test_access_groups_read_from_litellm_metadata():
    kwargs = {"litellm_params": {"litellm_metadata": {MODEL_ACCESS_GROUP_METADATA_KEY: ["pool-a"]}}}
    assert get_request_model_access_groups(kwargs) == ("pool-a",)


def test_standard_logging_payload_wins_over_metadata():
    kwargs = {
        "litellm_params": {"metadata": {MODEL_ACCESS_GROUP_METADATA_KEY: ["from-metadata"]}},
        "standard_logging_object": {"request_model_access_groups": ["from-payload"]},
    }
    assert get_request_model_access_groups(kwargs) == ("from-payload",)


def test_metadata_is_used_when_the_logging_payload_carries_no_groups():
    kwargs = {
        "litellm_params": {"metadata": {MODEL_ACCESS_GROUP_METADATA_KEY: ["from-metadata"]}},
        "standard_logging_object": {"request_model_access_groups": []},
    }
    assert get_request_model_access_groups(kwargs) == ("from-metadata",)


@pytest.mark.parametrize("stamped", ["pool-a", 7, {"pool-a": 1}])
def test_non_list_access_group_metadata_is_ignored(stamped):
    kwargs = {"litellm_params": {"metadata": {MODEL_ACCESS_GROUP_METADATA_KEY: stamped}}}
    assert get_request_model_access_groups(kwargs) == ()


def test_key_absent_from_metadata_yields_no_groups():
    """The chat path only stamps the key when something matched, so absent must mean nothing to debit."""
    kwargs = {"litellm_params": {"metadata": {"user_api_key_user_id": "u-1"}}}
    assert get_request_model_access_groups(kwargs) == ()


def test_explicit_none_yields_no_groups():
    """The pass-through path stamps the key unconditionally, so it can be present and None."""
    kwargs = {"litellm_params": {"metadata": {MODEL_ACCESS_GROUP_METADATA_KEY: None}}}
    assert get_request_model_access_groups(kwargs) == ()


@pytest.mark.parametrize(
    "metadata",
    [
        {"user_api_key_user_id": "u-1"},
        {MODEL_ACCESS_GROUP_METADATA_KEY: None},
    ],
    ids=["key-absent", "key-present-but-none"],
)
@pytest.mark.asyncio
async def test_neither_absent_nor_none_metadata_debits_anything(metadata):
    writer = DBSpendUpdateWriter()

    await writer._update_model_access_group_db(
        response_cost=0.5,
        request_model_access_groups=get_request_model_access_groups({"litellm_params": {"metadata": metadata}}),
        served_model_id="deployment-1",
        prisma_client=object(),
        router=_FakeRouter({"deployment-1": ["premium-pool"]}),
    )

    assert await _drain(writer.spend_update_queue) == []


def test_detached_sub_call_falls_back_to_the_auth_object():
    """Sub-calls inherit only the identity keys, so the groups come off user_api_key_auth there."""

    class _Auth:
        matched_model_access_groups = ["premium-pool"]

    kwargs = {"litellm_params": {"metadata": {"user_api_key_auth": _Auth()}}}
    assert get_request_model_access_groups(kwargs) == ("premium-pool",)


def test_stamped_metadata_wins_over_the_auth_object():
    class _Auth:
        matched_model_access_groups = ["stale-pool"]

    kwargs = {
        "litellm_params": {
            "metadata": {
                MODEL_ACCESS_GROUP_METADATA_KEY: ["fresh-pool"],
                "user_api_key_auth": _Auth(),
            }
        }
    }
    assert get_request_model_access_groups(kwargs) == ("fresh-pool",)


def test_auth_object_without_matched_groups_yields_no_groups():
    class _Auth:
        matched_model_access_groups = None

    kwargs = {"litellm_params": {"metadata": {"user_api_key_auth": _Auth()}}}
    assert get_request_model_access_groups(kwargs) == ()


def test_non_string_entries_are_dropped():
    kwargs = {"litellm_params": {"metadata": {MODEL_ACCESS_GROUP_METADATA_KEY: ["pool-a", None, "", 3]}}}
    assert get_request_model_access_groups(kwargs) == ("pool-a",)


def test_missing_metadata_yields_no_groups():
    assert get_request_model_access_groups(None) == ()
    assert get_request_model_access_groups({}) == ()
    assert get_request_model_access_groups({"litellm_params": {}}) == ()


# --- queue bucketing and redis round trip ----------------------------------


def test_access_group_updates_aggregate_into_their_own_bucket():
    queue = SpendUpdateQueue()

    transactions = queue.get_aggregated_db_spend_update_transactions(
        [
            SpendUpdateQueueItem(
                entity_type=Litellm_EntityType.MODEL_ACCESS_GROUP, entity_id="pool-a", response_cost=0.1
            ),
            SpendUpdateQueueItem(
                entity_type=Litellm_EntityType.MODEL_ACCESS_GROUP, entity_id="pool-a", response_cost=0.2
            ),
            SpendUpdateQueueItem(
                entity_type=Litellm_EntityType.MODEL_ACCESS_GROUP, entity_id="pool-b", response_cost=0.5
            ),
            SpendUpdateQueueItem(entity_type=Litellm_EntityType.TAG, entity_id="pool-a", response_cost=9.0),
        ]
    )

    assert transactions["model_access_group_list_transactions"] == {"pool-a": pytest.approx(0.3), "pool-b": 0.5}
    assert transactions["tag_list_transactions"] == {"pool-a": 9.0}


def test_access_group_transactions_survive_the_redis_buffer_merge():
    merged = RedisUpdateBuffer._combine_list_of_transactions(
        [
            _empty_transactions(model_access_group_list_transactions={"pool-a": 0.25}),
            _empty_transactions(model_access_group_list_transactions={"pool-a": 0.25, "pool-b": 1.0}),
        ]
    )

    assert merged["model_access_group_list_transactions"] == {"pool-a": 0.5, "pool-b": 1.0}


@pytest.mark.asyncio
async def test_redis_buffer_requeues_access_group_transactions_as_queue_items():
    queue = SpendUpdateQueue()
    daily_queue = DailySpendUpdateQueue()

    await RedisUpdateBuffer._restore_spend_updates_to_in_memory_queues(
        db_spend_update_transactions=_empty_transactions(model_access_group_list_transactions={"pool-a": 0.75}),
        daily_spend_update_transactions=None,
        daily_team_spend_update_transactions=None,
        daily_org_spend_update_transactions=None,
        daily_end_user_spend_update_transactions=None,
        daily_agent_spend_update_transactions=None,
        window_spend_update_transactions=None,
        spend_update_queue=queue,
        daily_spend_update_queue=daily_queue,
        daily_team_spend_update_queue=daily_queue,
        daily_org_spend_update_queue=daily_queue,
        daily_end_user_spend_update_queue=daily_queue,
        daily_agent_spend_update_queue=daily_queue,
        window_spend_update_queue=None,
    )

    updates = await _drain(queue)
    assert updates == [
        SpendUpdateQueueItem(
            entity_type=Litellm_EntityType.MODEL_ACCESS_GROUP,
            entity_id="pool-a",
            response_cost=0.75,
        )
    ]


# --- flush to postgres -----------------------------------------------------


@pytest.mark.asyncio
async def test_commit_increments_spend_on_the_model_access_group_budget_table():
    prisma_client = _FakePrismaClient()

    await DBSpendUpdateWriter()._commit_spend_updates_to_db(
        prisma_client=prisma_client,
        n_retry_times=0,
        proxy_logging_obj=None,
        db_spend_update_transactions=_empty_transactions(
            model_access_group_list_transactions={"pool-b": 0.5, "pool-a": 0.25}
        ),
    )

    assert prisma_client.batcher.tables["litellm_modelaccessgroupbudgettable"].calls == [
        ({"access_group_name": "pool-a"}, {"spend": {"increment": 0.25}}),
        ({"access_group_name": "pool-b"}, {"spend": {"increment": 0.5}}),
    ]
    assert "litellm_tagtable" not in prisma_client.batcher.tables


# --- end-to-end through the batched fan-out --------------------------------


@pytest.mark.asyncio
async def test_batch_database_updates_enqueues_access_group_spend():
    writer = DBSpendUpdateWriter()

    await writer._batch_database_updates(
        response_cost=0.15,
        user_id=None,
        hashed_token=None,
        team_id=None,
        org_id=None,
        end_user_id=None,
        prisma_client=object(),
        litellm_proxy_budget_name=None,
        payload={"model_id": "deployment-1", "spend": 0.15},
        request_model_access_groups=("pool-a", "pool-b"),
    )
    await asyncio.sleep(0)

    access_group_updates = [
        update
        for update in await _drain(writer.spend_update_queue)
        if update["entity_type"] is Litellm_EntityType.MODEL_ACCESS_GROUP
    ]
    assert [(update["entity_id"], update["response_cost"]) for update in access_group_updates] == [
        ("pool-a", 0.15),
        ("pool-b", 0.15),
    ]


@pytest.mark.asyncio
async def test_batch_database_updates_enqueues_nothing_without_access_groups():
    writer = DBSpendUpdateWriter()

    await writer._batch_database_updates(
        response_cost=0.15,
        user_id=None,
        hashed_token=None,
        team_id=None,
        org_id=None,
        end_user_id=None,
        prisma_client=object(),
        litellm_proxy_budget_name=None,
        payload={"model_id": "deployment-1", "spend": 0.15},
    )
    await asyncio.sleep(0)

    updates = await _drain(writer.spend_update_queue)
    assert [update for update in updates if update["entity_type"] is Litellm_EntityType.MODEL_ACCESS_GROUP] == []
