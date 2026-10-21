import asyncio
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

import pytest

import litellm.interactions.background_cost_polling as bg
from litellm import Router
from litellm.constants import BACKGROUND_INTERACTION_SETTLEMENT_CLAIM_LEASE_SECONDS
from litellm.interactions.background_cost_polling import (
    _create_context,
    BackgroundInteractionPollContext,
    configure_background_settlement_store,
    maybe_settle_background_interaction_before_delete,
    PendingBackgroundInteraction,
    poll_and_log_background_interaction_cost,
    PollSchedule,
    resume_unsettled_background_interactions,
)
from litellm.litellm_core_utils.litellm_logging import Logging as LitellmLogging
from litellm.proxy.spend_tracking.background_interaction_settlement import (
    configure_background_interaction_settlement,
    fetch_with_deployment_credentials,
    install_background_interaction_settlement,
    PrismaBackgroundSettlementStore,
)
from litellm.types.interactions import InteractionsAPIResponse

USAGE_BLOCK = {
    "total_tokens": 175,
    "total_input_tokens": 100,
    "input_tokens_by_modality": [{"modality": "text", "tokens": 100}],
    "total_cached_tokens": 0,
    "total_output_tokens": 50,
    "output_tokens_by_modality": [{"modality": "text", "tokens": 50}],
    "total_tool_use_tokens": 0,
    "total_thought_tokens": 25,
}

FAST_SCHEDULE = PollSchedule(initial_interval_seconds=0.001, max_interval_seconds=0.002, timeout_seconds=1.0)


@dataclass
class _Row:
    interaction_id: str
    custom_llm_provider: str
    create_context: object
    created_at: datetime
    claimed_at: Optional[datetime] = None
    claimed_by: Optional[str] = None
    settled_at: Optional[datetime] = None
    outcome: Optional[str] = None


class _FakeSettlementTable:
    """Just enough of prisma's per-model actions: Json is stored as the data it wraps and read back parsed."""

    def __init__(self, rows: tuple[_Row, ...] = ()):
        self.rows = {row.interaction_id: row for row in rows}

    async def create(self, *, data):
        row = _Row(
            interaction_id=data["interaction_id"],
            custom_llm_provider=data["custom_llm_provider"],
            create_context=data["create_context"].data,
            created_at=data["created_at"],
        )
        self.rows[row.interaction_id] = row
        return row

    async def find_unique(self, *, where):
        return self.rows.get(where["interaction_id"])

    async def find_many(self, *, where):
        return self._matching(where)

    async def update_many(self, *, data, where):
        matched = self._matching(where)
        for row in matched:
            for column, value in data.items():
                setattr(row, column, getattr(value, "data", value) if column == "create_context" else value)
        return len(matched)

    def _matching(self, where) -> list:
        return [row for row in self.rows.values() if _matches(row, where)]


def _matches(row: "_Row", where: dict) -> bool:
    return all(_satisfies(row, column, condition) for column, condition in where.items())


def _satisfies(row: "_Row", column: str, condition) -> bool:
    value = getattr(row, column)
    if isinstance(condition, dict):
        return value is not None and value < condition["lt"]
    return value == condition


class _FakeSpendLogs:
    """The spend log table, holding only the request ids of the rows written so far."""

    def __init__(self, request_ids: tuple[str, ...] = ()):
        self.request_ids = request_ids

    async def find_unique(self, *, where):
        return where["request_id"] if where["request_id"] in self.request_ids else None


def _store(
    table,
    claimed_by: str,
    claim_lease_seconds: float = BACKGROUND_INTERACTION_SETTLEMENT_CLAIM_LEASE_SECONDS,
) -> PrismaBackgroundSettlementStore:
    return PrismaBackgroundSettlementStore(
        table=table,
        spend_logs=_FakeSpendLogs(),
        spend_logs_kept=lambda: True,
        claimed_by=claimed_by,
        claim_lease_seconds=claim_lease_seconds,
    )


def _logging_obj(metadata: Optional[dict] = None) -> LitellmLogging:
    logging_obj = LitellmLogging(
        model="gemini-2.5-flash",
        messages=[],
        stream=False,
        call_type="acreate_interaction",
        start_time=time.time(),
        litellm_call_id="bg-settlement-call-id",
        function_id="bg-settlement-fn-id",
    )
    logging_obj.update_environment_variables(
        litellm_params={"metadata": metadata or {"user_api_key": "0123456789abcdef" * 4}},
        optional_params={},
        model="gemini-2.5-flash",
        custom_llm_provider="gemini",
        input="hi",
    )
    return logging_obj


def _pending(interaction_id: str) -> PendingBackgroundInteraction:
    return PendingBackgroundInteraction(
        interaction_id=interaction_id,
        custom_llm_provider="gemini",
        create_context=_create_context(_logging_obj(), "gemini"),
        created_at=datetime.now(timezone.utc),
    )


def _stored_row(
    interaction_id: str,
    claimed_at: Optional[datetime] = None,
    settled: bool = False,
    create_context: Optional[object] = None,
) -> _Row:
    return _Row(
        interaction_id=interaction_id,
        custom_llm_provider="gemini",
        create_context=(
            create_context
            if create_context is not None
            else _create_context(_logging_obj(), "gemini").model_dump(mode="json")
        ),
        created_at=datetime.now(timezone.utc),
        claimed_at=claimed_at,
        claimed_by="replica-a:1" if claimed_at is not None else None,
        settled_at=claimed_at if settled else None,
        outcome="billed" if settled else None,
    )


def _minutes_ago(minutes: float) -> datetime:
    return datetime.now(timezone.utc) - timedelta(minutes=minutes)


def _past_the_default_lease() -> datetime:
    return datetime.now(timezone.utc) - timedelta(seconds=BACKGROUND_INTERACTION_SETTLEMENT_CLAIM_LEASE_SECONDS + 60)


def _completed(interaction_id: str) -> InteractionsAPIResponse:
    return InteractionsAPIResponse(
        id=interaction_id, model="gemini-2.5-flash", status="completed", steps=[], usage=dict(USAGE_BLOCK)
    )


def _capturing_fetch():
    captured = []

    async def fetch(context):
        captured.append(context)
        return _completed(context.interaction_id)

    return fetch, captured


@pytest.mark.asyncio
async def test_registered_row_reads_back_as_the_same_pending_interaction():
    table = _FakeSettlementTable()
    store = _store(table, claimed_by="replica-a:1")
    pending = _pending("interactions/bg-1")

    await store.register(pending)

    assert await store.pending("interactions/bg-1") == pending
    assert await store.unsettled() == (pending,)


@pytest.mark.asyncio
async def test_claim_is_won_by_exactly_one_settler():
    table = _FakeSettlementTable()
    replica_a = _store(table, claimed_by="replica-a:1")
    replica_b = _store(table, claimed_by="replica-b:1")
    await replica_a.register(_pending("interactions/bg-1"))

    assert await replica_b.claim("interactions/bg-1") is True
    assert await replica_a.claim("interactions/bg-1") is False
    assert await replica_a.is_claimed("interactions/bg-1") is True
    assert await replica_a.pending("interactions/bg-1") is None
    assert table.rows["interactions/bg-1"].claimed_by == "replica-b:1"


class _MissingSettlementTable:
    """Prisma's per-model actions against a database whose migration for this table was held back."""

    async def create(self, *, data):
        raise self._missing()

    async def find_unique(self, *, where):
        raise self._missing()

    async def find_many(self, *, where):
        raise self._missing()

    async def update_many(self, *, data, where):
        raise self._missing()

    def _missing(self):
        from prisma.errors import TableNotFoundError

        return TableNotFoundError(
            {
                "user_facing_error": {
                    "error_code": "P2021",
                    "meta": {"table": "public.LiteLLM_BackgroundInteractionSettlement"},
                    "message": "The table does not exist in the current database.",
                }
            }
        )


@pytest.mark.asyncio
async def test_a_missing_table_holds_no_rows_and_takes_no_registration():
    from prisma.errors import TableNotFoundError

    store = _store(_MissingSettlementTable(), claimed_by="replica-a:1")

    with pytest.raises(TableNotFoundError):
        await store.register(_pending("interactions/bg-1"))
    assert await store.pending("interactions/bg-1") is None
    assert await store.is_claimed("interactions/bg-1") is False
    assert await store.claim("interactions/bg-1") is False
    with pytest.raises(TableNotFoundError):
        await store.unsettled()


@pytest.mark.asyncio
async def test_unsettled_lists_claimed_rows_with_no_outcome_and_when_their_lease_ends():
    claimed_at = _minutes_ago(1)
    table = _FakeSettlementTable(
        rows=(
            _stored_row("interactions/bg-orphaned"),
            _stored_row("interactions/bg-mid-settlement", claimed_at=claimed_at),
            _stored_row("interactions/bg-settled", claimed_at=claimed_at, settled=True),
            _stored_row("interactions/bg-from-the-future", create_context={"schema": "unknown"}),
        )
    )
    store = _store(table, claimed_by="replica-b:1", claim_lease_seconds=300)

    unsettled = await store.unsettled()

    assert [(row.interaction_id, row.claim_expires_at) for row in unsettled] == [
        ("interactions/bg-orphaned", None),
        ("interactions/bg-mid-settlement", claimed_at + timedelta(seconds=300)),
    ]


@pytest.mark.asyncio
async def test_a_claim_whose_settler_died_is_taken_over_once_its_lease_runs_out():
    """
    The regression: a settler that died between its claim and its outcome left
    the row claimed forever, so no replica, restart, or delete ever billed it.
    """
    table = _FakeSettlementTable(
        rows=(
            _stored_row("interactions/bg-dead-settler", claimed_at=_minutes_ago(10)),
            _stored_row("interactions/bg-settled", claimed_at=_minutes_ago(10), settled=True),
        )
    )
    replica_b = _store(table, claimed_by="replica-b:1", claim_lease_seconds=300)
    replica_c = _store(table, claimed_by="replica-c:1", claim_lease_seconds=300)

    assert await replica_b.is_claimed("interactions/bg-dead-settler") is False
    assert await replica_b.pending("interactions/bg-dead-settler") is not None
    assert await replica_b.claim("interactions/bg-dead-settler") is True
    assert await replica_c.claim("interactions/bg-dead-settler") is False
    assert await replica_c.is_claimed("interactions/bg-dead-settler") is True
    assert table.rows["interactions/bg-dead-settler"].claimed_by == "replica-b:1"

    assert await replica_b.is_claimed("interactions/bg-settled") is True
    assert await replica_b.pending("interactions/bg-settled") is None
    assert await replica_b.claim("interactions/bg-settled") is False
    assert table.rows["interactions/bg-settled"].claimed_by == "replica-a:1"


@pytest.mark.asyncio
async def test_record_outcome_keeps_the_audit_trail_and_drops_the_stored_request_context():
    table = _FakeSettlementTable()
    store = _store(table, claimed_by="replica-a:1")
    await store.register(_pending("interactions/bg-1"))
    assert await store.claim("interactions/bg-1")
    assert table.rows["interactions/bg-1"].create_context

    await store.record_outcome("interactions/bg-1", "billed")

    row = table.rows["interactions/bg-1"]
    assert row.outcome == "billed"
    assert row.settled_at is not None
    assert row.claimed_at <= row.settled_at
    assert row.create_context == {}


@pytest.mark.asyncio
async def test_a_takeover_records_the_bill_a_settler_wrote_before_dying_instead_of_billing_again():
    """
    The regression: a settler that billed and then died, or failed to record
    the outcome, left its claim to lapse with no outcome, and the replica that
    took the claim over billed the interaction a second time.
    """
    table = _FakeSettlementTable(
        rows=(
            _stored_row("interactions/bg-billed-then-died", claimed_at=_past_the_default_lease()),
            _stored_row("interactions/bg-died-before-billing", claimed_at=_past_the_default_lease()),
        )
    )
    fetch, _ = _capturing_fetch()
    previous_store = bg._STORE.store
    try:
        resumed = await configure_background_interaction_settlement(
            table=table,
            spend_logs=_FakeSpendLogs(request_ids=("interactions/bg-billed-then-died",)),
            spend_logs_kept=lambda: True,
            claimed_by="replica-b:1",
            fetch_interaction=fetch,
            schedule=FAST_SCHEDULE,
        )

        assert [await asyncio.wait_for(task, timeout=5) for task in resumed] == [None, "billed"]
        assert [(row.claimed_by, row.outcome) for row in table.rows.values()] == [
            ("replica-b:1", "billed"),
            ("replica-b:1", "billed"),
        ]
        assert await bg._STORE.store.unsettled() == ()
    finally:
        configure_background_settlement_store(previous_store)


@pytest.mark.asyncio
async def test_with_spend_logs_off_a_takeover_leaves_the_interaction_unbilled_instead_of_billing_it_twice():
    """
    The regression: with spend logs off a settler that billed and then died
    leaves no spend row to find, so the replica that took over its lapsed
    claim billed the interaction a second time.
    """
    table = _FakeSettlementTable(
        rows=(
            _stored_row("interactions/bg-claimed-then-died", claimed_at=_past_the_default_lease()),
            _stored_row("interactions/bg-orphaned"),
        )
    )
    fetch, _ = _capturing_fetch()
    previous_store = bg._STORE.store
    try:
        resumed = await configure_background_interaction_settlement(
            table=table,
            spend_logs=_FakeSpendLogs(),
            spend_logs_kept=lambda: False,
            claimed_by="replica-b:1",
            fetch_interaction=fetch,
            schedule=FAST_SCHEDULE,
        )

        assert [await asyncio.wait_for(task, timeout=5) for task in resumed] == [None, "billed"]
        assert [(row.claimed_by, row.outcome) for row in table.rows.values()] == [
            ("replica-b:1", "unsettled"),
            ("replica-b:1", "billed"),
        ]
    finally:
        configure_background_settlement_store(previous_store)


class _SpendLogsWhoseFirstReadFails(_FakeSpendLogs):
    def __init__(self):
        super().__init__()
        self.reads = 0

    async def find_unique(self, *, where):
        self.reads += 1
        if self.reads == 1:
            raise RuntimeError("read replica unavailable")
        return await super().find_unique(where=where)


@pytest.mark.asyncio
async def test_a_takeover_whose_spend_log_read_failed_retries_and_bills_instead_of_stranding_the_row():
    """
    The regression: a takeover wrote its claim before reading the spend log,
    so a failed read left the row under a fresh claim of its own, which the
    poll then took for another settler's and left unbilled until a reboot.
    """
    table = _FakeSettlementTable(
        rows=(_stored_row("interactions/bg-dead-settler", claimed_at=_past_the_default_lease()),)
    )
    fetch, _ = _capturing_fetch()
    previous_store = bg._STORE.store
    try:
        (resumed,) = await configure_background_interaction_settlement(
            table=table,
            spend_logs=_SpendLogsWhoseFirstReadFails(),
            spend_logs_kept=lambda: True,
            claimed_by="replica-b:1",
            fetch_interaction=fetch,
            schedule=FAST_SCHEDULE,
        )

        assert await asyncio.wait_for(resumed, timeout=5) == "billed"
        assert table.rows["interactions/bg-dead-settler"].outcome == "billed"
    finally:
        configure_background_settlement_store(previous_store)


def _logging_obj_whose_bill_takes(seconds: float, bills: list) -> LitellmLogging:
    logging_obj = _logging_obj()

    async def bill(result):
        bills.append(result.id)
        await asyncio.sleep(seconds)

    logging_obj.async_log_background_interaction_completion = bill
    return logging_obj


@pytest.mark.asyncio
async def test_a_settle_that_outlasts_the_lease_is_not_taken_over_and_billed_twice():
    """
    The regression: a success callback that held the settle past the claim
    lease let a booting replica take the live claim over and bill the same
    interaction a second time.
    """
    lease_seconds = 0.6
    table = _FakeSettlementTable()
    creator = _store(table, claimed_by="replica-a:1", claim_lease_seconds=lease_seconds)
    resumer = _store(table, claimed_by="replica-b:1", claim_lease_seconds=lease_seconds)
    await creator.register(_pending("interactions/bg-slow-settle"))
    fetch, _ = _capturing_fetch()
    bills = []

    def poll(store, resumed: bool, start_delay_seconds: float):
        return poll_and_log_background_interaction_cost(
            BackgroundInteractionPollContext(
                interaction_id="interactions/bg-slow-settle",
                custom_llm_provider="gemini",
                logging_obj=_logging_obj_whose_bill_takes(lease_seconds * 2.5, bills),
                initial_interval_seconds=0.001,
                max_interval_seconds=0.002,
                timeout_seconds=lease_seconds * 3,
                start_delay_seconds=start_delay_seconds,
                claim_renewal_interval_seconds=lease_seconds / 10,
                store=store,
                resumed=resumed,
            ),
            fetch,
        )

    outcomes = await asyncio.gather(
        poll(creator, resumed=False, start_delay_seconds=0.0),
        poll(resumer, resumed=True, start_delay_seconds=lease_seconds * 1.5),
    )

    assert outcomes == ["billed", None]
    assert bills == ["interactions/bg-slow-settle"]
    assert table.rows["interactions/bg-slow-settle"].claimed_by == "replica-a:1"


@pytest.mark.asyncio
async def test_configure_installs_the_store_and_resumes_the_orphaned_rows():
    table = _FakeSettlementTable(
        rows=(
            _stored_row("interactions/bg-orphaned"),
            _stored_row("interactions/bg-dead-settler", claimed_at=_past_the_default_lease()),
            _stored_row("interactions/bg-settled", claimed_at=_minutes_ago(1), settled=True),
        )
    )
    fetch, captured = _capturing_fetch()
    previous_store = bg._STORE.store
    try:
        resumed = await configure_background_interaction_settlement(
            table=table,
            spend_logs=_FakeSpendLogs(),
            spend_logs_kept=lambda: True,
            claimed_by="replica-b:1",
            fetch_interaction=fetch,
            schedule=FAST_SCHEDULE,
        )

        assert [await asyncio.wait_for(task, timeout=5) for task in resumed] == ["billed", "billed"]
        assert sorted(context.interaction_id for context in captured) == [
            "interactions/bg-dead-settler",
            "interactions/bg-orphaned",
        ]
        for interaction_id in ("interactions/bg-orphaned", "interactions/bg-dead-settler"):
            assert table.rows[interaction_id].claimed_by == "replica-b:1"
            assert table.rows[interaction_id].outcome == "billed"
        assert table.rows["interactions/bg-settled"].claimed_by == "replica-a:1"

        await table.create(
            data={
                "interaction_id": "interactions/bg-created-elsewhere",
                "custom_llm_provider": "gemini",
                "create_context": _JsonLike(_create_context(_logging_obj(), "gemini").model_dump(mode="json")),
                "created_at": datetime.now(timezone.utc),
            }
        )
        outcome = await maybe_settle_background_interaction_before_delete(
            interaction_id="interactions/bg-created-elsewhere", delete_kwargs={}, fetch_interaction=fetch
        )

        assert outcome == "billed"
        assert table.rows["interactions/bg-created-elsewhere"].claimed_by == "replica-b:1"
    finally:
        configure_background_settlement_store(previous_store)


class _PrismaClientWithoutSettlementTable:
    pass


@pytest.mark.asyncio
async def test_install_keeps_booting_when_the_settlement_table_is_unreachable():
    previous_store = bg._STORE.store

    await install_background_interaction_settlement(
        _PrismaClientWithoutSettlementTable(), router=lambda: None, spend_logs_kept=lambda: True
    )

    assert bg._STORE.store is previous_store


@dataclass(frozen=True)
class _JsonLike:
    data: object


@pytest.mark.asyncio
async def test_a_resumed_row_still_under_a_live_claim_waits_for_the_lease_before_settling():
    claimed_at = datetime.now(timezone.utc)
    table = _FakeSettlementTable(rows=(_stored_row("interactions/bg-mid-settlement", claimed_at=claimed_at),))
    store = _store(table, claimed_by="replica-b:1", claim_lease_seconds=0.2)
    fetched_at = []

    async def fetch(context):
        fetched_at.append(datetime.now(timezone.utc))
        return _completed(context.interaction_id)

    (resumed,) = await resume_unsettled_background_interactions(store, fetch, schedule=FAST_SCHEDULE)

    assert await asyncio.wait_for(resumed, timeout=5) == "billed"
    assert fetched_at[0] >= claimed_at + timedelta(seconds=0.2)
    assert table.rows["interactions/bg-mid-settlement"].claimed_by == "replica-b:1"


def _router_with_a_deployment_scoped_key() -> Router:
    return Router(
        model_list=[
            {
                "model_name": "gemini-background",
                "litellm_params": {
                    "model": "gemini/gemini-2.5-flash",
                    "api_key": "deployment-scoped-key",
                    "api_base": "https://deployment-scoped.example",
                },
                "model_info": {"id": "deployment-1"},
            }
        ]
    )


def _router_with_a_catch_all_route() -> Router:
    return Router(model_list=[{"model_name": "*", "litellm_params": {"model": "gemini/*", "api_key": "catch-all-key"}}])


def _router_with_two_deployments_in_the_group() -> Router:
    return Router(
        model_list=[
            {
                "model_name": "gemini-background",
                "litellm_params": {"model": "gemini/gemini-2.5-flash", "api_key": f"deployment-key-{index}"},
                "model_info": {"id": f"deployment-{index}"},
            }
            for index in (2, 3)
        ]
    )


def _router_with_a_vertex_deployment() -> Router:
    return Router(
        model_list=[
            {
                "model_name": "vertex-background",
                "litellm_params": {
                    "model": "vertex_ai/gemini-2.5-flash",
                    "vertex_project": "deployment-project",
                    "vertex_location": "europe-west4",
                    "vertex_credentials": "/secrets/deployment-service-account.json",
                },
                "model_info": {"id": "vertex-deployment-1"},
            }
        ]
    )


def _create_context_served_by(deployment_id: str) -> dict:
    metadata = {
        "user_api_key": "0123456789abcdef" * 4,
        "model_info": {"id": deployment_id},
        "model_group": "gemini-background",
    }
    return _create_context(_logging_obj(metadata), "gemini").model_dump(mode="json")


@pytest.mark.parametrize("deployment_id", ["deployment-1", "deployment-1-before-an-edit"])
@pytest.mark.asyncio
async def test_a_resumed_poll_fetches_with_the_credentials_of_the_deployment_that_served_the_create(deployment_id):
    """
    The regression: a resumed poll has no request to take credentials from, so
    it fetched with the environment's key, which a deployment-scoped key is
    not, and every fetch failed until the row was given up unbilled. Editing
    the deployment since gives it a new id, and its model group still names it.
    """
    table = _FakeSettlementTable(
        rows=(_stored_row("interactions/bg-1", create_context=_create_context_served_by(deployment_id)),)
    )
    router = _router_with_a_deployment_scoped_key()
    fetch, captured = _capturing_fetch()
    previous_store = bg._STORE.store
    try:
        (resumed,) = await configure_background_interaction_settlement(
            table=table,
            spend_logs=_FakeSpendLogs(),
            spend_logs_kept=lambda: True,
            claimed_by="replica-b:1",
            fetch_interaction=fetch_with_deployment_credentials(lambda: router, fetch),
            schedule=FAST_SCHEDULE,
        )

        assert await asyncio.wait_for(resumed, timeout=5) == "billed"
        assert (captured[0].credentials.api_key, captured[0].credentials.api_base) == (
            "deployment-scoped-key",
            "https://deployment-scoped.example",
        )
    finally:
        configure_background_settlement_store(previous_store)


@pytest.mark.parametrize(
    "router, deployment_id, model_group",
    [
        (_router_with_a_deployment_scoped_key, "deployment-removed-since", None),
        (_router_with_a_deployment_scoped_key, None, None),
        (lambda: None, "deployment-1", "gemini-background"),
        (_router_with_a_catch_all_route, "deployment-removed-since", None),
        (_router_with_two_deployments_in_the_group, "deployment-removed-since", "gemini-background"),
        (_router_with_a_vertex_deployment, "deployment-removed-since", "vertex-background"),
        (_router_with_a_vertex_deployment, "vertex-deployment-1", None),
    ],
)
@pytest.mark.asyncio
async def test_a_resumed_poll_without_a_known_deployment_fetches_with_the_environment_credentials(
    router, deployment_id, model_group
):
    live_router = router()
    fetch, captured = _capturing_fetch()
    context = BackgroundInteractionPollContext(
        interaction_id="interactions/bg-1",
        custom_llm_provider="gemini",
        logging_obj=_logging_obj(),
        deployment_id=deployment_id,
        model_group=model_group,
    )

    await fetch_with_deployment_credentials(lambda: live_router, fetch)(context)

    assert captured == [context]


@pytest.mark.asyncio
async def test_a_resumed_vertex_poll_fetches_with_the_project_location_and_credentials_of_its_deployment():
    router = _router_with_a_vertex_deployment()
    fetch, captured = _capturing_fetch()
    context = BackgroundInteractionPollContext(
        interaction_id="interactions/bg-1",
        custom_llm_provider="vertex_ai",
        logging_obj=_logging_obj(),
        deployment_id="vertex-deployment-1",
    )

    await fetch_with_deployment_credentials(lambda: router, fetch)(context)

    credentials = captured[0].credentials
    assert (credentials.vertex_project, credentials.vertex_location, credentials.vertex_credentials) == (
        "deployment-project",
        "europe-west4",
        "/secrets/deployment-service-account.json",
    )
