import os
from collections.abc import AsyncIterator, Iterator, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Final
from urllib.parse import urlsplit, urlunsplit

import psycopg
import pytest
import pytest_asyncio
from prisma import Prisma
from psycopg.types.json import Jsonb

from litellm.models.task_insights import (
    UNCLASSIFIED_TASK_ID,
    UNKNOWN_TASK_ID,
    Decision,
    DiscoveredTask,
    DraftTaskEdit,
    InsightsWindow,
    MapPlacement,
)
from litellm.repositories.accounting_delivery_repository import AccountingDeliveryRepository
from litellm.repositories.task_insights_demo_repository import TaskInsightsDemoRepository
from litellm.repositories.task_insights_repository import TaskInsightsRepository
from litellm.task_insights.demo_content import DEMO_PREFIX
from litellm.task_insights.demo_seed import ForeignTaxonomyError, demo_requests, seed_demo
from litellm.task_insights.observation import observe_applied_deliveries
from tests.integration._support.database import scratch_database

_MIGRATIONS: Final = Path(__file__).resolve().parents[3] / "litellm-proxy-extras/litellm_proxy_extras/migrations"
_NOW: Final = datetime(2026, 11, 11, 12, 0, tzinfo=timezone.utc)
_WINDOW: Final = InsightsWindow(
    start=datetime(2026, 11, 1, tzinfo=timezone.utc), end=datetime(2026, 11, 12, tzinfo=timezone.utc)
)


@pytest.fixture
def insights_database() -> Iterator[psycopg.Connection[tuple[object, ...]]]:
    with scratch_database("task_insights") as url, psycopg.connect(url, autocommit=True) as connection:
        connection.execute((_MIGRATIONS / "20261027000000_add_accounting_ledger_tables/migration.sql").read_text())
        connection.execute("""CREATE TABLE "LiteLLM_SpendLogs" (
            request_id text PRIMARY KEY, call_type text NOT NULL, api_key text NOT NULL DEFAULT '',
            spend float8 NOT NULL DEFAULT 0, total_tokens integer NOT NULL DEFAULT 0,
            prompt_tokens integer NOT NULL DEFAULT 0, completion_tokens integer NOT NULL DEFAULT 0,
            "startTime" timestamp(3) NOT NULL, "endTime" timestamp(3), request_duration_ms integer,
            model text NOT NULL DEFAULT '', model_id text, model_group text, custom_llm_provider text, team_id text,
            session_id text, status text, metadata jsonb DEFAULT '{}', request_tags jsonb DEFAULT '[]',
            proxy_server_request jsonb DEFAULT '{}'
        )""")
        connection.execute("""CREATE TABLE "LiteLLM_TeamTable" (
            team_id text PRIMARY KEY, team_alias text, admins text[], members text[], models text[],
            metadata jsonb NOT NULL DEFAULT '{}'
        )""")
        connection.execute((_MIGRATIONS / "20261103010000_spendlogs_accounting_delivery/migration.sql").read_text())
        connection.execute((_MIGRATIONS / "20261111000000_add_task_insights/migration.sql").read_text())
        yield connection


@pytest_asyncio.fixture(loop_scope="function")
async def insights_client(insights_database: psycopg.Connection[tuple[object, ...]]) -> AsyncIterator[Prisma]:
    url: Final = urlunsplit(urlsplit(os.environ["DATABASE_URL"])._replace(path=f"/{insights_database.info.dbname}"))
    client: Final = Prisma(datasource={"url": url})
    await client.connect()
    try:
        yield client
    finally:
        await client.disconnect()


def _log(
    connection: psycopg.Connection[tuple[object, ...]],
    request_id: str,
    *,
    model: str = "claude-sonnet",
    model_id: str | None = "config-sonnet",
    messages: list[Mapping[str, object]] | None = None,
    session_id: str | None = None,
) -> None:
    body: Final = {"messages": messages} if messages is not None else {}
    connection.execute(
        'INSERT INTO "LiteLLM_SpendLogs" (request_id, call_type, "startTime", model, model_id, session_id, '
        "proxy_server_request) VALUES (%s, 'anthropic_messages', '2026-11-02 12:00:00', %s, %s, %s, %s)",
        (request_id, model, model_id, session_id, Jsonb(body)),
    )


def _deliver(
    connection: psycopg.Connection[tuple[object, ...]],
    delivery_id: str,
    request_id: str,
    *,
    applied_days_ago: int = 0,
    state: str = "applied",
    **payload: object,
) -> None:
    connection.execute(
        'INSERT INTO "LiteLLM_SpendLogDelivery" (delivery_id, request_id, accounting_payload, is_batch, state, '
        'applied_at) VALUES (%s, %s, %s, false, %s::"AccountingDeliveryState", '
        "CASE WHEN %s = 'applied' THEN CURRENT_TIMESTAMP - make_interval(days => %s) END)",
        (
            delivery_id,
            request_id,
            Jsonb(
                {
                    "request_id": request_id,
                    "startTime": "2026-11-02T12:00:00+00:00",
                    "call_type": "acompletion",
                    **payload,
                }
            ),
            state,
            state,
            applied_days_ago,
        ),
    )


async def _observe(client: Prisma, prefix: str | None = None) -> int:
    return await observe_applied_deliveries(
        TaskInsightsRepository(client), now=lambda: _NOW, batch_size=50, max_batches=10, request_id_prefix=prefix
    )


def _task(task_id: str, name: str) -> DiscoveredTask:
    return DiscoveredTask(
        task_id=task_id,
        name=name,
        description=f"{name} description",
        inclusion_criteria="in",
        exclusion_criteria="out",
        examples=("example",),
        required_capabilities=("function_calling",),
        map_x=10.0,
        map_y=20.0,
        discovery_size=3,
    )


async def _freeze(repository: TaskInsightsRepository, *tasks: DiscoveredTask) -> int:
    await repository.save_draft_taxonomy(tasks, (), {"source": "test"}, _NOW)
    return await repository.freeze_draft(_NOW)


def _assigned(task_id: str | None) -> Decision:
    return Decision(
        outcome="assigned" if task_id else "unknown",
        task_id=task_id,
        probabilities={"t1": 0.9},
        top_probability=0.9,
        fit_probability=0.9,
        decision_model="jev-test",
    )


@pytest.mark.asyncio
async def test_retried_attempt_keeps_its_charge_but_not_the_final_deployment(insights_database, insights_client):
    _log(
        insights_database,
        "r1",
        model="claude-opus",
        model_id="config-opus",
        messages=[{"role": "user", "content": "Fix the flaky test"}],
    )
    _deliver(
        insights_database,
        "r1-attempt-1",
        "r1",
        spend=0.25,
        model="claude-sonnet",
        status="failure",
        request_duration_ms=900,
    )
    _deliver(insights_database, "r1", "r1", spend=1.0, model="claude-opus", status="success", request_duration_ms=4000)

    assert await _observe(insights_client) == 2
    rows = insights_database.execute(
        'SELECT delivery_id, model_config_id, spend, latency_ms FROM "LiteLLM_TaskObservation" ORDER BY delivery_id'
    ).fetchall()
    assert rows == [("r1", "config-opus", 1.0, 4000.0), ("r1-attempt-1", None, 0.25, None)]

    repository = TaskInsightsRepository(insights_client)
    volumes = {volume.task_id: volume for volume in await repository.task_volumes(_WINDOW)}
    assert volumes[UNCLASSIFIED_TASK_ID].request_count == 1
    assert volumes[UNCLASSIFIED_TASK_ID].delivery_count == 2
    assert volumes[UNCLASSIFIED_TASK_ID].spend == pytest.approx(1.25)
    cells = await repository.matrix_cells(_WINDOW)
    assert [(cell.model_config_id, cell.request_count, cell.avg_cost_usd, cell.p95_latency_ms) for cell in cells] == [
        ("config-opus", 1, pytest.approx(1.0), pytest.approx(4000.0))
    ]


@pytest.mark.asyncio
async def test_a_request_with_any_unrecorded_charge_has_unknown_cost(insights_database, insights_client):
    _log(insights_database, "partial")
    _deliver(insights_database, "partial-1", "partial", spend=0.5, model="claude-sonnet", status="failure")
    _deliver(insights_database, "partial-2", "partial", model="claude-sonnet", status="success")
    _log(insights_database, "known")
    _deliver(insights_database, "known", "known", spend=0.2, model="claude-sonnet", status="success")
    await _observe(insights_client)

    repository = TaskInsightsRepository(insights_client)
    volume = (await repository.task_volumes(_WINDOW))[0]
    assert (volume.request_count, volume.cost_observations, volume.spend) == (2, 1, pytest.approx(0.2))
    cell = (await repository.matrix_cells(_WINDOW))[0]
    assert (cell.request_count, cell.cost_observations, cell.avg_cost_usd) == (2, 1, pytest.approx(0.2))


@pytest.mark.asyncio
async def test_cleanup_waits_for_the_projection_only_when_task_insights_is_on(insights_database, insights_client):
    for request_id in ("old-a", "old-b"):
        _log(insights_database, request_id)
        _deliver(insights_database, request_id, request_id, applied_days_ago=3, spend=0.1, model="claude-sonnet")
    ledger = AccountingDeliveryRepository(insights_client)

    assert await ledger.cleanup_applied(retention_days=1, keep_unobserved=True) == 0
    await _observe(insights_client)
    assert await ledger.cleanup_applied(retention_days=1, keep_unobserved=True) == 2
    assert insights_database.execute('SELECT COUNT(*) FROM "LiteLLM_TaskObservation"').fetchone() == (2,)

    _log(insights_database, "old-c")
    _deliver(insights_database, "old-c", "old-c", applied_days_ago=3, spend=0.1)
    assert await ledger.cleanup_applied(retention_days=1) == 1


@pytest.mark.asyncio
async def test_projection_reads_only_applied_deliveries_once(insights_database, insights_client):
    _log(insights_database, "pending")
    _deliver(insights_database, "pending", "pending", state="pending", spend=0.1)
    _log(insights_database, "applied")
    _deliver(insights_database, "applied", "applied", spend=0.1)

    assert await _observe(insights_client) == 1
    assert await _observe(insights_client) == 0


@pytest.mark.asyncio
async def test_agent_turns_repeating_one_instruction_share_a_text_hash(insights_database, insights_client):
    history: Final = [
        {"role": "user", "content": "Add pagination to the orders endpoint"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "a", "name": "edit", "input": {}}]},
    ]
    _log(insights_database, "turn-1", messages=history[:1], session_id="s")
    _log(
        insights_database,
        "turn-2",
        messages=[
            *history,
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "a", "content": "ok"}]},
        ],
        session_id="s",
    )
    for request_id in ("turn-1", "turn-2"):
        _deliver(insights_database, request_id, request_id, spend=0.1, model="claude-sonnet")
    await _observe(insights_client)

    rows = insights_database.execute(
        'SELECT request_text, context_text, text_hash, session_id FROM "LiteLLM_TaskObservation" ORDER BY request_id'
    ).fetchall()
    assert rows[0] == rows[1]
    assert rows[0][:2] == ("Add pagination to the orders endpoint", None)


@pytest.mark.asyncio
async def test_freezing_a_draft_replaces_the_taxonomy_and_clears_assignments(insights_database, insights_client):
    repository = TaskInsightsRepository(insights_client)
    _log(insights_database, "r", messages=[{"role": "user", "content": "Explain this regex"}])
    _deliver(insights_database, "r", "r", spend=0.1, model="claude-sonnet")
    await _observe(insights_client)
    assert await _freeze(repository, _task("old", "Old task")) == 1
    pending = await repository.pending_requests(10)
    await repository.save_decisions(((pending[0], _assigned("old")),), _NOW, await repository.taxonomy_version())
    assert {volume.task_id for volume in await repository.task_volumes(_WINDOW)} == {"old"}

    await repository.save_draft_taxonomy(
        (_task("new", "New task"),), (MapPlacement(request_id="r", task_id="new", x=1, y=2),), {}, _NOW
    )
    assert [task.task_id for task in await repository.frozen_definitions()] == ["old"]
    assert await repository.edit_draft("old", DraftTaskEdit(name="Renamed")) is None
    assert (await repository.edit_draft("new", DraftTaskEdit(name="Renamed"))).name == "Renamed"

    assert await repository.freeze_draft(_NOW) == 1
    assert [task.name for task in await repository.frozen_definitions()] == ["Renamed"]
    assert {volume.task_id for volume in await repository.task_volumes(_WINDOW)} == {UNCLASSIFIED_TASK_ID}
    assert [(point.x, point.y) for point in await repository.map_points(_WINDOW, 10)] == [(1.0, 2.0)]
    assert await repository.freeze_draft(_NOW) == 0


@pytest.mark.asyncio
async def test_unknown_assignments_and_reused_decisions(insights_database, insights_client):
    repository = TaskInsightsRepository(insights_client)
    _log(insights_database, "poem", messages=[{"role": "user", "content": "Write a poem"}])
    _deliver(insights_database, "poem", "poem", spend=0.1, model="claude-sonnet")
    await _observe(insights_client)
    await _freeze(repository, _task("t1", "Task"))
    pending = (await repository.pending_requests(10))[0]
    await repository.save_decisions(((pending, _assigned(None)),), _NOW, await repository.taxonomy_version())

    assert [volume.task_id for volume in await repository.task_volumes(_WINDOW)] == [UNKNOWN_TASK_ID]
    assert await repository.pending_requests(10) == ()
    reused = await repository.decisions_by_hash((pending.text_hash,))
    assert reused[pending.text_hash].outcome == "unknown"


@pytest.mark.asyncio
async def test_failed_decisions_retry_until_the_attempt_limit(insights_database, insights_client):
    repository = TaskInsightsRepository(insights_client)
    _log(insights_database, "r", messages=[{"role": "user", "content": "Fix the build"}])
    _deliver(insights_database, "r", "r", spend=0.1, model="claude-sonnet")
    await _observe(insights_client)
    await _freeze(repository, _task("t1", "Task"))
    failed = Decision(outcome="failed", task_id=None, decision_model="jev-test", error="HTTPError")

    for _ in range(3):
        pending = await repository.pending_requests(10)
        assert len(pending) == 1
        await repository.save_decisions(((pending[0], failed),), _NOW, await repository.taxonomy_version())

    assert await repository.pending_requests(10) == ()
    assert await repository.decisions_by_hash((pending[0].text_hash,)) == {}


@pytest.mark.asyncio
async def test_only_one_discovery_is_queued_at_a_time(insights_client):
    repository = TaskInsightsRepository(insights_client)

    assert await repository.request_discovery(_WINDOW, 200, _NOW) is True
    assert await repository.request_discovery(_WINDOW, 200, _NOW) is False
    run = await repository.claim_discovery()
    assert run is not None and run.max_requests == 200
    assert await repository.claim_discovery() is None
    await repository.fail_discovery("boom", _NOW)
    assert (await repository.get_state()).discovery_status == "failed"
    assert await repository.request_discovery(_WINDOW, 200, _NOW) is True


@pytest.mark.asyncio
async def test_seeding_twice_gives_the_same_sample_and_leaves_other_records_alone(insights_database, insights_client):
    _log(insights_database, "someone-else", messages=[{"role": "user", "content": "Mine"}])
    _deliver(insights_database, "someone-else", "someone-else", spend=0.3, model="claude-sonnet")

    first = await seed_demo(insights_client, lambda: _NOW)
    counts = insights_database.execute(
        'SELECT (SELECT COUNT(*) FROM "LiteLLM_TaskObservation"), (SELECT COUNT(*) FROM "LiteLLM_TaskAssignment"), '
        '(SELECT COUNT(*) FROM "LiteLLM_TaskDefinition")'
    ).fetchone()
    second = await seed_demo(insights_client, lambda: _NOW)

    assert first == second > 0
    assert (
        insights_database.execute(
            'SELECT (SELECT COUNT(*) FROM "LiteLLM_TaskObservation"), (SELECT COUNT(*) FROM "LiteLLM_TaskAssignment"), '
            '(SELECT COUNT(*) FROM "LiteLLM_TaskDefinition")'
        ).fetchone()
        == counts
    )
    assert insights_database.execute(
        'SELECT COUNT(*) FROM "LiteLLM_TaskObservation" WHERE request_id NOT LIKE %s', (f"{DEMO_PREFIX}%",)
    ).fetchone() == (0,)
    assert insights_database.execute(
        'SELECT state::text FROM "LiteLLM_SpendLogDelivery" WHERE request_id = %s', ("someone-else",)
    ).fetchone() == ("applied",)
    expected_requests = len({request.request_id for request in demo_requests(_NOW) if not request.analysis})
    totals = await TaskInsightsRepository(insights_client).window_totals(
        InsightsWindow(start=datetime(2026, 10, 1, tzinfo=timezone.utc), end=_WINDOW.end)
    )
    assert totals.request_count == expected_requests
    assert totals.analysis_request_count > 0
    assert (await TaskInsightsRepository(insights_client).get_state()).discovery_provenance["source"] == "sample"

    await TaskInsightsDemoRepository(insights_client, DEMO_PREFIX).reset()

    assert insights_database.execute(
        'SELECT (SELECT COUNT(*) FROM "LiteLLM_TaskObservation"), (SELECT COUNT(*) FROM "LiteLLM_TaskDefinition"), '
        '(SELECT COUNT(*) FROM "LiteLLM_SpendLogs"), (SELECT COUNT(*) FROM "LiteLLM_TeamTable")'
    ).fetchone() == (0, 0, 1, 0)


@pytest.mark.asyncio
async def test_seeding_refuses_to_replace_a_real_taxonomy(insights_client):
    await _freeze(TaskInsightsRepository(insights_client), _task("real", "Real task"))

    with pytest.raises(ForeignTaxonomyError):
        await seed_demo(insights_client, lambda: _NOW)


@pytest.mark.asyncio
async def test_a_freeze_during_bucketing_discards_the_stale_batch(insights_database, insights_client):
    import asyncio

    from litellm.task_insights.bucketing import bucket_pending

    repository = TaskInsightsRepository(insights_client)
    _log(insights_database, "r", messages=[{"role": "user", "content": "Fix the flaky test"}])
    _deliver(insights_database, "r", "r", spend=0.1, model="claude-sonnet")
    await _observe(insights_client)
    await _freeze(repository, _task("old", "Old task"))

    class _BlockingDecider:
        decision_model = "jev-test"

        def __init__(self) -> None:
            self.entered = asyncio.Event()
            self.release = asyncio.Event()

        async def decide(self, tasks, request):
            self.entered.set()
            await self.release.wait()
            return _assigned("old")

    decider = _BlockingDecider()
    batch = asyncio.create_task(bucket_pending(repository, decider, now=lambda: _NOW, batch_size=10))
    await decider.entered.wait()
    await repository.save_draft_taxonomy((_task("new", "New task"),), (), {}, _NOW)
    assert await repository.freeze_draft(datetime(2026, 11, 11, 13, 0, tzinfo=timezone.utc)) == 1
    decider.release.set()

    assert await batch == 0
    assert insights_database.execute('SELECT COUNT(*) FROM "LiteLLM_TaskAssignment"').fetchone() == (0,)
    assert [pending.request_id for pending in await repository.pending_requests(10)] == ["r"]


@pytest.mark.asyncio
async def test_request_samples_follow_the_task_and_deployment_filters(insights_database, insights_client):
    repository = TaskInsightsRepository(insights_client)
    _log(
        insights_database,
        "on-sonnet",
        messages=[{"role": "user", "content": "Explain the cache lock"}],
        session_id="s1",
    )
    _deliver(insights_database, "on-sonnet", "on-sonnet", spend=0.2, model="claude-sonnet", request_duration_ms=900)
    _log(
        insights_database,
        "on-opus",
        model="claude-opus",
        model_id="config-opus",
        messages=[{"role": "user", "content": "Explain the router"}],
    )
    _deliver(insights_database, "on-opus", "on-opus", spend=0.9, model="claude-opus")
    await _observe(insights_client)
    await _freeze(repository, _task("explain", "Explain code"))
    pending = await repository.pending_requests(10)
    await repository.save_decisions(
        tuple((item, _assigned("explain")) for item in pending), _NOW, await repository.taxonomy_version()
    )

    every = await repository.request_samples(_WINDOW, "explain", None, 10)
    on_sonnet = await repository.request_samples(_WINDOW, "explain", "config-sonnet", 10)

    assert {sample.request_id for sample in every} == {"on-sonnet", "on-opus"}
    assert [(sample.request_id, sample.excerpt, sample.session_id, sample.latency_ms) for sample in on_sonnet] == [
        ("on-sonnet", "Explain the cache lock", "s1", 900.0)
    ]
    assert await repository.request_samples(_WINDOW, UNKNOWN_TASK_ID, None, 10) == ()
    assert (await repository.freshness()).last_bucketed_at is not None
    assert await repository.assigned_task_names(("Explain the router", "never sent")) == {
        "Explain the router": "Explain code"
    }


@pytest.mark.asyncio
async def test_discovery_samples_are_distinct_texts_inside_the_window_without_analysis_traffic(
    insights_database, insights_client
):
    repository = TaskInsightsRepository(insights_client)
    for request_id in ("first", "replay"):
        _log(insights_database, request_id, messages=[{"role": "user", "content": "Fix the build"}])
        _deliver(insights_database, request_id, request_id, spend=0.1, model="claude-sonnet")
    _log(insights_database, "analysis", messages=[{"role": "user", "content": "Distill these"}])
    _deliver(
        insights_database,
        "analysis",
        "analysis",
        spend=0.1,
        model="claude-sonnet",
        request_tags='["litellm-task-insights-analysis"]',
    )
    await _observe(insights_client)

    assert await repository.request_discovery(_WINDOW, 50, _NOW)
    run = await repository.claim_discovery()
    samples = await repository.discovery_samples(run)
    outside = await repository.discovery_samples(run.model_copy(update={"window": (_NOW, _NOW)}))

    assert [sample.request_text for sample in samples] == ["Fix the build"]
    assert outside == ()


@pytest.mark.asyncio
async def test_a_discovery_left_running_by_a_stopped_worker_is_failed_on_restart(insights_client):
    repository = TaskInsightsRepository(insights_client)
    await repository.request_discovery(_WINDOW, 50, _NOW)
    await repository.claim_discovery()

    assert await repository.recover_discovery() == 1
    state = await repository.get_state()
    assert (state.discovery_status, state.discovery_error) == (
        "failed",
        "The analysis worker restarted during discovery",
    )
    assert await repository.recover_discovery() == 0


@pytest.mark.asyncio
async def test_empty_writes_and_lookups_touch_nothing(insights_client):
    repository = TaskInsightsRepository(insights_client)

    assert await repository.insert_observations(()) == 0
    assert await repository.decisions_by_hash(()) == {}
    assert await repository.save_decisions((), _NOW, None) == 0
    assert await repository.assigned_task_names(()) == {}


@pytest.mark.asyncio
async def test_a_failed_management_request_is_left_out_while_a_failed_model_call_counts(
    insights_database, insights_client
):
    repository = TaskInsightsRepository(insights_client)
    _log(insights_database, "login", model="", model_id=None)
    _deliver(
        insights_database, "login", "login", call_type="/management/v1/users/me", status="failure", spend=0.0, model=""
    )
    _log(insights_database, "model-call", messages=[{"role": "user", "content": "Fix the flaky test"}])
    _deliver(insights_database, "model-call", "model-call", status="failure", spend=0.005, model="claude-sonnet")

    assert await _observe(insights_client) == 2

    totals = await repository.window_totals(_WINDOW)
    assert (totals.request_count, totals.analysis_request_count) == (1, 0)
    volume = (await repository.task_volumes(_WINDOW))[0]
    assert (volume.request_count, volume.spend) == (1, pytest.approx(0.005))
    assert [request.request_id for request in await repository.pending_requests(10)] == ["model-call"]
    ledger = AccountingDeliveryRepository(insights_client)
    insights_database.execute(
        'UPDATE "LiteLLM_SpendLogDelivery" SET applied_at = CURRENT_TIMESTAMP - make_interval(days => 3)'
    )
    assert await ledger.cleanup_applied(retention_days=1, keep_unobserved=True) == 2
