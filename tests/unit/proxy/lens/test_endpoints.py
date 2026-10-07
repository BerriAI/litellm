import asyncio
from collections.abc import Callable, Mapping
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Final

import pytest
from fastapi import HTTPException
from pydantic import TypeAdapter, ValidationError

import litellm
from litellm import Router
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.lens.endpoints import (
    claim_due,
    get_signals,
    list_agents,
    put_signals,
    read_reviews,
    result,
    run_settings,
    run_window,
    trace_findings,
    trace_signal_statuses,
    user_scope,
    validate_model,
    validate_signal_model,
    watchable,
    watching,
    worker_supports_model,
)
from litellm.proxy.lens.endpoints import (
    sample as worker_sample,
)
from litellm.proxy.lens.models import (
    ActivitySelection,
    Coverage,
    Lens,
    LensSettings,
    Result,
    ReviewVersion,
    RunAssessment,
    RunRequest,
    Sample,
    Scope,
    TraceFindingsRequest,
    TraceIdentity,
    Worker,
)
from litellm.proxy.lens.repository import DueLens, Row
from litellm.proxy.lens.signals import SignalConfig, StoredTraceSignal
from litellm.proxy.lens.state import claim_job, queue_job, replace_job
from litellm.rust_bridge.trace.generated.models import ExecutionRow, LensSampleParams
from tests.unit.proxy.lens.test_agent_workspace import execution
from tests.unit.proxy.lens.test_state import NOW, lens, worker


class ResultDatabase:
    def __init__(self, stored: Lens) -> None:
        self.stored = stored
        self.completed: tuple[ReviewVersion, ...] = ()

    async def query_raw(self, query: str, *args: object) -> tuple[Row, ...]:
        if query.startswith("SELECT data FROM"):
            return (Row(data=self.stored.model_dump(mode="json")),)
        payload: Final = args[0]
        assert isinstance(payload, str)
        self.stored = Lens.model_validate_json(payload)
        return (Row(data=1),)

    async def execute_raw(self, query: str, *args: object) -> int:
        from pydantic import TypeAdapter

        payload: Final = args[2]
        assert isinstance(payload, str)
        self.completed = TypeAdapter(tuple[ReviewVersion, ...]).validate_json(payload)
        return len(self.completed)


class SignalStatusDatabase:
    def __init__(self, config: SignalConfig, rows: Mapping[str, StoredTraceSignal]) -> None:
        self.config: Final = config
        self.rows: Final = rows
        self.saved: Final[asyncio.Queue[tuple[object, ...]]] = asyncio.Queue()

    async def query_raw(self, query: str, *args: object) -> object:
        if '"LiteLLM_LensSignalConfig"' in query:
            return ({"data": self.config.model_dump(mode="json")},)
        payload: Final = args[0]
        assert isinstance(payload, str)
        requested: Final = TypeAdapter(tuple[TraceIdentity, ...]).validate_json(payload)
        return tuple(
            {"data": row.model_dump(mode="json")}
            for identity in requested
            if (row := self.rows.get(identity.trace_id)) is not None
        )

    async def execute_raw(self, query: str, *args: object) -> int:
        await self.saved.put(args)
        return 1


def signal_router() -> Router:
    return Router(
        model_list=[
            {
                "model_name": "decision",
                "litellm_params": {"model": "openai/test-decision", "api_key": "test-key"},
                "model_info": {"mode": "evaluation"},
            },
            {
                "model_name": "chat",
                "litellm_params": {"model": "openai/test-chat", "api_key": "test-key"},
                "model_info": {"mode": "chat"},
            },
        ]
    )


@pytest.mark.asyncio
async def test_worker_sample_retries_oversized_pages_and_keeps_all_executions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from litellm.proxy import proxy_server

    claimed: Final = claim_job(queue_job(lens(), NOW, "job"), worker(), NOW)
    active: Final = claimed.jobs[0].model_copy(update={"lease_until": datetime.max.replace(tzinfo=timezone.utc)})
    db: Final = ResultDatabase(replace_job(claimed, active))
    monkeypatch.setattr(proxy_server, "prisma_client", SimpleNamespace(db=db))
    rows: Final = tuple(
        ExecutionRow(
            source="traces",
            trace_id=trace_id,
            team_id="team",
            name=trace_id,
            start_time="",
            span_count=1,
            root_seen=1,
            eligible=3,
            selected=3,
            selection_key=trace_id,
        )
        for trace_id in ("trace-1", "trace-2", "trace-3")
    )

    class SampleStorage:
        def __init__(self) -> None:
            self.limits: tuple[int, ...] = ()

        async def lens_sample(self, parameters: LensSampleParams) -> tuple[ExecutionRow, ...]:
            self.limits = (*self.limits, parameters.limit)
            if parameters.limit > 2_500:
                raise RuntimeError("ClickHouse query exceeded the response size limit")
            return rows

    storage: Final = SampleStorage()
    selected: Final = await worker_sample("lens", "job", worker(), storage)
    assert storage.limits == (10_000, 5_000, 2_500)
    assert tuple(execution.trace_id for execution in selected.executions) == ("trace-1", "trace-2", "trace-3")
    assert selected.selected == 3


@pytest.mark.asyncio
async def test_worker_sample_propagates_response_too_large_at_minimum_page_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from litellm.proxy import proxy_server

    claimed: Final = claim_job(queue_job(lens(), NOW, "job"), worker(), NOW)
    active: Final = claimed.jobs[0].model_copy(update={"lease_until": datetime.max.replace(tzinfo=timezone.utc)})
    db: Final = ResultDatabase(replace_job(claimed, active))
    monkeypatch.setattr(proxy_server, "prisma_client", SimpleNamespace(db=db))

    class SampleStorage:
        def __init__(self) -> None:
            self.limits: tuple[int, ...] = ()

        async def lens_sample(self, parameters: LensSampleParams) -> tuple[ExecutionRow, ...]:
            self.limits = (*self.limits, parameters.limit)
            raise RuntimeError("ClickHouse query exceeded the response size limit")

    storage: Final = SampleStorage()
    with pytest.raises(RuntimeError, match="response size limit"):
        await worker_sample("lens", "job", worker(), storage)
    assert storage.limits == (10_000, 5_000, 2_500, 1_250, 625, 312, 156, 100)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "selected,check_id,quoted",
    ((False, "retries", "run"), (True, "disabled", "run"), (True, "retries", "other")),
)
async def test_checkpoint_rejects_unselected_traces_disabled_checks_and_foreign_evidence(
    monkeypatch: pytest.MonkeyPatch, selected: bool, check_id: str, quoted: str
) -> None:
    from litellm.proxy import proxy_server
    from litellm.proxy.lens.endpoints import progress
    from litellm.proxy.lens.models import Evidence, Extraction, Observation, Progress, Review

    claimed: Final = claim_job(queue_job(lens(), NOW, "job"), worker(), NOW)
    active: Final = claimed.jobs[0].model_copy(
        update={
            "lease_until": datetime.max.replace(tzinfo=timezone.utc),
            "sample": Sample(executions=(execution("run"),), eligible=1) if selected else None,
        }
    )
    stored: Final = replace_job(claimed, active)
    db: Final = ResultDatabase(stored)
    monkeypatch.setattr(proxy_server, "prisma_client", SimpleNamespace(db=db))
    review: Final = Review(
        execution_id="run",
        trace_id="run",
        agent="agent",
        name="run",
        model="test",
        duration_ms=1,
        at=NOW,
        content_version="v1",
        extraction=Extraction(
            observations=(
                Observation(
                    check_id=check_id,
                    summary="Failure",
                    evidence=(Evidence(execution_id=quoted, span_id="s", quote="failed"),),
                ),
            )
        ),
    )
    with pytest.raises(HTTPException) as error:
        await progress("lens", "job", Progress(review=review), worker())
    assert error.value.status_code == 422
    assert db.stored == stored
    assert db.completed == ()


@pytest.mark.asyncio
@pytest.mark.parametrize("reference", ("existing", "merged"))
@pytest.mark.parametrize("foreign_kind", (False, True))
async def test_findings_cannot_merge_missing_ids_or_positive_patterns_into_issues(
    reference: str, foreign_kind: bool
) -> None:
    from litellm.proxy.lens.endpoints import validate_finding
    from litellm.proxy.lens.state import merge_finding
    from tests.unit.proxy.lens.test_state import finding

    saved: Final = merge_finding(lens(), finding("old"), 1, NOW, "previous").model_copy(update={"kind": "pattern"})
    identity: Final = saved.id if foreign_kind else "missing"
    draft: Final = finding("new").model_copy(
        update={
            "existing_finding_id": identity if reference == "existing" else None,
            "merged_finding_ids": (identity,) if reference == "merged" else (),
        }
    )
    with pytest.raises(HTTPException) as error:
        await validate_finding(
            lens().model_copy(update={"findings": (saved,)}), Sample(executions=(), eligible=0), draft, None
        )
    assert error.value.status_code == 422
    assert "finding must belong" in error.value.detail


@pytest.mark.asyncio
@pytest.mark.parametrize("selected", ("old-trace", "selected-without-quote"))
async def test_unchanged_rerun_does_not_rediscover_old_or_quoteless_occurrences(
    monkeypatch: pytest.MonkeyPatch, selected: str
) -> None:
    from litellm.proxy import proxy_server
    from litellm.proxy.lens.state import merge_finding
    from tests.unit.proxy.lens.test_state import finding

    saved_finding: Final = merge_finding(lens(), finding("old-trace"), 1, NOW, "original-run").model_copy(
        update={"occurrences": ("old-trace", "selected-without-quote")}
    )
    stored: Final = lens().model_copy(update={"findings": (saved_finding,)})
    claimed: Final = claim_job(queue_job(stored, NOW, "job"), worker(), NOW)
    active: Final = claimed.jobs[0].model_copy(
        update={
            "lease_until": datetime.max.replace(tzinfo=timezone.utc),
            "sample": Sample(executions=(execution(selected),), eligible=1),
        }
    )
    db: Final = ResultDatabase(replace_job(claimed, active))
    monkeypatch.setattr(proxy_server, "prisma_client", SimpleNamespace(db=db))
    body: Final = Result(coverage=Coverage(reused=1), assessments=(RunAssessment(execution_id=selected),))
    completed: Final = await result("lens", "job", body, worker(), None)
    assert completed.jobs[0].findings == ()
    assert completed.findings == (saved_finding,)
    assert Lens.model_validate_json(completed.model_dump_json()) == completed
    assert await result("lens", "job", body, worker(), None) == completed


@pytest.mark.asyncio
async def test_completed_checkpoints_are_sealed_despite_an_unrelated_trace_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from litellm.proxy import proxy_server

    claimed: Final = claim_job(queue_job(lens(), NOW, "job"), worker(), NOW)
    active: Final = claimed.jobs[0].model_copy(
        update={
            "lease_until": datetime.max.replace(tzinfo=timezone.utc),
            "sample": Sample(executions=(execution("valid"), execution("failed")), eligible=2),
        }
    )
    db: Final = ResultDatabase(replace_job(claimed, active))
    monkeypatch.setattr(proxy_server, "prisma_client", SimpleNamespace(db=db))
    versions: Final = (ReviewVersion(execution_id="valid", content_version="v1"),)
    body: Final = Result(
        coverage=Coverage(failed_tasks=1),
        assessments=(RunAssessment(execution_id="valid"), RunAssessment(execution_id="failed", cannot_assess=True)),
        review_versions=versions,
        error="Another trace failed",
    )
    completed: Final = await result("lens", "job", body, worker(), None)
    assert completed.jobs[0].status == "completed"
    assert db.completed == versions
    assert await result("lens", "job", body, worker(), None) == completed
    assert db.completed == versions


@pytest.mark.parametrize(
    "final_coverage,error,expected",
    (
        (
            Coverage(eligible=2, selected=2, screened=2, partial=1, unassessable=1),
            "Source unavailable during session review",
            Coverage(eligible=2, selected=2, screened=2, partial=1, unassessable=1),
        ),
        (
            Coverage(eligible=2, selected=2, screened=2, investigated=1, candidates=1, partial=1),
            "Source unavailable during investigation",
            Coverage(eligible=2, selected=2, screened=2, investigated=1, candidates=1, partial=1),
        ),
        (
            Coverage(),
            "Worker interrupted",
            Coverage(eligible=2, selected=2, screened=1),
        ),
        (Coverage(), "", Coverage()),
    ),
    ids=("review-diagnostic", "investigation-diagnostic", "interrupted-worker", "empty-success"),
)
@pytest.mark.asyncio
@pytest.mark.parametrize("assessed", (False, True))
async def test_result_persists_final_coverage_but_keeps_progress_when_worker_is_interrupted(
    monkeypatch: pytest.MonkeyPatch, final_coverage: Coverage, error: str, expected: Coverage, assessed: bool
) -> None:
    from litellm.proxy import proxy_server

    assigned: Final = claim_job(queue_job(lens(), NOW, "job"), worker(), NOW)
    active: Final = assigned.jobs[0].model_copy(
        update={
            "lease_until": datetime.max.replace(tzinfo=timezone.utc),
            "coverage": Coverage(eligible=2, selected=2, screened=1),
            "sample": Sample(executions=(execution("run"),), eligible=1),
        }
    )
    db: Final = ResultDatabase(replace_job(assigned, active))
    monkeypatch.setattr(proxy_server, "prisma_client", SimpleNamespace(db=db))
    assessments: Final = (RunAssessment(execution_id="run"),) if assessed else ()
    saved: Final = await result(
        "lens", "job", Result(coverage=final_coverage, error=error, assessments=assessments), worker(), None
    )

    assert saved == db.stored
    assert saved.jobs[0].coverage == expected
    assert saved.jobs[0].error == error
    assert saved.jobs[0].status == ("failed" if error and not assessed else "completed")
    assert saved.jobs[0].assessments == assessments
    assert saved.last_scan_at == (None if error else active.end)


@pytest.mark.asyncio
async def test_result_rejects_checkpoints_for_traces_outside_frozen_sample(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.proxy import proxy_server

    assigned: Final = claim_job(queue_job(lens(), NOW, "job"), worker(), NOW)
    active: Final = assigned.jobs[0].model_copy(
        update={
            "lease_until": datetime.max.replace(tzinfo=timezone.utc),
            "sample": Sample(executions=(execution("selected"),), eligible=1),
        }
    )
    db: Final = ResultDatabase(replace_job(assigned, active))
    monkeypatch.setattr(proxy_server, "prisma_client", SimpleNamespace(db=db))

    with pytest.raises(HTTPException) as raised:
        await result(
            "lens",
            "job",
            Result(coverage=Coverage(), review_versions=(ReviewVersion(execution_id="outside", content_version="v1"),)),
            worker(),
            None,
        )

    assert raised.value.status_code == 422
    assert db.stored.jobs[0] == active


@pytest.fixture
def analysis_router(monkeypatch: pytest.MonkeyPatch) -> Router:
    from litellm.proxy import proxy_server

    monkeypatch.setattr(litellm, "model_cost", {**litellm.model_cost})
    router: Final = Router(
        model_list=[
            {
                "model_name": "openai/*",
                "litellm_params": {
                    "model": "openai/*",
                    "api_key": "test-key",
                    "input_cost_per_token": 0.001,
                    "output_cost_per_token": 0.002,
                },
            },
            {
                "model_name": "analysis",
                "litellm_params": {
                    "model": "openai/test-analysis",
                    "api_key": "test-key",
                    "input_cost_per_token": 0.001,
                    "output_cost_per_token": 0.002,
                },
            },
            {"model_name": "unpriced/*", "litellm_params": {"model": "openai/*", "api_key": "test-key"}},
        ],
        model_group_alias={"analysis-alias": "analysis"},
    )
    monkeypatch.setattr(proxy_server, "llm_router", router)
    return router


@pytest.mark.parametrize("model", ("openai/test-analysis", "analysis", "analysis-alias"))
@pytest.mark.asyncio
async def test_analysis_accepts_models_served_by_configured_routes(analysis_router: Router, model: str) -> None:
    settings: Final = LensSettings(name="Research", model=model, context="Answer using cited sources")
    auth: Final = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN)
    assert analysis_router.get_model_list(model_name=model)
    await validate_model(settings, auth)


@pytest.mark.parametrize("model", ("unconfigured", "anthropic/test-analysis"))
@pytest.mark.asyncio
async def test_analysis_rejects_models_without_a_configured_route(analysis_router: Router, model: str) -> None:
    settings: Final = LensSettings(name="Research", model=model, context="Answer using cited sources")
    auth: Final = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN)
    assert not analysis_router.get_model_list(model_name=model)
    with pytest.raises(HTTPException) as error:
        await validate_model(settings, auth)
    assert error.value.status_code == 400


@pytest.mark.asyncio
async def test_analysis_route_resolution_preserves_key_model_restrictions(analysis_router: Router) -> None:
    settings: Final = LensSettings(name="Research", model="openai/test-analysis", context="Answer using cited sources")
    auth: Final = UserAPIKeyAuth(user_role=LitellmUserRoles.INTERNAL_USER, models=["analysis"])
    assert analysis_router.get_model_list(model_name=settings.model)
    with pytest.raises(HTTPException) as error:
        await validate_model(settings, auth)
    assert error.value.status_code == 403


@pytest.mark.asyncio
async def test_analysis_rejects_unpriced_wildcard_before_creating_a_run(analysis_router: Router) -> None:
    settings: Final = LensSettings(name="Research", model="unpriced/lens-unpriced-test", context="Answer questions")
    auth: Final = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN)
    assert analysis_router.get_model_list(model_name=settings.model)
    with pytest.raises(HTTPException) as error:
        await validate_model(settings, auth)
    assert error.value.status_code == 400
    assert "Pricing is not configured" in error.value.detail


@pytest.mark.parametrize("model,allowed", (("openai/test-analysis", "openai/*"), ("analysis-alias", "analysis")))
@pytest.mark.asyncio
async def test_analysis_key_accepts_wildcard_and_alias_access(
    analysis_router: Router, model: str, allowed: str
) -> None:
    auth: Final = UserAPIKeyAuth(user_role=LitellmUserRoles.INTERNAL_USER, models=[allowed])
    assert analysis_router.get_model_list(model_name=model)
    await validate_model(LensSettings(name="Research", model=model, context="Answer questions"), auth)


@pytest.mark.parametrize("revoked,key_id", ((True, "a" * 64), (False, None)))
@pytest.mark.asyncio
async def test_worker_without_active_billing_cannot_take_work(revoked: bool, key_id: str | None) -> None:
    from tests.unit.proxy.lens.test_state import worker

    inactive: Final = worker().model_copy(update={"revoked": revoked, "analysis_key_id": key_id})
    settings: Final = LensSettings(name="Research", model="analysis", context="Answer questions")
    assert not await worker_supports_model(inactive, settings)


@pytest.mark.parametrize("role", (LitellmUserRoles.PROXY_ADMIN, LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY))
@pytest.mark.asyncio
async def test_agent_discovery_without_trace_storage_is_empty(role: LitellmUserRoles) -> None:
    auth: Final = UserAPIKeyAuth(user_role=role)
    assert await list_agents(auth, None) == ()


@pytest.mark.asyncio
async def test_agent_discovery_without_trace_storage_still_requires_admin_access() -> None:
    auth: Final = UserAPIKeyAuth(user_role=LitellmUserRoles.INTERNAL_USER)
    with pytest.raises(HTTPException) as error:
        await list_agents(auth, None)
    assert error.value.status_code == 403


@pytest.mark.asyncio
async def test_trace_finding_counts_require_investigation_read_access() -> None:
    auth: Final = UserAPIKeyAuth(user_role=LitellmUserRoles.INTERNAL_USER)
    request: Final = TraceFindingsRequest(traces=(TraceIdentity(trace_id="trace"),))
    with pytest.raises(HTTPException) as error:
        await trace_findings(request, auth)
    assert error.value.status_code == 403


@pytest.mark.asyncio
async def test_signal_endpoints_return_statuses_in_request_order_for_admin_viewers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from litellm.proxy import proxy_server

    config: Final = SignalConfig(model="decision")
    rows: Final = {
        "pending": StoredTraceSignal(
            trace_id="pending",
            config_key=config.key(),
            span_count=1,
            claimed_until=NOW + timedelta(minutes=1),
            data={"status": "pending", "scores": {}, "model": "decision", "error": ""},
        ),
        "classified": StoredTraceSignal(
            trace_id="classified",
            config_key=config.key(),
            span_count=1,
            classified_at=NOW,
            data={
                "status": "classified",
                "scores": {"user_frustration": 0.7, "missing_capability": 0.8},
                "model": "decision",
                "error": "",
            },
        ),
        "failed": StoredTraceSignal(
            trace_id="failed",
            config_key=config.key(),
            span_count=1,
            classified_at=NOW,
            data={"status": "failed", "scores": {}, "model": "decision", "error": "classification failed"},
        ),
        "stale": StoredTraceSignal(
            trace_id="stale",
            config_key="old-config",
            span_count=1,
            classified_at=NOW,
            data={
                "status": "classified",
                "scores": {"user_frustration": 1.0},
                "model": "old",
                "error": "",
            },
        ),
    }
    database: Final = SignalStatusDatabase(config, rows)
    monkeypatch.setattr(proxy_server, "prisma_client", SimpleNamespace(db=database))
    viewer: Final = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY)
    request: Final = TraceFindingsRequest(
        traces=tuple(
            TraceIdentity(trace_id=trace_id) for trace_id in ("failed", "classified", "missing", "pending", "stale")
        )
    )

    assert await get_signals(viewer) == config
    results: Final = await trace_signal_statuses(request, viewer)

    assert tuple((result.trace_id, result.status) for result in results) == (
        ("failed", "failed"),
        ("classified", "classified"),
        ("missing", "unclassified"),
        ("pending", "pending"),
        ("stale", "unclassified"),
    )
    assert tuple((flag.signal_id, flag.name, flag.score) for flag in results[1].flags) == (
        ("missing_capability", "Missing capability", 0.8),
        ("user_frustration", "User frustration", 0.7),
    )


@pytest.mark.asyncio
async def test_signal_endpoints_require_connected_postgres(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.proxy import proxy_server

    monkeypatch.setattr(proxy_server, "prisma_client", None)
    auth: Final = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY)

    with pytest.raises(HTTPException) as error:
        await get_signals(auth)

    assert error.value.status_code == 503


@pytest.mark.asyncio
async def test_put_signals_saves_config_for_admin(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.proxy import proxy_server

    database: Final = SignalStatusDatabase(SignalConfig(), {})
    monkeypatch.setattr(proxy_server, "prisma_client", SimpleNamespace(db=database))
    monkeypatch.setattr(proxy_server, "llm_router", signal_router())
    auth: Final = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN)
    body: Final = SignalConfig(model="decision", threshold=0.7)

    assert await put_signals(body, auth) == body

    saved: Final = await database.saved.get()
    assert saved[0] == "global"
    assert isinstance(saved[1], str)
    assert SignalConfig.model_validate_json(saved[1]) == body


def test_signal_model_requires_a_ready_router() -> None:
    with pytest.raises(HTTPException) as error:
        validate_signal_model(SignalConfig(model="decision"), None)

    assert error.value.status_code == 400


@pytest.mark.asyncio
@pytest.mark.parametrize("role", (LitellmUserRoles.INTERNAL_USER, LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY))
async def test_put_signals_rejects_non_admin_roles(role: LitellmUserRoles) -> None:
    auth: Final = UserAPIKeyAuth(user_role=role)
    with pytest.raises(HTTPException) as error:
        await put_signals(SignalConfig(model="decision"), auth)
    assert error.value.status_code == 403


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ("chat", "unconfigured"))
async def test_put_signals_rejects_chat_and_unknown_model_groups(monkeypatch: pytest.MonkeyPatch, model: str) -> None:
    from litellm.proxy import proxy_server

    monkeypatch.setattr(proxy_server, "llm_router", signal_router())
    auth: Final = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN)

    with pytest.raises(HTTPException) as error:
        await put_signals(SignalConfig(model=model), auth)

    assert error.value.status_code == 400
    assert error.value.detail == "Choose a System 1 model (evaluation mode) configured on this proxy"


def test_signal_model_accepts_only_evaluation_mode_groups() -> None:
    assert validate_signal_model(SignalConfig(model="decision"), signal_router()) is None


@pytest.mark.parametrize(
    "role",
    (LitellmUserRoles.INTERNAL_USER, LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY, LitellmUserRoles.TEAM),
)
def test_non_admin_cannot_start_analysis_spending(role: LitellmUserRoles) -> None:
    auth: Final = UserAPIKeyAuth(user_role=role, team_id="team", token="hashed-test-key")
    with pytest.raises(HTTPException) as error:
        user_scope(auth, write=True)
    assert error.value.status_code == 403


def test_admin_can_configure_lens_and_viewer_can_only_read() -> None:
    admin: Final = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN)
    viewer: Final = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY)
    assert user_scope(admin, write=True).all_teams
    assert user_scope(viewer).all_teams


@pytest.mark.parametrize("identity", ("not-an-execution", "W10=", "WyJvdGhlciIsICIiLCAiaWQiXQ=="))
def test_invalid_explicit_execution_ids_are_rejected(identity: str) -> None:
    from litellm.proxy.lens.endpoints import validate_selection
    from tests.unit.proxy.lens.test_state import lens

    settings: Final = lens().settings.model_copy(update={"execution_ids": (identity,)})
    with pytest.raises(HTTPException) as error:
        validate_selection(settings)
    assert error.value.status_code == 422


@pytest.mark.parametrize("protocol_version", (1, 2, 3))
@pytest.mark.asyncio
async def test_incompatible_worker_is_rejected_before_claiming_work(
    protocol_version: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    from litellm.proxy.lens.endpoints import claim
    from tests.unit.proxy.lens.test_state import worker

    monkeypatch.setenv("LITELLM_RELEASE_TAG", "v1.2.3")
    with pytest.raises(HTTPException) as error:
        await claim(worker(), protocol_version=protocol_version)
    assert error.value.status_code == 409
    assert "Upgrade" in error.value.detail


@pytest.mark.parametrize("role", (LitellmUserRoles.INTERNAL_USER, LitellmUserRoles.TEAM, None))
@pytest.mark.asyncio
async def test_regular_keys_cannot_poll_live_reviews(role: LitellmUserRoles | None) -> None:
    auth: Final = UserAPIKeyAuth(user_role=role, team_id="team", token="hashed-test-key")
    with pytest.raises(HTTPException) as error:
        await read_reviews("lens", "job", auth)
    assert error.value.status_code == 403


@pytest.mark.parametrize("role", (LitellmUserRoles.INTERNAL_USER, LitellmUserRoles.TEAM, None))
def test_regular_keys_cannot_read_lens_results(role: LitellmUserRoles | None) -> None:
    auth: Final = UserAPIKeyAuth(user_role=role, team_id="team", token="hashed-test-key")
    with pytest.raises(HTTPException) as error:
        user_scope(auth)
    assert error.value.status_code == 403


def saved_lens() -> Lens:
    now: Final = datetime(2026, 1, 15, tzinfo=timezone.utc)
    return Lens(
        id="lens",
        scope=Scope(all_teams=True),
        settings=LensSettings(name="Support", model="analysis", context="Answer questions", agent_name="support"),
        created_at=now,
        next_run_at=now,
        budget_month="2026-01",
    )


def test_run_now_agent_override_only_changes_the_agent_for_that_run() -> None:
    lens: Final = saved_lens()
    overridden: Final = run_settings(lens, RunRequest(agent_name="billing"))
    assert overridden is not None
    assert overridden.agent_name == "billing"
    assert overridden.model_copy(update={"agent_name": "support"}) == lens.settings


def test_run_now_without_overrides_keeps_the_saved_settings() -> None:
    assert run_settings(saved_lens(), RunRequest()) is None


def test_run_now_rejects_a_window_that_is_missing_an_edge_or_backwards() -> None:
    now: Final = datetime(2026, 1, 15, tzinfo=timezone.utc)
    with pytest.raises(ValidationError, match="both a start and an end"):
        RunRequest(start=now)
    with pytest.raises(ValidationError, match="before end"):
        RunRequest(start=now, end=now - timedelta(hours=1))


def test_watching_switches_a_paused_investigation_on_and_records_the_change() -> None:
    paused: Final = saved_lens().model_copy(
        update={"settings": saved_lens().settings.model_copy(update={"enabled": False})}
    )
    watched: Final = watching(paused)
    assert watched.settings.enabled is True
    assert watched.revision == paused.revision + 1
    assert watched.settings.model_copy(update={"enabled": False}) == paused.settings


def test_watching_leaves_an_investigation_that_is_already_on_untouched() -> None:
    on: Final = saved_lens()
    assert watching(on) is on


async def test_watch_all_skips_an_investigation_whose_model_is_gone_instead_of_failing_them_all(
    analysis_router: Router,
) -> None:
    stale: Final = saved_lens().model_copy(
        update={"settings": saved_lens().settings.model_copy(update={"model": "retired-model", "enabled": False})}
    )
    skipped: Final = await watchable(stale, UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN))
    assert skipped is not None
    assert skipped.id == stale.id
    assert skipped.reason


def test_run_now_since_last_run_keeps_scanning_only_new_traces_even_with_an_agent_override() -> None:
    now: Final = datetime(2026, 1, 15, 12, tzinfo=timezone.utc)
    resumed: Final = saved_lens().model_copy(update={"last_scan_at": now - timedelta(hours=1)})
    window: Final = run_window(resumed, RunRequest(agent_name="billing"), now)
    assert window is not None
    assert window[0] == now - timedelta(hours=1)


def test_run_now_with_a_lookback_scans_that_lookback_instead_of_since_last_run() -> None:
    now: Final = datetime(2026, 1, 15, 12, tzinfo=timezone.utc)
    resumed: Final = saved_lens().model_copy(update={"last_scan_at": now - timedelta(hours=1)})
    assert run_window(resumed, RunRequest(lookback_hours=24), now) is None


@pytest.mark.parametrize("provider", (False, True))
def test_model_errors_reach_worker_with_status_and_redacted_provider_message(provider: bool) -> None:
    import httpx

    from litellm.proxy._types import ProxyException
    from litellm.proxy.lens.endpoints import model_failure
    from litellm.proxy.lens.worker import failure_message

    message: Final = "Token rate limit exceeded. api_key=secret-example-value-123456 Retry in 60 seconds."
    error: Final = model_failure(
        ProxyException(message, "rate_limit_error", None, 429, headers={"retry-after": "60"})
        if provider
        else HTTPException(429, message, headers={"retry-after": "60"})
    )
    request: Final = httpx.Request("POST", "https://proxy.test/lens/worker/lens/run/model")
    response: Final = httpx.Response(error.status_code, json={"detail": error.detail}, request=request)
    with pytest.raises(httpx.HTTPStatusError) as caught:
        response.raise_for_status()
    diagnostic: Final = failure_message(caught.value)
    assert diagnostic.startswith("Model request failed (HTTP 429):")
    assert "Token rate limit exceeded." in diagnostic
    assert "Retry in 60 seconds." in diagnostic
    assert "secret-example" not in diagnostic
    assert error.headers == {"retry-after": "60"}


@pytest.mark.asyncio
async def test_preview_samples_a_selection_without_investigation_settings() -> None:
    from litellm.proxy.lens.endpoints import Preview, preview_sample

    class SelectionStorage:
        async def lens_sample(self, parameters):
            assert (parameters.source, parameters.agent_name, parameters.selected_team) == ("requests", "billing", "t1")
            assert parameters.preview == 1 and parameters.offset == 3
            return []

    body: Final = Preview.model_validate(
        {"selection": {"source": "requests", "agent_name": "billing", "team_id": "t1"}, "offset": 3}
    )
    sample: Final = await preview_sample(
        body, UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN), SelectionStorage()
    )
    assert sample.eligible == 0 and not sample.executions


@pytest.mark.asyncio
async def test_preview_reports_calendar_overflow_as_a_validation_error() -> None:
    from datetime import datetime, timezone

    from litellm.proxy.lens.endpoints import Preview, preview_sample

    body: Final = Preview(
        selection=ActivitySelection(),
        as_of=datetime.min.replace(tzinfo=timezone.utc),
    )
    with pytest.raises(HTTPException) as error:
        await preview_sample(body, UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN), None)
    assert error.value.status_code == 422
    assert "supported calendar range" in error.value.detail


@pytest.mark.asyncio
@pytest.mark.parametrize("worker_release", ("", "v1.2.2", "branch-main-old"))
async def test_different_release_is_rejected_before_accessing_jobs(
    monkeypatch: pytest.MonkeyPatch, worker_release: str
) -> None:
    from litellm.proxy.lens.endpoints import claim
    from litellm.proxy.lens.release import PROTOCOL_VERSION
    from tests.unit.proxy.lens.test_state import worker

    monkeypatch.setenv("LITELLM_RELEASE_TAG", "v1.2.3")
    monkeypatch.delenv("LENS_WORKER_IMAGE", raising=False)
    with pytest.raises(HTTPException) as error:
        await claim(worker(), protocol_version=PROTOCOL_VERSION, worker_release=worker_release)
    assert error.value.status_code == 409
    assert "ghcr.io/berriai/litellm-lens-worker:v1.2.3" in error.value.detail


@pytest.mark.asyncio
async def test_unknown_gateway_release_refuses_registration_and_claims(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.proxy.lens.endpoints import WorkerName, claim, register_worker
    from litellm.proxy.lens.release import PROTOCOL_VERSION
    from tests.unit.proxy.lens.test_state import worker

    monkeypatch.setenv("LITELLM_RELEASE_TAG", "")
    monkeypatch.setenv("LENS_WORKER_IMAGE", "registry.example/lens-worker:old")
    with pytest.raises(HTTPException) as registration_error:
        await register_worker(
            WorkerName(analysis_key_id="a" * 64), UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN)
        )
    assert registration_error.value.status_code == 503
    assert "LITELLM_RELEASE_TAG" in registration_error.value.detail
    with pytest.raises(HTTPException) as claim_error:
        await claim(worker(), protocol_version=PROTOCOL_VERSION, worker_release="")
    assert claim_error.value.status_code == 503
    assert claim_error.value.detail == registration_error.value.detail


@pytest.mark.asyncio
async def test_claim_due_pages_through_more_than_a_thousand_full_pages() -> None:
    candidate_lens: Final = lens()

    def candidate_page(page_number: int, size: int) -> tuple[DueLens, ...]:
        return tuple(
            DueLens(
                lens=candidate_lens.model_copy(update={"id": f"lens-{page_number * 20 + offset:05}"}),
                due_at=NOW,
            )
            for offset in range(size)
        )

    full_pages: Final = tuple(candidate_page(page_number, 20) for page_number in range(1_200))
    pages: Final = (*full_pages, candidate_page(1_200, 1))
    assigned_worker: Final = worker()

    class PagingRepository:
        def __init__(self) -> None:
            self.after_calls: tuple[DueLens | None, ...] = ()

        async def due(
            self, scope: Scope, now: datetime, limit: int, after: DueLens | None = None
        ) -> tuple[DueLens, ...]:
            assert scope == assigned_worker.scope
            assert now == NOW
            assert limit == 20
            self.after_calls = (*self.after_calls, after)
            return pages[len(self.after_calls) - 1]

        async def sync_due(self, lens: Lens) -> None:
            return None

        async def update(
            self, lens_id: str, transform: Callable[[Lens], Lens], attempts: int, *, changed_only: bool
        ) -> Lens | None:
            raise AssertionError("Unsupported models must not update candidates")

    async def reject_model(_worker: Worker, _settings: LensSettings) -> bool:
        return False

    repository: Final = PagingRepository()
    claim: Final = await claim_due(assigned_worker, NOW, repository, reject_model)
    expected_after: Final = (None, *(page[-1] for page in pages[:-1]))

    assert claim is None
    assert len(repository.after_calls) == 1_201
    assert repository.after_calls == expected_after
