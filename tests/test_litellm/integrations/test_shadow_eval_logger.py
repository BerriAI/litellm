"""Unit tests for the shadow-eval logger: sampling, unmasking, the success hook's skip
paths, the detached pipeline, and the lifecycle loop's flush/finalize behavior."""

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.constants import INTERNAL_CALL_ORIGIN_METADATA_KEY
from litellm.integrations.shadow_eval_logger import (
    _MAX_CONCURRENT_SHADOW_TASKS,
    _MAX_JUDGE_PROMPT_CHARS,
    JUDGE_MAX_OUTPUT_TOKENS,
    ActiveShadowEvalJob,
    ShadowEvalLogger,
    _CallFailure,
    _judge_user_prompt,
    _sample_hits,
    _unmask_preference,
)
from litellm.types.utils import SHADOW_EVAL_JUDGE_CALL_ORIGIN, SHADOW_EVAL_ROUTER_CALL_ORIGIN


def _job(**overrides) -> ActiveShadowEvalJob:
    defaults = dict(
        id="job-1",
        router_name="my-router",
        shadow_percentage=100.0,
        judge_model="judge-model",
        status="running",
        cost_estimate=5.0,
        cost_actual=0.0,
        ends_at=datetime.now(timezone.utc) + timedelta(days=1),
    )
    return ActiveShadowEvalJob(**{**defaults, **overrides})


def _prisma() -> MagicMock:
    """Prisma wrapper mock. Lifecycle/counter writes use prisma.db; the pipeline's
    recording runs inside prisma.tx(), whose statements land on prisma.tx_mock."""
    prisma = MagicMock()
    prisma.db.litellm_shadowevaljob.find_many = AsyncMock(return_value=[])
    prisma.db.litellm_shadowevaljob.update_many = AsyncMock(return_value=1)
    prisma.db.litellm_shadowevalverdict.create = AsyncMock()
    tx = MagicMock()
    tx.litellm_shadowevaljob.update_many = AsyncMock(return_value=1)
    tx.litellm_shadowevalverdict.create = AsyncMock()
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=tx)
    ctx.__aexit__ = AsyncMock(return_value=False)
    prisma.tx = MagicMock(return_value=ctx)
    prisma.tx_mock = tx
    return prisma


def _router(shadow_text="shadow answer", judge_json='{"preference": "A", "confidence": 0.9, "reasoning": "x"}'):
    """One mock router serving the shadow call first, the judge call second. The shadow
    call's metadata receives the routing decision write-back, like the real router."""
    router = MagicMock()
    router.model_group_alias = {}
    router.get_model_list = MagicMock(return_value=[{"litellm_params": {"model": "openai/gpt-4o-mini"}}])

    async def acompletion(**kwargs):
        if kwargs["model"] == "my-router":
            kwargs["metadata"]["routing_decision"] = {"tier_label": "SIMPLE", "routed_model": "cheap-model"}
            return {"choices": [{"message": {"content": shadow_text}}], "usage": {"completion_tokens": 5}}
        return {"choices": [{"message": {"content": judge_json}}]}

    router.acompletion = MagicMock(side_effect=acompletion)
    return router


def _logger(router=None, prisma=None) -> ShadowEvalLogger:
    return ShadowEvalLogger(
        router_provider=lambda: router,
        prisma_provider=lambda: prisma,
    )


def _success_kwargs(request_id="req-1", api_key_hash="key-hash", request_metadata=None, call_type="acompletion"):
    return {
        "standard_logging_object": {
            "id": request_id,
            "call_type": call_type,
            "model": "claude-opus",
            "metadata": {"user_api_key_hash": api_key_hash},
            "model_parameters": {"temperature": 0.5, "stream": True},
        },
        "litellm_params": {"metadata": request_metadata or {}},
        "messages": [{"role": "user", "content": "what is 2+2"}],
    }


RESPONSE = {"choices": [{"message": {"content": "real answer"}}]}


async def _drain(logger: ShadowEvalLogger):
    for _ in range(100):
        if logger._inflight_shadow_tasks == 0:
            return
        await asyncio.sleep(0.01)
    raise AssertionError("shadow tasks never drained")


class TestSampling:
    def test_boundaries_and_determinism(self):
        assert not any(_sample_hits(f"req-{i}", "job", 0.0) for i in range(100))
        assert all(_sample_hits(f"req-{i}", "job", 100.0) for i in range(100))
        assert len({_sample_hits("req-1", "job-1", 50.0) for _ in range(10)}) == 1

    def test_distribution_close_to_percentage(self):
        hits = sum(_sample_hits(f"req-{i}", "job-x", 10.0) for i in range(10_000))
        assert 800 < hits < 1200

    def test_different_jobs_sample_independently(self):
        agreements = sum(
            _sample_hits(f"req-{i}", "job-a", 50.0) == _sample_hits(f"req-{i}", "job-b", 50.0) for i in range(1000)
        )
        assert 300 < agreements < 700


@pytest.mark.parametrize(
    "raw,real_is_a,expected",
    [
        ("A", True, "real"),
        ("a", True, "real"),
        ("A", False, "shadow"),
        ("B", True, "shadow"),
        ("B", False, "real"),
        ("tie", True, "tie"),
        ("garbage", True, "tie"),
        ("", False, "tie"),
    ],
)
def test_unmask_preference(raw, real_is_a, expected):
    assert _unmask_preference(raw, real_is_a) == expected


def test_judge_prompt_is_bounded_however_large_the_inputs():
    prompt = _judge_user_prompt("c" * 200_000, "a" * 200_000, "b" * 200_000)
    assert len(prompt) < _MAX_JUDGE_PROMPT_CHARS + 100
    assert prompt.endswith("Which response is better?")
    small = _judge_user_prompt("conv", "alpha", "beta")
    assert "conv" in small and "alpha" in small and "beta" in small


@pytest.mark.asyncio
class TestSuccessHookSkips:
    """Every skip path must leave no scheduled task; the sampled path must schedule one."""

    async def test_happy_path_writes_a_verdict_and_bumps_counters(self, monkeypatch: pytest.MonkeyPatch):
        import litellm as litellm_module

        monkeypatch.setattr(litellm_module, "completion_cost", lambda completion_response: 0.005)
        prisma = _prisma()
        router = _router()
        logger = _logger(router=router, prisma=prisma)
        logger._jobs_by_key = {"key-hash": _job()}

        await logger.async_log_success_event(_success_kwargs(), RESPONSE, None, None)
        await _drain(logger)

        create_kwargs = prisma.tx_mock.litellm_shadowevalverdict.create.call_args.kwargs["data"]
        assert create_kwargs["job_id"] == "job-1"
        assert create_kwargs["request_id"] == "req-1"
        assert create_kwargs["tier_classification"] == "SIMPLE"
        assert create_kwargs["real_model"] == "claude-opus"
        assert create_kwargs["shadow_model"] == "cheap-model"
        assert create_kwargs["judge_preference"] in ("real", "shadow")
        assert create_kwargs["judge_confidence"] == 0.9
        counter_update = prisma.tx_mock.litellm_shadowevaljob.update_many.call_args.kwargs
        assert counter_update["data"]["completed_count"] == {"increment": 1}
        assert counter_update["data"]["cost_actual"] == {"increment": 0.005}
        assert counter_update["where"]["status"] == {"in": ["pending", "running"]}
        assert logger._pending_seen == {"job-1": 1}

    @pytest.mark.parametrize(
        "kwargs_mutation,job_mutation",
        [
            ({"request_metadata": {INTERNAL_CALL_ORIGIN_METADATA_KEY: "shadow_eval_router"}}, {}),
            ({"api_key_hash": "other-key"}, {}),
            ({"call_type": "aembedding"}, {}),
            ({"call_type": None}, {}),
            ({"request_metadata": {"routing_decision": {"router_model_name": "my-router"}}}, {}),
            ({}, {"ends_at": datetime.now(timezone.utc) - timedelta(seconds=1)}),
            ({}, {"cost_estimate": 1.0, "cost_actual": 99.0}),
        ],
        ids=["internal-origin", "no-job-for-key", "non-chat", "missing-call-type", "self-shadow", "past-end", "over-spend-cap"],
    )
    async def test_skip_paths_schedule_nothing(self, kwargs_mutation, job_mutation):
        prisma = _prisma()
        logger = _logger(router=_router(), prisma=prisma)
        logger._jobs_by_key = {"key-hash": _job(**job_mutation)}

        await logger.async_log_success_event(_success_kwargs(**kwargs_mutation), RESPONSE, None, None)
        await _drain(logger)

        prisma.tx_mock.litellm_shadowevalverdict.create.assert_not_called()
        assert logger._inflight_shadow_tasks == 0

    async def test_inflight_cap_sheds_instead_of_queueing(self):
        prisma = _prisma()
        logger = _logger(router=_router(), prisma=prisma)
        logger._jobs_by_key = {"key-hash": _job()}
        logger._inflight_shadow_tasks = _MAX_CONCURRENT_SHADOW_TASKS

        await logger.async_log_success_event(_success_kwargs(), RESPONSE, None, None)

        assert logger._inflight_shadow_tasks == _MAX_CONCURRENT_SHADOW_TASKS
        prisma.tx_mock.litellm_shadowevalverdict.create.assert_not_called()

    async def test_sampled_out_request_still_counts_toward_request_count(self):
        logger = _logger(router=_router(), prisma=_prisma())
        logger._jobs_by_key = {"key-hash": _job(shadow_percentage=0.1)}

        for i in range(20):
            await logger.async_log_success_event(_success_kwargs(request_id=f"req-{i}"), RESPONSE, None, None)
        await _drain(logger)

        assert logger._pending_seen["job-1"] == 20


@pytest.mark.asyncio
class TestShadowPipeline:
    async def test_no_prisma_means_no_provider_spend(self):
        """The prisma gate sits above the shadow and judge dispatch: no verdict store,
        no spend."""
        router = _router()
        logger = _logger(router=router, prisma=None)

        await logger._run_shadow_eval(
            job=_job(),
            request_id="req-1",
            messages=({"role": "user", "content": "hi"},),
            response_obj=RESPONSE,
            real_model="claude-opus",
            model_parameters={},
            parent_metadata={},
        )

        router.acompletion.assert_not_called()

    async def test_over_budget_key_skips_before_any_call(self, monkeypatch: pytest.MonkeyPatch):
        """The gate delegates to the auth path's own budget owner, so an over-budget
        verdict there (BudgetExceededError) skips the shadow before any provider call."""
        import litellm.proxy.auth.auth_checks as auth_checks
        from litellm.exceptions import BudgetExceededError
        from litellm.proxy._types import UserAPIKeyAuth

        monkeypatch.setattr(
            auth_checks,
            "_virtual_key_max_budget_check",
            AsyncMock(side_effect=BudgetExceededError(current_cost=11.0, max_budget=10.0)),
        )
        router = _router()
        logger = _logger(router=router, prisma=_prisma())

        await logger._run_shadow_eval(
            job=_job(),
            request_id="req-1",
            messages=({"role": "user", "content": "hi"},),
            response_obj=RESPONSE,
            real_model="claude-opus",
            model_parameters={},
            parent_metadata={"user_api_key_auth": UserAPIKeyAuth(api_key="sk-abc", max_budget=10.0)},
        )

        router.acompletion.assert_not_called()

    async def test_shadow_failure_bumps_failed_count_with_last_error(self):
        prisma = _prisma()
        router = MagicMock()
        router.model_group_alias = {}
        router.get_model_list = MagicMock(return_value=None)
        router.acompletion = AsyncMock(side_effect=RuntimeError("provider exploded"))
        logger = _logger(router=router, prisma=prisma)

        await logger._run_shadow_eval(
            job=_job(),
            request_id="req-1",
            messages=({"role": "user", "content": "hi"},),
            response_obj=RESPONSE,
            real_model="claude-opus",
            model_parameters={},
            parent_metadata={},
        )

        prisma.tx_mock.litellm_shadowevalverdict.create.assert_not_called()
        bump = prisma.db.litellm_shadowevaljob.update_many.call_args.kwargs
        assert bump["data"]["failed_count"] == {"increment": 1}
        assert "provider exploded" in bump["data"]["last_error"]

    async def test_unparseable_judge_verdict_is_a_counted_failure(self):
        prisma = _prisma()
        logger = _logger(router=_router(judge_json="I prefer response A, definitely"), prisma=prisma)

        await logger._run_shadow_eval(
            job=_job(),
            request_id="req-1",
            messages=({"role": "user", "content": "hi"},),
            response_obj=RESPONSE,
            real_model="claude-opus",
            model_parameters={},
            parent_metadata={},
        )

        prisma.tx_mock.litellm_shadowevalverdict.create.assert_not_called()
        bump = prisma.db.litellm_shadowevaljob.update_many.call_args.kwargs
        assert bump["data"]["failed_count"] == {"increment": 1}

    async def test_redacted_requests_are_never_shadowed(self):
        """Redaction rewrites the logged messages before callbacks run, so this hook
        only ever sees placeholders for opted-out traffic: sampling it would judge
        garbage and put content the caller opted out of logging into sub-call rows.
        The skip uses the redactor's own predicate, so every redaction source counts."""
        prisma = _prisma()
        router = _router()
        logger = _logger(router=router, prisma=prisma)
        logger._jobs_by_key = {"key-hash": _job()}

        hook_kwargs = _success_kwargs()
        hook_kwargs["standard_callback_dynamic_params"] = {"turn_off_message_logging": True}
        await logger.async_log_success_event(hook_kwargs, RESPONSE, None, None)
        await _drain(logger)

        router.acompletion.assert_not_called()
        prisma.tx_mock.litellm_shadowevalverdict.create.assert_not_called()

    async def test_sub_calls_carry_identity_and_origin_but_never_parent_request_state(self):
        prisma = _prisma()
        router = _router()
        logger = _logger(router=router, prisma=prisma)
        parent_metadata = {
            "user_api_key_hash": "key-hash",
            "user_api_key_team_id": "team-1",
            "user_api_key_budget_reservation": {"amount": 1.0},
            "routing_decision": {"router_model_name": "other-router"},
        }

        await logger._run_shadow_eval(
            job=_job(),
            request_id="req-1",
            messages=({"role": "user", "content": "hi"},),
            response_obj=RESPONSE,
            real_model="claude-opus",
            model_parameters={"stream": True, "temperature": 0.2, "metadata": {"x": 1}},
            parent_metadata=parent_metadata,
        )

        shadow_call = router.acompletion.call_args_list[0].kwargs
        judge_call = router.acompletion.call_args_list[1].kwargs
        for call in (shadow_call, judge_call):
            assert call["num_retries"] == 0
            assert call["fallbacks"] == []
        assert shadow_call["metadata"][INTERNAL_CALL_ORIGIN_METADATA_KEY] == SHADOW_EVAL_ROUTER_CALL_ORIGIN
        assert judge_call["metadata"][INTERNAL_CALL_ORIGIN_METADATA_KEY] == SHADOW_EVAL_JUDGE_CALL_ORIGIN
        for call in (shadow_call, judge_call):
            assert call["metadata"]["user_api_key_hash"] == "key-hash"
            assert call["metadata"]["user_api_key_team_id"] == "team-1"
            assert "user_api_key_budget_reservation" not in call["metadata"]
        assert "routing_decision" not in judge_call["metadata"]
        assert "stream" not in shadow_call
        assert shadow_call["temperature"] == 0.2
        assert judge_call["max_tokens"] == JUDGE_MAX_OUTPUT_TOKENS



    async def test_job_stopped_mid_flight_drops_the_verdict(self):
        """The status-guarded counter update decides whether the verdict lands, so a
        completed job's results can never disagree with its frozen counts."""
        prisma = _prisma()
        prisma.tx_mock.litellm_shadowevaljob.update_many = AsyncMock(return_value=0)
        logger = _logger(router=_router(), prisma=prisma)

        await logger._run_shadow_eval(
            job=_job(),
            request_id="req-1",
            messages=({"role": "user", "content": "hi"},),
            response_obj=RESPONSE,
            real_model="claude-opus",
            model_parameters={},
            parent_metadata={},
        )

        prisma.tx_mock.litellm_shadowevalverdict.create.assert_not_called()

    async def test_failed_verdict_write_files_the_pipeline_under_failed_once(self):
        """A create that raises inside the transaction rolls the counters back, and the
        pipeline lands in failed_count exactly once, never in both buckets."""
        prisma = _prisma()
        prisma.tx_mock.litellm_shadowevalverdict.create = AsyncMock(side_effect=RuntimeError("db write failed"))
        logger = _logger(router=_router(), prisma=prisma)

        await logger._run_shadow_eval(
            job=_job(),
            request_id="req-1",
            messages=({"role": "user", "content": "hi"},),
            response_obj=RESPONSE,
            real_model="claude-opus",
            model_parameters={},
            parent_metadata={},
        )

        bumps = [c.kwargs for c in prisma.db.litellm_shadowevaljob.update_many.call_args_list]
        assert len(bumps) == 1
        assert bumps[0]["data"]["failed_count"] == {"increment": 1}
        assert "db write failed" in bumps[0]["data"]["last_error"]

    async def test_recording_runs_inside_one_transaction(self):
        """Counter and verdict must ride prisma.tx(), never separate prisma.db writes,
        so a failed verdict write cannot leave a counted-but-missing verdict."""
        prisma = _prisma()
        logger = _logger(router=_router(), prisma=prisma)

        await logger._run_shadow_eval(
            job=_job(),
            request_id="req-1",
            messages=({"role": "user", "content": "hi"},),
            response_obj=RESPONSE,
            real_model="claude-opus",
            model_parameters={},
            parent_metadata={},
        )

        prisma.tx.assert_called_once()
        prisma.tx_mock.litellm_shadowevaljob.update_many.assert_awaited_once()
        prisma.tx_mock.litellm_shadowevalverdict.create.assert_awaited_once()
        prisma.db.litellm_shadowevaljob.update_many.assert_not_called()
        prisma.db.litellm_shadowevalverdict.create.assert_not_called()

@pytest.mark.asyncio
class TestLifecycle:
    async def test_flush_is_status_guarded_and_buffer_resets(self):
        prisma = _prisma()
        logger = _logger(prisma=prisma)
        logger._pending_seen = {"job-1": 7}

        await logger._flush_seen_counts()

        flush = prisma.db.litellm_shadowevaljob.update_many.call_args.kwargs
        assert flush["where"] == {"id": "job-1", "status": {"in": ["pending", "running"]}}
        assert flush["data"] == {"request_count": {"increment": 7}}
        assert logger._pending_seen == {}

    async def test_tick_finalizes_expired_and_overspent_jobs(self):
        prisma = _prisma()
        expired = _job(id="job-expired", ends_at=datetime.now(timezone.utc) - timedelta(seconds=1))
        overspent = _job(id="job-overspent", cost_estimate=1.0, cost_actual=99.0)
        active = _job(id="job-active")
        prisma.db.litellm_shadowevaljob.find_many = AsyncMock(
            return_value=[
                MagicMock(
                    id=j.id,
                    api_key_id=f"key-{j.id}",
                    router_name=j.router_name,
                    shadow_percentage=j.shadow_percentage,
                    judge_model=j.judge_model,
                    status=j.status,
                    cost_estimate=j.cost_estimate,
                    cost_actual=j.cost_actual,
                    ends_at=j.ends_at,
                )
                for j in (expired, overspent, active)
            ]
        )
        logger = _logger(prisma=prisma)

        await logger._lifecycle_tick()

        finalized = {c.kwargs["where"]["id"] for c in prisma.db.litellm_shadowevaljob.update_many.call_args_list}
        assert finalized == {"job-expired", "job-overspent"}
        for call in prisma.db.litellm_shadowevaljob.update_many.call_args_list:
            assert call.kwargs["where"]["status"] == {"in": ["pending", "running"]}
            assert call.kwargs["data"]["status"] == "completed"
        assert set(logger._jobs_by_key) == {"key-job-active"}

    async def test_refresh_keeps_stale_snapshot_on_db_blip(self):
        prisma = _prisma()
        prisma.db.litellm_shadowevaljob.find_many = AsyncMock(side_effect=RuntimeError("db blip"))
        logger = _logger(prisma=prisma)
        logger._jobs_by_key = {"key-hash": _job()}

        await logger._refresh_active_jobs()

        assert set(logger._jobs_by_key) == {"key-hash"}

    async def test_start_lifecycle_loop_is_idempotent(self):
        logger = _logger(prisma=_prisma())
        logger.start_lifecycle_loop()
        first: asyncio.Task = logger._lifecycle_task
        logger.start_lifecycle_loop()
        assert logger._lifecycle_task is first
        first.cancel()
