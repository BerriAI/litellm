"""Unit tests for the shadow-eval logger: sampling, verdict parsing, unmasking, and the success hook's skip paths."""

import asyncio
import json
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.constants import INTERNAL_CALL_ORIGIN_METADATA_KEY
from litellm.integrations.shadow_eval_logger import (
    JUDGE_MAX_OUTPUT_TOKENS,
    ActiveShadowEvalJob,
    ShadowEvalLogger,
    _CallFailure,
    _JudgeVerdict,
    _key_or_team_is_over_budget,
    _parse_pairwise_verdict,
    _sample_hits,
    _ShadowResponse,
    _unmask_preference,
)
from litellm.types.utils import SHADOW_EVAL_JUDGE_CALL_ORIGIN, SHADOW_EVAL_ROUTER_CALL_ORIGIN


class TestSampling:
    def test_zero_percent_never_samples(self):
        assert not any(_sample_hits(f"req-{i}", "job", 0.0) for i in range(100))

    def test_hundred_percent_always_samples(self):
        assert all(_sample_hits(f"req-{i}", "job", 100.0) for i in range(100))

    def test_deterministic_for_same_inputs(self):
        results = [_sample_hits("req-1", "job-1", 50.0) for _ in range(10)]
        assert len(set(results)) == 1

    def test_distribution_close_to_percentage(self):
        hits = sum(_sample_hits(f"req-{i}", "job-x", 10.0) for i in range(10_000))
        assert 800 < hits < 1200

    def test_different_jobs_sample_independently(self):
        # The same request under different jobs should not always agree.
        agreements = sum(
            _sample_hits(f"req-{i}", "job-a", 50.0) == _sample_hits(f"req-{i}", "job-b", 50.0) for i in range(1000)
        )
        assert 300 < agreements < 700


class TestUnmaskPreference:
    @pytest.mark.parametrize(
        "raw,real_is_a,expected",
        [
            ("A", True, "real"),
            ("a", True, "real"),
            ("A", False, "shadow"),
            ("B", True, "shadow"),
            ("B", False, "real"),
            ("tie", True, "tie"),
            ("TIE", False, "tie"),
            ("garbage", True, "tie"),
            ("", False, "tie"),
        ],
    )
    def test_unmask(self, raw, real_is_a, expected):
        assert _unmask_preference(raw, real_is_a) == expected


class TestParsePairwiseVerdict:
    def test_plain_json(self):
        v = _parse_pairwise_verdict('{"preference": "A", "confidence": 0.9, "reasoning": "clearer"}')
        assert v.preference == "A"
        assert v.confidence == 0.9

    def test_fenced_json(self):
        raw = 'Here is my verdict:\n```json\n{"preference": "B", "confidence": 0.7, "reasoning": "x"}\n```\nDone.'
        assert _parse_pairwise_verdict(raw).preference == "B"

    def test_json_with_surrounding_prose(self):
        raw = 'Verdict: {"preference": "tie", "confidence": 0.5, "reasoning": "same"} — final.'
        assert _parse_pairwise_verdict(raw).preference == "tie"

    def test_non_object_raises(self):
        with pytest.raises(ValueError):
            _parse_pairwise_verdict('["not", "an", "object"]')

    def test_garbage_raises(self):
        with pytest.raises((json.JSONDecodeError, ValueError)):
            _parse_pairwise_verdict("no json here at all")


@pytest.mark.asyncio
class TestJudgeOutputBudget:
    async def test_judge_call_uses_the_named_output_budget(self, monkeypatch: pytest.MonkeyPatch):
        """Regression: a 200-token budget truncated ~12% of verdicts mid-JSON, losing them to failed_count."""
        import litellm as litellm_module

        acompletion = AsyncMock(
            return_value={
                "choices": [{"message": {"content": '{"preference": "A", "confidence": 0.8, "reasoning": "clearer"}'}}]
            }
        )
        monkeypatch.setattr(litellm_module, "acompletion", acompletion)

        logger = ShadowEvalLogger(router_provider=_router_mock, prisma_provider=lambda: MagicMock())
        verdict = await logger._call_judge(
            "gpt-4o-mini", [{"role": "user", "content": "hi"}], "real text", "shadow text", {}
        )

        assert not isinstance(verdict, _CallFailure)
        _, kwargs = acompletion.call_args
        assert kwargs["max_tokens"] == JUDGE_MAX_OUTPUT_TOKENS


class TestJudgeOutputBudgetStaticChecks:
    def test_budget_leaves_room_for_the_reasoning_field(self):
        """A tight budget truncates the JSON object the judge is asked to emit."""
        assert JUDGE_MAX_OUTPUT_TOKENS >= 500


def _job_record(job: ActiveShadowEvalJob, api_key_id: str = "key-hash") -> MagicMock:
    """A Prisma row double mirroring an ActiveShadowEvalJob, for snapshot refreshes."""
    record = MagicMock()
    record.id = job.id
    record.api_key_id = api_key_id
    record.router_name = job.router_name
    record.shadow_percentage = job.shadow_percentage
    record.judge_model = job.judge_model
    record.status = job.status
    record.cost_estimate = job.cost_estimate
    record.cost_actual = job.cost_actual
    record.ends_at = job.ends_at
    return record


def _router_mock(resolves_judge: bool = False):
    """A router double that, by default, does not resolve judge names, so judge
    dispatch falls through to the SDK exactly as it did for public model names."""
    router = MagicMock()
    router.model_group_alias = {}
    router.get_model_list = MagicMock(return_value=[{"litellm_params": {"model": "openai/gpt-4o"}}] if resolves_judge else None)
    return router


def _logger_with_mocks(job=None):
    prisma = MagicMock()
    router = _router_mock()
    logger = ShadowEvalLogger(router_provider=lambda: router, prisma_provider=lambda: prisma)
    if job is not None:
        logger._jobs_by_key["key-hash"] = job
    return logger, prisma, router


@pytest.mark.asyncio
class TestSuccessHookSkipPaths:
    async def test_skips_without_standard_logging_object(self):
        logger, prisma, _ = _logger_with_mocks()
        await logger.async_log_success_event({"messages": []}, MagicMock(), None, None)
        prisma.db.litellm_shadowevaljob.find_many.assert_not_called()

    @pytest.mark.parametrize(
        "origin",
        [SHADOW_EVAL_ROUTER_CALL_ORIGIN, SHADOW_EVAL_JUDGE_CALL_ORIGIN, "autorouter_classifier"],
    )
    async def test_skips_any_internal_sub_call(self, origin):
        """Regression: the auto-router's own classifier calls are logged as ordinary
        successes on the shadowed key; sampling them judges a tier label against a
        shadow answer and burns judge spend on traffic no user sent. One stamp,
        internal_call_origin, classifies every internal sub-call, including this
        logger's own shadow/judge traffic (the recursion guard)."""
        job = ActiveShadowEvalJob(id="j1", router_name="r", shadow_percentage=100.0, judge_model="m", status="running")
        logger, prisma, _ = _logger_with_mocks(job)
        kwargs = {
            "standard_logging_object": {
                "id": "req-1",
                "model": "gpt-4o",
                "metadata": {"user_api_key_hash": "key-hash"},
            },
            "litellm_params": {"metadata": {INTERNAL_CALL_ORIGIN_METADATA_KEY: origin}},
            "messages": [{"role": "user", "content": "hi"}],
        }
        await logger.async_log_success_event(kwargs, MagicMock(), None, None)
        assert logger._pending_seen == {}

    async def test_skips_when_no_active_job(self):
        logger, prisma, _ = _logger_with_mocks()
        prisma.db.litellm_shadowevaljob.find_many = AsyncMock(return_value=[])
        kwargs = {
            "standard_logging_object": {
                "id": "req-1",
                "model": "gpt-4o",
                "metadata": {"user_api_key_hash": "key-hash"},
            },
            "litellm_params": {"metadata": {}},
            "messages": [{"role": "user", "content": "hi"}],
        }
        await logger.async_log_success_event(kwargs, MagicMock(), None, None)
        assert logger._pending_seen == {}

    async def test_counts_seen_for_active_job(self):
        job = ActiveShadowEvalJob(id="j1", router_name="r", shadow_percentage=0.0, judge_model="m", status="running")
        logger, _, _ = _logger_with_mocks(job)
        kwargs = {
            "standard_logging_object": {
                "id": "req-1",
                "model": "gpt-4o",
                "call_type": "acompletion",
                "metadata": {"user_api_key_hash": "key-hash"},
            },
            "litellm_params": {"metadata": {}},
            "messages": [{"role": "user", "content": "hi"}],
        }
        await logger.async_log_success_event(kwargs, MagicMock(), None, None)
        # 0% sampling: request seen but never shadowed.
        assert logger._pending_seen == {"j1": 1}

    @staticmethod
    def _routed_kwargs(router_model_name: str):
        return {
            "standard_logging_object": {
                "id": "req-1",
                "model": "gpt-4o-mini",
                "call_type": "acompletion",
                "metadata": {"user_api_key_hash": "key-hash"},
            },
            "litellm_params": {"metadata": {"routing_decision": {"router_model_name": router_model_name}}},
            "messages": [{"role": "user", "content": "hi"}],
        }

    async def test_skips_requests_already_served_by_the_shadowed_router(self):
        """Duplicating the router's own traffic compares it to itself: paid ties, no signal."""
        job = ActiveShadowEvalJob(
            id="j1", router_name="claude-auto", shadow_percentage=100.0, judge_model="m", status="running"
        )
        logger, _, router = _logger_with_mocks(job)
        logger._run_shadow_eval = AsyncMock()

        await logger.async_log_success_event(self._routed_kwargs("claude-auto"), MagicMock(), None, None)
        seen_before_flush = dict(logger._pending_seen)
        await asyncio.sleep(0)

        logger._run_shadow_eval.assert_not_awaited()
        router.acompletion.assert_not_called()
        assert seen_before_flush == {"j1": 1}, "skipped for judging, but still counted toward requests seen"

    async def test_traffic_from_a_different_router_still_samples(self):
        job = ActiveShadowEvalJob(
            id="j1", router_name="claude-auto", shadow_percentage=100.0, judge_model="m", status="running"
        )
        logger, _, _ = _logger_with_mocks(job)
        logger._run_shadow_eval = AsyncMock()

        await logger.async_log_success_event(self._routed_kwargs("other-router"), MagicMock(), None, None)
        await asyncio.sleep(0.01)

        logger._run_shadow_eval.assert_awaited_once()


@pytest.mark.asyncio
class TestCallRouterShadowForwardsParameters:
    async def test_forwards_non_default_params(self):
        router = _router_mock()
        router.acompletion = AsyncMock(return_value={"choices": [{"message": {"content": "shadow reply"}}]})
        logger = ShadowEvalLogger(router_provider=lambda: router, prisma_provider=lambda: MagicMock())

        await logger._call_router_shadow(
            "claude-auto",
            [{"role": "user", "content": "hi"}],
            {"temperature": 0.2, "tools": [{"type": "function"}], "max_tokens": 500},
            {},
        )

        _, kwargs = router.acompletion.call_args
        assert kwargs["temperature"] == 0.2
        assert kwargs["tools"] == [{"type": "function"}]
        assert kwargs["max_tokens"] == 500

    async def test_drops_stream_and_metadata_from_forwarded_params(self):
        router = _router_mock()
        router.acompletion = AsyncMock(return_value={"choices": [{"message": {"content": "shadow reply"}}]})
        logger = ShadowEvalLogger(router_provider=lambda: router, prisma_provider=lambda: MagicMock())

        await logger._call_router_shadow(
            "claude-auto",
            [{"role": "user", "content": "hi"}],
            {"stream": True, "metadata": {"user_api_key_hash": "leaked"}, "temperature": 0.5},
            {},
        )

        _, kwargs = router.acompletion.call_args
        assert "stream" not in kwargs
        assert kwargs["temperature"] == 0.5
        # model_parameters carries a stale copy of the parent's metadata; the shadow call
        # builds its own from the parent metadata argument instead.
        assert "leaked" not in str(kwargs["metadata"])
        assert kwargs["metadata"][INTERNAL_CALL_ORIGIN_METADATA_KEY] == SHADOW_EVAL_ROUTER_CALL_ORIGIN


@pytest.mark.asyncio
class TestInflightTaskBacklog:
    async def test_drops_sample_when_at_capacity(self):
        job = ActiveShadowEvalJob(id="j1", router_name="r", shadow_percentage=100.0, judge_model="m", status="running")
        logger, _, _ = _logger_with_mocks(job)
        logger._inflight_shadow_tasks = 999999  # simulate saturation regardless of the real cap

        kwargs = {
            "standard_logging_object": {
                "id": "req-1",
                "model": "gpt-4o",
                "call_type": "acompletion",
                "metadata": {"user_api_key_hash": "key-hash"},
            },
            "litellm_params": {"metadata": {}},
            "messages": [{"role": "user", "content": "hi"}],
        }
        before = logger._inflight_shadow_tasks
        await logger.async_log_success_event(kwargs, MagicMock(), None, None)
        # No task should have been scheduled; the counter is untouched by the hook itself.
        assert logger._inflight_shadow_tasks == before

    async def test_schedules_and_decrements_when_under_capacity(self):
        job = ActiveShadowEvalJob(id="j1", router_name="r", shadow_percentage=100.0, judge_model="m", status="running")
        logger, _, router = _logger_with_mocks(job)
        router.acompletion = AsyncMock(side_effect=asyncio.sleep(0))  # keep the task alive briefly

        async def fake_run(*args, **kwargs):
            await asyncio.sleep(0.01)

        logger._run_shadow_eval = AsyncMock(side_effect=fake_run)

        kwargs = {
            "standard_logging_object": {
                "id": "req-1",
                "model": "gpt-4o",
                "call_type": "acompletion",
                "metadata": {"user_api_key_hash": "key-hash"},
            },
            "litellm_params": {"metadata": {}},
            "messages": [{"role": "user", "content": "hi"}],
        }
        await logger.async_log_success_event(kwargs, MagicMock(), None, None)
        assert logger._inflight_shadow_tasks == 1
        await asyncio.sleep(0.05)
        assert logger._inflight_shadow_tasks == 0


@pytest.mark.asyncio
class TestStoppedJobCannotBeReactivated:
    async def test_verdict_write_uses_conditional_status_guard(self):
        """Regression: a verdict written after stop() must not resurrect a completed job.

        stop_shadow_eval_job() sets status="completed". If a pipeline that started
        before the stop finishes afterwards, its counter update must not blindly
        set status back to "running" -- it has to filter on the job still being
        pending/running, so an already-completed job stays completed.
        """
        prisma = MagicMock()
        prisma.db.litellm_shadowevalverdict.create = AsyncMock()
        prisma.db.litellm_shadowevaljob.update_many = AsyncMock()
        # If the fix regresses to `.update(...)`, this test must fail loudly
        # rather than silently pass by mocking away the missing method.
        del prisma.db.litellm_shadowevaljob.update

        logger = ShadowEvalLogger(router_provider=lambda: MagicMock(), prisma_provider=lambda: prisma)
        logger._call_router_shadow = AsyncMock(
            return_value=_ShadowResponse(text="shadow text", model="shadow-model", tier="SIMPLE", completion_tokens=10, request_id="shadow-req-1")
        )
        logger._call_judge = AsyncMock(
            return_value=_JudgeVerdict(preference="real", confidence=0.9, reasoning="clearer", cost=0.01)
        )

        job = ActiveShadowEvalJob(id="j1", router_name="r", judge_model="m", shadow_percentage=100.0, status="running")
        await logger._run_shadow_eval(
            job=job,
            request_id="req-1",
            messages=[{"role": "user", "content": "hi"}],
            response_obj={"choices": [{"message": {"content": "real text"}}]},
            real_model="gpt-4o",
            real_response_tokens=42,
            model_parameters={},
            parent_metadata={},
        )

        prisma.db.litellm_shadowevaljob.update_many.assert_awaited_once()
        _, call_kwargs = prisma.db.litellm_shadowevaljob.update_many.call_args
        where = call_kwargs["where"]
        assert where["id"] == "j1"
        assert set(where["status"]["in"]) == {"pending", "running"}
        assert call_kwargs["data"]["status"] == "running"


@pytest.mark.asyncio
class TestNoPaidCallsWithoutAPlaceToRecordThem:
    async def test_pipeline_spends_nothing_when_prisma_is_unavailable(self):
        """Regression: the prisma check sat below the shadow and judge dispatch, so a
        DB outage mid-flight paid for both provider calls and then dropped the
        verdict. No verdict storage means no paid calls at all."""
        router = _router_mock()
        router.acompletion = AsyncMock()
        logger = ShadowEvalLogger(router_provider=lambda: router, prisma_provider=lambda: None)
        logger._call_judge = AsyncMock()

        job = ActiveShadowEvalJob(id="j1", router_name="r", judge_model="m", shadow_percentage=100.0, status="running")
        await logger._run_shadow_eval(
            job=job,
            request_id="req-1",
            messages=[{"role": "user", "content": "hi"}],
            response_obj={"choices": [{"message": {"content": "real text"}}]},
            real_model="gpt-4o",
            real_response_tokens=42,
            model_parameters={},
            parent_metadata={},
        )

        router.acompletion.assert_not_awaited()
        logger._call_judge.assert_not_awaited()


@pytest.mark.asyncio
class TestVerdictRowRecordsBothSides:
    async def test_verdict_row_carries_ids_and_token_counts_for_both_arms(self):
        """real_response_tokens next to shadow_response_tokens is what lets a reader
        check the judge's verdicts for verbosity bias, and shadow_request_id joins the
        verdict to the shadow call's own spend log for drill-down. Declared columns
        that are never written are worse than absent ones."""
        prisma = MagicMock()
        prisma.db.litellm_shadowevalverdict.create = AsyncMock()
        prisma.db.litellm_shadowevaljob.update_many = AsyncMock()
        logger = ShadowEvalLogger(router_provider=_router_mock, prisma_provider=lambda: prisma)
        logger._call_router_shadow = AsyncMock(
            return_value=_ShadowResponse(
                text="shadow text", model="shadow-model", tier="SIMPLE", completion_tokens=10, request_id="shadow-req-1"
            )
        )
        logger._call_judge = AsyncMock(
            return_value=_JudgeVerdict(preference="real", confidence=0.9, reasoning="clearer", cost=0.01)
        )

        job = ActiveShadowEvalJob(id="j1", router_name="r", judge_model="m", shadow_percentage=100.0, status="running")
        await logger._run_shadow_eval(
            job=job,
            request_id="req-1",
            messages=[{"role": "user", "content": "hi"}],
            response_obj={"choices": [{"message": {"content": "real text"}}]},
            real_model="gpt-4o",
            real_response_tokens=42,
            model_parameters={},
            parent_metadata={},
        )

        data = prisma.db.litellm_shadowevalverdict.create.call_args.kwargs["data"]
        assert data["shadow_request_id"] == "shadow-req-1"
        assert data["real_response_tokens"] == 42
        assert data["shadow_response_tokens"] == 10


@pytest.mark.asyncio
class TestVerdictWriteAccumulatesCost:
    async def test_cost_actual_uses_increment_not_a_raw_set(self):
        """Regression: cost_actual must accumulate across verdicts, not overwrite.

        The DB column defaults to 0 (not NULL) precisely so this increment lands
        on a real number instead of NULL + x = NULL. If this update ever
        regresses to a flat `"cost_actual": judge_cost` assignment, only the
        last verdict's cost would survive instead of the running total.
        """
        prisma = MagicMock()
        prisma.db.litellm_shadowevalverdict.create = AsyncMock()
        prisma.db.litellm_shadowevaljob.update_many = AsyncMock()

        logger = ShadowEvalLogger(router_provider=lambda: MagicMock(), prisma_provider=lambda: prisma)
        logger._call_router_shadow = AsyncMock(
            return_value=_ShadowResponse(text="shadow text", model="shadow-model", tier="SIMPLE", completion_tokens=10, request_id="shadow-req-1")
        )
        logger._call_judge = AsyncMock(
            return_value=_JudgeVerdict(preference="real", confidence=0.9, reasoning="clearer", cost=0.05)
        )

        job = ActiveShadowEvalJob(id="j1", router_name="r", judge_model="m", shadow_percentage=100.0, status="running")
        await logger._run_shadow_eval(
            job=job,
            request_id="req-1",
            messages=[{"role": "user", "content": "hi"}],
            response_obj={"choices": [{"message": {"content": "real text"}}]},
            real_model="gpt-4o",
            real_response_tokens=42,
            model_parameters={},
            parent_metadata={},
        )

        _, call_kwargs = prisma.db.litellm_shadowevaljob.update_many.call_args
        assert call_kwargs["data"]["cost_actual"] == {"increment": 0.05}


_PARENT_METADATA = {
    "user_api_key": "hashed-key",
    "user_api_key_alias": "team-prod-key",
    "user_api_key_team_id": "team-1",
    "user_api_key_org_id": "org-1",
    "user_api_key_user_id": "user-1",
    "user_api_key_end_user_id": "end-user-1",
    "user_api_key_budget_reservation": {"reserved_cost": 2.5, "finalized": False},
    "user_api_key_auth": {"models": ["gpt-4o"], "budget_reservation": {"reserved_cost": 2.5}},
    "routing_decision": {"tier": "SIMPLE"},
    "standard_logging_object": {"id": "parent-req"},
}

_IDENTITY_KEYS = (
    "user_api_key",
    "user_api_key_alias",
    "user_api_key_team_id",
    "user_api_key_org_id",
    "user_api_key_user_id",
    "user_api_key_end_user_id",
)


def _assert_attributed(metadata, expected_origin):
    """Every assertion the proxy's cost callback depends on to bill an internal sub-call."""
    for key in _IDENTITY_KEYS:
        assert metadata[key] == _PARENT_METADATA[key], f"{key} must reach the sub-call for spend attribution"
    assert metadata[INTERNAL_CALL_ORIGIN_METADATA_KEY] == expected_origin
    # The parent completion owns this reservation. A sub-call carrying it would let the
    # sub-call's cost callback finalize it, so the parent's own callback skips
    # incrementing the key/team counters and the parent's spend is lost.
    assert "user_api_key_budget_reservation" not in metadata
    assert "budget_reservation" not in metadata["user_api_key_auth"]
    assert metadata["user_api_key_auth"]["models"] == ["gpt-4o"]
    # Per-request state of a request that already finished says nothing true about this one.
    assert "routing_decision" not in metadata
    assert "standard_logging_object" not in metadata


@pytest.mark.asyncio
class TestSubCallsAreAttributedToTheShadowedKey:
    """Shadow and judge calls bill real provider spend nobody typed a prompt for.

    Without the caller's identity on their metadata, _PROXY_track_cost_callback drops
    the spend log and no budget counter moves, so an admin-enabled eval spends against
    a customer's key invisibly and past every configured limit.
    """

    async def test_judge_call_carries_the_key_identity_without_its_reservation(self, monkeypatch: pytest.MonkeyPatch):
        import litellm as litellm_module

        acompletion = AsyncMock(
            return_value={
                "choices": [{"message": {"content": '{"preference": "A", "confidence": 0.8, "reasoning": "clearer"}'}}]
            }
        )
        monkeypatch.setattr(litellm_module, "acompletion", acompletion)
        logger = ShadowEvalLogger(router_provider=_router_mock, prisma_provider=lambda: MagicMock())

        verdict = await logger._call_judge(
            "gpt-4o-mini", [{"role": "user", "content": "hi"}], "real text", "shadow text", _PARENT_METADATA
        )

        assert not isinstance(verdict, _CallFailure)
        metadata = acompletion.call_args.kwargs["metadata"]
        _assert_attributed(metadata, SHADOW_EVAL_JUDGE_CALL_ORIGIN)

    async def test_shadow_router_call_carries_the_key_identity_and_the_recursion_guard(self):
        router = _router_mock()
        router.acompletion = AsyncMock(return_value={"choices": [{"message": {"content": "shadow reply"}}]})
        logger = ShadowEvalLogger(router_provider=lambda: router, prisma_provider=lambda: MagicMock())

        result = await logger._call_router_shadow(
            "claude-auto", [{"role": "user", "content": "hi"}], {}, _PARENT_METADATA
        )

        assert not isinstance(result, _CallFailure)
        metadata = router.acompletion.call_args.kwargs["metadata"]
        # The origin stamp doubles as the recursion guard: the hook skips any
        # request carrying it, so the shadow call cannot shadow itself.
        _assert_attributed(metadata, SHADOW_EVAL_ROUTER_CALL_ORIGIN)

    async def test_shadow_metadata_stays_writable_for_the_routing_decision_read_back(self):
        """Tier attribution reads routing_decision back out of the dict the router was given."""
        router = _router_mock()

        async def acompletion(**kwargs):
            kwargs["metadata"]["routing_decision"] = {"tier_label": "COMPLEX", "routed_model": "o1"}
            return {"choices": [{"message": {"content": "shadow reply"}}]}

        router.acompletion = acompletion
        logger = ShadowEvalLogger(router_provider=lambda: router, prisma_provider=lambda: MagicMock())

        result = await logger._call_router_shadow(
            "claude-auto", [{"role": "user", "content": "hi"}], {}, _PARENT_METADATA
        )

        assert not isinstance(result, _CallFailure)
        assert result.tier == "COMPLEX"

    async def test_parent_metadata_reaches_both_legs_from_the_success_hook(self):
        """The hook is the only place the parent's metadata exists; a break here is invisible
        to unit tests of the two legs in isolation."""
        prisma = MagicMock()
        prisma.db.litellm_shadowevalverdict.create = AsyncMock()
        prisma.db.litellm_shadowevaljob.update_many = AsyncMock()
        job = ActiveShadowEvalJob(id="j1", router_name="r", shadow_percentage=100.0, judge_model="m", status="running")
        logger, _, _ = _logger_with_mocks(job)
        logger._prisma_provider = lambda: prisma
        logger._call_router_shadow = AsyncMock(
            return_value=_ShadowResponse(text="shadow text", model="shadow-model", tier="SIMPLE", completion_tokens=10, request_id="shadow-req-1")
        )
        logger._call_judge = AsyncMock(
            return_value=_JudgeVerdict(preference="real", confidence=0.9, reasoning="clearer", cost=0.01)
        )

        await logger.async_log_success_event(
            {
                "standard_logging_object": {
                    "id": "req-1",
                    "model": "gpt-4o",
                    "call_type": "acompletion",
                    "metadata": {"user_api_key_hash": "key-hash"},
                },
                "litellm_params": {"metadata": dict(_PARENT_METADATA)},
                "messages": [{"role": "user", "content": "hi"}],
            },
            {"choices": [{"message": {"content": "real text"}}]},
            None,
            None,
        )
        await asyncio.sleep(0.05)

        assert logger._call_router_shadow.await_args.args[3]["user_api_key"] == "hashed-key"
        assert logger._call_judge.await_args.kwargs["parent_metadata"]["user_api_key"] == "hashed-key"


@pytest.mark.asyncio
class TestKeyOrTeamIsOverBudget:
    """Shadow/judge calls run outside the normal auth path and never reserve budget
    for themselves, so an already-exhausted key or team must not be pushed further
    over by a background eval it never asked to run."""

    @staticmethod
    def _metadata(**overrides):
        return {
            "user_api_key_hash": "key-hash",
            "user_api_key_max_budget": 10.0,
            "user_api_key_spend": 4.0,
            **overrides,
        }

    async def test_under_key_budget_is_not_over_budget(self):
        with patch("litellm.proxy.proxy_server.get_current_spend", AsyncMock(return_value=4.0)):
            assert not await _key_or_team_is_over_budget(self._metadata())

    async def test_at_key_budget_is_over_budget(self):
        with patch("litellm.proxy.proxy_server.get_current_spend", AsyncMock(return_value=10.0)):
            assert await _key_or_team_is_over_budget(self._metadata())

    async def test_reads_the_cross_pod_counter_not_the_stale_metadata_spend(self):
        get_current_spend = AsyncMock(return_value=11.0)
        with patch("litellm.proxy.proxy_server.get_current_spend", get_current_spend):
            assert await _key_or_team_is_over_budget(self._metadata(user_api_key_spend=0.0))
        get_current_spend.assert_awaited_once_with(
            counter_key="spend:key:key-hash", fallback_spend=0.0, max_budget=10.0
        )

    async def test_team_over_budget_is_caught_even_when_key_has_room(self):
        metadata = self._metadata(
            user_api_key_team_id="team-1",
            user_api_key_team_max_budget=5.0,
            user_api_key_team_spend=5.0,
        )

        async def fake_spend(counter_key, fallback_spend, max_budget):
            return max_budget if counter_key == "spend:team:team-1" else 1.0

        with patch("litellm.proxy.proxy_server.get_current_spend", AsyncMock(side_effect=fake_spend)):
            assert await _key_or_team_is_over_budget(metadata)

    async def test_no_budget_configured_is_never_over_budget(self):
        assert not await _key_or_team_is_over_budget({"user_api_key_hash": "key-hash"})

    async def test_missing_proxy_server_fails_open_rather_than_blocking_logging(self):
        with patch.dict("sys.modules", {"litellm.proxy.proxy_server": None}):
            assert not await _key_or_team_is_over_budget(self._metadata())


@pytest.mark.asyncio
class TestBudgetGateRunsInTheDetachedTaskNotTheCallback:
    """get_current_spend can fall back to an authoritative DB read; the production
    success callback must never await that, so the gate lives in the detached task."""

    @staticmethod
    def _kwargs(spend: float):
        return {
            "standard_logging_object": {
                "id": "req-1",
                "model": "gpt-4o",
                "call_type": "acompletion",
                "metadata": {
                    "user_api_key_hash": "key-hash",
                    "user_api_key_max_budget": 10.0,
                    "user_api_key_spend": spend,
                },
            },
            "litellm_params": {"metadata": {}},
            "messages": [{"role": "user", "content": "hi"}],
        }

    @staticmethod
    def _job():
        return ActiveShadowEvalJob(id="j1", router_name="r", shadow_percentage=100.0, judge_model="m", status="running")

    async def test_over_budget_key_never_fires_a_shadow_call(self):
        logger, _, router = _logger_with_mocks(self._job())
        response = {"choices": [{"message": {"content": "real answer"}}]}

        with patch("litellm.proxy.proxy_server.get_current_spend", AsyncMock(return_value=10.0)):
            await logger.async_log_success_event(self._kwargs(spend=10.0), response, None, None)
            await asyncio.sleep(0.05)

        router.acompletion.assert_not_called()

    async def test_under_budget_key_fires_the_shadow_call(self):
        logger, _, router = _logger_with_mocks(self._job())
        router.acompletion = AsyncMock(return_value={"choices": [{"message": {"content": "shadow"}}]})
        response = {"choices": [{"message": {"content": "real answer"}}]}

        with patch("litellm.proxy.proxy_server.get_current_spend", AsyncMock(return_value=4.0)):
            await logger.async_log_success_event(self._kwargs(spend=4.0), response, None, None)
            await asyncio.sleep(0.05)

        router.acompletion.assert_awaited_once()

    async def test_the_callback_itself_never_awaits_the_spend_read(self):
        """The spend read must happen after the callback returns, inside the task."""
        logger, _, _ = _logger_with_mocks(self._job())
        get_current_spend = AsyncMock(return_value=0.0)
        response = {"choices": [{"message": {"content": "real answer"}}]}

        with patch("litellm.proxy.proxy_server.get_current_spend", get_current_spend):
            await logger.async_log_success_event(self._kwargs(spend=0.0), response, None, None)
            spend_reads_when_callback_returned = get_current_spend.await_count
            await asyncio.sleep(0.05)

        assert spend_reads_when_callback_returned == 0
        assert get_current_spend.await_count > 0


@pytest.mark.asyncio
class TestPerJobSpendCap:
    """Budgets bound the key; the cap bounds a single eval, so a bad estimate or a
    traffic spike cannot turn a quoted eval into a much larger bill."""

    @staticmethod
    def _success_kwargs():
        return {
            "standard_logging_object": {
                "id": "req-1",
                "model": "gpt-4o",
                "call_type": "acompletion",
                "metadata": {"user_api_key_hash": "key-hash"},
            },
            "litellm_params": {"metadata": {}},
            "messages": [{"role": "user", "content": "hi"}],
        }

    async def test_job_over_the_cap_stops_sampling(self):
        job = ActiveShadowEvalJob(
            id="j1",
            router_name="r",
            shadow_percentage=100.0,
            judge_model="m",
            status="running",
            cost_estimate=10.0,
            cost_actual=15.0,
        )
        logger, prisma, router = _logger_with_mocks(job)
        logger._run_shadow_eval = AsyncMock()

        await logger.async_log_success_event(self._success_kwargs(), MagicMock(), None, None)
        await asyncio.sleep(0)

        logger._run_shadow_eval.assert_not_awaited()
        router.acompletion.assert_not_called()

    async def test_lifecycle_tick_completes_a_job_over_the_cap(self):
        """The cap is enforced by the loop, so it lands even on a key gone quiet."""
        job = ActiveShadowEvalJob(
            id="j1",
            router_name="r",
            shadow_percentage=100.0,
            judge_model="m",
            status="running",
            cost_estimate=10.0,
            cost_actual=15.0,
        )
        logger, prisma, _ = _logger_with_mocks(job)
        prisma.db.litellm_shadowevaljob.update_many = AsyncMock()
        prisma.db.litellm_shadowevaljob.find_many = AsyncMock(return_value=[_job_record(job)])

        await logger._lifecycle_tick()

        call_kwargs = prisma.db.litellm_shadowevaljob.update_many.call_args.kwargs
        assert call_kwargs["where"]["id"] == "j1"
        # Guarded so a pod breaching the cap cannot resurrect a job an admin already stopped.
        assert set(call_kwargs["where"]["status"]["in"]) == {"pending", "running"}
        assert call_kwargs["data"]["status"] == "completed"
        assert call_kwargs["data"]["completed_at"] is not None

    async def test_job_just_under_the_cap_keeps_evaluating(self):
        job = ActiveShadowEvalJob(
            id="j1",
            router_name="r",
            shadow_percentage=100.0,
            judge_model="m",
            status="running",
            cost_estimate=10.0,
            cost_actual=14.99,
        )
        logger, prisma, _ = _logger_with_mocks(job)
        logger._run_shadow_eval = AsyncMock()

        await logger.async_log_success_event(self._success_kwargs(), MagicMock(), None, None)
        await asyncio.sleep(0.01)

        logger._run_shadow_eval.assert_awaited_once()

    async def test_job_without_an_estimate_is_uncapped(self):
        """A missing estimate is no multiple to compare against; treating it as 0 would
        stop every such job on its first request instead."""
        job = ActiveShadowEvalJob(
            id="j1",
            router_name="r",
            shadow_percentage=100.0,
            judge_model="m",
            status="running",
            cost_estimate=None,
            cost_actual=500.0,
        )
        logger, prisma, _ = _logger_with_mocks(job)
        logger._run_shadow_eval = AsyncMock()

        await logger.async_log_success_event(self._success_kwargs(), MagicMock(), None, None)
        await asyncio.sleep(0.01)

        logger._run_shadow_eval.assert_awaited_once()

    async def test_a_zero_estimate_job_is_still_capped_at_the_floor(self):
        """A key quiet during the lookback gets estimate $0.00; a later traffic spike on
        exactly that job must still hit the floor instead of billing until ends_at."""
        job = ActiveShadowEvalJob(
            id="j1",
            router_name="r",
            shadow_percentage=100.0,
            judge_model="m",
            status="running",
            cost_estimate=0.0,
            cost_actual=1.5,
        )
        logger, prisma, _ = _logger_with_mocks(job)
        prisma.db.litellm_shadowevaljob.update_many = AsyncMock()
        prisma.db.litellm_shadowevaljob.find_many = AsyncMock(return_value=[_job_record(job)])
        logger._run_shadow_eval = AsyncMock()

        await logger.async_log_success_event(self._success_kwargs(), MagicMock(), None, None)
        await asyncio.sleep(0)
        logger._run_shadow_eval.assert_not_awaited()

        await logger._lifecycle_tick()
        assert prisma.db.litellm_shadowevaljob.update_many.call_args.kwargs["data"]["status"] == "completed"

    async def test_a_cent_sized_estimate_is_not_stopped_by_its_first_verdict(self):
        """1.5x a $0.01 estimate is $0.015; without the floor a single judge call ends the job."""
        job = ActiveShadowEvalJob(
            id="j1",
            router_name="r",
            shadow_percentage=100.0,
            judge_model="m",
            status="running",
            cost_estimate=0.01,
            cost_actual=0.08,
        )
        logger, prisma, _ = _logger_with_mocks(job)
        prisma.db.litellm_shadowevaljob.update_many = AsyncMock()
        logger._run_shadow_eval = AsyncMock()

        await logger.async_log_success_event(self._success_kwargs(), MagicMock(), None, None)
        await asyncio.sleep(0.01)

        logger._run_shadow_eval.assert_awaited_once()

    async def test_finalizing_evicts_the_job_so_a_second_tick_does_not_rewrite_it(self):
        job = ActiveShadowEvalJob(
            id="j1",
            router_name="r",
            shadow_percentage=100.0,
            judge_model="m",
            status="running",
            cost_estimate=10.0,
            cost_actual=15.0,
        )
        logger, prisma, _ = _logger_with_mocks(job)
        prisma.db.litellm_shadowevaljob.update_many = AsyncMock()
        prisma.db.litellm_shadowevaljob.find_many = AsyncMock(side_effect=[[_job_record(job)], []])

        await logger._lifecycle_tick()
        await logger._lifecycle_tick()

        assert prisma.db.litellm_shadowevaljob.update_many.await_count == 1


@pytest.mark.asyncio
class TestActiveJobSnapshot:
    """One find_many per pod per tick serves every key, so DB load stays flat no
    matter how many distinct keys send traffic through the proxy."""

    async def test_hook_lookup_never_awaits_the_db(self):
        """The success hook reads the loop-maintained snapshot; a cold pod answers
        from the empty snapshot instead of blocking on Prisma."""
        logger, prisma, _ = _logger_with_mocks()
        prisma.db.litellm_shadowevaljob.find_many = AsyncMock(return_value=[])
        kwargs = {
            "standard_logging_object": {
                "id": "req-1",
                "model": "gpt-4o",
                "call_type": "acompletion",
                "metadata": {"user_api_key_hash": "key-1"},
            },
            "litellm_params": {"metadata": {}},
            "messages": [{"role": "user", "content": "hi"}],
        }
        await logger.async_log_success_event(kwargs, MagicMock(), None, None)
        prisma.db.litellm_shadowevaljob.find_many.assert_not_awaited()

    async def test_one_refresh_serves_many_keys(self):
        logger, prisma, _ = _logger_with_mocks()
        job = ActiveShadowEvalJob(id="j1", router_name="r", shadow_percentage=100.0, judge_model="m", status="running")
        prisma.db.litellm_shadowevaljob.find_many = AsyncMock(return_value=[_job_record(job, api_key_id="key-1")])

        await logger._refresh_active_jobs()

        assert logger._jobs_by_key["key-1"].id == "j1"
        assert all(logger._jobs_by_key.get(f"other-{i}") is None for i in range(50))
        assert prisma.db.litellm_shadowevaljob.find_many.await_count == 1

    async def test_newest_job_wins_when_a_key_somehow_has_two(self):
        logger, prisma, _ = _logger_with_mocks()
        newest = ActiveShadowEvalJob(
            id="j-newest", router_name="r", shadow_percentage=100.0, judge_model="m", status="running"
        )
        oldest = ActiveShadowEvalJob(
            id="j-oldest", router_name="r", shadow_percentage=100.0, judge_model="m", status="running"
        )
        prisma.db.litellm_shadowevaljob.find_many = AsyncMock(
            return_value=[_job_record(newest, "key-1"), _job_record(oldest, "key-1")]
        )

        await logger._refresh_active_jobs()

        assert logger._jobs_by_key["key-1"].id == "j-newest"

    async def test_db_blip_keeps_the_stale_snapshot_instead_of_disabling_the_feature(self):
        logger, prisma, _ = _logger_with_mocks()
        job = ActiveShadowEvalJob(id="j1", router_name="r", shadow_percentage=100.0, judge_model="m", status="running")
        prisma.db.litellm_shadowevaljob.find_many = AsyncMock(return_value=[_job_record(job, "key-1")])
        await logger._refresh_active_jobs()

        prisma.db.litellm_shadowevaljob.find_many = AsyncMock(side_effect=RuntimeError("db down"))
        await logger._refresh_active_jobs()

        assert logger._jobs_by_key["key-1"].id == "j1"

    async def test_a_failing_tick_does_not_kill_the_loop(self):
        logger, prisma, _ = _logger_with_mocks()
        prisma.db.litellm_shadowevaljob.find_many = AsyncMock(side_effect=RuntimeError("db down"))
        logger._pending_seen = {"j1": 3}
        prisma.db.litellm_shadowevaljob.update_many = AsyncMock(side_effect=RuntimeError("db down"))

        await logger._lifecycle_tick()

    async def test_start_lifecycle_loop_is_idempotent(self):
        logger, _, _ = _logger_with_mocks()
        logger.start_lifecycle_loop()
        first = logger._lifecycle_task
        logger.start_lifecycle_loop()
        assert logger._lifecycle_task is first
        first.cancel()


@pytest.mark.asyncio
class TestRequestCountFlush:
    """request_count is flushed by the loop, not by the next request arriving, so
    the final batch lands and a stopped job's counter freezes."""

    async def test_tick_flushes_buffered_counts_without_new_requests(self):
        """Regression: the old flush was piggybacked on a later request 10s+ after
        the last flush, so the final batch of a job was never written and idle jobs
        read 'N judged of 0 seen'."""
        logger, prisma, _ = _logger_with_mocks()
        prisma.db.litellm_shadowevaljob.update_many = AsyncMock()
        prisma.db.litellm_shadowevaljob.find_many = AsyncMock(return_value=[])
        logger._pending_seen = {"j1": 7}

        await logger._lifecycle_tick()

        flush_call = prisma.db.litellm_shadowevaljob.update_many.call_args_list[0]
        assert flush_call.kwargs["where"]["id"] == "j1"
        assert flush_call.kwargs["data"] == {"request_count": {"increment": 7}}
        assert logger._pending_seen == {}

    async def test_flush_is_guarded_on_the_job_still_being_active(self):
        """Stopping a job freezes its request_count: a pod serving a stale snapshot
        keeps buffering for up to one tick, and the status-guarded write drops those
        increments instead of growing a stopped job's counter for ~30s after stop."""
        logger, prisma, _ = _logger_with_mocks()
        prisma.db.litellm_shadowevaljob.update_many = AsyncMock()
        logger._pending_seen = {"j1": 2}

        await logger._flush_seen_counts()

        where = prisma.db.litellm_shadowevaljob.update_many.call_args.kwargs["where"]
        assert set(where["status"]["in"]) == {"pending", "running"}

    async def test_flush_before_finalize_lands_the_tail_batch_of_an_expiring_job(self):
        """The tick flushes first, so an expiring job's last counts pass the active
        guard before the same tick flips it to completed."""
        expired = ActiveShadowEvalJob(
            id="j1",
            router_name="r",
            shadow_percentage=100.0,
            judge_model="m",
            status="running",
            ends_at=datetime.now(timezone.utc) - timedelta(seconds=1),
        )
        logger, prisma, _ = _logger_with_mocks(expired)
        prisma.db.litellm_shadowevaljob.update_many = AsyncMock()
        prisma.db.litellm_shadowevaljob.find_many = AsyncMock(return_value=[_job_record(expired)])
        logger._pending_seen = {"j1": 4}

        await logger._lifecycle_tick()

        calls = prisma.db.litellm_shadowevaljob.update_many.call_args_list
        assert calls[0].kwargs["data"] == {"request_count": {"increment": 4}}
        assert calls[-1].kwargs["data"]["status"] == "completed"


@pytest.mark.asyncio
class TestJobsStopAtTheirScheduledEnd:
    """A shadow eval samples ongoing traffic, so a job whose window has closed must
    stop billing judge calls even if nobody remembers to stop it by hand, and even
    if its key never sends another request."""

    @staticmethod
    def _job(ends_at):
        return ActiveShadowEvalJob(
            id="j1",
            router_name="r",
            shadow_percentage=100.0,
            judge_model="m",
            status="running",
            cost_estimate=10.0,
            cost_actual=0.0,
            ends_at=ends_at,
        )

    async def test_job_past_its_end_stops_sampling(self):
        logger, prisma, router = _logger_with_mocks(self._job(datetime.now(timezone.utc) - timedelta(seconds=1)))
        logger._run_shadow_eval = AsyncMock()

        await logger.async_log_success_event(TestPerJobSpendCap._success_kwargs(), MagicMock(), None, None)
        await asyncio.sleep(0)

        logger._run_shadow_eval.assert_not_awaited()
        router.acompletion.assert_not_called()

    async def test_lifecycle_tick_completes_an_expired_job_with_no_traffic_at_all(self):
        """Regression: expiry used to be checked only inside the success hook, so a
        job on a key gone quiet stayed pending/running indefinitely."""
        expired = self._job(datetime.now(timezone.utc) - timedelta(seconds=1))
        logger, prisma, _ = _logger_with_mocks()
        prisma.db.litellm_shadowevaljob.update_many = AsyncMock()
        prisma.db.litellm_shadowevaljob.find_many = AsyncMock(return_value=[_job_record(expired)])

        await logger._lifecycle_tick()

        call_kwargs = prisma.db.litellm_shadowevaljob.update_many.call_args.kwargs
        assert call_kwargs["where"]["id"] == "j1"
        assert set(call_kwargs["where"]["status"]["in"]) == {"pending", "running"}
        assert call_kwargs["data"]["status"] == "completed"

    async def test_job_inside_its_window_keeps_evaluating(self):
        logger, prisma, _ = _logger_with_mocks(self._job(datetime.now(timezone.utc) + timedelta(hours=1)))
        prisma.db.litellm_shadowevaljob.update_many = AsyncMock()
        logger._run_shadow_eval = AsyncMock()

        await logger.async_log_success_event(TestPerJobSpendCap._success_kwargs(), MagicMock(), None, None)
        await asyncio.sleep(0.01)

        logger._run_shadow_eval.assert_awaited_once()
        prisma.db.litellm_shadowevaljob.update_many.assert_not_awaited()

    async def test_job_without_an_end_is_not_expired(self):
        logger, prisma, _ = _logger_with_mocks(self._job(None))
        prisma.db.litellm_shadowevaljob.update_many = AsyncMock()
        logger._run_shadow_eval = AsyncMock()

        await logger.async_log_success_event(TestPerJobSpendCap._success_kwargs(), MagicMock(), None, None)
        await asyncio.sleep(0.01)

        logger._run_shadow_eval.assert_awaited_once()

    async def test_naive_db_datetime_is_treated_as_utc(self):
        logger, prisma, _ = _logger_with_mocks()
        naive_expired = self._job(datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(seconds=1))
        prisma.db.litellm_shadowevaljob.find_many = AsyncMock(return_value=[_job_record(naive_expired)])

        await logger._refresh_active_jobs()
        job = logger._jobs_by_key["key-hash"]

        assert job.ends_at is not None and job.ends_at.tzinfo is not None
        from litellm.integrations.shadow_eval_logger import _job_is_past_its_end

        assert _job_is_past_its_end(job)


@pytest.mark.asyncio
class TestJudgeFailureModes:
    """A judge outage or malformed verdict must be a counted failure, never a crash,
    and must never write a verdict row."""

    @staticmethod
    def _logger():
        prisma = MagicMock()
        router = _router_mock()
        return ShadowEvalLogger(router_provider=lambda: router, prisma_provider=lambda: prisma), prisma, router

    async def test_judge_provider_error_is_a_described_failure(self, monkeypatch: pytest.MonkeyPatch):
        import litellm as litellm_module

        logger, _, _ = self._logger()
        monkeypatch.setattr(litellm_module, "acompletion", AsyncMock(side_effect=RuntimeError("provider down")))

        verdict = await logger._call_judge("m", [{"role": "user", "content": "hi"}], "real", "shadow", {})

        assert isinstance(verdict, _CallFailure)
        assert "provider down" in verdict.error

    async def test_unparseable_judge_output_is_a_described_failure(self, monkeypatch: pytest.MonkeyPatch):
        import litellm as litellm_module

        logger, _, _ = self._logger()
        monkeypatch.setattr(
            litellm_module,
            "acompletion",
            AsyncMock(return_value={"choices": [{"message": {"content": "I prefer response A because"}}]}),
        )

        verdict = await logger._call_judge("m", [{"role": "user", "content": "hi"}], "real", "shadow", {})

        assert isinstance(verdict, _CallFailure)
        assert "unparseable" in verdict.error

    async def test_failed_judge_bumps_failed_count_with_last_error_and_writes_no_verdict(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """Regression: a misconfigured judge model failed every turn with nothing but
        a debug log, so the admin saw only a growing failed_count and null results.
        The most recent failure is persisted on the job for the UI to show."""
        import litellm as litellm_module

        logger, prisma, router = self._logger()
        monkeypatch.setattr(litellm_module, "acompletion", AsyncMock(side_effect=RuntimeError("down")))
        router.acompletion = AsyncMock(return_value={"choices": [{"message": {"content": "shadow says"}}]})
        prisma.db.litellm_shadowevaljob.update_many = AsyncMock()
        job = ActiveShadowEvalJob(id="j1", router_name="r", shadow_percentage=100.0, judge_model="m", status="running")

        await logger._run_shadow_eval(
            job=job,
            request_id="req-1",
            messages=({"role": "user", "content": "hi"},),
            response_obj={"choices": [{"message": {"content": "real says"}}]},
            real_model="gpt-4o",
            real_response_tokens=42,
            model_parameters={},
            parent_metadata={},
        )

        prisma.db.litellm_shadowevalverdict.create.assert_not_called()
        call_kwargs = prisma.db.litellm_shadowevaljob.update_many.call_args.kwargs
        assert call_kwargs["data"]["failed_count"] == {"increment": 1}
        assert "down" in call_kwargs["data"]["last_error"]
        assert set(call_kwargs["where"]["status"]["in"]) == {"pending", "running"}

    async def test_judge_configured_as_a_proxy_deployment_dispatches_through_the_router(self):
        """Regression: the judge always went through the SDK, so a judge named after a
        proxy deployment failed every turn with 'LLM Provider NOT provided' while
        still paying for the shadow call."""
        logger, _, router = self._logger()
        router.get_model_list = MagicMock(return_value=[{"litellm_params": {"model": "openai/gpt-4o"}}])
        router.acompletion = AsyncMock(
            return_value={
                "choices": [{"message": {"content": '{"preference": "A", "confidence": 0.8, "reasoning": "x"}'}}]
            }
        )

        verdict = await logger._call_judge(
            "my-deployment", [{"role": "user", "content": "hi"}], "real", "shadow", {}
        )

        assert not isinstance(verdict, _CallFailure)
        router.acompletion.assert_awaited_once()
        assert router.acompletion.call_args.kwargs["model"] == "my-deployment"


class TestExtractResponseText:
    def test_dict_response(self):
        resp = {"choices": [{"message": {"content": "hello"}}]}
        assert ShadowEvalLogger._extract_response_text(resp) == "hello"

    def test_object_response(self):
        msg = MagicMock()
        msg.content = "world"
        choice = MagicMock()
        choice.message = msg
        resp = MagicMock()
        resp.choices = [choice]
        assert ShadowEvalLogger._extract_response_text(resp) == "world"

    def test_empty_on_malformed(self):
        assert ShadowEvalLogger._extract_response_text({"nope": True}) == ""
        assert ShadowEvalLogger._extract_response_text(None) == ""
