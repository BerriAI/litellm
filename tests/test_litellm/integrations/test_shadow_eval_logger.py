"""Unit tests for the shadow-eval logger: sampling, verdict parsing, unmasking, and the success hook's skip paths."""

import asyncio
import json
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.constants import INTERNAL_CALL_ORIGIN_METADATA_KEY
from litellm.integrations.shadow_eval_logger import (
    JUDGE_MAX_OUTPUT_TOKENS,
    SHADOW_EVAL_INTERNAL_MARKER,
    ActiveShadowEvalJob,
    ShadowEvalLogger,
    _key_or_team_is_over_budget,
    _parse_pairwise_verdict,
    _sample_hits,
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

        logger = ShadowEvalLogger(router_provider=lambda: MagicMock(), prisma_provider=lambda: MagicMock())
        verdict = await logger._call_judge(
            "gpt-4o-mini", [{"role": "user", "content": "hi"}], "real text", "shadow text", {}
        )

        assert verdict is not None
        _, kwargs = acompletion.call_args
        assert kwargs["max_tokens"] == JUDGE_MAX_OUTPUT_TOKENS


class TestJudgeOutputBudgetStaticChecks:
    def test_budget_leaves_room_for_the_reasoning_field(self):
        """A tight budget truncates the JSON object the judge is asked to emit."""
        assert JUDGE_MAX_OUTPUT_TOKENS >= 500


def _logger_with_mocks(job=None):
    prisma = MagicMock()
    router = MagicMock()
    logger = ShadowEvalLogger(router_provider=lambda: router, prisma_provider=lambda: prisma)
    if job is not None:
        # Pre-warm the cache so no DB call is needed.
        loop_time = asyncio.get_event_loop().time()
        logger._job_cache["key-hash"] = (loop_time, job)
    return logger, prisma, router


@pytest.mark.asyncio
class TestSuccessHookSkipPaths:
    async def test_skips_without_standard_logging_object(self):
        logger, prisma, _ = _logger_with_mocks()
        await logger.async_log_success_event({"messages": []}, MagicMock(), None, None)
        prisma.db.litellm_shadowevaljob.find_first.assert_not_called()

    async def test_skips_own_internal_traffic(self):
        job = ActiveShadowEvalJob(id="j1", router_name="r", shadow_percentage=100.0, judge_model="m", status="running")
        logger, prisma, _ = _logger_with_mocks(job)
        kwargs = {
            "standard_logging_object": {
                "id": "req-1",
                "model": "gpt-4o",
                "metadata": {"user_api_key_hash": "key-hash"},
            },
            "litellm_params": {"metadata": {SHADOW_EVAL_INTERNAL_MARKER: True}},
            "messages": [{"role": "user", "content": "hi"}],
        }
        await logger.async_log_success_event(kwargs, MagicMock(), None, None)
        assert logger._pending_seen == {}

    async def test_skips_when_no_active_job(self):
        logger, prisma, _ = _logger_with_mocks()
        prisma.db.litellm_shadowevaljob.find_first = AsyncMock(return_value=None)
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


@pytest.mark.asyncio
class TestCallRouterShadowForwardsParameters:
    async def test_forwards_non_default_params(self):
        router = MagicMock()
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
        router = MagicMock()
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
        assert kwargs["metadata"][SHADOW_EVAL_INTERNAL_MARKER] is True


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
        logger._call_router_shadow = AsyncMock(return_value=("shadow text", "shadow-model", "SIMPLE", 10))
        logger._call_judge = AsyncMock(return_value=("real", 0.9, "clearer", 0.01))

        job = ActiveShadowEvalJob(id="j1", router_name="r", judge_model="m", shadow_percentage=100.0, status="running")
        await logger._run_shadow_eval(
            job=job,
            request_id="req-1",
            messages=[{"role": "user", "content": "hi"}],
            response_obj={"choices": [{"message": {"content": "real text"}}]},
            real_model="gpt-4o",
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
        logger._call_router_shadow = AsyncMock(return_value=("shadow text", "shadow-model", "SIMPLE", 10))
        logger._call_judge = AsyncMock(return_value=("real", 0.9, "clearer", 0.05))

        job = ActiveShadowEvalJob(id="j1", router_name="r", judge_model="m", shadow_percentage=100.0, status="running")
        await logger._run_shadow_eval(
            job=job,
            request_id="req-1",
            messages=[{"role": "user", "content": "hi"}],
            response_obj={"choices": [{"message": {"content": "real text"}}]},
            real_model="gpt-4o",
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
        logger = ShadowEvalLogger(router_provider=lambda: MagicMock(), prisma_provider=lambda: MagicMock())

        verdict = await logger._call_judge(
            "gpt-4o-mini", [{"role": "user", "content": "hi"}], "real text", "shadow text", _PARENT_METADATA
        )

        assert verdict is not None
        metadata = acompletion.call_args.kwargs["metadata"]
        _assert_attributed(metadata, SHADOW_EVAL_JUDGE_CALL_ORIGIN)
        assert metadata[SHADOW_EVAL_INTERNAL_MARKER] is True

    async def test_shadow_router_call_carries_the_key_identity_and_the_recursion_guard(self):
        router = MagicMock()
        router.acompletion = AsyncMock(return_value={"choices": [{"message": {"content": "shadow reply"}}]})
        logger = ShadowEvalLogger(router_provider=lambda: router, prisma_provider=lambda: MagicMock())

        result = await logger._call_router_shadow(
            "claude-auto", [{"role": "user", "content": "hi"}], {}, _PARENT_METADATA
        )

        assert result is not None
        metadata = router.acompletion.call_args.kwargs["metadata"]
        _assert_attributed(metadata, SHADOW_EVAL_ROUTER_CALL_ORIGIN)
        # Without the marker the shadow call's own success event would shadow itself.
        assert metadata[SHADOW_EVAL_INTERNAL_MARKER] is True

    async def test_shadow_metadata_stays_writable_for_the_routing_decision_read_back(self):
        """Tier attribution reads routing_decision back out of the dict the router was given."""
        router = MagicMock()

        async def acompletion(**kwargs):
            kwargs["metadata"]["routing_decision"] = {"tier_label": "COMPLEX", "routed_model": "o1"}
            return {"choices": [{"message": {"content": "shadow reply"}}]}

        router.acompletion = acompletion
        logger = ShadowEvalLogger(router_provider=lambda: router, prisma_provider=lambda: MagicMock())

        result = await logger._call_router_shadow(
            "claude-auto", [{"role": "user", "content": "hi"}], {}, _PARENT_METADATA
        )

        assert result is not None
        assert result[2] == "COMPLEX"

    async def test_parent_metadata_reaches_both_legs_from_the_success_hook(self):
        """The hook is the only place the parent's metadata exists; a break here is invisible
        to unit tests of the two legs in isolation."""
        prisma = MagicMock()
        prisma.db.litellm_shadowevalverdict.create = AsyncMock()
        prisma.db.litellm_shadowevaljob.update_many = AsyncMock()
        job = ActiveShadowEvalJob(id="j1", router_name="r", shadow_percentage=100.0, judge_model="m", status="running")
        logger, _, _ = _logger_with_mocks(job)
        logger._prisma_provider = lambda: prisma
        logger._call_router_shadow = AsyncMock(return_value=("shadow text", "shadow-model", "SIMPLE", 10))
        logger._call_judge = AsyncMock(return_value=("real", 0.9, "clearer", 0.01))

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
class TestSuccessHookSkipsWhenOverBudget:
    async def test_over_budget_key_is_skipped_before_scheduling_the_shadow_task(self):
        job = ActiveShadowEvalJob(id="j1", router_name="r", shadow_percentage=100.0, judge_model="m", status="running")
        logger, _, router = _logger_with_mocks(job)
        logger._run_shadow_eval = AsyncMock()
        kwargs = {
            "standard_logging_object": {
                "id": "req-1",
                "model": "gpt-4o",
                "call_type": "acompletion",
                "metadata": {
                    "user_api_key_hash": "key-hash",
                    "user_api_key_max_budget": 10.0,
                    "user_api_key_spend": 10.0,
                },
            },
            "litellm_params": {"metadata": {}},
            "messages": [{"role": "user", "content": "hi"}],
        }

        with patch("litellm.proxy.proxy_server.get_current_spend", AsyncMock(return_value=10.0)):
            await logger.async_log_success_event(kwargs, MagicMock(), None, None)
        await asyncio.sleep(0)

        logger._run_shadow_eval.assert_not_awaited()
        router.acompletion.assert_not_called()

    async def test_under_budget_key_still_schedules_the_shadow_task(self):
        job = ActiveShadowEvalJob(id="j1", router_name="r", shadow_percentage=100.0, judge_model="m", status="running")
        logger, _, router = _logger_with_mocks(job)
        logger._run_shadow_eval = AsyncMock()
        kwargs = {
            "standard_logging_object": {
                "id": "req-1",
                "model": "gpt-4o",
                "call_type": "acompletion",
                "metadata": {
                    "user_api_key_hash": "key-hash",
                    "user_api_key_max_budget": 10.0,
                    "user_api_key_spend": 4.0,
                },
            },
            "litellm_params": {"metadata": {}},
            "messages": [{"role": "user", "content": "hi"}],
        }

        with patch("litellm.proxy.proxy_server.get_current_spend", AsyncMock(return_value=4.0)):
            await logger.async_log_success_event(kwargs, MagicMock(), None, None)
        await asyncio.sleep(0)

        logger._run_shadow_eval.assert_awaited_once()


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

    async def test_job_over_the_cap_stops_sampling_and_completes_the_job(self):
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
        prisma.db.litellm_shadowevaljob.update_many = AsyncMock()
        logger._run_shadow_eval = AsyncMock()

        await logger.async_log_success_event(self._success_kwargs(), MagicMock(), None, None)
        await asyncio.sleep(0)

        logger._run_shadow_eval.assert_not_awaited()
        router.acompletion.assert_not_called()
        prisma.db.litellm_shadowevaljob.update_many.assert_awaited_once()
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
        prisma.db.litellm_shadowevaljob.update_many = AsyncMock()
        logger._run_shadow_eval = AsyncMock()

        await logger.async_log_success_event(self._success_kwargs(), MagicMock(), None, None)
        await asyncio.sleep(0.01)

        logger._run_shadow_eval.assert_awaited_once()
        prisma.db.litellm_shadowevaljob.update_many.assert_not_awaited()

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
        prisma.db.litellm_shadowevaljob.update_many = AsyncMock()
        logger._run_shadow_eval = AsyncMock()

        await logger.async_log_success_event(self._success_kwargs(), MagicMock(), None, None)
        await asyncio.sleep(0.01)

        logger._run_shadow_eval.assert_awaited_once()
        prisma.db.litellm_shadowevaljob.update_many.assert_not_awaited()

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

    async def test_stopping_evicts_the_cached_job_so_later_requests_do_not_rewrite_it(self):
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
        prisma.db.litellm_shadowevaljob.find_first = AsyncMock(return_value=None)

        await logger.async_log_success_event(self._success_kwargs(), MagicMock(), None, None)
        await logger.async_log_success_event(self._success_kwargs(), MagicMock(), None, None)

        assert prisma.db.litellm_shadowevaljob.update_many.await_count == 1


@pytest.mark.asyncio
class TestJobsStopAtTheirScheduledEnd:
    """A shadow eval samples ongoing traffic, so a job whose window has closed must
    stop billing judge calls even if nobody remembers to stop it by hand."""

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

    async def test_job_past_its_end_stops_sampling_and_completes(self):
        logger, prisma, router = _logger_with_mocks(self._job(datetime.now(timezone.utc) - timedelta(seconds=1)))
        prisma.db.litellm_shadowevaljob.update_many = AsyncMock()
        logger._run_shadow_eval = AsyncMock()

        await logger.async_log_success_event(TestPerJobSpendCap._success_kwargs(), MagicMock(), None, None)
        await asyncio.sleep(0)

        logger._run_shadow_eval.assert_not_awaited()
        router.acompletion.assert_not_called()
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

    async def test_expiry_evicts_the_cached_job_so_later_requests_do_not_rewrite_it(self):
        logger, prisma, _ = _logger_with_mocks(self._job(datetime.now(timezone.utc) - timedelta(seconds=1)))
        prisma.db.litellm_shadowevaljob.update_many = AsyncMock()
        prisma.db.litellm_shadowevaljob.find_first = AsyncMock(return_value=None)

        await logger.async_log_success_event(TestPerJobSpendCap._success_kwargs(), MagicMock(), None, None)
        await logger.async_log_success_event(TestPerJobSpendCap._success_kwargs(), MagicMock(), None, None)

        assert prisma.db.litellm_shadowevaljob.update_many.await_count == 1

    async def test_naive_db_datetime_is_treated_as_utc(self):
        logger, prisma, _ = _logger_with_mocks()
        record = MagicMock()
        record.id = "j1"
        record.router_name = "r"
        record.shadow_percentage = 100.0
        record.judge_model = "m"
        record.status = "running"
        record.cost_estimate = 10.0
        record.cost_actual = 0.0
        record.ends_at = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(seconds=1)
        prisma.db.litellm_shadowevaljob.find_first = AsyncMock(return_value=record)

        job = await logger._get_active_job("key-hash")

        assert job is not None
        assert job.ends_at is not None and job.ends_at.tzinfo is not None
        from litellm.integrations.shadow_eval_logger import _job_is_past_its_end

        assert _job_is_past_its_end(job)


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
