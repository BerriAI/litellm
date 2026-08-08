"""Unit tests for the shadow-eval logger: sampling, verdict parsing, unmasking, and the success hook's skip paths."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.integrations.shadow_eval_logger import (
    SHADOW_EVAL_INTERNAL_MARKER,
    ActiveShadowEvalJob,
    ShadowEvalLogger,
    _parse_pairwise_verdict,
    _sample_hits,
    _unmask_preference,
)


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
        )

        _, kwargs = router.acompletion.call_args
        assert "stream" not in kwargs
        assert kwargs["temperature"] == 0.5
        # metadata must stay the logger's own internal marker, not the caller's
        assert kwargs["metadata"] == {SHADOW_EVAL_INTERNAL_MARKER: True}


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
        )

        _, call_kwargs = prisma.db.litellm_shadowevaljob.update_many.call_args
        assert call_kwargs["data"]["cost_actual"] == {"increment": 0.05}


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
