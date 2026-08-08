"""Unit tests for the shadow-eval logger: sampling, verdict parsing, unmasking, and the success hook's skip paths."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.integrations.shadow_eval_logger import (
    SHADOW_EVAL_INTERNAL_MARKER,
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
        assert v["preference"] == "A"
        assert v["confidence"] == 0.9

    def test_fenced_json(self):
        raw = 'Here is my verdict:\n```json\n{"preference": "B", "confidence": 0.7, "reasoning": "x"}\n```\nDone.'
        assert _parse_pairwise_verdict(raw)["preference"] == "B"

    def test_json_with_surrounding_prose(self):
        raw = 'Verdict: {"preference": "tie", "confidence": 0.5, "reasoning": "same"} — final.'
        assert _parse_pairwise_verdict(raw)["preference"] == "tie"

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
        job = {"id": "j1", "router_name": "r", "shadow_percentage": 100.0, "judge_model": "m", "status": "running"}
        logger, prisma, _ = _logger_with_mocks(job)
        kwargs = {
            "standard_logging_object": {"id": "req-1", "model": "gpt-4o", "metadata": {"user_api_key_hash": "key-hash"}},
            "litellm_params": {"metadata": {SHADOW_EVAL_INTERNAL_MARKER: True}},
            "messages": [{"role": "user", "content": "hi"}],
        }
        await logger.async_log_success_event(kwargs, MagicMock(), None, None)
        assert logger._pending_seen == {}

    async def test_skips_when_no_active_job(self):
        logger, prisma, _ = _logger_with_mocks()
        prisma.db.litellm_shadowevaljob.find_first = AsyncMock(return_value=None)
        kwargs = {
            "standard_logging_object": {"id": "req-1", "model": "gpt-4o", "metadata": {"user_api_key_hash": "key-hash"}},
            "litellm_params": {"metadata": {}},
            "messages": [{"role": "user", "content": "hi"}],
        }
        await logger.async_log_success_event(kwargs, MagicMock(), None, None)
        assert logger._pending_seen == {}

    async def test_counts_seen_for_active_job(self):
        job = {"id": "j1", "router_name": "r", "shadow_percentage": 0.0, "judge_model": "m", "status": "running"}
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
