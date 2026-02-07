"""Tests for guardrail retry logic (guardrail_retries.py)."""

import pytest
from unittest.mock import AsyncMock, patch

import litellm
from litellm.integrations.custom_guardrail import ModifyResponseException

from litellm.proxy.guardrails.guardrail_retries import (
    DEFAULT_GUARDRAIL_NUM_RETRIES,
    DEFAULT_GUARDRAIL_RETRY_AFTER,
    get_guardrail_retry_config,
    run_guardrail_with_retries,
    should_retry_guardrail_error,
)


class TestShouldRetryGuardrailError:
    """Test which errors are considered retriable."""

    def test_retries_on_429(self):
        e = Exception("rate limit")
        e.status_code = 429
        assert should_retry_guardrail_error(e) is True

    def test_retries_on_500(self):
        e = Exception("server error")
        e.status_code = 500
        assert should_retry_guardrail_error(e) is True

    def test_no_retry_on_404(self):
        e = Exception("not found")
        e.status_code = 404
        assert should_retry_guardrail_error(e) is False

    def test_no_retry_on_modify_response_exception(self):
        e = ModifyResponseException(
            message="blocked",
            model="gpt-4",
            request_data={},
        )
        assert should_retry_guardrail_error(e) is False

    def test_no_retry_on_content_policy_violation(self):
        try:
            e = litellm.ContentPolicyViolationError(
                message="policy", model="gpt-4", llm_provider="openai"
            )
            assert should_retry_guardrail_error(e) is False
        except (AttributeError, TypeError):
            pytest.skip("ContentPolicyViolationError not available")


class TestGetGuardrailRetryConfig:
    """Test reading num_retries and retry_after from guardrail."""

    def test_defaults(self):
        guardrail = type("G", (), {})()
        num_retries, retry_after = get_guardrail_retry_config(guardrail)
        assert num_retries == DEFAULT_GUARDRAIL_NUM_RETRIES
        assert retry_after == DEFAULT_GUARDRAIL_RETRY_AFTER

    def test_from_optional_params(self):
        guardrail = type("G", (), {"optional_params": {"num_retries": 5, "retry_after": 2.0}})()
        num_retries, retry_after = get_guardrail_retry_config(guardrail)
        assert num_retries == 5
        assert retry_after == 2.0

    def test_from_guardrail_config(self):
        guardrail = type("G", (), {"guardrail_config": {"num_retries": 3, "retry_after": 1}})()
        num_retries, retry_after = get_guardrail_retry_config(guardrail)
        assert num_retries == 3
        assert retry_after == 1


class TestRunGuardrailWithRetries:
    """Test run_guardrail_with_retries behavior."""

    @pytest.mark.asyncio
    async def test_succeeds_first_try(self):
        async def ok_coro():
            return {"done": True}

        result = await run_guardrail_with_retries(
            coro_factory=lambda: ok_coro(),
            num_retries=2,
            retry_after=0,
            guardrail_name="test",
        )
        assert result == {"done": True}

    @pytest.mark.asyncio
    async def test_succeeds_on_second_attempt(self):
        call_count = 0

        async def fail_once():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                e = Exception("retriable")
                e.status_code = 429
                raise e
            return {"done": True}

        with patch("litellm.proxy.guardrails.guardrail_retries.asyncio.sleep", new_callable=AsyncMock):
            result = await run_guardrail_with_retries(
                coro_factory=lambda: fail_once(),
                num_retries=2,
                retry_after=0,
                guardrail_name="test",
            )
        assert result == {"done": True}
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_raises_after_exhausting_retries(self):
        async def always_fail():
            e = Exception("rate limit")
            e.status_code = 429
            raise e

        with patch("litellm.proxy.guardrails.guardrail_retries.asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(Exception, match="rate limit"):
                await run_guardrail_with_retries(
                    coro_factory=lambda: always_fail(),
                    num_retries=2,
                    retry_after=0,
                    guardrail_name="test",
                )

    @pytest.mark.asyncio
    async def test_no_retry_when_num_retries_zero(self):
        call_count = 0

        async def fail_once():
            nonlocal call_count
            call_count += 1
            raise Exception("boom")

        with patch("litellm.proxy.guardrails.guardrail_retries.asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(Exception, match="boom"):
                await run_guardrail_with_retries(
                    coro_factory=lambda: fail_once(),
                    num_retries=0,
                    retry_after=0,
                    guardrail_name="test_guardrail",
                )
        assert call_count == 1
