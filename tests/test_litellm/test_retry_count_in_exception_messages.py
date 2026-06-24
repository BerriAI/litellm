"""
Tests for accurate retry count stamping in exception messages.

Verifies that:
1. retry_policy with TimeoutErrorRetries=0 stamps num_retries=0 / max_retries=0
   on the exception — not a stale value from litellm.num_retries.
2. exceptions.py __str__ displays 0 retries (is-not-None check, not truthy).
3. The @client decorator does not inject litellm.num_retries into exceptions
   that come from router/proxy calls.

Regression tests for https://github.com/BerriAI/litellm/pull/29903
"""

from unittest.mock import AsyncMock, patch

import pytest

import litellm
from litellm import Router
from litellm.exceptions import Timeout, InternalServerError
from litellm.types.router import RetryPolicy

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_timeout(message="Connection timed out"):
    return Timeout(
        message=message,
        model="claude-opus-4-6",
        llm_provider="vertex_ai",
    )


def _make_internal_server_error(message="vLLM connection refused"):
    return InternalServerError(
        message=message,
        model="hosted_vllm/llama-3.1",
        llm_provider="hosted_vllm",
    )


def _make_router(retry_policy: dict) -> Router:
    return Router(
        model_list=[
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "openai/gpt-4o",
                    "api_key": "fake-key",
                },
            }
        ],
        retry_policy=RetryPolicy(**retry_policy),
        num_retries=3,
    )


# ---------------------------------------------------------------------------
# exceptions.py: is-not-None check
# ---------------------------------------------------------------------------


class TestExceptionStrFormat:
    def test_zero_num_retries_is_displayed(self):
        """num_retries=0 should appear in __str__, not be silently suppressed."""
        exc = _make_timeout()
        exc.num_retries = 0
        exc.max_retries = 0
        s = str(exc)
        assert "LiteLLM Retried: 0 times" in s
        assert "LiteLLM Max Retries: 0" in s

    def test_none_num_retries_is_not_displayed(self):
        """num_retries=None (unset) should not produce any retry text."""
        exc = _make_timeout()
        exc.num_retries = None
        exc.max_retries = None
        s = str(exc)
        assert "LiteLLM Retried" not in s
        assert "LiteLLM Max Retries" not in s

    def test_nonzero_num_retries_displayed_normally(self):
        """num_retries=3 should display as before."""
        exc = _make_internal_server_error()
        exc.num_retries = 3
        exc.max_retries = 3
        s = str(exc)
        assert "LiteLLM Retried: 3 times" in s
        assert "LiteLLM Max Retries: 3" in s


# ---------------------------------------------------------------------------
# router.py: TimeoutErrorRetries=0 stamps 0 on exception
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_timeout_zero_retries_stamps_zero_on_exception():
    """
    With TimeoutErrorRetries=0, the router should stamp num_retries=0 and
    max_retries=0 on the exception before raising — not leave a stale value.
    """
    router = _make_router({"TimeoutErrorRetries": 0})

    timeout_exc = _make_timeout()

    with patch.object(router, "_acompletion", new_callable=AsyncMock) as mock_call:
        mock_call.side_effect = timeout_exc

        with pytest.raises(Timeout) as exc_info:
            await router.acompletion(
                model="test-model",
                messages=[{"role": "user", "content": "hi"}],
            )

    raised = exc_info.value
    assert raised.num_retries == 0, (
        f"Expected num_retries=0, got {raised.num_retries}. "
        "TimeoutErrorRetries=0 should stamp 0 retries, not litellm.num_retries."
    )
    assert raised.max_retries == 0, f"Expected max_retries=0, got {raised.max_retries}."


@pytest.mark.asyncio
async def test_internal_server_error_three_retries_stamps_correctly():
    """
    With InternalServerErrorRetries=3, the router should retry 3 times and
    stamp num_retries=3 / max_retries=3 on the final exception.
    """
    router = _make_router({"InternalServerErrorRetries": 3})

    with patch.object(router, "_acompletion", new_callable=AsyncMock) as mock_call:
        mock_call.side_effect = _make_internal_server_error()

        with pytest.raises(InternalServerError) as exc_info:
            await router.acompletion(
                model="test-model",
                messages=[{"role": "user", "content": "hi"}],
            )

    raised = exc_info.value
    assert raised.num_retries == 3, f"Expected num_retries=3, got {raised.num_retries}."
    assert raised.max_retries == 3, f"Expected max_retries=3, got {raised.max_retries}."
    assert mock_call.call_count == 4  # 1 initial + 3 retries


# ---------------------------------------------------------------------------
# utils.py: decorator does not inject litellm.num_retries for router calls
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_decorator_does_not_inject_global_num_retries_for_router_calls():
    """
    When a call comes from the router (metadata contains model_group),
    the @client decorator must not stamp litellm.num_retries onto the
    exception. The router manages its own retry counts.
    """
    original_num_retries = litellm.num_retries
    litellm.num_retries = 99  # set a clearly wrong global value

    try:
        router = _make_router({"TimeoutErrorRetries": 0})
        timeout_exc = _make_timeout()

        with patch.object(router, "_acompletion", new_callable=AsyncMock) as mock_call:
            mock_call.side_effect = timeout_exc

            with pytest.raises(Timeout) as exc_info:
                await router.acompletion(
                    model="test-model",
                    messages=[{"role": "user", "content": "hi"}],
                )

        raised = exc_info.value
        assert raised.num_retries == 0, (
            f"Expected num_retries=0 (router stamped actual count), "
            f"got {raised.num_retries}. "
            "Router calls should not be polluted by the global litellm.num_retries value."
        )
    finally:
        litellm.num_retries = original_num_retries
