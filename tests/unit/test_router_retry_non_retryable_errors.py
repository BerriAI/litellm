"""
Test that the Router retry loop correctly handles non-retryable errors.

Verifies that:
1. Non-retryable errors (e.g., 400 ContextWindowExceeded) inside the retry loop
   break out immediately instead of being swallowed.
2. original_exception is updated to the latest error, not stuck on the first.
3. Retryable errors (e.g., 429 RateLimitError) still retry normally.

Regression tests for https://github.com/BerriAI/litellm/issues/21343
"""

from collections.abc import Iterator

import httpx
import pytest
import respx

import litellm
from litellm import Router


@pytest.fixture(autouse=True)
def httpx_transport(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    litellm.in_memory_llm_clients_cache.flush_cache()
    yield
    litellm.in_memory_llm_clients_cache.flush_cache()


def _error(status: int, message: str, code: str | None = None) -> httpx.Response:
    return httpx.Response(
        status,
        headers={"retry-after": "0"},
        json={"error": {"message": message, "type": "invalid_request_error", "code": code}},
    )


def _rate_limit(attempt: int) -> httpx.Response:
    return _error(429, f"Rate limit attempt {attempt}")


def _create_router(num_retries=2):
    """Create a Router with two deployments for testing."""
    return Router(
        model_list=[
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "openai/gpt-4",
                    "api_key": f"fake-key-{i}",
                    "api_base": "https://api.openai.com/v1",
                },
            }
            for i in (1, 2)
        ],
        num_retries=num_retries,
    )


async def _call(router: Router, responses: list[httpx.Response]) -> None:
    """Run one acompletion against upstream replies served in order."""
    with respx.mock() as mock_router:
        mock_router.post("https://api.openai.com/v1/chat/completions").mock(side_effect=responses)
        await router.acompletion(model="test-model", messages=[{"role": "user", "content": "test"}])


@pytest.mark.asyncio
async def test_non_retryable_error_in_retry_loop_raises_immediately():
    """
    When a non-retryable error (400 ContextWindowExceeded) occurs inside the
    retry loop, the router should raise it immediately instead of swallowing it
    and raising the original error.

    Scenario: First call -> 429, Retry -> 400 (non-retryable)
    Expected: ContextWindowExceededError is raised, NOT RateLimitError
    """
    router = _create_router(num_retries=2)
    replies = [
        _rate_limit(1),
        _error(400, "This model's maximum context length is 8192 tokens", "context_length_exceeded"),
        _rate_limit(3),
    ]

    with pytest.raises(litellm.ContextWindowExceededError):
        await _call(router, replies)


@pytest.mark.asyncio
async def test_bad_request_error_in_retry_loop_raises_immediately():
    """
    A generic 400 BadRequestError inside the retry loop should also break out
    immediately since 400 is not retryable.
    """
    router = _create_router(num_retries=2)

    with pytest.raises(litellm.BadRequestError):
        await _call(router, [_rate_limit(1), _error(400, "Invalid request"), _rate_limit(3)])


@pytest.mark.asyncio
async def test_original_exception_updated_to_latest_error():
    """
    When all retries are exhausted with retryable errors, the LAST error
    should be raised, not the first one.
    """
    router = _create_router(num_retries=2)

    with pytest.raises(litellm.RateLimitError) as exc_info:
        await _call(router, [_rate_limit(i) for i in range(1, 6)])
    # Should be the LAST error, not the first
    assert "Rate limit attempt 3" in str(exc_info.value)


@pytest.mark.asyncio
async def test_retryable_errors_still_retry_normally():
    """
    Retryable errors (429 RateLimitError) should still be retried the
    configured number of times before raising.
    """
    router = _create_router(num_retries=3)
    replies = [_rate_limit(i) for i in range(1, 9)]

    with respx.mock() as mock_router:
        route = mock_router.post("https://api.openai.com/v1/chat/completions").mock(side_effect=replies)
        with pytest.raises(litellm.RateLimitError):
            await router.acompletion(model="test-model", messages=[{"role": "user", "content": "test"}])

    # Initial call + 3 retries = 4 total calls
    assert route.call_count == 4


@pytest.mark.asyncio
async def test_not_found_error_in_retry_loop_raises_immediately():
    """
    A 404 NotFoundError inside the retry loop should break out immediately.
    """
    router = _create_router(num_retries=2)
    replies = [_rate_limit(1), _error(404, "Model not found"), _rate_limit(3)]

    with respx.mock() as mock_router:
        route = mock_router.post("https://api.openai.com/v1/chat/completions").mock(side_effect=replies)
        with pytest.raises(litellm.NotFoundError):
            await router.acompletion(model="test-model", messages=[{"role": "user", "content": "test"}])

    # Only 2 calls: initial + first retry that hits non-retryable
    assert route.call_count == 2
