"""
Tests for router retry backoff behavior.
"""

import asyncio
from unittest.mock import patch

import httpx
import pytest
import respx

import litellm
from litellm import Router


@pytest.mark.asyncio
async def test_retry_backoff_uses_current_exception_headers(monkeypatch: pytest.MonkeyPatch):
    """
    Ensure retry backoff uses the current retry exception, not the initial one.
    """
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    litellm.in_memory_llm_clients_cache.flush_cache()
    router = Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": "sk-test",
                },
            }
        ],
        num_retries=2,
    )

    retry_after_values = (1, 15, 30)
    replies = [
        httpx.Response(
            429,
            headers={"retry-after": str(retry_after)},
            json={"error": {"message": f"Rate limited, retry after {retry_after}", "type": "rate_limit"}},
        )
        for retry_after in retry_after_values
    ]
    real_sleep = asyncio.sleep
    slept: list[float] = []

    async def record_sleep(delay: float, *args: object, **kwargs: object) -> None:
        slept.append(delay)
        await real_sleep(0)

    with respx.mock() as mock_router:
        mock_router.post("https://api.openai.com/v1/chat/completions").mock(side_effect=replies)
        with patch("asyncio.sleep", record_sleep):
            with pytest.raises(litellm.RateLimitError):
                await router.acompletion(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": "Hello"}],
                )
    litellm.in_memory_llm_clients_cache.flush_cache()

    # Router backs off once after the initial failure, then once per failed retry.
    # Each backoff honours that attempt's retry-after (plus jitter), so each sleep
    # sits between its own header value and the next one.
    assert len(slept) == router.num_retries + 1
    assert retry_after_values[0] <= slept[0] < retry_after_values[1]
    assert retry_after_values[1] <= slept[1] < retry_after_values[2]
    assert retry_after_values[2] <= slept[2]
