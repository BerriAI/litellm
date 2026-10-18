from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import openai
import pytest

import litellm
from litellm import (
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    RateLimitError,
    ServiceUnavailableError,
    acompletion_with_retries,
    completion_with_retries,
)


def _make_rate_limit_error(retry_after: str | None = None) -> RateLimitError:
    headers = {"retry-after": retry_after} if retry_after else {}
    resp = MagicMock()
    resp.headers = headers
    resp.status_code = 429
    err = RateLimitError(
        message="Rate limit reached",
        response=resp,
        llm_provider="openai",
        model="gpt-4o",
    )
    err.status_code = 429
    err.response = resp
    return err


def _make_auth_error() -> AuthenticationError:
    resp = MagicMock()
    resp.headers = {}
    resp.status_code = 401
    err = AuthenticationError(
        message="Invalid API Key",
        response=resp,
        llm_provider="openai",
        model="gpt-4o",
    )
    err.status_code = 401
    return err


def _make_bad_request_error() -> BadRequestError:
    resp = MagicMock()
    resp.headers = {}
    resp.status_code = 400
    err = BadRequestError(
        message="Invalid argument",
        response=resp,
        llm_provider="openai",
        model="gpt-4o",
    )
    err.status_code = 400
    return err


def _make_service_unavailable_error() -> ServiceUnavailableError:
    resp = MagicMock()
    resp.headers = {}
    resp.status_code = 503
    err = ServiceUnavailableError(
        message="The server is temporarily overloaded",
        response=resp,
        llm_provider="openai",
        model="gpt-4o",
    )
    err.status_code = 503
    return err


def _make_internal_server_error() -> InternalServerError:
    resp = MagicMock()
    resp.headers = {}
    resp.status_code = 500
    err = InternalServerError(
        message="Internal server error",
        response=resp,
        llm_provider="openai",
        model="gpt-4o",
    )
    err.status_code = 500
    return err


def test_completion_with_retries_honors_retry_after():
    """Verify completion_with_retries reads Retry-After and applies jitter."""
    calls = []

    def mock_completion(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise _make_rate_limit_error(retry_after="2")
        return {"status": "ok"}

    with patch("time.sleep") as mock_sleep:
        res = completion_with_retries(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
            num_retries=3,
            original_function=mock_completion,
        )
        assert res == {"status": "ok"}
        assert len(calls) == 2
        mock_sleep.assert_called_once()
        sleep_arg = mock_sleep.call_args[0][0]
        assert 2.0 <= sleep_arg <= 2.8


def test_completion_with_retries_non_transient_auth_error_fails_immediately():
    """Non-transient errors like 401 should fail on attempt 1 without retrying."""
    calls = []

    def mock_completion(*args, **kwargs):
        calls.append(1)
        raise _make_auth_error()

    with patch("time.sleep") as mock_sleep:
        with pytest.raises(AuthenticationError):
            completion_with_retries(
                model="gpt-4o",
                messages=[{"role": "user", "content": "hi"}],
                num_retries=3,
                original_function=mock_completion,
            )
        assert len(calls) == 1
        mock_sleep.assert_not_called()


def test_completion_with_retries_bad_request_fails_immediately():
    """Non-transient errors like 400 Bad Request should fail on attempt 1 without retrying."""
    calls = []

    def mock_completion(*args, **kwargs):
        calls.append(1)
        raise _make_bad_request_error()

    with patch("time.sleep") as mock_sleep:
        with pytest.raises(BadRequestError):
            completion_with_retries(
                model="gpt-4o",
                messages=[{"role": "user", "content": "hi"}],
                num_retries=3,
                original_function=mock_completion,
            )
        assert len(calls) == 1
        mock_sleep.assert_not_called()


def test_completion_with_retries_service_unavailable_503():
    """503 Service Unavailable should be retried with exponential backoff."""
    calls = []

    def mock_completion(*args, **kwargs):
        calls.append(1)
        if len(calls) < 3:
            raise _make_service_unavailable_error()
        return {"status": "recovered"}

    with patch("time.sleep") as mock_sleep:
        res = completion_with_retries(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
            num_retries=3,
            original_function=mock_completion,
        )
        assert res == {"status": "recovered"}
        assert len(calls) == 3
        assert mock_sleep.call_count == 2


def test_completion_with_retries_internal_server_error_500():
    """500 Internal Server Error should be retried."""
    calls = []

    def mock_completion(*args, **kwargs):
        calls.append(1)
        if len(calls) < 2:
            raise _make_internal_server_error()
        return {"status": "recovered"}

    with patch("time.sleep") as mock_sleep:
        res = completion_with_retries(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
            num_retries=2,
            original_function=mock_completion,
        )
        assert res == {"status": "recovered"}
        assert len(calls) == 2
        mock_sleep.assert_called_once()


@pytest.mark.asyncio
async def test_acompletion_with_retries_honors_retry_after():
    """Verify acompletion_with_retries reads Retry-After and applies jitter."""
    calls = []

    async def mock_acompletion(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise _make_rate_limit_error(retry_after="3")
        return {"status": "ok"}

    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        res = await acompletion_with_retries(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
            num_retries=3,
            original_function=mock_acompletion,
        )
        assert res == {"status": "ok"}
        assert len(calls) == 2
        mock_sleep.assert_called_once()
        sleep_arg = mock_sleep.call_args[0][0]
        assert 3.0 <= sleep_arg <= 3.8


@pytest.mark.asyncio
async def test_acompletion_with_retries_non_transient_auth_error_fails_immediately():
    """Async 401 should fail on attempt 1 without retrying."""
    calls = []

    async def mock_acompletion(*args, **kwargs):
        calls.append(1)
        raise _make_auth_error()

    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        with pytest.raises(AuthenticationError):
            await acompletion_with_retries(
                model="gpt-4o",
                messages=[{"role": "user", "content": "hi"}],
                num_retries=3,
                original_function=mock_acompletion,
            )
        assert len(calls) == 1
        mock_sleep.assert_not_called()


def test_completion_decorator_retries_transient_error():
    """Verify litellm.completion(..., num_retries=2) retries on RateLimitError and sleeps on first retry."""
    calls = []

    def mock_chat_fn(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise _make_rate_limit_error(retry_after="1")
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        return mock_response

    with (
        patch("time.sleep") as mock_sleep,
        patch.object(litellm.main.openai_chat_completions, "completion", side_effect=mock_chat_fn),
    ):
        litellm.completion(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
            num_retries=2,
        )
        assert len(calls) == 2
        mock_sleep.assert_called_once()
        assert 1.0 <= mock_sleep.call_args[0][0] <= 1.8


def test_completion_decorator_fails_fast_on_auth_error():
    """Verify litellm.completion(..., num_retries=2) fails immediately on AuthenticationError without retry."""
    calls = []

    def mock_chat_fn(*args, **kwargs):
        calls.append(1)
        raise _make_auth_error()

    with (
        patch("time.sleep") as mock_sleep,
        patch.object(litellm.main.openai_chat_completions, "completion", side_effect=mock_chat_fn),
    ):
        with pytest.raises(AuthenticationError):
            litellm.completion(
                model="gpt-4o",
                messages=[{"role": "user", "content": "hi"}],
                num_retries=2,
            )
        assert len(calls) == 1
        mock_sleep.assert_not_called()


@pytest.mark.asyncio
async def test_acompletion_decorator_preserves_retried_count_on_failure():
    """Verify litellm.acompletion(..., num_retries=3) attaches num_retries to the raised exception."""
    calls = []

    async def mock_chat_fn(*args, **kwargs):
        calls.append(1)
        raise _make_internal_server_error()

    with (
        patch("asyncio.sleep", new_callable=AsyncMock),
        patch.object(litellm.main.openai_chat_completions, "acompletion", side_effect=mock_chat_fn),
    ):
        with pytest.raises(InternalServerError, match="LiteLLM Retried: 3 times"):
            await litellm.acompletion(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "hi"}],
                api_key="test-key",
                num_retries=3,
            )
        assert len(calls) == 4


def test_completion_with_retries_connection_error_with_constant_retry():
    """Verify APIConnectionError (no status_code) retries and constant_retry waits 0.0s."""
    calls = []

    def mock_completion(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise openai.APIConnectionError(request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"))
        return {"status": "recovered"}

    with patch("time.sleep") as mock_sleep:
        res = completion_with_retries(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
            num_retries=2,
            retry_strategy="constant_retry",
            original_function=mock_completion,
        )
        assert res == {"status": "recovered"}
        assert len(calls) == 2
        mock_sleep.assert_called_once_with(0.0)


@pytest.mark.asyncio
async def test_acompletion_decorator_retries_transient_error():
    """Verify litellm.acompletion(..., num_retries=2) retries on RateLimitError and sleeps on first retry."""
    calls = []

    async def mock_chat_fn(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise _make_rate_limit_error(retry_after="1")
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        return mock_response

    with (
        patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
        patch.object(litellm.main.openai_chat_completions, "acompletion", side_effect=mock_chat_fn),
    ):
        await litellm.acompletion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hi"}],
            api_key="test-key",
            num_retries=2,
        )
        assert len(calls) == 2
        mock_sleep.assert_called_once()
        assert 1.0 <= mock_sleep.call_args[0][0] <= 1.8


def test_completion_retry_policy_bad_request_honors_full_count():
    """Explicit RetryPolicy(BadRequestErrorRetries=3) should execute all 3 retries (4 total attempts)."""
    from litellm.types.router import RetryPolicy

    calls = []

    def mock_chat_fn(*args, **kwargs):
        calls.append(1)
        raise _make_bad_request_error()

    with (
        patch("time.sleep") as mock_sleep,
        patch.object(litellm.main.openai_chat_completions, "completion", side_effect=mock_chat_fn),
    ):
        with pytest.raises(BadRequestError):
            litellm.completion(
                model="gpt-4o",
                messages=[{"role": "user", "content": "hi"}],
                retry_policy=RetryPolicy(BadRequestErrorRetries=3),
            )
        assert len(calls) == 4
        assert mock_sleep.call_count == 3


@pytest.mark.asyncio
async def test_acompletion_retry_policy_bad_request_honors_full_count():
    """Explicit RetryPolicy(BadRequestErrorRetries=3) on async should execute all 3 retries (4 total attempts)."""
    from litellm.types.router import RetryPolicy

    calls = []

    async def mock_chat_fn(*args, **kwargs):
        calls.append(1)
        raise _make_bad_request_error()

    with (
        patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
        patch.object(litellm.main.openai_chat_completions, "acompletion", side_effect=mock_chat_fn),
    ):
        with pytest.raises(BadRequestError):
            await litellm.acompletion(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "hi"}],
                api_key="test-key",
                retry_policy=RetryPolicy(BadRequestErrorRetries=3),
            )
        assert len(calls) == 4
        assert mock_sleep.call_count == 3


def test_completion_unconfigured_retry_policy_fails_fast_on_auth_error():
    """An unconfigured RetryPolicy() alongside num_retries=2 on sync should not retry AuthenticationError."""
    from litellm.types.router import RetryPolicy

    calls = []

    def mock_chat_fn(*args, **kwargs):
        calls.append(1)
        raise _make_auth_error()

    with (
        patch("time.sleep") as mock_sleep,
        patch.object(litellm.main.openai_chat_completions, "completion", side_effect=mock_chat_fn),
    ):
        with pytest.raises(AuthenticationError):
            litellm.completion(
                model="gpt-4o",
                messages=[{"role": "user", "content": "hi"}],
                num_retries=2,
                retry_policy=RetryPolicy(),
            )
        assert len(calls) == 1
        mock_sleep.assert_not_called()


@pytest.mark.asyncio
async def test_acompletion_unconfigured_retry_policy_fails_fast_on_auth_error():
    """An unconfigured RetryPolicy() alongside num_retries=2 should not retry AuthenticationError."""
    from litellm.types.router import RetryPolicy

    calls = []

    async def mock_chat_fn(*args, **kwargs):
        calls.append(1)
        raise _make_auth_error()

    with (
        patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
        patch.object(litellm.main.openai_chat_completions, "acompletion", side_effect=mock_chat_fn),
    ):
        with pytest.raises(AuthenticationError):
            await litellm.acompletion(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "hi"}],
                api_key="test-key",
                num_retries=2,
                retry_policy=RetryPolicy(),
            )
        assert len(calls) == 1
        mock_sleep.assert_not_called()
