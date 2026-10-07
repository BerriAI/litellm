from __future__ import annotations

import asyncio
from typing import Final
from unittest.mock import AsyncMock

import httpx
import openai
import pytest
from pydantic import TypeAdapter

import litellm
from litellm import Router
from litellm.integrations.custom_logger import CustomLogger
from litellm.llms.custom_httpx.http_handler import HTTPHandler
from litellm.router import AllowedFailsPolicy, RetryPolicy


class _RetryAttemptTracker(CustomLogger):
    previous_models: int = 0

    def log_pre_api_call(self, model: str, messages: list[object], kwargs: dict[str, object]) -> None:
        params: Final = TypeAdapter(dict[str, object]).validate_python(kwargs["litellm_params"])
        raw_metadata: Final = params.get("metadata")
        metadata: Final = (
            TypeAdapter(dict[str, object]).validate_python(raw_metadata) if isinstance(raw_metadata, dict) else {}
        )
        previous_models: Final = metadata.get("previous_models", ())
        self.previous_models = len(previous_models) if isinstance(previous_models, (list, tuple)) else 0


def _retry_test_router() -> Router:
    return Router(
        model_list=[
            {
                "model_name": "retry-test-model",
                "litellm_params": {
                    "model": "openai/retry-test-model",
                    "api_key": "test-key",
                },
            }
        ]
    )


def _rate_limit_error(retry_after: str | None = None) -> openai.RateLimitError:
    headers: Final = {} if retry_after is None else {"retry-after": retry_after}
    response: Final = httpx.Response(
        status_code=429,
        headers=headers,
        request=httpx.Request("POST", "https://example.invalid/v1"),
    )
    return openai.RateLimitError(
        message="Rate limit exceeded",
        response=response,
        body={"error": {"type": "rate_limit_exceeded"}},
    )


def _retry_tracking_router(num_retries: int) -> Router:
    model_list: Final = [
        {
            "model_name": "retry-test-model",
            "litellm_params": {
                "model": "openai/retry-test-model",
                "api_key": "test-key",
            },
            "model_info": {"id": f"model-{index}"},
        }
        for index in range(num_retries + 1)
    ]
    return Router(
        model_list=model_list,
        num_retries=num_retries,
        allowed_fails_policy=AllowedFailsPolicy(RateLimitErrorAllowedFails=100),
    )


def _patch_asyncio_sleep(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    sleep: Final = AsyncMock()
    monkeypatch.setattr(asyncio, "sleep", sleep)
    return sleep


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.parametrize("error_type", ["API Error"])
@pytest.mark.asyncio
async def test_router_retries_errors(sync_mode: bool, error_type: str, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_asyncio_sleep(monkeypatch)
    tracker: Final = _RetryAttemptTracker()
    monkeypatch.setattr(litellm, "callbacks", [tracker])
    router: Final = Router(
        model_list=[
            {
                "model_name": "retry-test-model",
                "litellm_params": {
                    "model": "openai/retry-test-model",
                    "api_key": "test-key",
                },
            },
            {
                "model_name": "retry-test-model",
                "litellm_params": {
                    "model": "openai/retry-test-model",
                    "api_key": "test-key",
                },
            },
            {
                "model_name": "retry-test-model",
                "litellm_params": {
                    "model": "openai/retry-test-model",
                    "api_key": "test-key",
                },
            },
        ],
        num_retries=2,
    )
    mock_responses: Final = {"API Error": Exception("Invalid Request")}
    mock_response: Final = mock_responses[error_type]

    if sync_mode:
        with pytest.raises(openai.APIError):
            router.completion(
                model="retry-test-model",
                messages=[{"role": "user", "content": "Retry test"}],
                mock_response=mock_response,
            )
    else:
        with pytest.raises(openai.APIError):
            await router.acompletion(
                model="retry-test-model",
                messages=[{"role": "user", "content": "Retry test"}],
                mock_response=mock_response,
            )

    assert tracker.previous_models == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error_type",
    ["ContentPolicyViolationErrorRetries"],
)
async def test_router_retry_policy(error_type: str, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_asyncio_sleep(monkeypatch)
    tracker: Final = _RetryAttemptTracker()
    monkeypatch.setattr(litellm, "callbacks", [tracker])
    router: Final = Router(
        model_list=[
            {
                "model_name": "retry-policy-test-model",
                "litellm_params": {
                    "model": "azure/retry-policy-test-model",
                    "api_key": "test-key",
                },
            }
        ],
        retry_policy=RetryPolicy(ContentPolicyViolationErrorRetries=3),
        allowed_fails_policy=AllowedFailsPolicy(
            ContentPolicyViolationErrorAllowedFails=1000,
            RateLimitErrorAllowedFails=100,
        ),
    )
    mock_responses: Final = {"ContentPolicyViolationErrorRetries": "Exception: content_filter_policy"}

    with pytest.raises(litellm.ContentPolicyViolationError):
        await router.acompletion(
            model="retry-policy-test-model",
            messages=[{"role": "user", "content": "Retry policy test"}],
            mock_response=mock_responses[error_type],
        )

    assert tracker.previous_models == 3


@pytest.mark.parametrize("model_group", ["gpt-3.5-turbo"])
@pytest.mark.asyncio
async def test_dynamic_router_retry_policy(model_group: str, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_asyncio_sleep(monkeypatch)
    tracker: Final = _RetryAttemptTracker()
    monkeypatch.setattr(litellm, "callbacks", [tracker])
    router: Final = Router(
        model_list=[
            {
                "model_name": model_group,
                "litellm_params": {
                    "model": "azure/retry-policy-test-model",
                    "api_key": "test-key",
                },
                "model_info": {"id": "model-0"},
            },
            {
                "model_name": model_group,
                "litellm_params": {
                    "model": "azure/retry-policy-test-model",
                    "api_key": "test-key",
                },
                "model_info": {"id": "model-1"},
            },
            {
                "model_name": model_group,
                "litellm_params": {
                    "model": "azure/retry-policy-test-model",
                    "api_key": "test-key",
                },
                "model_info": {"id": "model-2"},
            },
        ],
        model_group_retry_policy={model_group: RetryPolicy(ContentPolicyViolationErrorRetries=2)},
        allowed_fails_policy=AllowedFailsPolicy(
            ContentPolicyViolationErrorAllowedFails=1000,
            RateLimitErrorAllowedFails=100,
        ),
    )

    with pytest.raises(litellm.ContentPolicyViolationError):
        await router.acompletion(
            model=model_group,
            messages=[{"role": "user", "content": "Retry policy test"}],
            mock_response="Exception: content_filter_policy",
        )

    assert tracker.previous_models == 2


def test_retry_rate_limit_error_with_healthy_deployments() -> None:
    router: Final = _retry_test_router()

    should_retry: Final = router.should_retry_this_error(
        error=_rate_limit_error(),
        healthy_deployments=[{"model_name": "retry-test-model"}],
    )

    assert should_retry is True


def test_do_retry_rate_limit_error_with_no_fallbacks_and_no_healthy_deployments() -> None:
    router: Final = _retry_test_router()
    error: Final = _rate_limit_error()

    with pytest.raises(openai.RateLimitError) as error_info:
        router.should_retry_this_error(error=error, healthy_deployments=[])

    assert error_info.value is error


def test_raise_context_window_exceeded_error() -> None:
    router: Final = _retry_test_router()
    error: Final = litellm.ContextWindowExceededError(
        message="Context window exceeded",
        response=httpx.Response(
            status_code=400,
            request=httpx.Request("POST", "https://example.invalid/v1"),
        ),
        llm_provider="openai",
        model="retry-test-model",
    )

    with pytest.raises(litellm.ContextWindowExceededError) as error_info:
        router.should_retry_this_error(
            error=error,
            healthy_deployments=None,
            context_window_fallbacks=[{"retry-test-model": ["fallback-model"]}],
        )

    assert error_info.value is error


def test_raise_context_window_exceeded_error_no_retry() -> None:
    router: Final = _retry_test_router()
    error: Final = litellm.ContextWindowExceededError(
        message="Context window exceeded",
        response=httpx.Response(
            status_code=400,
            request=httpx.Request("POST", "https://example.invalid/v1"),
        ),
        llm_provider="openai",
        model="retry-test-model",
    )

    with pytest.raises(litellm.ContextWindowExceededError) as error_info:
        router.should_retry_this_error(
            error=error,
            healthy_deployments=[{"model_name": "retry-test-model"}],
            context_window_fallbacks=None,
        )

    assert error_info.value is error


@pytest.mark.parametrize(
    "num_deployments, expected_timeout",
    [(1, 60), (2, 0.0)],
)
def test_timeout_for_rate_limit_error_with_healthy_deployments(num_deployments: int, expected_timeout: float) -> None:
    deployment_names: Final = (
        ("retry-test-model",) if num_deployments == 1 else ("retry-test-model", "other-test-model")
    )
    model_list: Final = [
        {
            "model_name": model_name,
            "litellm_params": {
                "model": "openai/retry-test-model",
                "api_key": "test-key",
            },
        }
        for model_name in deployment_names
    ]
    router: Final = Router(model_list=model_list)
    timeout: Final = router._time_to_sleep_before_retry(
        e=_rate_limit_error("60"),
        remaining_retries=2,
        num_retries=2,
        healthy_deployments=[{"model_name": "retry-test-model"}],
        all_deployments=model_list,
    )

    if expected_timeout == 0.0:
        assert timeout == 0.0
    else:
        assert timeout >= expected_timeout


def test_timeout_for_rate_limit_error_with_no_healthy_deployments() -> None:
    router: Final = _retry_test_router()
    timeout: Final = router._time_to_sleep_before_retry(
        e=_rate_limit_error("30"),
        remaining_retries=4,
        num_retries=4,
        healthy_deployments=[],
        all_deployments=[{"model_name": "retry-test-model"}],
    )

    assert timeout >= 30


def test_no_retry_for_not_found_error_404() -> None:
    router: Final = _retry_test_router()
    error: Final = litellm.NotFoundError(
        message="Model not found",
        model="retry-test-model",
        llm_provider="openai",
    )

    with pytest.raises(litellm.NotFoundError) as error_info:
        router.should_retry_this_error(
            error=error,
            healthy_deployments=[],
        )

    assert error_info.value is error


def test_no_retry_for_bad_request_error_400() -> None:
    router: Final = _retry_test_router()
    error: Final = litellm.BadRequestError(
        message="Invalid request",
        model="retry-test-model",
        llm_provider="openai",
    )

    with pytest.raises(litellm.BadRequestError) as error_info:
        router.should_retry_this_error(
            error=error,
            healthy_deployments=[{"model_name": "retry-test-model"}],
        )

    assert error_info.value is error


def test_no_retry_for_unprocessable_entity_error_422() -> None:
    router: Final = _retry_test_router()
    error: Final = litellm.UnprocessableEntityError(
        message="Unprocessable entity",
        model="retry-test-model",
        llm_provider="openai",
        response=httpx.Response(
            status_code=422,
            request=httpx.Request("POST", "https://example.invalid/v1"),
        ),
    )

    with pytest.raises(litellm.UnprocessableEntityError) as error_info:
        router.should_retry_this_error(
            error=error,
            healthy_deployments=[{"model_name": "retry-test-model"}],
        )

    assert error_info.value is error


def test_no_retry_when_no_healthy_deployments() -> None:
    router: Final = _retry_test_router()
    errors: Final = (
        litellm.InternalServerError(
            message="Internal server error",
            model="retry-test-model",
            llm_provider="openai",
        ),
        _rate_limit_error(),
        litellm.ServiceUnavailableError(
            message="Service unavailable",
            model="retry-test-model",
            llm_provider="openai",
        ),
        litellm.Timeout(
            message="Request timed out",
            model="retry-test-model",
            llm_provider="openai",
        ),
    )

    for error in errors:
        with pytest.raises(type(error)) as error_info:
            router.should_retry_this_error(
                error=error,
                healthy_deployments=[],
            )
        assert error_info.value is error


@pytest.mark.asyncio
async def test_router_retries_model_specific_and_global(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_asyncio_sleep(monkeypatch)
    monkeypatch.setattr(litellm, "num_retries", 0)
    router: Final = Router(
        model_list=[
            {
                "model_name": "retry-test-model",
                "litellm_params": {
                    "model": "openai/retry-test-model",
                    "api_key": "test-key",
                    "num_retries": 1,
                },
            }
        ]
    )
    with pytest.raises(litellm.RateLimitError) as error_info:
        await router.acompletion(
            model="retry-test-model",
            messages=[{"role": "user", "content": "Retry configuration test"}],
            mock_response="litellm.RateLimitError",
        )

    assert error_info.value.num_retries == 1
    assert error_info.value.max_retries == 1


@pytest.mark.asyncio
async def test_router_timeout_model_specific_and_global(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model: Final = "claude-haiku-4-5-20251001"
    router: Final = Router(
        model_list=[
            {
                "model_name": "anthropic-claude",
                "litellm_params": {
                    "model": f"anthropic/{model}",
                    "api_key": "test-key",
                    "timeout": 1,
                },
            }
        ],
        timeout=10,
    )
    client: Final = HTTPHandler()
    response: Final = httpx.Response(
        status_code=200,
        json={
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [{"type": "text", "text": "Retry timeout test"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        },
    )
    client_post: Final = AsyncMock(return_value=response)
    monkeypatch.setattr(client, "post", client_post)

    result: Final = await router.acompletion(
        model="anthropic-claude",
        messages=[{"role": "user", "content": "Retry timeout test"}],
        client=client,
    )

    assert result.choices[0].message.content == "Retry timeout test"
    call_args: Final = client_post.call_args
    assert call_args is not None
    assert call_args.kwargs["timeout"] == 1


@pytest.mark.asyncio
async def test_router_retry_num_retries_tracking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_asyncio_sleep(monkeypatch)
    router: Final = _retry_tracking_router(num_retries=3)

    with pytest.raises(litellm.RateLimitError) as error_info:
        await router.acompletion(
            model="retry-test-model",
            messages=[{"role": "user", "content": "Retry tracking test"}],
            mock_response="litellm.RateLimitError",
        )

    error: Final = error_info.value
    assert error.num_retries == 3
    assert error.max_retries == 3
    assert "LiteLLM Retried: 3 times" in str(error)
    assert "LiteLLM Max Retries: 3" in str(error)


@pytest.mark.asyncio
async def test_router_retry_num_retries_single_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_asyncio_sleep(monkeypatch)
    router: Final = _retry_tracking_router(num_retries=1)

    with pytest.raises(litellm.RateLimitError) as error_info:
        await router.acompletion(
            model="retry-test-model",
            messages=[{"role": "user", "content": "Single retry tracking test"}],
            mock_response="litellm.RateLimitError",
        )

    error: Final = error_info.value
    assert error.num_retries == 1
    assert error.max_retries == 1
