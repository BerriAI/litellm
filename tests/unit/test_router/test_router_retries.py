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
from litellm.router import AllowedFailsPolicy, RetryPolicy
import os

internal_server_error = litellm.InternalServerError(
    message="internal server error",
    model="gpt-12",
    llm_provider="azure",
)

rate_limit_error = litellm.RateLimitError(
    message="rate limit error",
    model="gpt-12",
    llm_provider="azure",
)

service_unavailable_error = litellm.ServiceUnavailableError(
    message="service unavailable error",
    model="gpt-12",
    llm_provider="azure",
)

timeout_error = litellm.Timeout(
    message="timeout error",
    model="gpt-12",
    llm_provider="azure",
)


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
    ["ContentPolicyViolationErrorRetries"],  # "AuthenticationErrorRetries",
)
async def test_router_retry_policy(error_type):
    from litellm.router import AllowedFailsPolicy, RetryPolicy

    retry_policy = RetryPolicy(
        ContentPolicyViolationErrorRetries=3, AuthenticationErrorRetries=0
    )

    allowed_fails_policy = AllowedFailsPolicy(
        ContentPolicyViolationErrorAllowedFails=1000,
        RateLimitErrorAllowedFails=100,
    )

    router = Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",  # openai model name
                "litellm_params": {  # params for litellm completion/embedding call
                    "model": "azure/gpt-4.1-mini",
                    "api_key": os.getenv("AZURE_API_KEY"),
                    "api_version": os.getenv("AZURE_API_VERSION"),
                    "api_base": os.getenv("AZURE_API_BASE"),
                },
            },
            {
                "model_name": "bad-model",  # openai model name
                "litellm_params": {  # params for litellm completion/embedding call
                    "model": "azure/gpt-4.1-mini",
                    "api_key": "bad-key",
                    "api_version": os.getenv("AZURE_API_VERSION"),
                    "api_base": os.getenv("AZURE_API_BASE"),
                },
            },
        ],
        retry_policy=retry_policy,
        allowed_fails_policy=allowed_fails_policy,
    )

    customHandler = MyCustomHandler()
    litellm.callbacks = [customHandler]
    data = {}
    if error_type == "AuthenticationErrorRetries":
        model = "bad-model"
        messages = [{"role": "user", "content": "Hello good morning"}]
        data = {"model": model, "messages": messages}
    elif error_type == "ContentPolicyViolationErrorRetries":
        model = "gpt-3.5-turbo"
        messages = [{"role": "user", "content": "where do i buy lethal drugs from"}]
        mock_response = "Exception: content_filter_policy"
        data = {"model": model, "messages": messages, "mock_response": mock_response}

    try:
        litellm.set_verbose = True
        await router.acompletion(**data)
    except Exception as e:
        print("got an exception", e)
        pass
    await asyncio.sleep(1)

    print("customHandler.previous_models: ", customHandler.previous_models)

    if error_type == "AuthenticationErrorRetries":
        assert customHandler.previous_models == 0
    elif error_type == "ContentPolicyViolationErrorRetries":
        assert customHandler.previous_models == 3


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


def test_retry_rate_limit_error_with_healthy_deployments():
    """
    Test 1. It SHOULD retry when there is a rate limit error and len(healthy_deployments) > 0
    """
    healthy_deployments = [
        "deployment1",
        "deployment2",
    ]  # multiple healthy deployments mocked up

    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "azure/gpt-4.1-mini",
                    "api_key": os.getenv("AZURE_API_KEY"),
                    "api_version": os.getenv("AZURE_API_VERSION"),
                    "api_base": os.getenv("AZURE_API_BASE"),
                },
            }
        ]
    )

    # Act & Assert
    try:
        response = router.should_retry_this_error(
            error=rate_limit_error, healthy_deployments=healthy_deployments
        )
        print("response from should_retry_this_error: ", response)
    except Exception as e:
        pytest.fail(
            "Should not have raised an error, since there are healthy deployments. Raises",
            e,
        )


def test_do_retry_rate_limit_error_with_no_fallbacks_and_no_healthy_deployments():
    """
    Test 2. It SHOULD NOT Retry, when healthy_deployments is [] and fallbacks is None
    """
    healthy_deployments = []

    router = Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "azure/gpt-4.1-mini",
                    "api_key": os.getenv("AZURE_API_KEY"),
                    "api_version": os.getenv("AZURE_API_VERSION"),
                    "api_base": os.getenv("AZURE_API_BASE"),
                },
            }
        ]
    )

    # Act & Assert
    try:
        response = router.should_retry_this_error(
            error=rate_limit_error, healthy_deployments=healthy_deployments
        )
        pytest.fail("Should have raised an error")
    except Exception as e:
        print("got an exception", e)
        pass


def test_raise_context_window_exceeded_error():
    """
    Trigger Context Window fallback, when context_window_fallbacks is not None
    """
    context_window_error = litellm.ContextWindowExceededError(
        message="Context window exceeded",
        response=httpx.Response(
            status_code=400,
            request=httpx.Request(method="POST", url="https://api.openai.com/v1"),
        ),
        llm_provider="azure",
        model="gpt-3.5-turbo",
    )
    context_window_fallbacks = [{"gpt-3.5-turbo": ["azure/gpt-4.1-mini"]}]

    router = Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "azure/gpt-4.1-mini",
                    "api_key": os.getenv("AZURE_API_KEY"),
                    "api_version": os.getenv("AZURE_API_VERSION"),
                    "api_base": os.getenv("AZURE_API_BASE"),
                },
            }
        ]
    )

    try:
        response = router.should_retry_this_error(
            error=context_window_error,
            healthy_deployments=None,
            context_window_fallbacks=context_window_fallbacks,
        )
        pytest.fail(
            "Expected to raise context window exceeded error -> trigger fallback"
        )
    except Exception as e:
        pass


def test_raise_context_window_exceeded_error_no_retry():
    """
    Do not Retry Context Window Exceeded Error, when context_window_fallbacks is None
    """
    context_window_error = litellm.ContextWindowExceededError(
        message="Context window exceeded",
        response=httpx.Response(
            status_code=400,
            request=httpx.Request(method="POST", url="https://api.openai.com/v1"),
        ),
        llm_provider="azure",
        model="gpt-3.5-turbo",
    )
    context_window_fallbacks = None

    router = Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "azure/gpt-4.1-mini",
                    "api_key": os.getenv("AZURE_API_KEY"),
                    "api_version": os.getenv("AZURE_API_VERSION"),
                    "api_base": os.getenv("AZURE_API_BASE"),
                },
            }
        ]
    )

    try:
        response = router.should_retry_this_error(
            error=context_window_error,
            healthy_deployments=None,
            context_window_fallbacks=context_window_fallbacks,
        )
        assert (
            response == True
        ), "Should not have raised exception since we do not have context window fallbacks"
    except litellm.ContextWindowExceededError:
        pass


@pytest.mark.parametrize("num_deployments, expected_timeout", [(1, 60), (2, 0.0)])
def test_timeout_for_rate_limit_error_with_healthy_deployments(
    num_deployments, expected_timeout
):
    """
    Test 1. Timeout is 0.0 when RateLimit Error and healthy deployments are > 0
    """
    cooldown_time = 60
    rate_limit_error = litellm.RateLimitError(
        message="{RouterErrors.no_deployments_available.value}. 12345 Passed model={model_group}. Deployments={deployment_dict}",
        llm_provider="",
        model="gpt-3.5-turbo",
        response=httpx.Response(
            status_code=429,
            content="",
            headers={"retry-after": str(cooldown_time)},  # type: ignore
            request=httpx.Request(method="tpm_rpm_limits", url="https://github.com/BerriAI/litellm"),  # type: ignore
        ),
    )
    model_list = [
        {
            "model_name": "gpt-3.5-turbo",
            "litellm_params": {
                "model": "azure/gpt-4.1-mini",
                "api_key": os.getenv("AZURE_API_KEY"),
                "api_version": os.getenv("AZURE_API_VERSION"),
                "api_base": os.getenv("AZURE_API_BASE"),
            },
        }
    ]
    if num_deployments == 2:
        model_list.append(
            {
                "model_name": "gpt-4",
                "litellm_params": {"model": "gpt-3.5-turbo"},
            }
        )

    router = litellm.Router(model_list=model_list)

    _timeout = router._time_to_sleep_before_retry(
        e=rate_limit_error,
        remaining_retries=2,
        num_retries=2,
        healthy_deployments=[
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "api_key": "my-key",
                    "api_base": "https://openai-gpt-4-test-v-1.openai.azure.com",
                    "model": "azure/gpt-4.1-mini",
                },
                "model_info": {
                    "id": "0e30bc8a63fa91ae4415d4234e231b3f9e6dd900cac57d118ce13a720d95e9d6",
                    "db_model": False,
                },
            }
        ],
        all_deployments=model_list,
    )

    if expected_timeout == 0.0:
        assert _timeout == expected_timeout
    else:
        assert _timeout > 0.0


def test_timeout_for_rate_limit_error_with_no_healthy_deployments():
    """
    Test 2. Timeout is > 0.0 when RateLimit Error and healthy deployments == 0
    """
    healthy_deployments = []
    model_list = [
        {
            "model_name": "gpt-3.5-turbo",
            "litellm_params": {
                "model": "azure/gpt-4.1-mini",
                "api_key": os.getenv("AZURE_API_KEY"),
                "api_version": os.getenv("AZURE_API_VERSION"),
                "api_base": os.getenv("AZURE_API_BASE"),
            },
        }
    ]

    router = litellm.Router(model_list=model_list)

    _timeout = router._time_to_sleep_before_retry(
        e=rate_limit_error,
        remaining_retries=4,
        num_retries=4,
        healthy_deployments=healthy_deployments,
        all_deployments=model_list,
    )

    print(
        "timeout=",
        _timeout,
        "error is rate_limit_error and there are no healthy deployments",
    )

    assert _timeout > 0.0


def test_no_retry_for_not_found_error_404():
    healthy_deployments = []

    router = Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "azure/gpt-4.1-mini",
                    "api_key": os.getenv("AZURE_API_KEY"),
                    "api_version": os.getenv("AZURE_API_VERSION"),
                    "api_base": os.getenv("AZURE_API_BASE"),
                },
            }
        ]
    )

    # Act & Assert
    error = litellm.NotFoundError(
        message="404 model not found",
        model="gpt-12",
        llm_provider="azure",
    )
    try:
        response = router.should_retry_this_error(
            error=error, healthy_deployments=healthy_deployments
        )
        pytest.fail(
            "Should have raised an exception 404 NotFoundError should never be retried, it's typically model_not_found error"
        )
    except Exception as e:
        print("got exception", e)


def test_no_retry_for_bad_request_error_400():
    """
    Test that 400 BadRequestError is NOT retried, even if healthy deployments exist.
    This tests the fix for GitHub issue #19216.
    """
    healthy_deployments = ["deployment1", "deployment2"]  # Multiple healthy deployments

    router = Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "azure/gpt-4.1-mini",
                    "api_key": os.getenv("AZURE_API_KEY"),
                    "api_version": os.getenv("AZURE_API_VERSION"),
                    "api_base": os.getenv("AZURE_API_BASE"),
                },
            }
        ]
    )

    # Act & Assert
    error = litellm.BadRequestError(
        message="400 Invalid request parameters",
        model="gpt-3.5-turbo",
        llm_provider="azure",
    )
    try:
        response = router.should_retry_this_error(
            error=error, healthy_deployments=healthy_deployments
        )
        pytest.fail(
            "Should have raised BadRequestError - 400 errors should never be retried"
        )
    except litellm.BadRequestError as e:
        print("Correctly raised BadRequestError without retry:", e)


def test_no_retry_for_unprocessable_entity_error_422():
    """
    Test that 422 UnprocessableEntityError is NOT retried, even if healthy deployments exist.
    """
    healthy_deployments = ["deployment1", "deployment2"]  # Multiple healthy deployments

    router = Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "azure/gpt-4.1-mini",
                    "api_key": os.getenv("AZURE_API_KEY"),
                    "api_version": os.getenv("AZURE_API_VERSION"),
                    "api_base": os.getenv("AZURE_API_BASE"),
                },
            }
        ]
    )

    # Act & Assert
    error = litellm.UnprocessableEntityError(
        message="422 Unprocessable Entity",
        model="gpt-3.5-turbo",
        llm_provider="azure",
        response=httpx.Response(
            status_code=422,
            request=httpx.Request(method="POST", url="https://api.openai.com/v1"),
        ),
    )
    try:
        response = router.should_retry_this_error(
            error=error, healthy_deployments=healthy_deployments
        )
        pytest.fail(
            "Should have raised UnprocessableEntityError - 422 errors should never be retried"
        )
    except litellm.UnprocessableEntityError as e:
        print("Correctly raised UnprocessableEntityError without retry:", e)


def test_no_retry_when_no_healthy_deployments():
    healthy_deployments = []

    router = Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "azure/gpt-4.1-mini",
                    "api_key": os.getenv("AZURE_API_KEY"),
                    "api_version": os.getenv("AZURE_API_VERSION"),
                    "api_base": os.getenv("AZURE_API_BASE"),
                },
            }
        ]
    )

    for error in [
        internal_server_error,
        rate_limit_error,
        service_unavailable_error,
        timeout_error,
    ]:
        try:
            response = router.should_retry_this_error(
                error=error, healthy_deployments=healthy_deployments
            )
            pytest.fail(
                "Should have raised an exception,  there's no point retrying an error when there are 0 healthy deployments"
            )
        except Exception as e:
            print("got exception", e)


@pytest.mark.asyncio
async def test_router_retries_model_specific_and_global():
    from unittest.mock import MagicMock, patch

    litellm.num_retries = 0
    router = Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                    "num_retries": 1,
                },
            }
        ]
    )

    with patch.object(
        router, "_time_to_sleep_before_retry"
    ) as mock_async_function_with_retries:
        try:
            await router.acompletion(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hello, how are you?"}],
                mock_response="litellm.RateLimitError",
            )
        except Exception as e:
            print("got exception", e)

        mock_async_function_with_retries.assert_called_once()

        assert mock_async_function_with_retries.call_args.kwargs["num_retries"] == 1


@pytest.mark.usefixtures("fake_provider_credentials")
@pytest.mark.asyncio
async def test_router_timeout_model_specific_and_global():
    from unittest.mock import MagicMock, patch

    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    router = Router(
        model_list=[
            {
                "model_name": "anthropic-claude",
                "litellm_params": {
                    "model": f"anthropic/{os.environ.get('CI_CD_DEFAULT_ANTHROPIC_MODEL', 'claude-haiku-4-5-20251001')}",
                    "timeout": 1,
                },
            }
        ],
        timeout=10,
    )

    client = HTTPHandler()

    with patch.object(client, "post") as mock_client:
        try:
            await router.acompletion(
                model="anthropic-claude",
                messages=[{"role": "user", "content": "Hello, how are you?"}],
                client=client,
            )
        except Exception as e:
            print("got exception", e)

        mock_client.assert_called()

        assert mock_client.call_args.kwargs["timeout"] == 1


@pytest.mark.asyncio
async def test_router_retry_num_retries_tracking():
    """
    Test that num_retries attribute is correctly set on exceptions when all retries are exhausted.

    This verifies the fix for the bug where num_retries was incorrectly set to current_attempt
    (0-indexed) instead of the actual number of retries attempted.
    """
    from unittest.mock import AsyncMock, patch

    router = Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                },
            }
        ],
        num_retries=3,  # Set at router level to ensure it's used
    )

    # Mock make_call to always raise a RateLimitError
    async def mock_make_call(*args, **kwargs):
        raise litellm.RateLimitError(
            message="Rate limit exceeded",
            model="gpt-3.5-turbo",
            llm_provider="openai",
        )

    with patch.object(router, "make_call", side_effect=mock_make_call):
        with patch.object(
            router,
            "_async_get_healthy_deployments",
            return_value=(
                [{"model_info": {"id": "test-id"}}],
                [{"model_info": {"id": "test-id"}}],
            ),
        ):
            with patch.object(
                router, "_time_to_sleep_before_retry", return_value=0.01
            ):  # Fast retries for testing
                with pytest.raises(litellm.RateLimitError) as exc_info:
                    await router.acompletion(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": "Hello"}],
                    )
                e = exc_info.value
                assert hasattr(
                    e, "num_retries"
                ), "Exception should have num_retries attribute"
                assert hasattr(
                    e, "max_retries"
                ), "Exception should have max_retries attribute"
                assert (
                    e.num_retries == 3
                ), f"Expected num_retries to be 3, got {e.num_retries}"
                assert (
                    e.max_retries == 3
                ), f"Expected max_retries to be 3, got {e.max_retries}"

                # Verify the error message includes correct retry information
                error_str = str(e)
                assert (
                    "LiteLLM Retried: 3 times" in error_str
                ), f"Error message should indicate 3 retries: {error_str}"
                assert (
                    "LiteLLM Max Retries: 3" in error_str
                ), f"Error message should show max retries: {error_str}"


@pytest.mark.asyncio
async def test_router_retry_num_retries_single_retry():
    """
    Test num_retries tracking with a single retry to verify edge case handling.
    """
    from unittest.mock import patch

    router = Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                },
            }
        ],
        num_retries=1,  # Set at router level - single retry
    )

    # Mock make_call to always raise a Timeout error
    async def mock_make_call(*args, **kwargs):
        raise litellm.Timeout(
            message="Request timed out",
            model="gpt-3.5-turbo",
            llm_provider="openai",
        )

    with patch.object(router, "make_call", side_effect=mock_make_call):
        with patch.object(
            router,
            "_async_get_healthy_deployments",
            return_value=(
                [{"model_info": {"id": "test-id"}}],
                [{"model_info": {"id": "test-id"}}],
            ),
        ):
            with patch.object(router, "_time_to_sleep_before_retry", return_value=0.01):
                with pytest.raises(litellm.Timeout) as exc_info:
                    await router.acompletion(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": "Hello"}],
                    )
                e = exc_info.value
                assert (
                    e.num_retries == 1
                ), f"Expected num_retries to be 1, got {e.num_retries}"
                assert (
                    e.max_retries == 1
                ), f"Expected max_retries to be 1, got {e.max_retries}"


class MyCustomHandler(CustomLogger):
    success: bool = False
    failure: bool = False
    previous_models: int = 0

    def log_pre_api_call(self, model, messages, kwargs):
        print(f"Pre-API Call")
        print(
            f"previous_models: {kwargs['litellm_params']['metadata'].get('previous_models', None)}"
        )
        self.previous_models = len(
            kwargs["litellm_params"]["metadata"].get("previous_models", [])
        )  # {"previous_models": [{"model": litellm_model_name, "exception_type": AuthenticationError, "exception_string": <complete_traceback>}]}
        print(f"self.previous_models: {self.previous_models}")

    def log_post_api_call(self, kwargs, response_obj, start_time, end_time):
        print(
            f"Post-API Call - response object: {response_obj}; model: {kwargs['model']}"
        )

    def log_stream_event(self, kwargs, response_obj, start_time, end_time):
        print(f"On Stream")

    def async_log_stream_event(self, kwargs, response_obj, start_time, end_time):
        print(f"On Stream")

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        print(f"On Success")

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        print(f"On Success")

    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        print(f"On Failure")


internal_server_error = litellm.InternalServerError(
    message="internal server error",
    model="gpt-12",
    llm_provider="azure",
)


service_unavailable_error = litellm.ServiceUnavailableError(
    message="service unavailable error",
    model="gpt-12",
    llm_provider="azure",
)


timeout_error = litellm.Timeout(
    message="timeout error",
    model="gpt-12",
    llm_provider="azure",
)
