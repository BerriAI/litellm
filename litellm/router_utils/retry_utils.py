import asyncio
import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import httpx
import openai

import litellm
from litellm.litellm_core_utils.core_helpers import _get_parent_otel_span_from_kwargs
from litellm.types.router import DeploymentTypedDict, RetryPolicy


def run_with_retries(
    original_function: Callable,
    original_function_args: Tuple,
    original_function_kwargs: Dict[str, Any],
    num_retries: int,
    retry_after: int,  # min time to wait before retrying a failed request
    retry_policy: Optional[
        Union[RetryPolicy, Dict]
    ],  # set custom retries for different exceptions
    fallbacks: List,
    context_window_fallbacks: List,
    content_policy_fallbacks: List,
    get_healthy_deployments: Callable,
    log_retry: Callable,
    model_list: Optional[List[DeploymentTypedDict]],
):
    """
    Runs the specified function with retries and fallbacks.
    """
    async_task = async_run_with_retries(
        original_function=original_function,
        original_function_args=original_function_args,
        original_function_kwargs=original_function_kwargs,
        num_retries=num_retries,
        retry_after=retry_after,
        retry_policy=retry_policy,
        fallbacks=fallbacks,
        context_window_fallbacks=context_window_fallbacks,
        content_policy_fallbacks=content_policy_fallbacks,
        get_healthy_deployments=get_healthy_deployments,
        log_retry=log_retry,
        model_list=model_list,
    )
    try:
        # Check if an event loop is already running
        loop = asyncio.get_running_loop()
        # If running in an async context, return the coroutine for awaiting
        return async_task
    except RuntimeError:
        # If no event loop is running, create a new one and run the task
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(async_task)
        finally:
            loop.close()


async def async_run_with_retries(
    original_function: Callable,
    original_function_args: Tuple,
    original_function_kwargs: Dict[str, Any],
    num_retries: int,
    retry_after: int,  # min time to wait before retrying a failed request
    retry_policy: Optional[
        Union[RetryPolicy, Dict]
    ],  # set custom retries for different exceptions
    fallbacks: List,
    context_window_fallbacks: List,
    content_policy_fallbacks: List,
    get_healthy_deployments: Callable,
    log_retry: Callable,
    model_list: Optional[List[DeploymentTypedDict]],
):
    """
    Runs the specified function asynchronously with retries and fallbacks.
    """
    parent_otel_span = _get_parent_otel_span_from_kwargs(original_function_kwargs)

    ## ADD MODEL GROUP SIZE TO METADATA - used for model_group_rate_limit_error tracking
    model_group: Optional[str] = original_function_kwargs.get("model")
    _metadata: dict = original_function_kwargs.get("metadata") or {}
    if "model_group" in _metadata and isinstance(_metadata["model_group"], str):
        if model_list is not None:
            _metadata.update({"model_group_size": len(model_list)})

    try:
        handle_mock_testing_rate_limit_error(
            model_group=model_group, kwargs=original_function_kwargs
        )
        # if the function call is successful, no exception will be raised and we'll break out of the loop
        response = await _make_call(
            original_function, *original_function_args, **original_function_kwargs
        )
        return response
    except Exception as e:
        current_attempt = None
        original_exception = e

        """
        Retry Logic
        """
        _healthy_deployments, _all_deployments = await get_healthy_deployments(
            model=original_function_kwargs.get("model") or "",
            parent_otel_span=parent_otel_span,
        )

        # raises an exception if this error should not be retried
        should_retry_this_error(
            error=e,
            healthy_deployments=_healthy_deployments,
            all_deployments=_all_deployments,
            context_window_fallbacks=context_window_fallbacks,
            regular_fallbacks=fallbacks,
            content_policy_fallbacks=content_policy_fallbacks,
        )

        if retry_policy is not None:
            # get num_retries from retry policy
            _retry_policy_retries = _get_num_retries_from_retry_policy(
                retry_policy=retry_policy,
                exception=original_exception,
            )
            if _retry_policy_retries is not None:
                num_retries = _retry_policy_retries
        ## LOGGING
        if num_retries > 0:
            original_function_kwargs = log_retry(
                kwargs=original_function_kwargs,
                e=original_exception,
            )
        else:
            raise

        # decides how long to sleep before retry
        retry_after = time_to_sleep_before_retry(  # type: ignore
            e=original_exception,
            remaining_retries=num_retries,
            num_retries=num_retries,
            retry_after=retry_after,
            healthy_deployments=_healthy_deployments,
        )

        await asyncio.sleep(retry_after)
        for current_attempt in range(num_retries):
            try:
                # if the function call is successful, no exception will be raised and we'll break out of the loop
                response = await _make_call(
                    original_function,
                    *original_function_args,
                    **original_function_kwargs,
                )
                if inspect.iscoroutinefunction(
                    response
                ):  # async errors are often returned as coroutines
                    response = await response
                return response

            except Exception as e:
                ## LOGGING
                original_function_kwargs = log_retry(
                    kwargs=original_function_kwargs,
                    e=e,
                )
                remaining_retries = num_retries - current_attempt
                _model: Optional[str] = original_function_kwargs.get("model")  # type: ignore
                if _model is not None:
                    _healthy_deployments, _ = await get_healthy_deployments(
                        model=_model,
                        parent_otel_span=parent_otel_span,
                    )
                else:
                    _healthy_deployments = []
                _timeout = time_to_sleep_before_retry(
                    e=original_exception,
                    remaining_retries=remaining_retries,
                    num_retries=num_retries,
                    retry_after=retry_after,
                    healthy_deployments=_healthy_deployments,
                )
                await asyncio.sleep(_timeout)

        if type(original_exception) in litellm.LITELLM_EXCEPTION_TYPES:
            setattr(original_exception, "max_retries", num_retries)
            setattr(original_exception, "num_retries", current_attempt)

        raise original_exception


def should_retry_this_error(
    error: Exception,
    healthy_deployments: Optional[List] = None,
    all_deployments: Optional[List] = None,
    context_window_fallbacks: Optional[List] = None,
    content_policy_fallbacks: Optional[List] = None,
    regular_fallbacks: Optional[List] = None,
):
    """
    1. raise an exception for ContextWindowExceededError if context_window_fallbacks is not None
    2. raise an exception for ContentPolicyViolationError if content_policy_fallbacks is not None

    2. raise an exception for RateLimitError if
        - there are no fallbacks
        - there are no healthy deployments in the same model group
    """
    _num_healthy_deployments = 0
    if healthy_deployments is not None and isinstance(healthy_deployments, list):
        _num_healthy_deployments = len(healthy_deployments)
    _num_all_deployments = 0
    if all_deployments is not None and isinstance(all_deployments, list):
        _num_all_deployments = len(all_deployments)

    ### CHECK IF RATE LIMIT / CONTEXT WINDOW ERROR / CONTENT POLICY VIOLATION ERROR w/ fallbacks available / Bad Request Error
    if (
        isinstance(error, litellm.ContextWindowExceededError)
        and context_window_fallbacks is not None
    ):
        raise error

    if (
        isinstance(error, litellm.ContentPolicyViolationError)
        and content_policy_fallbacks is not None
    ):
        raise error

    if isinstance(error, litellm.NotFoundError):
        raise error
    # Error we should only retry if there are other deployments
    if isinstance(error, openai.RateLimitError):
        if (
            _num_healthy_deployments <= 0  # if no healthy deployments
            and regular_fallbacks is not None  # and fallbacks available
            and len(regular_fallbacks) > 0
        ):
            raise error  # then raise the error

    if isinstance(error, openai.AuthenticationError):
        """
        - if other deployments available -> retry
        - else -> raise error
        """
        if (
            _num_all_deployments <= 1
        ):  # if there is only 1 deployment for this model group then don't retry
            raise error  # then raise error

    # Do not retry if there are no healthy deployments
    # just raise the error
    if _num_healthy_deployments <= 0:  # if no healthy deployments
        raise error

    return True


def handle_mock_testing_rate_limit_error(
    kwargs: dict, model_group: Optional[str] = None
):
    """
    Helper function to raise a mock litellm.RateLimitError error for testing purposes.

    Raises:
        litellm.RateLimitError error when `mock_testing_rate_limit_error=True` passed in request params
    """
    mock_testing_rate_limit_error: Optional[bool] = kwargs.pop(
        "mock_testing_rate_limit_error", None
    )
    if (
        mock_testing_rate_limit_error is not None
        and mock_testing_rate_limit_error is True
    ):
        # TODO: Figure out logging - take a logger arg?
        # verbose_router_logger.info(
        #     f"litellm.router.py::_mock_rate_limit_error() - Raising mock RateLimitError for model={model_group}"
        # )
        raise litellm.RateLimitError(
            model=model_group,
            llm_provider="",
            message=f"This is a mock exception for model={model_group}, to trigger a rate limit error.",
        )


async def _make_call(original_function: Any, *args, **kwargs):
    """
    Handler for making a call to the .completion()/.embeddings()/etc. functions.
    """
    model_group = kwargs.get("model")
    response = original_function(*args, **kwargs)
    if inspect.iscoroutinefunction(response) or inspect.isawaitable(response):
        response = await response
    ## PROCESS RESPONSE HEADERS
    await _set_response_headers(response=response, model_group=model_group)

    return response


async def _set_response_headers(
    response: Any, model_group: Optional[str] = None
) -> Any:
    """
    Add the most accurate rate limit headers for a given model response.

    ## TODO: add model group rate limit headers
    # - if healthy_deployments > 1, return model group rate limit headers
    # - else return the model's rate limit headers
    """
    return response


def _get_num_retries_from_retry_policy(
    retry_policy: Union[RetryPolicy, Dict],
    exception: Exception,
):
    """
    BadRequestErrorRetries: Optional[int] = None
    AuthenticationErrorRetries: Optional[int] = None
    TimeoutErrorRetries: Optional[int] = None
    RateLimitErrorRetries: Optional[int] = None
    ContentPolicyViolationErrorRetries: Optional[int] = None
    """
    if isinstance(retry_policy, dict):
        retry_policy = RetryPolicy(**retry_policy)

    if (
        isinstance(exception, litellm.BadRequestError)
        and retry_policy.BadRequestErrorRetries is not None
    ):
        return retry_policy.BadRequestErrorRetries
    if (
        isinstance(exception, litellm.AuthenticationError)
        and retry_policy.AuthenticationErrorRetries is not None
    ):
        return retry_policy.AuthenticationErrorRetries
    if (
        isinstance(exception, litellm.Timeout)
        and retry_policy.TimeoutErrorRetries is not None
    ):
        return retry_policy.TimeoutErrorRetries
    if (
        isinstance(exception, litellm.RateLimitError)
        and retry_policy.RateLimitErrorRetries is not None
    ):
        return retry_policy.RateLimitErrorRetries
    if (
        isinstance(exception, litellm.ContentPolicyViolationError)
        and retry_policy.ContentPolicyViolationErrorRetries is not None
    ):
        return retry_policy.ContentPolicyViolationErrorRetries


def time_to_sleep_before_retry(
    e: Exception,
    remaining_retries: int,
    num_retries: int,
    retry_after: int,
    healthy_deployments: Optional[List] = None,
) -> Union[int, float]:
    """
    Calculate back-off, then retry

    It should instantly retry only when:
        1. there are healthy deployments in the same model group
        2. there are fallbacks for the completion call
    """
    if (
        healthy_deployments is not None
        and isinstance(healthy_deployments, list)
        and len(healthy_deployments) > 1
    ):
        return 0

    response_headers: Optional[httpx.Headers] = None
    if hasattr(e, "response") and hasattr(e.response, "headers"):  # type: ignore
        response_headers = e.response.headers  # type: ignore
    if hasattr(e, "litellm_response_headers"):
        response_headers = e.litellm_response_headers  # type: ignore

    if response_headers is not None:
        timeout = litellm._calculate_retry_after(
            remaining_retries=remaining_retries,
            max_retries=num_retries,
            response_headers=response_headers,
            min_timeout=retry_after,
        )

    else:
        timeout = litellm._calculate_retry_after(
            remaining_retries=remaining_retries,
            max_retries=num_retries,
            min_timeout=retry_after,
        )

    return timeout
