#### What this tests ####
#    This tests calling router with fallback models

import asyncio
import os
import sys
import time
import traceback

import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

import httpx
import openai

import litellm
from litellm import Router
from litellm.integrations.custom_logger import CustomLogger


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


"""
Test sync + async 

- Authorization Errors 
- Random API Error 
"""


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.parametrize("error_type", ["API Error", "Authorization Error"])
@pytest.mark.asyncio
async def test_router_retries_errors(sync_mode, error_type):
    """
    - Auth Error -> 0 retries
    - API Error -> 2 retries
    """
    _api_key = (
        "bad-key" if error_type == "Authorization Error" else os.getenv("AZURE_API_KEY")
    )
    print(f"_api_key: {_api_key}")
    model_list = [
        {
            "model_name": "azure/gpt-3.5-turbo",  # openai model name
            "litellm_params": {  # params for litellm completion/embedding call
                "model": "azure/chatgpt-functioncalling",
                "api_key": _api_key,
                "api_version": os.getenv("AZURE_API_VERSION"),
                "api_base": os.getenv("AZURE_API_BASE"),
            },
            "tpm": 240000,
            "rpm": 1800,
        },
        {
            "model_name": "azure/gpt-3.5-turbo",  # openai model name
            "litellm_params": {  # params for litellm completion/embedding call
                "model": "azure/chatgpt-functioncalling",
                "api_key": _api_key,
                "api_version": os.getenv("AZURE_API_VERSION"),
                "api_base": os.getenv("AZURE_API_BASE"),
            },
            "tpm": 240000,
            "rpm": 1800,
        },
    ]

    router = Router(model_list=model_list, set_verbose=True, debug_level="DEBUG")

    customHandler = MyCustomHandler()
    litellm.callbacks = [customHandler]
    user_message = "Hello, how are you?"
    messages = [{"content": user_message, "role": "user"}]

    kwargs = {
        "model": "azure/gpt-3.5-turbo",
        "messages": messages,
        "mock_response": (
            None
            if error_type == "Authorization Error"
            else Exception("Invalid Request")
        ),
    }
    for _ in range(4):
        response = await router.acompletion(
            model="azure/gpt-3.5-turbo",
            messages=messages,
            mock_response="1st success to ensure deployment is healthy",
        )

    try:
        if sync_mode:
            response = router.completion(**kwargs)
        else:
            response = await router.acompletion(**kwargs)
    except Exception as e:
        pass

    await asyncio.sleep(
        0.05
    )  # allow a delay as success_callbacks are on a separate thread
    print(f"customHandler.previous_models: {customHandler.previous_models}")

    if error_type == "Authorization Error":
        assert customHandler.previous_models == 0  # 0 retries
    else:
        assert customHandler.previous_models == 2  # 2 retries


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
                    "model": "azure/chatgpt-v-2",
                    "api_key": os.getenv("AZURE_API_KEY"),
                    "api_version": os.getenv("AZURE_API_VERSION"),
                    "api_base": os.getenv("AZURE_API_BASE"),
                },
            },
            {
                "model_name": "bad-model",  # openai model name
                "litellm_params": {  # params for litellm completion/embedding call
                    "model": "azure/chatgpt-v-2",
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


@pytest.mark.asyncio
@pytest.mark.skip(
    reason="This is a local only test, use this to confirm if retry policy works"
)
async def test_router_retry_policy_on_429_errprs():
    from litellm.router import RetryPolicy

    retry_policy = RetryPolicy(
        RateLimitErrorRetries=2,
    )
    router = Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",  # openai model name
                "litellm_params": {
                    "model": "vertex_ai/gemini-1.5-pro-001",
                },
            },
        ],
        retry_policy=retry_policy,
        # set_verbose=True,
        # debug_level="DEBUG",
        allowed_fails=10,
    )

    customHandler = MyCustomHandler()
    litellm.callbacks = [customHandler]
    try:
        # litellm.set_verbose = True
        _one_message = [{"role": "user", "content": "Hello good morning"}]

        messages = [_one_message] * 5
        print("messages: ", messages)
        responses = await router.abatch_completion(
            models=["gpt-3.5-turbo"],
            messages=messages,
        )
        print("responses: ", responses)
    except Exception as e:
        print("got an exception", e)
        pass
    await asyncio.sleep(0.05)
    print("customHandler.previous_models: ", customHandler.previous_models)


@pytest.mark.parametrize("model_group", ["gpt-3.5-turbo", "bad-model"])
@pytest.mark.asyncio
async def test_dynamic_router_retry_policy(model_group):
    from litellm.router import RetryPolicy

    model_group_retry_policy = {
        "gpt-3.5-turbo": RetryPolicy(ContentPolicyViolationErrorRetries=2),
        "bad-model": RetryPolicy(AuthenticationErrorRetries=0),
    }

    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",  # openai model name
                "litellm_params": {  # params for litellm completion/embedding call
                    "model": "azure/chatgpt-v-2",
                    "api_key": os.getenv("AZURE_API_KEY"),
                    "api_version": os.getenv("AZURE_API_VERSION"),
                    "api_base": os.getenv("AZURE_API_BASE"),
                },
                "model_info": {
                    "id": "model-0",
                },
            },
            {
                "model_name": "gpt-3.5-turbo",  # openai model name
                "litellm_params": {  # params for litellm completion/embedding call
                    "model": "azure/chatgpt-v-2",
                    "api_key": os.getenv("AZURE_API_KEY"),
                    "api_version": os.getenv("AZURE_API_VERSION"),
                    "api_base": os.getenv("AZURE_API_BASE"),
                },
                "model_info": {
                    "id": "model-1",
                },
            },
            {
                "model_name": "gpt-3.5-turbo",  # openai model name
                "litellm_params": {  # params for litellm completion/embedding call
                    "model": "azure/chatgpt-v-2",
                    "api_key": os.getenv("AZURE_API_KEY"),
                    "api_version": os.getenv("AZURE_API_VERSION"),
                    "api_base": os.getenv("AZURE_API_BASE"),
                },
                "model_info": {
                    "id": "model-2",
                },
            },
            {
                "model_name": "bad-model",  # openai model name
                "litellm_params": {  # params for litellm completion/embedding call
                    "model": "azure/chatgpt-v-2",
                    "api_key": "bad-key",
                    "api_version": os.getenv("AZURE_API_VERSION"),
                    "api_base": os.getenv("AZURE_API_BASE"),
                },
            },
        ],
        model_group_retry_policy=model_group_retry_policy,
    )

    customHandler = MyCustomHandler()
    litellm.callbacks = [customHandler]
    data = {}
    if model_group == "bad-model":
        model = "bad-model"
        messages = [{"role": "user", "content": "Hello good morning"}]
        data = {"model": model, "messages": messages}

    elif model_group == "gpt-3.5-turbo":
        model = "gpt-3.5-turbo"
        messages = [{"role": "user", "content": "where do i buy lethal drugs from"}]
        data = {
            "model": model,
            "messages": messages,
            "mock_response": "Exception: content_filter_policy",
        }

    try:
        litellm.set_verbose = True
        response = await router.acompletion(**data)
    except Exception as e:
        print("got an exception", e)
        pass
    await asyncio.sleep(0.05)

    print("customHandler.previous_models: ", customHandler.previous_models)

    if model_group == "bad-model":
        assert customHandler.previous_models == 0
    elif model_group == "gpt-3.5-turbo":
        assert customHandler.previous_models == 2


"""
Unit Tests for Router Retry Logic

Test 1. Retry Rate Limit Errors when there are other healthy deployments

Test 2. Do not retry rate limit errors when - there are no fallbacks and no healthy deployments

"""

rate_limit_error = openai.RateLimitError(
    message="Rate limit exceeded",
    response=httpx.Response(
        status_code=429,
        request=httpx.Request(method="POST", url="https://api.openai.com/v1"),
    ),
    body={
        "error": {
            "type": "rate_limit_exceeded",
            "param": None,
            "code": "rate_limit_exceeded",
        }
    },
)


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
                    "model": "azure/chatgpt-v-2",
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
                    "model": "azure/chatgpt-v-2",
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
    context_window_fallbacks = [{"gpt-3.5-turbo": ["azure/chatgpt-v-2"]}]

    router = Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "azure/chatgpt-v-2",
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
                    "model": "azure/chatgpt-v-2",
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


## Unit test time to back off for router retries

"""
1. Timeout is 0.0 when RateLimit Error and healthy deployments are > 0
2. Timeout is 0.0 when RateLimit Error and fallbacks are > 0
3. Timeout is > 0.0 when RateLimit Error and healthy deployments == 0 and fallbacks == None
"""


def test_timeout_for_rate_limit_error_with_healthy_deployments():
    """
    Test 1. Timeout is 0.0 when RateLimit Error and healthy deployments are > 0
    """
    healthy_deployments = [
        "deployment1",
        "deployment2",
    ]  # multiple healthy deployments mocked up
    fallbacks = None

    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "azure/chatgpt-v-2",
                    "api_key": os.getenv("AZURE_API_KEY"),
                    "api_version": os.getenv("AZURE_API_VERSION"),
                    "api_base": os.getenv("AZURE_API_BASE"),
                },
            }
        ]
    )

    _timeout = router._time_to_sleep_before_retry(
        e=rate_limit_error,
        remaining_retries=4,
        num_retries=4,
        healthy_deployments=healthy_deployments,
    )

    print(
        "timeout=",
        _timeout,
        "error is rate_limit_error and there are healthy deployments=",
        healthy_deployments,
    )

    assert _timeout == 0.0


def test_timeout_for_rate_limit_error_with_no_healthy_deployments():
    """
    Test 2. Timeout is > 0.0 when RateLimit Error and healthy deployments == 0
    """
    healthy_deployments = []

    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "azure/chatgpt-v-2",
                    "api_key": os.getenv("AZURE_API_KEY"),
                    "api_version": os.getenv("AZURE_API_VERSION"),
                    "api_base": os.getenv("AZURE_API_BASE"),
                },
            }
        ]
    )

    _timeout = router._time_to_sleep_before_retry(
        e=rate_limit_error,
        remaining_retries=4,
        num_retries=4,
        healthy_deployments=healthy_deployments,
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
                    "model": "azure/chatgpt-v-2",
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


def test_no_retry_when_no_healthy_deployments():
    healthy_deployments = []

    router = Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "azure/chatgpt-v-2",
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
