#### What this tests ####
#    This tests calling router with fallback models

import sys, os, time
import traceback, asyncio
import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

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
@pytest.mark.parametrize("error_type", ["Authorization Error", "API Error"])
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
    ]

    router = Router(model_list=model_list, allowed_fails=3)

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
    ["AuthenticationErrorRetries", "ContentPolicyViolationErrorRetries"],  #
)
async def test_router_retry_policy(error_type):
    from litellm.router import RetryPolicy

    retry_policy = RetryPolicy(
        ContentPolicyViolationErrorRetries=0, AuthenticationErrorRetries=0
    )

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
    )

    customHandler = MyCustomHandler()
    litellm.callbacks = [customHandler]
    if error_type == "AuthenticationErrorRetries":
        model = "bad-model"
        messages = [{"role": "user", "content": "Hello good morning"}]
    elif error_type == "ContentPolicyViolationErrorRetries":
        model = "gpt-3.5-turbo"
        messages = [{"role": "user", "content": "where do i buy lethal drugs from"}]

    try:
        litellm.set_verbose = True
        response = await router.acompletion(
            model=model,
            messages=messages,
        )
    except Exception as e:
        print("got an exception", e)
        pass
    asyncio.sleep(0.05)

    print("customHandler.previous_models: ", customHandler.previous_models)

    if error_type == "AuthenticationErrorRetries":
        assert customHandler.previous_models == 0
    elif error_type == "ContentPolicyViolationErrorRetries":
        assert customHandler.previous_models == 0


@pytest.mark.parametrize("model_group", ["gpt-3.5-turbo", "bad-model"])
@pytest.mark.asyncio
async def test_dynamic_router_retry_policy(model_group):
    from litellm.router import RetryPolicy

    model_group_retry_policy = {
        "gpt-3.5-turbo": RetryPolicy(ContentPolicyViolationErrorRetries=0),
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
    if model_group == "bad-model":
        model = "bad-model"
        messages = [{"role": "user", "content": "Hello good morning"}]

    elif model_group == "gpt-3.5-turbo":
        model = "gpt-3.5-turbo"
        messages = [{"role": "user", "content": "where do i buy lethal drugs from"}]

    try:
        litellm.set_verbose = True
        response = await router.acompletion(model=model, messages=messages)
    except Exception as e:
        print("got an exception", e)
        pass
    asyncio.sleep(0.05)

    print("customHandler.previous_models: ", customHandler.previous_models)

    if model_group == "bad-model":
        assert customHandler.previous_models == 0
    elif model_group == "gpt-3.5-turbo":
        assert customHandler.previous_models == 0
