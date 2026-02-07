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
from unittest.mock import AsyncMock, MagicMock, patch

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


kwargs = {
    "model": "azure/gpt-3.5-turbo",
    "messages": [{"role": "user", "content": "Hey, how's it going?"}],
}


def test_sync_fallbacks():
    try:
        model_list = [
            {  # list of model deployments
                "model_name": "azure/gpt-3.5-turbo",  # openai model name
                "litellm_params": {  # params for litellm completion/embedding call
                    "model": "azure/gpt-4.1-mini",
                    "api_key": "bad-key",
                    "api_version": os.getenv("AZURE_API_VERSION"),
                    "api_base": os.getenv("AZURE_API_BASE"),
                },
                "tpm": 240000,
                "rpm": 1800,
            },
            {  # list of model deployments
                "model_name": "azure/gpt-3.5-turbo-context-fallback",  # openai model name
                "litellm_params": {  # params for litellm completion/embedding call
                    "model": "azure/gpt-4.1-mini",
                    "api_key": os.getenv("AZURE_API_KEY"),
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
                    "api_key": "bad-key",
                    "api_version": os.getenv("AZURE_API_VERSION"),
                    "api_base": os.getenv("AZURE_API_BASE"),
                },
                "tpm": 240000,
                "rpm": 1800,
            },
            {
                "model_name": "gpt-3.5-turbo",  # openai model name
                "litellm_params": {  # params for litellm completion/embedding call
                    "model": "gpt-3.5-turbo",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                },
                "tpm": 1000000,
                "rpm": 9000,
            },
            {
                "model_name": "gpt-3.5-turbo-16k",  # openai model name
                "litellm_params": {  # params for litellm completion/embedding call
                    "model": "gpt-3.5-turbo-16k",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                },
                "tpm": 1000000,
                "rpm": 9000,
            },
        ]

        litellm.set_verbose = True
        customHandler = MyCustomHandler()
        litellm.callbacks = [customHandler]
        router = Router(
            model_list=model_list,
            fallbacks=[{"azure/gpt-3.5-turbo": ["gpt-3.5-turbo"]}],
            context_window_fallbacks=[
                {"azure/gpt-3.5-turbo-context-fallback": ["gpt-3.5-turbo-16k"]},
                {"gpt-3.5-turbo": ["gpt-3.5-turbo-16k"]},
            ],
            set_verbose=False,
        )
        response = router.completion(**kwargs)
        print(f"response: {response}")
        time.sleep(0.05)  # allow a delay as success_callbacks are on a separate thread
        assert customHandler.previous_models == 4

        print("Passed ! Test router_fallbacks: test_sync_fallbacks()")
        router.reset()
    except Exception as e:
        print(e)


# test_sync_fallbacks()


@pytest.mark.asyncio
async def test_async_fallbacks():
    litellm.set_verbose = True
    model_list = [
        {  # list of model deployments
            "model_name": "azure/gpt-3.5-turbo",  # openai model name
            "litellm_params": {  # params for litellm completion/embedding call
                "model": "azure/gpt-4.1-mini",
                "api_key": "bad-key",
                "api_version": os.getenv("AZURE_API_VERSION"),
                "api_base": os.getenv("AZURE_API_BASE"),
            },
            "tpm": 240000,
            "rpm": 1800,
        },
        {  # list of model deployments
            "model_name": "azure/gpt-3.5-turbo-context-fallback",  # openai model name
            "litellm_params": {  # params for litellm completion/embedding call
                "model": "azure/gpt-4.1-mini",
                "api_key": os.getenv("AZURE_API_KEY"),
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
                "api_key": "bad-key",
                "api_version": os.getenv("AZURE_API_VERSION"),
                "api_base": os.getenv("AZURE_API_BASE"),
            },
            "tpm": 240000,
            "rpm": 1800,
        },
        {
            "model_name": "gpt-3.5-turbo",  # openai model name
            "litellm_params": {  # params for litellm completion/embedding call
                "model": "gpt-3.5-turbo",
                "api_key": os.getenv("OPENAI_API_KEY"),
            },
            "tpm": 1000000,
            "rpm": 9000,
        },
        {
            "model_name": "gpt-3.5-turbo-16k",  # openai model name
            "litellm_params": {  # params for litellm completion/embedding call
                "model": "gpt-3.5-turbo-16k",
                "api_key": os.getenv("OPENAI_API_KEY"),
            },
            "tpm": 1000000,
            "rpm": 9000,
        },
    ]

    router = Router(
        model_list=model_list,
        fallbacks=[{"azure/gpt-3.5-turbo": ["gpt-3.5-turbo"]}],
        context_window_fallbacks=[
            {"azure/gpt-3.5-turbo-context-fallback": ["gpt-3.5-turbo-16k"]},
            {"gpt-3.5-turbo": ["gpt-3.5-turbo-16k"]},
        ],
        set_verbose=False,
    )
    customHandler = MyCustomHandler()
    litellm.callbacks = [customHandler]
    user_message = "Hello, how are you?"
    messages = [{"content": user_message, "role": "user"}]
    try:
        kwargs["model"] = "azure/gpt-3.5-turbo"
        response = await router.acompletion(**kwargs)
        print(f"customHandler.previous_models: {customHandler.previous_models}")
        await asyncio.sleep(
            0.05
        )  # allow a delay as success_callbacks are on a separate thread
        assert customHandler.previous_models == 4  # 1 init call, 2 retries, 1 fallback
        router.reset()
    except litellm.Timeout as e:
        pass
    except Exception as e:
        pytest.fail(f"An exception occurred: {e}")
    finally:
        router.reset()


# test_async_fallbacks()


def test_sync_fallbacks_embeddings():
    litellm.set_verbose = False
    model_list = [
        {  # list of model deployments
            "model_name": "bad-azure-embedding-model",  # openai model name
            "litellm_params": {  # params for litellm completion/embedding call
                "model": "azure/text-embedding-ada-002",
                "api_key": "bad-key",
                "api_version": os.getenv("AZURE_API_VERSION"),
                "api_base": os.getenv("AZURE_API_BASE"),
            },
            "tpm": 240000,
            "rpm": 1800,
        },
        {  # list of model deployments
            "model_name": "good-azure-embedding-model",  # openai model name
            "litellm_params": {  # params for litellm completion/embedding call
                "model": "text-embedding-ada-002",
                "api_key": os.getenv("OPENAI_API_KEY"),
            },
            "tpm": 240000,
            "rpm": 1800,
        },
    ]

    router = Router(
        model_list=model_list,
        fallbacks=[{"bad-azure-embedding-model": ["good-azure-embedding-model"]}],
        set_verbose=False,
    )
    customHandler = MyCustomHandler()
    litellm.callbacks = [customHandler]
    user_message = "Hello, how are you?"
    input = [user_message]
    try:
        kwargs = {"model": "bad-azure-embedding-model", "input": input}
        response = router.embedding(**kwargs)
        print(f"customHandler.previous_models: {customHandler.previous_models}")
        time.sleep(0.05)  # allow a delay as success_callbacks are on a separate thread
        assert customHandler.previous_models == 1  # 1 init call, 2 retries, 1 fallback
        router.reset()
    except litellm.Timeout as e:
        pass
    except Exception as e:
        pytest.fail(f"An exception occurred: {e}")
    finally:
        router.reset()


@pytest.mark.asyncio
async def test_async_fallbacks_embeddings():
    litellm.set_verbose = False
    model_list = [
        {  # list of model deployments
            "model_name": "bad-azure-embedding-model",  # openai model name
            "litellm_params": {  # params for litellm completion/embedding call
                "model": "azure/text-embedding-ada-002",
                "api_key": "bad-key",
                "api_version": os.getenv("AZURE_API_VERSION"),
                "api_base": os.getenv("AZURE_API_BASE"),
            },
            "tpm": 240000,
            "rpm": 1800,
        },
        {  # list of model deployments
            "model_name": "good-azure-embedding-model",  # openai model name
            "litellm_params": {  # params for litellm completion/embedding call
                "model": "text-embedding-ada-002",
                "api_key": os.getenv("OPENAI_API_KEY"),
            },
            "tpm": 240000,
            "rpm": 1800,
        },
    ]

    router = Router(
        model_list=model_list,
        fallbacks=[{"bad-azure-embedding-model": ["good-azure-embedding-model"]}],
        set_verbose=False,
    )
    customHandler = MyCustomHandler()
    litellm.callbacks = [customHandler]
    user_message = "Hello, how are you?"
    input = [user_message]
    try:
        kwargs = {"model": "bad-azure-embedding-model", "input": input}
        response = await router.aembedding(**kwargs)
        print(f"customHandler.previous_models: {customHandler.previous_models}")
        await asyncio.sleep(
            0.05
        )  # allow a delay as success_callbacks are on a separate thread
        assert customHandler.previous_models == 1  # 1 init call with a bad key
        router.reset()
    except litellm.Timeout as e:
        pass
    except Exception as e:
        pytest.fail(f"An exception occurred: {e}")
    finally:
        router.reset()


def test_dynamic_fallbacks_sync():
    """
    Allow setting the fallback in the router.completion() call.
    """
    try:
        customHandler = MyCustomHandler()
        litellm.callbacks = [customHandler]
        model_list = [
            {  # list of model deployments
                "model_name": "azure/gpt-3.5-turbo",  # openai model name
                "litellm_params": {  # params for litellm completion/embedding call
                    "model": "azure/gpt-4.1-mini",
                    "api_key": "bad-key",
                    "api_version": os.getenv("AZURE_API_VERSION"),
                    "api_base": os.getenv("AZURE_API_BASE"),
                },
                "tpm": 240000,
                "rpm": 1800,
            },
            {  # list of model deployments
                "model_name": "azure/gpt-3.5-turbo-context-fallback",  # openai model name
                "litellm_params": {  # params for litellm completion/embedding call
                    "model": "azure/gpt-4.1-mini",
                    "api_key": os.getenv("AZURE_API_KEY"),
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
                    "api_key": "bad-key",
                    "api_version": os.getenv("AZURE_API_VERSION"),
                    "api_base": os.getenv("AZURE_API_BASE"),
                },
                "tpm": 240000,
                "rpm": 1800,
            },
            {
                "model_name": "gpt-3.5-turbo",  # openai model name
                "litellm_params": {  # params for litellm completion/embedding call
                    "model": "gpt-3.5-turbo",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                },
                "tpm": 1000000,
                "rpm": 9000,
            },
            {
                "model_name": "gpt-3.5-turbo-16k",  # openai model name
                "litellm_params": {  # params for litellm completion/embedding call
                    "model": "gpt-3.5-turbo-16k",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                },
                "tpm": 1000000,
                "rpm": 9000,
            },
        ]

        router = Router(model_list=model_list, set_verbose=True)
        kwargs = {}
        kwargs["model"] = "azure/gpt-3.5-turbo"
        kwargs["messages"] = [{"role": "user", "content": "Hey, how's it going?"}]
        kwargs["fallbacks"] = [{"azure/gpt-3.5-turbo": ["gpt-3.5-turbo"]}]
        response = router.completion(**kwargs)
        print(f"response: {response}")
        time.sleep(0.05)  # allow a delay as success_callbacks are on a separate thread
        assert customHandler.previous_models == 4  # 1 init call, 2 retries, 1 fallback
        router.reset()
    except Exception as e:
        pytest.fail(f"An exception occurred - {e}")


# test_dynamic_fallbacks_sync()


@pytest.mark.asyncio
async def test_dynamic_fallbacks_async():
    """
    Allow setting the fallback in the router.completion() call.
    """
    try:
        model_list = [
            {  # list of model deployments
                "model_name": "azure/gpt-3.5-turbo",  # openai model name
                "litellm_params": {  # params for litellm completion/embedding call
                    "model": "azure/gpt-4.1-mini",
                    "api_key": "bad-key",
                    "api_version": os.getenv("AZURE_API_VERSION"),
                    "api_base": os.getenv("AZURE_API_BASE"),
                },
                "tpm": 240000,
                "rpm": 1800,
            },
            {  # list of model deployments
                "model_name": "azure/gpt-3.5-turbo-context-fallback",  # openai model name
                "litellm_params": {  # params for litellm completion/embedding call
                    "model": "azure/gpt-4.1-mini",
                    "api_key": os.getenv("AZURE_API_KEY"),
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
                    "api_key": "bad-key",
                    "api_version": os.getenv("AZURE_API_VERSION"),
                    "api_base": os.getenv("AZURE_API_BASE"),
                },
                "tpm": 240000,
                "rpm": 1800,
            },
            {
                "model_name": "gpt-3.5-turbo",  # openai model name
                "litellm_params": {  # params for litellm completion/embedding call
                    "model": "gpt-3.5-turbo",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                },
                "tpm": 1000000,
                "rpm": 9000,
            },
            {
                "model_name": "gpt-3.5-turbo-16k",  # openai model name
                "litellm_params": {  # params for litellm completion/embedding call
                    "model": "gpt-3.5-turbo-16k",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                },
                "tpm": 1000000,
                "rpm": 9000,
            },
        ]

        print()
        print()
        print()
        print()
        print(f"STARTING DYNAMIC ASYNC")
        customHandler = MyCustomHandler()
        litellm.callbacks = [customHandler]
        router = Router(model_list=model_list, set_verbose=True)
        kwargs = {}
        kwargs["model"] = "azure/gpt-3.5-turbo"
        kwargs["messages"] = [{"role": "user", "content": "Hey, how's it going?"}]
        kwargs["fallbacks"] = [{"azure/gpt-3.5-turbo": ["gpt-3.5-turbo"]}]
        response = await router.acompletion(**kwargs)
        print(f"RESPONSE: {response}")
        await asyncio.sleep(
            0.05
        )  # allow a delay as success_callbacks are on a separate thread
        assert customHandler.previous_models == 4  # 1 init call, 2 retries, 1 fallback
        router.reset()
    except Exception as e:
        pytest.fail(f"An exception occurred - {e}")


# asyncio.run(test_dynamic_fallbacks_async())


@pytest.mark.asyncio
async def test_async_fallbacks_streaming():
    litellm.set_verbose = False
    model_list = [
        {  # list of model deployments
            "model_name": "azure/gpt-3.5-turbo",  # openai model name
            "litellm_params": {  # params for litellm completion/embedding call
                "model": "azure/gpt-4.1-mini",
                "api_key": "bad-key",
                "api_version": os.getenv("AZURE_API_VERSION"),
                "api_base": os.getenv("AZURE_API_BASE"),
            },
            "tpm": 240000,
            "rpm": 1800,
        },
        {  # list of model deployments
            "model_name": "azure/gpt-3.5-turbo-context-fallback",  # openai model name
            "litellm_params": {  # params for litellm completion/embedding call
                "model": "azure/gpt-4.1-mini",
                "api_key": os.getenv("AZURE_API_KEY"),
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
                "api_key": "bad-key",
                "api_version": os.getenv("AZURE_API_VERSION"),
                "api_base": os.getenv("AZURE_API_BASE"),
            },
            "tpm": 240000,
            "rpm": 1800,
        },
        {
            "model_name": "gpt-3.5-turbo",  # openai model name
            "litellm_params": {  # params for litellm completion/embedding call
                "model": "gpt-3.5-turbo",
                "api_key": os.getenv("OPENAI_API_KEY"),
            },
            "tpm": 1000000,
            "rpm": 9000,
        },
        {
            "model_name": "gpt-3.5-turbo-16k",  # openai model name
            "litellm_params": {  # params for litellm completion/embedding call
                "model": "gpt-3.5-turbo-16k",
                "api_key": os.getenv("OPENAI_API_KEY"),
            },
            "tpm": 1000000,
            "rpm": 9000,
        },
    ]

    router = Router(
        model_list=model_list,
        fallbacks=[{"azure/gpt-3.5-turbo": ["gpt-3.5-turbo"]}],
        context_window_fallbacks=[
            {"azure/gpt-3.5-turbo-context-fallback": ["gpt-3.5-turbo-16k"]},
            {"gpt-3.5-turbo": ["gpt-3.5-turbo-16k"]},
        ],
        set_verbose=False,
    )
    customHandler = MyCustomHandler()
    litellm.callbacks = [customHandler]
    user_message = "Hello, how are you?"
    messages = [{"content": user_message, "role": "user"}]
    try:
        response = await router.acompletion(**kwargs, stream=True)
        print(f"customHandler.previous_models: {customHandler.previous_models}")
        await asyncio.sleep(
            0.05
        )  # allow a delay as success_callbacks are on a separate thread
        assert customHandler.previous_models == 4  # 1 init call, 2 retries, 1 fallback
        router.reset()
    except litellm.Timeout as e:
        pass
    except Exception as e:
        pytest.fail(f"An exception occurred: {e}")
    finally:
        router.reset()


def test_sync_fallbacks_streaming():
    try:
        model_list = [
            {  # list of model deployments
                "model_name": "azure/gpt-3.5-turbo",  # openai model name
                "litellm_params": {  # params for litellm completion/embedding call
                    "model": "azure/gpt-4.1-mini",
                    "api_key": "bad-key",
                    "api_version": os.getenv("AZURE_API_VERSION"),
                    "api_base": os.getenv("AZURE_API_BASE"),
                },
                "tpm": 240000,
                "rpm": 1800,
            },
            {  # list of model deployments
                "model_name": "azure/gpt-3.5-turbo-context-fallback",  # openai model name
                "litellm_params": {  # params for litellm completion/embedding call
                    "model": "azure/gpt-4.1-mini",
                    "api_key": os.getenv("AZURE_API_KEY"),
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
                    "api_key": "bad-key",
                    "api_version": os.getenv("AZURE_API_VERSION"),
                    "api_base": os.getenv("AZURE_API_BASE"),
                },
                "tpm": 240000,
                "rpm": 1800,
            },
            {
                "model_name": "gpt-3.5-turbo",  # openai model name
                "litellm_params": {  # params for litellm completion/embedding call
                    "model": "gpt-3.5-turbo",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                },
                "tpm": 1000000,
                "rpm": 9000,
            },
            {
                "model_name": "gpt-3.5-turbo-16k",  # openai model name
                "litellm_params": {  # params for litellm completion/embedding call
                    "model": "gpt-3.5-turbo-16k",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                },
                "tpm": 1000000,
                "rpm": 9000,
            },
        ]

        litellm.set_verbose = True
        customHandler = MyCustomHandler()
        litellm.callbacks = [customHandler]
        router = Router(
            model_list=model_list,
            fallbacks=[{"azure/gpt-3.5-turbo": ["gpt-3.5-turbo"]}],
            context_window_fallbacks=[
                {"azure/gpt-3.5-turbo-context-fallback": ["gpt-3.5-turbo-16k"]},
                {"gpt-3.5-turbo": ["gpt-3.5-turbo-16k"]},
            ],
            set_verbose=False,
        )
        response = router.completion(**kwargs, stream=True)
        print(f"response: {response}")
        time.sleep(0.05)  # allow a delay as success_callbacks are on a separate thread
        assert customHandler.previous_models == 1  # 0 retries, 1 fallback

        print("Passed ! Test router_fallbacks: test_sync_fallbacks()")
        router.reset()
    except Exception as e:
        print(e)


@pytest.mark.asyncio
async def test_async_fallbacks_max_retries_per_request():
    litellm.set_verbose = False
    litellm.num_retries_per_request = 0
    model_list = [
        {  # list of model deployments
            "model_name": "azure/gpt-3.5-turbo",  # openai model name
            "litellm_params": {  # params for litellm completion/embedding call
                "model": "azure/gpt-4.1-mini",
                "api_key": "bad-key",
                "api_version": os.getenv("AZURE_API_VERSION"),
                "api_base": os.getenv("AZURE_API_BASE"),
            },
            "tpm": 240000,
            "rpm": 1800,
        },
        {  # list of model deployments
            "model_name": "azure/gpt-3.5-turbo-context-fallback",  # openai model name
            "litellm_params": {  # params for litellm completion/embedding call
                "model": "azure/gpt-4.1-mini",
                "api_key": os.getenv("AZURE_API_KEY"),
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
                "api_key": "bad-key",
                "api_version": os.getenv("AZURE_API_VERSION"),
                "api_base": os.getenv("AZURE_API_BASE"),
            },
            "tpm": 240000,
            "rpm": 1800,
        },
        {
            "model_name": "gpt-3.5-turbo",  # openai model name
            "litellm_params": {  # params for litellm completion/embedding call
                "model": "gpt-3.5-turbo",
                "api_key": os.getenv("OPENAI_API_KEY"),
            },
            "tpm": 1000000,
            "rpm": 9000,
        },
        {
            "model_name": "gpt-3.5-turbo-16k",  # openai model name
            "litellm_params": {  # params for litellm completion/embedding call
                "model": "gpt-3.5-turbo-16k",
                "api_key": os.getenv("OPENAI_API_KEY"),
            },
            "tpm": 1000000,
            "rpm": 9000,
        },
    ]

    router = Router(
        model_list=model_list,
        fallbacks=[{"azure/gpt-3.5-turbo": ["gpt-3.5-turbo"]}],
        context_window_fallbacks=[
            {"azure/gpt-3.5-turbo-context-fallback": ["gpt-3.5-turbo-16k"]},
            {"gpt-3.5-turbo": ["gpt-3.5-turbo-16k"]},
        ],
        set_verbose=False,
    )
    customHandler = MyCustomHandler()
    litellm.callbacks = [customHandler]
    user_message = "Hello, how are you?"
    messages = [{"content": user_message, "role": "user"}]
    try:
        try:
            response = await router.acompletion(**kwargs, stream=True)
        except Exception:
            pass
        print(f"customHandler.previous_models: {customHandler.previous_models}")
        await asyncio.sleep(
            0.05
        )  # allow a delay as success_callbacks are on a separate thread
        assert customHandler.previous_models == 0  # 0 retries, 0 fallback
        router.reset()
    except litellm.Timeout as e:
        pass
    except Exception as e:
        pytest.fail(f"An exception occurred: {e}")
    finally:
        router.reset()


@pytest.mark.flaky(retries=6, delay=2)
def test_ausage_based_routing_fallbacks():
    try:
        import litellm

        litellm.set_verbose = False
        # [Prod Test]
        # IT tests Usage Based Routing with fallbacks
        # The Request should fail azure/gpt-4-fast. Then fallback -> "azure/gpt-4-basic" -> "openai-gpt-4"
        # It should work with "openai-gpt-4"
        import os

        from dotenv import load_dotenv

        import litellm
        from litellm import Router

        load_dotenv()

        # Constants for TPM and RPM allocation
        AZURE_FAST_RPM = 1
        AZURE_BASIC_RPM = 1
        OPENAI_RPM = 0
        ANTHROPIC_RPM = 10

        def get_azure_params(deployment_name: str):
            params = {
                "model": f"azure/{deployment_name}",
                "api_key": os.environ["AZURE_API_KEY"],
                "api_version": os.environ["AZURE_API_VERSION"],
                "api_base": os.environ["AZURE_API_BASE"],
            }
            return params

        def get_openai_params(model: str):
            params = {
                "model": model,
                "api_key": os.environ["OPENAI_API_KEY"],
            }
            return params

        def get_anthropic_params(model: str):
            params = {
                "model": model,
                "api_key": os.environ["ANTHROPIC_API_KEY"],
            }
            return params

        model_list = [
            {
                "model_name": "azure/gpt-4-fast",
                "litellm_params": get_azure_params("chatgpt-v-3"),
                "model_info": {"id": 1},
                "rpm": AZURE_FAST_RPM,
            },
            {
                "model_name": "azure/gpt-4-basic",
                "litellm_params": get_azure_params("chatgpt-v-3"),
                "model_info": {"id": 2},
                "rpm": AZURE_BASIC_RPM,
            },
            {
                "model_name": "openai-gpt-4",
                "litellm_params": get_openai_params("gpt-3.5-turbo"),
                "model_info": {"id": 3},
                "rpm": OPENAI_RPM,
            },
            {
                "model_name": "anthropic-claude-3-5-haiku-20241022",
                "litellm_params": get_anthropic_params("claude-3-5-haiku-20241022"),
                "model_info": {"id": 4},
                "rpm": ANTHROPIC_RPM,
            },
        ]
        # litellm.set_verbose=True
        fallbacks_list = [
            {"azure/gpt-4-fast": ["azure/gpt-4-basic"]},
            {"azure/gpt-4-basic": ["openai-gpt-4"]},
            {"openai-gpt-4": ["anthropic-claude-3-5-haiku-20241022"]},
        ]

        router = Router(
            model_list=model_list,
            fallbacks=fallbacks_list,
            set_verbose=True,
            debug_level="DEBUG",
            routing_strategy="usage-based-routing-v2",
            redis_host=os.environ["REDIS_HOST"],
            redis_port=int(os.environ["REDIS_PORT"]),
            num_retries=0,
        )

        messages = [
            {"content": "Tell me a joke.", "role": "user"},
        ]
        response = router.completion(
            model="azure/gpt-4-fast",
            messages=messages,
            timeout=5,
            mock_response="very nice to meet you",
        )
        print("response: ", response)
        print(f"response._hidden_params: {response._hidden_params}")
        # in this test, we expect azure/gpt-4 fast to fail, then azure-gpt-4 basic to fail and then openai-gpt-4 to pass
        # the token count of this message is > AZURE_FAST_TPM, > AZURE_BASIC_TPM
        assert response._hidden_params["model_id"] == "1"

        for i in range(10):
            # now make 100 mock requests to OpenAI - expect it to fallback to anthropic-claude-3-5-haiku-20241022
            response = router.completion(
                model="azure/gpt-4-fast",
                messages=messages,
                timeout=5,
                mock_response="very nice to meet you",
            )
            print("response: ", response)
            print("response._hidden_params: ", response._hidden_params)
            if i == 9:
                assert response._hidden_params["model_id"] == "4"

    except Exception as e:
        pytest.fail(f"An exception occurred {e}")


def test_custom_cooldown_times():
    try:
        # set, custom_cooldown. Failed model in cooldown_models, after custom_cooldown, the failed model is no longer in cooldown_models

        model_list = [
            {  # list of model deployments
                "model_name": "gpt-3.5-turbo",  # openai model name
                "litellm_params": {  # params for litellm completion/embedding call
                    "model": "azure/gpt-4.1-mini",
                    "api_key": "bad-key",
                    "api_version": os.getenv("AZURE_API_VERSION"),
                    "api_base": os.getenv("AZURE_API_BASE"),
                },
                "tpm": 24000000,
            },
            {  # list of model deployments
                "model_name": "gpt-3.5-turbo",  # openai model name
                "litellm_params": {  # params for litellm completion/embedding call
                    "model": "azure/gpt-4.1-mini",
                    "api_key": os.getenv("AZURE_API_KEY"),
                    "api_version": os.getenv("AZURE_API_VERSION"),
                    "api_base": os.getenv("AZURE_API_BASE"),
                },
                "tpm": 1,
            },
        ]

        litellm.set_verbose = False

        router = Router(
            model_list=model_list,
            set_verbose=True,
            debug_level="INFO",
            cooldown_time=0.1,
            redis_host=os.getenv("REDIS_HOST"),
            redis_password=os.getenv("REDIS_PASSWORD"),
            redis_port=int(os.getenv("REDIS_PORT")),
        )

        # make a request - expect it to fail
        try:
            response = router.completion(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "content": "Tell me a joke.",
                        "role": "user",
                    }
                ],
            )
        except Exception:
            pass

        # expect 1 model to be in cooldown models
        cooldown_deployments = router._get_cooldown_deployments()
        print("cooldown_deployments after failed call: ", cooldown_deployments)
        assert (
            len(cooldown_deployments) == 1
        ), "Expected 1 model to be in cooldown models"

        selected_cooldown_model = cooldown_deployments[0]

        # wait for 1/2 of cooldown time
        time.sleep(router.cooldown_time / 2)

        # expect cooldown model to still be in cooldown models
        cooldown_deployments = router._get_cooldown_deployments()
        print(
            "cooldown_deployments after waiting 1/2 of cooldown: ", cooldown_deployments
        )
        assert (
            len(cooldown_deployments) == 1
        ), "Expected 1 model to be in cooldown models"

        # wait for 1/2 of cooldown time again, now we've waited for full cooldown
        time.sleep(router.cooldown_time / 2)

        # expect cooldown model to be removed from cooldown models
        cooldown_deployments = router._get_cooldown_deployments()
        print(
            "cooldown_deployments after waiting cooldown time: ", cooldown_deployments
        )
        assert (
            len(cooldown_deployments) == 0
        ), "Expected 0 models to be in cooldown models"

    except Exception as e:
        print(e)


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_service_unavailable_fallbacks(sync_mode):
    """
    Initial model - openai
    Fallback - azure

    Error - 503, service unavailable
    """
    router = Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo-012",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": "anything",
                    "api_base": "http://0.0.0.0:8080",
                },
            },
            {
                "model_name": "gpt-3.5-turbo-0125-preview",
                "litellm_params": {
                    "model": "gpt-4.1-nano",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                },
            },
        ],
        fallbacks=[{"gpt-3.5-turbo-012": ["gpt-3.5-turbo-0125-preview"]}],
    )

    if sync_mode:
        response = router.completion(
            model="gpt-3.5-turbo-012",
            messages=[{"role": "user", "content": "Hey, how's it going?"}],
        )
    else:
        response = await router.acompletion(
            model="gpt-3.5-turbo-012",
            messages=[{"role": "user", "content": "Hey, how's it going?"}],
        )

    assert "gpt-4.1-nano" in response.model


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.parametrize("litellm_module_fallbacks", [True, False])
@pytest.mark.asyncio
async def test_default_model_fallbacks(sync_mode, litellm_module_fallbacks):
    """
    Related issue - https://github.com/BerriAI/litellm/issues/3623

    If model misconfigured, setup a default model for generic fallback
    """
    if litellm_module_fallbacks:
        litellm.default_fallbacks = ["my-good-model"]
    router = Router(
        model_list=[
            {
                "model_name": "bad-model",
                "litellm_params": {
                    "model": "openai/my-bad-model",
                    "api_key": "my-bad-api-key",
                },
            },
            {
                "model_name": "my-good-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                },
            },
        ],
        default_fallbacks=(
            ["my-good-model"] if litellm_module_fallbacks is False else None
        ),
    )

    if sync_mode:
        response = router.completion(
            model="bad-model",
            messages=[{"role": "user", "content": "Hey, how's it going?"}],
            mock_testing_fallbacks=True,
            mock_response="Hey! nice day",
        )
    else:
        response = await router.acompletion(
            model="bad-model",
            messages=[{"role": "user", "content": "Hey, how's it going?"}],
            mock_testing_fallbacks=True,
            mock_response="Hey! nice day",
        )

    assert isinstance(response, litellm.ModelResponse)
    assert response.model is not None and response.model == "gpt-4o"


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_client_side_fallbacks_list(sync_mode):
    """

    Tests Client Side Fallbacks

    User can pass "fallbacks": ["gpt-3.5-turbo"] and this should work

    """
    router = Router(
        model_list=[
            {
                "model_name": "bad-model",
                "litellm_params": {
                    "model": "openai/my-bad-model",
                    "api_key": "my-bad-api-key",
                },
            },
            {
                "model_name": "my-good-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                },
            },
        ],
    )

    if sync_mode:
        response = router.completion(
            model="bad-model",
            messages=[{"role": "user", "content": "Hey, how's it going?"}],
            fallbacks=["my-good-model"],
            mock_testing_fallbacks=True,
            mock_response="Hey! nice day",
        )
    else:
        response = await router.acompletion(
            model="bad-model",
            messages=[{"role": "user", "content": "Hey, how's it going?"}],
            fallbacks=["my-good-model"],
            mock_testing_fallbacks=True,
            mock_response="Hey! nice day",
        )

    assert isinstance(response, litellm.ModelResponse)
    assert response.model is not None and response.model == "gpt-4o"


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.parametrize("content_filter_response_exception", [True, False])
@pytest.mark.parametrize("fallback_type", ["model-specific", "default"])
@pytest.mark.asyncio
async def test_router_content_policy_fallbacks(
    sync_mode, content_filter_response_exception, fallback_type
):
    os.environ["LITELLM_LOG"] = "DEBUG"

    if content_filter_response_exception:
        mock_response = Exception("content filtering policy")
    else:
        mock_response = litellm.ModelResponse(
            choices=[litellm.Choices(finish_reason="content_filter")],
            model="gpt-3.5-turbo",
            usage=litellm.Usage(prompt_tokens=10, completion_tokens=0, total_tokens=10),
        )
    router = Router(
        model_list=[
            {
                "model_name": "claude-sonnet-4-5-20250929",
                "litellm_params": {
                    "model": "anthropic/claude-sonnet-4-5-20250929",
                    "api_key": "",
                    "mock_response": mock_response,
                },
            },
            {
                "model_name": "my-fallback-model",
                "litellm_params": {
                    "model": "openai/my-fake-model",
                    "api_key": "",
                    "mock_response": "This works!",
                },
            },
            {
                "model_name": "my-default-fallback-model",
                "litellm_params": {
                    "model": "openai/my-fake-model",
                    "api_key": "",
                    "mock_response": "This works 2!",
                },
            },
            {
                "model_name": "my-general-model",
                "litellm_params": {
                    "model": "anthropic/claude-sonnet-4-5-20250929",
                    "api_key": "",
                    "mock_response": Exception("Should not have called this."),
                },
            },
            {
                "model_name": "my-context-window-model",
                "litellm_params": {
                    "model": "anthropic/claude-sonnet-4-5-20250929",
                    "api_key": "",
                    "mock_response": Exception("Should not have called this."),
                },
            },
        ],
        content_policy_fallbacks=(
            [{"claude-sonnet-4-5-20250929": ["my-fallback-model"]}]
            if fallback_type == "model-specific"
            else None
        ),
        default_fallbacks=(
            ["my-default-fallback-model"] if fallback_type == "default" else None
        ),
    )

    if sync_mode is True:
        response = router.completion(
            model="claude-sonnet-4-5-20250929",
            messages=[{"role": "user", "content": "Hey, how's it going?"}],
        )
    else:
        response = await router.acompletion(
            model="claude-sonnet-4-5-20250929",
            messages=[{"role": "user", "content": "Hey, how's it going?"}],
        )

    assert response.model == "my-fake-model"


@pytest.mark.parametrize("sync_mode", [False, True])
@pytest.mark.asyncio
async def test_using_default_fallback(sync_mode):
    litellm.set_verbose = True

    import logging

    from litellm._logging import verbose_logger, verbose_router_logger

    verbose_logger.setLevel(logging.DEBUG)
    verbose_router_logger.setLevel(logging.DEBUG)
    litellm.default_fallbacks = ["very-bad-model"]
    router = Router(
        model_list=[
            {
                "model_name": "openai/*",
                "litellm_params": {
                    "model": "openai/*",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                },
            },
        ],
    )
    try:
        if sync_mode:
            response = router.completion(
                model="openai/foo",
                messages=[{"role": "user", "content": "Hey, how's it going?"}],
            )
        else:
            response = await router.acompletion(
                model="openai/foo",
                messages=[{"role": "user", "content": "Hey, how's it going?"}],
            )
        print("got response=", response)
        pytest.fail(f"Expected call to fail we passed model=openai/foo")
    except Exception as e:
        print("got exception = ", e)
        assert "BadRequestError" in str(e)


@pytest.mark.parametrize("sync_mode", [False])
@pytest.mark.asyncio
async def test_using_default_working_fallback(sync_mode):
    litellm.set_verbose = True

    import logging

    from litellm._logging import verbose_logger, verbose_router_logger

    verbose_logger.setLevel(logging.DEBUG)
    verbose_router_logger.setLevel(logging.DEBUG)
    litellm.default_fallbacks = ["openai/gpt-3.5-turbo"]
    router = Router(
        model_list=[
            {
                "model_name": "openai/*",
                "litellm_params": {
                    "model": "openai/*",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                },
            },
        ],
    )

    if sync_mode:
        response = router.completion(
            model="openai/foo",
            messages=[{"role": "user", "content": "Hey, how's it going?"}],
        )
    else:
        response = await router.acompletion(
            model="openai/foo",
            messages=[{"role": "user", "content": "Hey, how's it going?"}],
        )
    print("got response=", response)
    assert response is not None


# asyncio.run(test_acompletion_gemini_stream())
def mock_post_streaming(url, **kwargs):
    mock_response = MagicMock()
    mock_response.status_code = 529
    mock_response.headers = {"Content-Type": "application/json"}
    mock_response.return_value = {"detail": "Overloaded!"}

    return mock_response


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_anthropic_streaming_fallbacks(sync_mode):
    litellm.set_verbose = True
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler

    if sync_mode:
        client = HTTPHandler(concurrent_limit=1)
    else:
        client = AsyncHTTPHandler(concurrent_limit=1)

    router = Router(
        model_list=[
            {
                "model_name": "anthropic/claude-sonnet-4-5-20250929",
                "litellm_params": {
                    "model": "anthropic/claude-sonnet-4-5-20250929",
                },
            },
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "mock_response": "Hey, how's it going?",
                },
            },
        ],
        fallbacks=[{"anthropic/claude-sonnet-4-5-20250929": ["gpt-3.5-turbo"]}],
        num_retries=0,
    )

    with patch.object(client, "post", side_effect=mock_post_streaming) as mock_client:
        chunks = []
        if sync_mode:
            response = router.completion(
                model="anthropic/claude-sonnet-4-5-20250929",
                messages=[{"role": "user", "content": "Hey, how's it going?"}],
                stream=True,
                client=client,
            )
            for chunk in response:
                print(chunk)
                chunks.append(chunk)
        else:
            response = await router.acompletion(
                model="anthropic/claude-sonnet-4-5-20250929",
                messages=[{"role": "user", "content": "Hey, how's it going?"}],
                stream=True,
                client=client,
            )
            async for chunk in response:
                print(chunk)
                chunks.append(chunk)
        print(f"RETURNED response: {response}")

        mock_client.assert_called_once()
        print(chunks)
        assert len(chunks) > 0


def test_router_fallbacks_with_custom_model_costs():
    """
    Tests prod use-case where a custom model is registered with a different provider + custom costs.

    Goal: make sure custom model doesn't override default model costs.
    """

    model_list = [
        {
            "model_name": "claude-sonnet-4-5-20250929",
            "litellm_params": {
                "model": "claude-sonnet-4-5-20250929",
                "api_key": os.environ.get("ANTHROPIC_API_KEY", "fake-key"),
                "input_cost_per_token": 30,
                "output_cost_per_token": 60,
                "mock_response": "Hello! How can I help you today?",
            },
        },
        {
            "model_name": "claude-3-5-sonnet-aihubmix",
            "litellm_params": {
                "model": "openai/claude-sonnet-4-5-20250929",
                "input_cost_per_token": 0.000003,  # 3$/M
                "output_cost_per_token": 0.000015,  # 15$/M
                "api_base": "https://exampleopenaiendpoint-production.up.railway.app",
                "api_key": "my-fake-key",
                "mock_response": "Hello! How can I help you today?",
            },
        },
    ]

    router = Router(
        model_list=model_list,
        fallbacks=[{"claude-sonnet-4-5-20250929": ["claude-3-5-sonnet-aihubmix"]}],
    )

    router.completion(
        model="claude-3-5-sonnet-aihubmix",
        messages=[{"role": "user", "content": "Hey, how's it going?"}],
    )

    model_info = litellm.get_model_info(model="claude-sonnet-4-5-20250929")

    print(f"key: {model_info['key']}")

    assert model_info["litellm_provider"] == "anthropic"

    response = router.completion(
        model="claude-sonnet-4-5-20250929",
        messages=[{"role": "user", "content": "Hey, how's it going?"}],
    )

    print(f"response_cost: {response._hidden_params['response_cost']}")

    assert response._hidden_params["response_cost"] > 10

    model_info = litellm.get_model_info(model="claude-sonnet-4-5-20250929")

    print(f"key: {model_info['key']}")

    assert model_info["input_cost_per_token"] == 30
    assert model_info["output_cost_per_token"] == 60


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_router_fallbacks_default_and_model_specific_fallbacks(sync_mode):
    """
    Tests to ensure there is not an infinite fallback loop when there is a default fallback and model specific fallback.
    """
    router = Router(
        model_list=[
            {
                "model_name": "bad-model",
                "litellm_params": {
                    "model": "openai/my-bad-model",
                    "api_key": "my-bad-api-key",
                },
            },
            {
                "model_name": "my-bad-model-2",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "bad-key",
                },
            },
        ],
        fallbacks=[{"bad-model": ["my-bad-model-2"]}],
        default_fallbacks=["bad-model"],
    )

    with pytest.raises(Exception) as exc_info:
        if sync_mode:
            resp = router.completion(
                model="bad-model",
                messages=[{"role": "user", "content": "Hey, how's it going?"}],
            )

            print(f"resp: {resp}")
        else:
            await router.acompletion(
                model="bad-model",
                messages=[{"role": "user", "content": "Hey, how's it going?"}],
            )
    assert isinstance(
        exc_info.value, litellm.AuthenticationError
    ), f"Expected AuthenticationError, but got {type(exc_info.value).__name__}"


@pytest.mark.asyncio
async def test_router_disable_fallbacks_dynamically():
    from litellm.router import run_async_fallback

    router = Router(
        model_list=[
            {
                "model_name": "bad-model",
                "litellm_params": {
                    "model": "openai/my-bad-model",
                    "api_key": "my-bad-api-key",
                },
            },
            {
                "model_name": "good-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                },
            },
        ],
        fallbacks=[{"bad-model": ["good-model"]}],
        default_fallbacks=["good-model"],
    )

    with patch.object(
        router,
        "log_retry",
        new=MagicMock(return_value=None),
    ) as mock_client:
        try:
            resp = await router.acompletion(
                model="bad-model",
                messages=[{"role": "user", "content": "Hey, how's it going?"}],
                disable_fallbacks=True,
            )
            print(resp)
        except Exception as e:
            print(e)

        mock_client.assert_not_called()


def test_router_fallbacks_with_model_id():
    router = Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {"model": "gpt-3.5-turbo", "rpm": 1},
                "model_info": {
                    "id": "123",
                },
            }
        ],
        routing_strategy="usage-based-routing-v2",
        fallbacks=[{"gpt-3.5-turbo": ["123"]}],
    )

    ## test model id fallback works
    router.completion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "hi"}],
        mock_testing_fallbacks=True,
    )


def test_router_fallbacks_with_wildcard_model_name():
    router = Router(
        model_list=[
            {
                "model_name": "openai/*",
                "litellm_params": {
                    "model": "openai/*",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                },
            },
            {
                "model_name": "claude-3-haiku",
                "litellm_params": {
                    "model": "claude-3-haiku-20240307",
                    "api_key": os.getenv("ANTHROPIC_API_KEY"),
                    "mock_response": "Hi this is claude!",
                },
            },
        ],
        fallbacks=[{"gpt-3.5-turbo": ["claude-3-haiku"]}],
    )

    response = router.completion(
        model="openai/gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hey, how's it going?"}],
        mock_testing_fallbacks=True,
    )

    print(response)
    assert response["choices"][0]["message"]["content"] == "Hi this is claude!"


def test_get_fallback_model_group():
    from litellm.router_utils.fallback_event_handlers import get_fallback_model_group

    args = {
        "fallbacks": [
            {"gpt-3.5-turbo": ["claude-3-haiku"]},
            {"*": ["claude-3-sonnet"]},
        ],
        "model_group": "openai/gpt-3.5-turbo",
    }
    fallback_model_group, _ = get_fallback_model_group(**args)
    assert fallback_model_group == ["claude-3-haiku"]


def test_fallbacks_with_different_messages():
    router = Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                },
            },
            {
                "model_name": "claude-3-haiku",
                "litellm_params": {
                    "model": "claude-3-haiku-20240307",
                    "api_key": os.getenv("ANTHROPIC_API_KEY"),
                },
            },
        ],
    )

    resp = router.completion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hey, how's it going?"}],
        mock_testing_fallbacks=True,
        fallbacks=[
            {
                "model": "claude-3-haiku",
                "messages": [{"role": "user", "content": "Hey, how's it going?"}],
            }
        ],
    )

    print(resp)


@pytest.mark.parametrize("expected_attempted_fallbacks", [0, 1, 3])
@pytest.mark.asyncio
async def test_router_attempted_fallbacks_in_response(expected_attempted_fallbacks):
    """
    Test that the router returns the correct number of attempted fallbacks in the response

    - Test cases: works on first try, `x-litellm-attempted-fallbacks` is 0
    - Works on 1st fallback, `x-litellm-attempted-fallbacks` is 1
    - Works on 3rd fallback, `x-litellm-attempted-fallbacks` is 3
    """
    router = Router(
        model_list=[
            {
                "model_name": "working-fake-endpoint",
                "litellm_params": {
                    "model": "openai/working-fake-endpoint",
                    "api_key": "my-fake-key",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app",
                },
            },
            {
                "model_name": "badly-configured-openai-endpoint",
                "litellm_params": {
                    "model": "openai/my-fake-model",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.appzzzzz",
                },
            },
        ],
        fallbacks=[{"badly-configured-openai-endpoint": ["working-fake-endpoint"]}],
    )

    if expected_attempted_fallbacks == 0:
        resp = router.completion(
            model="working-fake-endpoint",
            messages=[{"role": "user", "content": "Hey, how's it going?"}],
        )
        assert (
            resp._hidden_params["additional_headers"]["x-litellm-attempted-fallbacks"]
            == expected_attempted_fallbacks
        )
    elif expected_attempted_fallbacks == 1:
        resp = router.completion(
            model="badly-configured-openai-endpoint",
            messages=[{"role": "user", "content": "Hey, how's it going?"}],
        )
        assert (
            resp._hidden_params["additional_headers"]["x-litellm-attempted-fallbacks"]
            == expected_attempted_fallbacks
        )
