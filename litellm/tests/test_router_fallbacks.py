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
            f"previous_models: {kwargs['litellm_params']['metadata']['previous_models']}"
        )
        self.previous_models += len(
            kwargs["litellm_params"]["metadata"]["previous_models"]
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
                    "model": "azure/chatgpt-v-2",
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
                    "model": "azure/chatgpt-v-2",
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
        assert customHandler.previous_models == 1  # 0 retries, 1 fallback

        print("Passed ! Test router_fallbacks: test_sync_fallbacks()")
        router.reset()
    except Exception as e:
        print(e)


# test_sync_fallbacks()


@pytest.mark.asyncio
async def test_async_fallbacks():
    litellm.set_verbose = False
    model_list = [
        {  # list of model deployments
            "model_name": "azure/gpt-3.5-turbo",  # openai model name
            "litellm_params": {  # params for litellm completion/embedding call
                "model": "azure/chatgpt-v-2",
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
                "model": "azure/chatgpt-v-2",
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
        response = await router.acompletion(**kwargs)
        print(f"customHandler.previous_models: {customHandler.previous_models}")
        await asyncio.sleep(
            0.05
        )  # allow a delay as success_callbacks are on a separate thread
        assert customHandler.previous_models == 1  # 0 retries, 1 fallback
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
                "model": "azure/azure-embedding-model",
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
                "model": "azure/azure-embedding-model",
                "api_key": os.getenv("AZURE_API_KEY"),
                "api_version": os.getenv("AZURE_API_VERSION"),
                "api_base": os.getenv("AZURE_API_BASE"),
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
        assert customHandler.previous_models == 1  # 0 retries, 1 fallback
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
                "model": "azure/azure-embedding-model",
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
                "model": "azure/azure-embedding-model",
                "api_key": os.getenv("AZURE_API_KEY"),
                "api_version": os.getenv("AZURE_API_VERSION"),
                "api_base": os.getenv("AZURE_API_BASE"),
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
        assert customHandler.previous_models == 1  # 0 retries, 1 fallback
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
                    "model": "azure/chatgpt-v-2",
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
                    "model": "azure/chatgpt-v-2",
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
        assert customHandler.previous_models == 1  # 0 retries, 1 fallback
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
                    "model": "azure/chatgpt-v-2",
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
                    "model": "azure/chatgpt-v-2",
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
        assert customHandler.previous_models == 1  # 0 retries, 1 fallback
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
                "model": "azure/chatgpt-v-2",
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
                "model": "azure/chatgpt-v-2",
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
        assert customHandler.previous_models == 1  # 0 retries, 1 fallback
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
                    "model": "azure/chatgpt-v-2",
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
                    "model": "azure/chatgpt-v-2",
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
                "model": "azure/chatgpt-v-2",
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
                "model": "azure/chatgpt-v-2",
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
        except:
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


def test_usage_based_routing_fallbacks():
    try:
        # [Prod Test]
        # IT tests Usage Based Routing with fallbacks
        # The Request should fail azure/gpt-4-fast. Then fallback -> "azure/gpt-4-basic" -> "openai-gpt-4"
        # It should work with "openai-gpt-4"
        import os
        import litellm
        from litellm import Router
        from dotenv import load_dotenv

        load_dotenv()

        # Constants for TPM and RPM allocation
        AZURE_FAST_TPM = 3
        AZURE_BASIC_TPM = 4
        OPENAI_TPM = 400
        ANTHROPIC_TPM = 100000

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
                "litellm_params": get_azure_params("chatgpt-v-2"),
                "tpm": AZURE_FAST_TPM,
            },
            {
                "model_name": "azure/gpt-4-basic",
                "litellm_params": get_azure_params("chatgpt-v-2"),
                "tpm": AZURE_BASIC_TPM,
            },
            {
                "model_name": "openai-gpt-4",
                "litellm_params": get_openai_params("gpt-3.5-turbo"),
                "tpm": OPENAI_TPM,
            },
            {
                "model_name": "anthropic-claude-instant-1.2",
                "litellm_params": get_anthropic_params("claude-instant-1.2"),
                "tpm": ANTHROPIC_TPM,
            },
        ]
        # litellm.set_verbose=True
        fallbacks_list = [
            {"azure/gpt-4-fast": ["azure/gpt-4-basic"]},
            {"azure/gpt-4-basic": ["openai-gpt-4"]},
            {"openai-gpt-4": ["anthropic-claude-instant-1.2"]},
        ]

        router = Router(
            model_list=model_list,
            fallbacks=fallbacks_list,
            set_verbose=True,
            debug_level="DEBUG",
            routing_strategy="usage-based-routing",
            redis_host=os.environ["REDIS_HOST"],
            redis_port=os.environ["REDIS_PORT"],
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
        print("response._hidden_params: ", response._hidden_params)
        # in this test, we expect azure/gpt-4 fast to fail, then azure-gpt-4 basic to fail and then openai-gpt-4 to pass
        # the token count of this message is > AZURE_FAST_TPM, > AZURE_BASIC_TPM
        assert response._hidden_params["custom_llm_provider"] == "openai"

        # now make 100 mock requests to OpenAI - expect it to fallback to anthropic-claude-instant-1.2
        for i in range(20):
            response = router.completion(
                model="azure/gpt-4-fast",
                messages=messages,
                timeout=5,
                mock_response="very nice to meet you",
            )
            print("response: ", response)
            print("response._hidden_params: ", response._hidden_params)
            if i == 19:
                # by the 19th call we should have hit TPM LIMIT for OpenAI, it should fallback to anthropic-claude-instant-1.2
                assert response._hidden_params["custom_llm_provider"] == "anthropic"

    except Exception as e:
        pytest.fail(f"An exception occurred {e}")


def test_custom_cooldown_times():
    try:
        # set, custom_cooldown. Failed model in cooldown_models, after custom_cooldown, the failed model is no longer in cooldown_models

        model_list = [
            {  # list of model deployments
                "model_name": "gpt-3.5-turbo",  # openai model name
                "litellm_params": {  # params for litellm completion/embedding call
                    "model": "azure/chatgpt-v-2",
                    "api_key": "bad-key",
                    "api_version": os.getenv("AZURE_API_VERSION"),
                    "api_base": os.getenv("AZURE_API_BASE"),
                },
                "tpm": 24000000,
            },
            {  # list of model deployments
                "model_name": "gpt-3.5-turbo",  # openai model name
                "litellm_params": {  # params for litellm completion/embedding call
                    "model": "azure/chatgpt-v-2",
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
        except:
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
