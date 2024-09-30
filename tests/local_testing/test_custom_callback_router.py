### What this tests ####
## This test asserts the type of data passed into each method of the custom callback handler
import asyncio
import inspect
import os
import sys
import time
import traceback
from datetime import datetime

import pytest

sys.path.insert(0, os.path.abspath("../.."))
from typing import List, Literal, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import litellm
from litellm import Cache, Router
from litellm.integrations.custom_logger import CustomLogger

# Test Scenarios (test across completion, streaming, embedding)
## 1: Pre-API-Call
## 2: Post-API-Call
## 3: On LiteLLM Call success
## 4: On LiteLLM Call failure
## fallbacks
## retries

# Test cases
## 1. Simple Azure OpenAI acompletion + streaming call
## 2. Simple Azure OpenAI aembedding call
## 3. Azure OpenAI acompletion + streaming  call with retries
## 4. Azure OpenAI aembedding call with retries
## 5. Azure OpenAI acompletion + streaming call with fallbacks
## 6. Azure OpenAI aembedding call with fallbacks

# Test interfaces
## 1. router.completion() + router.embeddings()
## 2. proxy.completions + proxy.embeddings

litellm.num_retries = 0


class CompletionCustomHandler(
    CustomLogger
):  # https://docs.litellm.ai/docs/observability/custom_callback#callback-class
    """
    The set of expected inputs to a custom handler for a
    """

    # Class variables or attributes
    def __init__(self):
        self.errors = []
        self.states: Optional[
            List[
                Literal[
                    "sync_pre_api_call",
                    "async_pre_api_call",
                    "post_api_call",
                    "sync_stream",
                    "async_stream",
                    "sync_success",
                    "async_success",
                    "sync_failure",
                    "async_failure",
                ]
            ]
        ] = []

    def log_pre_api_call(self, model, messages, kwargs):
        try:
            print(f"received kwargs in pre-input: {kwargs}")
            self.states.append("sync_pre_api_call")
            ## MODEL
            assert isinstance(model, str)
            ## MESSAGES
            assert isinstance(messages, list)
            ## KWARGS
            assert isinstance(kwargs["model"], str)
            assert isinstance(kwargs["messages"], list)
            assert isinstance(kwargs["optional_params"], dict)
            assert isinstance(kwargs["litellm_params"], dict)
            assert isinstance(kwargs["start_time"], (datetime, type(None)))
            assert isinstance(kwargs["stream"], bool)
            assert isinstance(kwargs["user"], (str, type(None)))
            ### ROUTER-SPECIFIC KWARGS
            assert isinstance(kwargs["litellm_params"]["metadata"], dict)
            assert isinstance(kwargs["litellm_params"]["metadata"]["model_group"], str)
            assert isinstance(kwargs["litellm_params"]["metadata"]["deployment"], str)
            assert isinstance(kwargs["litellm_params"]["model_info"], dict)
            assert isinstance(kwargs["litellm_params"]["model_info"]["id"], str)
            assert isinstance(
                kwargs["litellm_params"]["proxy_server_request"], (str, type(None))
            )
            assert isinstance(
                kwargs["litellm_params"]["preset_cache_key"], (str, type(None))
            )
            assert isinstance(kwargs["litellm_params"]["stream_response"], dict)
        except Exception as e:
            print(f"Assertion Error: {traceback.format_exc()}")
            self.errors.append(traceback.format_exc())

    def log_post_api_call(self, kwargs, response_obj, start_time, end_time):
        try:
            self.states.append("post_api_call")
            ## START TIME
            assert isinstance(start_time, datetime)
            ## END TIME
            assert end_time == None
            ## RESPONSE OBJECT
            assert response_obj == None
            ## KWARGS
            assert isinstance(kwargs["model"], str)
            assert isinstance(kwargs["messages"], list)
            assert isinstance(kwargs["optional_params"], dict)
            assert isinstance(kwargs["litellm_params"], dict)
            assert isinstance(kwargs["start_time"], (datetime, type(None)))
            assert isinstance(kwargs["stream"], bool)
            assert isinstance(kwargs["user"], (str, type(None)))
            assert isinstance(kwargs["input"], (list, dict, str))
            assert isinstance(kwargs["api_key"], (str, type(None)))
            assert (
                isinstance(
                    kwargs["original_response"], (str, litellm.CustomStreamWrapper)
                )
                or inspect.iscoroutine(kwargs["original_response"])
                or inspect.isasyncgen(kwargs["original_response"])
            )
            assert isinstance(kwargs["additional_args"], (dict, type(None)))
            assert isinstance(kwargs["log_event_type"], str)
            ### ROUTER-SPECIFIC KWARGS
            assert isinstance(kwargs["litellm_params"]["metadata"], dict)
            assert isinstance(kwargs["litellm_params"]["metadata"]["model_group"], str)
            assert isinstance(kwargs["litellm_params"]["metadata"]["deployment"], str)
            assert isinstance(kwargs["litellm_params"]["model_info"], dict)
            assert isinstance(kwargs["litellm_params"]["model_info"]["id"], str)
            assert isinstance(
                kwargs["litellm_params"]["proxy_server_request"], (str, type(None))
            )
            assert isinstance(
                kwargs["litellm_params"]["preset_cache_key"], (str, type(None))
            )
            assert isinstance(kwargs["litellm_params"]["stream_response"], dict)
        except:
            print(f"Assertion Error: {traceback.format_exc()}")
            self.errors.append(traceback.format_exc())

    async def async_log_stream_event(self, kwargs, response_obj, start_time, end_time):
        try:
            self.states.append("async_stream")
            ## START TIME
            assert isinstance(start_time, datetime)
            ## END TIME
            assert isinstance(end_time, datetime)
            ## RESPONSE OBJECT
            assert isinstance(response_obj, litellm.ModelResponse)
            ## KWARGS
            assert isinstance(kwargs["model"], str)
            assert isinstance(kwargs["messages"], list) and isinstance(
                kwargs["messages"][0], dict
            )
            assert isinstance(kwargs["optional_params"], dict)
            assert isinstance(kwargs["litellm_params"], dict)
            assert isinstance(kwargs["start_time"], (datetime, type(None)))
            assert isinstance(kwargs["stream"], bool)
            assert isinstance(kwargs["user"], (str, type(None)))
            assert (
                isinstance(kwargs["input"], list)
                and isinstance(kwargs["input"][0], dict)
            ) or isinstance(kwargs["input"], (dict, str))
            assert isinstance(kwargs["api_key"], (str, type(None)))
            assert (
                isinstance(
                    kwargs["original_response"], (str, litellm.CustomStreamWrapper)
                )
                or inspect.isasyncgen(kwargs["original_response"])
                or inspect.iscoroutine(kwargs["original_response"])
            )
            assert isinstance(kwargs["additional_args"], (dict, type(None)))
            assert isinstance(kwargs["log_event_type"], str)
        except:
            print(f"Assertion Error: {traceback.format_exc()}")
            self.errors.append(traceback.format_exc())

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        try:
            self.states.append("sync_success")
            ## START TIME
            assert isinstance(start_time, datetime)
            ## END TIME
            assert isinstance(end_time, datetime)
            ## RESPONSE OBJECT
            assert isinstance(response_obj, litellm.ModelResponse)
            ## KWARGS
            assert isinstance(kwargs["model"], str)
            assert isinstance(kwargs["messages"], list) and isinstance(
                kwargs["messages"][0], dict
            )
            assert isinstance(kwargs["optional_params"], dict)
            assert isinstance(kwargs["litellm_params"], dict)
            assert isinstance(kwargs["start_time"], (datetime, type(None)))
            assert isinstance(kwargs["stream"], bool)
            assert isinstance(kwargs["user"], (str, type(None)))
            assert (
                isinstance(kwargs["input"], list)
                and isinstance(kwargs["input"][0], dict)
            ) or isinstance(kwargs["input"], (dict, str))
            assert isinstance(kwargs["api_key"], (str, type(None)))
            assert isinstance(
                kwargs["original_response"], (str, litellm.CustomStreamWrapper)
            )
            assert isinstance(kwargs["additional_args"], (dict, type(None)))
            assert isinstance(kwargs["log_event_type"], str)
            assert kwargs["cache_hit"] is None or isinstance(kwargs["cache_hit"], bool)
        except:
            print(f"Assertion Error: {traceback.format_exc()}")
            self.errors.append(traceback.format_exc())

    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        try:
            self.states.append("sync_failure")
            ## START TIME
            assert isinstance(start_time, datetime)
            ## END TIME
            assert isinstance(end_time, datetime)
            ## RESPONSE OBJECT
            assert response_obj == None
            ## KWARGS
            assert isinstance(kwargs["model"], str)
            assert isinstance(kwargs["messages"], list) and isinstance(
                kwargs["messages"][0], dict
            )
            assert isinstance(kwargs["optional_params"], dict)
            assert isinstance(kwargs["litellm_params"], dict)
            assert isinstance(kwargs["start_time"], (datetime, type(None)))
            assert isinstance(kwargs["stream"], bool)
            assert isinstance(kwargs["user"], (str, type(None)))
            assert (
                isinstance(kwargs["input"], list)
                and isinstance(kwargs["input"][0], dict)
            ) or isinstance(kwargs["input"], (dict, str))
            assert isinstance(kwargs["api_key"], (str, type(None)))
            assert (
                isinstance(
                    kwargs["original_response"], (str, litellm.CustomStreamWrapper)
                )
                or kwargs["original_response"] == None
            )
            assert isinstance(kwargs["additional_args"], (dict, type(None)))
            assert isinstance(kwargs["log_event_type"], str)
        except:
            print(f"Assertion Error: {traceback.format_exc()}")
            self.errors.append(traceback.format_exc())

    async def async_log_pre_api_call(self, model, messages, kwargs):
        try:
            """
            No-op.
            Not implemented yet.
            """
            pass
        except Exception as e:
            print(f"Assertion Error: {traceback.format_exc()}")
            self.errors.append(traceback.format_exc())

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        try:
            self.states.append("async_success")
            print("in async success, kwargs: ", kwargs)
            ## START TIME
            assert isinstance(start_time, datetime)
            ## END TIME
            assert isinstance(end_time, datetime)
            ## RESPONSE OBJECT
            assert isinstance(
                response_obj, (litellm.ModelResponse, litellm.EmbeddingResponse)
            )
            ## KWARGS
            assert isinstance(kwargs["model"], str)

            # checking we use base_model for azure cost calculation
            base_model = litellm.utils._get_base_model_from_metadata(
                model_call_details=kwargs
            )

            if (
                kwargs["model"] == "chatgpt-v-2"
                and base_model is not None
                and kwargs["stream"] != True
            ):
                # when base_model is set for azure, we should use pricing for the base_model
                # this checks response_cost == litellm.cost_per_token(model=base_model)
                assert isinstance(kwargs["response_cost"], float)
                response_cost = kwargs["response_cost"]
                print(
                    f"response_cost: {response_cost}, for model: {kwargs['model']} and base_model: {base_model}"
                )
                prompt_tokens = response_obj.usage.prompt_tokens
                completion_tokens = response_obj.usage.completion_tokens
                # ensure the pricing is based on the base_model here
                prompt_price, completion_price = litellm.cost_per_token(
                    model=base_model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )
                expected_price = prompt_price + completion_price
                print(f"expected price: {expected_price}")
                assert (
                    response_cost == expected_price
                ), f"response_cost: {response_cost} != expected_price: {expected_price}. For model: {kwargs['model']} and base_model: {base_model}. should have used base_model for price"

            assert isinstance(kwargs["messages"], list)
            assert isinstance(kwargs["optional_params"], dict)
            assert isinstance(kwargs["litellm_params"], dict)
            assert isinstance(kwargs["start_time"], (datetime, type(None)))
            assert isinstance(kwargs["stream"], bool)
            assert isinstance(kwargs["user"], (str, type(None)))
            assert isinstance(kwargs["input"], (list, dict, str))
            assert isinstance(kwargs["api_key"], (str, type(None)))
            assert (
                isinstance(
                    kwargs["original_response"], (str, litellm.CustomStreamWrapper)
                )
                or inspect.isasyncgen(kwargs["original_response"])
                or inspect.iscoroutine(kwargs["original_response"])
            )
            assert isinstance(kwargs["additional_args"], (dict, type(None)))
            assert isinstance(kwargs["log_event_type"], str)
            assert kwargs["cache_hit"] is None or isinstance(kwargs["cache_hit"], bool)
            ### ROUTER-SPECIFIC KWARGS
            assert isinstance(kwargs["litellm_params"]["metadata"], dict)
            assert isinstance(kwargs["litellm_params"]["metadata"]["model_group"], str)
            assert isinstance(kwargs["litellm_params"]["metadata"]["deployment"], str)
            assert isinstance(kwargs["litellm_params"]["model_info"], dict)
            assert isinstance(kwargs["litellm_params"]["model_info"]["id"], str)
            assert isinstance(
                kwargs["litellm_params"]["proxy_server_request"], (str, type(None))
            )
            assert isinstance(
                kwargs["litellm_params"]["preset_cache_key"], (str, type(None))
            )
            assert isinstance(kwargs["litellm_params"]["stream_response"], dict)
        except:
            print(f"Assertion Error: {traceback.format_exc()}")
            self.errors.append(traceback.format_exc())

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        try:
            print(f"received original response: {kwargs['original_response']}")
            self.states.append("async_failure")
            ## START TIME
            assert isinstance(start_time, datetime)
            ## END TIME
            assert isinstance(end_time, datetime)
            ## RESPONSE OBJECT
            assert response_obj == None
            ## KWARGS
            assert isinstance(kwargs["model"], str)
            assert isinstance(kwargs["messages"], list)
            assert isinstance(kwargs["optional_params"], dict)
            assert isinstance(kwargs["litellm_params"], dict)
            assert isinstance(kwargs["start_time"], (datetime, type(None)))
            assert isinstance(kwargs["stream"], bool)
            assert isinstance(kwargs["user"], (str, type(None)))
            assert isinstance(kwargs["input"], (list, str, dict))
            assert isinstance(kwargs["api_key"], (str, type(None)))
            assert (
                isinstance(
                    kwargs["original_response"], (str, litellm.CustomStreamWrapper)
                )
                or inspect.isasyncgen(kwargs["original_response"])
                or inspect.iscoroutine(kwargs["original_response"])
                or kwargs["original_response"] == None
            )
            assert isinstance(kwargs["additional_args"], (dict, type(None)))
            assert isinstance(kwargs["log_event_type"], str)
        except:
            print(f"Assertion Error: {traceback.format_exc()}")
            self.errors.append(traceback.format_exc())


# Simple Azure OpenAI call
## COMPLETION
@pytest.mark.flaky(retries=5, delay=1)
@pytest.mark.asyncio
async def test_async_chat_azure():
    try:
        customHandler_completion_azure_router = CompletionCustomHandler()
        customHandler_streaming_azure_router = CompletionCustomHandler()
        customHandler_failure = CompletionCustomHandler()
        litellm.callbacks = [customHandler_completion_azure_router]
        litellm.set_verbose = True
        model_list = [
            {
                "model_name": "gpt-3.5-turbo",  # openai model name
                "litellm_params": {  # params for litellm completion/embedding call
                    "model": "azure/chatgpt-v-2",
                    "api_key": os.getenv("AZURE_API_KEY"),
                    "api_version": os.getenv("AZURE_API_VERSION"),
                    "api_base": os.getenv("AZURE_API_BASE"),
                },
                "model_info": {"base_model": "azure/gpt-4-1106-preview"},
                "tpm": 240000,
                "rpm": 1800,
            },
        ]
        router = Router(model_list=model_list, num_retries=0)  # type: ignore
        response = await router.acompletion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hi 👋 - i'm openai"}],
        )
        await asyncio.sleep(2)
        assert len(customHandler_completion_azure_router.errors) == 0
        assert (
            len(customHandler_completion_azure_router.states) == 3
        )  # pre, post, success
        # streaming
        litellm.callbacks = [customHandler_streaming_azure_router]
        router2 = Router(model_list=model_list, num_retries=0)  # type: ignore
        response = await router2.acompletion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hi 👋 - i'm openai"}],
            stream=True,
        )
        async for chunk in response:
            print(f"async azure router chunk: {chunk}")
            continue
        await asyncio.sleep(1)
        print(f"customHandler.states: {customHandler_streaming_azure_router.states}")
        assert len(customHandler_streaming_azure_router.errors) == 0
        assert (
            len(customHandler_streaming_azure_router.states) >= 4
        )  # pre, post, stream (multiple times), success
        # failure
        model_list = [
            {
                "model_name": "gpt-3.5-turbo",  # openai model name
                "litellm_params": {  # params for litellm completion/embedding call
                    "model": "azure/chatgpt-v-2",
                    "api_key": "my-bad-key",
                    "api_version": os.getenv("AZURE_API_VERSION"),
                    "api_base": os.getenv("AZURE_API_BASE"),
                },
                "tpm": 240000,
                "rpm": 1800,
            },
        ]
        litellm.callbacks = [customHandler_failure]
        router3 = Router(model_list=model_list, num_retries=0)  # type: ignore
        try:
            response = await router3.acompletion(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hi 👋 - i'm openai"}],
            )
            print(f"response in router3 acompletion: {response}")
        except:
            pass
        await asyncio.sleep(1)
        print(f"customHandler.states: {customHandler_failure.states}")
        assert len(customHandler_failure.errors) == 0
        assert len(customHandler_failure.states) == 3  # pre, post, failure
        assert "async_failure" in customHandler_failure.states
    except Exception as e:
        print(f"Assertion Error: {traceback.format_exc()}")
        pytest.fail(f"An exception occurred - {str(e)}")


# asyncio.run(test_async_chat_azure())
## EMBEDDING
@pytest.mark.asyncio
async def test_async_embedding_azure():
    try:
        customHandler = CompletionCustomHandler()
        customHandler_failure = CompletionCustomHandler()
        litellm.callbacks = [customHandler]
        model_list = [
            {
                "model_name": "azure-embedding-model",  # openai model name
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
        router = Router(model_list=model_list)  # type: ignore
        response = await router.aembedding(
            model="azure-embedding-model", input=["hello from litellm!"]
        )
        await asyncio.sleep(2)
        assert len(customHandler.errors) == 0
        assert len(customHandler.states) == 3  # pre, post, success
        # failure
        model_list = [
            {
                "model_name": "azure-embedding-model",  # openai model name
                "litellm_params": {  # params for litellm completion/embedding call
                    "model": "azure/azure-embedding-model",
                    "api_key": "my-bad-key",
                    "api_version": os.getenv("AZURE_API_VERSION"),
                    "api_base": os.getenv("AZURE_API_BASE"),
                },
                "tpm": 240000,
                "rpm": 1800,
            },
        ]
        litellm.callbacks = [customHandler_failure]
        router3 = Router(model_list=model_list, num_retries=0)  # type: ignore
        try:
            response = await router3.aembedding(
                model="azure-embedding-model", input=["hello from litellm!"]
            )
            print(f"response in router3 aembedding: {response}")
        except:
            pass
        await asyncio.sleep(1)
        print(f"customHandler.states: {customHandler_failure.states}")
        assert len(customHandler_failure.errors) == 0
        assert len(customHandler_failure.states) == 3  # pre, post, failure
        assert "async_failure" in customHandler_failure.states
    except Exception as e:
        print(f"Assertion Error: {traceback.format_exc()}")
        pytest.fail(f"An exception occurred - {str(e)}")


# asyncio.run(test_async_embedding_azure())
# Azure OpenAI call w/ Fallbacks
## COMPLETION
@pytest.mark.asyncio
async def test_async_chat_azure_with_fallbacks():
    try:
        customHandler_fallbacks = CompletionCustomHandler()
        litellm.callbacks = [customHandler_fallbacks]
        litellm.set_verbose = True
        # with fallbacks
        model_list = [
            {
                "model_name": "gpt-3.5-turbo",  # openai model name
                "litellm_params": {  # params for litellm completion/embedding call
                    "model": "azure/chatgpt-v-2",
                    "api_key": "my-bad-key",
                    "api_version": os.getenv("AZURE_API_VERSION"),
                    "api_base": os.getenv("AZURE_API_BASE"),
                },
                "tpm": 240000,
                "rpm": 1800,
            },
            {
                "model_name": "gpt-3.5-turbo-16k",
                "litellm_params": {
                    "model": "gpt-3.5-turbo-16k",
                },
                "tpm": 240000,
                "rpm": 1800,
            },
        ]
        router = Router(
            model_list=model_list,
            fallbacks=[{"gpt-3.5-turbo": ["gpt-3.5-turbo-16k"]}],
            retry_policy=litellm.router.RetryPolicy(
                AuthenticationErrorRetries=0,
            ),
        )  # type: ignore
        response = await router.acompletion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hi 👋 - i'm openai"}],
        )
        await asyncio.sleep(2)
        print(f"customHandler_fallbacks.states: {customHandler_fallbacks.states}")
        assert len(customHandler_fallbacks.errors) == 0
        assert (
            len(customHandler_fallbacks.states) == 6
        )  # pre, post, failure, pre, post, success
        litellm.callbacks = []
    except Exception as e:
        print(f"Assertion Error: {traceback.format_exc()}")
        pytest.fail(f"An exception occurred - {str(e)}")


# asyncio.run(test_async_chat_azure_with_fallbacks())


# CACHING
## Test Azure - completion, embedding
@pytest.mark.asyncio
@pytest.mark.flaky(retries=3, delay=1)
async def test_async_completion_azure_caching():
    customHandler_caching = CompletionCustomHandler()
    litellm.cache = Cache(
        type="redis",
        host=os.environ["REDIS_HOST"],
        port=os.environ["REDIS_PORT"],
        password=os.environ["REDIS_PASSWORD"],
    )
    litellm.callbacks = [customHandler_caching]
    unique_time = time.time()
    model_list = [
        {
            "model_name": "gpt-3.5-turbo",  # openai model name
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
            "model_name": "gpt-3.5-turbo-16k",
            "litellm_params": {
                "model": "gpt-3.5-turbo-16k",
            },
            "tpm": 240000,
            "rpm": 1800,
        },
    ]
    router = Router(model_list=model_list)  # type: ignore
    response1 = await router.acompletion(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": f"Hi 👋 - i'm async azure {unique_time}"}
        ],
        caching=True,
    )
    await asyncio.sleep(1)
    print(f"customHandler_caching.states pre-cache hit: {customHandler_caching.states}")
    response2 = await router.acompletion(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": f"Hi 👋 - i'm async azure {unique_time}"}
        ],
        caching=True,
    )
    await asyncio.sleep(1)  # success callbacks are done in parallel
    print(
        f"customHandler_caching.states post-cache hit: {customHandler_caching.states}"
    )
    assert len(customHandler_caching.errors) == 0
    assert len(customHandler_caching.states) == 4  # pre, post, success, success


@pytest.mark.asyncio
async def test_rate_limit_error_callback():
    """
    Assert a callback is hit, if a model group starts hitting rate limit errors

    Relevant issue: https://github.com/BerriAI/litellm/issues/4096
    """
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLogging

    customHandler = CompletionCustomHandler()
    litellm.callbacks = [customHandler]
    litellm.success_callback = []

    router = Router(
        model_list=[
            {
                "model_name": "my-test-gpt",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "mock_response": "litellm.RateLimitError",
                },
            }
        ],
        allowed_fails=2,
        num_retries=0,
    )

    litellm_logging_obj = LiteLLMLogging(
        model="my-test-gpt",
        messages=[{"role": "user", "content": "hi"}],
        stream=False,
        call_type="acompletion",
        litellm_call_id="1234",
        start_time=datetime.now(),
        function_id="1234",
    )

    try:
        _ = await router.acompletion(
            model="my-test-gpt",
            messages=[{"role": "user", "content": "Hey, how's it going?"}],
        )
    except Exception:
        pass

    with patch.object(
        customHandler, "log_model_group_rate_limit_error", new=AsyncMock()
    ) as mock_client:

        print(
            f"customHandler.log_model_group_rate_limit_error: {customHandler.log_model_group_rate_limit_error}"
        )

        try:
            _ = await router.acompletion(
                model="my-test-gpt",
                messages=[{"role": "user", "content": "Hey, how's it going?"}],
                litellm_logging_obj=litellm_logging_obj,
            )
        except (litellm.RateLimitError, ValueError):
            pass

        await asyncio.sleep(3)
        mock_client.assert_called_once()

        assert "original_model_group" in mock_client.call_args.kwargs
        assert mock_client.call_args.kwargs["original_model_group"] == "my-test-gpt"
