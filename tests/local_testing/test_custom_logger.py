### What this tests ####
import asyncio
import inspect
import os
import sys
import time
import traceback

import pytest

sys.path.insert(0, os.path.abspath("../.."))

import litellm
from litellm import completion, embedding
from litellm.integrations.custom_logger import CustomLogger


class MyCustomHandler(CustomLogger):
    complete_streaming_response_in_callback = ""

    def __init__(self):
        self.success: bool = False  # type: ignore
        self.failure: bool = False  # type: ignore
        self.async_success: bool = False  # type: ignore
        self.async_success_embedding: bool = False  # type: ignore
        self.async_failure: bool = False  # type: ignore
        self.async_failure_embedding: bool = False  # type: ignore

        self.async_completion_kwargs = None  # type: ignore
        self.async_embedding_kwargs = None  # type: ignore
        self.async_embedding_response = None  # type: ignore

        self.async_completion_kwargs_fail = None  # type: ignore
        self.async_embedding_kwargs_fail = None  # type: ignore

        self.stream_collected_response = None  # type: ignore
        self.sync_stream_collected_response = None  # type: ignore
        self.user = None  # type: ignore
        self.data_sent_to_api: dict = {}
        self.response_cost = 0

    def log_pre_api_call(self, model, messages, kwargs):
        print("Pre-API Call")
        self.data_sent_to_api = kwargs["additional_args"].get("complete_input_dict", {})

    def log_post_api_call(self, kwargs, response_obj, start_time, end_time):
        print("Post-API Call")

    def log_stream_event(self, kwargs, response_obj, start_time, end_time):
        print("On Stream")

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        print(f"On Success")
        self.success = True
        if kwargs.get("stream") == True:
            self.sync_stream_collected_response = response_obj
        print(f"response cost in log_success_event: {kwargs.get('response_cost')}")
        self.response_cost = kwargs.get("response_cost", 0)

    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        print(f"On Failure")
        self.failure = True

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        print(f"On Async success")
        print(f"received kwargs user: {kwargs['user']}")
        self.async_success = True
        if kwargs.get("model") == "text-embedding-ada-002":
            self.async_success_embedding = True
            self.async_embedding_kwargs = kwargs
            self.async_embedding_response = response_obj
        if kwargs.get("stream") == True:
            self.stream_collected_response = response_obj
        self.async_completion_kwargs = kwargs
        self.user = kwargs.get("user", None)
        print(
            f"response cost in log_async_success_event: {kwargs.get('response_cost')}"
        )
        self.response_cost = kwargs.get("response_cost", 0)

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        print(f"On Async Failure")
        self.async_failure = True
        if kwargs.get("model") == "text-embedding-ada-002":
            self.async_failure_embedding = True
            self.async_embedding_kwargs_fail = kwargs

        self.async_completion_kwargs_fail = kwargs


class TmpFunction:
    complete_streaming_response_in_callback = ""
    async_success: bool = False

    async def async_test_logging_fn(self, kwargs, completion_obj, start_time, end_time):
        print(f"ON ASYNC LOGGING")
        self.async_success = True
        print(
            f'kwargs.get("async_complete_streaming_response"): {kwargs.get("async_complete_streaming_response")}'
        )
        self.complete_streaming_response_in_callback = kwargs.get(
            "async_complete_streaming_response"
        )


@pytest.mark.asyncio
async def test_async_chat_openai_stream():
    try:
        tmp_function = TmpFunction()
        litellm.set_verbose = True
        litellm.success_callback = [tmp_function.async_test_logging_fn]
        complete_streaming_response = ""

        response = await litellm.acompletion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hi 👋 - i'm openai"}],
            stream=True,
        )
        async for chunk in response:
            complete_streaming_response += chunk["choices"][0]["delta"]["content"] or ""
            print(complete_streaming_response)

        complete_streaming_response = complete_streaming_response.strip("'")

        print(f"complete_streaming_response: {complete_streaming_response}")

        await asyncio.sleep(3)

        print(
            f"tmp_function.complete_streaming_response_in_callback: {tmp_function.complete_streaming_response_in_callback}"
        )
        # problematic line
        response1 = tmp_function.complete_streaming_response_in_callback["choices"][0][
            "message"
        ]["content"]
        response2 = complete_streaming_response
        # assert [ord(c) for c in response1] == [ord(c) for c in response2]
        print(f"response1: {response1}")
        print(f"response2: {response2}")
        assert response1 == response2
        assert tmp_function.async_success == True
    except Exception as e:
        print(e)
        pytest.fail(f"An error occurred - {str(e)}\n\n{traceback.format_exc()}")


# test_async_chat_openai_stream()


def test_completion_azure_stream_moderation_failure():
    try:
        customHandler = MyCustomHandler()
        litellm.callbacks = [customHandler]
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": "how do i kill someone",
            },
        ]
        try:
            response = completion(
                model="azure/gpt-4.1-nano",
                messages=messages,
                mock_response="Exception: content_filter_policy",
                stream=True,
            )
            for chunk in response:
                print(f"chunk: {chunk}")
                continue
        except Exception as e:
            print(e)
        time.sleep(1)
        assert customHandler.failure == True
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


def test_async_custom_handler_stream():
    try:
        # [PROD Test] - Do not DELETE
        # checks if the model response available in the async + stream callbacks is equal to the received response
        customHandler2 = MyCustomHandler()
        litellm.callbacks = [customHandler2]
        litellm.set_verbose = False
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": "write 1 sentence about litellm being amazing",
            },
        ]
        complete_streaming_response = ""

        async def test_1():
            nonlocal complete_streaming_response
            response = await litellm.acompletion(
                model="azure/gpt-4.1-nano", messages=messages, stream=True
            )
            async for chunk in response:
                complete_streaming_response += (
                    chunk["choices"][0]["delta"]["content"] or ""
                )
                print(complete_streaming_response)

        asyncio.run(test_1())

        response_in_success_handler = customHandler2.stream_collected_response
        response_in_success_handler = response_in_success_handler["choices"][0][
            "message"
        ]["content"]
        print("\n\n")
        print("response_in_success_handler: ", response_in_success_handler)
        print("complete_streaming_response: ", complete_streaming_response)
        assert response_in_success_handler == complete_streaming_response
    except Exception as e:
        pytest.fail(f"Error occurred: {e}\n{traceback.format_exc()}")


# test_async_custom_handler_stream()


@pytest.mark.skip(reason="Flaky test")
def test_azure_completion_stream():
    # [PROD Test] - Do not DELETE
    # test if completion() + sync custom logger get the same complete stream response
    try:
        # checks if the model response available in the async + stream callbacks is equal to the received response
        customHandler2 = MyCustomHandler()
        litellm.callbacks = [customHandler2]
        litellm.set_verbose = True
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": f"write 1 sentence about litellm being amazing {time.time()}",
            },
        ]
        complete_streaming_response = ""

        response = litellm.completion(
            model="azure/gpt-4.1-nano", messages=messages, stream=True
        )
        for chunk in response:
            complete_streaming_response += chunk["choices"][0]["delta"]["content"] or ""
            print(complete_streaming_response)

        time.sleep(0.5)  # wait 1/2 second before checking callbacks
        response_in_success_handler = customHandler2.sync_stream_collected_response
        response_in_success_handler = response_in_success_handler["choices"][0][
            "message"
        ]["content"]
        print("\n\n")
        print("response_in_success_handler: ", response_in_success_handler)
        print("complete_streaming_response: ", complete_streaming_response)
        assert response_in_success_handler == complete_streaming_response
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


@pytest.mark.asyncio
async def test_async_custom_handler_completion():
    try:
        litellm._turn_on_debug
        customHandler_success = MyCustomHandler()
        customHandler_failure = MyCustomHandler()
        # success
        assert customHandler_success.async_success == False
        litellm.callbacks = [customHandler_success]
        response = await litellm.acompletion(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "user",
                    "content": "hello from litellm test",
                }
            ],
        )
        await asyncio.sleep(1)
        assert (
            customHandler_success.async_success == True
        ), "async success is not set to True even after success"
        assert (
            customHandler_success.async_completion_kwargs.get("model")
            == "gpt-3.5-turbo"
        )
        # failure
        litellm.logging_callback_manager._reset_all_callbacks()
        litellm.callbacks = [customHandler_failure]
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": "how do i kill someone",
            },
        ]

        assert customHandler_failure.async_failure == False
        try:
            response = await litellm.acompletion(
                model="gpt-3.5-turbo",
                messages=messages,
                api_key="my-bad-key",
            )
        except Exception:
            pass
        assert (
            customHandler_failure.async_failure == True
        ), "async failure is not set to True even after failure"
        assert (
            customHandler_failure.async_completion_kwargs_fail.get("model")
            == "gpt-3.5-turbo"
        )
        assert (
            len(
                str(customHandler_failure.async_completion_kwargs_fail.get("exception"))
            )
            > 10
        )  # expect APIError("OpenAIException - Error code: 401 - {'error': {'message': 'Incorrect API key provided: test. You can find your API key at https://platform.openai.com/account/api-keys.', 'type': 'invalid_request_error', 'param': None, 'code': 'invalid_api_key'}}"), 'traceback_exception': 'Traceback (most recent call last):\n  File "/Users/ishaanjaffer/Github/litellm/litellm/llms/openai.py", line 269, in acompletion\n    response = await openai_aclient.chat.completions.create(**data)\n  File "/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages/openai/resources/chat/completions.py", line 119
        litellm.callbacks = []
        print("Passed setting async failure")
    except Exception as e:
        pytest.fail(f"An exception occurred - {str(e)}")


# asyncio.run(test_async_custom_handler_completion())


@pytest.mark.asyncio
async def test_async_custom_handler_embedding():
    try:
        customHandler_embedding = MyCustomHandler()
        litellm.callbacks = [customHandler_embedding]
        # success
        assert customHandler_embedding.async_success_embedding == False
        response = await litellm.aembedding(
            model="text-embedding-ada-002",
            input=["hello world"],
        )
        await asyncio.sleep(1)
        assert (
            customHandler_embedding.async_success_embedding == True
        ), "async_success_embedding is not set to True even after success"
        assert (
            customHandler_embedding.async_embedding_kwargs.get("model")
            == "text-embedding-ada-002"
        )
        assert (
            customHandler_embedding.async_embedding_response["usage"]["prompt_tokens"]
            == 2
        )
        print("Passed setting async success: Embedding")
        # failure
        assert customHandler_embedding.async_failure_embedding == False
        try:
            response = await litellm.aembedding(
                model="text-embedding-ada-002",
                input=["hello world"],
                api_key="my-bad-key",
            )
        except Exception:
            pass
        assert (
            customHandler_embedding.async_failure_embedding == True
        ), "async failure embedding is not set to True even after failure"
        assert (
            customHandler_embedding.async_embedding_kwargs_fail.get("model")
            == "text-embedding-ada-002"
        )
        assert (
            len(
                str(
                    customHandler_embedding.async_embedding_kwargs_fail.get("exception")
                )
            )
            > 10
        )  # exppect APIError("OpenAIException - Error code: 401 - {'error': {'message': 'Incorrect API key provided: test. You can find your API key at https://platform.openai.com/account/api-keys.', 'type': 'invalid_request_error', 'param': None, 'code': 'invalid_api_key'}}"), 'traceback_exception': 'Traceback (most recent call last):\n  File "/Users/ishaanjaffer/Github/litellm/litellm/llms/openai.py", line 269, in acompletion\n    response = await openai_aclient.chat.completions.create(**data)\n  File "/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages/openai/resources/chat/completions.py", line 119
    except Exception as e:
        pytest.fail(f"An exception occurred - {str(e)}")


# asyncio.run(test_async_custom_handler_embedding())


@pytest.mark.asyncio
async def test_async_custom_handler_embedding_optional_param():
    """
    Tests if the openai optional params for embedding - user + encoding_format,
    are logged
    """
    litellm.set_verbose = True
    customHandler_optional_params = MyCustomHandler()
    litellm.callbacks = [customHandler_optional_params]
    response = await litellm.aembedding(
        model="text-embedding-ada-002", input=["hello world"], user="John"
    )
    await asyncio.sleep(1)  # success callback is async
    assert customHandler_optional_params.user == "John"
    assert (
        customHandler_optional_params.user
        == customHandler_optional_params.data_sent_to_api["user"]
    )


# asyncio.run(test_async_custom_handler_embedding_optional_param())


@pytest.mark.skip(reason="AWS Account suspended. Pending their approval")
@pytest.mark.asyncio
async def test_async_custom_handler_embedding_optional_param_bedrock():
    """
    Tests if the openai optional params for embedding - user + encoding_format,
    are logged

    but makes sure these are not sent to the non-openai/azure endpoint (raises errors).
    """
    litellm.drop_params = True
    litellm.set_verbose = True
    customHandler_optional_params = MyCustomHandler()
    litellm.callbacks = [customHandler_optional_params]
    response = await litellm.aembedding(
        model="bedrock/amazon.titan-embed-text-v1", input=["hello world"], user="John"
    )
    await asyncio.sleep(1)  # success callback is async
    assert customHandler_optional_params.user == "John"
    assert "user" not in customHandler_optional_params.data_sent_to_api


@pytest.mark.asyncio
@pytest.mark.flaky(retries=3, delay=1)
async def test_cost_tracking_with_caching():
    """
    Important Test - This tests if that cost is 0 for cached responses
    """
    from litellm import Cache

    litellm.set_verbose = True
    litellm.cache = Cache(
        type="redis",
        host=os.environ["REDIS_HOST"],
        port=os.environ["REDIS_PORT"],
        password=os.environ["REDIS_PASSWORD"],
    )
    customHandler_optional_params = MyCustomHandler()
    litellm.callbacks = [customHandler_optional_params]
    messages = [
        {
            "role": "user",
            "content": f"write a one sentence poem about: {time.time()}",
        }
    ]
    response1 = await litellm.acompletion(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=40,
        temperature=0.2,
        caching=True,
        mock_response="Hey, i'm doing well!",
    )
    await asyncio.sleep(3)  # success callback is async
    response_cost = customHandler_optional_params.response_cost
    assert response_cost > 0
    response2 = await litellm.acompletion(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=40,
        temperature=0.2,
        caching=True,
    )
    await asyncio.sleep(1)  # success callback is async
    response_cost_2 = customHandler_optional_params.response_cost
    assert response_cost_2 == 0


def test_redis_cache_completion_stream():
    # Important Test - This tests if we can add to streaming cache, when custom callbacks are set
    import random

    from litellm import Cache

    try:
        print("\nrunning test_redis_cache_completion_stream")
        litellm.set_verbose = True
        random_number = random.randint(
            1, 100000
        )  # add a random number to ensure it's always adding / reading from cache
        messages = [
            {
                "role": "user",
                "content": f"write a one sentence poem about: {random_number}",
            }
        ]
        litellm.cache = Cache(
            type="redis",
            host=os.environ["REDIS_HOST"],
            port=os.environ["REDIS_PORT"],
            password=os.environ["REDIS_PASSWORD"],
        )
        print("test for caching, streaming + completion")
        response1 = completion(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=40,
            temperature=0.2,
            stream=True,
            caching=True,
        )
        response_1_content = ""
        response_1_id = None
        for chunk in response1:
            response_1_id = chunk.id
            print(chunk)
            response_1_content += chunk.choices[0].delta.content or ""
        print(response_1_content)

        time.sleep(1)  # sleep for 0.1 seconds allow set cache to occur
        response2 = completion(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=40,
            temperature=0.2,
            stream=True,
            caching=True,
        )
        response_2_content = ""
        response_2_id = None
        for chunk in response2:
            response_2_id = chunk.id
            print(chunk)
            response_2_content += chunk.choices[0].delta.content or ""
        print(
            f"\nresponse 1: {response_1_content}",
        )
        print(f"\nresponse 2: {response_2_content}")
        assert (
            response_1_id == response_2_id
        ), f"Response 1 != Response 2. Same params, Response 1{response_1_content} != Response 2{response_2_content}"
        # assert (
        #     response_1_content == response_2_content
        # ), f"Response 1 != Response 2. Same params, Response 1{response_1_content} != Response 2{response_2_content}"
        litellm.success_callback = []
        litellm._async_success_callback = []
        litellm.cache = None
    except Exception as e:
        print(e)
        litellm.success_callback = []
        raise e


# test_redis_cache_completion_stream()
