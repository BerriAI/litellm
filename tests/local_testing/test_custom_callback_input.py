### What this tests ####
## This test asserts the type of data passed into each method of the custom callback handler
import asyncio
import inspect
import os
import sys
import time
import traceback
from litellm._uuid import uuid
from datetime import datetime

import pytest
from pydantic import BaseModel

sys.path.insert(0, os.path.abspath("../.."))
from typing import List, Literal, Optional, Union
from unittest.mock import AsyncMock, MagicMock, patch

import litellm
from litellm import Cache, completion, embedding
from litellm.integrations.custom_logger import CustomLogger
from litellm.types.utils import LiteLLMCommonStrings

# Test Scenarios (test across completion, streaming, embedding)
## 1: Pre-API-Call
## 2: Post-API-Call
## 3: On LiteLLM Call success
## 4: On LiteLLM Call failure
## 5. Caching

# Test models
## 1. OpenAI
## 2. Azure OpenAI
## 3. Non-OpenAI/Azure - e.g. Bedrock

# Test interfaces
## 1. litellm.completion() + litellm.embeddings()
## refer to test_custom_callback_input_router.py for the router +  proxy tests


class CompletionCustomHandler(
    CustomLogger
):  # https://docs.litellm.ai/docs/observability/custom_callback#callback-class
    """
    The set of expected inputs to a custom handler for a
    """

    # Class variables or attributes
    def __init__(self):
        self.errors = []
        self.states: List[
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
        ] = []

    def log_pre_api_call(self, model, messages, kwargs):
        try:
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
            ### METADATA
            metadata_value = kwargs["litellm_params"].get("metadata")
            assert metadata_value is None or isinstance(metadata_value, dict)
            if metadata_value is not None:
                if litellm.turn_off_message_logging is True:
                    assert (
                        metadata_value["raw_request"]
                        is LiteLLMCommonStrings.redacted_by_litellm.value
                    )
                else:
                    assert "raw_request" not in metadata_value or isinstance(
                        metadata_value["raw_request"], str
                    )
        except Exception:
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
                    kwargs["original_response"],
                    (str, litellm.CustomStreamWrapper, BaseModel),
                )
                or inspect.iscoroutine(kwargs["original_response"])
                or inspect.isasyncgen(kwargs["original_response"])
            )
            assert isinstance(kwargs["additional_args"], (dict, type(None)))
            assert isinstance(kwargs["log_event_type"], str)
        except Exception:
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
            assert isinstance(response_obj, litellm.ModelResponseStream)
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
        except Exception:
            print(f"Assertion Error: {traceback.format_exc()}")
            self.errors.append(traceback.format_exc())

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        try:
            print(f"\n\nkwargs={kwargs}\n\n")
            print(
                json.dumps(kwargs, default=str)
            )  # this is a test to confirm no circular references are in the logging object

            self.states.append("sync_success")
            ## START TIME
            assert isinstance(start_time, datetime)
            ## END TIME
            assert isinstance(end_time, datetime)
            ## RESPONSE OBJECT
            assert isinstance(
                response_obj,
                (
                    litellm.ModelResponse,
                    litellm.EmbeddingResponse,
                    litellm.ImageResponse,
                ),
            )
            ## KWARGS
            assert isinstance(kwargs["model"], str)
            assert isinstance(kwargs["messages"], list) and isinstance(
                kwargs["messages"][0], dict
            )
            assert isinstance(kwargs["optional_params"], dict)
            assert isinstance(kwargs["litellm_params"], dict)
            assert isinstance(kwargs["litellm_params"]["api_base"], str)
            assert kwargs["cache_hit"] is None or isinstance(kwargs["cache_hit"], bool)
            assert isinstance(kwargs["start_time"], (datetime, type(None)))
            assert isinstance(kwargs["stream"], bool)
            assert isinstance(kwargs["user"], (str, type(None)))
            assert (
                isinstance(kwargs["input"], list)
                and (
                    isinstance(kwargs["input"][0], dict)
                    or isinstance(kwargs["input"][0], str)
                )
            ) or isinstance(kwargs["input"], (dict, str))
            assert isinstance(kwargs["api_key"], (str, type(None)))
            assert isinstance(
                kwargs["original_response"],
                (str, litellm.CustomStreamWrapper, BaseModel),
            ), "Original Response={}. Allowed types=[str, litellm.CustomStreamWrapper, BaseModel]".format(
                kwargs["original_response"]
            )
            assert isinstance(kwargs["additional_args"], (dict, type(None)))
            assert isinstance(kwargs["log_event_type"], str)
            assert isinstance(kwargs["response_cost"], (float, type(None)))
        except Exception:
            print(f"Assertion Error: {traceback.format_exc()}")
            self.errors.append(traceback.format_exc())

    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        try:
            print(f"kwargs: {kwargs}")
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
            assert isinstance(kwargs["litellm_params"]["metadata"], Optional[dict])
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
        except Exception:
            print(f"Assertion Error: {traceback.format_exc()}")
            self.errors.append(traceback.format_exc())

    async def async_log_pre_api_call(self, model, messages, kwargs):
        try:
            self.states.append("async_pre_api_call")
            ## MODEL
            assert isinstance(model, str)
            ## MESSAGES
            assert isinstance(messages, list) and isinstance(messages[0], dict)
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
        except Exception as e:
            print(f"Assertion Error: {traceback.format_exc()}")
            self.errors.append(traceback.format_exc())

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        try:
            print(
                "in async_log_success_event", kwargs, response_obj, start_time, end_time
            )
            self.states.append("async_success")
            ## START TIME
            assert isinstance(start_time, datetime)
            ## END TIME
            assert isinstance(end_time, datetime)
            ## RESPONSE OBJECT
            assert isinstance(
                response_obj,
                (
                    litellm.ModelResponse,
                    litellm.EmbeddingResponse,
                    litellm.TextCompletionResponse,
                ),
            )
            ## KWARGS
            assert isinstance(kwargs["model"], str)
            assert isinstance(kwargs["messages"], list)
            assert isinstance(kwargs["optional_params"], dict)
            assert isinstance(kwargs["litellm_params"], dict)
            assert isinstance(kwargs["litellm_params"]["api_base"], str)
            assert isinstance(kwargs["start_time"], (datetime, type(None)))
            assert isinstance(kwargs["stream"], bool)
            assert isinstance(kwargs["completion_start_time"], datetime)
            assert kwargs["cache_hit"] is None or isinstance(kwargs["cache_hit"], bool)
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
            assert isinstance(kwargs["response_cost"], (float, type(None)))
        except Exception:
            print(f"Assertion Error: {traceback.format_exc()}")
            self.errors.append(traceback.format_exc())

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        try:
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
        except Exception:
            print(f"Assertion Error: {traceback.format_exc()}")
            self.errors.append(traceback.format_exc())


# COMPLETION
## Test OpenAI + sync
def test_chat_openai_stream():
    try:
        customHandler = CompletionCustomHandler()
        litellm.callbacks = [customHandler]
        response = litellm.completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hi 👋 - i'm sync openai"}],
        )
        ## test streaming
        response = litellm.completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hi 👋 - i'm openai"}],
            stream=True,
        )
        for chunk in response:
            continue
        ## test failure callback
        try:
            response = litellm.completion(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hi 👋 - i'm openai"}],
                api_key="my-bad-key",
                stream=True,
            )
            for chunk in response:
                continue
        except Exception:
            pass
        time.sleep(1)
        print(f"customHandler.errors: {customHandler.errors}")
        assert len(customHandler.errors) == 0
        litellm.callbacks = []
    except Exception as e:
        pytest.fail(f"An exception occurred: {str(e)}")


# test_chat_openai_stream()


## Test OpenAI + Async
@pytest.mark.asyncio
async def test_async_chat_openai_stream():
    try:
        customHandler = CompletionCustomHandler()
        litellm.callbacks = [customHandler]
        response = await litellm.acompletion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hi 👋 - i'm openai"}],
        )
        ## test streaming
        response = await litellm.acompletion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hi 👋 - i'm openai"}],
            stream=True,
        )
        async for chunk in response:
            continue

        await asyncio.sleep(1)
        ## test failure callback
        try:
            response = await litellm.acompletion(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hi 👋 - i'm openai"}],
                api_key="my-bad-key",
                stream=True,
            )
            async for chunk in response:
                continue
            await asyncio.sleep(1)
        except Exception:
            pass
        time.sleep(1)
        print(f"customHandler.errors: {customHandler.errors}")
        assert len(customHandler.errors) == 0
        litellm.callbacks = []
    except Exception as e:
        pytest.fail(f"An exception occurred: {str(e)}")


# asyncio.run(test_async_chat_openai_stream())


## Test Azure + sync
def test_chat_azure_stream():
    try:
        customHandler = CompletionCustomHandler()
        litellm.callbacks = [customHandler]
        response = litellm.completion(
            model="azure/gpt-4o-new-test",
            messages=[{"role": "user", "content": "Hi 👋 - i'm sync azure"}],
        )
        # test streaming
        response = litellm.completion(
            model="azure/gpt-4o-new-test",
            messages=[{"role": "user", "content": "Hi 👋 - i'm sync azure"}],
            stream=True,
        )
        for chunk in response:
            continue
        # test failure callback
        try:
            response = litellm.completion(
                model="azure/gpt-4o-new-test",
                messages=[{"role": "user", "content": "Hi 👋 - i'm sync azure"}],
                api_key="my-bad-key",
                stream=True,
            )
            for chunk in response:
                continue
        except Exception:
            pass
        time.sleep(1)
        print(f"customHandler.errors: {customHandler.errors}")
        assert len(customHandler.errors) == 0
        litellm.callbacks = []
    except Exception as e:
        pytest.fail(f"An exception occurred: {str(e)}")


# test_chat_azure_stream()


## Test Azure + Async
@pytest.mark.asyncio
async def test_async_chat_azure_stream():
    try:
        customHandler = CompletionCustomHandler()
        litellm.callbacks = [customHandler]
        response = await litellm.acompletion(
            model="azure/chatgpt-v-3",
            messages=[{"role": "user", "content": "Hi 👋 - i'm async azure"}],
        )
        ## test streaming
        response = await litellm.acompletion(
            model="azure/chatgpt-v-3",
            messages=[{"role": "user", "content": "Hi 👋 - i'm async azure"}],
            stream=True,
        )
        async for chunk in response:
            continue

        await asyncio.sleep(1)
        # test failure callback
        try:
            response = await litellm.acompletion(
                model="azure/chatgpt-v-3",
                messages=[{"role": "user", "content": "Hi 👋 - i'm async azure"}],
                api_key="my-bad-key",
                stream=True,
            )
            async for chunk in response:
                continue
            await asyncio.sleep(1)
        except Exception:
            pass
        await asyncio.sleep(1)
        print(f"customHandler.errors: {customHandler.errors}")
        assert len(customHandler.errors) == 0
        litellm.callbacks = []
    except Exception as e:
        pytest.fail(f"An exception occurred: {str(e)}")


# asyncio.run(test_async_chat_azure_stream())


@pytest.mark.asyncio
async def test_async_chat_openai_stream_options():
    try:
        litellm.set_verbose = True
        customHandler = CompletionCustomHandler()
        litellm.callbacks = [customHandler]
        with patch.object(
            customHandler, "async_log_success_event", new=AsyncMock()
        ) as mock_client:
            response = await litellm.acompletion(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hi 👋 - i'm async openai"}],
                stream=True,
                stream_options={"include_usage": True},
            )

            async for chunk in response:
                continue

            await asyncio.sleep(1)
            print("mock client args list=", mock_client.await_args_list)
            mock_client.assert_awaited_once()
    except Exception as e:
        pytest.fail(f"An exception occurred: {str(e)}")


# asyncio.run(test_async_chat_bedrock_stream())


## Test Sagemaker + Async
@pytest.mark.skip(reason="AWS Suspended Account")
@pytest.mark.asyncio
async def test_async_chat_sagemaker_stream():
    try:
        customHandler = CompletionCustomHandler()
        litellm.callbacks = [customHandler]
        response = await litellm.acompletion(
            model="sagemaker/berri-benchmarking-Llama-2-70b-chat-hf-4",
            messages=[{"role": "user", "content": "Hi 👋 - i'm async sagemaker"}],
        )
        # test streaming
        response = await litellm.acompletion(
            model="sagemaker/berri-benchmarking-Llama-2-70b-chat-hf-4",
            messages=[{"role": "user", "content": "Hi 👋 - i'm async sagemaker"}],
            stream=True,
        )
        print(f"response: {response}")
        async for chunk in response:
            print(f"chunk: {chunk}")
            continue
        ## test failure callback
        try:
            response = await litellm.acompletion(
                model="sagemaker/berri-benchmarking-Llama-2-70b-chat-hf-4",
                messages=[{"role": "user", "content": "Hi 👋 - i'm async sagemaker"}],
                aws_region_name="my-bad-key",
                stream=True,
            )
            async for chunk in response:
                continue
        except Exception:
            pass
        time.sleep(1)
        print(f"customHandler.errors: {customHandler.errors}")
        assert len(customHandler.errors) == 0
        litellm.callbacks = []
    except Exception as e:
        pytest.fail(f"An exception occurred: {str(e)}")


## Test Vertex AI + Async
import json
import tempfile


def load_vertex_ai_credentials():
    # Define the path to the vertex_key.json file
    print("loading vertex ai credentials")
    filepath = os.path.dirname(os.path.abspath(__file__))
    vertex_key_path = filepath + "/vertex_key.json"

    # Read the existing content of the file or create an empty dictionary
    try:
        with open(vertex_key_path, "r") as file:
            # Read the file content
            print("Read vertexai file path")
            content = file.read()

            # If the file is empty or not valid JSON, create an empty dictionary
            if not content or not content.strip():
                service_account_key_data = {}
            else:
                # Attempt to load the existing JSON content
                file.seek(0)
                service_account_key_data = json.load(file)
    except FileNotFoundError:
        # If the file doesn't exist, create an empty dictionary
        service_account_key_data = {}

    # Update the service_account_key_data with environment variables
    private_key_id = os.environ.get("VERTEX_AI_PRIVATE_KEY_ID", "")
    private_key = os.environ.get("VERTEX_AI_PRIVATE_KEY", "")
    private_key = private_key.replace("\\n", "\n")
    service_account_key_data["private_key_id"] = private_key_id
    service_account_key_data["private_key"] = private_key

    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as temp_file:
        # Write the updated content to the temporary file
        json.dump(service_account_key_data, temp_file, indent=2)

    # Export the temporary file as GOOGLE_APPLICATION_CREDENTIALS
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.abspath(temp_file.name)


@pytest.mark.skip(reason="Vertex AI Hanging")
@pytest.mark.asyncio
async def test_async_chat_vertex_ai_stream():
    try:
        load_vertex_ai_credentials()
        customHandler = CompletionCustomHandler()
        litellm.set_verbose = True
        litellm.callbacks = [customHandler]
        # test streaming
        response = await litellm.acompletion(
            model="gemini-pro",
            messages=[
                {
                    "role": "user",
                    "content": f"Hi 👋 - i'm async vertex_ai {uuid.uuid4()}",
                }
            ],
            stream=True,
        )
        print(f"response: {response}")
        async for chunk in response:
            print(f"chunk: {chunk}")
            continue
        await asyncio.sleep(10)
        print(f"customHandler.states: {customHandler.states}")
        assert (
            customHandler.states.count("async_success") == 1
        )  # pre, post, success, pre, post, failure
        assert len(customHandler.states) >= 3  # pre, post, success
    except Exception as e:
        pytest.fail(f"An exception occurred: {str(e)}")


# Text Completion


@pytest.mark.asyncio
@pytest.mark.skip(reason="temp-skip to see what else is failing")
async def test_async_text_completion_bedrock():
    try:
        customHandler = CompletionCustomHandler()
        litellm.callbacks = [customHandler]
        response = await litellm.atext_completion(
            model="bedrock/anthropic.claude-3-haiku-20240307-v1:0",
            prompt=["Hi 👋 - i'm async text completion bedrock"],
        )
        # test streaming
        response = await litellm.atext_completion(
            model="bedrock/anthropic.claude-3-haiku-20240307-v1:0",
            prompt=["Hi 👋 - i'm async text completion bedrock"],
            stream=True,
        )
        async for chunk in response:
            print(f"chunk: {chunk}")
            continue

        await asyncio.sleep(1)
        ## test failure callback
        try:
            response = await litellm.atext_completion(
                model="bedrock/",
                prompt=["Hi 👋 - i'm async text completion bedrock"],
                stream=True,
                api_key="my-bad-key",
            )
            async for chunk in response:
                continue

            await asyncio.sleep(1)
        except Exception:
            pass
        time.sleep(1)
        print(f"customHandler.errors: {customHandler.errors}")
        assert len(customHandler.errors) == 0
        litellm.callbacks = []
    except Exception as e:
        pytest.fail(f"An exception occurred: {str(e)}")


## Test OpenAI text completion + Async
@pytest.mark.asyncio
async def test_async_text_completion_openai_stream():
    try:
        customHandler = CompletionCustomHandler()
        litellm.callbacks = [customHandler]
        response = await litellm.atext_completion(
            model="gpt-3.5-turbo",
            prompt="Hi 👋 - i'm async text completion openai",
        )
        # test streaming
        response = await litellm.atext_completion(
            model="gpt-3.5-turbo",
            prompt="Hi 👋 - i'm async text completion openai",
            stream=True,
        )
        async for chunk in response:
            print(f"chunk: {chunk}")
            continue

        await asyncio.sleep(1)
        ## test failure callback
        try:
            response = await litellm.atext_completion(
                model="gpt-3.5-turbo",
                prompt="Hi 👋 - i'm async text completion openai",
                stream=True,
                api_key="my-bad-key",
            )
            async for chunk in response:
                continue

            await asyncio.sleep(1)
        except Exception:
            pass
        time.sleep(1)
        print(f"customHandler.errors: {customHandler.errors}")
        assert len(customHandler.errors) == 0
        litellm.callbacks = []
    except Exception as e:
        pytest.fail(f"An exception occurred: {str(e)}")


# EMBEDDING
## Test OpenAI + Async
@pytest.mark.asyncio
async def test_async_embedding_openai():
    try:
        customHandler_success = CompletionCustomHandler()
        customHandler_failure = CompletionCustomHandler()
        litellm.callbacks = [customHandler_success]
        response = await litellm.aembedding(
            model="azure/azure-embedding-model", input=["good morning from litellm"]
        )
        await asyncio.sleep(1)
        print(f"customHandler_success.errors: {customHandler_success.errors}")
        print(f"customHandler_success.states: {customHandler_success.states}")
        assert len(customHandler_success.errors) == 0
        assert len(customHandler_success.states) == 3  # pre, post, success
        # test failure callback
        litellm.logging_callback_manager._reset_all_callbacks()
        litellm.callbacks = [customHandler_failure]
        try:
            response = await litellm.aembedding(
                model="text-embedding-ada-002",
                input=["good morning from litellm"],
                api_key="my-bad-key",
            )
        except Exception:
            pass
        await asyncio.sleep(1)
        print(f"customHandler_failure.errors: {customHandler_failure.errors}")
        print(f"customHandler_failure.states: {customHandler_failure.states}")
        assert len(customHandler_failure.errors) == 0
        assert len(customHandler_failure.states) == 3  # pre, post, failure
    except Exception as e:
        pytest.fail(f"An exception occurred: {str(e)}")


# asyncio.run(test_async_embedding_openai())


## Test Azure + Async
def test_amazing_sync_embedding():
    try:
        customHandler_success = CompletionCustomHandler()
        customHandler_failure = CompletionCustomHandler()
        litellm.callbacks = [customHandler_success]
        response = litellm.embedding(
            model="azure/azure-embedding-model", input=["good morning from litellm"]
        )
        print(f"customHandler_success.errors: {customHandler_success.errors}")
        print(f"customHandler_success.states: {customHandler_success.states}")
        time.sleep(2)
        assert len(customHandler_success.errors) == 0
        assert len(customHandler_success.states) == 3  # pre, post, success
        # test failure callback
        litellm.logging_callback_manager._reset_all_callbacks()
        litellm.callbacks = [customHandler_failure]
        try:
            response = litellm.embedding(
                model="azure/azure-embedding-model",
                input=["good morning from litellm"],
                api_key="my-bad-key",
            )
        except Exception:
            pass
        print(f"customHandler_failure.errors: {customHandler_failure.errors}")
        print(f"customHandler_failure.states: {customHandler_failure.states}")
        time.sleep(2)
        assert len(customHandler_failure.errors) == 1
        assert len(customHandler_failure.states) == 3  # pre, post, failure
    except Exception as e:
        pytest.fail(f"An exception occurred: {str(e)}")


## Test Azure + Async
@pytest.mark.asyncio
async def test_async_embedding_azure():
    try:
        customHandler_success = CompletionCustomHandler()
        customHandler_failure = CompletionCustomHandler()
        litellm.callbacks = [customHandler_success]
        response = await litellm.aembedding(
            model="azure/azure-embedding-model", input=["good morning from litellm"]
        )
        await asyncio.sleep(1)
        print(f"customHandler_success.errors: {customHandler_success.errors}")
        print(f"customHandler_success.states: {customHandler_success.states}")
        assert len(customHandler_success.errors) == 0
        assert len(customHandler_success.states) == 3  # pre, post, success
        # test failure callback
        litellm.logging_callback_manager._reset_all_callbacks()
        litellm.callbacks = [customHandler_failure]
        try:
            response = await litellm.aembedding(
                model="azure/azure-embedding-model",
                input=["good morning from litellm"],
                api_key="my-bad-key",
            )
        except Exception:
            pass
        await asyncio.sleep(1)
        print(f"customHandler_failure.errors: {customHandler_failure.errors}")
        print(f"customHandler_failure.states: {customHandler_failure.states}")
        assert len(customHandler_failure.errors) == 0
        assert len(customHandler_failure.states) == 3  # pre, post, success
    except Exception as e:
        pytest.fail(f"An exception occurred: {str(e)}")


# asyncio.run(test_async_embedding_azure())


## Test Bedrock + Async
@pytest.mark.asyncio
async def test_async_embedding_bedrock():
    try:
        customHandler_success = CompletionCustomHandler()
        customHandler_failure = CompletionCustomHandler()
        litellm.callbacks = [customHandler_success]
        litellm.set_verbose = True
        response = await litellm.aembedding(
            model="bedrock/cohere.embed-multilingual-v3",
            input=["good morning from litellm"],
            aws_region_name="us-east-1",
        )
        await asyncio.sleep(1)
        print(f"customHandler_success.errors: {customHandler_success.errors}")
        print(f"customHandler_success.states: {customHandler_success.states}")
        assert len(customHandler_success.errors) == 0
        assert len(customHandler_success.states) == 3  # pre, post, success
        # test failure callback
        litellm.logging_callback_manager._reset_all_callbacks()
        litellm.callbacks = [customHandler_failure]
        try:
            response = await litellm.aembedding(
                model="bedrock/cohere.embed-multilingual-v3",
                input=["good morning from litellm"],
                aws_region_name="my-bad-region",
            )
        except Exception:
            pass
        await asyncio.sleep(1)
        print(f"customHandler_failure.errors: {customHandler_failure.errors}")
        print(f"customHandler_failure.states: {customHandler_failure.states}")
        assert len(customHandler_failure.errors) == 0
        assert len(customHandler_failure.states) == 3  # pre, post, success
    except Exception as e:
        pytest.fail(f"An exception occurred: {str(e)}")


# asyncio.run(test_async_embedding_bedrock())


# CACHING
## Test Azure - completion, embedding
@pytest.mark.asyncio
@pytest.mark.flaky(retries=3, delay=1)
async def test_async_completion_azure_caching():
    litellm.set_verbose = True
    customHandler_caching = CompletionCustomHandler()
    litellm.cache = Cache(
        type="redis",
        host=os.environ["REDIS_HOST"],
        port=os.environ["REDIS_PORT"],
        password=os.environ["REDIS_PASSWORD"],
    )
    litellm.callbacks = [customHandler_caching]
    unique_time = time.time()
    response1 = await litellm.acompletion(
        model="azure/chatgpt-v-3",
        messages=[
            {"role": "user", "content": f"Hi 👋 - i'm async azure {unique_time}"}
        ],
        caching=True,
    )
    await asyncio.sleep(1)
    print(f"customHandler_caching.states pre-cache hit: {customHandler_caching.states}")
    response2 = await litellm.acompletion(
        model="azure/chatgpt-v-3",
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
async def test_async_completion_azure_caching_streaming():
    import copy

    litellm.set_verbose = True
    customHandler_caching = CompletionCustomHandler()
    litellm.cache = Cache(
        type="redis",
        host=os.environ["REDIS_HOST"],
        port=os.environ["REDIS_PORT"],
        password=os.environ["REDIS_PASSWORD"],
    )
    litellm.callbacks = [customHandler_caching]
    unique_time = uuid.uuid4()
    response1 = await litellm.acompletion(
        model="azure/chatgpt-v-3",
        messages=[
            {"role": "user", "content": f"Hi 👋 - i'm async azure {unique_time}"}
        ],
        caching=True,
        stream=True,
    )
    async for chunk in response1:
        print(f"chunk in response1: {chunk}")
    await asyncio.sleep(1)
    initial_customhandler_caching_states = len(customHandler_caching.states)
    print(f"customHandler_caching.states pre-cache hit: {customHandler_caching.states}")
    response2 = await litellm.acompletion(
        model="azure/chatgpt-v-3",
        messages=[
            {"role": "user", "content": f"Hi 👋 - i'm async azure {unique_time}"}
        ],
        caching=True,
        stream=True,
    )
    async for chunk in response2:
        print(f"chunk in response2: {chunk}")
    await asyncio.sleep(1)  # success callbacks are done in parallel
    print(
        f"customHandler_caching.states post-cache hit: {customHandler_caching.states}"
    )
    assert len(customHandler_caching.errors) == 0
    assert (
        len(customHandler_caching.states) > initial_customhandler_caching_states
    )  # pre, post, streaming .., success, success


@pytest.mark.asyncio
@pytest.mark.flaky(retries=3, delay=2)
async def test_async_embedding_azure_caching():
    print("Testing custom callback input - Azure Caching")
    customHandler_caching = CompletionCustomHandler()
    litellm.cache = Cache(
        type="redis",
        host=os.environ["REDIS_HOST"],
        port=os.environ["REDIS_PORT"],
        password=os.environ["REDIS_PASSWORD"],
    )
    litellm.callbacks = [customHandler_caching]
    unique_time = time.time()
    response1 = await litellm.aembedding(
        model="azure/azure-embedding-model",
        input=[f"good morning from litellm1 {unique_time}"],
        caching=True,
    )
    await asyncio.sleep(1)  # set cache is async for aembedding()
    response2 = await litellm.aembedding(
        model="azure/azure-embedding-model",
        input=[f"good morning from litellm1 {unique_time}"],
        caching=True,
    )
    await asyncio.sleep(1)  # success callbacks are done in parallel
    print(customHandler_caching.states)
    print(customHandler_caching.errors)
    assert len(customHandler_caching.errors) == 0
    assert len(customHandler_caching.states) == 4  # pre, post, success, success


# Image Generation


## Test OpenAI + Sync
@pytest.mark.flaky(retries=3, delay=1)
def test_image_generation_openai():
    try:
        customHandler_success = CompletionCustomHandler()
        customHandler_failure = CompletionCustomHandler()
        litellm.callbacks = [customHandler_success]

        litellm.set_verbose = True

        response = litellm.image_generation(
            prompt="A cute baby sea otter",
            model="azure/dall-e-3-test",
            api_version="2023-12-01-preview",
            api_base=os.getenv("AZURE_SWEDEN_API_BASE"),
            api_key=os.getenv("AZURE_SWEDEN_API_KEY"),
        )

        print(f"response: {response}")
        assert len(response.data) > 0

        print(f"customHandler_success.errors: {customHandler_success.errors}")
        print(f"customHandler_success.states: {customHandler_success.states}")
        time.sleep(2)
        assert len(customHandler_success.errors) == 0
        assert len(customHandler_success.states) == 3  # pre, post, success
        # test failure callback
        litellm.logging_callback_manager._reset_all_callbacks()
        litellm.callbacks = [customHandler_failure]
        try:
            response = litellm.image_generation(
                prompt="A cute baby sea otter",
                model="dall-e-2",
                api_key="my-bad-api-key",
            )
        except Exception:
            pass
        print(f"customHandler_failure.errors: {customHandler_failure.errors}")
        print(f"customHandler_failure.states: {customHandler_failure.states}")
        assert len(customHandler_failure.errors) == 0
        assert len(customHandler_failure.states) == 3  # pre, post, failure
    except litellm.RateLimitError as e:
        pass
    except litellm.ContentPolicyViolationError:
        pass  # OpenAI randomly raises these errors - skip when they occur
    except Exception as e:
        pytest.fail(f"An exception occurred - {str(e)}")


# test_image_generation_openai()
## Test OpenAI + Async

## Test Azure + Sync

## Test Azure + Async

##### PII REDACTION ######


def test_turn_off_message_logging():
    """
    If 'turn_off_message_logging' is true, assert no user request information is logged.
    """
    litellm.turn_off_message_logging = True

    # sync completion
    customHandler = CompletionCustomHandler()
    litellm.callbacks = [customHandler]

    _ = litellm.completion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hey, how's it going?"}],
        mock_response="Going well!",
    )

    time.sleep(2)
    assert len(customHandler.errors) == 0


##### VALID JSON ######


@pytest.mark.parametrize(
    "model",
    [
        "ft:gpt-3.5-turbo:my-org:custom_suffix:id"
    ],  # "gpt-3.5-turbo", "azure/chatgpt-v-3",
)
@pytest.mark.parametrize(
    "turn_off_message_logging",
    [
        True,
    ],
)  # False
def test_standard_logging_payload(model, turn_off_message_logging):
    """
    Ensure valid standard_logging_payload is passed for logging calls to s3

    Motivation: provide a standard set of things that are logged to s3/gcs/future integrations across all llm calls
    """
    from litellm.types.utils import StandardLoggingPayload

    # sync completion
    customHandler = CompletionCustomHandler()
    litellm.callbacks = [customHandler]

    litellm.turn_off_message_logging = turn_off_message_logging

    with patch.object(
        customHandler, "log_success_event", new=MagicMock()
    ) as mock_client:
        _ = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": "Hey, how's it going?"}],
            mock_response="Going well!",
        )

        time.sleep(2)
        mock_client.assert_called_once()

        print(
            f"mock_client_post.call_args: {mock_client.call_args.kwargs['kwargs'].keys()}"
        )
        assert "standard_logging_object" in mock_client.call_args.kwargs["kwargs"]
        assert (
            mock_client.call_args.kwargs["kwargs"]["standard_logging_object"]
            is not None
        )

        print(
            "Standard Logging Object - {}".format(
                mock_client.call_args.kwargs["kwargs"]["standard_logging_object"]
            )
        )

        keys_list = list(StandardLoggingPayload.__annotations__.keys())

        for k in keys_list:
            assert (
                k in mock_client.call_args.kwargs["kwargs"]["standard_logging_object"]
            )

        ## json serializable
        json_str_payload = json.dumps(
            mock_client.call_args.kwargs["kwargs"]["standard_logging_object"]
        )
        json.loads(json_str_payload)

        ## response cost
        assert (
            mock_client.call_args.kwargs["kwargs"]["standard_logging_object"][
                "response_cost"
            ]
            > 0
        )
        assert (
            mock_client.call_args.kwargs["kwargs"]["standard_logging_object"][
                "model_map_information"
            ]["model_map_value"]
            is not None
        )

        ## turn off message logging
        slobject: StandardLoggingPayload = mock_client.call_args.kwargs["kwargs"][
            "standard_logging_object"
        ]
        if turn_off_message_logging:
            print("checks redacted-by-litellm")
            assert "redacted-by-litellm" == slobject["messages"][0]["content"]
            assert {"text": "redacted-by-litellm"} == slobject["response"]


@pytest.mark.parametrize(
    "stream",
    [True, False],
)
@pytest.mark.parametrize(
    "turn_off_message_logging",
    [
        True,
    ],
)  # False
def test_standard_logging_payload_audio(turn_off_message_logging, stream):
    """
    Ensure valid standard_logging_payload is passed for logging calls to s3

    Motivation: provide a standard set of things that are logged to s3/gcs/future integrations across all llm calls
    """
    from litellm.types.utils import StandardLoggingPayload

    # sync completion
    customHandler = CompletionCustomHandler()
    litellm.callbacks = [customHandler]

    litellm.turn_off_message_logging = turn_off_message_logging

    with patch.object(
        customHandler, "log_success_event", new=MagicMock()
    ) as mock_client:
        try:
            response = litellm.completion(
                model="gpt-4o-audio-preview",
                modalities=["text", "audio"],
                audio={"voice": "alloy", "format": "pcm16"},
                messages=[
                    {"role": "user", "content": "response in 1 word - yes or no"}
                ],
                stream=stream,
            )
        except Exception as e:
            if "openai-internal" in str(e):
                pytest.skip("Skipping test due to openai-internal error")

        if stream:
            for chunk in response:
                continue

        time.sleep(2)
        mock_client.assert_called()

        print(
            f"mock_client_post.call_args: {mock_client.call_args.kwargs['kwargs'].keys()}"
        )
        assert "standard_logging_object" in mock_client.call_args.kwargs["kwargs"]
        assert (
            mock_client.call_args.kwargs["kwargs"]["standard_logging_object"]
            is not None
        )

        print(
            "Standard Logging Object - {}".format(
                mock_client.call_args.kwargs["kwargs"]["standard_logging_object"]
            )
        )

        keys_list = list(StandardLoggingPayload.__annotations__.keys())

        for k in keys_list:
            assert (
                k in mock_client.call_args.kwargs["kwargs"]["standard_logging_object"]
            )

        ## json serializable
        json_str_payload = json.dumps(
            mock_client.call_args.kwargs["kwargs"]["standard_logging_object"]
        )
        json.loads(json_str_payload)

        ## response cost
        assert (
            mock_client.call_args.kwargs["kwargs"]["standard_logging_object"][
                "response_cost"
            ]
            > 0
        )
        assert (
            mock_client.call_args.kwargs["kwargs"]["standard_logging_object"][
                "model_map_information"
            ]["model_map_value"]
            is not None
        )

        ## turn off message logging
        slobject: StandardLoggingPayload = mock_client.call_args.kwargs["kwargs"][
            "standard_logging_object"
        ]
        if turn_off_message_logging:
            print("checks redacted-by-litellm")
            assert "redacted-by-litellm" == slobject["messages"][0]["content"]
            assert {"text": "redacted-by-litellm"} == slobject["response"]


@pytest.mark.skip(reason="Works locally. Flaky on ci/cd")
def test_aaastandard_logging_payload_cache_hit():
    from litellm.types.utils import StandardLoggingPayload

    # sync completion

    litellm.cache = Cache()

    _ = litellm.completion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hey, how's it going?"}],
        caching=True,
    )

    customHandler = CompletionCustomHandler()
    litellm.callbacks = [customHandler]
    litellm.success_callback = []

    with patch.object(
        customHandler, "log_success_event", new=MagicMock()
    ) as mock_client:
        _ = litellm.completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hey, how's it going?"}],
            caching=True,
        )

        time.sleep(2)
        mock_client.assert_called_once()

        assert "standard_logging_object" in mock_client.call_args.kwargs["kwargs"]
        assert (
            mock_client.call_args.kwargs["kwargs"]["standard_logging_object"]
            is not None
        )

        standard_logging_object: StandardLoggingPayload = mock_client.call_args.kwargs[
            "kwargs"
        ]["standard_logging_object"]

        assert standard_logging_object["cache_hit"] is True
        assert standard_logging_object["response_cost"] == 0
        assert standard_logging_object["saved_cache_cost"] > 0


@pytest.mark.parametrize(
    "turn_off_message_logging",
    [False, True],
)  # False
def test_logging_async_cache_hit_sync_call(turn_off_message_logging):
    from litellm.types.utils import StandardLoggingPayload

    litellm.turn_off_message_logging = turn_off_message_logging

    litellm.cache = Cache()

    response = litellm.completion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hey, how's it going?"}],
        caching=True,
        stream=True,
    )
    for chunk in response:
        print(chunk)

    time.sleep(3)
    customHandler = CompletionCustomHandler()
    litellm.callbacks = [customHandler]
    litellm.success_callback = []

    with patch.object(
        customHandler, "log_success_event", new=MagicMock()
    ) as mock_client:
        resp = litellm.completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hey, how's it going?"}],
            caching=True,
            stream=True,
        )

        for chunk in resp:
            print(chunk)

        time.sleep(2)
        mock_client.assert_called_once()

        assert "standard_logging_object" in mock_client.call_args.kwargs["kwargs"]
        assert (
            mock_client.call_args.kwargs["kwargs"]["standard_logging_object"]
            is not None
        )

        standard_logging_object: StandardLoggingPayload = mock_client.call_args.kwargs[
            "kwargs"
        ]["standard_logging_object"]

        assert standard_logging_object["cache_hit"] is True
        assert standard_logging_object["response_cost"] == 0
        assert standard_logging_object["saved_cache_cost"] > 0

        if turn_off_message_logging:
            print("checks redacted-by-litellm")
            assert (
                "redacted-by-litellm"
                == standard_logging_object["messages"][0]["content"]
            )
            assert {"text": "redacted-by-litellm"} == standard_logging_object[
                "response"
            ]


def test_logging_standard_payload_failure_call():
    from litellm.types.utils import StandardLoggingPayload

    customHandler = CompletionCustomHandler()
    litellm.callbacks = [customHandler]

    with patch.object(
        customHandler, "log_failure_event", new=MagicMock()
    ) as mock_client:
        try:
            resp = litellm.completion(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hey, how's it going?"}],
                api_key="my-bad-api-key",
            )
        except litellm.AuthenticationError:
            pass

        mock_client.assert_called_once()

        assert "standard_logging_object" in mock_client.call_args.kwargs["kwargs"]
        assert (
            mock_client.call_args.kwargs["kwargs"]["standard_logging_object"]
            is not None
        )

        standard_logging_object: StandardLoggingPayload = mock_client.call_args.kwargs[
            "kwargs"
        ]["standard_logging_object"]
        assert "additional_headers" in standard_logging_object["hidden_params"]


@pytest.mark.parametrize("stream", [False, True])
def test_logging_standard_payload_llm_headers(stream):
    from litellm.types.utils import StandardLoggingPayload

    # sync completion
    customHandler = CompletionCustomHandler()
    litellm.callbacks = [customHandler]

    with patch.object(
        customHandler, "log_success_event", new=MagicMock()
    ) as mock_client:

        resp = litellm.completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hey, how's it going?"}],
            stream=stream,
        )

        if stream:
            for chunk in resp:
                continue

        time.sleep(2)
        mock_client.assert_called()

        standard_logging_object: StandardLoggingPayload = mock_client.call_args.kwargs[
            "kwargs"
        ]["standard_logging_object"]

        print(standard_logging_object["hidden_params"]["additional_headers"])


def test_logging_key_masking_gemini():
    customHandler = CompletionCustomHandler()
    litellm.callbacks = [customHandler]
    litellm.success_callback = []

    with patch.object(
        customHandler, "log_pre_api_call", new=MagicMock()
    ) as mock_client:
        try:
            resp = litellm.completion(
                model="gemini/gemini-1.5-pro",
                messages=[{"role": "user", "content": "Hey, how's it going?"}],
                api_key="LEAVE_ONLY_LAST_4_CHAR_UNMASKED_THIS_PART",
            )
        except litellm.AuthenticationError:
            pass

        mock_client.assert_called()

        print(f"mock_client.call_args.kwargs: {mock_client.call_args.kwargs}")
        assert (
            "LEAVE_ONLY_LAST_4_CHAR_UNMASKED_THIS_PART"
            not in mock_client.call_args.kwargs["kwargs"]["litellm_params"]["api_base"]
        )
        key = mock_client.call_args.kwargs["kwargs"]["litellm_params"]["api_base"]
        trimmed_key = key.split("key=")[1]
        trimmed_key = trimmed_key.replace("*", "")
        assert "PART" == trimmed_key


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_standard_logging_payload_stream_usage(sync_mode):
    """
    Even if stream_options is not provided, correct usage should be logged
    """
    from litellm.types.utils import StandardLoggingPayload
    from litellm.main import stream_chunk_builder

    stream = True
    try:
        # sync completion
        customHandler = CompletionCustomHandler()
        litellm.callbacks = [customHandler]

        if sync_mode:
            patch_event = "log_success_event"
            return_val = MagicMock()
        else:
            patch_event = "async_log_success_event"
            return_val = AsyncMock()

        with patch.object(customHandler, patch_event, new=return_val) as mock_client:
            if sync_mode:
                resp = litellm.completion(
                    model="anthropic/claude-3-5-sonnet-20240620",
                    messages=[{"role": "user", "content": "Hey, how's it going?"}],
                    stream=stream,
                )

                chunks = []
                for chunk in resp:
                    chunks.append(chunk)
                time.sleep(2)
            else:
                resp = await litellm.acompletion(
                    model="anthropic/claude-3-5-sonnet-20240620",
                    messages=[{"role": "user", "content": "Hey, how's it going?"}],
                    stream=stream,
                )

                chunks = []
                async for chunk in resp:
                    chunks.append(chunk)
                await asyncio.sleep(2)

            mock_client.assert_called_once()

            standard_logging_object: StandardLoggingPayload = (
                mock_client.call_args.kwargs["kwargs"]["standard_logging_object"]
            )

            built_response = stream_chunk_builder(chunks=chunks)
            print(f"built_response: {built_response}")
            assert (
                built_response.usage.total_tokens
                == standard_logging_object["total_tokens"]
            )
            print(f"standard_logging_object usage: {built_response.usage}")
    except litellm.InternalServerError:
        pass


def test_standard_logging_retries():
    """
    know if a request was retried.
    """
    from litellm.types.utils import StandardLoggingPayload
    from litellm.router import Router

    customHandler = CompletionCustomHandler()
    litellm.callbacks = [customHandler]

    router = Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "openai/gpt-3.5-turbo",
                    "api_key": "test-api-key",
                },
            }
        ]
    )

    with patch.object(
        customHandler, "log_failure_event", new=MagicMock()
    ) as mock_client:
        try:
            router.completion(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hey, how's it going?"}],
                num_retries=1,
                mock_response="litellm.RateLimitError",
            )
        except litellm.RateLimitError:
            pass

        assert mock_client.call_count == 2
        assert (
            mock_client.call_args_list[0].kwargs["kwargs"]["standard_logging_object"][
                "trace_id"
            ]
            is not None
        )
        assert (
            mock_client.call_args_list[0].kwargs["kwargs"]["standard_logging_object"][
                "trace_id"
            ]
            == mock_client.call_args_list[1].kwargs["kwargs"][
                "standard_logging_object"
            ]["trace_id"]
        )


@pytest.mark.parametrize("disable_no_log_param", [True, False])
def test_litellm_logging_no_log_param(monkeypatch, disable_no_log_param):
    monkeypatch.setattr(litellm, "global_disable_no_log_param", disable_no_log_param)
    from litellm.litellm_core_utils.litellm_logging import Logging

    litellm.success_callback = ["langfuse"]
    litellm_call_id = "my-unique-call-id"
    litellm_logging_obj = Logging(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "hi"}],
        stream=False,
        call_type="acompletion",
        litellm_call_id=litellm_call_id,
        start_time=datetime.now(),
        function_id="1234",
    )

    should_run = litellm_logging_obj.should_run_callback(
        callback="langfuse",
        litellm_params={"no-log": True},
        event_hook="success_handler",
    )

    if disable_no_log_param:
        assert should_run is True
    else:
        assert should_run is False
