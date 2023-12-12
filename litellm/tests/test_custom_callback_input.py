### What this tests ####
## This test asserts the type of data passed into each method of the custom callback handler
import sys, os, time, inspect, asyncio, traceback
from datetime import datetime
import pytest
sys.path.insert(0, os.path.abspath('../..'))
from typing import Optional, Literal, List, Union
from litellm import completion, embedding
import litellm
from litellm.integrations.custom_logger import CustomLogger

# Test Scenarios (test across completion, streaming, embedding)
## 1: Pre-API-Call
## 2: Post-API-Call
## 3: On LiteLLM Call success
## 4: On LiteLLM Call failure

# Test models 
## 1. OpenAI 
## 2. Azure OpenAI 
## 3. Non-OpenAI/Azure - e.g. Bedrock

# Test interfaces
## 1. litellm.completion() + litellm.embeddings()
## refer to test_custom_callback_input_router.py for the router +  proxy tests

class CompletionCustomHandler(CustomLogger): # https://docs.litellm.ai/docs/observability/custom_callback#callback-class
    """
    The set of expected inputs to a custom handler for a 
    """
    # Class variables or attributes
    def __init__(self):
        self.errors = []
        self.states: Optional[List[Literal["sync_pre_api_call", "async_pre_api_call", "post_api_call", "sync_stream", "async_stream", "sync_success", "async_success", "sync_failure", "async_failure"]]] = []

    def log_pre_api_call(self, model, messages, kwargs): 
        try: 
            self.states.append("sync_pre_api_call")
            ## MODEL
            assert isinstance(model, str)
            ## MESSAGES
            assert isinstance(messages, list)
            ## KWARGS
            assert isinstance(kwargs['model'], str)
            assert isinstance(kwargs['messages'], list) 
            assert isinstance(kwargs['optional_params'], dict)
            assert isinstance(kwargs['litellm_params'], dict) 
            assert isinstance(kwargs['start_time'], (datetime, type(None)))
            assert isinstance(kwargs['stream'], bool)
            assert isinstance(kwargs['user'], (str, type(None)))
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
            assert isinstance(kwargs['model'], str)
            assert isinstance(kwargs['messages'], list)
            assert isinstance(kwargs['optional_params'], dict)
            assert isinstance(kwargs['litellm_params'], dict) 
            assert isinstance(kwargs['start_time'], (datetime, type(None)))
            assert isinstance(kwargs['stream'], bool)
            assert isinstance(kwargs['user'], (str, type(None)))
            assert isinstance(kwargs['input'], (list, dict, str))
            assert isinstance(kwargs['api_key'], (str, type(None)))
            assert isinstance(kwargs['original_response'], (str, litellm.CustomStreamWrapper)) or inspect.iscoroutine(kwargs['original_response']) or inspect.isasyncgen(kwargs['original_response'])
            assert isinstance(kwargs['additional_args'], (dict, type(None)))
            assert isinstance(kwargs['log_event_type'], str) 
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
            assert isinstance(kwargs['model'], str)
            assert isinstance(kwargs['messages'], list) and isinstance(kwargs['messages'][0], dict)
            assert isinstance(kwargs['optional_params'], dict)
            assert isinstance(kwargs['litellm_params'], dict) 
            assert isinstance(kwargs['start_time'], (datetime, type(None)))
            assert isinstance(kwargs['stream'], bool)
            assert isinstance(kwargs['user'], (str, type(None)))
            assert (isinstance(kwargs['input'], list) and isinstance(kwargs['input'][0], dict)) or isinstance(kwargs['input'], (dict, str))
            assert isinstance(kwargs['api_key'], (str, type(None)))
            assert isinstance(kwargs['original_response'], (str, litellm.CustomStreamWrapper)) or inspect.isasyncgen(kwargs['original_response']) or inspect.iscoroutine(kwargs['original_response'])
            assert isinstance(kwargs['additional_args'], (dict, type(None)))
            assert isinstance(kwargs['log_event_type'], str) 
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
            assert isinstance(kwargs['model'], str)
            assert isinstance(kwargs['messages'], list) and isinstance(kwargs['messages'][0], dict)
            assert isinstance(kwargs['optional_params'], dict)
            assert isinstance(kwargs['litellm_params'], dict) 
            assert isinstance(kwargs['start_time'], (datetime, type(None)))
            assert isinstance(kwargs['stream'], bool)
            assert isinstance(kwargs['user'], (str, type(None)))
            assert (isinstance(kwargs['input'], list) and isinstance(kwargs['input'][0], dict)) or isinstance(kwargs['input'], (dict, str))
            assert isinstance(kwargs['api_key'], (str, type(None)))
            assert isinstance(kwargs['original_response'], (str, litellm.CustomStreamWrapper))
            assert isinstance(kwargs['additional_args'], (dict, type(None)))
            assert isinstance(kwargs['log_event_type'], str) 
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
            assert isinstance(kwargs['model'], str)
            assert isinstance(kwargs['messages'], list) and isinstance(kwargs['messages'][0], dict)
            assert isinstance(kwargs['optional_params'], dict)
            assert isinstance(kwargs['litellm_params'], dict) 
            assert isinstance(kwargs['start_time'], (datetime, type(None)))
            assert isinstance(kwargs['stream'], bool)
            assert isinstance(kwargs['user'], (str, type(None)))
            assert (isinstance(kwargs['input'], list) and isinstance(kwargs['input'][0], dict)) or isinstance(kwargs['input'], (dict, str))
            assert isinstance(kwargs['api_key'], (str, type(None)))
            assert isinstance(kwargs['original_response'], (str, litellm.CustomStreamWrapper)) or kwargs["original_response"] == None
            assert isinstance(kwargs['additional_args'], (dict, type(None)))
            assert isinstance(kwargs['log_event_type'], str) 
        except: 
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
            assert isinstance(kwargs['model'], str)
            assert isinstance(kwargs['messages'], list) and isinstance(kwargs['messages'][0], dict)
            assert isinstance(kwargs['optional_params'], dict)
            assert isinstance(kwargs['litellm_params'], dict) 
            assert isinstance(kwargs['start_time'], (datetime, type(None)))
            assert isinstance(kwargs['stream'], bool)
            assert isinstance(kwargs['user'], (str, type(None)))
        except Exception as e: 
            print(f"Assertion Error: {traceback.format_exc()}")
            self.errors.append(traceback.format_exc())

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        try: 
            self.states.append("async_success")
            ## START TIME 
            assert isinstance(start_time, datetime)
            ## END TIME 
            assert isinstance(end_time, datetime)
            ## RESPONSE OBJECT 
            assert isinstance(response_obj, (litellm.ModelResponse, litellm.EmbeddingResponse))
            ## KWARGS
            assert isinstance(kwargs['model'], str)
            assert isinstance(kwargs['messages'], list)
            assert isinstance(kwargs['optional_params'], dict)
            assert isinstance(kwargs['litellm_params'], dict) 
            assert isinstance(kwargs['start_time'], (datetime, type(None)))
            assert isinstance(kwargs['stream'], bool)
            assert isinstance(kwargs['user'], (str, type(None)))
            assert isinstance(kwargs['input'], (list, dict, str))
            assert isinstance(kwargs['api_key'], (str, type(None)))
            assert isinstance(kwargs['original_response'], (str, litellm.CustomStreamWrapper)) or inspect.isasyncgen(kwargs['original_response']) or inspect.iscoroutine(kwargs['original_response'])
            assert isinstance(kwargs['additional_args'], (dict, type(None)))
            assert isinstance(kwargs['log_event_type'], str) 

        except: 
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
            assert isinstance(kwargs['model'], str)
            assert isinstance(kwargs['messages'], list)
            assert isinstance(kwargs['optional_params'], dict)
            assert isinstance(kwargs['litellm_params'], dict) 
            assert isinstance(kwargs['start_time'], (datetime, type(None)))
            assert isinstance(kwargs['stream'], bool)
            assert isinstance(kwargs['user'], (str, type(None)))
            assert isinstance(kwargs['input'], (list, str, dict))
            assert isinstance(kwargs['api_key'], (str, type(None)))
            assert isinstance(kwargs['original_response'], (str, litellm.CustomStreamWrapper)) or inspect.isasyncgen(kwargs['original_response']) or kwargs['original_response'] == None
            assert isinstance(kwargs['additional_args'], (dict, type(None)))
            assert isinstance(kwargs['log_event_type'], str) 
        except: 
            print(f"Assertion Error: {traceback.format_exc()}")
            self.errors.append(traceback.format_exc())


# COMPLETION
## Test OpenAI + sync
def test_chat_openai_stream():
    try: 
        customHandler = CompletionCustomHandler()
        litellm.callbacks = [customHandler]
        response = litellm.completion(model="gpt-3.5-turbo",
                                messages=[{
                                    "role": "user",
                                    "content": "Hi 👋 - i'm sync openai"
                                }])
        ## test streaming
        response = litellm.completion(model="gpt-3.5-turbo",
                                messages=[{
                                    "role": "user",
                                    "content": "Hi 👋 - i'm openai"
                                }],
                                stream=True)
        for chunk in response: 
            continue
        ## test failure callback
        try: 
            response = litellm.completion(model="gpt-3.5-turbo",
                                    messages=[{
                                        "role": "user",
                                        "content": "Hi 👋 - i'm openai"
                                    }],
                                    api_key="my-bad-key",
                                    stream=True)
            for chunk in response: 
                continue
        except:
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
        response = await litellm.acompletion(model="gpt-3.5-turbo",
                                messages=[{
                                    "role": "user",
                                    "content": "Hi 👋 - i'm openai"
                                }])
        ## test streaming
        response = await litellm.acompletion(model="gpt-3.5-turbo",
                                messages=[{
                                    "role": "user",
                                    "content": "Hi 👋 - i'm openai"
                                }],
                                stream=True)
        async for chunk in response: 
            continue
        ## test failure callback
        try: 
            response = await litellm.acompletion(model="gpt-3.5-turbo",
                                    messages=[{
                                        "role": "user",
                                        "content": "Hi 👋 - i'm openai"
                                    }],
                                    api_key="my-bad-key",
                                    stream=True)
            async for chunk in response: 
                continue
        except:
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
        response = litellm.completion(model="azure/chatgpt-v-2",
                                messages=[{
                                    "role": "user",
                                    "content": "Hi 👋 - i'm sync azure"
                                }])
        # test streaming
        response = litellm.completion(model="azure/chatgpt-v-2",
                                messages=[{
                                    "role": "user",
                                    "content": "Hi 👋 - i'm sync azure"
                                }],
                                stream=True)
        for chunk in response: 
            continue
        # test failure callback
        try: 
            response = litellm.completion(model="azure/chatgpt-v-2",
                                    messages=[{
                                        "role": "user",
                                        "content": "Hi 👋 - i'm sync azure"
                                    }],
                                    api_key="my-bad-key",
                                    stream=True)
            for chunk in response: 
                continue
        except:
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
        response = await litellm.acompletion(model="azure/chatgpt-v-2",
                                messages=[{
                                    "role": "user",
                                    "content": "Hi 👋 - i'm async azure"
                                }])
        ## test streaming
        response = await litellm.acompletion(model="azure/chatgpt-v-2",
                                messages=[{
                                    "role": "user",
                                    "content": "Hi 👋 - i'm async azure"
                                }],
                                stream=True)
        async for chunk in response: 
            continue
        ## test failure callback
        try: 
            response = await litellm.acompletion(model="azure/chatgpt-v-2",
                                    messages=[{
                                        "role": "user",
                                        "content": "Hi 👋 - i'm async azure"
                                    }],
                                    api_key="my-bad-key",
                                    stream=True)
            async for chunk in response: 
                continue
        except:
            pass
        await asyncio.sleep(1)
        print(f"customHandler.errors: {customHandler.errors}")
        assert len(customHandler.errors) == 0
        litellm.callbacks = []
    except Exception as e: 
        pytest.fail(f"An exception occurred: {str(e)}")

# asyncio.run(test_async_chat_azure_stream())

## Test Bedrock + sync
def test_chat_bedrock_stream():
    try: 
        customHandler = CompletionCustomHandler()
        litellm.callbacks = [customHandler]
        response = litellm.completion(model="bedrock/anthropic.claude-v1",
                                messages=[{
                                    "role": "user",
                                    "content": "Hi 👋 - i'm sync bedrock"
                                }])
        # test streaming
        response = litellm.completion(model="bedrock/anthropic.claude-v1",
                                messages=[{
                                    "role": "user",
                                    "content": "Hi 👋 - i'm sync bedrock"
                                }],
                                stream=True)
        for chunk in response: 
            continue
        # test failure callback
        try: 
            response = litellm.completion(model="bedrock/anthropic.claude-v1",
                                    messages=[{
                                        "role": "user",
                                        "content": "Hi 👋 - i'm sync bedrock"
                                    }],
                                    aws_region_name="my-bad-region",
                                    stream=True)
            for chunk in response: 
                continue
        except:
            pass
        time.sleep(1)
        print(f"customHandler.errors: {customHandler.errors}")
        assert len(customHandler.errors) == 0
        litellm.callbacks = []
    except Exception as e: 
        pytest.fail(f"An exception occurred: {str(e)}")

# test_chat_bedrock_stream()

## Test Bedrock + Async
@pytest.mark.asyncio
async def test_async_chat_bedrock_stream():
    try: 
        customHandler = CompletionCustomHandler()
        litellm.callbacks = [customHandler]
        response = await litellm.acompletion(model="bedrock/anthropic.claude-v1",
                                messages=[{
                                    "role": "user",
                                    "content": "Hi 👋 - i'm async bedrock"
                                }])
        # test streaming
        response = await litellm.acompletion(model="bedrock/anthropic.claude-v1",
                                messages=[{
                                    "role": "user",
                                    "content": "Hi 👋 - i'm async bedrock"
                                }],
                                stream=True)
        print(f"response: {response}")
        async for chunk in response: 
            print(f"chunk: {chunk}")
            continue
        ## test failure callback
        try: 
            response = await litellm.acompletion(model="bedrock/anthropic.claude-v1",
                                    messages=[{
                                        "role": "user",
                                        "content": "Hi 👋 - i'm async bedrock"
                                    }],
                                    aws_region_name="my-bad-key",
                                    stream=True)
            async for chunk in response: 
                continue
        except:
            pass
        time.sleep(1)
        print(f"customHandler.errors: {customHandler.errors}")
        assert len(customHandler.errors) == 0
        litellm.callbacks = []
    except Exception as e: 
        pytest.fail(f"An exception occurred: {str(e)}")

# asyncio.run(test_async_chat_bedrock_stream())

# EMBEDDING
## Test OpenAI + Async
@pytest.mark.asyncio
async def test_async_embedding_openai():
    try: 
        customHandler_success = CompletionCustomHandler()
        customHandler_failure = CompletionCustomHandler()
        litellm.callbacks = [customHandler_success]
        response = await litellm.aembedding(model="azure/azure-embedding-model",
                                input=["good morning from litellm"])
        await asyncio.sleep(1)
        print(f"customHandler_success.errors: {customHandler_success.errors}")
        print(f"customHandler_success.states: {customHandler_success.states}")
        assert len(customHandler_success.errors) == 0
        assert len(customHandler_success.states) == 3 # pre, post, success
        # test failure callback
        litellm.callbacks = [customHandler_failure]
        try: 
            response = await litellm.aembedding(model="text-embedding-ada-002",
                                    input=["good morning from litellm"],
                                    api_key="my-bad-key")
        except:
            pass
        await asyncio.sleep(1)
        print(f"customHandler_failure.errors: {customHandler_failure.errors}")
        print(f"customHandler_failure.states: {customHandler_failure.states}")
        assert len(customHandler_failure.errors) == 0
        assert len(customHandler_failure.states) == 3 # pre, post, failure
    except Exception as e: 
        pytest.fail(f"An exception occurred: {str(e)}")

# asyncio.run(test_async_embedding_openai())

## Test Azure + Async
@pytest.mark.asyncio
async def test_async_embedding_azure():
    try: 
        customHandler_success = CompletionCustomHandler()
        customHandler_failure = CompletionCustomHandler()
        litellm.callbacks = [customHandler_success]
        response = await litellm.aembedding(model="azure/azure-embedding-model",
                                input=["good morning from litellm"])
        await asyncio.sleep(1)
        print(f"customHandler_success.errors: {customHandler_success.errors}")
        print(f"customHandler_success.states: {customHandler_success.states}")
        assert len(customHandler_success.errors) == 0
        assert len(customHandler_success.states) == 3 # pre, post, success
        # test failure callback
        litellm.callbacks = [customHandler_failure]
        try: 
            response = await litellm.aembedding(model="azure/azure-embedding-model",
                                    input=["good morning from litellm"],
                                    api_key="my-bad-key")
        except:
            pass
        await asyncio.sleep(1)
        print(f"customHandler_failure.errors: {customHandler_failure.errors}")
        print(f"customHandler_failure.states: {customHandler_failure.states}")
        assert len(customHandler_failure.errors) == 0
        assert len(customHandler_failure.states) == 3 # pre, post, success
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
        response = await litellm.aembedding(model="bedrock/cohere.embed-multilingual-v3",
                                input=["good morning from litellm"], aws_region_name="os.environ/AWS_REGION_NAME_2")
        await asyncio.sleep(1)
        print(f"customHandler_success.errors: {customHandler_success.errors}")
        print(f"customHandler_success.states: {customHandler_success.states}")
        assert len(customHandler_success.errors) == 0
        assert len(customHandler_success.states) == 3 # pre, post, success
        # test failure callback
        litellm.callbacks = [customHandler_failure]
        try: 
            response = await litellm.aembedding(model="bedrock/cohere.embed-multilingual-v3",
                                    input=["good morning from litellm"],
                                    aws_region_name="my-bad-region")
        except:
            pass
        await asyncio.sleep(1)
        print(f"customHandler_failure.errors: {customHandler_failure.errors}")
        print(f"customHandler_failure.states: {customHandler_failure.states}")
        assert len(customHandler_failure.errors) == 0
        assert len(customHandler_failure.states) == 3 # pre, post, success
    except Exception as e: 
        pytest.fail(f"An exception occurred: {str(e)}")

# asyncio.run(test_async_embedding_bedrock())