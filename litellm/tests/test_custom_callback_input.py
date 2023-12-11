### What this tests ####
## This test asserts the type of data passed into each method of the custom callback handler
import sys, os, time, inspect, asyncio, traceback
from datetime import datetime
import pytest
sys.path.insert(0, os.path.abspath('../..'))
from typing import Optional
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
## 2. router.completion() + router.embeddings() 
## 3. proxy.completions + proxy.embeddings 

class CompletionCustomHandler(CustomLogger): # https://docs.litellm.ai/docs/observability/custom_callback#callback-class
    """
    The set of expected inputs to a custom handler for a 
    """
    # Class variables or attributes
    def __init__(self):
        self.errors = []

    def log_pre_api_call(self, model, messages, kwargs): 
        try: 
            ## MODEL
            assert isinstance(model, str)
            ## MESSAGES
            assert isinstance(messages, list) and isinstance(messages[0], dict)
            ## KWARGS
            assert isinstance(kwargs['model'], str)
            assert isinstance(kwargs['messages'], list) and isinstance(kwargs['messages'][0], dict)
            assert isinstance(kwargs['optional_params'], dict)
            assert isinstance(kwargs['litellm_params'], dict) 
            assert isinstance(kwargs['start_time'], Optional[datetime])
            assert isinstance(kwargs['stream'], bool)
            assert isinstance(kwargs['user'], Optional[str])
        except Exception as e: 
            print(f"Assertion Error: {traceback.format_exc()}")
            self.errors.append(traceback.format_exc())

    def log_post_api_call(self, kwargs, response_obj, start_time, end_time): 
        try:
            ## START TIME 
            assert isinstance(start_time, datetime)
            ## END TIME 
            assert end_time == None
            ## RESPONSE OBJECT 
            assert response_obj == None
            ## KWARGS 
            assert isinstance(kwargs['model'], str)
            assert isinstance(kwargs['messages'], list) and isinstance(kwargs['messages'][0], dict)
            assert isinstance(kwargs['optional_params'], dict)
            assert isinstance(kwargs['litellm_params'], dict) 
            assert isinstance(kwargs['start_time'], Optional[datetime])
            assert isinstance(kwargs['stream'], bool)
            assert isinstance(kwargs['user'], Optional[str])
            assert isinstance(kwargs['input'], list) and isinstance(kwargs['input'][0], dict)
            assert isinstance(kwargs['api_key'], str)
            assert isinstance(kwargs['original_response'], (str, litellm.CustomStreamWrapper)) or inspect.iscoroutine(kwargs['original_response']) or inspect.isasyncgen(kwargs['original_response'])
            assert isinstance(kwargs['additional_args'], Optional[dict])
            assert isinstance(kwargs['log_event_type'], str) 
        except: 
            print(f"Assertion Error: {traceback.format_exc()}")
            self.errors.append(traceback.format_exc())
    
    async def async_log_stream_event(self, kwargs, response_obj, start_time, end_time):
        try:
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
            assert isinstance(kwargs['start_time'], Optional[datetime])
            assert isinstance(kwargs['stream'], bool)
            assert isinstance(kwargs['user'], Optional[str])
            assert isinstance(kwargs['input'], list) and isinstance(kwargs['input'][0], dict)
            assert isinstance(kwargs['api_key'], str)
            assert inspect.isasyncgen(kwargs['original_response']) or inspect.iscoroutine(kwargs['original_response'])
            assert isinstance(kwargs['additional_args'], Optional[dict])
            assert isinstance(kwargs['log_event_type'], str) 
        except: 
            print(f"Assertion Error: {traceback.format_exc()}")
            self.errors.append(traceback.format_exc())

    def log_success_event(self, kwargs, response_obj, start_time, end_time): 
        try:
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
            assert isinstance(kwargs['start_time'], Optional[datetime])
            assert isinstance(kwargs['stream'], bool)
            assert isinstance(kwargs['user'], Optional[str])
            assert isinstance(kwargs['input'], list) and isinstance(kwargs['input'][0], dict)
            assert isinstance(kwargs['api_key'], str)
            assert isinstance(kwargs['original_response'], (str, litellm.CustomStreamWrapper))
            assert isinstance(kwargs['additional_args'], Optional[dict])
            assert isinstance(kwargs['log_event_type'], str) 
        except:
            print(f"Assertion Error: {traceback.format_exc()}")
            self.errors.append(traceback.format_exc())

    def log_failure_event(self, kwargs, response_obj, start_time, end_time): 
        try:
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
            assert isinstance(kwargs['start_time'], Optional[datetime])
            assert isinstance(kwargs['stream'], bool)
            assert isinstance(kwargs['user'], Optional[str])
            assert isinstance(kwargs['input'], list) and isinstance(kwargs['input'][0], dict)
            assert isinstance(kwargs['api_key'], str)
            assert isinstance(kwargs['original_response'], (str, litellm.CustomStreamWrapper)) or kwargs["original_response"] == None
            assert isinstance(kwargs['additional_args'], Optional[dict])
            assert isinstance(kwargs['log_event_type'], str) 
        except: 
            print(f"Assertion Error: {traceback.format_exc()}")
            self.errors.append(traceback.format_exc())
    
    async def async_log_pre_api_call(self, model, messages, kwargs):
        try: 
            ## MODEL
            assert isinstance(model, str)
            ## MESSAGES
            assert isinstance(messages, list) and isinstance(messages[0], dict)
            ## KWARGS
            assert isinstance(kwargs['model'], str)
            assert isinstance(kwargs['messages'], list) and isinstance(kwargs['messages'][0], dict)
            assert isinstance(kwargs['optional_params'], dict)
            assert isinstance(kwargs['litellm_params'], dict) 
            assert isinstance(kwargs['start_time'], Optional[datetime])
            assert isinstance(kwargs['stream'], bool)
            assert isinstance(kwargs['user'], Optional[str])
        except Exception as e: 
            print(f"Assertion Error: {traceback.format_exc()}")
            self.errors.append(traceback.format_exc())

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        try: 
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
            assert isinstance(kwargs['start_time'], Optional[datetime])
            assert isinstance(kwargs['stream'], bool)
            assert isinstance(kwargs['user'], Optional[str])
            assert isinstance(kwargs['input'], list) and isinstance(kwargs['input'][0], dict)
            assert isinstance(kwargs['api_key'], str)
            assert isinstance(kwargs['original_response'], str) or inspect.isasyncgen(kwargs['original_response']) or inspect.iscoroutine(kwargs['original_response'])
            assert isinstance(kwargs['additional_args'], Optional[dict])
            assert isinstance(kwargs['log_event_type'], str) 
        except: 
            print(f"Assertion Error: {traceback.format_exc()}")
            self.errors.append(traceback.format_exc())

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        try:
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
            assert isinstance(kwargs['start_time'], Optional[datetime])
            assert isinstance(kwargs['stream'], bool)
            assert isinstance(kwargs['user'], Optional[str])
            assert isinstance(kwargs['input'], list) and isinstance(kwargs['input'][0], dict)
            assert isinstance(kwargs['api_key'], str)
            assert isinstance(kwargs['original_response'], (str, litellm.CustomStreamWrapper)) or inspect.isasyncgen(kwargs['original_response'])
            assert isinstance(kwargs['additional_args'], Optional[dict])
            assert isinstance(kwargs['log_event_type'], str) 
        except: 
            print(f"Assertion Error: {traceback.format_exc()}")
            self.errors.append(traceback.format_exc())

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

## Test OpenAI + Async
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
        time.sleep(1)
        print(f"customHandler.errors: {customHandler.errors}")
        assert len(customHandler.errors) == 0
        litellm.callbacks = []
    except Exception as e: 
        pytest.fail(f"An exception occurred: {str(e)}")

# asyncio.run(test_async_chat_azure_stream())