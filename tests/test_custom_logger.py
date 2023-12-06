### What this tests ####
import sys, os, time, inspect, asyncio
import pytest
sys.path.insert(0, os.path.abspath('../..'))

from litellm import completion, embedding
import litellm
from litellm.integrations.custom_logger import CustomLogger

async_success = False
class MyCustomHandler(CustomLogger):
    success: bool = False
    failure: bool = False

    def log_pre_api_call(self, model, messages, kwargs): 
        print(f"Pre-API Call")
    
    def log_post_api_call(self, kwargs, response_obj, start_time, end_time): 
        print(f"Post-API Call")
    
    def log_stream_event(self, kwargs, response_obj, start_time, end_time):
        print(f"On Stream")
        
    def log_success_event(self, kwargs, response_obj, start_time, end_time): 
        print(f"On Success")
        self.success = True

    def log_failure_event(self, kwargs, response_obj, start_time, end_time): 
        print(f"On Failure")
        self.failure = True


async def async_test_logging_fn(kwargs, completion_obj, start_time, end_time):
    global async_success
    print(f"ON ASYNC LOGGING")
    async_success = True

@pytest.mark.asyncio
async def test_chat_openai():
    try:
        # litellm.set_verbose = True
        litellm.success_callback = [async_test_logging_fn]
        response = await litellm.acompletion(model="gpt-3.5-turbo",
                              messages=[{
                                  "role": "user",
                                  "content": "Hi 👋 - i'm openai"
                              }],
                              stream=True)
        async for chunk in response: 
            continue
        assert async_success == True
    except Exception as e:
        print(e)
        pytest.fail(f"An error occurred - {str(e)}")

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
                model="azure/chatgpt-v-2", messages=messages, stream=True
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
