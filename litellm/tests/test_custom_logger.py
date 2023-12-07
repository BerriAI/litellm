### What this tests ####
import sys, os, time, inspect, asyncio
import pytest
sys.path.insert(0, os.path.abspath('../..'))

from litellm import completion, embedding
import litellm
from litellm.integrations.custom_logger import CustomLogger

async_success = False
class MyCustomHandler(CustomLogger):
    def __init__(self):
        self.success: bool = False
        self.failure: bool = False
        self.async_success: bool = False
        self.async_failure: bool = False

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

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time): 
        print(f"On Async success")
        self.async_success = True
        print("Value of async success: ", self.async_success)
    
    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time): 
        print(f"On Async Failure")
        self.async_failure = True
        print("Value of async failure: ", self.async_failure)


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
# test_chat_openai()

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


def test_async_custom_handler():
    try:
        customHandler2 = MyCustomHandler()
        litellm.callbacks = [customHandler2]
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": "how do i kill someone",
            },
        ]
        async def test_1():
            try:
                response = await litellm.acompletion(
                    model="gpt-3.5", 
                    messages=messages
                )
            except:
                pass

        assert customHandler2.async_failure == False 
        asyncio.run(test_1())
        assert customHandler2.async_failure == True, "async failure is not set to True even after failure"
        print("Passed setting async failure")

        async def test_2():
            response = await litellm.acompletion(
                model="gpt-3.5-turbo", 
                messages=messages
            )
            print("\n response", response)
        assert customHandler2.async_success == False
        asyncio.run(test_2())
        assert customHandler2.async_success == True, "async success is not set to True even after success"
        print("Passed setting async success")
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")
# test_async_custom_handler()