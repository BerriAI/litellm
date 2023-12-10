### What this tests ####
import sys, os, time, inspect, asyncio
import pytest
sys.path.insert(0, os.path.abspath('../..'))

from litellm import completion, embedding
import litellm
from litellm.integrations.custom_logger import CustomLogger
 
async_success = False
complete_streaming_response_in_callback = ""
class MyCustomHandler(CustomLogger):
    def __init__(self):
        self.success: bool = False                  # type: ignore
        self.failure: bool = False                  # type: ignore
        self.async_success: bool = False            # type: ignore
        self.async_success_embedding: bool = False  # type: ignore
        self.async_failure: bool = False            # type: ignore
        self.async_failure_embedding: bool = False  # type: ignore

        self.async_completion_kwargs = None         # type: ignore
        self.async_embedding_kwargs = None          # type: ignore
        self.async_embedding_response = None        # type: ignore

        self.async_completion_kwargs_fail = None    # type: ignore
        self.async_embedding_kwargs_fail = None     # type: ignore

        self.stream_collected_response = None       # type: ignore
        self.sync_stream_collected_response = None       # type: ignore

    def log_pre_api_call(self, model, messages, kwargs): 
        print(f"Pre-API Call")
    
    def log_post_api_call(self, kwargs, response_obj, start_time, end_time): 
        print(f"Post-API Call")
    
    def log_stream_event(self, kwargs, response_obj, start_time, end_time):
        print(f"On Stream")
        
    def log_success_event(self, kwargs, response_obj, start_time, end_time): 
        print(f"On Success")
        self.success = True
        if kwargs.get("stream") == True:
            self.sync_stream_collected_response = response_obj


    def log_failure_event(self, kwargs, response_obj, start_time, end_time): 
        print(f"On Failure")
        self.failure = True

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time): 
        print(f"On Async success")
        self.async_success = True
        print("Value of async success: ", self.async_success)
        print("\n kwargs: ", kwargs)
        if kwargs.get("model") == "text-embedding-ada-002":
            self.async_success_embedding = True
            self.async_embedding_kwargs = kwargs
            self.async_embedding_response = response_obj
        if kwargs.get("stream") == True:
            self.stream_collected_response = response_obj
        self.async_completion_kwargs = kwargs
    
    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time): 
        print(f"On Async Failure")
        self.async_failure = True
        print("Value of async failure: ", self.async_failure)
        print("\n kwargs: ", kwargs)
        if kwargs.get("model") == "text-embedding-ada-002":
            self.async_failure_embedding = True
            self.async_embedding_kwargs_fail = kwargs
        
        self.async_completion_kwargs_fail = kwargs

async def async_test_logging_fn(kwargs, completion_obj, start_time, end_time):
    global async_success, complete_streaming_response_in_callback
    print(f"ON ASYNC LOGGING")
    async_success = True
    print("\nKWARGS", kwargs)
    complete_streaming_response_in_callback = kwargs.get("complete_streaming_response")


def test_async_chat_openai_stream():
    try:
        global complete_streaming_response_in_callback
        # litellm.set_verbose = True
        litellm.success_callback = [async_test_logging_fn]
        complete_streaming_response = ""
        async def call_gpt():
            nonlocal complete_streaming_response
            response = await litellm.acompletion(model="gpt-3.5-turbo",
                                messages=[{
                                    "role": "user",
                                    "content": "Hi 👋 - i'm openai"
                                }],
                                stream=True)
            async for chunk in response: 
                complete_streaming_response += chunk["choices"][0]["delta"]["content"] or ""
                print(complete_streaming_response)
        asyncio.run(call_gpt())
        assert complete_streaming_response_in_callback["choices"][0]["message"]["content"] == complete_streaming_response
        assert async_success == True
    except Exception as e:
        print(e)
        pytest.fail(f"An error occurred - {str(e)}")
test_async_chat_openai_stream()

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
                model="azure/chatgpt-v-2", 
                messages=messages,
                stream=True
            )
            async for chunk in response: 
                complete_streaming_response += chunk["choices"][0]["delta"]["content"] or ""
                print(complete_streaming_response)
        
        asyncio.run(test_1())

        response_in_success_handler = customHandler2.stream_collected_response
        response_in_success_handler = response_in_success_handler["choices"][0]["message"]["content"]
        print("\n\n")
        print("response_in_success_handler: ", response_in_success_handler)
        print("complete_streaming_response: ", complete_streaming_response)
        assert response_in_success_handler == complete_streaming_response
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")
# test_async_custom_handler_stream()


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
                "content": "write 1 sentence about litellm being amazing",
            },
        ]
        complete_streaming_response = ""

        response = litellm.completion(
            model="azure/chatgpt-v-2", 
            messages=messages,
            stream=True
        )
        for chunk in response: 
            complete_streaming_response += chunk["choices"][0]["delta"]["content"] or ""
            print(complete_streaming_response)
        
        time.sleep(0.5) # wait 1/2 second before checking callbacks
        response_in_success_handler = customHandler2.sync_stream_collected_response
        response_in_success_handler = response_in_success_handler["choices"][0]["message"]["content"]
        print("\n\n")
        print("response_in_success_handler: ", response_in_success_handler)
        print("complete_streaming_response: ", complete_streaming_response)
        assert response_in_success_handler == complete_streaming_response
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")
test_azure_completion_stream()

def test_async_custom_handler():
    try:
        customHandler2 = MyCustomHandler()
        litellm.callbacks = [customHandler2]
        litellm.set_verbose = True
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
                    model="gpt-3.5-turbo", 
                    messages=messages,
                    api_key="test",
                )
            except:
                pass

        assert customHandler2.async_failure == False 
        asyncio.run(test_1())
        assert customHandler2.async_failure == True, "async failure is not set to True even after failure"        
        assert customHandler2.async_completion_kwargs_fail.get("model") == "gpt-3.5-turbo"
        assert len(str(customHandler2.async_completion_kwargs_fail.get("exception"))) > 10 # exppect APIError("OpenAIException - Error code: 401 - {'error': {'message': 'Incorrect API key provided: test. You can find your API key at https://platform.openai.com/account/api-keys.', 'type': 'invalid_request_error', 'param': None, 'code': 'invalid_api_key'}}"), 'traceback_exception': 'Traceback (most recent call last):\n  File "/Users/ishaanjaffer/Github/litellm/litellm/llms/openai.py", line 269, in acompletion\n    response = await openai_aclient.chat.completions.create(**data)\n  File "/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages/openai/resources/chat/completions.py", line 119
        print("Passed setting async failure")

        async def test_2():
            response = await litellm.acompletion(
                model="gpt-3.5-turbo", 
                messages=[{
                    "role": "user",
                    "content": "hello from litellm test",
                }]
            )
            print("\n response", response)
        assert customHandler2.async_success == False
        asyncio.run(test_2())
        assert customHandler2.async_success == True, "async success is not set to True even after success"
        assert customHandler2.async_completion_kwargs.get("model") == "gpt-3.5-turbo"


        async def test_3():
            response = await litellm.aembedding(
                model="text-embedding-ada-002", 
                input = ["hello world"],
            )
            print("\n response", response)
        assert customHandler2.async_success_embedding == False
        asyncio.run(test_3())
        assert customHandler2.async_success_embedding == True, "async_success_embedding is not set to True even after success"
        assert customHandler2.async_embedding_kwargs.get("model") == "text-embedding-ada-002"
        assert customHandler2.async_embedding_response["usage"]["prompt_tokens"] ==2
        print("Passed setting async success: Embedding")


        print("Testing custom failure callback for embedding")

        async def test_4():
            try:
                response = await litellm.aembedding(
                    model="text-embedding-ada-002", 
                    input = ["hello world"],
                    api_key="test",
                )
            except:
                pass

        assert customHandler2.async_failure_embedding == False 
        asyncio.run(test_4())
        assert customHandler2.async_failure_embedding == True, "async failure embedding is not set to True even after failure"        
        assert customHandler2.async_embedding_kwargs_fail.get("model") == "text-embedding-ada-002"
        assert len(str(customHandler2.async_embedding_kwargs_fail.get("exception"))) > 10 # exppect APIError("OpenAIException - Error code: 401 - {'error': {'message': 'Incorrect API key provided: test. You can find your API key at https://platform.openai.com/account/api-keys.', 'type': 'invalid_request_error', 'param': None, 'code': 'invalid_api_key'}}"), 'traceback_exception': 'Traceback (most recent call last):\n  File "/Users/ishaanjaffer/Github/litellm/litellm/llms/openai.py", line 269, in acompletion\n    response = await openai_aclient.chat.completions.create(**data)\n  File "/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages/openai/resources/chat/completions.py", line 119
        print("Passed setting async failure")

    except Exception as e:
        pytest.fail(f"Error occurred: {e}")
# test_async_custom_handler()

from litellm import Cache
def test_redis_cache_completion_stream():
    # Important Test - This tests if we can add to streaming cache, when custom callbacks are set 
    import random
    try:
        print("\nrunning test_redis_cache_completion_stream")
        litellm.set_verbose = True
        random_number = random.randint(1, 100000) # add a random number to ensure it's always adding / reading from cache
        messages = [{"role": "user", "content": f"write a one sentence poem about: {random_number}"}]
        litellm.cache = Cache(type="redis", host=os.environ['REDIS_HOST'], port=os.environ['REDIS_PORT'], password=os.environ['REDIS_PASSWORD'])
        print("test for caching, streaming + completion")
        response1 = completion(model="gpt-3.5-turbo", messages=messages, max_tokens=40, temperature=0.2, stream=True)
        response_1_content = ""
        for chunk in response1:
            print(chunk)
            response_1_content += chunk.choices[0].delta.content or ""
        print(response_1_content)

        time.sleep(0.1) # sleep for 0.1 seconds allow set cache to occur
        response2 = completion(model="gpt-3.5-turbo", messages=messages, max_tokens=40, temperature=0.2, stream=True)
        response_2_content = ""
        for chunk in response2:
            print(chunk)
            response_2_content += chunk.choices[0].delta.content or ""
        print("\nresponse 1", response_1_content)
        print("\nresponse 2", response_2_content)
        assert response_1_content == response_2_content, f"Response 1 != Response 2. Same params, Response 1{response_1_content} != Response 2{response_2_content}"
        litellm.success_callback = []
        litellm.cache = None
    except Exception as e:
        print(e)
        litellm.success_callback = []
        raise e
    """

    1 & 2 should be exactly the same 
    """
# test_redis_cache_completion_stream()