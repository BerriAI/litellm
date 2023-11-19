#### What this tests ####
#    This tests the the acompletion function #

import sys, os
import pytest
import traceback
import asyncio, logging

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import litellm
from litellm import completion, acompletion, acreate
litellm.num_retries = 3

def test_sync_response():
    litellm.set_verbose = False
    user_message = "Hello, how are you?"
    messages = [{"content": user_message, "role": "user"}]
    try:
        response = completion(model="gpt-3.5-turbo", messages=messages, timeout=5)
        print(f"response: {response}")
    except litellm.Timeout as e: 
        pass
    except Exception as e:
        pytest.fail(f"An exception occurred: {e}")
# test_sync_response()

def test_sync_response_anyscale():
    litellm.set_verbose = False
    user_message = "Hello, how are you?"
    messages = [{"content": user_message, "role": "user"}]
    try:
        response = completion(model="anyscale/mistralai/Mistral-7B-Instruct-v0.1", messages=messages, timeout=5)
    except litellm.Timeout as e: 
        pass
    except Exception as e:
        pytest.fail(f"An exception occurred: {e}")
# test_sync_response_anyscale()

def test_async_response_openai():
    import asyncio
    litellm.set_verbose = True
    async def test_get_response():
        user_message = "Hello, how are you?"
        messages = [{"content": user_message, "role": "user"}]
        try:
            response = await acompletion(model="gpt-3.5-turbo", messages=messages, timeout=5)
            print(f"response: {response}")
        except litellm.Timeout as e: 
            pass
        except Exception as e:
            pytest.fail(f"An exception occurred: {e}")
            print(e)

    asyncio.run(test_get_response())

# test_async_response_openai()

def test_async_response_azure():
    import asyncio
    litellm.set_verbose = True
    async def test_get_response():
        user_message = "What do you know?"
        messages = [{"content": user_message, "role": "user"}]
        try:
            response = await acompletion(model="azure/chatgpt-v-2", messages=messages, timeout=5)
            print(f"response: {response}")
        except litellm.Timeout as e: 
            pass
        except Exception as e:
            pytest.fail(f"An exception occurred: {e}")

    asyncio.run(test_get_response())

def test_async_anyscale_response():
    import asyncio
    litellm.set_verbose = True
    async def test_get_response():
        user_message = "Hello, how are you?"
        messages = [{"content": user_message, "role": "user"}]
        try:
            response = await acompletion(model="anyscale/mistralai/Mistral-7B-Instruct-v0.1", messages=messages, timeout=5)
            # response = await response
            print(f"response: {response}")
        except litellm.Timeout as e: 
            pass
        except Exception as e:
            pytest.fail(f"An exception occurred: {e}")

    asyncio.run(test_get_response())

# test_async_anyscale_response()

def test_get_response_streaming():
    import asyncio
    async def test_async_call():
        user_message = "write a short poem in one sentence"
        messages = [{"content": user_message, "role": "user"}]
        try:
            litellm.set_verbose = True
            response = await acompletion(model="gpt-3.5-turbo", messages=messages, stream=True, timeout=5)
            print(type(response))

            import inspect

            is_async_generator = inspect.isasyncgen(response)
            print(is_async_generator)

            output = ""
            i = 0
            async for chunk in response:
                token = chunk["choices"][0]["delta"].get("content", "")
                if token == None:
                    continue # openai v1.0.0 returns content=None
                output += token
            assert output is not None, "output cannot be None."
            assert isinstance(output, str), "output needs to be of type str"
            assert len(output) > 0, "Length of output needs to be greater than 0."
            print(f'output: {output}')
        except litellm.Timeout as e: 
            pass
        except Exception as e:
            pytest.fail(f"An exception occurred: {e}")
    asyncio.run(test_async_call())

# test_get_response_streaming()

def test_get_response_non_openai_streaming():
    import asyncio
    litellm.set_verbose = True
    async def test_async_call():
        user_message = "Hello, how are you?"
        messages = [{"content": user_message, "role": "user"}]
        try:
            response = await acompletion(model="anyscale/mistralai/Mistral-7B-Instruct-v0.1", messages=messages, stream=True, timeout=5)
            print(type(response))

            import inspect

            is_async_generator = inspect.isasyncgen(response)
            print(is_async_generator)

            output = ""
            i = 0
            async for chunk in response:
                token = chunk["choices"][0]["delta"].get("content", None)
                if token == None:
                    continue
                print(token)
                output += token
            print(f"output: {output}")
            assert output is not None, "output cannot be None."
            assert isinstance(output, str), "output needs to be of type str"
            assert len(output) > 0, "Length of output needs to be greater than 0."
        except litellm.Timeout as e: 
            pass
        except Exception as e:
            pytest.fail(f"An exception occurred: {e}")
        return response
    asyncio.run(test_async_call())

test_get_response_non_openai_streaming()