#### What this tests ####
#    This tests the the acompletion function #

import sys, os
import pytest
import traceback
import asyncio

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import litellm
from litellm import completion, acompletion, acreate

def test_sync_response():
    litellm.set_verbose = True
    user_message = "Hello, how are you?"
    messages = [{"content": user_message, "role": "user"}]
    try:
        response = completion(model="gpt-3.5-turbo", messages=messages, api_key=os.environ["OPENAI_API_KEY"])
    except Exception as e:
        pytest.fail(f"An exception occurred: {e}")


def test_async_response():
    import asyncio
    async def test_get_response():
        user_message = "Hello, how are you?"
        messages = [{"content": user_message, "role": "user"}]
        try:
            response = await acompletion(model="gpt-3.5-turbo", messages=messages)
            print(f"response: {response}")
            response = await acompletion(model="azure/chatgpt-v-2", messages=messages)
            print(f"response: {response}")
        except Exception as e:
            pytest.fail(f"An exception occurred: {e}")

    response = asyncio.run(test_get_response())
# print(response)
# test_async_response()

def test_get_response_streaming():
    import asyncio
    async def test_async_call():
        user_message = "Hello, how are you?"
        messages = [{"content": user_message, "role": "user"}]
        try:
            response = await acompletion(model="azure/chatgpt-v-2", messages=messages, stream=True)
            # response = await acompletion(model="gpt-3.5-turbo", messages=messages, stream=True)
            print(type(response))

            import inspect

            is_async_generator = inspect.isasyncgen(response)
            print(is_async_generator)

            output = ""
            i = 0
            async for chunk in response:
                token = chunk["choices"][0]["delta"].get("content", "")
                output += token
            print(f"output: {output}")
            assert output is not None, "output cannot be None."
            assert isinstance(output, str), "output needs to be of type str"
            assert len(output) > 0, "Length of output needs to be greater than 0."

        except Exception as e:
            pytest.fail(f"An exception occurred: {e}")
        return response
    asyncio.run(test_async_call())


test_get_response_streaming()

def test_get_response_non_openai_streaming():
    import asyncio
    async def test_async_call():
        user_message = "Hello, how are you?"
        messages = [{"content": user_message, "role": "user"}]
        try:
            response = await acompletion(model="command-nightly", messages=messages, stream=True)
            print(type(response))

            import inspect

            is_async_generator = inspect.isasyncgen(response)
            print(is_async_generator)

            output = ""
            i = 0
            async for chunk in response:
                token = chunk["choices"][0]["delta"].get("content", "")
                output += token
            print(f"output: {output}")
            assert output is not None, "output cannot be None."
            assert isinstance(output, str), "output needs to be of type str"
            assert len(output) > 0, "Length of output needs to be greater than 0."

        except Exception as e:
            pytest.fail(f"An exception occurred: {e}")
        return response
    asyncio.run(test_async_call())

# test_get_response_non_openai_streaming()
