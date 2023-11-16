#### What this tests ####
#    This tests the the acompletion function #

import sys, os
import pytest
import traceback
import asyncio, logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import litellm
from litellm import completion, acompletion, acreate
litellm.num_retries = 3

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
    litellm.set_verbose = True
    async def test_get_response():
        user_message = "Hello, how are you?"
        messages = [{"content": user_message, "role": "user"}]
        try:
            response = await acompletion(model="huggingface/HuggingFaceH4/zephyr-7b-beta", messages=messages)
            print(f"response: {response}")
        except Exception as e:
            pytest.fail(f"An exception occurred: {e}")

    asyncio.run(test_get_response())
# test_async_response()

def test_get_response_streaming():
    import asyncio
    async def test_async_call():
        user_message = "write a short poem in one sentence"
        messages = [{"content": user_message, "role": "user"}]
        try:
            litellm.set_verbose = True
            response = await acompletion(model="gpt-3.5-turbo", messages=messages, stream=True)
            print(type(response))

            import inspect

            is_async_generator = inspect.isasyncgen(response)
            print(is_async_generator)

            output = ""
            i = 0
            async for chunk in response:
                token = chunk["choices"][0]["delta"].get("content", "")
                output += token
            assert output is not None, "output cannot be None."
            assert isinstance(output, str), "output needs to be of type str"
            assert len(output) > 0, "Length of output needs to be greater than 0."
            print(f'output: {output}')
        except Exception as e:
            pytest.fail(f"An exception occurred: {e}")
        return response
    asyncio.run(test_async_call())


# test_get_response_streaming()

def test_get_response_non_openai_streaming():
    import asyncio
    litellm.set_verbose = True
    async def test_async_call():
        user_message = "Hello, how are you?"
        messages = [{"content": user_message, "role": "user"}]
        try:
            response = await acompletion(model="huggingface/HuggingFaceH4/zephyr-7b-beta", messages=messages, stream=True)
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
