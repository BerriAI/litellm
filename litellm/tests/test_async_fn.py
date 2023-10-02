#### What this tests ####
#    This tests the the acompletion function #

import sys, os
import pytest
import traceback
import asyncio

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
from litellm import acompletion, acreate

@pytest.mark.asyncio
async def test_get_response():
    user_message = "Hello, how are you?"
    messages = [{"content": user_message, "role": "user"}]
    try:
        response = await acompletion(model="gpt-3.5-turbo", messages=messages)
    except Exception as e:
        pass

response = asyncio.run(test_get_response())
# print(response)

@pytest.mark.asyncio
async def test_get_response_streaming():
    user_message = "Hello, how are you?"
    messages = [{"content": user_message, "role": "user"}]
    try:
        response = await acompletion(model="gpt-3.5-turbo", messages=messages, stream=True)
        print(type(response))

        import inspect

        is_async_generator = inspect.isasyncgen(response)
        print(is_async_generator)

        output = ""
        async for chunk in response:
            token = chunk["choices"][0]["delta"].get("content", "")
            output += token
            print(output)

        assert output is not None, "output cannot be None."
        assert isinstance(output, str), "output needs to be of type str"
        assert len(output) > 0, "Length of output needs to be greater than 0."

    except Exception as e:
        pass
    return response

# response = asyncio.run(test_get_response_streaming())
# print(response)


