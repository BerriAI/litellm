#### What this tests ####
#    This tests the the acompletion function

import sys, os
import pytest
import traceback
import asyncio

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
from litellm import acompletion, acreate


async def test_get_response():
    user_message = "Hello, how are you?"
    messages = [{"content": user_message, "role": "user"}]
    try:
        response = await acompletion(model="gpt-3.5-turbo", messages=messages)
    except Exception as e:
        pytest.fail(f"error occurred: {e}")
    return response


response = asyncio.run(test_get_response())
print(response)

# async def test_get_response():
#     user_message = "Hello, how are you?"
#     messages = [{"content": user_message, "role": "user"}]
#     try:
#         response = await acreate(model="gpt-3.5-turbo", messages=messages)
#     except Exception as e:
#         pytest.fail(f"error occurred: {e}")
#     return response


# response = asyncio.run(test_get_response())
# print(response)
