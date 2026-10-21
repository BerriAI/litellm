#### What this tests ####
#    This tests the the acompletion function #

import asyncio

import pytest

import litellm
from litellm import acompletion

litellm.num_retries = 3




# test_sync_response_anyscale()


def test_async_response_openai():

    litellm.set_verbose = True

    async def test_get_response():
        user_message = "Hello, how are you?"
        messages = [{"content": user_message, "role": "user"}]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_current_weather",
                    "description": "Get the current weather in a given location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g. San Francisco, CA",
                            },
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                            },
                        },
                        "required": ["location"],
                    },
                },
            }
        ]
        try:
            response = await acompletion(
                model="gpt-3.5-turbo",
                messages=messages,
                tools=tools,
                parallel_tool_calls=True,
                timeout=5,
            )
            print(f"response: {response}")
            print(f"response ms: {response._response_ms}")
        except litellm.Timeout as e:
            pass
        except Exception as e:
            pytest.fail(f"An exception occurred: {e}")
            print(e)

    asyncio.run(test_get_response())


# test_async_response_openai()




# test_async_anyscale_response()




# test_async_completion_cloudflare()






# test_get_cloudflare_response_streaming()


# test_get_response_streaming()




# test_get_response_non_openai_streaming()
