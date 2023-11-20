import sys
import os
import io, asyncio
# import logging
# logging.basicConfig(level=logging.DEBUG)
sys.path.insert(0, os.path.abspath('../..'))

from litellm import completion
import litellm
litellm.num_retries = 3
litellm.success_callback = ["langfuse"]
# litellm.set_verbose = True
import time
import pytest

def test_langfuse_logging_async(): 
    try: 
        litellm.set_verbose = True
        async def _test_langfuse():
            return await litellm.acompletion(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content":"This is a test"}],
                max_tokens=1000,
                temperature=0.7,
                timeout=5
            )
        response = asyncio.run(_test_langfuse())
        print(f"response: {response}")
    except litellm.Timeout as e: 
        pass
    except Exception as e: 
        pytest.fail(f"An exception occurred - {e}")

# test_langfuse_logging_async()

def test_langfuse_logging():
    try:
        response = completion(model="claude-instant-1.2",
                              messages=[{
                                  "role": "user",
                                  "content": "Hi 👋 - i'm claude"
                              }],
                              max_tokens=10,
                              temperature=0.2
                              )
        print(response)
    except litellm.Timeout as e: 
        pass
    except Exception as e:
        print(e)

# test_langfuse_logging()


def test_langfuse_logging_stream():
    try:
        litellm.set_verbose=True
        response = completion(model="anyscale/meta-llama/Llama-2-7b-chat-hf",
                              messages=[{
                                  "role": "user",
                                  "content": "this is a streaming test for llama2 + langfuse"
                              }],
                              max_tokens=20,
                              temperature=0.2,
                              stream=True
                              )
        print(response)
        for chunk in response:
            pass
            # print(chunk)
    except litellm.Timeout as e: 
        pass
    except Exception as e:
        print(e)

# test_langfuse_logging_stream()

def test_langfuse_logging_custom_generation_name():
    try:
        response = completion(model="gpt-3.5-turbo",
                              messages=[{
                                  "role": "user",
                                  "content": "Hi 👋 - i'm claude"
                              }],
                              max_tokens=10,
                              metadata = {
                                "generation_name": "litellm-ishaan-gen", # set langfuse generation name
                                # custom metadata fields
                                "project": "litellm-proxy" 
                              }
        )
        print(response)
    except litellm.Timeout as e: 
        pass
    except Exception as e:
        print(e)

# test_langfuse_logging_custom_generation_name()

def test_langfuse_logging_function_calling():
    function1 = [
        {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        }
    ]
    try:
        response = completion(model="gpt-3.5-turbo",
                              messages=[{
                                  "role": "user",
                                  "content": "what's the weather in boston"
                              }],
                              temperature=0.1,
                              functions=function1,
            )
        print(response)
    except litellm.Timeout as e: 
        pass
    except Exception as e:
        print(e)

# test_langfuse_logging_function_calling()



