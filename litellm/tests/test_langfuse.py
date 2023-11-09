import sys
import os
import io, asyncio
import logging
logging.basicConfig(level=logging.DEBUG)
sys.path.insert(0, os.path.abspath('../..'))

from litellm import completion
import litellm

litellm.success_callback = ["langfuse"]
# litellm.set_verbose = True
import time

def test_langfuse_logging_async(): 
    async def _test_langfuse():
        return await litellm.acompletion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content":"This is a test"}],
            max_tokens=1000,
            temperature=0.7,
        )
    response = asyncio.run(_test_langfuse())
    print(f"response: {response}")

test_langfuse_logging_async()

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
    except Exception as e:
        print(e)

# test_langfuse_logging()

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
    except Exception as e:
        print(e)

# test_langfuse_logging_function_calling()



