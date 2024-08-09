import os
import litellm

# METHOD 1: Using keywordsai as the LLM proxy
litellm.api_base = "https://api.keywordsai.co/api/"
KEYWORDS_AI_API_KEY = os.getenv("KEYWORDS_AI_API_KEY")

response = litellm.completion(
    api_key=KEYWORDS_AI_API_KEY, # !!!!!!! Use the keyowrdsai api key in your completion call !!!!!!!
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "How does a court case get to the Supreme Court?"}]
)

print(response)
# Go to https://platform.keywordsai.co/ to see the log

# ================================================================================================================
# METHOD 2: Use keywordsai callback:



import litellm
from litellm import completion, acompletion
from litellm.integrations.keywordsai import KeywordsAILogger
litellm.set_verbose = True
litellm.api_base = None
litellm.callbacks = [KeywordsAILogger()]
extra_body = {
    "keywordsai_params": {
        "customer_identifier": "test_litellm_logging",
        "thread_identifier": "test_litellm_thread",
        "metadata": {"key": "value"},
    }
}
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
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    }
]
tool_choice = {"type": "function", "function": {"name": "get_current_weather"}}
messages = [{"role": "user", "content": "Hi 👋 - i'm openai"}]
tool_messages = [
    {"role": "user", "content": "Get the current weather in San Francisco, CA"}
]
model = "gpt-3.5-turbo"
# sync
print(f"==================== Sync ====================")
## No tools
### non-streaming
print(f"==================== No Tools ====================")
response = completion(model=model, messages=messages, extra_body=extra_body)
## streaming
response = completion(model=model, messages=messages,extra_body=extra_body, stream=True)
for chunk in response:
    continue
# ## With tools
print(f"==================== With Tools ====================")
### non-streaming
response = completion(
    model=model,
    messages=tool_messages,
    tools=tools,
    tool_choice=tool_choice,
    extra_body=extra_body
)
### streaming
response = completion(
    model=model,
    messages=tool_messages,
    tools=tools,
    tool_choice=tool_choice,
    extra_body=extra_body,
    stream=True
)
for chunk in response:
    continue


# ## async
import asyncio


async def completion(stream=False, tools=None, tool_choice=None):
    response = await acompletion(
        model=model, messages=messages, extra_body=extra_body, stream=stream, tools=tools, tool_choice=tool_choice
    )
    if stream:
        async for chunk in response:
            continue
print(f"==================== Async ====================")
## No tools
print(f"==================== No Tools ====================")
# asyncio.run(completion())
asyncio.run(completion(stream=True))
# ## With tools
# print(f"==================== With Tools ====================")
asyncio.run(completion(tools=tools, tool_choice=tool_choice))
asyncio.run(completion(stream=True, tools=tools, tool_choice=tool_choice))

