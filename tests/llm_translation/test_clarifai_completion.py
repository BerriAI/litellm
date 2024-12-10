import sys, os
import traceback
from dotenv import load_dotenv
import asyncio, logging

load_dotenv()
import os, io

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import pytest
import litellm
from litellm import (
    embedding,
    completion,
    acompletion,
    acreate,
    completion_cost,
    Timeout,
    ModelResponse,
)
from litellm import RateLimitError

# litellm.num_retries = 3
litellm.cache = None
litellm.success_callback = []
user_message = "Write a short poem about the sky"
messages = [{"content": user_message, "role": "user"}]


def test_completion_clarifai_claude_2_1():
    print("calling clarifai claude completion")
    litellm.set_verbose = True
    import os

    clarifai_pat = os.environ["CLARIFAI_API_KEY"]

    try:
        response = completion(
            model="clarifai/openai.chat-completion.GPT-4",
            num_retries=3,
            messages=messages,
            max_tokens=10,
            temperature=0.1,
        )
        print(response)

    except RateLimitError:
        pass

    except Exception as e:
        pytest.fail(f"Error occured: {e}")


def test_completion_clarifai_mistral_large():
    try:
        litellm.set_verbose = True
        response = completion(
            model="clarifai/mistralai.completion.mistral-small",
            messages=messages,
            num_retries=3,
            max_tokens=10,
            temperature=0.78,
        )
        # Add any assertions here to check the response
        assert len(response.choices) > 0
        assert len(response.choices[0].message.content) > 0
    except RateLimitError:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


@pytest.mark.asyncio
async def test_async_completion_clarifai():
    litellm.set_verbose = True
    user_message = "Hello, how are you?"
    messages = [{"content": user_message, "role": "user"}]
    response = await acompletion(
        model="clarifai/openai.chat-completion.GPT-4",
        messages=messages,
        num_retries=3,
        timeout=10,
        api_key=os.getenv("CLARIFAI_API_KEY"),
    )
    print(response)
