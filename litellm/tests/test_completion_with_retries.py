import sys, os
import traceback
from dotenv import load_dotenv

load_dotenv()
import os

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import pytest
import openai
import litellm
from litellm import completion_with_retries, completion
from litellm import (
    AuthenticationError,
    BadRequestError,
    RateLimitError,
    ServiceUnavailableError,
    OpenAIError,
)

user_message = "Hello, whats the weather in San Francisco??"
messages = [{"content": user_message, "role": "user"}]


def logger_fn(user_model_dict):
    # print(f"user_model_dict: {user_model_dict}")
    pass

# normal call
def test_completion_custom_provider_model_name():
    try:
        response = completion_with_retries(
            model="together_ai/togethercomputer/llama-2-70b-chat",
            messages=messages,
            logger_fn=logger_fn,
        )
        # Add any assertions here to check the response
        print(response)
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")

# completion with num retries + impact on exception mapping
def test_completion_with_num_retries(): 
    try: 
        response = completion(model="j2-ultra", messages=[{"messages": "vibe", "bad": "message"}], num_retries=2)
        pytest.fail(f"Unmapped exception occurred")
    except Exception as e: 
        pass

# test_completion_with_num_retries()