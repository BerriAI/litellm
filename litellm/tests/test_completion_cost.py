
import sys, os
import traceback
from dotenv import load_dotenv

load_dotenv()
import os

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import pytest
import litellm
from litellm import embedding, completion, text_completion
from litellm.utils import completion_cost


user_message = "Write a short poem about the sky"
messages = [{"content": user_message, "role": "user"}]


def test_completion_togetherai_cost():
    try:
        response = completion(
            model="together_ai/togethercomputer/llama-2-70b-chat",
            messages=messages,
            request_timeout=200,
        )
        # Add any assertions here to check the response
        print(response)
        print("Completion Cost: for togethercomputer/llama-2-70b-chat")
        cost = completion_cost(completion_response=response)
        formatted_string = f"${float(cost):.10f}"
        print(formatted_string)
        
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")
test_completion_togetherai_cost()