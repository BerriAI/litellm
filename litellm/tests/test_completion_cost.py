
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
# test_completion_togetherai_cost()


def test_completion_replicate_llama_2():
    model_name = "replicate/llama-2-70b-chat:2796ee9483c3fd7aa2e171d38f4ca12251a30609463dcfd4cd76703f22e96cdf"
    try:
        response = completion(
            model=model_name, 
            messages=messages, 
            max_tokens=20,
            custom_llm_provider="replicate"
        )
        print(response)
        # Add any assertions here to check the response
        response_str = response["choices"][0]["message"]["content"]
        print(response_str)

        # Add any assertions here to check the response
        print(response)
        print("Completion Cost: for togethercomputer/llama-2-70b-chat")
        cost = completion_cost(completion_response=response)
        formatted_string = f"${float(cost):.10f}"
        print(formatted_string)
        
        if type(response_str) != str:
            pytest.fail(f"Error occurred: {e}")
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")

# v1
# test_completion_replicate_llama_2()