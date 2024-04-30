import sys, os
import traceback
from dotenv import load_dotenv

load_dotenv()
import os, io

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import pytest
import litellm
from litellm import embedding, completion, completion_cost, Timeout, ModelResponse
from litellm import RateLimitError

# litellm.num_retries = 3
litellm.cache = None
litellm.success_callback = []
user_message = "Write a short poem about the sky"
messages = [{"content": user_message, "role": "user"}]

@pytest.fixture(autouse=True)
def reset_callbacks():
    print("\npytest fixture - resetting callbacks")
    litellm.success_callback = []
    litellm._async_success_callback = []
    litellm.failure_callback = []
    litellm.callbacks = []
    
def test_completion_clarifai_claude_2_1():
    print("calling clarifai claude completion")
    import os 
    
    clarifai_pat = os.environ["CLARIFAI_API_KEY"]
    
    try:
        response =  completion(
            model="clarifai/anthropic.completion.claude-2_1",
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
        response: ModelResponse = completion(
            model="clarifai/mistralai.completion.mistral-small",
            messages=messages,
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
