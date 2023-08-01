import sys, os
import traceback
from dotenv import load_dotenv
load_dotenv()
import os
sys.path.insert(0, os.path.abspath('../..'))  # Adds the parent directory to the system path
import pytest
import litellm
from litellm import embedding, completion

litellm.set_verbose = True

user_message = "Hello, whats the weather in San Francisco??"
messages = [{ "content": user_message,"role": "user"}]

def test_completion_openai():
    try:
        response = completion(model="gpt-3.5-turbo", messages=messages)
        # Add any assertions here to check the response
        print(response)
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")

def test_completion_openai_with_optional_params():
    try:
        response = completion(model="gpt-3.5-turbo", messages=messages, temperature=0.5, top_p=0.1, user="ishaan_dev@berri.ai")
        # Add any assertions here to check the response
        print(response)
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")

def test_completion_openai_with_more_optional_params():
    try:
        response = completion(model="gpt-3.5-turbo", messages=messages, temperature=0.5, top_p=0.1, n=2, max_tokens=150, presence_penalty=0.5, frequency_penalty=-0.5, logit_bias={123: 5}, user="ishaan_dev@berri.ai")
        # Add any assertions here to check the response
        print(response)
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")

def test_completion_openai_with_stream():
    try:
        response = completion(model="gpt-3.5-turbo", messages=messages, temperature=0.5, top_p=0.1, n=2, max_tokens=150, presence_penalty=0.5, stream=True, frequency_penalty=-0.5, logit_bias={27000: 5}, user="ishaan_dev@berri.ai")
        # Add any assertions here to check the response
        print(response)
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")

def test_completion_openai_with_functions():
    function1 = [
        {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"]
                    }
                },
                "required": ["location"]
            }
        }
    ]
    try:
        response = completion(model="gpt-3.5-turbo", messages=messages, functions=function1)
        # Add any assertions here to check the response
        print(response)
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")

def test_completion_azure():
    try:
        response = completion(model="chatgpt-test", messages=messages, azure=True)
        # Add any assertions here to check the response
        print(response)
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")

def test_completion_claude():
    try:
        response = completion(model="claude-instant-1", messages=messages)
        # Add any assertions here to check the response
        print(response)
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")

def test_completion_cohere():
    try:
        response = completion(model="command-nightly", messages=messages, max_tokens=500)
        # Add any assertions here to check the response
        print(response)
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# def test_completion_replicate_llama():
#     model_name = "replicate/llama-2-70b-chat:2c1608e18606fad2812020dc541930f2d0495ce32eee50074220b87300bc16e1"
#     try:
#         response = completion(model=model_name, messages=messages, max_tokens=500)
#         # Add any assertions here to check the response
#         print(response)
#     except Exception as e:
#         print(f"in replicate llama, got error {e}")
#         pass
#         if e == "FunctionTimedOut":
#             pass
#         else:
#             pytest.fail(f"Error occurred: {e}")
