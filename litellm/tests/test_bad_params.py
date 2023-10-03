#### What this tests ####
#    This tests chaos monkeys - if random parts of the system are broken / things aren't sent correctly - what happens.
#    Expect to add more edge cases to this over time.

import sys, os
import traceback
import pytest
sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import litellm
from litellm import embedding, completion


# litellm.set_verbose = True
user_message = "Hello, how are you?"
messages = [{"content": user_message, "role": "user"}]
model_val = None

def test_completion_with_no_model():
    # test on empty
    with pytest.raises(ValueError):
        response = completion(messages=messages)


def test_completion_with_empty_model():
    # test on empty
    try:
        response = completion(model=model_val, messages=messages)
    except Exception as e:
        print(f"error occurred: {e}")
        pass

def test_completion_return_full_text_hf():
    try: 
        response = completion(model="dolphin", messages=messages, remove_input=True)
        # check if input in response 
        assert "Hello, how are you?" not in response["choices"][0]["message"]["content"]
    except Exception as e: 
        if "Function calling is not supported by this provider" in str(e): 
            pass
        else: 
            pytest.fail(f'An error occurred {e}')

# test_completion_return_full_text_hf() 

def test_completion_invalid_param_cohere():
    try: 
        response = completion(model="command-nightly", messages=messages, top_p=1)
        print(f"response: {response}")
    except Exception as e: 
        if "Unsupported parameters passed: top_p" in str(e): 
            pass
        else: 
            pytest.fail(f'An error occurred {e}')

test_completion_invalid_param_cohere()

def test_completion_function_call_cohere():
    try: 
        response = completion(model="command-nightly", messages=messages, function_call="TEST-FUNCTION")
    except Exception as e: 
        if "Function calling is not supported by this provider" in str(e): 
            pass
        else: 
            pytest.fail(f'An error occurred {e}')

def test_completion_function_call_openai(): 
    try: 
        messages = [{"role": "user", "content": "What is the weather like in Boston?"}]
        response = completion(model="gpt-3.5-turbo", messages=messages, functions=[
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
        ])
        print(f"response: {response}")
    except: 
        pass

# test_completion_function_call_openai() 

def test_completion_with_no_provider():
    # test on empty
    try:
        model = "cerebras/btlm-3b-8k-base"
        response = completion(model=model, messages=messages)
    except Exception as e:
        print(f"error occurred: {e}")
        pass

# test_completion_with_no_provider()
# # bad key
# temp_key = os.environ.get("OPENAI_API_KEY")
# os.environ["OPENAI_API_KEY"] = "bad-key"
# # test on openai completion call
# try:
#     response = completion(model="gpt-3.5-turbo", messages=messages)
#     print(f"response: {response}")
# except:
#     print(f"error occurred: {traceback.format_exc()}")
#     pass
# os.environ["OPENAI_API_KEY"] = str(temp_key)  # this passes linting#5

