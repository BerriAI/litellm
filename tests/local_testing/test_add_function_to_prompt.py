#### What this tests ####
#  Allow the user to map the function to the prompt, if the model doesn't support function calling

import sys, os, pytest
import traceback

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import litellm


## case 1: set_function_to_prompt not set
def test_function_call_non_openai_model():
    try:
        model = "claude-instant-1"
        messages = [{"role": "user", "content": "what's the weather in sf?"}]
        functions = [
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
        response = litellm.completion(
            model=model, messages=messages, functions=functions
        )
        pytest.fail(f"An error occurred")
    except Exception as e:
        print(e)
        pass


# test_function_call_non_openai_model()


## case 2: add_function_to_prompt set
@pytest.mark.skip(reason="Anthropic now supports tool calling")
def test_function_call_non_openai_model_litellm_mod_set():
    litellm.add_function_to_prompt = True
    litellm.set_verbose = True
    try:
        model = "claude-instant-1.2"
        messages = [{"role": "user", "content": "what's the weather in sf?"}]
        functions = [
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
        response = litellm.completion(
            model=model, messages=messages, functions=functions
        )
        print(f"response: {response}")
    except Exception as e:
        pytest.fail(f"An error occurred {e}")


# test_function_call_non_openai_model_litellm_mod_set()
