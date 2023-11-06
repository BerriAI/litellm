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
        messages=[{"role": "user", "content": "what's the weather in sf?"}]
        functions = [
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
        response = litellm.completion(model=model, messages=messages, functions=functions)
        pytest.fail(f'An error occurred')
    except Exception as e: 
        print(e)
        pass


## case 2: add_function_to_prompt set 
def test_function_call_non_openai_model_litellm_mod_set():
    litellm.add_function_to_prompt = True
    try: 
        # model = "claude-instant-1"
        model = "claude-2"
        messages=[{"role": "user", "content": "what's the weather in sf?"}]
        functions = [
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
        response = litellm.completion(model=model, messages=messages, functions=functions)
        print(f'response: {response}')
    except Exception as e: 
        pytest.fail(f'An error occurred {e}')

## case 3: compared with gpt for the function call result
def test_function_call_with_gpt_model():
    try: 
        model = "gpt-3.5-turbo"
        messages=[{"role": "user", "content": "what's the weather in sf?"}]
        functions = [
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
        response = litellm.completion(model=model, messages=messages, functions=functions)
        print(f'response: {response}')
    except Exception as e: 
        pytest.fail(f'An error occurred {e}')

test_function_call_non_openai_model()
# test_function_call_non_openai_model_litellm_mod_set()
# test_function_call_with_gpt_model()

"""
# the function call response of gpt:
"choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": null,
        "function_call": {
          "name": "get_current_weather",
          "arguments": "{\n  \"location\": \"San Francisco, CA\"\n}"
        }
      },
      "finish_reason": "function_call"
    }
  ],

# the response of claude-instant-1
"choices": [
    {
      "finish_reason": "stop_sequence",
      "index": 0,
      "message": {
        "content": "You asked about the weather in San Francisco. Let me check the current weather conditions.",
        "role": "assistant",
        "function_call": {
          "name": "get_current_weather",
          "arguments": "{\"location\": \"sf\", \"unit\": \"fahrenheit\"}"
        }
      }
    }
  ],


# the response of claude-2
"choices": [
    {
      "finish_reason": "stop_sequence",
      "index": 0,
      "message": {
        "content": "To get the current weather in San Francisco, I will invoke the get_current_weather function.",
        "role": "assistant",
        "function_call": {
          "name": "get_current_weather",
          "arguments": "{\"location\": \"sf\", \"unit\": \"fahrenheit\"}"
        }
      }
    }
  ],

"""