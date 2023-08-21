#### What this tests ####
#    This tests the ability to set api key's via the params instead of as environment variables

import sys, os
import traceback

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import litellm
from litellm import embedding, completion

litellm.set_verbose = False


def logger_fn(model_call_object: dict):
    print(f"model call details: {model_call_object}")


user_message = "Hello, how are you?"
messages = [{"content": user_message, "role": "user"}]

## Test 1: Setting key dynamically
temp_key = os.environ.get("ANTHROPIC_API_KEY", "")
os.environ["ANTHROPIC_API_KEY"] = "bad-key"
# test on openai completion call
try:
    response = completion(
        model="claude-instant-1",
        messages=messages,
        logger_fn=logger_fn,
        api_key=temp_key,
    )
    print(f"response: {response}")
except:
    print(f"error occurred: {traceback.format_exc()}")
    pass
os.environ["ANTHROPIC_API_KEY"] = temp_key


## Test 2: Setting key via __init__ params
litellm.anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
os.environ.pop("ANTHROPIC_API_KEY")
# test on openai completion call
try:
    response = completion(
        model="claude-instant-1", messages=messages, logger_fn=logger_fn
    )
    print(f"response: {response}")
except:
    print(f"error occurred: {traceback.format_exc()}")
    pass
os.environ["ANTHROPIC_API_KEY"] = temp_key
