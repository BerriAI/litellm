#### What this tests ####
#    This tests the ability to set api key's via the params instead of as environment variables

import sys, os
import traceback
sys.path.insert(0, os.path.abspath('../..'))  # Adds the parent directory to the system path
import litellm
from litellm import embedding, completion

litellm.set_verbose = False

def logger_fn(model_call_object: dict):
    print(f"model call details: {model_call_object}")

user_message = "Hello, how are you?"
messages = [{ "content": user_message,"role": "user"}]

print(os.environ)
temp_key = os.environ.get("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = "bad-key"
# test on openai completion call 
try:
    response = completion(model="gpt-3.5-turbo", messages=messages, logger_fn=logger_fn, api_key=temp_key)
    print(f"response: {response}")
except:
    print(f"error occurred: {traceback.format_exc()}") 
    pass
os.environ["OPENAI_API_KEY"] = temp_key
