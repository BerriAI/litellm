#### What this tests ####
#    This tests streaming for the completion endpoint

import sys, os
import traceback
sys.path.insert(0, os.path.abspath('../..'))  # Adds the parent directory to the system path
import litellm
from litellm import completion

litellm.set_verbose = False

score = 0

def logger_fn(model_call_object: dict):
    print(f"model call details: {model_call_object}")

user_message = "Hello, how are you?"
messages = [{ "content": user_message,"role": "user"}]

# test on anthropic completion call 
try:
    response = completion(model="claude-instant-1", messages=messages, stream=True, logger_fn=logger_fn)
    for chunk in response:
        print(chunk['choices'][0]['delta'])
    score +=1 
except:
    print(f"error occurred: {traceback.format_exc()}") 
    pass