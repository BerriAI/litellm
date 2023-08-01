import sys, os
import traceback

# Get the current directory of the script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory by joining the current directory with '..'
parent_dir = os.path.join(current_dir, '..')

# Add the parent directory to the system path
sys.path.append(parent_dir)

import main
from main import embedding, completion
main.success_callback = ["posthog"]
main.failure_callback = ["slack", "sentry", "posthog"]

main.set_verbose = True

user_message = "Hello, how are you?"
messages = [{ "content": user_message,"role": "user"}]
model_val = "krrish is a model"
# test on empty
try:
    response = completion(model=model_val, messages=messages)
except:
    print(f"error occurred: {traceback.format_exc()}") 
    pass
