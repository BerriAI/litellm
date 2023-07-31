import sys, os
import traceback
sys.path.append('..')  # Adds the parent directory to the system path
import main
from main import embedding, completion
main.success_callback = ["posthog"]
main.failure_callback = ["slack", "sentry", "posthog"]

main.set_verbose = True

user_message = "Hello, how are you?"
messages = [{ "content": user_message,"role": "user"}]
model_val = None
# test on empty
try:
    response = completion(model=model_val, messages=messages)
except:
    print(f"error occurred: {traceback.format_exc()}") 
    pass
