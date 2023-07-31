import sys, os
import traceback
sys.path.append('..')  # Adds the parent directory to the system path
import main
from main import litellm_client
client = litellm_client(success_callback=["posthog"], failure_callback=["slack", "sentry", "posthog"], verbose=True)
completion = client.completion
embedding = client.embedding

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
