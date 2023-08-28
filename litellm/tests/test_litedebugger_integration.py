#### What this tests ####
#    This tests if logging to the litedebugger integration actually works
# pytest mistakes intentional bad calls as failed tests -> [TODO] fix this
import sys, os
import traceback
import pytest

sys.path.insert(0, os.path.abspath('../..'))  # Adds the parent directory to the system path
import litellm
from litellm import embedding, completion

litellm.set_verbose = True

litellm.use_client = True

user_message = "Hello, how are you?"
messages = [{ "content": user_message,"role": "user"}]


# Test 1: On completion call
response = completion(model="claude-instant-1", messages=[{"role": "user", "content": "Hi 👋 - i'm openai"}])
# print(f"response: {response}")

# # Test 2: On embedding call
response = embedding(model="text-embedding-ada-002", input=["sample text"])
# print(f"response: {response}")

# # Test 3: On streaming completion call
response = completion(model="replicate/llama-2-70b-chat:58d078176e02c219e11eb4da5a02a7830a283b14cf8f94537af893ccff5ee781", messages=[{"role": "user", "content": "Hi 👋 - i'm openai"}], stream=True)
print(f"response: {response}")