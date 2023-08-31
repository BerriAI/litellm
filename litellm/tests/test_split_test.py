#### What this tests ####
#    This tests the 'completion_with_split_tests' function to enable a/b testing between llm models

import sys, os
import traceback

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import litellm
from litellm import completion_with_split_tests
litellm.set_verbose = True
split_per_model = {
	"gpt-4": 0.7, 
	"claude-instant-1.2": 0.3
}

messages = [{ "content": "Hello, how are you?","role": "user"}]

# print(completion_with_split_tests(models=split_per_model, messages=messages))

# test with client 

print(completion_with_split_tests(models=split_per_model, messages=messages, use_client=True, id=1234))