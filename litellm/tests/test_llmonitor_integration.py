#### What this tests ####
#    This tests if logging to the llmonitor integration actually works
# Adds the parent directory to the system path
import sys
import os

sys.path.insert(0, os.path.abspath('../..'))

from litellm import completion
import litellm

litellm.success_callback = ["llmonitor"]
litellm.failure_callback = ["llmonitor"]

litellm.set_verbose = True

# openai call
first_success_test = completion(model="gpt-3.5-turbo",
                                messages=[{
                                    "role": "user",
                                    "content": "Hi 👋 - i'm openai"
                                }])

print(first_success_test)
