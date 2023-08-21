#### What this tests ####
#    This tests if logging to the llmonitor integration actually works
# Adds the parent directory to the system path
import sys
import os

sys.path.insert(0, os.path.abspath('../..'))

from litellm import completion
import litellm

litellm.input_callback = ["llmonitor"]
litellm.success_callback = ["llmonitor"]
litellm.error_callback = ["llmonitor"]

litellm.set_verbose = True

# openai call
# response = completion(model="gpt-3.5-turbo",
#                       messages=[{
#                           "role": "user",
#                           "content": "Hi 👋 - i'm openai"
#                       }])

# print(response)

# #bad request call
# response = completion(model="chatgpt-test", messages=[{"role": "user", "content": "Hi 👋 - i'm a bad request"}])

# cohere call
response = completion(model="command-nightly",
                      messages=[{
                          "role": "user",
                          "content": "Hi 👋 - i'm cohere"
                      }])
print(response)
