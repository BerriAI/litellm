import sys
import os
import io

sys.path.insert(0, os.path.abspath('../..'))

from litellm import completion
import litellm

litellm.success_callback = ["wandb"]

litellm.set_verbose = True
import time



def test_wandb_logging():
    try:
        response = completion(model="claude-instant-1.2",
                              messages=[{
                                  "role": "user",
                                  "content": "Hi 👋 - i'm claude"
                              }],
                              max_tokens=10,
                              temperature=0.2
                              )
        print(response)
    except Exception as e:
        print(e)

test_wandb_logging()

# def test_langfuse_logging_custom_generation_name():
#     try:
#         response = completion(model="gpt-3.5-turbo",
#                               messages=[{
#                                   "role": "user",
#                                   "content": "Hi 👋 - i'm claude"
#                               }],
#                               max_tokens=10,
#                               metadata = {
#                                 "generation_name": "litellm-ishaan-gen", # set langfuse generation name
#                                 # custom metadata fields
#                                 "project": "litellm-proxy" 
#                               }
#         )
#         print(response)
#     except Exception as e:
#         print(e)

# test_langfuse_logging_custom_generation_name()




