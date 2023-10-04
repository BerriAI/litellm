import sys
import os
import io

sys.path.insert(0, os.path.abspath('../..'))

from litellm import completion
import litellm

litellm.success_callback = ["langfuse"]
# litellm.set_verbose = True
import time



def test_langfuse_logging():
    try:
        response = completion(model="claude-instant-1.2",
                              messages=[{
                                  "role": "user",
                                  "content": "Hi 👋 - i'm claude"
                              }])
        print(response)
    except Exception as e:
        print(e)

test_langfuse_logging()



