import sys
import os
import io

sys.path.insert(0, os.path.abspath('../..'))

from litellm import completion
import litellm

litellm.success_callback = ["langsmith"]
# litellm.set_verbose = True
import time


def test_langsmith_logging():
    try:
        response = completion(model="claude-instant-1.2",
                              messages=[{
                                  "role": "user",
                                  "content": "what llm are u"
                              }],
                              max_tokens=10,
                              temperature=0.2
                              )
        print(response)
    except Exception as e:
        print(e)

test_langsmith_logging()


def test_langsmith_logging_with_metadata():
    try:
        response = completion(model="gpt-3.5-turbo",
                              messages=[{
                                  "role": "user",
                                  "content": "what llm are u"
                              }],
                              max_tokens=10,
                              temperature=0.2,
                              metadata={
                                  "run_name": "litellmRUN",
                                  "project_name": "litellm-completion",
                              }
                              )
        print(response)
    except Exception as e:
        print(e)

test_langsmith_logging_with_metadata()
