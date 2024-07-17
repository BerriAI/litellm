import io
import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

import litellm
from litellm import completion

litellm.set_verbose = True
import time


def test_langsmith_logging():
    try:
        litellm.set_verbose = True
        litellm.success_callback = ["langsmith"]
        response = completion(
            model="claude-instant-1.2",
            messages=[{"role": "user", "content": "what llm are u"}],
            max_tokens=10,
            temperature=0.2,
        )
        print(response)
        time.sleep(3)
    except Exception as e:
        print(e)


# test_langsmith_logging()


def test_langsmith_logging_with_metadata():
    try:
        litellm.success_callback = ["langsmith"]
        response = completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "what llm are u"}],
            max_tokens=10,
            temperature=0.2,
        )
        print(response)
        time.sleep(3)
    except Exception as e:
        print(e)


# test_langsmith_logging_with_metadata()


def test_langsmith_logging_with_streaming_and_metadata():
    try:
        litellm.success_callback = ["langsmith"]
        response = completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "what llm are u"}],
            max_tokens=10,
            temperature=0.2,
            stream=True,
        )
        for chunk in response:
            continue
        time.sleep(3)
    except Exception as e:
        print(e)


# test_langsmith_logging_with_streaming_and_metadata()
