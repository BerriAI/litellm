#### What this tests ####
#    This tests the timeout decorator

import sys, os
import traceback

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import time
import litellm
import openai
import pytest


def test_timeout():
    # this Will Raise a timeout
    litellm.set_verbose=False
    try:
        response = litellm.completion(
            model="gpt-4",
            timeout=0.01,
            messages=[
                {
                    "role": "user",
                    "content": "hello, write a 20 pg essay"
                }
            ]
        )
    except openai.APITimeoutError as e:
        print("Passed: Raised correct exception. Got openai.APITimeoutError\nGood Job", e)
        print(type(e))
        pass
    except Exception as e:
        pytest.fail(f"Did not raise error `openai.APITimeoutError`. Instead raised error type: {type(e)}, Error: {e}")
test_timeout()