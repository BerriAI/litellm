import sys, os
import traceback
from dotenv import load_dotenv

load_dotenv()
import os

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import pytest
import litellm
from litellm import embedding, completion

messages = [{"role": "user", "content": "who is ishaan Github?  "}]


# test if response cached
def test_caching():
    try:
        litellm.caching = True
        response1 = completion(model="gpt-3.5-turbo", messages=messages)
        response2 = completion(model="gpt-3.5-turbo", messages=messages)
        print(f"response1: {response1}")
        print(f"response2: {response2}")
        litellm.caching = False
        if response2 != response1:
            print(f"response1: {response1}")
            print(f"response2: {response2}")
            pytest.fail(f"Error occurred: {e}")
    except Exception as e:
        litellm.caching = False
        print(f"error occurred: {traceback.format_exc()}")
        pytest.fail(f"Error occurred: {e}")


def test_caching_with_models():
    litellm.caching_with_models = True
    response2 = completion(model="gpt-3.5-turbo", messages=messages)
    response3 = completion(model="command-nightly", messages=messages)
    print(f"response2: {response2}")
    print(f"response3: {response3}")
    litellm.caching_with_models = False
    if response3 == response2:
        # if models are different, it should not return cached response
        print(f"response2: {response2}")
        print(f"response3: {response3}")
        pytest.fail(f"Error occurred: {e}")
