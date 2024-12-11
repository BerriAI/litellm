import os
import sys
import traceback

from dotenv import load_dotenv

load_dotenv()
import io
import os

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import json

import pytest

import litellm
from litellm import RateLimitError, Timeout, completion, completion_cost, embedding


def test_baseten():
    litellm.set_verbose = True
    try:
        response = completion(
            model="baseten/8w61582q",
            messages=[{"role": "user", "content": "hi who are you"}],
        )
        print(response)
        for chunk in response:
            print(chunk)
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")
