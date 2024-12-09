import os
import sys

import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
from litellm import completion
import litellm


def test_completion_ai21():
    print("running ai21 j2light test")
    litellm.set_verbose = True
    model_name = "jamba-1.5-mini"
    try:
        response = completion(
            model=model_name,
            messages=[{"role": "user", "content": "What is the capital of France?"}],
            max_tokens=100,
            temperature=0.8,
        )
        # Add any assertions here to check the response
        print(response)
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")
