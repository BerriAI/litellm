import sys
import os
import io

sys.path.insert(0, os.path.abspath("../.."))

from litellm import completion
import litellm
import pytest

import time


@pytest.mark.skip(reason="beta test - this is a new feature")
def test_datadog_logging():
    try:
        litellm.success_callback = ["datadog"]
        litellm.set_verbose = True
        response = completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "what llm are u"}],
            max_tokens=10,
            temperature=0.2,
        )
        print(response)
    except Exception as e:
        print(e)
