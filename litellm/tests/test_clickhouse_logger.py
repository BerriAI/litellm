import sys
import os
import io, asyncio

# import logging
# logging.basicConfig(level=logging.DEBUG)
sys.path.insert(0, os.path.abspath("../.."))
print("Modified sys.path:", sys.path)


from litellm import completion
import litellm

litellm.num_retries = 3

import time, random
import pytest


@pytest.mark.asyncio
async def test_custom_api_logging():
    try:
        litellm.success_callback = ["clickhouse"]
        litellm.set_verbose = True
        await litellm.acompletion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": f"This is a test"}],
            max_tokens=10,
            temperature=0.7,
            user="ishaan-2",
        )

    except Exception as e:
        pytest.fail(f"An exception occurred - {e}")
    finally:
        # post, close log file and verify
        # Reset stdout to the original value
        print("Passed!")
