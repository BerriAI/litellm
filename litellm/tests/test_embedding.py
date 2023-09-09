import sys, os
import traceback
import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import litellm
from litellm import embedding, completion

litellm.set_verbose = True


def test_openai_embedding():
    try:
        response = embedding(
            model="text-embedding-ada-002", input=["good morning from litellm"]
        )
        # Add any assertions here to check the response
        # print(f"response: {str(response)}")
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# test_openai_embedding()
