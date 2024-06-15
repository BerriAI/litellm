import os
import sys
import traceback

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import pytest

import litellm

try:
    print(litellm.get_model_cost_map(url="fake-url"))
except Exception as e:
    pytest.fail(f"An exception occurred: {e}")
