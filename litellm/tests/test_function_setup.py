# What is this?
## Unit tests for the 'function_setup()' function
import os
import sys
import traceback

from dotenv import load_dotenv

load_dotenv()
import io
import os

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the, system path
import uuid
from datetime import datetime

import pytest

from litellm.utils import Rules, function_setup


def test_empty_content():
    """
    Make a chat completions request with empty content -> expect this to work
    """
    rules_obj = Rules()

    def completion():
        pass

    function_setup(
        original_function="completion",
        rules_obj=rules_obj,
        start_time=datetime.now(),
        messages=[],
        litellm_call_id=str(uuid.uuid4()),
    )
