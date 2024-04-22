# What is this?
## Unit tests for the 'function_setup()' function
import sys, os
import traceback
from dotenv import load_dotenv

load_dotenv()
import os, io

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the, system path
import pytest, uuid
from litellm.utils import function_setup, Rules
from datetime import datetime


def test_empty_content():
    """
    Make a chat completions request with empty content -> expect this to work
    """
    rules_obj = Rules()

    def completion():
        pass

    function_setup(
        original_function=completion.__name__,
        rules_obj=rules_obj,
        start_time=datetime.now(),
        messages=[],
        litellm_call_id=str(uuid.uuid4()),
    )
