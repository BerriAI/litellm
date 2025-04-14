import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

import time

from litellm.litellm_core_utils.litellm_logging import Logging as LitellmLogging


@pytest.fixture
def logging_obj():
    return LitellmLogging(
        model="bedrock/claude-3-5-sonnet-20240620-v1:0",
        messages=[{"role": "user", "content": "Hey"}],
        stream=True,
        call_type="completion",
        start_time=time.time(),
        litellm_call_id="12345",
        function_id="1245",
    )


def test_get_masked_api_base(logging_obj):
    api_base = "https://api.openai.com/v1"
    masked_api_base = logging_obj._get_masked_api_base(api_base)
    assert masked_api_base == "https://api.openai.com/v1"
    assert type(masked_api_base) == str
