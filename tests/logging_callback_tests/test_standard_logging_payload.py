"""
Unit tests for StandardLoggingPayloadSetup
"""

import json
import os
import sys
from datetime import datetime
from unittest.mock import AsyncMock

from pydantic.main import Model

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system-path

import pytest
import litellm
from litellm.types.utils import Usage
from litellm.litellm_core_utils.litellm_logging import StandardLoggingPayloadSetup


@pytest.mark.parametrize(
    "response_obj,expected_values",
    [
        # Test None input
        (None, (0, 0, 0)),
        # Test empty dict
        ({}, (0, 0, 0)),
        # Test valid usage dict
        (
            {
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_tokens": 30,
                }
            },
            (10, 20, 30),
        ),
        # Test with litellm.Usage object
        (
            {"usage": Usage(prompt_tokens=15, completion_tokens=25, total_tokens=40)},
            (15, 25, 40),
        ),
        # Test invalid usage type
        ({"usage": "invalid"}, (0, 0, 0)),
        # Test None usage
        ({"usage": None}, (0, 0, 0)),
    ],
)
def test_get_usage(response_obj, expected_values):
    """
    Make sure values returned from get_usage are always integers
    """

    usage = StandardLoggingPayloadSetup.get_usage_from_response_obj(response_obj)

    # Check types
    assert isinstance(usage.prompt_tokens, int)
    assert isinstance(usage.completion_tokens, int)
    assert isinstance(usage.total_tokens, int)

    # Check values
    assert usage.prompt_tokens == expected_values[0]
    assert usage.completion_tokens == expected_values[1]
    assert usage.total_tokens == expected_values[2]
