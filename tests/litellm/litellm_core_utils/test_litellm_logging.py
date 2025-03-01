import json
import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from litellm.litellm_core_utils.litellm_logging import StandardLoggingPayloadSetup


def test_remove_message_content_from_dict():
    # Test with None input
    assert StandardLoggingPayloadSetup._remove_message_content_from_dict(None) == {}

    # Test with empty dict
    assert StandardLoggingPayloadSetup._remove_message_content_from_dict({}) == {}

    # Test with sensitive content
    input_dict = {
        "messages": "sensitive content",
        "input": "secret prompt",
        "prompt": "confidential text",
        "safe_key": "safe value",
        "temperature": 0.7,
    }

    expected_output = {"safe_key": "safe value", "temperature": 0.7}

    result = StandardLoggingPayloadSetup._remove_message_content_from_dict(input_dict)
    assert result == expected_output


def test_get_model_parameters():
    # Test with empty kwargs
    assert StandardLoggingPayloadSetup._get_model_parameters({}) == {}

    # Test with None optional_params
    assert (
        StandardLoggingPayloadSetup._get_model_parameters({"optional_params": None})
        == {}
    )

    # Test with actual parameters
    kwargs = {
        "optional_params": {
            "temperature": 0.8,
            "messages": "sensitive data",
            "max_tokens": 100,
            "prompt": "secret prompt",
        }
    }

    expected_output = {"temperature": 0.8, "max_tokens": 100}

    result = StandardLoggingPayloadSetup._get_model_parameters(kwargs)
    assert result == expected_output


def test_get_model_parameters_complex():
    # Test with more complex optional parameters
    kwargs = {
        "optional_params": {
            "temperature": 0.8,
            "messages": [{"role": "user", "content": "sensitive"}],
            "max_tokens": 100,
            "stop": ["\n", "stop"],
            "presence_penalty": 0.5,
            "frequency_penalty": 0.3,
            "prompt": "secret prompt",
            "n": 1,
            "best_of": 2,
            "logit_bias": {"50256": -100},
        }
    }

    expected_output = {
        "temperature": 0.8,
        "max_tokens": 100,
        "stop": ["\n", "stop"],
        "presence_penalty": 0.5,
        "frequency_penalty": 0.3,
        "n": 1,
        "best_of": 2,
        "logit_bias": {"50256": -100},
    }

    result = StandardLoggingPayloadSetup._get_model_parameters(kwargs)
    assert result == expected_output


def test_get_model_parameters_empty_optional_params():
    # Test with empty optional_params dictionary
    kwargs = {"optional_params": {}}
    assert StandardLoggingPayloadSetup._get_model_parameters(kwargs) == {}


def test_get_model_parameters_missing_optional_params():
    # Test with missing optional_params key
    kwargs = {"model": "gpt-4", "api_key": "test"}
    assert StandardLoggingPayloadSetup._get_model_parameters(kwargs) == {}
