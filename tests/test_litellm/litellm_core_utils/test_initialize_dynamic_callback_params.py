import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../../.."))

from litellm.litellm_core_utils.initialize_dynamic_callback_params import (
    initialize_standard_callback_dynamic_params,
)


def test_resolves_plain_values_at_top_level():
    kwargs = {
        "langfuse_public_key": "pk-test",
        "langfuse_secret_key": "sk-test",
    }

    params = initialize_standard_callback_dynamic_params(kwargs)

    assert params.get("langfuse_public_key") == "pk-test"
    assert params.get("langfuse_secret_key") == "sk-test"


def test_resolves_plain_values_from_metadata():
    kwargs = {
        "metadata": {
            "langfuse_public_key": "pk-meta",
            "langfuse_host": "https://test.langfuse.com",
        }
    }

    params = initialize_standard_callback_dynamic_params(kwargs)

    assert params.get("langfuse_public_key") == "pk-meta"
    assert params.get("langfuse_host") == "https://test.langfuse.com"


def test_env_reference_at_top_level_raises_with_guidance():
    kwargs = {"langfuse_public_key": "os.environ/LANGFUSE_PUBLIC_KEY"}

    with pytest.raises(ValueError) as exc_info:
        initialize_standard_callback_dynamic_params(kwargs)

    message = str(exc_info.value)
    assert "langfuse_public_key" in message
    assert "request body" in message
    assert "os.environ/" in message
    assert "config.yaml" in message


def test_env_reference_in_metadata_raises_with_guidance():
    kwargs = {
        "metadata": {
            "langsmith_api_key": "os.environ/LANGSMITH_API_KEY",
        }
    }

    with pytest.raises(ValueError) as exc_info:
        initialize_standard_callback_dynamic_params(kwargs)

    message = str(exc_info.value)
    assert "langsmith_api_key" in message
    assert "metadata" in message


def test_env_reference_in_litellm_params_metadata_raises():
    kwargs = {
        "litellm_params": {
            "metadata": {
                "gcs_bucket_name": "os.environ/GCS_BUCKET",
            }
        }
    }

    with pytest.raises(ValueError) as exc_info:
        initialize_standard_callback_dynamic_params(kwargs)

    assert "gcs_bucket_name" in str(exc_info.value)


def test_non_string_values_are_not_flagged():
    kwargs = {
        "langsmith_sampling_rate": 0.5,
    }

    params = initialize_standard_callback_dynamic_params(kwargs)

    assert params.get("langsmith_sampling_rate") == 0.5


def test_turn_off_message_logging_not_extracted_from_request():
    """turn_off_message_logging is admin-only — must not be settable via request."""
    kwargs = {"turn_off_message_logging": True}
    params = initialize_standard_callback_dynamic_params(kwargs)
    assert params.get("turn_off_message_logging") is None


def test_empty_kwargs_returns_empty_params():
    params = initialize_standard_callback_dynamic_params(None)
    assert dict(params) == {}

    params = initialize_standard_callback_dynamic_params({})
    assert dict(params) == {}
