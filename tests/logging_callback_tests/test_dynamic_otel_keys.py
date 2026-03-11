import sys
import os

sys.path.insert(0, os.path.abspath("../.."))

from litellm.litellm_core_utils.initialize_dynamic_callback_params import (
    initialize_standard_callback_dynamic_params,
    scrub_callback_config_params_from_dict,
)


def test_dynamic_key_extraction_from_metadata():
    """
    Test extraction of langfuse keys from metadata in kwargs.
    This simulates a Proxy request where keys are passed in metadata.
    """
    kwargs = {
        "metadata": {
            "langfuse_public_key": "pk-test",
            "langfuse_secret_key": "sk-test",
            "langfuse_host": "https://test.langfuse.com",
        }
    }

    params = initialize_standard_callback_dynamic_params(kwargs)

    assert params.get("langfuse_public_key") == "pk-test"
    assert params.get("langfuse_secret_key") == "sk-test"
    assert params.get("langfuse_host") == "https://test.langfuse.com"


def test_dynamic_key_extraction_from_litellm_params_metadata():
    """
    Test extraction of langfuse keys from litellm_params.metadata.
    """
    kwargs = {
        "litellm_params": {
            "metadata": {
                "langfuse_public_key": "pk-litellm",
                "langfuse_secret_key": "sk-litellm",
            }
        }
    }

    params = initialize_standard_callback_dynamic_params(kwargs)

    assert params.get("langfuse_public_key") == "pk-litellm"
    assert params.get("langfuse_secret_key") == "sk-litellm"


def test_scrub_callback_config_params_removes_credentials():
    """Scrub removes callback config/credentials before passing to loggers."""
    data = {
        "messages": "[{'role': 'user', 'content': 'Hi'}]",
        "langfuse_public_key": "pk-lf-xxx",
        "langfuse_secret": "sk-lf-4be1b432-61f4-4771-82a2-a33d3822ad65",
        "langfuse_host": "http://localhost:9999",
        "litellm_logging_obj": "<Logging object>",
    }
    result = scrub_callback_config_params_from_dict(data)
    assert "messages" in result
    assert "langfuse_secret" not in result
    assert "langfuse_public_key" not in result
    assert "langfuse_host" not in result
    assert "litellm_logging_obj" not in result


def test_scrub_callback_config_params_pattern_matching():
    """Test suffix-pattern scrubbing for custom integration credentials."""
    data = {
        "model": "gpt-4",
        "temperature": 0.7,
        "my_custom_service_api_key": "secret-api-key",
        "some_integration_secret": "s3cr3t",
        "another_service_secret_key": "sk-xxxx",
        "keep_this_value": "important_data",
    }
    result = scrub_callback_config_params_from_dict(data)
    
    # Verify safe values are kept
    assert "model" in result and result["model"] == "gpt-4"
    assert "temperature" in result and result["temperature"] == 0.7
    assert "keep_this_value" in result and result["keep_this_value"] == "important_data"
    
    # Verify credentials are scrubbed by suffix patterns
    assert "my_custom_service_api_key" not in result
    assert "some_integration_secret" not in result
    assert "another_service_secret_key" not in result


if __name__ == "__main__":
    test_dynamic_key_extraction_from_metadata()
    test_dynamic_key_extraction_from_litellm_params_metadata()
    test_scrub_callback_config_params_removes_credentials()
    test_scrub_callback_config_params_pattern_matching()
