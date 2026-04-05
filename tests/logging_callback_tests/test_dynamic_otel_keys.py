import sys
import os

sys.path.insert(0, os.path.abspath("../.."))

from litellm.litellm_core_utils.initialize_dynamic_callback_params import (
    initialize_standard_callback_dynamic_params,
    sanitize_metadata_for_key_info,
    REDACTED_VALUE,
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


def test_sanitize_metadata_for_key_info():
    """
    Test that sanitize_metadata_for_key_info correctly redacts sensitive callback credentials.

    Regression test for: https://github.com/BerriAI/litellm/issues/23776
    """
    metadata = {
        "langfuse_public_key": "pk-lf-12345",
        "langfuse_secret_key": "sk-lf-secret",
        "langfuse_secret": "sk-lf-alt-secret",
        "langfuse_host": "https://langfuse.example.com",
        "langsmith_api_key": "ls-api-key",
        "arize_api_key": "arize-key",
        "arize_space_key": "arize-space",
        "humanloop_api_key": "humanloop-key",
        "posthog_api_key": "posthog-key",
        "braintrust_api_key": "braintrust-key",
        "slack_webhook_url": "https://hooks.slack.com/services/xxx",
        "lunary_public_key": "lunary-key",
        "gcs_path_service_account": "gcs-service-account-json",
        # Non-sensitive fields that should remain intact
        "safe_field": "visible-value",
        "model_rpm_limit": {"gpt-4": 100},
        "trace_id": "trace-123",
    }

    sanitized = sanitize_metadata_for_key_info(metadata)

    # Verify sensitive fields are redacted
    assert sanitized["langfuse_public_key"] == REDACTED_VALUE
    assert sanitized["langfuse_secret_key"] == REDACTED_VALUE
    assert sanitized["langfuse_secret"] == REDACTED_VALUE
    assert sanitized["langfuse_host"] == REDACTED_VALUE
    assert sanitized["langsmith_api_key"] == REDACTED_VALUE
    assert sanitized["arize_api_key"] == REDACTED_VALUE
    assert sanitized["arize_space_key"] == REDACTED_VALUE
    assert sanitized["humanloop_api_key"] == REDACTED_VALUE
    assert sanitized["posthog_api_key"] == REDACTED_VALUE
    assert sanitized["braintrust_api_key"] == REDACTED_VALUE
    assert sanitized["slack_webhook_url"] == REDACTED_VALUE
    assert sanitized["lunary_public_key"] == REDACTED_VALUE
    assert sanitized["gcs_path_service_account"] == REDACTED_VALUE

    # Verify non-sensitive fields remain intact
    assert sanitized["safe_field"] == "visible-value"
    assert sanitized["model_rpm_limit"] == {"gpt-4": 100}
    assert sanitized["trace_id"] == "trace-123"

    # Verify original metadata is not modified
    assert metadata["langfuse_public_key"] == "pk-lf-12345"


def test_sanitize_metadata_for_key_info_none():
    """
    Test that sanitize_metadata_for_key_info handles None input.
    """
    result = sanitize_metadata_for_key_info(None)
    assert result is None


def test_sanitize_metadata_for_key_info_empty():
    """
    Test that sanitize_metadata_for_key_info handles empty metadata.
    """
    result = sanitize_metadata_for_key_info({})
    assert result == {}


if __name__ == "__main__":
    test_dynamic_key_extraction_from_metadata()
    test_dynamic_key_extraction_from_litellm_params_metadata()
    test_sanitize_metadata_for_key_info()
    test_sanitize_metadata_for_key_info_none()
    test_sanitize_metadata_for_key_info_empty()
    print("All tests passed!")

