import sys
import os

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from litellm.proxy.common_utils.callback_utils import (
    encrypt_logging_callback_vars,
    get_remaining_tokens_and_requests_from_request_data,
    normalize_callback_names,
    redact_sensitive_logging_metadata,
)

from unittest.mock import patch
from litellm.proxy.common_utils.callback_utils import process_callback


def test_get_remaining_tokens_and_requests_from_request_data():
    model_group = "openrouter/google/gemini-2.0-flash-001"
    casedata = {
        "metadata": {
            "model_group": model_group,
            f"litellm-key-remaining-requests-{model_group}": 100,
            f"litellm-key-remaining-tokens-{model_group}": 200,
        }
    }

    headers = get_remaining_tokens_and_requests_from_request_data(casedata)

    expected_name = "openrouter-google-gemini-2.0-flash-001"
    assert headers == {
        f"x-litellm-key-remaining-requests-{expected_name}": 100,
        f"x-litellm-key-remaining-tokens-{expected_name}": 200,
    }


@patch(
    "litellm.proxy.common_utils.callback_utils.CustomLogger.get_callback_env_vars",
    return_value=["API_KEY", "MISSING_VAR"],
)
def test_process_callback_with_env_vars(mock_get_env_vars):
    environment_variables = {
        "API_KEY": "PLAIN_VALUE",
        "UNUSED": "SHOULD_BE_IGNORED",
    }

    result = process_callback(
        _callback="my_callback",
        callback_type="input",
        environment_variables=environment_variables,
    )

    assert result["name"] == "my_callback"
    assert result["type"] == "input"
    assert result["variables"] == {
        "API_KEY": "PLAIN_VALUE",
        "MISSING_VAR": None,
    }


@patch(
    "litellm.proxy.common_utils.callback_utils.CustomLogger.get_callback_env_vars",
    return_value=[],
)
def test_process_callback_with_no_required_env_vars(mock_get_env_vars):
    result = process_callback(
        _callback="another_callback",
        callback_type="output",
        environment_variables={"SHOULD_NOT_BE_USED": "VALUE"},
    )

    assert result["name"] == "another_callback"
    assert result["type"] == "output"
    assert result["variables"] == {}


def test_normalize_callback_names_none_returns_empty_list():
    assert normalize_callback_names(None) == []
    assert normalize_callback_names([]) == []


def test_normalize_callback_names_lowercases_strings():
    assert normalize_callback_names(["SQS", "S3", "CUSTOM_CALLBACK"]) == [
        "sqs",
        "s3",
        "custom_callback",
    ]


# ---------------------------------------------------------------------------
# redact_sensitive_logging_metadata tests
# ---------------------------------------------------------------------------


def test_redact_scrubs_real_secret_values():
    """Real credential values must be partially masked showing last 3 chars."""
    metadata = {
        "logging": [
            {
                "callback_name": "langfuse",
                "callback_type": "success_and_failure",
                "callback_vars": {
                    "langfuse_public_key": "pk-lf-abc123",
                    "langfuse_secret_key": "sk-lf-supersecret",
                    "langfuse_host": "https://us.cloud.langfuse.com",
                },
            }
        ]
    }
    result = redact_sensitive_logging_metadata(metadata)
    vars_ = result["logging"][0]["callback_vars"]
    assert vars_["langfuse_public_key"] == "...123"
    assert vars_["langfuse_secret_key"] == "...ret"
    assert vars_["langfuse_host"] == "...com"


def test_redact_keeps_env_var_references():
    """os.environ/ pointers are not secrets — they must be preserved."""
    metadata = {
        "logging": [
            {
                "callback_name": "langfuse",
                "callback_vars": {
                    "langfuse_public_key": "os.environ/LANGFUSE_PUBLIC_KEY",
                    "langfuse_secret_key": "os.environ/LANGFUSE_SECRET_KEY",
                },
            }
        ]
    }
    result = redact_sensitive_logging_metadata(metadata)
    vars_ = result["logging"][0]["callback_vars"]
    assert vars_["langfuse_public_key"] == "os.environ/LANGFUSE_PUBLIC_KEY"
    assert vars_["langfuse_secret_key"] == "os.environ/LANGFUSE_SECRET_KEY"


def test_redact_does_not_mutate_original():
    """The original metadata dict must not be modified."""
    metadata = {
        "logging": [
            {
                "callback_name": "langfuse",
                "callback_vars": {"langfuse_secret_key": "sk-real-secret"},
            }
        ]
    }
    original_value = metadata["logging"][0]["callback_vars"]["langfuse_secret_key"]
    redact_sensitive_logging_metadata(metadata)
    assert (
        metadata["logging"][0]["callback_vars"]["langfuse_secret_key"] == original_value
    )


def test_redact_returns_none_for_none_input():
    assert redact_sensitive_logging_metadata(None) is None


def test_redact_returns_unchanged_when_no_logging_key():
    """Metadata without a 'logging' key passes through untouched."""
    metadata = {"some_other_key": "value"}
    result = redact_sensitive_logging_metadata(metadata)
    assert result == {"some_other_key": "value"}


def test_redact_mixed_env_and_real_values():
    """Only real values are scrubbed; env-var pointers in the same dict survive."""
    metadata = {
        "logging": [
            {
                "callback_name": "langfuse",
                "callback_vars": {
                    "langfuse_public_key": "os.environ/LANGFUSE_PUBLIC_KEY",
                    "langfuse_secret_key": "sk-real-secret",
                },
            }
        ]
    }
    result = redact_sensitive_logging_metadata(metadata)
    vars_ = result["logging"][0]["callback_vars"]
    assert vars_["langfuse_public_key"] == "os.environ/LANGFUSE_PUBLIC_KEY"
    assert vars_["langfuse_secret_key"] == "...ret"


# ---------------------------------------------------------------------------
# encrypt_logging_callback_vars tests
# ---------------------------------------------------------------------------


def _make_metadata(callback_vars: dict) -> dict:
    return {
        "logging": [
            {
                "callback_name": "langfuse",
                "callback_type": "success_and_failure",
                "callback_vars": callback_vars,
            }
        ]
    }


def test_encrypt_returns_none_for_none():
    assert encrypt_logging_callback_vars(None) is None


def test_encrypt_returns_unchanged_when_no_logging_key():
    metadata = {"other_key": "value"}
    result = encrypt_logging_callback_vars(metadata)
    assert result == {"other_key": "value"}


def test_encrypt_leaves_env_var_pointers_unchanged(monkeypatch):
    monkeypatch.setenv("LITELLM_SALT_KEY", "test-salt-key-1234567890123456")
    metadata = _make_metadata({"langfuse_secret_key": "os.environ/LANGFUSE_SECRET_KEY"})
    result = encrypt_logging_callback_vars(metadata)
    assert (
        result["logging"][0]["callback_vars"]["langfuse_secret_key"]
        == "os.environ/LANGFUSE_SECRET_KEY"
    )


def test_encrypt_real_values_are_changed(monkeypatch):
    monkeypatch.setenv("LITELLM_SALT_KEY", "test-salt-key-1234567890123456")
    plaintext = "sk-lf-supersecret"
    metadata = _make_metadata({"langfuse_secret_key": plaintext})
    result = encrypt_logging_callback_vars(metadata)
    encrypted = result["logging"][0]["callback_vars"]["langfuse_secret_key"]
    assert encrypted != plaintext
    assert isinstance(encrypted, str)


def test_encrypt_then_decrypt_roundtrip(monkeypatch):
    """Encrypt a value and confirm decrypt_value_helper returns the original."""
    monkeypatch.setenv("LITELLM_SALT_KEY", "test-salt-key-1234567890123456")
    from litellm.proxy.common_utils.encrypt_decrypt_utils import decrypt_value_helper

    plaintext = "pk-lf-abc123"
    metadata = _make_metadata({"langfuse_public_key": plaintext})
    encrypt_logging_callback_vars(metadata)
    encrypted = metadata["logging"][0]["callback_vars"]["langfuse_public_key"]
    assert encrypted != plaintext

    recovered = decrypt_value_helper(
        value=encrypted, key="langfuse_public_key", return_original_value=True
    )
    assert recovered == plaintext


def test_encrypt_mutates_in_place(monkeypatch):
    """encrypt_logging_callback_vars modifies the dict in-place."""
    monkeypatch.setenv("LITELLM_SALT_KEY", "test-salt-key-1234567890123456")
    metadata = _make_metadata({"langfuse_secret_key": "sk-lf-real"})
    returned = encrypt_logging_callback_vars(metadata)
    assert returned is metadata
    assert (
        metadata["logging"][0]["callback_vars"]["langfuse_secret_key"] != "sk-lf-real"
    )
