"""
Tests for litellm.litellm_core_utils.redact_messages
"""

import pytest


def test_redact_sensitive_keys():
    """
    Test that redact_sensitive_keys recursively redacts keys matching the predicate.
    """
    from litellm.litellm_core_utils.redact_messages import redact_sensitive_keys

    metadata = {
        "user_api_key_auth_metadata": {
            "logging": [
                {
                    "callback_name": "langfuse",
                    "callback_vars": {
                        "langfuse_public_key": "pk-lf-xxx",
                        "langfuse_secret_key": "sk-lf-secret",
                        "langfuse_secret": "another-secret",
                    },
                }
            ],
            "other_key": "keep_this",
        },
        "user_api_key_auth": {
            "token": "abc123",
            "metadata": {
                "logging": [
                    {
                        "callback_name": "langfuse",
                        "callback_vars": {"langfuse_secret_key": "secret"},
                    }
                ],
            },
        },
    }

    predicate = lambda k: k.endswith("_secret_key") or k.endswith("_secret")
    result = redact_sensitive_keys(metadata, predicate)

    assert result["user_api_key_auth_metadata"]["logging"][0]["callback_vars"] == {
        "langfuse_public_key": "pk-lf-xxx",
        "langfuse_secret_key": "scrubbed_by_litellm",
        "langfuse_secret": "scrubbed_by_litellm",
    }
    assert result["user_api_key_auth_metadata"]["other_key"] == "keep_this"
    assert result["user_api_key_auth"]["token"] == "abc123"
    assert result["user_api_key_auth"]["metadata"]["logging"][0]["callback_vars"] == {
        "langfuse_secret_key": "scrubbed_by_litellm",
    }


def test_redact_sensitive_keys_with_list():
    """
    Test that redact_sensitive_keys handles lists correctly.
    """
    from litellm.litellm_core_utils.redact_messages import redact_sensitive_keys

    data = [
        {"api_secret": "secret1", "name": "first"},
        {"api_secret": "secret2", "name": "second"},
    ]

    predicate = lambda k: k.endswith("_secret")
    result = redact_sensitive_keys(data, predicate)

    assert result[0] == {"api_secret": "scrubbed_by_litellm", "name": "first"}
    assert result[1] == {"api_secret": "scrubbed_by_litellm", "name": "second"}


def test_redact_sensitive_keys_with_primitives():
    """
    Test that redact_sensitive_keys returns primitives unchanged.
    """
    from litellm.litellm_core_utils.redact_messages import redact_sensitive_keys

    predicate = lambda k: k.endswith("_secret")

    assert redact_sensitive_keys("string", predicate) == "string"
    assert redact_sensitive_keys(123, predicate) == 123
    assert redact_sensitive_keys(None, predicate) is None
    assert redact_sensitive_keys(True, predicate) is True
