"""
Unit tests for SensitiveDataMasker - List Preservation
"""

import os
import sys

import pytest

# Add the parent directory to the system path
sys.path.insert(0, os.path.abspath("../../.."))

from litellm.litellm_core_utils.sensitive_data_masker import SensitiveDataMasker


def test_lists_are_preserved_not_converted_to_strings():
    """
    Regression test: Ensure lists are preserved as JSON arrays, not converted to strings.
    Previously, tags field in /model/info was returned as "['tag1', 'tag2']" instead of ["tag1", "tag2"]
    """
    masker = SensitiveDataMasker()

    data = {
        "tags": ["East US 2", "production", "test"],
    }

    masked = masker.mask_dict(data)

    # Must be a list, not a string
    assert isinstance(masked["tags"], list)
    assert masked["tags"] == ["East US 2", "production", "test"]


def test_excluded_keys_exact_match():
    """
    Test that excluded_keys prevents masking of specific keys (exact match).
    """
    masker = SensitiveDataMasker()

    data = {
        "api_key": "sk-1234567890abcdef",
        "litellm_credentials_name": "my-credential-name",
        "access_token": "token-12345",
        "port": 6379,
    }

    # Without excluded_keys, sensitive keys should be masked
    masked = masker.mask_dict(data)
    assert masked["api_key"] != "sk-1234567890abcdef"
    assert "*" in masked["api_key"]
    assert masked["access_token"] != "token-12345"
    assert "*" in masked["access_token"]

    # With excluded_keys, litellm_credentials_name should NOT be masked (exact match)
    # This ensures that even if pattern matching logic changes, excluded keys won't be masked
    masked = masker.mask_dict(data, excluded_keys={"litellm_credentials_name"})
    assert masked["litellm_credentials_name"] == "my-credential-name"

    # Other sensitive keys should still be masked
    assert masked["api_key"] != "sk-1234567890abcdef"
    assert "*" in masked["api_key"]
    assert masked["access_token"] != "token-12345"
    assert "*" in masked["access_token"]

    # Non-sensitive keys should remain unchanged
    assert masked["port"] == 6379

    # Test case sensitivity - excluded_keys should be exact match. Supplying an
    # uppercase variant must NOT exclude the lowercase key, so the field falls
    # back to pattern matching ("credentials" is in the sensitive pattern set)
    # and is masked.
    masked = masker.mask_dict(data, excluded_keys={"LITELLM_CREDENTIALS_NAME"})
    assert masked["litellm_credentials_name"] != "my-credential-name"
    assert "*" in masked["litellm_credentials_name"]

    # Test with api_key in excluded_keys to verify it works for keys that would be masked
    masked = masker.mask_dict(data, excluded_keys={"api_key"})
    assert masked["api_key"] == "sk-1234567890abcdef"  # Should NOT be masked
    assert masked["access_token"] != "token-12345"  # Should still be masked
    assert "*" in masked["access_token"]


def test_extra_headers_are_masked_recursively():
    """
    Ensure nested dictionaries (like extra_headers) are masked.
    """
    masker = SensitiveDataMasker()

    data = {
        "litellm_params": {
            "model": "openai/gpt-4",
            "extra_headers": {
                "rits_api_key": "sk-secret-12345-very-sensitive",
                "Authorization": "Bearer token123",
            },
        }
    }

    masked = masker.mask_dict(data)
    extra_headers = masked["litellm_params"]["extra_headers"]

    assert extra_headers["rits_api_key"] != "sk-secret-12345-very-sensitive"
    assert "*" in extra_headers["rits_api_key"]
    assert extra_headers["Authorization"] != "Bearer token123"
    assert "*" in extra_headers["Authorization"]


def test_lists_with_sensitive_keys_are_masked():
    """
    Lists belonging to sensitive keys should have their values masked.
    """
    masker = SensitiveDataMasker()
    data = {
        "api_key": ["sk-1234567890abcdef", "sk-9876543210fedcba"],
        "tags": ["prod", "test"],
    }

    masked = masker.mask_dict(data)
    # sensitive key list entries should be masked
    assert masked["api_key"][0] != "sk-1234567890abcdef"
    assert "*" in masked["api_key"][0]
    assert masked["api_key"][1] != "sk-9876543210fedcba"
    assert "*" in masked["api_key"][1]

    # non-sensitive list should remain unchanged
    assert masked["tags"] == ["prod", "test"]


def test_short_secrets_are_fully_masked():
    """
    Regression test: secrets at or below the reveal threshold (visible_prefix +
    visible_suffix, 8 by default) were returned verbatim instead of masked.
    An exactly-8-char value hit masked_length == 0 and round-tripped unchanged;
    anything shorter hit the early return. Both leaked short credentials (e.g. an
    8-char redis password) in plaintext through mask_dict.
    """
    masker = SensitiveDataMasker()

    # Boundary: exactly 8 chars previously returned verbatim.
    assert masker._mask_value("abcd1234") == "********"
    # Below threshold previously hit the early return and leaked verbatim.
    assert masker._mask_value("sk-12") == "*****"
    # Values above the threshold must still partially reveal, not over-mask.
    assert masker._mask_value("abcd12345") == "abcd*2345"

    masked = masker.mask_dict({"redis_password": "pass1234", "api_key": "sk-7a"})
    assert masked["redis_password"] == "********"
    assert masked["api_key"] == "*****"


def test_mask_short_values_false_keeps_short_values_readable():
    """
    mask_short_values=False opts out of full masking so short values are returned
    as-is. This preserves the truncation use (e.g. CooldownCache shows the first 50
    chars of an exception and only masks longer tails), while longer values are still
    partially masked.
    """
    masker = SensitiveDataMasker(
        visible_prefix=50, visible_suffix=0, mask_short_values=False
    )

    short = "Test exception for structure validation"
    assert masker._mask_value(short) == short

    long_value = "x" * 60
    masked = masker._mask_value(long_value)
    assert masked.startswith("x" * 50)
    assert masked.endswith("*" * 10)
    assert len(masked) == 60


def test_cost_per_token_fields_not_masked():
    """
    Regression test: cost fields like input_cost_per_token contain "token" in their name
    but should NOT be masked — they are pricing fields, not secrets.
    Previously, these would be displayed as e.g. "3.60*******e-06" in the UI.
    """
    masker = SensitiveDataMasker()
    data = {
        "input_cost_per_token": 3.6e-06,
        "output_cost_per_token": 1.2e-05,
        "cache_read_input_token_cost": 9.0e-07,
        "cache_creation_input_token_cost": 3.75e-06,
        # Real secret fields should still be masked
        "api_key": "sk-1234567890abcdef",
        "access_token": "my-secret-token",
    }

    masked = masker.mask_dict(data)

    # Cost fields must not be masked
    assert masked["input_cost_per_token"] == 3.6e-06
    assert masked["output_cost_per_token"] == 1.2e-05
    assert masked["cache_read_input_token_cost"] == 9.0e-07
    assert masked["cache_creation_input_token_cost"] == 3.75e-06

    # Actual secrets must still be masked
    assert "*" in masked["api_key"]
    assert "*" in masked["access_token"]


def test_mask_sensitive_structure_passes_through_plain_topology_names():
    """Fallback groups are usually lists of model-group name strings; those
    carry no secrets and must survive verbatim so opt-in debug output stays useful."""
    from litellm.litellm_core_utils.sensitive_data_masker import mask_sensitive_structure

    assert mask_sensitive_structure(["gpt-4", "claude-3-haiku"]) == ["gpt-4", "claude-3-haiku"]
    assert mask_sensitive_structure([{"gpt-3.5-turbo": ["claude-3-haiku"]}]) == [
        {"gpt-3.5-turbo": ["claude-3-haiku"]}
    ]
    assert mask_sensitive_structure(None) is None


def test_mask_sensitive_structure_masks_credentials_in_inline_fallback_dicts():
    """An inline-dict fallback can carry provider credentials; those values must be
    masked before the structure is embedded in a client-facing error message."""
    from litellm.litellm_core_utils.sensitive_data_masker import mask_sensitive_structure

    secret = "sk-INLINEFALLBACKSECRET1234567890"
    aws_secret = "wJalrXUtnFEMIK7MDENGbPxRfiCYSECRETKEY"
    masked = mask_sensitive_structure(
        [{"model": "openai/gpt-4", "api_key": secret, "aws_secret_access_key": aws_secret}]
    )

    rendered = str(masked)
    assert secret not in rendered
    assert aws_secret not in rendered
    # Non-secret keys stay visible so the fallback wiring remains debuggable
    assert masked[0]["model"] == "openai/gpt-4"
    assert "*" in masked[0]["api_key"]
    assert "*" in masked[0]["aws_secret_access_key"]


def test_mask_sensitive_structure_masks_credentials_nested_in_config_shape():
    """Credentials nested inside the {group: [fallbacks]} config shape must also be masked."""
    from litellm.litellm_core_utils.sensitive_data_masker import mask_sensitive_structure

    secret = "sk-NESTEDINLINESECRET0987654321"
    masked = mask_sensitive_structure(
        [{"primary-group": [{"model": "gpt-4o", "api_key": secret}]}]
    )
    assert secret not in str(masked)
