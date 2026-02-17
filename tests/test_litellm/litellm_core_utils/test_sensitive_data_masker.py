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
    
    # Test case sensitivity - excluded_keys should be exact match
    masked = masker.mask_dict(data, excluded_keys={"LITELLM_CREDENTIALS_NAME"})
    # Should still be masked because case doesn't match (exact match required)
    assert masked["litellm_credentials_name"] == "my-credential-name"  # Not masked because it doesn't match patterns anyway
    
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
