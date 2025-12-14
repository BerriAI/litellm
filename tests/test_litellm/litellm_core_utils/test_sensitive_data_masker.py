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
