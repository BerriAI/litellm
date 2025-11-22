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
