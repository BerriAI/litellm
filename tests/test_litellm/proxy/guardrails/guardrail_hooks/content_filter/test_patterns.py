"""
Tests for content filter pattern loading from JSON
"""
import json
import os
import sys

import pytest

sys.path.insert(
    0, os.path.abspath("../../")
)

from litellm.proxy.guardrails.guardrail_hooks.litellm_content_filter.patterns import (
    PATTERN_CATEGORIES,
    PATTERN_DESCRIPTIONS,
    PATTERN_DISPLAY_NAMES,
    PREBUILT_PATTERNS,
    get_all_pattern_names,
    get_compiled_pattern,
    get_pattern_metadata,
)


def test_patterns_loaded_from_json():
    """
    Test that patterns are successfully loaded from JSON file
    """
    assert len(PREBUILT_PATTERNS) > 0
    assert len(PATTERN_CATEGORIES) > 0
    assert len(PATTERN_DISPLAY_NAMES) > 0
    assert len(PATTERN_DESCRIPTIONS) > 0


def test_pattern_metadata_structure():
    """
    Test that get_pattern_metadata returns correctly structured data
    """
    patterns = get_pattern_metadata()
    
    assert len(patterns) > 0
    
    for pattern in patterns:
        assert "name" in pattern
        assert "display_name" in pattern
        assert "category" in pattern
        assert "description" in pattern
        
        assert isinstance(pattern["name"], str)
        assert isinstance(pattern["display_name"], str)
        assert isinstance(pattern["category"], str)
        assert isinstance(pattern["description"], str)


def test_display_names_user_friendly():
    """
    Test that display names are user-friendly and different from internal names
    """
    patterns = get_pattern_metadata()
    
    ssn_pattern = next((p for p in patterns if p["name"] == "us_ssn"), None)
    assert ssn_pattern is not None
    assert ssn_pattern["display_name"] == "SSN (Social Security Number)"
    
    email_pattern = next((p for p in patterns if p["name"] == "email"), None)
    assert email_pattern is not None
    assert email_pattern["display_name"] == "Email Address"


def test_pattern_compilation():
    """
    Test that patterns can be compiled into regex objects
    """
    pattern_names = get_all_pattern_names()
    
    assert len(pattern_names) > 0
    
    for pattern_name in pattern_names:
        compiled_pattern = get_compiled_pattern(pattern_name)
        assert compiled_pattern is not None
        assert hasattr(compiled_pattern, "match")


def test_pattern_compilation_error():
    """
    Test that invalid pattern names raise ValueError
    """
    with pytest.raises(ValueError, match="Unknown pattern name"):
        get_compiled_pattern("invalid_pattern_name")


def test_categories_contain_patterns():
    """
    Test that each category contains valid pattern names
    """
    all_pattern_names = set(PREBUILT_PATTERNS.keys())
    
    for category, patterns in PATTERN_CATEGORIES.items():
        assert len(patterns) > 0
        for pattern_name in patterns:
            assert pattern_name in all_pattern_names


def test_json_file_exists():
    """
    Test that patterns.json file exists and is valid JSON
    """
    json_path = os.path.join(
        os.path.dirname(__file__),
        "../../../../../../litellm/proxy/guardrails/guardrail_hooks/litellm_content_filter/patterns.json"
    )
    
    assert os.path.exists(json_path)
    
    with open(json_path, "r") as f:
        data = json.load(f)
        assert "patterns" in data
        assert len(data["patterns"]) > 0


def test_json_pattern_structure():
    """
    Test that each pattern in JSON has required fields
    """
    json_path = os.path.join(
        os.path.dirname(__file__),
        "../../../../../../litellm/proxy/guardrails/guardrail_hooks/litellm_content_filter/patterns.json"
    )
    
    with open(json_path, "r") as f:
        data = json.load(f)
        
        for pattern in data["patterns"]:
            assert "name" in pattern
            assert "display_name" in pattern
            assert "pattern" in pattern
            assert "category" in pattern
            assert "description" in pattern
            
            assert isinstance(pattern["name"], str)
            assert isinstance(pattern["display_name"], str)
            assert isinstance(pattern["pattern"], str)
            assert isinstance(pattern["category"], str)
            assert isinstance(pattern["description"], str)


def test_all_dictionaries_consistent():
    """
    Test that all pattern dictionaries have consistent keys
    """
    pattern_names_from_patterns = set(PREBUILT_PATTERNS.keys())
    pattern_names_from_display = set(PATTERN_DISPLAY_NAMES.keys())
    pattern_names_from_descriptions = set(PATTERN_DESCRIPTIONS.keys())
    
    assert pattern_names_from_patterns == pattern_names_from_display
    assert pattern_names_from_patterns == pattern_names_from_descriptions

