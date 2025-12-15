"""Simple tests for lazy import functionality."""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../.."))

import litellm
from litellm._lazy_imports import (
    COST_CALCULATOR_NAMES,
    LITELLM_LOGGING_NAMES,
    UTILS_NAMES,
    _lazy_import_cost_calculator,
    _lazy_import_litellm_logging,
    _lazy_import_utils,
)


def test_cost_calculator_lazy_imports():
    """Test that all cost calculator functions can be lazy imported."""
    # Test each name individually - only that name should be imported
    for name in COST_CALCULATOR_NAMES:
        # Clear all names before importing just one
        for n in COST_CALCULATOR_NAMES:
            if n in litellm.__dict__:
                del litellm.__dict__[n]
        
        func = _lazy_import_cost_calculator(name)
        assert func is not None
        assert callable(func)
        assert name in litellm.__dict__
        
        # Verify only the requested name is in globals, not the others
        for other_name in COST_CALCULATOR_NAMES:
            if other_name != name:
                assert other_name not in litellm.__dict__, f"{other_name} should not be imported when importing {name}"


def test_litellm_logging_lazy_imports():
    """Test that all litellm_logging items can be lazy imported."""
    # Test each name individually - only that name should be imported
    for name in LITELLM_LOGGING_NAMES:
        # Clear all names before importing just one
        for n in LITELLM_LOGGING_NAMES:
            if n in litellm.__dict__:
                del litellm.__dict__[n]
        
        item = _lazy_import_litellm_logging(name)
        assert item is not None
        assert name in litellm.__dict__
        
        # Verify only the requested name is in globals, not the others
        for other_name in LITELLM_LOGGING_NAMES:
            if other_name != name:
                assert other_name not in litellm.__dict__, f"{other_name} should not be imported when importing {name}"


def test_utils_lazy_imports():
    """Test that all utils functions can be lazy imported."""
    # Test each name individually - only that name should be imported
    for name in UTILS_NAMES:
        # Clear all names before importing just one
        for n in UTILS_NAMES:
            if n in litellm.__dict__:
                del litellm.__dict__[n]
        
        attr = _lazy_import_utils(name)
        assert attr is not None
        assert name in litellm.__dict__
        
        # Verify only the requested name is in globals, not the others
        for other_name in UTILS_NAMES:
            if other_name != name:
                assert other_name not in litellm.__dict__, f"{other_name} should not be imported when importing {name}"


def test_unknown_attribute_raises_error():
    """Test that unknown attributes raise AttributeError."""
    with pytest.raises(AttributeError):
        _lazy_import_cost_calculator("unknown")
    
    with pytest.raises(AttributeError):
        _lazy_import_litellm_logging("unknown")
    
    with pytest.raises(AttributeError):
        _lazy_import_utils("unknown")

