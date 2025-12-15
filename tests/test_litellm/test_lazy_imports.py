"""Simple tests for lazy import functionality."""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../.."))

import litellm
from litellm._lazy_imports import (
    UTILS_NAMES,
    _lazy_import_cost_calculator,
    _lazy_import_litellm_logging,
    _lazy_import_utils,
)


def test_cost_calculator_lazy_imports():
    """Test that all cost calculator functions can be lazy imported."""
    # Clear from globals to test fresh import
    for name in ["completion_cost", "cost_per_token", "response_cost_calculator"]:
        if name in litellm.__dict__:
            del litellm.__dict__[name]
        
        func = _lazy_import_cost_calculator(name)
        assert func is not None
        assert callable(func)
        assert name in litellm.__dict__


def test_litellm_logging_lazy_imports():
    """Test that all litellm_logging items can be lazy imported."""
    for name in ["Logging", "modify_integration"]:
        if name in litellm.__dict__:
            del litellm.__dict__[name]
        
        item = _lazy_import_litellm_logging(name)
        assert item is not None
        assert name in litellm.__dict__


def test_utils_lazy_imports():
    """Test that all utils functions can be lazy imported."""
    for name in UTILS_NAMES:
        if name in litellm.__dict__:
            del litellm.__dict__[name]
        
        attr = _lazy_import_utils(name)
        assert attr is not None
        assert name in litellm.__dict__


def test_unknown_attribute_raises_error():
    """Test that unknown attributes raise AttributeError."""
    with pytest.raises(AttributeError):
        _lazy_import_cost_calculator("unknown")
    
    with pytest.raises(AttributeError):
        _lazy_import_litellm_logging("unknown")
    
    with pytest.raises(AttributeError):
        _lazy_import_utils("unknown")

