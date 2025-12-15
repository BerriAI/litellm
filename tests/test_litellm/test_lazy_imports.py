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
    CACHING_NAMES,
    HTTP_HANDLER_NAMES,
    _lazy_import_cost_calculator,
    _lazy_import_litellm_logging,
    _lazy_import_utils,
    _lazy_import_caching,
    _lazy_import_http_handlers,
)


def _clear_names_from_globals(names: tuple):
    """Clear all names from litellm globals."""
    for name in names:
        if name in litellm.__dict__:
            del litellm.__dict__[name]


def _verify_only_requested_name_imported(name: str, all_names: tuple):
    """Verify that only the requested name is in globals, not the others."""
    for other_name in all_names:
        if other_name != name:
            assert other_name not in litellm.__dict__, f"{other_name} should not be imported when importing {name}"


def test_cost_calculator_lazy_imports():
    """Test that all cost calculator functions can be lazy imported."""
    # Test each name individually - only that name should be imported
    for name in COST_CALCULATOR_NAMES:
        # Clear all names before importing just one
        _clear_names_from_globals(COST_CALCULATOR_NAMES)
        
        func = _lazy_import_cost_calculator(name)
        assert func is not None
        assert callable(func)
        assert name in litellm.__dict__
        
        # Verify only the requested name is in globals, not the others
        _verify_only_requested_name_imported(name, COST_CALCULATOR_NAMES)


def test_litellm_logging_lazy_imports():
    """Test that all litellm_logging items can be lazy imported."""
    # Test each name individually - only that name should be imported
    for name in LITELLM_LOGGING_NAMES:
        # Clear all names before importing just one
        _clear_names_from_globals(LITELLM_LOGGING_NAMES)
        
        item = _lazy_import_litellm_logging(name)
        assert item is not None
        assert name in litellm.__dict__
        
        # Verify only the requested name is in globals, not the others
        _verify_only_requested_name_imported(name, LITELLM_LOGGING_NAMES)


def test_utils_lazy_imports():
    """Test that all utils functions can be lazy imported."""
    # Test each name individually - only that name should be imported
    for name in UTILS_NAMES:
        # Clear all names before importing just one
        _clear_names_from_globals(UTILS_NAMES)
        
        attr = _lazy_import_utils(name)
        assert attr is not None
        assert name in litellm.__dict__
        
        # Verify only the requested name is in globals, not the others
        _verify_only_requested_name_imported(name, UTILS_NAMES)


def test_caching_lazy_imports():
    """Test that all caching classes can be lazy imported."""
    # Test each name individually - only that name should be imported
    for name in CACHING_NAMES:
        # Clear all names before importing just one
        _clear_names_from_globals(CACHING_NAMES)
        
        cls = _lazy_import_caching(name)
        assert cls is not None
        assert name in litellm.__dict__
        
        # Verify only the requested name is in globals, not the others
        _verify_only_requested_name_imported(name, CACHING_NAMES)


def test_http_handler_lazy_imports():
    """Test that HTTP handler singletons can be lazy imported."""
    for name in HTTP_HANDLER_NAMES:
        _clear_names_from_globals(HTTP_HANDLER_NAMES)

        handler = _lazy_import_http_handlers(name)
        assert handler is not None
        assert name in litellm.__dict__

        _verify_only_requested_name_imported(name, HTTP_HANDLER_NAMES)


def test_unknown_attribute_raises_error():
    """Test that unknown attributes raise AttributeError."""
    with pytest.raises(AttributeError):
        _lazy_import_cost_calculator("unknown")
    
    with pytest.raises(AttributeError):
        _lazy_import_litellm_logging("unknown")
    
    with pytest.raises(AttributeError):
        _lazy_import_utils("unknown")

    with pytest.raises(AttributeError):
        _lazy_import_caching("unknown")

