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
    TOKEN_COUNTER_NAMES,
    CACHING_NAMES,
    BEDROCK_TYPES_NAMES,
    TYPES_UTILS_NAMES,
    LLM_CLIENT_CACHE_NAMES,
    HTTP_HANDLER_NAMES,
    _lazy_import_cost_calculator,
    _lazy_import_litellm_logging,
    _lazy_import_utils,
    _lazy_import_token_counter,
    _lazy_import_bedrock_types,
    _lazy_import_types_utils,
    _lazy_import_caching,
    _lazy_import_llm_client_cache,
    _lazy_import_http_handlers,
    DOTPROMPT_NAMES,
    _lazy_import_dotprompt,
    LLM_CONFIG_NAMES,
    _lazy_import_llm_configs,
    TYPES_NAMES,
    _lazy_import_types,
    LLM_PROVIDER_LOGIC_NAMES,
    _lazy_import_llm_provider_logic,
    UTILS_MODULE_NAMES,
    _lazy_import_utils_module,
)


def _clear_names_from_globals(names: tuple):
    """Clear all names from litellm globals."""
    # Get the actual globals dict, not a copy
    litellm_globals = sys.modules["litellm"].__dict__
    for name in names:
        if name in litellm_globals:
            del litellm_globals[name]


def _clear_names_from_utils_globals(names: tuple):
    """Clear all names from litellm.utils globals."""
    # Get the actual globals dict, not a copy
    utils_globals = sys.modules["litellm.utils"].__dict__
    for name in names:
        if name in utils_globals:
            del utils_globals[name]


def _verify_only_requested_name_imported(name: str, all_names: tuple):
    """Verify that only the requested name is in globals, not the others."""
    # Get the actual globals dict, not a copy
    litellm_globals = sys.modules["litellm"].__dict__
    for other_name in all_names:
        if other_name != name:
            assert other_name not in litellm_globals, f"{other_name} should not be imported when importing {name}"


def _verify_only_requested_name_imported_in_utils(name: str, all_names: tuple):
    """Verify that only the requested name is in utils globals, not the others."""
    # Get the actual globals dict, not a copy
    utils_globals = sys.modules["litellm.utils"].__dict__
    for other_name in all_names:
        if other_name != name:
            assert other_name not in utils_globals, f"{other_name} should not be imported when importing {name}"


def test_cost_calculator_lazy_imports():
    """Test that all cost calculator functions can be lazy imported."""
    # Get the actual globals dict, not a copy
    litellm_globals = sys.modules["litellm"].__dict__
    
    # Test each name individually - only that name should be imported
    for name in COST_CALCULATOR_NAMES:
        # Clear all names before importing just one
        _clear_names_from_globals(COST_CALCULATOR_NAMES)
        
        func = _lazy_import_cost_calculator(name)
        assert func is not None
        assert callable(func)
        assert name in litellm_globals
        
        # Verify only the requested name is in globals, not the others
        _verify_only_requested_name_imported(name, COST_CALCULATOR_NAMES)


def test_litellm_logging_lazy_imports():
    """Test that all litellm_logging items can be lazy imported."""
    # Get the actual globals dict, not a copy
    litellm_globals = sys.modules["litellm"].__dict__
    
    # Test each name individually - only that name should be imported
    for name in LITELLM_LOGGING_NAMES:
        # Clear all names before importing just one
        _clear_names_from_globals(LITELLM_LOGGING_NAMES)
        
        item = _lazy_import_litellm_logging(name)
        assert item is not None
        assert name in litellm_globals
        
        # Verify only the requested name is in globals, not the others
        _verify_only_requested_name_imported(name, LITELLM_LOGGING_NAMES)


def test_utils_lazy_imports():
    """Test that all utils functions can be lazy imported."""
    # Get the actual globals dict, not a copy
    litellm_globals = sys.modules["litellm"].__dict__
    
    # Test each name individually - only that name should be imported
    for name in UTILS_NAMES:
        # Clear all names before importing just one
        _clear_names_from_globals(UTILS_NAMES)
        
        attr = _lazy_import_utils(name)
        assert attr is not None
        assert name in litellm_globals
        
        # Verify only the requested name is in globals, not the others
        _verify_only_requested_name_imported(name, UTILS_NAMES)


def test_caching_lazy_imports():
    """Test that all caching classes can be lazy imported."""
    # Get the actual globals dict, not a copy
    litellm_globals = sys.modules["litellm"].__dict__
    
    # Test each name individually - only that name should be imported
    for name in CACHING_NAMES:
        # Clear all names before importing just one
        _clear_names_from_globals(CACHING_NAMES)
        
        cls = _lazy_import_caching(name)
        assert cls is not None
        assert name in litellm_globals
        
        # Verify only the requested name is in globals, not the others
        _verify_only_requested_name_imported(name, CACHING_NAMES)


def test_token_counter_lazy_imports():
    """Test that token counter utilities can be lazy imported."""
    # Get the actual globals dict, not a copy
    litellm_globals = sys.modules["litellm"].__dict__
    
    for name in TOKEN_COUNTER_NAMES:
        _clear_names_from_globals(TOKEN_COUNTER_NAMES)

        func = _lazy_import_token_counter(name)
        assert func is not None
        assert name in litellm_globals

        _verify_only_requested_name_imported(name, TOKEN_COUNTER_NAMES)


def test_bedrock_types_lazy_imports():
    """Test that Bedrock type aliases can be lazy imported."""
    # Get the actual globals dict, not a copy
    litellm_globals = sys.modules["litellm"].__dict__
    
    for name in BEDROCK_TYPES_NAMES:
        _clear_names_from_globals(BEDROCK_TYPES_NAMES)

        alias = _lazy_import_bedrock_types(name)
        assert alias is not None
        assert name in litellm_globals

        _verify_only_requested_name_imported(name, BEDROCK_TYPES_NAMES)


def test_types_utils_lazy_imports():
    """Test that common types.utils symbols can be lazy imported."""
    # Get the actual globals dict, not a copy
    litellm_globals = sys.modules["litellm"].__dict__
    
    for name in TYPES_UTILS_NAMES:
        _clear_names_from_globals(TYPES_UTILS_NAMES)

        obj = _lazy_import_types_utils(name)
        assert obj is not None
        assert name in litellm_globals

        _verify_only_requested_name_imported(name, TYPES_UTILS_NAMES)


def test_llm_client_cache_lazy_imports():
    """Test that LLM client cache class and singleton can be lazy imported."""
    # Get the actual globals dict, not a copy
    litellm_globals = sys.modules["litellm"].__dict__
    
    for name in LLM_CLIENT_CACHE_NAMES:
        _clear_names_from_globals(LLM_CLIENT_CACHE_NAMES)

        obj = _lazy_import_llm_client_cache(name)
        assert obj is not None
        assert name in litellm_globals

        _verify_only_requested_name_imported(name, LLM_CLIENT_CACHE_NAMES)


def test_http_handler_lazy_imports():
    """Test that HTTP handler singletons can be lazy imported."""
    # Get the actual globals dict, not a copy
    litellm_globals = sys.modules["litellm"].__dict__
    
    for name in HTTP_HANDLER_NAMES:
        _clear_names_from_globals(HTTP_HANDLER_NAMES)

        handler = _lazy_import_http_handlers(name)
        assert handler is not None
        assert name in litellm_globals

        _verify_only_requested_name_imported(name, HTTP_HANDLER_NAMES)


def test_dotprompt_lazy_imports():
    """Test that dotprompt globals can be lazy imported."""
    # Get the actual globals dict, not a copy
    litellm_globals = sys.modules["litellm"].__dict__
    
    for name in DOTPROMPT_NAMES:
        _clear_names_from_globals(DOTPROMPT_NAMES)

        obj = _lazy_import_dotprompt(name)
        assert name in litellm_globals

        # Only the setter must be callable; others may be None by default
        if name == "set_global_prompt_directory":
            assert callable(obj), f"{name} should be callable"

        _verify_only_requested_name_imported(name, DOTPROMPT_NAMES)


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

    with pytest.raises(AttributeError):
        _lazy_import_token_counter("unknown")

    with pytest.raises(AttributeError):
        _lazy_import_llm_client_cache("unknown")

    with pytest.raises(AttributeError):
        _lazy_import_bedrock_types("unknown")

    with pytest.raises(AttributeError):
        _lazy_import_types_utils("unknown")

    with pytest.raises(AttributeError):
        _lazy_import_llm_configs("unknown")

    with pytest.raises(AttributeError):
        _lazy_import_types("unknown")

    with pytest.raises(AttributeError):
        _lazy_import_llm_provider_logic("unknown")

    with pytest.raises(AttributeError):
        _lazy_import_utils_module("unknown")


def test_llm_config_lazy_imports():
    """Test that LLM config classes can be lazy imported."""
    # Get the actual globals dict, not a copy
    litellm_globals = sys.modules["litellm"].__dict__
    
    for name in LLM_CONFIG_NAMES:
        _clear_names_from_globals(LLM_CONFIG_NAMES)

        obj = _lazy_import_llm_configs(name)
        assert obj is not None
        assert name in litellm_globals
        # Config classes should be classes/types
        assert isinstance(obj, type), f"{name} should be a class"

        _verify_only_requested_name_imported(name, LLM_CONFIG_NAMES)


def test_types_lazy_imports():
    """Test that type classes can be lazy imported."""
    # Get the actual globals dict, not a copy
    litellm_globals = sys.modules["litellm"].__dict__
    
    for name in TYPES_NAMES:
        _clear_names_from_globals(TYPES_NAMES)

        obj = _lazy_import_types(name)
        assert obj is not None
        assert name in litellm_globals
        # Type classes should be classes/types
        assert isinstance(obj, type), f"{name} should be a class"

        _verify_only_requested_name_imported(name, TYPES_NAMES)


def test_llm_provider_logic_lazy_imports():
    """Test that LLM provider logic functions can be lazy imported."""
    # Get the actual globals dict, not a copy
    litellm_globals = sys.modules["litellm"].__dict__
    
    for name in LLM_PROVIDER_LOGIC_NAMES:
        _clear_names_from_globals(LLM_PROVIDER_LOGIC_NAMES)

        func = _lazy_import_llm_provider_logic(name)
        assert func is not None
        assert callable(func)
        assert name in litellm_globals

        _verify_only_requested_name_imported(name, LLM_PROVIDER_LOGIC_NAMES)


def test_utils_module_lazy_imports():
    """Test that utils module attributes can be lazy imported."""
    # Get the actual globals dict, not a copy
    utils_globals = sys.modules["litellm.utils"].__dict__
    
    for name in UTILS_MODULE_NAMES:
        _clear_names_from_utils_globals(UTILS_MODULE_NAMES)

        obj = _lazy_import_utils_module(name)
        assert obj is not None
        assert name in utils_globals

        _verify_only_requested_name_imported_in_utils(name, UTILS_MODULE_NAMES)

