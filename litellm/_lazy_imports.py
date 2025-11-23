"""Lazy import helper functions for litellm module.

This module contains helper functions that handle lazy loading of various
litellm components to reduce import-time memory consumption.
"""
import sys
from typing import Any


def _get_litellm_globals() -> dict:
    """Helper to get the globals dictionary of the litellm module."""
    return sys.modules["litellm"].__dict__


def _lazy_import_cost_calculator(name: str) -> Any:
    """Lazy import for cost_calculator functions."""
    _globals = _get_litellm_globals()
    if name == "completion_cost":
        from .cost_calculator import completion_cost as _completion_cost
        _globals["completion_cost"] = _completion_cost
        return _completion_cost
    
    if name == "cost_per_token":
        from .cost_calculator import cost_per_token as _cost_per_token
        _globals["cost_per_token"] = _cost_per_token
        return _cost_per_token
    
    if name == "response_cost_calculator":
        from .cost_calculator import response_cost_calculator as _response_cost_calculator
        _globals["response_cost_calculator"] = _response_cost_calculator
        return _response_cost_calculator
    
    raise AttributeError(f"Cost calculator lazy import: unknown attribute {name!r}")

