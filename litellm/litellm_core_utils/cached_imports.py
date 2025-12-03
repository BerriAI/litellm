"""
Cached imports module for LiteLLM.

This module provides cached import functionality to avoid repeated imports
inside functions that are critical to performance.
"""

from typing import TYPE_CHECKING, Any, Callable, Optional, Type

# Type annotations for cached imports
if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging
    from litellm.litellm_core_utils.coroutine_checker import CoroutineChecker

# Global cache variables
_LiteLLMLogging: Optional[Type["Logging"]] = None
_coroutine_checker: Optional["CoroutineChecker"] = None
_set_callbacks: Optional[Callable] = None
_tiktoken_module: Optional[Any] = None
_default_encoding: Optional[Any] = None


def get_litellm_logging_class() -> Type["Logging"]:
    """Get the cached LiteLLM Logging class, initializing if needed."""
    global _LiteLLMLogging
    if _LiteLLMLogging is not None:
        return _LiteLLMLogging
    from litellm.litellm_core_utils.litellm_logging import Logging
    _LiteLLMLogging = Logging
    return _LiteLLMLogging


def get_coroutine_checker() -> "CoroutineChecker":
    """Get the cached coroutine checker instance, initializing if needed."""
    global _coroutine_checker
    if _coroutine_checker is not None:
        return _coroutine_checker
    from litellm.litellm_core_utils.coroutine_checker import coroutine_checker
    _coroutine_checker = coroutine_checker
    return _coroutine_checker


def get_set_callbacks() -> Callable:
    """Get the cached set_callbacks function, initializing if needed."""
    global _set_callbacks
    if _set_callbacks is not None:
        return _set_callbacks
    from litellm.litellm_core_utils.litellm_logging import set_callbacks
    _set_callbacks = set_callbacks
    return _set_callbacks


def get_tiktoken_module():
    """
    Get the cached tiktoken module, initializing if needed.
    
    This avoids repeated import overhead in hot paths like text_completion() 
    and token_counter functions.
    """
    global _tiktoken_module
    if _tiktoken_module is not None:
        return _tiktoken_module
    import tiktoken
    _tiktoken_module = tiktoken
    return _tiktoken_module


def get_default_encoding():
    """
    Get the cached default encoding (cl100k_base), initializing if needed.
    
    This avoids repeated import overhead in token counting functions.
    The default_encoding module imports tiktoken at module level, so this
    ensures tiktoken is only loaded once and cached.
    """
    global _default_encoding
    if _default_encoding is not None:
        return _default_encoding
    from litellm.litellm_core_utils.default_encoding import encoding
    _default_encoding = encoding
    return _default_encoding


def clear_cached_imports() -> None:
    """Clear all cached imports. Useful for testing or memory management."""
    global _LiteLLMLogging, _coroutine_checker, _set_callbacks, _tiktoken_module, _default_encoding
    _LiteLLMLogging = None
    _coroutine_checker = None
    _set_callbacks = None
    _tiktoken_module = None
    _default_encoding = None
