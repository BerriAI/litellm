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
_get_modified_max_tokens: Optional[Callable] = None
_default_encoding: Optional[Any] = None
_tiktoken_module: Optional[Any] = None


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


def get_modified_max_tokens() -> Callable:
    """Get the cached get_modified_max_tokens function, initializing if needed.
    
    Lazy imports on first call to avoid loading token_counter (which imports
    default_encoding and tiktoken) at import time. Subsequent calls use cached
    function for better performance.
    """
    global _get_modified_max_tokens
    if _get_modified_max_tokens is not None:
        return _get_modified_max_tokens
    from litellm.litellm_core_utils.token_counter import get_modified_max_tokens
    _get_modified_max_tokens = get_modified_max_tokens
    return _get_modified_max_tokens


def get_default_encoding() -> Any:
    """Get the cached default encoding object, initializing if needed.
    
    Lazy imports on first call to avoid loading default_encoding (which imports
    tiktoken) at import time. Subsequent calls use cached encoding object for
    better performance.
    """
    global _default_encoding
    if _default_encoding is not None:
        return _default_encoding
    from litellm.litellm_core_utils.default_encoding import encoding
    _default_encoding = encoding
    return _default_encoding


def get_tiktoken_module() -> Any:
    """Get the cached tiktoken module, initializing if needed.
    
    Lazy imports on first call to avoid loading tiktoken at import time.
    Subsequent calls use cached module for better performance.
    """
    global _tiktoken_module
    if _tiktoken_module is not None:
        return _tiktoken_module
    import tiktoken
    _tiktoken_module = tiktoken
    return _tiktoken_module


def clear_cached_imports() -> None:
    """Clear all cached imports. Useful for testing or memory management."""
    global _LiteLLMLogging, _coroutine_checker, _set_callbacks, _get_modified_max_tokens, _default_encoding, _tiktoken_module
    _LiteLLMLogging = None
    _coroutine_checker = None
    _set_callbacks = None
    _get_modified_max_tokens = None
    _default_encoding = None
    _tiktoken_module = None
