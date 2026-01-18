"""
Lazy Import System

This module implements lazy loading for LiteLLM attributes. Instead of importing
everything when the module loads, we only import things when they're actually used.

How it works:
1. When someone accesses `litellm.some_attribute`, Python calls __getattr__ in __init__.py
2. __getattr__ looks up the attribute name in a registry
3. The registry points to a handler function (like _lazy_import_utils)
4. The handler function imports the module and returns the attribute
5. The result is cached so we don't import it again

This makes importing litellm much faster because we don't load heavy dependencies
until they're actually needed.
"""
import importlib
import sys
from typing import Any, Optional, cast, Callable

# Import all the data structures that define what can be lazy-loaded
# These are just lists of names and maps of where to find them
from ._lazy_imports_registry import (
    # Name tuples
    COST_CALCULATOR_NAMES,
    LITELLM_LOGGING_NAMES,
    UTILS_NAMES,
    TOKEN_COUNTER_NAMES,
    LLM_CLIENT_CACHE_NAMES,
    BEDROCK_TYPES_NAMES,
    TYPES_UTILS_NAMES,
    CACHING_NAMES,
    HTTP_HANDLER_NAMES,
    DOTPROMPT_NAMES,
    LLM_CONFIG_NAMES,
    TYPES_NAMES,
    LLM_PROVIDER_LOGIC_NAMES,
    UTILS_MODULE_NAMES,
    # Import maps
    _UTILS_IMPORT_MAP,
    _COST_CALCULATOR_IMPORT_MAP,
    _TYPES_UTILS_IMPORT_MAP,
    _TOKEN_COUNTER_IMPORT_MAP,
    _BEDROCK_TYPES_IMPORT_MAP,
    _CACHING_IMPORT_MAP,
    _LITELLM_LOGGING_IMPORT_MAP,
    _DOTPROMPT_IMPORT_MAP,
    _TYPES_IMPORT_MAP,
    _LLM_CONFIGS_IMPORT_MAP,
    _LLM_PROVIDER_LOGIC_IMPORT_MAP,
    _UTILS_MODULE_IMPORT_MAP,
)


def _get_litellm_globals() -> dict:
    """
    Get the globals dictionary of the litellm module.
    
    This is where we cache imported attributes so we don't import them twice.
    When you do `litellm.some_function`, it gets stored in this dictionary.
    """
    return sys.modules["litellm"].__dict__


def _get_utils_globals() -> dict:
    """
    Get the globals dictionary of the utils module.
    
    This is where we cache imported attributes so we don't import them twice.
    When you do `litellm.utils.some_function`, it gets stored in this dictionary.
    """
    return sys.modules["litellm.utils"].__dict__

# These are special lazy loaders for things that are used internally
# They're separate from the main lazy import system because they have specific use cases

# Lazy loader for default encoding - avoids importing heavy tiktoken library at startup
_default_encoding: Optional[Any] = None


def _get_default_encoding() -> Any:
    """
    Lazily load and cache the default OpenAI encoding.
    
    This avoids importing `litellm.litellm_core_utils.default_encoding` (and thus tiktoken)
    at `litellm` import time. The encoding is cached after the first import.
    
    This is used internally by utils.py functions that need the encoding but shouldn't
    trigger its import during module load.
    """
    global _default_encoding
    if _default_encoding is None:
        from litellm.litellm_core_utils.default_encoding import encoding

        _default_encoding = encoding
    return _default_encoding


# Lazy loader for get_modified_max_tokens to avoid importing token_counter at module import time
_get_modified_max_tokens_func: Optional[Any] = None


def _get_modified_max_tokens() -> Any:
    """
    Lazily load and cache the get_modified_max_tokens function.
    
    This avoids importing `litellm.litellm_core_utils.token_counter` at `litellm` import time.
    The function is cached after the first import.
    
    This is used internally by utils.py functions that need the token counter but shouldn't
    trigger its import during module load.
    """
    global _get_modified_max_tokens_func
    if _get_modified_max_tokens_func is None:
        from litellm.litellm_core_utils.token_counter import (
            get_modified_max_tokens as _get_modified_max_tokens_imported,
        )

        _get_modified_max_tokens_func = _get_modified_max_tokens_imported
    return _get_modified_max_tokens_func


# Lazy loader for token_counter to avoid importing token_counter module at module import time
_token_counter_new_func: Optional[Any] = None


def _get_token_counter_new() -> Any:
    """
    Lazily load and cache the token_counter function (aliased as token_counter_new).
    
    This avoids importing `litellm.litellm_core_utils.token_counter` at `litellm` import time.
    The function is cached after the first import.
    
    This is used internally by utils.py functions that need the token counter but shouldn't
    trigger its import during module load.
    """
    global _token_counter_new_func
    if _token_counter_new_func is None:
        from litellm.litellm_core_utils.token_counter import (
            token_counter as _token_counter_imported,
        )

        _token_counter_new_func = _token_counter_imported
    return _token_counter_new_func


# ============================================================================
# MAIN LAZY IMPORT SYSTEM
# ============================================================================

# This registry maps attribute names (like "ModelResponse") to handler functions
# It's built once the first time someone accesses a lazy-loaded attribute
# Example: {"ModelResponse": _lazy_import_utils, "Cache": _lazy_import_caching, ...}
_LAZY_IMPORT_REGISTRY: Optional[dict[str, Callable[[str], Any]]] = None


def _get_lazy_import_registry() -> dict[str, Callable[[str], Any]]:
    """
    Build the registry that maps attribute names to their handler functions.
    
    This is called once, the first time someone accesses a lazy-loaded attribute.
    After that, we just look up the handler function in this dictionary.
    
    Returns:
        Dictionary like {"ModelResponse": _lazy_import_utils, ...}
    """
    global _LAZY_IMPORT_REGISTRY
    if _LAZY_IMPORT_REGISTRY is None:
        # Build the registry by going through each category and mapping
        # all the names in that category to their handler function
        _LAZY_IMPORT_REGISTRY = {}
        # For each category, map all its names to the handler function
        # Example: All names in UTILS_NAMES get mapped to _lazy_import_utils
        for name in COST_CALCULATOR_NAMES:
            _LAZY_IMPORT_REGISTRY[name] = _lazy_import_cost_calculator
        for name in LITELLM_LOGGING_NAMES:
            _LAZY_IMPORT_REGISTRY[name] = _lazy_import_litellm_logging
        for name in UTILS_NAMES:
            _LAZY_IMPORT_REGISTRY[name] = _lazy_import_utils
        for name in TOKEN_COUNTER_NAMES:
            _LAZY_IMPORT_REGISTRY[name] = _lazy_import_token_counter
        for name in LLM_CLIENT_CACHE_NAMES:
            _LAZY_IMPORT_REGISTRY[name] = _lazy_import_llm_client_cache
        for name in BEDROCK_TYPES_NAMES:
            _LAZY_IMPORT_REGISTRY[name] = _lazy_import_bedrock_types
        for name in TYPES_UTILS_NAMES:
            _LAZY_IMPORT_REGISTRY[name] = _lazy_import_types_utils
        for name in CACHING_NAMES:
            _LAZY_IMPORT_REGISTRY[name] = _lazy_import_caching
        for name in HTTP_HANDLER_NAMES:
            _LAZY_IMPORT_REGISTRY[name] = _lazy_import_http_handlers
        for name in DOTPROMPT_NAMES:
            _LAZY_IMPORT_REGISTRY[name] = _lazy_import_dotprompt
        for name in LLM_CONFIG_NAMES:
            _LAZY_IMPORT_REGISTRY[name] = _lazy_import_llm_configs
        for name in TYPES_NAMES:
            _LAZY_IMPORT_REGISTRY[name] = _lazy_import_types
        for name in LLM_PROVIDER_LOGIC_NAMES:
            _LAZY_IMPORT_REGISTRY[name] = _lazy_import_llm_provider_logic
        for name in UTILS_MODULE_NAMES:
            _LAZY_IMPORT_REGISTRY[name] = _lazy_import_utils_module
    
    return _LAZY_IMPORT_REGISTRY


def _generic_lazy_import(name: str, import_map: dict[str, tuple[str, str]], category: str) -> Any:
    """
    Generic function that handles lazy importing for most attributes.
    
    This is the workhorse function - it does the actual importing and caching.
    Most handler functions just call this with their specific import map.
    
    Steps:
    1. Check if the name exists in the import map (if not, raise error)
    2. Check if we've already imported it (if yes, return cached value)
    3. Look up where to find it (module_path and attr_name from the map)
    4. Import the module (Python caches this automatically)
    5. Get the attribute from the module
    6. Cache it in _globals so we don't import again
    7. Return it
    
    Args:
        name: The attribute name someone is trying to access (e.g., "ModelResponse")
        import_map: Dictionary telling us where to find each attribute
                   Format: {"ModelResponse": (".utils", "ModelResponse")}
        category: Just for error messages (e.g., "Utils", "Cost calculator")
    """
    # Step 1: Make sure this attribute exists in our map
    if name not in import_map:
        raise AttributeError(f"{category} lazy import: unknown attribute {name!r}")
    
    # Step 2: Get the cache (where we store imported things)
    _globals = _get_litellm_globals()
    
    # Step 3: If we've already imported it, just return the cached version
    if name in _globals:
        return _globals[name]
    
    # Step 4: Look up where to find this attribute
    # The map tells us: (module_path, attribute_name)
    # Example: (".utils", "ModelResponse") means "look in .utils module, get ModelResponse"
    module_path, attr_name = import_map[name]
    
    # Step 5: Import the module
    # Python automatically caches modules in sys.modules, so calling this twice is fast
    # If module_path starts with ".", it's a relative import (needs package="litellm")
    # Otherwise it's an absolute import (like "litellm.caching.caching")
    if module_path.startswith("."):
        module = importlib.import_module(module_path, package="litellm")
    else:
        module = importlib.import_module(module_path)
    
    # Step 6: Get the actual attribute from the module
    # Example: getattr(utils_module, "ModelResponse") returns the ModelResponse class
    value = getattr(module, attr_name)
    
    # Step 7: Cache it so we don't have to import again next time
    _globals[name] = value
    
    # Step 8: Return it
    return value


# ============================================================================
# HANDLER FUNCTIONS
# ============================================================================
# These functions are called when someone accesses a lazy-loaded attribute.
# Most of them just call _generic_lazy_import with their specific import map.
# The registry (above) maps attribute names to these handler functions.

def _lazy_import_utils(name: str) -> Any:
    """Handler for utils module attributes (ModelResponse, token_counter, etc.)"""
    return _generic_lazy_import(name, _UTILS_IMPORT_MAP, "Utils")


def _lazy_import_cost_calculator(name: str) -> Any:
    """Handler for cost calculator functions (completion_cost, cost_per_token, etc.)"""
    return _generic_lazy_import(name, _COST_CALCULATOR_IMPORT_MAP, "Cost calculator")


def _lazy_import_token_counter(name: str) -> Any:
    """Handler for token counter utilities"""
    return _generic_lazy_import(name, _TOKEN_COUNTER_IMPORT_MAP, "Token counter")


def _lazy_import_bedrock_types(name: str) -> Any:
    """Handler for Bedrock type aliases"""
    return _generic_lazy_import(name, _BEDROCK_TYPES_IMPORT_MAP, "Bedrock types")


def _lazy_import_types_utils(name: str) -> Any:
    """Handler for types from litellm.types.utils (BudgetConfig, ImageObject, etc.)"""
    return _generic_lazy_import(name, _TYPES_UTILS_IMPORT_MAP, "Types utils")


def _lazy_import_caching(name: str) -> Any:
    """Handler for caching classes (Cache, DualCache, RedisCache, etc.)"""
    return _generic_lazy_import(name, _CACHING_IMPORT_MAP, "Caching")

def _lazy_import_dotprompt(name: str) -> Any:
    """Handler for dotprompt integration globals"""
    return _generic_lazy_import(name, _DOTPROMPT_IMPORT_MAP, "Dotprompt")


def _lazy_import_types(name: str) -> Any:
    """Handler for type classes (GuardrailItem, etc.)"""
    return _generic_lazy_import(name, _TYPES_IMPORT_MAP, "Types")


def _lazy_import_llm_configs(name: str) -> Any:
    """Handler for LLM config classes (AnthropicConfig, OpenAILikeChatConfig, etc.)"""
    return _generic_lazy_import(name, _LLM_CONFIGS_IMPORT_MAP, "LLM config")

def _lazy_import_litellm_logging(name: str) -> Any:
    """Handler for litellm_logging module (Logging, modify_integration)"""
    return _generic_lazy_import(name, _LITELLM_LOGGING_IMPORT_MAP, "Litellm logging")


def _lazy_import_llm_provider_logic(name: str) -> Any:
    """Handler for LLM provider logic functions (get_llm_provider, etc.)"""
    return _generic_lazy_import(name, _LLM_PROVIDER_LOGIC_IMPORT_MAP, "LLM provider logic")


def _lazy_import_utils_module(name: str) -> Any:
    """
    Handler for utils module lazy imports.
    
    This uses a custom implementation because utils module needs to use
    _get_utils_globals() instead of _get_litellm_globals() for caching.
    """
    # Check if this attribute exists in our map
    if name not in _UTILS_MODULE_IMPORT_MAP:
        raise AttributeError(f"Utils module lazy import: unknown attribute {name!r}")
    
    # Get the cache (where we store imported things) - use utils globals
    _globals = _get_utils_globals()
    
    # If we've already imported it, just return the cached version
    if name in _globals:
        return _globals[name]
    
    # Look up where to find this attribute
    module_path, attr_name = _UTILS_MODULE_IMPORT_MAP[name]
    
    # Import the module
    if module_path.startswith("."):
        module = importlib.import_module(module_path, package="litellm")
    else:
        module = importlib.import_module(module_path)
    
    # Get the actual attribute from the module
    value = getattr(module, attr_name)
    
    # Cache it so we don't have to import again next time
    _globals[name] = value
    
    # Return it
    return value

# ============================================================================
# SPECIAL HANDLERS
# ============================================================================
# These handlers have custom logic that doesn't fit the generic pattern

def _lazy_import_llm_client_cache(name: str) -> Any:
    """
    Handler for LLM client cache - has special logic for singleton instance.
    
    This one is different because:
    - "LLMClientCache" is the class itself
    - "in_memory_llm_clients_cache" is a singleton instance of that class
    So we need custom logic to handle both cases.
    """
    _globals = _get_litellm_globals()
    
    # If already cached, return it
    if name in _globals:
        return _globals[name]
    
    # Import the class
    module = importlib.import_module("litellm.caching.llm_caching_handler")
    LLMClientCache = getattr(module, "LLMClientCache")
    
    # If they want the class itself, return it
    if name == "LLMClientCache":
        _globals["LLMClientCache"] = LLMClientCache
        return LLMClientCache
    
    # If they want the singleton instance, create it (only once)
    if name == "in_memory_llm_clients_cache":
        instance = LLMClientCache()
        _globals["in_memory_llm_clients_cache"] = instance
        return instance
    
    raise AttributeError(f"LLM client cache lazy import: unknown attribute {name!r}")


def _lazy_import_http_handlers(name: str) -> Any:
    """
    Handler for HTTP clients - has special logic for creating client instances.
    
    This one is different because:
    - These aren't just imports, they're actual client instances that need to be created
    - They need configuration (timeout, etc.) from the module globals
    - They use factory functions instead of direct instantiation
    """
    _globals = _get_litellm_globals()

    if name == "module_level_aclient":
        # Create an async HTTP client using the factory function
        from litellm.llms.custom_httpx.http_handler import get_async_httpx_client

        # Get timeout from module config (if set)
        timeout = _globals.get("request_timeout")
        params = {"timeout": timeout, "client_alias": "module level aclient"}
        
        # Create the client instance
        provider_id = cast(Any, "litellm_module_level_client")
        async_client = get_async_httpx_client(
            llm_provider=provider_id,
            params=params,
        )
        
        # Cache it so we don't create it again
        _globals["module_level_aclient"] = async_client
        return async_client

    if name == "module_level_client":
        # Create a sync HTTP client
        from litellm.llms.custom_httpx.http_handler import HTTPHandler

        timeout = _globals.get("request_timeout")
        sync_client = HTTPHandler(timeout=timeout)
        
        # Cache it
        _globals["module_level_client"] = sync_client
        return sync_client

    raise AttributeError(f"HTTP handlers lazy import: unknown attribute {name!r}")
