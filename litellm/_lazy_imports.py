import sys
from typing import Any, Optional, cast, Callable

# Import all name tuples and import maps from the registry
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
)


def _get_litellm_globals() -> dict:
    """Helper to get the globals dictionary of the litellm module."""
    return sys.modules["litellm"].__dict__

# Lazy loader for default encoding to avoid importing tiktoken at module import time
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


# Cached registry for lazy imports - built once on first access
# Maps attribute names to their handler function
_LAZY_IMPORT_REGISTRY: Optional[dict[str, Callable[[str], Any]]] = None


def _get_lazy_import_registry() -> dict[str, Callable[[str], Any]]:
    """
    Build and cache the lazy import registry (only once).
    
    Returns a dictionary mapping attribute names to their handler functions.
    This avoids importing all name tuples on every __getattr__ call.
    """
    global _LAZY_IMPORT_REGISTRY
    if _LAZY_IMPORT_REGISTRY is None:
        # Build unified registry mapping names directly to handler functions
        # All name tuples and handler functions are already in this module
        _LAZY_IMPORT_REGISTRY = {}
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
    
    return _LAZY_IMPORT_REGISTRY



# Lazy import for utils module - imports only the requested item by name.
def _lazy_import_utils(name: str) -> Any:
    """Lazy import for utils module - imports only the requested item by name."""
    if name not in _UTILS_IMPORT_MAP:
        raise AttributeError(f"Utils lazy import: unknown attribute {name!r}")
    
    _globals = _get_litellm_globals()
    
    # Check if already cached
    if name in _globals:
        return _globals[name]
    
    module_path, attr_name = _UTILS_IMPORT_MAP[name]
    
    # Cache module reference to avoid repeated importlib calls
    _module_cache_key = f"_cached_module_{module_path}"
    if _module_cache_key not in _globals:
        import importlib
        _globals[_module_cache_key] = importlib.import_module(module_path, package="litellm")
    
    module = _globals[_module_cache_key]
    value = getattr(module, attr_name)
    
    _globals[name] = value
    return value


def _lazy_import_cost_calculator(name: str) -> Any:
    """Lazy import for cost_calculator functions."""
    if name not in _COST_CALCULATOR_IMPORT_MAP:
        raise AttributeError(f"Cost calculator lazy import: unknown attribute {name!r}")
    
    _globals = _get_litellm_globals()
    
    # Check if already cached
    if name in _globals:
        return _globals[name]
    
    module_path, attr_name = _COST_CALCULATOR_IMPORT_MAP[name]
    
    # Cache module reference to avoid repeated importlib calls
    _module_cache_key = f"_cached_module_{module_path}"
    if _module_cache_key not in _globals:
        import importlib
        _globals[_module_cache_key] = importlib.import_module(module_path, package="litellm")
    
    module = _globals[_module_cache_key]
    value = getattr(module, attr_name)
    
    _globals[name] = value
    return value


def _lazy_import_token_counter(name: str) -> Any:
    """Lazy import for token_counter utilities."""
    if name not in _TOKEN_COUNTER_IMPORT_MAP:
        raise AttributeError(f"Token counter lazy import: unknown attribute {name!r}")
    
    _globals = _get_litellm_globals()
    
    # Check if already cached
    if name in _globals:
        return _globals[name]
    
    module_path, attr_name = _TOKEN_COUNTER_IMPORT_MAP[name]
    
    # Cache module reference to avoid repeated importlib calls
    _module_cache_key = f"_cached_module_{module_path}"
    if _module_cache_key not in _globals:
        import importlib
        _globals[_module_cache_key] = importlib.import_module(module_path)
    
    module = _globals[_module_cache_key]
    value = getattr(module, attr_name)
    
    _globals[name] = value
    return value


def _lazy_import_bedrock_types(name: str) -> Any:
    """Lazy import for Bedrock type aliases."""
    if name not in _BEDROCK_TYPES_IMPORT_MAP:
        raise AttributeError(f"Bedrock types lazy import: unknown attribute {name!r}")
    
    _globals = _get_litellm_globals()
    
    # Check if already cached
    if name in _globals:
        return _globals[name]
    
    module_path, attr_name = _BEDROCK_TYPES_IMPORT_MAP[name]
    
    # Cache module reference to avoid repeated importlib calls
    _module_cache_key = f"_cached_module_{module_path}"
    if _module_cache_key not in _globals:
        import importlib
        _globals[_module_cache_key] = importlib.import_module(module_path)
    
    module = _globals[_module_cache_key]
    value = getattr(module, attr_name)
    
    _globals[name] = value
    return value


def _lazy_import_types_utils(name: str) -> Any:
    """Lazy import for common types and constants from litellm.types.utils."""
    if name not in _TYPES_UTILS_IMPORT_MAP:
        raise AttributeError(f"Types utils lazy import: unknown attribute {name!r}")
    
    _globals = _get_litellm_globals()
    
    # Check if already cached
    if name in _globals:
        return _globals[name]
    
    module_path, attr_name = _TYPES_UTILS_IMPORT_MAP[name]
    
    # Cache module reference to avoid repeated importlib calls
    _module_cache_key = f"_cached_module_{module_path}"
    if _module_cache_key not in _globals:
        import importlib
        _globals[_module_cache_key] = importlib.import_module(module_path, package="litellm")
    
    module = _globals[_module_cache_key]
    value = getattr(module, attr_name)
    
    _globals[name] = value
    return value


def _lazy_import_caching(name: str) -> Any:
    """Lazy import for caching module classes."""
    if name not in _CACHING_IMPORT_MAP:
        raise AttributeError(f"Caching lazy import: unknown attribute {name!r}")
    
    _globals = _get_litellm_globals()
    
    # Check if already cached
    if name in _globals:
        return _globals[name]
    
    module_path, attr_name = _CACHING_IMPORT_MAP[name]
    
    # Cache module reference to avoid repeated importlib calls
    _module_cache_key = f"_cached_module_{module_path}"
    if _module_cache_key not in _globals:
        import importlib
        _globals[_module_cache_key] = importlib.import_module(module_path)
    
    module = _globals[_module_cache_key]
    value = getattr(module, attr_name)
    
    _globals[name] = value
    return value


def _lazy_import_llm_client_cache(name: str) -> Any:
    """Lazy import for LLM client cache class and singleton."""
    _globals = _get_litellm_globals()
    
    # Check if already cached
    if name in _globals:
        return _globals[name]
    
    # Import the class
    module_path = "litellm.caching.llm_caching_handler"
    _module_cache_key = f"_cached_module_{module_path}"
    if _module_cache_key not in _globals:
        import importlib
        _globals[_module_cache_key] = importlib.import_module(module_path)
    
    module = _globals[_module_cache_key]
    LLMClientCache = getattr(module, "LLMClientCache")
    
    if name == "LLMClientCache":
        _globals["LLMClientCache"] = LLMClientCache
        return LLMClientCache
    
    if name == "in_memory_llm_clients_cache":
        instance = LLMClientCache()
        # Only populate the requested singleton name to keep lazy-import
        # semantics consistent with other helpers (no extra symbols).
        _globals["in_memory_llm_clients_cache"] = instance
        return instance
    
    raise AttributeError(f"LLM client cache lazy import: unknown attribute {name!r}")


def _lazy_import_litellm_logging(name: str) -> Any:
    """Lazy import for litellm_logging module."""
    if name not in _LITELLM_LOGGING_IMPORT_MAP:
        raise AttributeError(f"Litellm logging lazy import: unknown attribute {name!r}")
    
    _globals = _get_litellm_globals()
    
    # Check if already cached
    if name in _globals:
        return _globals[name]
    
    module_path, attr_name = _LITELLM_LOGGING_IMPORT_MAP[name]
    
    # Cache module reference to avoid repeated importlib calls
    _module_cache_key = f"_cached_module_{module_path}"
    if _module_cache_key not in _globals:
        import importlib
        _globals[_module_cache_key] = importlib.import_module(module_path)
    
    module = _globals[_module_cache_key]
    value = getattr(module, attr_name)
    
    _globals[name] = value
    return value


def _lazy_import_http_handlers(name: str) -> Any:
    """Lazy import and instantiate module-level HTTP handlers."""
    _globals = _get_litellm_globals()

    if name == "module_level_aclient":
        # Use shared async client factory instead of directly instantiating AsyncHTTPHandler
        from litellm.llms.custom_httpx.http_handler import get_async_httpx_client

        timeout = _globals.get("request_timeout")
        params = {"timeout": timeout, "client_alias": "module level aclient"}
        # llm_provider is only used for cache keying; use a string identifier but
        # cast to Any so static type checkers don't complain about the literal.
        provider_id = cast(Any, "litellm_module_level_client")
        async_client = get_async_httpx_client(
            llm_provider=provider_id,
            params=params,
        )
        _globals["module_level_aclient"] = async_client
        return async_client

    if name == "module_level_client":
        # Import handler type locally to avoid heavy imports at module load time
        from litellm.llms.custom_httpx.http_handler import HTTPHandler

        timeout = _globals.get("request_timeout")
        sync_client = HTTPHandler(timeout=timeout)
        _globals["module_level_client"] = sync_client
        return sync_client

    raise AttributeError(f"HTTP handlers lazy import: unknown attribute {name!r}")


def _lazy_import_dotprompt(name: str) -> Any:
    """Lazy import for dotprompt integration globals."""
    if name not in _DOTPROMPT_IMPORT_MAP:
        raise AttributeError(f"Dotprompt lazy import: unknown attribute {name!r}")
    
    _globals = _get_litellm_globals()
    
    # Check if already cached
    if name in _globals:
        return _globals[name]
    
    module_path, attr_name = _DOTPROMPT_IMPORT_MAP[name]
    
    # Cache module reference to avoid repeated importlib calls
    _module_cache_key = f"_cached_module_{module_path}"
    if _module_cache_key not in _globals:
        import importlib
        _globals[_module_cache_key] = importlib.import_module(module_path)
    
    module = _globals[_module_cache_key]
    value = getattr(module, attr_name)
    
    _globals[name] = value
    return value


def _lazy_import_types(name: str) -> Any:
    """Lazy import for type classes."""
    if name not in _TYPES_IMPORT_MAP:
        raise AttributeError(f"Types lazy import: unknown attribute {name!r}")
    
    _globals = _get_litellm_globals()
    
    # Check if already cached
    if name in _globals:
        return _globals[name]
    
    module_path, attr_name = _TYPES_IMPORT_MAP[name]
    
    # Cache module reference to avoid repeated importlib calls
    _module_cache_key = f"_cached_module_{module_path}"
    if _module_cache_key not in _globals:
        import importlib
        _globals[_module_cache_key] = importlib.import_module(module_path)
    
    module = _globals[_module_cache_key]
    value = getattr(module, attr_name)
    
    _globals[name] = value
    return value


def _lazy_import_llm_configs(name: str) -> Any:
    """Lazy import for LLM config classes."""
    if name not in _LLM_CONFIGS_IMPORT_MAP:
        raise AttributeError(f"LLM config lazy import: unknown attribute {name!r}")
    
    _globals = _get_litellm_globals()
    
    # Check if already cached
    if name in _globals:
        return _globals[name]
    
    module_path, attr_name = _LLM_CONFIGS_IMPORT_MAP[name]
    
    # Cache module reference to avoid repeated importlib calls
    _module_cache_key = f"_cached_module_{module_path}"
    if _module_cache_key not in _globals:
        import importlib
        _globals[_module_cache_key] = importlib.import_module(module_path, package="litellm")
    
    module = _globals[_module_cache_key]
    value = getattr(module, attr_name)
    
    _globals[name] = value
    return value