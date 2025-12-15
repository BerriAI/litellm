from typing import Any, cast
import sys

def _get_litellm_globals() -> dict:
    """Helper to get the globals dictionary of the litellm module."""
    return sys.modules["litellm"].__dict__

# Cost calculator names that support lazy loading via _lazy_import_cost_calculator
COST_CALCULATOR_NAMES = (
    "completion_cost",
    "cost_per_token",
    "response_cost_calculator",
)

# Litellm logging names that support lazy loading via _lazy_import_litellm_logging
LITELLM_LOGGING_NAMES = (
    "Logging",
    "modify_integration",
)

# Utils names that support lazy loading via _lazy_import_utils
UTILS_NAMES = (
    "exception_type", "get_optional_params", "get_response_string", "token_counter",
    "create_pretrained_tokenizer", "create_tokenizer", "supports_function_calling",
    "supports_web_search", "supports_url_context", "supports_response_schema",
    "supports_parallel_function_calling", "supports_vision", "supports_audio_input",
    "supports_audio_output", "supports_system_messages", "supports_reasoning",
    "get_litellm_params", "acreate", "get_max_tokens", "get_model_info",
    "register_prompt_template", "validate_environment", "check_valid_key",
    "register_model", "encode", "decode", "_calculate_retry_after", "_should_retry",
    "get_supported_openai_params", "get_api_base", "get_first_chars_messages",
    "ModelResponse", "ModelResponseStream", "EmbeddingResponse", "ImageResponse",
    "TranscriptionResponse", "TextCompletionResponse", "get_provider_fields",
    "ModelResponseListIterator", "get_valid_models",
)

# Caching / cache classes that support lazy loading via _lazy_import_caching
CACHING_NAMES = (
    "Cache",
    "DualCache",
    "RedisCache",
    "InMemoryCache",
)

# HTTP handler names that support lazy loading via _lazy_import_http_handlers
HTTP_HANDLER_NAMES = (
    "module_level_aclient",
    "module_level_client",
)

# Lazy import for utils module - imports only the requested item by name.
# Note: PLR0915 (too many statements) is suppressed because the many if statements
# are intentional - each attribute is imported individually only when requested,
# ensuring true lazy imports rather than importing the entire utils module.
def _lazy_import_utils(name: str) -> Any:  # noqa: PLR0915
    """Lazy import for utils module - imports only the requested item by name."""
    _globals = _get_litellm_globals()
    if name == "exception_type":
        from .utils import exception_type as _exception_type
        _globals["exception_type"] = _exception_type
        return _exception_type
    
    if name == "get_optional_params":
        from .utils import get_optional_params as _get_optional_params
        _globals["get_optional_params"] = _get_optional_params
        return _get_optional_params
    
    if name == "get_response_string":
        from .utils import get_response_string as _get_response_string
        _globals["get_response_string"] = _get_response_string
        return _get_response_string
    
    if name == "token_counter":
        from .utils import token_counter as _token_counter
        _globals["token_counter"] = _token_counter
        return _token_counter
    
    if name == "create_pretrained_tokenizer":
        from .utils import create_pretrained_tokenizer as _create_pretrained_tokenizer
        _globals["create_pretrained_tokenizer"] = _create_pretrained_tokenizer
        return _create_pretrained_tokenizer
    
    if name == "create_tokenizer":
        from .utils import create_tokenizer as _create_tokenizer
        _globals["create_tokenizer"] = _create_tokenizer
        return _create_tokenizer
    
    if name == "supports_function_calling":
        from .utils import supports_function_calling as _supports_function_calling
        _globals["supports_function_calling"] = _supports_function_calling
        return _supports_function_calling
    
    if name == "supports_web_search":
        from .utils import supports_web_search as _supports_web_search
        _globals["supports_web_search"] = _supports_web_search
        return _supports_web_search
    
    if name == "supports_url_context":
        from .utils import supports_url_context as _supports_url_context
        _globals["supports_url_context"] = _supports_url_context
        return _supports_url_context
    
    if name == "supports_response_schema":
        from .utils import supports_response_schema as _supports_response_schema
        _globals["supports_response_schema"] = _supports_response_schema
        return _supports_response_schema
    
    if name == "supports_parallel_function_calling":
        from .utils import supports_parallel_function_calling as _supports_parallel_function_calling
        _globals["supports_parallel_function_calling"] = _supports_parallel_function_calling
        return _supports_parallel_function_calling
    
    if name == "supports_vision":
        from .utils import supports_vision as _supports_vision
        _globals["supports_vision"] = _supports_vision
        return _supports_vision
    
    if name == "supports_audio_input":
        from .utils import supports_audio_input as _supports_audio_input
        _globals["supports_audio_input"] = _supports_audio_input
        return _supports_audio_input
    
    if name == "supports_audio_output":
        from .utils import supports_audio_output as _supports_audio_output
        _globals["supports_audio_output"] = _supports_audio_output
        return _supports_audio_output
    
    if name == "supports_system_messages":
        from .utils import supports_system_messages as _supports_system_messages
        _globals["supports_system_messages"] = _supports_system_messages
        return _supports_system_messages
    
    if name == "supports_reasoning":
        from .utils import supports_reasoning as _supports_reasoning
        _globals["supports_reasoning"] = _supports_reasoning
        return _supports_reasoning
    
    if name == "get_litellm_params":
        from .utils import get_litellm_params as _get_litellm_params
        _globals["get_litellm_params"] = _get_litellm_params
        return _get_litellm_params
    
    if name == "acreate":
        from .utils import acreate as _acreate
        _globals["acreate"] = _acreate
        return _acreate
    
    if name == "get_max_tokens":
        from .utils import get_max_tokens as _get_max_tokens
        _globals["get_max_tokens"] = _get_max_tokens
        return _get_max_tokens
    
    if name == "get_model_info":
        from .utils import get_model_info as _get_model_info
        _globals["get_model_info"] = _get_model_info
        return _get_model_info
    
    if name == "register_prompt_template":
        from .utils import register_prompt_template as _register_prompt_template
        _globals["register_prompt_template"] = _register_prompt_template
        return _register_prompt_template
    
    if name == "validate_environment":
        from .utils import validate_environment as _validate_environment
        _globals["validate_environment"] = _validate_environment
        return _validate_environment
    
    if name == "check_valid_key":
        from .utils import check_valid_key as _check_valid_key
        _globals["check_valid_key"] = _check_valid_key
        return _check_valid_key
    
    if name == "register_model":
        from .utils import register_model as _register_model
        _globals["register_model"] = _register_model
        return _register_model
    
    if name == "encode":
        from .utils import encode as _encode
        _globals["encode"] = _encode
        return _encode
    
    if name == "decode":
        from .utils import decode as _decode
        _globals["decode"] = _decode
        return _decode
    
    if name == "_calculate_retry_after":
        from .utils import _calculate_retry_after as __calculate_retry_after
        _globals["_calculate_retry_after"] = __calculate_retry_after
        return __calculate_retry_after
    
    if name == "_should_retry":
        from .utils import _should_retry as __should_retry
        _globals["_should_retry"] = __should_retry
        return __should_retry
    
    if name == "get_supported_openai_params":
        from .utils import get_supported_openai_params as _get_supported_openai_params
        _globals["get_supported_openai_params"] = _get_supported_openai_params
        return _get_supported_openai_params
    
    if name == "get_api_base":
        from .utils import get_api_base as _get_api_base
        _globals["get_api_base"] = _get_api_base
        return _get_api_base
    
    if name == "get_first_chars_messages":
        from .utils import get_first_chars_messages as _get_first_chars_messages
        _globals["get_first_chars_messages"] = _get_first_chars_messages
        return _get_first_chars_messages
    
    if name == "ModelResponse":
        from .utils import ModelResponse as _ModelResponse
        _globals["ModelResponse"] = _ModelResponse
        return _ModelResponse
    
    if name == "ModelResponseStream":
        from .utils import ModelResponseStream as _ModelResponseStream
        _globals["ModelResponseStream"] = _ModelResponseStream
        return _ModelResponseStream
    
    if name == "EmbeddingResponse":
        from .utils import EmbeddingResponse as _EmbeddingResponse
        _globals["EmbeddingResponse"] = _EmbeddingResponse
        return _EmbeddingResponse
    
    if name == "ImageResponse":
        from .utils import ImageResponse as _ImageResponse
        _globals["ImageResponse"] = _ImageResponse
        return _ImageResponse
    
    if name == "TranscriptionResponse":
        from .utils import TranscriptionResponse as _TranscriptionResponse
        _globals["TranscriptionResponse"] = _TranscriptionResponse
        return _TranscriptionResponse
    
    if name == "TextCompletionResponse":
        from .utils import TextCompletionResponse as _TextCompletionResponse
        _globals["TextCompletionResponse"] = _TextCompletionResponse
        return _TextCompletionResponse
    
    if name == "get_provider_fields":
        from .utils import get_provider_fields as _get_provider_fields
        _globals["get_provider_fields"] = _get_provider_fields
        return _get_provider_fields
    
    if name == "ModelResponseListIterator":
        from .utils import ModelResponseListIterator as _ModelResponseListIterator
        _globals["ModelResponseListIterator"] = _ModelResponseListIterator
        return _ModelResponseListIterator
    
    if name == "get_valid_models":
        from .utils import get_valid_models as _get_valid_models
        _globals["get_valid_models"] = _get_valid_models
        return _get_valid_models
    
    raise AttributeError(f"Utils lazy import: unknown attribute {name!r}")


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


def _lazy_import_caching(name: str) -> Any:
    """Lazy import for caching module classes."""
    _globals = _get_litellm_globals()

    if name == "Cache":
        from litellm.caching.caching import Cache as _Cache

        _globals["Cache"] = _Cache
        return _Cache

    if name == "DualCache":
        from litellm.caching.caching import DualCache as _DualCache

        _globals["DualCache"] = _DualCache
        return _DualCache

    if name == "RedisCache":
        from litellm.caching.caching import RedisCache as _RedisCache

        _globals["RedisCache"] = _RedisCache
        return _RedisCache

    if name == "InMemoryCache":
        from litellm.caching.caching import InMemoryCache as _InMemoryCache

        _globals["InMemoryCache"] = _InMemoryCache
        return _InMemoryCache

    raise AttributeError(f"Caching lazy import: unknown attribute {name!r}")


def _lazy_import_litellm_logging(name: str) -> Any:
    """Lazy import for litellm_logging module."""
    _globals = _get_litellm_globals()
    if name == "Logging":
        from litellm.litellm_core_utils.litellm_logging import Logging as _Logging
        _globals["Logging"] = _Logging
        return _Logging
    
    if name == "modify_integration":
        from litellm.litellm_core_utils.litellm_logging import modify_integration as _modify_integration
        _globals["modify_integration"] = _modify_integration
        return _modify_integration
    
    raise AttributeError(f"Litellm logging lazy import: unknown attribute {name!r}")


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