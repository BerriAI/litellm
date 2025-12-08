from typing import Any
import sys

def _get_litellm_globals() -> dict:
    """Helper to get the globals dictionary of the litellm module."""
    return sys.modules["litellm"].__dict__

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
    from .cost_calculator import (
        completion_cost as _completion_cost,
        cost_per_token as _cost_per_token,
        response_cost_calculator as _response_cost_calculator,
    )
    
    _cost_functions = {
        "completion_cost": _completion_cost,
        "cost_per_token": _cost_per_token,
        "response_cost_calculator": _response_cost_calculator,
    }
    
    func = _cost_functions[name]
    _globals[name] = func
    return func


def _lazy_import_litellm_logging(name: str) -> Any:
    """Lazy import for litellm_logging module."""
    _globals = _get_litellm_globals()
    try:
        from litellm.litellm_core_utils.litellm_logging import (
            Logging as _Logging,
            modify_integration as _modify_integration,
        )
        
        _logging_objects = {
            "Logging": _Logging,
            "modify_integration": _modify_integration,
        }
        
        obj = _logging_objects[name]
        _globals[name] = obj
        return obj
    except Exception as e:
        raise AttributeError(
            f"module 'litellm' has no attribute {name!r}. "
            f"Lazy import failed: {e}"
        ) from e