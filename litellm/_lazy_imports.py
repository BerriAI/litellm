"""Lazy import helper functions for litellm module.

This module contains helper functions that handle lazy loading of various
litellm components to reduce import-time memory consumption.
"""
import importlib
import sys
from typing import Any, Callable, Optional


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


def _lazy_import_utils(name: str) -> Any:
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


def _lazy_import_http_handlers(name: str) -> Any:
    """Lazy import for HTTP handler instances and classes - imports only what's needed per name."""
    _globals = _get_litellm_globals()
    _litellm_module = sys.modules["litellm"]
    request_timeout = _litellm_module.request_timeout
    
    # Handle HTTP handler instances
    if name == "module_level_aclient":
        from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler as _AsyncHTTPHandler
        _module_level_aclient = _AsyncHTTPHandler(
            timeout=request_timeout, client_alias="module level aclient"
        )
        _globals["module_level_aclient"] = _module_level_aclient
        return _module_level_aclient
    
    if name == "module_level_client":
        from litellm.llms.custom_httpx.http_handler import HTTPHandler as _HTTPHandler
        _module_level_client = _HTTPHandler(timeout=request_timeout)
        _globals["module_level_client"] = _module_level_client
        return _module_level_client
    
    # Handle HTTP handler classes for backward compatibility
    if name == "AsyncHTTPHandler":
        from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler as _AsyncHTTPHandler
        _globals["AsyncHTTPHandler"] = _AsyncHTTPHandler
        return _AsyncHTTPHandler
    
    if name == "HTTPHandler":
        from litellm.llms.custom_httpx.http_handler import HTTPHandler as _HTTPHandler
        _globals["HTTPHandler"] = _HTTPHandler
        return _HTTPHandler
    
    raise AttributeError(f"HTTP handler lazy import: unknown attribute {name!r}")


def _lazy_import_caching(name: str) -> Any:
    """Lazy import for caching classes - imports only the requested class by name."""
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
    
    if name == "LLMClientCache":
        from litellm.caching.llm_caching_handler import LLMClientCache as _LLMClientCache
        _globals["LLMClientCache"] = _LLMClientCache
        return _LLMClientCache
    
    raise AttributeError(f"Caching lazy import: unknown attribute {name!r}")


def _lazy_import_types_utils(name: str) -> Any:
    """Lazy import for types.utils module - imports only the requested item by name."""
    _globals = _get_litellm_globals()
    if name == "ImageObject":
        from litellm.types.utils import ImageObject as _ImageObject
        _globals["ImageObject"] = _ImageObject
        return _ImageObject
    
    if name == "BudgetConfig":
        from litellm.types.utils import BudgetConfig as _BudgetConfig
        _globals["BudgetConfig"] = _BudgetConfig
        return _BudgetConfig
    
    if name == "all_litellm_params":
        from litellm.types.utils import all_litellm_params as _all_litellm_params
        _globals["all_litellm_params"] = _all_litellm_params
        return _all_litellm_params
    
    if name == "_litellm_completion_params":
        from litellm.types.utils import all_litellm_params as _all_litellm_params
        _globals["_litellm_completion_params"] = _all_litellm_params
        return _all_litellm_params
    
    if name == "CredentialItem":
        from litellm.types.utils import CredentialItem as _CredentialItem
        _globals["CredentialItem"] = _CredentialItem
        return _CredentialItem
    
    if name == "PriorityReservationDict":
        from litellm.types.utils import PriorityReservationDict as _PriorityReservationDict
        _globals["PriorityReservationDict"] = _PriorityReservationDict
        return _PriorityReservationDict
    
    if name == "StandardKeyGenerationConfig":
        from litellm.types.utils import StandardKeyGenerationConfig as _StandardKeyGenerationConfig
        _globals["StandardKeyGenerationConfig"] = _StandardKeyGenerationConfig
        return _StandardKeyGenerationConfig
    
    if name == "LlmProviders":
        from litellm.types.utils import LlmProviders as _LlmProviders
        _globals["LlmProviders"] = _LlmProviders
        return _LlmProviders
    
    if name == "SearchProviders":
        from litellm.types.utils import SearchProviders as _SearchProviders
        _globals["SearchProviders"] = _SearchProviders
        return _SearchProviders
    
    if name == "PriorityReservationSettings":
        from litellm.types.utils import PriorityReservationSettings as _PriorityReservationSettings
        _globals["PriorityReservationSettings"] = _PriorityReservationSettings
        return _PriorityReservationSettings
    
    raise AttributeError(f"Types utils lazy import: unknown attribute {name!r}")


def _lazy_import_ui_sso(name: str) -> Any:
    """Lazy import for types.proxy.management_endpoints.ui_sso module - imports only the requested item by name."""
    _globals = _get_litellm_globals()
    if name == "DefaultTeamSSOParams":
        from litellm.types.proxy.management_endpoints.ui_sso import DefaultTeamSSOParams as _DefaultTeamSSOParams
        _globals["DefaultTeamSSOParams"] = _DefaultTeamSSOParams
        return _DefaultTeamSSOParams
    
    if name == "LiteLLM_UpperboundKeyGenerateParams":
        from litellm.types.proxy.management_endpoints.ui_sso import LiteLLM_UpperboundKeyGenerateParams as _LiteLLM_UpperboundKeyGenerateParams
        _globals["LiteLLM_UpperboundKeyGenerateParams"] = _LiteLLM_UpperboundKeyGenerateParams
        return _LiteLLM_UpperboundKeyGenerateParams
    
    raise AttributeError(f"UI SSO lazy import: unknown attribute {name!r}")


def _lazy_import_secret_managers(name: str) -> Any:
    """Lazy import for types.secret_managers.main module - imports only the requested item by name."""
    _globals = _get_litellm_globals()
    if name == "KeyManagementSystem":
        from litellm.types.secret_managers.main import KeyManagementSystem as _KeyManagementSystem
        _globals["KeyManagementSystem"] = _KeyManagementSystem
        return _KeyManagementSystem
    
    if name == "KeyManagementSettings":
        from litellm.types.secret_managers.main import KeyManagementSettings as _KeyManagementSettings
        _globals["KeyManagementSettings"] = _KeyManagementSettings
        return _KeyManagementSettings
    
    raise AttributeError(f"Secret managers lazy import: unknown attribute {name!r}")


def _lazy_import_logging_integrations(name: str) -> Any:
    """Lazy import for logging-related integrations - imports only the requested item by name."""
    _globals = _get_litellm_globals()
    if name == "CustomLogger":
        from litellm.integrations.custom_logger import CustomLogger as _CustomLogger
        _globals["CustomLogger"] = _CustomLogger
        return _CustomLogger
    
    if name == "LoggingCallbackManager":
        from litellm.litellm_core_utils.logging_callback_manager import LoggingCallbackManager as _LoggingCallbackManager
        _globals["LoggingCallbackManager"] = _LoggingCallbackManager
        return _LoggingCallbackManager
    
    raise AttributeError(f"Logging integrations lazy import: unknown attribute {name!r}")

def _lazy_import_nvidia_nim_configs(name: str) -> Any:
    """Lazy import for NvidiaNim config classes - imports only the requested class."""
    _globals = _get_litellm_globals()
    if name == "NvidiaNimConfig":
        from .llms.nvidia_nim.chat.transformation import NvidiaNimConfig as _NvidiaNimConfig
        _globals["NvidiaNimConfig"] = _NvidiaNimConfig
        return _NvidiaNimConfig
    
    if name == "nvidiaNimConfig":
        from .llms.nvidia_nim.chat.transformation import NvidiaNimConfig as _NvidiaNimConfig
        _nvidiaNimConfig = _NvidiaNimConfig()
        _globals["NvidiaNimConfig"] = _NvidiaNimConfig
        _globals["nvidiaNimConfig"] = _nvidiaNimConfig
        return _nvidiaNimConfig
    
    raise AttributeError(f"NvidiaNim configs lazy import: unknown attribute {name!r}")

def _lazy_import_openai_gpt_configs(name: str) -> Any:
    """Lazy import for OpenAI GPT config classes - imports only the requested class."""
    _globals = _get_litellm_globals()
    if name == "OpenAIGPTConfig":
        from .llms.openai.chat.gpt_transformation import OpenAIGPTConfig as _OpenAIGPTConfig
        _globals["OpenAIGPTConfig"] = _OpenAIGPTConfig
        return _OpenAIGPTConfig
    
    if name == "openAIGPTConfig":
        from .llms.openai.chat.gpt_transformation import OpenAIGPTConfig as _OpenAIGPTConfig
        _openAIGPTConfig = _OpenAIGPTConfig()
        _globals["OpenAIGPTConfig"] = _OpenAIGPTConfig
        _globals["openAIGPTConfig"] = _openAIGPTConfig
        return _openAIGPTConfig
    
    if name == "OpenAIGPT5Config":
        from .llms.openai.chat.gpt_5_transformation import OpenAIGPT5Config as _OpenAIGPT5Config
        _globals["OpenAIGPT5Config"] = _OpenAIGPT5Config
        return _OpenAIGPT5Config
    
    if name == "openAIGPT5Config":
        from .llms.openai.chat.gpt_5_transformation import OpenAIGPT5Config as _OpenAIGPT5Config
        _openAIGPT5Config = _OpenAIGPT5Config()
        _globals["OpenAIGPT5Config"] = _OpenAIGPT5Config
        _globals["openAIGPT5Config"] = _openAIGPT5Config
        return _openAIGPT5Config
    
    if name == "OpenAIGPTAudioConfig":
        from .llms.openai.chat.gpt_audio_transformation import OpenAIGPTAudioConfig as _OpenAIGPTAudioConfig
        _globals["OpenAIGPTAudioConfig"] = _OpenAIGPTAudioConfig
        return _OpenAIGPTAudioConfig
    
    if name == "openAIGPTAudioConfig":
        from .llms.openai.chat.gpt_audio_transformation import OpenAIGPTAudioConfig as _OpenAIGPTAudioConfig
        _openAIGPTAudioConfig = _OpenAIGPTAudioConfig()
        _globals["OpenAIGPTAudioConfig"] = _OpenAIGPTAudioConfig
        _globals["openAIGPTAudioConfig"] = _openAIGPTAudioConfig
        return _openAIGPTAudioConfig
    
    raise AttributeError(f"OpenAI GPT configs lazy import: unknown attribute {name!r}")

# Lazy import helper functions are imported inside __getattr__ to avoid any import-time overhead

def _lazy_import_dotprompt(name: str) -> Any:
    """Lazy import for dotprompt module - imports only the requested item by name."""
    _globals = _get_litellm_globals()
    if name == "global_prompt_manager":
        from litellm.integrations.dotprompt import global_prompt_manager as _global_prompt_manager
        _globals["global_prompt_manager"] = _global_prompt_manager
        return _global_prompt_manager
    
    if name == "global_prompt_directory":
        from litellm.integrations.dotprompt import global_prompt_directory as _global_prompt_directory
        _globals["global_prompt_directory"] = _global_prompt_directory
        return _global_prompt_directory
    
    if name == "set_global_prompt_directory":
        from litellm.integrations.dotprompt import set_global_prompt_directory as _set_global_prompt_directory
        _globals["set_global_prompt_directory"] = _set_global_prompt_directory
        return _set_global_prompt_directory
    
    raise AttributeError(f"Dotprompt lazy import: unknown attribute {name!r}")


def _lazy_import_type_items(name: str) -> Any:
    """Lazy import for type-related items - imports only the requested item by name."""
    _globals = _get_litellm_globals()
    if name == "COHERE_EMBEDDING_INPUT_TYPES":
        from litellm.types.llms.bedrock import COHERE_EMBEDDING_INPUT_TYPES as _COHERE_EMBEDDING_INPUT_TYPES
        _globals["COHERE_EMBEDDING_INPUT_TYPES"] = _COHERE_EMBEDDING_INPUT_TYPES
        return _COHERE_EMBEDDING_INPUT_TYPES
    
    if name == "GuardrailItem":
        from litellm.types.guardrails import GuardrailItem as _GuardrailItem
        _globals["GuardrailItem"] = _GuardrailItem
        return _GuardrailItem
    
    raise AttributeError(f"Type items lazy import: unknown attribute {name!r}")


def _lazy_import_core_helpers(name: str) -> Any:
    """Lazy import for core helper functions - imports only the requested item by name."""
    _globals = _get_litellm_globals()
    if name == "remove_index_from_tool_calls":
        from litellm.litellm_core_utils.core_helpers import remove_index_from_tool_calls as _remove_index_from_tool_calls
        _globals["remove_index_from_tool_calls"] = _remove_index_from_tool_calls
        return _remove_index_from_tool_calls
    
    raise AttributeError(f"Core helpers lazy import: unknown attribute {name!r}")


def _lazy_import_openai_like_configs(name: str) -> Any:
    """Lazy import for OpenAI-like config classes - imports only the requested class."""
    _globals = _get_litellm_globals()
    if name == "OpenAILikeChatConfig":
        from .llms.openai_like.chat.handler import OpenAILikeChatConfig as _OpenAILikeChatConfig
        _globals["OpenAILikeChatConfig"] = _OpenAILikeChatConfig
        return _OpenAILikeChatConfig
    
    if name == "AiohttpOpenAIChatConfig":
        from .llms.aiohttp_openai.chat.transformation import AiohttpOpenAIChatConfig as _AiohttpOpenAIChatConfig
        _globals["AiohttpOpenAIChatConfig"] = _AiohttpOpenAIChatConfig
        return _AiohttpOpenAIChatConfig
    
    raise AttributeError(f"OpenAI-like configs lazy import: unknown attribute {name!r}")


def _lazy_import_small_provider_chat_configs(name: str) -> Any:
    """Lazy import for smaller provider chat config classes - imports only the requested class."""
    _globals = _get_litellm_globals()
    if name == "GaladrielChatConfig":
        from .llms.galadriel.chat.transformation import GaladrielChatConfig as _GaladrielChatConfig
        _globals["GaladrielChatConfig"] = _GaladrielChatConfig
        return _GaladrielChatConfig
    
    if name == "GithubChatConfig":
        from .llms.github.chat.transformation import GithubChatConfig as _GithubChatConfig
        _globals["GithubChatConfig"] = _GithubChatConfig
        return _GithubChatConfig
    
    if name == "CompactifAIChatConfig":
        from .llms.compactifai.chat.transformation import CompactifAIChatConfig as _CompactifAIChatConfig
        _globals["CompactifAIChatConfig"] = _CompactifAIChatConfig
        return _CompactifAIChatConfig
    
    if name == "EmpowerChatConfig":
        from .llms.empower.chat.transformation import EmpowerChatConfig as _EmpowerChatConfig
        _globals["EmpowerChatConfig"] = _EmpowerChatConfig
        return _EmpowerChatConfig
    
    if name == "FeatherlessAIConfig":
        from .llms.featherless_ai.chat.transformation import FeatherlessAIConfig as _FeatherlessAIConfig
        _globals["FeatherlessAIConfig"] = _FeatherlessAIConfig
        return _FeatherlessAIConfig
    
    if name == "CerebrasConfig":
        from .llms.cerebras.chat import CerebrasConfig as _CerebrasConfig
        _globals["CerebrasConfig"] = _CerebrasConfig
        return _CerebrasConfig
    
    if name == "BasetenConfig":
        from .llms.baseten.chat import BasetenConfig as _BasetenConfig
        _globals["BasetenConfig"] = _BasetenConfig
        return _BasetenConfig
    
    if name == "SambanovaConfig":
        from .llms.sambanova.chat import SambanovaConfig as _SambanovaConfig
        _globals["SambanovaConfig"] = _SambanovaConfig
        return _SambanovaConfig
    
    if name == "FireworksAIConfig":
        from .llms.fireworks_ai.chat.transformation import FireworksAIConfig as _FireworksAIConfig
        _globals["FireworksAIConfig"] = _FireworksAIConfig
        return _FireworksAIConfig
    
    if name == "FriendliaiChatConfig":
        from .llms.friendliai.chat.transformation import FriendliaiChatConfig as _FriendliaiChatConfig
        _globals["FriendliaiChatConfig"] = _FriendliaiChatConfig
        return _FriendliaiChatConfig
    
    if name == "XAIChatConfig":
        from .llms.xai.chat.transformation import XAIChatConfig as _XAIChatConfig
        _globals["XAIChatConfig"] = _XAIChatConfig
        return _XAIChatConfig
    
    if name == "AIMLChatConfig":
        from .llms.aiml.chat.transformation import AIMLChatConfig as _AIMLChatConfig
        _globals["AIMLChatConfig"] = _AIMLChatConfig
        return _AIMLChatConfig
    
    if name == "VolcEngineConfig":
        from .llms.volcengine.chat.transformation import VolcEngineChatConfig as _VolcEngineChatConfig
        _globals["VolcEngineChatConfig"] = _VolcEngineChatConfig
        _globals["VolcEngineConfig"] = _VolcEngineChatConfig  # alias
        return _VolcEngineChatConfig
    
    if name == "VolcEngineChatConfig":
        from .llms.volcengine.chat.transformation import VolcEngineChatConfig as _VolcEngineChatConfig
        _globals["VolcEngineChatConfig"] = _VolcEngineChatConfig
        _globals["VolcEngineConfig"] = _VolcEngineChatConfig  # alias
        return _VolcEngineChatConfig
    
    if name == "HerokuChatConfig":
        from .llms.heroku.chat.transformation import HerokuChatConfig as _HerokuChatConfig
        _globals["HerokuChatConfig"] = _HerokuChatConfig
        return _HerokuChatConfig
    
    if name == "CometAPIConfig":
        from .llms.cometapi.chat.transformation import CometAPIConfig as _CometAPIConfig
        _globals["CometAPIConfig"] = _CometAPIConfig
        return _CometAPIConfig
    
    if name == "HostedVLLMChatConfig":
        from .llms.hosted_vllm.chat.transformation import HostedVLLMChatConfig as _HostedVLLMChatConfig
        _globals["HostedVLLMChatConfig"] = _HostedVLLMChatConfig
        return _HostedVLLMChatConfig
    
    if name == "LlamafileChatConfig":
        from .llms.llamafile.chat.transformation import LlamafileChatConfig as _LlamafileChatConfig
        _globals["LlamafileChatConfig"] = _LlamafileChatConfig
        return _LlamafileChatConfig
    
    if name == "LiteLLMProxyChatConfig":
        from .llms.litellm_proxy.chat.transformation import LiteLLMProxyChatConfig as _LiteLLMProxyChatConfig
        _globals["LiteLLMProxyChatConfig"] = _LiteLLMProxyChatConfig
        return _LiteLLMProxyChatConfig
    
    if name == "DeepSeekChatConfig":
        from .llms.deepseek.chat.transformation import DeepSeekChatConfig as _DeepSeekChatConfig
        _globals["DeepSeekChatConfig"] = _DeepSeekChatConfig
        return _DeepSeekChatConfig
    
    if name == "LMStudioChatConfig":
        from .llms.lm_studio.chat.transformation import LMStudioChatConfig as _LMStudioChatConfig
        _globals["LMStudioChatConfig"] = _LMStudioChatConfig
        return _LMStudioChatConfig
    
    if name == "NscaleConfig":
        from .llms.nscale.chat.transformation import NscaleConfig as _NscaleConfig
        _globals["NscaleConfig"] = _NscaleConfig
        return _NscaleConfig
    
    if name == "PerplexityChatConfig":
        from .llms.perplexity.chat.transformation import PerplexityChatConfig as _PerplexityChatConfig
        _globals["PerplexityChatConfig"] = _PerplexityChatConfig
        return _PerplexityChatConfig
    
    if name == "IBMWatsonXChatConfig":
        from .llms.watsonx.chat.transformation import IBMWatsonXChatConfig as _IBMWatsonXChatConfig
        _globals["IBMWatsonXChatConfig"] = _IBMWatsonXChatConfig
        return _IBMWatsonXChatConfig
    
    if name == "GithubCopilotConfig":
        from .llms.github_copilot.chat.transformation import GithubCopilotConfig as _GithubCopilotConfig
        _globals["GithubCopilotConfig"] = _GithubCopilotConfig
        return _GithubCopilotConfig
    
    if name == "NebiusConfig":
        from .llms.nebius.chat.transformation import NebiusConfig as _NebiusConfig
        _globals["NebiusConfig"] = _NebiusConfig
        return _NebiusConfig
    
    if name == "WandbConfig":
        from .llms.wandb.chat.transformation import WandbConfig as _WandbConfig
        _globals["WandbConfig"] = _WandbConfig
        return _WandbConfig
    
    if name == "DashScopeChatConfig":
        from .llms.dashscope.chat.transformation import DashScopeChatConfig as _DashScopeChatConfig
        _globals["DashScopeChatConfig"] = _DashScopeChatConfig
        return _DashScopeChatConfig
    
    if name == "MoonshotChatConfig":
        from .llms.moonshot.chat.transformation import MoonshotChatConfig as _MoonshotChatConfig
        _globals["MoonshotChatConfig"] = _MoonshotChatConfig
        return _MoonshotChatConfig
    
    if name == "DockerModelRunnerChatConfig":
        from .llms.docker_model_runner.chat.transformation import DockerModelRunnerChatConfig as _DockerModelRunnerChatConfig
        _globals["DockerModelRunnerChatConfig"] = _DockerModelRunnerChatConfig
        return _DockerModelRunnerChatConfig
    
    if name == "V0ChatConfig":
        from .llms.v0.chat.transformation import V0ChatConfig as _V0ChatConfig
        _globals["V0ChatConfig"] = _V0ChatConfig
        return _V0ChatConfig
    
    if name == "OCIChatConfig":
        from .llms.oci.chat.transformation import OCIChatConfig as _OCIChatConfig
        _globals["OCIChatConfig"] = _OCIChatConfig
        return _OCIChatConfig
    
    if name == "MorphChatConfig":
        from .llms.morph.chat.transformation import MorphChatConfig as _MorphChatConfig
        _globals["MorphChatConfig"] = _MorphChatConfig
        return _MorphChatConfig
    
    if name == "LambdaAIChatConfig":
        from .llms.lambda_ai.chat.transformation import LambdaAIChatConfig as _LambdaAIChatConfig
        _globals["LambdaAIChatConfig"] = _LambdaAIChatConfig
        return _LambdaAIChatConfig
    
    if name == "HyperbolicChatConfig":
        from .llms.hyperbolic.chat.transformation import HyperbolicChatConfig as _HyperbolicChatConfig
        _globals["HyperbolicChatConfig"] = _HyperbolicChatConfig
        return _HyperbolicChatConfig
    
    if name == "VercelAIGatewayConfig":
        from .llms.vercel_ai_gateway.chat.transformation import VercelAIGatewayConfig as _VercelAIGatewayConfig
        _globals["VercelAIGatewayConfig"] = _VercelAIGatewayConfig
        return _VercelAIGatewayConfig
    
    if name == "OVHCloudChatConfig":
        from .llms.ovhcloud.chat.transformation import OVHCloudChatConfig as _OVHCloudChatConfig
        _globals["OVHCloudChatConfig"] = _OVHCloudChatConfig
        return _OVHCloudChatConfig
    
    if name == "LemonadeChatConfig":
        from .llms.lemonade.chat.transformation import LemonadeChatConfig as _LemonadeChatConfig
        _globals["LemonadeChatConfig"] = _LemonadeChatConfig
        return _LemonadeChatConfig
    
    raise AttributeError(f"Small provider chat configs lazy import: unknown attribute {name!r}")


def _lazy_import_data_platform_configs(name: str) -> Any:
    """Lazy import for data platform provider chat config classes - imports only the requested class."""
    _globals = _get_litellm_globals()
    if name == "DatabricksConfig":
        from .llms.databricks.chat.transformation import DatabricksConfig as _DatabricksConfig
        _globals["DatabricksConfig"] = _DatabricksConfig
        return _DatabricksConfig
    
    if name == "PredibaseConfig":
        from .llms.predibase.chat.transformation import PredibaseConfig as _PredibaseConfig
        _globals["PredibaseConfig"] = _PredibaseConfig
        return _PredibaseConfig
    
    if name == "SnowflakeConfig":
        from .llms.snowflake.chat.transformation import SnowflakeConfig as _SnowflakeConfig
        _globals["SnowflakeConfig"] = _SnowflakeConfig
        return _SnowflakeConfig
    
    raise AttributeError(f"Data platform configs lazy import: unknown attribute {name!r}")


def _lazy_import_huggingface_configs(name: str) -> Any:
    """Lazy import for HuggingFace config classes - imports only the requested class."""
    _globals = _get_litellm_globals()
    if name == "HuggingFaceChatConfig":
        from .llms.huggingface.chat.transformation import HuggingFaceChatConfig as _HuggingFaceChatConfig
        _globals["HuggingFaceChatConfig"] = _HuggingFaceChatConfig
        return _HuggingFaceChatConfig
    
    if name == "HuggingFaceEmbeddingConfig":
        from .llms.huggingface.embedding.transformation import HuggingFaceEmbeddingConfig as _HuggingFaceEmbeddingConfig
        _globals["HuggingFaceEmbeddingConfig"] = _HuggingFaceEmbeddingConfig
        return _HuggingFaceEmbeddingConfig
    
    raise AttributeError(f"HuggingFace configs lazy import: unknown attribute {name!r}")


def _lazy_import_anthropic_configs(name: str) -> Any:
    """Lazy import for Anthropic config classes - imports only the requested class."""
    _globals = _get_litellm_globals()
    if name == "AnthropicConfig":
        from .llms.anthropic.chat.transformation import AnthropicConfig as _AnthropicConfig
        _globals["AnthropicConfig"] = _AnthropicConfig
        return _AnthropicConfig
    
    if name == "AnthropicTextConfig":
        from .llms.anthropic.completion.transformation import AnthropicTextConfig as _AnthropicTextConfig
        _globals["AnthropicTextConfig"] = _AnthropicTextConfig
        return _AnthropicTextConfig
    
    if name == "AnthropicMessagesConfig":
        from .llms.anthropic.experimental_pass_through.messages.transformation import AnthropicMessagesConfig as _AnthropicMessagesConfig
        _globals["AnthropicMessagesConfig"] = _AnthropicMessagesConfig
        return _AnthropicMessagesConfig
    
    raise AttributeError(f"Anthropic configs lazy import: unknown attribute {name!r}")


def _lazy_import_triton_configs(name: str) -> Any:
    """Lazy import for Triton config classes - imports only the requested class."""
    _globals = _get_litellm_globals()
    if name == "TritonConfig":
        from .llms.triton.completion.transformation import TritonConfig as _TritonConfig
        _globals["TritonConfig"] = _TritonConfig
        return _TritonConfig
    
    if name == "TritonEmbeddingConfig":
        from .llms.triton.embedding.transformation import TritonEmbeddingConfig as _TritonEmbeddingConfig
        _globals["TritonEmbeddingConfig"] = _TritonEmbeddingConfig
        return _TritonEmbeddingConfig
    
    raise AttributeError(f"Triton configs lazy import: unknown attribute {name!r}")


def _lazy_import_ai21_configs(name: str) -> Any:
    """Lazy import for AI21 config classes - imports only the requested class."""
    _globals = _get_litellm_globals()
    if name == "AI21ChatConfig":
        from .llms.ai21.chat.transformation import AI21ChatConfig as _AI21ChatConfig
        _globals["AI21ChatConfig"] = _AI21ChatConfig
        _globals["AI21Config"] = _AI21ChatConfig  # alias
        return _AI21ChatConfig
    
    if name == "AI21Config":
        from .llms.ai21.chat.transformation import AI21ChatConfig as _AI21ChatConfig
        _globals["AI21ChatConfig"] = _AI21ChatConfig
        _globals["AI21Config"] = _AI21ChatConfig  # alias
        return _AI21ChatConfig
    
    raise AttributeError(f"AI21 configs lazy import: unknown attribute {name!r}")


def _lazy_import_ollama_configs(name: str) -> Any:
    """Lazy import for Ollama config classes - imports only the requested class."""
    _globals = _get_litellm_globals()
    if name == "OllamaChatConfig":
        from .llms.ollama.chat.transformation import OllamaChatConfig as _OllamaChatConfig
        _globals["OllamaChatConfig"] = _OllamaChatConfig
        return _OllamaChatConfig
    
    if name == "OllamaConfig":
        from .llms.ollama.completion.transformation import OllamaConfig as _OllamaConfig
        _globals["OllamaConfig"] = _OllamaConfig
        return _OllamaConfig
    
    raise AttributeError(f"Ollama configs lazy import: unknown attribute {name!r}")


def _lazy_import_sagemaker_configs(name: str) -> Any:
    """Lazy import for Sagemaker config classes - imports only the requested class."""
    _globals = _get_litellm_globals()
    if name == "SagemakerConfig":
        from .llms.sagemaker.completion.transformation import SagemakerConfig as _SagemakerConfig
        _globals["SagemakerConfig"] = _SagemakerConfig
        return _SagemakerConfig
    
    if name == "SagemakerChatConfig":
        from .llms.sagemaker.chat.transformation import SagemakerChatConfig as _SagemakerChatConfig
        _globals["SagemakerChatConfig"] = _SagemakerChatConfig
        return _SagemakerChatConfig
    
    raise AttributeError(f"Sagemaker configs lazy import: unknown attribute {name!r}")


def _lazy_import_cohere_chat_configs(name: str) -> Any:
    """Lazy import for Cohere chat config classes - imports only the requested class."""
    _globals = _get_litellm_globals()
    if name == "CohereChatConfig":
        from .llms.cohere.chat.transformation import CohereChatConfig as _CohereChatConfig
        _globals["CohereChatConfig"] = _CohereChatConfig
        return _CohereChatConfig
    
    if name == "CohereV2ChatConfig":
        from .llms.cohere.chat.v2_transformation import CohereV2ChatConfig as _CohereV2ChatConfig
        _globals["CohereV2ChatConfig"] = _CohereV2ChatConfig
        return _CohereV2ChatConfig
    
    raise AttributeError(f"Cohere chat configs lazy import: unknown attribute {name!r}")


def _lazy_import_rerank_configs(name: str) -> Any:
    """Lazy import for rerank config classes - imports only the requested class."""
    _globals = _get_litellm_globals()
    if name == "HuggingFaceRerankConfig":
        from .llms.huggingface.rerank.transformation import HuggingFaceRerankConfig as _HuggingFaceRerankConfig
        _globals["HuggingFaceRerankConfig"] = _HuggingFaceRerankConfig
        return _HuggingFaceRerankConfig
    
    if name == "CohereRerankConfig":
        from .llms.cohere.rerank.transformation import CohereRerankConfig as _CohereRerankConfig
        _globals["CohereRerankConfig"] = _CohereRerankConfig
        return _CohereRerankConfig
    
    if name == "CohereRerankV2Config":
        from .llms.cohere.rerank_v2.transformation import CohereRerankV2Config as _CohereRerankV2Config
        _globals["CohereRerankV2Config"] = _CohereRerankV2Config
        return _CohereRerankV2Config
    
    if name == "AzureAIRerankConfig":
        from .llms.azure_ai.rerank.transformation import AzureAIRerankConfig as _AzureAIRerankConfig
        _globals["AzureAIRerankConfig"] = _AzureAIRerankConfig
        return _AzureAIRerankConfig
    
    if name == "InfinityRerankConfig":
        from .llms.infinity.rerank.transformation import InfinityRerankConfig as _InfinityRerankConfig
        _globals["InfinityRerankConfig"] = _InfinityRerankConfig
        return _InfinityRerankConfig
    
    if name == "JinaAIRerankConfig":
        from .llms.jina_ai.rerank.transformation import JinaAIRerankConfig as _JinaAIRerankConfig
        _globals["JinaAIRerankConfig"] = _JinaAIRerankConfig
        return _JinaAIRerankConfig
    
    if name == "DeepinfraRerankConfig":
        from .llms.deepinfra.rerank.transformation import DeepinfraRerankConfig as _DeepinfraRerankConfig
        _globals["DeepinfraRerankConfig"] = _DeepinfraRerankConfig
        return _DeepinfraRerankConfig
    
    if name == "HostedVLLMRerankConfig":
        from .llms.hosted_vllm.rerank.transformation import HostedVLLMRerankConfig as _HostedVLLMRerankConfig
        _globals["HostedVLLMRerankConfig"] = _HostedVLLMRerankConfig
        return _HostedVLLMRerankConfig
    
    if name == "NvidiaNimRerankConfig":
        from .llms.nvidia_nim.rerank.transformation import NvidiaNimRerankConfig as _NvidiaNimRerankConfig
        _globals["NvidiaNimRerankConfig"] = _NvidiaNimRerankConfig
        return _NvidiaNimRerankConfig
    
    if name == "VertexAIRerankConfig":
        from .llms.vertex_ai.rerank.transformation import VertexAIRerankConfig as _VertexAIRerankConfig
        _globals["VertexAIRerankConfig"] = _VertexAIRerankConfig
        return _VertexAIRerankConfig
    
    raise AttributeError(f"Rerank configs lazy import: unknown attribute {name!r}")


def _lazy_import_vertex_ai_configs(name: str) -> Any:
    """Lazy import for Vertex AI config classes - imports only the requested class."""
    _globals = _get_litellm_globals()
    if name == "VertexGeminiConfig":
        from .llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import VertexGeminiConfig as _VertexGeminiConfig
        _globals["VertexGeminiConfig"] = _VertexGeminiConfig
        _globals["VertexAIConfig"] = _VertexGeminiConfig  # alias
        return _VertexGeminiConfig
    
    if name == "VertexAIConfig":
        from .llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import VertexGeminiConfig as _VertexGeminiConfig
        _globals["VertexGeminiConfig"] = _VertexGeminiConfig
        _globals["VertexAIConfig"] = _VertexGeminiConfig  # alias
        return _VertexGeminiConfig
    
    if name == "GoogleAIStudioGeminiConfig":
        from .llms.gemini.chat.transformation import GoogleAIStudioGeminiConfig as _GoogleAIStudioGeminiConfig
        _globals["GoogleAIStudioGeminiConfig"] = _GoogleAIStudioGeminiConfig
        _globals["GeminiConfig"] = _GoogleAIStudioGeminiConfig  # alias
        return _GoogleAIStudioGeminiConfig
    
    if name == "GeminiConfig":
        from .llms.gemini.chat.transformation import GoogleAIStudioGeminiConfig as _GoogleAIStudioGeminiConfig
        _globals["GoogleAIStudioGeminiConfig"] = _GoogleAIStudioGeminiConfig
        _globals["GeminiConfig"] = _GoogleAIStudioGeminiConfig  # alias
        return _GoogleAIStudioGeminiConfig
    
    if name == "VertexAIAnthropicConfig":
        from .llms.vertex_ai.vertex_ai_partner_models.anthropic.transformation import VertexAIAnthropicConfig as _VertexAIAnthropicConfig
        _globals["VertexAIAnthropicConfig"] = _VertexAIAnthropicConfig
        return _VertexAIAnthropicConfig
    
    if name == "VertexAILlama3Config":
        from .llms.vertex_ai.vertex_ai_partner_models.llama3.transformation import VertexAILlama3Config as _VertexAILlama3Config
        _globals["VertexAILlama3Config"] = _VertexAILlama3Config
        return _VertexAILlama3Config
    
    if name == "VertexAIAi21Config":
        from .llms.vertex_ai.vertex_ai_partner_models.ai21.transformation import VertexAIAi21Config as _VertexAIAi21Config
        _globals["VertexAIAi21Config"] = _VertexAIAi21Config
        return _VertexAIAi21Config
    
    raise AttributeError(f"Vertex AI configs lazy import: unknown attribute {name!r}")


def _lazy_import_amazon_bedrock_configs(name: str) -> Any:
    """Lazy import for Amazon Bedrock config classes - imports only the requested class."""
    _globals = _get_litellm_globals()
    if name == "AmazonCohereChatConfig":
        from .llms.bedrock.chat.invoke_handler import AmazonCohereChatConfig as _AmazonCohereChatConfig
        _globals["AmazonCohereChatConfig"] = _AmazonCohereChatConfig
        return _AmazonCohereChatConfig
    
    if name == "AmazonBedrockGlobalConfig":
        from .llms.bedrock.common_utils import AmazonBedrockGlobalConfig as _AmazonBedrockGlobalConfig
        _globals["AmazonBedrockGlobalConfig"] = _AmazonBedrockGlobalConfig
        return _AmazonBedrockGlobalConfig
    
    if name == "AmazonAI21Config":
        from .llms.bedrock.chat.invoke_transformations.amazon_ai21_transformation import AmazonAI21Config as _AmazonAI21Config
        _globals["AmazonAI21Config"] = _AmazonAI21Config
        return _AmazonAI21Config
    
    if name == "AmazonAnthropicConfig":
        from .llms.bedrock.chat.invoke_transformations.anthropic_claude2_transformation import AmazonAnthropicConfig as _AmazonAnthropicConfig
        _globals["AmazonAnthropicConfig"] = _AmazonAnthropicConfig
        return _AmazonAnthropicConfig
    
    if name == "AmazonAnthropicClaudeConfig":
        from .llms.bedrock.chat.invoke_transformations.anthropic_claude3_transformation import AmazonAnthropicClaudeConfig as _AmazonAnthropicClaudeConfig
        _globals["AmazonAnthropicClaudeConfig"] = _AmazonAnthropicClaudeConfig
        return _AmazonAnthropicClaudeConfig
    
    if name == "AmazonTitanG1Config":
        from .llms.bedrock.embed.amazon_titan_g1_transformation import AmazonTitanG1Config as _AmazonTitanG1Config
        _globals["AmazonTitanG1Config"] = _AmazonTitanG1Config
        return _AmazonTitanG1Config
    
    if name == "AmazonTitanMultimodalEmbeddingG1Config":
        from .llms.bedrock.embed.amazon_titan_multimodal_transformation import AmazonTitanMultimodalEmbeddingG1Config as _AmazonTitanMultimodalEmbeddingG1Config
        _globals["AmazonTitanMultimodalEmbeddingG1Config"] = _AmazonTitanMultimodalEmbeddingG1Config
        return _AmazonTitanMultimodalEmbeddingG1Config
    
    if name == "AmazonTitanV2Config":
        from .llms.bedrock.embed.amazon_titan_v2_transformation import AmazonTitanV2Config as _AmazonTitanV2Config
        _globals["AmazonTitanV2Config"] = _AmazonTitanV2Config
        return _AmazonTitanV2Config
    
    if name == "BedrockCohereEmbeddingConfig":
        from .llms.bedrock.embed.cohere_transformation import BedrockCohereEmbeddingConfig as _BedrockCohereEmbeddingConfig
        _globals["BedrockCohereEmbeddingConfig"] = _BedrockCohereEmbeddingConfig
        return _BedrockCohereEmbeddingConfig
    
    raise AttributeError(f"Amazon Bedrock configs lazy import: unknown attribute {name!r}")


def _lazy_import_deprecated_provider_configs(name: str) -> Any:
    """Lazy import for deprecated provider config classes - imports only the requested class."""
    _globals = _get_litellm_globals()
    if name == "PalmConfig":
        from .llms.deprecated_providers.palm import PalmConfig as _PalmConfig
        _globals["PalmConfig"] = _PalmConfig
        return _PalmConfig
    
    if name == "AlephAlphaConfig":
        from .llms.deprecated_providers.aleph_alpha import AlephAlphaConfig as _AlephAlphaConfig
        _globals["AlephAlphaConfig"] = _AlephAlphaConfig
        return _AlephAlphaConfig
    
    raise AttributeError(f"Deprecated provider configs lazy import: unknown attribute {name!r}")


def _lazy_import_azure_responses_configs(name: str) -> Any:
    """Lazy import for Azure OpenAI Responses API config classes - imports only the requested class."""
    _globals = _get_litellm_globals()
    if name == "AzureOpenAIResponsesAPIConfig":
        from .llms.azure.responses.transformation import AzureOpenAIResponsesAPIConfig as _AzureOpenAIResponsesAPIConfig
        _globals["AzureOpenAIResponsesAPIConfig"] = _AzureOpenAIResponsesAPIConfig
        return _AzureOpenAIResponsesAPIConfig
    
    if name == "AzureOpenAIOSeriesResponsesAPIConfig":
        from .llms.azure.responses.o_series_transformation import AzureOpenAIOSeriesResponsesAPIConfig as _AzureOpenAIOSeriesResponsesAPIConfig
        _globals["AzureOpenAIOSeriesResponsesAPIConfig"] = _AzureOpenAIOSeriesResponsesAPIConfig
        return _AzureOpenAIOSeriesResponsesAPIConfig
    
    if name == "GithubCopilotResponsesAPIConfig":
        from .llms.github_copilot.responses.transformation import GithubCopilotResponsesAPIConfig as _GithubCopilotResponsesAPIConfig
        _globals["GithubCopilotResponsesAPIConfig"] = _GithubCopilotResponsesAPIConfig
        return _GithubCopilotResponsesAPIConfig
    
    raise AttributeError(f"Azure Responses API configs lazy import: unknown attribute {name!r}")


def _lazy_import_azure_openai_configs(name: str) -> Any:
    """Lazy import for Azure OpenAI config classes - imports only the requested class."""
    _globals = _get_litellm_globals()
    if name == "AzureOpenAIConfig":
        from .llms.azure.chat.gpt_transformation import AzureOpenAIConfig as _AzureOpenAIConfig
        _globals["AzureOpenAIConfig"] = _AzureOpenAIConfig
        return _AzureOpenAIConfig
    
    if name == "AzureOpenAIGPT5Config":
        from .llms.azure.chat.gpt_5_transformation import AzureOpenAIGPT5Config as _AzureOpenAIGPT5Config
        _globals["AzureOpenAIGPT5Config"] = _AzureOpenAIGPT5Config
        return _AzureOpenAIGPT5Config
    
    if name == "AzureOpenAITextConfig":
        from .llms.azure.completion.transformation import AzureOpenAITextConfig as _AzureOpenAITextConfig
        _globals["AzureOpenAITextConfig"] = _AzureOpenAITextConfig
        return _AzureOpenAITextConfig
    
    if name == "AzureOpenAIAssistantsAPIConfig":
        from .llms.azure.azure import AzureOpenAIAssistantsAPIConfig as _AzureOpenAIAssistantsAPIConfig
        _globals["AzureOpenAIAssistantsAPIConfig"] = _AzureOpenAIAssistantsAPIConfig
        return _AzureOpenAIAssistantsAPIConfig
    
    raise AttributeError(f"Azure OpenAI configs lazy import: unknown attribute {name!r}")


def _lazy_import_openai_o_series_configs(name: str) -> Any:
    """Lazy import for OpenAI O-Series config classes - imports only the requested class."""
    _globals = _get_litellm_globals()
    if name == "OpenAIOSeriesConfig":
        from .llms.openai.chat.o_series_transformation import OpenAIOSeriesConfig as _OpenAIOSeriesConfig
        _globals["OpenAIOSeriesConfig"] = _OpenAIOSeriesConfig
        return _OpenAIOSeriesConfig
    
    if name == "OpenAIO1Config":
        from .llms.openai.chat.o_series_transformation import OpenAIOSeriesConfig as _OpenAIOSeriesConfig
        _globals["OpenAIOSeriesConfig"] = _OpenAIOSeriesConfig
        _globals["OpenAIO1Config"] = _OpenAIOSeriesConfig  # alias
        return _OpenAIOSeriesConfig
    
    if name == "openaiOSeriesConfig":
        from .llms.openai.chat.o_series_transformation import OpenAIOSeriesConfig as _OpenAIOSeriesConfig
        _openaiOSeriesConfig = _OpenAIOSeriesConfig()
        _globals["OpenAIOSeriesConfig"] = _OpenAIOSeriesConfig
        _globals["openaiOSeriesConfig"] = _openaiOSeriesConfig
        return _openaiOSeriesConfig
    
    raise AttributeError(f"OpenAI O-Series configs lazy import: unknown attribute {name!r}")



def _lazy_import_misc_transformation_configs(name: str) -> Any:
    """Lazy import for miscellaneous transformation config classes - imports only the requested class."""
    _globals = _get_litellm_globals()
    if name == "DeepInfraConfig":
        from .llms.deepinfra.chat.transformation import DeepInfraConfig as _DeepInfraConfig
        _globals["DeepInfraConfig"] = _DeepInfraConfig
        return _DeepInfraConfig
    
    if name == "GroqChatConfig":
        from .llms.groq.chat.transformation import GroqChatConfig as _GroqChatConfig
        _globals["GroqChatConfig"] = _GroqChatConfig
        return _GroqChatConfig
    
    if name == "VoyageEmbeddingConfig":
        from .llms.voyage.embedding.transformation import VoyageEmbeddingConfig as _VoyageEmbeddingConfig
        _globals["VoyageEmbeddingConfig"] = _VoyageEmbeddingConfig
        return _VoyageEmbeddingConfig
    
    if name == "InfinityEmbeddingConfig":
        from .llms.infinity.embedding.transformation import InfinityEmbeddingConfig as _InfinityEmbeddingConfig
        _globals["InfinityEmbeddingConfig"] = _InfinityEmbeddingConfig
        return _InfinityEmbeddingConfig
    
    if name == "AzureAIStudioConfig":
        from .llms.azure_ai.chat.transformation import AzureAIStudioConfig as _AzureAIStudioConfig
        _globals["AzureAIStudioConfig"] = _AzureAIStudioConfig
        return _AzureAIStudioConfig
    
    if name == "MistralConfig":
        from .llms.mistral.chat.transformation import MistralConfig as _MistralConfig
        _globals["MistralConfig"] = _MistralConfig
        return _MistralConfig
    
    if name == "SambaNovaEmbeddingConfig":
        from .llms.sambanova.embedding.transformation import SambaNovaEmbeddingConfig as _SambaNovaEmbeddingConfig
        _globals["SambaNovaEmbeddingConfig"] = _SambaNovaEmbeddingConfig
        return _SambaNovaEmbeddingConfig
    
    if name == "FireworksAITextCompletionConfig":
        from .llms.fireworks_ai.completion.transformation import FireworksAITextCompletionConfig as _FireworksAITextCompletionConfig
        _globals["FireworksAITextCompletionConfig"] = _FireworksAITextCompletionConfig
        return _FireworksAITextCompletionConfig
    
    if name == "JinaAIEmbeddingConfig":
        from .llms.jina_ai.embedding.transformation import JinaAIEmbeddingConfig as _JinaAIEmbeddingConfig
        _globals["JinaAIEmbeddingConfig"] = _JinaAIEmbeddingConfig
        return _JinaAIEmbeddingConfig
    
    if name == "CodestralTextCompletionConfig":
        from .llms.codestral.completion.transformation import CodestralTextCompletionConfig as _CodestralTextCompletionConfig
        _globals["CodestralTextCompletionConfig"] = _CodestralTextCompletionConfig
        return _CodestralTextCompletionConfig
    
    if name == "VLLMConfig":
        from .llms.vllm.completion.transformation import VLLMConfig as _VLLMConfig
        _globals["VLLMConfig"] = _VLLMConfig
        return _VLLMConfig
    
    if name == "IBMWatsonXAIConfig":
        from .llms.watsonx.completion.transformation import IBMWatsonXAIConfig as _IBMWatsonXAIConfig
        _globals["IBMWatsonXAIConfig"] = _IBMWatsonXAIConfig
        return _IBMWatsonXAIConfig
    
    if name == "LmStudioEmbeddingConfig":
        from .llms.lm_studio.embed.transformation import LmStudioEmbeddingConfig as _LmStudioEmbeddingConfig
        _globals["LmStudioEmbeddingConfig"] = _LmStudioEmbeddingConfig
        return _LmStudioEmbeddingConfig
    
    if name == "IBMWatsonXEmbeddingConfig":
        from .llms.watsonx.embed.transformation import IBMWatsonXEmbeddingConfig as _IBMWatsonXEmbeddingConfig
        _globals["IBMWatsonXEmbeddingConfig"] = _IBMWatsonXEmbeddingConfig
        return _IBMWatsonXEmbeddingConfig
    
    if name == "OVHCloudEmbeddingConfig":
        from .llms.ovhcloud.embedding.transformation import OVHCloudEmbeddingConfig as _OVHCloudEmbeddingConfig
        _globals["OVHCloudEmbeddingConfig"] = _OVHCloudEmbeddingConfig
        return _OVHCloudEmbeddingConfig
    
    if name == "CometAPIEmbeddingConfig":
        from .llms.cometapi.embed.transformation import CometAPIEmbeddingConfig as _CometAPIEmbeddingConfig
        _globals["CometAPIEmbeddingConfig"] = _CometAPIEmbeddingConfig
        return _CometAPIEmbeddingConfig
    
    if name == "SnowflakeEmbeddingConfig":
        from .llms.snowflake.embedding.transformation import SnowflakeEmbeddingConfig as _SnowflakeEmbeddingConfig
        _globals["SnowflakeEmbeddingConfig"] = _SnowflakeEmbeddingConfig
        return _SnowflakeEmbeddingConfig
    
    raise AttributeError(f"Misc transformation configs lazy import: unknown attribute {name!r}")

def _lazy_import_main_functions(name: str) -> Any:
    """Lazy import for main module functions and classes - dynamically imports from main.
    
    Optimized to check if module is already loaded before importing, and uses importlib
    for better clarity. Note: Python's import system doesn't support partial imports,
    so the entire litellm.main module will be loaded on first access.
    """
    _globals = _get_litellm_globals()
    
    # Check if module is already loaded to avoid re-importing
    main_module_name = "litellm.main"
    main_module = sys.modules.get(main_module_name)
    
    if main_module is None:
        # Only import if not already loaded
        try:
            main_module = importlib.import_module(main_module_name)
        except ImportError as e:
            raise AttributeError(f"Failed to lazy import {name!r} from litellm.main: {e}") from e
    
    # Get the requested attribute - use try/except for EAFP (more Pythonic)
    try:
        attr = getattr(main_module, name)
        _globals[name] = attr
        return attr
    except AttributeError:
        raise AttributeError(f"module 'litellm.main' has no attribute {name!r}")


# Module-level cache for hot-path functions to avoid repeated import overhead
_get_llm_provider_cached: Optional[Callable] = None


def get_cached_llm_provider() -> Callable:
    """
    Get cached get_llm_provider function with lazy loading.
    This avoids repeated import overhead in hot-path functions.
    
    This is a shared utility for modules that need to call get_llm_provider
    frequently (e.g., routing, realtime API) without the overhead of
    repeated imports or __getattr__ lookups.
    
    Returns:
        The get_llm_provider function
    """
    global _get_llm_provider_cached
    if _get_llm_provider_cached is None:
        from litellm import get_llm_provider
        _get_llm_provider_cached = get_llm_provider
    return _get_llm_provider_cached