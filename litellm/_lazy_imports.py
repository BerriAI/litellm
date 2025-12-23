import sys
from typing import Any, Optional, cast, Callable


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

# Token counter names that support lazy loading via _lazy_import_token_counter
TOKEN_COUNTER_NAMES = (
    "get_modified_max_tokens",
)

# LLM client cache names that support lazy loading via _lazy_import_llm_client_cache
LLM_CLIENT_CACHE_NAMES = (
    "LLMClientCache",
    "in_memory_llm_clients_cache",
)

# Bedrock type names that support lazy loading via _lazy_import_bedrock_types
BEDROCK_TYPES_NAMES = (
    "COHERE_EMBEDDING_INPUT_TYPES",
)

# Common types from litellm.types.utils that support lazy loading via
# _lazy_import_types_utils
TYPES_UTILS_NAMES = (
    "ImageObject",
    "BudgetConfig",
    "all_litellm_params",
    "_litellm_completion_params",
    "CredentialItem",
    "PriorityReservationDict",
    "StandardKeyGenerationConfig",
    "SearchProviders",
    "GenericStreamingChunk",
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

# Dotprompt integration names that support lazy loading via _lazy_import_dotprompt
DOTPROMPT_NAMES = (
    "global_prompt_manager",
    "global_prompt_directory",
    "set_global_prompt_directory",
)

# LLM config classes that support lazy loading via _lazy_import_llm_configs
LLM_CONFIG_NAMES = (
    "AmazonConverseConfig",
    "OpenAILikeChatConfig",
    "GaladrielChatConfig",
    "GithubChatConfig",
    "AzureAnthropicConfig",
    "BytezChatConfig",
    "CompactifAIChatConfig",
    "EmpowerChatConfig",
    "MinimaxChatConfig",
    "AiohttpOpenAIChatConfig",
    "HuggingFaceChatConfig",
    "HuggingFaceEmbeddingConfig",
    "OobaboogaConfig",
    "MaritalkConfig",
    "OpenrouterConfig",
    "DataRobotConfig",
    "AnthropicConfig",
    "AnthropicTextConfig",
    "GroqSTTConfig",
    "TritonConfig",
    "TritonGenerateConfig",
    "TritonInferConfig",
    "TritonEmbeddingConfig",
    "HuggingFaceRerankConfig",
    "DatabricksConfig",
    "DatabricksEmbeddingConfig",
    "PredibaseConfig",
    "ReplicateConfig",
    "SnowflakeConfig",
    "CohereRerankConfig",
    "CohereRerankV2Config",
    "AzureAIRerankConfig",
    "InfinityRerankConfig",
    "JinaAIRerankConfig",
    "DeepinfraRerankConfig",
    "HostedVLLMRerankConfig",
    "NvidiaNimRerankConfig",
    "NvidiaNimRankingConfig",
    "VertexAIRerankConfig",
    "FireworksAIRerankConfig",
    "VoyageRerankConfig",
    "ClarifaiConfig",
    "AI21ChatConfig",
)

# Types that support lazy loading via _lazy_import_types
TYPES_NAMES = (
    "GuardrailItem",
)

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


# Import maps for registry pattern - reduces repetition
_UTILS_IMPORT_MAP = {
    "exception_type": (".utils", "exception_type"),
    "get_optional_params": (".utils", "get_optional_params"),
    "get_response_string": (".utils", "get_response_string"),
    "token_counter": (".utils", "token_counter"),
    "create_pretrained_tokenizer": (".utils", "create_pretrained_tokenizer"),
    "create_tokenizer": (".utils", "create_tokenizer"),
    "supports_function_calling": (".utils", "supports_function_calling"),
    "supports_web_search": (".utils", "supports_web_search"),
    "supports_url_context": (".utils", "supports_url_context"),
    "supports_response_schema": (".utils", "supports_response_schema"),
    "supports_parallel_function_calling": (".utils", "supports_parallel_function_calling"),
    "supports_vision": (".utils", "supports_vision"),
    "supports_audio_input": (".utils", "supports_audio_input"),
    "supports_audio_output": (".utils", "supports_audio_output"),
    "supports_system_messages": (".utils", "supports_system_messages"),
    "supports_reasoning": (".utils", "supports_reasoning"),
    "get_litellm_params": (".utils", "get_litellm_params"),
    "acreate": (".utils", "acreate"),
    "get_max_tokens": (".utils", "get_max_tokens"),
    "get_model_info": (".utils", "get_model_info"),
    "register_prompt_template": (".utils", "register_prompt_template"),
    "validate_environment": (".utils", "validate_environment"),
    "check_valid_key": (".utils", "check_valid_key"),
    "register_model": (".utils", "register_model"),
    "encode": (".utils", "encode"),
    "decode": (".utils", "decode"),
    "_calculate_retry_after": (".utils", "_calculate_retry_after"),
    "_should_retry": (".utils", "_should_retry"),
    "get_supported_openai_params": (".utils", "get_supported_openai_params"),
    "get_api_base": (".utils", "get_api_base"),
    "get_first_chars_messages": (".utils", "get_first_chars_messages"),
    "ModelResponse": (".utils", "ModelResponse"),
    "ModelResponseStream": (".utils", "ModelResponseStream"),
    "EmbeddingResponse": (".utils", "EmbeddingResponse"),
    "ImageResponse": (".utils", "ImageResponse"),
    "TranscriptionResponse": (".utils", "TranscriptionResponse"),
    "TextCompletionResponse": (".utils", "TextCompletionResponse"),
    "get_provider_fields": (".utils", "get_provider_fields"),
    "ModelResponseListIterator": (".utils", "ModelResponseListIterator"),
    "get_valid_models": (".utils", "get_valid_models"),
}

_COST_CALCULATOR_IMPORT_MAP = {
    "completion_cost": (".cost_calculator", "completion_cost"),
    "cost_per_token": (".cost_calculator", "cost_per_token"),
    "response_cost_calculator": (".cost_calculator", "response_cost_calculator"),
}

_TYPES_UTILS_IMPORT_MAP = {
    "ImageObject": (".types.utils", "ImageObject"),
    "BudgetConfig": (".types.utils", "BudgetConfig"),
    "all_litellm_params": (".types.utils", "all_litellm_params"),
    "_litellm_completion_params": (".types.utils", "all_litellm_params"),  # Alias
    "CredentialItem": (".types.utils", "CredentialItem"),
    "PriorityReservationDict": (".types.utils", "PriorityReservationDict"),
    "StandardKeyGenerationConfig": (".types.utils", "StandardKeyGenerationConfig"),
    "SearchProviders": (".types.utils", "SearchProviders"),
    "GenericStreamingChunk": (".types.utils", "GenericStreamingChunk"),
}

_TOKEN_COUNTER_IMPORT_MAP = {
    "get_modified_max_tokens": ("litellm.litellm_core_utils.token_counter", "get_modified_max_tokens"),
}

_BEDROCK_TYPES_IMPORT_MAP = {
    "COHERE_EMBEDDING_INPUT_TYPES": ("litellm.types.llms.bedrock", "COHERE_EMBEDDING_INPUT_TYPES"),
}

_CACHING_IMPORT_MAP = {
    "Cache": ("litellm.caching.caching", "Cache"),
    "DualCache": ("litellm.caching.caching", "DualCache"),
    "RedisCache": ("litellm.caching.caching", "RedisCache"),
    "InMemoryCache": ("litellm.caching.caching", "InMemoryCache"),
}

_LITELLM_LOGGING_IMPORT_MAP = {
    "Logging": ("litellm.litellm_core_utils.litellm_logging", "Logging"),
    "modify_integration": ("litellm.litellm_core_utils.litellm_logging", "modify_integration"),
}

_DOTPROMPT_IMPORT_MAP = {
    "global_prompt_manager": ("litellm.integrations.dotprompt", "global_prompt_manager"),
    "global_prompt_directory": ("litellm.integrations.dotprompt", "global_prompt_directory"),
    "set_global_prompt_directory": ("litellm.integrations.dotprompt", "set_global_prompt_directory"),
}

_TYPES_IMPORT_MAP = {
    "GuardrailItem": ("litellm.types.guardrails", "GuardrailItem"),
}

_LLM_CONFIGS_IMPORT_MAP = {
    "AmazonConverseConfig": (".llms.bedrock.chat.converse_transformation", "AmazonConverseConfig"),
    "OpenAILikeChatConfig": (".llms.openai_like.chat.handler", "OpenAILikeChatConfig"),
    "GaladrielChatConfig": (".llms.galadriel.chat.transformation", "GaladrielChatConfig"),
    "GithubChatConfig": (".llms.github.chat.transformation", "GithubChatConfig"),
    "AzureAnthropicConfig": (".llms.azure_ai.anthropic.transformation", "AzureAnthropicConfig"),
    "BytezChatConfig": (".llms.bytez.chat.transformation", "BytezChatConfig"),
    "CompactifAIChatConfig": (".llms.compactifai.chat.transformation", "CompactifAIChatConfig"),
    "EmpowerChatConfig": (".llms.empower.chat.transformation", "EmpowerChatConfig"),
    "MinimaxChatConfig": (".llms.minimax.chat.transformation", "MinimaxChatConfig"),
    "AiohttpOpenAIChatConfig": (".llms.aiohttp_openai.chat.transformation", "AiohttpOpenAIChatConfig"),
    "HuggingFaceChatConfig": (".llms.huggingface.chat.transformation", "HuggingFaceChatConfig"),
    "HuggingFaceEmbeddingConfig": (".llms.huggingface.embedding.transformation", "HuggingFaceEmbeddingConfig"),
    "OobaboogaConfig": (".llms.oobabooga.chat.transformation", "OobaboogaConfig"),
    "MaritalkConfig": (".llms.maritalk", "MaritalkConfig"),
    "OpenrouterConfig": (".llms.openrouter.chat.transformation", "OpenrouterConfig"),
    "DataRobotConfig": (".llms.datarobot.chat.transformation", "DataRobotConfig"),
    "AnthropicConfig": (".llms.anthropic.chat.transformation", "AnthropicConfig"),
    "AnthropicTextConfig": (".llms.anthropic.completion.transformation", "AnthropicTextConfig"),
    "GroqSTTConfig": (".llms.groq.stt.transformation", "GroqSTTConfig"),
    "TritonConfig": (".llms.triton.completion.transformation", "TritonConfig"),
    "TritonGenerateConfig": (".llms.triton.completion.transformation", "TritonGenerateConfig"),
    "TritonInferConfig": (".llms.triton.completion.transformation", "TritonInferConfig"),
    "TritonEmbeddingConfig": (".llms.triton.embedding.transformation", "TritonEmbeddingConfig"),
    "HuggingFaceRerankConfig": (".llms.huggingface.rerank.transformation", "HuggingFaceRerankConfig"),
    "DatabricksConfig": (".llms.databricks.chat.transformation", "DatabricksConfig"),
    "DatabricksEmbeddingConfig": (".llms.databricks.embed.transformation", "DatabricksEmbeddingConfig"),
    "PredibaseConfig": (".llms.predibase.chat.transformation", "PredibaseConfig"),
    "ReplicateConfig": (".llms.replicate.chat.transformation", "ReplicateConfig"),
    "SnowflakeConfig": (".llms.snowflake.chat.transformation", "SnowflakeConfig"),
    "CohereRerankConfig": (".llms.cohere.rerank.transformation", "CohereRerankConfig"),
    "CohereRerankV2Config": (".llms.cohere.rerank_v2.transformation", "CohereRerankV2Config"),
    "AzureAIRerankConfig": (".llms.azure_ai.rerank.transformation", "AzureAIRerankConfig"),
    "InfinityRerankConfig": (".llms.infinity.rerank.transformation", "InfinityRerankConfig"),
    "JinaAIRerankConfig": (".llms.jina_ai.rerank.transformation", "JinaAIRerankConfig"),
    "DeepinfraRerankConfig": (".llms.deepinfra.rerank.transformation", "DeepinfraRerankConfig"),
    "HostedVLLMRerankConfig": (".llms.hosted_vllm.rerank.transformation", "HostedVLLMRerankConfig"),
    "NvidiaNimRerankConfig": (".llms.nvidia_nim.rerank.transformation", "NvidiaNimRerankConfig"),
    "NvidiaNimRankingConfig": (".llms.nvidia_nim.rerank.ranking_transformation", "NvidiaNimRankingConfig"),
    "VertexAIRerankConfig": (".llms.vertex_ai.rerank.transformation", "VertexAIRerankConfig"),
    "FireworksAIRerankConfig": (".llms.fireworks_ai.rerank.transformation", "FireworksAIRerankConfig"),
    "VoyageRerankConfig": (".llms.voyage.rerank.transformation", "VoyageRerankConfig"),
    "ClarifaiConfig": (".llms.clarifai.chat.transformation", "ClarifaiConfig"),
    "AI21ChatConfig": (".llms.ai21.chat.transformation", "AI21ChatConfig"),
}

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