"""
Registry data for lazy imports.

This module contains all the name tuples and import maps used by the lazy import system.
Separated from the handler functions for better organization.
"""

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
    "exception_type",
    "get_optional_params",
    "get_response_string",
    "token_counter",
    "create_pretrained_tokenizer",
    "create_tokenizer",
    "supports_function_calling",
    "supports_web_search",
    "supports_url_context",
    "supports_response_schema",
    "supports_parallel_function_calling",
    "supports_vision",
    "supports_audio_input",
    "supports_audio_output",
    "supports_system_messages",
    "supports_reasoning",
    "get_litellm_params",
    "acreate",
    "get_max_tokens",
    "get_model_info",
    "register_prompt_template",
    "validate_environment",
    "check_valid_key",
    "register_model",
    "encode",
    "decode",
    "_calculate_retry_after",
    "_should_retry",
    "get_supported_openai_params",
    "get_api_base",
    "get_first_chars_messages",
    "ModelResponse",
    "ModelResponseStream",
    "EmbeddingResponse",
    "ImageResponse",
    "TranscriptionResponse",
    "TextCompletionResponse",
    "get_provider_fields",
    "ModelResponseListIterator",
    "get_valid_models",
    "timeout",
    "get_llm_provider",
    "remove_index_from_tool_calls",
)

# Token counter names that support lazy loading via _lazy_import_token_counter
TOKEN_COUNTER_NAMES = ("get_modified_max_tokens",)

# LLM client cache names that support lazy loading via _lazy_import_llm_client_cache
LLM_CLIENT_CACHE_NAMES = (
    "LLMClientCache",
    "in_memory_llm_clients_cache",
)

# Bedrock type names that support lazy loading via _lazy_import_bedrock_types
BEDROCK_TYPES_NAMES = ("COHERE_EMBEDDING_INPUT_TYPES",)

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
    "LlamaAPIConfig",
    "TogetherAITextCompletionConfig",
    "CloudflareChatConfig",
    "NovitaConfig",
    "PetalsConfig",
    "OllamaChatConfig",
    "OllamaConfig",
    "SagemakerConfig",
    "SagemakerChatConfig",
    "CohereChatConfig",
    "AnthropicMessagesConfig",
    "AmazonAnthropicClaudeMessagesConfig",
    "TogetherAIConfig",
    "NLPCloudConfig",
    "VertexGeminiConfig",
    "GoogleAIStudioGeminiConfig",
    "VertexAIAnthropicConfig",
    "VertexAILlama3Config",
    "VertexAIAi21Config",
    "AmazonCohereChatConfig",
    "AmazonBedrockGlobalConfig",
    "AmazonAI21Config",
    "AmazonInvokeNovaConfig",
    "AmazonQwen2Config",
    "AmazonQwen3Config",
    # Aliases for backwards compatibility
    "VertexAIConfig",  # Alias for VertexGeminiConfig
    "GeminiConfig",  # Alias for GoogleAIStudioGeminiConfig
    "AmazonAnthropicConfig",
    "AmazonAnthropicClaudeConfig",
    "AmazonCohereConfig",
    "AmazonLlamaConfig",
    "AmazonDeepSeekR1Config",
    "AmazonMistralConfig",
    "AmazonMoonshotConfig",
    "AmazonTitanConfig",
    "AmazonTwelveLabsPegasusConfig",
    "AmazonInvokeConfig",
    "AmazonBedrockOpenAIConfig",
    "AmazonStabilityConfig",
    "AmazonStability3Config",
    "AmazonNovaCanvasConfig",
    "AmazonTitanG1Config",
    "AmazonTitanMultimodalEmbeddingG1Config",
    "CohereV2ChatConfig",
    "BedrockCohereEmbeddingConfig",
    "TwelveLabsMarengoEmbeddingConfig",
    "AmazonNovaEmbeddingConfig",
    "OpenAIConfig",
    "MistralEmbeddingConfig",
    "OpenAIImageVariationConfig",
    "DeepInfraConfig",
    "DeepgramAudioTranscriptionConfig",
    "TopazImageVariationConfig",
    "OpenAITextCompletionConfig",
    "GroqChatConfig",
    "A2AConfig",
    "GenAIHubOrchestrationConfig",
    "VoyageEmbeddingConfig",
    "VoyageContextualEmbeddingConfig",
    "InfinityEmbeddingConfig",
    "AzureAIStudioConfig",
    "MistralConfig",
    "OpenAIResponsesAPIConfig",
    "AzureOpenAIResponsesAPIConfig",
    "AzureOpenAIOSeriesResponsesAPIConfig",
    "XAIResponsesAPIConfig",
    "LiteLLMProxyResponsesAPIConfig",
    "VolcEngineResponsesAPIConfig",
    "PerplexityResponsesConfig",
    "GoogleAIStudioInteractionsConfig",
    "OpenAIOSeriesConfig",
    "AnthropicSkillsConfig",
    "BaseSkillsAPIConfig",
    "GradientAIConfig",
    # Alias for backwards compatibility
    "OpenAIO1Config",  # Alias for OpenAIOSeriesConfig
    "OpenAIGPTConfig",
    "OpenAIGPT5Config",
    "OpenAIWhisperAudioTranscriptionConfig",
    "OpenAIGPTAudioTranscriptionConfig",
    "OpenAIGPTAudioConfig",
    "NvidiaNimConfig",
    "NvidiaNimEmbeddingConfig",
    "FeatherlessAIConfig",
    "CerebrasConfig",
    "BasetenConfig",
    "SambanovaConfig",
    "SambaNovaEmbeddingConfig",
    "FireworksAIConfig",
    "FireworksAITextCompletionConfig",
    "FireworksAIAudioTranscriptionConfig",
    "FireworksAIEmbeddingConfig",
    "FriendliaiChatConfig",
    "JinaAIEmbeddingConfig",
    "XAIChatConfig",
    "ZAIChatConfig",
    "AIMLChatConfig",
    "VolcEngineChatConfig",
    "CodestralTextCompletionConfig",
    "AzureOpenAIAssistantsAPIConfig",
    "HerokuChatConfig",
    "CometAPIConfig",
    "AzureOpenAIConfig",
    "AzureOpenAIGPT5Config",
    "AzureOpenAITextConfig",
    "HostedVLLMChatConfig",
    "HostedVLLMEmbeddingConfig",
    # Alias for backwards compatibility
    "VolcEngineConfig",  # Alias for VolcEngineChatConfig
    "LlamafileChatConfig",
    "LiteLLMProxyChatConfig",
    "VLLMConfig",
    "DeepSeekChatConfig",
    "LMStudioChatConfig",
    "LmStudioEmbeddingConfig",
    "NscaleConfig",
    "PerplexityChatConfig",
    "AzureOpenAIO1Config",
    "IBMWatsonXAIConfig",
    "IBMWatsonXChatConfig",
    "IBMWatsonXEmbeddingConfig",
    "GenAIHubEmbeddingConfig",
    "IBMWatsonXAudioTranscriptionConfig",
    "GithubCopilotConfig",
    "GithubCopilotResponsesAPIConfig",
    "ChatGPTConfig",
    "ChatGPTResponsesAPIConfig",
    "ManusResponsesAPIConfig",
    "GithubCopilotEmbeddingConfig",
    "NebiusConfig",
    "WandbConfig",
    "GigaChatConfig",
    "GigaChatEmbeddingConfig",
    "DashScopeChatConfig",
    "MoonshotChatConfig",
    "DockerModelRunnerChatConfig",
    "V0ChatConfig",
    "OCIChatConfig",
    "MorphChatConfig",
    "RAGFlowConfig",
    "LambdaAIChatConfig",
    "HyperbolicChatConfig",
    "VercelAIGatewayConfig",
    "OVHCloudChatConfig",
    "OVHCloudEmbeddingConfig",
    "CometAPIEmbeddingConfig",
    "LemonadeChatConfig",
    "SnowflakeEmbeddingConfig",
    "AmazonNovaChatConfig",
)

# Types that support lazy loading via _lazy_import_types
TYPES_NAMES = (
    "GuardrailItem",
    "DefaultTeamSSOParams",
    "LiteLLM_UpperboundKeyGenerateParams",
    "KeyManagementSystem",
    "PriorityReservationSettings",
    "CustomLogger",
    "LoggingCallbackManager",
    "DatadogLLMObsInitParams",
    # Note: LlmProviders is NOT lazy-loaded because it's imported during import time
    # in multiple places including openai.py (via main import)
    # Note: KeyManagementSettings is NOT lazy-loaded because _key_management_settings
    # is accessed during import time in secret_managers/main.py
)

# LLM provider logic names that support lazy loading via _lazy_import_llm_provider_logic
LLM_PROVIDER_LOGIC_NAMES = (
    "get_llm_provider",
    "remove_index_from_tool_calls",
)

# Utils module names that support lazy loading via _lazy_import_utils_module
# These are attributes accessed from litellm.utils module
UTILS_MODULE_NAMES = (
    "encoding",
    "BaseVectorStore",
    "CredentialAccessor",
    "exception_type",
    "get_error_message",
    "_get_response_headers",
    "get_llm_provider",
    "_is_non_openai_azure_model",
    "get_supported_openai_params",
    "LiteLLMResponseObjectHandler",
    "_handle_invalid_parallel_tool_calls",
    "convert_to_model_response_object",
    "convert_to_streaming_response",
    "convert_to_streaming_response_async",
    "get_api_base",
    "ResponseMetadata",
    "_parse_content_for_reasoning",
    "LiteLLMLoggingObject",
    "redact_message_input_output_from_logging",
    "CustomStreamWrapper",
    "BaseGoogleGenAIGenerateContentConfig",
    "BaseOCRConfig",
    "BaseSearchConfig",
    "BaseTextToSpeechConfig",
    "BedrockModelInfo",
    "CohereModelInfo",
    "MistralOCRConfig",
    "Rules",
    "AsyncHTTPHandler",
    "HTTPHandler",
    "get_num_retries_from_retry_policy",
    "reset_retry_policy",
    "get_secret",
    "get_coroutine_checker",
    "get_litellm_logging_class",
    "get_set_callbacks",
    "get_litellm_metadata_from_kwargs",
    "map_finish_reason",
    "process_response_headers",
    "delete_nested_value",
    "is_nested_path",
    "_get_base_model_from_litellm_call_metadata",
    "get_litellm_params",
    "_ensure_extra_body_is_safe",
    "get_formatted_prompt",
    "get_response_headers",
    "update_response_metadata",
    "executor",
    "BaseAnthropicMessagesConfig",
    "BaseAudioTranscriptionConfig",
    "BaseBatchesConfig",
    "BaseContainerConfig",
    "BaseEmbeddingConfig",
    "BaseImageEditConfig",
    "BaseImageGenerationConfig",
    "BaseImageVariationConfig",
    "BasePassthroughConfig",
    "BaseRealtimeConfig",
    "BaseRerankConfig",
    "BaseVectorStoreConfig",
    "BaseVectorStoreFilesConfig",
    "BaseVideoConfig",
    "ANTHROPIC_API_ONLY_HEADERS",
    "AnthropicThinkingParam",
    "RerankResponse",
    "ChatCompletionDeltaToolCallChunk",
    "ChatCompletionToolCallChunk",
    "ChatCompletionToolCallFunctionChunk",
    "LiteLLM_Params",
)

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
    "supports_parallel_function_calling": (
        ".utils",
        "supports_parallel_function_calling",
    ),
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
    "timeout": (".timeout", "timeout"),
    "get_llm_provider": (
        "litellm.litellm_core_utils.get_llm_provider_logic",
        "get_llm_provider",
    ),
    "remove_index_from_tool_calls": (
        "litellm.litellm_core_utils.core_helpers",
        "remove_index_from_tool_calls",
    ),
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
    "get_modified_max_tokens": (
        "litellm.litellm_core_utils.token_counter",
        "get_modified_max_tokens",
    ),
}

_BEDROCK_TYPES_IMPORT_MAP = {
    "COHERE_EMBEDDING_INPUT_TYPES": (
        "litellm.types.llms.bedrock",
        "COHERE_EMBEDDING_INPUT_TYPES",
    ),
}

_CACHING_IMPORT_MAP = {
    "Cache": ("litellm.caching.caching", "Cache"),
    "DualCache": ("litellm.caching.caching", "DualCache"),
    "RedisCache": ("litellm.caching.caching", "RedisCache"),
    "InMemoryCache": ("litellm.caching.caching", "InMemoryCache"),
}

_LITELLM_LOGGING_IMPORT_MAP = {
    "Logging": ("litellm.litellm_core_utils.litellm_logging", "Logging"),
    "modify_integration": (
        "litellm.litellm_core_utils.litellm_logging",
        "modify_integration",
    ),
}

_DOTPROMPT_IMPORT_MAP = {
    "global_prompt_manager": (
        "litellm.integrations.dotprompt",
        "global_prompt_manager",
    ),
    "global_prompt_directory": (
        "litellm.integrations.dotprompt",
        "global_prompt_directory",
    ),
    "set_global_prompt_directory": (
        "litellm.integrations.dotprompt",
        "set_global_prompt_directory",
    ),
}

_TYPES_IMPORT_MAP = {
    "GuardrailItem": ("litellm.types.guardrails", "GuardrailItem"),
    "DefaultTeamSSOParams": (
        "litellm.types.proxy.management_endpoints.ui_sso",
        "DefaultTeamSSOParams",
    ),
    "LiteLLM_UpperboundKeyGenerateParams": (
        "litellm.types.proxy.management_endpoints.ui_sso",
        "LiteLLM_UpperboundKeyGenerateParams",
    ),
    "KeyManagementSystem": (
        "litellm.types.secret_managers.main",
        "KeyManagementSystem",
    ),
    "PriorityReservationSettings": (
        "litellm.types.utils",
        "PriorityReservationSettings",
    ),
    "CustomLogger": ("litellm.integrations.custom_logger", "CustomLogger"),
    "LoggingCallbackManager": (
        "litellm.litellm_core_utils.logging_callback_manager",
        "LoggingCallbackManager",
    ),
    "DatadogLLMObsInitParams": (
        "litellm.types.integrations.datadog_llm_obs",
        "DatadogLLMObsInitParams",
    ),
}

_LLM_PROVIDER_LOGIC_IMPORT_MAP = {
    "get_llm_provider": (
        "litellm.litellm_core_utils.get_llm_provider_logic",
        "get_llm_provider",
    ),
    "remove_index_from_tool_calls": (
        "litellm.litellm_core_utils.core_helpers",
        "remove_index_from_tool_calls",
    ),
}

_LLM_CONFIGS_IMPORT_MAP = {
    "AmazonConverseConfig": (
        ".llms.bedrock.chat.converse_transformation",
        "AmazonConverseConfig",
    ),
    "OpenAILikeChatConfig": (".llms.openai_like.chat.handler", "OpenAILikeChatConfig"),
    "GaladrielChatConfig": (
        ".llms.galadriel.chat.transformation",
        "GaladrielChatConfig",
    ),
    "GithubChatConfig": (".llms.github.chat.transformation", "GithubChatConfig"),
    "AzureAnthropicConfig": (
        ".llms.azure_ai.anthropic.transformation",
        "AzureAnthropicConfig",
    ),
    "BytezChatConfig": (".llms.bytez.chat.transformation", "BytezChatConfig"),
    "CompactifAIChatConfig": (
        ".llms.compactifai.chat.transformation",
        "CompactifAIChatConfig",
    ),
    "EmpowerChatConfig": (".llms.empower.chat.transformation", "EmpowerChatConfig"),
    "MinimaxChatConfig": (".llms.minimax.chat.transformation", "MinimaxChatConfig"),
    "AiohttpOpenAIChatConfig": (
        ".llms.aiohttp_openai.chat.transformation",
        "AiohttpOpenAIChatConfig",
    ),
    "HuggingFaceChatConfig": (
        ".llms.huggingface.chat.transformation",
        "HuggingFaceChatConfig",
    ),
    "HuggingFaceEmbeddingConfig": (
        ".llms.huggingface.embedding.transformation",
        "HuggingFaceEmbeddingConfig",
    ),
    "OobaboogaConfig": (".llms.oobabooga.chat.transformation", "OobaboogaConfig"),
    "MaritalkConfig": (".llms.maritalk", "MaritalkConfig"),
    "OpenrouterConfig": (".llms.openrouter.chat.transformation", "OpenrouterConfig"),
    "DataRobotConfig": (".llms.datarobot.chat.transformation", "DataRobotConfig"),
    "AnthropicConfig": (".llms.anthropic.chat.transformation", "AnthropicConfig"),
    "AnthropicTextConfig": (
        ".llms.anthropic.completion.transformation",
        "AnthropicTextConfig",
    ),
    "GroqSTTConfig": (".llms.groq.stt.transformation", "GroqSTTConfig"),
    "TritonConfig": (".llms.triton.completion.transformation", "TritonConfig"),
    "TritonGenerateConfig": (
        ".llms.triton.completion.transformation",
        "TritonGenerateConfig",
    ),
    "TritonInferConfig": (
        ".llms.triton.completion.transformation",
        "TritonInferConfig",
    ),
    "TritonEmbeddingConfig": (
        ".llms.triton.embedding.transformation",
        "TritonEmbeddingConfig",
    ),
    "HuggingFaceRerankConfig": (
        ".llms.huggingface.rerank.transformation",
        "HuggingFaceRerankConfig",
    ),
    "DatabricksConfig": (".llms.databricks.chat.transformation", "DatabricksConfig"),
    "DatabricksEmbeddingConfig": (
        ".llms.databricks.embed.transformation",
        "DatabricksEmbeddingConfig",
    ),
    "PredibaseConfig": (".llms.predibase.chat.transformation", "PredibaseConfig"),
    "ReplicateConfig": (".llms.replicate.chat.transformation", "ReplicateConfig"),
    "SnowflakeConfig": (".llms.snowflake.chat.transformation", "SnowflakeConfig"),
    "CohereRerankConfig": (".llms.cohere.rerank.transformation", "CohereRerankConfig"),
    "CohereRerankV2Config": (
        ".llms.cohere.rerank_v2.transformation",
        "CohereRerankV2Config",
    ),
    "AzureAIRerankConfig": (
        ".llms.azure_ai.rerank.transformation",
        "AzureAIRerankConfig",
    ),
    "InfinityRerankConfig": (
        ".llms.infinity.rerank.transformation",
        "InfinityRerankConfig",
    ),
    "JinaAIRerankConfig": (".llms.jina_ai.rerank.transformation", "JinaAIRerankConfig"),
    "DeepinfraRerankConfig": (
        ".llms.deepinfra.rerank.transformation",
        "DeepinfraRerankConfig",
    ),
    "HostedVLLMRerankConfig": (
        ".llms.hosted_vllm.rerank.transformation",
        "HostedVLLMRerankConfig",
    ),
    "NvidiaNimRerankConfig": (
        ".llms.nvidia_nim.rerank.transformation",
        "NvidiaNimRerankConfig",
    ),
    "NvidiaNimRankingConfig": (
        ".llms.nvidia_nim.rerank.ranking_transformation",
        "NvidiaNimRankingConfig",
    ),
    "VertexAIRerankConfig": (
        ".llms.vertex_ai.rerank.transformation",
        "VertexAIRerankConfig",
    ),
    "FireworksAIRerankConfig": (
        ".llms.fireworks_ai.rerank.transformation",
        "FireworksAIRerankConfig",
    ),
    "VoyageRerankConfig": (".llms.voyage.rerank.transformation", "VoyageRerankConfig"),
    "ClarifaiConfig": (".llms.clarifai.chat.transformation", "ClarifaiConfig"),
    "AI21ChatConfig": (".llms.ai21.chat.transformation", "AI21ChatConfig"),
    "LlamaAPIConfig": (".llms.meta_llama.chat.transformation", "LlamaAPIConfig"),
    "TogetherAITextCompletionConfig": (
        ".llms.together_ai.completion.transformation",
        "TogetherAITextCompletionConfig",
    ),
    "CloudflareChatConfig": (
        ".llms.cloudflare.chat.transformation",
        "CloudflareChatConfig",
    ),
    "NovitaConfig": (".llms.novita.chat.transformation", "NovitaConfig"),
    "PetalsConfig": (".llms.petals.completion.transformation", "PetalsConfig"),
    "OllamaChatConfig": (".llms.ollama.chat.transformation", "OllamaChatConfig"),
    "OllamaConfig": (".llms.ollama.completion.transformation", "OllamaConfig"),
    "SagemakerConfig": (".llms.sagemaker.completion.transformation", "SagemakerConfig"),
    "SagemakerChatConfig": (
        ".llms.sagemaker.chat.transformation",
        "SagemakerChatConfig",
    ),
    "CohereChatConfig": (".llms.cohere.chat.transformation", "CohereChatConfig"),
    "AnthropicMessagesConfig": (
        ".llms.anthropic.experimental_pass_through.messages.transformation",
        "AnthropicMessagesConfig",
    ),
    "AmazonAnthropicClaudeMessagesConfig": (
        ".llms.bedrock.messages.invoke_transformations.anthropic_claude3_transformation",
        "AmazonAnthropicClaudeMessagesConfig",
    ),
    "TogetherAIConfig": (".llms.together_ai.chat", "TogetherAIConfig"),
    "NLPCloudConfig": (".llms.nlp_cloud.chat.handler", "NLPCloudConfig"),
    "VertexGeminiConfig": (
        ".llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini",
        "VertexGeminiConfig",
    ),
    "GoogleAIStudioGeminiConfig": (
        ".llms.gemini.chat.transformation",
        "GoogleAIStudioGeminiConfig",
    ),
    "VertexAIAnthropicConfig": (
        ".llms.vertex_ai.vertex_ai_partner_models.anthropic.transformation",
        "VertexAIAnthropicConfig",
    ),
    "VertexAILlama3Config": (
        ".llms.vertex_ai.vertex_ai_partner_models.llama3.transformation",
        "VertexAILlama3Config",
    ),
    "VertexAIAi21Config": (
        ".llms.vertex_ai.vertex_ai_partner_models.ai21.transformation",
        "VertexAIAi21Config",
    ),
    "AmazonCohereChatConfig": (
        ".llms.bedrock.chat.invoke_handler",
        "AmazonCohereChatConfig",
    ),
    "AmazonBedrockGlobalConfig": (
        ".llms.bedrock.common_utils",
        "AmazonBedrockGlobalConfig",
    ),
    "AmazonAI21Config": (
        ".llms.bedrock.chat.invoke_transformations.amazon_ai21_transformation",
        "AmazonAI21Config",
    ),
    "AmazonInvokeNovaConfig": (
        ".llms.bedrock.chat.invoke_transformations.amazon_nova_transformation",
        "AmazonInvokeNovaConfig",
    ),
    "AmazonQwen2Config": (
        ".llms.bedrock.chat.invoke_transformations.amazon_qwen2_transformation",
        "AmazonQwen2Config",
    ),
    "AmazonQwen3Config": (
        ".llms.bedrock.chat.invoke_transformations.amazon_qwen3_transformation",
        "AmazonQwen3Config",
    ),
    # Aliases for backwards compatibility
    "VertexAIConfig": (
        ".llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini",
        "VertexGeminiConfig",
    ),  # Alias
    "GeminiConfig": (
        ".llms.gemini.chat.transformation",
        "GoogleAIStudioGeminiConfig",
    ),  # Alias
    "AmazonAnthropicConfig": (
        ".llms.bedrock.chat.invoke_transformations.anthropic_claude2_transformation",
        "AmazonAnthropicConfig",
    ),
    "AmazonAnthropicClaudeConfig": (
        ".llms.bedrock.chat.invoke_transformations.anthropic_claude3_transformation",
        "AmazonAnthropicClaudeConfig",
    ),
    "AmazonCohereConfig": (
        ".llms.bedrock.chat.invoke_transformations.amazon_cohere_transformation",
        "AmazonCohereConfig",
    ),
    "AmazonLlamaConfig": (
        ".llms.bedrock.chat.invoke_transformations.amazon_llama_transformation",
        "AmazonLlamaConfig",
    ),
    "AmazonDeepSeekR1Config": (
        ".llms.bedrock.chat.invoke_transformations.amazon_deepseek_transformation",
        "AmazonDeepSeekR1Config",
    ),
    "AmazonMistralConfig": (
        ".llms.bedrock.chat.invoke_transformations.amazon_mistral_transformation",
        "AmazonMistralConfig",
    ),
    "AmazonMoonshotConfig": (
        ".llms.bedrock.chat.invoke_transformations.amazon_moonshot_transformation",
        "AmazonMoonshotConfig",
    ),
    "AmazonTitanConfig": (
        ".llms.bedrock.chat.invoke_transformations.amazon_titan_transformation",
        "AmazonTitanConfig",
    ),
    "AmazonTwelveLabsPegasusConfig": (
        ".llms.bedrock.chat.invoke_transformations.amazon_twelvelabs_pegasus_transformation",
        "AmazonTwelveLabsPegasusConfig",
    ),
    "AmazonInvokeConfig": (
        ".llms.bedrock.chat.invoke_transformations.base_invoke_transformation",
        "AmazonInvokeConfig",
    ),
    "AmazonBedrockOpenAIConfig": (
        ".llms.bedrock.chat.invoke_transformations.amazon_openai_transformation",
        "AmazonBedrockOpenAIConfig",
    ),
    "AmazonStabilityConfig": (
        ".llms.bedrock.image_generation.amazon_stability1_transformation",
        "AmazonStabilityConfig",
    ),
    "AmazonStability3Config": (
        ".llms.bedrock.image_generation.amazon_stability3_transformation",
        "AmazonStability3Config",
    ),
    "AmazonNovaCanvasConfig": (
        ".llms.bedrock.image_generation.amazon_nova_canvas_transformation",
        "AmazonNovaCanvasConfig",
    ),
    "AmazonTitanG1Config": (
        ".llms.bedrock.embed.amazon_titan_g1_transformation",
        "AmazonTitanG1Config",
    ),
    "AmazonTitanMultimodalEmbeddingG1Config": (
        ".llms.bedrock.embed.amazon_titan_multimodal_transformation",
        "AmazonTitanMultimodalEmbeddingG1Config",
    ),
    "CohereV2ChatConfig": (".llms.cohere.chat.v2_transformation", "CohereV2ChatConfig"),
    "BedrockCohereEmbeddingConfig": (
        ".llms.bedrock.embed.cohere_transformation",
        "BedrockCohereEmbeddingConfig",
    ),
    "TwelveLabsMarengoEmbeddingConfig": (
        ".llms.bedrock.embed.twelvelabs_marengo_transformation",
        "TwelveLabsMarengoEmbeddingConfig",
    ),
    "AmazonNovaEmbeddingConfig": (
        ".llms.bedrock.embed.amazon_nova_transformation",
        "AmazonNovaEmbeddingConfig",
    ),
    "OpenAIConfig": (".llms.openai.openai", "OpenAIConfig"),
    "MistralEmbeddingConfig": (".llms.openai.openai", "MistralEmbeddingConfig"),
    "OpenAIImageVariationConfig": (
        ".llms.openai.image_variations.transformation",
        "OpenAIImageVariationConfig",
    ),
    "DeepInfraConfig": (".llms.deepinfra.chat.transformation", "DeepInfraConfig"),
    "DeepgramAudioTranscriptionConfig": (
        ".llms.deepgram.audio_transcription.transformation",
        "DeepgramAudioTranscriptionConfig",
    ),
    "TopazImageVariationConfig": (
        ".llms.topaz.image_variations.transformation",
        "TopazImageVariationConfig",
    ),
    "OpenAITextCompletionConfig": (
        "litellm.llms.openai.completion.transformation",
        "OpenAITextCompletionConfig",
    ),
    "GroqChatConfig": (".llms.groq.chat.transformation", "GroqChatConfig"),
    "A2AConfig": (".llms.a2a.chat.transformation", "A2AConfig"),
    "GenAIHubOrchestrationConfig": (
        ".llms.sap.chat.transformation",
        "GenAIHubOrchestrationConfig",
    ),
    "VoyageEmbeddingConfig": (
        ".llms.voyage.embedding.transformation",
        "VoyageEmbeddingConfig",
    ),
    "VoyageContextualEmbeddingConfig": (
        ".llms.voyage.embedding.transformation_contextual",
        "VoyageContextualEmbeddingConfig",
    ),
    "InfinityEmbeddingConfig": (
        ".llms.infinity.embedding.transformation",
        "InfinityEmbeddingConfig",
    ),
    "AzureAIStudioConfig": (
        ".llms.azure_ai.chat.transformation",
        "AzureAIStudioConfig",
    ),
    "MistralConfig": (".llms.mistral.chat.transformation", "MistralConfig"),
    "OpenAIResponsesAPIConfig": (
        ".llms.openai.responses.transformation",
        "OpenAIResponsesAPIConfig",
    ),
    "AzureOpenAIResponsesAPIConfig": (
        ".llms.azure.responses.transformation",
        "AzureOpenAIResponsesAPIConfig",
    ),
    "AzureOpenAIOSeriesResponsesAPIConfig": (
        ".llms.azure.responses.o_series_transformation",
        "AzureOpenAIOSeriesResponsesAPIConfig",
    ),
    "XAIResponsesAPIConfig": (
        ".llms.xai.responses.transformation",
        "XAIResponsesAPIConfig",
    ),
    "LiteLLMProxyResponsesAPIConfig": (
        ".llms.litellm_proxy.responses.transformation",
        "LiteLLMProxyResponsesAPIConfig",
    ),
    "VolcEngineResponsesAPIConfig": (
        ".llms.volcengine.responses.transformation",
        "VolcEngineResponsesAPIConfig",
    ),
    "ManusResponsesAPIConfig": (
        ".llms.manus.responses.transformation",
        "ManusResponsesAPIConfig",
    ),
    "PerplexityResponsesConfig": (
        ".llms.perplexity.responses.transformation",
        "PerplexityResponsesConfig",
    ),
    "GoogleAIStudioInteractionsConfig": (
        ".llms.gemini.interactions.transformation",
        "GoogleAIStudioInteractionsConfig",
    ),
    "OpenAIOSeriesConfig": (
        ".llms.openai.chat.o_series_transformation",
        "OpenAIOSeriesConfig",
    ),
    "AnthropicSkillsConfig": (
        ".llms.anthropic.skills.transformation",
        "AnthropicSkillsConfig",
    ),
    "BaseSkillsAPIConfig": (
        ".llms.base_llm.skills.transformation",
        "BaseSkillsAPIConfig",
    ),
    "GradientAIConfig": (".llms.gradient_ai.chat.transformation", "GradientAIConfig"),
    # Alias for backwards compatibility
    "OpenAIO1Config": (
        ".llms.openai.chat.o_series_transformation",
        "OpenAIOSeriesConfig",
    ),  # Alias
    "OpenAIGPTConfig": (".llms.openai.chat.gpt_transformation", "OpenAIGPTConfig"),
    "OpenAIGPT5Config": (".llms.openai.chat.gpt_5_transformation", "OpenAIGPT5Config"),
    "OpenAIWhisperAudioTranscriptionConfig": (
        ".llms.openai.transcriptions.whisper_transformation",
        "OpenAIWhisperAudioTranscriptionConfig",
    ),
    "OpenAIGPTAudioTranscriptionConfig": (
        ".llms.openai.transcriptions.gpt_transformation",
        "OpenAIGPTAudioTranscriptionConfig",
    ),
    "OpenAIGPTAudioConfig": (
        ".llms.openai.chat.gpt_audio_transformation",
        "OpenAIGPTAudioConfig",
    ),
    "NvidiaNimConfig": (".llms.nvidia_nim.chat.transformation", "NvidiaNimConfig"),
    "NvidiaNimEmbeddingConfig": (".llms.nvidia_nim.embed", "NvidiaNimEmbeddingConfig"),
    "FeatherlessAIConfig": (
        ".llms.featherless_ai.chat.transformation",
        "FeatherlessAIConfig",
    ),
    "CerebrasConfig": (".llms.cerebras.chat", "CerebrasConfig"),
    "BasetenConfig": (".llms.baseten.chat", "BasetenConfig"),
    "SambanovaConfig": (".llms.sambanova.chat", "SambanovaConfig"),
    "SambaNovaEmbeddingConfig": (
        ".llms.sambanova.embedding.transformation",
        "SambaNovaEmbeddingConfig",
    ),
    "FireworksAIConfig": (
        ".llms.fireworks_ai.chat.transformation",
        "FireworksAIConfig",
    ),
    "FireworksAITextCompletionConfig": (
        ".llms.fireworks_ai.completion.transformation",
        "FireworksAITextCompletionConfig",
    ),
    "FireworksAIAudioTranscriptionConfig": (
        ".llms.fireworks_ai.audio_transcription.transformation",
        "FireworksAIAudioTranscriptionConfig",
    ),
    "FireworksAIEmbeddingConfig": (
        ".llms.fireworks_ai.embed.fireworks_ai_transformation",
        "FireworksAIEmbeddingConfig",
    ),
    "FriendliaiChatConfig": (
        ".llms.friendliai.chat.transformation",
        "FriendliaiChatConfig",
    ),
    "JinaAIEmbeddingConfig": (
        ".llms.jina_ai.embedding.transformation",
        "JinaAIEmbeddingConfig",
    ),
    "XAIChatConfig": (".llms.xai.chat.transformation", "XAIChatConfig"),
    "ZAIChatConfig": (".llms.zai.chat.transformation", "ZAIChatConfig"),
    "AIMLChatConfig": (".llms.aiml.chat.transformation", "AIMLChatConfig"),
    "VolcEngineChatConfig": (
        ".llms.volcengine.chat.transformation",
        "VolcEngineChatConfig",
    ),
    "CodestralTextCompletionConfig": (
        ".llms.codestral.completion.transformation",
        "CodestralTextCompletionConfig",
    ),
    "AzureOpenAIAssistantsAPIConfig": (
        ".llms.azure.azure",
        "AzureOpenAIAssistantsAPIConfig",
    ),
    "HerokuChatConfig": (".llms.heroku.chat.transformation", "HerokuChatConfig"),
    "CometAPIConfig": (".llms.cometapi.chat.transformation", "CometAPIConfig"),
    "AzureOpenAIConfig": (".llms.azure.chat.gpt_transformation", "AzureOpenAIConfig"),
    "AzureOpenAIGPT5Config": (
        ".llms.azure.chat.gpt_5_transformation",
        "AzureOpenAIGPT5Config",
    ),
    "AzureOpenAITextConfig": (
        ".llms.azure.completion.transformation",
        "AzureOpenAITextConfig",
    ),
    "HostedVLLMChatConfig": (
        ".llms.hosted_vllm.chat.transformation",
        "HostedVLLMChatConfig",
    ),
    "HostedVLLMEmbeddingConfig": (
        ".llms.hosted_vllm.embedding.transformation",
        "HostedVLLMEmbeddingConfig",
    ),
    # Alias for backwards compatibility
    "VolcEngineConfig": (
        ".llms.volcengine.chat.transformation",
        "VolcEngineChatConfig",
    ),  # Alias
    "LlamafileChatConfig": (
        ".llms.llamafile.chat.transformation",
        "LlamafileChatConfig",
    ),
    "LiteLLMProxyChatConfig": (
        ".llms.litellm_proxy.chat.transformation",
        "LiteLLMProxyChatConfig",
    ),
    "VLLMConfig": (".llms.vllm.completion.transformation", "VLLMConfig"),
    "DeepSeekChatConfig": (".llms.deepseek.chat.transformation", "DeepSeekChatConfig"),
    "LMStudioChatConfig": (".llms.lm_studio.chat.transformation", "LMStudioChatConfig"),
    "LmStudioEmbeddingConfig": (
        ".llms.lm_studio.embed.transformation",
        "LmStudioEmbeddingConfig",
    ),
    "NscaleConfig": (".llms.nscale.chat.transformation", "NscaleConfig"),
    "PerplexityChatConfig": (
        ".llms.perplexity.chat.transformation",
        "PerplexityChatConfig",
    ),
    "AzureOpenAIO1Config": (
        ".llms.azure.chat.o_series_transformation",
        "AzureOpenAIO1Config",
    ),
    "IBMWatsonXAIConfig": (
        ".llms.watsonx.completion.transformation",
        "IBMWatsonXAIConfig",
    ),
    "IBMWatsonXChatConfig": (
        ".llms.watsonx.chat.transformation",
        "IBMWatsonXChatConfig",
    ),
    "IBMWatsonXEmbeddingConfig": (
        ".llms.watsonx.embed.transformation",
        "IBMWatsonXEmbeddingConfig",
    ),
    "GenAIHubEmbeddingConfig": (
        ".llms.sap.embed.transformation",
        "GenAIHubEmbeddingConfig",
    ),
    "IBMWatsonXAudioTranscriptionConfig": (
        ".llms.watsonx.audio_transcription.transformation",
        "IBMWatsonXAudioTranscriptionConfig",
    ),
    "GithubCopilotConfig": (
        ".llms.github_copilot.chat.transformation",
        "GithubCopilotConfig",
    ),
    "GithubCopilotResponsesAPIConfig": (
        ".llms.github_copilot.responses.transformation",
        "GithubCopilotResponsesAPIConfig",
    ),
    "GithubCopilotEmbeddingConfig": (
        ".llms.github_copilot.embedding.transformation",
        "GithubCopilotEmbeddingConfig",
    ),
    "ChatGPTConfig": (".llms.chatgpt.chat.transformation", "ChatGPTConfig"),
    "ChatGPTResponsesAPIConfig": (
        ".llms.chatgpt.responses.transformation",
        "ChatGPTResponsesAPIConfig",
    ),
    "NebiusConfig": (".llms.nebius.chat.transformation", "NebiusConfig"),
    "WandbConfig": (".llms.wandb.chat.transformation", "WandbConfig"),
    "GigaChatConfig": (".llms.gigachat.chat.transformation", "GigaChatConfig"),
    "GigaChatEmbeddingConfig": (
        ".llms.gigachat.embedding.transformation",
        "GigaChatEmbeddingConfig",
    ),
    "DashScopeChatConfig": (
        ".llms.dashscope.chat.transformation",
        "DashScopeChatConfig",
    ),
    "MoonshotChatConfig": (".llms.moonshot.chat.transformation", "MoonshotChatConfig"),
    "DockerModelRunnerChatConfig": (
        ".llms.docker_model_runner.chat.transformation",
        "DockerModelRunnerChatConfig",
    ),
    "V0ChatConfig": (".llms.v0.chat.transformation", "V0ChatConfig"),
    "OCIChatConfig": (".llms.oci.chat.transformation", "OCIChatConfig"),
    "MorphChatConfig": (".llms.morph.chat.transformation", "MorphChatConfig"),
    "RAGFlowConfig": (".llms.ragflow.chat.transformation", "RAGFlowConfig"),
    "LambdaAIChatConfig": (".llms.lambda_ai.chat.transformation", "LambdaAIChatConfig"),
    "HyperbolicChatConfig": (
        ".llms.hyperbolic.chat.transformation",
        "HyperbolicChatConfig",
    ),
    "VercelAIGatewayConfig": (
        ".llms.vercel_ai_gateway.chat.transformation",
        "VercelAIGatewayConfig",
    ),
    "OVHCloudChatConfig": (".llms.ovhcloud.chat.transformation", "OVHCloudChatConfig"),
    "OVHCloudEmbeddingConfig": (
        ".llms.ovhcloud.embedding.transformation",
        "OVHCloudEmbeddingConfig",
    ),
    "CometAPIEmbeddingConfig": (
        ".llms.cometapi.embed.transformation",
        "CometAPIEmbeddingConfig",
    ),
    "LemonadeChatConfig": (".llms.lemonade.chat.transformation", "LemonadeChatConfig"),
    "SnowflakeEmbeddingConfig": (
        ".llms.snowflake.embedding.transformation",
        "SnowflakeEmbeddingConfig",
    ),
    "AmazonNovaChatConfig": (
        ".llms.amazon_nova.chat.transformation",
        "AmazonNovaChatConfig",
    ),
}

# Import map for utils module lazy imports
_UTILS_MODULE_IMPORT_MAP = {
    "encoding": ("litellm.main", "encoding"),
    "BaseVectorStore": (
        "litellm.integrations.vector_store_integrations.base_vector_store",
        "BaseVectorStore",
    ),
    "CredentialAccessor": (
        "litellm.litellm_core_utils.credential_accessor",
        "CredentialAccessor",
    ),
    "exception_type": (
        "litellm.litellm_core_utils.exception_mapping_utils",
        "exception_type",
    ),
    "get_error_message": (
        "litellm.litellm_core_utils.exception_mapping_utils",
        "get_error_message",
    ),
    "_get_response_headers": (
        "litellm.litellm_core_utils.exception_mapping_utils",
        "_get_response_headers",
    ),
    "get_llm_provider": (
        "litellm.litellm_core_utils.get_llm_provider_logic",
        "get_llm_provider",
    ),
    "_is_non_openai_azure_model": (
        "litellm.litellm_core_utils.get_llm_provider_logic",
        "_is_non_openai_azure_model",
    ),
    "get_supported_openai_params": (
        "litellm.litellm_core_utils.get_supported_openai_params",
        "get_supported_openai_params",
    ),
    "LiteLLMResponseObjectHandler": (
        "litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response",
        "LiteLLMResponseObjectHandler",
    ),
    "_handle_invalid_parallel_tool_calls": (
        "litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response",
        "_handle_invalid_parallel_tool_calls",
    ),
    "convert_to_model_response_object": (
        "litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response",
        "convert_to_model_response_object",
    ),
    "convert_to_streaming_response": (
        "litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response",
        "convert_to_streaming_response",
    ),
    "convert_to_streaming_response_async": (
        "litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response",
        "convert_to_streaming_response_async",
    ),
    "get_api_base": (
        "litellm.litellm_core_utils.llm_response_utils.get_api_base",
        "get_api_base",
    ),
    "ResponseMetadata": (
        "litellm.litellm_core_utils.llm_response_utils.response_metadata",
        "ResponseMetadata",
    ),
    "_parse_content_for_reasoning": (
        "litellm.litellm_core_utils.prompt_templates.common_utils",
        "_parse_content_for_reasoning",
    ),
    "LiteLLMLoggingObject": (
        "litellm.litellm_core_utils.redact_messages",
        "LiteLLMLoggingObject",
    ),
    "redact_message_input_output_from_logging": (
        "litellm.litellm_core_utils.redact_messages",
        "redact_message_input_output_from_logging",
    ),
    "CustomStreamWrapper": (
        "litellm.litellm_core_utils.streaming_handler",
        "CustomStreamWrapper",
    ),
    "BaseGoogleGenAIGenerateContentConfig": (
        "litellm.llms.base_llm.google_genai.transformation",
        "BaseGoogleGenAIGenerateContentConfig",
    ),
    "BaseOCRConfig": ("litellm.llms.base_llm.ocr.transformation", "BaseOCRConfig"),
    "BaseSearchConfig": (
        "litellm.llms.base_llm.search.transformation",
        "BaseSearchConfig",
    ),
    "BaseTextToSpeechConfig": (
        "litellm.llms.base_llm.text_to_speech.transformation",
        "BaseTextToSpeechConfig",
    ),
    "BedrockModelInfo": ("litellm.llms.bedrock.common_utils", "BedrockModelInfo"),
    "CohereModelInfo": ("litellm.llms.cohere.common_utils", "CohereModelInfo"),
    "MistralOCRConfig": ("litellm.llms.mistral.ocr.transformation", "MistralOCRConfig"),
    "Rules": ("litellm.litellm_core_utils.rules", "Rules"),
    "AsyncHTTPHandler": ("litellm.llms.custom_httpx.http_handler", "AsyncHTTPHandler"),
    "HTTPHandler": ("litellm.llms.custom_httpx.http_handler", "HTTPHandler"),
    "get_num_retries_from_retry_policy": (
        "litellm.router_utils.get_retry_from_policy",
        "get_num_retries_from_retry_policy",
    ),
    "reset_retry_policy": (
        "litellm.router_utils.get_retry_from_policy",
        "reset_retry_policy",
    ),
    "get_secret": ("litellm.secret_managers.main", "get_secret"),
    "get_coroutine_checker": (
        "litellm.litellm_core_utils.cached_imports",
        "get_coroutine_checker",
    ),
    "get_litellm_logging_class": (
        "litellm.litellm_core_utils.cached_imports",
        "get_litellm_logging_class",
    ),
    "get_set_callbacks": (
        "litellm.litellm_core_utils.cached_imports",
        "get_set_callbacks",
    ),
    "get_litellm_metadata_from_kwargs": (
        "litellm.litellm_core_utils.core_helpers",
        "get_litellm_metadata_from_kwargs",
    ),
    "map_finish_reason": (
        "litellm.litellm_core_utils.core_helpers",
        "map_finish_reason",
    ),
    "process_response_headers": (
        "litellm.litellm_core_utils.core_helpers",
        "process_response_headers",
    ),
    "delete_nested_value": (
        "litellm.litellm_core_utils.dot_notation_indexing",
        "delete_nested_value",
    ),
    "is_nested_path": (
        "litellm.litellm_core_utils.dot_notation_indexing",
        "is_nested_path",
    ),
    "_get_base_model_from_litellm_call_metadata": (
        "litellm.litellm_core_utils.get_litellm_params",
        "_get_base_model_from_litellm_call_metadata",
    ),
    "get_litellm_params": (
        "litellm.litellm_core_utils.get_litellm_params",
        "get_litellm_params",
    ),
    "_ensure_extra_body_is_safe": (
        "litellm.litellm_core_utils.llm_request_utils",
        "_ensure_extra_body_is_safe",
    ),
    "get_formatted_prompt": (
        "litellm.litellm_core_utils.llm_response_utils.get_formatted_prompt",
        "get_formatted_prompt",
    ),
    "get_response_headers": (
        "litellm.litellm_core_utils.llm_response_utils.get_headers",
        "get_response_headers",
    ),
    "update_response_metadata": (
        "litellm.litellm_core_utils.llm_response_utils.response_metadata",
        "update_response_metadata",
    ),
    "executor": ("litellm.litellm_core_utils.thread_pool_executor", "executor"),
    "BaseAnthropicMessagesConfig": (
        "litellm.llms.base_llm.anthropic_messages.transformation",
        "BaseAnthropicMessagesConfig",
    ),
    "BaseAudioTranscriptionConfig": (
        "litellm.llms.base_llm.audio_transcription.transformation",
        "BaseAudioTranscriptionConfig",
    ),
    "BaseBatchesConfig": (
        "litellm.llms.base_llm.batches.transformation",
        "BaseBatchesConfig",
    ),
    "BaseContainerConfig": (
        "litellm.llms.base_llm.containers.transformation",
        "BaseContainerConfig",
    ),
    "BaseEmbeddingConfig": (
        "litellm.llms.base_llm.embedding.transformation",
        "BaseEmbeddingConfig",
    ),
    "BaseImageEditConfig": (
        "litellm.llms.base_llm.image_edit.transformation",
        "BaseImageEditConfig",
    ),
    "BaseImageGenerationConfig": (
        "litellm.llms.base_llm.image_generation.transformation",
        "BaseImageGenerationConfig",
    ),
    "BaseImageVariationConfig": (
        "litellm.llms.base_llm.image_variations.transformation",
        "BaseImageVariationConfig",
    ),
    "BasePassthroughConfig": (
        "litellm.llms.base_llm.passthrough.transformation",
        "BasePassthroughConfig",
    ),
    "BaseRealtimeConfig": (
        "litellm.llms.base_llm.realtime.transformation",
        "BaseRealtimeConfig",
    ),
    "BaseRerankConfig": (
        "litellm.llms.base_llm.rerank.transformation",
        "BaseRerankConfig",
    ),
    "BaseVectorStoreConfig": (
        "litellm.llms.base_llm.vector_store.transformation",
        "BaseVectorStoreConfig",
    ),
    "BaseVectorStoreFilesConfig": (
        "litellm.llms.base_llm.vector_store_files.transformation",
        "BaseVectorStoreFilesConfig",
    ),
    "BaseVideoConfig": (
        "litellm.llms.base_llm.videos.transformation",
        "BaseVideoConfig",
    ),
    "ANTHROPIC_API_ONLY_HEADERS": (
        "litellm.types.llms.anthropic",
        "ANTHROPIC_API_ONLY_HEADERS",
    ),
    "AnthropicThinkingParam": (
        "litellm.types.llms.anthropic",
        "AnthropicThinkingParam",
    ),
    "RerankResponse": ("litellm.types.rerank", "RerankResponse"),
    "ChatCompletionDeltaToolCallChunk": (
        "litellm.types.llms.openai",
        "ChatCompletionDeltaToolCallChunk",
    ),
    "ChatCompletionToolCallChunk": (
        "litellm.types.llms.openai",
        "ChatCompletionToolCallChunk",
    ),
    "ChatCompletionToolCallFunctionChunk": (
        "litellm.types.llms.openai",
        "ChatCompletionToolCallFunctionChunk",
    ),
    "LiteLLM_Params": ("litellm.types.router", "LiteLLM_Params"),
}

# Export all name tuples and import maps for use in _lazy_imports.py
__all__ = [
    # Name tuples
    "COST_CALCULATOR_NAMES",
    "LITELLM_LOGGING_NAMES",
    "UTILS_NAMES",
    "TOKEN_COUNTER_NAMES",
    "LLM_CLIENT_CACHE_NAMES",
    "BEDROCK_TYPES_NAMES",
    "TYPES_UTILS_NAMES",
    "CACHING_NAMES",
    "HTTP_HANDLER_NAMES",
    "DOTPROMPT_NAMES",
    "LLM_CONFIG_NAMES",
    "TYPES_NAMES",
    "LLM_PROVIDER_LOGIC_NAMES",
    "UTILS_MODULE_NAMES",
    # Import maps
    "_UTILS_IMPORT_MAP",
    "_COST_CALCULATOR_IMPORT_MAP",
    "_TYPES_UTILS_IMPORT_MAP",
    "_TOKEN_COUNTER_IMPORT_MAP",
    "_BEDROCK_TYPES_IMPORT_MAP",
    "_CACHING_IMPORT_MAP",
    "_LITELLM_LOGGING_IMPORT_MAP",
    "_DOTPROMPT_IMPORT_MAP",
    "_TYPES_IMPORT_MAP",
    "_LLM_CONFIGS_IMPORT_MAP",
    "_LLM_PROVIDER_LOGIC_IMPORT_MAP",
    "_UTILS_MODULE_IMPORT_MAP",
]
