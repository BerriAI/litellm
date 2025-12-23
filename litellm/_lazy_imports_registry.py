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
    "GenAIHubOrchestrationConfig",
    "VoyageEmbeddingConfig",
    "VoyageContextualEmbeddingConfig",
)

# Types that support lazy loading via _lazy_import_types
TYPES_NAMES = (
    "GuardrailItem",
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
    "LlamaAPIConfig": (".llms.meta_llama.chat.transformation", "LlamaAPIConfig"),
    "TogetherAITextCompletionConfig": (".llms.together_ai.completion.transformation", "TogetherAITextCompletionConfig"),
    "CloudflareChatConfig": (".llms.cloudflare.chat.transformation", "CloudflareChatConfig"),
    "NovitaConfig": (".llms.novita.chat.transformation", "NovitaConfig"),
    "PetalsConfig": (".llms.petals.completion.transformation", "PetalsConfig"),
    "OllamaChatConfig": (".llms.ollama.chat.transformation", "OllamaChatConfig"),
    "OllamaConfig": (".llms.ollama.completion.transformation", "OllamaConfig"),
    "SagemakerConfig": (".llms.sagemaker.completion.transformation", "SagemakerConfig"),
    "SagemakerChatConfig": (".llms.sagemaker.chat.transformation", "SagemakerChatConfig"),
    "CohereChatConfig": (".llms.cohere.chat.transformation", "CohereChatConfig"),
    "AnthropicMessagesConfig": (".llms.anthropic.experimental_pass_through.messages.transformation", "AnthropicMessagesConfig"),
    "AmazonAnthropicClaudeMessagesConfig": (".llms.bedrock.messages.invoke_transformations.anthropic_claude3_transformation", "AmazonAnthropicClaudeMessagesConfig"),
    "TogetherAIConfig": (".llms.together_ai.chat", "TogetherAIConfig"),
    "NLPCloudConfig": (".llms.nlp_cloud.chat.handler", "NLPCloudConfig"),
    "VertexGeminiConfig": (".llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini", "VertexGeminiConfig"),
    "GoogleAIStudioGeminiConfig": (".llms.gemini.chat.transformation", "GoogleAIStudioGeminiConfig"),
    "VertexAIAnthropicConfig": (".llms.vertex_ai.vertex_ai_partner_models.anthropic.transformation", "VertexAIAnthropicConfig"),
    "VertexAILlama3Config": (".llms.vertex_ai.vertex_ai_partner_models.llama3.transformation", "VertexAILlama3Config"),
    "VertexAIAi21Config": (".llms.vertex_ai.vertex_ai_partner_models.ai21.transformation", "VertexAIAi21Config"),
    "AmazonCohereChatConfig": (".llms.bedrock.chat.invoke_handler", "AmazonCohereChatConfig"),
    "AmazonBedrockGlobalConfig": (".llms.bedrock.common_utils", "AmazonBedrockGlobalConfig"),
    "AmazonAI21Config": (".llms.bedrock.chat.invoke_transformations.amazon_ai21_transformation", "AmazonAI21Config"),
    "AmazonInvokeNovaConfig": (".llms.bedrock.chat.invoke_transformations.amazon_nova_transformation", "AmazonInvokeNovaConfig"),
    "AmazonQwen2Config": (".llms.bedrock.chat.invoke_transformations.amazon_qwen2_transformation", "AmazonQwen2Config"),
    "AmazonQwen3Config": (".llms.bedrock.chat.invoke_transformations.amazon_qwen3_transformation", "AmazonQwen3Config"),
    # Aliases for backwards compatibility
    "VertexAIConfig": (".llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini", "VertexGeminiConfig"),  # Alias
    "GeminiConfig": (".llms.gemini.chat.transformation", "GoogleAIStudioGeminiConfig"),  # Alias
    "AmazonAnthropicConfig": (".llms.bedrock.chat.invoke_transformations.anthropic_claude2_transformation", "AmazonAnthropicConfig"),
    "AmazonAnthropicClaudeConfig": (".llms.bedrock.chat.invoke_transformations.anthropic_claude3_transformation", "AmazonAnthropicClaudeConfig"),
    "AmazonCohereConfig": (".llms.bedrock.chat.invoke_transformations.amazon_cohere_transformation", "AmazonCohereConfig"),
    "AmazonLlamaConfig": (".llms.bedrock.chat.invoke_transformations.amazon_llama_transformation", "AmazonLlamaConfig"),
    "AmazonDeepSeekR1Config": (".llms.bedrock.chat.invoke_transformations.amazon_deepseek_transformation", "AmazonDeepSeekR1Config"),
    "AmazonMistralConfig": (".llms.bedrock.chat.invoke_transformations.amazon_mistral_transformation", "AmazonMistralConfig"),
    "AmazonTitanConfig": (".llms.bedrock.chat.invoke_transformations.amazon_titan_transformation", "AmazonTitanConfig"),
    "AmazonTwelveLabsPegasusConfig": (".llms.bedrock.chat.invoke_transformations.amazon_twelvelabs_pegasus_transformation", "AmazonTwelveLabsPegasusConfig"),
    "AmazonInvokeConfig": (".llms.bedrock.chat.invoke_transformations.base_invoke_transformation", "AmazonInvokeConfig"),
    "AmazonBedrockOpenAIConfig": (".llms.bedrock.chat.invoke_transformations.amazon_openai_transformation", "AmazonBedrockOpenAIConfig"),
    "AmazonStabilityConfig": (".llms.bedrock.image_generation.amazon_stability1_transformation", "AmazonStabilityConfig"),
    "AmazonStability3Config": (".llms.bedrock.image_generation.amazon_stability3_transformation", "AmazonStability3Config"),
    "AmazonNovaCanvasConfig": (".llms.bedrock.image_generation.amazon_nova_canvas_transformation", "AmazonNovaCanvasConfig"),
    "AmazonTitanG1Config": (".llms.bedrock.embed.amazon_titan_g1_transformation", "AmazonTitanG1Config"),
    "AmazonTitanMultimodalEmbeddingG1Config": (".llms.bedrock.embed.amazon_titan_multimodal_transformation", "AmazonTitanMultimodalEmbeddingG1Config"),
    "CohereV2ChatConfig": (".llms.cohere.chat.v2_transformation", "CohereV2ChatConfig"),
    "BedrockCohereEmbeddingConfig": (".llms.bedrock.embed.cohere_transformation", "BedrockCohereEmbeddingConfig"),
    "TwelveLabsMarengoEmbeddingConfig": (".llms.bedrock.embed.twelvelabs_marengo_transformation", "TwelveLabsMarengoEmbeddingConfig"),
    "AmazonNovaEmbeddingConfig": (".llms.bedrock.embed.amazon_nova_transformation", "AmazonNovaEmbeddingConfig"),
    "OpenAIConfig": (".llms.openai.openai", "OpenAIConfig"),
    "MistralEmbeddingConfig": (".llms.openai.openai", "MistralEmbeddingConfig"),
    "OpenAIImageVariationConfig": (".llms.openai.image_variations.transformation", "OpenAIImageVariationConfig"),
    "DeepInfraConfig": (".llms.deepinfra.chat.transformation", "DeepInfraConfig"),
    "DeepgramAudioTranscriptionConfig": (".llms.deepgram.audio_transcription.transformation", "DeepgramAudioTranscriptionConfig"),
    "TopazImageVariationConfig": (".llms.topaz.image_variations.transformation", "TopazImageVariationConfig"),
    "OpenAITextCompletionConfig": ("litellm.llms.openai.completion.transformation", "OpenAITextCompletionConfig"),
    "GroqChatConfig": (".llms.groq.chat.transformation", "GroqChatConfig"),
    "GenAIHubOrchestrationConfig": (".llms.sap.chat.transformation", "GenAIHubOrchestrationConfig"),
    "VoyageEmbeddingConfig": (".llms.voyage.embedding.transformation", "VoyageEmbeddingConfig"),
    "VoyageContextualEmbeddingConfig": (".llms.voyage.embedding.transformation_contextual", "VoyageContextualEmbeddingConfig"),
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
]

