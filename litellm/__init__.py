### Hide pydantic namespace conflict warnings globally ###
import warnings

from litellm._lazy_module import _Alias, _ImportStructure, _LazyModule

warnings.filterwarnings("ignore", message=".*conflict with protected namespace.*")
import os
import re

### INIT VARIABLES ###########
import threading
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Union,
    get_args,
)

import dotenv
import httpx

_LITELLM_LAZY_IMPORTS = os.getenv("LITELLM_LAZY_IMPORT", "true").lower() == "true"


_import_structures: _ImportStructure = {}
if TYPE_CHECKING and _LITELLM_LAZY_IMPORTS:
    from litellm._logging import (
        _turn_on_debug,
        _turn_on_json,
        json_logs,
        log_level,
        set_verbose,
        verbose_logger,
    )
    from litellm.caching.caching import Cache, DualCache, InMemoryCache, RedisCache
    from litellm.caching.llm_caching_handler import LLMClientCache
    from litellm.constants import (
        BEDROCK_INVOKE_PROVIDERS_LITERAL,
        DEFAULT_BATCH_SIZE,
        DEFAULT_FLUSH_INTERVAL_SECONDS,
        DEFAULT_MAX_RETRIES,
        DEFAULT_REPLICATE_POLLING_DELAY_SECONDS,
        DEFAULT_REPLICATE_POLLING_RETRIES,
        HUMANLOOP_PROMPT_CACHE_TTL_SECONDS,
        LITELLM_CHAT_PROVIDERS,
    )
    from litellm.constants import OPENAI_CHAT_COMPLETION_PARAMS
    from litellm.constants import (
        OPENAI_CHAT_COMPLETION_PARAMS as _openai_completion_params,  # backwards compatibility
    )
    from litellm.constants import OPENAI_FINISH_REASONS
    from litellm.constants import (
        OPENAI_FINISH_REASONS as _openai_finish_reasons,  # backwards compatibility
    )
    from litellm.constants import (
        REPEATED_STREAMING_CHUNK_LIMIT,
        ROUTER_MAX_FALLBACKS,
        _openai_like_providers,
        baseten_models,
        bedrock_embedding_models,
        clarifai_models,
        cohere_embedding_models,
        empower_models,
        huggingface_models,
        known_tokenizer_config,
        open_ai_embedding_models,
        openai_compatible_endpoints,
        openai_compatible_providers,
        openai_text_completion_compatible_providers,
        replicate_models,
        request_timeout,
        together_ai_models,
    )
    from litellm.integrations.custom_logger import CustomLogger
    from litellm.litellm_core_utils.logging_callback_manager import (
        LoggingCallbackManager,
    )
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
    from litellm.proxy._types import (
        KeyManagementSettings,
        KeyManagementSystem,
        LiteLLM_UpperboundKeyGenerateParams,
    )
    from litellm.types.guardrails import GuardrailItem
    from litellm.types.llms.bedrock import COHERE_EMBEDDING_INPUT_TYPES
    from litellm.types.llms.openai import CreateFileRequest
    from litellm.types.utils import (
        all_litellm_params,  # maintain backwards compatibility for root param
    )
    from litellm.types.utils import (
        BudgetConfig,
        Choices,
        CredentialItem,
        ImageObject,
        LlmProviders,
        Message,
        StandardKeyGenerationConfig,
    )
else:
    _import_structures.setdefault("litellm._logging", []).extend(
        [
            "_turn_on_debug",
            "_turn_on_json",
            "json_logs",
            "log_level",
            "set_verbose",
            "verbose_logger",
        ]
    )
    _import_structures.setdefault("litellm.caching.caching", []).extend(
        [
            "Cache",
            "DualCache",
            "InMemoryCache",
            "RedisCache",
        ]
    )
    _import_structures.setdefault("litellm.caching.llm_caching_handler", []).extend(
        ["LLMClientCache"]
    )
    _import_structures.setdefault("litellm.constants", []).extend(
        [
            "BEDROCK_INVOKE_PROVIDERS_LITERAL",
            "DEFAULT_BATCH_SIZE",
            "DEFAULT_FLUSH_INTERVAL_SECONDS",
            "DEFAULT_MAX_RETRIES",
            "DEFAULT_REPLICATE_POLLING_DELAY_SECONDS",
            "DEFAULT_REPLICATE_POLLING_RETRIES",
            "HUMANLOOP_PROMPT_CACHE_TTL_SECONDS",
            "LITELLM_CHAT_PROVIDERS",
        ]
    )
    _import_structures.setdefault("litellm.constants", []).extend(
        [
            "OPENAI_CHAT_COMPLETION_PARAMS",
            _Alias(
                "OPENAI_CHAT_COMPLETION_PARAMS", "_openai_completion_params"
            ),  # backwards compatibility
            "OPENAI_FINISH_REASONS",
            _Alias(
                "OPENAI_FINISH_REASONS", "_openai_finish_reasons"
            ),  # backwards compatibility
            "REPEATED_STREAMING_CHUNK_LIMIT",
            "ROUTER_MAX_FALLBACKS",
            "_openai_like_providers",
            "baseten_models",
            "bedrock_embedding_models",
            "clarifai_models",
            "cohere_embedding_models",
            "empower_models",
            "huggingface_models",
            "known_tokenizer_config",
            "open_ai_embedding_models",
            "openai_compatible_endpoints",
            "openai_compatible_providers",
            "openai_text_completion_compatible_providers",
            "replicate_models",
            "request_timeout",
            "together_ai_models",
        ]
    )
    _import_structures.setdefault("litellm.integrations.custom_logger", []).extend(
        ["CustomLogger"]
    )
    _import_structures.setdefault(
        "litellm.litellm_core_utils.logging_callback_manager", []
    ).extend(
        [
            "LoggingCallbackManager",
        ]
    )
    _import_structures.setdefault("litellm.llms.custom_httpx.http_handler", []).extend(
        [
            "AsyncHTTPHandler",
            "HTTPHandler",
        ]
    )
    _import_structures.setdefault("litellm.proxy._types", []).extend(
        [
            "KeyManagementSettings",
            "KeyManagementSystem",
            "LiteLLM_UpperboundKeyGenerateParams",
        ]
    )
    _import_structures.setdefault("litellm.types.guardrails", []).extend(
        ["GuardrailItem"]
    )
    _import_structures.setdefault("litellm.types.llms.bedrock", []).extend(
        ["COHERE_EMBEDDING_INPUT_TYPES"]
    )
    _import_structures.setdefault("litellm.types.llms.openai", []).extend(
        ["CreateFileRequest"]
    )
    _import_structures.setdefault("litellm.types.utils", []).extend(
        [
            "all_litellm_params",  # maintain backwards compatibility for root param
            "BudgetConfig",
            "CredentialItem",
            "ImageObject",
            "LlmProviders",
            "StandardKeyGenerationConfig",
            "Message",
            "Choices",
        ]
    )


################################################
### Callbacks /Logging / Success / Failure Handlers #####
if TYPE_CHECKING and _LITELLM_LAZY_IMPORTS:
    from litellm._variables.handlers import (
        CALLBACK_TYPES,
        _async_failure_callback,
        _async_input_callback,
        _async_success_callback,
        _custom_logger_compatible_callbacks_literal,
        _known_custom_logger_compatible_callbacks,
        add_user_information_to_llm_headers,
        argilla_batch_size,
        argilla_transformation_object,
        callbacks,
        datadog_use_v1,
        failure_callback,
        filter_invalid_headers,
        gcs_pub_sub_use_v1,
        input_callback,
        langfuse_default_tags,
        langsmith_batch_size,
        log_raw_request_response,
        logged_real_time_event_types,
        logging_callback_manager,
        post_call_rules,
        pre_call_rules,
        prometheus_initialize_budget_metrics,
        redact_messages_in_exceptions,
        redact_user_api_key_info,
        service_callback,
        store_audit_logs,
        success_callback,
        turn_off_message_logging,
    )
else:
    _import_structures.setdefault("litellm._variables.handlers", []).extend(
        [
            "CALLBACK_TYPES",
            "_custom_logger_compatible_callbacks_literal",
            "failure_callback",
            "input_callback",
            "logging_callback_manager",
            "service_callback",
            "success_callback",
            "logged_real_time_event_types",
            "_known_custom_logger_compatible_callbacks",
            "callbacks",
            "langfuse_default_tags",
            "langsmith_batch_size",
            "prometheus_initialize_budget_metrics",
            "argilla_batch_size",
            "datadog_use_v1",
            "gcs_pub_sub_use_v1",
            "argilla_transformation_object",
            "_async_input_callback",
            "_async_success_callback",
            "_async_failure_callback",
            "pre_call_rules",
            "post_call_rules",
            "turn_off_message_logging",
            "log_raw_request_response",
            "redact_messages_in_exceptions",
            "redact_user_api_key_info",
            "filter_invalid_headers",
            "add_user_information_to_llm_headers",
            "store_audit_logs",
        ]
    )
### end of callbacks #############

if TYPE_CHECKING and _LITELLM_LAZY_IMPORTS:
    from litellm._variables.auth import (
        ai21_key,
        aleph_alpha_key,
        anthropic_key,
        api_key,
        azure_key,
        baseten_key,
        clarifai_key,
        cloudflare_api_key,
        cohere_key,
        common_cloud_provider_auth_params,
        databricks_key,
        disable_add_transform_inline_image_block,
        disable_streaming_logging,
        drop_params,
        email,
        enable_azure_ad_token_refresh,
        groq_key,
        huggingface_key,
        in_memory_llm_clients_cache,
        infinity_key,
        maritalk_key,
        max_tokens,
        modify_params,
        nlp_cloud_key,
        ollama_key,
        openai_key,
        openai_like_key,
        openrouter_key,
        predibase_key,
        predibase_tenant_id,
        replicate_key,
        retry,
        safe_memory_mode,
        snowflake_key,
        ssl_certificate,
        ssl_verify,
        telemetry,
        togetherai_api_key,
        token,
        use_client,
        vertex_location,
        vertex_project,
    )
else:
    _import_structures.setdefault("litellm._variables.auth", []).extend(
        [
            "ai21_key",
            "aleph_alpha_key",
            "anthropic_key",
            "api_key",
            "azure_key",
            "baseten_key",
            "clarifai_key",
            "cloudflare_api_key",
            "cohere_key",
            "common_cloud_provider_auth_params",
            "databricks_key",
            "disable_add_transform_inline_image_block",
            "disable_streaming_logging",
            "drop_params",
            "email",
            "enable_azure_ad_token_refresh",
            "groq_key",
            "huggingface_key",
            "infinity_key",
            "maritalk_key",
            "max_tokens",
            "modify_params",
            "nlp_cloud_key",
            "ollama_key",
            "openai_key",
            "openai_like_key",
            "openrouter_key",
            "predibase_key",
            "predibase_tenant_id",
            "replicate_key",
            "retry",
            "safe_memory_mode",
            "snowflake_key",
            "ssl_certificate",
            "ssl_verify",
            "telemetry",
            "togetherai_api_key",
            "token",
            "use_client",
            "vertex_location",
            "vertex_project",
            "in_memory_llm_clients_cache",
        ]
    )

if TYPE_CHECKING and _LITELLM_LAZY_IMPORTS:
    from litellm._variables.model_lists import (
        AZURE_DEFAULT_API_VERSION,
        BEDROCK_CONVERSE_MODELS,
        COHERE_DEFAULT_EMBEDDING_INPUT_TYPE,
        WATSONX_DEFAULT_API_VERSION,
        MyLocal,
        _current_cost,
        _google_kms_resource_name,
        _key_management_settings,
        _key_management_system,
        _thread_context,
        aclient_session,
        add_function_to_prompt,
        add_known_models,
        ai21_chat_models,
        ai21_models,
        aleph_alpha_models,
        all_embedding_models,
        allowed_fails,
        anthropic_models,
        anyscale_models,
        api_base,
        api_version,
        assemblyai_models,
        azure_ai_models,
        azure_embedding_models,
        azure_llms,
        azure_models,
        azure_text_models,
        banned_keywords_list,
        bedrock_converse_models,
        bedrock_models,
        blocked_user_list,
        budget_duration,
        cache,
        caching,
        caching_with_models,
        cerebras_models,
        client_session,
        cloudflare_models,
        codestral_models,
        cohere_chat_models,
        cohere_models,
        config_path,
        content_policy_fallbacks,
        context_window_fallbacks,
        credential_list,
        custom_prometheus_metadata_labels,
        custom_prompt_dict,
        databricks_models,
        deepinfra_models,
        deepseek_models,
        default_fallbacks,
        default_in_memory_ttl,
        default_internal_user_params,
        default_key_generate_params,
        default_max_internal_user_budget,
        default_redis_batch_cache_expiry,
        default_redis_ttl,
        default_soft_budget,
        default_team_settings,
        disable_end_user_cost_tracking,
        disable_end_user_cost_tracking_prometheus_only,
        dynamodb_table_name,
        enable_caching_on_provider_specific_optional_params,
        enable_json_schema_validation,
        enable_loadbalancing_on_batch_endpoints,
        enable_preview_features,
        error_logs,
        fallbacks,
        fireworks_ai_embedding_models,
        fireworks_ai_models,
        force_ipv4,
        forward_traceparent_to_llm_provider,
        friendliai_models,
        galadriel_models,
        gemini_models,
        generic_logger_headers,
        google_moderation_confidence_threshold,
        groq_models,
        guardrail_name_config_map,
        headers,
        identify,
        internal_user_budget_duration,
        is_bedrock_pricing_only_model,
        is_openai_finetune_model,
        jina_ai_models,
        key_generation_settings,
        llamaguard_model_name,
        llamaguard_unsafe_content_categories,
        llm_guard_mode,
        logging,
        longer_context_model_fallback_dict,
        maritalk_models,
        max_budget,
        max_end_user_budget,
        max_fallbacks,
        max_internal_user_budget,
        max_ui_session_budget,
        max_user_budget,
        mistral_chat_models,
        model_alias_map,
        model_cost,
        model_cost_map_url,
        model_fallbacks,
        model_group_alias_map,
        model_list,
        model_list_set,
        models_by_provider,
        module_level_aclient,
        module_level_client,
        nlp_cloud_models,
        num_retries,
        num_retries_per_request,
        ollama_models,
        open_ai_chat_completion_models,
        open_ai_text_completion_models,
        openai_image_generation_models,
        openai_moderations_model_name,
        openrouter_models,
        organization,
        output_parse_pii,
        palm_models,
        perplexity_models,
        petals_models,
        presidio_ad_hoc_recognizers,
        priority_reservation,
        project,
        provider_list,
        return_response_headers,
        s3_callback_params,
        sambanova_models,
        secret_manager_client,
        snowflake_models,
        suppress_debug_info,
        tag_budget_config,
        text_completion_codestral_models,
        upperbound_key_generate_params,
        vertex_ai_ai21_models,
        vertex_ai_image_models,
        vertex_ai_safety_settings,
        vertex_anthropic_models,
        vertex_chat_models,
        vertex_code_chat_models,
        vertex_code_text_models,
        vertex_embedding_models,
        vertex_language_models,
        vertex_llama3_models,
        vertex_mistral_models,
        vertex_text_models,
        vertex_vision_models,
        voyage_models,
        watsonx_models,
        xai_models,
    )
else:
    _import_structures.setdefault("litellm._variables.model_lists", []).extend(
        [
            "AZURE_DEFAULT_API_VERSION",
            "BEDROCK_CONVERSE_MODELS",
            "COHERE_DEFAULT_EMBEDDING_INPUT_TYPE",
            "WATSONX_DEFAULT_API_VERSION",
            "MyLocal",
            "_current_cost",
            "_google_kms_resource_name",
            "_key_management_settings",
            "_key_management_system",
            "_thread_context",
            "aclient_session",
            "add_function_to_prompt",
            "add_known_models",
            "ai21_chat_models",
            "ai21_models",
            "aleph_alpha_models",
            "all_embedding_models",
            "allowed_fails",
            "anthropic_models",
            "anyscale_models",
            "api_base",
            "api_version",
            "assemblyai_models",
            "azure_ai_models",
            "azure_embedding_models",
            "azure_llms",
            "azure_models",
            "azure_text_models",
            "banned_keywords_list",
            "bedrock_converse_models",
            "bedrock_models",
            "blocked_user_list",
            "budget_duration",
            "cache",
            "caching",
            "caching_with_models",
            "cerebras_models",
            "client_session",
            "cloudflare_models",
            "codestral_models",
            "cohere_chat_models",
            "cohere_models",
            "config_path",
            "content_policy_fallbacks",
            "context_window_fallbacks",
            "credential_list",
            "custom_prometheus_metadata_labels",
            "custom_prompt_dict",
            "databricks_models",
            "deepinfra_models",
            "deepseek_models",
            "default_fallbacks",
            "default_in_memory_ttl",
            "default_internal_user_params",
            "default_key_generate_params",
            "default_max_internal_user_budget",
            "default_redis_batch_cache_expiry",
            "default_redis_ttl",
            "default_soft_budget",
            "default_team_settings",
            "disable_end_user_cost_tracking",
            "disable_end_user_cost_tracking_prometheus_only",
            "dynamodb_table_name",
            "enable_caching_on_provider_specific_optional_params",
            "enable_json_schema_validation",
            "enable_loadbalancing_on_batch_endpoints",
            "enable_preview_features",
            "error_logs",
            "fallbacks",
            "fireworks_ai_embedding_models",
            "fireworks_ai_models",
            "force_ipv4",
            "forward_traceparent_to_llm_provider",
            "friendliai_models",
            "galadriel_models",
            "gemini_models",
            "generic_logger_headers",
            "google_moderation_confidence_threshold",
            "groq_models",
            "guardrail_name_config_map",
            "headers",
            "identify",
            "internal_user_budget_duration",
            "is_bedrock_pricing_only_model",
            "is_openai_finetune_model",
            "jina_ai_models",
            "key_generation_settings",
            "llamaguard_model_name",
            "llamaguard_unsafe_content_categories",
            "llm_guard_mode",
            "logging",
            "longer_context_model_fallback_dict",
            "maritalk_models",
            "max_budget",
            "max_end_user_budget",
            "max_fallbacks",
            "max_internal_user_budget",
            "max_ui_session_budget",
            "max_user_budget",
            "mistral_chat_models",
            "model_alias_map",
            "model_cost",
            "model_cost_map_url",
            "model_fallbacks",
            "model_group_alias_map",
            "model_list",
            "model_list_set",
            "models_by_provider",
            "module_level_aclient",
            "module_level_client",
            "nlp_cloud_models",
            "num_retries",
            "num_retries_per_request",
            "ollama_models",
            "open_ai_chat_completion_models",
            "open_ai_text_completion_models",
            "openai_image_generation_models",
            "openai_moderations_model_name",
            "openrouter_models",
            "organization",
            "output_parse_pii",
            "palm_models",
            "perplexity_models",
            "petals_models",
            "presidio_ad_hoc_recognizers",
            "priority_reservation",
            "project",
            "provider_list",
            "return_response_headers",
            "s3_callback_params",
            "sambanova_models",
            "secret_manager_client",
            "snowflake_models",
            "suppress_debug_info",
            "tag_budget_config",
            "text_completion_codestral_models",
            "upperbound_key_generate_params",
            "vertex_ai_ai21_models",
            "vertex_ai_image_models",
            "vertex_ai_safety_settings",
            "vertex_anthropic_models",
            "vertex_chat_models",
            "vertex_code_chat_models",
            "vertex_code_text_models",
            "vertex_embedding_models",
            "vertex_language_models",
            "vertex_llama3_models",
            "vertex_mistral_models",
            "vertex_text_models",
            "vertex_vision_models",
            "voyage_models",
            "watsonx_models",
            "xai_models",
        ]
    )

if TYPE_CHECKING and _LITELLM_LAZY_IMPORTS:
    from litellm.litellm_core_utils.core_helpers import remove_index_from_tool_calls
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
    from litellm.litellm_core_utils.litellm_logging import Logging, modify_integration
    from litellm.litellm_core_utils.token_counter import get_modified_max_tokens
    from litellm.utils import (
        ALL_LITELLM_RESPONSE_TYPES,
        OPENAI_RESPONSE_HEADERS,
        BudgetConfig,
        CallTypes,
        ChatCompletionDeltaToolCall,
        ChatCompletionMessageToolCall,
        CostPerToken,
        CredentialItem,
        CustomHuggingfaceTokenizer,
        CustomStreamWrapper,
        Delta,
        Embedding,
        EmbeddingResponse,
        Function,
        ImageObject,
        ImageResponse,
        LlmProviders,
        LlmProvidersSet,
        ModelInfo,
        ModelInfoBase,
        ModelResponse,
        ModelResponseListIterator,
        ModelResponseStream,
        ProviderField,
        ProviderSpecificModelInfo,
        RawRequestTypedDict,
        SelectTokenizerResponse,
        StandardKeyGenerationConfig,
        StreamingChoices,
        TextChoices,
        TextCompletionResponse,
        TranscriptionResponse,
        Usage,
        _calculate_retry_after,
        _should_retry,
        acreate,
        all_litellm_params,
        check_valid_key,
        client,
        create_pretrained_tokenizer,
        create_tokenizer,
        decode,
        encode,
        exception_type,
        get_api_base,
        get_first_chars_messages,
        get_litellm_params,
        get_max_tokens,
        get_model_info,
        get_optional_params,
        get_provider_fields,
        get_response_string,
        get_supported_openai_params,
        register_model,
        register_prompt_template,
        supports_audio_input,
        supports_audio_output,
        supports_function_calling,
        supports_parallel_function_calling,
        supports_response_schema,
        supports_system_messages,
        supports_vision,
        supports_web_search,
        token_counter,
        validate_environment,
    )

    from .cost_calculator import completion_cost
    from .llms.ai21.chat.transformation import AI21ChatConfig
    from .llms.ai21.chat.transformation import AI21ChatConfig as AI21Config
    from .llms.aiohttp_openai.chat.transformation import AiohttpOpenAIChatConfig
    from .llms.anthropic.chat.transformation import AnthropicConfig
    from .llms.anthropic.common_utils import AnthropicModelInfo
    from .llms.anthropic.completion.transformation import AnthropicTextConfig
    from .llms.anthropic.experimental_pass_through.messages.transformation import (
        AnthropicMessagesConfig,
    )
    from .llms.azure_ai.rerank.transformation import AzureAIRerankConfig
    from .llms.bedrock.chat.converse_transformation import AmazonConverseConfig
    from .llms.clarifai.chat.transformation import ClarifaiConfig
    from .llms.cloudflare.chat.transformation import CloudflareChatConfig
    from .llms.cohere.completion.transformation import CohereTextConfig as CohereConfig
    from .llms.cohere.rerank.transformation import CohereRerankConfig
    from .llms.cohere.rerank_v2.transformation import CohereRerankV2Config
    from .llms.custom_llm import CustomLLM
    from .llms.databricks.chat.transformation import DatabricksConfig
    from .llms.databricks.embed.transformation import DatabricksEmbeddingConfig
    from .llms.deprecated_providers.aleph_alpha import AlephAlphaConfig
    from .llms.deprecated_providers.palm import (  # here to prevent breaking changes
        PalmConfig,
    )
    from .llms.empower.chat.transformation import EmpowerChatConfig
    from .llms.galadriel.chat.transformation import GaladrielChatConfig
    from .llms.gemini.chat.transformation import GoogleAIStudioGeminiConfig
    from .llms.gemini.chat.transformation import (
        GoogleAIStudioGeminiConfig as GeminiConfig,  # aliased to maintain backwards compatibility
    )
    from .llms.gemini.common_utils import GeminiModelInfo
    from .llms.github.chat.transformation import GithubChatConfig
    from .llms.groq.stt.transformation import GroqSTTConfig
    from .llms.huggingface.chat.transformation import HuggingFaceChatConfig
    from .llms.huggingface.embedding.transformation import HuggingFaceEmbeddingConfig
    from .llms.infinity.rerank.transformation import InfinityRerankConfig
    from .llms.jina_ai.rerank.transformation import JinaAIRerankConfig
    from .llms.maritalk import MaritalkConfig
    from .llms.nlp_cloud.chat.handler import NLPCloudConfig
    from .llms.oobabooga.chat.transformation import OobaboogaConfig
    from .llms.openai_like.chat.handler import OpenAILikeChatConfig
    from .llms.openrouter.chat.transformation import OpenrouterConfig
    from .llms.petals.completion.transformation import PetalsConfig
    from .llms.predibase.chat.transformation import PredibaseConfig
    from .llms.replicate.chat.transformation import ReplicateConfig
    from .llms.snowflake.chat.transformation import SnowflakeConfig
    from .llms.together_ai.chat import TogetherAIConfig
    from .llms.together_ai.completion.transformation import (
        TogetherAITextCompletionConfig,
    )
    from .llms.triton.completion.transformation import (
        TritonConfig,
        TritonGenerateConfig,
        TritonInferConfig,
    )
    from .llms.triton.embedding.transformation import TritonEmbeddingConfig
    from .llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
        VertexGeminiConfig,
    )
    from .llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
        VertexGeminiConfig as VertexAIConfig,
    )
    from .llms.vertex_ai.vertex_embeddings.transformation import (
        VertexAITextEmbeddingConfig,
    )
    from .timeout import timeout
    from .utils import (
        ALL_LITELLM_RESPONSE_TYPES,
        EmbeddingResponse,
        ImageResponse,
        ModelResponse,
        ModelResponseListIterator,
        ModelResponseStream,
        TextCompletionResponse,
        TranscriptionResponse,
        _calculate_retry_after,
        _should_retry,
        acreate,
        check_valid_key,
        client,
        create_pretrained_tokenizer,
        create_tokenizer,
        decode,
        encode,
        exception_type,
        get_api_base,
        get_first_chars_messages,
        get_litellm_params,
        get_max_tokens,
        get_model_info,
        get_optional_params,
        get_provider_fields,
        get_response_string,
        get_supported_openai_params,
        register_model,
        register_prompt_template,
        supports_audio_input,
        supports_audio_output,
        supports_function_calling,
        supports_parallel_function_calling,
        supports_response_schema,
        supports_system_messages,
        supports_vision,
        supports_web_search,
        token_counter,
        validate_environment,
    )
else:
    _import_structures.setdefault("litellm.litellm_core_utils.core_helpers", []).extend(
        [
            "remove_index_from_tool_calls",
        ]
    )
    _import_structures.setdefault(
        "litellm.litellm_core_utils.get_llm_provider_logic", []
    ).extend(
        [
            "get_llm_provider",
        ]
    )
    _import_structures.setdefault(
        "litellm.litellm_core_utils.litellm_logging", []
    ).extend(
        [
            "Logging",
            "modify_integration",
        ]
    )
    _import_structures.setdefault(
        "litellm.litellm_core_utils.token_counter", []
    ).extend(
        [
            "get_modified_max_tokens",
        ]
    )

    _import_structures.setdefault("litellm.cost_calculator", []).extend(
        ["completion_cost"]
    )
    _import_structures.setdefault("litellm.llms.ai21.chat.transformation", []).extend(
        [
            "AI21ChatConfig",
            _Alias("AI21ChatConfig", "AI21Config"),
        ]
    )
    _import_structures.setdefault(
        "litellm.llms.aiohttp_openai.chat.transformation", []
    ).extend(["AiohttpOpenAIChatConfig"])
    _import_structures.setdefault(
        "litellm.llms.anthropic.chat.transformation", []
    ).extend(["AnthropicConfig"])
    _import_structures.setdefault("litellm.llms.anthropic.common_utils", []).extend(
        ["AnthropicModelInfo"]
    )
    _import_structures.setdefault(
        "litellm.llms.anthropic.completion.transformation", []
    ).extend(["AnthropicTextConfig"])
    _import_structures.setdefault(
        "litellm.llms.anthropic.experimental_pass_through.messages.transformation", []
    ).extend(
        [
            "AnthropicMessagesConfig",
        ]
    )
    _import_structures.setdefault(
        "litellm.llms.azure_ai.rerank.transformation", []
    ).extend(["AzureAIRerankConfig"])
    _import_structures.setdefault(
        "litellm.llms.bedrock.chat.converse_transformation", []
    ).extend(["AmazonConverseConfig"])
    _import_structures.setdefault(
        "litellm.llms.clarifai.chat.transformation", []
    ).extend(["ClarifaiConfig"])
    _import_structures.setdefault(
        "litellm.llms.cloudflare.chat.transformation", []
    ).extend(["CloudflareChatConfig"])
    _import_structures.setdefault(
        "litellm.llms.cohere.completion.transformation", []
    ).extend([_Alias("CohereTextConfig", "CohereConfig")])
    _import_structures.setdefault(
        "litellm.llms.cohere.rerank.transformation", []
    ).extend(["CohereRerankConfig"])
    _import_structures.setdefault(
        "litellm.llms.cohere.rerank_v2.transformation", []
    ).extend(["CohereRerankV2Config"])
    _import_structures.setdefault("litellm.llms.custom_llm", []).extend(["CustomLLM"])
    _import_structures.setdefault(
        "litellm.llms.databricks.chat.transformation", []
    ).extend(["DatabricksConfig"])
    _import_structures.setdefault(
        "litellm.llms.databricks.embed.transformation", []
    ).extend(["DatabricksEmbeddingConfig"])
    _import_structures.setdefault(
        "litellm.llms.deprecated_providers.aleph_alpha", []
    ).extend(["AlephAlphaConfig"])
    _import_structures.setdefault("litellm.llms.deprecated_providers.palm", []).extend(
        [
            "PalmConfig",  # here to prevent breaking changes
        ]
    )
    _import_structures.setdefault(
        "litellm.llms.empower.chat.transformation", []
    ).extend(
        [
            "EmpowerChatConfig",
        ]
    )
    _import_structures.setdefault(
        "litellm.llms.galadriel.chat.transformation", []
    ).extend(
        [
            "GaladrielChatConfig",
        ]
    )
    _import_structures.setdefault("litellm.llms.gemini.chat.transformation", []).extend(
        [
            "GoogleAIStudioGeminiConfig",
            "GeminiConfig",  # aliased to maintain backwards compatibility
        ]
    )
    _import_structures.setdefault("litellm.llms.gemini.common_utils", []).extend(
        [
            "GeminiModelInfo",
        ]
    )
    _import_structures.setdefault("litellm.llms.github.chat.transformation", []).extend(
        [
            "GithubChatConfig",
        ]
    )
    _import_structures.setdefault("litellm.llms.groq.stt.transformation", []).extend(
        [
            "GroqSTTConfig",
        ]
    )
    _import_structures.setdefault(
        "litellm.llms.huggingface.chat.transformation", []
    ).extend(["HuggingFaceChatConfig"])
    _import_structures.setdefault(
        "litellm.llms.huggingface.embedding.transformation", []
    ).extend(["HuggingFaceEmbeddingConfig"])
    _import_structures.setdefault(
        "litellm.llms.infinity.rerank.transformation", []
    ).extend(
        [
            "InfinityRerankConfig",
        ]
    )
    _import_structures.setdefault(
        "litellm.llms.jina_ai.rerank.transformation", []
    ).extend(
        [
            "JinaAIRerankConfig",
        ]
    )
    _import_structures.setdefault("litellm.llms.maritalk", []).extend(
        [
            "MaritalkConfig",
        ]
    )
    _import_structures.setdefault("litellm.llms.nlp_cloud.chat.handler", []).extend(
        [
            "NLPCloudConfig",
        ]
    )
    _import_structures.setdefault(
        "litellm.llms.oobabooga.chat.transformation", []
    ).extend(
        [
            "OobaboogaConfig",
        ]
    )
    _import_structures.setdefault("litellm.llms.openai_like.chat.handler", []).extend(
        [
            "OpenAILikeChatConfig",
        ]
    )
    _import_structures.setdefault(
        "litellm.llms.openrouter.chat.transformation", []
    ).extend(
        [
            "OpenrouterConfig",
        ]
    )
    _import_structures.setdefault(
        "litellm.llms.petals.completion.transformation", []
    ).extend(
        [
            "PetalsConfig",
        ]
    )
    _import_structures.setdefault(
        "litellm.llms.predibase.chat.transformation", []
    ).extend(
        [
            "PredibaseConfig",
        ]
    )
    _import_structures.setdefault(
        "litellm.llms.replicate.chat.transformation", []
    ).extend(
        [
            "ReplicateConfig",
        ]
    )
    _import_structures.setdefault(
        "litellm.llms.snowflake.chat.transformation", []
    ).extend(
        [
            "SnowflakeConfig",
        ]
    )
    _import_structures.setdefault("litellm.llms.together_ai.chat", []).extend(
        [
            "TogetherAIConfig",
        ]
    )
    _import_structures.setdefault(
        "litellm.llms.together_ai.completion.transformation", []
    ).extend(
        [
            "TogetherAITextCompletionConfig",
        ]
    )
    _import_structures.setdefault(
        "litellm.llms.triton.completion.transformation", []
    ).extend(
        [
            "TritonConfig",
            "TritonGenerateConfig",
            "TritonInferConfig",
        ]
    )
    _import_structures.setdefault(
        "litellm.llms.triton.embedding.transformation", []
    ).extend(
        [
            "TritonEmbeddingConfig",
        ]
    )
    _import_structures.setdefault(
        "litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini", []
    ).extend(
        [
            "VertexGeminiConfig",
            "VertexAIConfig",
        ]
    )
    _import_structures.setdefault(
        "litellm.llms.vertex_ai.vertex_embeddings.transformation", []
    ).extend(
        [
            "VertexAITextEmbeddingConfig",
        ]
    )
    _import_structures.setdefault("litellm.timeout", []).extend(
        [
            "timeout",
        ]
    )
    _import_structures.setdefault("litellm.utils", []).extend(
        [
            "CustomStreamWrapper",
            "TextChoices",
            "Usage",
            "PromptTokenDetailsWrapper",
            "CompletionTokensDetailsWrapper",
            "ChatCompletionAudioResponse",
            "ChatCompletionMessageToolCall",
            "ChatCompletionDeltaToolCall",
            "ChoiceLogprobs",
            "Function",
            "FunctionCall",
            "CostPerToken",
            "SupportedCacheControls",
            "LiteLLMCommonStrings",
            "StreamingChoices",
            "StreamingChatCompletionChunk",
            "ModelResponseBase",
            "ModelResponseStream",
            "ModelResponse",
            "Embedding",
            "EmbeddingResponse",
            "Logprobs",
            "TextChoices",
            "ImageObject",
            "TextCompletionResponse",
            "LiteLLMPydanticObjectBase",
            "CallTypesLiteral",
            "ProviderSpecificModelInfo",
            "ProviderField",
            "CallTypes",
            "SearchContextCostPerQuery",
            "GenericStreamingChunk",
            "ModelInfoBase",
            "ModelInfo",
            "ImageResponse",
            "TranscriptionResponse",
            "GenericImageParsingChunk",
            "ResponseFormatChunk",
            "LoggedLiteLLMParams",
            "AdapterCompletionStreamWrapper",
            "StandardLoggingUserAPIKeyMetadata",
            "StandardLoggingMCPToolCall",
            "StandardBuiltInToolsParams",
            "StandardLoggingPromptManagementMetadata",
            "StandardLoggingMetadata",
            "StandardLoggingAdditionalHeaders",
            "StandardLoggingHiddenParams",
            "StandardLoggingModelInformation",
            "StandardLoggingModelCostFailureDebugInformation",
            "StandardLoggingPayloadErrorInformation",
            "StandardLoggingGuardrailInformation",
            "StandardLoggingPayloadStatus",
            "CustomStreamingDecoder",
            "StandardLoggingPayload",
            "StandardPassThroughResponseObject",
            "OPENAI_RESPONSE_HEADERS",
            "StandardCallbackDynamicParams",
            "all_litellm_params",
            "KeyGenerationConfig",
            "TeamUIKeyGenerationConfig",
            "PersonalUIKeyGenerationConfig",
            "StandardKeyGenerationConfig",
            "BudgetConfig",
            "GenericBudgetConfigType",
            "LlmProviders",
            "LlmProvidersSet",
            "LiteLLMLoggingBaseClass",
            "CustomHuggingfaceTokenizer",
            "LITELLM_IMAGE_VARIATION_PROVIDERS",
            "HttpHandlerRequestFields",
            "ProviderSpecificHeader",
            "SelectTokenizerResponse",
            "LiteLLMBatch",
            "RawRequestTypedDict",
            "CredentialBase",
            "CredentialItem",
            "CreateCredentialItem",
            "HiddenParams",
            "Delta",
            "ChatCompletionTokenLogprob",
            "PassthroughCallTypes",
            "ALL_LITELLM_RESPONSE_TYPES",
            "EmbeddingResponse",
            "ImageResponse",
            "ModelResponse",
            "ModelResponseListIterator",
            "ModelResponseStream",
            "TextCompletionResponse",
            "TranscriptionResponse",
            "_calculate_retry_after",
            "_should_retry",
            "acreate",
            "check_valid_key",
            "client",
            "create_pretrained_tokenizer",
            "create_tokenizer",
            "decode",
            "encode",
            "exception_type",
            "get_api_base",
            "get_first_chars_messages",
            "get_litellm_params",
            "get_max_tokens",
            "get_model_info",
            "get_optional_params",
            "get_provider_fields",
            "get_response_string",
            "get_supported_openai_params",
            "register_model",
            "register_prompt_template",
            "supports_audio_input",
            "supports_audio_output",
            "supports_function_calling",
            "supports_parallel_function_calling",
            "supports_response_schema",
            "supports_system_messages",
            "supports_vision",
            "supports_web_search",
            "token_counter",
            "validate_environment",
        ]
    )


if TYPE_CHECKING and _LITELLM_LAZY_IMPORTS:
    from litellm.llms.openai.completion.transformation import OpenAITextCompletionConfig

    from .llms.azure_ai.chat.transformation import AzureAIStudioConfig
    from .llms.bedrock.chat.invoke_handler import (
        AmazonCohereChatConfig,
        AWSEventStreamDecoder,
        BedrockLLM,
        bedrock_tool_name_mappings,
    )
    from .llms.bedrock.chat.invoke_transformations.amazon_ai21_transformation import (
        AmazonAI21Config,
    )
    from .llms.bedrock.chat.invoke_transformations.amazon_cohere_transformation import (
        AmazonCohereConfig,
    )
    from .llms.bedrock.chat.invoke_transformations.amazon_deepseek_transformation import (
        AmazonDeepSeekR1Config,
    )
    from .llms.bedrock.chat.invoke_transformations.amazon_llama_transformation import (
        AmazonLlamaConfig,
    )
    from .llms.bedrock.chat.invoke_transformations.amazon_mistral_transformation import (
        AmazonMistralConfig,
    )
    from .llms.bedrock.chat.invoke_transformations.amazon_nova_transformation import (
        AmazonInvokeNovaConfig,
    )
    from .llms.bedrock.chat.invoke_transformations.amazon_titan_transformation import (
        AmazonTitanConfig,
    )
    from .llms.bedrock.chat.invoke_transformations.anthropic_claude2_transformation import (
        AmazonAnthropicConfig,
    )
    from .llms.bedrock.chat.invoke_transformations.anthropic_claude3_transformation import (
        AmazonAnthropicClaude3Config,
    )
    from .llms.bedrock.chat.invoke_transformations.base_invoke_transformation import (
        AmazonInvokeConfig,
    )
    from .llms.bedrock.common_utils import AmazonBedrockGlobalConfig
    from .llms.bedrock.embed.amazon_titan_g1_transformation import AmazonTitanG1Config
    from .llms.bedrock.embed.amazon_titan_multimodal_transformation import (
        AmazonTitanMultimodalEmbeddingG1Config,
    )
    from .llms.bedrock.embed.amazon_titan_v2_transformation import AmazonTitanV2Config
    from .llms.bedrock.embed.cohere_transformation import BedrockCohereEmbeddingConfig
    from .llms.bedrock.image.amazon_nova_canvas_transformation import (
        AmazonNovaCanvasConfig,
    )
    from .llms.bedrock.image.amazon_stability1_transformation import (
        AmazonStabilityConfig,
    )
    from .llms.bedrock.image.amazon_stability3_transformation import (
        AmazonStability3Config,
    )
    from .llms.cohere.chat.transformation import CohereChatConfig
    from .llms.deepgram.audio_transcription.transformation import (
        DeepgramAudioTranscriptionConfig,
    )
    from .llms.deepinfra.chat.transformation import DeepInfraConfig
    from .llms.groq.chat.transformation import GroqChatConfig
    from .llms.mistral.mistral_chat_transformation import MistralConfig
    from .llms.ollama.completion.transformation import OllamaConfig
    from .llms.ollama_chat import OllamaChatConfig
    from .llms.openai.chat.o_series_transformation import OpenAIOSeriesConfig
    from .llms.openai.chat.o_series_transformation import (
        OpenAIOSeriesConfig as OpenAIO1Config,  # maintain backwards compatibility
    )
    from .llms.openai.image_variations.transformation import OpenAIImageVariationConfig
    from .llms.openai.openai import MistralEmbeddingConfig, OpenAIConfig
    from .llms.openai.responses.transformation import OpenAIResponsesAPIConfig
    from .llms.sagemaker.chat.transformation import SagemakerChatConfig
    from .llms.sagemaker.completion.transformation import SagemakerConfig
    from .llms.snowflake.chat.transformation import SnowflakeConfig
    from .llms.topaz.common_utils import TopazModelInfo
    from .llms.topaz.image_variations.transformation import TopazImageVariationConfig
    from .llms.vertex_ai.vertex_ai_partner_models.ai21.transformation import (
        VertexAIAi21Config,
    )
    from .llms.vertex_ai.vertex_ai_partner_models.anthropic.transformation import (
        VertexAIAnthropicConfig,
    )
    from .llms.vertex_ai.vertex_ai_partner_models.llama3.transformation import (
        VertexAILlama3Config,
    )
    from .llms.voyage.embedding.transformation import VoyageEmbeddingConfig
else:
    _import_structures.setdefault(
        "litellm.llms.openai.completion.transformation", []
    ).extend(
        [
            "OpenAITextCompletionConfig",
        ]
    )
    _import_structures.setdefault(
        "litellm.llms.azure_ai.chat.transformation", []
    ).extend(
        [
            "AzureAIStudioConfig",
        ]
    )
    _import_structures.setdefault(
        "litellm.llms.bedrock.chat.invoke_handler", []
    ).extend(
        [
            "BedrockLLM",
            "AWSEventStreamDecoder",
            "AmazonCohereChatConfig",
            "bedrock_tool_name_mappings",
        ]
    )
    _import_structures.setdefault(
        "litellm.llms.bedrock.chat.invoke_transformations.amazon_ai21_transformation",
        [],
    ).extend(
        [
            "AmazonAI21Config",
        ]
    )
    _import_structures.setdefault(
        "litellm.llms.bedrock.chat.invoke_transformations.amazon_cohere_transformation",
        [],
    ).extend(
        [
            "AmazonCohereConfig",
        ]
    )
    _import_structures.setdefault(
        "litellm.llms.bedrock.chat.invoke_transformations.amazon_deepseek_transformation",
        [],
    ).extend(
        [
            "AmazonDeepSeekR1Config",
        ]
    )
    _import_structures.setdefault(
        "litellm.llms.bedrock.chat.invoke_transformations.amazon_llama_transformation",
        [],
    ).extend(
        [
            "AmazonLlamaConfig",
        ]
    )
    _import_structures.setdefault(
        "litellm.llms.bedrock.chat.invoke_transformations.amazon_mistral_transformation",
        [],
    ).extend(
        [
            "AmazonMistralConfig",
        ]
    )
    _import_structures.setdefault(
        "litellm.llms.bedrock.chat.invoke_transformations.amazon_nova_transformation",
        [],
    ).extend(
        [
            "AmazonInvokeNovaConfig",
        ]
    )
    _import_structures.setdefault(
        "litellm.llms.bedrock.chat.invoke_transformations.amazon_titan_transformation",
        [],
    ).extend(
        [
            "AmazonTitanConfig",
        ]
    )
    _import_structures.setdefault(
        "litellm.llms.bedrock.chat.invoke_transformations.anthropic_claude2_transformation",
        [],
    ).extend(
        [
            "AmazonAnthropicConfig",
        ]
    )
    _import_structures.setdefault(
        "litellm.llms.bedrock.chat.invoke_transformations.anthropic_claude3_transformation",
        [],
    ).extend(
        [
            "AmazonAnthropicClaude3Config",
        ]
    )
    _import_structures.setdefault(
        "litellm.llms.bedrock.chat.invoke_transformations.base_invoke_transformation",
        [],
    ).extend(
        [
            "AmazonInvokeConfig",
        ]
    )
    _import_structures.setdefault("litellm.llms.bedrock.common_utils", []).extend(
        [
            "AmazonBedrockGlobalConfig",
        ]
    )
    _import_structures.setdefault(
        "litellm.llms.bedrock.embed.amazon_titan_g1_transformation", []
    ).extend(
        [
            "AmazonTitanG1Config",
        ]
    )
    _import_structures.setdefault(
        "litellm.llms.bedrock.embed.amazon_titan_multimodal_transformation", []
    ).extend(
        [
            "AmazonTitanMultimodalEmbeddingG1Config",
        ]
    )
    _import_structures.setdefault(
        "litellm.llms.bedrock.embed.amazon_titan_v2_transformation", []
    ).extend(
        [
            "AmazonTitanV2Config",
        ]
    )
    _import_structures.setdefault(
        "litellm.llms.bedrock.embed.cohere_transformation", []
    ).extend(
        [
            "BedrockCohereEmbeddingConfig",
        ]
    )
    _import_structures.setdefault(
        "litellm.llms.bedrock.image.amazon_nova_canvas_transformation", []
    ).extend(
        [
            "AmazonNovaCanvasConfig",
        ]
    )
    _import_structures.setdefault(
        "litellm.llms.bedrock.image.amazon_stability1_transformation", []
    ).extend(
        [
            "AmazonStabilityConfig",
        ]
    )
    _import_structures.setdefault(
        "litellm.llms.bedrock.image.amazon_stability3_transformation", []
    ).extend(
        [
            "AmazonStability3Config",
        ]
    )
    _import_structures.setdefault("litellm.llms.cohere.chat.transformation", []).extend(
        [
            "CohereChatConfig",
        ]
    )
    _import_structures.setdefault(
        "litellm.llms.deepgram.audio_transcription.transformation", []
    ).extend(
        [
            "DeepgramAudioTranscriptionConfig",
        ]
    )
    _import_structures.setdefault(
        "litellm.llms.deepinfra.chat.transformation", []
    ).extend(
        [
            "DeepInfraConfig",
        ]
    )
    _import_structures.setdefault("litellm.llms.groq.chat.transformation", []).extend(
        [
            "GroqChatConfig",
        ]
    )
    _import_structures.setdefault(
        "litellm.llms.mistral.mistral_chat_transformation", []
    ).extend(
        [
            "MistralConfig",
        ]
    )
    _import_structures.setdefault(
        "litellm.llms.ollama.completion.transformation", []
    ).extend(
        [
            "OllamaConfig",
        ]
    )
    _import_structures.setdefault("litellm.llms.ollama_chat", []).extend(
        [
            "OllamaChatConfig",
        ]
    )
    _import_structures.setdefault(
        "litellm.llms.openai.chat.o_series_transformation", []
    ).extend(
        [
            "OpenAIOSeriesConfig",
            _Alias(
                "OpenAIOSeriesConfig", "OpenAIO1Config"
            ),  # maintain backwards compatibility
        ]
    )
    _import_structures.setdefault(
        "litellm.llms.openai.image_variations.transformation", []
    ).extend(
        [
            "OpenAIImageVariationConfig",
        ]
    )
    _import_structures.setdefault("litellm.llms.openai.openai", []).extend(
        [
            "MistralEmbeddingConfig",
            "OpenAIConfig",
        ]
    )
    _import_structures.setdefault(
        "litellm.llms.openai.responses.transformation", []
    ).extend(
        [
            "OpenAIResponsesAPIConfig",
        ]
    )
    _import_structures.setdefault(
        "litellm.llms.sagemaker.chat.transformation", []
    ).extend(
        [
            "SagemakerChatConfig",
        ]
    )
    _import_structures.setdefault(
        "litellm.llms.sagemaker.completion.transformation", []
    ).extend(
        [
            "SagemakerConfig",
        ]
    )
    _import_structures.setdefault(
        "litellm.llms.snowflake.chat.transformation", []
    ).extend(
        [
            "SnowflakeConfig",
        ]
    )
    _import_structures.setdefault("litellm.llms.topaz.common_utils", []).extend(
        [
            "TopazModelInfo",
        ]
    )
    _import_structures.setdefault(
        "litellm.llms.topaz.image_variations.transformation", []
    ).extend(
        [
            "TopazImageVariationConfig",
        ]
    )
    _import_structures.setdefault(
        "litellm.llms.vertex_ai.vertex_ai_partner_models.ai21.transformation", []
    ).extend(
        [
            "VertexAIAi21Config",
        ]
    )
    _import_structures.setdefault(
        "litellm.llms.vertex_ai.vertex_ai_partner_models.anthropic.transformation", []
    ).extend(
        [
            "VertexAIAnthropicConfig",
        ]
    )
    _import_structures.setdefault(
        "litellm.llms.vertex_ai.vertex_ai_partner_models.llama3.transformation", []
    ).extend(
        [
            "VertexAILlama3Config",
        ]
    )
    _import_structures.setdefault(
        "litellm.llms.voyage.embedding.transformation", []
    ).extend(
        [
            "VoyageEmbeddingConfig",
        ]
    )

    if TYPE_CHECKING and _LITELLM_LAZY_IMPORTS:
        from .llms.nvidia_nim.chat import NvidiaNimConfig
        from .llms.nvidia_nim.embed import NvidiaNimEmbeddingConfig
        from .llms.openai.chat.gpt_audio_transformation import OpenAIGPTAudioConfig
        from .llms.openai.chat.gpt_transformation import OpenAIGPTConfig
        from .llms.openai.transcriptions.gpt_transformation import (
            OpenAIGPTAudioTranscriptionConfig,
        )
        from .llms.openai.transcriptions.whisper_transformation import (
            OpenAIWhisperAudioTranscriptionConfig,
        )
    else:
        _import_structures.setdefault(
            "litellm.llms.openai.chat.gpt_audio_transformation", []
        ).extend(["OpenAIGPTAudioConfig"])
        _import_structures.setdefault(
            "litellm.llms.openai.chat.gpt_transformation", []
        ).extend(["OpenAIGPTConfig"])
        _import_structures.setdefault(
            "litellm.llms.openai.transcriptions.gpt_transformation", []
        ).extend(["OpenAIGPTAudioTranscriptionConfig"])
        _import_structures.setdefault(
            "litellm.llms.openai.transcriptions.whisper_transformation", []
        ).extend(["OpenAIWhisperAudioTranscriptionConfig"])
        _import_structures.setdefault("litellm.llms.nvidia_nim.chat", []).extend(
            ["NvidiaNimConfig"]
        )
        _import_structures.setdefault("litellm.llms.nvidia_nim.embed", []).extend(
            ["NvidiaNimEmbeddingConfig"]
        )


if TYPE_CHECKING and _LITELLM_LAZY_IMPORTS:
    import litellm.anthropic_interface as anthropic
else:
    _import_structures.setdefault("litellm.anthropic_interface", []).extend(
        ["anthropic"]
    )

if TYPE_CHECKING and _LITELLM_LAZY_IMPORTS:
    from litellm._variables.config import (
        nvidiaNimConfig,
        nvidiaNimEmbeddingConfig,
        openAIGPTAudioConfig,
        openAIGPTConfig,
        openaiOSeriesConfig,
        vertexAITextEmbeddingConfig,
    )
    from litellm.assistants.main import (
        a_add_message,
        acreate_assistants,
        acreate_thread,
        add_message,
        adelete_assistant,
        aget_assistants,
        aget_messages,
        aget_thread,
        arun_thread,
        arun_thread_stream,
        azure_assistants_api,
        create_assistants,
        create_thread,
        delete_assistant,
        get_assistants,
        get_messages,
        get_thread,
        openai_assistants_api,
        run_thread,
        run_thread_stream,
    )
    from litellm.main import (
        AsyncCompletions,
        Chat,
        Completions,
        LiteLLM,
        _async_streaming,
        _handle_mock_potential_exceptions,
        _handle_mock_timeout,
        _handle_mock_timeout_async,
        _sleep_for_timeout,
        _sleep_for_timeout_async,
        aadapter_completion,
        acompletion,
        acompletion_with_retries,
        adapter_completion,
        aembedding,
        ahealth_check,
        ahealth_check_wildcard_models,
        aimage_generation,
        aimage_variation,
        amoderation,
        anthropic_chat_completions,
        aspeech,
        atext_completion,
        atranscription,
        azure_ai_embedding,
        azure_audio_transcriptions,
        azure_chat_completions,
        azure_o1_chat_completions,
        azure_text_completions,
        base_llm_aiohttp_handler,
        base_llm_http_handler,
        bedrock_converse_chat_completion,
        bedrock_embedding,
        bedrock_image_generation,
        codestral_text_completions,
        completion,
        completion_with_retries,
        config_completion,
        databricks_embedding,
        embedding,
        google_batch_embeddings,
        groq_chat_completions,
        image_generation,
        image_variation,
        mock_completion,
        moderation,
        openai_audio_transcriptions,
        openai_chat_completions,
        openai_image_variations,
        openai_like_chat_completion,
        openai_like_embedding,
        openai_text_completions,
        predibase_chat_completions,
        print_verbose,
        sagemaker_chat_completion,
        sagemaker_llm,
        speech,
        stream_chunk_builder,
        stream_chunk_builder_text_completion,
        text_completion,
        transcription,
        vertex_chat_completion,
        vertex_embedding,
        vertex_image_generation,
        vertex_model_garden_chat_completion,
        vertex_multimodal_embedding,
        vertex_partner_models_chat_completion,
        vertex_text_to_speech,
        watsonx_chat_completion,
    )

    from ._variables.misc import (
        adapters,
        custom_provider_map,
        disable_hf_tokenizer_download,
        global_disable_no_log_param,
    )
    from .batch_completion.main import (
        batch_completion,
        batch_completion_models,
        batch_completion_models_all_responses,
    )
    from .batches.main import (
        acancel_batch,
        acreate_batch,
        alist_batches,
        aretrieve_batch,
        cancel_batch,
        create_batch,
        list_batches,
        retrieve_batch,
    )
    from .budget_manager import BudgetManager
    from .cost_calculator import cost_per_token, response_cost_calculator
    from .exceptions import (
        LITELLM_EXCEPTION_TYPES,
        APIConnectionError,
        APIError,
        APIResponseValidationError,
        AuthenticationError,
        BadRequestError,
        BudgetExceededError,
        ContentPolicyViolationError,
        ContextWindowExceededError,
        InternalServerError,
        InvalidRequestError,
        JSONSchemaValidationError,
        MockException,
        NotFoundError,
        OpenAIError,
        RateLimitError,
        ServiceUnavailableError,
        Timeout,
        UnprocessableEntityError,
        UnsupportedParamsError,
    )
    from .files.main import *
    from .fine_tuning.main import *
    from .integrations import *
    from .litellm_core_utils.get_model_cost_map import get_model_cost_map
    from .llms.ai21.chat.transformation import AI21ChatConfig
    from .llms.azure.azure import AzureOpenAIAssistantsAPIConfig, AzureOpenAIError
    from .llms.azure.chat.gpt_transformation import AzureOpenAIConfig
    from .llms.azure.chat.o_series_transformation import AzureOpenAIO1Config
    from .llms.azure.completion.transformation import AzureOpenAITextConfig
    from .llms.base_llm.chat.transformation import BaseConfig, BaseLLMException
    from .llms.cerebras.chat import CerebrasConfig
    from .llms.codestral.completion.transformation import CodestralTextCompletionConfig
    from .llms.deepseek.chat.transformation import DeepSeekChatConfig
    from .llms.fireworks_ai.audio_transcription.transformation import (
        FireworksAIAudioTranscriptionConfig,
    )
    from .llms.fireworks_ai.chat.transformation import FireworksAIConfig
    from .llms.fireworks_ai.completion.transformation import (
        FireworksAITextCompletionConfig,
    )
    from .llms.fireworks_ai.embed.fireworks_ai_transformation import (
        FireworksAIEmbeddingConfig,
    )
    from .llms.friendliai.chat.transformation import FriendliaiChatConfig
    from .llms.hosted_vllm.chat.transformation import HostedVLLMChatConfig
    from .llms.jina_ai.embedding.transformation import JinaAIEmbeddingConfig
    from .llms.litellm_proxy.chat.transformation import LiteLLMProxyChatConfig
    from .llms.lm_studio.chat.transformation import LMStudioChatConfig
    from .llms.lm_studio.embed.transformation import LmStudioEmbeddingConfig
    from .llms.perplexity.chat.transformation import PerplexityChatConfig
    from .llms.sambanova.chat import SambanovaConfig
    from .llms.vllm.completion.transformation import VLLMConfig
    from .llms.volcengine import VolcEngineConfig
    from .llms.watsonx.chat.transformation import IBMWatsonXChatConfig
    from .llms.watsonx.completion.transformation import IBMWatsonXAIConfig
    from .llms.watsonx.embed.transformation import IBMWatsonXEmbeddingConfig
    from .llms.xai.chat.transformation import XAIChatConfig
    from .llms.xai.common_utils import XAIModelInfo
    from .proxy.proxy_cli import run_server
    from .realtime_api.main import _arealtime
    from .rerank_api.main import *
    from .responses.main import *
    from .router import Router
    from .scheduler import *
    from .secret_managers.main import get_secret, get_secret_str

    ### ADAPTERS ###
    from .types.adapter import AdapterItem

    ### CUSTOM LLMs ###
    from .types.llms.custom_llm import CustomLLMItem
    from .types.utils import GenericStreamingChunk

else:
    _import_structures.setdefault("litellm._variables.config", []).extend(
        [
            "vertexAITextEmbeddingConfig",
            "openaiOSeriesConfig",
            "openAIGPTConfig",
            "openAIGPTAudioConfig",
            "nvidiaNimConfig",
            "nvidiaNimEmbeddingConfig",
        ]
    )
    _import_structures.setdefault("litellm.assistants.main", []).extend(
        [
            "openai_assistants_api",
            "azure_assistants_api",
            "aget_assistants",
            "get_assistants",
            "acreate_assistants",
            "create_assistants",
            "adelete_assistant",
            "delete_assistant",
            "acreate_thread",
            "create_thread",
            "aget_thread",
            "get_thread",
            "a_add_message",
            "add_message",
            "aget_messages",
            "get_messages",
            "arun_thread_stream",
            "arun_thread",
            "run_thread_stream",
            "run_thread",
        ]
    )
    _import_structures.setdefault("litellm.batch_completion.main", []).extend(
        [
            "batch_completion",
            "batch_completion_models",
            "batch_completion_models_all_responses",
        ]
    )
    _import_structures.setdefault("litellm.batches.main", []).extend(
        [
            "acancel_batch",
            "acreate_batch",
            "alist_batches",
            "aretrieve_batch",
            "cancel_batch",
            "create_batch",
            "list_batches",
            "retrieve_batch",
        ]
    )
    _import_structures.setdefault("litellm.budget_manager", []).extend(
        [
            "BudgetManager",
        ]
    )
    _import_structures.setdefault("litellm.cost_calculator", []).extend(
        [
            "cost_per_token",
            "response_cost_calculator",
        ]
    )
    _import_structures.setdefault("litellm.exceptions", []).extend(
        [
            "LITELLM_EXCEPTION_TYPES",
            "APIConnectionError",
            "APIError",
            "APIResponseValidationError",
            "AuthenticationError",
            "BadRequestError",
            "BudgetExceededError",
            "ContentPolicyViolationError",
            "ContextWindowExceededError",
            "InternalServerError",
            "InvalidRequestError",
            "JSONSchemaValidationError",
            "MockException",
            "NotFoundError",
            "OpenAIError",
            "RateLimitError",
            "ServiceUnavailableError",
            "Timeout",
            "UnprocessableEntityError",
            "UnsupportedParamsError",
        ]
    )
    _import_structures.setdefault("litellm.files.main", []).extend(
        [
            "openai_files_instance",
            "azure_files_instance",
            "vertex_ai_files_instance",
            "afile_retrieve",
            "file_retrieve",
            "afile_delete",
            "file_delete",
            "afile_list",
            "file_list",
            "acreate_file",
            "create_file",
            "afile_content",
            "file_content",
        ]
    )
    _import_structures.setdefault("litellm.fine_tuning.main", []).extend(
        [
            "openai_fine_tuning_instance",
            "azure_fine_tuning_instance",
            "vertex_ai_fine_tuning_instance",
            "acreate_fine_tuning_job",
            "create_fine_tuning_job",
            "acancel_fine_tuning_job",
            "cancel_fine_tuning_job",
            "alist_fine_tuning_jobs",
            "list_fine_tuning_jobs",
            "aretrieve_fine_tuning_job",
            "retrieve_fine_tuning_job",
        ]
    )
    _import_structures.setdefault("litellm.integrations", []).extend(
        [
            "custom_logger",
            "additional_logging_utils",
            "datadog",
            "custom_batch_logger",
            "opentelemetry",
            "prometheus_services",
            "custom_guardrail",
            "arize",
            "mlflow",
            "pagerduty",
            "SlackAlerting",
            "email_templates",
            "argilla",
            "athina",
            "azure_storage",
            "braintrust_logging",
            "prompt_management_base",
            "custom_prompt_management",
            "dynamodb",
            "galileo",
            "gcs_bucket",
            "gcs_pubsub",
            "greenscale",
            "helicone",
            "humanloop",
            "lago",
            "langfuse",
            "langsmith",
            "literal_ai",
            "logfire_logger",
            "lunary",
            "openmeter",
            "opik",
            "prometheus",
            "prompt_layer",
            "s3",
            "supabase",
            "traceloop",
            "weights_biases",
        ]
    )
    _import_structures.setdefault(
        "litellm.llms.anthropic.experimental_pass_through.messages.handler", []
    ).extend(
        [
            "AnthropicMessagesHandler",
            "AnthropicMessagesResponse",
            "AsyncHTTPHandler",
            "AsyncIterator",
            "BaseAnthropicMessagesConfig",
            "GenericLiteLLMParams",
            "LiteLLMLoggingObj",
            "ProviderConfigManager",
            "ProviderSpecificHeader",
            "anthropic_messages",
        ]
    )
    _import_structures.setdefault("litellm.llms.ai21.chat.transformation", []).extend(
        ["AI21ChatConfig"]
    )
    _import_structures.setdefault("litellm.llms.azure.azure", []).extend(
        [
            "AzureOpenAIAssistantsAPIConfig",
            "AzureOpenAIError",
        ]
    )
    _import_structures.setdefault(
        "litellm.llms.azure.chat.gpt_transformation", []
    ).extend(
        [
            "AzureOpenAIConfig",
        ]
    )
    _import_structures.setdefault(
        "litellm.llms.azure.chat.o_series_transformation", []
    ).extend(
        [
            "AzureOpenAIO1Config",
        ]
    )
    _import_structures.setdefault(
        "litellm.llms.azure.completion.transformation", []
    ).extend(
        [
            "AzureOpenAITextConfig",
        ]
    )
    _import_structures.setdefault("litellm.llms.cerebras.chat", []).extend(
        [
            "CerebrasConfig",
        ]
    )
    _import_structures.setdefault(
        "litellm.llms.codestral.completion.transformation", []
    ).extend(
        [
            "CodestralTextCompletionConfig",
        ]
    )
    _import_structures.setdefault(
        "litellm.llms.deepseek.chat.transformation", []
    ).extend(
        [
            "DeepSeekChatConfig",
        ]
    )
    _import_structures.setdefault(
        "litellm.llms.fireworks_ai.audio_transcription.transformation", []
    ).extend(
        [
            "FireworksAIAudioTranscriptionConfig",
        ]
    )
    _import_structures.setdefault(
        "litellm.llms.fireworks_ai.chat.transformation", []
    ).extend(
        [
            "FireworksAIConfig",
        ]
    )
    _import_structures.setdefault(
        "litellm.llms.fireworks_ai.completion.transformation", []
    ).extend(
        [
            "FireworksAITextCompletionConfig",
        ]
    )
    _import_structures.setdefault(
        "litellm.llms.fireworks_ai.embed.fireworks_ai_transformation", []
    ).extend(
        [
            "FireworksAIEmbeddingConfig",
        ]
    )
    _import_structures.setdefault(
        "litellm.llms.friendliai.chat.transformation", []
    ).extend(
        [
            "FriendliaiChatConfig",
        ]
    )
    _import_structures.setdefault(
        "litellm.llms.hosted_vllm.chat.transformation", []
    ).extend(
        [
            "HostedVLLMChatConfig",
        ]
    )
    _import_structures.setdefault(
        "litellm.llms.jina_ai.embedding.transformation", []
    ).extend(
        [
            "JinaAIEmbeddingConfig",
        ]
    )
    _import_structures.setdefault(
        "litellm.llms.litellm_proxy.chat.transformation", []
    ).extend(
        [
            "LiteLLMProxyChatConfig",
        ]
    )
    _import_structures.setdefault(
        "litellm.llms.lm_studio.chat.transformation", []
    ).extend(
        [
            "LMStudioChatConfig",
        ]
    )
    _import_structures.setdefault(
        "litellm.llms.lm_studio.embed.transformation", []
    ).extend(
        [
            "LmStudioEmbeddingConfig",
        ]
    )
    _import_structures.setdefault(
        "litellm.llms.perplexity.chat.transformation", []
    ).extend(
        [
            "PerplexityChatConfig",
        ]
    )
    _import_structures.setdefault("litellm.llms.sambanova.chat", []).extend(
        [
            "SambanovaConfig",
        ]
    )
    _import_structures.setdefault(
        "litellm.llms.vllm.completion.transformation", []
    ).extend(
        [
            "VLLMConfig",
        ]
    )
    _import_structures.setdefault("litellm.llms.volcengine", []).extend(
        [
            "VolcEngineConfig",
        ]
    )
    _import_structures.setdefault(
        "litellm.llms.watsonx.chat.transformation", []
    ).extend(
        [
            "IBMWatsonXChatConfig",
        ]
    )
    _import_structures.setdefault(
        "litellm.llms.watsonx.completion.transformation", []
    ).extend(
        [
            "IBMWatsonXAIConfig",
        ]
    )
    _import_structures.setdefault(
        "litellm.llms.watsonx.embed.transformation", []
    ).extend(
        [
            "IBMWatsonXEmbeddingConfig",
        ]
    )
    _import_structures.setdefault("litellm.llms.xai.chat.transformation", []).extend(
        [
            "XAIChatConfig",
        ]
    )
    _import_structures.setdefault("litellm.llms.xai.common_utils", []).extend(
        [
            "XAIModelInfo",
        ]
    )
    _import_structures.setdefault(
        "litellm.litellm_core_utils.get_model_cost_map", []
    ).extend(
        [
            "get_model_cost_map",
        ]
    )
    _import_structures.setdefault("litellm.secret_managers.main", []).extend(
        ["get_secret_str", "get_secret"]
    )
    _import_structures.setdefault("litellm.main", []).extend(
        [
            "anthropic_chat_completions",
            "azure_ai_embedding",
            "azure_audio_transcriptions",
            "azure_chat_completions",
            "azure_o1_chat_completions",
            "azure_text_completions",
            "base_llm_aiohttp_handler",
            "base_llm_http_handler",
            "bedrock_converse_chat_completion",
            "bedrock_embedding",
            "bedrock_image_generation",
            "codestral_text_completions",
            "databricks_chat_completions",
            "databricks_embedding",
            "google_batch_embeddings",
            "groq_chat_completions",
            "huggingface",
            "openai_audio_transcriptions",
            "openai_chat_completions",
            "openai_image_variations",
            "openai_like_chat_completion",
            "openai_like_embedding",
            "openai_text_completions",
            "predibase_chat_completions",
            "sagemaker_chat_completion",
            "sagemaker_llm",
            "vertex_chat_completion",
            "vertex_embedding",
            "vertex_image_generation",
            "vertex_model_garden_chat_completion",
            "vertex_multimodal_embedding",
            "vertex_partner_models_chat_completion",
            "vertex_text_to_speech",
            "watsonx_chat_completion",
            "LiteLLM",
            "Chat",
            "Completions",
            "AsyncCompletions",
            "acompletion",
            "_async_streaming",
            "_handle_mock_potential_exceptions",
            "_handle_mock_timeout",
            "_handle_mock_timeout_async",
            "_sleep_for_timeout",
            "_sleep_for_timeout_async",
            "mock_completion",
            "completion",
            "completion_with_retries",
            "acompletion_with_retries",
            "aembedding",
            "embedding",
            "atext_completion",
            "text_completion",
            "aadapter_completion",
            "adapter_completion",
            "moderation",
            "amoderation",
            "aimage_generation",
            "image_generation",
            "aimage_variation",
            "image_variation",
            "atranscription",
            "transcription",
            "aspeech",
            "speech",
            "ahealth_check_wildcard_models",
            "ahealth_check",
            "print_verbose",
            "config_completion",
            "stream_chunk_builder_text_completion",
            "stream_chunk_builder",
        ]
    )

    _import_structures.setdefault("litellm.proxy.proxy_cli", []).extend(["run_server"])
    _import_structures.setdefault("litellm.realtime_api.main", []).extend(
        ["_arealtime"]
    )

    _import_structures.setdefault("litellm.rerank_api.main", []).extend(
        [
            "together_rerank",
            "bedrock_rerank",
            "base_llm_http_handler",
            "arerank",
            "rerank",
        ]
    )
    _import_structures.setdefault("litellm.responses.main", []).extend(
        [
            "base_llm_http_handler",
            "aresponses",
            "responses",
        ]
    )
    _import_structures.setdefault("litellm.router", []).extend(["Router"])
    _import_structures.setdefault("litellm.scheduler", []).extend(
        [
            "SchedulerCacheKeys",
            "DefaultPriorities",
            "FlowItem",
            "Scheduler",
        ]
    )
    _import_structures.setdefault("litellm.types.adapter", []).extend(["AdapterItem"])
    _import_structures.setdefault("litellm.types.llms.custom_llm", []).extend(
        ["CustomLLMItem"]
    )
    _import_structures.setdefault(
        "litellm.llms.base_llm.chat.transformation", []
    ).extend(["BaseLLMException", "BaseConfig"])
    _import_structures.setdefault("litellm.types.utils", []).extend(
        ["GenericStreamingChunk"]
    )
    _import_structures.setdefault("litellm._variables.misc", []).extend(
        [
            "adapters",
            "custom_provider_map",
            "disable_hf_tokenizer_download",
            "global_disable_no_log_param",
            "_custom_providers",
        ]
    )
    _import_structures.setdefault("litellm._lazy_module", []).extend(["_LazyModule"])

    # Lazy import for litellm module
    import sys

    _module = _LazyModule.create(
        __name__,
        globals()["__file__"],
        _import_structures,
        module_spec=__spec__,
    )
    sys.modules[__name__] = _module

import sys

litellm_mode = os.getenv("LITELLM_MODE", "DEV")  # "PRODUCTION", "DEV"
if litellm_mode == "DEV":
    dotenv.load_dotenv()
################################################
_litellm = sys.modules["litellm"]
if _litellm.set_verbose == True:
    _litellm._turn_on_debug()
