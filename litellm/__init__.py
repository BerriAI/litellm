### Hide pydantic namespace conflict warnings globally ###
import warnings

warnings.filterwarnings("ignore", message=".*conflict with protected namespace.*")
# Suppress Pydantic 2.11+ deprecation warning about accessing model_fields on instances
# This warning can accumulate during streaming and cause memory leaks
warnings.filterwarnings(
    "ignore", message=".*Accessing the.*attribute on the instance is deprecated.*"
)
### INIT VARIABLES #######################
import threading
import os
from typing import (
    Callable,
    List,
    Optional,
    Dict,
    Union,
    Any,
    Literal,
    get_args,
    TYPE_CHECKING,
)
from litellm.types.integrations.datadog_llm_obs import DatadogLLMObsInitParams
from litellm.types.integrations.datadog import DatadogInitParams
# HTTP handlers are lazy-loaded to reduce import-time memory cost
# from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
# Caching classes are lazy-loaded to reduce import-time memory cost
# from litellm.caching.caching import Cache, DualCache, RedisCache, InMemoryCache

from litellm._logging import (
    set_verbose,
    _turn_on_debug,
    verbose_logger,
    json_logs,
    _turn_on_json,
    log_level,
)
import re
from litellm.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_FLUSH_INTERVAL_SECONDS,
    ROUTER_MAX_FALLBACKS,
    DEFAULT_MAX_RETRIES,
    DEFAULT_REPLICATE_POLLING_RETRIES,
    DEFAULT_REPLICATE_POLLING_DELAY_SECONDS,
    LITELLM_CHAT_PROVIDERS,
    HUMANLOOP_PROMPT_CACHE_TTL_SECONDS,
    OPENAI_CHAT_COMPLETION_PARAMS,
    OPENAI_CHAT_COMPLETION_PARAMS as _openai_completion_params,  # backwards compatibility
    OPENAI_FINISH_REASONS,
    OPENAI_FINISH_REASONS as _openai_finish_reasons,  # backwards compatibility
    openai_compatible_endpoints,
    openai_compatible_providers,
    openai_text_completion_compatible_providers,
    _openai_like_providers,
    replicate_models,
    clarifai_models,
    huggingface_models,
    empower_models,
    together_ai_models,
    baseten_models,
    WANDB_MODELS,
    REPEATED_STREAMING_CHUNK_LIMIT,
    request_timeout,
    open_ai_embedding_models,
    cohere_embedding_models,
    bedrock_embedding_models,
    known_tokenizer_config,
    BEDROCK_INVOKE_PROVIDERS_LITERAL,
    BEDROCK_EMBEDDING_PROVIDERS_LITERAL,
    BEDROCK_CONVERSE_MODELS,
    DEFAULT_MAX_TOKENS,
    DEFAULT_SOFT_BUDGET,
    DEFAULT_ALLOWED_FAILS,
)
from litellm.types.utils import LlmProviders, PriorityReservationSettings
if TYPE_CHECKING:
    from litellm.integrations.custom_logger import CustomLogger
    from litellm.types.llms.bedrock import COHERE_EMBEDDING_INPUT_TYPES
    from litellm.types.guardrails import GuardrailItem
    from litellm.types.utils import CredentialItem, BudgetConfig, PriorityReservationDict, StandardKeyGenerationConfig, LlmProviders, PriorityReservationSettings
    from litellm.types.proxy.management_endpoints.ui_sso import DefaultTeamSSOParams, LiteLLM_UpperboundKeyGenerateParams
    from litellm.types.secret_managers.main import KeyManagementSystem, KeyManagementSettings
    from litellm.llms.openai_like.chat.handler import OpenAILikeChatConfig
    from litellm.llms.aiohttp_openai.chat.transformation import AiohttpOpenAIChatConfig
    from litellm.llms.galadriel.chat.transformation import GaladrielChatConfig
    from litellm.llms.github.chat.transformation import GithubChatConfig
    from litellm.llms.compactifai.chat.transformation import CompactifAIChatConfig
    from litellm.llms.empower.chat.transformation import EmpowerChatConfig
    from litellm.llms.huggingface.chat.transformation import HuggingFaceChatConfig
    from litellm.llms.openrouter.chat.transformation import OpenrouterConfig
    from litellm.llms.anthropic.chat.transformation import AnthropicConfig
    from litellm.llms.databricks.chat.transformation import DatabricksConfig
    from litellm.llms.predibase.chat.transformation import PredibaseConfig
    from litellm.llms.replicate.chat.transformation import ReplicateConfig
    from litellm.llms.snowflake.chat.transformation import SnowflakeConfig
    from litellm.llms.huggingface.embedding.transformation import HuggingFaceEmbeddingConfig
    from litellm.llms.oobabooga.chat.transformation import OobaboogaConfig
    from litellm.llms.maritalk import MaritalkConfig
    from litellm.llms.datarobot.chat.transformation import DataRobotConfig
    from litellm.llms.groq.stt.transformation import GroqSTTConfig
    from litellm.llms.anthropic.completion.transformation import AnthropicTextConfig
    from litellm.llms.triton.completion.transformation import TritonConfig
    from litellm.llms.triton.embedding.transformation import TritonEmbeddingConfig
    from litellm.llms.clarifai.chat.transformation import ClarifaiConfig
    from litellm.llms.ai21.chat.transformation import AI21ChatConfig
    from litellm.llms.meta_llama.chat.transformation import LlamaAPIConfig
    from litellm.llms.together_ai.chat import TogetherAIConfig
    from litellm.llms.cloudflare.chat.transformation import CloudflareChatConfig
    from litellm.llms.novita.chat.transformation import NovitaConfig
    from litellm.llms.nlp_cloud.chat.handler import NLPCloudConfig
    from litellm.llms.petals.completion.transformation import PetalsConfig
    from litellm.llms.ollama.chat.transformation import OllamaChatConfig
    from litellm.llms.ollama.completion.transformation import OllamaConfig
    from litellm.llms.sagemaker.completion.transformation import SagemakerConfig
    from litellm.llms.sagemaker.chat.transformation import SagemakerChatConfig
    from litellm.llms.cohere.chat.transformation import CohereChatConfig
    from litellm.llms.cohere.chat.v2_transformation import CohereV2ChatConfig
    from litellm.llms.openai.openai import OpenAIConfig, MistralEmbeddingConfig
    from litellm.llms.openai.completion.transformation import OpenAITextCompletionConfig
    from litellm.llms.deepinfra.chat.transformation import DeepInfraConfig
    from litellm.llms.groq.chat.transformation import GroqChatConfig
    from litellm.llms.voyage.embedding.transformation import VoyageEmbeddingConfig
    from litellm.llms.voyage.embedding.transformation_contextual import VoyageContextualEmbeddingConfig
    from litellm.llms.infinity.embedding.transformation import InfinityEmbeddingConfig
    from litellm.llms.azure_ai.chat.transformation import AzureAIStudioConfig
    from litellm.llms.mistral.chat.transformation import MistralConfig
    from litellm.llms.huggingface.rerank.transformation import HuggingFaceRerankConfig
    from litellm.llms.cohere.rerank.transformation import CohereRerankConfig
    from litellm.llms.cohere.rerank_v2.transformation import CohereRerankV2Config
    from litellm.llms.azure_ai.rerank.transformation import AzureAIRerankConfig
    from litellm.llms.infinity.rerank.transformation import InfinityRerankConfig
    from litellm.llms.jina_ai.rerank.transformation import JinaAIRerankConfig
    from litellm.llms.deepinfra.rerank.transformation import DeepinfraRerankConfig
    from litellm.llms.hosted_vllm.rerank.transformation import HostedVLLMRerankConfig
    from litellm.llms.nvidia_nim.rerank.transformation import NvidiaNimRerankConfig
    from litellm.llms.vertex_ai.rerank.transformation import VertexAIRerankConfig
    from litellm.llms.anthropic.experimental_pass_through.messages.transformation import AnthropicMessagesConfig
    from litellm.llms.together_ai.completion.transformation import TogetherAITextCompletionConfig
    from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import VertexGeminiConfig
    from litellm.llms.gemini.chat.transformation import GoogleAIStudioGeminiConfig
    from litellm.llms.vertex_ai.vertex_ai_partner_models.anthropic.transformation import VertexAIAnthropicConfig
    from litellm.llms.vertex_ai.vertex_ai_partner_models.llama3.transformation import VertexAILlama3Config
    from litellm.llms.vertex_ai.vertex_ai_partner_models.ai21.transformation import VertexAIAi21Config
    from litellm.llms.bedrock.chat.invoke_handler import AmazonCohereChatConfig
    from litellm.llms.bedrock.common_utils import AmazonBedrockGlobalConfig
    from litellm.llms.bedrock.chat.invoke_transformations.amazon_ai21_transformation import AmazonAI21Config
    from litellm.llms.bedrock.chat.invoke_transformations.base_invoke_transformation import AmazonInvokeConfig
    from litellm.llms.anthropic.common_utils import AnthropicModelInfo
    from litellm.llms.deprecated_providers.palm import PalmConfig
    from litellm.llms.deprecated_providers.aleph_alpha import AlephAlphaConfig
    from litellm.llms.azure.responses.transformation import AzureOpenAIResponsesAPIConfig
    from litellm.llms.azure.responses.o_series_transformation import AzureOpenAIOSeriesResponsesAPIConfig
    from litellm.llms.openai.chat.o_series_transformation import OpenAIOSeriesConfig
    from litellm.llms.azure.chat.o_series_transformation import AzureOpenAIO1Config
    from litellm.llms.gradient_ai.chat.transformation import GradientAIConfig
    from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig
    from litellm.llms.openai.chat.gpt_5_transformation import OpenAIGPT5Config
    from litellm.llms.openai.chat.gpt_audio_transformation import OpenAIGPTAudioConfig
    from litellm.llms.nvidia_nim.chat.transformation import NvidiaNimConfig
import httpx
import dotenv
from litellm.llms.custom_httpx.async_client_cleanup import register_async_client_cleanup

litellm_mode = os.getenv("LITELLM_MODE", "DEV")  # "PRODUCTION", "DEV"
if litellm_mode == "DEV":
    dotenv.load_dotenv()

# Register async client cleanup to prevent resource leaks
register_async_client_cleanup()
####################################################
if set_verbose:
    _turn_on_debug()
####################################################
### Callbacks /Logging / Success / Failure Handlers #####
CALLBACK_TYPES = Union[str, Callable, "CustomLogger"]
input_callback: List[CALLBACK_TYPES] = []
success_callback: List[CALLBACK_TYPES] = []
failure_callback: List[CALLBACK_TYPES] = []
service_callback: List[CALLBACK_TYPES] = []
_logging_callback_manager_instance: Optional[Any] = None

class _LazyLoggingCallbackManagerWrapper:
    """Wrapper to lazy-load LoggingCallbackManager instance."""
    def _get_instance(self) -> Any:
        """Lazy initialization of logging_callback_manager."""
        global _logging_callback_manager_instance
        if _logging_callback_manager_instance is None:
            from litellm.litellm_core_utils.logging_callback_manager import LoggingCallbackManager
            _logging_callback_manager_instance = LoggingCallbackManager()
        return _logging_callback_manager_instance
    
    def __getattr__(self, name: str) -> Any:
        return getattr(self._get_instance(), name)

logging_callback_manager: Any = _LazyLoggingCallbackManagerWrapper()
_custom_logger_compatible_callbacks_literal = Literal[
    "lago",
    "openmeter",
    "logfire",
    "literalai",
    "dynamic_rate_limiter",
    "dynamic_rate_limiter_v3",
    "langsmith",
    "prometheus",
    "otel",
    "datadog",
    "datadog_llm_observability",
    "galileo",
    "braintrust",
    "arize",
    "arize_phoenix",
    "langtrace",
    "gcs_bucket",
    "azure_storage",
    "opik",
    "argilla",
    "mlflow",
    "langfuse",
    "langfuse_otel",
    "pagerduty",
    "humanloop",
    "gcs_pubsub",
    "agentops",
    "anthropic_cache_control_hook",
    "generic_api",
    "resend_email",
    "smtp_email",
    "deepeval",
    "s3_v2",
    "aws_sqs",
    "vector_store_pre_call_hook",
    "dotprompt",
    "bitbucket",
    "gitlab",
    "cloudzero",
    "posthog",
]
cold_storage_custom_logger: Optional[_custom_logger_compatible_callbacks_literal] = None
logged_real_time_event_types: Optional[Union[List[str], Literal["*"]]] = None
_known_custom_logger_compatible_callbacks: List = list(
    get_args(_custom_logger_compatible_callbacks_literal)
)
callbacks: List[
    Union[Callable, _custom_logger_compatible_callbacks_literal, "CustomLogger"]
] = []
initialized_langfuse_clients: int = 0
langfuse_default_tags: Optional[List[str]] = None
langsmith_batch_size: Optional[int] = None
prometheus_initialize_budget_metrics: Optional[bool] = False
require_auth_for_metrics_endpoint: Optional[bool] = False
argilla_batch_size: Optional[int] = None
datadog_use_v1: Optional[bool] = False  # if you want to use v1 datadog logged payload.
gcs_pub_sub_use_v1: Optional[bool] = (
    False  # if you want to use v1 gcs pubsub logged payload
)
generic_api_use_v1: Optional[bool] = (
    False  # if you want to use v1 generic api logged payload
)
argilla_transformation_object: Optional[Dict[str, Any]] = None
_async_input_callback: List[Union[str, Callable, "CustomLogger"]] = (
    []
)  # internal variable - async custom callbacks are routed here.
_async_success_callback: List[Union[str, Callable, "CustomLogger"]] = (
    []
)  # internal variable - async custom callbacks are routed here.
_async_failure_callback: List[Union[str, Callable, "CustomLogger"]] = (
    []
)  # internal variable - async custom callbacks are routed here.
pre_call_rules: List[Callable] = []
post_call_rules: List[Callable] = []
turn_off_message_logging: Optional[bool] = False
log_raw_request_response: bool = False
redact_messages_in_exceptions: Optional[bool] = False
redact_user_api_key_info: Optional[bool] = False
filter_invalid_headers: Optional[bool] = False
add_user_information_to_llm_headers: Optional[bool] = (
    None  # adds user_id, team_id, token hash (params from StandardLoggingMetadata) to request headers
)
store_audit_logs = False  # Enterprise feature, allow users to see audit logs
### end of callbacks #############

email: Optional[str] = (
    None  # Not used anymore, will be removed in next MAJOR release - https://github.com/BerriAI/litellm/discussions/648
)
token: Optional[str] = (
    None  # Not used anymore, will be removed in next MAJOR release - https://github.com/BerriAI/litellm/discussions/648
)
telemetry = True
max_tokens: int = DEFAULT_MAX_TOKENS  # OpenAI Defaults
drop_params = bool(os.getenv("LITELLM_DROP_PARAMS", False))
modify_params = bool(os.getenv("LITELLM_MODIFY_PARAMS", False))
retry = True
### AUTH ###
api_key: Optional[str] = None
openai_key: Optional[str] = None
groq_key: Optional[str] = None
databricks_key: Optional[str] = None
openai_like_key: Optional[str] = None
azure_key: Optional[str] = None
anthropic_key: Optional[str] = None
replicate_key: Optional[str] = None
bytez_key: Optional[str] = None
cohere_key: Optional[str] = None
infinity_key: Optional[str] = None
clarifai_key: Optional[str] = None
maritalk_key: Optional[str] = None
ai21_key: Optional[str] = None
ollama_key: Optional[str] = None
openrouter_key: Optional[str] = None
datarobot_key: Optional[str] = None
predibase_key: Optional[str] = None
huggingface_key: Optional[str] = None
vertex_project: Optional[str] = None
vertex_location: Optional[str] = None
predibase_tenant_id: Optional[str] = None
togetherai_api_key: Optional[str] = None
cloudflare_api_key: Optional[str] = None
vercel_ai_gateway_key: Optional[str] = None
baseten_key: Optional[str] = None
llama_api_key: Optional[str] = None
aleph_alpha_key: Optional[str] = None
nlp_cloud_key: Optional[str] = None
novita_api_key: Optional[str] = None
snowflake_key: Optional[str] = None
gradient_ai_api_key: Optional[str] = None
nebius_key: Optional[str] = None
wandb_key: Optional[str] = None
heroku_key: Optional[str] = None
cometapi_key: Optional[str] = None
ovhcloud_key: Optional[str] = None
lemonade_key: Optional[str] = None
common_cloud_provider_auth_params: dict = {
    "params": ["project", "region_name", "token"],
    "providers": ["vertex_ai", "bedrock", "watsonx", "azure", "vertex_ai_beta"],
}
use_litellm_proxy: bool = (
    False  # when True, requests will be sent to the specified litellm proxy endpoint
)
use_client: bool = False
ssl_verify: Union[str, bool] = True
ssl_security_level: Optional[str] = None
ssl_certificate: Optional[str] = None
ssl_ecdh_curve: Optional[str] = (
    None  # Set to 'X25519' to disable PQC and improve performance
)
disable_streaming_logging: bool = False
disable_token_counter: bool = False
disable_add_transform_inline_image_block: bool = False
disable_add_user_agent_to_request_tags: bool = False
extra_spend_tag_headers: Optional[List[str]] = None
_in_memory_llm_clients_cache_instance: Optional[Any] = None

class _LazyLLMClientCacheWrapper:
    """Wrapper to lazy-load LLMClientCache instance."""
    def _get_instance(self) -> Any:
        """Lazy initialization of in_memory_llm_clients_cache."""
        global _in_memory_llm_clients_cache_instance
        if _in_memory_llm_clients_cache_instance is None:
            from litellm.caching.llm_caching_handler import LLMClientCache
            _in_memory_llm_clients_cache_instance = LLMClientCache()
        return _in_memory_llm_clients_cache_instance
    
    def __getattr__(self, name: str) -> Any:
        return getattr(self._get_instance(), name)

in_memory_llm_clients_cache: Any = _LazyLLMClientCacheWrapper()
safe_memory_mode: bool = False
enable_azure_ad_token_refresh: Optional[bool] = False
### DEFAULT AZURE API VERSION ###
AZURE_DEFAULT_API_VERSION = "2025-02-01-preview"  # this is updated to the latest
### DEFAULT WATSONX API VERSION ###
WATSONX_DEFAULT_API_VERSION = "2024-03-13"
### COHERE EMBEDDINGS DEFAULT TYPE ###
COHERE_DEFAULT_EMBEDDING_INPUT_TYPE: "COHERE_EMBEDDING_INPUT_TYPES" = "search_document"
### CREDENTIALS ###
credential_list: List["CredentialItem"] = []
### GUARDRAILS ###
llamaguard_model_name: Optional[str] = None
openai_moderations_model_name: Optional[str] = None
presidio_ad_hoc_recognizers: Optional[str] = None
google_moderation_confidence_threshold: Optional[float] = None
llamaguard_unsafe_content_categories: Optional[str] = None
blocked_user_list: Optional[Union[str, List]] = None
banned_keywords_list: Optional[Union[str, List]] = None
llm_guard_mode: Literal["all", "key-specific", "request-specific"] = "all"
guardrail_name_config_map: Dict[str, "GuardrailItem"] = {}
include_cost_in_streaming_usage: bool = False
### PROMPTS ####
from litellm.types.prompts.init_prompts import PromptSpec

prompt_name_config_map: Dict[str, PromptSpec] = {}

##################
### PREVIEW FEATURES ###
enable_preview_features: bool = False
return_response_headers: bool = (
    False  # get response headers from LLM Api providers - example x-remaining-requests,
)
enable_json_schema_validation: bool = False
####################
logging: bool = True
enable_loadbalancing_on_batch_endpoints: Optional[bool] = None
enable_caching_on_provider_specific_optional_params: bool = (
    False  # feature-flag for caching on optional params - e.g. 'top_k'
)
caching: bool = (
    False  # Not used anymore, will be removed in next MAJOR release - https://github.com/BerriAI/litellm/discussions/648
)
caching_with_models: bool = (
    False  # # Not used anymore, will be removed in next MAJOR release - https://github.com/BerriAI/litellm/discussions/648
)
cache: Optional["Cache"] = (  # type: ignore[name-defined]
    None  # cache object <- use this - https://docs.litellm.ai/docs/caching
)
default_in_memory_ttl: Optional[float] = None
default_redis_ttl: Optional[float] = None
default_redis_batch_cache_expiry: Optional[float] = None
model_alias_map: Dict[str, str] = {}
model_group_settings: Optional["ModelGroupSettings"] = None
max_budget: float = 0.0  # set the max budget across all providers
budget_duration: Optional[str] = (
    None  # proxy only - resets budget after fixed duration. You can set duration as seconds ("30s"), minutes ("30m"), hours ("30h"), days ("30d").
)
default_soft_budget: float = (
    DEFAULT_SOFT_BUDGET  # by default all litellm proxy keys have a soft budget of 50.0
)
forward_traceparent_to_llm_provider: bool = False


_current_cost = 0.0  # private variable, used if max budget is set
error_logs: Dict = {}
add_function_to_prompt: bool = (
    False  # if function calling not supported by api, append function call details to system prompt
)
client_session: Optional[httpx.Client] = None
aclient_session: Optional[httpx.AsyncClient] = None
model_fallbacks: Optional[List] = None  # Deprecated for 'litellm.fallbacks'
model_cost_map_url: str = os.getenv(
    "LITELLM_MODEL_COST_MAP_URL",
    "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json",
)
suppress_debug_info = False
dynamodb_table_name: Optional[str] = None
s3_callback_params: Optional[Dict] = None
datadog_llm_observability_params: Optional[Union[DatadogLLMObsInitParams, Dict]] = None
datadog_params: Optional[Union[DatadogInitParams, Dict]] = None
aws_sqs_callback_params: Optional[Dict] = None
generic_logger_headers: Optional[Dict] = None
default_key_generate_params: Optional[Dict] = None
upperbound_key_generate_params: Optional["LiteLLM_UpperboundKeyGenerateParams"] = None
key_generation_settings: Optional["StandardKeyGenerationConfig"] = None
default_internal_user_params: Optional[Dict] = None
default_team_params: Optional[Union["DefaultTeamSSOParams", Dict]] = None
default_team_settings: Optional[List] = None
max_user_budget: Optional[float] = None
default_max_internal_user_budget: Optional[float] = None
max_internal_user_budget: Optional[float] = None
max_ui_session_budget: Optional[float] = 10  # $10 USD budgets for UI Chat sessions
internal_user_budget_duration: Optional[str] = None
tag_budget_config: Optional[Dict[str, "BudgetConfig"]] = None
max_end_user_budget: Optional[float] = None
max_end_user_budget_id: Optional[str] = None
disable_end_user_cost_tracking: Optional[bool] = None
disable_end_user_cost_tracking_prometheus_only: Optional[bool] = None
enable_end_user_cost_tracking_prometheus_only: Optional[bool] = None
custom_prometheus_metadata_labels: List[str] = []
custom_prometheus_tags: List[str] = []
prometheus_metrics_config: Optional[List] = None
disable_add_prefix_to_prompt: bool = (
    False  # used by anthropic, to disable adding prefix to prompt
)
disable_copilot_system_to_assistant: bool = (
    False  # If false (default), converts all 'system' role messages to 'assistant' for GitHub Copilot compatibility. Set to true to disable this behavior.
)
public_mcp_servers: Optional[List[str]] = None
public_model_groups: Optional[List[str]] = None
public_agent_groups: Optional[List[str]] = None
public_model_groups_links: Dict[str, str] = {}
#### REQUEST PRIORITIZATION #######
priority_reservation: Optional[Dict[str, Union[float, "PriorityReservationDict"]]] = None


######## Networking Settings ########
use_aiohttp_transport: bool = (
    True  # Older variable, aiohttp is now the default. use disable_aiohttp_transport instead.
)
aiohttp_trust_env: bool = False  # set to true to use HTTP_ Proxy settings
disable_aiohttp_transport: bool = False  # Set this to true to use httpx instead
disable_aiohttp_trust_env: bool = (
    False  # When False, aiohttp will respect HTTP(S)_PROXY env vars
)
force_ipv4: bool = (
    False  # when True, litellm will force ipv4 for all LLM requests. Some users have seen httpx ConnectionError when using ipv6.
)
# module_level_aclient and module_level_client are lazy-loaded to reduce import-time memory cost
# They are created on first access via __getattr__

#### RETRIES ####
num_retries: Optional[int] = None  # per model endpoint
max_fallbacks: Optional[int] = None
default_fallbacks: Optional[List] = None
fallbacks: Optional[List] = None
context_window_fallbacks: Optional[List] = None
content_policy_fallbacks: Optional[List] = None
allowed_fails: int = 3
allow_dynamic_callback_disabling: bool = True
num_retries_per_request: Optional[int] = (
    None  # for the request overall (incl. fallbacks + model retries)
)
####### SECRET MANAGERS #####################
secret_manager_client: Optional[Any] = (
    None  # list of instantiated key management clients - e.g. azure kv, infisical, etc.
)
_google_kms_resource_name: Optional[str] = None
_key_management_system: Optional["KeyManagementSystem"] = None
# KeyManagementSettings must be imported directly because _key_management_settings
# is accessed during import (in dd_tracing.py via get_secret)
from litellm.types.secret_managers.main import KeyManagementSettings
_key_management_settings: "KeyManagementSettings" = KeyManagementSettings()
#### PII MASKING ####
output_parse_pii: bool = False
#############################################
from litellm.litellm_core_utils.get_model_cost_map import get_model_cost_map

model_cost = get_model_cost_map(url=model_cost_map_url)
cost_discount_config: Dict[str, float] = (
    {}
)  # Provider-specific cost discounts {"vertex_ai": 0.05} = 5% discount
custom_prompt_dict: Dict[str, dict] = {}
check_provider_endpoint = False


####### THREAD-SPECIFIC DATA ####################
class MyLocal(threading.local):
    def __init__(self):
        self.user = "Hello World"


_thread_context = MyLocal()


def identify(event_details):
    # Store user in thread local data
    if "user" in event_details:
        _thread_context.user = event_details["user"]


####### ADDITIONAL PARAMS ################### configurable params if you use proxy models like Helicone, map spend to org id, etc.
api_base: Optional[str] = None
headers = None
api_version: Optional[str] = None
organization = None
project = None
config_path = None
vertex_ai_safety_settings: Optional[dict] = None

####### COMPLETION MODELS ###################
from typing import Set

open_ai_chat_completion_models: Set = set()
open_ai_text_completion_models: Set = set()
cohere_models: Set = set()
cohere_chat_models: Set = set()
mistral_chat_models: Set = set()
text_completion_codestral_models: Set = set()
anthropic_models: Set = set()
openrouter_models: Set = set()
datarobot_models: Set = set()
vertex_language_models: Set = set()
vertex_vision_models: Set = set()
vertex_chat_models: Set = set()
vertex_code_chat_models: Set = set()
vertex_ai_image_models: Set = set()
vertex_ai_video_models: Set = set()
vertex_text_models: Set = set()
vertex_code_text_models: Set = set()
vertex_embedding_models: Set = set()
vertex_anthropic_models: Set = set()
vertex_llama3_models: Set = set()
vertex_deepseek_models: Set = set()
vertex_ai_ai21_models: Set = set()
vertex_mistral_models: Set = set()
vertex_openai_models: Set = set()
vertex_minimax_models: Set = set()
vertex_moonshot_models: Set = set()
ai21_models: Set = set()
ai21_chat_models: Set = set()
nlp_cloud_models: Set = set()
aleph_alpha_models: Set = set()
bedrock_models: Set = set()
bedrock_converse_models: Set = set(BEDROCK_CONVERSE_MODELS)
fal_ai_models: Set = set()
fireworks_ai_models: Set = set()
fireworks_ai_embedding_models: Set = set()
deepinfra_models: Set = set()
perplexity_models: Set = set()
watsonx_models: Set = set()
gemini_models: Set = set()
xai_models: Set = set()
deepseek_models: Set = set()
runwayml_models: Set = set()
azure_ai_models: Set = set()
jina_ai_models: Set = set()
voyage_models: Set = set()
infinity_models: Set = set()
heroku_models: Set = set()
databricks_models: Set = set()
cloudflare_models: Set = set()
codestral_models: Set = set()
friendliai_models: Set = set()
featherless_ai_models: Set = set()
palm_models: Set = set()
groq_models: Set = set()
azure_models: Set = set()
azure_text_models: Set = set()
anyscale_models: Set = set()
cerebras_models: Set = set()
galadriel_models: Set = set()
nvidia_nim_models: Set = set()
sambanova_models: Set = set()
sambanova_embedding_models: Set = set()
novita_models: Set = set()
assemblyai_models: Set = set()
snowflake_models: Set = set()
gradient_ai_models: Set = set()
llama_models: Set = set()
nscale_models: Set = set()
nebius_models: Set = set()
nebius_embedding_models: Set = set()
aiml_models: Set = set()
deepgram_models: Set = set()
elevenlabs_models: Set = set()
dashscope_models: Set = set()
moonshot_models: Set = set()
v0_models: Set = set()
morph_models: Set = set()
lambda_ai_models: Set = set()
hyperbolic_models: Set = set()
recraft_models: Set = set()
cometapi_models: Set = set()
oci_models: Set = set()
vercel_ai_gateway_models: Set = set()
volcengine_models: Set = set()
wandb_models: Set = set(WANDB_MODELS)
ovhcloud_models: Set = set()
ovhcloud_embedding_models: Set = set()
lemonade_models: Set = set()
docker_model_runner_models: Set = set()


def is_bedrock_pricing_only_model(key: str) -> bool:
    """
    Excludes keys with the pattern 'bedrock/<region>/<model>'. These are in the model_prices_and_context_window.json file for pricing purposes only.

    Args:
        key (str): A key to filter.

    Returns:
        bool: True if the key matches the Bedrock pattern, False otherwise.
    """
    # Regex to match 'bedrock/<region>/<model>'
    bedrock_pattern = re.compile(r"^bedrock/[a-zA-Z0-9_-]+/.+$")

    if "month-commitment" in key:
        return True

    is_match = bedrock_pattern.match(key)
    return is_match is not None


def is_openai_finetune_model(key: str) -> bool:
    """
    Excludes model cost keys with the pattern 'ft:<model>'. These are in the model_prices_and_context_window.json file for pricing purposes only.

    Args:
        key (str): A key to filter.

    Returns:
        bool: True if the key matches the OpenAI finetune pattern, False otherwise.
    """
    return key.startswith("ft:") and not key.count(":") > 1


def add_known_models():
    for key, value in model_cost.items():
        if value.get("litellm_provider") == "openai" and not is_openai_finetune_model(
            key
        ):
            open_ai_chat_completion_models.add(key)
        elif value.get("litellm_provider") == "text-completion-openai":
            open_ai_text_completion_models.add(key)
        elif value.get("litellm_provider") == "azure_text":
            azure_text_models.add(key)
        elif value.get("litellm_provider") == "cohere":
            cohere_models.add(key)
        elif value.get("litellm_provider") == "cohere_chat":
            cohere_chat_models.add(key)
        elif value.get("litellm_provider") == "mistral":
            mistral_chat_models.add(key)
        elif value.get("litellm_provider") == "anthropic":
            anthropic_models.add(key)
        elif value.get("litellm_provider") == "empower":
            empower_models.add(key)
        elif value.get("litellm_provider") == "openrouter":
            openrouter_models.add(key)
        elif value.get("litellm_provider") == "vercel_ai_gateway":
            vercel_ai_gateway_models.add(key)
        elif value.get("litellm_provider") == "datarobot":
            datarobot_models.add(key)
        elif value.get("litellm_provider") == "vertex_ai-text-models":
            vertex_text_models.add(key)
        elif value.get("litellm_provider") == "vertex_ai-code-text-models":
            vertex_code_text_models.add(key)
        elif value.get("litellm_provider") == "vertex_ai-language-models":
            vertex_language_models.add(key)
        elif value.get("litellm_provider") == "vertex_ai-vision-models":
            vertex_vision_models.add(key)
        elif value.get("litellm_provider") == "vertex_ai-chat-models":
            vertex_chat_models.add(key)
        elif value.get("litellm_provider") == "vertex_ai-code-chat-models":
            vertex_code_chat_models.add(key)
        elif value.get("litellm_provider") == "vertex_ai-embedding-models":
            vertex_embedding_models.add(key)
        elif value.get("litellm_provider") == "vertex_ai-anthropic_models":
            key = key.replace("vertex_ai/", "")
            vertex_anthropic_models.add(key)
        elif value.get("litellm_provider") == "vertex_ai-llama_models":
            key = key.replace("vertex_ai/", "")
            vertex_llama3_models.add(key)
        elif value.get("litellm_provider") == "vertex_ai-deepseek_models":
            key = key.replace("vertex_ai/", "")
            vertex_deepseek_models.add(key)
        elif value.get("litellm_provider") == "vertex_ai-mistral_models":
            key = key.replace("vertex_ai/", "")
            vertex_mistral_models.add(key)
        elif value.get("litellm_provider") == "vertex_ai-ai21_models":
            key = key.replace("vertex_ai/", "")
            vertex_ai_ai21_models.add(key)
        elif value.get("litellm_provider") == "vertex_ai-image-models":
            key = key.replace("vertex_ai/", "")
            vertex_ai_image_models.add(key)
        elif value.get("litellm_provider") == "vertex_ai-video-models":
            key = key.replace("vertex_ai/", "")
            vertex_ai_video_models.add(key)
        elif value.get("litellm_provider") == "vertex_ai-openai_models":
            key = key.replace("vertex_ai/", "")
            vertex_openai_models.add(key)
        elif value.get("litellm_provider") == "vertex_ai-minimax_models":
            key = key.replace("vertex_ai/", "")
            vertex_minimax_models.add(key)
        elif value.get("litellm_provider") == "vertex_ai-moonshot_models":
            key = key.replace("vertex_ai/", "")
            vertex_moonshot_models.add(key)
        elif value.get("litellm_provider") == "ai21":
            if value.get("mode") == "chat":
                ai21_chat_models.add(key)
            else:
                ai21_models.add(key)
        elif value.get("litellm_provider") == "nlp_cloud":
            nlp_cloud_models.add(key)
        elif value.get("litellm_provider") == "aleph_alpha":
            aleph_alpha_models.add(key)
        elif value.get(
            "litellm_provider"
        ) == "bedrock" and not is_bedrock_pricing_only_model(key):
            bedrock_models.add(key)
        elif value.get("litellm_provider") == "bedrock_converse":
            bedrock_converse_models.add(key)
        elif value.get("litellm_provider") == "deepinfra":
            deepinfra_models.add(key)
        elif value.get("litellm_provider") == "perplexity":
            perplexity_models.add(key)
        elif value.get("litellm_provider") == "watsonx":
            watsonx_models.add(key)
        elif value.get("litellm_provider") == "gemini":
            gemini_models.add(key)
        elif value.get("litellm_provider") == "fireworks_ai":
            # ignore the 'up-to', '-to-' model names -> not real models. just for cost tracking based on model params.
            if "-to-" not in key and "fireworks-ai-default" not in key:
                fireworks_ai_models.add(key)
        elif value.get("litellm_provider") == "fireworks_ai-embedding-models":
            # ignore the 'up-to', '-to-' model names -> not real models. just for cost tracking based on model params.
            if "-to-" not in key:
                fireworks_ai_embedding_models.add(key)
        elif value.get("litellm_provider") == "text-completion-codestral":
            text_completion_codestral_models.add(key)
        elif value.get("litellm_provider") == "xai":
            xai_models.add(key)
        elif value.get("litellm_provider") == "fal_ai":
            fal_ai_models.add(key)
        elif value.get("litellm_provider") == "deepseek":
            deepseek_models.add(key)
        elif value.get("litellm_provider") == "runwayml":
            runwayml_models.add(key)
        elif value.get("litellm_provider") == "meta_llama":
            llama_models.add(key)
        elif value.get("litellm_provider") == "nscale":
            nscale_models.add(key)
        elif value.get("litellm_provider") == "azure_ai":
            azure_ai_models.add(key)
        elif value.get("litellm_provider") == "voyage":
            voyage_models.add(key)
        elif value.get("litellm_provider") == "infinity":
            infinity_models.add(key)
        elif value.get("litellm_provider") == "databricks":
            databricks_models.add(key)
        elif value.get("litellm_provider") == "cloudflare":
            cloudflare_models.add(key)
        elif value.get("litellm_provider") == "codestral":
            codestral_models.add(key)
        elif value.get("litellm_provider") == "friendliai":
            friendliai_models.add(key)
        elif value.get("litellm_provider") == "palm":
            palm_models.add(key)
        elif value.get("litellm_provider") == "groq":
            groq_models.add(key)
        elif value.get("litellm_provider") == "azure":
            azure_models.add(key)
        elif value.get("litellm_provider") == "anyscale":
            anyscale_models.add(key)
        elif value.get("litellm_provider") == "cerebras":
            cerebras_models.add(key)
        elif value.get("litellm_provider") == "galadriel":
            galadriel_models.add(key)
        elif value.get("litellm_provider") == "nvidia_nim":
            nvidia_nim_models.add(key)
        elif value.get("litellm_provider") == "sambanova":
            sambanova_models.add(key)
        elif value.get("litellm_provider") == "sambanova-embedding-models":
            sambanova_embedding_models.add(key)
        elif value.get("litellm_provider") == "novita":
            novita_models.add(key)
        elif value.get("litellm_provider") == "nebius-chat-models":
            nebius_models.add(key)
        elif value.get("litellm_provider") == "nebius-embedding-models":
            nebius_embedding_models.add(key)
        elif value.get("litellm_provider") == "aiml":
            aiml_models.add(key)
        elif value.get("litellm_provider") == "assemblyai":
            assemblyai_models.add(key)
        elif value.get("litellm_provider") == "jina_ai":
            jina_ai_models.add(key)
        elif value.get("litellm_provider") == "snowflake":
            snowflake_models.add(key)
        elif value.get("litellm_provider") == "gradient_ai":
            gradient_ai_models.add(key)
        elif value.get("litellm_provider") == "featherless_ai":
            featherless_ai_models.add(key)
        elif value.get("litellm_provider") == "deepgram":
            deepgram_models.add(key)
        elif value.get("litellm_provider") == "elevenlabs":
            elevenlabs_models.add(key)
        elif value.get("litellm_provider") == "heroku":
            heroku_models.add(key)
        elif value.get("litellm_provider") == "dashscope":
            dashscope_models.add(key)
        elif value.get("litellm_provider") == "moonshot":
            moonshot_models.add(key)
        elif value.get("litellm_provider") == "v0":
            v0_models.add(key)
        elif value.get("litellm_provider") == "morph":
            morph_models.add(key)
        elif value.get("litellm_provider") == "lambda_ai":
            lambda_ai_models.add(key)
        elif value.get("litellm_provider") == "hyperbolic":
            hyperbolic_models.add(key)
        elif value.get("litellm_provider") == "recraft":
            recraft_models.add(key)
        elif value.get("litellm_provider") == "cometapi":
            cometapi_models.add(key)
        elif value.get("litellm_provider") == "oci":
            oci_models.add(key)
        elif value.get("litellm_provider") == "volcengine":
            volcengine_models.add(key)
        elif value.get("litellm_provider") == "wandb":
            wandb_models.add(key)
        elif value.get("litellm_provider") == "ovhcloud":
            ovhcloud_models.add(key)
        elif value.get("litellm_provider") == "ovhcloud-embedding-models":
            ovhcloud_embedding_models.add(key)
        elif value.get("litellm_provider") == "lemonade":
            lemonade_models.add(key)
        elif value.get("litellm_provider") == "docker_model_runner":
            docker_model_runner_models.add(key)


add_known_models()
# known openai compatible endpoints - we'll eventually move this list to the model_prices_and_context_window.json dictionary

# this is maintained for Exception Mapping


# used for Cost Tracking & Token counting
# https://azure.microsoft.com/en-in/pricing/details/cognitive-services/openai-service/
# Azure returns gpt-35-turbo in their responses, we need to map this to azure/gpt-3.5-turbo for token counting
azure_llms = {
    "gpt-35-turbo": "azure/gpt-35-turbo",
    "gpt-35-turbo-16k": "azure/gpt-35-turbo-16k",
    "gpt-35-turbo-instruct": "azure/gpt-35-turbo-instruct",
    "azure/gpt-41": "gpt-4.1",
    "azure/gpt-41-mini": "gpt-4.1-mini",
    "azure/gpt-41-nano": "gpt-4.1-nano",
}

azure_embedding_models = {
    "ada": "azure/ada",
}

petals_models = [
    "petals-team/StableBeluga2",
]

ollama_models = ["llama2"]

maritalk_models = ["maritalk"]

model_list = list(
    open_ai_chat_completion_models
    | open_ai_text_completion_models
    | cohere_models
    | cohere_chat_models
    | anthropic_models
    | set(replicate_models)
    | openrouter_models
    | datarobot_models
    | set(huggingface_models)
    | vertex_chat_models
    | vertex_text_models
    | ai21_models
    | ai21_chat_models
    | set(together_ai_models)
    | set(baseten_models)
    | aleph_alpha_models
    | nlp_cloud_models
    | set(ollama_models)
    | bedrock_models
    | deepinfra_models
    | perplexity_models
    | set(maritalk_models)
    | runwayml_models
    | vertex_language_models
    | watsonx_models
    | gemini_models
    | text_completion_codestral_models
    | xai_models
    | fal_ai_models
    | deepseek_models
    | azure_ai_models
    | voyage_models
    | infinity_models
    | databricks_models
    | cloudflare_models
    | codestral_models
    | friendliai_models
    | palm_models
    | groq_models
    | azure_models
    | anyscale_models
    | cerebras_models
    | galadriel_models
    | nvidia_nim_models
    | sambanova_models
    | azure_text_models
    | novita_models
    | assemblyai_models
    | jina_ai_models
    | snowflake_models
    | gradient_ai_models
    | llama_models
    | featherless_ai_models
    | nscale_models
    | deepgram_models
    | elevenlabs_models
    | dashscope_models
    | moonshot_models
    | v0_models
    | morph_models
    | lambda_ai_models
    | recraft_models
    | cometapi_models
    | oci_models
    | heroku_models
    | vercel_ai_gateway_models
    | volcengine_models
    | wandb_models
    | ovhcloud_models
    | lemonade_models
    | docker_model_runner_models
    | set(clarifai_models)
)

model_list_set = set(model_list)



models_by_provider: dict = {
    "openai": open_ai_chat_completion_models | open_ai_text_completion_models,
    "text-completion-openai": open_ai_text_completion_models,
    "cohere": cohere_models | cohere_chat_models,
    "cohere_chat": cohere_chat_models,
    "anthropic": anthropic_models,
    "replicate": replicate_models,
    "huggingface": huggingface_models,
    "together_ai": together_ai_models,
    "baseten": baseten_models,
    "openrouter": openrouter_models,
    "vercel_ai_gateway": vercel_ai_gateway_models,
    "datarobot": datarobot_models,
    "vertex_ai": vertex_chat_models
    | vertex_text_models
    | vertex_anthropic_models
    | vertex_vision_models
    | vertex_language_models
    | vertex_deepseek_models
    | vertex_minimax_models
    | vertex_moonshot_models,
    "ai21": ai21_models,
    "bedrock": bedrock_models | bedrock_converse_models,
    "petals": petals_models,
    "ollama": ollama_models,
    "ollama_chat": ollama_models,
    "deepinfra": deepinfra_models,
    "perplexity": perplexity_models,
    "maritalk": maritalk_models,
    "watsonx": watsonx_models,
    "gemini": gemini_models,
    "fireworks_ai": fireworks_ai_models | fireworks_ai_embedding_models,
    "aleph_alpha": aleph_alpha_models,
    "text-completion-codestral": text_completion_codestral_models,
    "xai": xai_models,
    "fal_ai": fal_ai_models,
    "deepseek": deepseek_models,
    "runwayml": runwayml_models,
    "mistral": mistral_chat_models,
    "azure_ai": azure_ai_models,
    "voyage": voyage_models,
    "infinity": infinity_models,
    "databricks": databricks_models,
    "cloudflare": cloudflare_models,
    "codestral": codestral_models,
    "nlp_cloud": nlp_cloud_models,
    "friendliai": friendliai_models,
    "palm": palm_models,
    "groq": groq_models,
    "azure": azure_models | azure_text_models,
    "azure_text": azure_text_models,
    "anyscale": anyscale_models,
    "cerebras": cerebras_models,
    "galadriel": galadriel_models,
    "nvidia_nim": nvidia_nim_models,
    "sambanova": sambanova_models | sambanova_embedding_models,
    "novita": novita_models,
    "nebius": nebius_models | nebius_embedding_models,
    "aiml": aiml_models,
    "assemblyai": assemblyai_models,
    "jina_ai": jina_ai_models,
    "snowflake": snowflake_models,
    "gradient_ai": gradient_ai_models,
    "meta_llama": llama_models,
    "nscale": nscale_models,
    "featherless_ai": featherless_ai_models,
    "deepgram": deepgram_models,
    "elevenlabs": elevenlabs_models,
    "heroku": heroku_models,
    "dashscope": dashscope_models,
    "moonshot": moonshot_models,
    "v0": v0_models,
    "morph": morph_models,
    "lambda_ai": lambda_ai_models,
    "hyperbolic": hyperbolic_models,
    "recraft": recraft_models,
    "cometapi": cometapi_models,
    "oci": oci_models,
    "volcengine": volcengine_models,
    "wandb": wandb_models,
    "ovhcloud": ovhcloud_models | ovhcloud_embedding_models,
    "lemonade": lemonade_models,
    "clarifai": clarifai_models,
}

# mapping for those models which have larger equivalents
longer_context_model_fallback_dict: dict = {
    # openai chat completion models
    "gpt-3.5-turbo": "gpt-3.5-turbo-16k",
    "gpt-3.5-turbo-0301": "gpt-3.5-turbo-16k-0301",
    "gpt-3.5-turbo-0613": "gpt-3.5-turbo-16k-0613",
    "gpt-4": "gpt-4-32k",
    "gpt-4-0314": "gpt-4-32k-0314",
    "gpt-4-0613": "gpt-4-32k-0613",
    # anthropic
    "claude-instant-1": "claude-2",
    "claude-instant-1.2": "claude-2",
    # vertexai
    "chat-bison": "chat-bison-32k",
    "chat-bison@001": "chat-bison-32k",
    "codechat-bison": "codechat-bison-32k",
    "codechat-bison@001": "codechat-bison-32k",
    # openrouter
    "openrouter/openai/gpt-3.5-turbo": "openrouter/openai/gpt-3.5-turbo-16k",
    "openrouter/anthropic/claude-instant-v1": "openrouter/anthropic/claude-2",
}

####### EMBEDDING MODELS ###################

all_embedding_models = (
    open_ai_embedding_models
    | set(cohere_embedding_models)
    | set(bedrock_embedding_models)
    | vertex_embedding_models
    | fireworks_ai_embedding_models
    | nebius_embedding_models
    | sambanova_embedding_models
    | ovhcloud_embedding_models
)

####### IMAGE GENERATION MODELS ###################
openai_image_generation_models = ["dall-e-2", "dall-e-3"]

####### VIDEO GENERATION MODELS ###################
openai_video_generation_models = ["sora-2"]

from .timeout import timeout
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
# Note: remove_index_from_tool_calls is lazy-loaded via __getattr__ to reduce import-time memory cost
# Note: Most other utils imports are lazy-loaded via __getattr__ to avoid loading utils.py 
# (which imports tiktoken) at import time

from .llms.triton.completion.transformation import TritonGenerateConfig
from .llms.triton.completion.transformation import TritonInferConfig
from .llms.gemini.common_utils import GeminiModelInfo


from .llms.vertex_ai.vertex_embeddings.transformation import (
    VertexAITextEmbeddingConfig,
)

vertexAITextEmbeddingConfig = VertexAITextEmbeddingConfig()

from .llms.bedrock.embed.twelvelabs_marengo_transformation import (
    TwelveLabsMarengoEmbeddingConfig,
)
from .llms.openai.image_variations.transformation import OpenAIImageVariationConfig
from .llms.deepgram.audio_transcription.transformation import (
    DeepgramAudioTranscriptionConfig,
)
from .llms.topaz.common_utils import TopazModelInfo
from .llms.topaz.image_variations.transformation import TopazImageVariationConfig
from .llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from .llms.xai.responses.transformation import XAIResponsesAPIConfig
from .llms.litellm_proxy.responses.transformation import (
    LiteLLMProxyResponsesAPIConfig,
)
from .llms.openai.transcriptions.whisper_transformation import (
    OpenAIWhisperAudioTranscriptionConfig,
)
from .llms.openai.transcriptions.gpt_transformation import (
    OpenAIGPTAudioTranscriptionConfig,
)

from .llms.nvidia_nim.embed import NvidiaNimEmbeddingConfig

nvidiaNimEmbeddingConfig = NvidiaNimEmbeddingConfig()

from .llms.featherless_ai.chat.transformation import FeatherlessAIConfig
from .llms.cerebras.chat import CerebrasConfig
from .llms.baseten.chat import BasetenConfig
from .llms.sambanova.chat import SambanovaConfig
from .llms.sambanova.embedding.transformation import SambaNovaEmbeddingConfig
from .llms.fireworks_ai.chat.transformation import FireworksAIConfig
from .llms.fireworks_ai.completion.transformation import FireworksAITextCompletionConfig
from .llms.fireworks_ai.audio_transcription.transformation import (
    FireworksAIAudioTranscriptionConfig,
)
from .llms.fireworks_ai.embed.fireworks_ai_transformation import (
    FireworksAIEmbeddingConfig,
)
from .llms.friendliai.chat.transformation import FriendliaiChatConfig
from .llms.jina_ai.embedding.transformation import JinaAIEmbeddingConfig
from .llms.xai.chat.transformation import XAIChatConfig
from .llms.xai.common_utils import XAIModelInfo
from .llms.aiml.chat.transformation import AIMLChatConfig
from .llms.volcengine.chat.transformation import (
    VolcEngineChatConfig as VolcEngineConfig,
)
from .llms.codestral.completion.transformation import CodestralTextCompletionConfig
from .llms.azure.azure import (
    AzureOpenAIError,
    AzureOpenAIAssistantsAPIConfig,
)
from .llms.heroku.chat.transformation import HerokuChatConfig
from .llms.cometapi.chat.transformation import CometAPIConfig
from .llms.azure.chat.gpt_transformation import AzureOpenAIConfig
from .llms.azure.chat.gpt_5_transformation import AzureOpenAIGPT5Config
from .llms.azure.completion.transformation import AzureOpenAITextConfig
from .llms.hosted_vllm.chat.transformation import HostedVLLMChatConfig
from .llms.llamafile.chat.transformation import LlamafileChatConfig
from .llms.litellm_proxy.chat.transformation import LiteLLMProxyChatConfig
from .llms.vllm.completion.transformation import VLLMConfig
from .llms.deepseek.chat.transformation import DeepSeekChatConfig
from .llms.lm_studio.chat.transformation import LMStudioChatConfig
from .llms.lm_studio.embed.transformation import LmStudioEmbeddingConfig
from .llms.nscale.chat.transformation import NscaleConfig
from .llms.perplexity.chat.transformation import PerplexityChatConfig
from .llms.watsonx.completion.transformation import IBMWatsonXAIConfig
from .llms.watsonx.chat.transformation import IBMWatsonXChatConfig
from .llms.watsonx.embed.transformation import IBMWatsonXEmbeddingConfig
from .llms.github_copilot.chat.transformation import GithubCopilotConfig
from .llms.github_copilot.responses.transformation import (
    GithubCopilotResponsesAPIConfig,
)
from .llms.nebius.chat.transformation import NebiusConfig
from .llms.wandb.chat.transformation import WandbConfig
from .llms.dashscope.chat.transformation import DashScopeChatConfig
from .llms.moonshot.chat.transformation import MoonshotChatConfig
from .llms.docker_model_runner.chat.transformation import DockerModelRunnerChatConfig
from .llms.v0.chat.transformation import V0ChatConfig
from .llms.oci.chat.transformation import OCIChatConfig
from .llms.morph.chat.transformation import MorphChatConfig
from .llms.lambda_ai.chat.transformation import LambdaAIChatConfig
from .llms.hyperbolic.chat.transformation import HyperbolicChatConfig
from .llms.vercel_ai_gateway.chat.transformation import VercelAIGatewayConfig
from .llms.ovhcloud.chat.transformation import OVHCloudChatConfig
from .llms.ovhcloud.embedding.transformation import OVHCloudEmbeddingConfig
from .llms.cometapi.embed.transformation import CometAPIEmbeddingConfig
from .llms.lemonade.chat.transformation import LemonadeChatConfig
from .llms.snowflake.embedding.transformation import SnowflakeEmbeddingConfig
from .utils import client
from .main import *  # type: ignore
from .integrations import *
from .llms.custom_httpx.async_client_cleanup import close_litellm_async_clients
from .exceptions import (
    AuthenticationError,
    InvalidRequestError,
    BadRequestError,
    ImageFetchError,
    NotFoundError,
    RateLimitError,
    ServiceUnavailableError,
    BadGatewayError,
    OpenAIError,
    ContextWindowExceededError,
    ContentPolicyViolationError,
    BudgetExceededError,
    APIError,
    Timeout,
    APIConnectionError,
    UnsupportedParamsError,
    APIResponseValidationError,
    UnprocessableEntityError,
    InternalServerError,
    JSONSchemaValidationError,
    LITELLM_EXCEPTION_TYPES,
    MockException,
)
from .budget_manager import BudgetManager
from .proxy.proxy_cli import run_server
from .router import Router
from .assistants.main import *
from .batches.main import *
from .images.main import *
from .videos.main import *
from .batch_completion.main import *  # type: ignore
from .rerank_api.main import *
from .llms.anthropic.experimental_pass_through.messages.handler import *
from .responses.main import *
from .containers.main import *
from .ocr.main import *
from .search.main import *
from .realtime_api.main import _arealtime
from .fine_tuning.main import *
from .files.main import *
from .vector_store_files.main import (
    acreate as avector_store_file_create,
    adelete as avector_store_file_delete,
    alist as avector_store_file_list,
    aretrieve as avector_store_file_retrieve,
    aretrieve_content as avector_store_file_content,
    aupdate as avector_store_file_update,
    create as vector_store_file_create,
    delete as vector_store_file_delete,
    list as vector_store_file_list,
    retrieve as vector_store_file_retrieve,
    retrieve_content as vector_store_file_content,
    update as vector_store_file_update,
)
from .scheduler import *
# Note: response_cost_calculator and cost_per_token are imported lazily via __getattr__ 
# to avoid loading cost_calculator.py at import time

### ADAPTERS ###
from .types.adapter import AdapterItem
import litellm.anthropic_interface as anthropic

adapters: List[AdapterItem] = []

### Vector Store Registry ###
from .vector_stores.vector_store_registry import (
    VectorStoreRegistry,
    VectorStoreIndexRegistry,
)

vector_store_registry: Optional[VectorStoreRegistry] = None
vector_store_index_registry: Optional[VectorStoreIndexRegistry] = None

### CUSTOM LLMs ###
from .types.llms.custom_llm import CustomLLMItem
from .types.utils import GenericStreamingChunk

custom_provider_map: List[CustomLLMItem] = []
_custom_providers: List[str] = (
    []
)  # internal helper util, used to track names of custom providers
disable_hf_tokenizer_download: Optional[bool] = (
    None  # disable huggingface tokenizer download. Defaults to openai clk100
)
global_disable_no_log_param: bool = False

### CLI UTILITIES ###
from litellm.litellm_core_utils.cli_token_utils import get_litellm_gateway_api_key

### PASSTHROUGH ###
from .passthrough import allm_passthrough_route, llm_passthrough_route
from .google_genai import agenerate_content

### GLOBAL CONFIG ###
global_bitbucket_config: Optional[Dict[str, Any]] = None


def set_global_bitbucket_config(config: Dict[str, Any]) -> None:
    """Set global BitBucket configuration for prompt management."""
    global global_bitbucket_config
    global_bitbucket_config = config


### GLOBAL CONFIG ###
global_gitlab_config: Optional[Dict[str, Any]] = None


def set_global_gitlab_config(config: Dict[str, Any]) -> None:
    """Set global BitBucket configuration for prompt management."""
    global global_gitlab_config
    global_gitlab_config = config


# Lazy import helper functions are imported inside __getattr__ to avoid any import-time overhead





# Lazy import for HTTP handlers to reduce import-time memory cost
def _lazy_import_http_handlers(name: str) -> Any:
    """Lazy import for HTTP handler instances and classes - imports only what's needed per name."""
    # Handle HTTP handler instances
    if name == "module_level_aclient":
        from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler as _AsyncHTTPHandler
        _module_level_aclient = _AsyncHTTPHandler(
            timeout=request_timeout, client_alias="module level aclient"
        )
        globals()["module_level_aclient"] = _module_level_aclient
        return _module_level_aclient
    
    if name == "module_level_client":
        from litellm.llms.custom_httpx.http_handler import HTTPHandler as _HTTPHandler
        _module_level_client = _HTTPHandler(timeout=request_timeout)
        globals()["module_level_client"] = _module_level_client
        return _module_level_client
    
    # Handle HTTP handler classes for backward compatibility
    if name == "AsyncHTTPHandler":
        from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler as _AsyncHTTPHandler
        globals()["AsyncHTTPHandler"] = _AsyncHTTPHandler
        return _AsyncHTTPHandler
    
    if name == "HTTPHandler":
        from litellm.llms.custom_httpx.http_handler import HTTPHandler as _HTTPHandler
        globals()["HTTPHandler"] = _HTTPHandler
        return _HTTPHandler
    
    raise AttributeError(f"HTTP handler lazy import: unknown attribute {name!r}")


# Lazy import for caching classes to reduce import-time memory cost
def _lazy_import_caching(name: str) -> Any:
    """Lazy import for caching classes - imports only the requested class by name."""
    if name == "Cache":
        from litellm.caching.caching import Cache as _Cache
        globals()["Cache"] = _Cache
        return _Cache
    
    if name == "DualCache":
        from litellm.caching.caching import DualCache as _DualCache
        globals()["DualCache"] = _DualCache
        return _DualCache
    
    if name == "RedisCache":
        from litellm.caching.caching import RedisCache as _RedisCache
        globals()["RedisCache"] = _RedisCache
        return _RedisCache
    
    if name == "InMemoryCache":
        from litellm.caching.caching import InMemoryCache as _InMemoryCache
        globals()["InMemoryCache"] = _InMemoryCache
        return _InMemoryCache
    
    if name == "LLMClientCache":
        from litellm.caching.llm_caching_handler import LLMClientCache as _LLMClientCache
        globals()["LLMClientCache"] = _LLMClientCache
        return _LLMClientCache
    
    raise AttributeError(f"Caching lazy import: unknown attribute {name!r}")


def _lazy_import_types_utils(name: str) -> Any:
    """Lazy import for types.utils module - imports only the requested item by name."""
    if name == "ImageObject":
        from litellm.types.utils import ImageObject as _ImageObject
        globals()["ImageObject"] = _ImageObject
        return _ImageObject
    
    if name == "BudgetConfig":
        from litellm.types.utils import BudgetConfig as _BudgetConfig
        globals()["BudgetConfig"] = _BudgetConfig
        return _BudgetConfig
    
    if name == "all_litellm_params":
        from litellm.types.utils import all_litellm_params as _all_litellm_params
        globals()["all_litellm_params"] = _all_litellm_params
        return _all_litellm_params
    
    if name == "_litellm_completion_params":
        from litellm.types.utils import all_litellm_params as _all_litellm_params
        globals()["_litellm_completion_params"] = _all_litellm_params
        return _all_litellm_params
    
    if name == "CredentialItem":
        from litellm.types.utils import CredentialItem as _CredentialItem
        globals()["CredentialItem"] = _CredentialItem
        return _CredentialItem
    
    if name == "PriorityReservationDict":
        from litellm.types.utils import PriorityReservationDict as _PriorityReservationDict
        globals()["PriorityReservationDict"] = _PriorityReservationDict
        return _PriorityReservationDict
    
    if name == "StandardKeyGenerationConfig":
        from litellm.types.utils import StandardKeyGenerationConfig as _StandardKeyGenerationConfig
        globals()["StandardKeyGenerationConfig"] = _StandardKeyGenerationConfig
        return _StandardKeyGenerationConfig
    
    if name == "LlmProviders":
        from litellm.types.utils import LlmProviders as _LlmProviders
        globals()["LlmProviders"] = _LlmProviders
        return _LlmProviders
    
    if name == "SearchProviders":
        from litellm.types.utils import SearchProviders as _SearchProviders
        globals()["SearchProviders"] = _SearchProviders
        return _SearchProviders
    
    if name == "PriorityReservationSettings":
        from litellm.types.utils import PriorityReservationSettings as _PriorityReservationSettings
        globals()["PriorityReservationSettings"] = _PriorityReservationSettings
        return _PriorityReservationSettings
    
    raise AttributeError(f"Types utils lazy import: unknown attribute {name!r}")


def _lazy_import_ui_sso(name: str) -> Any:
    """Lazy import for types.proxy.management_endpoints.ui_sso module - imports only the requested item by name."""
    if name == "DefaultTeamSSOParams":
        from litellm.types.proxy.management_endpoints.ui_sso import DefaultTeamSSOParams as _DefaultTeamSSOParams
        globals()["DefaultTeamSSOParams"] = _DefaultTeamSSOParams
        return _DefaultTeamSSOParams
    
    if name == "LiteLLM_UpperboundKeyGenerateParams":
        from litellm.types.proxy.management_endpoints.ui_sso import LiteLLM_UpperboundKeyGenerateParams as _LiteLLM_UpperboundKeyGenerateParams
        globals()["LiteLLM_UpperboundKeyGenerateParams"] = _LiteLLM_UpperboundKeyGenerateParams
        return _LiteLLM_UpperboundKeyGenerateParams
    
    raise AttributeError(f"UI SSO lazy import: unknown attribute {name!r}")


def _lazy_import_secret_managers(name: str) -> Any:
    """Lazy import for types.secret_managers.main module - imports only the requested item by name."""
    if name == "KeyManagementSystem":
        from litellm.types.secret_managers.main import KeyManagementSystem as _KeyManagementSystem
        globals()["KeyManagementSystem"] = _KeyManagementSystem
        return _KeyManagementSystem
    
    if name == "KeyManagementSettings":
        from litellm.types.secret_managers.main import KeyManagementSettings as _KeyManagementSettings
        globals()["KeyManagementSettings"] = _KeyManagementSettings
        return _KeyManagementSettings
    
    raise AttributeError(f"Secret managers lazy import: unknown attribute {name!r}")


def _lazy_import_logging_integrations(name: str) -> Any:
    """Lazy import for logging-related integrations - imports only the requested item by name."""
    if name == "CustomLogger":
        from litellm.integrations.custom_logger import CustomLogger as _CustomLogger
        globals()["CustomLogger"] = _CustomLogger
        return _CustomLogger
    
    if name == "LoggingCallbackManager":
        from litellm.litellm_core_utils.logging_callback_manager import LoggingCallbackManager as _LoggingCallbackManager
        globals()["LoggingCallbackManager"] = _LoggingCallbackManager
        return _LoggingCallbackManager
    
    raise AttributeError(f"Logging integrations lazy import: unknown attribute {name!r}")


def _lazy_import_dotprompt(name: str) -> Any:
    """Lazy import for dotprompt module - imports only the requested item by name."""
    if name == "global_prompt_manager":
        from litellm.integrations.dotprompt import global_prompt_manager as _global_prompt_manager
        globals()["global_prompt_manager"] = _global_prompt_manager
        return _global_prompt_manager
    
    if name == "global_prompt_directory":
        from litellm.integrations.dotprompt import global_prompt_directory as _global_prompt_directory
        globals()["global_prompt_directory"] = _global_prompt_directory
        return _global_prompt_directory
    
    if name == "set_global_prompt_directory":
        from litellm.integrations.dotprompt import set_global_prompt_directory as _set_global_prompt_directory
        globals()["set_global_prompt_directory"] = _set_global_prompt_directory
        return _set_global_prompt_directory
    
    raise AttributeError(f"Dotprompt lazy import: unknown attribute {name!r}")


def _lazy_import_type_items(name: str) -> Any:
    """Lazy import for type-related items - imports only the requested item by name."""
    if name == "COHERE_EMBEDDING_INPUT_TYPES":
        from litellm.types.llms.bedrock import COHERE_EMBEDDING_INPUT_TYPES as _COHERE_EMBEDDING_INPUT_TYPES
        globals()["COHERE_EMBEDDING_INPUT_TYPES"] = _COHERE_EMBEDDING_INPUT_TYPES
        return _COHERE_EMBEDDING_INPUT_TYPES
    
    if name == "GuardrailItem":
        from litellm.types.guardrails import GuardrailItem as _GuardrailItem
        globals()["GuardrailItem"] = _GuardrailItem
        return _GuardrailItem
    
    raise AttributeError(f"Type items lazy import: unknown attribute {name!r}")


def _lazy_import_core_helpers(name: str) -> Any:
    """Lazy import for core helper functions - imports only the requested item by name."""
    if name == "remove_index_from_tool_calls":
        from litellm.litellm_core_utils.core_helpers import remove_index_from_tool_calls as _remove_index_from_tool_calls
        globals()["remove_index_from_tool_calls"] = _remove_index_from_tool_calls
        return _remove_index_from_tool_calls
    
    raise AttributeError(f"Core helpers lazy import: unknown attribute {name!r}")


def _lazy_import_openai_like_configs(name: str) -> Any:
    """Lazy import for OpenAI-like config classes - imports only the requested class."""
    if name == "OpenAILikeChatConfig":
        from .llms.openai_like.chat.handler import OpenAILikeChatConfig as _OpenAILikeChatConfig
        globals()["OpenAILikeChatConfig"] = _OpenAILikeChatConfig
        return _OpenAILikeChatConfig
    
    if name == "AiohttpOpenAIChatConfig":
        from .llms.aiohttp_openai.chat.transformation import AiohttpOpenAIChatConfig as _AiohttpOpenAIChatConfig
        globals()["AiohttpOpenAIChatConfig"] = _AiohttpOpenAIChatConfig
        return _AiohttpOpenAIChatConfig
    
    raise AttributeError(f"OpenAI-like configs lazy import: unknown attribute {name!r}")


def _lazy_import_small_provider_chat_configs(name: str) -> Any:
    """Lazy import for smaller provider chat config classes - imports only the requested class."""
    if name == "GaladrielChatConfig":
        from .llms.galadriel.chat.transformation import GaladrielChatConfig as _GaladrielChatConfig
        globals()["GaladrielChatConfig"] = _GaladrielChatConfig
        return _GaladrielChatConfig
    
    if name == "GithubChatConfig":
        from .llms.github.chat.transformation import GithubChatConfig as _GithubChatConfig
        globals()["GithubChatConfig"] = _GithubChatConfig
        return _GithubChatConfig
    
    if name == "CompactifAIChatConfig":
        from .llms.compactifai.chat.transformation import CompactifAIChatConfig as _CompactifAIChatConfig
        globals()["CompactifAIChatConfig"] = _CompactifAIChatConfig
        return _CompactifAIChatConfig
    
    if name == "EmpowerChatConfig":
        from .llms.empower.chat.transformation import EmpowerChatConfig as _EmpowerChatConfig
        globals()["EmpowerChatConfig"] = _EmpowerChatConfig
        return _EmpowerChatConfig
    
    raise AttributeError(f"Small provider chat configs lazy import: unknown attribute {name!r}")


def _lazy_import_data_platform_configs(name: str) -> Any:
    """Lazy import for data platform provider chat config classes - imports only the requested class."""
    if name == "DatabricksConfig":
        from .llms.databricks.chat.transformation import DatabricksConfig as _DatabricksConfig
        globals()["DatabricksConfig"] = _DatabricksConfig
        return _DatabricksConfig
    
    if name == "PredibaseConfig":
        from .llms.predibase.chat.transformation import PredibaseConfig as _PredibaseConfig
        globals()["PredibaseConfig"] = _PredibaseConfig
        return _PredibaseConfig
    
    if name == "SnowflakeConfig":
        from .llms.snowflake.chat.transformation import SnowflakeConfig as _SnowflakeConfig
        globals()["SnowflakeConfig"] = _SnowflakeConfig
        return _SnowflakeConfig
    
    raise AttributeError(f"Data platform configs lazy import: unknown attribute {name!r}")


def _lazy_import_huggingface_configs(name: str) -> Any:
    """Lazy import for HuggingFace config classes - imports only the requested class."""
    if name == "HuggingFaceChatConfig":
        from .llms.huggingface.chat.transformation import HuggingFaceChatConfig as _HuggingFaceChatConfig
        globals()["HuggingFaceChatConfig"] = _HuggingFaceChatConfig
        return _HuggingFaceChatConfig
    
    if name == "HuggingFaceEmbeddingConfig":
        from .llms.huggingface.embedding.transformation import HuggingFaceEmbeddingConfig as _HuggingFaceEmbeddingConfig
        globals()["HuggingFaceEmbeddingConfig"] = _HuggingFaceEmbeddingConfig
        return _HuggingFaceEmbeddingConfig
    
    raise AttributeError(f"HuggingFace configs lazy import: unknown attribute {name!r}")


def _lazy_import_anthropic_configs(name: str) -> Any:
    """Lazy import for Anthropic config classes - imports only the requested class."""
    if name == "AnthropicConfig":
        from .llms.anthropic.chat.transformation import AnthropicConfig as _AnthropicConfig
        globals()["AnthropicConfig"] = _AnthropicConfig
        return _AnthropicConfig
    
    if name == "AnthropicTextConfig":
        from .llms.anthropic.completion.transformation import AnthropicTextConfig as _AnthropicTextConfig
        globals()["AnthropicTextConfig"] = _AnthropicTextConfig
        return _AnthropicTextConfig
    
    if name == "AnthropicMessagesConfig":
        from .llms.anthropic.experimental_pass_through.messages.transformation import AnthropicMessagesConfig as _AnthropicMessagesConfig
        globals()["AnthropicMessagesConfig"] = _AnthropicMessagesConfig
        return _AnthropicMessagesConfig
    
    raise AttributeError(f"Anthropic configs lazy import: unknown attribute {name!r}")


def _lazy_import_triton_configs(name: str) -> Any:
    """Lazy import for Triton config classes - imports only the requested class."""
    if name == "TritonConfig":
        from .llms.triton.completion.transformation import TritonConfig as _TritonConfig
        globals()["TritonConfig"] = _TritonConfig
        return _TritonConfig
    
    if name == "TritonEmbeddingConfig":
        from .llms.triton.embedding.transformation import TritonEmbeddingConfig as _TritonEmbeddingConfig
        globals()["TritonEmbeddingConfig"] = _TritonEmbeddingConfig
        return _TritonEmbeddingConfig
    
    raise AttributeError(f"Triton configs lazy import: unknown attribute {name!r}")


def _lazy_import_ai21_configs(name: str) -> Any:
    """Lazy import for AI21 config classes - imports only the requested class."""
    if name == "AI21ChatConfig":
        from .llms.ai21.chat.transformation import AI21ChatConfig as _AI21ChatConfig
        globals()["AI21ChatConfig"] = _AI21ChatConfig
        globals()["AI21Config"] = _AI21ChatConfig  # alias
        return _AI21ChatConfig
    
    if name == "AI21Config":
        from .llms.ai21.chat.transformation import AI21ChatConfig as _AI21ChatConfig
        globals()["AI21ChatConfig"] = _AI21ChatConfig
        globals()["AI21Config"] = _AI21ChatConfig  # alias
        return _AI21ChatConfig
    
    raise AttributeError(f"AI21 configs lazy import: unknown attribute {name!r}")


def _lazy_import_ollama_configs(name: str) -> Any:
    """Lazy import for Ollama config classes - imports only the requested class."""
    if name == "OllamaChatConfig":
        from .llms.ollama.chat.transformation import OllamaChatConfig as _OllamaChatConfig
        globals()["OllamaChatConfig"] = _OllamaChatConfig
        return _OllamaChatConfig
    
    if name == "OllamaConfig":
        from .llms.ollama.completion.transformation import OllamaConfig as _OllamaConfig
        globals()["OllamaConfig"] = _OllamaConfig
        return _OllamaConfig
    
    raise AttributeError(f"Ollama configs lazy import: unknown attribute {name!r}")


def _lazy_import_sagemaker_configs(name: str) -> Any:
    """Lazy import for Sagemaker config classes - imports only the requested class."""
    if name == "SagemakerConfig":
        from .llms.sagemaker.completion.transformation import SagemakerConfig as _SagemakerConfig
        globals()["SagemakerConfig"] = _SagemakerConfig
        return _SagemakerConfig
    
    if name == "SagemakerChatConfig":
        from .llms.sagemaker.chat.transformation import SagemakerChatConfig as _SagemakerChatConfig
        globals()["SagemakerChatConfig"] = _SagemakerChatConfig
        return _SagemakerChatConfig
    
    raise AttributeError(f"Sagemaker configs lazy import: unknown attribute {name!r}")


def _lazy_import_cohere_chat_configs(name: str) -> Any:
    """Lazy import for Cohere chat config classes - imports only the requested class."""
    if name == "CohereChatConfig":
        from .llms.cohere.chat.transformation import CohereChatConfig as _CohereChatConfig
        globals()["CohereChatConfig"] = _CohereChatConfig
        return _CohereChatConfig
    
    if name == "CohereV2ChatConfig":
        from .llms.cohere.chat.v2_transformation import CohereV2ChatConfig as _CohereV2ChatConfig
        globals()["CohereV2ChatConfig"] = _CohereV2ChatConfig
        return _CohereV2ChatConfig
    
    raise AttributeError(f"Cohere chat configs lazy import: unknown attribute {name!r}")


def _lazy_import_rerank_configs(name: str) -> Any:
    """Lazy import for rerank config classes - imports only the requested class."""
    if name == "HuggingFaceRerankConfig":
        from .llms.huggingface.rerank.transformation import HuggingFaceRerankConfig as _HuggingFaceRerankConfig
        globals()["HuggingFaceRerankConfig"] = _HuggingFaceRerankConfig
        return _HuggingFaceRerankConfig
    
    if name == "CohereRerankConfig":
        from .llms.cohere.rerank.transformation import CohereRerankConfig as _CohereRerankConfig
        globals()["CohereRerankConfig"] = _CohereRerankConfig
        return _CohereRerankConfig
    
    if name == "CohereRerankV2Config":
        from .llms.cohere.rerank_v2.transformation import CohereRerankV2Config as _CohereRerankV2Config
        globals()["CohereRerankV2Config"] = _CohereRerankV2Config
        return _CohereRerankV2Config
    
    if name == "AzureAIRerankConfig":
        from .llms.azure_ai.rerank.transformation import AzureAIRerankConfig as _AzureAIRerankConfig
        globals()["AzureAIRerankConfig"] = _AzureAIRerankConfig
        return _AzureAIRerankConfig
    
    if name == "InfinityRerankConfig":
        from .llms.infinity.rerank.transformation import InfinityRerankConfig as _InfinityRerankConfig
        globals()["InfinityRerankConfig"] = _InfinityRerankConfig
        return _InfinityRerankConfig
    
    if name == "JinaAIRerankConfig":
        from .llms.jina_ai.rerank.transformation import JinaAIRerankConfig as _JinaAIRerankConfig
        globals()["JinaAIRerankConfig"] = _JinaAIRerankConfig
        return _JinaAIRerankConfig
    
    if name == "DeepinfraRerankConfig":
        from .llms.deepinfra.rerank.transformation import DeepinfraRerankConfig as _DeepinfraRerankConfig
        globals()["DeepinfraRerankConfig"] = _DeepinfraRerankConfig
        return _DeepinfraRerankConfig
    
    if name == "HostedVLLMRerankConfig":
        from .llms.hosted_vllm.rerank.transformation import HostedVLLMRerankConfig as _HostedVLLMRerankConfig
        globals()["HostedVLLMRerankConfig"] = _HostedVLLMRerankConfig
        return _HostedVLLMRerankConfig
    
    if name == "NvidiaNimRerankConfig":
        from .llms.nvidia_nim.rerank.transformation import NvidiaNimRerankConfig as _NvidiaNimRerankConfig
        globals()["NvidiaNimRerankConfig"] = _NvidiaNimRerankConfig
        return _NvidiaNimRerankConfig
    
    if name == "VertexAIRerankConfig":
        from .llms.vertex_ai.rerank.transformation import VertexAIRerankConfig as _VertexAIRerankConfig
        globals()["VertexAIRerankConfig"] = _VertexAIRerankConfig
        return _VertexAIRerankConfig
    
    raise AttributeError(f"Rerank configs lazy import: unknown attribute {name!r}")


def _lazy_import_vertex_ai_configs(name: str) -> Any:
    """Lazy import for Vertex AI config classes - imports only the requested class."""
    if name == "VertexGeminiConfig":
        from .llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import VertexGeminiConfig as _VertexGeminiConfig
        globals()["VertexGeminiConfig"] = _VertexGeminiConfig
        globals()["VertexAIConfig"] = _VertexGeminiConfig  # alias
        return _VertexGeminiConfig
    
    if name == "VertexAIConfig":
        from .llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import VertexGeminiConfig as _VertexGeminiConfig
        globals()["VertexGeminiConfig"] = _VertexGeminiConfig
        globals()["VertexAIConfig"] = _VertexGeminiConfig  # alias
        return _VertexGeminiConfig
    
    if name == "GoogleAIStudioGeminiConfig":
        from .llms.gemini.chat.transformation import GoogleAIStudioGeminiConfig as _GoogleAIStudioGeminiConfig
        globals()["GoogleAIStudioGeminiConfig"] = _GoogleAIStudioGeminiConfig
        globals()["GeminiConfig"] = _GoogleAIStudioGeminiConfig  # alias
        return _GoogleAIStudioGeminiConfig
    
    if name == "GeminiConfig":
        from .llms.gemini.chat.transformation import GoogleAIStudioGeminiConfig as _GoogleAIStudioGeminiConfig
        globals()["GoogleAIStudioGeminiConfig"] = _GoogleAIStudioGeminiConfig
        globals()["GeminiConfig"] = _GoogleAIStudioGeminiConfig  # alias
        return _GoogleAIStudioGeminiConfig
    
    if name == "VertexAIAnthropicConfig":
        from .llms.vertex_ai.vertex_ai_partner_models.anthropic.transformation import VertexAIAnthropicConfig as _VertexAIAnthropicConfig
        globals()["VertexAIAnthropicConfig"] = _VertexAIAnthropicConfig
        return _VertexAIAnthropicConfig
    
    if name == "VertexAILlama3Config":
        from .llms.vertex_ai.vertex_ai_partner_models.llama3.transformation import VertexAILlama3Config as _VertexAILlama3Config
        globals()["VertexAILlama3Config"] = _VertexAILlama3Config
        return _VertexAILlama3Config
    
    if name == "VertexAIAi21Config":
        from .llms.vertex_ai.vertex_ai_partner_models.ai21.transformation import VertexAIAi21Config as _VertexAIAi21Config
        globals()["VertexAIAi21Config"] = _VertexAIAi21Config
        return _VertexAIAi21Config
    
    raise AttributeError(f"Vertex AI configs lazy import: unknown attribute {name!r}")


def _lazy_import_amazon_bedrock_configs(name: str) -> Any:
    """Lazy import for Amazon Bedrock config classes - imports only the requested class."""
    if name == "AmazonCohereChatConfig":
        from .llms.bedrock.chat.invoke_handler import AmazonCohereChatConfig as _AmazonCohereChatConfig
        globals()["AmazonCohereChatConfig"] = _AmazonCohereChatConfig
        return _AmazonCohereChatConfig
    
    if name == "AmazonBedrockGlobalConfig":
        from .llms.bedrock.common_utils import AmazonBedrockGlobalConfig as _AmazonBedrockGlobalConfig
        globals()["AmazonBedrockGlobalConfig"] = _AmazonBedrockGlobalConfig
        return _AmazonBedrockGlobalConfig
    
    if name == "AmazonAI21Config":
        from .llms.bedrock.chat.invoke_transformations.amazon_ai21_transformation import AmazonAI21Config as _AmazonAI21Config
        globals()["AmazonAI21Config"] = _AmazonAI21Config
        return _AmazonAI21Config
    
    if name == "AmazonAnthropicConfig":
        from .llms.bedrock.chat.invoke_transformations.anthropic_claude2_transformation import AmazonAnthropicConfig as _AmazonAnthropicConfig
        globals()["AmazonAnthropicConfig"] = _AmazonAnthropicConfig
        return _AmazonAnthropicConfig
    
    if name == "AmazonAnthropicClaudeConfig":
        from .llms.bedrock.chat.invoke_transformations.anthropic_claude3_transformation import AmazonAnthropicClaudeConfig as _AmazonAnthropicClaudeConfig
        globals()["AmazonAnthropicClaudeConfig"] = _AmazonAnthropicClaudeConfig
        return _AmazonAnthropicClaudeConfig
    
    if name == "AmazonTitanG1Config":
        from .llms.bedrock.embed.amazon_titan_g1_transformation import AmazonTitanG1Config as _AmazonTitanG1Config
        globals()["AmazonTitanG1Config"] = _AmazonTitanG1Config
        return _AmazonTitanG1Config
    
    if name == "AmazonTitanMultimodalEmbeddingG1Config":
        from .llms.bedrock.embed.amazon_titan_multimodal_transformation import AmazonTitanMultimodalEmbeddingG1Config as _AmazonTitanMultimodalEmbeddingG1Config
        globals()["AmazonTitanMultimodalEmbeddingG1Config"] = _AmazonTitanMultimodalEmbeddingG1Config
        return _AmazonTitanMultimodalEmbeddingG1Config
    
    if name == "AmazonTitanV2Config":
        from .llms.bedrock.embed.amazon_titan_v2_transformation import AmazonTitanV2Config as _AmazonTitanV2Config
        globals()["AmazonTitanV2Config"] = _AmazonTitanV2Config
        return _AmazonTitanV2Config
    
    if name == "BedrockCohereEmbeddingConfig":
        from .llms.bedrock.embed.cohere_transformation import BedrockCohereEmbeddingConfig as _BedrockCohereEmbeddingConfig
        globals()["BedrockCohereEmbeddingConfig"] = _BedrockCohereEmbeddingConfig
        return _BedrockCohereEmbeddingConfig
    
    raise AttributeError(f"Amazon Bedrock configs lazy import: unknown attribute {name!r}")


def _lazy_import_deprecated_provider_configs(name: str) -> Any:
    """Lazy import for deprecated provider config classes - imports only the requested class."""
    if name == "PalmConfig":
        from .llms.deprecated_providers.palm import PalmConfig as _PalmConfig
        globals()["PalmConfig"] = _PalmConfig
        return _PalmConfig
    
    if name == "AlephAlphaConfig":
        from .llms.deprecated_providers.aleph_alpha import AlephAlphaConfig as _AlephAlphaConfig
        globals()["AlephAlphaConfig"] = _AlephAlphaConfig
        return _AlephAlphaConfig
    
    raise AttributeError(f"Deprecated provider configs lazy import: unknown attribute {name!r}")


def _lazy_import_azure_responses_configs(name: str) -> Any:
    """Lazy import for Azure OpenAI Responses API config classes - imports only the requested class."""
    if name == "AzureOpenAIResponsesAPIConfig":
        from .llms.azure.responses.transformation import AzureOpenAIResponsesAPIConfig as _AzureOpenAIResponsesAPIConfig
        globals()["AzureOpenAIResponsesAPIConfig"] = _AzureOpenAIResponsesAPIConfig
        return _AzureOpenAIResponsesAPIConfig
    
    if name == "AzureOpenAIOSeriesResponsesAPIConfig":
        from .llms.azure.responses.o_series_transformation import AzureOpenAIOSeriesResponsesAPIConfig as _AzureOpenAIOSeriesResponsesAPIConfig
        globals()["AzureOpenAIOSeriesResponsesAPIConfig"] = _AzureOpenAIOSeriesResponsesAPIConfig
        return _AzureOpenAIOSeriesResponsesAPIConfig
    
    raise AttributeError(f"Azure Responses API configs lazy import: unknown attribute {name!r}")


def _lazy_import_openai_o_series_configs(name: str) -> Any:
    """Lazy import for OpenAI O-Series config classes - imports only the requested class."""
    if name == "OpenAIOSeriesConfig":
        from .llms.openai.chat.o_series_transformation import OpenAIOSeriesConfig as _OpenAIOSeriesConfig
        globals()["OpenAIOSeriesConfig"] = _OpenAIOSeriesConfig
        return _OpenAIOSeriesConfig
    
    if name == "OpenAIO1Config":
        from .llms.openai.chat.o_series_transformation import OpenAIOSeriesConfig as _OpenAIOSeriesConfig
        globals()["OpenAIOSeriesConfig"] = _OpenAIOSeriesConfig
        globals()["OpenAIO1Config"] = _OpenAIOSeriesConfig  # alias
        return _OpenAIOSeriesConfig
    
    if name == "openaiOSeriesConfig":
        from .llms.openai.chat.o_series_transformation import OpenAIOSeriesConfig as _OpenAIOSeriesConfig
        _openaiOSeriesConfig = _OpenAIOSeriesConfig()
        globals()["OpenAIOSeriesConfig"] = _OpenAIOSeriesConfig
        globals()["openaiOSeriesConfig"] = _openaiOSeriesConfig
        return _openaiOSeriesConfig
    
    raise AttributeError(f"OpenAI O-Series configs lazy import: unknown attribute {name!r}")


def _lazy_import_openai_gpt_configs(name: str) -> Any:
    """Lazy import for OpenAI GPT config classes - imports only the requested class."""
    if name == "OpenAIGPTConfig":
        from .llms.openai.chat.gpt_transformation import OpenAIGPTConfig as _OpenAIGPTConfig
        globals()["OpenAIGPTConfig"] = _OpenAIGPTConfig
        return _OpenAIGPTConfig
    
    if name == "openAIGPTConfig":
        from .llms.openai.chat.gpt_transformation import OpenAIGPTConfig as _OpenAIGPTConfig
        _openAIGPTConfig = _OpenAIGPTConfig()
        globals()["OpenAIGPTConfig"] = _OpenAIGPTConfig
        globals()["openAIGPTConfig"] = _openAIGPTConfig
        return _openAIGPTConfig
    
    if name == "OpenAIGPT5Config":
        from .llms.openai.chat.gpt_5_transformation import OpenAIGPT5Config as _OpenAIGPT5Config
        globals()["OpenAIGPT5Config"] = _OpenAIGPT5Config
        return _OpenAIGPT5Config
    
    if name == "openAIGPT5Config":
        from .llms.openai.chat.gpt_5_transformation import OpenAIGPT5Config as _OpenAIGPT5Config
        _openAIGPT5Config = _OpenAIGPT5Config()
        globals()["OpenAIGPT5Config"] = _OpenAIGPT5Config
        globals()["openAIGPT5Config"] = _openAIGPT5Config
        return _openAIGPT5Config
    
    if name == "OpenAIGPTAudioConfig":
        from .llms.openai.chat.gpt_audio_transformation import OpenAIGPTAudioConfig as _OpenAIGPTAudioConfig
        globals()["OpenAIGPTAudioConfig"] = _OpenAIGPTAudioConfig
        return _OpenAIGPTAudioConfig
    
    if name == "openAIGPTAudioConfig":
        from .llms.openai.chat.gpt_audio_transformation import OpenAIGPTAudioConfig as _OpenAIGPTAudioConfig
        _openAIGPTAudioConfig = _OpenAIGPTAudioConfig()
        globals()["OpenAIGPTAudioConfig"] = _OpenAIGPTAudioConfig
        globals()["openAIGPTAudioConfig"] = _openAIGPTAudioConfig
        return _openAIGPTAudioConfig
    
    raise AttributeError(f"OpenAI GPT configs lazy import: unknown attribute {name!r}")


def _lazy_import_nvidia_nim_configs(name: str) -> Any:
    """Lazy import for NvidiaNim config classes - imports only the requested class."""
    if name == "NvidiaNimConfig":
        from .llms.nvidia_nim.chat.transformation import NvidiaNimConfig as _NvidiaNimConfig
        globals()["NvidiaNimConfig"] = _NvidiaNimConfig
        return _NvidiaNimConfig
    
    if name == "nvidiaNimConfig":
        from .llms.nvidia_nim.chat.transformation import NvidiaNimConfig as _NvidiaNimConfig
        _nvidiaNimConfig = _NvidiaNimConfig()
        globals()["NvidiaNimConfig"] = _NvidiaNimConfig
        globals()["nvidiaNimConfig"] = _nvidiaNimConfig
        return _nvidiaNimConfig
    
    raise AttributeError(f"NvidiaNim configs lazy import: unknown attribute {name!r}")


def _lazy_import_misc_transformation_configs(name: str) -> Any:
    """Lazy import for miscellaneous transformation config classes - imports only the requested class."""
    if name == "DeepInfraConfig":
        from .llms.deepinfra.chat.transformation import DeepInfraConfig as _DeepInfraConfig
        globals()["DeepInfraConfig"] = _DeepInfraConfig
        return _DeepInfraConfig
    
    if name == "GroqChatConfig":
        from .llms.groq.chat.transformation import GroqChatConfig as _GroqChatConfig
        globals()["GroqChatConfig"] = _GroqChatConfig
        return _GroqChatConfig
    
    if name == "VoyageEmbeddingConfig":
        from .llms.voyage.embedding.transformation import VoyageEmbeddingConfig as _VoyageEmbeddingConfig
        globals()["VoyageEmbeddingConfig"] = _VoyageEmbeddingConfig
        return _VoyageEmbeddingConfig
    
    if name == "InfinityEmbeddingConfig":
        from .llms.infinity.embedding.transformation import InfinityEmbeddingConfig as _InfinityEmbeddingConfig
        globals()["InfinityEmbeddingConfig"] = _InfinityEmbeddingConfig
        return _InfinityEmbeddingConfig
    
    if name == "AzureAIStudioConfig":
        from .llms.azure_ai.chat.transformation import AzureAIStudioConfig as _AzureAIStudioConfig
        globals()["AzureAIStudioConfig"] = _AzureAIStudioConfig
        return _AzureAIStudioConfig
    
    if name == "MistralConfig":
        from .llms.mistral.chat.transformation import MistralConfig as _MistralConfig
        globals()["MistralConfig"] = _MistralConfig
        return _MistralConfig
    
    raise AttributeError(f"Misc transformation configs lazy import: unknown attribute {name!r}")


def __getattr__(name: str) -> Any:
    """Lazy import for cost_calculator, litellm_logging, and utils functions."""
    if name in {"completion_cost", "response_cost_calculator", "cost_per_token"}:
        from ._lazy_imports import _lazy_import_cost_calculator
        return _lazy_import_cost_calculator(name)
    
    if name in {"Logging", "modify_integration"}:
        from ._lazy_imports import _lazy_import_litellm_logging
        return _lazy_import_litellm_logging(name)
    
    # Lazy load utils functions
    _utils_names = {
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
    }
    if name in _utils_names:
        from ._lazy_imports import _lazy_import_utils
        return _lazy_import_utils(name)
    
    # Lazy-load encoding to avoid loading tiktoken at import time
    if name == "encoding":
        from litellm.litellm_core_utils.default_encoding import encoding as _encoding
        globals()["encoding"] = _encoding
        return _encoding
    
    # Lazy-load HTTP handlers to reduce import-time memory cost
    if name in {"module_level_aclient", "module_level_client", "AsyncHTTPHandler", "HTTPHandler"}:
        return _lazy_import_http_handlers(name)
    
    # Lazy-load caching classes to reduce import-time memory cost
    if name in {"Cache", "DualCache", "RedisCache", "InMemoryCache", "LLMClientCache"}:
        return _lazy_import_caching(name)
    
    # Lazy-load types.utils to reduce import-time memory cost
    _types_utils_names = {
        "ImageObject", "BudgetConfig", "all_litellm_params", "_litellm_completion_params",
        "CredentialItem", "PriorityReservationDict", "StandardKeyGenerationConfig",
        "LlmProviders", "SearchProviders", "PriorityReservationSettings",
    }
    if name in _types_utils_names:
        return _lazy_import_types_utils(name)
    
    if name in {"DefaultTeamSSOParams", "LiteLLM_UpperboundKeyGenerateParams"}:
        return _lazy_import_ui_sso(name)
    
    if name == "KeyManagementSystem":
        return _lazy_import_secret_managers(name)
    
    if name == "provider_list":
        provider_list_val = list(LlmProviders)
        globals()["provider_list"] = provider_list_val
        return provider_list_val
    
    if name == "priority_reservation_settings":
        prs_val = PriorityReservationSettings()
        globals()["priority_reservation_settings"] = prs_val
        return prs_val
    
    # Lazy-load logging integrations to avoid circular imports
    if name in {"CustomLogger", "LoggingCallbackManager"}:
        return _lazy_import_logging_integrations(name)
    
    # Lazy-load dotprompt imports to avoid circular imports
    if name in {"global_prompt_manager", "global_prompt_directory", "set_global_prompt_directory"}:
        return _lazy_import_dotprompt(name)
    
    # Lazy-load type-related items to reduce import-time memory cost
    if name in {"COHERE_EMBEDDING_INPUT_TYPES", "GuardrailItem"}:
        return _lazy_import_type_items(name)
    
    # Lazy-load core helpers to reduce import-time memory cost
    if name == "remove_index_from_tool_calls":
        return _lazy_import_core_helpers(name)
    
    # Lazy-load BytezChatConfig to reduce import-time memory cost
    if name == "BytezChatConfig":
        from .llms.bytez.chat.transformation import BytezChatConfig as _BytezChatConfig
        globals()["BytezChatConfig"] = _BytezChatConfig
        return _BytezChatConfig
    
    # Lazy-load CustomLLM to reduce import-time memory cost
    if name == "CustomLLM":
        from .llms.custom_llm import CustomLLM as _CustomLLM
        globals()["CustomLLM"] = _CustomLLM
        return _CustomLLM
    
    # Lazy-load AmazonConverseConfig to reduce import-time memory cost
    if name == "AmazonConverseConfig":
        from .llms.bedrock.chat.converse_transformation import AmazonConverseConfig as _AmazonConverseConfig
        globals()["AmazonConverseConfig"] = _AmazonConverseConfig
        return _AmazonConverseConfig
    
    # Lazy-load OpenAI-like configs to reduce import-time memory cost
    if name in {"OpenAILikeChatConfig", "AiohttpOpenAIChatConfig"}:
        return _lazy_import_openai_like_configs(name)
    
    # Lazy-load small provider chat configs to reduce import-time memory cost
    if name in {"GaladrielChatConfig", "GithubChatConfig", "CompactifAIChatConfig", "EmpowerChatConfig"}:
        return _lazy_import_small_provider_chat_configs(name)
    
    # Lazy-load HuggingFace configs to reduce import-time memory cost
    if name in {"HuggingFaceChatConfig", "HuggingFaceEmbeddingConfig"}:
        return _lazy_import_huggingface_configs(name)
    
    # Lazy-load OpenrouterConfig to reduce import-time memory cost
    if name == "OpenrouterConfig":
        from .llms.openrouter.chat.transformation import OpenrouterConfig as _OpenrouterConfig
        globals()["OpenrouterConfig"] = _OpenrouterConfig
        return _OpenrouterConfig
    
    # Lazy-load Anthropic configs to reduce import-time memory cost
    if name in {"AnthropicConfig", "AnthropicTextConfig", "AnthropicMessagesConfig"}:
        return _lazy_import_anthropic_configs(name)
    
    # Lazy-load data platform configs to reduce import-time memory cost
    if name in {"DatabricksConfig", "PredibaseConfig", "SnowflakeConfig"}:
        return _lazy_import_data_platform_configs(name)
    
    # Lazy-load ReplicateConfig to reduce import-time memory cost
    if name == "ReplicateConfig":
        from .llms.replicate.chat.transformation import ReplicateConfig as _ReplicateConfig
        globals()["ReplicateConfig"] = _ReplicateConfig
        return _ReplicateConfig
    
    # Lazy-load OobaboogaConfig to reduce import-time memory cost
    if name == "OobaboogaConfig":
        from .llms.oobabooga.chat.transformation import OobaboogaConfig as _OobaboogaConfig
        globals()["OobaboogaConfig"] = _OobaboogaConfig
        return _OobaboogaConfig
    
    # Lazy-load MaritalkConfig to reduce import-time memory cost
    if name == "MaritalkConfig":
        from .llms.maritalk import MaritalkConfig as _MaritalkConfig
        globals()["MaritalkConfig"] = _MaritalkConfig
        return _MaritalkConfig
    
    # Lazy-load DataRobotConfig to reduce import-time memory cost
    if name == "DataRobotConfig":
        from .llms.datarobot.chat.transformation import DataRobotConfig as _DataRobotConfig
        globals()["DataRobotConfig"] = _DataRobotConfig
        return _DataRobotConfig
    
    # Lazy-load GroqSTTConfig to reduce import-time memory cost
    if name == "GroqSTTConfig":
        from .llms.groq.stt.transformation import GroqSTTConfig as _GroqSTTConfig
        globals()["GroqSTTConfig"] = _GroqSTTConfig
        return _GroqSTTConfig
    
    # Lazy-load Triton configs to reduce import-time memory cost
    if name in {"TritonConfig", "TritonEmbeddingConfig"}:
        return _lazy_import_triton_configs(name)
    
    # Lazy-load ClarifaiConfig to reduce import-time memory cost
    if name == "ClarifaiConfig":
        from .llms.clarifai.chat.transformation import ClarifaiConfig as _ClarifaiConfig
        globals()["ClarifaiConfig"] = _ClarifaiConfig
        return _ClarifaiConfig
    
    # Lazy-load AI21 configs to reduce import-time memory cost
    if name in {"AI21ChatConfig", "AI21Config"}:
        return _lazy_import_ai21_configs(name)
    
    # Lazy-load LlamaAPIConfig to reduce import-time memory cost
    if name == "LlamaAPIConfig":
        from .llms.meta_llama.chat.transformation import LlamaAPIConfig as _LlamaAPIConfig
        globals()["LlamaAPIConfig"] = _LlamaAPIConfig
        return _LlamaAPIConfig
    
    # Lazy-load TogetherAIConfig to reduce import-time memory cost
    if name == "TogetherAIConfig":
        from .llms.together_ai.chat import TogetherAIConfig as _TogetherAIConfig
        globals()["TogetherAIConfig"] = _TogetherAIConfig
        return _TogetherAIConfig
    
    # Lazy-load CloudflareChatConfig to reduce import-time memory cost
    if name == "CloudflareChatConfig":
        from .llms.cloudflare.chat.transformation import CloudflareChatConfig as _CloudflareChatConfig
        globals()["CloudflareChatConfig"] = _CloudflareChatConfig
        return _CloudflareChatConfig
    
    # Lazy-load NovitaConfig to reduce import-time memory cost
    if name == "NovitaConfig":
        from .llms.novita.chat.transformation import NovitaConfig as _NovitaConfig
        globals()["NovitaConfig"] = _NovitaConfig
        return _NovitaConfig
    
    # Lazy-load NLPCloudConfig to reduce import-time memory cost
    if name == "NLPCloudConfig":
        from .llms.nlp_cloud.chat.handler import NLPCloudConfig as _NLPCloudConfig
        globals()["NLPCloudConfig"] = _NLPCloudConfig
        return _NLPCloudConfig
    
    # Lazy-load PetalsConfig to reduce import-time memory cost
    if name == "PetalsConfig":
        from .llms.petals.completion.transformation import PetalsConfig as _PetalsConfig
        globals()["PetalsConfig"] = _PetalsConfig
        return _PetalsConfig
    
    # Lazy-load Ollama configs to reduce import-time memory cost
    if name in {"OllamaChatConfig", "OllamaConfig"}:
        return _lazy_import_ollama_configs(name)
    
    # Lazy-load Sagemaker configs to reduce import-time memory cost
    if name in {"SagemakerConfig", "SagemakerChatConfig"}:
        return _lazy_import_sagemaker_configs(name)
    
    # Lazy-load Cohere chat configs to reduce import-time memory cost
    if name in {"CohereChatConfig", "CohereV2ChatConfig"}:
        return _lazy_import_cohere_chat_configs(name)
    
    # Lazy-load OpenAIConfig to reduce import-time memory cost
    if name == "OpenAIConfig":
        from .llms.openai.openai import OpenAIConfig as _OpenAIConfig
        globals()["OpenAIConfig"] = _OpenAIConfig
        return _OpenAIConfig
    
    # Lazy-load miscellaneous transformation configs to reduce import-time memory cost
    _misc_transformation_config_names = {
        "DeepInfraConfig", "GroqChatConfig", "VoyageEmbeddingConfig",
        "InfinityEmbeddingConfig", "AzureAIStudioConfig", "MistralConfig",
    }
    if name in _misc_transformation_config_names:
        return _lazy_import_misc_transformation_configs(name)
    
    # Lazy-load rerank configs to reduce import-time memory cost
    _rerank_config_names = {
        "HuggingFaceRerankConfig", "CohereRerankConfig", "CohereRerankV2Config",
        "AzureAIRerankConfig", "InfinityRerankConfig", "JinaAIRerankConfig",
        "DeepinfraRerankConfig", "HostedVLLMRerankConfig", "NvidiaNimRerankConfig",
        "VertexAIRerankConfig",
    }
    if name in _rerank_config_names:
        return _lazy_import_rerank_configs(name)
    
    # Lazy-load TogetherAITextCompletionConfig to reduce import-time memory cost
    if name == "TogetherAITextCompletionConfig":
        from .llms.together_ai.completion.transformation import TogetherAITextCompletionConfig as _TogetherAITextCompletionConfig
        globals()["TogetherAITextCompletionConfig"] = _TogetherAITextCompletionConfig
        return _TogetherAITextCompletionConfig
    
    # Lazy-load Vertex AI configs to reduce import-time memory cost
    _vertex_ai_config_names = {
        "VertexGeminiConfig", "VertexAIConfig", "GoogleAIStudioGeminiConfig",
        "GeminiConfig", "VertexAIAnthropicConfig", "VertexAILlama3Config",
        "VertexAIAi21Config",
    }
    if name in _vertex_ai_config_names:
        return _lazy_import_vertex_ai_configs(name)
    
    # Lazy-load Amazon Bedrock configs to reduce import-time memory cost
    _amazon_bedrock_config_names = {
        "AmazonCohereChatConfig", "AmazonBedrockGlobalConfig", "AmazonAI21Config",
        "AmazonAnthropicConfig", "AmazonAnthropicClaudeConfig", "AmazonTitanG1Config",
        "AmazonTitanMultimodalEmbeddingG1Config", "AmazonTitanV2Config",
        "BedrockCohereEmbeddingConfig",
    }
    if name in _amazon_bedrock_config_names:
        return _lazy_import_amazon_bedrock_configs(name)
    
    # Lazy-load AnthropicModelInfo to reduce import-time memory cost
    if name == "AnthropicModelInfo":
        from .llms.anthropic.common_utils import AnthropicModelInfo as _AnthropicModelInfo
        globals()["AnthropicModelInfo"] = _AnthropicModelInfo
        return _AnthropicModelInfo
    
    # Lazy-load deprecated provider configs to reduce import-time memory cost
    if name in {"PalmConfig", "AlephAlphaConfig"}:
        return _lazy_import_deprecated_provider_configs(name)
    
    # Lazy-load bedrock_tool_name_mappings to reduce import-time memory cost
    if name == "bedrock_tool_name_mappings":
        from .llms.bedrock.chat.invoke_handler import bedrock_tool_name_mappings as _bedrock_tool_name_mappings
        globals()["bedrock_tool_name_mappings"] = _bedrock_tool_name_mappings
        return _bedrock_tool_name_mappings
    
    # Lazy-load AmazonInvokeConfig to reduce import-time memory cost
    if name == "AmazonInvokeConfig":
        from .llms.bedrock.chat.invoke_transformations.base_invoke_transformation import AmazonInvokeConfig as _AmazonInvokeConfig
        globals()["AmazonInvokeConfig"] = _AmazonInvokeConfig
        return _AmazonInvokeConfig
    
    # Lazy-load MistralEmbeddingConfig to reduce import-time memory cost
    if name == "MistralEmbeddingConfig":
        from .llms.openai.openai import MistralEmbeddingConfig as _MistralEmbeddingConfig
        globals()["MistralEmbeddingConfig"] = _MistralEmbeddingConfig
        return _MistralEmbeddingConfig
    
    # Lazy-load OpenAITextCompletionConfig to reduce import-time memory cost
    if name == "OpenAITextCompletionConfig":
        from .llms.openai.completion.transformation import OpenAITextCompletionConfig as _OpenAITextCompletionConfig
        globals()["OpenAITextCompletionConfig"] = _OpenAITextCompletionConfig
        return _OpenAITextCompletionConfig
    
    # Lazy-load VoyageContextualEmbeddingConfig to reduce import-time memory cost
    if name == "VoyageContextualEmbeddingConfig":
        from .llms.voyage.embedding.transformation_contextual import VoyageContextualEmbeddingConfig as _VoyageContextualEmbeddingConfig
        globals()["VoyageContextualEmbeddingConfig"] = _VoyageContextualEmbeddingConfig
        return _VoyageContextualEmbeddingConfig
    
    # Lazy-load Azure Responses API configs to reduce import-time memory cost
    if name in {"AzureOpenAIResponsesAPIConfig", "AzureOpenAIOSeriesResponsesAPIConfig"}:
        return _lazy_import_azure_responses_configs(name)
    
    # Lazy-load OpenAI O-Series configs to reduce import-time memory cost
    if name in {"OpenAIOSeriesConfig", "OpenAIO1Config", "openaiOSeriesConfig"}:
        return _lazy_import_openai_o_series_configs(name)
    
    # Lazy-load AzureOpenAIO1Config to reduce import-time memory cost
    if name == "AzureOpenAIO1Config":
        from .llms.azure.chat.o_series_transformation import AzureOpenAIO1Config as _AzureOpenAIO1Config
        globals()["AzureOpenAIO1Config"] = _AzureOpenAIO1Config
        return _AzureOpenAIO1Config
    
    # Lazy-load GradientAIConfig to reduce import-time memory cost
    if name == "GradientAIConfig":
        from .llms.gradient_ai.chat.transformation import GradientAIConfig as _GradientAIConfig
        globals()["GradientAIConfig"] = _GradientAIConfig
        return _GradientAIConfig
    
    # Lazy-load OpenAI GPT configs to reduce import-time memory cost
    _openai_gpt_config_names = {
        "OpenAIGPTConfig", "openAIGPTConfig", "OpenAIGPT5Config",
        "openAIGPT5Config", "OpenAIGPTAudioConfig", "openAIGPTAudioConfig",
    }
    if name in _openai_gpt_config_names:
        return _lazy_import_openai_gpt_configs(name)
    
    # Lazy-load NvidiaNim configs to reduce import-time memory cost
    if name in {"NvidiaNimConfig", "nvidiaNimConfig"}:
        return _lazy_import_nvidia_nim_configs(name)
    
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
