### Hide pydantic namespace conflict warnings globally ###
import warnings

warnings.filterwarnings("ignore", message=".*conflict with protected namespace.*")
### INIT VARIABLES ####################
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
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.caching.caching import Cache, DualCache, RedisCache, InMemoryCache
from litellm.caching.llm_caching_handler import LLMClientCache
from litellm.types.llms.bedrock import COHERE_EMBEDDING_INPUT_TYPES
from litellm.types.utils import (
    ImageObject,
    BudgetConfig,
    all_litellm_params,
    all_litellm_params as _litellm_completion_params,
    CredentialItem,
)  # maintain backwards compatibility for root param
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
from litellm.integrations.dotprompt import (
    global_prompt_manager,
    global_prompt_directory,
    set_global_prompt_directory,
)
from litellm.types.guardrails import GuardrailItem
from litellm.types.secret_managers.main import (
    KeyManagementSystem,
    KeyManagementSettings,
)
from litellm.types.proxy.management_endpoints.ui_sso import (
    DefaultTeamSSOParams,
    LiteLLM_UpperboundKeyGenerateParams,
)
from litellm.types.utils import StandardKeyGenerationConfig, LlmProviders
from litellm.types.utils import PriorityReservationSettings
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.logging_callback_manager import LoggingCallbackManager
import httpx
import dotenv
from litellm.llms.custom_httpx.async_client_cleanup import register_async_client_cleanup

litellm_mode = os.getenv("LITELLM_MODE", "DEV")  # "PRODUCTION", "DEV"
if litellm_mode == "DEV":
    dotenv.load_dotenv()

# Register async client cleanup to prevent resource leaks
register_async_client_cleanup()
####################################################
if set_verbose == True:
    _turn_on_debug()
####################################################
### Callbacks /Logging / Success / Failure Handlers #####
CALLBACK_TYPES = Union[str, Callable, CustomLogger]
input_callback: List[CALLBACK_TYPES] = []
success_callback: List[CALLBACK_TYPES] = []
failure_callback: List[CALLBACK_TYPES] = []
service_callback: List[CALLBACK_TYPES] = []
logging_callback_manager = LoggingCallbackManager()
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
    "cloudzero",
    "posthog",
]
configured_cold_storage_logger: Optional[
    _custom_logger_compatible_callbacks_literal
] = None
logged_real_time_event_types: Optional[Union[List[str], Literal["*"]]] = None
_known_custom_logger_compatible_callbacks: List = list(
    get_args(_custom_logger_compatible_callbacks_literal)
)
callbacks: List[
    Union[Callable, _custom_logger_compatible_callbacks_literal, CustomLogger]
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
_async_input_callback: List[Union[str, Callable, CustomLogger]] = (
    []
)  # internal variable - async custom callbacks are routed here.
_async_success_callback: List[Union[str, Callable, CustomLogger]] = (
    []
)  # internal variable - async custom callbacks are routed here.
_async_failure_callback: List[Union[str, Callable, CustomLogger]] = (
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
disable_streaming_logging: bool = False
disable_token_counter: bool = False
disable_add_transform_inline_image_block: bool = False
disable_add_user_agent_to_request_tags: bool = False
extra_spend_tag_headers: Optional[List[str]] = None
in_memory_llm_clients_cache: LLMClientCache = LLMClientCache()
safe_memory_mode: bool = False
enable_azure_ad_token_refresh: Optional[bool] = False
### DEFAULT AZURE API VERSION ###
AZURE_DEFAULT_API_VERSION = "2025-02-01-preview"  # this is updated to the latest
### DEFAULT WATSONX API VERSION ###
WATSONX_DEFAULT_API_VERSION = "2024-03-13"
### COHERE EMBEDDINGS DEFAULT TYPE ###
COHERE_DEFAULT_EMBEDDING_INPUT_TYPE: COHERE_EMBEDDING_INPUT_TYPES = "search_document"
### CREDENTIALS ###
credential_list: List[CredentialItem] = []
### GUARDRAILS ###
llamaguard_model_name: Optional[str] = None
openai_moderations_model_name: Optional[str] = None
presidio_ad_hoc_recognizers: Optional[str] = None
google_moderation_confidence_threshold: Optional[float] = None
llamaguard_unsafe_content_categories: Optional[str] = None
blocked_user_list: Optional[Union[str, List]] = None
banned_keywords_list: Optional[Union[str, List]] = None
llm_guard_mode: Literal["all", "key-specific", "request-specific"] = "all"
guardrail_name_config_map: Dict[str, GuardrailItem] = {}
include_cost_in_streaming_usage: bool = False
### PROMPTS ###
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
cache: Optional[Cache] = (
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
model_cost_map_url: str = (
    "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"
)
suppress_debug_info = False
dynamodb_table_name: Optional[str] = None
s3_callback_params: Optional[Dict] = None
datadog_llm_observability_params: Optional[Union[DatadogLLMObsInitParams, Dict]] = None
datadog_params: Optional[Union[DatadogInitParams, Dict]] = None
aws_sqs_callback_params: Optional[Dict] = None
generic_logger_headers: Optional[Dict] = None
default_key_generate_params: Optional[Dict] = None
upperbound_key_generate_params: Optional[LiteLLM_UpperboundKeyGenerateParams] = None
key_generation_settings: Optional[StandardKeyGenerationConfig] = None
default_internal_user_params: Optional[Dict] = None
default_team_params: Optional[Union[DefaultTeamSSOParams, Dict]] = None
default_team_settings: Optional[List] = None
max_user_budget: Optional[float] = None
default_max_internal_user_budget: Optional[float] = None
max_internal_user_budget: Optional[float] = None
max_ui_session_budget: Optional[float] = 10  # $10 USD budgets for UI Chat sessions
internal_user_budget_duration: Optional[str] = None
tag_budget_config: Optional[Dict[str, BudgetConfig]] = None
max_end_user_budget: Optional[float] = None
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
public_model_groups: Optional[List[str]] = None
public_model_groups_links: Dict[str, str] = {}
#### REQUEST PRIORITIZATION ######
priority_reservation: Optional[Dict[str, float]] = None
priority_reservation_settings: "PriorityReservationSettings" = (
    PriorityReservationSettings()
)


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
module_level_aclient = AsyncHTTPHandler(
    timeout=request_timeout, client_alias="module level aclient"
)
module_level_client = HTTPHandler(timeout=request_timeout)

#### RETRIES ####
num_retries: Optional[int] = None  # per model endpoint
max_fallbacks: Optional[int] = None
default_fallbacks: Optional[List] = None
fallbacks: Optional[List] = None
context_window_fallbacks: Optional[List] = None
content_policy_fallbacks: Optional[List] = None
allowed_fails: int = 3
num_retries_per_request: Optional[int] = (
    None  # for the request overall (incl. fallbacks + model retries)
)
####### SECRET MANAGERS #####################
secret_manager_client: Optional[Any] = (
    None  # list of instantiated key management clients - e.g. azure kv, infisical, etc.
)
_google_kms_resource_name: Optional[str] = None
_key_management_system: Optional[KeyManagementSystem] = None
_key_management_settings: KeyManagementSettings = KeyManagementSettings()
#### PII MASKING ####
output_parse_pii: bool = False
#############################################
from litellm.litellm_core_utils.get_model_cost_map import get_model_cost_map

model_cost = get_model_cost_map(url=model_cost_map_url)
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
ai21_models: Set = set()
ai21_chat_models: Set = set()
nlp_cloud_models: Set = set()
aleph_alpha_models: Set = set()
bedrock_models: Set = set()
bedrock_converse_models: Set = set(BEDROCK_CONVERSE_MODELS)
fireworks_ai_models: Set = set()
fireworks_ai_embedding_models: Set = set()
deepinfra_models: Set = set()
perplexity_models: Set = set()
watsonx_models: Set = set()
gemini_models: Set = set()
xai_models: Set = set()
deepseek_models: Set = set()
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
        elif value.get("litellm_provider") == "deepseek":
            deepseek_models.add(key)
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
    | vertex_language_models
    | watsonx_models
    | gemini_models
    | text_completion_codestral_models
    | xai_models
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
)

model_list_set = set(model_list)

provider_list: List[Union[LlmProviders, str]] = list(LlmProviders)


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
    | vertex_deepseek_models,
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
    "deepseek": deepseek_models,
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

from .timeout import timeout
from .cost_calculator import completion_cost
from litellm.litellm_core_utils.litellm_logging import Logging, modify_integration
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.litellm_core_utils.core_helpers import remove_index_from_tool_calls
from litellm.litellm_core_utils.token_counter import get_modified_max_tokens
from .utils import (
    client,
    exception_type,
    get_optional_params,
    get_response_string,
    token_counter,
    create_pretrained_tokenizer,
    create_tokenizer,
    supports_function_calling,
    supports_web_search,
    supports_url_context,
    supports_response_schema,
    supports_parallel_function_calling,
    supports_vision,
    supports_audio_input,
    supports_audio_output,
    supports_system_messages,
    supports_reasoning,
    get_litellm_params,
    acreate,
    get_max_tokens,
    get_model_info,
    register_prompt_template,
    validate_environment,
    check_valid_key,
    register_model,
    encode,
    decode,
    _calculate_retry_after,
    _should_retry,
    get_supported_openai_params,
    get_api_base,
    get_first_chars_messages,
    ModelResponse,
    ModelResponseStream,
    EmbeddingResponse,
    ImageResponse,
    TranscriptionResponse,
    TextCompletionResponse,
    get_provider_fields,
    ModelResponseListIterator,
    get_valid_models,
)

ALL_LITELLM_RESPONSE_TYPES = [
    ModelResponse,
    EmbeddingResponse,
    ImageResponse,
    TranscriptionResponse,
    TextCompletionResponse,
]

from .llms.bytez.chat.transformation import BytezChatConfig
from .llms.custom_llm import CustomLLM
from .llms.bedrock.chat.converse_transformation import AmazonConverseConfig
from .llms.openai_like.chat.handler import OpenAILikeChatConfig
from .llms.aiohttp_openai.chat.transformation import AiohttpOpenAIChatConfig
from .llms.galadriel.chat.transformation import GaladrielChatConfig
from .llms.github.chat.transformation import GithubChatConfig
from .llms.compactifai.chat.transformation import CompactifAIChatConfig
from .llms.empower.chat.transformation import EmpowerChatConfig
from .llms.huggingface.chat.transformation import HuggingFaceChatConfig
from .llms.huggingface.embedding.transformation import HuggingFaceEmbeddingConfig
from .llms.oobabooga.chat.transformation import OobaboogaConfig
from .llms.maritalk import MaritalkConfig
from .llms.openrouter.chat.transformation import OpenrouterConfig
from .llms.datarobot.chat.transformation import DataRobotConfig
from .llms.anthropic.chat.transformation import AnthropicConfig
from .llms.anthropic.common_utils import AnthropicModelInfo
from .llms.groq.stt.transformation import GroqSTTConfig
from .llms.anthropic.completion.transformation import AnthropicTextConfig
from .llms.triton.completion.transformation import TritonConfig
from .llms.triton.completion.transformation import TritonGenerateConfig
from .llms.triton.completion.transformation import TritonInferConfig
from .llms.triton.embedding.transformation import TritonEmbeddingConfig
from .llms.huggingface.rerank.transformation import HuggingFaceRerankConfig
from .llms.databricks.chat.transformation import DatabricksConfig
from .llms.databricks.embed.transformation import DatabricksEmbeddingConfig
from .llms.predibase.chat.transformation import PredibaseConfig
from .llms.replicate.chat.transformation import ReplicateConfig
from .llms.snowflake.chat.transformation import SnowflakeConfig
from .llms.cohere.rerank.transformation import CohereRerankConfig
from .llms.cohere.rerank_v2.transformation import CohereRerankV2Config
from .llms.azure_ai.rerank.transformation import AzureAIRerankConfig
from .llms.infinity.rerank.transformation import InfinityRerankConfig
from .llms.jina_ai.rerank.transformation import JinaAIRerankConfig
from .llms.deepinfra.rerank.transformation import DeepinfraRerankConfig
from .llms.clarifai.chat.transformation import ClarifaiConfig
from .llms.ai21.chat.transformation import AI21ChatConfig, AI21ChatConfig as AI21Config
from .llms.meta_llama.chat.transformation import LlamaAPIConfig
from .llms.anthropic.experimental_pass_through.messages.transformation import (
    AnthropicMessagesConfig,
)
from .llms.bedrock.messages.invoke_transformations.anthropic_claude3_transformation import (
    AmazonAnthropicClaudeMessagesConfig,
)
from .llms.together_ai.chat import TogetherAIConfig
from .llms.together_ai.completion.transformation import TogetherAITextCompletionConfig
from .llms.cloudflare.chat.transformation import CloudflareChatConfig
from .llms.novita.chat.transformation import NovitaConfig
from .llms.deprecated_providers.palm import (
    PalmConfig,
)  # here to prevent breaking changes
from .llms.nlp_cloud.chat.handler import NLPCloudConfig
from .llms.petals.completion.transformation import PetalsConfig
from .llms.deprecated_providers.aleph_alpha import AlephAlphaConfig
from .llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
    VertexGeminiConfig,
    VertexGeminiConfig as VertexAIConfig,
)
from .llms.gemini.common_utils import GeminiModelInfo
from .llms.gemini.chat.transformation import (
    GoogleAIStudioGeminiConfig,
    GoogleAIStudioGeminiConfig as GeminiConfig,  # aliased to maintain backwards compatibility
)


from .llms.vertex_ai.vertex_embeddings.transformation import (
    VertexAITextEmbeddingConfig,
)

vertexAITextEmbeddingConfig = VertexAITextEmbeddingConfig()

from .llms.vertex_ai.vertex_ai_partner_models.anthropic.transformation import (
    VertexAIAnthropicConfig,
)
from .llms.vertex_ai.vertex_ai_partner_models.llama3.transformation import (
    VertexAILlama3Config,
)
from .llms.vertex_ai.vertex_ai_partner_models.ai21.transformation import (
    VertexAIAi21Config,
)
from .llms.ollama.chat.transformation import OllamaChatConfig
from .llms.ollama.completion.transformation import OllamaConfig
from .llms.sagemaker.completion.transformation import SagemakerConfig
from .llms.sagemaker.chat.transformation import SagemakerChatConfig
from .llms.bedrock.chat.invoke_handler import (
    AmazonCohereChatConfig,
    bedrock_tool_name_mappings,
)

from .llms.bedrock.common_utils import (
    AmazonBedrockGlobalConfig,
)
from .llms.bedrock.chat.invoke_transformations.amazon_ai21_transformation import (
    AmazonAI21Config,
)
from .llms.bedrock.chat.invoke_transformations.amazon_nova_transformation import (
    AmazonInvokeNovaConfig,
)
from .llms.bedrock.chat.invoke_transformations.anthropic_claude2_transformation import (
    AmazonAnthropicConfig,
)
from .llms.bedrock.chat.invoke_transformations.anthropic_claude3_transformation import (
    AmazonAnthropicClaudeConfig,
)
from .llms.bedrock.chat.invoke_transformations.amazon_cohere_transformation import (
    AmazonCohereConfig,
)
from .llms.bedrock.chat.invoke_transformations.amazon_llama_transformation import (
    AmazonLlamaConfig,
)
from .llms.bedrock.chat.invoke_transformations.amazon_deepseek_transformation import (
    AmazonDeepSeekR1Config,
)
from .llms.bedrock.chat.invoke_transformations.amazon_mistral_transformation import (
    AmazonMistralConfig,
)
from .llms.bedrock.chat.invoke_transformations.amazon_titan_transformation import (
    AmazonTitanConfig,
)
from .llms.bedrock.chat.invoke_transformations.base_invoke_transformation import (
    AmazonInvokeConfig,
)

from .llms.bedrock.image.amazon_stability1_transformation import AmazonStabilityConfig
from .llms.bedrock.image.amazon_stability3_transformation import AmazonStability3Config
from .llms.bedrock.image.amazon_nova_canvas_transformation import AmazonNovaCanvasConfig
from .llms.bedrock.embed.amazon_titan_g1_transformation import AmazonTitanG1Config
from .llms.bedrock.embed.amazon_titan_multimodal_transformation import (
    AmazonTitanMultimodalEmbeddingG1Config,
)
from .llms.bedrock.embed.amazon_titan_v2_transformation import (
    AmazonTitanV2Config,
)
from .llms.cohere.chat.transformation import CohereChatConfig
from .llms.bedrock.embed.cohere_transformation import BedrockCohereEmbeddingConfig
from .llms.openai.openai import OpenAIConfig, MistralEmbeddingConfig
from .llms.openai.image_variations.transformation import OpenAIImageVariationConfig
from .llms.deepinfra.chat.transformation import DeepInfraConfig
from .llms.deepgram.audio_transcription.transformation import (
    DeepgramAudioTranscriptionConfig,
)
from .llms.topaz.common_utils import TopazModelInfo
from .llms.topaz.image_variations.transformation import TopazImageVariationConfig
from litellm.llms.openai.completion.transformation import OpenAITextCompletionConfig
from .llms.groq.chat.transformation import GroqChatConfig
from .llms.voyage.embedding.transformation import VoyageEmbeddingConfig
from .llms.voyage.embedding.transformation_contextual import (
    VoyageContextualEmbeddingConfig,
)
from .llms.infinity.embedding.transformation import InfinityEmbeddingConfig
from .llms.azure_ai.chat.transformation import AzureAIStudioConfig
from .llms.mistral.chat.transformation import MistralConfig
from .llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from .llms.azure.responses.transformation import AzureOpenAIResponsesAPIConfig
from .llms.azure.responses.o_series_transformation import (
    AzureOpenAIOSeriesResponsesAPIConfig,
)
from .llms.openai.chat.o_series_transformation import (
    OpenAIOSeriesConfig as OpenAIO1Config,  # maintain backwards compatibility
    OpenAIOSeriesConfig,
)

from .llms.snowflake.chat.transformation import SnowflakeConfig
from .llms.gradient_ai.chat.transformation import GradientAIConfig

openaiOSeriesConfig = OpenAIOSeriesConfig()
from .llms.openai.chat.gpt_transformation import (
    OpenAIGPTConfig,
)
from .llms.openai.chat.gpt_5_transformation import (
    OpenAIGPT5Config,
)
from .llms.openai.transcriptions.whisper_transformation import (
    OpenAIWhisperAudioTranscriptionConfig,
)
from .llms.openai.transcriptions.gpt_transformation import (
    OpenAIGPTAudioTranscriptionConfig,
)

openAIGPTConfig = OpenAIGPTConfig()
from .llms.openai.chat.gpt_audio_transformation import (
    OpenAIGPTAudioConfig,
)

openAIGPTAudioConfig = OpenAIGPTAudioConfig()
openAIGPT5Config = OpenAIGPT5Config()

from .llms.nvidia_nim.chat.transformation import NvidiaNimConfig
from .llms.nvidia_nim.embed import NvidiaNimEmbeddingConfig

nvidiaNimConfig = NvidiaNimConfig()
nvidiaNimEmbeddingConfig = NvidiaNimEmbeddingConfig()

from .llms.featherless_ai.chat.transformation import FeatherlessAIConfig
from .llms.cerebras.chat import CerebrasConfig
from .llms.baseten.chat import BasetenConfig
from .llms.sambanova.chat import SambanovaConfig
from .llms.sambanova.embedding.transformation import SambaNovaEmbeddingConfig
from .llms.ai21.chat.transformation import AI21ChatConfig
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
from .llms.azure.chat.o_series_transformation import AzureOpenAIO1Config
from .llms.watsonx.completion.transformation import IBMWatsonXAIConfig
from .llms.watsonx.chat.transformation import IBMWatsonXChatConfig
from .llms.watsonx.embed.transformation import IBMWatsonXEmbeddingConfig
from .llms.github_copilot.chat.transformation import GithubCopilotConfig
from .llms.nebius.chat.transformation import NebiusConfig
from .llms.wandb.chat.transformation import WandbConfig
from .llms.dashscope.chat.transformation import DashScopeChatConfig
from .llms.moonshot.chat.transformation import MoonshotChatConfig
from .llms.v0.chat.transformation import V0ChatConfig
from .llms.oci.chat.transformation import OCIChatConfig
from .llms.morph.chat.transformation import MorphChatConfig
from .llms.lambda_ai.chat.transformation import LambdaAIChatConfig
from .llms.hyperbolic.chat.transformation import HyperbolicChatConfig
from .llms.vercel_ai_gateway.chat.transformation import VercelAIGatewayConfig
from .llms.ovhcloud.chat.transformation import OVHCloudChatConfig
from .llms.ovhcloud.embedding.transformation import OVHCloudEmbeddingConfig
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
from .batch_completion.main import *  # type: ignore
from .rerank_api.main import *
from .llms.anthropic.experimental_pass_through.messages.handler import *
from .responses.main import *
from .realtime_api.main import _arealtime
from .fine_tuning.main import *
from .files.main import *
from .scheduler import *
from .cost_calculator import response_cost_calculator, cost_per_token

### ADAPTERS ###
from .types.adapter import AdapterItem
import litellm.anthropic_interface as anthropic

adapters: List[AdapterItem] = []

### Vector Store Registry ###
from .vector_stores.vector_store_registry import VectorStoreRegistry

vector_store_registry: Optional[VectorStoreRegistry] = None

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

### GLOBAL CONFIG ###
global_bitbucket_config: Optional[Dict[str, Any]] = None


def set_global_bitbucket_config(config: Dict[str, Any]) -> None:
    """Set global BitBucket configuration for prompt management."""
    global global_bitbucket_config
    global_bitbucket_config = config
