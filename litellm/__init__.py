### Hide pydantic namespace conflict warnings globally ###
import warnings

warnings.filterwarnings("ignore", message=".*conflict with protected namespace.*")
### INIT VARIABLES ###
import threading, requests, os
from typing import Callable, List, Optional, Dict, Union, Any, Literal, get_args
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.caching import Cache
from litellm._logging import (
    set_verbose,
    _turn_on_debug,
    verbose_logger,
    json_logs,
    _turn_on_json,
    log_level,
)

from litellm.types.guardrails import GuardrailItem
from litellm.proxy._types import (
    KeyManagementSystem,
    KeyManagementSettings,
    LiteLLM_UpperboundKeyGenerateParams,
)
import httpx
import dotenv
from enum import Enum

litellm_mode = os.getenv("LITELLM_MODE", "DEV")  # "PRODUCTION", "DEV"
if litellm_mode == "DEV":
    dotenv.load_dotenv()
#############################################
if set_verbose == True:
    _turn_on_debug()
#############################################
### Callbacks /Logging / Success / Failure Handlers ###
input_callback: List[Union[str, Callable]] = []
success_callback: List[Union[str, Callable]] = []
failure_callback: List[Union[str, Callable]] = []
service_callback: List[Union[str, Callable]] = []
_custom_logger_compatible_callbacks_literal = Literal[
    "lago",
    "openmeter",
    "logfire",
    "dynamic_rate_limiter",
    "langsmith",
    "prometheus",
    "datadog",
    "galileo",
    "braintrust",
    "arize",
    "gcs_bucket",
]
_known_custom_logger_compatible_callbacks: List = list(
    get_args(_custom_logger_compatible_callbacks_literal)
)
callbacks: List[Union[Callable, _custom_logger_compatible_callbacks_literal]] = []
langfuse_default_tags: Optional[List[str]] = None
langsmith_batch_size: Optional[int] = None
_async_input_callback: List[Callable] = (
    []
)  # internal variable - async custom callbacks are routed here.
_async_success_callback: List[Union[str, Callable]] = (
    []
)  # internal variable - async custom callbacks are routed here.
_async_failure_callback: List[Callable] = (
    []
)  # internal variable - async custom callbacks are routed here.
pre_call_rules: List[Callable] = []
post_call_rules: List[Callable] = []
turn_off_message_logging: Optional[bool] = False
log_raw_request_response: bool = False
redact_messages_in_exceptions: Optional[bool] = False
redact_user_api_key_info: Optional[bool] = False
store_audit_logs = False  # Enterprise feature, allow users to see audit logs
## end of callbacks #############

email: Optional[str] = (
    None  # Not used anymore, will be removed in next MAJOR release - https://github.com/BerriAI/litellm/discussions/648
)
token: Optional[str] = (
    None  # Not used anymore, will be removed in next MAJOR release - https://github.com/BerriAI/litellm/discussions/648
)
telemetry = True
max_tokens = 256  # OpenAI Defaults
drop_params = bool(os.getenv("LITELLM_DROP_PARAMS", False))
modify_params = False
retry = True
### AUTH ###
api_key: Optional[str] = None
openai_key: Optional[str] = None
databricks_key: Optional[str] = None
azure_key: Optional[str] = None
anthropic_key: Optional[str] = None
replicate_key: Optional[str] = None
cohere_key: Optional[str] = None
clarifai_key: Optional[str] = None
maritalk_key: Optional[str] = None
ai21_key: Optional[str] = None
ollama_key: Optional[str] = None
openrouter_key: Optional[str] = None
predibase_key: Optional[str] = None
huggingface_key: Optional[str] = None
vertex_project: Optional[str] = None
vertex_location: Optional[str] = None
predibase_tenant_id: Optional[str] = None
togetherai_api_key: Optional[str] = None
cloudflare_api_key: Optional[str] = None
baseten_key: Optional[str] = None
aleph_alpha_key: Optional[str] = None
nlp_cloud_key: Optional[str] = None
common_cloud_provider_auth_params: dict = {
    "params": ["project", "region_name", "token"],
    "providers": ["vertex_ai", "bedrock", "watsonx", "azure", "vertex_ai_beta"],
}
use_client: bool = False
ssl_verify: Union[str, bool] = True
ssl_certificate: Optional[str] = None
disable_streaming_logging: bool = False
in_memory_llm_clients_cache: dict = {}
safe_memory_mode: bool = False
enable_azure_ad_token_refresh: Optional[bool] = False
### DEFAULT AZURE API VERSION ###
AZURE_DEFAULT_API_VERSION = "2024-08-01-preview"  # this is updated to the latest
### COHERE EMBEDDINGS DEFAULT TYPE ###
COHERE_DEFAULT_EMBEDDING_INPUT_TYPE = "search_document"
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
##################
### PREVIEW FEATURES ###
enable_preview_features: bool = False
return_response_headers: bool = (
    False  # get response headers from LLM Api providers - example x-remaining-requests,
)
enable_json_schema_validation: bool = False
##################
logging: bool = True
enable_loadbalancing_on_batch_endpoints: Optional[bool] = None
enable_caching_on_provider_specific_optional_params: bool = (
    False  # feature-flag for caching on optional params - e.g. 'top_k'
)
caching: bool = (
    False  # Not used anymore, will be removed in next MAJOR release - https://github.com/BerriAI/litellm/discussions/648
)
always_read_redis: bool = (
    True  # always use redis for rate limiting logic on litellm proxy
)
caching_with_models: bool = (
    False  # # Not used anymore, will be removed in next MAJOR release - https://github.com/BerriAI/litellm/discussions/648
)
cache: Optional[Cache] = (
    None  # cache object <- use this - https://docs.litellm.ai/docs/caching
)
default_in_memory_ttl: Optional[float] = None
default_redis_ttl: Optional[float] = None
model_alias_map: Dict[str, str] = {}
model_group_alias_map: Dict[str, str] = {}
max_budget: float = 0.0  # set the max budget across all providers
budget_duration: Optional[str] = (
    None  # proxy only - resets budget after fixed duration. You can set duration as seconds ("30s"), minutes ("30m"), hours ("30h"), days ("30d").
)
default_soft_budget: float = (
    50.0  # by default all litellm proxy keys have a soft budget of 50.0
)
forward_traceparent_to_llm_provider: bool = False
_openai_finish_reasons = ["stop", "length", "function_call", "content_filter", "null"]
_openai_completion_params = [
    "functions",
    "function_call",
    "temperature",
    "temperature",
    "top_p",
    "n",
    "stream",
    "stop",
    "max_tokens",
    "presence_penalty",
    "frequency_penalty",
    "logit_bias",
    "user",
    "request_timeout",
    "api_base",
    "api_version",
    "api_key",
    "deployment_id",
    "organization",
    "base_url",
    "default_headers",
    "timeout",
    "response_format",
    "seed",
    "tools",
    "tool_choice",
    "max_retries",
]
_litellm_completion_params = [
    "metadata",
    "acompletion",
    "caching",
    "mock_response",
    "api_key",
    "api_version",
    "api_base",
    "force_timeout",
    "logger_fn",
    "verbose",
    "custom_llm_provider",
    "litellm_logging_obj",
    "litellm_call_id",
    "use_client",
    "id",
    "fallbacks",
    "azure",
    "headers",
    "model_list",
    "num_retries",
    "context_window_fallback_dict",
    "roles",
    "final_prompt_value",
    "bos_token",
    "eos_token",
    "request_timeout",
    "complete_response",
    "self",
    "client",
    "rpm",
    "tpm",
    "input_cost_per_token",
    "output_cost_per_token",
    "hf_model_name",
    "model_info",
    "proxy_server_request",
    "preset_cache_key",
]
_current_cost = 0  # private variable, used if max budget is set
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
generic_logger_headers: Optional[Dict] = None
default_key_generate_params: Optional[Dict] = None
upperbound_key_generate_params: Optional[LiteLLM_UpperboundKeyGenerateParams] = None
default_user_params: Optional[Dict] = None
default_team_settings: Optional[List] = None
max_user_budget: Optional[float] = None
default_max_internal_user_budget: Optional[float] = None
max_internal_user_budget: Optional[float] = None
internal_user_budget_duration: Optional[str] = None
max_end_user_budget: Optional[float] = None
#### REQUEST PRIORITIZATION ####
priority_reservation: Optional[Dict[str, float]] = None
#### RELIABILITY ####
REPEATED_STREAMING_CHUNK_LIMIT = 100  # catch if model starts looping the same chunk while streaming. Uses high default to prevent false positives.
request_timeout: float = 6000
module_level_aclient = AsyncHTTPHandler(timeout=request_timeout)
module_level_client = HTTPHandler(timeout=request_timeout)
num_retries: Optional[int] = None  # per model endpoint
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
_key_management_settings: Optional[KeyManagementSettings] = None
#### PII MASKING ####
output_parse_pii: bool = False
#############################################


def get_model_cost_map(url: str):
    if (
        os.getenv("LITELLM_LOCAL_MODEL_COST_MAP", False) == True
        or os.getenv("LITELLM_LOCAL_MODEL_COST_MAP", False) == "True"
    ):
        import importlib.resources
        import json

        with importlib.resources.open_text(
            "litellm", "model_prices_and_context_window_backup.json"
        ) as f:
            content = json.load(f)
            return content

    try:
        with requests.get(
            url, timeout=5
        ) as response:  # set a 5 second timeout for the get request
            response.raise_for_status()  # Raise an exception if the request is unsuccessful
            content = response.json()
            return content
    except Exception as e:
        import importlib.resources
        import json

        with importlib.resources.open_text(
            "litellm", "model_prices_and_context_window_backup.json"
        ) as f:
            content = json.load(f)
            return content


model_cost = get_model_cost_map(url=model_cost_map_url)
custom_prompt_dict: Dict[str, dict] = {}


####### THREAD-SPECIFIC DATA ###################
class MyLocal(threading.local):
    def __init__(self):
        self.user = "Hello World"


_thread_context = MyLocal()


def identify(event_details):
    # Store user in thread local data
    if "user" in event_details:
        _thread_context.user = event_details["user"]


####### ADDITIONAL PARAMS ################### configurable params if you use proxy models like Helicone, map spend to org id, etc.
api_base = None
headers = None
api_version = None
organization = None
project = None
config_path = None
vertex_ai_safety_settings: Optional[dict] = None
####### COMPLETION MODELS ###################
open_ai_chat_completion_models: List = []
open_ai_text_completion_models: List = []
cohere_models: List = []
cohere_chat_models: List = []
mistral_chat_models: List = []
anthropic_models: List = []
empower_models: List = []
openrouter_models: List = []
vertex_language_models: List = []
vertex_vision_models: List = []
vertex_chat_models: List = []
vertex_code_chat_models: List = []
vertex_ai_image_models: List = []
vertex_text_models: List = []
vertex_code_text_models: List = []
vertex_embedding_models: List = []
vertex_anthropic_models: List = []
vertex_llama3_models: List = []
vertex_ai_ai21_models: List = []
vertex_mistral_models: List = []
ai21_models: List = []
ai21_chat_models: List = []
nlp_cloud_models: List = []
aleph_alpha_models: List = []
bedrock_models: List = []
fireworks_ai_models: List = []
deepinfra_models: List = []
perplexity_models: List = []
watsonx_models: List = []
gemini_models: List = []
for key, value in model_cost.items():
    if value.get("litellm_provider") == "openai":
        open_ai_chat_completion_models.append(key)
    elif value.get("litellm_provider") == "text-completion-openai":
        open_ai_text_completion_models.append(key)
    elif value.get("litellm_provider") == "cohere":
        cohere_models.append(key)
    elif value.get("litellm_provider") == "cohere_chat":
        cohere_chat_models.append(key)
    elif value.get("litellm_provider") == "mistral":
        mistral_chat_models.append(key)
    elif value.get("litellm_provider") == "anthropic":
        anthropic_models.append(key)
    elif value.get("litellm_provider") == "empower":
        empower_models.append(key)
    elif value.get("litellm_provider") == "openrouter":
        openrouter_models.append(key)
    elif value.get("litellm_provider") == "vertex_ai-text-models":
        vertex_text_models.append(key)
    elif value.get("litellm_provider") == "vertex_ai-code-text-models":
        vertex_code_text_models.append(key)
    elif value.get("litellm_provider") == "vertex_ai-language-models":
        vertex_language_models.append(key)
    elif value.get("litellm_provider") == "vertex_ai-vision-models":
        vertex_vision_models.append(key)
    elif value.get("litellm_provider") == "vertex_ai-chat-models":
        vertex_chat_models.append(key)
    elif value.get("litellm_provider") == "vertex_ai-code-chat-models":
        vertex_code_chat_models.append(key)
    elif value.get("litellm_provider") == "vertex_ai-embedding-models":
        vertex_embedding_models.append(key)
    elif value.get("litellm_provider") == "vertex_ai-anthropic_models":
        key = key.replace("vertex_ai/", "")
        vertex_anthropic_models.append(key)
    elif value.get("litellm_provider") == "vertex_ai-llama_models":
        key = key.replace("vertex_ai/", "")
        vertex_llama3_models.append(key)
    elif value.get("litellm_provider") == "vertex_ai-mistral_models":
        key = key.replace("vertex_ai/", "")
        vertex_mistral_models.append(key)
    elif value.get("litellm_provider") == "vertex_ai-ai21_models":
        key = key.replace("vertex_ai/", "")
        vertex_ai_ai21_models.append(key)
    elif value.get("litellm_provider") == "vertex_ai-image-models":
        key = key.replace("vertex_ai/", "")
        vertex_ai_image_models.append(key)
    elif value.get("litellm_provider") == "ai21":
        if value.get("mode") == "chat":
            ai21_chat_models.append(key)
        else:
            ai21_models.append(key)
    elif value.get("litellm_provider") == "nlp_cloud":
        nlp_cloud_models.append(key)
    elif value.get("litellm_provider") == "aleph_alpha":
        aleph_alpha_models.append(key)
    elif value.get("litellm_provider") == "bedrock":
        bedrock_models.append(key)
    elif value.get("litellm_provider") == "deepinfra":
        deepinfra_models.append(key)
    elif value.get("litellm_provider") == "perplexity":
        perplexity_models.append(key)
    elif value.get("litellm_provider") == "watsonx":
        watsonx_models.append(key)
    elif value.get("litellm_provider") == "gemini":
        gemini_models.append(key)
    elif value.get("litellm_provider") == "fireworks_ai":
        fireworks_ai_models.append(key)
# known openai compatible endpoints - we'll eventually move this list to the model_prices_and_context_window.json dictionary
openai_compatible_endpoints: List = [
    "api.perplexity.ai",
    "api.endpoints.anyscale.com/v1",
    "api.deepinfra.com/v1/openai",
    "api.mistral.ai/v1",
    "codestral.mistral.ai/v1/chat/completions",
    "codestral.mistral.ai/v1/fim/completions",
    "api.groq.com/openai/v1",
    "https://integrate.api.nvidia.com/v1",
    "api.deepseek.com/v1",
    "api.together.xyz/v1",
    "app.empower.dev/api/v1",
    "inference.friendli.ai/v1",
    "api.sambanova.ai/v1",
]

# this is maintained for Exception Mapping
openai_compatible_providers: List = [
    "anyscale",
    "mistral",
    "groq",
    "nvidia_nim",
    "cerebras",
    "sambanova",
    "ai21_chat",
    "volcengine",
    "codestral",
    "deepseek",
    "deepinfra",
    "perplexity",
    "xinference",
    "together_ai",
    "fireworks_ai",
    "empower",
    "friendliai",
    "azure_ai",
    "github",
]
openai_text_completion_compatible_providers: List = (
    [  # providers that support `/v1/completions`
        "together_ai",
        "fireworks_ai",
    ]
)

# well supported replicate llms
replicate_models: List = [
    # llama replicate supported LLMs
    "replicate/llama-2-70b-chat:2796ee9483c3fd7aa2e171d38f4ca12251a30609463dcfd4cd76703f22e96cdf",
    "a16z-infra/llama-2-13b-chat:2a7f981751ec7fdf87b5b91ad4db53683a98082e9ff7bfd12c8cd5ea85980a52",
    "meta/codellama-13b:1c914d844307b0588599b8393480a3ba917b660c7e9dfae681542b5325f228db",
    # Vicuna
    "replicate/vicuna-13b:6282abe6a492de4145d7bb601023762212f9ddbbe78278bd6771c8b3b2f2a13b",
    "joehoover/instructblip-vicuna13b:c4c54e3c8c97cd50c2d2fec9be3b6065563ccf7d43787fb99f84151b867178fe",
    # Flan T-5
    "daanelson/flan-t5-large:ce962b3f6792a57074a601d3979db5839697add2e4e02696b3ced4c022d4767f",
    # Others
    "replicate/dolly-v2-12b:ef0e1aefc61f8e096ebe4db6b2bacc297daf2ef6899f0f7e001ec445893500e5",
    "replit/replit-code-v1-3b:b84f4c074b807211cd75e3e8b1589b6399052125b4c27106e43d47189e8415ad",
]

clarifai_models: List = [
    "clarifai/meta.Llama-3.Llama-3-8B-Instruct",
    "clarifai/gcp.generate.gemma-1_1-7b-it",
    "clarifai/mistralai.completion.mixtral-8x22B",
    "clarifai/cohere.generate.command-r-plus",
    "clarifai/databricks.drbx.dbrx-instruct",
    "clarifai/mistralai.completion.mistral-large",
    "clarifai/mistralai.completion.mistral-medium",
    "clarifai/mistralai.completion.mistral-small",
    "clarifai/mistralai.completion.mixtral-8x7B-Instruct-v0_1",
    "clarifai/gcp.generate.gemma-2b-it",
    "clarifai/gcp.generate.gemma-7b-it",
    "clarifai/deci.decilm.deciLM-7B-instruct",
    "clarifai/mistralai.completion.mistral-7B-Instruct",
    "clarifai/gcp.generate.gemini-pro",
    "clarifai/anthropic.completion.claude-v1",
    "clarifai/anthropic.completion.claude-instant-1_2",
    "clarifai/anthropic.completion.claude-instant",
    "clarifai/anthropic.completion.claude-v2",
    "clarifai/anthropic.completion.claude-2_1",
    "clarifai/meta.Llama-2.codeLlama-70b-Python",
    "clarifai/meta.Llama-2.codeLlama-70b-Instruct",
    "clarifai/openai.completion.gpt-3_5-turbo-instruct",
    "clarifai/meta.Llama-2.llama2-7b-chat",
    "clarifai/meta.Llama-2.llama2-13b-chat",
    "clarifai/meta.Llama-2.llama2-70b-chat",
    "clarifai/openai.chat-completion.gpt-4-turbo",
    "clarifai/microsoft.text-generation.phi-2",
    "clarifai/meta.Llama-2.llama2-7b-chat-vllm",
    "clarifai/upstage.solar.solar-10_7b-instruct",
    "clarifai/openchat.openchat.openchat-3_5-1210",
    "clarifai/togethercomputer.stripedHyena.stripedHyena-Nous-7B",
    "clarifai/gcp.generate.text-bison",
    "clarifai/meta.Llama-2.llamaGuard-7b",
    "clarifai/fblgit.una-cybertron.una-cybertron-7b-v2",
    "clarifai/openai.chat-completion.GPT-4",
    "clarifai/openai.chat-completion.GPT-3_5-turbo",
    "clarifai/ai21.complete.Jurassic2-Grande",
    "clarifai/ai21.complete.Jurassic2-Grande-Instruct",
    "clarifai/ai21.complete.Jurassic2-Jumbo-Instruct",
    "clarifai/ai21.complete.Jurassic2-Jumbo",
    "clarifai/ai21.complete.Jurassic2-Large",
    "clarifai/cohere.generate.cohere-generate-command",
    "clarifai/wizardlm.generate.wizardCoder-Python-34B",
    "clarifai/wizardlm.generate.wizardLM-70B",
    "clarifai/tiiuae.falcon.falcon-40b-instruct",
    "clarifai/togethercomputer.RedPajama.RedPajama-INCITE-7B-Chat",
    "clarifai/gcp.generate.code-gecko",
    "clarifai/gcp.generate.code-bison",
    "clarifai/mistralai.completion.mistral-7B-OpenOrca",
    "clarifai/mistralai.completion.openHermes-2-mistral-7B",
    "clarifai/wizardlm.generate.wizardLM-13B",
    "clarifai/huggingface-research.zephyr.zephyr-7B-alpha",
    "clarifai/wizardlm.generate.wizardCoder-15B",
    "clarifai/microsoft.text-generation.phi-1_5",
    "clarifai/databricks.Dolly-v2.dolly-v2-12b",
    "clarifai/bigcode.code.StarCoder",
    "clarifai/salesforce.xgen.xgen-7b-8k-instruct",
    "clarifai/mosaicml.mpt.mpt-7b-instruct",
    "clarifai/anthropic.completion.claude-3-opus",
    "clarifai/anthropic.completion.claude-3-sonnet",
    "clarifai/gcp.generate.gemini-1_5-pro",
    "clarifai/gcp.generate.imagen-2",
    "clarifai/salesforce.blip.general-english-image-caption-blip-2",
]


huggingface_models: List = [
    "meta-llama/Llama-2-7b-hf",
    "meta-llama/Llama-2-7b-chat-hf",
    "meta-llama/Llama-2-13b-hf",
    "meta-llama/Llama-2-13b-chat-hf",
    "meta-llama/Llama-2-70b-hf",
    "meta-llama/Llama-2-70b-chat-hf",
    "meta-llama/Llama-2-7b",
    "meta-llama/Llama-2-7b-chat",
    "meta-llama/Llama-2-13b",
    "meta-llama/Llama-2-13b-chat",
    "meta-llama/Llama-2-70b",
    "meta-llama/Llama-2-70b-chat",
]  # these have been tested on extensively. But by default all text2text-generation and text-generation models are supported by liteLLM. - https://docs.litellm.ai/docs/providers
empower_models = [
    "empower/empower-functions",
    "empower/empower-functions-small",
]

together_ai_models: List = [
    # llama llms - chat
    "togethercomputer/llama-2-70b-chat",
    # llama llms - language / instruct
    "togethercomputer/llama-2-70b",
    "togethercomputer/LLaMA-2-7B-32K",
    "togethercomputer/Llama-2-7B-32K-Instruct",
    "togethercomputer/llama-2-7b",
    # falcon llms
    "togethercomputer/falcon-40b-instruct",
    "togethercomputer/falcon-7b-instruct",
    # alpaca
    "togethercomputer/alpaca-7b",
    # chat llms
    "HuggingFaceH4/starchat-alpha",
    # code llms
    "togethercomputer/CodeLlama-34b",
    "togethercomputer/CodeLlama-34b-Instruct",
    "togethercomputer/CodeLlama-34b-Python",
    "defog/sqlcoder",
    "NumbersStation/nsql-llama-2-7B",
    "WizardLM/WizardCoder-15B-V1.0",
    "WizardLM/WizardCoder-Python-34B-V1.0",
    # language llms
    "NousResearch/Nous-Hermes-Llama2-13b",
    "Austism/chronos-hermes-13b",
    "upstage/SOLAR-0-70b-16bit",
    "WizardLM/WizardLM-70B-V1.0",
]  # supports all together ai models, just pass in the model id e.g. completion(model="together_computer/replit_code_3b",...)


baseten_models: List = [
    "qvv0xeq",
    "q841o8w",
    "31dxrj3",
]  # FALCON 7B  # WizardLM  # Mosaic ML


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

model_list = (
    open_ai_chat_completion_models
    + open_ai_text_completion_models
    + cohere_models
    + cohere_chat_models
    + anthropic_models
    + replicate_models
    + openrouter_models
    + huggingface_models
    + vertex_chat_models
    + vertex_text_models
    + ai21_models
    + ai21_chat_models
    + together_ai_models
    + baseten_models
    + aleph_alpha_models
    + nlp_cloud_models
    + ollama_models
    + bedrock_models
    + deepinfra_models
    + perplexity_models
    + maritalk_models
    + vertex_language_models
    + watsonx_models
    + gemini_models
)


class LlmProviders(str, Enum):
    OPENAI = "openai"
    CUSTOM_OPENAI = "custom_openai"
    TEXT_COMPLETION_OPENAI = "text-completion-openai"
    COHERE = "cohere"
    COHERE_CHAT = "cohere_chat"
    CLARIFAI = "clarifai"
    ANTHROPIC = "anthropic"
    REPLICATE = "replicate"
    HUGGINGFACE = "huggingface"
    TOGETHER_AI = "together_ai"
    OPENROUTER = "openrouter"
    VERTEX_AI = "vertex_ai"
    VERTEX_AI_BETA = "vertex_ai_beta"
    PALM = "palm"
    GEMINI = "gemini"
    AI21 = "ai21"
    BASETEN = "baseten"
    AZURE = "azure"
    AZURE_TEXT = "azure_text"
    AZURE_AI = "azure_ai"
    SAGEMAKER = "sagemaker"
    SAGEMAKER_CHAT = "sagemaker_chat"
    BEDROCK = "bedrock"
    VLLM = "vllm"
    NLP_CLOUD = "nlp_cloud"
    PETALS = "petals"
    OOBABOOGA = "oobabooga"
    OLLAMA = "ollama"
    OLLAMA_CHAT = "ollama_chat"
    DEEPINFRA = "deepinfra"
    PERPLEXITY = "perplexity"
    ANYSCALE = "anyscale"
    MISTRAL = "mistral"
    GROQ = "groq"
    NVIDIA_NIM = "nvidia_nim"
    CEREBRAS = "cerebras"
    AI21_CHAT = "ai21_chat"
    VOLCENGINE = "volcengine"
    CODESTRAL = "codestral"
    TEXT_COMPLETION_CODESTRAL = "text-completion-codestral"
    DEEPSEEK = "deepseek"
    SAMBANOVA = "sambanova"
    MARITALK = "maritalk"
    VOYAGE = "voyage"
    CLOUDFLARE = "cloudflare"
    XINFERENCE = "xinference"
    FIREWORKS_AI = "fireworks_ai"
    FRIENDLIAI = "friendliai"
    WATSONX = "watsonx"
    TRITON = "triton"
    PREDIBASE = "predibase"
    DATABRICKS = "databricks"
    EMPOWER = "empower"
    GITHUB = "github"
    CUSTOM = "custom"


provider_list: List[Union[LlmProviders, str]] = list(LlmProviders)


models_by_provider: dict = {
    "openai": open_ai_chat_completion_models + open_ai_text_completion_models,
    "cohere": cohere_models + cohere_chat_models,
    "cohere_chat": cohere_chat_models,
    "anthropic": anthropic_models,
    "replicate": replicate_models,
    "huggingface": huggingface_models,
    "together_ai": together_ai_models,
    "baseten": baseten_models,
    "openrouter": openrouter_models,
    "vertex_ai": vertex_chat_models
    + vertex_text_models
    + vertex_anthropic_models
    + vertex_vision_models
    + vertex_language_models,
    "ai21": ai21_models,
    "bedrock": bedrock_models,
    "petals": petals_models,
    "ollama": ollama_models,
    "deepinfra": deepinfra_models,
    "perplexity": perplexity_models,
    "maritalk": maritalk_models,
    "watsonx": watsonx_models,
    "gemini": gemini_models,
    "fireworks_ai": fireworks_ai_models,
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
open_ai_embedding_models: List = ["text-embedding-ada-002"]
cohere_embedding_models: List = [
    "embed-english-v3.0",
    "embed-english-light-v3.0",
    "embed-multilingual-v3.0",
    "embed-english-v2.0",
    "embed-english-light-v2.0",
    "embed-multilingual-v2.0",
]
bedrock_embedding_models: List = [
    "amazon.titan-embed-text-v1",
    "cohere.embed-english-v3",
    "cohere.embed-multilingual-v3",
]

all_embedding_models = (
    open_ai_embedding_models
    + cohere_embedding_models
    + bedrock_embedding_models
    + vertex_embedding_models
)

####### IMAGE GENERATION MODELS ###################
openai_image_generation_models = ["dall-e-2", "dall-e-3"]

from .timeout import timeout
from .cost_calculator import completion_cost
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.litellm_core_utils.core_helpers import remove_index_from_tool_calls
from litellm.litellm_core_utils.token_counter import get_modified_max_tokens
from .utils import (
    client,
    exception_type,
    get_optional_params,
    get_response_string,
    modify_integration,
    token_counter,
    create_pretrained_tokenizer,
    create_tokenizer,
    supports_function_calling,
    supports_response_schema,
    supports_parallel_function_calling,
    supports_vision,
    supports_system_messages,
    get_litellm_params,
    acreate,
    get_model_list,
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
    EmbeddingResponse,
    ImageResponse,
    TranscriptionResponse,
    TextCompletionResponse,
    get_provider_fields,
    ModelResponseListIterator,
)

ALL_LITELLM_RESPONSE_TYPES = [
    ModelResponse,
    EmbeddingResponse,
    ImageResponse,
    TranscriptionResponse,
    TextCompletionResponse,
]

from .types.utils import ImageObject
from .llms.custom_llm import CustomLLM
from .llms.huggingface_restapi import HuggingfaceConfig
from .llms.anthropic.chat import AnthropicConfig
from .llms.anthropic.completion import AnthropicTextConfig
from .llms.databricks.chat import DatabricksConfig, DatabricksEmbeddingConfig
from .llms.predibase import PredibaseConfig
from .llms.replicate import ReplicateConfig
from .llms.cohere.completion import CohereConfig
from .llms.clarifai import ClarifaiConfig
from .llms.AI21.completion import AI21Config
from .llms.AI21.chat import AI21ChatConfig
from .llms.together_ai import TogetherAIConfig
from .llms.cloudflare import CloudflareConfig
from .llms.palm import PalmConfig
from .llms.gemini import GeminiConfig
from .llms.nlp_cloud import NLPCloudConfig
from .llms.aleph_alpha import AlephAlphaConfig
from .llms.petals import PetalsConfig
from .llms.vertex_ai_and_google_ai_studio.gemini.vertex_and_google_ai_studio_gemini import (
    VertexGeminiConfig,
    GoogleAIStudioGeminiConfig,
    VertexAIConfig,
)
from .llms.vertex_ai_and_google_ai_studio.vertex_embeddings.embedding_handler import (
    VertexAITextEmbeddingConfig,
)
from .llms.vertex_ai_and_google_ai_studio.vertex_ai_anthropic import (
    VertexAIAnthropicConfig,
)
from .llms.vertex_ai_and_google_ai_studio.vertex_ai_partner_models.llama3.transformation import (
    VertexAILlama3Config,
)
from .llms.vertex_ai_and_google_ai_studio.vertex_ai_partner_models.ai21.transformation import (
    VertexAIAi21Config,
)

from .llms.sagemaker.sagemaker import SagemakerConfig
from .llms.ollama import OllamaConfig
from .llms.ollama_chat import OllamaChatConfig
from .llms.maritalk import MaritTalkConfig
from .llms.bedrock.chat.invoke_handler import (
    AmazonCohereChatConfig,
    AmazonConverseConfig,
    BEDROCK_CONVERSE_MODELS,
    bedrock_tool_name_mappings,
)
from .llms.bedrock.common_utils import (
    AmazonTitanConfig,
    AmazonAI21Config,
    AmazonAnthropicConfig,
    AmazonAnthropicClaude3Config,
    AmazonCohereConfig,
    AmazonLlamaConfig,
    AmazonStabilityConfig,
    AmazonMistralConfig,
    AmazonBedrockGlobalConfig,
)
from .llms.bedrock.embed.amazon_titan_g1_transformation import AmazonTitanG1Config
from .llms.bedrock.embed.amazon_titan_multimodal_transformation import (
    AmazonTitanMultimodalEmbeddingG1Config,
)
from .llms.bedrock.embed.amazon_titan_v2_transformation import (
    AmazonTitanV2Config,
)
from .llms.bedrock.embed.cohere_transformation import BedrockCohereEmbeddingConfig
from .llms.OpenAI.openai import (
    OpenAIConfig,
    OpenAITextCompletionConfig,
    MistralEmbeddingConfig,
    DeepInfraConfig,
    GroqConfig,
    AzureAIStudioConfig,
)
from .llms.mistral.mistral_chat_transformation import MistralConfig
from .llms.OpenAI.chat.o1_transformation import (
    OpenAIO1Config,
)
from .llms.OpenAI.chat.gpt_transformation import (
    OpenAIGPTConfig,
)
from .llms.nvidia_nim import NvidiaNimConfig
from .llms.cerebras.chat import CerebrasConfig
from .llms.sambanova.chat import SambanovaConfig
from .llms.AI21.chat import AI21ChatConfig
from .llms.fireworks_ai import FireworksAIConfig
from .llms.volcengine import VolcEngineConfig
from .llms.text_completion_codestral import MistralTextCompletionConfig
from .llms.AzureOpenAI.azure import (
    AzureOpenAIConfig,
    AzureOpenAIError,
    AzureOpenAIAssistantsAPIConfig,
)
from .llms.watsonx import IBMWatsonXAIConfig
from .main import *  # type: ignore
from .integrations import *
from .exceptions import (
    AuthenticationError,
    InvalidRequestError,
    BadRequestError,
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
from .rerank_api.main import *
from .fine_tuning.main import *
from .files.main import *
from .scheduler import *
from .cost_calculator import response_cost_calculator, cost_per_token

### ADAPTERS ###
from .types.adapter import AdapterItem

adapters: List[AdapterItem] = []

### CUSTOM LLMs ###
from .types.llms.custom_llm import CustomLLMItem
from .types.utils import GenericStreamingChunk

custom_provider_map: List[CustomLLMItem] = []
_custom_providers: List[str] = (
    []
)  # internal helper util, used to track names of custom providers
