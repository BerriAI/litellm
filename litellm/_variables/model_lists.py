import re
import threading
from typing import Any, Dict, List, Literal, Optional, Union

import httpx

from litellm.caching.caching import Cache
from litellm.constants import (
    baseten_models,
    bedrock_embedding_models,
    cohere_embedding_models,
    empower_models,
    huggingface_models,
    open_ai_embedding_models,
    replicate_models,
    request_timeout,
    together_ai_models,
)
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.proxy._types import (
    KeyManagementSettings,
    KeyManagementSystem,
    LiteLLM_UpperboundKeyGenerateParams,
)
from litellm.types.guardrails import GuardrailItem
from litellm.types.llms.bedrock import COHERE_EMBEDDING_INPUT_TYPES
from litellm.types.utils import (
    BudgetConfig,
    CredentialItem,
    LlmProviders,
    StandardKeyGenerationConfig,
)

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
model_group_alias_map: Dict[str, str] = {}
max_budget: float = 0.0  # set the max budget across all providers
budget_duration: Optional[str] = (
    None  # proxy only - resets budget after fixed duration. You can set duration as seconds ("30s"), minutes ("30m"), hours ("30h"), days ("30d").
)
default_soft_budget: float = (
    50.0  # by default all litellm proxy keys have a soft budget of 50.0
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
generic_logger_headers: Optional[Dict] = None
default_key_generate_params: Optional[Dict] = None
upperbound_key_generate_params: Optional[LiteLLM_UpperboundKeyGenerateParams] = None
key_generation_settings: Optional[StandardKeyGenerationConfig] = None
default_internal_user_params: Optional[Dict] = None
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
custom_prometheus_metadata_labels: List[str] = []
#### REQUEST PRIORITIZATION ####
priority_reservation: Optional[Dict[str, float]] = None
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
api_version = None
organization = None
project = None
config_path = None
vertex_ai_safety_settings: Optional[dict] = None
BEDROCK_CONVERSE_MODELS = [
    "anthropic.claude-3-5-haiku-20241022-v1:0",
    "anthropic.claude-3-5-sonnet-20241022-v2:0",
    "anthropic.claude-3-5-sonnet-20240620-v1:0",
    "anthropic.claude-3-opus-20240229-v1:0",
    "anthropic.claude-3-sonnet-20240229-v1:0",
    "anthropic.claude-3-haiku-20240307-v1:0",
    "anthropic.claude-v2",
    "anthropic.claude-v2:1",
    "anthropic.claude-v1",
    "anthropic.claude-instant-v1",
    "ai21.jamba-instruct-v1:0",
    "meta.llama3-70b-instruct-v1:0",
    "meta.llama3-8b-instruct-v1:0",
    "meta.llama3-1-8b-instruct-v1:0",
    "meta.llama3-1-70b-instruct-v1:0",
    "meta.llama3-1-405b-instruct-v1:0",
    "meta.llama3-70b-instruct-v1:0",
    "mistral.mistral-large-2407-v1:0",
    "mistral.mistral-large-2402-v1:0",
    "meta.llama3-2-1b-instruct-v1:0",
    "meta.llama3-2-3b-instruct-v1:0",
    "meta.llama3-2-11b-instruct-v1:0",
    "meta.llama3-2-90b-instruct-v1:0",
]

####### COMPLETION MODELS ###################
open_ai_chat_completion_models: List = []
open_ai_text_completion_models: List = []
cohere_models: List = []
cohere_chat_models: List = []
mistral_chat_models: List = []
text_completion_codestral_models: List = []
anthropic_models: List = []
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
bedrock_converse_models: List = BEDROCK_CONVERSE_MODELS
fireworks_ai_models: List = []
fireworks_ai_embedding_models: List = []
deepinfra_models: List = []
perplexity_models: List = []
watsonx_models: List = []
gemini_models: List = []
xai_models: List = []
deepseek_models: List = []
azure_ai_models: List = []
jina_ai_models: List = []
voyage_models: List = []
databricks_models: List = []
cloudflare_models: List = []
codestral_models: List = []
friendliai_models: List = []
palm_models: List = []
groq_models: List = []
azure_models: List = []
azure_text_models: List = []
anyscale_models: List = []
cerebras_models: List = []
galadriel_models: List = []
sambanova_models: List = []
assemblyai_models: List = []
snowflake_models: List = []


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
            open_ai_chat_completion_models.append(key)
        elif value.get("litellm_provider") == "text-completion-openai":
            open_ai_text_completion_models.append(key)
        elif value.get("litellm_provider") == "azure_text":
            azure_text_models.append(key)
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
        elif value.get(
            "litellm_provider"
        ) == "bedrock" and not is_bedrock_pricing_only_model(key):
            bedrock_models.append(key)
        elif value.get("litellm_provider") == "bedrock_converse":
            bedrock_converse_models.append(key)
        elif value.get("litellm_provider") == "deepinfra":
            deepinfra_models.append(key)
        elif value.get("litellm_provider") == "perplexity":
            perplexity_models.append(key)
        elif value.get("litellm_provider") == "watsonx":
            watsonx_models.append(key)
        elif value.get("litellm_provider") == "gemini":
            gemini_models.append(key)
        elif value.get("litellm_provider") == "fireworks_ai":
            # ignore the 'up-to', '-to-' model names -> not real models. just for cost tracking based on model params.
            if "-to-" not in key and "fireworks-ai-default" not in key:
                fireworks_ai_models.append(key)
        elif value.get("litellm_provider") == "fireworks_ai-embedding-models":
            # ignore the 'up-to', '-to-' model names -> not real models. just for cost tracking based on model params.
            if "-to-" not in key:
                fireworks_ai_embedding_models.append(key)
        elif value.get("litellm_provider") == "text-completion-codestral":
            text_completion_codestral_models.append(key)
        elif value.get("litellm_provider") == "xai":
            xai_models.append(key)
        elif value.get("litellm_provider") == "deepseek":
            deepseek_models.append(key)
        elif value.get("litellm_provider") == "azure_ai":
            azure_ai_models.append(key)
        elif value.get("litellm_provider") == "voyage":
            voyage_models.append(key)
        elif value.get("litellm_provider") == "databricks":
            databricks_models.append(key)
        elif value.get("litellm_provider") == "cloudflare":
            cloudflare_models.append(key)
        elif value.get("litellm_provider") == "codestral":
            codestral_models.append(key)
        elif value.get("litellm_provider") == "friendliai":
            friendliai_models.append(key)
        elif value.get("litellm_provider") == "palm":
            palm_models.append(key)
        elif value.get("litellm_provider") == "groq":
            groq_models.append(key)
        elif value.get("litellm_provider") == "azure":
            azure_models.append(key)
        elif value.get("litellm_provider") == "anyscale":
            anyscale_models.append(key)
        elif value.get("litellm_provider") == "cerebras":
            cerebras_models.append(key)
        elif value.get("litellm_provider") == "galadriel":
            galadriel_models.append(key)
        elif value.get("litellm_provider") == "sambanova_models":
            sambanova_models.append(key)
        elif value.get("litellm_provider") == "assemblyai":
            assemblyai_models.append(key)
        elif value.get("litellm_provider") == "jina_ai":
            jina_ai_models.append(key)
        elif value.get("litellm_provider") == "snowflake":
            snowflake_models.append(key)


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
    + text_completion_codestral_models
    + xai_models
    + deepseek_models
    + azure_ai_models
    + voyage_models
    + databricks_models
    + cloudflare_models
    + codestral_models
    + friendliai_models
    + palm_models
    + groq_models
    + azure_models
    + anyscale_models
    + cerebras_models
    + galadriel_models
    + sambanova_models
    + azure_text_models
    + assemblyai_models
    + jina_ai_models
    + snowflake_models
)

model_list_set = set(model_list)

provider_list: List[Union[LlmProviders, str]] = list(LlmProviders)


models_by_provider: dict = {
    "openai": open_ai_chat_completion_models + open_ai_text_completion_models,
    "text-completion-openai": open_ai_text_completion_models,
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
    "bedrock": bedrock_models + bedrock_converse_models,
    "petals": petals_models,
    "ollama": ollama_models,
    "deepinfra": deepinfra_models,
    "perplexity": perplexity_models,
    "maritalk": maritalk_models,
    "watsonx": watsonx_models,
    "gemini": gemini_models,
    "fireworks_ai": fireworks_ai_models + fireworks_ai_embedding_models,
    "aleph_alpha": aleph_alpha_models,
    "text-completion-codestral": text_completion_codestral_models,
    "xai": xai_models,
    "deepseek": deepseek_models,
    "mistral": mistral_chat_models,
    "azure_ai": azure_ai_models,
    "voyage": voyage_models,
    "databricks": databricks_models,
    "cloudflare": cloudflare_models,
    "codestral": codestral_models,
    "nlp_cloud": nlp_cloud_models,
    "friendliai": friendliai_models,
    "palm": palm_models,
    "groq": groq_models,
    "azure": azure_models + azure_text_models,
    "azure_text": azure_text_models,
    "anyscale": anyscale_models,
    "cerebras": cerebras_models,
    "galadriel": galadriel_models,
    "sambanova": sambanova_models,
    "assemblyai": assemblyai_models,
    "jina_ai": jina_ai_models,
    "snowflake": snowflake_models,
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
    + cohere_embedding_models
    + bedrock_embedding_models
    + vertex_embedding_models
    + fireworks_ai_embedding_models
)

####### IMAGE GENERATION MODELS ###################
openai_image_generation_models = ["dall-e-2", "dall-e-3"]
