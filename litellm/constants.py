import os
from typing import List, Literal

ROUTER_MAX_FALLBACKS = int(os.getenv("ROUTER_MAX_FALLBACKS", 5))
DEFAULT_BATCH_SIZE = int(os.getenv("DEFAULT_BATCH_SIZE", 512))
DEFAULT_FLUSH_INTERVAL_SECONDS = int(os.getenv("DEFAULT_FLUSH_INTERVAL_SECONDS", 5))
DEFAULT_S3_FLUSH_INTERVAL_SECONDS = int(
    os.getenv("DEFAULT_S3_FLUSH_INTERVAL_SECONDS", 10)
)
DEFAULT_S3_BATCH_SIZE = int(os.getenv("DEFAULT_S3_BATCH_SIZE", 512))
DEFAULT_SQS_FLUSH_INTERVAL_SECONDS = int(
    os.getenv("DEFAULT_SQS_FLUSH_INTERVAL_SECONDS", 10)
)
DEFAULT_SQS_BATCH_SIZE = int(os.getenv("DEFAULT_SQS_BATCH_SIZE", 512))
SQS_SEND_MESSAGE_ACTION = "SendMessage"
SQS_API_VERSION = "2012-11-05"
DEFAULT_MAX_RETRIES = int(os.getenv("DEFAULT_MAX_RETRIES", 2))
DEFAULT_MAX_RECURSE_DEPTH = int(os.getenv("DEFAULT_MAX_RECURSE_DEPTH", 100))
DEFAULT_MAX_RECURSE_DEPTH_SENSITIVE_DATA_MASKER = int(
    os.getenv("DEFAULT_MAX_RECURSE_DEPTH_SENSITIVE_DATA_MASKER", 10)
)
DEFAULT_FAILURE_THRESHOLD_PERCENT = float(
    os.getenv("DEFAULT_FAILURE_THRESHOLD_PERCENT", 0.5)
)  # default cooldown a deployment if 50% of requests fail in a given minute
DEFAULT_MAX_TOKENS = int(os.getenv("DEFAULT_MAX_TOKENS", 4096))
DEFAULT_ALLOWED_FAILS = int(os.getenv("DEFAULT_ALLOWED_FAILS", 3))
DEFAULT_REDIS_SYNC_INTERVAL = int(os.getenv("DEFAULT_REDIS_SYNC_INTERVAL", 1))
DEFAULT_COOLDOWN_TIME_SECONDS = int(os.getenv("DEFAULT_COOLDOWN_TIME_SECONDS", 5))
DEFAULT_REPLICATE_POLLING_RETRIES = int(
    os.getenv("DEFAULT_REPLICATE_POLLING_RETRIES", 5)
)
DEFAULT_REPLICATE_POLLING_DELAY_SECONDS = int(
    os.getenv("DEFAULT_REPLICATE_POLLING_DELAY_SECONDS", 1)
)
DEFAULT_IMAGE_TOKEN_COUNT = int(os.getenv("DEFAULT_IMAGE_TOKEN_COUNT", 250))
DEFAULT_IMAGE_WIDTH = int(os.getenv("DEFAULT_IMAGE_WIDTH", 300))
DEFAULT_IMAGE_HEIGHT = int(os.getenv("DEFAULT_IMAGE_HEIGHT", 300))
MAX_SIZE_PER_ITEM_IN_MEMORY_CACHE_IN_KB = int(
    os.getenv("MAX_SIZE_PER_ITEM_IN_MEMORY_CACHE_IN_KB", 1024)
)  # 1MB = 1024KB
SINGLE_DEPLOYMENT_TRAFFIC_FAILURE_THRESHOLD = int(
    os.getenv("SINGLE_DEPLOYMENT_TRAFFIC_FAILURE_THRESHOLD", 1000)
)  # Minimum number of requests to consider "reasonable traffic". Used for single-deployment cooldown logic.

DEFAULT_REASONING_EFFORT_DISABLE_THINKING_BUDGET = int(
    os.getenv("DEFAULT_REASONING_EFFORT_DISABLE_THINKING_BUDGET", 0)
)
DEFAULT_REASONING_EFFORT_LOW_THINKING_BUDGET = int(
    os.getenv("DEFAULT_REASONING_EFFORT_LOW_THINKING_BUDGET", 1024)
)
DEFAULT_REASONING_EFFORT_MEDIUM_THINKING_BUDGET = int(
    os.getenv("DEFAULT_REASONING_EFFORT_MEDIUM_THINKING_BUDGET", 2048)
)
DEFAULT_REASONING_EFFORT_HIGH_THINKING_BUDGET = int(
    os.getenv("DEFAULT_REASONING_EFFORT_HIGH_THINKING_BUDGET", 4096)
)
MAX_TOKEN_TRIMMING_ATTEMPTS = int(
    os.getenv("MAX_TOKEN_TRIMMING_ATTEMPTS", 10)
)  # Maximum number of attempts to trim the message


########## Networking constants ##############################################################
_DEFAULT_TTL_FOR_HTTPX_CLIENTS = 3600  # 1 hour, re-use the same httpx client for 1 hour

########### v2 Architecture constants for managing writing updates to the database ###########
REDIS_UPDATE_BUFFER_KEY = "litellm_spend_update_buffer"
REDIS_DAILY_SPEND_UPDATE_BUFFER_KEY = "litellm_daily_spend_update_buffer"
REDIS_DAILY_TEAM_SPEND_UPDATE_BUFFER_KEY = "litellm_daily_team_spend_update_buffer"
REDIS_DAILY_TAG_SPEND_UPDATE_BUFFER_KEY = "litellm_daily_tag_spend_update_buffer"
MAX_REDIS_BUFFER_DEQUEUE_COUNT = int(os.getenv("MAX_REDIS_BUFFER_DEQUEUE_COUNT", 100))
MAX_SIZE_IN_MEMORY_QUEUE = int(os.getenv("MAX_SIZE_IN_MEMORY_QUEUE", 10000))
MAX_IN_MEMORY_QUEUE_FLUSH_COUNT = int(
    os.getenv("MAX_IN_MEMORY_QUEUE_FLUSH_COUNT", 1000)
)
###############################################################################################
MINIMUM_PROMPT_CACHE_TOKEN_COUNT = int(
    os.getenv("MINIMUM_PROMPT_CACHE_TOKEN_COUNT", 1024)
)  # minimum number of tokens to cache a prompt by Anthropic
DEFAULT_TRIM_RATIO = float(
    os.getenv("DEFAULT_TRIM_RATIO", 0.75)
)  # default ratio of tokens to trim from the end of a prompt
HOURS_IN_A_DAY = int(os.getenv("HOURS_IN_A_DAY", 24))
DAYS_IN_A_WEEK = int(os.getenv("DAYS_IN_A_WEEK", 7))
DAYS_IN_A_MONTH = int(os.getenv("DAYS_IN_A_MONTH", 28))
DAYS_IN_A_YEAR = int(os.getenv("DAYS_IN_A_YEAR", 365))
REPLICATE_MODEL_NAME_WITH_ID_LENGTH = int(
    os.getenv("REPLICATE_MODEL_NAME_WITH_ID_LENGTH", 64)
)
#### TOKEN COUNTING ####
FUNCTION_DEFINITION_TOKEN_COUNT = int(os.getenv("FUNCTION_DEFINITION_TOKEN_COUNT", 9))
SYSTEM_MESSAGE_TOKEN_COUNT = int(os.getenv("SYSTEM_MESSAGE_TOKEN_COUNT", 4))
TOOL_CHOICE_OBJECT_TOKEN_COUNT = int(os.getenv("TOOL_CHOICE_OBJECT_TOKEN_COUNT", 4))
DEFAULT_MOCK_RESPONSE_PROMPT_TOKEN_COUNT = int(
    os.getenv("DEFAULT_MOCK_RESPONSE_PROMPT_TOKEN_COUNT", 10)
)
DEFAULT_MOCK_RESPONSE_COMPLETION_TOKEN_COUNT = int(
    os.getenv("DEFAULT_MOCK_RESPONSE_COMPLETION_TOKEN_COUNT", 20)
)
MAX_SHORT_SIDE_FOR_IMAGE_HIGH_RES = int(
    os.getenv("MAX_SHORT_SIDE_FOR_IMAGE_HIGH_RES", 768)
)
MAX_LONG_SIDE_FOR_IMAGE_HIGH_RES = int(
    os.getenv("MAX_LONG_SIDE_FOR_IMAGE_HIGH_RES", 2000)
)
MAX_TILE_WIDTH = int(os.getenv("MAX_TILE_WIDTH", 512))
MAX_TILE_HEIGHT = int(os.getenv("MAX_TILE_HEIGHT", 512))
OPENAI_FILE_SEARCH_COST_PER_1K_CALLS = float(
    os.getenv("OPENAI_FILE_SEARCH_COST_PER_1K_CALLS", 2.5 / 1000)
)
# Azure OpenAI Assistants feature costs
# Source: https://azure.microsoft.com/en-us/pricing/details/cognitive-services/openai-service/
AZURE_FILE_SEARCH_COST_PER_GB_PER_DAY = float(
    os.getenv("AZURE_FILE_SEARCH_COST_PER_GB_PER_DAY", 0.1)  # $0.1 USD per 1 GB/Day
)
AZURE_CODE_INTERPRETER_COST_PER_SESSION = float(
    os.getenv("AZURE_CODE_INTERPRETER_COST_PER_SESSION", 0.03)  # $0.03 USD per 1 Session
)
AZURE_COMPUTER_USE_INPUT_COST_PER_1K_TOKENS = float(
    os.getenv("AZURE_COMPUTER_USE_INPUT_COST_PER_1K_TOKENS", 3.0)  # $0.003 USD per 1K Tokens
)
AZURE_COMPUTER_USE_OUTPUT_COST_PER_1K_TOKENS = float(
    os.getenv("AZURE_COMPUTER_USE_OUTPUT_COST_PER_1K_TOKENS", 12.0)  # $0.012 USD per 1K Tokens
)
AZURE_VECTOR_STORE_COST_PER_GB_PER_DAY = float(
    os.getenv("AZURE_VECTOR_STORE_COST_PER_GB_PER_DAY", 0.1)  # $0.1 USD per 1 GB/Day (same as file search)
)
MIN_NON_ZERO_TEMPERATURE = float(os.getenv("MIN_NON_ZERO_TEMPERATURE", 0.0001))
#### RELIABILITY ####
REPEATED_STREAMING_CHUNK_LIMIT = int(
    os.getenv("REPEATED_STREAMING_CHUNK_LIMIT", 100)
)  # catch if model starts looping the same chunk while streaming. Uses high default to prevent false positives.
DEFAULT_MAX_LRU_CACHE_SIZE = int(os.getenv("DEFAULT_MAX_LRU_CACHE_SIZE", 16))
INITIAL_RETRY_DELAY = float(os.getenv("INITIAL_RETRY_DELAY", 0.5))
MAX_RETRY_DELAY = float(os.getenv("MAX_RETRY_DELAY", 8.0))
JITTER = float(os.getenv("JITTER", 0.75))
DEFAULT_IN_MEMORY_TTL = int(
    os.getenv("DEFAULT_IN_MEMORY_TTL", 5)
)  # default time to live for the in-memory cache
DEFAULT_POLLING_INTERVAL = float(
    os.getenv("DEFAULT_POLLING_INTERVAL", 0.03)
)  # default polling interval for the scheduler
AZURE_OPERATION_POLLING_TIMEOUT = int(os.getenv("AZURE_OPERATION_POLLING_TIMEOUT", 120))
REDIS_SOCKET_TIMEOUT = float(os.getenv("REDIS_SOCKET_TIMEOUT", 0.1))
REDIS_CONNECTION_POOL_TIMEOUT = int(os.getenv("REDIS_CONNECTION_POOL_TIMEOUT", 5))
NON_LLM_CONNECTION_TIMEOUT = int(
    os.getenv("NON_LLM_CONNECTION_TIMEOUT", 15)
)  # timeout for adjacent services (e.g. jwt auth)
MAX_EXCEPTION_MESSAGE_LENGTH = int(os.getenv("MAX_EXCEPTION_MESSAGE_LENGTH", 2000))
BEDROCK_MAX_POLICY_SIZE = int(os.getenv("BEDROCK_MAX_POLICY_SIZE", 75))
REPLICATE_POLLING_DELAY_SECONDS = float(
    os.getenv("REPLICATE_POLLING_DELAY_SECONDS", 0.5)
)
DEFAULT_ANTHROPIC_CHAT_MAX_TOKENS = int(
    os.getenv("DEFAULT_ANTHROPIC_CHAT_MAX_TOKENS", 4096)
)
TOGETHER_AI_4_B = int(os.getenv("TOGETHER_AI_4_B", 4))
TOGETHER_AI_8_B = int(os.getenv("TOGETHER_AI_8_B", 8))
TOGETHER_AI_21_B = int(os.getenv("TOGETHER_AI_21_B", 21))
TOGETHER_AI_41_B = int(os.getenv("TOGETHER_AI_41_B", 41))
TOGETHER_AI_80_B = int(os.getenv("TOGETHER_AI_80_B", 80))
TOGETHER_AI_110_B = int(os.getenv("TOGETHER_AI_110_B", 110))
TOGETHER_AI_EMBEDDING_150_M = int(os.getenv("TOGETHER_AI_EMBEDDING_150_M", 150))
TOGETHER_AI_EMBEDDING_350_M = int(os.getenv("TOGETHER_AI_EMBEDDING_350_M", 350))
QDRANT_SCALAR_QUANTILE = float(os.getenv("QDRANT_SCALAR_QUANTILE", 0.99))
QDRANT_VECTOR_SIZE = int(os.getenv("QDRANT_VECTOR_SIZE", 1536))
CACHED_STREAMING_CHUNK_DELAY = float(os.getenv("CACHED_STREAMING_CHUNK_DELAY", 0.02))
MAX_SIZE_PER_ITEM_IN_MEMORY_CACHE_IN_KB = int(
    os.getenv("MAX_SIZE_PER_ITEM_IN_MEMORY_CACHE_IN_KB", 512)
)
DEFAULT_MAX_TOKENS_FOR_TRITON = int(os.getenv("DEFAULT_MAX_TOKENS_FOR_TRITON", 2000))
#### Networking settings ####
request_timeout: float = float(os.getenv("REQUEST_TIMEOUT", 6000))  # time in seconds
STREAM_SSE_DONE_STRING: str = "[DONE]"
STREAM_SSE_DATA_PREFIX: str = "data: "
### SPEND TRACKING ###
DEFAULT_REPLICATE_GPU_PRICE_PER_SECOND = float(
    os.getenv("DEFAULT_REPLICATE_GPU_PRICE_PER_SECOND", 0.001400)
)  # price per second for a100 80GB
FIREWORKS_AI_56_B_MOE = int(os.getenv("FIREWORKS_AI_56_B_MOE", 56))
FIREWORKS_AI_176_B_MOE = int(os.getenv("FIREWORKS_AI_176_B_MOE", 176))
FIREWORKS_AI_4_B = int(os.getenv("FIREWORKS_AI_4_B", 4))
FIREWORKS_AI_16_B = int(os.getenv("FIREWORKS_AI_16_B", 16))
FIREWORKS_AI_80_B = int(os.getenv("FIREWORKS_AI_80_B", 80))
#### Logging callback constants ####
REDACTED_BY_LITELM_STRING = "REDACTED_BY_LITELM"
MAX_LANGFUSE_INITIALIZED_CLIENTS = int(
    os.getenv("MAX_LANGFUSE_INITIALIZED_CLIENTS", 50)
)
DD_TRACER_STREAMING_CHUNK_YIELD_RESOURCE = os.getenv(
    "DD_TRACER_STREAMING_CHUNK_YIELD_RESOURCE", "streaming.chunk.yield"
)

############### LLM Provider Constants ###############
### ANTHROPIC CONSTANTS ###
ANTHROPIC_WEB_SEARCH_TOOL_MAX_USES = {
    "low": 1,
    "medium": 5,
    "high": 10,
}
DEFAULT_IMAGE_ENDPOINT_MODEL = "dall-e-2"

LITELLM_CHAT_PROVIDERS = [
    "openai",
    "openai_like",
    "xai",
    "custom_openai",
    "text-completion-openai",
    "cohere",
    "cohere_chat",
    "clarifai",
    "anthropic",
    "anthropic_text",
    "replicate",
    "huggingface",
    "together_ai",
    "datarobot",
    "openrouter",
    "vertex_ai",
    "vertex_ai_beta",
    "gemini",
    "ai21",
    "baseten",
    "azure",
    "azure_text",
    "azure_ai",
    "sagemaker",
    "sagemaker_chat",
    "bedrock",
    "vllm",
    "nlp_cloud",
    "petals",
    "oobabooga",
    "ollama",
    "ollama_chat",
    "deepinfra",
    "perplexity",
    "mistral",
    "groq",
    "nvidia_nim",
    "cerebras",
    "ai21_chat",
    "volcengine",
    "codestral",
    "text-completion-codestral",
    "deepseek",
    "sambanova",
    "maritalk",
    "cloudflare",
    "fireworks_ai",
    "friendliai",
    "watsonx",
    "watsonx_text",
    "triton",
    "predibase",
    "databricks",
    "empower",
    "github",
    "custom",
    "litellm_proxy",
    "hosted_vllm",
    "llamafile",
    "lm_studio",
    "galadriel",
    "github_copilot",  # GitHub Copilot Chat API
    "novita",
    "meta_llama",
    "featherless_ai",
    "nscale",
    "nebius",
]

LITELLM_EMBEDDING_PROVIDERS_SUPPORTING_INPUT_ARRAY_OF_TOKENS = [
    "openai",
    "azure",
    "hosted_vllm",
    "nebius",
]


OPENAI_CHAT_COMPLETION_PARAMS = [
    "functions",
    "function_call",
    "temperature",
    "temperature",
    "top_p",
    "n",
    "stream",
    "stream_options",
    "stop",
    "max_completion_tokens",
    "modalities",
    "prediction",
    "audio",
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
    "parallel_tool_calls",
    "logprobs",
    "top_logprobs",
    "reasoning_effort",
    "extra_headers",
    "thinking",
    "web_search_options",
]

OPENAI_TRANSCRIPTION_PARAMS = [
    "language",
    "response_format",
    "timestamp_granularities",
]

OPENAI_EMBEDDING_PARAMS = ["dimensions", "encoding_format", "user"]

DEFAULT_EMBEDDING_PARAM_VALUES = {
    **{k: None for k in OPENAI_EMBEDDING_PARAMS},
    "model": None,
    "custom_llm_provider": "",
    "input": None,
}

DEFAULT_CHAT_COMPLETION_PARAM_VALUES = {
    "functions": None,
    "function_call": None,
    "temperature": None,
    "top_p": None,
    "n": None,
    "stream": None,
    "stream_options": None,
    "stop": None,
    "max_tokens": None,
    "max_completion_tokens": None,
    "modalities": None,
    "prediction": None,
    "audio": None,
    "presence_penalty": None,
    "frequency_penalty": None,
    "logit_bias": None,
    "user": None,
    "model": None,
    "custom_llm_provider": "",
    "response_format": None,
    "seed": None,
    "tools": None,
    "tool_choice": None,
    "max_retries": None,
    "logprobs": None,
    "top_logprobs": None,
    "extra_headers": None,
    "api_version": None,
    "parallel_tool_calls": None,
    "drop_params": None,
    "allowed_openai_params": None,
    "additional_drop_params": None,
    "messages": None,
    "reasoning_effort": None,
    "thinking": None,
    "web_search_options": None,
}

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
    "https://api.friendli.ai/serverless/v1",
    "api.sambanova.ai/v1",
    "api.x.ai/v1",
    "api.galadriel.ai/v1",
    "api.llama.com/compat/v1/",
    "api.featherless.ai/v1",
    "inference.api.nscale.com/v1",
    "api.studio.nebius.ai/v1",
]


openai_compatible_providers: List = [
    "anyscale",
    "groq",
    "nvidia_nim",
    "cerebras",
    "sambanova",
    "ai21_chat",
    "ai21",
    "volcengine",
    "codestral",
    "deepseek",
    "deepinfra",
    "perplexity",
    "xinference",
    "xai",
    "together_ai",
    "fireworks_ai",
    "empower",
    "friendliai",
    "azure_ai",
    "github",
    "litellm_proxy",
    "hosted_vllm",
    "llamafile",
    "lm_studio",
    "galadriel",
    "github_copilot",  # GitHub Copilot Chat API
    "novita",
    "meta_llama",
    "featherless_ai",
    "nscale",
    "nebius",
]
openai_text_completion_compatible_providers: List = (
    [  # providers that support `/v1/completions`
        "together_ai",
        "fireworks_ai",
        "hosted_vllm",
        "meta_llama",
        "llamafile",
        "featherless_ai",
        "nebius",
    ]
)
_openai_like_providers: List = [
    "predibase",
    "databricks",
    "watsonx",
]  # private helper. similar to openai but require some custom auth / endpoint handling, so can't use the openai sdk
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

featherless_ai_models: List = [
    "featherless-ai/Qwerky-72B",
    "featherless-ai/Qwerky-QwQ-32B",
    "Qwen/Qwen2.5-72B-Instruct",
    "all-hands/openhands-lm-32b-v0.1",
    "Qwen/Qwen2.5-Coder-32B-Instruct",
    "deepseek-ai/DeepSeek-V3-0324",
    "mistralai/Mistral-Small-24B-Instruct-2501",
    "mistralai/Mistral-Nemo-Instruct-2407",
    "ProdeusUnity/Stellar-Odyssey-12b-v0.0",
]

nebius_models: List = [
    "Qwen/Qwen3-235B-A22B",
    "Qwen/Qwen3-30B-A3B-fast",
    "Qwen/Qwen3-32B",
    "Qwen/Qwen3-14B",
    "nvidia/Llama-3_1-Nemotron-Ultra-253B-v1",
    "deepseek-ai/DeepSeek-V3-0324",
    "deepseek-ai/DeepSeek-V3-0324-fast",
    "deepseek-ai/DeepSeek-R1",
    "deepseek-ai/DeepSeek-R1-fast",
    "meta-llama/Llama-3.3-70B-Instruct-fast",
    "Qwen/Qwen2.5-32B-Instruct-fast",
    "Qwen/Qwen2.5-Coder-32B-Instruct-fast",
]

nebius_embedding_models: List = [
    "BAAI/bge-en-icl",
    "BAAI/bge-multilingual-gemma2",
    "intfloat/e5-mistral-7b-instruct",
]

BEDROCK_INVOKE_PROVIDERS_LITERAL = Literal[
    "cohere",
    "anthropic",
    "mistral",
    "amazon",
    "meta",
    "llama",
    "ai21",
    "nova",
    "deepseek_r1",
]

open_ai_embedding_models: List = ["text-embedding-ada-002"]
cohere_embedding_models: List = [
    "embed-v4.0",
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

known_tokenizer_config = {
    "mistralai/Mistral-7B-Instruct-v0.1": {
        "tokenizer": {
            "chat_template": "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token + ' ' }}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}",
            "bos_token": "<s>",
            "eos_token": "</s>",
        },
        "status": "success",
    },
    "meta-llama/Meta-Llama-3-8B-Instruct": {
        "tokenizer": {
            "chat_template": "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}",
            "bos_token": "<|begin_of_text|>",
            "eos_token": "",
        },
        "status": "success",
    },
    "deepseek-r1/deepseek-r1-7b-instruct": {
        "tokenizer": {
            "add_bos_token": True,
            "add_eos_token": False,
            "bos_token": {
                "__type": "AddedToken",
                "content": "<｜begin▁of▁sentence｜>",
                "lstrip": False,
                "normalized": True,
                "rstrip": False,
                "single_word": False,
            },
            "clean_up_tokenization_spaces": False,
            "eos_token": {
                "__type": "AddedToken",
                "content": "<｜end▁of▁sentence｜>",
                "lstrip": False,
                "normalized": True,
                "rstrip": False,
                "single_word": False,
            },
            "legacy": True,
            "model_max_length": 16384,
            "pad_token": {
                "__type": "AddedToken",
                "content": "<｜end▁of▁sentence｜>",
                "lstrip": False,
                "normalized": True,
                "rstrip": False,
                "single_word": False,
            },
            "sp_model_kwargs": {},
            "unk_token": None,
            "tokenizer_class": "LlamaTokenizerFast",
            "chat_template": "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% set ns = namespace(is_first=false, is_tool=false, is_output_first=true, system_prompt='') %}{%- for message in messages %}{%- if message['role'] == 'system' %}{% set ns.system_prompt = message['content'] %}{%- endif %}{%- endfor %}{{bos_token}}{{ns.system_prompt}}{%- for message in messages %}{%- if message['role'] == 'user' %}{%- set ns.is_tool = false -%}{{'<｜User｜>' + message['content']}}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is none %}{%- set ns.is_tool = false -%}{%- for tool in message['tool_calls']%}{%- if not ns.is_first %}{{'<｜Assistant｜><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '<｜tool▁call▁end｜>'}}{%- set ns.is_first = true -%}{%- else %}{{'\\n' + '<｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '<｜tool▁call▁end｜>'}}{{'<｜tool▁calls▁end｜><｜end▁of▁sentence｜>'}}{%- endif %}{%- endfor %}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is not none %}{%- if ns.is_tool %}{{'<｜tool▁outputs▁end｜>' + message['content'] + '<｜end▁of▁sentence｜>'}}{%- set ns.is_tool = false -%}{%- else %}{% set content = message['content'] %}{% if '</think>' in content %}{% set content = content.split('</think>')[-1] %}{% endif %}{{'<｜Assistant｜>' + content + '<｜end▁of▁sentence｜>'}}{%- endif %}{%- endif %}{%- if message['role'] == 'tool' %}{%- set ns.is_tool = true -%}{%- if ns.is_output_first %}{{'<｜tool▁outputs▁begin｜><｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- set ns.is_output_first = false %}{%- else %}{{'\\n<｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- endif %}{%- endif %}{%- endfor -%}{% if ns.is_tool %}{{'<｜tool▁outputs▁end｜>'}}{% endif %}{% if add_generation_prompt and not ns.is_tool %}{{'<｜Assistant｜><think>\\n'}}{% endif %}",
        },
        "status": "success",
    },
}


OPENAI_FINISH_REASONS = ["stop", "length", "function_call", "content_filter", "null"]
HUMANLOOP_PROMPT_CACHE_TTL_SECONDS = int(
    os.getenv("HUMANLOOP_PROMPT_CACHE_TTL_SECONDS", 60)
)  # 1 minute
RESPONSE_FORMAT_TOOL_NAME = "json_tool_call"  # default tool name used when converting response format to tool call

########################### Logging Callback Constants ###########################
AZURE_STORAGE_MSFT_VERSION = "2019-07-07"
PROMETHEUS_BUDGET_METRICS_REFRESH_INTERVAL_MINUTES = int(
    os.getenv("PROMETHEUS_BUDGET_METRICS_REFRESH_INTERVAL_MINUTES", 5)
)
MCP_TOOL_NAME_PREFIX = "mcp_tool"
MAXIMUM_TRACEBACK_LINES_TO_LOG = int(os.getenv("MAXIMUM_TRACEBACK_LINES_TO_LOG", 100))

# Headers to control callbacks
X_LITELLM_DISABLE_CALLBACKS = "x-litellm-disable-callbacks"
LITELLM_METADATA_FIELD = "litellm_metadata"
OLD_LITELLM_METADATA_FIELD = "metadata"

########################### LiteLLM Proxy Specific Constants ###########################
########################################################################################
MAX_SPENDLOG_ROWS_TO_QUERY = int(
    os.getenv("MAX_SPENDLOG_ROWS_TO_QUERY", 1_000_000)
)  # if spendLogs has more than 1M rows, do not query the DB
DEFAULT_SOFT_BUDGET = float(
    os.getenv("DEFAULT_SOFT_BUDGET", 50.0)
)  # by default all litellm proxy keys have a soft budget of 50.0
# makes it clear this is a rate limit error for a litellm virtual key
RATE_LIMIT_ERROR_MESSAGE_FOR_VIRTUAL_KEY = "LiteLLM Virtual Key user_api_key_hash"

# pass through route constansts
BEDROCK_AGENT_RUNTIME_PASS_THROUGH_ROUTES = [
    "agents/",
    "knowledgebases/",
    "flows/",
    "retrieveAndGenerate/",
    "rerank/",
    "generateQuery/",
    "optimize-prompt/",
]
BASE_MCP_ROUTE = "/mcp"

BATCH_STATUS_POLL_INTERVAL_SECONDS = int(
    os.getenv("BATCH_STATUS_POLL_INTERVAL_SECONDS", 3600)
)  # 1 hour
BATCH_STATUS_POLL_MAX_ATTEMPTS = int(
    os.getenv("BATCH_STATUS_POLL_MAX_ATTEMPTS", 24)
)  # for 24 hours

HEALTH_CHECK_TIMEOUT_SECONDS = int(
    os.getenv("HEALTH_CHECK_TIMEOUT_SECONDS", 60)
)  # 60 seconds

UI_SESSION_TOKEN_TEAM_ID = "litellm-dashboard"
LITELLM_PROXY_ADMIN_NAME = "default_user_id"

########################### CLI SSO AUTHENTICATION CONSTANTS ###########################
LITELLM_CLI_SOURCE_IDENTIFIER = "litellm-cli"
LITELLM_CLI_SESSION_TOKEN_PREFIX = "litellm-session-token"

########################### DB CRON JOB NAMES ###########################
DB_SPEND_UPDATE_JOB_NAME = "db_spend_update_job"
PROMETHEUS_EMIT_BUDGET_METRICS_JOB_NAME = "prometheus_emit_budget_metrics"
SPEND_LOG_CLEANUP_JOB_NAME = "spend_log_cleanup"
SPEND_LOG_RUN_LOOPS = int(os.getenv("SPEND_LOG_RUN_LOOPS", 500))
SPEND_LOG_CLEANUP_BATCH_SIZE = int(os.getenv("SPEND_LOG_CLEANUP_BATCH_SIZE", 1000))
DEFAULT_CRON_JOB_LOCK_TTL_SECONDS = int(
    os.getenv("DEFAULT_CRON_JOB_LOCK_TTL_SECONDS", 60)
)  # 1 minute
PROXY_BUDGET_RESCHEDULER_MIN_TIME = int(
    os.getenv("PROXY_BUDGET_RESCHEDULER_MIN_TIME", 597)
)
PROXY_BUDGET_RESCHEDULER_MAX_TIME = int(
    os.getenv("PROXY_BUDGET_RESCHEDULER_MAX_TIME", 605)
)
PROXY_BATCH_WRITE_AT = int(os.getenv("PROXY_BATCH_WRITE_AT", 10))  # in seconds
DEFAULT_HEALTH_CHECK_INTERVAL = int(
    os.getenv("DEFAULT_HEALTH_CHECK_INTERVAL", 300)
)  # 5 minutes
PROMETHEUS_FALLBACK_STATS_SEND_TIME_HOURS = int(
    os.getenv("PROMETHEUS_FALLBACK_STATS_SEND_TIME_HOURS", 9)
)
DEFAULT_MODEL_CREATED_AT_TIME = int(
    os.getenv("DEFAULT_MODEL_CREATED_AT_TIME", 1677610602)
)  # returns on `/models` endpoint
DEFAULT_SLACK_ALERTING_THRESHOLD = int(
    os.getenv("DEFAULT_SLACK_ALERTING_THRESHOLD", 300)
)
MAX_TEAM_LIST_LIMIT = int(os.getenv("MAX_TEAM_LIST_LIMIT", 20))
DEFAULT_PROMPT_INJECTION_SIMILARITY_THRESHOLD = float(
    os.getenv("DEFAULT_PROMPT_INJECTION_SIMILARITY_THRESHOLD", 0.7)
)
LENGTH_OF_LITELLM_GENERATED_KEY = int(os.getenv("LENGTH_OF_LITELLM_GENERATED_KEY", 16))
SECRET_MANAGER_REFRESH_INTERVAL = int(
    os.getenv("SECRET_MANAGER_REFRESH_INTERVAL", 86400)
)
LITELLM_SETTINGS_SAFE_DB_OVERRIDES = ["default_internal_user_params"]
SPECIAL_LITELLM_AUTH_TOKEN = ["ui-token"]
DEFAULT_MANAGEMENT_OBJECT_IN_MEMORY_CACHE_TTL = int(
    os.getenv("DEFAULT_MANAGEMENT_OBJECT_IN_MEMORY_CACHE_TTL", 60)
)

# Sentry Scrubbing Configuration
SENTRY_DENYLIST = [
    # API Keys and Tokens
    "api_key", "token", "key", "secret", "password", "auth", "credential",
    "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "AZURE_API_KEY", "COHERE_API_KEY",
    "REPLICATE_API_KEY", "HUGGINGFACE_API_KEY", "TOGETHERAI_API_KEY",
    "CLOUDFLARE_API_KEY", "BASETEN_KEY", "OPENROUTER_KEY", "DATAROBOT_API_TOKEN",
    "FIREWORKS_API_KEY", "FIREWORKS_AI_API_KEY", "FIREWORKSAI_API_KEY",
    # Database and Connection Strings
    "database_url", "redis_url", "connection_string",
    # Authentication and Security
    "master_key", "LITELLM_MASTER_KEY", "auth_token", "jwt_token", "private_key",
    "SLACK_WEBHOOK_URL", "webhook_url", "LANGFUSE_SECRET_KEY",
    # Email Configuration
    "SMTP_PASSWORD", "SMTP_USERNAME", "email_password",
    # Cloud Provider Credentials
    "aws_access_key", "aws_secret_key", "gcp_credentials",
    "azure_credentials", "HCP_VAULT_TOKEN", "CIRCLE_OIDC_TOKEN",
    # Proxy and Environment Settings
    "proxy_url", "proxy_key", "environment_variables"
]
SENTRY_PII_DENYLIST = [
    "user_id", "email", "phone", "address", "ip_address",
    "SMTP_SENDER_EMAIL", "TEST_EMAIL_ADDRESS"
]