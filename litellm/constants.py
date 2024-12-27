ROUTER_MAX_FALLBACKS = 5
DEFAULT_BATCH_SIZE = 512
DEFAULT_FLUSH_INTERVAL_SECONDS = 5
DEFAULT_MAX_RETRIES = 2
DEFAULT_REPLICATE_POLLING_RETRIES = 5
DEFAULT_REPLICATE_POLLING_DELAY_SECONDS = 1
DEFAULT_IMAGE_TOKEN_COUNT = 250
DEFAULT_IMAGE_WIDTH = 300
DEFAULT_IMAGE_HEIGHT = 300
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
    "lm_studio",
    "galadriel",
]

RESPONSE_FORMAT_TOOL_NAME = "json_tool_call"  # default tool name used when converting response format to tool call

########################### Logging Callback Constants ###########################
AZURE_STORAGE_MSFT_VERSION = "2019-07-07"

########################### LiteLLM Proxy Specific Constants ###########################
########################################################################################
MAX_SPENDLOG_ROWS_TO_QUERY = (
    1_000_000  # if spendLogs has more than 1M rows, do not query the DB
)
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

BATCH_STATUS_POLL_INTERVAL_SECONDS = 3600  # 1 hour
BATCH_STATUS_POLL_MAX_ATTEMPTS = 24  # for 24 hours
