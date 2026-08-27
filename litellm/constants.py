import os
import sys
from types import MappingProxyType
from typing import Final, Literal

from litellm.litellm_core_utils.env_utils import get_env_int, get_env_int_in_range, get_env_int_or_none

DEFAULT_HEALTH_CHECK_PROMPT: Final = str(os.getenv("DEFAULT_HEALTH_CHECK_PROMPT", "test from litellm"))
AZURE_DEFAULT_RESPONSES_API_VERSION: Final = str(os.getenv("AZURE_DEFAULT_RESPONSES_API_VERSION", "preview"))
ROUTER_MAX_FALLBACKS: Final = int(os.getenv("ROUTER_MAX_FALLBACKS", 5))
ROUTER_FALLBACK_ERROR_DETAIL_MAX_CHARS: Final = 2000
DEFAULT_BATCH_SIZE: Final = int(os.getenv("DEFAULT_BATCH_SIZE", 512))
DEFAULT_FLUSH_INTERVAL_SECONDS: Final = int(os.getenv("DEFAULT_FLUSH_INTERVAL_SECONDS", 5))
DEFAULT_S3_FLUSH_INTERVAL_SECONDS: Final = int(os.getenv("DEFAULT_S3_FLUSH_INTERVAL_SECONDS", 10))
DEFAULT_S3_BATCH_SIZE: Final = int(os.getenv("DEFAULT_S3_BATCH_SIZE", 512))
DEFAULT_SQS_FLUSH_INTERVAL_SECONDS: Final = int(os.getenv("DEFAULT_SQS_FLUSH_INTERVAL_SECONDS", 10))
DEFAULT_NUM_WORKERS_LITELLM_PROXY: Final = int(os.getenv("DEFAULT_NUM_WORKERS_LITELLM_PROXY", 1))
DYNAMIC_RATE_LIMIT_ERROR_THRESHOLD_PER_MINUTE = int(os.getenv("DYNAMIC_RATE_LIMIT_ERROR_THRESHOLD_PER_MINUTE", 1))
DEFAULT_SQS_BATCH_SIZE: Final = int(os.getenv("DEFAULT_SQS_BATCH_SIZE", 512))
SQS_SEND_MESSAGE_ACTION: Final = "SendMessage"
SQS_API_VERSION: Final = "2012-11-05"
DEFAULT_MAX_RETRIES: Final = int(os.getenv("DEFAULT_MAX_RETRIES", 2))
# Max records accepted in one POST /v1/callbacks/logs batch. Bounds the blast
# radius: each record fans out to spend logs + every callback integration.
MAX_CALLBACK_LOG_RECORDS: Final = 1000
DEFAULT_MAX_RECURSE_DEPTH: Final = int(os.getenv("DEFAULT_MAX_RECURSE_DEPTH", 100))
DEFAULT_MAX_RECURSE_DEPTH_SENSITIVE_DATA_MASKER = int(os.getenv("DEFAULT_MAX_RECURSE_DEPTH_SENSITIVE_DATA_MASKER", 10))
DEFAULT_FAILURE_THRESHOLD_PERCENT: Final = float(
    os.getenv("DEFAULT_FAILURE_THRESHOLD_PERCENT", 0.5)
)  # default cooldown a deployment if 50% of requests fail in a given minute
DEFAULT_MAX_TOKENS: Final = int(os.getenv("DEFAULT_MAX_TOKENS", 4096))
DEFAULT_ALLOWED_FAILS: Final = int(os.getenv("DEFAULT_ALLOWED_FAILS", 3))
DEFAULT_REDIS_SYNC_INTERVAL: Final = int(os.getenv("DEFAULT_REDIS_SYNC_INTERVAL", 1))
DEFAULT_COOLDOWN_TIME_SECONDS: Final = int(os.getenv("DEFAULT_COOLDOWN_TIME_SECONDS", 5))
DEFAULT_REPLICATE_POLLING_RETRIES: Final = int(os.getenv("DEFAULT_REPLICATE_POLLING_RETRIES", 5))
DEFAULT_REPLICATE_POLLING_DELAY_SECONDS: Final = int(os.getenv("DEFAULT_REPLICATE_POLLING_DELAY_SECONDS", 1))
DEFAULT_IMAGE_TOKEN_COUNT: Final = int(os.getenv("DEFAULT_IMAGE_TOKEN_COUNT", 250))

# Maximum wall-clock seconds a streaming response is allowed to run.
# Streams exceeding this duration are terminated with a Timeout error.
# None (default) = no limit.  Set env var to a number of seconds to enable globally.
_max_stream_duration_env: Final = os.getenv("LITELLM_MAX_STREAMING_DURATION_SECONDS", None)
LITELLM_MAX_STREAMING_DURATION_SECONDS: Final = (
    float(_max_stream_duration_env) if _max_stream_duration_env is not None else None
)

# Maximum number of base64 characters to keep in logging payloads.
# Data URIs exceeding this are replaced with a size placeholder.
# Set to 0 to disable truncation.
MAX_BASE64_LENGTH_FOR_LOGGING: Final = int(os.getenv("MAX_BASE64_LENGTH_FOR_LOGGING", 64))
REDACTED_BY_LITELLM: Final = "redacted-by-litellm"
# in-memory stand-in handed to provider converters for redacted arguments; never stored
REDACTED_TOOL_CALL_ARGUMENTS_PLACEHOLDER: Final = "{}"

MAX_STRING_LENGTH_STDOUT_LOG: Final = get_env_int("MAX_STRING_LENGTH_STDOUT_LOG", 4096)

# When true, adds detailed per-phase timing breakdown headers to responses.
# Headers: x-litellm-timing-{pre-processing,llm-api,post-processing,message-copy}-ms
LITELLM_DETAILED_TIMING: Final = os.getenv("LITELLM_DETAILED_TIMING", "false").lower() == "true"

# Model cost map validation constants
MODEL_COST_MAP_MIN_MODEL_COUNT: Final = int(
    os.getenv("MODEL_COST_MAP_MIN_MODEL_COUNT", 50)
)  # Minimum number of models a fetched cost map must contain to be considered valid
MODEL_COST_MAP_MAX_SHRINK_RATIO: Final = float(
    os.getenv("MODEL_COST_MAP_MAX_SHRINK_RATIO", 0.5)
)  # Maximum allowed shrinkage ratio vs local backup (0.5 = reject if fetched map is <50% of backup)
DEFAULT_IMAGE_WIDTH: Final = int(os.getenv("DEFAULT_IMAGE_WIDTH", 300))
DEFAULT_IMAGE_HEIGHT: Final = int(os.getenv("DEFAULT_IMAGE_HEIGHT", 300))
# Maximum size for image URL downloads in MB (default 50MB, set to 0 to disable limit)
# This prevents memory issues from downloading very large images
# Maps to OpenAI's 50 MB payload limit - requests with images exceeding this size will be rejected
# Set MAX_IMAGE_URL_DOWNLOAD_SIZE_MB=0 to disable image URL handling entirely
MAX_IMAGE_URL_DOWNLOAD_SIZE_MB: Final = float(os.getenv("MAX_IMAGE_URL_DOWNLOAD_SIZE_MB", 50))
MAX_SIZE_PER_ITEM_IN_MEMORY_CACHE_IN_KB: Final = int(
    os.getenv("MAX_SIZE_PER_ITEM_IN_MEMORY_CACHE_IN_KB", 1024)
)  # 1MB = 1024KB
# Surrogate-repair fallback in _read_request_body runs two full-body re.sub passes
# that block the event loop on multi-MB malformed bodies. Skip the repair above this
# size and raise the existing 400 immediately. Set to 0 to disable the cap.
MAX_REQUEST_BODY_SIZE_TO_REPAIR_MB: Final = get_env_int("MAX_REQUEST_BODY_SIZE_TO_REPAIR_MB", 1)
SINGLE_DEPLOYMENT_TRAFFIC_FAILURE_THRESHOLD: Final = int(
    os.getenv("SINGLE_DEPLOYMENT_TRAFFIC_FAILURE_THRESHOLD", 1000)
)  # Minimum number of requests to consider "reasonable traffic". Used for single-deployment cooldown logic.
DEFAULT_FAILURE_THRESHOLD_MINIMUM_REQUESTS: Final = int(
    os.getenv("DEFAULT_FAILURE_THRESHOLD_MINIMUM_REQUESTS", 5)
)  # Minimum number of requests before applying error rate cooldown. Prevents cooldown from triggering on first failure.

DEFAULT_REASONING_EFFORT_DISABLE_THINKING_BUDGET = int(os.getenv("DEFAULT_REASONING_EFFORT_DISABLE_THINKING_BUDGET", 0))

# MCP Semantic Tool Filter Defaults
DEFAULT_MCP_SEMANTIC_FILTER_EMBEDDING_MODEL: Final = str(
    os.getenv("DEFAULT_MCP_SEMANTIC_FILTER_EMBEDDING_MODEL", "text-embedding-3-small")
)
DEFAULT_MCP_SEMANTIC_FILTER_TOP_K: Final = int(os.getenv("DEFAULT_MCP_SEMANTIC_FILTER_TOP_K", 10))
DEFAULT_MCP_SEMANTIC_FILTER_SIMILARITY_THRESHOLD: Final = float(
    os.getenv("DEFAULT_MCP_SEMANTIC_FILTER_SIMILARITY_THRESHOLD", 0.3)
)
MAX_MCP_SEMANTIC_FILTER_TOOLS_HEADER_LENGTH: Final = int(os.getenv("MAX_MCP_SEMANTIC_FILTER_TOOLS_HEADER_LENGTH", 150))

DEFAULT_AUTO_ROUTER_MAX_INPUT_CHARS: Final = 2000

# Semantic Guard Defaults
DEFAULT_SEMANTIC_GUARD_EMBEDDING_MODEL: Final = str(
    os.getenv("DEFAULT_SEMANTIC_GUARD_EMBEDDING_MODEL", "text-embedding-3-small")
)
DEFAULT_SEMANTIC_GUARD_SIMILARITY_THRESHOLD = float(os.getenv("DEFAULT_SEMANTIC_GUARD_SIMILARITY_THRESHOLD", 0.75))

# MCP OAuth2 Client Credentials Defaults
MCP_OAUTH2_TOKEN_EXPIRY_BUFFER_SECONDS: Final = int(os.getenv("MCP_OAUTH2_TOKEN_EXPIRY_BUFFER_SECONDS", "60"))
MCP_OAUTH2_TOKEN_CACHE_MAX_SIZE: Final = int(os.getenv("MCP_OAUTH2_TOKEN_CACHE_MAX_SIZE", "200"))
MCP_OAUTH2_TOKEN_CACHE_DEFAULT_TTL: Final = int(os.getenv("MCP_OAUTH2_TOKEN_CACHE_DEFAULT_TTL", "3600"))

# Default npm cache directory for STDIO MCP servers.
# npm/npx needs a writable cache dir; in containers the default (~/.npm)
# may not exist or be read-only. /tmp is always writable.
MCP_NPM_CACHE_DIR: Final = os.getenv("MCP_NPM_CACHE_DIR", "/tmp/.npm_mcp_cache")
MCP_OAUTH2_TOKEN_CACHE_MIN_TTL: Final = int(os.getenv("MCP_OAUTH2_TOKEN_CACHE_MIN_TTL", "10"))

# Per-user OAuth token Redis cache (for server-side token storage)
MCP_PER_USER_TOKEN_REDIS_KEY_PREFIX: Final = "mcp:per_user_token"
MCP_PER_USER_TOKEN_DEFAULT_TTL: Final = int(
    os.getenv("MCP_PER_USER_TOKEN_DEFAULT_TTL", "43200")  # 12 hours
)
MCP_PER_USER_TOKEN_EXPIRY_BUFFER_SECONDS: Final = int(os.getenv("MCP_PER_USER_TOKEN_EXPIRY_BUFFER_SECONDS", "60"))

# MCP timeout defaults (seconds). Override via env vars for slow/custom MCP servers.
MCP_CLIENT_TIMEOUT: Final = float(os.getenv("LITELLM_MCP_CLIENT_TIMEOUT", "60.0"))
MCP_TOOL_LISTING_TIMEOUT: Final = float(os.getenv("LITELLM_MCP_TOOL_LISTING_TIMEOUT", "30.0"))
MCP_METADATA_TIMEOUT: Final = float(os.getenv("LITELLM_MCP_METADATA_TIMEOUT", "10.0"))
MCP_HEALTH_CHECK_TIMEOUT: Final = float(os.getenv("LITELLM_MCP_HEALTH_CHECK_TIMEOUT", "10.0"))

# Allowlist of commands permitted for MCP stdio transport.
# Prevents arbitrary command execution via /mcp-rest/test/* endpoints or server creation.
# Note: allowlisted runtimes can still execute code via args (e.g. python -c "...").
# This is an accepted residual risk since these endpoints require PROXY_ADMIN.
# Extend via LITELLM_MCP_STDIO_EXTRA_COMMANDS env var (comma-separated).
_MCP_STDIO_EXTRA_COMMANDS: Final = os.getenv("LITELLM_MCP_STDIO_EXTRA_COMMANDS", "")
MCP_STDIO_ALLOWED_COMMANDS: Final[frozenset] = frozenset(
    {"npx", "uvx", "python", "python3", "node", "docker", "deno"} | (set(_MCP_STDIO_EXTRA_COMMANDS.split(",")) - {""})
)

# MCP OAuth2 Token Exchange (OBO) Defaults
MCP_TOKEN_EXCHANGE_CACHE_MAX_SIZE: Final = int(os.getenv("MCP_TOKEN_EXCHANGE_CACHE_MAX_SIZE", "500"))

LITELLM_UI_ALLOW_HEADERS: Final = [
    "x-litellm-semantic-filter",
    "x-litellm-semantic-filter-tools",
    "x-litellm-adaptive-router-model",
    "x-litellm-applied-guardrails",
    "x-litellm-guardrail-scan-id",
    "x-litellm-cache-key",
]

# Gemini model-specific minimal thinking budget constants
DEFAULT_REASONING_EFFORT_MINIMAL_THINKING_BUDGET_GEMINI_2_5_FLASH: Final = int(
    os.getenv("DEFAULT_REASONING_EFFORT_MINIMAL_THINKING_BUDGET_GEMINI_2_5_FLASH", 1)
)
DEFAULT_REASONING_EFFORT_MINIMAL_THINKING_BUDGET_GEMINI_2_5_PRO: Final = int(
    os.getenv("DEFAULT_REASONING_EFFORT_MINIMAL_THINKING_BUDGET_GEMINI_2_5_PRO", 128)
)
DEFAULT_REASONING_EFFORT_MINIMAL_THINKING_BUDGET_GEMINI_2_5_FLASH_LITE: Final = int(
    os.getenv("DEFAULT_REASONING_EFFORT_MINIMAL_THINKING_BUDGET_GEMINI_2_5_FLASH_LITE", 512)
)

# Maximum number of callbacks that can be registered
# This prevents callbacks from exponentially growing and consuming CPU resources
# Override with LITELLM_MAX_CALLBACKS env var for large deployments (e.g., many teams with guardrails)
MAX_CALLBACKS: Final = get_env_int("LITELLM_MAX_CALLBACKS", 100)

# Metadata key recording which pre_call guardrails the proxy loop already ran,
# so the deployment-level hook does not re-run them for the same request
PRE_CALL_EXECUTED_GUARDRAILS_KEY: Final = "_pre_call_executed_guardrails"

# Generic fallback for unknown models
DEFAULT_REASONING_EFFORT_MINIMAL_THINKING_BUDGET: Final = int(
    os.getenv("DEFAULT_REASONING_EFFORT_MINIMAL_THINKING_BUDGET", 128)
)

# Provider-specific API base URLs
XAI_API_BASE: Final = "https://api.x.ai/v1"
OPEN_SANDBOX_API_BASE_ENV_VAR: Final = "OPEN_SANDBOX_API_BASE"
OPEN_SANDBOX_API_KEY_ENV_VAR: Final = "OPEN_SANDBOX_API_KEY"
OPEN_SANDBOX_DEFAULT_TEMPLATE: Final = "opensandbox/code-interpreter:v1.1.0"
_OPEN_SANDBOX_FALLBACK_ENTRYPOINT: Final = "/opt/code-interpreter/code-interpreter.sh"
OPEN_SANDBOX_DEFAULT_ENTRYPOINT: Final = (_OPEN_SANDBOX_FALLBACK_ENTRYPOINT,)
OPEN_SANDBOX_DEFAULT_LANGUAGE: Final = "python"
OPEN_SANDBOX_DEFAULT_CPU_LIMIT: Final = "1"
OPEN_SANDBOX_DEFAULT_MEMORY_LIMIT: Final = "2Gi"
OPEN_SANDBOX_EXECD_PORT: Final = 44772
OPEN_SANDBOX_DEFAULT_TIMEOUT: Final = 300
OPEN_SANDBOX_READY_TIMEOUT: Final = 30.0
OPEN_SANDBOX_POLL_INTERVAL: Final = 0.2

DEFAULT_REASONING_EFFORT_LOW_THINKING_BUDGET = int(os.getenv("DEFAULT_REASONING_EFFORT_LOW_THINKING_BUDGET", 1024))
DEFAULT_REASONING_EFFORT_MEDIUM_THINKING_BUDGET: Final = int(
    os.getenv("DEFAULT_REASONING_EFFORT_MEDIUM_THINKING_BUDGET", 2048)
)
DEFAULT_REASONING_EFFORT_HIGH_THINKING_BUDGET = int(os.getenv("DEFAULT_REASONING_EFFORT_HIGH_THINKING_BUDGET", 4096))
DEFAULT_REASONING_EFFORT_XHIGH_THINKING_BUDGET = int(os.getenv("DEFAULT_REASONING_EFFORT_XHIGH_THINKING_BUDGET", 8192))
DEFAULT_REASONING_EFFORT_MAX_THINKING_BUDGET = int(os.getenv("DEFAULT_REASONING_EFFORT_MAX_THINKING_BUDGET", 16384))
MAX_TOKEN_TRIMMING_ATTEMPTS: Final = int(
    os.getenv("MAX_TOKEN_TRIMMING_ATTEMPTS", 10)
)  # Maximum number of attempts to trim the message

RUNWAYML_DEFAULT_API_VERSION: Final = str(os.getenv("RUNWAYML_DEFAULT_API_VERSION", "2024-11-06"))
RUNWAYML_POLLING_TIMEOUT = int(os.getenv("RUNWAYML_POLLING_TIMEOUT", 600))  # 10 minutes default for image generation

########## Networking constants ##############################################################
_DEFAULT_TTL_FOR_HTTPX_CLIENTS: Final = 3600  # 1 hour, re-use the same httpx client for 1 hour

# The earliest an evicted, litellm-created client may be closed. A request handed the
# client just before eviction is still using it, so nothing is closed inside this window;
# past it, the client is closed once it reports no connection in flight.
EVICTED_LLM_CLIENT_CLOSE_GRACE_SECONDS: Final = 900

# How many evicted clients may be queued for closing at once. Past this, an evicted client
# is left to the collector rather than letting a cache-churning workload grow the queue
# without bound. Each queued entry is ~100 bytes and comes due within one grace window.
EVICTED_LLM_CLIENT_CLOSE_MAX_PENDING: Final = 10_000

# Aiohttp connection pooling - prevents memory leaks from unbounded connection growth
# Set to 0 for unlimited (not recommended for production)
AIOHTTP_CONNECTOR_LIMIT: Final = int(os.getenv("AIOHTTP_CONNECTOR_LIMIT", 1000))
AIOHTTP_CONNECTOR_LIMIT_PER_HOST: Final = int(os.getenv("AIOHTTP_CONNECTOR_LIMIT_PER_HOST", 500))
AIOHTTP_KEEPALIVE_TIMEOUT: Final = int(os.getenv("AIOHTTP_KEEPALIVE_TIMEOUT", 120))
AIOHTTP_TTL_DNS_CACHE: Final = int(os.getenv("AIOHTTP_TTL_DNS_CACHE", 300))
# TCP keep-alive (SO_KEEPALIVE) — opt-in. Required when running behind NAT/LBs
# whose idle timeout is shorter than provider response timeouts (e.g. AWS NAT
# Gateway: 350s vs OpenAI/Azure: 600s). Without this, the kernel sends nothing
# during a long provider call and the NAT reaps the flow before the response
# arrives. Enabling SO_KEEPALIVE makes the kernel emit TCP probes that reset
# the NAT idle timer.
AIOHTTP_SO_KEEPALIVE: Final = os.getenv("AIOHTTP_SO_KEEPALIVE", "False").lower() == "true"
AIOHTTP_TCP_KEEPIDLE: Final = int(os.getenv("AIOHTTP_TCP_KEEPIDLE", 60))
AIOHTTP_TCP_KEEPINTVL: Final = int(os.getenv("AIOHTTP_TCP_KEEPINTVL", 30))
AIOHTTP_TCP_KEEPCNT: Final = int(os.getenv("AIOHTTP_TCP_KEEPCNT", 5))
# enable_cleanup_closed is only needed for Python versions with the SSL leak bug
# Fixed in Python 3.12.7+ and 3.13.1+ (see https://github.com/python/cpython/pull/118960)
# Reference: https://github.com/aio-libs/aiohttp/blob/master/aiohttp/connector.py#L74-L78
AIOHTTP_NEEDS_CLEANUP_CLOSED: Final = (3, 13, 0) <= sys.version_info < (
    3,
    13,
    1,
) or sys.version_info < (3, 12, 7)

# WebSocket constants
# Default to None (unlimited) to match OpenAI's official agents SDK behavior
# https://github.com/openai/openai-agents-python/blob/cf1b933660e44fd37b4350c41febab8221801409/src/agents/realtime/openai_realtime.py#L235
_max_size_env: Final = os.getenv("REALTIME_WEBSOCKET_MAX_MESSAGE_SIZE_BYTES")
REALTIME_WEBSOCKET_MAX_MESSAGE_SIZE_BYTES: Final = int(_max_size_env) if _max_size_env is not None else None
REALTIME_CREDENTIAL_RESOLUTION_TIMEOUT_SECONDS: Final = float(
    os.getenv("REALTIME_CREDENTIAL_RESOLUTION_TIMEOUT_SECONDS", "20.0")
)

# RFC 6455 caps the close frame payload at 125 bytes, 2 of which carry the status code
WEBSOCKET_CLOSE_REASON_MAX_BYTES: Final = 123

# SSL/TLS cipher configuration for faster handshakes
# Strategy: Strongly prefer fast modern ciphers, but allow fallback to commonly supported ones
# This balances performance with broad compatibility
DEFAULT_SSL_CIPHERS: Final = os.getenv(
    "LITELLM_SSL_CIPHERS",
    # Priority 1: TLS 1.3 ciphers (fastest, ~50ms handshake)
    "TLS_AES_256_GCM_SHA384:"  # Fastest observed in testing
    "TLS_AES_128_GCM_SHA256:"  # Slightly faster than 256-bit
    "TLS_CHACHA20_POLY1305_SHA256:"  # Fast on ARM/mobile
    # Priority 2: TLS 1.2 ECDHE+GCM (fast, ~100ms handshake, widely supported)
    "ECDHE-RSA-AES256-GCM-SHA384:"
    "ECDHE-RSA-AES128-GCM-SHA256:"
    "ECDHE-ECDSA-AES256-GCM-SHA384:"
    "ECDHE-ECDSA-AES128-GCM-SHA256:"
    # Priority 3: Additional modern ciphers (good balance)
    "ECDHE-RSA-CHACHA20-POLY1305:"
    "ECDHE-ECDSA-CHACHA20-POLY1305:"
    # Priority 4: Widely compatible fallbacks (slower but universally supported)
    "ECDHE-RSA-AES256-SHA384:"  # Common fallback
    "ECDHE-RSA-AES128-SHA256:"  # Very widely supported
    "AES256-GCM-SHA384:"  # Non-PFS fallback (compatibility)
    "AES128-GCM-SHA256",  # Last resort (maximum compatibility)
)

########### v2 Architecture constants for managing writing updates to the database ###########
REDIS_UPDATE_BUFFER_KEY: Final = "litellm_spend_update_buffer"
REDIS_DAILY_SPEND_UPDATE_BUFFER_KEY: Final = "litellm_daily_spend_update_buffer"
REDIS_DAILY_TEAM_SPEND_UPDATE_BUFFER_KEY: Final = "litellm_daily_team_spend_update_buffer"
REDIS_DAILY_ORG_SPEND_UPDATE_BUFFER_KEY: Final = "litellm_daily_org_spend_update_buffer"
REDIS_DAILY_END_USER_SPEND_UPDATE_BUFFER_KEY: Final = "litellm_daily_end_user_spend_update_buffer"
REDIS_DAILY_AGENT_SPEND_UPDATE_BUFFER_KEY: Final = "litellm_daily_agent_spend_update_buffer"
REDIS_DAILY_TAG_SPEND_UPDATE_BUFFER_KEY: Final = "litellm_daily_tag_spend_update_buffer"
MAX_REDIS_BUFFER_DEQUEUE_COUNT: Final = int(os.getenv("MAX_REDIS_BUFFER_DEQUEUE_COUNT", 100))
# Bounds asyncio.Queue() instances (log queues, spend update queues, etc.) to prevent unbounded memory growth
LITELLM_ASYNCIO_QUEUE_MAXSIZE: Final = int(os.getenv("LITELLM_ASYNCIO_QUEUE_MAXSIZE", 1000))
TOOL_POLICY_CACHE_TTL_SECONDS: Final = int(os.getenv("TOOL_POLICY_CACHE_TTL_SECONDS", 60))
GUARDRAIL_SCANNED_MESSAGES_CACHE_TTL_SECONDS: Final = int(
    os.getenv("GUARDRAIL_SCANNED_MESSAGES_CACHE_TTL_SECONDS", 24 * 60 * 60)
)
BEDROCK_APPLY_GUARDRAIL_CHUNK_BUDGET_CHARS: Final = 25_000
# Aggregation threshold: default to 80% of the asyncio queue maxsize so the check can always trigger.
# Must be < LITELLM_ASYNCIO_QUEUE_MAXSIZE; if set higher the aggregation logic will never fire.
MAX_SIZE_IN_MEMORY_QUEUE: Final = int(os.getenv("MAX_SIZE_IN_MEMORY_QUEUE", int(LITELLM_ASYNCIO_QUEUE_MAXSIZE * 0.8)))
MAX_IN_MEMORY_QUEUE_FLUSH_COUNT: Final = int(os.getenv("MAX_IN_MEMORY_QUEUE_FLUSH_COUNT", 1000))
###############################################################################################
# Providers will not cache a prefix below a minimum size. That minimum is per-model, not global:
# Anthropic's ranges from 512 to 4096 depending on the model, and can differ per platform for the
# same model. The real minimum is resolved from `prompt_cache_min_tokens` in the model cost map;
# this value is only the fallback for models the cost map has no entry for, and doubles as a global
# escape hatch when `MINIMUM_PROMPT_CACHE_TOKEN_COUNT` is explicitly set.
MINIMUM_PROMPT_CACHE_TOKEN_COUNT_OVERRIDE: Final[int | None] = get_env_int_or_none("MINIMUM_PROMPT_CACHE_TOKEN_COUNT")
DEFAULT_MINIMUM_PROMPT_CACHE_TOKEN_COUNT: Final = 1024
MINIMUM_PROMPT_CACHE_TOKEN_COUNT: Final = (
    MINIMUM_PROMPT_CACHE_TOKEN_COUNT_OVERRIDE
    if MINIMUM_PROMPT_CACHE_TOKEN_COUNT_OVERRIDE is not None
    else DEFAULT_MINIMUM_PROMPT_CACHE_TOKEN_COUNT
)
DEFAULT_TRIM_RATIO: Final = float(
    os.getenv("DEFAULT_TRIM_RATIO", 0.75)
)  # default ratio of tokens to trim from the end of a prompt
HOURS_IN_A_DAY: Final = int(os.getenv("HOURS_IN_A_DAY", 24))
DAYS_IN_A_WEEK: Final = int(os.getenv("DAYS_IN_A_WEEK", 7))
DAYS_IN_A_MONTH: Final = int(os.getenv("DAYS_IN_A_MONTH", 28))
DAYS_IN_A_YEAR: Final = int(os.getenv("DAYS_IN_A_YEAR", 365))
REPLICATE_MODEL_NAME_WITH_ID_LENGTH: Final = int(os.getenv("REPLICATE_MODEL_NAME_WITH_ID_LENGTH", 64))
#### TOKEN COUNTING ####
FUNCTION_DEFINITION_TOKEN_COUNT: Final = int(os.getenv("FUNCTION_DEFINITION_TOKEN_COUNT", 9))
SYSTEM_MESSAGE_TOKEN_COUNT: Final = int(os.getenv("SYSTEM_MESSAGE_TOKEN_COUNT", 4))
TOOL_CHOICE_OBJECT_TOKEN_COUNT: Final = int(os.getenv("TOOL_CHOICE_OBJECT_TOKEN_COUNT", 4))
DEFAULT_MOCK_RESPONSE_PROMPT_TOKEN_COUNT: Final = int(os.getenv("DEFAULT_MOCK_RESPONSE_PROMPT_TOKEN_COUNT", 10))
DEFAULT_MOCK_RESPONSE_COMPLETION_TOKEN_COUNT: Final = int(os.getenv("DEFAULT_MOCK_RESPONSE_COMPLETION_TOKEN_COUNT", 20))
MAX_SHORT_SIDE_FOR_IMAGE_HIGH_RES: Final = int(os.getenv("MAX_SHORT_SIDE_FOR_IMAGE_HIGH_RES", 768))
MAX_LONG_SIDE_FOR_IMAGE_HIGH_RES: Final = int(os.getenv("MAX_LONG_SIDE_FOR_IMAGE_HIGH_RES", 2000))
# tiktoken's BPE merge loop is quadratic in the length of a single regex piece, so a long run of one
# repeated character (dot leaders, whitespace, zero-padded base64) can take minutes on a multi-MB payload.
# Encoding in chunks makes the cost linear, at a drift of at most ~1 token per chunk boundary. The upper
# bound keeps a misconfigured chunk size from restoring the quadratic cost this exists to remove.
TIKTOKEN_ENCODE_MAX_CHUNK_SIZE_CHARS: Final = 4096
TIKTOKEN_ENCODE_CHUNK_SIZE_CHARS: Final = get_env_int_in_range(
    "TIKTOKEN_ENCODE_CHUNK_SIZE_CHARS",
    default=1024,
    minimum=1,
    maximum=TIKTOKEN_ENCODE_MAX_CHUNK_SIZE_CHARS,
)
MAX_TILE_WIDTH: Final = int(os.getenv("MAX_TILE_WIDTH", 512))
MAX_TILE_HEIGHT: Final = int(os.getenv("MAX_TILE_HEIGHT", 512))
OPENAI_FILE_SEARCH_COST_PER_1K_CALLS: Final = float(os.getenv("OPENAI_FILE_SEARCH_COST_PER_1K_CALLS", 2.5 / 1000))
GROQ_BROWSER_VISIT_WEBSITE_COST_PER_CALL: Final = 1.0 / 1000
# Azure OpenAI Assistants feature costs
# Source: https://azure.microsoft.com/en-us/pricing/details/cognitive-services/openai-service/
AZURE_FILE_SEARCH_COST_PER_GB_PER_DAY: Final = float(
    os.getenv("AZURE_FILE_SEARCH_COST_PER_GB_PER_DAY", 0.1)  # $0.1 USD per 1 GB/Day
)
AZURE_COMPUTER_USE_INPUT_COST_PER_1K_TOKENS: Final = float(
    os.getenv("AZURE_COMPUTER_USE_INPUT_COST_PER_1K_TOKENS", 3.0)  # $0.003 USD per 1K Tokens
)
AZURE_COMPUTER_USE_OUTPUT_COST_PER_1K_TOKENS: Final = float(
    os.getenv("AZURE_COMPUTER_USE_OUTPUT_COST_PER_1K_TOKENS", 12.0)  # $0.012 USD per 1K Tokens
)
AZURE_VECTOR_STORE_COST_PER_GB_PER_DAY: Final = float(
    os.getenv("AZURE_VECTOR_STORE_COST_PER_GB_PER_DAY", 0.1)  # $0.1 USD per 1 GB/Day (same as file search)
)
MIN_NON_ZERO_TEMPERATURE: Final = float(os.getenv("MIN_NON_ZERO_TEMPERATURE", 0.0001))
#### RELIABILITY ####
REPEATED_STREAMING_CHUNK_LIMIT: Final = int(
    os.getenv("REPEATED_STREAMING_CHUNK_LIMIT", 100)
)  # catch if model starts looping the same chunk while streaming. Uses high default to prevent false positives.
# Shared maxsize for functools.lru_cache usage across hot paths.
# Defaulted to 64 to avoid cache thrash in multi-model production workloads.
DEFAULT_MAX_LRU_CACHE_SIZE: Final = int(os.getenv("DEFAULT_MAX_LRU_CACHE_SIZE", 64))
_REALTIME_BODY_CACHE_SIZE = 1000  # Keep realtime helper caches bounded; workloads rarely exceed 1k models/intents
INITIAL_RETRY_DELAY: Final = float(os.getenv("INITIAL_RETRY_DELAY", 0.5))
MAX_RETRY_DELAY: Final = float(os.getenv("MAX_RETRY_DELAY", 8.0))
JITTER: Final = float(os.getenv("JITTER", 0.75))
DEFAULT_IN_MEMORY_TTL = int(os.getenv("DEFAULT_IN_MEMORY_TTL", 5))  # default time to live for the in-memory cache
DEFAULT_MAX_REDIS_BATCH_CACHE_SIZE: Final = int(
    os.getenv("DEFAULT_MAX_REDIS_BATCH_CACHE_SIZE", 1000)
)  # default max size for redis batch cache
DEFAULT_POLLING_INTERVAL: Final = float(
    os.getenv("DEFAULT_POLLING_INTERVAL", 0.03)
)  # default polling interval for the scheduler
AZURE_OPERATION_POLLING_TIMEOUT: Final = int(os.getenv("AZURE_OPERATION_POLLING_TIMEOUT", 120))
AZURE_DOCUMENT_INTELLIGENCE_API_VERSION: Final = str(os.getenv("AZURE_DOCUMENT_INTELLIGENCE_API_VERSION", "2024-11-30"))
AZURE_DOCUMENT_INTELLIGENCE_DEFAULT_DPI: Final = int(os.getenv("AZURE_DOCUMENT_INTELLIGENCE_DEFAULT_DPI", 96))
REDIS_SOCKET_TIMEOUT: Final = float(os.getenv("REDIS_SOCKET_TIMEOUT", 0.1))
CACHE_WRITE_SHUTDOWN_FLUSH_TIMEOUT_SECONDS: Final[float] = 5.0
REDIS_CONNECTION_POOL_TIMEOUT: Final = int(os.getenv("REDIS_CONNECTION_POOL_TIMEOUT", 5))
REDIS_CIRCUIT_BREAKER_FAILURE_THRESHOLD: Final = int(os.getenv("REDIS_CIRCUIT_BREAKER_FAILURE_THRESHOLD", 5))
REDIS_CIRCUIT_BREAKER_RECOVERY_TIMEOUT: Final = int(os.getenv("REDIS_CIRCUIT_BREAKER_RECOVERY_TIMEOUT", 60))
REDIS_CIRCUIT_BREAKER_ENABLED: Final = os.getenv("REDIS_CIRCUIT_BREAKER_ENABLED", "true").lower() == "true"
# Seconds of idle before a Redis cluster connection is validated with a PING and
# reconnected if dead, so a connection silently dropped by a cluster restart
# (e.g. ElastiCache Serverless maintenance) is not reused while broken
REDIS_CLUSTER_HEALTH_CHECK_INTERVAL: Final = 25
# Default Redis major version to assume when version cannot be determined
# Using 7 as it's the modern version that supports LPOP with count parameter
DEFAULT_REDIS_MAJOR_VERSION: Final = int(os.getenv("DEFAULT_REDIS_MAJOR_VERSION", 7))
NON_LLM_CONNECTION_TIMEOUT: Final = int(
    os.getenv("NON_LLM_CONNECTION_TIMEOUT", 15)
)  # timeout for adjacent services (e.g. jwt auth)
MAX_EXCEPTION_MESSAGE_LENGTH: Final = int(os.getenv("MAX_EXCEPTION_MESSAGE_LENGTH", 2000))
MAX_STRING_LENGTH_PROMPT_IN_DB: Final = int(os.getenv("MAX_STRING_LENGTH_PROMPT_IN_DB", 2048))
BEDROCK_MAX_POLICY_SIZE: Final = int(os.getenv("BEDROCK_MAX_POLICY_SIZE", 75))
# One entry per distinct AWS credential-argument set. Per-user cost attribution passes the attributed
# identity as aws_session_name, so this bounds how many attributed identities keep a cached STS session.
BEDROCK_IAM_CACHE_MAX_ENTRIES: Final = 1000
# Single-flight lock stripes over that cache. Only keys landing on the same stripe wait for each
# other, so a burst of distinct identities still resolves its credentials in parallel.
BEDROCK_IAM_CACHE_FETCH_LOCK_STRIPES: Final = 64
# Retire a cached STS credential this many seconds before AWS expires it, so a request that reads it
# still has a usable credential for the whole call.
STS_CREDENTIAL_EXPIRY_SAFETY_MARGIN_SECONDS: Final = 60
BEDROCK_MIN_THINKING_BUDGET_TOKENS: Final = int(os.getenv("BEDROCK_MIN_THINKING_BUDGET_TOKENS", 1024))
# Anthropic's Messages API rejects thinking.budget_tokens < 1024.
ANTHROPIC_MIN_THINKING_BUDGET_TOKENS: Final = 1024
REPLICATE_POLLING_DELAY_SECONDS: Final = float(os.getenv("REPLICATE_POLLING_DELAY_SECONDS", 0.5))
DEFAULT_ANTHROPIC_CHAT_MAX_TOKENS: Final = int(os.getenv("DEFAULT_ANTHROPIC_CHAT_MAX_TOKENS", 4096))
DEFAULT_OCI_CHAT_MAX_TOKENS: Final = 4096
TOGETHER_AI_4_B: Final = int(os.getenv("TOGETHER_AI_4_B", 4))
TOGETHER_AI_8_B: Final = int(os.getenv("TOGETHER_AI_8_B", 8))
TOGETHER_AI_21_B: Final = int(os.getenv("TOGETHER_AI_21_B", 21))
TOGETHER_AI_41_B: Final = int(os.getenv("TOGETHER_AI_41_B", 41))
TOGETHER_AI_80_B: Final = int(os.getenv("TOGETHER_AI_80_B", 80))
TOGETHER_AI_110_B: Final = int(os.getenv("TOGETHER_AI_110_B", 110))
TOGETHER_AI_EMBEDDING_150_M: Final = int(os.getenv("TOGETHER_AI_EMBEDDING_150_M", 150))
TOGETHER_AI_EMBEDDING_350_M: Final = int(os.getenv("TOGETHER_AI_EMBEDDING_350_M", 350))
QDRANT_SCALAR_QUANTILE: Final = float(os.getenv("QDRANT_SCALAR_QUANTILE", 0.99))
QDRANT_VECTOR_SIZE: Final = int(os.getenv("QDRANT_VECTOR_SIZE", 1536))
CACHED_STREAMING_CHUNK_DELAY: Final = float(os.getenv("CACHED_STREAMING_CHUNK_DELAY", 0.02))
AUDIO_SPEECH_CHUNK_SIZE: Final = int(
    os.getenv("AUDIO_SPEECH_CHUNK_SIZE", 8192)
)  # chunk_size for audio speech streaming. Balance between latency and memory usage
DEFAULT_MAX_TOKENS_FOR_TRITON: Final = int(os.getenv("DEFAULT_MAX_TOKENS_FOR_TRITON", 2000))
#### Networking settings ####
# Sentinel used when `REQUEST_TIMEOUT` is unset: `litellm.request_timeout` keeps this
# value so longer-running surfaces (Router `timeout or litellm.request_timeout`,
# speech/TTS, responses, vector stores, etc.) get a long HTTP deadline. Chat
# `completion()` maps this sentinel down to 600s when the caller did not set a
# per-request/model timeout—see ``CompletionTimeout.resolve`` in completion_timeout.py. MCP uses
# dedicated timeouts (e.g. `MCP_CLIENT_TIMEOUT`), not `request_timeout`.
DEFAULT_REQUEST_TIMEOUT_SECONDS: Final[float] = 6000.0
# Pair used for default httpx clients when no custom timeout is passed: read/write
# deadline and connect handshake (see ``http_handler`` cached handler paths).
COMPLETION_HTTP_FALLBACK_SECONDS: Final[float] = 600.0
HTTP_HANDLER_CONNECT_TIMEOUT_SECONDS: Final[float] = 5.0
SEMANTIC_CACHE_EMBEDDING_TIMEOUT_SECONDS: Final[float] = float(
    os.getenv("SEMANTIC_CACHE_EMBEDDING_TIMEOUT_SECONDS", "5.0")
)
request_timeout: float = float(os.getenv("REQUEST_TIMEOUT", str(int(DEFAULT_REQUEST_TIMEOUT_SECONDS))))
request_timeout_explicitly_set: bool = "REQUEST_TIMEOUT" in os.environ
DEFAULT_A2A_AGENT_TIMEOUT: Final[float] = float(os.getenv("DEFAULT_A2A_AGENT_TIMEOUT", 6000))  # 10 minutes
# Patterns that indicate a localhost/internal URL in A2A agent cards that should be
# replaced with the original base_url. This is a common misconfiguration where
# developers deploy agents with development URLs in their agent cards.
LOCALHOST_URL_PATTERNS: Final[list[str]] = [
    "localhost",
    "127.0.0.1",
    "0.0.0.0",
    "[::1]",  # IPv6 localhost
]
# Patterns in error messages that indicate a connection failure
CONNECTION_ERROR_PATTERNS: Final[list[str]] = [
    "connect",
    "connection",
    "network",
    "refused",
]
STREAM_SSE_DONE_STRING: Final[str] = "[DONE]"
STREAM_SSE_DATA_PREFIX: Final[str] = "data: "
STREAM_SSE_KEEPALIVE_PING_CHUNK: Final[str] = 'event: ping\ndata: {"type": "ping"}\n\n'
STREAM_SSE_KEEPALIVE_PING_BYTES: Final[bytes] = STREAM_SSE_KEEPALIVE_PING_CHUNK.encode("utf-8")
### SPEND TRACKING ###
DEFAULT_REPLICATE_GPU_PRICE_PER_SECOND: Final = float(
    os.getenv("DEFAULT_REPLICATE_GPU_PRICE_PER_SECOND", 0.001400)
)  # price per second for a100 80GB
FIREWORKS_AI_56_B_MOE: Final = int(os.getenv("FIREWORKS_AI_56_B_MOE", 56))
FIREWORKS_AI_176_B_MOE: Final = int(os.getenv("FIREWORKS_AI_176_B_MOE", 176))
FIREWORKS_AI_4_B: Final = int(os.getenv("FIREWORKS_AI_4_B", 4))
FIREWORKS_AI_16_B: Final = int(os.getenv("FIREWORKS_AI_16_B", 16))
FIREWORKS_AI_80_B: Final = int(os.getenv("FIREWORKS_AI_80_B", 80))
#### Logging callback constants ####
REDACTED_BY_LITELM_STRING: Final = "REDACTED_BY_LITELM"
MAX_LANGFUSE_INITIALIZED_CLIENTS: Final = int(os.getenv("MAX_LANGFUSE_INITIALIZED_CLIENTS", 50))
LOGGING_WORKER_CONCURRENCY: Final = int(os.getenv("LOGGING_WORKER_CONCURRENCY", 100))  # Must be above 0
LOGGING_WORKER_MAX_QUEUE_SIZE: Final = int(os.getenv("LOGGING_WORKER_MAX_QUEUE_SIZE", 50_000))
LOGGING_WORKER_MAX_TIME_PER_COROUTINE: Final = float(os.getenv("LOGGING_WORKER_MAX_TIME_PER_COROUTINE", 20.0))
LOGGING_WORKER_CLEAR_PERCENTAGE: Final = int(
    os.getenv("LOGGING_WORKER_CLEAR_PERCENTAGE", 50)
)  # Percentage of queue to clear (default: 50%)
MAX_ITERATIONS_TO_CLEAR_QUEUE: Final = int(os.getenv("MAX_ITERATIONS_TO_CLEAR_QUEUE", 200))
MAX_TIME_TO_CLEAR_QUEUE: Final = float(os.getenv("MAX_TIME_TO_CLEAR_QUEUE", 5.0))
LOGGING_WORKER_AGGRESSIVE_CLEAR_COOLDOWN_SECONDS: Final = float(
    os.getenv("LOGGING_WORKER_AGGRESSIVE_CLEAR_COOLDOWN_SECONDS", 0.5)
)  # Cooldown time in seconds before allowing another aggressive clear (default: 0.5s)
LOGGING_EXECUTOR_MAX_THREADS: Final = get_env_int("LOGGING_EXECUTOR_MAX_THREADS", 100)
LOGGING_EXECUTOR_MAX_PENDING_TASKS: Final = get_env_int("LOGGING_EXECUTOR_MAX_PENDING_TASKS", 10_000)
LOGGING_EXECUTOR_DROPPED_TASK_LOG_INTERVAL_SECONDS: Final = 30.0
DD_TRACER_STREAMING_CHUNK_YIELD_RESOURCE: Final = os.getenv(
    "DD_TRACER_STREAMING_CHUNK_YIELD_RESOURCE", "streaming.chunk.yield"
)

LITELLM_HTTP_STATUS_CLIENT_DISCONNECTED: Final = 499

EMAIL_BUDGET_ALERT_TTL: Final = int(os.getenv("EMAIL_BUDGET_ALERT_TTL", 24 * 60 * 60))  # 24 hours in seconds
EMAIL_BUDGET_ALERT_MAX_SPEND_ALERT_PERCENTAGE: Final = float(
    os.getenv("EMAIL_BUDGET_ALERT_MAX_SPEND_ALERT_PERCENTAGE", 0.8)
)  # 80% of max budget
############### LLM Provider Constants ###############
### ANTHROPIC CONSTANTS ###
ANTHROPIC_TOKEN_COUNTING_BETA_VERSION = os.getenv("ANTHROPIC_TOKEN_COUNTING_BETA_VERSION", "token-counting-2024-11-01")
ANTHROPIC_SKILLS_API_BETA_VERSION: Final = "skills-2025-10-02"
ANTHROPIC_BATCHES_ROUTE: Final = "/v1/messages/batches"
VERTEX_BATCH_PREDICTION_JOBS_ROUTE: Final = "batchPredictionJobs"
ANTHROPIC_WEB_SEARCH_TOOL_MAX_USES: Final = {
    "low": 1,
    "medium": 5,
    "high": 10,
}

# LiteLLM standard web search tool name
# Used for web search interception across providers
LITELLM_WEB_SEARCH_TOOL_NAME: Final = "litellm_web_search"

DEFAULT_IMAGE_ENDPOINT_MODEL: Final = "dall-e-2"
DEFAULT_VIDEO_ENDPOINT_MODEL: Final = "sora-2"

DEFAULT_GOOGLE_VIDEO_DURATION_SECONDS: Final = int(os.getenv("DEFAULT_GOOGLE_VIDEO_DURATION_SECONDS", 8))

### DATAFORSEO CONSTANTS ###
DEFAULT_DATAFORSEO_LOCATION_CODE: Final = int(
    os.getenv("DEFAULT_DATAFORSEO_LOCATION_CODE", 2250)
)  # Default to France (2250) - lower number, commonly used location

LITELLM_CHAT_PROVIDERS: Final = [
    "openai",
    "openai_like",
    "bytez",
    "gdc",
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
    "helicone",
    "openrouter",
    "cometapi",
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
    "sagemaker_nova",
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
    "gigachat",
    "nvidia_nim",
    "cerebras",
    "baseten",
    "ai21_chat",
    "volcengine",
    "codestral",
    "text-completion-codestral",
    "text-completion-inception",
    "deepseek",
    "tencent",
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
    "gradient_ai",
    "github_copilot",  # GitHub Copilot Chat API
    "chatgpt",  # ChatGPT subscription API
    "novita",
    "meta_llama",
    "featherless_ai",
    "nscale",
    "nebius",
    "dashscope",
    "modelscope",
    "moonshot",
    "publicai",
    "v0",
    "heroku",
    "oci",
    "morph",
    "lambda_ai",
    "inception",
    "vercel_ai_gateway",
    "wandb",
    "ovhcloud",
    "lemonade",
    "docker_model_runner",
    "amazon_nova",
]

LITELLM_EMBEDDING_PROVIDERS_SUPPORTING_INPUT_ARRAY_OF_TOKENS: Final = [
    "openai",
    "azure",
    "hosted_vllm",
    "nebius",
]


OPENAI_CHAT_COMPLETION_PARAMS: Final = [
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
    "include_server_side_tool_invocations",
    "service_tier",
    "prompt_cache_key",
    "prompt_cache_retention",
    "safety_identifier",
    "verbosity",
    "store",
]

OPENAI_TRANSCRIPTION_PARAMS: Final = [
    "language",
    "response_format",
    "timestamp_granularities",
]

OPENAI_EMBEDDING_PARAMS: Final = ["dimensions", "encoding_format", "user"]

DEFAULT_EMBEDDING_PARAM_VALUES: Final = {
    **{k: None for k in OPENAI_EMBEDDING_PARAMS},
    "model": None,
    "custom_llm_provider": "",
    "input": None,
}

DEFAULT_CHAT_COMPLETION_PARAM_VALUES: Final = {
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
    "verbosity": None,
    "thinking": None,
    "web_search_options": None,
    "include_server_side_tool_invocations": None,
    "service_tier": None,
    "safety_identifier": None,
    "prompt_cache_key": None,
    "prompt_cache_retention": None,
    "store": None,
    "metadata": None,
    "context_management": None,
}

openai_compatible_endpoints: Final[list] = [
    "api.perplexity.ai",
    "api.endpoints.anyscale.com/v1",
    "api.deepinfra.com/v1/openai",
    "api.mistral.ai/v1",
    "codestral.mistral.ai/v1/chat/completions",
    "codestral.mistral.ai/v1/fim/completions",
    "api.groq.com/openai/v1",
    "https://integrate.api.nvidia.com/v1",
    "api.deepseek.com/v1",
    "api.together.ai/v1",
    "api.together.xyz/v1",
    "app.empower.dev/api/v1",
    "https://api.friendli.ai/serverless/v1",
    "api.sambanova.ai/v1",
    "api.x.ai/v1",
    "ollama.com",
    "api.galadriel.ai/v1",
    "api.llama.com/compat/v1/",
    "api.featherless.ai/v1",
    "inference.api.nscale.com/v1",
    "api.studio.nebius.ai/v1",
    "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    "https://api-inference.modelscope.cn/v1",
    "https://api.moonshot.ai/v1",
    "https://api.publicai.co/v1",
    "https://api.synthetic.new/openai/v1",
    "https://serverless.tensormesh.ai/v1",
    "https://api.stima.tech/v1",
    "https://nano-gpt.com/api/v1",
    "https://api.poe.com/v1",
    "https://llm.chutes.ai/v1/",
    "https://api.v0.dev/v1",
    "https://api.morphllm.com/v1",
    "https://api.lambda.ai/v1",
    "https://api.inceptionlabs.ai/v1",
    "https://api.hyperbolic.xyz/v1",
    "https://ai-gateway.helicone.ai/",
    "https://ai-gateway.vercel.sh/v1",
    "https://api.inference.wandb.ai/v1",
    "https://api.clarifai.com/v2/ext/openai/v1",
    "https://api.libertai.io/v1",
    "https://pinstripes.io/v1",
    "https://api.meta.ai/v1",
    "https://api.cognition.ai/v1",
    "https://api.scx.ai/v1",
]


openai_compatible_providers: Final[list] = [
    "anyscale",
    "groq",
    "nvidia_nim",
    "cerebras",
    "baseten",
    "sambanova",
    "ai21_chat",
    "ai21",
    "volcengine",
    "codestral",
    "deepseek",
    "tencent",
    "deepinfra",
    "perplexity",
    "xinference",
    "xai",
    "zai",
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
    "chatgpt",  # ChatGPT subscription API
    "novita",
    "meta_llama",
    "publicai",  # PublicAI - JSON-configured provider
    "synthetic",  # Synthetic - JSON-configured provider
    "tensormesh",  # Tensormesh - JSON-configured provider
    "apertis",  # Apertis - JSON-configured provider
    "nano-gpt",  # Nano-GPT - JSON-configured provider
    "poe",  # Poe - JSON-configured provider
    "chutes",  # Chutes - JSON-configured provider
    "parasail",  # Parasail - JSON-configured provider
    "libertai",  # LibertAI - JSON-configured provider
    "featherless_ai",
    "nscale",
    "nebius",
    "dashscope",
    "modelscope",
    "moonshot",
    "v0",
    "helicone",
    "morph",
    "lambda_ai",
    "inception",
    "hyperbolic",
    "vercel_ai_gateway",
    "aiml",
    "wandb",
    "cometapi",
    "clarifai",
    "docker_model_runner",
    "ragflow",
    "pinstripes",  # Pinstripes - JSON-configured provider
    "darkbloom",
    "meta",  # Meta Model API (Muse Spark) - JSON-configured provider
    "cognition",
    "scx-ai",
]
openai_text_completion_compatible_providers: Final[list] = [  # providers that support `/v1/completions`
    "together_ai",
    "fireworks_ai",
    "hosted_vllm",
    "meta_llama",
    "llamafile",
    "featherless_ai",
    "nebius",
    "dashscope",
    "modelscope",
    "moonshot",
    "publicai",
    "synthetic",
    "tensormesh",
    "apertis",
    "nano-gpt",
    "poe",
    "chutes",
    "v0",
    "lambda_ai",
    "hyperbolic",
    "wandb",
]
_openai_like_providers: Final[list] = [
    "predibase",
    "databricks",
    "lemonade",
    "watsonx",
]  # private helper. similar to openai but require some custom auth / endpoint handling, so can't use the openai sdk
# well supported replicate llms
replicate_models: Final[set] = set(
    [
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
)

clarifai_models: Final[set] = set(
    [
        "clarifai/openai.chat-completion.gpt-oss-20b",
        "clarifai/qwen.qwenLM.Qwen3-30B-A3B-Instruct-2507",
        "clarifai/qwen.qwen3.qwen3-next-80B-A3B-Thinking",
        "clarifai/openai.chat-completion.gpt-oss-120b",
        "clarifai/qwen.qwenLM.Qwen3-30B-A3B-Thinking-2507clarifai/openai.chat-completion.gpt-5-nano",
        "clarifai/openai.chat-completion.gpt-4o",
        "clarifai/gcp.generate.gemini-2_5-pro",
        "clarifai/anthropic.completion.claude-sonnet-4",
        "clarifai/xai.chat-completion.grok-2-vision-1212",
        "clarifai/openbmb.miniCPM.MiniCPM-o-2_6-language",
        "clarifai/microsoft.text-generation.Phi-4-reasoning-plus",
        "clarifai/openbmb.miniCPM.MiniCPM3-4B",
        "clarifai/openbmb.miniCPM.MiniCPM4-8B",
        "clarifai/xai.chat-completion.grok-2-1212",
        "clarifai/anthropic.completion.claude-opus-4",
        "clarifai/xai.chat-completion.grok-code-fast-1",
        "clarifai/qwen.qwenCoder.Qwen3-Coder-30B-A3B-Instruct",
        "clarifai/deepseek-ai.deepseek-chat.DeepSeek-R1-0528-Qwen3-8B",
        "clarifai/openai.chat-completion.gpt-5-mini",
        "clarifai/microsoft.text-generation.phi-4",
        "clarifai/openai.chat-completion.gpt-5",
        "clarifai/meta.Llama-3.Llama-3_2-3B-Instruct",
        "clarifai/xai.image-generation.grok-2-image-1212",
        "clarifai/xai.chat-completion.grok-3",
        "clarifai/openai.chat-completion.o3",
        "clarifai/qwen.qwen-VL.Qwen2_5-VL-7B-Instruct",
        "clarifai/qwen.qwenLM.Qwen3-14B",
        "clarifai/qwen.qwenLM.QwQ-32B-AWQ",
        "clarifai/anthropic.completion.claude-3_5-haiku",
        "clarifai/anthropic.completion.claude-3_7-sonnet",
    ]
)


huggingface_models: Final[set] = set(
    [
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
    ]
)  # these have been tested on extensively. But by default all text2text-generation and text-generation models are supported by liteLLM. - https://docs.litellm.ai/docs/providers
empower_models: Final = set(
    [
        "empower/empower-functions",
        "empower/empower-functions-small",
    ]
)

together_ai_models: Final[set] = set(
    [
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
    ]
)
# supports all together ai models, just pass in the model id e.g. completion(model="together_computer/replit_code_3b",...)


baseten_models: Final[set] = set(
    [
        "qvv0xeq",
        "q841o8w",
        "31dxrj3",
    ]
)  # FALCON 7B  # WizardLM  # Mosaic ML

featherless_ai_models: Final[set] = set(
    [
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
)

nebius_models: Final[set] = set(
    [
        # deepseek models
        "deepseek-ai/DeepSeek-R1-0528",
        "deepseek-ai/DeepSeek-V3-0324",
        "deepseek-ai/DeepSeek-V3",
        "deepseek-ai/DeepSeek-R1",
        "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        # google models
        "google/gemma-2-2b-it",
        "google/gemma-2-9b-it-fast",
        # llama models
        "meta-llama/Llama-3.3-70B-Instruct",
        "meta-llama/Meta-Llama-3.1-70B-Instruct",
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "meta-llama/Meta-Llama-3.1-405B-Instruct",
        "NousResearch/Hermes-3-Llama-405B",
        # microsoft models
        "microsoft/phi-4",
        # mistral models
        "mistralai/Mistral-Nemo-Instruct-2407",
        "mistralai/Devstral-Small-2505",
        # moonshot models
        "moonshotai/Kimi-K2-Instruct",
        # nvidia models
        "nvidia/Llama-3_1-Nemotron-Ultra-253B-v1",
        "nvidia/Llama-3_3-Nemotron-Super-49B-v1",
        # openai models
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        # qwen models
        "Qwen/Qwen3-Coder-480B-A35B-Instruct",
        "Qwen/Qwen3-235B-A22B-Instruct-2507",
        "Qwen/Qwen3-235B-A22B",
        "Qwen/Qwen3-30B-A3B",
        "Qwen/Qwen3-32B",
        "Qwen/Qwen3-14B",
        "Qwen/Qwen3-4B-fast",
        "Qwen/Qwen2.5-Coder-7B",
        "Qwen/Qwen2.5-Coder-32B-Instruct",
        "Qwen/Qwen2.5-72B-Instruct",
        "Qwen/QwQ-32B",
        "Qwen/Qwen3-30B-A3B-Thinking-2507",
        "Qwen/Qwen3-30B-A3B-Instruct-2507",
        # zai models
        "zai-org/GLM-4.5",
        "zai-org/GLM-4.5-Air",
        # other models
        "aaditya/Llama3-OpenBioLLM-70B",
        "ProdeusUnity/Stellar-Odyssey-12b-v0.0",
        "all-hands/openhands-lm-32b-v0.1",
    ]
)

dashscope_models: Final[set] = set(
    [
        "qwen-turbo",
        "qwen-plus",
        "qwen-max",
        "qwen-turbo-latest",
        "qwen-plus-latest",
        "qwen-max-latest",
        "qwq-32b",
        "qwen3-235b-a22b",
        "qwen3-32b",
        "qwen3-30b-a3b",
    ]
)

nebius_embedding_models: Final[set] = set(
    [
        "BAAI/bge-en-icl",
        "BAAI/bge-multilingual-gemma2",
        "intfloat/e5-mistral-7b-instruct",
    ]
)

WANDB_MODELS: Final[set] = set(
    [
        # openai models
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        # zai-org models
        "zai-org/GLM-4.5",
        # Qwen models
        "Qwen/Qwen3-235B-A22B-Instruct-2507",
        "Qwen/Qwen3-Coder-480B-A35B-Instruct",
        "Qwen/Qwen3-235B-A22B-Thinking-2507",
        # moonshotai
        "moonshotai/Kimi-K2-Instruct",
        "moonshotai/Kimi-K2.5",
        # MiniMaxAI
        "MiniMaxAI/MiniMax-M2.5",
        # meta models
        "meta-llama/Llama-3.1-8B-Instruct",
        "meta-llama/Llama-3.3-70B-Instruct",
        "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        # deepseek-ai
        "deepseek-ai/DeepSeek-V3.1",
        "deepseek-ai/DeepSeek-R1-0528",
        "deepseek-ai/DeepSeek-V3-0324",
        # microsoft
        "microsoft/Phi-4-mini-instruct",
    ]
)

modelscope_models: Final[set] = set(
    [
        # Qwen series models
        "Qwen/Qwen3-0.6B",
        "Qwen/Qwen3-1.7B",
        "Qwen/Qwen3-4B",
        "Qwen/Qwen3-8B",
        "Qwen/Qwen3-14B",
        "Qwen/Qwen3-30B-A3B",
        "Qwen/Qwen3-32B",
        "Qwen/Qwen3-235B-A22B",
        "Qwen/Qwen3-235B-A22B-Instruct-2507",
        "Qwen/Qwen3-235B-A22B-Thinking-2507",
        "Qwen/Qwen3-30B-A3B-Thinking-2507",
        "Qwen/Qwen3-Coder-30B-A3B-Instruct",
        "Qwen/Qwen3-Coder-480B-A35B-Instruct",
        "Qwen/Qwen3-Next-80B-A3B-Instruct",
        "Qwen/Qwen3-Next-80B-A3B-Thinking",
        "Qwen/Qwen3-VL-235B-A22B-Instruct",
        "Qwen/Qwen3-VL-8B-Instruct",
        "Qwen/Qwen3-VL-8B-Thinking",
        "Qwen/Qwen3.5-122B-A10B",
        "Qwen/Qwen3.5-27B",
        "Qwen/Qwen3.5-35B-A3B",
        "Qwen/Qwen3.5-397B-A17B",
        "Qwen/QwQ-32B",
        "Qwen/QwQ-32B-Preview",
        "Qwen/QVQ-72B-Preview",
        "Qwen/Qwen-Image-Edit",
        # DeepSeek series models
        "deepseek-ai/DeepSeek-R1-0528",
        "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "deepseek-ai/DeepSeek-V3.2",
        "deepseek-ai/DeepSeek-V4-Flash",
    ]
)

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
    "qwen3",
    "qwen2",
    "twelvelabs",
    "openai",
    "stability",
    "moonshot",
]

BEDROCK_EMBEDDING_PROVIDERS_LITERAL = Literal[
    "cohere",
    "amazon",
    "twelvelabs",
    "nova",
]

BEDROCK_CONVERSE_MODELS: Final = [
    "qwen.qwen3-coder-480b-a35b-v1:0",
    "qwen.qwen3-coder-next",
    "qwen.qwen3-235b-a22b-2507-v1:0",
    "qwen.qwen3-coder-30b-a3b-v1:0",
    "qwen.qwen3-32b-v1:0",
    "deepseek.v3-v1:0",
    "deepseek.v3.2",
    "openai.gpt-oss-20b-1:0",
    "openai.gpt-oss-120b-1:0",
    "anthropic.claude-haiku-4-5-20251001-v1:0",
    "anthropic.claude-sonnet-4-5-20250929-v1:0",
    "anthropic.claude-fable-5",
    "anthropic.claude-sonnet-5",
    "anthropic.claude-opus-5",
    "anthropic.claude-opus-4-8",
    "anthropic.claude-opus-4-7",
    "anthropic.claude-opus-4-6-v1:0",
    "anthropic.claude-opus-4-6-v1",
    "anthropic.claude-sonnet-4-6",
    "anthropic.claude-opus-4-1-20250805-v1:0",
    "anthropic.claude-opus-4-20250514-v1:0",
    "anthropic.claude-sonnet-4-20250514-v1:0",
    "anthropic.claude-3-7-sonnet-20250219-v1:0",
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
    "ai21.jamba-1-5-mini-v1:0",
    "ai21.jamba-1-5-large-v1:0",
    "meta.llama3-70b-instruct-v1:0",
    "meta.llama3-8b-instruct-v1:0",
    "meta.llama3-1-8b-instruct-v1:0",
    "meta.llama3-1-70b-instruct-v1:0",
    "meta.llama3-1-405b-instruct-v1:0",
    "meta.llama3-70b-instruct-v1:0",
    "mistral.mistral-large-2407-v1:0",
    "mistral.mistral-large-2402-v1:0",
    "mistral.mistral-small-2402-v1:0",
    "meta.llama3-2-1b-instruct-v1:0",
    "meta.llama3-2-3b-instruct-v1:0",
    "meta.llama3-2-11b-instruct-v1:0",
    "meta.llama3-2-90b-instruct-v1:0",
    "amazon.nova-lite-v1:0",
    "amazon.nova-2-lite-v1:0",
    "amazon.nova-2-pro-preview-20251202-v1:0",
    "amazon.nova-pro-v1:0",
    "writer.palmyra-x4-v1:0",
    "writer.palmyra-x5-v1:0",
    "minimax.minimax-m2.1",
    "moonshotai.kimi-k2.5",
]


open_ai_embedding_models: Final[set] = set(["text-embedding-ada-002"])
cohere_embedding_models: Final[set] = set(
    [
        "embed-v4.0",
        "embed-english-v3.0",
        "embed-english-light-v3.0",
        "embed-multilingual-v3.0",
        "embed-english-v2.0",
        "embed-english-light-v2.0",
        "embed-multilingual-v2.0",
    ]
)
bedrock_embedding_models: Final[set] = set(
    [
        "amazon.titan-embed-text-v1",
        "amazon.nova-2-multimodal-embeddings-v1:0",
        "cohere.embed-english-v3",
        "cohere.embed-multilingual-v3",
        "cohere.embed-v4:0",
        "twelvelabs.marengo-embed-2-7-v1:0",
    ]
)

known_tokenizer_config: Final = {
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


OPENAI_FINISH_REASONS: Final = [
    "stop",
    "length",
    "function_call",
    "tool_calls",
    "content_filter",
]
HUMANLOOP_PROMPT_CACHE_TTL_SECONDS: Final = int(os.getenv("HUMANLOOP_PROMPT_CACHE_TTL_SECONDS", 60))  # 1 minute
RESPONSE_FORMAT_TOOL_NAME = "json_tool_call"  # default tool name used when converting response format to tool call

########################### Logging Callback Constants ###########################
AZURE_STORAGE_MSFT_VERSION: Final = "2019-07-07"
AZURE_STORAGE_DEFAULT_ENDPOINT_SUFFIX: Final = "core.windows.net"
PROMETHEUS_BUDGET_METRICS_REFRESH_INTERVAL_MINUTES: Final = int(
    os.getenv("PROMETHEUS_BUDGET_METRICS_REFRESH_INTERVAL_MINUTES", 5)
)
CLOUDZERO_EXPORT_INTERVAL_MINUTES: Final = int(os.getenv("CLOUDZERO_EXPORT_INTERVAL_MINUTES", 60))
MCP_TOOL_NAME_PREFIX: Final = "mcp_tool"
MAXIMUM_TRACEBACK_LINES_TO_LOG: Final = int(os.getenv("MAXIMUM_TRACEBACK_LINES_TO_LOG", 100))

# Headers to control callbacks
X_LITELLM_DISABLE_CALLBACKS: Final = "x-litellm-disable-callbacks"
LITELLM_METADATA_FIELD: Final = "litellm_metadata"
OLD_LITELLM_METADATA_FIELD: Final = "metadata"
RETURN_RAW_MODEL_NAME_METADATA_KEY: Final = "_complexity_router_return_raw_model_name"
SESSION_DEPLOYMENT_AFFINITY_TTL_METADATA_KEY: Final = "_session_deployment_affinity_ttl"
CONSUMED_REQUEST_TAGS_METADATA_KEY: Final = "_consumed_request_tags"
INTERNAL_CALL_ORIGIN_METADATA_KEY: Final = "internal_call_origin"
LITELLM_TRUNCATED_PAYLOAD_FIELD: Final = "litellm_truncated"
LITELLM_TRUNCATION_DB_SAFEGUARD_NOTE: Final = (
    "Truncation is a DB storage safeguard. "
    "Full, untruncated data is logged to logging callbacks (OTEL, Datadog, etc.). "
    "To increase the truncation limit, set `MAX_STRING_LENGTH_PROMPT_IN_DB` in your env."
)
LITELLM_TRUNCATION_STDOUT_SAFEGUARD_NOTE: Final = (
    "Truncation is a stdout logging safeguard. "
    "Full, untruncated data is logged to logging callbacks (OTEL, Datadog, etc.) and at DEBUG level. "
    "To increase the truncation limit, set `MAX_STRING_LENGTH_STDOUT_LOG` in your env."
)

########################### LiteLLM Proxy Specific Constants ###########################
########################################################################################

# Standard headers that are always checked for customer/end-user ID (no configuration required)
# These headers work out-of-the-box for tools like Claude Code that support custom headers
STANDARD_CUSTOMER_ID_HEADERS: Final = [
    "x-litellm-customer-id",
    "x-litellm-end-user-id",
]
MAX_SPENDLOG_ROWS_TO_QUERY: Final = int(
    os.getenv("MAX_SPENDLOG_ROWS_TO_QUERY", 1_000_000)
)  # if spendLogs has more than 1M rows, do not query the DB
DEFAULT_SOFT_BUDGET: Final = float(
    os.getenv("DEFAULT_SOFT_BUDGET", 50.0)
)  # by default all litellm proxy keys have a soft budget of 50.0
# makes it clear this is a rate limit error for a litellm virtual key
RATE_LIMIT_ERROR_MESSAGE_FOR_VIRTUAL_KEY: Final = "LiteLLM Virtual Key user_api_key_hash"

# Python garbage collection threshold configuration
# Format: "gen0,gen1,gen2" e.g., "1000,50,50"
PYTHON_GC_THRESHOLD: Final = os.getenv("PYTHON_GC_THRESHOLD")

# pass through route constansts
BEDROCK_AGENT_RUNTIME_PASS_THROUGH_ROUTES: Final = [
    "agents/",
    "knowledgebases/",
    "flows/",
    "retrieveAndGenerate/",
    "rerank/",
    "generateQuery/",
    "optimize-prompt/",
]


# Headers that are safe to forward from incoming requests to Vertex AI
# Using an allowlist approach for security - only forward headers we explicitly trust
ALLOWED_VERTEX_AI_PASSTHROUGH_HEADERS: Final = {
    "anthropic-beta",  # Required for Anthropic features like extended context windows
    "content-type",  # Required for request body parsing
}

# Prefix for headers that should be forwarded to the provider with the prefix stripped
# e.g., 'x-pass-anthropic-beta: value' becomes 'anthropic-beta: value'
# Works for all LLM pass-through endpoints (Vertex AI, Anthropic, Bedrock, etc.)
PASS_THROUGH_HEADER_PREFIX: Final = "x-pass-"

BASE_MCP_ROUTE: Final = "/mcp"

BATCH_STATUS_POLL_INTERVAL_SECONDS: Final = int(os.getenv("BATCH_STATUS_POLL_INTERVAL_SECONDS", 3600))  # 1 hour
BATCH_STATUS_POLL_MAX_ATTEMPTS: Final = int(os.getenv("BATCH_STATUS_POLL_MAX_ATTEMPTS", 24))  # for 24 hours

HEALTH_CHECK_TIMEOUT_SECONDS: Final = int(os.getenv("HEALTH_CHECK_TIMEOUT_SECONDS", 60))  # 60 seconds
_background_health_check_max_tokens_env: Final = os.getenv("BACKGROUND_HEALTH_CHECK_MAX_TOKENS")
try:
    _raw_background_health_check_max_tokens: Final = (
        _background_health_check_max_tokens_env.strip() if _background_health_check_max_tokens_env is not None else ""
    )
    BACKGROUND_HEALTH_CHECK_MAX_TOKENS: int | None = (
        int(_raw_background_health_check_max_tokens) if _raw_background_health_check_max_tokens else None
    )
except (ValueError, TypeError):
    BACKGROUND_HEALTH_CHECK_MAX_TOKENS = None


_background_health_check_max_tokens_reasoning_env: Final = os.getenv("BACKGROUND_HEALTH_CHECK_MAX_TOKENS_REASONING")
try:
    _raw_background_health_check_max_tokens_reasoning: Final = (
        _background_health_check_max_tokens_reasoning_env.strip()
        if _background_health_check_max_tokens_reasoning_env is not None
        else ""
    )
    BACKGROUND_HEALTH_CHECK_MAX_TOKENS_REASONING: int | None = (
        int(_raw_background_health_check_max_tokens_reasoning)
        if _raw_background_health_check_max_tokens_reasoning
        else None
    )
except (ValueError, TypeError):
    BACKGROUND_HEALTH_CHECK_MAX_TOKENS_REASONING = None

LITTELM_INTERNAL_HEALTH_SERVICE_ACCOUNT_NAME: Final = "litellm-internal-health-check"
LITTELM_CLI_SERVICE_ACCOUNT_NAME: Final = "litellm-cli"
LITELLM_INTERNAL_JOBS_SERVICE_ACCOUNT_NAME: Final = "litellm_internal_jobs"
# Stable identifier substituted in place of the master key on UserAPIKeyAuth
# objects so the master key (or its hash) never propagates to spend logs,
# Prometheus metrics, audit trails, or any other downstream consumer.
LITELLM_PROXY_MASTER_KEY_ALIAS: Final = "litellm_proxy_master_key"

# Marker placed in ``model_call_details`` on a synthetic ``Logging`` object that
# records a proxy-gate error (auth/rate-limit rejection) for a request that never
# reached an upstream provider. Tracing callbacks key off it to avoid fabricating
# an LLM-call span for a call that did not happen. See
# ``ProxyLogging._handle_logging_proxy_only_error``.
LITELLM_LOGGING_NO_UPSTREAM_LLM_CALL: Final = "litellm_no_upstream_llm_call"

# Key/team metadata fields naming the OTel Resource ``service.name``, highest
# precedence first. Shared between the OTel v2 tenant router (which reads them
# out of ``user_api_key_auth_metadata``) and proxy request setup (which re-applies
# the key's values after the team metadata merge so a key outranks its team).
OTEL_SERVICE_NAME_METADATA_KEYS: Final = ("otel_service_name_override", "otel_service_name")

# Key Rotation Constants
LITELLM_KEY_ROTATION_ENABLED: Final = os.getenv("LITELLM_KEY_ROTATION_ENABLED", "false")
LITELLM_KEY_ROTATION_CHECK_INTERVAL_SECONDS: Final = int(
    os.getenv("LITELLM_KEY_ROTATION_CHECK_INTERVAL_SECONDS", 86400)
)  # 24 hours default
LITELLM_KEY_ROTATION_GRACE_PERIOD: Final[str] = os.getenv(
    "LITELLM_KEY_ROTATION_GRACE_PERIOD", ""
)  # Duration to keep old key valid after rotation (e.g. "24h", "2d"); empty = immediate revoke (default)
LITELLM_KEY_ROTATION_LOCK_TTL_SECONDS: Final = int(
    os.getenv("LITELLM_KEY_ROTATION_LOCK_TTL_SECONDS", 600)
)  # 10 minutes default — caps the deadlock window if a pod crashes mid-rotation
UI_SESSION_TOKEN_TEAM_ID: Final = "litellm-dashboard"
LITELLM_EXPIRED_UI_SESSION_KEY_CLEANUP_ENABLED = os.getenv("LITELLM_EXPIRED_UI_SESSION_KEY_CLEANUP_ENABLED", "false")
LITELLM_EXPIRED_UI_SESSION_KEY_CLEANUP_INTERVAL_SECONDS: Final = int(
    os.getenv("LITELLM_EXPIRED_UI_SESSION_KEY_CLEANUP_INTERVAL_SECONDS", 86400)
)  # 24 hours default
LITELLM_EXPIRED_UI_SESSION_KEY_CLEANUP_BATCH_SIZE: Final = int(
    os.getenv("LITELLM_EXPIRED_UI_SESSION_KEY_CLEANUP_BATCH_SIZE", 1000)
)
LITELLM_PROXY_ADMIN_NAME: Final = "default_user_id"
LITELLM_PROXY_BUDGET_NAME: Final = "litellm-proxy-budget"
GLOBAL_PROXY_SPEND_CACHE_KEY: Final = f"{LITELLM_PROXY_ADMIN_NAME}:spend"

########################### CLI SSO AUTHENTICATION CONSTANTS ###########################
LITELLM_CLI_SOURCE_IDENTIFIER: Final = "litellm-cli"
LITELLM_CLI_SESSION_TOKEN_PREFIX: Final = "litellm-session-token"
CLI_SSO_SESSION_CACHE_KEY_PREFIX: Final = "cli_sso_session"
CLI_SSO_SESSION_TTL_SECONDS: Final = 600
CLI_SESSION_KEY_PREFIX: Final = "cli-session"
# Support both CLI_JWT_EXPIRATION_HOURS and LITELLM_CLI_JWT_EXPIRATION_HOURS for backwards compatibility
CLI_JWT_EXPIRATION_HOURS: Final = int(
    os.getenv("CLI_JWT_EXPIRATION_HOURS") or os.getenv("LITELLM_CLI_JWT_EXPIRATION_HOURS") or 24
)
# Comma-separated allowlisted OIDC claim map for CLI SSO polling, e.g.
# "employment_type->acme_employment_type,org_info.department->department"
CLI_SSO_CLAIM_MAP: Final = os.getenv("CLI_SSO_CLAIM_MAP") or os.getenv("LITELLM_CLI_SSO_CLAIM_MAP") or ""
CLI_SSO_CLAIM_MAX_SCALAR_LENGTH: Final = 1024

########################### UI SESSION DURATION ###########################
# Duration for UI login session (username/password, SSO, invitation links). Format: "30s", "30m", "24h", "7d"
# Does NOT apply to EXPERIMENTAL_UI_LOGIN flow, which intentionally uses a fixed 10-minute expiry for security.
LITELLM_UI_SESSION_DURATION: Final = os.getenv("LITELLM_UI_SESSION_DURATION", "24h")

########################### DB CRON JOB NAMES ###########################
DB_SPEND_UPDATE_JOB_NAME: Final = "db_spend_update_job"
DB_DAILY_TAG_SPEND_UPDATE_JOB_NAME: Final = "db_daily_tag_spend_update_job"
PROMETHEUS_EMIT_BUDGET_METRICS_JOB_NAME: Final = "prometheus_emit_budget_metrics"
CLOUDZERO_EXPORT_USAGE_DATA_JOB_NAME: Final = "cloudzero_export_usage_data"
MAVVRIK_FOCUS_EXPORT_JOB_NAME: Final = "mavvrik_focus_export_usage_data"
CLOUDZERO_MAX_FETCHED_DATA_RECORDS: Final = int(os.getenv("CLOUDZERO_MAX_FETCHED_DATA_RECORDS", 50000))
SPEND_LOG_CLEANUP_JOB_NAME: Final = "spend_log_cleanup"
KEY_ROTATION_JOB_NAME: Final = "litellm_key_rotation_job"
EXPIRED_UI_SESSION_KEY_CLEANUP_JOB_NAME: Final = "litellm_expired_ui_session_key_cleanup_job"
WEEKLY_SPEND_REPORT_JOB_ID: Final = "weekly_spend_report_job"
MONTHLY_SPEND_REPORT_JOB_ID: Final = "monthly_spend_report_job"
PROMETHEUS_FALLBACK_STATS_JOB_ID: Final = "prometheus_fallback_stats_job"
SLACK_DAILY_REPORT_LOCK_ID: Final = "slack_daily_report"
SLACK_MODEL_DEPRECATION_LOCK_ID: Final = "slack_model_deprecation_warning"
SPEND_LOG_RUN_LOOPS: Final = int(os.getenv("SPEND_LOG_RUN_LOOPS", 500))
SPEND_LOG_CLEANUP_BATCH_SIZE: Final = int(os.getenv("SPEND_LOG_CLEANUP_BATCH_SIZE", 1000))
SPEND_LOG_CLEANUP_MAX_CONSECUTIVE_BATCH_FAILURES = int(os.getenv("SPEND_LOG_CLEANUP_MAX_CONSECUTIVE_BATCH_FAILURES", 3))
SPEND_LOG_CLEANUP_BATCH_FAILURE_BACKOFF_SECONDS: Final = float(
    os.getenv("SPEND_LOG_CLEANUP_BATCH_FAILURE_BACKOFF_SECONDS", 0.5)
)
SPEND_LOG_CLEANUP_RUN_BUDGET_SECONDS: Final = float(os.getenv("SPEND_LOG_CLEANUP_RUN_BUDGET_SECONDS", "300"))
SPEND_LOG_CLEANUP_BATCH_TIMEOUT_SECONDS: Final = float(os.getenv("SPEND_LOG_CLEANUP_BATCH_TIMEOUT_SECONDS", "30"))
SPEND_LOG_CLEANUP_REMAINING_COUNT_CAP: Final = int(os.getenv("SPEND_LOG_CLEANUP_REMAINING_COUNT_CAP", "100000"))
TOOL_SPEND_TOP_TOOLS: Final = 100
SPEND_LOG_PARTITION_INTERVAL: Final = os.getenv("SPEND_LOG_PARTITION_INTERVAL", "day")
SPEND_LOG_PARTITION_PRECREATE_AHEAD: Final = int(os.getenv("SPEND_LOG_PARTITION_PRECREATE_AHEAD", 7))
SPEND_LOG_WRITE_BATCH_MAX_BYTES: Final = max(1, int(os.getenv("SPEND_LOG_WRITE_BATCH_MAX_BYTES", 2_000_000)))
SPEND_LOG_WRITE_BATCH_MAX_ROWS: Final = max(1, int(os.getenv("SPEND_LOG_WRITE_BATCH_MAX_ROWS", "100")))
SPEND_LOG_QUEUE_SIZE_THRESHOLD: Final = int(os.getenv("SPEND_LOG_QUEUE_SIZE_THRESHOLD", 100))
SPEND_LOG_QUEUE_MAX_BYTES: Final = max(1, int(os.getenv("SPEND_LOG_QUEUE_MAX_BYTES", "64000000")))
SPEND_LOG_QUEUE_POLL_INTERVAL: Final = float(os.getenv("SPEND_LOG_QUEUE_POLL_INTERVAL", 2.0))
RESPONSES_SESSION_LOOKUP_MAX_ATTEMPTS: Final = max(1, int(os.getenv("RESPONSES_SESSION_LOOKUP_MAX_ATTEMPTS", "3")))
RESPONSES_SESSION_LOOKUP_RETRY_INTERVAL: Final = float(os.getenv("RESPONSES_SESSION_LOOKUP_RETRY_INTERVAL", "0.2"))
SPEND_COUNTER_RESEED_LOCKS_MAX_SIZE: Final = int(os.getenv("SPEND_COUNTER_RESEED_LOCKS_MAX_SIZE", 10000))
DEFAULT_CRON_JOB_LOCK_TTL_SECONDS: Final = int(os.getenv("DEFAULT_CRON_JOB_LOCK_TTL_SECONDS", 60))  # 1 minute
PROXY_BUDGET_RESCHEDULER_MIN_TIME: Final = int(os.getenv("PROXY_BUDGET_RESCHEDULER_MIN_TIME", 597))
RESET_BUDGET_JOB_BATCH_SIZE: Final = max(1, int(os.getenv("RESET_BUDGET_JOB_BATCH_SIZE", "500")))
RESET_BUDGET_JOB_MAX_CHUNKS_PER_RUN: Final = max(1, int(os.getenv("RESET_BUDGET_JOB_MAX_CHUNKS_PER_RUN", "100")))
RESET_BUDGET_JOB_NAME: Final = "reset_budget_job"
# Comfortably longer than one PROXY_BUDGET_RESCHEDULER_MIN_TIME tick, so a healthy
# leader keeps the lease across its own run, and a crashed one strands the sweep for
# at most a single tick.
RESET_BUDGET_JOB_LOCK_TTL_SECONDS: Final[int] = 900
PROXY_BATCH_POLLING_INTERVAL: Final = int(os.getenv("PROXY_BATCH_POLLING_INTERVAL", 3600))
MAX_OBJECTS_PER_POLL_CYCLE: Final = max(1, int(os.getenv("MAX_OBJECTS_PER_POLL_CYCLE", 50)))
MANAGED_OBJECT_STALENESS_CUTOFF_DAYS: Final = max(1, int(os.getenv("MANAGED_OBJECT_STALENESS_CUTOFF_DAYS", 7)))
STALE_OBJECT_CLEANUP_BATCH_SIZE: Final = max(1, int(os.getenv("STALE_OBJECT_CLEANUP_BATCH_SIZE", 1000)))
# Set PROXY_BATCH_POLLING_ENABLED=false to disable the CheckBatchCost and
# CheckResponsesCost background polling jobs entirely (e.g. to avoid DB load on
# installations with large numbers of stale managed objects).
_batch_polling_env: Final = os.getenv("PROXY_BATCH_POLLING_ENABLED", "true").lower()
PROXY_BATCH_POLLING_ENABLED: Final = _batch_polling_env == "true"
BACKGROUND_INTERACTION_COST_POLL_INITIAL_INTERVAL_SECONDS: Final = float(
    os.getenv("BACKGROUND_INTERACTION_COST_POLL_INITIAL_INTERVAL_SECONDS", "5")
)
BACKGROUND_INTERACTION_COST_POLL_MAX_INTERVAL_SECONDS: Final = float(
    os.getenv("BACKGROUND_INTERACTION_COST_POLL_MAX_INTERVAL_SECONDS", "60")
)
BACKGROUND_INTERACTION_COST_POLL_TIMEOUT_SECONDS: Final = float(
    os.getenv("BACKGROUND_INTERACTION_COST_POLL_TIMEOUT_SECONDS", "3600")
)
_background_interaction_cost_polling_env: Final = os.getenv(
    "BACKGROUND_INTERACTION_COST_POLLING_ENABLED", "true"
).lower()
BACKGROUND_INTERACTION_COST_POLLING_ENABLED: Final = _background_interaction_cost_polling_env == "true"
PROXY_BUDGET_RESCHEDULER_MAX_TIME: Final = int(os.getenv("PROXY_BUDGET_RESCHEDULER_MAX_TIME", 605))
PROXY_BATCH_WRITE_AT: Final = int(os.getenv("PROXY_BATCH_WRITE_AT", 10))  # in seconds, increased from 10
PROXY_CONFIG_RELOAD_INTERVAL_SECONDS: Final = get_env_int("PROXY_CONFIG_RELOAD_INTERVAL_SECONDS", 30)

# APScheduler Configuration - MEMORY LEAK FIX
# These settings prevent memory leaks in APScheduler's normalize() and _apply_jitter() functions
APSCHEDULER_COALESCE: Final = os.getenv("APSCHEDULER_COALESCE", "True").lower() in [
    "true",
    "1",
]  # collapse many missed runs into one
APSCHEDULER_MISFIRE_GRACE_TIME: Final = int(
    os.getenv("APSCHEDULER_MISFIRE_GRACE_TIME", 3600)
)  # ignore runs older than 1 hour (was 120)
APSCHEDULER_MAX_INSTANCES: Final = int(os.getenv("APSCHEDULER_MAX_INSTANCES", 1))  # prevent concurrent job instances
APSCHEDULER_REPLACE_EXISTING: Final = os.getenv("APSCHEDULER_REPLACE_EXISTING", "True").lower() in [
    "true",
    "1",
]  # always replace existing jobs

# Width of the window scheduled background jobs are spread across, so they do not all fire
# on one instant on every replica. Tunable per deployment via general_settings.
DEFAULT_STAGGER_WINDOW_SECONDS: Final = 300

# The number of tag entries are higher than number of user, team entries. This leads to a higher QPS.
# This will run tag spcific tasks at a later time to smooth QPS
DAILY_TAG_SPEND_BATCH_MULTIPLIER: Final = 2.3

DEFAULT_HEALTH_CHECK_INTERVAL: Final = int(os.getenv("DEFAULT_HEALTH_CHECK_INTERVAL", 300))  # 5 minutes
DEFAULT_SHARED_HEALTH_CHECK_TTL: Final = int(
    os.getenv("DEFAULT_SHARED_HEALTH_CHECK_TTL", 300)
)  # 5 minutes - TTL for cached health check results
DEFAULT_SHARED_HEALTH_CHECK_LOCK_TTL: Final = int(
    os.getenv("DEFAULT_SHARED_HEALTH_CHECK_LOCK_TTL", 60)
)  # 1 minute - TTL for health check lock
DEFAULT_HEALTH_CHECK_STALENESS_MULTIPLIER: Final = 2  # health state is stale after interval * this
PROMETHEUS_FALLBACK_STATS_SEND_TIME_HOURS: Final = int(os.getenv("PROMETHEUS_FALLBACK_STATS_SEND_TIME_HOURS", 9))
DEFAULT_MODEL_CREATED_AT_TIME: Final = int(
    os.getenv("DEFAULT_MODEL_CREATED_AT_TIME", 1677610602)
)  # returns on `/models` endpoint
DEFAULT_SLACK_ALERTING_THRESHOLD: Final = int(os.getenv("DEFAULT_SLACK_ALERTING_THRESHOLD", 300))
MAX_TEAM_LIST_LIMIT: Final = int(os.getenv("MAX_TEAM_LIST_LIMIT", 20))
MAX_POLICY_ESTIMATE_IMPACT_ROWS: Final = int(os.getenv("MAX_POLICY_ESTIMATE_IMPACT_ROWS", 1000))
DEFAULT_PROMPT_INJECTION_SIMILARITY_THRESHOLD = float(os.getenv("DEFAULT_PROMPT_INJECTION_SIMILARITY_THRESHOLD", 0.7))
LENGTH_OF_LITELLM_GENERATED_KEY: Final = int(os.getenv("LENGTH_OF_LITELLM_GENERATED_KEY", 16))
MINIMUM_CUSTOM_KEY_LENGTH: Final = int(os.getenv("MINIMUM_CUSTOM_KEY_LENGTH", 16))
SECRET_MANAGER_REFRESH_INTERVAL: Final = int(os.getenv("SECRET_MANAGER_REFRESH_INTERVAL", 86400))
LITELLM_SETTINGS_SAFE_DB_OVERRIDES: Final = [
    "default_internal_user_params",
    "default_team_params",
    "public_mcp_servers",
    "public_agent_groups",
    "public_model_groups",
    "public_model_groups_links",
    "cost_discount_config",
    "cost_margin_config",
    "block_requests_for_models_without_pricing",
    "budget_exceeded_throttle_percentage",
    # Every field editable from the Admin UI (proxy_server._GENERAL_SETTINGS_UI_LITELLM_FIELDS)
    # must be listed here so a DB write from one worker overrides the live litellm attribute on
    # the others when config reloads; otherwise peer workers stay on their startup value.
    # test_general_settings_ui_fields_are_db_overridable enforces that pairing.
    "enable_anthropic_prompt_caching",
    "anthropic_prompt_caching_ttl",
    "max_ui_session_budget",
    "budget_rollover",
]
SPECIAL_LITELLM_AUTH_TOKEN: Final = ["ui-token"]
DEFAULT_MANAGEMENT_OBJECT_IN_MEMORY_CACHE_TTL = int(os.getenv("DEFAULT_MANAGEMENT_OBJECT_IN_MEMORY_CACHE_TTL", 60))
DEFAULT_ACCESS_GROUP_CACHE_TTL: Final = int(os.getenv("DEFAULT_ACCESS_GROUP_CACHE_TTL", 600))
# Short TTL for negative MCP access-group existence lookups. Keeps unauthenticated
# callers from forcing a DB query per request for unknown names, while bounding
# staleness so a transient DB error (which surfaces as an empty list) cannot
# hide a real group for long.
DEFAULT_MCP_ACCESS_GROUP_NEGATIVE_CACHE_TTL: Final = 10
# Maximum number of comma-separated MCP server / access-group tokens accepted
# in a single ``/{name1,name2,...}/mcp`` URL. Bounds the per-request DB / cache
# fan-out an authenticated caller can trigger by stuffing the path with tokens.
DEFAULT_MCP_NAMESPACE_CSV_MAX_TOKENS: Final = 16
# Ceilings on the cached auth registries; larger tables fall back to per-row lookups
# instead of holding an unbounded id set in every worker.
TAG_REGISTRY_MAX_SIZE: Final = 5000
END_USER_RESTRICTED_REGISTRY_MAX_SIZE: Final = 5000
# How long a failed registry load is remembered as "unusable", so a degraded Postgres
# is not re-scanned on every request on top of the per-id lookups it falls back to.
REGISTRY_ERROR_NEGATIVE_CACHE_TTL: Final = 30

# Sentry Scrubbing Configuration
SENTRY_DENYLIST: Final = [
    # API Keys and Tokens
    "api_key",
    "token",
    "key",
    "secret",
    "password",
    "auth",
    "credential",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "AZURE_API_KEY",
    "COHERE_API_KEY",
    "REPLICATE_API_KEY",
    "HUGGINGFACE_API_KEY",
    "TOGETHERAI_API_KEY",
    "CLOUDFLARE_API_KEY",
    "BASETEN_KEY",
    "OPENROUTER_KEY",
    "COMETAPI_KEY",
    "DATAROBOT_API_TOKEN",
    "FIREWORKS_API_KEY",
    "FIREWORKS_AI_API_KEY",
    "FIREWORKSAI_API_KEY",
    "OVHCLOUD_API_KEY",
    "CLARIFAI_API_KEY",
    # Database and Connection Strings
    "database_url",
    "redis_url",
    "connection_string",
    # Authentication and Security
    "master_key",
    "LITELLM_MASTER_KEY",
    "auth_token",
    "jwt_token",
    "private_key",
    "SLACK_WEBHOOK_URL",
    "webhook_url",
    "LANGFUSE_SECRET_KEY",
    # Email Configuration
    "SMTP_PASSWORD",
    "SMTP_USERNAME",
    "email_password",
    # Cloud Provider Credentials
    "aws_access_key",
    "aws_secret_key",
    "gcp_credentials",
    "azure_credentials",
    "HCP_VAULT_TOKEN",
    "CIRCLE_OIDC_TOKEN",
    # Proxy and Environment Settings
    "proxy_url",
    "proxy_key",
    "environment_variables",
]
SENTRY_PII_DENYLIST: Final = [
    "user_id",
    "email",
    "phone",
    "address",
    "ip_address",
    "SMTP_SENDER_EMAIL",
    "TEST_EMAIL_ADDRESS",
]

# CoroutineChecker cache configuration
COROUTINE_CHECKER_MAX_SIZE_IN_MEMORY: Final = int(os.getenv("COROUTINE_CHECKER_MAX_SIZE_IN_MEMORY", 1000))

########################### RAG Text Splitter Constants ###########################
DEFAULT_CHUNK_SIZE: Final = int(os.getenv("DEFAULT_CHUNK_SIZE", 1000))
DEFAULT_CHUNK_OVERLAP: Final = int(os.getenv("DEFAULT_CHUNK_OVERLAP", 200))

########################### S3 Vectors RAG Constants ###########################
S3_VECTORS_DEFAULT_DIMENSION: Final = int(os.getenv("S3_VECTORS_DEFAULT_DIMENSION", 1024))
S3_VECTORS_DEFAULT_DISTANCE_METRIC: Final = str(os.getenv("S3_VECTORS_DEFAULT_DISTANCE_METRIC", "cosine"))
S3_VECTORS_DEFAULT_NON_FILTERABLE_METADATA_KEYS: Final = ["source_text"]

########################### Microsoft SSO Constants ###########################
MICROSOFT_USER_EMAIL_ATTRIBUTE: Final = str(os.getenv("MICROSOFT_USER_EMAIL_ATTRIBUTE", "userPrincipalName"))
MICROSOFT_USER_DISPLAY_NAME_ATTRIBUTE: Final = str(os.getenv("MICROSOFT_USER_DISPLAY_NAME_ATTRIBUTE", "displayName"))
MICROSOFT_USER_ID_ATTRIBUTE: Final = str(os.getenv("MICROSOFT_USER_ID_ATTRIBUTE", "id"))
MICROSOFT_USER_FIRST_NAME_ATTRIBUTE: Final = str(os.getenv("MICROSOFT_USER_FIRST_NAME_ATTRIBUTE", "givenName"))
MICROSOFT_USER_LAST_NAME_ATTRIBUTE: Final = str(os.getenv("MICROSOFT_USER_LAST_NAME_ATTRIBUTE", "surname"))

# Maximum payload size (in bytes) to fully serialize for DEBUG logging.
# Payloads larger than this are truncated to avoid multi-second json.dumps blocking the response.
MAX_PAYLOAD_SIZE_FOR_DEBUG_LOG: Final = int(os.getenv("MAX_PAYLOAD_SIZE_FOR_DEBUG_LOG", 102400))  # 100 KB

# Policy template enrichment
MAX_COMPETITOR_NAMES: Final = int(os.getenv("MAX_COMPETITOR_NAMES", 100))
COMPETITOR_LLM_TEMPERATURE: Final = float(os.getenv("COMPETITOR_LLM_TEMPERATURE", 0.3))
DEFAULT_COMPETITOR_DISCOVERY_MODEL: Final = "gpt-4o-mini"

# Advisor tool orchestration
# Providers that support advisor_20260301 natively (no LiteLLM orchestration needed).
# Add vertex_ai here once verified.
ADVISOR_NATIVE_PROVIDERS: Final[frozenset] = frozenset({"anthropic"})
# Hard cap on advisor iterations per request to prevent runaway loops.
ADVISOR_MAX_USES: Final[int] = 5
# Description injected into the synthetic advisor tool definition sent to non-native providers.
ADVISOR_TOOL_DESCRIPTION: Final[str] = (
    "Consult a highly intelligent advisor model when you need expert guidance, "
    "want to verify your reasoning, or face a complex decision. "
    "Describe your question or challenge clearly in the 'question' field."
)

# Headers that must be stripped from a provider exception before it's forwarded as
# the proxy's own HTTP response, or they conflict with the framing the proxy sets.
HTTP_FRAMING_HEADERS: Final[frozenset[str]] = frozenset(
    {
        "content-length",
        "transfer-encoding",
        "content-encoding",
        "content-type",
        "set-cookie",
        "cookie",
        "proxy-authenticate",
        "proxy-authorization",
    }
)

# Browser-facing security headers that a malicious or misconfigured upstream
# provider must not be able to set on the proxy's own response.
BROWSER_SECURITY_HEADERS: Final[frozenset[str]] = frozenset(
    {
        "access-control-allow-origin",
        "access-control-allow-credentials",
        "access-control-allow-methods",
        "access-control-allow-headers",
        "access-control-expose-headers",
        "content-security-policy",
        "content-security-policy-report-only",
        "clear-site-data",
        "strict-transport-security",
        "x-frame-options",
        "cross-origin-opener-policy",
        "cross-origin-embedder-policy",
        "cross-origin-resource-policy",
    }
)

UNSAFE_PROXY_RESPONSE_HEADERS: Final[frozenset[str]] = HTTP_FRAMING_HEADERS | BROWSER_SECURITY_HEADERS

# A retrieved response replays the usage of the call that created it, so pricing these
# read/management routes like inference bills the same tokens twice.
NON_INFERENCE_CALL_TYPES: Final[frozenset[str]] = frozenset(
    {
        "get_responses",
        "aget_responses",
        "delete_responses",
        "adelete_responses",
        "cancel_responses",
        "acancel_responses",
        "list_input_items",
        "alist_input_items",
        "vector_store_create",
        "avector_store_create",
        "vector_store_retrieve",
        "avector_store_retrieve",
        "vector_store_list",
        "avector_store_list",
        "vector_store_update",
        "avector_store_update",
        "vector_store_delete",
        "avector_store_delete",
        "vector_store_file_create",
        "avector_store_file_create",
        "vector_store_file_list",
        "avector_store_file_list",
        "vector_store_file_retrieve",
        "avector_store_file_retrieve",
        "vector_store_file_content",
        "avector_store_file_content",
        "vector_store_file_update",
        "avector_store_file_update",
        "vector_store_file_delete",
        "avector_store_file_delete",
    }
)

# PTU reservation rollup writes rows to LiteLLM_DailyTeamSpend with this
# sentinel api_key so PTU flat cost stays distinguishable from real per-request
# spend under the table's composite unique constraint.
PTU_SENTINEL_API_KEY: Final[str] = "__ptu_flat_cost__"
PTU_ROLLUP_JOB_ID: Final[str] = "ptu_flat_cost_rollup_job"
PTU_ROLLUP_LOCK_TTL_SECONDS: Final[int] = 900
# Furthest back the catch-up pass looks for unpriced PTU days when a deployment
# declares no ptu_effective_from, bounding the scan for an open-ended window.
PTU_ROLLUP_MAX_BACKFILL_DAYS: Final[int] = 90
# Deployments named in the lapsed-window alert before it is truncated, so a fleet-wide
# expiry cannot produce an alert too large for the channel delivering it.
PTU_LAPSED_ALERT_LIMIT: Final[int] = 10
# Slack allowed when deciding a sentinel row is stale. The row's updated_at and the
# run's cutoff are stamped by different hosts, so clock skew between them must not let
# one run delete a charge another just wrote. A stale row is hours old and a concurrent
# one is seconds old, so a few minutes separates them.
PTU_PRUNE_SKEW_GRACE_SECONDS: Final[int] = 300

# How long enqueued-token reservations for batches live without a refund. Providers
# complete or expire batches within their completion window (24h for OpenAI), so a
# reservation still unrefunded after 8 days belongs to a batch whose terminal state
# was never observed (e.g. proxy restart); expiry returns the tokens to the caller.
BATCH_ENQUEUED_TOKEN_TTL_SECONDS: Final[int] = 8 * 24 * 60 * 60

# Key/team metadata field that opts batches into enqueued-token limiting. Only proxy
# admins may write it: when present it replaces the standard RPM/TPM checks for
# batch submissions.
BATCH_ENQUEUED_TOKEN_LIMIT_METADATA_KEY: Final = "batch_enqueued_token_limit"

# Shared read-only empty mapping, for defaulting optional Mapping parameters without
# constructing a fresh mutable dict at each call site.
EMPTY_MAPPING: Final = MappingProxyType({})
