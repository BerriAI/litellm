"""Source of truth #1 for the LiteLLM OpenTelemetry instrumentation: attribute keys.

This module is intentionally free of any ``opentelemetry`` import so it can be
imported in environments where the OTel SDK is not installed. It is the *only*
place where a span-attribute key string or metric name is written.

Keys follow the OpenTelemetry GenAI semantic conventions (experimental). Anything
without a semconv equivalent lives under the ``litellm.*`` vendor namespace.
"""

from enum import Enum
from typing import Dict, Final, Optional, Tuple


class GenAIOperation(str, Enum):
    """Values for ``gen_ai.operation.name``."""

    CHAT = "chat"
    TEXT_COMPLETION = "text_completion"
    EMBEDDINGS = "embeddings"
    GENERATE_CONTENT = "generate_content"
    CREATE_AGENT = "create_agent"  # reserved for future agent spans
    INVOKE_AGENT = "invoke_agent"  # reserved for future agent spans
    EXECUTE_TOOL = "execute_tool"  # reserved for future tool spans


class GenAIProvider(str, Enum):
    """Common values for ``gen_ai.provider.name`` (replaces ``gen_ai.system``)."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AWS_BEDROCK = "aws.bedrock"
    AZURE_AI_OPENAI = "azure.ai.openai"
    AZURE_AI_INFERENCE = "azure.ai.inference"
    GCP_GEMINI = "gcp.gemini"
    GCP_VERTEX_AI = "gcp.vertex_ai"
    COHERE = "cohere"
    MISTRAL_AI = "mistral_ai"
    DEEPSEEK = "deepseek"
    GROQ = "groq"
    PERPLEXITY = "perplexity"
    X_AI = "x_ai"
    IBM_WATSONX_AI = "ibm.watsonx.ai"


class GenAI:
    """Canonical OTel GenAI span-attribute keys."""

    # request
    OPERATION_NAME: Final = "gen_ai.operation.name"
    PROVIDER_NAME: Final = "gen_ai.provider.name"
    REQUEST_MODEL: Final = "gen_ai.request.model"
    REQUEST_TEMPERATURE: Final = "gen_ai.request.temperature"
    REQUEST_TOP_P: Final = "gen_ai.request.top_p"
    REQUEST_TOP_K: Final = "gen_ai.request.top_k"
    REQUEST_MAX_TOKENS: Final = "gen_ai.request.max_tokens"
    REQUEST_FREQUENCY_PENALTY: Final = "gen_ai.request.frequency_penalty"
    REQUEST_PRESENCE_PENALTY: Final = "gen_ai.request.presence_penalty"
    REQUEST_STOP_SEQUENCES: Final = "gen_ai.request.stop_sequences"
    REQUEST_SEED: Final = "gen_ai.request.seed"
    REQUEST_CHOICE_COUNT: Final = "gen_ai.request.choice.count"
    REQUEST_ENCODING_FORMATS: Final = "gen_ai.request.encoding_formats"
    # response
    RESPONSE_ID: Final = "gen_ai.response.id"
    RESPONSE_MODEL: Final = "gen_ai.response.model"
    RESPONSE_FINISH_REASONS: Final = "gen_ai.response.finish_reasons"
    # usage
    USAGE_INPUT_TOKENS: Final = "gen_ai.usage.input_tokens"
    USAGE_OUTPUT_TOKENS: Final = "gen_ai.usage.output_tokens"
    # content (opt-in, gated by capture mode)
    INPUT_MESSAGES: Final = "gen_ai.input.messages"
    OUTPUT_MESSAGES: Final = "gen_ai.output.messages"
    SYSTEM_INSTRUCTIONS: Final = "gen_ai.system_instructions"
    OUTPUT_TYPE: Final = "gen_ai.output.type"
    CONVERSATION_ID: Final = "gen_ai.conversation.id"
    # agent / tool (reserved)
    AGENT_ID: Final = "gen_ai.agent.id"
    AGENT_NAME: Final = "gen_ai.agent.name"
    TOOL_NAME: Final = "gen_ai.tool.name"
    TOOL_CALL_ID: Final = "gen_ai.tool.call.id"


class Error:
    TYPE: Final = "error.type"


class Server:
    ADDRESS: Final = "server.address"
    PORT: Final = "server.port"


class HTTP:
    """HTTP server-span keys. Belong on the SERVER span only (never promoted)."""

    REQUEST_METHOD: Final = "http.request.method"
    ROUTE: Final = "http.route"
    RESPONSE_STATUS_CODE: Final = "http.response.status_code"
    URL_PATH: Final = "url.path"


class LiteLLM:
    """Vendor-extension keys (no semconv equivalent). Always ``litellm.*``."""

    CALL_ID: Final = "litellm.call_id"
    COST_PREFIX: Final = "litellm.cost."
    METADATA_PREFIX: Final = "litellm.metadata."
    TEAM_ID: Final = "litellm.team.id"
    TEAM_ALIAS: Final = "litellm.team.alias"
    KEY_HASH: Final = "litellm.api_key.hash"
    END_USER: Final = "litellm.end_user.id"
    REQUEST_STREAMING: Final = "litellm.request.streaming"
    GUARDRAIL_NAME: Final = "litellm.guardrail.name"
    GUARDRAIL_MODE: Final = "litellm.guardrail.mode"
    GUARDRAIL_STATUS: Final = "litellm.guardrail.status"
    SERVICE_NAME: Final = "litellm.service.name"
    SERVICE_CALL_TYPE: Final = "litellm.service.call_type"
    PREPROCESSING_MS: Final = "litellm.preprocessing.duration_ms"


class Metric:
    """GenAI metric instrument names."""

    TOKEN_USAGE: Final = "gen_ai.client.token.usage"
    OPERATION_DURATION: Final = "gen_ai.client.operation.duration"


# Identity keys promoted onto every span via Baggage (bounded allowlist).
# NOTE: http.* is deliberately excluded — it belongs on the SERVER span only.
BAGGAGE_PROMOTED_KEYS: Final[Tuple[str, ...]] = (
    LiteLLM.TEAM_ID,
    LiteLLM.TEAM_ALIAS,
    LiteLLM.KEY_HASH,
    GenAI.REQUEST_MODEL,
)

# Default metadata sub-keys eligible for baggage promotion. The full ``metadata``
# blob is never promoted; only this explicit, config-overridable allowlist is.
DEFAULT_BAGGAGE_METADATA_KEYS: Final[Tuple[str, ...]] = (
    "user_api_key_org_id",
    "user_api_key_user_id",
    "user_api_key_alias",
    "user_api_key_end_user_id",
    "requester_ip_address",
)


# litellm ``custom_llm_provider`` -> ``gen_ai.provider.name`` value.
_PROVIDER_BY_LITELLM: Dict[str, GenAIProvider] = {
    "openai": GenAIProvider.OPENAI,
    "text-completion-openai": GenAIProvider.OPENAI,
    "azure": GenAIProvider.AZURE_AI_OPENAI,
    "azure_ai": GenAIProvider.AZURE_AI_INFERENCE,
    "anthropic": GenAIProvider.ANTHROPIC,
    "bedrock": GenAIProvider.AWS_BEDROCK,
    "bedrock_converse": GenAIProvider.AWS_BEDROCK,
    "vertex_ai": GenAIProvider.GCP_VERTEX_AI,
    "vertex_ai_beta": GenAIProvider.GCP_VERTEX_AI,
    "gemini": GenAIProvider.GCP_GEMINI,
    "cohere": GenAIProvider.COHERE,
    "cohere_chat": GenAIProvider.COHERE,
    "mistral": GenAIProvider.MISTRAL_AI,
    "deepseek": GenAIProvider.DEEPSEEK,
    "groq": GenAIProvider.GROQ,
    "perplexity": GenAIProvider.PERPLEXITY,
    "xai": GenAIProvider.X_AI,
    "watsonx": GenAIProvider.IBM_WATSONX_AI,
}

# litellm ``call_type`` -> ``gen_ai.operation.name``.
_OPERATION_BY_CALL_TYPE: Dict[str, GenAIOperation] = {
    "completion": GenAIOperation.CHAT,
    "acompletion": GenAIOperation.CHAT,
    "completion_with_retries": GenAIOperation.CHAT,
    "text_completion": GenAIOperation.TEXT_COMPLETION,
    "atext_completion": GenAIOperation.TEXT_COMPLETION,
    "embedding": GenAIOperation.EMBEDDINGS,
    "aembedding": GenAIOperation.EMBEDDINGS,
    "responses": GenAIOperation.CHAT,
    "aresponses": GenAIOperation.CHAT,
}


def resolve_provider(custom_llm_provider: Optional[str]) -> str:
    """Map a litellm provider string to a ``gen_ai.provider.name`` value.

    Unknown providers pass through verbatim — the convention explicitly allows
    provider-specific values, so an unmapped name is still valid.
    """
    if not custom_llm_provider:
        return ""
    mapped = _PROVIDER_BY_LITELLM.get(custom_llm_provider.lower())
    return mapped.value if mapped is not None else custom_llm_provider


def resolve_operation(call_type: Optional[str]) -> GenAIOperation:
    """Map a litellm ``call_type`` to a ``gen_ai.operation.name`` value."""
    if not call_type:
        return GenAIOperation.CHAT
    return _OPERATION_BY_CALL_TYPE.get(call_type.lower(), GenAIOperation.CHAT)
