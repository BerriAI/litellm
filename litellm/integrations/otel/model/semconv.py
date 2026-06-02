"""
Keys follow the OpenTelemetry GenAI semantic conventions (experimental). Anything
without a semconv equivalent lives under the ``litellm.*`` vendor namespace.
"""

from enum import Enum
from typing import Final


class GenAIOperation(str, Enum):
    """Values for ``gen_ai.operation.name``."""

    CHAT = "chat"
    TEXT_COMPLETION = "text_completion"
    EMBEDDINGS = "embeddings"
    GENERATE_CONTENT = "generate_content"
    CREATE_AGENT = "create_agent"  # reserved for future agent spans
    INVOKE_AGENT = "invoke_agent"  # reserved for future agent spans
    EXECUTE_TOOL = "execute_tool"  # MCP tool-call spans


class GenAIProvider(str, Enum):
    """Common values for the ``gen_ai.provider.name`` attribute."""

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


class MCPMethod(str, Enum):
    """Well-known values for ``mcp.method.name`` that litellm's MCP gateway
    serves. The value is the JSON-RPC method exactly as it travels on the wire."""

    TOOLS_CALL = "tools/call"
    TOOLS_LIST = "tools/list"
    PROMPTS_GET = "prompts/get"
    PROMPTS_LIST = "prompts/list"


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
    # agent (reserved)
    AGENT_ID: Final = "gen_ai.agent.id"
    AGENT_NAME: Final = "gen_ai.agent.name"
    # tool / tool-call (stamped on MCP tool-call spans). Arguments and result are
    # the tool's input/output payloads — sensitive, so they're opt-in and gated by
    # the same content-capture mode as prompt/response content.
    TOOL_NAME: Final = "gen_ai.tool.name"
    TOOL_CALL_ID: Final = "gen_ai.tool.call.id"
    TOOL_CALL_ARGUMENTS: Final = "gen_ai.tool.call.arguments"
    TOOL_CALL_RESULT: Final = "gen_ai.tool.call.result"
    # prompt (MCP ``prompts/get`` etc.)
    PROMPT_NAME: Final = "gen_ai.prompt.name"


class MCP:
    """OTel GenAI MCP (Model Context Protocol) span-attribute keys.

    ``METHOD_NAME`` is the only key litellm populates from a closed request today;
    the rest are part of the convention's vocabulary and are stamped when the
    corresponding signal (session, protocol version, resource) is available.
    """

    METHOD_NAME: Final = "mcp.method.name"
    SESSION_ID: Final = "mcp.session.id"
    PROTOCOL_VERSION: Final = "mcp.protocol.version"
    RESOURCE_URI: Final = "mcp.resource.uri"


class JsonRpc:
    """JSON-RPC keys carried on MCP spans. The error/status code lives in the
    ``rpc.*`` namespace per semconv, not ``jsonrpc.*``."""

    REQUEST_ID: Final = "jsonrpc.request.id"
    PROTOCOL_VERSION: Final = "jsonrpc.protocol.version"
    RESPONSE_STATUS_CODE: Final = "rpc.response.status_code"


class NetworkTransport(str, Enum):
    """Well-known values for ``network.transport``."""

    TCP = "tcp"
    UDP = "udp"
    QUIC = "quic"
    UNIX = "unix"
    PIPE = "pipe"


class Network:
    """OTel network keys, recommended on MCP spans to describe the transport
    carrying the JSON-RPC messages (stdio pipe, HTTP, websocket, …)."""

    PROTOCOL_NAME: Final = "network.protocol.name"
    PROTOCOL_VERSION: Final = "network.protocol.version"
    TRANSPORT: Final = "network.transport"


class Client:
    """Peer (client) network keys, stamped on MCP *server* spans the same way
    ``server.*`` is stamped on client spans."""

    ADDRESS: Final = "client.address"
    PORT: Final = "client.port"


class Error:
    TYPE: Final = "error.type"


class Server:
    ADDRESS: Final = "server.address"
    PORT: Final = "server.port"


class DB:
    """Database / cache client-span keys (OTel ``db.*`` semconv).

    Stamped on ``DB_CALL`` spans (redis / postgres), which are CLIENT spans for
    outbound datastore calls — not on the INTERNAL ``SERVICE`` spans.
    """

    SYSTEM_NAME: Final = "db.system.name"
    OPERATION_NAME: Final = "db.operation.name"


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
    # The team's free-form metadata dict, JSON-serialized into a single value.
    TEAM_METADATA: Final = "litellm.team.metadata"
    KEY_HASH: Final = "litellm.api_key.hash"
    END_USER: Final = "litellm.end_user.id"
    # The model string litellm actually sent to the provider (the deployment's
    # ``litellm_params.model``), distinct from the user-facing ``gen_ai.request.model``.
    PROVIDER_MODEL: Final = "litellm.provider.model"
    REQUEST_STREAMING: Final = "litellm.request.streaming"
    GUARDRAIL_NAME: Final = "litellm.guardrail.name"
    GUARDRAIL_MODE: Final = "litellm.guardrail.mode"
    GUARDRAIL_STATUS: Final = "litellm.guardrail.status"
    GUARDRAIL_PROVIDER: Final = "litellm.guardrail.provider"
    GUARDRAIL_ACTION: Final = "litellm.guardrail.action"
    GUARDRAIL_RESPONSE: Final = "litellm.guardrail.response"
    GUARDRAIL_VIOLATION_CATEGORIES: Final = "litellm.guardrail.violation_categories"
    GUARDRAIL_CONFIDENCE_SCORE: Final = "litellm.guardrail.confidence_score"
    GUARDRAIL_RISK_SCORE: Final = "litellm.guardrail.risk_score"
    GUARDRAIL_MASKED_ENTITY_COUNT: Final = "litellm.guardrail.masked_entity_count"
    GUARDRAIL_DURATION: Final = "litellm.guardrail.duration"
    GUARDRAIL_ID: Final = "litellm.guardrail.id"
    GUARDRAIL_POLICY_TEMPLATE: Final = "litellm.guardrail.policy_template"
    GUARDRAIL_DETECTION_METHOD: Final = "litellm.guardrail.detection_method"
    SERVICE_NAME: Final = "litellm.service.name"
    SERVICE_CALL_TYPE: Final = "litellm.service.call_type"
    PREPROCESSING_MS: Final = "litellm.preprocessing.duration_ms"
    # The logical name of the MCP server a tool call was routed to. There is no
    # semconv key for an MCP server's *name* (the convention uses ``server.address``
    # for its network location), so it lives under the vendor namespace.
    MCP_SERVER_NAME: Final = "litellm.mcp.server.name"


class Metric:
    """GenAI metric instrument names."""

    TOKEN_USAGE: Final = "gen_ai.client.token.usage"
    OPERATION_DURATION: Final = "gen_ai.client.operation.duration"


# litellm ``custom_llm_provider`` -> ``gen_ai.provider.name`` value.
_PROVIDER_BY_LITELLM: dict[str, GenAIProvider] = {
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
_OPERATION_BY_CALL_TYPE: dict[str, GenAIOperation] = {
    "completion": GenAIOperation.CHAT,
    "acompletion": GenAIOperation.CHAT,
    "completion_with_retries": GenAIOperation.CHAT,
    "text_completion": GenAIOperation.TEXT_COMPLETION,
    "atext_completion": GenAIOperation.TEXT_COMPLETION,
    "embedding": GenAIOperation.EMBEDDINGS,
    "aembedding": GenAIOperation.EMBEDDINGS,
    "responses": GenAIOperation.CHAT,
    "aresponses": GenAIOperation.CHAT,
    "call_mcp_tool": GenAIOperation.EXECUTE_TOOL,
}


def resolve_provider(custom_llm_provider: str | None) -> str:
    """Map a litellm provider string to a ``gen_ai.provider.name`` value.

    Unknown providers pass through verbatim — the convention explicitly allows
    provider-specific values, so an unmapped name is still valid.
    """
    if not custom_llm_provider:
        return ""
    mapped = _PROVIDER_BY_LITELLM.get(custom_llm_provider.lower())
    return mapped.value if mapped is not None else custom_llm_provider


def resolve_operation(call_type: str | None) -> GenAIOperation:
    """Map a litellm ``call_type`` to a ``gen_ai.operation.name`` value."""
    if not call_type:
        return GenAIOperation.CHAT
    return _OPERATION_BY_CALL_TYPE.get(call_type.lower(), GenAIOperation.CHAT)
