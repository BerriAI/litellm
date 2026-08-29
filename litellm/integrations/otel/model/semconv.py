"""
Keys follow the OpenTelemetry GenAI semantic conventions (experimental). Anything
without a semconv equivalent lives under the ``litellm.*`` vendor namespace.
"""

from collections.abc import Mapping
from enum import Enum
from types import MappingProxyType
from typing import Final

from litellm._logging import verbose_logger


class GenAIOperation(str, Enum):
    """Values for ``gen_ai.operation.name``.

    The first block is the convention's own vocabulary. The ``LITELLM_`` members
    are vendor values for operations the convention names nothing for; its note
    on this attribute directs instrumentation to use a system-specific name in
    exactly that case, the same allowance :func:`resolve_provider` relies on for
    unmapped providers. They stay under the ``litellm.`` prefix so a value the
    convention adds later can never collide with one of ours.
    """

    CHAT = "chat"
    TEXT_COMPLETION = "text_completion"
    EMBEDDINGS = "embeddings"
    GENERATE_CONTENT = "generate_content"
    RETRIEVAL = "retrieval"  # vector-store search / RAG query spans
    CREATE_AGENT = "create_agent"  # reserved for future agent spans
    INVOKE_AGENT = "invoke_agent"  # agent (A2A) message spans
    EXECUTE_TOOL = "execute_tool"  # MCP tool-call spans
    LITELLM_VECTOR_STORE_MANAGEMENT = "litellm.vector_store_management"
    LITELLM_VECTOR_STORE_FILE_MANAGEMENT = "litellm.vector_store_file_management"
    LITELLM_RESPONSES_MANAGEMENT = "litellm.responses_management"
    LITELLM_MODERATION = "litellm.moderation"


class GenAIOutputType(str, Enum):
    """Values for ``gen_ai.output.type``, the modality the client asked for.

    It is what separates the inference routes that share ``generate_content``:
    image generation requests ``image``, speech requests ``speech``, and
    transcription and OCR both request ``text``.
    """

    TEXT = "text"
    JSON = "json"
    IMAGE = "image"
    SPEECH = "speech"


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
    """Canonical OTel GenAI attribute keys.

    ``SYSTEM`` is the one exception: the convention deprecated it in favor of
    ``PROVIDER_NAME``, and it survives here only so already-shipped series keep
    resolving for consumers that query it. Nothing new should use it.
    """

    # request
    OPERATION_NAME: Final = "gen_ai.operation.name"
    PROVIDER_NAME: Final = "gen_ai.provider.name"
    SYSTEM: Final = "gen_ai.system"
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
    RESPONSE_TIME_TO_FIRST_CHUNK: Final = "gen_ai.response.time_to_first_chunk"
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

    SYSTEM: Final = "rpc.system"
    REQUEST_ID: Final = "jsonrpc.request.id"
    PROTOCOL_VERSION: Final = "jsonrpc.protocol.version"
    RESPONSE_STATUS_CODE: Final = "rpc.response.status_code"


class RpcSystem(str, Enum):
    """Well-known values for ``rpc.system``. MCP frames every message as JSON-RPC 2.0.

    Naming the system also classifies the span: a CLIENT span carrying none of the
    ``rpc.*``/``http.*``/``db.*``/``messaging.*`` families records no span type or
    subtype in backends that derive those from the attribute family. It is emitted
    only alongside ``server.address``/``server.port``, since a backend that reads it
    as a downstream dependency names that dependency from the server address.
    """

    JSONRPC = "jsonrpc"


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
    """OTel-defined error attribute keys, from the semconv ``error.*`` registry.
    ``MESSAGE`` is marked *Deprecated* upstream in favor of domain-specific
    error message keys plus ``exception.message`` on the exception event, but
    litellm still stamps it."""

    TYPE: Final = "error.type"
    MESSAGE: Final = "error.message"


class LiteLLMError:
    """Detail keys for the mapped provider exception of a failed LLM call.
    OTel semconv does not define these, so they live under the ``litellm.*``
    vendor namespace rather than squatting on the semconv-owned ``error.*``
    namespace."""

    CODE: Final = "litellm.provider.error.code"
    STACK_TRACE: Final = "litellm.provider.error.stack_trace"
    LLM_PROVIDER: Final = "litellm.provider.error.llm_provider"


class ExceptionEvent:
    """OTel exception-event name and attribute keys (semconv ``exception.*``).

    The full error message rides ``exception.message`` on a span event rather than
    a custom string attribute. Backends recognise these semantic-convention names
    and map them as full text; an unrecognised key (e.g. ``error_message``) falls
    into the default dynamic template, which truncates strings to a 1024-char
    ``keyword``.
    """

    NAME: Final = "exception"
    TYPE: Final = "exception.type"
    MESSAGE: Final = "exception.message"
    STACKTRACE: Final = "exception.stacktrace"


class GenAIEvent:
    """GenAI semconv event names, from the GenAI registry's *events* section.

    ``gen_ai.client.operation.exception`` is defined as a log-based event
    (severity WARN) carrying the ``exception.*`` trio, correlated to the failed
    span via the trace/span ids — the semconv-compliant home for GenAI failure
    details, unlike the deprecated ``error.message`` span attribute.
    """

    OPERATION_EXCEPTION: Final = "gen_ai.client.operation.exception"


class Server:
    ADDRESS: Final = "server.address"
    PORT: Final = "server.port"


class DB:
    """Database / cache client-span keys (OTel ``db.*`` semconv).

    Stamped on ``DB_CALL`` spans (redis / postgres), which are CLIENT spans for
    outbound datastore calls — not on the INTERNAL ``SERVICE`` spans.
    """

    SYSTEM_NAME: Final = "db.system.name"
    # Superseded by SYSTEM_NAME, dual-emitted because Datadog's OTLP intake
    # still infers a span's database type from this key.
    SYSTEM_LEGACY: Final = "db.system"
    OPERATION_NAME: Final = "db.operation.name"
    NAMESPACE: Final = "db.namespace"


class HTTP:
    """HTTP server-span keys. Belong on the SERVER span only (never promoted)."""

    REQUEST_METHOD: Final = "http.request.method"
    ROUTE: Final = "http.route"
    RESPONSE_STATUS_CODE: Final = "http.response.status_code"
    URL_PATH: Final = "url.path"


class LiteLLM:
    """Vendor-extension keys (no semconv equivalent). Always ``litellm.*``."""

    CALL_ID: Final = "litellm.call_id"
    # The litellm route that produced the call. Needed because the convention maps
    # several routes onto one operation: transcription and OCR are both
    # ``generate_content`` with a ``text`` output type, so this is the only thing
    # that tells them apart.
    CALL_TYPE: Final = "litellm.call_type"
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
    TOOLS_DECLARED: Final = "litellm.request.tools.declared"
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
    # Provider-reported billable usage counters, JSON-serialized into one value.
    GUARDRAIL_USAGE: Final = "litellm.guardrail.usage"
    # Numeric USD cost of the guardrail invocation; lives under the litellm.cost.*
    # namespace (COST_PREFIX) beside the LLM call's litellm.cost.total.
    GUARDRAIL_COST: Final = "litellm.cost.guardrail"
    # Whether litellm.cost.guardrail is already inside litellm.cost.total (True,
    # the billed default) or reported alongside it (False) — without this a trace
    # consumer cannot tell whether adding the two double-counts.
    GUARDRAIL_COST_IN_SPEND: Final = "litellm.guardrail.cost_in_spend"
    SERVICE_NAME: Final = "litellm.service.name"
    SERVICE_CALL_TYPE: Final = "litellm.service.call_type"
    PREPROCESSING_MS: Final = "litellm.preprocessing.duration_ms"
    # The logical name of the MCP server a tool call was routed to. There is no
    # semconv key for an MCP server's *name* (the convention uses ``server.address``
    # for its network location), so it lives under the vendor namespace.
    MCP_SERVER_NAME: Final = "litellm.mcp.server.name"


class Metric:
    """GenAI metric instrument names.

    Every name here that a convention or a backend defines uses that name, so a
    consumer charting GenAI telemetry finds litellm's series where it looks for
    them. ``TOKEN_USAGE``, ``OPERATION_DURATION``, ``TIME_TO_FIRST_TOKEN`` and
    ``TIME_PER_OUTPUT_TOKEN`` are semconv instruments, defined in the GenAI
    conventions; the ``gen_ai.client.response.*`` spellings litellm used for the
    latter two are not conventions at all, so nothing downstream could chart
    them. Cost has no semconv instrument, so it takes ``gen_ai.usage.cost``, the
    name backends already query for spend.

    ``RESPONSE_DURATION`` keeps its vendor spelling deliberately: the closest
    convention, ``gen_ai.server.request.duration``, would collide in meaning with
    ``OPERATION_DURATION``, which litellm already emits for the whole operation.
    """

    TOKEN_USAGE: Final = "gen_ai.client.token.usage"
    OPERATION_DURATION: Final = "gen_ai.client.operation.duration"
    TOKEN_COST: Final = "gen_ai.usage.cost"
    TIME_TO_FIRST_TOKEN: Final = "gen_ai.server.time_to_first_token"
    TIME_PER_OUTPUT_TOKEN: Final = "gen_ai.server.time_per_output_token"
    RESPONSE_DURATION: Final = "gen_ai.client.response.duration"


# litellm ``custom_llm_provider`` -> ``gen_ai.provider.name`` value.
_PROVIDER_BY_LITELLM: Final[dict[str, GenAIProvider]] = {
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
_OPERATION_BY_CALL_TYPE: Final[dict[str, GenAIOperation]] = {
    "completion": GenAIOperation.CHAT,
    "acompletion": GenAIOperation.CHAT,
    "completion_with_retries": GenAIOperation.CHAT,
    "text_completion": GenAIOperation.TEXT_COMPLETION,
    "atext_completion": GenAIOperation.TEXT_COMPLETION,
    "embedding": GenAIOperation.EMBEDDINGS,
    "aembedding": GenAIOperation.EMBEDDINGS,
    "responses": GenAIOperation.CHAT,
    "aresponses": GenAIOperation.CHAT,
    "get_responses": GenAIOperation.LITELLM_RESPONSES_MANAGEMENT,
    "aget_responses": GenAIOperation.LITELLM_RESPONSES_MANAGEMENT,
    "delete_responses": GenAIOperation.LITELLM_RESPONSES_MANAGEMENT,
    "adelete_responses": GenAIOperation.LITELLM_RESPONSES_MANAGEMENT,
    "cancel_responses": GenAIOperation.LITELLM_RESPONSES_MANAGEMENT,
    "acancel_responses": GenAIOperation.LITELLM_RESPONSES_MANAGEMENT,
    "list_input_items": GenAIOperation.LITELLM_RESPONSES_MANAGEMENT,
    "alist_input_items": GenAIOperation.LITELLM_RESPONSES_MANAGEMENT,
    "image_generation": GenAIOperation.GENERATE_CONTENT,
    "aimage_generation": GenAIOperation.GENERATE_CONTENT,
    "moderation": GenAIOperation.LITELLM_MODERATION,
    "amoderation": GenAIOperation.LITELLM_MODERATION,
    "ocr": GenAIOperation.GENERATE_CONTENT,
    "aocr": GenAIOperation.GENERATE_CONTENT,
    "speech": GenAIOperation.GENERATE_CONTENT,
    "aspeech": GenAIOperation.GENERATE_CONTENT,
    "transcription": GenAIOperation.GENERATE_CONTENT,
    "atranscription": GenAIOperation.GENERATE_CONTENT,
    "call_mcp_tool": GenAIOperation.EXECUTE_TOOL,
    "vector_store_search": GenAIOperation.RETRIEVAL,
    "avector_store_search": GenAIOperation.RETRIEVAL,
    "query": GenAIOperation.RETRIEVAL,
    "aquery": GenAIOperation.RETRIEVAL,
    "send_message": GenAIOperation.INVOKE_AGENT,
    "asend_message": GenAIOperation.INVOKE_AGENT,
    "asend_message_streaming": GenAIOperation.INVOKE_AGENT,
    "vector_store_create": GenAIOperation.LITELLM_VECTOR_STORE_MANAGEMENT,
    "avector_store_create": GenAIOperation.LITELLM_VECTOR_STORE_MANAGEMENT,
    "vector_store_retrieve": GenAIOperation.LITELLM_VECTOR_STORE_MANAGEMENT,
    "avector_store_retrieve": GenAIOperation.LITELLM_VECTOR_STORE_MANAGEMENT,
    "vector_store_list": GenAIOperation.LITELLM_VECTOR_STORE_MANAGEMENT,
    "avector_store_list": GenAIOperation.LITELLM_VECTOR_STORE_MANAGEMENT,
    "vector_store_update": GenAIOperation.LITELLM_VECTOR_STORE_MANAGEMENT,
    "avector_store_update": GenAIOperation.LITELLM_VECTOR_STORE_MANAGEMENT,
    "vector_store_delete": GenAIOperation.LITELLM_VECTOR_STORE_MANAGEMENT,
    "avector_store_delete": GenAIOperation.LITELLM_VECTOR_STORE_MANAGEMENT,
    "vector_store_file_create": GenAIOperation.LITELLM_VECTOR_STORE_FILE_MANAGEMENT,
    "avector_store_file_create": GenAIOperation.LITELLM_VECTOR_STORE_FILE_MANAGEMENT,
    "vector_store_file_list": GenAIOperation.LITELLM_VECTOR_STORE_FILE_MANAGEMENT,
    "avector_store_file_list": GenAIOperation.LITELLM_VECTOR_STORE_FILE_MANAGEMENT,
    "vector_store_file_retrieve": GenAIOperation.LITELLM_VECTOR_STORE_FILE_MANAGEMENT,
    "avector_store_file_retrieve": GenAIOperation.LITELLM_VECTOR_STORE_FILE_MANAGEMENT,
    "vector_store_file_content": GenAIOperation.LITELLM_VECTOR_STORE_FILE_MANAGEMENT,
    "avector_store_file_content": GenAIOperation.LITELLM_VECTOR_STORE_FILE_MANAGEMENT,
    "vector_store_file_update": GenAIOperation.LITELLM_VECTOR_STORE_FILE_MANAGEMENT,
    "avector_store_file_update": GenAIOperation.LITELLM_VECTOR_STORE_FILE_MANAGEMENT,
    "vector_store_file_delete": GenAIOperation.LITELLM_VECTOR_STORE_FILE_MANAGEMENT,
    "avector_store_file_delete": GenAIOperation.LITELLM_VECTOR_STORE_FILE_MANAGEMENT,
}


# litellm ``call_type`` -> ``gen_ai.output.type``. Only the call types whose route
# fixes the requested modality are listed; the attribute is conditionally required
# on a request that asks for an output format, so anything else is left unstamped.
_OUTPUT_TYPE_BY_CALL_TYPE: Final[Mapping[str, GenAIOutputType]] = MappingProxyType(
    {
        "image_generation": GenAIOutputType.IMAGE,
        "aimage_generation": GenAIOutputType.IMAGE,
        "speech": GenAIOutputType.SPEECH,
        "aspeech": GenAIOutputType.SPEECH,
        "transcription": GenAIOutputType.TEXT,
        "atranscription": GenAIOutputType.TEXT,
        "ocr": GenAIOutputType.TEXT,
        "aocr": GenAIOutputType.TEXT,
    }
)


def resolve_provider(custom_llm_provider: str | None) -> str:
    """Map a litellm provider string to a ``gen_ai.provider.name`` value.

    Unknown providers pass through verbatim — the convention explicitly allows
    provider-specific values, so an unmapped name is still valid.
    """
    if not custom_llm_provider:
        return ""
    mapped: Final = _PROVIDER_BY_LITELLM.get(custom_llm_provider.lower())
    return mapped.value if mapped is not None else custom_llm_provider


def resolve_operation(call_type: str | None) -> GenAIOperation:
    """Map a litellm ``call_type`` to a ``gen_ai.operation.name`` value.

    An unmapped call type still falls back to ``chat`` so every series keeps an
    operation label, but it logs at debug rather than falling through silently:
    a new call type mislabelled as ``chat`` mixes its latency and cost into
    everyone's chat charts, which is invisible until someone reads the numbers.
    """
    if not call_type:
        return GenAIOperation.CHAT
    mapped: Final = _OPERATION_BY_CALL_TYPE.get(call_type.lower())
    if mapped is not None:
        return mapped
    verbose_logger.debug(
        "otel: call_type %r has no gen_ai.operation.name mapping; labelling it %r. Add it to _OPERATION_BY_CALL_TYPE.",
        call_type,
        GenAIOperation.CHAT.value,
    )
    return GenAIOperation.CHAT


def resolve_output_type(call_type: str | None) -> GenAIOutputType | None:
    """Map a litellm ``call_type`` to a ``gen_ai.output.type`` value, or ``None``
    for a route that doesn't pin the output modality."""
    if not call_type:
        return None
    return _OUTPUT_TYPE_BY_CALL_TYPE.get(call_type.lower())
