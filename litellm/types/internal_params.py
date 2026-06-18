from enum import Enum


class LiteLLMInternalParam(str, Enum):
    """optional_params keys LiteLLM consumes internally and must never serialize into a provider request body.

    Strict-schema providers (Bedrock and a growing set of others) reject unknown
    fields with a hard 400, so any of these leaking into the wire payload fails
    the whole request. This enum is the single source of truth: the request-body
    filter derives its key set here, so a newly added internal knob is covered by
    adding one member instead of remembering to pop it at every splat site.
    """

    SKIP_MCP_HANDLER = "skip_mcp_handler"
    PRIVATE_SKIP_MCP_HANDLER = "_skip_mcp_handler"
    MCP_HANDLER_CONTEXT = "mcp_handler_context"
    STREAM_CHUNK_SIZE = "stream_chunk_size"
    FAKE_STREAM = "fake_stream"
    CACHE_CONTROL_INJECTION_POINTS = "cache_control_injection_points"


LITELLM_INTERNAL_REQUEST_BODY_PARAMS: frozenset[str] = frozenset(
    member.value for member in LiteLLMInternalParam
)

MCP_INTERNAL_PARAMS: frozenset[str] = frozenset(
    {
        LiteLLMInternalParam.SKIP_MCP_HANDLER.value,
        LiteLLMInternalParam.PRIVATE_SKIP_MCP_HANDLER.value,
        LiteLLMInternalParam.MCP_HANDLER_CONTEXT.value,
    }
)
