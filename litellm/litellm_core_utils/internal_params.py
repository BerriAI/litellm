from collections.abc import Mapping
from enum import Enum
from typing import Final, TypeVar


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


LITELLM_INTERNAL_REQUEST_BODY_PARAMS: Final[frozenset[str]] = frozenset(member.value for member in LiteLLMInternalParam)

LITELLM_CHAT_REQUEST_BODY_STRIP_PARAMS: Final[frozenset[str]] = LITELLM_INTERNAL_REQUEST_BODY_PARAMS - frozenset(
    {LiteLLMInternalParam.CACHE_CONTROL_INJECTION_POINTS.value}
)
"""Variant of `LITELLM_INTERNAL_REQUEST_BODY_PARAMS` for the chat-completion
boundary. `cache_control_injection_points` is consumed inside `transform_request`
by `AmazonConverseConfig` (it appends a `cachePoint` to the Bedrock tool list for
``location: "tool_config"``), so it must reach the transform on the
``converse_like/`` and other shared HTTP handler routes. The shared HTTP handler
re-applies the full strip to the body returned by `transform_request`, so
splat-style transforms cannot leak the preserved key into the wire payload."""

MCP_INTERNAL_PARAMS: Final[frozenset[str]] = frozenset(
    {
        LiteLLMInternalParam.SKIP_MCP_HANDLER.value,
        LiteLLMInternalParam.PRIVATE_SKIP_MCP_HANDLER.value,
        LiteLLMInternalParam.MCP_HANDLER_CONTEXT.value,
    }
)


_RequestParamValue = TypeVar("_RequestParamValue")


def strip_internal_params_from_request_body(data: Mapping[str, _RequestParamValue]) -> dict[str, _RequestParamValue]:
    """
    Remove every LiteLLM-internal optional_params key from a provider request body.

    Applied at the serialization boundary (where optional_params becomes a request
    body) so internal control knobs can never reach a provider that rejects unknown
    fields. See `litellm.litellm_core_utils.internal_params.LiteLLMInternalParam` for the registry.
    """
    return {k: v for k, v in data.items() if k not in LITELLM_INTERNAL_REQUEST_BODY_PARAMS}


def strip_internal_params_from_chat_request_body(
    data: Mapping[str, _RequestParamValue],
) -> dict[str, _RequestParamValue]:
    """
    Strip variant for the chat-completion boundary that preserves keys consumed
    inside `transform_request` (currently `cache_control_injection_points`, which
    `AmazonConverseConfig` reads to append a `cachePoint` to Bedrock tool_config).
    The shared chat handler re-applies `strip_internal_params_from_request_body`
    to the body returned by `transform_request`, so splat-style transforms that
    splat `**optional_params` into the wire body cannot leak the preserved key.
    """
    return {k: v for k, v in data.items() if k not in LITELLM_CHAT_REQUEST_BODY_STRIP_PARAMS}
