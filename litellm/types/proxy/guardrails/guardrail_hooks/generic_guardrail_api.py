from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import TypedDict

from litellm.types.llms.openai import (
    AllMessageValues,
    ChatCompletionToolCallChunk,
)
from litellm.types.proxy.guardrails.guardrail_hooks.base import GuardrailConfigModel
from litellm.types.utils import ChatCompletionMessageToolCall


class GuardrailToolParam(BaseModel):
    """A tool forwarded verbatim to the guardrail for inspection.

    Built-in tools (code_interpreter, file_search, ...) have no ``function`` block
    and stash their config in tool-specific keys, so only ``type`` is required and
    ``extra="allow"`` preserves the rest instead of stripping it.
    """

    model_config = ConfigDict(extra="allow")
    type: str


class GenericGuardrailAPIMetadata(TypedDict, total=False):
    user_api_key_hash: str | None
    user_api_key_alias: str | None
    user_api_key_user_id: str | None
    user_api_key_user_email: str | None
    user_api_key_team_id: str | None
    user_api_key_team_alias: str | None
    user_api_key_end_user_id: str | None
    user_api_key_org_id: str | None


class GenericGuardrailAPIOptionalParams(BaseModel):
    """Optional parameters for the Generic Guardrail API"""

    additional_provider_specific_params: dict[str, Any] | None = Field(
        default=None,
        description="Additional provider-specific parameters to send with the guardrail request",
    )

    unreachable_fallback: Literal["fail_closed", "fail_open"] | None = Field(
        default="fail_closed",
        description=(
            "Behavior when the guardrail endpoint is unreachable due to network errors. "
            "'fail_closed' raises an error (default). 'fail_open' logs a critical error and allows the request to proceed."
        ),
    )

    fail_on_error: bool | None = Field(
        default=True,
        description=(
            "Behavior on any guardrail error, not just unreachability. "
            "True (default) raises and blocks the request on error. "
            "False logs a critical error and allows the request to proceed, so only a valid "
            "guardrail response can block or modify it; broader than unreachable_fallback."
        ),
    )

    streaming_end_of_stream_only: bool | None = Field(
        default=None,
        description=(
            "If False (default when unset), the guardrail runs on sampled chunks during "
            "the stream at the cadence set by streaming_sampling_rate, and an in-flight "
            "BLOCKED stops further chunks from streaming. If True, the guardrail runs "
            "once at end of stream over the assembled response; lower cost and latency, "
            "but flagged content has already streamed to the client before the terminal "
            "block. Defaults are applied in GenericGuardrailAPI.__init__ when None so "
            "unset optional_params does not shadow top-level litellm_params."
        ),
    )

    streaming_sampling_rate: int | None = Field(
        default=None,
        ge=1,
        description=(
            "When streaming_end_of_stream_only is False, the guardrail runs every Nth "
            "streamed chunk. Ignored when streaming_end_of_stream_only is True. "
            "Must be >= 1 when set. Defaults to 5 in GenericGuardrailAPI.__init__ "
            "when None so unset optional_params does not shadow top-level litellm_params."
        ),
    )

    streaming_transform_mode: Literal["block_only", "incremental_diff"] | None = Field(
        default=None,
        description=(
            "Controls whether text modifications returned by the guardrail (action="
            "GUARDRAIL_INTERVENED with modified texts) reach the client on the streaming "
            "path. 'block_only' (default) preserves the historical behavior: the raw "
            "upstream chunks are streamed and only a BLOCK terminates the stream; text "
            "rewrites are dropped. 'incremental_diff' withholds the raw chunks and instead "
            "emits the guardrailed text as new deltas computed by diffing the mutated "
            "accumulated text against what has already been sent, enabling PII masking, "
            "pseudonym reversal, redaction and similar rewrites over HTTP. Only supported "
            "for the OpenAI chat completions streaming path (string delta.content) and "
            "ignored when streaming_end_of_stream_only is True except for a single "
            "post-stream synthetic chunk. Defaults to 'block_only' in "
            "GenericGuardrailAPI.__init__ when None."
        ),
    )


class GenericGuardrailAPIConfigModel(
    GuardrailConfigModel[GenericGuardrailAPIOptionalParams],
):
    """Configuration parameters for the Generic Guardrail API guardrail"""

    optional_params: GenericGuardrailAPIOptionalParams | None = Field(
        default_factory=GenericGuardrailAPIOptionalParams,
        description="Optional parameters for the Generic Guardrail API guardrail",
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "Generic Guardrail API"


class GenericGuardrailAPIRequest(BaseModel):
    """Request model for the Generic Guardrail API"""

    input_type: Literal["request", "response"]
    litellm_call_id: str | None = None  # the call id of the individual LLM call
    litellm_trace_id: str | None = (
        None  # the trace id of the LLM call - useful if there are multiple LLM calls for the same conversation
    )
    structured_messages: list[AllMessageValues] | None = None
    images: list[str] | None = None
    tools: list[GuardrailToolParam] | None = None
    texts: list[str] | None = None
    request_data: GenericGuardrailAPIMetadata
    request_headers: dict[str, str] | None = Field(
        default=None,
        description="Sanitized inbound request headers from the original proxy request.",
    )
    litellm_version: str | None = Field(
        default=None,
        description="LiteLLM library version running this proxy.",
    )
    additional_provider_specific_params: dict[str, Any] | None = None
    tool_calls: list[ChatCompletionToolCallChunk] | list[ChatCompletionMessageToolCall] | None = None
    model: str | None = None  # the model being used for the LLM call


def coerce_stream_holdback_value(value: Any) -> int:
    """Coerce a single ``stream_holdback_chars`` entry to a non-negative int.

    A guardrail returning a null, non-numeric, or negative holdback element must
    not abort the streaming round, so malformed values degrade to 0 (no holdback)
    rather than raising. Shared by response parsing (``from_dict``) and the
    handler that applies holdback to in-process guardrail return values.
    """
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


class GenericGuardrailAPIResponse:
    """Response model for the Generic Guardrail API"""

    texts: list[str] | None
    images: list[str] | None
    tools: list[GuardrailToolParam] | None
    action: str
    blocked_reason: str | None
    stream_holdback_chars: list[int] | None

    def __init__(
        self,
        action: str,
        texts: list[str] | None = None,
        blocked_reason: str | None = None,
        images: list[str] | None = None,
        tools: list[GuardrailToolParam] | None = None,
        stream_holdback_chars: list[int] | None = None,
    ) -> None:
        self.action = action
        self.blocked_reason = blocked_reason
        self.texts = texts
        self.images = images
        self.tools = tools
        # Number of trailing chars, indexed the same as ``texts``, that the
        # framework must withhold from streaming emission until the next
        # processing round (word-boundary safety for text transformations).
        self.stream_holdback_chars = stream_holdback_chars

    @classmethod
    def from_dict(cls, data: dict) -> "GenericGuardrailAPIResponse":
        raw_holdback: Final = data.get("stream_holdback_chars")
        stream_holdback_chars: Final = (
            [coerce_stream_holdback_value(value) for value in raw_holdback] if isinstance(raw_holdback, list) else None
        )
        return cls(
            action=data.get("action", "NONE"),
            blocked_reason=data.get("blocked_reason"),
            texts=data.get("texts"),
            images=data.get("images"),
            tools=data.get("tools"),
            stream_holdback_chars=stream_holdback_chars,
        )
