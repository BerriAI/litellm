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

    fire_and_forget: bool | None = Field(
        default=None,
        description=(
            "If True, the guardrail HTTP call is dispatched as a background task and the "
            "request proceeds immediately without awaiting the response. Applies to every "
            "mode (pre_call, post_call, during_call), adding ~0 latency. Because the "
            "response is never awaited, the guardrail is observe-only: action=BLOCKED and "
            "action=GUARDRAIL_INTERVENED are ignored. Also forces "
            "streaming_end_of_stream_only=True so a stream dispatches one background call "
            "instead of one per sampled chunk. Defaults to False in "
            "GenericGuardrailAPI.__init__ when None."
        ),
    )

    fire_and_forget_max_inflight: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Maximum number of fire_and_forget calls in flight at once. Async dispatch "
            "decouples the request rate from the guardrail endpoint's throughput, so a slow "
            "endpoint would otherwise pile up tasks without bound. Calls beyond this limit "
            "are dropped and counted, with a rate-limited warning. Ignored unless "
            "fire_and_forget is True. Defaults to 100 in GenericGuardrailAPI.__init__ when None."
        ),
    )

    send_images: bool | None = Field(
        default=None,
        description=(
            "If False, base64 image data is omitted from the guardrail request even when the "
            "LLM request contains images. Large payload saver for guardrails that only "
            "inspect text. Images the guardrail never received cannot be rewritten by its "
            "response. Defaults to True in GenericGuardrailAPI.__init__ when None."
        ),
    )

    exclude_payload_fields: tuple[str, ...] | None = Field(
        default=None,
        description=(
            "Top-level GenericGuardrailAPIRequest keys to omit from the payload (e.g. "
            "['images','texts','request_headers']), for providers that do not consume them. "
            "Unknown keys and the routing-critical keys input_type and litellm_call_id are "
            "ignored with a warning at init. A component that is not sent cannot be "
            "rewritten by the guardrail response."
        ),
    )

    max_messages: int | None = Field(
        default=None,
        ge=1,
        description=(
            "If set, only the last N entries of structured_messages and the last N text "
            "blocks are sent. Bounds payload size when the full conversation is re-sent "
            "every turn. Note the system prompt and early context fall out of the window "
            "once the session exceeds N. LOSSY: windowing shifts text positions, so the "
            "guardrail can no longer rewrite text on this call (action=BLOCKED still "
            "applies); leave it unset on a guardrail that masks or redacts."
        ),
    )

    max_text_chars: int | None = Field(
        default=None,
        ge=1,
        description=(
            "If set, each individual text block is truncated to this many characters before "
            "sending, for providers that only need a prefix. LOSSY: truncated text blocks "
            "keep their original content on write-back, so the guardrail cannot rewrite them."
        ),
    )

    strip_patterns: tuple[str, ...] | None = Field(
        default=None,
        description=(
            "Regex patterns applied to text content (texts[] and the text of each "
            "structured_messages entry) before the guardrail request is sent. Matches are "
            "removed. Intended to drop volatile boilerplate the provider does not need. "
            "Applied only to string text fields, never to JSON structure, tool schemas, ids "
            "or metadata. LOSSY: stripped content cannot be inspected by the guardrail, and "
            "stripped text blocks keep their original content on write-back. An invalid "
            "regex raises at init. Patterns run synchronously on the request path against "
            "caller-supplied text, and Python's re has no match timeout, so a pattern that "
            "backtracks catastrophically (nested quantifiers such as (a+)+) can stall the "
            "worker: keep patterns linear-time. Text over 100k characters is not matched at "
            "all, and is sent unstripped so an enforcing guardrail still sees it."
        ),
    )

    skip_if_system_prompt_matches: tuple[str, ...] | None = Field(
        default=None,
        description=(
            "Regex patterns matched against the request's system message (role=system or "
            "developer). On a match the guardrail is skipped for this call: no request is "
            "sent, and the paired response is skipped too, so both telemetry and enforcement "
            "are suppressed for matched requests. Matching the system message only (not "
            "arbitrary user text) avoids false positives from pasted content. Requires a "
            "request-side hook (pre_call or during_call) in the mode list. "
            "TRUST BOUNDARY: the system message comes from the request body, so any caller "
            "who knows the configured pattern can add a system message and exempt itself "
            "from this guardrail. Treat this as traffic scoping, not as enforcement, and "
            "prefer skip_if_key_alias_in / skip_if_team_id_in when the exemption has to "
            "hold against the caller. Patterns run synchronously against caller-supplied "
            "text, so they must be linear-time for the same reason as strip_patterns."
        ),
    )

    skip_if_first_role_in: tuple[str, ...] | None = Field(
        default=None,
        description=(
            "If the first message's role is in this list (e.g. ['developer']), skip the "
            "guardrail for the call, with the same request/response semantics as, and the "
            "same trust boundary as, skip_if_system_prompt_matches: the caller chooses the "
            "roles it sends."
        ),
    )

    skip_if_key_alias_in: tuple[str, ...] | None = Field(
        default=None,
        description=(
            "If the calling virtual key's alias is in this list, skip the guardrail for the "
            "call. Unlike the message-based filters this reads what authentication "
            "established, so a caller cannot exempt itself by changing its request body. "
            "Same request/response semantics: neither side is sent."
        ),
    )

    skip_if_team_id_in: tuple[str, ...] | None = Field(
        default=None,
        description=(
            "If the calling key's team id is in this list, skip the guardrail for the call. "
            "Admin-controlled like skip_if_key_alias_in, and matched on the same "
            "authenticated metadata."
        ),
    )

    run_only_on_call_types: tuple[str, ...] | None = Field(
        default=None,
        description=(
            "If set, the guardrail runs ONLY for these call types (e.g. "
            "['completion','acompletion','anthropic_messages','responses','aresponses']). All "
            "other call types, including embeddings, image generation, audio, rerank and "
            "moderation, are skipped before any request is sent. Allowlist; takes precedence "
            "over skip_call_types. Values are CallTypes names. An unresolvable call type runs."
        ),
    )

    skip_call_types: tuple[str, ...] | None = Field(
        default=None,
        description=(
            "If set (and run_only_on_call_types is not), the guardrail is skipped for these "
            "call types (e.g. ['embedding','aembedding','image_generation','transcription',"
            "'speech','rerank','moderation']). Denylist. Values are CallTypes names."
        ),
    )

    guardrail_information_scope: Literal["per_call", "per_session", "off"] | None = Field(
        default=None,
        description=(
            "How often this guardrail records a StandardLoggingGuardrailInformation entry "
            "into request metadata (spend logs / OTEL). Every invocation records one today, "
            "with no cap and no dedup, so a long agent session accumulates one entry per "
            "guardrail call in the same row. 'per_call' (default) keeps that behavior. "
            "'per_session' records only the first call of a session, and needs a resolvable "
            "session id (litellm_session_id or metadata.session_id); without one it behaves "
            "as 'per_call' rather than dropping every entry. 'off' records nothing on "
            "success. Blocks and guardrail failures are recorded under every scope. Dedup "
            "is per proxy process, so a session spread across workers records once per "
            "worker. Defaults to 'per_call' in GenericGuardrailAPI.__init__ when None."
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
