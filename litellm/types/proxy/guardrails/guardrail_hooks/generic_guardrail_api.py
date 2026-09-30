from collections.abc import Mapping, Sequence
from typing import Any, Final, Literal, cast  # noqa: TID251  # JSON chat rows have no typed constructor across roles

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

    send_images: bool | None = Field(
        default=None,
        description=(
            "If False, the top-level images field is not sent, and every inline image_url part in "
            "structured_messages keeps its place but has its URL replaced by '[omitted]'. File, "
            "audio and video parts are still sent as they are. Saves payload size for guardrails "
            "that only inspect text. The guardrail cannot replace the caller's images: a rewritten "
            "message gets the caller's image back in each part still holding '[omitted]', and a "
            "rewrite that changes or moves an image part is rejected. Defaults to True in "
            "GenericGuardrailAPI.__init__ when None."
        ),
    )

    exclude_payload_fields: tuple[str, ...] | None = Field(
        default=None,
        description=(
            "Top-level guardrail request fields to leave out of the payload, e.g. "
            "['request_headers', 'tools'], for guardrails that do not use them. Unknown fields "
            "are ignored with a warning at init, and input_type and litellm_call_id are always "
            "sent. A field that is not sent (texts, structured_messages, images or tools) cannot "
            "be rewritten by the guardrail response."
        ),
    )

    max_messages: int | None = Field(
        default=None,
        ge=1,
        description=(
            "If set and a request has more than N structured_messages, only the last N are sent, "
            "and texts is rebuilt from the text of those N messages. Calls without "
            "structured_messages, such as embeddings, rerank or an LLM response, are not affected. "
            "images and tool_calls are not windowed. Bounds payload size when the whole conversation "
            "is re-sent every turn, but the system prompt and early turns fall out of the window. "
            "For block-only or observe-only guardrails: on a windowed call BLOCKED still applies, "
            "but any rewrite the guardrail returns fails the call. A failed request or response is "
            "rejected with an error, and a failed stream is cut off after the chunks already sent."
        ),
    )

    max_text_chars: int | None = Field(
        default=None,
        ge=1,
        description=(
            "If set, every text in texts and in structured_messages content is cut to this many "
            "characters before sending, so a caller can put content the guardrail never sees after "
            "the first N characters. For block-only or observe-only guardrails: when any text was "
            "cut, BLOCKED still applies, but any rewrite the guardrail returns fails the call, with "
            "the same errors as max_messages."
        ),
    )

    strip_patterns: tuple[str, ...] | None = Field(
        default=None,
        description=(
            "Regexes whose matches are removed from every text in texts and in structured_messages "
            "content before sending, e.g. volatile boilerplate the guardrail does not need. Roles, "
            "ids, tool calls, tools and metadata are never touched. A caller can hide content from "
            "the guardrail by wrapping it in something a pattern matches. For block-only or "
            "observe-only guardrails: when any text was stripped, BLOCKED still applies, but any "
            "rewrite the guardrail returns fails the call, with the same errors as max_messages. "
            "An invalid regex raises at init. Patterns use the regex package and run against "
            "caller requests and LLM responses alike. Each pattern removes at most 64 matches per "
            "text. Per guardrail call, only the first 100,000 characters of distinct text are "
            "stripped and stripping stops after 0.1 seconds. A text past either limit is sent "
            "unstripped in full with a warning. Stripping runs on the worker's event loop, so a slow "
            "pattern blocks that worker, and every request on it, for up to 0.1 seconds per "
            "guardrail call. Keep patterns linear-time: no nested quantifiers such as (a+)+ and no "
            "lazy match up to a closing delimiter such as <!--.*?-->."
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


def structured_messages_from_json(value: object) -> Sequence[AllMessageValues] | None:
    if not isinstance(value, list):
        return None
    if not all(isinstance(message, Mapping) and isinstance(message.get("role"), str) for message in value):
        return None
    return cast("Sequence[AllMessageValues]", value)  # cast-ok: JSON rows checked for a role, the same trust texts get


class GenericGuardrailAPIResponse:
    """Response model for the Generic Guardrail API"""

    texts: list[str] | None
    images: list[str] | None
    tools: list[GuardrailToolParam] | None
    structured_messages: Sequence[AllMessageValues] | None
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
        structured_messages: Sequence[AllMessageValues] | None = None,
    ) -> None:
        self.action = action
        self.blocked_reason = blocked_reason
        self.texts = texts
        self.images = images
        self.tools = tools
        self.structured_messages = structured_messages
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
            structured_messages=structured_messages_from_json(data.get("structured_messages")),
        )
