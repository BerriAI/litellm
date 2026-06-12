"""Frozen intermediate representation for chat-shaped requests.

This is the hub of the hub-and-spoke: inbound parsers map each accepted
request schema into these types, and provider serializers map these types
onto a wire format. Everything here is immutable. Product types are frozen
dataclasses, sum types are Expression tagged unions matched on their
``Literal`` tag (with ``assert_never`` on the final arm), and collections are
Expression ``Block``s. Boundary-validated JSON the package never inspects
(tool arguments, JSON schemas) rides as an opaque ``JsonBlob`` instead of
being recursively frozen, so the hot path does no freeze/thaw churn.
Nothing in this module performs I/O or imports a provider.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, fields
from typing import Literal, TypeVar, Union

from expression import Nothing, Option, case, tag, tagged_union
from expression.collections import Block, Map

Json = Union[None, bool, int, float, str, "Block[Json]", "Map[str, Json]"]

PlainJson = Union[
    None, bool, int, float, str, "list[PlainJson]", "dict[str, PlainJson]"
]

Body = dict[str, PlainJson]

Sampling = int | float
"""A numeric parameter kept as the caller sent it (int stays int on the wire)."""


@dataclass(frozen=True)
class Unit:
    """The single inhabited value for tagged-union cases that carry no data."""


UNIT = Unit()

_TUnion = TypeVar("_TUnion")


def _case_maker(cls: type[_TUnion], name: str) -> Callable[[object], _TUnion]:
    """Allocation-fast constructor for one case of an Expression tagged union.

    Produces instances bit-identical to ``cls(name=value)`` (same ``tag``,
    case attribute, ``_index`` and per-case ``__dataclass_fields__``), but
    precomputes the per-case constants once instead of rebuilding them on
    every call. Content blocks are built per message part on the hot path;
    the stock ``__init__`` costs ~4x more (pattern-auditor perf budget:
    v2 <= 1.5x v1 at 600-message histories).
    """
    field_list = fields(cls)  # type: ignore[arg-type]
    field_names = tuple(f.name for f in field_list)
    index = field_names.index(name)
    case_fields = {f.name: f for f in field_list if f.name in (name, "tag")}
    new = cls.__new__  # type: ignore[attr-defined]
    set_attr = object.__setattr__

    def make(value: object) -> _TUnion:
        instance = new(cls)
        set_attr(instance, "tag", name)
        set_attr(instance, name, value)
        set_attr(instance, "_index", index)
        set_attr(instance, "__dataclass_fields__", case_fields)
        return instance

    return make


@dataclass(frozen=True)
class JsonBlob:
    """Boundary-validated plain JSON the package treats as opaque.

    Built only by ``boundary.as_plain_json``, which deep-copies and checks
    every leaf, so the wrapped value has no aliases outside the package.
    Nothing in the package mutates it; serializers deep-copy on emit because
    returned bodies are plain dicts that downstream v1-era code may mutate.
    """

    value: PlainJson


@dataclass(frozen=True)
class CacheControl:
    type: str
    ttl: Option[str]


@dataclass(frozen=True)
class Base64Source:
    media_type: str
    data: str


@dataclass(frozen=True)
class UrlSource:
    url: str
    format: Option[str] = Nothing
    """Caller-supplied media type override; the anthropic family ignores it
    for URL sources (so does v1) but gemini uses it for ``file_data`` mime."""


@tagged_union(frozen=True)
class ImageSource:
    tag: Literal["base64", "url"] = tag()

    base64: Base64Source = case()
    url: UrlSource = case()

    @staticmethod
    def of_base64(value: Base64Source) -> ImageSource:
        return ImageSource(base64=value)

    @staticmethod
    def of_url(value: UrlSource) -> ImageSource:
        return ImageSource(url=value)


@dataclass(frozen=True)
class Text:
    text: str
    cache: Option[CacheControl]


@dataclass(frozen=True)
class Image:
    source: ImageSource
    cache: Option[CacheControl]


@dataclass(frozen=True)
class ToolUse:
    id: str
    name: str
    arguments: JsonBlob
    cache: Option[CacheControl]
    arguments_raw: Option[str] = Nothing
    """The verbatim wire argument string when the block came from an
    openai-format tool_call (replayed histories): same-family serializers
    re-emit these bytes so real compact-spaced (or blank) arguments
    round-trip byte-faithfully instead of falling back over re-dump
    spacing. Cross-family serializers use the parsed ``arguments``."""


@tagged_union(frozen=True)
class ToolResultContent:
    tag: Literal["text", "parts"] = tag()

    text: str = case()
    parts: Block[Text] = case()

    @staticmethod
    def of_text(value: str) -> ToolResultContent:
        return _tool_result_text(value)

    @staticmethod
    def of_parts(value: Block[Text]) -> ToolResultContent:
        return _tool_result_parts(value)


_tool_result_text = _case_maker(ToolResultContent, "text")
_tool_result_parts = _case_maker(ToolResultContent, "parts")


@dataclass(frozen=True)
class ToolResult:
    tool_use_id: str
    content: ToolResultContent
    cache: Option[CacheControl]


@dataclass(frozen=True)
class Thinking:
    thinking: str
    signature: Option[str]
    cache: Option[CacheControl]


@dataclass(frozen=True)
class RedactedThinking:
    data: str


@tagged_union(frozen=True)
class ContentBlock:
    tag: Literal[
        "text", "image", "tool_use", "tool_result", "thinking", "redacted_thinking"
    ] = tag()

    text: Text = case()
    image: Image = case()
    tool_use: ToolUse = case()
    tool_result: ToolResult = case()
    thinking: Thinking = case()
    redacted_thinking: RedactedThinking = case()

    @staticmethod
    def of_text(value: Text) -> ContentBlock:
        return _content_text(value)

    @staticmethod
    def of_image(value: Image) -> ContentBlock:
        return _content_image(value)

    @staticmethod
    def of_tool_use(value: ToolUse) -> ContentBlock:
        return _content_tool_use(value)

    @staticmethod
    def of_tool_result(value: ToolResult) -> ContentBlock:
        return _content_tool_result(value)

    @staticmethod
    def of_thinking(value: Thinking) -> ContentBlock:
        return _content_thinking(value)

    @staticmethod
    def of_redacted_thinking(value: RedactedThinking) -> ContentBlock:
        return _content_redacted_thinking(value)


_content_text = _case_maker(ContentBlock, "text")
_content_image = _case_maker(ContentBlock, "image")
_content_tool_use = _case_maker(ContentBlock, "tool_use")
_content_tool_result = _case_maker(ContentBlock, "tool_result")
_content_thinking = _case_maker(ContentBlock, "thinking")
_content_redacted_thinking = _case_maker(ContentBlock, "redacted_thinking")


Role = Literal["user", "assistant"]


@dataclass(frozen=True)
class Message:
    role: Role
    content: Block[ContentBlock]


@dataclass(frozen=True)
class SystemText:
    text: str
    cache: Option[CacheControl]


@dataclass(frozen=True)
class ToolDef:
    name: str
    description: Option[str]
    parameters: Option[JsonBlob]
    cache: Option[CacheControl]
    strict: Option[bool] = Nothing
    """OpenAI structured-outputs flag; passthrough providers re-emit it,
    anthropic-family serializers ignore it (v1 drops it there too)."""


@tagged_union(frozen=True)
class ToolChoice:
    tag: Literal["auto", "required", "none", "specific"] = tag()

    auto: Unit = case()
    required: Unit = case()
    none: Unit = case()
    specific: str = case()

    @staticmethod
    def of_auto() -> ToolChoice:
        return ToolChoice(auto=UNIT)

    @staticmethod
    def of_required() -> ToolChoice:
        return ToolChoice(required=UNIT)

    @staticmethod
    def of_none() -> ToolChoice:
        return ToolChoice(none=UNIT)

    @staticmethod
    def of_specific(name: str) -> ToolChoice:
        return ToolChoice(specific=name)


@dataclass(frozen=True)
class JsonSchemaSpec:
    schema: JsonBlob
    name: Option[str] = Nothing
    description: Option[str] = Nothing
    strict: Option[bool] = Nothing
    """``name``/``description``/``strict`` ride beside the schema on the
    OpenAI wire; passthrough providers re-emit them, anthropic-family
    serializers read only ``schema`` (v1 parity on both sides)."""


@tagged_union(frozen=True)
class ResponseFormat:
    tag: Literal["text", "json_object", "json_schema"] = tag()

    text: Unit = case()
    json_object: Unit = case()
    json_schema: JsonSchemaSpec = case()

    @staticmethod
    def of_text() -> ResponseFormat:
        return ResponseFormat(text=UNIT)

    @staticmethod
    def of_json_object() -> ResponseFormat:
        return ResponseFormat(json_object=UNIT)

    @staticmethod
    def of_json_schema(value: JsonSchemaSpec) -> ResponseFormat:
        return ResponseFormat(json_schema=value)


@dataclass(frozen=True)
class ThinkingEnabled:
    budget_tokens: Option[int]


@tagged_union(frozen=True)
class ThinkingParam:
    tag: Literal["enabled", "disabled", "adaptive"] = tag()

    enabled: ThinkingEnabled = case()
    disabled: Unit = case()
    adaptive: Unit = case()

    @staticmethod
    def of_enabled(budget_tokens: Option[int]) -> ThinkingParam:
        return ThinkingParam(enabled=ThinkingEnabled(budget_tokens=budget_tokens))

    @staticmethod
    def of_disabled() -> ThinkingParam:
        return ThinkingParam(disabled=UNIT)

    @staticmethod
    def of_adaptive() -> ThinkingParam:
        return ThinkingParam(adaptive=UNIT)


ReasoningEffort = Literal["minimal", "low", "medium", "high", "xhigh", "max", "none"]


@dataclass(frozen=True)
class InferenceParams:
    max_tokens: Option[int]
    temperature: Option[Sampling]
    top_p: Option[Sampling]
    top_k: Option[Sampling]
    stop: Block[str]
    max_completion_tokens: Option[int] = Nothing
    """The caller's verbatim ``max_completion_tokens`` when that key was sent.
    ``max_tokens`` above stays the collapsed value every non-passthrough
    provider reads; OpenAI-passthrough serializers re-emit the original key
    (the raw guard rejects requests carrying both keys)."""


@dataclass(frozen=True)
class ChatRequest:
    model: str
    system: Block[SystemText]
    messages: Block[Message]
    tools: Block[ToolDef]
    tool_choice: Option[ToolChoice]
    parallel_tool_calls: Option[bool]
    response_format: Option[ResponseFormat]
    thinking: Option[ThinkingParam]
    reasoning_effort: Option[ReasoningEffort]
    user: Option[str]
    params: InferenceParams
    stream: bool


def has_tool_blocks(messages: Block[Message]) -> bool:
    """True when any message carries a tool_use or tool_result content block."""
    return any(
        block.tag in ("tool_use", "tool_result")
        for message in messages
        for block in message.content
    )


# --------------------------------------------------------------------------
# Response IR: what a provider's parse_response yields and an inbound
# serialize_response consumes. Content reuses ContentBlock (text, tool_use,
# thinking, redacted_thinking are the cases a v2-sent request can produce;
# anything else fails parse loudly because the fail-closed request surface
# cannot trigger it).
# --------------------------------------------------------------------------

FinishReason = Literal["stop", "length", "tool_calls", "content_filter"]


@dataclass(frozen=True)
class CacheCreationDetails:
    five_minute: Option[int]
    one_hour: Option[int]


@dataclass(frozen=True)
class ModalityTokens:
    """Per-modality token breakdown (gemini ``usageMetadata`` details).

    Values are post-processing in v1's ``_calculate_usage`` terms: prompt text
    tokens already have cached tokens subtracted, completion text tokens are
    computed from the candidate total minus the other modalities.
    """

    text: Option[int]
    audio: Option[int]
    image: Option[int]
    video: Option[int]


@dataclass(frozen=True)
class ResponseUsage:
    """Provider-reported token counts, provider-neutral.

    ``total_tokens`` is the wire-reported total when the provider sends one
    (bedrock converse); ``Nothing`` means the inbound serializer computes
    prompt + completion (anthropic, whose wire total never includes cache).
    """

    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int
    cache_read_input_tokens: int
    cache_creation: Option[CacheCreationDetails]
    total_tokens: Option[int]
    reasoning_tokens: Option[int] = Nothing
    """Wire-reported reasoning tokens (gemini ``thoughtsTokenCount``);
    ``Nothing`` for providers whose dialect estimates them (anthropic)."""
    prompt_modalities: Option[ModalityTokens] = Nothing
    completion_modalities: Option[ModalityTokens] = Nothing
    cache_read_reported: bool = True
    """False when the wire omitted ``cachedContentTokenCount`` entirely (the
    gemini dialect then emits null cached-token fields instead of 0)."""


@dataclass(frozen=True)
class ChatResponse:
    id: str
    model: str
    content: Block[ContentBlock]
    finish: FinishReason
    usage: ResponseUsage
    synthesized_json_content: bool
    """True when the provider rewrote a forced json_tool_call into plain
    content (v1 then emits a bare message: no provider fields, no thinking)."""
    wire: Option[JsonBlob] = Nothing
    """The normalized outbound chat-completion body, set by providers whose
    outbound dialect is wire-derived (openai_compat: v1's
    convert_to_model_response_object is a near-passthrough, so byte parity
    needs the wire fields the semantic IR does not model: system_fingerprint,
    refusal, verbatim usage details). The provider's parse_response builds it;
    the ``openai`` response dialect emits it unchanged."""


# --------------------------------------------------------------------------
# Stream IR: provider stream parsers map wire events onto these; the inbound
# stream serializer folds them into outbound chunks. One event per provider
# wire event that carries information; keep-alives map to nothing.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class StreamStart:
    id: str
    model: str
    usage: ResponseUsage


@dataclass(frozen=True)
class TextDelta:
    index: int
    text: str


@dataclass(frozen=True)
class ToolUseStart:
    index: int
    id: str
    name: str


@dataclass(frozen=True)
class ToolArgsDelta:
    index: int
    partial_json: str


@dataclass(frozen=True)
class ThinkingDelta:
    index: int
    thinking: str


@dataclass(frozen=True)
class SignatureDelta:
    index: int
    signature: str


@dataclass(frozen=True)
class StreamFinish:
    """Anthropic message_delta: the final stop reason plus output usage."""

    finish: FinishReason
    output_tokens: int


@dataclass(frozen=True)
class StreamToolCall:
    """A COMPLETE tool call inside one stream chunk (gemini never streams
    partial arguments). An empty ``id`` (or one starting with the
    thought-signature separator) means the wire had no native id and the seam
    mints v1's ambient ``call_<uuid>`` prefix."""

    id: str
    name: str
    arguments_json: str


@dataclass(frozen=True)
class CompositeChunk:
    """One gemini ``GenerateContentResponse`` stream event: text, thinking,
    complete tool calls, the finish reason, and usage can all ride together,
    so the wire event maps onto one composite IR event (and back onto exactly
    one outbound chunk, matching v1's ``ModelResponseIterator``)."""

    id: str
    text: Option[str]
    reasoning: Option[str]
    signatures: Block[str]
    tool_calls: Block[StreamToolCall]
    finish: Option[FinishReason]
    usage: Option[ResponseUsage]


@tagged_union(frozen=True)
class StreamEvent:
    tag: Literal[
        "start",
        "text_delta",
        "tool_use_start",
        "tool_args_delta",
        "thinking_delta",
        "signature_delta",
        "finish",
        "stop",
        "wire_chunk",
        "chunk",
    ] = tag()

    start: StreamStart = case()
    text_delta: TextDelta = case()
    tool_use_start: ToolUseStart = case()
    tool_args_delta: ToolArgsDelta = case()
    thinking_delta: ThinkingDelta = case()
    signature_delta: SignatureDelta = case()
    finish: StreamFinish = case()
    stop: Unit = case()
    wire_chunk: JsonBlob = case()
    """A same-family provider chunk carried verbatim (openai_compat): the
    outbound chunk IS the inbound family, so a semantic re-encode would lose
    wire bytes (refusal, system_fingerprint, logprobs). The openai chunk
    dialect folds these; cross-family parsers never emit them."""
    chunk: CompositeChunk = case()

    @staticmethod
    def of_start(value: StreamStart) -> StreamEvent:
        return _stream_start(value)

    @staticmethod
    def of_text_delta(value: TextDelta) -> StreamEvent:
        return _stream_text_delta(value)

    @staticmethod
    def of_tool_use_start(value: ToolUseStart) -> StreamEvent:
        return _stream_tool_use_start(value)

    @staticmethod
    def of_tool_args_delta(value: ToolArgsDelta) -> StreamEvent:
        return _stream_tool_args_delta(value)

    @staticmethod
    def of_thinking_delta(value: ThinkingDelta) -> StreamEvent:
        return _stream_thinking_delta(value)

    @staticmethod
    def of_signature_delta(value: SignatureDelta) -> StreamEvent:
        return _stream_signature_delta(value)

    @staticmethod
    def of_finish(value: StreamFinish) -> StreamEvent:
        return _stream_finish(value)

    @staticmethod
    def of_stop() -> StreamEvent:
        return _stream_stop(UNIT)

    @staticmethod
    def of_wire_chunk(value: JsonBlob) -> StreamEvent:
        return _stream_wire_chunk(value)

    @staticmethod
    def of_chunk(value: CompositeChunk) -> StreamEvent:
        return _stream_chunk(value)


_stream_start = _case_maker(StreamEvent, "start")
_stream_text_delta = _case_maker(StreamEvent, "text_delta")
_stream_tool_use_start = _case_maker(StreamEvent, "tool_use_start")
_stream_tool_args_delta = _case_maker(StreamEvent, "tool_args_delta")
_stream_thinking_delta = _case_maker(StreamEvent, "thinking_delta")
_stream_signature_delta = _case_maker(StreamEvent, "signature_delta")
_stream_finish = _case_maker(StreamEvent, "finish")
_stream_stop = _case_maker(StreamEvent, "stop")
_stream_wire_chunk = _case_maker(StreamEvent, "wire_chunk")
_stream_chunk = _case_maker(StreamEvent, "chunk")
