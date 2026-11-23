# +-----------------------------------------------+
# |                                               |
# |               PII Masking                     |
# |         with Microsoft Presidio               |
# |   https://github.com/BerriAI/litellm/issues/  |
# +-----------------------------------------------+
#
#  Tell us how we can improve! - Krrish & Ishaan


import asyncio
import json
import re
import sys
import threading
from bisect import bisect_left, bisect_right
from collections.abc import (
    AsyncGenerator,
    AsyncIterable,
    AsyncIterator,
    Awaitable,
    Callable,
    Iterator,
    Mapping,
    Sequence,
)
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from datetime import datetime
from itertools import accumulate, chain, count, groupby, pairwise
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, Optional, Protocol, TypeAlias, cast

import aiohttp
from pydantic import ConfigDict, JsonValue, TypeAdapter, with_config
from typing_extensions import NotRequired, ReadOnly, TypedDict

import litellm
from litellm import get_secret
from litellm._logging import verbose_proxy_logger
from litellm.constants import (
    DEFAULT_PRESIDIO_ANALYZE_CHUNK_SIZE_BYTES,
    PRESIDIO_ANALYZE_CHUNK_CONCURRENCY,
    PRESIDIO_ANALYZE_CHUNK_OVERLAP_CHARS,
)
from litellm.litellm_core_utils.core_helpers import (
    attribute_items,
    attribute_or_none,
    float_of,
    int_of,
    require_attributes,
)
from litellm.types.utils import GenericGuardrailAPIInputs

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

from litellm.caching.caching import DualCache
from litellm.exceptions import BlockedPiiEntityError, GuardrailRaisedException
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,
)
from litellm.llms.anthropic.common_utils import (
    anthropic_server_tool_call_params,
    with_server_tool_call_arguments,
    write_anthropic_server_tool_calls,
)
from litellm.llms.anthropic.messages.stream_assembly import (
    anthropic_sse_error_event_of,
    anthropic_sse_event_types,
    anthropic_sse_frames,
)
from litellm.llms.base_llm.guardrail_translation.utils import stream_item_field
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_utils.sse_keepalive import split_complete_sse_frames
from litellm.proxy.guardrails.anthropic_sse import (
    anthropic_sse_blocks,
    anthropic_sse_chunks_of_events,
    anthropic_sse_chunks_rewritten,
    has_unreadable_json_event,
    is_anthropic_sse_stream,
)
from litellm.types.guardrails import (
    GuardrailEventHooks,
    LitellmParams,
    PiiAction,
    PiiEntityType,
    PresidioPerRequestConfig,
)
from litellm.types.llms.openai import ChatCompletionToolCallChunk
from litellm.types.proxy.guardrails.guardrail_hooks.presidio import (
    PresidioAnalyzeRequest,
    PresidioAnalyzeResponseItem,
)
from litellm.types.utils import ChatCompletionMessageToolCall, GuardrailStatus, Message, StreamingChoices
from litellm.utils import (
    EmbeddingResponse,
    ImageResponse,
    ModelResponse,
    ModelResponseStream,
)


@with_config(ConfigDict(extra="allow", strict=True))
class _PresidioAnonymizeItem(TypedDict, total=False):
    entity_type: ReadOnly[str | None]


@with_config(ConfigDict(extra="allow", strict=True))
class _PresidioAnonymizeResponse(TypedDict):
    text: ReadOnly[str]
    items: ReadOnly[NotRequired[list[_PresidioAnonymizeItem]]]


_PRESIDIO_ANONYMIZE_ADAPTER: Final[TypeAdapter[_PresidioAnonymizeResponse | None]] = TypeAdapter(
    _PresidioAnonymizeResponse | None
)
_DECODED_JSON: Final = TypeAdapter(object)


class _JsonResponse(Protocol):
    def json(self) -> Awaitable[object]: ...


async def _json_body(response: _JsonResponse) -> object:
    return await response.json()


def _write_attribute(target: object, name: str, value: object) -> None:
    setattr(target, name, value)


_LoopSemaphores = dict[asyncio.AbstractEventLoop, asyncio.Semaphore]


def _json_escaped_len(text: str) -> int:
    """
    Byte length of ``text`` as it appears serialized inside the JSON request
    body sent to Presidio (``json.dumps`` escapes non-ASCII characters, so a
    3-byte UTF-8 character can occupy 6+ bytes on the wire).
    """
    return len(json.dumps(text).encode("utf-8")) - 2  # strip the surrounding quotes


_MAX_FIRST_SSE_FRAME_BYTES: Final = 64 * 1024


@dataclass(frozen=True, slots=True)
class _SsePreface:
    """Complete leading SSE frames that carry no data, relayed verbatim before the stream shape is decided."""

    raw: bytes


_SSE_FRAME_END: Final = re.compile(rb"\r\n\r\n|\n\n|\r\r")


def _carries_data(frame: bytes) -> bool:
    return any(line.startswith(b"data:") and line[len(b"data:") :].strip() for line in frame.splitlines())


def _split_sse_preface(complete_frames: bytes) -> tuple[bytes, bytes]:
    """Split complete frames into ``(frames before the first data-bearing frame, that frame and everything after)``."""
    start = 0
    for end in _SSE_FRAME_END.finditer(complete_frames):
        frame = complete_frames[start : end.end()]
        if _carries_data(frame):
            return complete_frames[:start], complete_frames[start:]
        start = end.end()
    return complete_frames, b""


def _holds_json_object_frame(frames: bytes) -> bool:
    return any(
        line.startswith(b"data:") and line[len(b"data:") :].lstrip().startswith(b"{") for line in frames.splitlines()
    )


def _flush_unmaskable_buffer(all_chunks: list[ModelResponseStream]) -> Iterator[ModelResponseStream]:
    """Buffered chunks flushed unmasked when a mixed stream shape makes reconstruction impossible."""
    if not all_chunks:
        return
    verbose_proxy_logger.warning(
        "Presidio apply_to_output: mixed stream detected (ModelResponseStream + unknown event). "
        "Flushing %d buffered chunks without PII masking and switching to transparent passthrough.",
        len(all_chunks),
    )
    yield from all_chunks


_JSON_TOKEN: Final = re.compile(
    r'"(?P<string>[^"\\]*(?:\\.[^"\\]*)*)(?:"|\\?\Z)|(?P<scalar>[^\s\ufeff,:\[\]{}"]+)', re.DOTALL
)
_JSON_KEY_END: Final = re.compile(r"\s*:")
_JSON_NUMBER: Final = re.compile(r"-?[0-9]*(?:\.[0-9]*)?(?:[eE][+-]?[0-9]*)?")
_JSON_LITERALS: Final = ("true", "false", "null", "NaN", "Infinity", "-Infinity")
_JSON_ESCAPE: Final = re.compile(r'\\(?:u(?P<code>[0-9a-fA-F]{4})|(?P<char>["\\/bfnrt]))')
_JSON_ESCAPED_CHARS: Final = MappingProxyType({"b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t"})
_SEPARATORS: Final = range(0xE000, sys.maxunicode + 1)


@dataclass(frozen=True, slots=True)
class _TextPart:
    """A part of a scanned text that is masked on its own, and the text between it and the part before."""

    text: str
    before: str = ""
    is_json_key: bool = False


@dataclass(frozen=True, slots=True)
class _PiiProcessor:
    text: Callable[[str], Awaitable[str]]
    parts: Callable[[Sequence[_TextPart]], Awaitable[tuple[str, ...]]]

    async def texts(self, texts: Sequence[str]) -> tuple[str, ...]:
        if sum(map(bool, texts)) > 1:
            return await self.parts(tuple(map(_TextPart, texts)))
        processed: Final = await self.text("".join(texts))
        return tuple(text and processed for text in texts)

    async def json(self, text: str) -> str:
        """The JSON text with each string and scalar processed on its own, so it stays JSON whatever they become.

        Every string is processed as the value a JSON reader takes it for, with its escapes decoded. A string or
        scalar that does not change stays byte for byte, and a scalar that changes becomes a string. JSON that was
        cut short, as by the output token limit, is processed the same way, and any other text is processed whole.
        """
        tokens: Final = _json_tokens(text)
        if not tokens or not _is_json_or_cut_json(text, tokens):
            return await self.text(text)
        values: Final = await self.parts(tuple(_TextPart(token.value, token.before, token.is_key) for token in tokens))
        return _with_json_values(text, tokens, values)


@dataclass(frozen=True, slots=True)
class _JsonToken:
    before: str
    raw: str
    value: str
    is_string: bool
    is_key: bool

    def encoded(self, value: str) -> str:
        if value == self.value:
            return self.raw
        encoded: Final = json.dumps(value, ensure_ascii=self.raw.isascii())
        return encoded[1:-1] if self.is_string else encoded


def _is_json_scalar_or_cut_scalar(raw: str) -> bool:
    return _JSON_NUMBER.fullmatch(raw) is not None or any(literal.startswith(raw) for literal in _JSON_LITERALS)


def _is_json_or_cut_json(text: str, tokens: Sequence[_JsonToken]) -> bool:
    """Whether the tokens hold all the text's content, with only JSON punctuation around them."""
    if not all(token.is_string or _is_json_scalar_or_cut_scalar(token.raw) for token in tokens):
        return False
    content: Final = text.strip("\ufeff \t\r\n")
    return content.startswith(("{", "[")) or (len(tokens) == 1 and content in (tokens[0].raw, f'"{tokens[0].raw}"'))


def _json_escape_value(escape: re.Match[str]) -> str:
    code: Final = escape["code"]
    return chr(int(code, 16)) if code else _JSON_ESCAPED_CHARS.get(escape["char"], escape["char"])


def _json_string_value(raw: str) -> str:
    """The string a JSON reader decodes, with each unpaired surrogate replaced, since Presidio cannot read one."""
    decoded: Final = _JSON_ESCAPE.sub(_json_escape_value, raw)
    return decoded.encode("utf-16", "surrogatepass").decode("utf-16", "replace")


def _json_token(match: re.Match[str], previous_end: int) -> _JsonToken:
    is_string: Final = match["string"] is not None
    start, end = match.span("string" if is_string else "scalar")
    raw: Final = match.string[start:end]
    return _JsonToken(
        before=match.string[previous_end:start],
        raw=raw,
        value=_json_string_value(raw) if is_string else raw,
        is_string=is_string,
        is_key=_JSON_KEY_END.match(match.string, match.end()) is not None,
    )


def _json_tokens(text: str) -> tuple[_JsonToken, ...]:
    """Every string and scalar of the JSON text. The quotes of a string belong to the text between the tokens."""
    matches: Final = tuple(_JSON_TOKEN.finditer(text))
    ends: Final = (match.end(match.lastgroup or 0) for match in matches)
    return tuple(map(_json_token, matches, chain((0,), ends)))


def _free_key_names(masked: str, kept: frozenset[str], taken: frozenset[str]) -> Iterator[str]:
    if masked not in kept:
        yield masked
    yield from (name for name in (f"{masked}_{number}" for number in count(2)) if name not in taken)


def _masked_name(key: tuple[str, str]) -> str:
    return key[1]


def _processed_key_names(keys: Sequence[tuple[str, str]]) -> Mapping[str, str]:
    """A name for each changed key, by its value, so that entries masked to the same text stay separate entries."""
    changed: Final = MappingProxyType({value: name for value, name in keys if value != name})
    kept: Final = frozenset(value for value, name in keys if value == name)
    taken: Final = kept.union(changed.values())
    by_name: Final = groupby(sorted(changed.items(), key=_masked_name), key=_masked_name)
    named: Final = (zip(group, _free_key_names(name, kept, taken), strict=False) for name, group in by_name)
    return MappingProxyType({value: name for (value, _), name in chain.from_iterable(named)})


def _with_json_values(text: str, tokens: Sequence[_JsonToken], values: Sequence[str]) -> str:
    processed: Final = tuple(zip(tokens, values, strict=True))
    key_names: Final = _processed_key_names(tuple((token.value, value) for token, value in processed if token.is_key))
    encoded: Final = (
        token.before + token.encoded(key_names.get(token.value, value) if token.is_key else value)
        for token, value in processed
    )
    after: Final = text[sum(len(token.before) + len(token.raw) for token in tokens) :]
    return "".join((*encoded, after))


@dataclass(frozen=True, slots=True)
class _PartDetection:
    """What a detection covers of one part, placed in that part."""

    position: int
    start: int
    end: int
    is_whole: bool
    item: PresidioAnalyzeResponseItem


def _part_spans(parts: Sequence[_TextPart]) -> tuple[tuple[int, int], ...]:
    ends: Final = accumulate(len(part.before) + len(part.text) for part in parts)
    return tuple((end - len(part.text), end) for end, part in zip(ends, parts, strict=True))


def _detections_by_part(
    parts: Sequence[_TextPart], analyze_results: Sequence[PresidioAnalyzeResponseItem]
) -> Iterator[_PartDetection]:
    spans: Final = _part_spans(parts)
    starts: Final = tuple(start for start, _ in spans)
    ends: Final = tuple(end for _, end in spans)
    for item in analyze_results:
        item_start: int = item.get("start") or 0
        item_end: int = ends[-1] if (end := item.get("end")) is None else end
        for position in range(bisect_right(ends, item_start), bisect_left(starts, item_end)):
            part_start, part_end = starts[position], ends[position]
            if part_start < part_end:
                yield _PartDetection(
                    position=position,
                    start=max(item_start, part_start) - part_start,
                    end=min(item_end, part_end) - part_start,
                    is_whole=part_start <= item_start and item_end <= part_end,
                    item=item,
                )


def _cut_keys(parts: Sequence[_TextPart], analyze_results: Sequence[PresidioAnalyzeResponseItem]) -> tuple[str, ...]:
    """The JSON keys that a detection covers together with text outside the key."""
    cut_parts: Final = (parts[cut.position] for cut in _detections_by_part(parts, analyze_results) if not cut.is_whole)
    return tuple(dict.fromkeys(part.text for part in cut_parts if part.is_json_key))


def _detections_in_keys(
    parts: Sequence[_TextPart], found: Mapping[str, Sequence[PresidioAnalyzeResponseItem]]
) -> Iterator[PresidioAnalyzeResponseItem]:
    """The detections found in keys scanned on their own, moved to where the keys are among the parts."""
    for part, (key_start, _) in zip(parts, _part_spans(parts), strict=True):
        if part.is_json_key:
            for detection in _detections_by_part((_TextPart(part.text),), found.get(part.text, ())):
                yield {**detection.item, "start": key_start + detection.start, "end": key_start + detection.end}


def _detections_in_parts(
    parts: Sequence[_TextPart], analyze_results: Sequence[PresidioAnalyzeResponseItem]
) -> Iterator[PresidioAnalyzeResponseItem]:
    """The detections, each cut to the parts it covers and moved onto the parts joined by one-character separators.

    A JSON key is masked only for a detection inside it, since one that spans several values would rename the fields
    between them.
    """
    joined_starts: Final = tuple(accumulate((len(part.text) + 1 for part in parts), initial=0))
    for detection in _detections_by_part(parts, analyze_results):
        if detection.is_whole or not parts[detection.position].is_json_key:
            moved = joined_starts[detection.position]
            yield {**detection.item, "start": moved + detection.start, "end": moved + detection.end}


_Detections: TypeAlias = Sequence[PresidioAnalyzeResponseItem] | _PresidioAnonymizeResponse


async def _with_cut_keys_detected(
    parts: Sequence[_TextPart], analyze_results: _Detections, detect: Callable[[str], Awaitable[_Detections]]
) -> _Detections:
    """The detections, with those of each JSON key that one of them covers together with text outside the key.

    Such a detection does not say how much of it is the key's own, so the key is scanned again on its own.
    """
    if isinstance(analyze_results, dict):
        return analyze_results
    keys: Final = _cut_keys(parts, analyze_results)
    if not keys:
        return analyze_results
    found: Final = zip(keys, await asyncio.gather(*map(detect, keys)), strict=True)
    in_keys: Final = MappingProxyType({key: results for key, results in found if not isinstance(results, dict)})
    return [*analyze_results, *_detections_in_keys(parts, in_keys)]


async def _anonymized_parts(
    parts: Sequence[_TextPart],
    analyze_results: _Detections,
    anonymize: Callable[[str, _Detections], Awaitable[str]],
) -> tuple[str, ...]:
    if isinstance(analyze_results, dict) or (len(parts) == 1 and not parts[0].before):
        return (await anonymize(parts[0].text, analyze_results),)
    texts: Final = tuple(part.text for part in parts)
    characters: Final = frozenset(chain.from_iterable(texts))
    separator: Final = next(char for char in map(chr, _SEPARATORS) if char not in characters)
    anonymized: Final = await anonymize(separator.join(texts), tuple(_detections_in_parts(parts, analyze_results)))
    return tuple(anonymized.split(separator))


async def _processed_tool_call(
    tool_call: ChatCompletionToolCallChunk | ChatCompletionMessageToolCall,
    process_arguments: Callable[[str], Awaitable[str]],
) -> ChatCompletionToolCallChunk | ChatCompletionMessageToolCall:
    arguments: Final = stream_item_field(stream_item_field(tool_call, "function"), "arguments")
    if not isinstance(arguments, str):
        return tool_call
    processed: Final = await process_arguments(arguments)
    if isinstance(tool_call, dict):
        return {**tool_call, "function": {**tool_call["function"], "arguments": processed}}
    return tool_call.model_copy(update={"function": tool_call.function.model_copy(update={"arguments": processed})})


async def _coalesce_first_sse_frame(stream: AsyncIterator[object]) -> AsyncGenerator[object, None]:
    """
    Relay leading data-less SSE frames (comment keepalives, events with no
    data) as they complete, and join raw ``bytes`` chunks until they
    hold one complete SSE event with a JSON object on its data line, so the
    stream shape is decided on a whole frame rather than a transport fragment
    or a line that says nothing about the shape. Everything after that frame
    is forwarded untouched. The byte cap bounds a single unterminated frame
    and a run of frames with no JSON object.
    """
    pending = b""
    try:
        async for chunk in stream:
            if not isinstance(chunk, bytes):
                yield chunk
                continue
            pending += chunk
            complete_frames, tail = split_complete_sse_frames(pending)
            preface, classifiable = _split_sse_preface(complete_frames)
            if preface:
                yield _SsePreface(preface)
            pending = classifiable + tail
            if _holds_json_object_frame(classifiable) or len(pending) >= _MAX_FIRST_SSE_FRAME_BYTES:
                break
        else:
            if pending:
                yield pending
            return
    except Exception:
        if pending:
            yield pending
        raise
    yield pending
    async for chunk in stream:
        yield chunk


@dataclass(frozen=True, slots=True)
class _Detection:
    entity_type: str
    start: int
    end: int
    score: float


def _strictly_contains(outer: _Detection, inner: _Detection) -> bool:
    return (
        outer.start <= inner.start and inner.end <= outer.end and (outer.start, outer.end) != (inner.start, inner.end)
    )


def _collapse_run(run: Sequence[_Detection]) -> _Detection:
    label: Final = max(
        (detection for detection in run if not any(_strictly_contains(other, detection) for other in run)),
        key=lambda detection: (detection.score, detection.end - detection.start, detection.entity_type),
    )
    return _Detection(label.entity_type, run[0].start, max(detection.end for detection in run), label.score)


class _DetectionFields(Protocol):
    def get(self, key: str, /) -> object: ...


def _to_detection(item: _DetectionFields) -> _Detection:
    return _Detection(
        str(item.get("entity_type")),
        int_of(item.get("start") or 0),
        int_of(item.get("end") or 0),
        float_of(item.get("score") or 0.0),
    )


def _analyze_items(
    analyze_results: Sequence[PresidioAnalyzeResponseItem] | _PresidioAnonymizeResponse,
) -> Sequence[_DetectionFields]:
    if isinstance(analyze_results, dict):
        return tuple(require_attributes(key, _DetectionFields, "get") for key in analyze_results)
    return analyze_results


def _resolve_overlapping_spans(analyze_results: Sequence[_DetectionFields]) -> tuple[_Detection, ...]:
    """
    Collapse each run of overlapping detections into one detection covering the whole run.

    The run takes its entity type from its highest-scoring detection that no other
    detection in the run strictly contains, the same containment rule Presidio's
    anonymizer applies, so a phone number with a date-shaped fragment inside it stays
    a PHONE_NUMBER. Runs come back sorted by start and never overlap.
    """
    detections: Final = sorted(
        (detection for detection in map(_to_detection, analyze_results) if detection.start < detection.end),
        key=lambda detection: detection.start,
    )
    run_ends: Final = tuple(accumulate((detection.end for detection in detections), max))
    boundaries: Final = (
        0,
        *(index for index in range(1, len(detections)) if detections[index].start >= run_ends[index - 1]),
        len(detections),
    )
    return tuple(_collapse_run(detections[first:last]) for first, last in pairwise(boundaries) if first < last)


class _OPTIONAL_PresidioPIIMasking(CustomGuardrail):
    user_api_key_cache = None
    ad_hoc_recognizers: list[str] | None = None

    @classmethod
    def get_supported_event_hooks(cls) -> list[GuardrailEventHooks]:
        return [
            GuardrailEventHooks.pre_call,
            GuardrailEventHooks.during_call,
            GuardrailEventHooks.post_call,
            GuardrailEventHooks.logging_only,
            GuardrailEventHooks.pre_mcp_call,
            GuardrailEventHooks.post_mcp_call,
        ]

    # Class variables or attributes
    def __init__(
        self,
        mock_testing: bool = False,
        mock_redacted_text: _PresidioAnonymizeResponse | None = None,
        presidio_analyzer_api_base: str | None = None,
        presidio_anonymizer_api_base: str | None = None,
        output_parse_pii: bool | None = False,
        apply_to_output: bool = False,
        presidio_ad_hoc_recognizers: str | None = None,
        logging_only: bool | None = None,
        pii_entities_config: dict[PiiEntityType | str, PiiAction] | None = None,
        presidio_language: str | None = None,
        presidio_score_thresholds: dict[PiiEntityType | str, float] | None = None,
        presidio_entities_deny_list: list[PiiEntityType | str] | None = None,
        presidio_analyze_chunk_size_bytes: int | None = None,
        mask_tool_calls: bool = False,
        _callback_role: Literal["scan", "restore"] | None = None,
        **kwargs,
    ):
        if logging_only is True:
            self.logging_only = True
            kwargs["event_hook"] = GuardrailEventHooks.logging_only
        kwargs.setdefault("supported_event_hooks", list(self.get_supported_event_hooks()))
        super().__init__(**kwargs)
        self.guardrail_provider = "presidio"
        self.pii_tokens: dict[
            str, str
        ] = {}  # mapping of PII token to original text - only used with Presidio `replace` operation
        self.mock_redacted_text = mock_redacted_text
        self.output_parse_pii = output_parse_pii or False
        self.apply_to_output = apply_to_output
        self.mask_tool_calls: Final = mask_tool_calls
        self._callback_role = _callback_role

        # When output_parse_pii or apply_to_output is enabled, the guardrail must
        # also run on post_call to unmask/mask the response.  Expand the event_hook
        # so should_run_guardrail returns True for both pre_call and post_call.
        if _callback_role is None and (self.output_parse_pii or self.apply_to_output) and not logging_only:
            current_hook: Final = self.event_hook
            if isinstance(current_hook, str) and current_hook != "post_call":
                self.event_hook = cast(list[GuardrailEventHooks], [current_hook, "post_call"])
            elif isinstance(current_hook, list) and "post_call" not in current_hook:
                self.event_hook = cast(list[GuardrailEventHooks], current_hook + ["post_call"])
        self.pii_entities_config: dict[PiiEntityType | str, PiiAction] = pii_entities_config or {}
        self.presidio_score_thresholds: dict[PiiEntityType | str, float] = presidio_score_thresholds or {}
        self.presidio_entities_deny_list: list[PiiEntityType | str] = presidio_entities_deny_list or []
        self.presidio_language = presidio_language or "en"
        self.presidio_analyze_chunk_size_bytes: int = self._coerce_analyze_chunk_size(presidio_analyze_chunk_size_bytes)
        # Shared HTTP session to prevent memory leaks (issue #14540)
        self._http_session: aiohttp.ClientSession | None = None
        # Lock to prevent race conditions when creating session under concurrent load
        # Note: asyncio.Lock() can be created without an event loop; it only needs one when awaited
        self._session_lock: asyncio.Lock = asyncio.Lock()

        # Track main thread ID to safely identity when we are running in main loop vs background thread

        self._main_thread_id = threading.get_ident()

        # Loop-bound session cache for background threads
        self._loop_sessions: dict[asyncio.AbstractEventLoop, aiohttp.ClientSession] = {}

        # Per-loop semaphores bounding chunked-analyze fan-out across ALL
        # concurrent oversized blocks/requests on this instance, not per call
        self._loop_chunk_semaphores: _LoopSemaphores = {}

        if mock_testing is True:  # for testing purposes only
            return

        ad_hoc_recognizers: Final = presidio_ad_hoc_recognizers
        if ad_hoc_recognizers is not None:
            try:
                with open(ad_hoc_recognizers, "r") as file:
                    self.ad_hoc_recognizers = json.load(file)
            except FileNotFoundError:
                raise Exception(f"File not found. file_path={ad_hoc_recognizers}")
            except json.JSONDecodeError as e:
                raise Exception(f"Error decoding JSON file: {e}, file_path={ad_hoc_recognizers}")
            except Exception as e:
                raise Exception(f"An error occurred: {e}, file_path={ad_hoc_recognizers}")
        self.validate_environment(
            presidio_analyzer_api_base=presidio_analyzer_api_base,
            presidio_anonymizer_api_base=presidio_anonymizer_api_base,
        )

    def validate_environment(
        self,
        presidio_analyzer_api_base: str | None = None,
        presidio_anonymizer_api_base: str | None = None,
    ) -> None:
        self.presidio_analyzer_api_base: str | None = presidio_analyzer_api_base or get_secret(
            "PRESIDIO_ANALYZER_API_BASE", None
        )
        self.presidio_anonymizer_api_base: str | None = presidio_anonymizer_api_base or litellm.get_secret(
            "PRESIDIO_ANONYMIZER_API_BASE", None
        )

        if self.presidio_analyzer_api_base is None:
            raise Exception("Missing `PRESIDIO_ANALYZER_API_BASE` from environment")
        if not self.presidio_analyzer_api_base.endswith("/"):
            self.presidio_analyzer_api_base += "/"
        if not (
            self.presidio_analyzer_api_base.startswith("http://")
            or self.presidio_analyzer_api_base.startswith("https://")
        ):
            # add http:// if unset, assume communicating over private network - e.g. render
            self.presidio_analyzer_api_base = "http://" + self.presidio_analyzer_api_base

        if self.presidio_anonymizer_api_base is None:
            raise Exception("Missing `PRESIDIO_ANONYMIZER_API_BASE` from environment")
        if not self.presidio_anonymizer_api_base.endswith("/"):
            self.presidio_anonymizer_api_base += "/"
        if not (
            self.presidio_anonymizer_api_base.startswith("http://")
            or self.presidio_anonymizer_api_base.startswith("https://")
        ):
            # add http:// if unset, assume communicating over private network - e.g. render
            self.presidio_anonymizer_api_base = "http://" + self.presidio_anonymizer_api_base

    @asynccontextmanager
    async def _get_session_iterator(
        self,
    ) -> AsyncGenerator[aiohttp.ClientSession, None]:
        """
        Async context manager for yielding an HTTP session.

        Logic:
        1. If running in the main thread (where the object was initialized/destined to live normally),
           use the shared `self._http_session` (protected by a lock).
        2. If running in a background thread (e.g. logging hook), use a cached session for that loop.
        """
        current_loop: Final = asyncio.get_running_loop()

        # Check if we are in the stored main thread
        if threading.get_ident() == self._main_thread_id:
            # Main thread -> use shared session
            async with self._session_lock:
                if self._http_session is None or self._http_session.closed:
                    self._http_session = aiohttp.ClientSession()
                yield self._http_session
        else:
            # Background thread/loop -> use loop-bound session cache
            # This avoids "attached to a different loop" or "no running event loop" errors
            # when accessing the shared session created in the main loop
            if current_loop not in self._loop_sessions or self._loop_sessions[current_loop].closed:
                self._loop_sessions[current_loop] = aiohttp.ClientSession()
            yield self._loop_sessions[current_loop]

    async def _close_http_session(self) -> None:
        """Close all cached HTTP sessions."""
        if self._http_session is not None and not self._http_session.closed:
            await self._http_session.close()
            self._http_session = None

        for session in self._loop_sessions.values():
            if not session.closed:
                await session.close()
        self._loop_sessions.clear()

    def __del__(self):
        """Cleanup: we try to close, but doing async cleanup in __del__ is risky."""

    def _has_block_action(self) -> bool:
        """Return True if pii_entities_config has any BLOCK action (fail-closed on analyzer errors)."""
        if not self.pii_entities_config:
            return False
        return any(action == PiiAction.BLOCK for action in self.pii_entities_config.values())

    def _get_presidio_analyze_request_payload(
        self,
        text: str,
        presidio_config: PresidioPerRequestConfig | None,
        request_data: dict,
    ) -> PresidioAnalyzeRequest:
        """
        Construct the payload for the Presidio analyze request

        API Ref: https://microsoft.github.io/presidio/api-docs/api-docs.html#tag/Analyzer/paths/~1analyze/post
        """
        analyze_payload: Final[PresidioAnalyzeRequest] = PresidioAnalyzeRequest(
            text=text,
            language=self.presidio_language,
        )
        ##################################################################
        ###### Check if user has configured any params for this guardrail
        ################################################################
        if self.ad_hoc_recognizers is not None:
            analyze_payload["ad_hoc_recognizers"] = self.ad_hoc_recognizers

        if self.pii_entities_config:
            analyze_payload["entities"] = list(self.pii_entities_config.keys())

        ##################################################################
        ######### End of adding config params
        ##################################################################

        # Check if client side request passed any dynamic params
        if presidio_config and presidio_config.language:
            analyze_payload["language"] = presidio_config.language

        casted_analyze_payload: Final[dict] = cast(dict, analyze_payload)
        casted_analyze_payload.update(self.get_guardrail_dynamic_request_body_params(request_data=request_data))
        return cast(PresidioAnalyzeRequest, casted_analyze_payload)

    async def analyze_text(
        self,
        text: str,
        presidio_config: PresidioPerRequestConfig | None,
        request_data: dict,
    ) -> list[PresidioAnalyzeResponseItem] | _PresidioAnonymizeResponse:
        """
        Send text to the Presidio analyzer endpoint and get analysis results

        Texts larger than ``presidio_analyze_chunk_size_bytes`` (UTF-8) are split
        into overlapping chunks, analyzed per chunk, and the per-chunk results
        are remapped onto the original text. Presidio analyzer deployments
        commonly cap the /analyze request body size (e.g. at 1 MB), and analyzer
        latency grows with payload size.
        """
        # Chunk oversized texts before the try block so that a failing chunk
        # keeps the same sanitized error message a single call would produce.
        # A single-character text can never be split further, so it always
        # takes the single-call path regardless of its encoded width.
        if (
            text
            and len(text) > 1
            and self.mock_redacted_text is None
            and _json_escaped_len(text) > self.presidio_analyze_chunk_size_bytes
        ):
            return await self._analyze_text_chunked(
                text=text,
                presidio_config=presidio_config,
                request_data=request_data,
            )
        try:
            # Skip empty or whitespace-only text to avoid Presidio errors
            # Common in tool/function calling where assistant content is empty
            if not text or len(text.strip()) == 0:
                verbose_proxy_logger.debug("Skipping Presidio analysis for empty/whitespace-only text")
                return []

            if self.mock_redacted_text is not None:
                return self.mock_redacted_text

            # Use shared session to prevent memory leak (issue #14540)
            async with self._get_session_iterator() as session:
                # Make the request to /analyze
                analyze_url: Final = f"{self.presidio_analyzer_api_base}analyze"

                analyze_payload: Final[PresidioAnalyzeRequest] = self._get_presidio_analyze_request_payload(
                    text=text,
                    presidio_config=presidio_config,
                    request_data=request_data,
                )

                verbose_proxy_logger.debug(
                    "Making request to: %s with payload: %s",
                    analyze_url,
                    analyze_payload,
                )

                def _fail_on_invalid_response(
                    reason: str,
                ) -> list[PresidioAnalyzeResponseItem]:
                    should_fail_closed = bool(self.pii_entities_config) or self.output_parse_pii or self.apply_to_output
                    if should_fail_closed:
                        raise GuardrailRaisedException(
                            guardrail_name=self.guardrail_name,
                            message=f"Presidio analyzer returned invalid response; cannot verify PII when PII protection is configured: {reason}",
                            should_wrap_with_default_message=False,
                        )
                    verbose_proxy_logger.warning("Presidio analyzer %s, returning empty list", reason)
                    return []

                async with session.post(
                    analyze_url,
                    json=analyze_payload,
                    headers={"Accept": "application/json"},
                    timeout=(
                        aiohttp.ClientTimeout(total=self.timeout)
                        if isinstance(self.timeout, (int, float))
                        else aiohttp.client.DEFAULT_TIMEOUT
                    ),
                ) as response:
                    # Validate HTTP status
                    if response.status >= 400:
                        error_body = await response.text()
                        return _fail_on_invalid_response(
                            f"HTTP {response.status} from Presidio analyzer: {error_body[:200]}"
                        )

                    # Validate Content-Type is JSON
                    content_type: Final = getattr(
                        response,
                        "content_type",
                        response.headers.get("Content-Type", ""),
                    )
                    if "application/json" not in content_type:
                        error_body = await response.text()
                        return _fail_on_invalid_response(
                            f"expected application/json Content-Type but received '{content_type}'; body: '{error_body[:200]}'"
                        )

                    analyze_results: Final = await _json_body(response)
                    verbose_proxy_logger.debug("analyze_results: %s", analyze_results)

                # Handle error responses from Presidio (e.g., {'error': 'No text provided'})
                # Presidio may return a dict instead of a list when errors occur

                if isinstance(analyze_results, dict):
                    if "error" in analyze_results:
                        return _fail_on_invalid_response(f"error: {analyze_results.get('error')}")
                    # If it's a dict but not an error, try to process it as a single item
                    verbose_proxy_logger.debug(
                        "Presidio returned dict (not list), attempting to process as single item"
                    )
                    try:
                        return [PresidioAnalyzeResponseItem(**analyze_results)]
                    except Exception as e:
                        return _fail_on_invalid_response(f"failed to parse dict response: {e}")

                # Handle unexpected types (str, None, etc.) - e.g. from malformed/error
                if not isinstance(analyze_results, list):
                    return _fail_on_invalid_response(
                        f"unexpected type {type(analyze_results).__name__} (expected list or dict), response: {str(analyze_results)[:200]}"
                    )

                # Normal case: list of results
                final_results: Final = []
                for item in analyze_results:
                    if not isinstance(item, dict):
                        verbose_proxy_logger.warning(
                            "Skipping invalid Presidio result item (expected dict, got %s): %s",
                            type(item).__name__,
                            str(item)[:100],
                        )
                        continue
                    try:
                        final_results.append(PresidioAnalyzeResponseItem(**item))
                    except Exception as e:
                        verbose_proxy_logger.warning(
                            "Failed to parse Presidio result item: %s (error: %s)",
                            item,
                            e,
                        )
                        continue
                return final_results
        except GuardrailRaisedException:
            # Re-raise GuardrailRaisedException without wrapping
            raise
        except Exception as e:
            # Sanitize exception to avoid leaking the original text (which may
            # contain API keys or other secrets) in error responses.
            raise Exception(f"Presidio PII analysis failed: {type(e).__name__}") from e

    async def _analyze_text_chunked(
        self,
        text: str,
        presidio_config: PresidioPerRequestConfig | None,
        request_data: dict,  # mutable-ok: shared per-request state dict, matching analyze_text's parameter
    ) -> list[PresidioAnalyzeResponseItem]:  # mutable-ok: analyze_text's declared return type requires list
        """
        Analyze an oversized text by splitting it into overlapping chunks.

        Each chunk serializes to at most ``presidio_analyze_chunk_size_bytes``
        bytes inside the JSON request body, so every /analyze call stays below
        the analyzer deployment's request body limit; per-chunk results are remapped onto the original text and
        merged. Raises exactly like a single ``analyze_text`` call if any chunk
        fails.

        Only the analyzer side is chunked: the later anonymize call still
        receives the full original text, so texts above the anonymizer's own
        body limit that contain detections keep failing there.
        """
        text_chunks: Final = self._split_text_for_analysis(
            text=text,
            chunk_size_bytes=self.presidio_analyze_chunk_size_bytes,
            overlap_chars=PRESIDIO_ANALYZE_CHUNK_OVERLAP_CHARS,
        )
        verbose_proxy_logger.debug(
            "Presidio analyze: text exceeds %s bytes, analyzing in %s overlapping chunks",
            self.presidio_analyze_chunk_size_bytes,
            len(text_chunks),
        )
        # Bound the fan-out so oversized requests cannot saturate the analyzer.
        # The semaphore is shared per event loop across every chunked call on
        # this instance, so many oversized blocks in one request (or many
        # concurrent requests) still hold at most this many analyzer calls in
        # flight. On the proxy's main thread the shared-session lock in
        # _get_session_iterator additionally serializes the HTTP calls; the
        # bound matters for loop-bound sessions (background threads).
        analyze_semaphore: Final = self._get_chunk_semaphore()

        async def _analyze_chunk_bounded(
            chunk_text: str,
        ) -> Sequence[PresidioAnalyzeResponseItem] | _PresidioAnonymizeResponse:
            async with analyze_semaphore:
                return await self.analyze_text(
                    text=chunk_text,
                    presidio_config=presidio_config,
                    request_data=request_data,
                )

        gathered: Final = await asyncio.gather(
            *(_analyze_chunk_bounded(chunk_text) for _, chunk_text in text_chunks),
            return_exceptions=True,
        )
        chunk_results: Final = []
        for result in gathered:
            if isinstance(result, BaseException):
                raise result
            # analyze_text only returns a non-list shape when mock_redacted_text
            # is set, and the chunked path is never entered in that case.
            typed_result = cast("list[PresidioAnalyzeResponseItem]", result)  # cast-ok: gather() erases element type
            # Apply the configured score thresholds and deny list BEFORE the
            # overlap merge: a below-threshold detection must not win overlap
            # resolution against one the thresholds would keep. The same filter
            # runs again downstream in check_pii, where it is a no-op for the
            # already-filtered items.
            filtered_result = self.filter_analyze_results_by_score(analyze_results=typed_result)
            chunk_results.append(
                cast("list[PresidioAnalyzeResponseItem]", filtered_result)  # cast-ok: list input yields list
            )
        return self._merge_chunked_analyze_results(text_chunks=text_chunks, chunk_results=chunk_results)

    def _get_chunk_semaphore(self) -> asyncio.Semaphore:
        """Per-event-loop semaphore shared by all chunked analyze calls on this instance."""
        loop: Final = asyncio.get_running_loop()
        existing: Final = self._loop_chunk_semaphores.get(loop)
        if existing is not None:
            return existing
        created: Final = asyncio.Semaphore(PRESIDIO_ANALYZE_CHUNK_CONCURRENCY)
        self._loop_chunk_semaphores[loop] = created
        return created

    @staticmethod
    def _coerce_analyze_chunk_size(value: int | None) -> int:
        """
        Validate a configured chunk size, falling back to the default.

        Non-positive values would either bypass chunking entirely or degenerate
        it into per-character splits (silently disabling detection), so they are
        replaced by the default; values below 4 bytes are floored to 4 and the
        splitter always emits at least one character per chunk, so the chunked
        path can never re-enter itself.
        """
        if not value or value <= 0:
            return DEFAULT_PRESIDIO_ANALYZE_CHUNK_SIZE_BYTES
        return max(value, 4)

    @staticmethod
    def _split_text_for_analysis(
        text: str,
        chunk_size_bytes: int,
        overlap_chars: int,
    ) -> Sequence[tuple[int, str]]:
        """
        Split ``text`` into chunks whose JSON-serialized form is at most
        ``chunk_size_bytes`` bytes (the analyzer body limit applies to the
        JSON request body, where non-ASCII characters are escaped and larger
        than their raw UTF-8 encoding).

        Consecutive chunks overlap by up to ``overlap_chars`` characters so a
        PII entity up to that length lying across a chunk boundary is still
        seen whole by one of the chunks (longer boundary-straddling entities
        may be seen only truncated); ``_merge_chunked_analyze_results`` resolves
        the duplicate and truncated detections this produces. Returns
        ``(char_offset, chunk_text)`` pairs where ``char_offset`` is the
        chunk's start position in the original text.
        """
        chunks: Final = []
        text_len: Final = len(text)
        start = 0  # rebind-ok: chunk cursor advances across the loop
        while start < text_len:
            # Serialized length of a character is at least 1 byte, so a slice
            # of chunk_size_bytes characters is a sufficient search window.
            candidate = text[start : start + chunk_size_bytes]
            if _json_escaped_len(candidate) <= chunk_size_bytes:
                chunk = candidate
            else:
                # Largest prefix whose serialized form fits the budget.
                low, high = 1, len(candidate)
                while low < high:
                    mid = (low + high + 1) // 2
                    if _json_escaped_len(candidate[:mid]) <= chunk_size_bytes:
                        low = mid
                    else:
                        high = mid - 1
                # low >= 1 keeps the loop advancing even when a single
                # character serializes over a (floored, tiny) budget.
                chunk = candidate[:low]
            end = start + len(chunk)
            chunks.append((start, chunk))
            if end >= text_len:
                break
            # Cap the overlap so the next chunk always makes forward progress.
            effective_overlap = min(overlap_chars, len(chunk) // 2)
            start = max(start + 1, end - effective_overlap)
        return chunks

    @staticmethod
    def _merge_chunked_analyze_results(
        text_chunks: Sequence[tuple[int, str]],
        chunk_results: Sequence[Sequence[PresidioAnalyzeResponseItem]],
    ) -> list[PresidioAnalyzeResponseItem]:  # mutable-ok: analyze_text's declared return type requires list
        """
        Remap per-chunk analyzer offsets onto the original text and merge.

        A detection in an overlap region is reported by both neighbouring
        chunks, and a boundary entity can additionally be reported truncated by
        the chunk that saw only its head or tail. Same-entity-type detections
        with overlapping remapped spans are therefore resolved by keeping the
        longest span (highest score on ties) — mirroring the same-type conflict
        removal Presidio's AnalyzerEngine applies within a single call, and
        keeping overlapping spans from corrupting the numbered-token rewriter.
        Detections of DIFFERENT entity types may still overlap, exactly as in a
        single-call response. The merged list is sorted by position.
        """
        remapped: Final[list[PresidioAnalyzeResponseItem]] = []
        for (char_offset, _), results in zip(text_chunks, chunk_results, strict=True):
            for item in results:
                item_start = item.get("start")
                item_end = item.get("end")
                if item_start is not None:
                    item["start"] = item_start + char_offset
                if item_end is not None:
                    item["end"] = item_end + char_offset
                remapped.append(item)

        def _priority(item: PresidioAnalyzeResponseItem) -> tuple[int, float]:
            span_start: Final = item.get("start") or 0
            span_end: Final = item.get("end") or 0
            return (-(span_end - span_start), -(item.get("score") or 0.0))

        merged: Final[list[PresidioAnalyzeResponseItem]] = []
        kept_spans_by_type: Final[dict[str, list[tuple[int, int]]]] = {}
        for item in sorted(remapped, key=_priority):
            item_start = item.get("start")
            item_end = item.get("end")
            if item_start is None or item_end is None:
                merged.append(item)
                continue
            kept_spans = kept_spans_by_type.setdefault(str(item.get("entity_type")), [])
            if any(item_start < kept_end and kept_start < item_end for kept_start, kept_end in kept_spans):
                continue
            kept_spans.append((item_start, item_end))
            merged.append(item)
        merged.sort(key=lambda r: (r.get("start") or 0, r.get("end") or 0))
        return merged

    async def _post_presidio_anonymize(
        self,
        text: str,
        analyze_results: list[PresidioAnalyzeResponseItem] | _PresidioAnonymizeResponse,
    ) -> _PresidioAnonymizeResponse | None:
        """POST to Presidio anonymize; returns parsed JSON body."""
        # Use shared session to prevent memory leak (issue #14540)
        async with self._get_session_iterator() as session:
            anonymize_url: Final = f"{self.presidio_anonymizer_api_base}anonymize"
            verbose_proxy_logger.debug("Making request to: %s", anonymize_url)
            anonymize_payload: Final = {
                "text": text,
                "analyzer_results": analyze_results,
            }
            async with session.post(
                anonymize_url,
                json=anonymize_payload,
                headers={"Accept": "application/json"},
                timeout=(
                    aiohttp.ClientTimeout(total=self.timeout)
                    if isinstance(self.timeout, (int, float))
                    else aiohttp.client.DEFAULT_TIMEOUT
                ),
            ) as response:
                if response.status >= 400:
                    error_body = await response.text()
                    raise Exception(f"Presidio anonymizer returned HTTP {response.status}: {error_body[:200]}")
                content_type: Final = getattr(
                    response,
                    "content_type",
                    response.headers.get("Content-Type", ""),
                )
                if "application/json" not in content_type:
                    error_body = await response.text()
                    raise Exception(
                        f"Presidio anonymizer returned non-JSON Content-Type '{content_type}'; body: '{error_body[:200]}'"
                    )
                return _PRESIDIO_ANONYMIZE_ADAPTER.validate_python(await response.json())

    def _finalize_presidio_anonymize_simple(
        self,
        redacted_text: _PresidioAnonymizeResponse,
        masked_entity_count: dict[str, int],
    ) -> str:
        # No need to build numbered tokens — just use Presidio's
        # already-anonymized text directly.  The old code incorrectly
        # applied anonymizer item positions (which reference the
        # *output* text) to the *original* text, causing offset errors.
        for item in redacted_text.get("items", []):
            entity_type = item.get("entity_type", None)
            if entity_type is not None:
                masked_entity_count[entity_type] = masked_entity_count.get(entity_type, 0) + 1
        return redacted_text["text"]

    def _finalize_presidio_anonymize_numbered_tokens(
        self,
        text: str,
        analyze_results: Sequence[PresidioAnalyzeResponseItem] | _PresidioAnonymizeResponse,
        request_data: dict | None,
        masked_entity_count: dict[str, int],
    ) -> str:
        # output_parse_pii is True — we need sequentially numbered
        # tokens and a pii_tokens mapping for later unmasking.
        # Use analyze_results positions (which reference the ORIGINAL
        # text) instead of anonymizer items (which reference the output).
        new_text = text
        if request_data is None:
            verbose_proxy_logger.warning(
                "Presidio anonymize_text called without request_data — "
                "PII tokens cannot be stored per-request. "
                "This may indicate a missing caller update."
            )
            request_data = {}
        if not request_data.get("metadata"):
            request_data["metadata"] = {}
        if "pii_tokens" not in request_data["metadata"]:
            request_data["metadata"]["pii_tokens"] = {}
        pii_tokens: Final = cast("dict[str, str]", request_data["metadata"]["pii_tokens"])  # cast-ok: this hook's map

        runs: Final = _resolve_overlapping_spans(_analyze_items(analyze_results))
        # Every message and content part numbers its runs from 1 into this one map, so a
        # token another part already holds for a different value takes a spare number.
        # No token in the map is numbered above its size, so spares start above both.
        spare_numbers: Final = count(max(len(pii_tokens), len(runs)) + 1)

        for seq, run in reversed(tuple(enumerate(runs, start=1))):
            original = text[run.start : run.end]
            numbered = f"<{run.entity_type}_{seq}>"
            replacement = (
                numbered
                if pii_tokens.get(numbered, original) == original
                else f"<{run.entity_type}_{next(spare_numbers)}>"
            )
            pii_tokens[replacement] = original
            new_text = new_text[: run.start] + replacement + new_text[run.end :]
            masked_entity_count[run.entity_type] = masked_entity_count.get(run.entity_type, 0) + 1
        return new_text

    async def anonymize_text(
        self,
        text: str,
        analyze_results: list[PresidioAnalyzeResponseItem] | _PresidioAnonymizeResponse,
        output_parse_pii: bool,
        masked_entity_count: dict[str, int],
        request_data: dict | None = None,
    ) -> str:
        """
        Send analysis results to the Presidio anonymizer endpoint to get redacted text
        """
        try:
            # If there are no detections after filtering, return the original text
            if isinstance(analyze_results, list) and len(analyze_results) == 0:
                return text

            redacted_text: Final = await self._post_presidio_anonymize(text, analyze_results)
            if redacted_text is None:
                raise Exception("Invalid anonymizer response: received None")

            verbose_proxy_logger.debug("redacted_text: %s", redacted_text)

            if not output_parse_pii:
                return self._finalize_presidio_anonymize_simple(redacted_text, masked_entity_count)

            return self._finalize_presidio_anonymize_numbered_tokens(
                text, analyze_results, request_data, masked_entity_count
            )
        except Exception as e:
            # Sanitize exception to avoid leaking the original text (which may
            # contain API keys or other secrets) in error responses.
            error_str: Final = str(e)
            if "Invalid anonymizer response" in error_str or "Presidio anonymizer returned" in error_str:
                raise
            raise Exception(f"Presidio PII anonymization failed: {type(e).__name__}") from e

    def filter_analyze_results_by_score(
        self, analyze_results: list[PresidioAnalyzeResponseItem] | _PresidioAnonymizeResponse
    ) -> list[PresidioAnalyzeResponseItem] | _PresidioAnonymizeResponse:
        """
        Drop detections that fall below configured per-entity score thresholds
        or match an entity type in the deny list.
        """
        if not self.presidio_score_thresholds and not self.presidio_entities_deny_list:
            return analyze_results

        if not isinstance(analyze_results, list):
            return analyze_results

        filtered_results: Final[list[PresidioAnalyzeResponseItem]] = []
        deny_list_strings: Final = [getattr(x, "value", str(x)) for x in self.presidio_entities_deny_list]
        for item in analyze_results:
            entity_type = item.get("entity_type")

            str_entity_type = str(
                getattr(entity_type, "value", entity_type) if entity_type is not None else entity_type
            )
            if entity_type and str_entity_type in deny_list_strings:
                continue

            if self.presidio_score_thresholds:
                score = item.get("score")
                threshold = None
                if entity_type is not None:
                    threshold = self.presidio_score_thresholds.get(entity_type)
                if threshold is None:
                    threshold = self.presidio_score_thresholds.get("ALL")

                if threshold is not None:
                    if score is None or score < threshold:
                        continue

            filtered_results.append(item)

        return filtered_results

    def raise_exception_if_blocked_entities_detected(
        self, analyze_results: list[PresidioAnalyzeResponseItem] | _PresidioAnonymizeResponse
    ) -> None:
        """
        Raise an exception if blocked entities are detected
        """
        if self.pii_entities_config is None:
            return

        if isinstance(analyze_results, dict):
            # if mock testing is enabled, analyze_results is a dict
            # we don't need to raise an exception in this case
            return

        for result in analyze_results:
            entity_type = result.get("entity_type")

            if entity_type:
                # Check if entity_type is in config (supports both enum and string)
                if entity_type in self.pii_entities_config and self.pii_entities_config[entity_type] == PiiAction.BLOCK:
                    raise BlockedPiiEntityError(
                        entity_type=entity_type,
                        guardrail_name=self.guardrail_name,
                    )

    async def check_pii(
        self,
        text: str,
        output_parse_pii: bool,
        presidio_config: PresidioPerRequestConfig | None,
        request_data: dict,
    ) -> str:
        """
        Calls Presidio Analyze + Anonymize endpoints for PII Analysis + Masking
        """
        (checked,) = await self._check_pii_parts((_TextPart(text),), output_parse_pii, presidio_config, request_data)
        return checked

    async def _check_pii_parts(
        self,
        parts: Sequence[_TextPart],
        output_parse_pii: bool,
        presidio_config: PresidioPerRequestConfig | None,
        request_data: dict[str, object],  # mutable-ok: the guardrail trace is recorded on the request dict
    ) -> tuple[str, ...]:
        async def anonymized(text: str, results: _Detections) -> str:
            return await self.anonymize_text(
                text=text,
                analyze_results=results if isinstance(results, dict) else list(results),
                output_parse_pii=output_parse_pii,
                masked_entity_count=masked_entity_count,
                request_data=request_data,
            )

        async def detections(text: str) -> _Detections:
            found: Final = await self.analyze_text(
                text=text, presidio_config=presidio_config, request_data=request_data
            )
            verbose_proxy_logger.debug("analyze_results: %s", found)
            kept: Final = self.filter_analyze_results_by_score(analyze_results=found)
            self.raise_exception_if_blocked_entities_detected(analyze_results=kept)
            return kept

        start_time: Final = datetime.now()
        analyze_results: _Detections | None = None
        status: GuardrailStatus = "success"
        masked_entity_count: Final[dict[str, int]] = {}
        exception_str: str = ""
        try:
            if self.mock_redacted_text is not None:
                redacted_text: Final = self.mock_redacted_text
            else:
                # First get analysis results
                analyze_results = await _with_cut_keys_detected(
                    parts, await detections("".join(part.before + part.text for part in parts)), detections
                )

                # Then anonymize the text using the analysis results
                return await _anonymized_parts(parts, analyze_results, anonymized)
            return tuple(redacted_text["text"] for _ in parts)
        except Exception as e:
            status = "guardrail_failed_to_respond"
            exception_str = str(e)
            raise e
        finally:
            ####################################################
            # Create Guardrail Trace for logging on Langfuse, Datadog, etc.
            ####################################################
            guardrail_json_response: Exception | str | dict | list[dict] = {}
            if status == "success":
                if isinstance(analyze_results, list):
                    guardrail_json_response = [dict(item) for item in analyze_results]
            else:
                guardrail_json_response = exception_str
            self.add_standard_logging_guardrail_information_to_request_data(
                guardrail_provider=self.guardrail_provider,
                guardrail_json_response=guardrail_json_response,
                request_data=request_data,
                guardrail_status=status,
                start_time=start_time.timestamp(),
                end_time=datetime.now().timestamp(),
                duration=(datetime.now() - start_time).total_seconds(),
                masked_entity_count=masked_entity_count,
            )

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: str,
    ) -> dict[str, object]:
        """
        - Check if request turned off pii
            - Check if user allowed to turn off pii (key permissions -> 'allow_pii_controls')

        - Take the request data
        - Call /analyze -> get the results
        - Call /anonymize w/ the analyze results -> get the redacted text

        For multiple messages in /chat/completions, we'll need to call them in parallel.
        """
        # Respect the configured event hook. In `logging_only` mode (and any config that
        # excludes pre_call) the live request must not be masked - masking is applied to a
        # copy at logging time via `async_logging_hook`. Without this gate the request sent
        # to the model would carry anonymization tokens and the response would echo them.
        if (
            self.should_run_guardrail(
                data=data,
                event_type=GuardrailEventHooks.pre_call,
            )
            is not True
        ):
            return data

        try:
            content_safety: Final = data.get("content_safety", None)
            verbose_proxy_logger.debug("content_safety: %s", content_safety)
            presidio_config: Final = self.get_presidio_settings_from_request_data(data)
            messages: Final = data.get("messages", None)
            if messages is None:
                return data
            tasks: Final = []
            task_mappings: list[tuple[int, int | None]] = []  # Track (message_index, content_index) for each task

            for msg_idx, m in enumerate(messages):
                content = m.get("content", None)
                if content is None:
                    continue
                if isinstance(content, str):
                    tasks.append(
                        self.check_pii(
                            text=content,
                            output_parse_pii=self.output_parse_pii,
                            presidio_config=presidio_config,
                            request_data=data,
                        )
                    )
                    task_mappings.append((msg_idx, None))  # None indicates string content
                elif isinstance(content, list):
                    for content_idx, c in enumerate(content):
                        text_str = c.get("text", None)
                        if text_str is None:
                            continue
                        tasks.append(
                            self.check_pii(
                                text=text_str,
                                output_parse_pii=self.output_parse_pii,
                                presidio_config=presidio_config,
                                request_data=data,
                            )
                        )
                        task_mappings.append((msg_idx, int(content_idx)))

            responses: Final = await asyncio.gather(*tasks)

            # Map responses back to the correct message and content item
            for task_idx, r in enumerate(responses):
                mapping = task_mappings[task_idx]
                msg_idx = cast(int, mapping[0])
                content_idx_optional = cast(int | None, mapping[1])
                content = messages[msg_idx].get("content", None)
                if content is None:
                    continue
                if isinstance(content, str) and content_idx_optional is None:
                    messages[msg_idx]["content"] = r  # replace content with redacted string
                elif isinstance(content, list) and content_idx_optional is not None:
                    messages[msg_idx]["content"][content_idx_optional]["text"] = r

            verbose_proxy_logger.debug("Presidio PII Masking: Redacted pii message: %s", data["messages"])
            data["messages"] = messages
            return data
        except Exception as e:
            raise e

    def logging_hook(self, kwargs: dict, result: object, call_type: str) -> tuple[dict, object]:
        from concurrent.futures import ThreadPoolExecutor

        def run_in_new_loop():
            """Run the coroutine in a new event loop within this thread."""
            new_loop: Final = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(new_loop)
                return new_loop.run_until_complete(
                    self.async_logging_hook(kwargs=kwargs, result=result, call_type=call_type)
                )
            finally:
                new_loop.close()
                asyncio.set_event_loop(None)

        try:
            # First, try to get the current event loop
            _ = asyncio.get_running_loop()
            # If we're already in an event loop, run in a separate thread
            # to avoid nested event loop issues
            with ThreadPoolExecutor(max_workers=1) as executor:
                future: Final = executor.submit(run_in_new_loop)
                return future.result()

        except RuntimeError:
            # No running event loop, we can safely run in this thread
            return run_in_new_loop()

    async def async_logging_hook(self, kwargs: dict, result: object, call_type: str) -> tuple[dict, object]:
        """
        Masks the input and output before logging to langfuse, datadog, etc.
        """
        import copy

        from litellm.types.utils import CallTypes

        supported_call_types: Final = {
            CallTypes.completion.value,
            CallTypes.acompletion.value,
            CallTypes.anthropic_messages.value,
            CallTypes.aanthropic_messages.value,
        }
        if call_type in supported_call_types or kwargs.get("messages") is not None:
            raw_messages: Final[list | None] = kwargs.get("messages", None)
            if raw_messages is None:
                return kwargs, result

            messages: Final[list] = copy.deepcopy(raw_messages)
            tasks: Final = []
            task_mappings: list[tuple[int, int | None]] = []  # Track (message_index, content_index) for each task

            presidio_config: Final = self.get_presidio_settings_from_request_data(kwargs)

            for msg_idx, m in enumerate(messages):
                if not isinstance(m, dict):
                    continue
                content = m.get("content", None)
                if content is None:
                    continue
                if isinstance(content, str):
                    tasks.append(
                        self.check_pii(
                            text=content,
                            output_parse_pii=False,
                            presidio_config=presidio_config,
                            request_data=kwargs,
                        )
                    )  # need to pass separately b/c presidio has context window limits
                    task_mappings.append((msg_idx, None))  # None indicates string content
                elif isinstance(content, list):
                    for content_idx, c in enumerate(content):
                        if not isinstance(c, dict):
                            continue
                        text_str = c.get("text", None)
                        if text_str is None:
                            continue
                        tasks.append(
                            self.check_pii(
                                text=text_str,
                                output_parse_pii=False,
                                presidio_config=presidio_config,
                                request_data=kwargs,
                            )
                        )
                        task_mappings.append((msg_idx, int(content_idx)))

            responses: Final = await asyncio.gather(*tasks)

            # Map responses back to the correct message and content item
            for task_idx, r in enumerate(responses):
                mapping = task_mappings[task_idx]
                msg_idx = cast(int, mapping[0])
                content_idx_optional = cast(int | None, mapping[1])
                content = messages[msg_idx].get("content", None)
                if content is None:
                    continue
                if isinstance(content, str) and content_idx_optional is None:
                    messages[msg_idx]["content"] = r  # replace content with redacted string
                elif isinstance(content, list) and content_idx_optional is not None:
                    messages[msg_idx]["content"][content_idx_optional]["text"] = r

            verbose_proxy_logger.debug("Presidio PII Masking: Redacted pii message: %s", messages)
            kwargs["messages"] = messages

            raw_system = kwargs.get("system")
            if raw_system is not None:
                masked_system = copy.deepcopy(raw_system)
                if isinstance(masked_system, str):
                    masked_system = await self.check_pii(
                        text=masked_system,
                        output_parse_pii=False,
                        presidio_config=presidio_config,
                        request_data=kwargs,
                    )
                elif isinstance(masked_system, list):
                    for block in masked_system:
                        if isinstance(block, dict) and isinstance(block.get("text"), str):
                            block["text"] = await self.check_pii(
                                text=block["text"],
                                output_parse_pii=False,
                                presidio_config=presidio_config,
                                request_data=kwargs,
                            )
                kwargs["system"] = masked_system

            standard_logging_object: Final = kwargs.get("standard_logging_object")
            if isinstance(standard_logging_object, dict) and "messages" in standard_logging_object:
                from litellm.litellm_core_utils.litellm_logging import StandardLoggingPayloadSetup
                from litellm.litellm_core_utils.logging_utils import truncate_base64_in_messages

                standard_logging_object["messages"] = truncate_base64_in_messages(
                    StandardLoggingPayloadSetup.append_system_prompt_messages(
                        kwargs=kwargs, messages=copy.deepcopy(messages)
                    )
                )

            if (
                isinstance(result, ModelResponse)
                and result.choices
                and not isinstance(result.choices[0], StreamingChoices)
            ):
                await self._process_response_for_pii(response=result, request_data=kwargs, mode="mask")
            elif isinstance(result, dict) and self._is_anthropic_message_response(result):
                await self._process_anthropic_response_for_pii(
                    response=result,
                    request_data=kwargs,
                    mode="mask",
                )

        return kwargs, result

    async def async_post_call_success_hook(
        self,
        data: dict[str, object],
        user_api_key_dict: UserAPIKeyAuth,
        response: ModelResponse | EmbeddingResponse | ImageResponse,
    ) -> dict[str, JsonValue] | ModelResponse | EmbeddingResponse | ImageResponse:
        """
        Output parse the response object to replace the masked tokens with user sent values
        """
        verbose_proxy_logger.debug(
            "PII Masking Args: self.output_parse_pii=%s; type of response=%s", self.output_parse_pii, type(response)
        )

        if self.apply_to_output is True:
            if self._is_anthropic_message_response(response):
                return await self._process_anthropic_response_for_pii(
                    response=cast(dict, response), request_data=data, mode="mask"
                )
            return await self._mask_output_response(response=response, request_data=data)

        if self.output_parse_pii is False and litellm.output_parse_pii is False:
            return response

        if isinstance(response, ModelResponse) and not isinstance(
            response.choices[0], StreamingChoices
        ):  # /chat/completions requests
            await self._process_response_for_pii(
                response=response,
                request_data=data,
                mode="unmask",
            )
        elif self._is_anthropic_message_response(response):
            await self._process_anthropic_response_for_pii(
                response=cast(dict, response), request_data=data, mode="unmask"
            )
        return response

    @staticmethod
    def _unmask_pii_text(text: str, pii_tokens: dict[str, str]) -> str:
        """
        Replace PII tokens in *text* with their original values.

        Includes a fallback for tokens that were truncated by ``max_tokens``:
        if the *end* of ``text`` matches the *beginning* of a token and the
        overlap is long enough, the truncated suffix is replaced with the
        original value.  The minimum overlap length is
        ``min(20, len(token) // 2)`` to reduce the risk of false positives
        when multiple tokens share a common prefix.
        """
        for token, original_text in pii_tokens.items():
            if token in text:
                text = text.replace(token, original_text)
            else:
                # FALLBACK: Handle truncated tokens (token cut off by max_tokens)
                # Only check at the very end of the text.
                min_overlap = min(20, len(token) // 2)
                for i in range(max(0, len(text) - len(token)), len(text)):
                    sub = text[i:]
                    if token.startswith(sub) and len(sub) >= min_overlap:
                        text = text[:i] + original_text
                        break
        return text

    @staticmethod
    def _is_anthropic_message_response(
        response: ModelResponse | EmbeddingResponse | ImageResponse | dict[str, object],
    ) -> bool:
        """Check if the response is an Anthropic native message dict."""
        return (
            isinstance(response, dict)
            and response.get("type") == "message"
            and isinstance(response.get("content"), list)
        )

    async def _process_anthropic_response_for_pii(
        self,
        response: dict,
        request_data: dict,
        mode: Literal["mask", "unmask"],
    ) -> dict:
        """
        Process an Anthropic native message dict for PII masking/unmasking.
        Handles content blocks with type == "text".
        """
        pii_processed_text: Final = self._pii_processor(request_data, mode).text
        content: Final = response.get("content")
        if not isinstance(content, list):
            return response

        for block in content:
            if not isinstance(block, dict) or block.get("type") != "text":
                continue
            text_value = block.get("text")
            if text_value is None:
                continue
            block["text"] = await pii_processed_text(text_value)

        return response

    def _pii_masker(
        self,
        output_parse_pii: bool,
        presidio_config: PresidioPerRequestConfig | None,
        request_data: dict[str, object],  # mutable-ok: check_pii records its guardrail trace on the request dict
    ) -> _PiiProcessor:
        async def masked_text(text: str) -> str:
            return await self.check_pii(
                text=text,
                output_parse_pii=output_parse_pii,
                presidio_config=presidio_config,
                request_data=request_data,
            )

        async def masked_parts(parts: Sequence[_TextPart]) -> tuple[str, ...]:
            return await self._check_pii_parts(parts, output_parse_pii, presidio_config, request_data)

        return _PiiProcessor(masked_text, masked_parts)

    def _pii_processor(
        self,
        request_data: dict,  # mutable-ok: check_pii records its guardrail trace on the request dict
        mode: Literal["mask", "unmask"],
    ) -> _PiiProcessor:
        if mode == "mask":
            return self._pii_masker(
                False, self.get_presidio_settings_from_request_data(request_data or {}), request_data
            )
        metadata: Final = (request_data.get("metadata") or {}) if request_data else {}
        pii_tokens: Final = metadata.get("pii_tokens", {})
        if not pii_tokens:
            verbose_proxy_logger.debug("No pii_tokens found in request_data['metadata'] — nothing to unmask")

        async def unmasked_text(text: str) -> str:
            return self._unmask_pii_text(text, pii_tokens)

        async def unmasked_parts(parts: Sequence[_TextPart]) -> tuple[str, ...]:
            return tuple(self._unmask_pii_text(part.text, pii_tokens) for part in parts)

        return _PiiProcessor(unmasked_text, unmasked_parts)

    async def _process_response_for_pii(
        self,
        response: ModelResponse,
        request_data: dict,
        mode: Literal["mask", "unmask"],
    ) -> ModelResponse:
        """
        Helper to recursively process a ModelResponse for PII.
        Handles all choices and tool calls.
        """
        processor: Final = self._pii_processor(request_data, mode)
        pii_processed_text: Final = processor.text
        pii_processed_arguments: Final = processor.json
        for choice in response.choices:
            message: Message | None = getattr(choice, "message", None)
            if message is None:
                continue

            # 1. Process content
            content = getattr(message, "content", None)
            if isinstance(content, str):
                message.content = await pii_processed_text(content)
            elif isinstance(content, list):
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    text_value = item.get("text")
                    if text_value is None:
                        continue
                    item["text"] = await pii_processed_text(text_value)

            await self._process_server_tool_calls_for_pii(message, pii_processed_arguments)

            # 2. Process tool calls
            for tool_call in attribute_items(message, "tool_calls"):
                function = attribute_or_none(tool_call, "function")
                args = attribute_or_none(function, "arguments") if function else None
                if isinstance(args, str):
                    _write_attribute(function, "arguments", await pii_processed_arguments(args))

            # 3. Process legacy function calls
            function_call = attribute_or_none(message, "function_call")
            legacy_args = attribute_or_none(function_call, "arguments") if function_call else None
            if isinstance(legacy_args, str):
                _write_attribute(function_call, "arguments", await pii_processed_arguments(legacy_args))
        return response

    @staticmethod
    async def _process_server_tool_calls_for_pii(
        message: Message,
        pii_processed_text: Callable[[str], Awaitable[str]],
    ) -> None:
        server_tool_calls: Final = anthropic_server_tool_call_params(message)
        if not server_tool_calls:
            return
        processed_arguments: Final = tuple(
            [
                await pii_processed_text(server_tool_call["function"].get("arguments", ""))
                for server_tool_call in server_tool_calls
            ]
        )
        write_anthropic_server_tool_calls(
            message,
            tuple(map(with_server_tool_call_arguments, server_tool_calls, processed_arguments)),
        )

    async def _mask_output_response(
        self,
        response: ModelResponse | EmbeddingResponse | ImageResponse,
        request_data: dict,
    ):
        """
        Apply Presidio masking on model responses (non-streaming).
        """
        if not isinstance(response, ModelResponse):
            return response

        # skip streaming here; handled in async_post_call_streaming_iterator_hook
        if isinstance(response, ModelResponseStream):
            return response

        await self._process_response_for_pii(
            response=response,
            request_data=request_data,
            mode="mask",
        )
        return response

    async def _mask_buffered_model_response_stream(
        self, all_chunks: Sequence[ModelResponseStream], request_data: dict
    ) -> tuple[object, ...]:
        from litellm.llms.base_llm.base_model_iterator import (
            convert_model_response_to_streaming,
        )
        from litellm.main import stream_chunk_builder
        from litellm.types.utils import ModelResponse

        assembled: Final = stream_chunk_builder(chunks=list(all_chunks), messages=request_data.get("messages"))
        if not isinstance(assembled, ModelResponse):
            return tuple(all_chunks)
        await self._process_response_for_pii(response=assembled, request_data=request_data, mode="mask")
        return (convert_model_response_to_streaming(assembled),)

    async def _stream_apply_output_masking(
        self,
        response: AsyncIterable[object],
        request_data: dict,
    ) -> AsyncGenerator[object, None]:
        """Apply Presidio masking to streaming output (apply_to_output=True path)."""
        all_chunks: list[ModelResponseStream] = []
        passthrough_due_to_unknown_stream_shape = False
        try:
            stream: Final = _coalesce_first_sse_frame(response.__aiter__())
            async for chunk in stream:
                if isinstance(chunk, ModelResponseStream):
                    if passthrough_due_to_unknown_stream_shape:
                        yield chunk
                    else:
                        all_chunks.append(chunk)
                elif isinstance(chunk, _SsePreface):
                    yield chunk.raw
                elif isinstance(chunk, bytes):
                    first_frame_is_anthropic = (
                        not passthrough_due_to_unknown_stream_shape
                        and not all_chunks
                        and self._is_anthropic_first_frame(chunk)
                    )
                    if not first_frame_is_anthropic:
                        passthrough_due_to_unknown_stream_shape = (
                            passthrough_due_to_unknown_stream_shape or not all_chunks
                        )
                        yield chunk
                        continue
                    for masked_chunk in await self._mask_anthropic_sse_stream(chunk, stream, request_data):
                        yield masked_chunk
                    return
                else:
                    for buffered_chunk in _flush_unmaskable_buffer(all_chunks):
                        yield buffered_chunk
                    all_chunks = []
                    passthrough_due_to_unknown_stream_shape = True
                    yield chunk
            if passthrough_due_to_unknown_stream_shape:
                verbose_proxy_logger.warning(
                    "Presidio apply_to_output: streaming response was not a parsed chat completion stream "
                    "(raw non-Anthropic SSE passthrough or /v1/responses events). "
                    "Output PII masking was skipped for this response."
                )
                return
            if not all_chunks:
                verbose_proxy_logger.warning(
                    "Presidio apply_to_output: streaming response contained no "
                    "ModelResponseStream chunks (an empty upstream stream). "
                    "Output PII masking was skipped for this response."
                )
                return

            for masked_chunk in await self._mask_buffered_model_response_stream(all_chunks, request_data):
                yield masked_chunk

        except Exception as e:
            if not all_chunks or isinstance(e, BlockedPiiEntityError):
                raise
            verbose_proxy_logger.error("Error masking streaming PII output: %s", e)
            for chunk in all_chunks:
                yield chunk

    def _is_anthropic_first_frame(self, chunk: bytes) -> bool:
        if is_anthropic_sse_stream((chunk,)):
            return True
        if has_unreadable_json_event((chunk,)):
            raise self._unreadable_stream_refusal()
        return False

    def _unreadable_stream_refusal(self) -> GuardrailRaisedException:
        return GuardrailRaisedException(
            guardrail_name=self.guardrail_name,
            message="Presidio could not read the upstream stream; cannot verify PII when PII protection is configured",
            should_wrap_with_default_message=False,
        )

    async def _mask_anthropic_sse_stream(
        self, first_chunk: bytes, rest: AsyncIterator[object], request_data: dict
    ) -> tuple[object, ...]:
        """The frames the client gets: only what Presidio read and masked, or a refusal.

        A stream keeps its shape, with the text and tool input Presidio masked replaced where the upstream sent them.
        Without ``mask_tool_calls`` a tool input is masked only along with text of its message. A stream a client could
        read another way is refused, since its PII could sit where Presidio never looked, and one that failed before
        its message started is its error alone.
        """
        rest_chunks: Final = [chunk async for chunk in rest]
        chunks: Final = (first_chunk, *rest_chunks)
        masker: Final = self._pii_processor(request_data, "mask")
        error_data: Final = tuple(
            frame.error_data for frame in anthropic_sse_frames(chunks) if frame.error_data is not None
        )
        masked_error_data: Final = MappingProxyType({data: await masker.json(data) for data in frozenset(error_data)})
        blocks: Final = anthropic_sse_blocks(chunks)
        if blocks is None:
            if not error_data or "message_start" in anthropic_sse_event_types(chunks):
                raise self._unreadable_stream_refusal()
            return anthropic_sse_chunks_of_events(
                tuple(anthropic_sse_error_event_of(masked_error_data[data]) for data in error_data)
            )
        texts: Final = await masker.texts(tuple(block.text for block in blocks))
        tool_inputs: Final = [await masker.json(block.tool_input) if block.tool_input else "" for block in blocks]
        masks_tool_inputs: Final = self.mask_tool_calls or texts != tuple(block.text for block in blocks)
        masked_blocks: Final = tuple(
            replace(block, text=text, tool_input=tool_input if masks_tool_inputs else block.tool_input)
            for block, text, tool_input in zip(blocks, texts, tool_inputs, strict=True)
        )
        return anthropic_sse_chunks_rewritten(chunks, blocks, masked_blocks, masked_error_data)

    @staticmethod
    def _unmask_sse_bytes_chunk(chunk: bytes, pii_tokens: dict[str, str]) -> bytes:
        try:
            text: Final = chunk.decode("utf-8")
        except UnicodeDecodeError:
            return chunk

        result_lines: Final[list[str]] = []
        for line in text.split("\n"):
            line = line.rstrip("\r")
            if line.startswith("data: ") and line != "data: [DONE]":
                raw_json = line[6:]
                try:
                    event = _DECODED_JSON.validate_python(json.loads(raw_json))
                    delta = event.get("delta") if isinstance(event, dict) else None
                    if (
                        isinstance(event, dict)
                        and isinstance(delta, dict)
                        and event.get("type") == "content_block_delta"
                        and delta.get("type") == "text_delta"
                        and isinstance(delta.get("text"), str)
                    ):
                        unmasked = OPTIONAL_PresidioPIIMasking._unmask_pii_text(delta["text"], pii_tokens)
                        if unmasked != delta["text"]:
                            event["delta"]["text"] = unmasked
                            line = "data: " + json.dumps(event, ensure_ascii=False)
                except (ValueError, RecursionError, KeyError, TypeError):
                    pass
            result_lines.append(line)

        return "\n".join(result_lines).encode("utf-8")

    def _unmask_responses_api_completed_chunk(self, chunk: object, pii_tokens: dict[str, str]) -> None:
        """
        Unmask PII tokens in-place for a ``response.completed`` / ``response.incomplete`` Responses API event.

        The chunk carries a ``response`` attribute (ResponsesAPIResponse) whose
        ``output`` list holds message items.  Each item has a ``content`` list of
        blocks; text blocks expose a ``.text`` string attribute.  We walk the tree
        and replace every PII token with its original value.
        """
        response_obj: Final[object] = getattr(chunk, "response", None)
        if response_obj is None:
            return

        output: Final = getattr(response_obj, "output", None) or []
        for output_item in output:
            content = getattr(output_item, "content", None) or []
            for content_block in content:
                if isinstance(content_block, dict):
                    if isinstance(content_block.get("text"), str):
                        content_block["text"] = self._unmask_pii_text(content_block["text"], pii_tokens)
                elif hasattr(content_block, "text") and isinstance(content_block.text, str):
                    content_block.text = self._unmask_pii_text(content_block.text, pii_tokens)

    async def _stream_pii_unmasking(
        self,
        response: AsyncIterable[object],
        request_data: dict,
    ) -> AsyncGenerator[object, None]:
        """Apply PII unmasking to streaming output (output_parse_pii=True path)."""
        from litellm.llms.base_llm.base_model_iterator import (
            convert_model_response_to_streaming,
        )
        from litellm.main import stream_chunk_builder
        from litellm.types.utils import ModelResponse

        metadata: Final = (request_data.get("metadata") or {}) if request_data else {}
        pii_tokens: Final[dict[str, str]] = metadata.get("pii_tokens", {})

        remaining_chunks: list[ModelResponseStream] = []
        saw_non_chat_chunk = False
        try:
            async for chunk in response:
                if isinstance(chunk, ModelResponseStream):
                    if saw_non_chat_chunk:
                        yield chunk
                    else:
                        remaining_chunks.append(chunk)
                elif isinstance(chunk, bytes):
                    if pii_tokens:
                        yield self._unmask_sse_bytes_chunk(chunk, pii_tokens)
                    else:
                        yield chunk
                    continue
                else:
                    # /v1/responses events: unmask response.completed text in-place.
                    # A mixed stream can't be reassembled, so flush buffered chat
                    # chunks in order before passthrough instead of dropping them.
                    if remaining_chunks and not saw_non_chat_chunk:
                        for buffered_chunk in remaining_chunks:
                            yield buffered_chunk
                        remaining_chunks = []
                    chunk_type = getattr(chunk, "type", None)
                    if chunk_type in ("response.completed", "response.incomplete") and pii_tokens:
                        self._unmask_responses_api_completed_chunk(chunk, pii_tokens)
                    saw_non_chat_chunk = True
                    yield chunk

            if saw_non_chat_chunk:
                return

            if not remaining_chunks:
                return

            assembled_model_response: Final = stream_chunk_builder(
                chunks=remaining_chunks, messages=request_data.get("messages")
            )

            if not isinstance(assembled_model_response, ModelResponse):
                for chunk in remaining_chunks:
                    yield chunk
                return

            self._preserve_usage_from_last_chunk(assembled_model_response, remaining_chunks)

            await self._process_response_for_pii(
                response=assembled_model_response,
                request_data=request_data,
                mode="unmask",
            )

            mock_response_stream: Final = convert_model_response_to_streaming(assembled_model_response)
            yield mock_response_stream

        except Exception as e:
            verbose_proxy_logger.error("Error in PII streaming processing: %s", e)
            for chunk in remaining_chunks:
                yield chunk

    async def async_post_call_streaming_iterator_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        response: AsyncIterable[object],
        request_data: dict,
    ) -> AsyncGenerator[object, None]:
        """
        Process streaming response chunks to unmask PII tokens when needed.

        Note: the return type includes `bytes` because Anthropic native SSE
        streaming sends raw bytes chunks that pass through untransformed.
        The base class declares ModelResponseStream only.
        """
        if self.apply_to_output:
            async for chunk in self._stream_apply_output_masking(response, request_data):
                yield chunk
            return

        metadata: Final = (request_data.get("metadata") or {}) if request_data else {}
        pii_tokens: Final = metadata.get("pii_tokens", {})
        if not pii_tokens and request_data:
            verbose_proxy_logger.debug("No pii_tokens in request_data['metadata'] for streaming unmask path")
        if not (self.output_parse_pii and pii_tokens):
            async for chunk in response:
                yield chunk
            return

        async for chunk in self._stream_pii_unmasking(response, request_data):
            yield chunk

    @staticmethod
    def _preserve_usage_from_last_chunk(
        assembled_model_response: ModelResponse,
        chunks: list[ModelResponseStream],
    ) -> None:
        """Copy usage metadata from the last chunk when stream_chunk_builder misses it."""
        if not getattr(assembled_model_response, "usage", None) and chunks:
            last_chunk_usage: Final = getattr(chunks[-1], "usage", None)
            if last_chunk_usage:
                setattr(assembled_model_response, "usage", last_chunk_usage)

    def get_presidio_settings_from_request_data(self, data: dict) -> PresidioPerRequestConfig | None:
        if "metadata" in data:
            _metadata: Final = data.get("metadata", None)
            if _metadata is None:
                return None
            _guardrail_config: Final = _metadata.get("guardrail_config")
            if _guardrail_config:
                _presidio_config: Final = PresidioPerRequestConfig(**_guardrail_config)
                return _presidio_config

        return None

    @log_guardrail_information
    async def apply_guardrail(
        self,
        inputs: "GenericGuardrailAPIInputs",
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional["LiteLLMLoggingObj"] = None,
    ) -> "GenericGuardrailAPIInputs":
        """
        UI will call this function to check:
            1. If the connection to the guardrail is working
            2. When Testing the guardrail with some text, this function will be called with the input text and returns a text after applying the guardrail
        """
        texts: Final = inputs.get("texts", [])

        metadata: Final = (request_data.get("metadata") or {}) if request_data else {}
        pii_tokens: Final = metadata.get("pii_tokens", {})

        if input_type == "response" and (
            self._callback_role == "restore"
            or (self._callback_role is None and not self.apply_to_output and pii_tokens)
        ):
            return {**inputs, "texts": [self._unmask_pii_text(text, pii_tokens) for text in texts]}

        masker: Final = self._pii_masker(self.output_parse_pii, None, request_data)
        scanned: Final[GenericGuardrailAPIInputs] = {**inputs, "texts": [await masker.text(text) for text in texts]}
        tool_calls: Final = inputs.get("tool_calls")
        if input_type == "request" or not self.mask_tool_calls or not tool_calls:
            return scanned
        scanned_tool_calls: Final = [await _processed_tool_call(call, masker.json) for call in tool_calls]
        return {
            **scanned,
            "tool_calls": cast(  # cast-ok: each call keeps the type it came in as
                list[ChatCompletionToolCallChunk], scanned_tool_calls
            ),
        }

    def supports_scan_tool_descriptions_as_text(self) -> bool:
        return True

    def update_in_memory_litellm_params(self, litellm_params: LitellmParams) -> None:
        """
        Update the guardrails litellm params in memory
        """
        super().update_in_memory_litellm_params(litellm_params)
        if self.apply_to_output:
            self.output_parse_pii = False
        if litellm_params.pii_entities_config:
            self.pii_entities_config = litellm_params.pii_entities_config
        if litellm_params.presidio_score_thresholds:
            self.presidio_score_thresholds = litellm_params.presidio_score_thresholds
        if litellm_params.presidio_entities_deny_list:
            self.presidio_entities_deny_list = litellm_params.presidio_entities_deny_list
        if litellm_params.presidio_analyze_chunk_size_bytes is not None:
            # Same validation as __init__: a non-positive value from a guardrail
            # update must not silently disable detection via degenerate chunking.
            self.presidio_analyze_chunk_size_bytes = self._coerce_analyze_chunk_size(
                litellm_params.presidio_analyze_chunk_size_bytes
            )


OPTIONAL_PresidioPIIMasking = _OPTIONAL_PresidioPIIMasking
