"""Assistant text the caller received, per choice, so the logging payload stores the response a
post-call guardrail rewrote rather than the provider response the proxy assembled before it ran."""

from __future__ import annotations

import json
import re
import threading
from array import array
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import chain
from typing import Final, Literal
from weakref import WeakKeyDictionary, WeakValueDictionary

import anyio
import anyio.lowlevel
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from litellm.litellm_core_utils.asyncify import asyncify
from litellm.types.llms.base import LiteLLMBaseModel
from litellm.types.utils import ModelResponse, ModelResponseStream, TextCompletionResponse

SERVED_OUTPUT_TEXTS_KEY: Final = "served_output_texts"
_SERVED_OUTPUT_SANITIZER_KEY: Final = "_served_output_sanitizer_id"
_SANITIZER_MAX_CONCURRENT_JOBS: Final = 4
_SANITIZER_OFFLOAD_LIMITER: Final = anyio.lowlevel.RunVar[anyio.CapacityLimiter](
    "litellm_served_output_sanitizer_limiter"
)
_JSON_NEEDS_ESCAPE_RE: Final = re.compile(r'[\x00-\x1f"\\]')
_MAX_SANITIZER_PATTERN_CHARS: Final = 2 * 1024 * 1024
_MAX_TRIE_KEY_LEN: Final = 1024
_MAX_TRIE_NODES: Final = 131_072
_MAX_FALLBACK_SCAN_OPS: Final = 32 * 1024 * 1024
_REDACTED_LOG_PLACEHOLDER: Final = "[ZTDS_REDACTED_OUTPUT]"

_JSON_OBJECT: Final = TypeAdapter(dict[str, object])
_JSON_LIST: Final = TypeAdapter(list[object])
_TEXTS: Final = TypeAdapter(tuple[str | None, ...])


def _sanitizer_offload_limiter_for_this_loop() -> anyio.CapacityLimiter:
    existing: Final = _SANITIZER_OFFLOAD_LIMITER.get(None)
    if existing is not None:
        return existing
    created: Final = anyio.CapacityLimiter(_SANITIZER_MAX_CONCURRENT_JOBS)
    _SANITIZER_OFFLOAD_LIMITER.set(created)
    return created


@dataclass(frozen=True)
class _CompactTrieData:
    first_ch: list[str]  # mutable-ok: primary edge character per trie state
    first_nxt: array[int]
    extra: dict[int, dict[str, int]]  # mutable-ok: sparse secondary edges for branching states
    fail: array[int]
    alt_fail: array[int]
    depth: array[int]
    max_len: array[int]


def _step_edge(data: _CompactTrieData, state: int, ch: str) -> int:
    if data.first_ch[state] == ch:
        return data.first_nxt[state]
    ed: Final = data.extra.get(state)
    return ed.get(ch, -1) if ed is not None else -1


def _step_fallback(data: _CompactTrieData, state: int, ch: str) -> int:
    cur = data.alt_fail[state]  # rebind-ok: walks shortcut failure links until character transition found
    while cur:
        nxt = _step_edge(data, cur, ch)
        if nxt >= 0:
            return nxt
        cur = data.alt_fail[cur]
    nxt0: Final = _step_edge(data, 0, ch)
    return max(nxt0, 0)


def _insert_reversed_key(data: _CompactTrieData, key: str) -> bool:
    node = 0  # rebind-ok: walks reversed trie from root
    for ch in reversed(key):
        nxt = _step_edge(data, node, ch)
        if nxt < 0:
            nxt = len(data.first_ch)
            if nxt >= _MAX_TRIE_NODES:
                return False
            if not data.first_ch[node]:
                data.first_ch[node] = ch  # rebind-ok: store first outgoing transition char
                data.first_nxt[node] = nxt  # rebind-ok: store first outgoing child state
            else:
                data.extra.setdefault(node, {})[ch] = nxt
            data.first_ch.append("")
            data.first_nxt.append(0)
            data.fail.append(0)
            data.alt_fail.append(0)
            data.depth.append(data.depth[node] + 1)
            data.max_len.append(0)
        node = nxt
    data.max_len[node] = data.depth[node]  # rebind-ok: record terminal key depth
    return True


def _find_failure_transition(data: _CompactTrieData, r: int, ch: str) -> int:
    f = data.fail[r]  # rebind-ok: climbs ancestor failure links until edge matches
    while f:
        fnxt = _step_edge(data, f, ch)
        if fnxt >= 0:
            return fnxt
        f = data.alt_fail[f]
    fnxt0: Final = _step_edge(data, 0, ch)
    return max(fnxt0, 0)


def _link_child(data: _CompactTrieData, r: int, ch: str, u: int) -> None:
    fu: Final = _find_failure_transition(data, r, ch)
    data.fail[u] = fu  # rebind-ok: assign failure link in BFS order
    fu_ch: Final = data.first_ch[fu]
    single_or_leaf: Final = fu > 0 and fu not in data.extra
    covered_by_u: Final = not fu_ch or _step_edge(data, u, fu_ch) >= 0
    data.alt_fail[u] = (  # rebind-ok: skip single-edge failure link covered by child
        data.alt_fail[fu] if (single_or_leaf and covered_by_u) else fu
    )
    data.max_len[u] = max(data.max_len[u], data.max_len[fu])  # rebind-ok: inherit longest suffix len


def _populate_links(data: _CompactTrieData) -> None:
    queue: Final[array[int]] = array("I")
    if data.first_ch[0]:
        queue.append(data.first_nxt[0])
        if 0 in data.extra:
            queue.extend(data.extra[0].values())
    head = 0  # rebind-ok: advances through BFS queue without popleft overhead
    while head < len(queue):
        r = queue[head]
        head += 1
        rc = data.first_ch[r]
        if not rc:
            continue
        r_extra = data.extra.get(r)
        children = ((rc, data.first_nxt[r]), *r_extra.items()) if r_extra else ((rc, data.first_nxt[r]),)
        for ch, u in children:
            queue.append(u)
            _link_child(data, r, ch, u)


class _ReverseAhoCorasickMatcher:
    """Compact array-backed O(N + M) leftmost-longest matcher over reversed bounded keys."""

    def __init__(self, data: _CompactTrieData) -> None:
        self._data: Final = data

    @classmethod
    def try_build(cls, keys: Sequence[str]) -> _ReverseAhoCorasickMatcher | None:
        if not keys:
            return None
        data: Final = _CompactTrieData(
            first_ch=[""],
            first_nxt=array("I", [0]),
            extra={},
            fail=array("I", [0]),
            alt_fail=array("I", [0]),
            depth=array("I", [0]),
            max_len=array("I", [0]),
        )
        for key in keys:
            if not key:
                continue
            if not _insert_reversed_key(data, key):
                return None
        _populate_links(data)
        return cls(data)

    def scan(self, text: str) -> tuple[array[int], array[int]]:
        data: Final = self._data
        first_ch: Final = data.first_ch
        first_nxt: Final = data.first_nxt
        extra: Final = data.extra
        max_len: Final = data.max_len
        starts: Final[array[int]] = array("I")
        lens: Final[array[int]] = array("I")
        state = 0  # rebind-ok: current reverse Aho-Corasick state

        for idx in range(len(text) - 1, -1, -1):
            ch = text[idx]
            if first_ch[state] == ch:
                state = first_nxt[state]
            else:
                ed = extra.get(state)
                nxt = ed.get(ch, -1) if ed is not None else -1
                state = nxt if nxt >= 0 else _step_fallback(data, state, ch)
            ml = max_len[state]
            if ml:
                starts.append(idx)
                lens.append(ml)
        return starts, lens


def _cleartext_variants(cleartext: str) -> tuple[str, ...]:
    if cleartext.isascii() and _JSON_NEEDS_ESCAPE_RE.search(cleartext) is None:
        return (cleartext,)
    return (
        cleartext,
        json.dumps(cleartext)[1:-1],
        json.dumps(cleartext, ensure_ascii=False)[1:-1],
    )


def _build_sanitization_mapping(pairs: Sequence[tuple[str, str]]) -> Mapping[str, str] | None:
    mapping: Final[dict[str, str]] = {}  # mutable-ok: build deduplicated cleartext-to-surrogate map locally
    total_chars = 0  # rebind-ok: enforces strict per-request pattern character budget
    for cleartext, token in pairs:
        if not cleartext or cleartext == token:
            continue
        if len(cleartext) > _MAX_SANITIZER_PATTERN_CHARS:
            return None
        for variant in _cleartext_variants(cleartext):
            if not variant or variant == token:
                continue
            if variant not in mapping:
                total_chars += len(variant)
                if total_chars > _MAX_SANITIZER_PATTERN_CHARS:
                    return None
            mapping[variant] = token
    return mapping


def _emit_trie_only(text: str, mapping: Mapping[str, str], starts: array[int], lens: array[int]) -> str:
    if not starts:
        return text
    out: Final[list[str]] = []  # mutable-ok: linear output piece buffer
    pos = 0  # rebind-ok: tracks end of last emitted replacement
    for idx in range(len(starts) - 1, -1, -1):
        s = starts[idx]
        if s < pos:
            continue
        e = s + lens[idx]
        if s > pos:
            out.append(text[pos:s])
        out.append(mapping[text[s:e]])
        pos = e
    if pos < len(text):
        out.append(text[pos:])
    return "".join(out)


def _next_long_match(
    text: str,
    eligible_long: tuple[str, ...],
    long_pos: list[int],  # mutable-ok: advances monotonic per-key search cursors in place
    active_long: list[int],  # mutable-ok: compacts remaining active long-key indices
    pos: int,
) -> tuple[int, int]:
    best_s = len(text) + 1  # rebind-ok: earliest long-key match offset >= pos
    best_ml = 0  # rebind-ok: longest long-key match length at best_s
    keep = 0  # rebind-ok: compaction write cursor in active_long
    for idx_j in range(len(active_long)):
        j = active_long[idx_j]
        lp = long_pos[j]
        if lp < pos:
            lp = text.find(eligible_long[j], pos)
            long_pos[j] = lp  # rebind-ok: advance monotonic search offset for long key
        if lp >= 0:
            active_long[keep] = j  # rebind-ok: compact active long-key list in place
            keep += 1
            kl = len(eligible_long[j])
            if lp < best_s or (lp == best_s and kl > best_ml):
                best_s = lp
                best_ml = kl
    del active_long[keep:]  # rebind-ok: prune exhausted long-key indices
    return best_s, best_ml


def _merge_hybrid_matches(
    text: str,
    mapping: Mapping[str, str],
    starts: array[int],
    lens: array[int],
    eligible_long: tuple[str, ...],
) -> str:
    long_pos: Final[list[int]] = [text.find(k, 0) for k in eligible_long]  # mutable-ok: per-key cursor table
    active_long: Final[list[int]] = [j for j, p in enumerate(long_pos) if p >= 0]  # mutable-ok: remaining active keys
    if not starts and not active_long:
        return text
    out: Final[list[str]] = []  # mutable-ok: linear output piece buffer
    pos = 0  # rebind-ok: tracks end of last emitted replacement
    ti = len(starts) - 1  # rebind-ok: walks reverse-order trie match arrays left-to-right
    n: Final = len(text)
    while pos < n:
        while ti >= 0 and starts[ti] < pos:
            ti -= 1
        trie_s = starts[ti] if ti >= 0 else n + 1
        trie_ml = lens[ti] if ti >= 0 else 0
        long_s, long_ml = _next_long_match(text, eligible_long, long_pos, active_long, pos)
        pick_long = long_s < trie_s or (long_s == trie_s and long_ml > trie_ml)
        best_s, best_ml = (long_s, long_ml) if pick_long else (trie_s, trie_ml)
        if best_s > n:
            break
        if best_s > pos:
            out.append(text[pos:best_s])
        out.append(mapping[text[best_s : best_s + best_ml]])
        pos = best_s + best_ml
    if pos < n:
        out.append(text[pos:])
    return "".join(out)


@dataclass(frozen=True)
class _CompiledSanitizer:
    budget_exceeded: bool
    mapping: Mapping[str, str]
    trie: _ReverseAhoCorasickMatcher | None
    long_keys: tuple[str, ...]


def _compile_sanitizer(pairs: Sequence[tuple[str, str]]) -> _CompiledSanitizer | None:
    mapping: Final = _build_sanitization_mapping(pairs)
    if mapping is None:
        return _CompiledSanitizer(budget_exceeded=True, mapping={}, trie=None, long_keys=())
    if not mapping:
        return None
    short_keys: Final = tuple(k for k in mapping if len(k) <= _MAX_TRIE_KEY_LEN)
    long_keys: Final = tuple(k for k in mapping if len(k) > _MAX_TRIE_KEY_LEN)
    trie: Final = _ReverseAhoCorasickMatcher.try_build(short_keys) if short_keys else None
    if short_keys and trie is None:
        return _CompiledSanitizer(budget_exceeded=False, mapping=mapping, trie=None, long_keys=tuple(mapping.keys()))
    return _CompiledSanitizer(budget_exceeded=False, mapping=mapping, trie=trie, long_keys=long_keys)


def _execute_hybrid_scan(text: str, compiled: _CompiledSanitizer) -> str:
    eligible_long: Final = tuple(k for k in compiled.long_keys if len(k) <= len(text))
    if len(eligible_long) * len(text) > _MAX_FALLBACK_SCAN_OPS:
        return _REDACTED_LOG_PLACEHOLDER
    starts, lens = compiled.trie.scan(text) if compiled.trie is not None else (array("I"), array("I"))
    if not eligible_long:
        return _emit_trie_only(text, compiled.mapping, starts, lens)
    return _merge_hybrid_matches(text, compiled.mapping, starts, lens, eligible_long)


class _OutputLogSanitizer:
    def __init__(self, pairs: tuple[tuple[str, str], ...]) -> None:
        self._pairs: tuple[tuple[str, str], ...] = pairs
        self._min_key_len: int = min((len(c) for c, t in pairs if c and c != t), default=0)
        self._lock: Final = threading.Lock()
        self._compiled: _CompiledSanitizer | None = None
        self._cleared: bool = False

    def matches_pairs(self, pairs: tuple[tuple[str, str], ...]) -> bool:
        with self._lock:
            return not self._cleared and self._pairs == pairs

    def is_active(self) -> bool:
        with self._lock:
            return not self._cleared and bool(self._pairs)

    def _get_compiled(self) -> _CompiledSanitizer | None:
        with self._lock:
            if self._cleared or not self._pairs:
                return None
            if self._compiled is not None:
                return self._compiled
            compiled: Final = _compile_sanitizer(self._pairs)
            if compiled is None:
                self._pairs = ()
                self._min_key_len = 0
                return None
            self._compiled = compiled
            return compiled

    def sanitize_text(self, text: str) -> str:
        if not text or self._min_key_len == 0 or len(text) < self._min_key_len:
            return text
        compiled: Final = self._get_compiled()
        if compiled is None:
            return text
        if compiled.budget_exceeded:
            return _REDACTED_LOG_PLACEHOLDER
        return _execute_hybrid_scan(text, compiled)

    def sanitize_texts(self, texts: ServedTexts) -> ServedTexts:
        return tuple(self.sanitize_text(t) if t is not None else None for t in texts)

    def clear(self) -> None:
        with self._lock:
            self._cleared = True
            self._pairs = ()
            self._min_key_len = 0
            self._compiled = None


_LOGGING_SANITIZERS: Final = WeakKeyDictionary[object, _OutputLogSanitizer]()
_DETAILS_SANITIZERS: Final = WeakValueDictionary[str, _OutputLogSanitizer]()

ServedTexts = tuple[str | None, ...]


class _TextBlock(LiteLLMBaseModel):
    type: str
    text: str | None = None


class _AnthropicMessage(LiteLLMBaseModel):
    type: Literal["message"]
    content: list[_TextBlock]


class _ResponsesOutputItem(LiteLLMBaseModel):
    type: str
    content: list[_TextBlock] = Field(default=[])


class _ResponsesResponse(LiteLLMBaseModel):
    object: Literal["response"]
    output: list[_ResponsesOutputItem]


class _ChatChoices(LiteLLMBaseModel):
    choices: list[object]


class _TextCompletionChoice(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    index: int | None = None
    text: str | None = None


class _TextCompletionChunk(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    choices: tuple[_TextCompletionChoice, ...]


def _as_json_object(response: object) -> dict[str, object] | None:
    candidate: Final = response.model_dump() if isinstance(response, BaseModel) else response
    try:
        return _JSON_OBJECT.validate_python(candidate)
    except ValidationError:
        return None


def _joined_block_texts(blocks: Sequence[_TextBlock], *, text_type: str) -> str | None:
    texts: Final = tuple(block.text for block in blocks if block.type == text_type and block.text is not None)
    return "".join(texts) if texts else None


def _chat_texts(response: ModelResponse) -> ServedTexts | None:
    texts: Final = tuple(
        choice.message.content if isinstance(choice.message.content, str) else None for choice in response.choices
    )
    return texts if any(text is not None for text in texts) else None


def _anthropic_message_text(response: dict[str, object]) -> str | None:
    try:
        message: Final = _AnthropicMessage.model_validate(response)
    except ValidationError:
        return None
    return _joined_block_texts(message.content, text_type="text")


def _responses_api_text(response: dict[str, object]) -> str | None:
    try:
        parsed: Final = _ResponsesResponse.model_validate(response)
    except ValidationError:
        return None
    texts: Final = tuple(
        text
        for item in parsed.output
        if item.type == "message" and (text := _joined_block_texts(item.content, text_type="output_text")) is not None
    )
    return "".join(texts) if texts else None


def _chat_dict_texts(response: dict[str, object]) -> ServedTexts | None:
    try:
        _ChatChoices.model_validate(response)
        return _chat_texts(ModelResponse(**response))
    except (ValidationError, TypeError, ValueError):
        return None


def served_output_texts(response: object) -> ServedTexts | None:
    if isinstance(response, ModelResponse):
        return _chat_texts(response)
    mapping: Final = _as_json_object(response)
    if mapping is None:
        return None
    chat_texts: Final = _chat_dict_texts(mapping)
    if chat_texts is not None:
        return chat_texts
    anthropic_text: Final = _anthropic_message_text(mapping)
    text: Final = anthropic_text if anthropic_text is not None else _responses_api_text(mapping)
    return (text,) if text is not None else None


def served_stream_output_texts(chunks: Sequence[object]) -> ServedTexts | None:
    chat_chunks: Final = tuple(chunk for chunk in chunks if isinstance(chunk, ModelResponseStream))
    if chunks and len(chat_chunks) == len(chunks):
        return _indexed_texts_by_choice(tuple(chain.from_iterable(_chat_chunk_texts(chunk) for chunk in chat_chunks)))
    if chunks and all(isinstance(chunk, TextCompletionResponse) for chunk in chunks):
        return _indexed_texts_by_choice(
            tuple(chain.from_iterable(_text_completion_chunk_texts(chunk) for chunk in chunks))
        )
    from litellm.llms.anthropic.messages.stream_assembly import (
        assemble_anthropic_sse_stream,
        is_anthropic_sse_stream,
    )

    if not is_anthropic_sse_stream(chunks):
        return None
    assembled: Final = assemble_anthropic_sse_stream(chunks)
    return _chat_texts(assembled) if assembled is not None else None


def _chat_chunk_texts(chunk: ModelResponseStream) -> tuple[tuple[int, str | None], ...]:
    return tuple(
        (choice.index or 0, content if isinstance(content := choice.delta.content, str) else None)
        for choice in chunk.choices
    )


def _text_completion_chunk_texts(chunk: object) -> tuple[tuple[int, str | None], ...]:
    parsed: Final = _TextCompletionChunk.model_validate(_as_json_object(chunk))
    return tuple((choice.index or 0, choice.text) for choice in parsed.choices)


def _joined_choice_text(indexed_texts: Sequence[tuple[int, str | None]], index: int) -> str | None:
    texts: Final = tuple(text for text_index, text in indexed_texts if text_index == index and text is not None)
    return "".join(texts) if texts else None


def _indexed_texts_by_choice(indexed_texts: Sequence[tuple[int, str | None]]) -> ServedTexts | None:
    choice_count: Final = max((index + 1 for index, _ in indexed_texts), default=0)
    texts: Final = tuple(_joined_choice_text(indexed_texts, index) for index in range(choice_count))
    return texts if any(text is not None for text in texts) else None


def _active_logging_sanitizer(request_data: Mapping[str, object]) -> _OutputLogSanitizer | None:
    logging_obj: Final = request_data.get("litellm_logging_obj")
    if logging_obj is None:
        return None
    try:
        sanitizer: Final = _LOGGING_SANITIZERS.get(logging_obj)
    except TypeError:
        return None
    return sanitizer if sanitizer is not None and sanitizer.is_active() else None


def _active_details_sanitizer(model_call_details: Mapping[str, object]) -> _OutputLogSanitizer | None:
    sanitizer_id: Final = model_call_details.get(_SERVED_OUTPUT_SANITIZER_KEY)
    if not isinstance(sanitizer_id, str):
        return None
    sanitizer: Final = _DETAILS_SANITIZERS.get(sanitizer_id)
    return sanitizer if sanitizer is not None and sanitizer.is_active() else None


def _extract_and_sanitize_response(response: object, sanitizer: _OutputLogSanitizer) -> ServedTexts | None:
    texts: Final = served_output_texts(response)
    return sanitizer.sanitize_texts(texts) if texts is not None else None


def register_served_output_sanitizer(
    logging_obj: object,
    model_call_details: dict[str, object] | None,  # mutable-ok: stamps sanitizer id on Logging.model_call_details
    pairs: Sequence[tuple[str, str]],
    sanitizer_id: str,
) -> None:
    if logging_obj is None or not pairs:
        return
    normalized_pairs: Final = tuple(pairs)
    try:
        existing: Final = _LOGGING_SANITIZERS.get(logging_obj)
    except TypeError:
        return
    sanitizer: Final = (
        existing
        if existing is not None and existing.matches_pairs(normalized_pairs)
        else _OutputLogSanitizer(normalized_pairs)
    )
    _LOGGING_SANITIZERS[logging_obj] = sanitizer
    _DETAILS_SANITIZERS[sanitizer_id] = sanitizer
    if model_call_details is not None:
        model_call_details[_SERVED_OUTPUT_SANITIZER_KEY] = sanitizer_id  # rebind-ok: bind sanitizer id to call details


def clear_served_output_sanitizer(logging_obj: object) -> None:
    if logging_obj is None:
        return
    try:
        sanitizer: Final = _LOGGING_SANITIZERS.pop(logging_obj, None)
    except TypeError:
        return
    if sanitizer is not None:
        sanitizer.clear()


def sanitize_logged_output_str(request_data: Mapping[str, object], rendered: str | None) -> str | None:
    if rendered is None:
        return None
    sanitizer: Final = _active_logging_sanitizer(request_data)
    return sanitizer.sanitize_text(rendered) if sanitizer is not None else rendered


async def async_sanitize_logged_output_str(request_data: Mapping[str, object], rendered: str | None) -> str | None:
    if rendered is None:
        return None
    sanitizer: Final = _active_logging_sanitizer(request_data)
    if sanitizer is None:
        return rendered
    return await asyncify(sanitizer.sanitize_text, limiter=_sanitizer_offload_limiter_for_this_loop())(rendered)


def record_served_output_texts(model_call_details: dict[str, object], texts: ServedTexts | None) -> None:
    if texts is None:
        return
    sanitizer: Final = _active_details_sanitizer(model_call_details)
    recorded: Final = sanitizer.sanitize_texts(texts) if sanitizer is not None else texts
    model_call_details[SERVED_OUTPUT_TEXTS_KEY] = recorded  # rebind-ok: model_call_details is the shared kwargs bag


async def async_record_served_output_texts(
    model_call_details: dict[str, object],  # mutable-ok: writes SERVED_OUTPUT_TEXTS_KEY on Logging.model_call_details
    texts: ServedTexts | None,
) -> None:
    if texts is None:
        return
    sanitizer: Final = _active_details_sanitizer(model_call_details)
    recorded: Final = (
        await asyncify(sanitizer.sanitize_texts, limiter=_sanitizer_offload_limiter_for_this_loop())(texts)
        if sanitizer is not None
        else texts
    )
    model_call_details[SERVED_OUTPUT_TEXTS_KEY] = recorded  # rebind-ok: model_call_details is the shared kwargs bag


async def async_record_served_response_output(
    model_call_details: dict[str, object],  # mutable-ok: writes SERVED_OUTPUT_TEXTS_KEY on Logging.model_call_details
    response: object,
) -> None:
    sanitizer: Final = _active_details_sanitizer(model_call_details)
    if sanitizer is None:
        record_served_output_texts(model_call_details, served_output_texts(response))
        return
    recorded: Final = await asyncify(
        _extract_and_sanitize_response, limiter=_sanitizer_offload_limiter_for_this_loop()
    )(response, sanitizer)
    if recorded is not None:
        model_call_details[SERVED_OUTPUT_TEXTS_KEY] = recorded  # rebind-ok: model_call_details is the shared kwargs bag


def overlay_served_output_texts(
    response_obj: dict[str, object] | str | list[object] | None, served_texts: object
) -> dict[str, object] | str | list[object] | None:
    if not isinstance(response_obj, dict):
        return response_obj
    logged: Final = _as_json_object(response_obj)
    if logged is None:
        return response_obj
    try:
        texts: Final = _TEXTS.validate_python(served_texts)
        choices: Final = _JSON_LIST.validate_python(logged.get("choices"))
    except ValidationError:
        return response_obj
    return {
        **logged,
        "choices": [
            _choice_with_text(choice, texts[index]) if index < len(texts) else choice
            for index, choice in enumerate(choices)
        ],
    }


def _choice_with_text(choice: object, text: str | None) -> object:
    choice_obj: Final = _as_json_object(choice)
    if choice_obj is None or text is None:
        return choice
    message: Final = _as_json_object(choice_obj.get("message"))
    if message is not None:
        return choice if message.get("content") == text else {**choice_obj, "message": {**message, "content": text}}
    if isinstance(choice_obj.get("text"), str):
        return {**choice_obj, "text": text}
    return choice
