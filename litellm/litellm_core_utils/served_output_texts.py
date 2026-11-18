"""Assistant text the caller received, per choice, so the logging payload stores the response a
post-call guardrail rewrote rather than the provider response the proxy assembled before it ran."""

from __future__ import annotations

import json
import threading
from collections.abc import Mapping, Sequence
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
_FAILURE_SKIP_MAX_EDGES: Final = 8

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


def _can_skip_failure_state(fu_edges: Mapping[str, int], u_edges: Mapping[str, int]) -> bool:
    return (
        len(fu_edges) <= _FAILURE_SKIP_MAX_EDGES and len(fu_edges) <= len(u_edges) and fu_edges.keys() <= u_edges.keys()
    )


class _ReverseAhoCorasickMatcher:
    """Linear-time O(N + M) leftmost-longest multi-pattern matcher over reversed keys.

    Scanning right-to-left means every suffix matching at step ``idx`` starts at the exact same
    forward index ``idx``, so each state only needs a single ``max_len`` integer (no dict-link walks).
    """

    def __init__(self, mapping: Mapping[str, str]) -> None:
        trie: Final[list[dict[str, int]]] = [{}]  # mutable-ok: automaton transition table built once
        fail: Final[list[int]] = [0]  # mutable-ok: failure links indexed by state id
        alt_fail: Final[list[int]] = [0]  # mutable-ok: shortcut failure links skipping subset key sets
        depth: Final[list[int]] = [0]  # mutable-ok: trie depth per state id
        max_len: Final[list[int]] = [0]  # mutable-ok: longest terminal key length per state id

        for key in mapping:
            if not key:
                continue
            node = 0
            for ch in reversed(key):
                nxt = trie[node].get(ch)
                if nxt is None:
                    nxt = len(trie)
                    trie[node][ch] = nxt
                    trie.append({})
                    fail.append(0)
                    alt_fail.append(0)
                    depth.append(depth[node] + 1)
                    max_len.append(0)
                node = nxt
            max_len[node] = depth[node]

        self._populate_links(trie, fail, alt_fail, max_len)
        self._trie: Sequence[Mapping[str, int]] = trie
        self._alt_fail: Sequence[int] = alt_fail
        self._max_len: Sequence[int] = max_len

    @staticmethod
    def _populate_links(
        trie: Sequence[Mapping[str, int]],
        fail: list[int],  # mutable-ok: populated in BFS order
        alt_fail: list[int],  # mutable-ok: populated in BFS order
        max_len: list[int],  # mutable-ok: propagated from failure ancestor in BFS order
    ) -> None:
        queue: Final[list[int]] = list(trie[0].values())  # mutable-ok: flat BFS state queue
        for u in queue:
            alt_fail[u] = 0  # rebind-ok: initialize root-child shortcut failure link
        head = 0  # rebind-ok: advances through BFS queue without popleft overhead
        while head < len(queue):
            r = queue[head]
            head += 1
            for ch, u in trie[r].items():
                queue.append(u)
                f = fail[r]
                while f and ch not in trie[f]:
                    f = alt_fail[f]
                fu = trie[f].get(ch, 0)
                fail[u] = fu  # rebind-ok: assign failure transition for child state
                alt_fail[u] = (  # rebind-ok: skip failure states whose key set is a subset of child's
                    alt_fail[fu] if (fu and _can_skip_failure_state(trie[fu], trie[u])) else fu
                )
                max_len[u] = max(max_len[u], max_len[fu])  # rebind-ok: inherit longest match length

    @staticmethod
    def _step_fallback(
        trie: Sequence[Mapping[str, int]],
        alt_fail: Sequence[int],
        state: int,
        ch: str,
    ) -> int:
        cur = alt_fail[state]  # rebind-ok: walks shortcut failure links until character transition found
        while cur and ch not in trie[cur]:
            cur = alt_fail[cur]
        return trie[cur].get(ch, 0)

    def _collect_matches_descending(self, text: str) -> Sequence[tuple[int, int]]:
        trie: Final = self._trie
        alt_fail: Final = self._alt_fail
        max_len: Final = self._max_len
        matches: Final[list[tuple[int, int]]] = []  # mutable-ok: accumulates (start, length) spans in descending order
        state = 0  # rebind-ok: current reverse Aho-Corasick state

        for idx in range(len(text) - 1, -1, -1):
            ch = text[idx]
            nxt = trie[state].get(ch)
            state = nxt if nxt is not None else self._step_fallback(trie, alt_fail, state, ch)
            ml = max_len[state]
            if ml:
                matches.append((idx, ml))
        return matches

    def sanitize(self, text: str, mapping: Mapping[str, str]) -> str:
        matches: Final = self._collect_matches_descending(text)
        if not matches:
            return text
        out: Final[list[str]] = []  # mutable-ok: linear output piece buffer
        pos = 0  # rebind-ok: tracks end of last emitted replacement
        for s, ml in reversed(matches):
            if s < pos:
                continue
            e = s + ml
            if s > pos:
                out.append(text[pos:s])
            out.append(mapping[text[s:e]])
            pos = e
        if pos < len(text):
            out.append(text[pos:])
        return "".join(out)


def _build_sanitization_mapping(
    pairs: Sequence[tuple[str, str]],
) -> dict[str, str]:  # mutable-ok: returns deduplicated cleartext-to-surrogate map built once
    mapping: Final[dict[str, str]] = {}  # mutable-ok: build deduplicated cleartext-to-surrogate map locally
    for cleartext, token in pairs:
        if not cleartext or cleartext == token:
            continue
        for variant in (
            cleartext,
            json.dumps(cleartext)[1:-1],
            json.dumps(cleartext, ensure_ascii=False)[1:-1],
        ):
            if variant and variant != token:
                mapping[variant] = token
    return mapping


class _OutputLogSanitizer:
    def __init__(self, pairs: tuple[tuple[str, str], ...]) -> None:
        self._pairs: tuple[tuple[str, str], ...] = pairs
        self._lock: Final = threading.Lock()
        self._compiled: tuple[Mapping[str, str], _ReverseAhoCorasickMatcher] | None = None
        self._cleared: bool = False

    def matches_pairs(self, pairs: tuple[tuple[str, str], ...]) -> bool:
        with self._lock:
            return not self._cleared and self._pairs == pairs

    def is_active(self) -> bool:
        with self._lock:
            return not self._cleared and bool(self._pairs)

    def _get_compiled(self) -> tuple[Mapping[str, str], _ReverseAhoCorasickMatcher] | None:
        with self._lock:
            if self._cleared or not self._pairs:
                return None
            if self._compiled is not None:
                return self._compiled
            mapping: Final = _build_sanitization_mapping(self._pairs)
            if not mapping:
                self._pairs = ()
                return None
            compiled: Final = (mapping, _ReverseAhoCorasickMatcher(mapping))
            self._compiled = compiled
            return compiled

    def sanitize_text(self, text: str) -> str:
        if not text:
            return text
        compiled: Final = self._get_compiled()
        if compiled is None:
            return text
        return compiled[1].sanitize(text, compiled[0])

    def sanitize_texts(self, texts: ServedTexts) -> ServedTexts:
        return tuple(self.sanitize_text(t) if t is not None else None for t in texts)

    def clear(self) -> None:
        with self._lock:
            self._cleared = True
            self._pairs = ()
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
