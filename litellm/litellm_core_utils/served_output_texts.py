"""Assistant text the caller received, per choice, so the logging payload stores the response a
post-call guardrail rewrote rather than the provider response the proxy assembled before it ran."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from itertools import chain
from typing import Final, Literal
from weakref import WeakKeyDictionary, WeakValueDictionary

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from litellm.types.llms.base import LiteLLMBaseModel
from litellm.types.utils import ModelResponse, ModelResponseStream, TextCompletionResponse

SERVED_OUTPUT_TEXTS_KEY: Final = "served_output_texts"
_SERVED_OUTPUT_SANITIZER_KEY: Final = "_served_output_sanitizer_id"

_JSON_OBJECT: Final = TypeAdapter(dict[str, object])
_JSON_LIST: Final = TypeAdapter(list[object])
_TEXTS: Final = TypeAdapter(tuple[str | None, ...])


class _AhoCorasickMatcher:
    def __init__(self, mapping: Mapping[str, str]) -> None:
        trie: Final[list[dict[str, int]]] = [{}]  # mutable-ok: automaton transition table built once
        fail: Final[list[int]] = [0]  # mutable-ok: failure links indexed by state id
        depth: Final[list[int]] = [0]  # mutable-ok: trie depth per state id
        own_len: Final[list[int]] = [0]  # mutable-ok: terminal key length per state id
        dict_link: Final[list[int]] = [0]  # mutable-ok: nearest terminal ancestor per state id

        for key in mapping:
            if not key:
                continue
            node = 0
            for ch in key:
                nxt = trie[node].get(ch)
                if nxt is None:
                    nxt = len(trie)
                    trie[node][ch] = nxt
                    trie.append({})
                    fail.append(0)
                    depth.append(depth[node] + 1)
                    own_len.append(0)
                    dict_link.append(0)
                node = nxt
            own_len[node] = depth[node]

        self._populate_links(trie, fail, own_len, dict_link)
        self._trie: list[dict[str, int]] = trie  # mutable-ok: cleared on explicit zeroization
        self._fail: list[int] = fail  # mutable-ok: cleared on explicit zeroization
        self._depth: list[int] = depth  # mutable-ok: cleared on explicit zeroization
        self._own_len: list[int] = own_len  # mutable-ok: cleared on explicit zeroization
        self._dict_link: list[int] = dict_link  # mutable-ok: cleared on explicit zeroization

    @staticmethod
    def _populate_links(
        trie: Sequence[Mapping[str, int]],
        fail: list[int],  # mutable-ok: populated in BFS order
        own_len: Sequence[int],
        dict_link: list[int],  # mutable-ok: populated in BFS order
    ) -> None:
        queue: Final[list[int]] = list(trie[0].values())  # mutable-ok: flat BFS state queue
        head = 0  # rebind-ok: advances through BFS queue without popleft overhead
        while head < len(queue):
            r = queue[head]
            head += 1
            for ch, u in trie[r].items():
                queue.append(u)
                f = fail[r]
                while f and ch not in trie[f]:
                    f = fail[f]
                fu = trie[f].get(ch, 0)
                fail[u] = fu  # rebind-ok: assign failure transition for child state
                dict_link[u] = fu if own_len[fu] else dict_link[fu]  # rebind-ok: link nearest word suffix

    @staticmethod
    def _step_state(trie: Sequence[Mapping[str, int]], fail: Sequence[int], state: int, ch: str) -> int:
        cur = state  # rebind-ok: walks failure links until character transition found
        while cur and ch not in trie[cur]:
            cur = fail[cur]
        return trie[cur].get(ch, 0)

    def _record_suffix_matches(
        self,
        start_state: int,
        end: int,
        guaranteed_until: int,
        best_end: dict[int, int],  # mutable-ok: updates earliest-start longest-end map in place
        current_max_end: int,
    ) -> int:
        own_len: Final = self._own_len
        dict_link: Final = self._dict_link
        max_end = current_max_end  # rebind-ok: tracks furthest match end seen so far
        u = start_state  # rebind-ok: follows terminal dict links in O(matches)
        while u:
            s = end - own_len[u]
            if s >= guaranteed_until:
                if end > best_end.get(s, 0):
                    best_end[s] = end  # rebind-ok: record longer match for start offset
                max_end = max(max_end, end)
            u = dict_link[u]
        return max_end

    def collect_best_ends(self, text: str) -> Mapping[int, int]:
        trie: Final = self._trie
        fail: Final = self._fail
        depth: Final = self._depth
        own_len: Final = self._own_len
        dict_link: Final = self._dict_link

        best_end: Final[dict[int, int]] = {}  # mutable-ok: start-index to longest-end-index table
        guaranteed_until = 0  # rebind-ok: advances as non-overlappable matches finalize
        max_recorded_end = 0  # rebind-ok: furthest match end recorded so far
        locked_start: int | None = None  # rebind-ok: active unoverlapped leftmost match start
        state = 0  # rebind-ok: current Aho-Corasick automaton state

        for idx, ch in enumerate(text):
            state = self._step_state(trie, fail, state, ch)
            if not state:
                if locked_start is not None:
                    guaranteed_until = best_end[locked_start]
                    locked_start = None
                continue

            end = idx + 1
            trie_start = end - depth[state]
            if locked_start is not None and trie_start > locked_start:
                guaranteed_until = best_end[locked_start]
                locked_start = None
                if guaranteed_until == idx:
                    state = trie[0].get(ch, 0)
                    if not state:
                        continue
                    trie_start = idx

            u = state if own_len[state] else dict_link[state]
            if not u:
                continue

            if own_len[state] and (trie_start >= max_recorded_end or trie_start == locked_start):
                best_end[trie_start] = end
                max_recorded_end = end
                locked_start = trie_start
                if not trie[state]:
                    guaranteed_until = end
                    locked_start = None
                    state = 0
                continue

            max_recorded_end = self._record_suffix_matches(u, end, guaranteed_until, best_end, max_recorded_end)

        return best_end

    def clear(self) -> None:
        self._trie.clear()
        self._fail.clear()
        self._depth.clear()
        self._own_len.clear()
        self._dict_link.clear()


class _OutputLogSanitizer:
    def __init__(self, pairs: Sequence[tuple[str, str]]) -> None:
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
        self._mapping: dict[str, str] = mapping  # mutable-ok: cleared on logging completion zeroization
        self._matcher: _AhoCorasickMatcher | None = None

    def sanitize_text(self, text: str) -> str:
        if not text or not self._mapping:
            return text
        matcher = self._matcher  # rebind-ok: lazily built linear-time Aho-Corasick automaton
        if matcher is None:
            matcher = _AhoCorasickMatcher(self._mapping)  # rebind-ok: cache built automaton
            self._matcher = matcher
        best_end: Final = matcher.collect_best_ends(text)
        if not best_end:
            return text
        mapping: Final = self._mapping
        out: Final[list[str]] = []  # mutable-ok: linear output piece buffer
        pos = 0  # rebind-ok: tracks end of last emitted replacement
        for s in sorted(best_end):
            if s < pos:
                continue
            e = best_end[s]
            if s > pos:
                out.append(text[pos:s])
            out.append(mapping[text[s:e]])
            pos = e
        if pos < len(text):
            out.append(text[pos:])
        return "".join(out)

    def clear(self) -> None:
        self._mapping.clear()
        if self._matcher is not None:
            self._matcher.clear()
            self._matcher = None


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


def register_served_output_sanitizer(
    logging_obj: object,
    model_call_details: dict[str, object] | None,  # mutable-ok: stamps sanitizer id on Logging.model_call_details
    pairs: Sequence[tuple[str, str]],
    sanitizer_id: str,
) -> None:
    if logging_obj is None or not pairs:
        return
    sanitizer: Final = _OutputLogSanitizer(pairs)
    try:
        _LOGGING_SANITIZERS[logging_obj] = sanitizer
    except TypeError:
        return
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
    logging_obj: Final = request_data.get("litellm_logging_obj")
    if logging_obj is None:
        return rendered
    try:
        sanitizer: Final = _LOGGING_SANITIZERS.get(logging_obj)
    except TypeError:
        return rendered
    return sanitizer.sanitize_text(rendered) if sanitizer is not None else rendered


def record_served_output_texts(model_call_details: dict[str, object], texts: ServedTexts | None) -> None:
    if texts is None:
        return
    sanitizer_id: Final = model_call_details.get(_SERVED_OUTPUT_SANITIZER_KEY)
    sanitizer: Final = _DETAILS_SANITIZERS.get(sanitizer_id) if isinstance(sanitizer_id, str) else None
    recorded: Final = (
        tuple(sanitizer.sanitize_text(t) if t is not None else None for t in texts) if sanitizer is not None else texts
    )
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
