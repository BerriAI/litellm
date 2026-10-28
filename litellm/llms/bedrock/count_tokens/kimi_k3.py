"""
Exact Bedrock token counts for Kimi K3, which has no CountTokens support.

Renders the Converse body litellm sends through Moonshot's Kimi K3 chat template and counts it with the
official tokenizer, downloaded once from Hugging Face at a pinned revision and checked against its SHA-256.
"""

import asyncio
import base64
import copy
import hashlib
import json
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import Future
from dataclasses import dataclass
from enum import Enum
from itertools import chain, groupby
from pathlib import Path
from typing import Annotated, Final, Literal, TypeAlias

import regex
import tiktoken
from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError, model_validator
from typing_extensions import Self

import litellm
from litellm._logging import verbose_logger
from litellm.litellm_core_utils.offline_mode import is_offline_mode
from litellm.litellm_core_utils.prompt_templates.factory import (
    chat_completion_tools_for,
    is_flat_responses_tool,
)
from litellm.llms.anthropic.pass_through.adapters.transformation import LiteLLMAnthropicMessagesAdapter
from litellm.types.llms.openai import AllMessageValues, ChatCompletionToolParam

KIMI_K3_REPO: Final = "moonshotai/Kimi-K3"
KIMI_K3_REVISION: Final = "f831ab66814297da540d832a5235f8e904f29d06"
KIMI_K3_RANKS_FILE: Final = "tiktoken.model"
KIMI_K3_RANKS_SHA256: Final = "b6c497a7469b33ced9c38afb1ad6e47f03f5e5dc05f15930799210ec050c5103"
_PATTERN: Final = "|".join(
    (
        r"[\p{Han}]+",
        r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
        r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
        r"\p{N}{1,3}",
        r" ?[^\s\p{L}\p{N}]+[\r\n]*",
        r"\s*[\r\n]+",
        r"\s+(?!\S)",
        r"\s+",
    )
)
# Ends chunks only where the pattern above always splits, so their counts sum to the whole text's count while BPE
# stays linear. Only a run of over 1024 characters with no such split, like 2000 spaces, is cut mid-piece.
_CHUNK: Final = regex.compile(
    r"(?s).{1,1024}(?:(?<=\S)(?= \S)|(?<=\p{N})(?=\P{N})|(?<=[^\s\p{N}])(?=\p{N})|(?<=\p{L})(?=[^\p{L}\p{M}'])"
    r"|(?<=[\r\n])(?=\S)|\Z)|.{1024}\p{N}*"
)


@dataclass(frozen=True, slots=True)
class KimiK3TokenizerSource:
    ranks_path: Callable[[], str]
    sha256: str


def _hugging_face_ranks_path() -> str:
    import huggingface_hub

    if litellm.disable_hf_tokenizer_download is True or is_offline_mode():
        cached: Final = huggingface_hub.try_to_load_from_cache(
            repo_id=KIMI_K3_REPO, filename=KIMI_K3_RANKS_FILE, revision=KIMI_K3_REVISION
        )
        if not isinstance(cached, str):
            raise FileNotFoundError(f"{KIMI_K3_RANKS_FILE} is not in the Hugging Face cache")
        return cached
    return huggingface_hub.hf_hub_download(repo_id=KIMI_K3_REPO, filename=KIMI_K3_RANKS_FILE, revision=KIMI_K3_REVISION)


HUGGING_FACE_SOURCE: Final = KimiK3TokenizerSource(ranks_path=_hugging_face_ranks_path, sha256=KIMI_K3_RANKS_SHA256)


def _loaded(source: KimiK3TokenizerSource) -> tiktoken.Encoding | str:
    try:
        data: Final = Path(source.ranks_path()).read_bytes()
    except Exception as e:  # noqa: BLE001  # a download or read failure of any kind falls back to the estimate
        return type(e).__name__
    if hashlib.sha256(data).hexdigest() != source.sha256:
        return "SHA-256 mismatch"
    rows: Final = (line.split() for line in data.splitlines() if line)
    return tiktoken.Encoding(
        name="kimi_k3",
        pat_str=_PATTERN,
        mergeable_ranks={base64.b64decode(token): int(rank) for token, rank in rows},
        special_tokens={},
    )


def load_kimi_k3_encoding(source: KimiK3TokenizerSource) -> tiktoken.Encoding | None:
    loaded: Final = _loaded(source)
    if isinstance(loaded, tiktoken.Encoding):
        return loaded
    verbose_logger.warning(
        "Kimi K3 tokenizer unavailable (%s), so Kimi K3 token counts are estimated until a later count loads it. "
        "Allow downloads of %s from Hugging Face repo %s at revision %s, or put it in the Hugging Face cache",
        loaded,
        KIMI_K3_RANKS_FILE,
        KIMI_K3_REPO,
        KIMI_K3_REVISION,
    )
    return None


def _started_load(source: KimiK3TokenizerSource) -> "Future[tiktoken.Encoding | None]":
    encoding: Final = Future[tiktoken.Encoding | None]()
    threading.Thread(
        target=lambda: encoding.set_result(load_kimi_k3_encoding(source)), name="litellm-kimi-k3-tokenizer", daemon=True
    ).start()
    return encoding


@dataclass(frozen=True, slots=True)
class _EncodingLoad:
    encoding: "Future[tiktoken.Encoding | None]"
    started_at: float
    is_retry: bool

    def retry_due(self, now: float, retry_after_seconds: float) -> bool:
        return self.encoding.done() and self.encoding.result() is None and now - self.started_at >= retry_after_seconds


@dataclass(slots=True)
class KimiK3Encodings:
    source: KimiK3TokenizerSource
    retry_after_seconds: float
    clock: Callable[[], float] = time.monotonic
    current: _EncodingLoad | None = None

    def _load(self) -> _EncodingLoad:
        now: Final = self.clock()
        current: Final = self.current
        if current is not None and not current.retry_due(now, self.retry_after_seconds):
            return current
        load: Final = _EncodingLoad(encoding=_started_load(self.source), started_at=now, is_retry=current is not None)
        self.current = load
        return load

    async def encoding(self, wait_seconds: float) -> tiktoken.Encoding | None:
        load: Final = self._load()
        remaining: Final = 0 if load.is_retry else load.started_at + wait_seconds - self.clock()
        if load.encoding.done() or remaining <= 0:
            return load.encoding.result() if load.encoding.done() else None
        await asyncio.wait((asyncio.wrap_future(load.encoding),), timeout=remaining)
        if load.encoding.done():
            return load.encoding.result()
        verbose_logger.warning(
            "Kimi K3 tokenizer still loading after %s s, so Kimi K3 token counts are estimated until it loads. "
            "Check that %s can be downloaded from Hugging Face repo %s at revision %s, or put it in the Hugging Face cache",
            wait_seconds,
            KIMI_K3_RANKS_FILE,
            KIMI_K3_REPO,
            KIMI_K3_REVISION,
        )
        return None


class _Converse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class _Text(_Converse):
    text: str


class _CachePoint(_Converse):
    cache_point: JsonValue = Field(alias="cachePoint")


class _ToolUse(_Converse):
    tool_use_id: str = Field(alias="toolUseId")
    name: str
    arguments: Mapping[str, JsonValue] = Field(alias="input")


class _ToolUseBlock(_Converse):
    tool_use: _ToolUse = Field(alias="toolUse")


class _ToolResult(_Converse):
    tool_use_id: str = Field(alias="toolUseId")
    content: tuple[_Text, ...]
    status: str | None = None


class _ToolResultBlock(_Converse):
    tool_result: _ToolResult = Field(alias="toolResult")


class _ReasoningText(_Converse):
    text: str
    signature: str | None = None


class _Reasoning(_Converse):
    reasoning_text: _ReasoningText = Field(alias="reasoningText")


class _ReasoningBlock(_Converse):
    reasoning_content: _Reasoning = Field(alias="reasoningContent")


class _UserMessage(_Converse):
    role: Literal["user"]
    content: tuple[_Text | _ToolResultBlock | _CachePoint, ...]


class _AssistantMessage(_Converse):
    role: Literal["assistant"]
    content: tuple[_Text | _ToolUseBlock | _ReasoningBlock | _CachePoint, ...]

    def calls(self) -> tuple[_ToolUse, ...]:
        return tuple(block.tool_use for block in self.content if isinstance(block, _ToolUseBlock))


class _InputSchema(_Converse):
    definition: JsonValue = Field(alias="json")


class _ToolSpec(_Converse):
    name: str
    description: str | None = None
    input_schema: _InputSchema = Field(alias="inputSchema")

    def declared(self) -> Mapping[str, JsonValue]:
        described: Final[Mapping[str, JsonValue]] = (
            {} if self.description is None else {"description": self.description}
        )
        return {"name": self.name, **described, "parameters": self.input_schema.definition}


class _ToolSpecBlock(_Converse):
    tool_spec: _ToolSpec = Field(alias="toolSpec")


class _ToolConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")
    tools: tuple[_ToolSpecBlock | _CachePoint, ...]


_Message: TypeAlias = Annotated[_UserMessage | _AssistantMessage, Field(discriminator="role")]


def _result_ids(message: _Message) -> frozenset[str]:
    if isinstance(message, _AssistantMessage):
        return frozenset()
    return frozenset(block.tool_result.tool_use_id for block in message.content if isinstance(block, _ToolResultBlock))


def _call_ids(message: _Message | None) -> frozenset[str]:
    if not isinstance(message, _AssistantMessage):
        return frozenset()
    return frozenset(call.tool_use_id for call in message.calls())


class _ConverseRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")
    messages: tuple[_Message, ...]
    system: tuple[_Text | _CachePoint, ...] = ()
    tool_config: _ToolConfig | None = Field(default=None, alias="toolConfig")

    @model_validator(mode="after")
    def _results_answer_the_previous_calls(self) -> Self:
        pairs: Final = zip((None, *self.messages), self.messages, strict=False)
        if not all(_result_ids(message) <= _call_ids(previous) for previous, message in pairs):
            raise ValueError("a toolResult does not answer a toolUse of the previous assistant message")
        return self


class _Control(Enum):
    OPEN = "<|open|>"
    CLOSE = "<|close|>"
    SEP = "<|sep|>"
    END_OF_MSG = "<|end_of_msg|>"


_Segment: TypeAlias = str | _Control
_Segments: TypeAlias = tuple[_Segment, ...]


def _open(tag: str, attrs: Iterable[tuple[str, str]] = ()) -> _Segments:
    escaped: Final = (
        (f" {key}", '="', value.replace("&", "&amp;").replace('"', "&quot;"), '"') for key, value in attrs
    )
    return (_Control.OPEN, tag, *chain.from_iterable(escaped), _Control.SEP)


def _close(tag: str) -> _Segments:
    return (_Control.CLOSE, tag, _Control.SEP)


def _message(attrs: Iterable[tuple[str, str]], body: Iterable[_Segment]) -> _Segments:
    return (*_open("message", attrs), *body, *_close("message"), _Control.END_OF_MSG)


def _tool_declare(specs: Sequence[_ToolSpec]) -> _Segments:
    tools: Final = json.dumps(
        [{"type": "function", "function": spec.declared()} for spec in specs],
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return _message(
        (("role", "system"), ("type", "tool-declare")),
        (f"# Tools\nHere are the available tools, described in JSONSchema.\n\n```json\n{tools}\n```",),
    )


def _text(blocks: Iterable[_Converse]) -> str:
    return "".join(block.text for block in blocks if isinstance(block, _Text))


def _json_type(value: JsonValue) -> str:
    match value:
        case bool():
            return "boolean"
        case None:
            return "null"
        case int() | float():
            return "number"
        case str():
            return "string"
        case dict():
            return "object"
        case _:
            return "array"


def _argument(key: str, value: JsonValue) -> _Segments:
    text: Final = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return (*_open("argument", (("key", key), ("type", _json_type(value)))), text, *_close("argument"))


def _call(index: int, call: _ToolUse) -> _Segments:
    arguments: Final = chain.from_iterable(_argument(key, value) for key, value in call.arguments.items())
    return (*_open("call", (("tool", call.name), ("index", str(index)))), *arguments, *_close("call"))


def _tool_calls(calls: Sequence[_ToolUse]) -> _Segments:
    if not calls:
        return ()
    rendered: Final = chain.from_iterable(_call(index, call) for index, call in enumerate(calls, start=1))
    return (*_open("tools"), *rendered, *_close("tools"))


def _assistant_message(message: _AssistantMessage) -> _Segments:
    reasoning: Final = "".join(
        block.reasoning_content.reasoning_text.text for block in message.content if isinstance(block, _ReasoningBlock)
    )
    return _message(
        (("role", "assistant"),),
        (
            *_open("think"),
            reasoning,
            *_close("think"),
            *_open("response"),
            _text(message.content),
            *_close("response"),
            *_tool_calls(message.calls()),
        ),
    )


def _tool_messages(results: Sequence[_ToolResult], previous: _AssistantMessage) -> _Segments:
    names: Final = {call.tool_use_id: call.name for call in previous.calls()}
    return tuple(
        chain.from_iterable(
            _message(
                (("role", "tool"), ("tool", names[result.tool_use_id]), ("index", str(index))),
                (_text(result.content),),
            )
            for index, result in enumerate(results, start=1)
        )
    )


def _user_run(blocks: Sequence[_Text | _ToolResultBlock], previous: _Message | None) -> _Segments:
    results: Final = tuple(block.tool_result for block in blocks if isinstance(block, _ToolResultBlock))
    if results and isinstance(previous, _AssistantMessage):
        return _tool_messages(results, previous)
    return _message((("role", "user"),), (_text(blocks),))


def _user_messages(message: _UserMessage, previous: _Message | None) -> _Segments:
    blocks: Final = tuple(block for block in message.content if not isinstance(block, _CachePoint))
    runs: Final = groupby(blocks, key=lambda block: isinstance(block, _ToolResultBlock))
    return tuple(chain.from_iterable(_user_run(tuple(run), previous) for _, run in runs))


def _turn(previous: _Message | None, message: _Message) -> _Segments:
    if isinstance(message, _AssistantMessage):
        return _assistant_message(message)
    return _user_messages(message, previous)


def _rendered(request: _ConverseRequest) -> _Segments:
    specs: Final = (
        ()
        if request.tool_config is None
        else tuple(block.tool_spec for block in request.tool_config.tools if isinstance(block, _ToolSpecBlock))
    )
    system: Final = _text(request.system)
    turns: Final = chain.from_iterable(
        _turn(previous, message) for previous, message in zip((None, *request.messages), request.messages, strict=False)
    )
    return (
        *(_tool_declare(specs) if specs else ()),
        *(_message((("role", "system"),), (system,)) if system else ()),
        *turns,
        *_open("message", (("role", "assistant"),)),
        *_open("think"),
    )


def _text_tokens(encoding: tiktoken.Encoding, text: str) -> int:
    spans: Final = (match.span() for match in _CHUNK.finditer(text))
    return sum(len(encoding.encode_ordinary(text[start:end])) for start, end in spans)


class _TextPart(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")
    type: Literal["text"]


class _TextOnlyMessage(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")
    content: str | tuple[_TextPart, ...] | None = None


def _text_only(message: AllMessageValues) -> bool:
    try:
        _TextOnlyMessage.model_validate(message)
    except ValidationError:
        return False
    return True


def _as_converse_input(tool: ChatCompletionToolParam) -> tuple[Mapping[str, object], ...]:
    return chat_completion_tools_for(tool) if is_flat_responses_tool(tool) else (tool,)


def _converse_request(
    model: str, messages: Sequence[AllMessageValues], tools: Sequence[ChatCompletionToolParam] | None
) -> _ConverseRequest | None:
    if not all(_text_only(message) for message in messages):
        return None
    try:
        chat_tools: Final = tuple(
            chat_tool for tool in tools or () for chat_tool in _as_converse_input(tool)
        )  # comprehension-ok: flattens each tool's translation
        return _ConverseRequest.model_validate(
            litellm.AmazonConverseConfig().transform_request(
                model=model,
                messages=copy.deepcopy(list(messages)),
                optional_params={"tools": copy.deepcopy(list(chat_tools))} if chat_tools else {},
                litellm_params={},
                headers={},
            )
        )
    except Exception:  # noqa: BLE001  # a body litellm cannot send to Bedrock, or cannot render, falls back to the estimate
        return None


def kimi_k3_tokens(
    encoding: tiktoken.Encoding,
    model: str,
    messages: Sequence[AllMessageValues],
    tools: Sequence[ChatCompletionToolParam] | None,
) -> int | None:
    request: Final = _converse_request(model, messages, tools)
    if request is None:
        return None
    return sum(
        1 if isinstance(segment, _Control) else _text_tokens(encoding, segment) for segment in _rendered(request)
    )


def kimi_k3_anthropic_tokens(encoding: tiktoken.Encoding, model: str, fields: Mapping[str, object]) -> int | None:
    body: Final = {**fields, "model": model, "max_tokens": 1}
    try:
        translated, _ = LiteLLMAnthropicMessagesAdapter().translate_anthropic_to_openai(
            copy.deepcopy(body),  # pyright: ignore[reportArgumentType]  # raw /v1/messages fields
            custom_llm_provider="bedrock",
        )
    except Exception:  # noqa: BLE001  # a request the adapter cannot translate falls back to the estimate
        return None
    match translated:
        case {"tools": tools}:
            return kimi_k3_tokens(encoding, model, translated["messages"], tools)
        case _:
            return kimi_k3_tokens(encoding, model, translated["messages"], None)


def kimi_k3_template_tokens(message_count: int, has_tools: bool) -> int:
    return 14 + 13 * max(message_count - 1, 0) + (40 if has_tools else 0)
