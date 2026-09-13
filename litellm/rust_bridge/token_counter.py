"""Thin Python wrapper for the native Rust input token counter."""

from __future__ import annotations

from collections.abc import Awaitable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Final, Literal, Protocol, cast  # noqa: TID251  # native extension exposes untyped callables

import orjson
from pydantic import TypeAdapter

import litellm
from litellm._logging import verbose_logger
from litellm.litellm_core_utils.default_encoding import cl100k_base_rank_file, o200k_base_rank_file
from litellm.litellm_core_utils.token_counter import openai_tokenizer_encoding, uses_legacy_message_accounting
from litellm.rust_bridge.bindings import NativeBinding
from litellm.rust_bridge.configuration import rust_enabled
from litellm.rust_bridge.runtime import BridgeErrorContext, RustHandled, aattempt
from litellm.utils import claude_json_str, huggingface_tokenizer_kind

RustTokenizer = Literal["anthropic", "cl100k_base", "o200k_base"]


class RustTokenCounter(Protocol):
    def acount_request(self, body: bytes) -> Awaitable[object]:
        raise NotImplementedError


class RustTokenCounterFactory(Protocol):
    def __call__(self, tokenizer_json: str) -> RustTokenCounter:
        raise NotImplementedError

    def from_cl100k_ranks(self, rank_file: str) -> RustTokenCounter:
        raise NotImplementedError

    def from_o200k_ranks(self, rank_file: str) -> RustTokenCounter:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class InputTokenCount:
    model: str | None
    input_tokens: int


_INPUT_TOKEN_COUNT: Final = TypeAdapter(InputTokenCount)


def _as_factory(value: object) -> RustTokenCounterFactory | None:
    return (
        cast(  # cast-ok: native extension protocol is runtime-defined
            RustTokenCounterFactory, value
        )
        if callable(value)
        else None
    )


TOKEN_COUNTER: Final = NativeBinding("TokenCounter", validate=_as_factory)


def rust_tokenizer(model: str) -> RustTokenizer | None:
    """The Rust counter for the tokenizer `litellm.token_counter` selects for `model`, `None` when Python must count.

    Mirrors `_select_tokenizer_helper`: the Anthropic tokenizer has a Rust port, the other HuggingFace
    downloads do not, and of the tiktoken encodings `cl100k_base` and `o200k_base` do (p50k/r50k do not). Rust
    prices every message with the default constants, so the legacy `gpt-3.5-turbo-0301` accounting stays in
    Python."""
    if litellm.disable_token_counter is True:
        return None
    kind: Final = None if litellm.disable_hf_tokenizer_download is True else huggingface_tokenizer_kind(model)
    if kind == "anthropic":
        return "anthropic"
    if kind is not None or uses_legacy_message_accounting(model):
        return None
    match openai_tokenizer_encoding(model).name:
        case "cl100k_base":
            return "cl100k_base"
        case "o200k_base":
            return "o200k_base"
        case _:
            return None


@lru_cache(maxsize=4)
def _counter(factory: RustTokenCounterFactory, tokenizer: RustTokenizer) -> RustTokenCounter:
    match tokenizer:
        case "anthropic":
            return factory(claude_json_str)
        case "cl100k_base":
            return factory.from_cl100k_ranks(cl100k_base_rank_file())
        case "o200k_base":
            return factory.from_o200k_ranks(o200k_base_rank_file())


async def count_input_tokens(body: bytes, tokenizer: RustTokenizer) -> InputTokenCount | None:
    if not rust_enabled():
        return None
    factory: Final = TOKEN_COUNTER.load()
    if factory is None:
        return None
    try:
        attempt: Final = await aattempt(
            native_call=lambda: _counter(factory, tokenizer).acount_request(body),
            adapt=_INPUT_TOKEN_COUNT.validate_python,
            context=BridgeErrorContext(route="token_counter", provider=tokenizer, model=""),
        )
    except (RuntimeError, ValueError) as error:
        verbose_logger.debug("Rust token counter (%s) failed, counting in Python: %s", tokenizer, error)
        return None
    if not isinstance(attempt, RustHandled):
        return None
    verbose_logger.debug("Rust token counter (%s) counted %d input tokens", tokenizer, attempt.value.input_tokens)
    return attempt.value


async def count_chat_input_tokens(
    model: str,
    messages: Sequence[object],
    tools: object = None,
    tool_choice: object = None,
) -> int | None:
    """Rust count of an already-parsed chat body, `None` when Python must count.

    Callers holding Python messages instead of the raw request bytes pay one `orjson.dumps` of the counted fields.
    Pydantic messages and other non-JSON objects are left to Python, as are the blocks Rust declines."""
    if not rust_enabled() or TOKEN_COUNTER.load() is None:
        return None
    tokenizer: Final = rust_tokenizer(model)
    if tokenizer is None:
        return None
    try:
        body: Final = orjson.dumps({"model": model, "messages": messages, "tools": tools, "tool_choice": tool_choice})
    except orjson.JSONEncodeError as error:
        verbose_logger.debug("Rust token counter (%s) skipped an unserializable body: %s", tokenizer, error)
        return None
    count: Final = await count_input_tokens(body, tokenizer)
    return None if count is None else count.input_tokens
