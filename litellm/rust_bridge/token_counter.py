"""Thin Python wrapper for the native Rust input token counter."""

from __future__ import annotations

from collections.abc import Awaitable
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Final, Literal, Protocol, cast  # noqa: TID251  # PyO3 binding validation

from pydantic import TypeAdapter
from typing_extensions import assert_never

import litellm
from litellm.litellm_core_utils.token_counter import openai_tokenizer_encoding_name, uses_legacy_message_accounting
from litellm.rust_bridge import tokenizer as tokenizer_dispatch
from litellm.rust_bridge.bindings import NativeBinding
from litellm.utils import huggingface_tokenizer_kind

if TYPE_CHECKING:
    from litellm.rust_bridge._native import Tokenizer as NativeTokenizer

RustTokenizer = Literal["anthropic", "cl100k_base", "o200k_base"]


class RustTokenCounter(Protocol):
    def acount_request(self, body: bytes) -> Awaitable[object]:
        raise NotImplementedError


class RustTokenCounterFactory(Protocol):
    def from_tokenizer(self, tokenizer: NativeTokenizer, fast: bool = False) -> RustTokenCounter:
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
        if callable(getattr(value, "from_tokenizer", None))
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
    match openai_tokenizer_encoding_name(model):
        case "cl100k_base":
            return "cl100k_base"
        case "o200k_base":
            return "o200k_base"
        case _:
            return None


@lru_cache(maxsize=4)
def _counter(factory: RustTokenCounterFactory, tokenizer: RustTokenizer) -> RustTokenCounter:
    return factory.from_tokenizer(_native_tokenizer(tokenizer))


def _native_tokenizer(tokenizer: RustTokenizer) -> NativeTokenizer:
    match tokenizer:
        case "anthropic":
            native = tokenizer_dispatch.native_anthropic()
        case "cl100k_base" | "o200k_base":
            native = tokenizer_dispatch.native_encoding(tokenizer)
        case _:
            assert_never(tokenizer)
    if native is None:
        raise RuntimeError(f"native {tokenizer} tokenizer is unavailable")
    return native


async def native_count(factory: RustTokenCounterFactory, tokenizer: RustTokenizer, body: bytes) -> InputTokenCount:
    """One native count, validated into the public shape.

    ``RustBridgeDeclined`` and upstream errors propagate so the caller's route
    runner can map them onto its fallback policy; other failures (RuntimeError,
    ValueError) propagate as-is."""
    return _INPUT_TOKEN_COUNT.validate_python(await _counter(factory, tokenizer).acount_request(body))
