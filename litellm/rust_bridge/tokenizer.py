from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Final, cast  # noqa: TID251  # native class is validated at the binding boundary

import tiktoken
from tokenizers import Tokenizer as PythonHuggingFaceTokenizer

from litellm.litellm_core_utils.tokenizer import Encoding, HuggingFace, HuggingFaceTokenizer, OpenAIEncoding
from litellm.rust_bridge import runtime
from litellm.rust_bridge.bindings import NativeBinding
from litellm.rust_bridge.catalog import Route, RouteContext

if TYPE_CHECKING:
    from litellm.rust_bridge._native import Tokenizer as NativeTokenizer


def _as_factory(value: object) -> type[NativeTokenizer] | None:
    return (
        cast(type["NativeTokenizer"], value)  # cast-ok: PyO3 class validated at the native boundary
        if isinstance(value, type)
        else None
    )


TOKENIZER: Final = NativeBinding("Tokenizer", validate=_as_factory)

# The catalog contexts the tokenizer factories dispatch on. Callers that cache a tokenizer per
# backend key their cache on `decision(...)` of the same context, so key and dispatch agree.
TIKTOKEN_CONTEXT: Final = RouteContext(Route.TOKENIZER, provider="tiktoken")
HUGGINGFACE_CONTEXT: Final = RouteContext(Route.TOKENIZER, provider="huggingface")


@lru_cache(maxsize=8)
def _native_tiktoken(factory: type[NativeTokenizer], name: str) -> NativeTokenizer:
    return factory.from_tiktoken(name)


@lru_cache(maxsize=1)
def _native_anthropic(factory: type[NativeTokenizer]) -> NativeTokenizer:
    from litellm.utils import claude_json_str

    return factory.from_json(claude_json_str)


@lru_cache(maxsize=8)
def _native_encoding(factory: type[NativeTokenizer], name: str) -> OpenAIEncoding:
    return OpenAIEncoding.wrap(_native_tiktoken(factory, name))


def native_encoding(name: str) -> NativeTokenizer | None:
    """The native tiktoken encoding behind `get_encoding(name)`, for a Rust route that counts
    with the same loaded model; `None` without the extension."""
    factory: Final = TOKENIZER.load()
    return None if factory is None else _native_tiktoken(factory, name)


def native_anthropic() -> NativeTokenizer | None:
    """The native packaged Anthropic tokenizer behind `anthropic()`, parsed once per process."""
    factory: Final = TOKENIZER.load()
    return None if factory is None else _native_anthropic(factory)


def _python_encoding(name: str) -> tiktoken.Encoding:
    from litellm.litellm_core_utils.default_encoding import encoding

    return encoding if name == encoding.name else tiktoken.get_encoding(name)


def get_encoding(name: str) -> Encoding:
    return runtime.run(
        TIKTOKEN_CONTEXT,
        binding=TOKENIZER,
        native=lambda factory: _native_encoding(factory, name),
        python=lambda: _python_encoding(name),
    )


def anthropic() -> HuggingFace:
    """The packaged Anthropic tokenizer on the selected backend."""
    from litellm.utils import claude_json_str

    return runtime.run(
        HUGGINGFACE_CONTEXT,
        binding=TOKENIZER,
        native=lambda factory: HuggingFaceTokenizer(_native_anthropic(factory)),
        python=lambda: PythonHuggingFaceTokenizer.from_str(claude_json_str),
    )


def from_str(json: str) -> HuggingFace:
    return runtime.run(
        HUGGINGFACE_CONTEXT,
        binding=TOKENIZER,
        native=lambda factory: HuggingFaceTokenizer(factory.from_json(json)),
        python=lambda: PythonHuggingFaceTokenizer.from_str(json),
    )


def from_pretrained(identifier: str, revision: str = "main", token: str | None = None) -> HuggingFace:
    return runtime.run(
        HUGGINGFACE_CONTEXT,
        binding=TOKENIZER,
        native=lambda factory: HuggingFaceTokenizer(
            factory.from_pretrained(identifier, revision=revision, token=token)
        ),
        python=lambda: PythonHuggingFaceTokenizer.from_pretrained(identifier, revision=revision, token=token),
    )
