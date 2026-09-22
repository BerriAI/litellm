from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Final, cast  # noqa: TID251  # native class is validated at the binding boundary

import tiktoken
from tokenizers import Tokenizer as PythonHuggingFaceTokenizer

from litellm.litellm_core_utils.tokenizer import Encoding, HuggingFace, HuggingFaceTokenizer, OpenAIEncoding
from litellm.rust_bridge import runtime
from litellm.rust_bridge.bindings import NativeBinding
from litellm.rust_bridge.catalog import Context, Route

if TYPE_CHECKING:
    from litellm.rust_bridge._native import Tokenizer as NativeTokenizer


def _as_factory(value: object) -> type[NativeTokenizer] | None:
    return (
        cast(type["NativeTokenizer"], value)  # cast-ok: PyO3 class validated at the native boundary
        if isinstance(value, type)
        else None
    )


TOKENIZER: Final = NativeBinding("Tokenizer", validate=_as_factory)


@lru_cache(maxsize=8)
def _native_encoding(factory: type[NativeTokenizer], name: str) -> OpenAIEncoding:
    native: Final = factory.from_tiktoken(name)
    return OpenAIEncoding(name, native, frozenset(native.special_tokens()))


def _python_encoding(name: str) -> tiktoken.Encoding:
    from litellm.litellm_core_utils.default_encoding import encoding

    return encoding if name == encoding.name else tiktoken.get_encoding(name)


def get_encoding(name: str) -> Encoding:
    return runtime.run(
        Context(Route.TOKENIZER, provider="tiktoken"),
        binding=TOKENIZER,
        native=lambda factory: _native_encoding(factory, name),
        python=lambda: _python_encoding(name),
    )


def from_str(json: str) -> HuggingFace:
    return runtime.run(
        Context(Route.TOKENIZER, provider="huggingface"),
        binding=TOKENIZER,
        native=lambda factory: HuggingFaceTokenizer(factory.from_json(json)),
        python=lambda: PythonHuggingFaceTokenizer.from_str(json),
    )


def from_pretrained(identifier: str, revision: str = "main", token: str | None = None) -> HuggingFace:
    return runtime.run(
        Context(Route.TOKENIZER, provider="huggingface"),
        binding=TOKENIZER,
        native=lambda factory: HuggingFaceTokenizer(
            factory.from_pretrained(identifier, revision=revision, token=token)
        ),
        python=lambda: PythonHuggingFaceTokenizer.from_pretrained(identifier, revision=revision, token=token),
    )
