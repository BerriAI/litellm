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

# The catalog contexts the tokenizer factories dispatch on. Callers that cache a tokenizer per
# backend key their cache on `decision(...)` of the same context, so key and dispatch agree.
TIKTOKEN_CONTEXT: Final = Context(Route.TOKENIZER, provider="tiktoken")
HUGGINGFACE_CONTEXT: Final = Context(Route.TOKENIZER, provider="huggingface")


@lru_cache(maxsize=8)
def _native_encoding(factory: type[NativeTokenizer], name: str) -> OpenAIEncoding:
    return OpenAIEncoding.wrap(factory.from_tiktoken(name))


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
