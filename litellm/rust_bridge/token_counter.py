"""Thin Python wrapper for the native Rust input token counter."""

from __future__ import annotations

from collections.abc import Awaitable
from dataclasses import dataclass
from functools import lru_cache
from typing import Final, Protocol, cast  # noqa: TID251  # native extension exposes dynamically typed callables

from pydantic import TypeAdapter

import litellm
from litellm._logging import verbose_logger
from litellm.rust_bridge.bindings import NativeBinding
from litellm.rust_bridge.configuration import rust_enabled
from litellm.rust_bridge.runtime import BridgeErrorContext, RustHandled, aattempt


class RustTokenCounter(Protocol):
    def acount_request(self, body: bytes) -> Awaitable[object]:
        raise NotImplementedError


class RustTokenCounterFactory(Protocol):
    def __call__(self, tokenizer_json: str) -> RustTokenCounter:
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


def uses_anthropic_tokenizer(model: str) -> bool:
    if litellm.disable_token_counter is True or litellm.disable_hf_tokenizer_download is True:
        return False
    return model in litellm.anthropic_models and "claude-3" not in model


@lru_cache(maxsize=4)
def _anthropic_counter(factory: RustTokenCounterFactory) -> RustTokenCounter:
    from litellm.utils import claude_json_str

    return factory(claude_json_str)


async def count_anthropic_input_tokens(body: bytes) -> InputTokenCount | None:
    if not rust_enabled():
        return None
    factory: Final = TOKEN_COUNTER.load()
    if factory is None:
        return None
    try:
        attempt: Final = await aattempt(
            native_call=lambda: _anthropic_counter(factory).acount_request(body),
            adapt=_INPUT_TOKEN_COUNT.validate_python,
            context=BridgeErrorContext(route="token_counter", provider="anthropic", model=""),
        )
    except (RuntimeError, ValueError) as error:
        verbose_logger.debug("Rust token counter failed, counting in Python: %s", error)
        return None
    return attempt.value if isinstance(attempt, RustHandled) else None
