from __future__ import annotations

from typing import Final, cast  # noqa: TID251  # native extension exposes untyped callables

from pydantic import TypeAdapter

import litellm
from litellm.litellm_core_utils.default_encoding import cl100k_base_rank_file, o200k_base_rank_file
from litellm.litellm_core_utils.token_counter import openai_tokenizer_encoding, uses_legacy_message_accounting
from litellm.rust_bridge.runtime import BridgeErrorContext, ainvoke
from litellm.rust_bridge.token_counter.definition import COMPONENT
from litellm.rust_bridge.token_counter.types import InputTokenCount, RustTokenCounter, RustTokenizer
from litellm.utils import claude_json_str, huggingface_tokenizer_kind

_INPUT_TOKEN_COUNT: Final = TypeAdapter(InputTokenCount)


def _as_counter(value: object) -> RustTokenCounter | None:
    return cast(RustTokenCounter, value) if callable(value) else None


TOKEN_COUNTER: Final = COMPONENT.bind("count_input_tokens", validate=_as_counter)


def rust_tokenizer(model: str) -> RustTokenizer | None:
    execution: Final = COMPONENT.resolve()
    if execution.select(TOKEN_COUNTER) is None:
        return None
    kind: Final = None if litellm.disable_hf_tokenizer_download is True else huggingface_tokenizer_kind(model)
    encoding: Final = openai_tokenizer_encoding(model).name if kind is None else ""
    return RustTokenizer(
        kind=kind,
        encoding=encoding,
        disabled=litellm.disable_token_counter is True,
        legacy_accounting=uses_legacy_message_accounting(model),
    )


def _tokenizer_resource(tokenizer: str) -> str:
    match tokenizer:
        case "anthropic":
            return claude_json_str
        case "cl100k_base":
            return cl100k_base_rank_file()
        case "o200k_base":
            return o200k_base_rank_file()
        case _:
            raise ValueError(f"unsupported Rust tokenizer resource: {tokenizer}")


async def count_input_tokens(body: bytes, tokenizer: RustTokenizer) -> InputTokenCount | None:
    execution: Final = COMPONENT.resolve()
    counter: Final = execution.select(TOKEN_COUNTER)

    async def python_fallback() -> None:
        return None

    return await ainvoke(
        execution=execution,
        native_call=(
            lambda: counter(
                body,
                tokenizer.kind,
                tokenizer.encoding,
                tokenizer.disabled,
                tokenizer.legacy_accounting,
                _tokenizer_resource,
            )
        )
        if counter is not None
        else None,
        python_fallback=python_fallback,
        adapt=_INPUT_TOKEN_COUNT.validate_python,
        context=BridgeErrorContext(route=COMPONENT.name.value, provider="", model=""),
    )
