from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

from litellm.litellm_core_utils.token_counter import (
    MessageCountFraming,
    message_count_framing,
    openai_tokenizer_encoding,
    openai_tokenizer_encoding_name,
)
from litellm.rust_bridge.tokenizer import get_encoding
from litellm.types.utils import SelectTokenizerResponse

STAND_IN_ENCODING: Final = "o200k_base"
STAND_IN_OVERCOUNT_MARGIN: Final = 2


@dataclass(frozen=True, slots=True)
class ContextWindow:
    openai_model: str | None
    max_input_tokens: int


@dataclass(frozen=True, slots=True)
class ContextWindowOverflow:
    max_input_tokens: int
    tokens: int
    certain: bool


@dataclass(frozen=True, slots=True)
class TokenCountKey:
    encoding: str
    framing: MessageCountFraming


def stand_in_tokenizer() -> SelectTokenizerResponse:
    return {"type": "openai_tokenizer", "tokenizer": get_encoding(STAND_IN_ENCODING)}


def pre_call_tokenizer(openai_model: str | None) -> SelectTokenizerResponse:
    if openai_model is None:
        return stand_in_tokenizer()
    return {"type": "openai_tokenizer", "tokenizer": openai_tokenizer_encoding(openai_model)}


def token_count_key(openai_model: str | None) -> TokenCountKey:
    return TokenCountKey(
        encoding=STAND_IN_ENCODING if openai_model is None else openai_tokenizer_encoding_name(openai_model),
        framing=message_count_framing(openai_model or ""),
    )


def context_window_overflow(
    window: ContextWindow | None, token_counts: Mapping[str | None, int] | None
) -> ContextWindowOverflow | None:
    """How a prompt overflows a deployment's window, or None when it fits or was not counted. A stand-in count is
    only an estimate, so it proves an overflow only past STAND_IN_OVERCOUNT_MARGIN times the window."""
    tokens: Final = token_counts.get(window.openai_model) if window is not None and token_counts is not None else None
    if window is None or tokens is None or tokens <= window.max_input_tokens:
        return None
    return ContextWindowOverflow(
        max_input_tokens=window.max_input_tokens,
        tokens=tokens,
        certain=window.openai_model is not None or tokens > window.max_input_tokens * STAND_IN_OVERCOUNT_MARGIN,
    )
