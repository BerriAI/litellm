"""Thin Python wrapper for the native Rust token counter bridge.

The Rust core owns the tiktoken encoding resolution and token counting for the
subset of inputs it supports (tiktoken-based text and message counting). This
module loads the native function and provides a fallback when the bridge is
unavailable or the input contains unsupported content (images, HF tokenizers).
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Protocol

from litellm._logging import verbose_logger
from litellm.rust_bridge.loader import get_native_bridge

_TRUTHY_ENV_VALUES: Final = frozenset({"1", "true", "yes", "on"})

_UNSUPPORTED_CONTENT_TYPES: Final = frozenset(
    {"image_url", "tool_use", "tool_result", "thinking", "tool_reference"}
)


class RustTokenCounter(Protocol):
    def __call__(
        self,
        model: str,
        text: str | None = None,
        messages: Sequence[Mapping[str, object]] | None = None,
        tools: Sequence[Mapping[str, object]] | None = None,
        tool_choice: object | None = None,
        count_response_tokens: bool = False,
        default_token_count: int | None = None,
    ) -> int:
        raise NotImplementedError


@dataclass(slots=True)
class _RustTokenCounterState:
    token_counter: RustTokenCounter | None = None


_STATE: Final[_RustTokenCounterState] = _RustTokenCounterState()


def set_rust_token_counter(token_counter: RustTokenCounter | None) -> None:
    """Inject the native callable, so tests can supply a double."""
    _STATE.token_counter = token_counter


def _env_enables_rust_token_counter() -> bool:
    return (
        os.getenv("LITELLM_RUST_TOKEN_COUNTER", "").strip().lower()
        in _TRUTHY_ENV_VALUES
    )


def load_rust_token_counter() -> RustTokenCounter | None:
    """Return the native token counter function, or None when unavailable."""
    if _STATE.token_counter is not None:
        return _STATE.token_counter

    native_bridge = get_native_bridge()
    if native_bridge is None:
        return None

    func = getattr(native_bridge, "token_counter", None)
    if func is not None:
        _STATE.token_counter = func
    return func


def _messages_contain_unsupported_content(
    messages: Sequence[Mapping[str, object]],
) -> bool:
    """Scan messages for content blocks the Rust path cannot handle.

    Image and Anthropic-specific blocks trigger Unsupported in Rust. When a
    default_token_count is set the Rust path returns it as fallback, so we let
    those through. This check is only used to decide whether to skip Rust
    entirely when no default is available.
    """
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, Mapping):
                    block_type = block.get("type")
                    if isinstance(block_type, str) and block_type in _UNSUPPORTED_CONTENT_TYPES:
                        return True
    return False


def try_rust_token_counter(
    *,
    model: str,
    text: str | None = None,
    messages: Sequence[Mapping[str, object]] | None = None,
    tools: Sequence[Mapping[str, object]] | None = None,
    tool_choice: object | None = None,
    count_response_tokens: bool = False,
    default_token_count: int | None = None,
) -> int | None:
    """Attempt to count tokens using the Rust bridge.

    Returns the token count on success, or None if the Rust path is unavailable
    or the input contains unsupported content without a default fallback.
    """
    if not _env_enables_rust_token_counter():
        return None

    if (
        default_token_count is None
        and messages is not None
        and _messages_contain_unsupported_content(messages)
    ):
        return None

    rust_func = load_rust_token_counter()
    if rust_func is None:
        return None

    try:
        return rust_func(
            model=model,
            text=text,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            count_response_tokens=count_response_tokens,
            default_token_count=default_token_count,
        )
    except Exception:
        verbose_logger.debug(
            "Rust token counter failed, falling back to Python", exc_info=True
        )
        return None
