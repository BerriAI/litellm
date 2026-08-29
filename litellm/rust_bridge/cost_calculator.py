"""Thin Python wrapper for the native Rust cost calculator bridge.

The Rust core owns the pricing lookup and cost arithmetic. This module
loads the native function and provides a fallback when the bridge is
unavailable or the model has no pricing entry.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, Protocol

from litellm._logging import verbose_logger
from litellm.rust_bridge.loader import get_native_bridge

_TRUTHY_ENV_VALUES: Final = frozenset({"1", "true", "yes", "on"})


class RustCompletionCost(Protocol):
    def __call__(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        custom_llm_provider: str | None = None,
        service_tier: str | None = None,
        cache_hit_tokens: int | None = None,
        cache_creation_tokens: int | None = None,
        audio_input_tokens: int | None = None,
        audio_output_tokens: int | None = None,
        reasoning_tokens: int | None = None,
    ) -> float:
        raise NotImplementedError


@dataclass(slots=True)
class _RustCompletionCostState:
    completion_cost: RustCompletionCost | None = None


_STATE: Final[_RustCompletionCostState] = _RustCompletionCostState()


def set_rust_completion_cost(completion_cost: RustCompletionCost | None) -> None:
    """Inject the native callable, so tests can supply a double."""
    _STATE.completion_cost = completion_cost


def _env_enables_rust_cost_calculator() -> bool:
    return (
        os.getenv("LITELLM_RUST_COST_CALCULATOR", "").strip().lower()
        in _TRUTHY_ENV_VALUES
    )


def load_rust_completion_cost() -> RustCompletionCost | None:
    """Return the native cost calculator function, or None when unavailable."""
    if _STATE.completion_cost is not None:
        return _STATE.completion_cost

    native_bridge = get_native_bridge()
    if native_bridge is None:
        return None

    func = getattr(native_bridge, "completion_cost", None)
    if func is not None:
        _STATE.completion_cost = func
    return func


def try_rust_completion_cost(
    *,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    custom_llm_provider: str | None = None,
    service_tier: str | None = None,
    cache_hit_tokens: int | None = None,
    cache_creation_tokens: int | None = None,
    audio_input_tokens: int | None = None,
    audio_output_tokens: int | None = None,
    reasoning_tokens: int | None = None,
) -> float | None:
    """Attempt to calculate cost using the Rust bridge.

    Returns the total cost in USD on success, or None if the Rust path is
    unavailable or the model has no pricing entry.
    """
    if not _env_enables_rust_cost_calculator():
        return None

    rust_func = load_rust_completion_cost()
    if rust_func is None:
        return None

    try:
        return rust_func(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            custom_llm_provider=custom_llm_provider,
            service_tier=service_tier,
            cache_hit_tokens=cache_hit_tokens,
            cache_creation_tokens=cache_creation_tokens,
            audio_input_tokens=audio_input_tokens,
            audio_output_tokens=audio_output_tokens,
            reasoning_tokens=reasoning_tokens,
        )
    except Exception:
        verbose_logger.debug(
            "Rust cost calculator failed, falling back to Python", exc_info=True
        )
        return None
