"""
Pattern pre-filter used by ContentFilterGuardrail to skip regex patterns that
cannot possibly match a given text, without changing matching/masking
semantics for patterns that might.
"""

from dataclasses import dataclass
from typing import Protocol


class PatternPrefilter(Protocol):
    def any_match(self, text: str) -> bool: ...


@dataclass(frozen=True)
class AlwaysMatchPrefilter:
    """Safe default: never skips a pattern. Used when the Rust extension
    isn't installed, or when no pattern was eligible for the fast path."""

    def any_match(self, text: str) -> bool:
        return True


def build_rust_pattern_prefilter(
    pattern_sources: list[str],
) -> tuple[PatternPrefilter, frozenset[int]]:
    """
    Build a Rust-backed prefilter for `pattern_sources`.

    Returns the prefilter and the indices of `pattern_sources` it does not
    cover (incompatible regex syntax, e.g. lookaround/backreferences). Those
    must keep going through the existing per-pattern path. Falls back to a
    pass-through prefilter covering nothing if the Rust extension isn't built.
    """
    try:
        import litellm_core  # pyright: ignore[reportMissingTypeStubs]
    except ImportError:
        return AlwaysMatchPrefilter(), frozenset()

    prefilter, rejected_indices = (
        litellm_core.build_pattern_prefilter(  # pyright: ignore[reportAttributeAccessIssue,reportUnknownVariableType,reportUnknownMemberType]
            pattern_sources
        )
    )
    return prefilter, frozenset(
        rejected_indices
    )  # pyright: ignore[reportUnknownArgumentType]
