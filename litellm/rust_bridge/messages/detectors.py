"""Detectors for Python-only Anthropic Messages behavior."""

from __future__ import annotations

from typing import Final

import litellm
from litellm.rust_bridge.public_call import NativeCall


def websearch_interception(_request: NativeCall) -> bool:
    """A WebSearchInterceptionLogger replacing its short-circuit may answer a native-search provider without an LLM call."""
    from litellm.integrations.websearch_interception.handler import WebSearchInterceptionLogger

    stock: Final[object] = WebSearchInterceptionLogger.try_short_circuit_search  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # the logger annotates bare dicts
    return any(
        isinstance(callback, WebSearchInterceptionLogger) and type(callback).try_short_circuit_search is not stock  # pyright: ignore[reportUnknownMemberType]  # the logger annotates bare dicts
        for callback in litellm.callbacks
    )
