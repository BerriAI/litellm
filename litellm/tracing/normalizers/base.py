from collections.abc import Mapping
from typing import Protocol

from litellm.tracing.types import SpanRow


class SpanNormalizer(Protocol):
    """Maps one tracing convention's span attributes onto the LiteLLM `SpanRow` columns."""

    @property
    def name(self) -> str: ...

    def matches(self, scope_name: str, attributes: Mapping[str, str]) -> bool: ...

    def normalize(self, row: SpanRow, attributes: Mapping[str, str]) -> None: ...


def to_int(value: str | None) -> int:
    try:
        return int(value) if value else 0
    except ValueError:
        return 0
