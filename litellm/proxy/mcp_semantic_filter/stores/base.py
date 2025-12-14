"""Interfaces for semantic MCP filter vector stores."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Protocol, Sequence, Tuple


@dataclass
class ToolVectorRecord:
    tool_id: str
    vector: Sequence[float]


class ToolVectorStore(Protocol):
    """Basic operations every vector backend must support."""

    def replace_records(self, records: Iterable[ToolVectorRecord]) -> None:
        """Replace the entire index with `records`."""

    def upsert_records(self, records: Iterable[ToolVectorRecord]) -> None:
        """Add or replace the provided records."""

    def remove_records(self, tool_ids: Iterable[str]) -> None:
        """Delete the specified tool ids from the index."""

    def query(self, vector: Sequence[float], top_k: int) -> List[Tuple[str, float]]:
        """Return `(tool_id, score)` pairs sorted by similarity."""
