"""azure_ai stream parsing.

The httpx iterator parses the same chat.completion.chunk family and
``CustomStreamWrapper``'s azure branch applies to ``azure_ai`` too
(streaming_handler.py:1448-1454 lists both providers for the per-chunk model
re-read), so the azure parser is re-exported and the fold runs the ``azure``
chunk dialect.
"""

from __future__ import annotations

from ..azure.stream import parse_event, parse_line

__all__ = ("parse_event", "parse_line")
