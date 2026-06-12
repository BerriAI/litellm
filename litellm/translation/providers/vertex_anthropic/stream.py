"""Vertex Claude stream parsing.

``:streamRawPredict`` emits genuine anthropic SSE through the anthropic
iterator (v1 wraps it with ``custom_llm_provider="anthropic"``), so the
provider's parser is the anthropic event parser re-exported, exactly like
bedrock_invoke.
"""

from __future__ import annotations

from ..anthropic.stream import parse_event, reverse_names

__all__ = ("parse_event", "reverse_names")
