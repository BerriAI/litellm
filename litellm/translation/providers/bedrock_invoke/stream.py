"""Bedrock Invoke stream parsing.

Invoke streaming is anthropic SSE parsing over AWS event-stream framing
(v1's ``AmazonAnthropicClaudeStreamDecoder`` wraps the anthropic
``chunk_parser``); pinned at the parsed-event seam, the events ARE anthropic
stream events, so the provider's parser is the anthropic event parser.
"""

from __future__ import annotations

from ..anthropic.stream import parse_event, reverse_names

__all__ = ("parse_event", "reverse_names")
