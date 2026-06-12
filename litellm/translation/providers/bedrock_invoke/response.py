"""Bedrock Invoke response parsing.

The invoke response body IS anthropic wire format (v1 delegates to
``AnthropicConfig.transform_response`` verbatim), so the provider's parser is
the anthropic parser re-exported under the route's name.
"""

from __future__ import annotations

from ..anthropic.response import parse_response

__all__ = ("parse_response",)
