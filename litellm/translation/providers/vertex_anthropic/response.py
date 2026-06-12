"""Vertex Claude response parsing.

The ``:rawPredict`` response body IS anthropic wire format; the only delta is
v1's ``VertexAIAnthropicConfig.transform_response`` restoring the REQUEST
model onto the response (the vertex body carries no model the caller knows).
"""

from __future__ import annotations

from dataclasses import replace

from expression import Result

from ...errors import TranslationError
from ...ir import ChatRequest, ChatResponse, PlainJson
from ..anthropic.response import parse_response as anthropic_parse_response

__all__ = ("parse_response",)


def parse_response(
    raw: PlainJson, request: ChatRequest
) -> Result[ChatResponse, TranslationError]:
    return anthropic_parse_response(raw, request).map(
        lambda response: replace(response, model=request.model)
    )
