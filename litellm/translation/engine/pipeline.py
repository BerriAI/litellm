"""Async-first composition and the public translation entry point.

For this slice the inbound schema is OpenAI chat and the only ported provider
is Anthropic. ``translate_chat_request`` composes the inbound parse with the
provider serialize, lifting the boundary failure into a ``TranslationError`` so
the whole pipeline returns one ``Result`` and never raises.
"""

from __future__ import annotations

from typing import Callable, Dict

from expression import Error

from ..dispatch import Provider
from ..errors import TranslateResult, TranslationError
from ..inbound.openai_chat import parse_request
from ..ir import Body, ChatRequest
from ..providers.anthropic import serialize_request

_SERIALIZERS: Dict[Provider, Callable[[ChatRequest], Body]] = {
    "anthropic": serialize_request,
}


def translate_chat_request(
    raw: Dict[str, object], provider: Provider
) -> TranslateResult:
    serializer = _SERIALIZERS.get(provider)
    if serializer is None:
        return Error(
            TranslationError.of_unsupported(
                f"provider {provider!r} has no v2 chat serializer yet"
            )
        )
    return parse_request(raw).map(serializer).map_error(TranslationError.of_boundary)
