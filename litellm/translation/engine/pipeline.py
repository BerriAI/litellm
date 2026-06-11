"""Composition and the public translate entry point.

``translate_chat_request`` composes the inbound parse with the provider
serialize under injected deps; the whole pipeline returns one ``Result`` and
never raises. Any error means "outside v2's proven surface" and the dispatch
seam falls back to v1, so no request ever loses a feature silently.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Callable, Mapping

from expression import Error, Result

from ..deps import TranslationDeps
from ..dispatch import Provider
from ..errors import TranslateResult, TranslationError
from ..inbound.openai_chat import parse_request
from ..ir import Body, ChatRequest
from ..providers.anthropic import serialize_request

_Serializer = Callable[[ChatRequest, TranslationDeps], Result[Body, TranslationError]]

_SERIALIZERS: Mapping[Provider, _Serializer] = MappingProxyType(
    {
        "anthropic": serialize_request,
    }
)


def translate_chat_request(raw: Mapping[str, object], provider: Provider, deps: TranslationDeps) -> TranslateResult:
    serializer = _SERIALIZERS.get(provider)
    if serializer is None:
        return Error(TranslationError.of_unsupported(f"provider {provider!r} has no v2 chat serializer yet"))
    return parse_request(raw).bind(lambda request: serializer(request, deps))
