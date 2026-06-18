"""litellm.translation: v2 chat translation behind the single ``LLM_TRANSLATION_V2`` flag.

Hub-and-spoke through a frozen IR: inbound parsers map an accepted request
schema into the IR, provider serializers map the IR onto a wire format, and
``dispatch.route`` decides v1 vs the same-family fast path vs the IR path.
One flag opts a deployment into v2 for every ported provider at once; which
providers are ported is the serializer registry (``ported_providers``), not
configuration. This module exposes the public surface only; ``ir``,
``boundary``, ``inbound``, ``providers``, and ``engine`` are internal and
reached through these names.
"""

from .dispatch import InboundSchema, Provider, Route, route
from .engine.pipeline import ported_providers, translate_chat_request
from .errors import BoundaryError, TranslationError
from .flag import TRANSLATION_V2_ENV, is_translation_v2_enabled
from .ir import ChatRequest

__all__ = [
    "BoundaryError",
    "ChatRequest",
    "InboundSchema",
    "Provider",
    "Route",
    "TRANSLATION_V2_ENV",
    "TranslationError",
    "is_translation_v2_enabled",
    "ported_providers",
    "route",
    "translate_chat_request",
]
