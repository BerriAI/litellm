"""litellm.translation: v2 chat translation behind a per-provider opt-in flag.

Hub-and-spoke through a frozen IR: inbound parsers map an accepted request
schema into the IR, provider serializers map the IR onto a wire format, and
``dispatch.route`` decides v1 vs the same-family fast path vs the IR path. This
module exposes the public surface only; ``ir``, ``boundary``, ``inbound``,
``providers``, and ``engine`` are internal and reached through these names.
"""

from .dispatch import InboundSchema, Provider, Route, route
from .engine.pipeline import translate_chat_request
from .errors import BoundaryError, TranslationError
from .ir import ChatRequest

__all__ = [
    "BoundaryError",
    "ChatRequest",
    "InboundSchema",
    "Provider",
    "Route",
    "TranslationError",
    "route",
    "translate_chat_request",
]
