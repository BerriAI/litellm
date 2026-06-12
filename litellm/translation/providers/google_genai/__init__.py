"""Google generateContent wire format (vertex_ai gemini route + AI Studio).

One serializer family; the two providers differ only by the drift-list
``target`` (see serialize.py) and their envelopes (seam-owned).
"""

from .guard import unsupported_request_shapes
from .response import parse_response
from .serialize import serialize_request_studio, serialize_request_vertex
from .stream import parse_event

__all__ = (
    "parse_event",
    "parse_response",
    "serialize_request_studio",
    "serialize_request_vertex",
    "unsupported_request_shapes",
)
