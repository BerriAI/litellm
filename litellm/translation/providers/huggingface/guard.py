"""Raw-shape fidelity guard for the huggingface serializer.

The httpx path serializes an explicitly-sent ``stream: false`` onto the
wire (the api_base arm's body carries every optional_params key), so that
arm runs first; then the shared openai guard with the FULL
message-``name`` fallback — the api_base route sends messages VERBATIM,
so every raw shape the IR cannot round-trip losslessly is a v1-serves
fallback.
"""

from __future__ import annotations

from ..openai_compat.guard import stream_false_then_unsupported_shapes

unsupported_request_shapes = stream_false_then_unsupported_shapes
