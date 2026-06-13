"""Raw-shape fidelity guard for the deepseek serializer.

The httpx path serializes an explicitly-sent ``stream: false`` onto the wire
(the shared family fact), so that arm runs first; then the shared openai
guard with the FULL message-``name`` fallback — v1's deepseek transform
chains the base ``_transform_messages`` after its flatten and nothing strips
names, so v1 forwards ``name`` verbatim on every role.
"""

from __future__ import annotations

from ..openai_compat.guard import stream_false_then_unsupported_shapes

unsupported_request_shapes = stream_false_then_unsupported_shapes
