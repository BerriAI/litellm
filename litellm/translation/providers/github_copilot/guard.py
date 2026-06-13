"""Raw-shape fidelity guard for the github_copilot serializer.

``GithubCopilotConfig`` extends ``OpenAIConfig`` over the SDK path: the
explicit ``stream: false`` arm applies (the SDK serializes the key — the
compat_sdk shape), then the shared openai guard with the FULL message-``name``
fallback (the base transform forwards names; the system->assistant rewrite is
a ``message.copy()`` that preserves every key, so a ``name`` would survive
into the rewritten message and the IR cannot carry it).
"""

from __future__ import annotations

from collections.abc import Mapping

from ...errors import TranslationError
from ..openai_compat.guard import stream_false_then_unsupported_shapes

_Raw = Mapping[str, object]


def unsupported_request_shapes(raw: _Raw) -> TranslationError | None:
    return stream_false_then_unsupported_shapes(raw)
