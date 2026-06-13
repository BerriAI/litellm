"""Raw-shape fidelity guard for the sagemaker_chat serializer.

``SagemakerChatConfig`` is a pure ``OpenAIGPTConfig`` over SigV4 transport:
the explicit ``stream: false`` arm applies (an explicitly-sent False rides
optional_params into the body — probed), then the shared openai guard with
the FULL message-``name`` fallback (the base transform forwards names).

SigV4 (``sign_request`` -> ``BaseAWSLLM._sign_request``, service
"sagemaker") signs the EXACT serialized body AFTER assembly — envelope
scope, the bedrock sign-after-body-final precedent; nothing here touches
credentials (semgrep-enforced). aws_* params ride v1's optional_params
INTO THE BODY (probed: ``aws_region_name`` appears in the wire body); in
v2 they are unknown inbound keys, so any aws-kwarg-bearing request falls
back at parse and v1 serves it.
"""

from __future__ import annotations

from collections.abc import Mapping

from ...errors import TranslationError
from ..openai_compat.guard import stream_false_then_unsupported_shapes

_Raw = Mapping[str, object]


def unsupported_request_shapes(raw: _Raw) -> TranslationError | None:
    # the ONE shared composition (critic-wave2b-alpha NIT-1; sibling-merge
    # sweep: this body re-declared it)
    return stream_false_then_unsupported_shapes(raw)
