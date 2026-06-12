"""Claude on Vertex: the anthropic serializer plus envelope deltas
(the bedrock_invoke wrapper pattern, dossier section 1.4).

v1's ``VertexAIAnthropicConfig`` IS ``AnthropicConfig`` plus an envelope:
``model`` pops into the URL (``:rawPredict``), ``anthropic_version:
vertex-2023-10-16`` is injected via optional_params, structured outputs are
forced onto the json-tool strategy by the same model spoof bedrock_invoke
uses, and the anthropic handler pops ``json_mode`` BEFORE the transform on
this path (so unlike invoke, the marker never reaches the body). v1's beta
machinery is suppressed for vertex except shapes the IR cannot express
(tool search, context management, web search), so a v2-built body never
carries ``anthropic_beta``; shapes that would (plus the output_config /
output_format families) return ``unsupported``.
"""

from __future__ import annotations

from dataclasses import replace

from expression import Error, Ok, Result

from ...deps import TranslationDeps
from ...errors import TranslationError
from ...ir import Body, ChatRequest
from ..anthropic import params as anthropic_params
from ..anthropic import serialize_request as anthropic_serialize_request
from ..bedrock_invoke.serialize import _RESPONSE_FORMAT_SPOOF_MODEL

_SerializeResult = Result[Body, TranslationError]

ANTHROPIC_VERSION = "vertex-2023-10-16"


def serialize_request(request: ChatRequest, deps: TranslationDeps) -> _SerializeResult:
    reasoning = request.thinking.is_some() or request.reasoning_effort.is_some()
    if request.response_format.is_some() and reasoning:
        return Error(
            TranslationError.of_unsupported(
                "response_format with thinking/reasoning_effort on vertex claude crosses the model spoof; v1 handles it"
            )
        )
    if (
        request.reasoning_effort.is_some()
        and anthropic_params.is_adaptive_thinking_model(request.model, deps)
    ):
        return Error(
            TranslationError.of_unsupported(
                "reasoning_effort on adaptive-thinking models takes v1's output_config/beta path"
            )
        )
    mapped = (
        replace(request, model=_RESPONSE_FORMAT_SPOOF_MODEL)
        if request.response_format.is_some()
        else request
    )
    return anthropic_serialize_request(mapped, deps).bind(_apply_envelope)


def _apply_envelope(body: Body) -> _SerializeResult:
    if "output_format" in body or "output_config" in body:
        return Error(
            TranslationError.of_unsupported(
                "output_format/output_config on vertex claude takes v1's sanitize/beta path"
            )
        )
    enveloped: Body = {
        key: value
        for key, value in body.items()
        if key not in ("model", "stream", "json_mode")
    }
    return Ok({**enveloped, "anthropic_version": ANTHROPIC_VERSION})
