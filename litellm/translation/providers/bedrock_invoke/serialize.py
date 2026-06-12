"""Bedrock Invoke (anthropic messages route): the anthropic serializer plus
envelope deltas.

v1's ``AmazonAnthropicClaudeConfig`` IS ``AnthropicConfig`` plus an envelope
(claude3 transformation): ``model``/``stream`` pop into the URL,
``anthropic_version: bedrock-2023-05-31`` is injected, and structured outputs
are forced onto the json-tool strategy by spoofing the mapping model to
``claude-3-sonnet-20240229``. Only those deltas live here; the message/tool/
param semantics are the one anthropic serializer. Image URL sources and
document blocks never reach this point (the inbound parser fails closed), so
v1's force-base64 conversions have nothing to do. Beta-header-producing
shapes (adaptive effort/output_config) return ``unsupported``.
"""

from __future__ import annotations

from dataclasses import replace

from expression import Error, Ok, Result

from ...deps import TranslationDeps
from ...errors import TranslationError
from ...ir import Body, ChatRequest
from ..anthropic import params as anthropic_params
from ..anthropic import serialize_request as anthropic_serialize_request
from ..bedrock_converse.params import is_anthropic_base

_SerializeResult = Result[Body, TranslationError]

ANTHROPIC_VERSION = "bedrock-2023-05-31"

# Shared with providers/vertex_anthropic: both routes force structured
# outputs onto the json-tool strategy by spoofing this mapping model.
RESPONSE_FORMAT_SPOOF_MODEL = "claude-3-sonnet-20240229"


def serialize_request(request: ChatRequest, deps: TranslationDeps) -> _SerializeResult:
    if not is_anthropic_base(request.model):
        return Error(
            TranslationError.of_unsupported(
                f"bedrock invoke v2 serves Claude models only; {request.model} stays on v1"
            )
        )
    reasoning = request.thinking.is_some() or request.reasoning_effort.is_some()
    if request.response_format.is_some() and reasoning:
        return Error(
            TranslationError.of_unsupported(
                "response_format with thinking/reasoning_effort on invoke crosses the model spoof; v1 handles it"
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
        replace(request, model=RESPONSE_FORMAT_SPOOF_MODEL)
        if request.response_format.is_some()
        else request
    )
    return anthropic_serialize_request(mapped, deps).bind(_apply_envelope)


def _apply_envelope(body: Body) -> _SerializeResult:
    if "output_format" in body or "output_config" in body:
        return Error(
            TranslationError.of_unsupported(
                "output_format/output_config on invoke takes v1's inline-schema path"
            )
        )
    enveloped: Body = {
        key: value for key, value in body.items() if key not in ("model", "stream")
    }
    return Ok({**enveloped, "anthropic_version": ANTHROPIC_VERSION})
