"""The azure_ai Claude route: the anthropic serializer plus envelope deltas.

v1 routes ``azure_ai`` models with "claude" in the name to
``AzureAnthropicChatCompletion`` with ``AzureAnthropicConfig(AnthropicConfig)``
(main.py azure_ai branch; api_base forced to ``.../anthropic/v1/messages``).
The config overrides NOTHING about the wire transform: ``map_openai_params``
is AnthropicConfig's with the REAL model name (so response_format picks
output_format vs json-tool by the model itself -- unlike bedrock_invoke and
vertex, which spoof ``RESPONSE_FORMAT_SPOOF_MODEL``), and
``transform_request`` only pops ``extra_body``/``max_retries``/
``stream_options`` (unreachable through the inbound schema) after the parent
transform. The two real deltas are envelope or fail-closed here:
api-key/AD-token auth headers stay in the seam, and
``should_strip_billing_metadata() is True`` drops
``x-anthropic-billing-header:`` system blocks v1-side, so requests carrying
them fall back instead of diverging. Response and stream are genuine
anthropic wire format: re-exports.
"""

from __future__ import annotations

from expression import Error, Result

from ...deps import TranslationDeps
from ...errors import TranslationError
from ...ir import Body, ChatRequest
from ..anthropic import serialize_request as anthropic_serialize_request
from ..anthropic.response import parse_response
from ..anthropic.stream import parse_event, reverse_names

_SerializeResult = Result[Body, TranslationError]

_BILLING_HEADER_PREFIX = "x-anthropic-billing-header:"


def serialize_request(request: ChatRequest, deps: TranslationDeps) -> _SerializeResult:
    if "claude" not in request.model.lower():
        return Error(
            TranslationError.of_unsupported(
                f"azure_ai claude route serves Claude models only; "
                f"{request.model} stays on v1 (main.py's 'claude in model' fork)"
            )
        )
    if any(system.text.startswith(_BILLING_HEADER_PREFIX) for system in request.system):
        return Error(
            TranslationError.of_unsupported(
                "x-anthropic-billing-header system block: azure_ai strips it "
                "(should_strip_billing_metadata, azure_ai/anthropic/"
                "transformation.py:43-44); v1 handles it"
            )
        )
    return anthropic_serialize_request(request, deps)


__all__ = ("parse_event", "parse_response", "reverse_names", "serialize_request")
