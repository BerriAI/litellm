"""azure_ai response parsing: the openai parser plus the model rename.

v1's httpx-path ``AzureAIStudioConfig.transform_response`` presets
``model_response.model = f"azure_ai/{request model}"`` and delegates to
``convert_to_model_response_object``, whose model branch
(convert_dict_to_response.py:699-711) then re-prefixes the RESPONSE body's
model: a slash-carrying preset plus a non-None wire model yields
``azure_ai/{wire model}``; otherwise the preset survives. The parser patches
the normalized wire body (and the semantic model) the same way.
"""

from __future__ import annotations

from dataclasses import replace

from expression import Error, Ok, Option, Result, Some
from expression.collections import Block

from ...errors import BoundaryError, TranslationError
from ...ir import ChatRequest, ChatResponse, JsonBlob, PlainJson
from ..openai_compat.response import parse_response as openai_parse_response

_ParseResult = Result[ChatResponse, TranslationError]


def parse_response(raw: PlainJson, request: ChatRequest) -> _ParseResult:
    return openai_parse_response(raw, request).bind(
        lambda response: _prefix_model(response, raw, request)
    )


def _prefix_model(
    response: ChatResponse, raw: PlainJson, request: ChatRequest
) -> _ParseResult:
    wire_model = raw.get("model") if isinstance(raw, dict) else None
    if wire_model is not None and not isinstance(wire_model, str):
        return Error(
            TranslationError.of_boundary(
                BoundaryError.of(Block.of_seq(["response 'model' is not a string"]))
            )
        )
    prefixed = f"azure_ai/{wire_model if wire_model is not None else request.model}"
    match response.wire:
        case Option(tag="some", some=wire):
            body = wire.value
        case _:
            return Error(
                TranslationError.of_unsupported(
                    "openai parser produced no wire body; unreachable"
                )
            )
    if not isinstance(body, dict):
        return Error(
            TranslationError.of_unsupported(
                "openai parser wire body is not an object; unreachable"
            )
        )
    patched: dict[str, PlainJson] = {**body, "model": prefixed}
    return Ok(replace(response, model=prefixed, wire=Some(JsonBlob(value=patched))))
