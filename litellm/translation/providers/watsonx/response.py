"""watsonx chat-completion response JSON -> IR ``ChatResponse``.

v1 is ``OpenAILikeChatConfig._transform_response`` called by the
``OpenAILikeChatHandler`` with ``custom_llm_provider="watsonx"`` — the ONE
wave-2b path where the openai_like prefix arm is LIVE:
``ModelResponse(**response_json)`` DIRECTLY (no cdr) and then
``model = "watsonx/" + (wire model or "")``. v2 mirrors the compat_httpx
"openai_like" convention: the shared openai parser for fail-closed shape
validation, then the VERBATIM raw body rides ``ChatResponse.wire`` with the
model overwritten to the PREFIXED WIRE model (NOT the request model — the
compactifai/amazon-nova/lemonade profiles differ here), so the seam's
``openai_like`` construction arm reproduces v1 byte-for-byte.

The verifier-longtail F2 arm carries over: a present NON-STRING wire model
raises pydantic ValidationError inside v1's construction BEFORE the prefix
overwrite, so it fails closed here; ``model: None``/absent constructs in v1
and serves as the literal ``"watsonx/"``. The OpenAILike usage-null
sanitize (``*_tokens: None`` -> 0) is observationally dead (``Usage``
coerces None -> 0 in the constructor — the family pin, re-pinned in the
watsonx response gate); the json_mode tool->content conversion is dormant
(watsonx never sets json_mode — probed).
"""

from __future__ import annotations

from dataclasses import replace

from expression import Error, Ok, Result, Some
from expression.collections import Block

from ...errors import BoundaryError, TranslationError
from ...ir import ChatRequest, ChatResponse, JsonBlob, PlainJson
from ..openai_compat.response import parse_response as openai_parse_response

_ParseResult = Result[ChatResponse, TranslationError]


def parse_response(raw: PlainJson, request: ChatRequest) -> _ParseResult:
    return openai_parse_response(raw, request).bind(
        lambda response: _prefixed_wire(response, raw)
    )


def _prefixed_wire(response: ChatResponse, raw: PlainJson) -> _ParseResult:
    if not isinstance(raw, dict):
        return Error(
            TranslationError.of_boundary(
                BoundaryError.of(
                    Block.of_seq(["non-dict response body reached _prefixed_wire"])
                )
            )
        )
    wire_model = raw.get("model")
    if wire_model is not None and not isinstance(wire_model, str):
        return Error(
            TranslationError.of_boundary(
                BoundaryError.of(
                    Block.of_seq(
                        [
                            "non-string wire model "
                            f"({type(wire_model).__name__}): v1's OpenAILike "
                            "construction raises ValidationError on it"
                        ]
                    )
                )
            )
        )
    model = f"watsonx/{wire_model if isinstance(wire_model, str) else ''}"
    body: dict[str, PlainJson] = {**raw, "model": model}
    return Ok(replace(response, model=model, wire=Some(JsonBlob(value=body))))
