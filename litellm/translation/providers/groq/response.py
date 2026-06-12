"""groq chat-completion response JSON -> IR ``ChatResponse``.

v1 is ``GroqChatConfig.transform_response``: the OpenAILike DIRECT
construction (``ModelResponse(**response_json)``, ``custom_llm_provider=
None`` -> the prefix arm is DEAD, BARE wire model — researcher-4 verified)
plus the groq ``service_tier`` post-step: the response's service_tier is
CLAMPED to {auto, default, flex} with None/unknown -> "auto", setattr'd on
every response. Unknown top-level keys (``x_groq``) survive onto the
ModelResponse via the direct construction.

v2 mirrors the compat_httpx "openai_like" convention: the shared openai
parser for fail-closed validation, then the VERBATIM raw body rides
``ChatResponse.wire`` with the clamped ``service_tier`` attached, so the
seam's ``openai_like`` construction arm reproduces v1 byte-for-byte. The
verifier-longtail F2 arm applies (non-string wire model raises
ValidationError inside v1's construction; ``model: None`` constructs and
serves).
"""

from __future__ import annotations

from dataclasses import replace

from expression import Error, Ok, Result, Some
from expression.collections import Block

from ...errors import BoundaryError, TranslationError
from ...ir import ChatRequest, ChatResponse, JsonBlob, PlainJson
from ..openai_compat.response import parse_response as openai_parse_response

_ParseResult = Result[ChatResponse, TranslationError]

_SERVICE_TIERS = frozenset({"auto", "default", "flex"})


def parse_response(raw: PlainJson, request: ChatRequest) -> _ParseResult:
    return openai_parse_response(raw, request).bind(
        lambda response: _groq_wire(response, raw)
    )


def _groq_wire(response: ChatResponse, raw: PlainJson) -> _ParseResult:
    if not isinstance(raw, dict):
        return Error(
            TranslationError.of_boundary(
                BoundaryError.of(
                    Block.of_seq(["non-dict response body reached _groq_wire"])
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
    if "service_tier" not in raw:
        # v1's post-step reads getattr(model_response, "service_tier") with
        # NO default: a response body without the key crashes v1 with
        # AttributeError (probed in-process) — fail closed, never serve a
        # synthesized "auto" where v1 errors.
        return Error(
            TranslationError.of_boundary(
                BoundaryError.of(
                    Block.of_seq(
                        [
                            "groq response without a service_tier key: v1's "
                            "clamp post-step raises AttributeError on it"
                        ]
                    )
                )
            )
        )
    tier = raw.get("service_tier")
    clamped = tier if isinstance(tier, str) and tier in _SERVICE_TIERS else "auto"
    body: dict[str, PlainJson] = {**raw, "service_tier": clamped}
    return Ok(replace(response, wire=Some(JsonBlob(value=body))))
