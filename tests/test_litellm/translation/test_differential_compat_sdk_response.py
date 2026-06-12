"""Differential parity for the compat_sdk (wave-1a) response path.

The live v1 normalizer for every provider here is the SAME
``convert_to_model_response_object`` the openai differential pins in depth —
the per-provider delta is the envelope preset: openai.py:676-677 sets
``model_response.model = f"{provider}/{model}"`` for every non-"openai"
compat consumer BEFORE conversion, and cdr's model arm (cdr:699-710) then
re-prefixes onto the wire model. These rows pin exactly that, per provider,
through the seam's ``_to_model_response_openai`` (the path-keyed B1 arm):
the preset must come out as ``{provider}/{wire_model}``, byte-identical to
v1. Depth (usage details, finish rewrites, reasoning extraction, unsupported
shapes) is the openai response differential's job; the parser is shared.
"""

import copy
import json

import pytest

from litellm.types.utils import ModelResponse
from litellm.utils import convert_to_model_response_object

from litellm.translation.inbound.openai_chat import parse_request
from litellm.translation.inbound.openai_chat.response import serialize_response
from litellm.translation.providers.openai_compat.response import parse_response
from litellm.translation_seam import build_translation_deps, to_model_response

from ._compat_sdk_corpus import PROVIDERS, SPECS
from .test_differential_openai_response import _RESPONSES

_RESPONSE_ROWS = (
    "text",
    "tool_calls_rewrites_stop",
    "cached_and_reasoning_usage_details",
)


def _request_for(provider: str) -> dict:
    return {
        "model": SPECS[provider].model,
        "messages": [{"role": "user", "content": "hi"}],
    }


def _v1_model_response(raw: dict, preset_model: str) -> dict:
    result = convert_to_model_response_object(
        response_object=copy.deepcopy(raw),
        model_response_object=ModelResponse(model=preset_model),
    )
    return result.model_dump()


def _v2_model_response(provider: str, raw: dict, preset_model: str) -> dict:
    parsed = parse_request(_request_for(provider))
    assert parsed.is_ok(), parsed.error.summary
    response = parse_response(copy.deepcopy(raw), parsed.ok)
    assert response.is_ok(), response.error.summary
    body = serialize_response(response.ok, build_translation_deps(), "openai")
    return to_model_response(
        body, ModelResponse(model=preset_model), usage_style="openai"
    ).model_dump()


def _norm(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, default=str)


def _rows():
    # Every family member is SDK-path (the preset/re-prefix arm is the SDK
    # seam's); httpx members live in compat_httpx with their own gates —
    # cometapi's NO-prefix pins are in test_differential_cometapi_response.py
    # (the xai R4 shape).
    return sorted((provider, name) for provider in PROVIDERS for name in _RESPONSE_ROWS)


@pytest.mark.parametrize("provider,name", _rows())
def test_preset_reprefix_matches_v1(provider: str, name: str, frozen_ambient) -> None:
    """SDK-path preset: {provider}/{request model} in, {provider}/{wire
    model} out, byte-identical dumps both sides."""
    raw = _RESPONSES[name]
    preset = f"{provider}/{SPECS[provider].model}"
    v1 = _v1_model_response(raw, preset)
    v2 = _v2_model_response(provider, raw, preset)
    assert _norm(v2) == _norm(v1)
    wire_model = raw["model"]
    assert v2["model"] == f"{provider}/{wire_model}"


@pytest.mark.parametrize("provider", PROVIDERS)
def test_preset_survives_when_wire_model_missing(provider: str, frozen_ambient) -> None:
    """cdr's elif arm needs a non-None wire model; without one the preset
    {provider}/{request model} survives verbatim on both sides."""
    raw = {k: v for k, v in _RESPONSES["text"].items() if k != "model"}
    preset = f"{provider}/{SPECS[provider].model}"
    v1 = _v1_model_response(raw, preset)
    v2 = _v2_model_response(provider, raw, preset)
    assert _norm(v2) == _norm(v1)
    assert v2["model"] == preset


_PERPLEXITY_CITATIONS_RESPONSE = {
    "id": "resp-cite",
    "object": "chat.completion",
    "created": 1718000000,
    "model": "sonar",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Answer [1]."},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 7, "completion_tokens": 4, "total_tokens": 11},
    "citations": ["https://a.example"],
    "search_results": [{"url": "https://a.example", "title": "A"}],
}


def test_perplexity_citation_enrichment_is_dormant(frozen_ambient) -> None:
    """PerplexityChatConfig.transform_response (citation-token estimation +
    citations->annotations) is DORMANT on the SDK path: the live normalizer
    is convert_to_model_response_object, where top-level citations/
    search_results survive only via the unknown-key mirror and NO
    annotations / usage.citation_tokens exist. v2 must match the dormant
    truth byte-identically — porting the enrichment would be inventing
    behavior v1 doesn't execute (researcher-4 entry 1)."""
    preset = "perplexity/sonar"
    v1 = _v1_model_response(_PERPLEXITY_CITATIONS_RESPONSE, preset)
    v2 = _v2_model_response("perplexity", _PERPLEXITY_CITATIONS_RESPONSE, preset)
    assert _norm(v2) == _norm(v1)
    assert v2["citations"] == ["https://a.example"]
    assert v2["search_results"] == [{"url": "https://a.example", "title": "A"}]
    assert v2["choices"][0]["message"].get("annotations") is None
    assert "citation_tokens" not in v2["usage"]
