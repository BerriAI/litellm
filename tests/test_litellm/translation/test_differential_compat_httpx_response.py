"""Differential parity for the compat_httpx (wave-1b) response path.

The v1 reference is each config's ``transform_response`` — LIVE on the
httpx path — over an ``httpx.Response`` with a FRESH ``ModelResponse()``
(no model preset, the xai R4 rule; the seam must never apply the compat_sdk
``{provider}/{model}`` preset to this family). The rows pin, per provider:

- the shared openai-body normalization (text / tool_calls finish rewrite /
  usage details) byte-identical through the family parser;
- the MODEL FIELD pin researcher-4 binds: bare wire model for six
  providers; ``compactifai/{REQUEST model}``, ``amazon-nova/{REQUEST
  model}`` (the literal hyphenated string), ``lemonade/{REQUEST model}``
  for the three prefixing configs (wire model ignored);
- the usage-null row proving the OpenAILike ``_sanitize_usage_obj`` delta
  is observationally dead at this layer (litellm's ``Usage`` zeros None
  token values on the OpenAIGPT-based configs too, and v2's parser already
  matches).
"""

import copy
import json

import pytest

from litellm.types.utils import ModelResponse

from litellm.translation.inbound.openai_chat import parse_request
from litellm.translation.inbound.openai_chat.response import serialize_response
from litellm.translation.providers.compat_httpx import PARSERS
from litellm.translation.providers.compat_httpx.response import RESPONSE_STYLES
from litellm.translation_seam import build_translation_deps, to_model_response

from ._compat_httpx_corpus import PROVIDERS, SPECS, run_v1_response_transform
from .test_differential_openai_response import _RESPONSES

_RESPONSE_ROWS = (
    "text",
    "tool_calls_rewrites_stop",
    "cached_and_reasoning_usage_details",
)

_NULL_USAGE_BODY = {
    "id": "resp-null-usage",
    "object": "chat.completion",
    "created": 1718000000,
    "model": "wire-model-x",
    "choices": [
        {
            "index": 0,
            "finish_reason": "stop",
            "logprobs": None,
            "message": {"role": "assistant", "content": "hello"},
        }
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": None, "total_tokens": 10},
}


def _request_for(provider: str) -> dict:
    return {
        "model": SPECS[provider].model,
        "messages": [{"role": "user", "content": "hi"}],
    }


def _v2_model_response(provider: str, raw: dict) -> dict:
    parsed = parse_request(_request_for(provider))
    assert parsed.is_ok(), parsed.error.summary
    response = PARSERS[provider](copy.deepcopy(raw), parsed.ok)
    assert response.is_ok(), response.error.summary
    body = serialize_response(response.ok, build_translation_deps(), "openai")
    # the per-provider construction style (cdr vs ModelResponse(**json)) is
    # family DATA the future seam fork must read the same way
    return to_model_response(
        body, ModelResponse(), usage_style=RESPONSE_STYLES[provider]
    ).model_dump()


def _norm(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, default=str)


def _rows():
    return sorted((provider, name) for provider in PROVIDERS for name in _RESPONSE_ROWS)


@pytest.mark.parametrize("provider,name", _rows())
def test_v2_response_matches_v1(provider: str, name: str, frozen_ambient) -> None:
    raw = _RESPONSES[name]
    v1 = run_v1_response_transform(provider, raw, SPECS[provider].model).model_dump()
    v2 = _v2_model_response(provider, raw)
    assert _norm(v2) == _norm(v1)


@pytest.mark.parametrize("provider", PROVIDERS)
def test_response_model_field_pin(provider: str, frozen_ambient) -> None:
    """researcher-4's R4-style pin per provider: bare wire model (no seam
    preset, no prefix) — or the {prefix}/{REQUEST model} rewrite for
    compactifai/amazon-nova/lemonade, wire model ignored."""
    raw = _RESPONSES["text"]
    spec = SPECS[provider]
    v1 = run_v1_response_transform(provider, raw, spec.model).model_dump()
    v2 = _v2_model_response(provider, raw)
    assert v1["model"] == v2["model"]
    if spec.prefix is None:
        assert v2["model"] == raw["model"]
    else:
        assert v2["model"] == f"{spec.prefix}/{spec.model}"


@pytest.mark.parametrize("provider", PROVIDERS)
def test_usage_null_tokens_zeroed_on_both_sides(provider: str, frozen_ambient) -> None:
    """The OpenAILike usage-null sanitize is observationally dead: Usage
    coerces None -> 0 on every config base, and v2's parser matches."""
    v1 = run_v1_response_transform(
        provider, _NULL_USAGE_BODY, SPECS[provider].model
    ).model_dump()
    v2 = _v2_model_response(provider, _NULL_USAGE_BODY)
    assert _norm(v2) == _norm(v1)
    assert v2["usage"]["completion_tokens"] == 0
