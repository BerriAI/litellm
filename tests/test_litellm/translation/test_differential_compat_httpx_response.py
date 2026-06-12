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
from typing import cast

import pydantic
import pytest

from litellm.types.utils import ModelResponse

from litellm.translation.dispatch import Provider
from litellm.translation.inbound.openai_chat import parse_request
from litellm.translation.inbound.openai_chat.response import serialize_response
from litellm.translation.providers.compat_httpx import PARSERS
from litellm.translation.providers.compat_httpx.response import RESPONSE_STYLES
from litellm.translation_seam import (
    UsageStyle,
    build_translation_deps,
    to_model_response,
)

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


def _v2_parse_result(provider: str, raw: dict):
    parsed = parse_request(_request_for(provider))
    assert parsed.is_ok(), parsed.error.summary
    return PARSERS[provider](copy.deepcopy(raw), parsed.ok)


_NON_STRING_MODEL_BODY = {
    "id": "resp-badmodel",
    "object": "chat.completion",
    "created": 1718000000,
    "model": 7,
    "choices": [
        {
            "index": 0,
            "finish_reason": "stop",
            "logprobs": None,
            "message": {"role": "assistant", "content": "hello"},
        }
    ],
    "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
}

_PREFIXING_MEMBERS = sorted(p for p in PROVIDERS if SPECS[p].prefix is not None)


@pytest.mark.parametrize("provider", _PREFIXING_MEMBERS)
def test_non_string_wire_model_falls_back_where_v1_raises(
    provider: str, frozen_ambient
) -> None:
    """verifier-longtail F2: v1's OpenAILike construction is
    ModelResponse(**response_json) BEFORE the model overwrite, so a
    non-string wire model RAISES pydantic ValidationError — while v2's
    prefix rewrite used to hide the bad value from the constructor and
    SERVE. The family parser now fails closed, so the typed v1 fallback
    reproduces v1's raise; the fork-wirer must keep this arm ahead of any
    compat_httpx completion() fork (the CLAUDE.md obligation)."""
    with pytest.raises(pydantic.ValidationError):
        run_v1_response_transform(
            provider, _NON_STRING_MODEL_BODY, SPECS[provider].model
        )
    result = _v2_parse_result(provider, _NON_STRING_MODEL_BODY)
    assert result.is_error(), provider
    assert "non-string wire model" in result.error.summary, result.error.summary


def test_non_string_wire_model_family_arm_canary(frozen_ambient) -> None:
    """The arm is family-wide and exactly as wide as v1's raise: a
    NON-prefix openai_like member (datarobot) raises the same
    ValidationError in v1 (its construction sees the bad value directly),
    so the typed fallback reproduces v1 there too — and ``model: None``,
    which v1 constructs and serves, stays SERVED byte-for-byte through the
    prefix rewrite. If v1's OpenAILike ever constructs AFTER the model
    overwrite, re-decide this arm."""
    with pytest.raises(pydantic.ValidationError):
        run_v1_response_transform(
            "datarobot", _NON_STRING_MODEL_BODY, SPECS["datarobot"].model
        )
    result = _v2_parse_result("datarobot", _NON_STRING_MODEL_BODY)
    assert result.is_error()
    assert "non-string wire model" in result.error.summary
    null_model_body = {**_NON_STRING_MODEL_BODY, "model": None}
    v1 = run_v1_response_transform(
        "compactifai", null_model_body, SPECS["compactifai"].model
    ).model_dump()
    v2 = _v2_model_response("compactifai", null_model_body)
    assert _norm(v2) == _norm(v1)
    assert v2["model"] == f"compactifai/{SPECS['compactifai'].model}"


_STOP_WITH_TOOL_CALLS_BODY = {
    "id": "resp-armgate",
    "object": "chat.completion",
    "created": 1718000000,
    "model": "wire-model-x",
    "choices": [
        {
            "index": 0,
            # The discriminator (verifier-wave1b): cdr REWRITES finish
            # "stop" -> "tool_calls" when tool_calls are present; the direct
            # ModelResponse(**json) construction keeps "stop" verbatim.
            "finish_reason": "stop",
            "logprobs": None,
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{}"},
                    }
                ],
            },
        }
    ],
    "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
}


def _v2_with_style(
    provider: str, raw: dict, style: UsageStyle, model_response: ModelResponse
) -> dict:
    parsed = parse_request(_request_for(provider))
    assert parsed.is_ok(), parsed.error.summary
    response = PARSERS[provider](copy.deepcopy(raw), parsed.ok)
    assert response.is_ok(), response.error.summary
    body = serialize_response(response.ok, build_translation_deps(), "openai")
    return to_model_response(
        body, model_response, usage_style=style
    ).model_dump()


@pytest.mark.parametrize("provider", PROVIDERS)
def test_construction_arm_must_come_from_response_styles(
    provider: str, frozen_ambient
) -> None:
    """The verifier-wave1b F3 trap, pinned per provider at HEAD. Wiring the
    compat_httpx completion() fork "the way the wired forks work today"
    means copying the compat_sdk template: preset
    ``ModelResponse(model=f"{provider}/{model}")`` + ``usage_style=
    response_dialect(provider)``. Both halves are wrong for this family and
    this gate pins each: (a) ``response_dialect()`` answers "openai" for
    every member (it is the OUTBOUND-BODY dialect — correct for
    serialize_response, NOT a construction-style source; for the
    openai_like members it contradicts RESPONSE_STYLES), and (b) the
    template combination rewrites the response model to
    ``{provider}/{wire model}`` where v1 serves the bare wire model /
    request-model prefix — a real dump divergence for EVERY member. The
    obligated combination (fresh ModelResponse + RESPONSE_STYLES) equals v1
    byte-for-byte. The arms-coincide note that used to live here was
    REFUTED (verifier-longtail F1): over parser-admissible bodies the two
    seam arms diverge on real shapes (choice-index enumerate rewrite,
    unknown choice keys, created/id/model/system_fingerprint coercion vs
    pydantic ValidationError), so the RESPONSE_STYLES read is a LIVE
    byte-parity obligation — pinned per openai_like member by
    ``test_construction_arms_diverge_on_wire_choice_index`` below — and
    the companion AST gate fails the moment a fork is wired through the
    dialect helper."""
    from litellm.translation.engine.pipeline import response_dialect

    # the corpus PROVIDERS tuple is str-typed; every member is a registered
    # pipeline Provider row (the critic-wave1b N4 cast shape, not Any)
    dialect_answer = response_dialect(cast(Provider, provider))
    assert dialect_answer == "openai", provider
    if RESPONSE_STYLES[provider] == "openai_like":
        # the table the fork MUST read disagrees with the dialect helper
        assert RESPONSE_STYLES[provider] != dialect_answer, provider
    raw = _STOP_WITH_TOOL_CALLS_BODY
    v1 = run_v1_response_transform(provider, raw, SPECS[provider].model).model_dump()
    correct = _v2_with_style(provider, raw, RESPONSE_STYLES[provider], ModelResponse())
    assert _norm(correct) == _norm(v1), provider
    # the compat_sdk fork template (preset + dialect shortcut) diverges on
    # the model field for every family member: NO preset is the R4 rule
    preset = ModelResponse(model=f"{provider}/{SPECS[provider].model}")
    template = _v2_with_style(provider, raw, dialect_answer, preset)
    if SPECS[provider].prefix is None:
        assert template["model"] == f"{provider}/{raw['model']}"
    else:
        # the family parser already rewrote the body model to
        # {prefix}/{REQUEST model}; the template re-prefixes ON TOP of it
        assert (
            template["model"]
            == f"{provider}/{SPECS[provider].prefix}/{SPECS[provider].model}"
        )
    assert template["model"] != v1["model"], (
        provider,
        "the compat_sdk preset template stopped diverging — re-decide the "
        "F3 gate before relying on it",
    )


_INDEX_REWRITE_BODY = {
    "id": "resp-armdiverge",
    "object": "chat.completion",
    "created": 1718000000,
    "model": "wire-model-x",
    "choices": [
        {
            # The arm discriminator (verifier-longtail F1, the cleanest of
            # its 31 counterexamples): the openai (cdr) arm REBUILDS choices
            # under enumerate (index becomes 0); the openai_like arm's
            # ModelResponse(**body) keeps the verbatim wire index.
            "index": 5,
            "finish_reason": "stop",
            "logprobs": None,
            "message": {"role": "assistant", "content": "hello"},
        }
    ],
    "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
}

_OPENAI_LIKE_MEMBERS = sorted(
    provider
    for provider in PROVIDERS
    if RESPONSE_STYLES[provider] == "openai_like"
)


@pytest.mark.parametrize("provider", _OPENAI_LIKE_MEMBERS)
def test_construction_arms_diverge_on_wire_choice_index(
    provider: str, frozen_ambient
) -> None:
    """The behavioral half of the F3 gate on the STYLE axis (verifier-longtail
    F1 refuted the arms-coincide note this branch shipped with): a
    parser-admissible body where the two seam arms produce DIFFERENT dumps
    with a fresh ModelResponse and no preset. The correct (openai_like) arm
    equals v1 byte-for-byte and keeps the verbatim wire choice index; the
    openai arm's enumerate rebuild rewrites it to 0 — so inheriting the
    wrong construction arm is a live byte divergence the differential
    catches, not an incidental no-op."""
    raw = _INDEX_REWRITE_BODY
    v1 = run_v1_response_transform(provider, raw, SPECS[provider].model).model_dump()
    correct = _v2_with_style(provider, raw, "openai_like", ModelResponse())
    wrong = _v2_with_style(provider, raw, "openai", ModelResponse())
    assert _norm(correct) == _norm(v1), provider
    assert correct["choices"][0]["index"] == 5, provider
    assert wrong["choices"][0]["index"] == 0, provider
    assert _norm(wrong) != _norm(v1), (
        provider,
        "the seam arms stopped diverging on the index-rewrite body — the "
        "style-axis half of the F3 gate lost its discriminator; re-decide "
        "before relying on it",
    )


def _dialect_coupled_functions(source: str) -> list[str]:
    """Functions whose body contains BOTH a ``response_dialect(...)`` call
    AND a ``to_model_response(...)`` call — however the value travels
    (keyword, local variable, positional argument, attribute access).
    Deliberately COARSER than matching a ``usage_style=`` keyword whose
    value mentions the dialect helper: that predicate was disarmed by a
    local-variable refactor or a positional third argument
    (critic-longtail MAJOR-1; both dodges are pinned below)."""
    import ast as ast_mod

    tree = ast_mod.parse(source)
    offenders: list[str] = []
    for node in ast_mod.walk(tree):
        if not isinstance(node, (ast_mod.FunctionDef, ast_mod.AsyncFunctionDef)):
            continue
        called = set()
        for call in ast_mod.walk(node):
            if not isinstance(call, ast_mod.Call):
                continue
            if isinstance(call.func, ast_mod.Name):
                called.add(call.func.id)
            if isinstance(call.func, ast_mod.Attribute):
                called.add(call.func.attr)
        if {"response_dialect", "to_model_response"} <= called:
            offenders.append(node.name)
    return offenders


def test_seam_forks_never_select_usage_style_via_response_dialect() -> None:
    """The MECHANICAL half of the F3 gate: any completion() fork in
    litellm/translation_seam.py that BOTH calls ``response_dialect(...)``
    and constructs through ``to_model_response(...)`` fails this test
    unless it is the one audited bedrock fork (whose dialects are exactly
    the two bedrock construction styles). The compat_httpx fork MUST read
    ``compat_httpx.RESPONSE_STYLES`` instead (the CLAUDE.md HARD
    OBLIGATION); wiring it through the dialect helper makes this test red
    before any differential row can."""
    import inspect

    import litellm.translation_seam as seam

    offenders = _dialect_coupled_functions(inspect.getsource(seam))
    assert offenders == ["_send_v2_bedrock"], (
        "a seam fork couples response_dialect() with to_model_response(); "
        "the compat_httpx family MUST read compat_httpx.RESPONSE_STYLES "
        f"(verifier-wave1b F3). Offending functions: {offenders}"
    )


_DODGE_LOCAL_VARIABLE = '''
def _send_v2_family(provider_key, body, model_response):
    style = response_dialect(provider_key)
    return to_model_response(body, model_response, usage_style=style)
'''

_DODGE_POSITIONAL_ARGUMENT = '''
def _send_v2_family(provider_key, body, model_response):
    return to_model_response(body, model_response, response_dialect(provider_key))
'''


@pytest.mark.parametrize(
    "dodge",
    [_DODGE_LOCAL_VARIABLE, _DODGE_POSITIONAL_ARGUMENT],
    ids=["local-variable", "positional-argument"],
)
def test_construction_arm_gate_catches_the_critic_dodges(dodge: str) -> None:
    """critic-longtail MAJOR-1's two disarm simulations, frozen as negative
    tests: both formatting-level refactors of the dialect-shortcut fork
    must stay offenders under the gate's predicate."""
    assert _dialect_coupled_functions(dodge) == ["_send_v2_family"]


def test_usage_style_is_keyword_only() -> None:
    """The signature half of the MAJOR-1 fix: ``usage_style`` can never be
    passed positionally (dodge B is a TypeError at runtime and a pyright
    error at the call site, before the AST gate even runs)."""
    import inspect

    parameter = inspect.signature(to_model_response).parameters["usage_style"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
