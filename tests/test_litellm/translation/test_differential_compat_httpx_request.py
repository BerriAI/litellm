"""Differential parity: v2 compat_httpx (wave-1b) vs the v1 httpx-path chain.

Same gate shape as the compat_sdk request differential, with the httpx
deltas: the v1 reference invoker runs each provider's LIVE
``transform_request`` (dedicated elifs into ``base_llm_http_handler``), the
raise rows pin v1's UnsupportedParamsError in-process (including
gradient_ai's OWN map-level raise on ``user``, which ``_check_valid_arg``
skips), and the mirror drift gates re-derive every hand-copied allowlist
from the v1 configs at HEAD. lemonade is excluded from the config mirror —
its chat config is unregistered at HEAD so the LIVE param surface is the
generic openai fallback list; the lemonade facts canary pins that state
instead.
"""

import copy
import json

import pytest

import litellm
from litellm.exceptions import UnsupportedParamsError

from litellm.translation import translate_chat_request
from litellm.translation.providers.compat_httpx import params as chp

from ._compat_httpx_corpus import (
    PROVIDERS,
    SPECS,
    WEATHER_TOOL,
    corpus_for,
    provider_config,
    run_v1_request_transform,
)
from .conftest import build_real_deps

_USER = [{"role": "user", "content": "x"}]
_TOOLS = [WEATHER_TOOL]


def _m(provider: str) -> str:
    return SPECS[provider].model


V1_RAISES = {}
for _p in PROVIDERS:
    _spec = SPECS[_p]
    if not _spec.tools:
        V1_RAISES[f"{_p}:tools"] = (
            _p,
            {"model": _m(_p), "tools": _TOOLS, "messages": _USER},
            "tools",
        )
    if not _spec.response_format:
        V1_RAISES[f"{_p}:response_format"] = (
            _p,
            {
                "model": _m(_p),
                "response_format": {"type": "json_object"},
                "messages": _USER,
            },
            "response_format",
        )
    if not _spec.parallel_tool_calls:
        V1_RAISES[f"{_p}:parallel_tool_calls"] = (
            _p,
            {"model": _m(_p), "parallel_tool_calls": False, "messages": _USER},
            "parallel_tool_calls",
        )
    if _spec.reasoning == "none":
        V1_RAISES[f"{_p}:reasoning_effort"] = (
            _p,
            {"model": _m(_p), "reasoning_effort": "high", "messages": _USER},
            "reasoning_effort",
        )

V1_RAISES.update(
    {
        "bedrock_mantle:reasoning_effort_non_reasoning_model": (
            "bedrock_mantle",
            {
                "model": "unflagged-model",
                "reasoning_effort": "high",
                "messages": _USER,
            },
            "reasoning_effort on non-reasoning bedrock_mantle model",
        ),
        "gradient_ai:user_map_level_raise": (
            "gradient_ai",
            {"model": _m("gradient_ai"), "user": "u1", "messages": _USER},
            "user on gradient_ai",
        ),
    }
)

EXPECTED_FALLBACKS = {}
for _p in PROVIDERS:
    EXPECTED_FALLBACKS[f"{_p}:explicit_stream_false"] = (
        _p,
        {"model": _m(_p), "stream": False, "messages": _USER},
        "explicit stream: false",
    )
    EXPECTED_FALLBACKS[f"{_p}:both_max_tokens_keys"] = (
        _p,
        {
            "model": _m(_p),
            "max_tokens": 5,
            "max_completion_tokens": 6,
            "messages": _USER,
        },
        "both max_tokens and max_completion_tokens",
    )
    EXPECTED_FALLBACKS[f"{_p}:string_form_stop"] = (
        _p,
        {"model": _m(_p), "stop": "END", "messages": _USER},
        "string-form stop",
    )
    EXPECTED_FALLBACKS[f"{_p}:message_name_field"] = (
        _p,
        {
            "model": _m(_p),
            "messages": [{"role": "user", "content": "x", "name": "alice"}],
        },
        "message name field",
    )
    EXPECTED_FALLBACKS[f"{_p}:seed_outside_ir"] = (
        _p,
        {"model": _m(_p), "seed": 42, "messages": _USER},
        "seed",
    )
    if _p != "gradient_ai":  # gradient's user RAISE is a V1_RAISES row above
        EXPECTED_FALLBACKS[f"{_p}:user_model_list_gate"] = (
            _p,
            {"model": _m(_p), "user": "u-1", "messages": _USER},
            "user",
        )

# cometapi's top_k extra_body crossing is wire-proven (verifier-wave2a §5:
# get_optional_params packs it, CometAPIConfig.transform_request merges it);
# the other nine members inherit the same default reason arm but have no
# pinned row (their wire merge is unverified — do not add rows claiming it).
EXPECTED_FALLBACKS["cometapi:top_k_extra_body"] = (
    "cometapi",
    {"model": _m("cometapi"), "top_k": 3, "messages": _USER},
    "extra_body",
)

EXPECTED_FALLBACKS.update(
    {
        "heroku:content_list_flatten": (
            "heroku",
            {
                "model": _m("heroku"),
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "a"},
                            {"type": "text", "text": "b"},
                        ],
                    }
                ],
            },
            "list-form message content",
        ),
        "minimax:cache_control_preserved": (
            "minimax",
            {
                "model": _m("minimax"),
                "messages": [
                    {
                        "role": "user",
                        "content": "x",
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            },
            "cache_control",
        ),
        "minimax:thinking_verbatim_copy": (
            "minimax",
            {
                "model": "MiniMax-M2.1",
                "thinking": {"type": "enabled"},
                "messages": _USER,
            },
            "thinking on minimax",
        ),
    }
)


def _v2(provider: str, case: dict[str, object]):
    return translate_chat_request(copy.deepcopy(case), provider, build_real_deps())


def _norm(body: dict) -> str:
    return json.dumps(body, sort_keys=True, default=str)


def _request_rows():
    return sorted(
        (provider, name) for provider in PROVIDERS for name in corpus_for(provider)
    )


@pytest.mark.parametrize("provider,name", _request_rows())
def test_v2_request_matches_v1(provider: str, name: str) -> None:
    case = corpus_for(provider)[name]
    result = _v2(provider, case)
    assert result.is_ok(), result.error.summary
    assert _norm(result.ok) == _norm(run_v1_request_transform(provider, case))


@pytest.mark.parametrize("name", sorted(V1_RAISES))
def test_v1_raise_rows_fall_back_typed(name: str) -> None:
    provider, case, reason_fragment = V1_RAISES[name]
    result = _v2(provider, case)
    assert result.is_error(), f"{name} unexpectedly translated: {result.ok!r}"
    assert reason_fragment in result.error.summary, result.error.summary
    with pytest.raises(UnsupportedParamsError):
        run_v1_request_transform(provider, case)


@pytest.mark.parametrize("name", sorted(EXPECTED_FALLBACKS))
def test_unsupported_shape_is_a_typed_fallback(name: str) -> None:
    provider, case, reason_fragment = EXPECTED_FALLBACKS[name]
    result = _v2(provider, case)
    assert result.is_error(), f"{name} unexpectedly translated: {result.ok!r}"
    assert reason_fragment in result.error.summary, result.error.summary


@pytest.mark.parametrize("provider", [p for p in PROVIDERS if SPECS[p].mct == "rename"])
def test_mct_rename_matches_v1(provider: str) -> None:
    case = {"model": _m(provider), "max_completion_tokens": 33, "messages": _USER}
    v1 = run_v1_request_transform(provider, case)
    assert v1.get("max_tokens") == 33 and "max_completion_tokens" not in v1
    result = _v2(provider, case)
    assert result.is_ok(), result.error.summary
    assert result.ok.get("max_tokens") == 33
    assert "max_completion_tokens" not in result.ok


@pytest.mark.parametrize(
    "provider", [p for p in PROVIDERS if SPECS[p].mct == "verbatim"]
)
def test_mct_verbatim_matches_v1(provider: str) -> None:
    case = {"model": _m(provider), "max_completion_tokens": 33, "messages": _USER}
    v1 = run_v1_request_transform(provider, case)
    assert v1.get("max_completion_tokens") == 33 and "max_tokens" not in v1
    result = _v2(provider, case)
    assert result.is_ok(), result.error.summary
    assert result.ok.get("max_completion_tokens") == 33
    assert "max_tokens" not in result.ok


# ---------------------------------------------------------------------------
# Supported-list mirror drift gates (the compat_sdk M4 pattern), over every
# {provider}/... chat row the model map knows plus fixed samples; lemonade
# excluded (unregistered config — facts canary below).
# ---------------------------------------------------------------------------

_MIRROR_KEYS = (
    "max_tokens",
    "max_completion_tokens",
    "temperature",
    "top_p",
    "stream",
    "stop",
    "tools",
    "tool_choice",
    "parallel_tool_calls",
    "response_format",
)

_RF_NAME_GATED = ("gpt-4", "gpt-3.5-turbo-16k")

_SAMPLE_MODELS = {
    "compactifai": ("cai-llama-3-1-8b-slim", "gpt-4"),
    "amazon_nova": ("nova-pro", "nova-lite"),
    "datarobot": ("datarobot-deployed-llm", "gpt-4"),
    "heroku": ("claude-3-7-sonnet", "gpt-4"),
    "bedrock_mantle": ("openai.gpt-oss-120b", "gpt-4"),
    "minimax": ("MiniMax-M2", "gpt-4"),
    "gradient_ai": ("llama3.3-70b-instruct",),
    "ovhcloud": ("Meta-Llama-3_1-70B-Instruct", "gpt-4"),
    # cometapi has no model-map rows at HEAD; fixed names cover the base
    # list, the gpt-4 rf name gate, and a non-openai name
    "cometapi": ("gpt-4o-mini", "gpt-4", "claude-sonnet-4-5"),
}

_MIRROR_PROVIDERS = tuple(p for p in PROVIDERS if p != "lemonade")


def _v2_allowed(provider: str, model: str, deps) -> frozenset[str]:
    allowed = chp.ALLOWED[provider]
    if provider == "bedrock_mantle" and not chp.supports_bedrock_mantle_reasoning(
        model, deps
    ):
        allowed = allowed - {"reasoning_effort"}
    if (
        "response_format" in allowed
        and provider not in ("amazon_nova", "gradient_ai")
        and model in _RF_NAME_GATED
    ):
        # the base/OpenAILike supported list drops response_format for these
        # names; amazon_nova and gradient_ai carry their own static lists
        allowed = allowed - {"response_format"}
    return allowed


def _mirror_models(provider: str) -> list:
    mapped = sorted(
        key.split("/", 1)[1]
        for key, info in litellm.model_cost.items()
        if key.startswith(f"{provider}/") and info.get("mode") == "chat"
    )
    return list(_SAMPLE_MODELS.get(provider, ())) + mapped


@pytest.mark.parametrize("provider", _MIRROR_PROVIDERS)
def test_supported_list_mirrors_track_v1_at_head(provider: str) -> None:
    deps = build_real_deps()
    models = _mirror_models(provider)
    assert len(models) >= 2 or provider == "gradient_ai", provider
    for model in models:
        supported = set(
            provider_config(provider, model).get_supported_openai_params(model)
        )
        allowed = _v2_allowed(provider, model, deps)
        assert allowed <= set(_MIRROR_KEYS) | {"reasoning_effort"}, (provider, model)
        for key in _MIRROR_KEYS:
            assert (key in allowed) == (key in supported), (provider, model, key)
        if provider == "bedrock_mantle":
            assert chp.supports_bedrock_mantle_reasoning(model, deps) == (
                "reasoning_effort" in supported
            ), model
        elif provider == "amazon_nova":
            assert "reasoning_effort" in supported, model
        else:
            assert "reasoning_effort" not in supported, (provider, model)
        if provider == "gradient_ai":
            # the map-level user raise depends on user staying OUTSIDE the
            # list (otherwise the gate note above goes stale)
            assert "user" not in supported, model


def test_lemonade_facts_canary() -> None:
    """lemonade's dual-path state at HEAD: chat-config resolution is None
    (param time rides the generic openai fallback list + the OpenAILike
    else-arm, so mct RENAMES), while completion() threads an explicit
    ``LemonadeChatConfig()`` for the transforms. If the resolution half
    fails, upstream registered the config — the IR-visible surface stays the
    same (base list + the OpenAILike super rename), but re-verify and drop
    this canary deliberately."""
    from litellm.llms.lemonade.chat.transformation import LemonadeChatConfig
    from litellm.main import lemonade_transformation
    from litellm.types.utils import LlmProviders
    from litellm.utils import ProviderConfigManager

    resolved = ProviderConfigManager.get_provider_chat_config(
        model=_m("lemonade"), provider=LlmProviders("lemonade")
    )
    assert resolved is None, "LemonadeChatConfig got registered; re-verify the mirror"
    assert isinstance(lemonade_transformation, LemonadeChatConfig)
    v1 = run_v1_request_transform(
        "lemonade",
        {"model": _m("lemonade"), "max_completion_tokens": 9, "messages": _USER},
    )
    assert v1.get("max_tokens") == 9 and "max_completion_tokens" not in v1
