"""Differential parity: v2 compat_sdk (wave-1a) vs the v1 SDK-path chain.

For every surviving wave-1a provider, the served corpus is compared row by
row against v1 in-process (``_compat_sdk_corpus.run_v1_request_transform``:
``get_optional_params(custom_llm_provider=p)`` + the resolved config's
``transform_request``, the chain main.py:2646 -> openai.py:727 runs). Rows
where v1's supported-list gate RAISES are pinned as raises + typed v2
fallbacks (the xai R2 pattern), and shapes the IR cannot round-trip are
typed fallbacks the seam serves through v1. The supported-list mirror drift
gates (critic-grok M4 pattern) re-derive every hand-copied allowlist from
the v1 configs at HEAD, over every model-map row where the provider has any.
"""

import copy
import json
from typing import get_args

import pytest

import litellm
from litellm.exceptions import UnsupportedParamsError

from litellm.translation import translate_chat_request
from litellm.translation.dispatch import Provider
from litellm.translation.engine import pipeline
from litellm.translation.providers import compat_sdk
from litellm.translation.providers.compat_sdk import params as csp

from ._compat_sdk_corpus import (
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
    return SPECS[provider]["model"]


# Rows where v1 RAISES UnsupportedParamsError (the supported-list gate); v2
# must be a typed fallback so v1 serves its own raise (or its drop under
# drop_params). The reason fragment is asserted on the v2 error; the raise
# is asserted on v1 in-process.
V1_RAISES = {}
for _p in PROVIDERS:
    _spec = SPECS[_p]
    if _spec["mct"] == "raise":
        V1_RAISES[f"{_p}:max_completion_tokens"] = (
            _p,
            {"model": _m(_p), "max_completion_tokens": 128, "messages": _USER},
            "max_completion_tokens",
        )
    if not _spec["tools"]:
        V1_RAISES[f"{_p}:tools"] = (
            _p,
            {"model": _m(_p), "tools": _TOOLS, "messages": _USER},
            "tools",
        )
    if not _spec["response_format"]:
        V1_RAISES[f"{_p}:response_format"] = (
            _p,
            {
                "model": _m(_p),
                "response_format": {"type": "json_object"},
                "messages": _USER,
            },
            "response_format",
        )
    if not _spec["parallel_tool_calls"]:
        V1_RAISES[f"{_p}:parallel_tool_calls"] = (
            _p,
            {"model": _m(_p), "parallel_tool_calls": False, "messages": _USER},
            "parallel_tool_calls",
        )
    # no provider in the family supports reasoning_effort except cerebras
    # (capability-gated, pinned below)
    if _p != "cerebras":
        V1_RAISES[f"{_p}:reasoning_effort"] = (
            _p,
            {"model": _m(_p), "reasoning_effort": "high", "messages": _USER},
            "reasoning_effort",
        )

V1_RAISES.update(
    {
        "cerebras:reasoning_effort_non_reasoning_model": (
            "cerebras",
            {"model": "llama3.1-8b", "reasoning_effort": "high", "messages": _USER},
            "reasoning_effort on non-reasoning cerebras model",
        ),
        "together_ai:tools_on_non_fc_model": (
            "together_ai",
            {
                "model": "Qwen/Qwen3-235B-A22B-fp8-tput",
                "tools": _TOOLS,
                "messages": _USER,
            },
            "tools",
        ),
        "together_ai:response_format_on_non_fc_model": (
            "together_ai",
            {
                "model": "Qwen/Qwen3-235B-A22B-fp8-tput",
                "response_format": {"type": "json_object"},
                "messages": _USER,
            },
            "response_format",
        ),
        "nvidia_nim:tools_on_gemma": (
            "nvidia_nim",
            {"model": "google/gemma-2-9b-it", "tools": _TOOLS, "messages": _USER},
            "tools",
        ),
        "nvidia_nim:max_completion_tokens_on_gemma": (
            "nvidia_nim",
            {
                "model": "google/gemma-2-9b-it",
                "max_completion_tokens": 7,
                "messages": _USER,
            },
            "max_completion_tokens",
        ),
        "nvidia_nim:stop_on_nemotron_instruct": (
            "nvidia_nim",
            {
                "model": "nvidia/nemotron-4-340b-instruct",
                "stop": ["END"],
                "messages": _USER,
            },
            "stop",
        ),
        "nvidia_nim:temperature_on_nemotron_reward": (
            "nvidia_nim",
            {
                "model": "nvidia/nemotron-4-340b-reward",
                "temperature": 0.1,
                "messages": _USER,
            },
            "temperature",
        ),
        "lm_studio:response_format_on_gpt4_name": (
            # the base supported list's gpt-4 name gate applies to every
            # config that inherits it, local providers included
            "lm_studio",
            {
                "model": "gpt-4",
                "response_format": {"type": "json_object"},
                "messages": _USER,
            },
            "outside v1's supported set",
        ),
    }
)

# Typed fallbacks where v1 SERVES the request (v1 is not invoked: the seam
# routes these to v1 untouched). Shared raw-guard/parse shapes per provider
# plus the provider-specific ones.
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
    if not SPECS[_p]["user"]:
        EXPECTED_FALLBACKS[f"{_p}:user_model_list_gate"] = (
            _p,
            {"model": _m(_p), "user": "u-1", "messages": _USER},
            "user",
        )

EXPECTED_FALLBACKS.update(
    {
        "volcengine:thinking_extra_body_packing": (
            "volcengine",
            {
                "model": _m("volcengine"),
                "thinking": {"type": "enabled"},
                "messages": _USER,
            },
            "extra_body",
        ),
        "lm_studio:bare_schema_response_format": (
            # v1's LMStudioChatConfig map wraps {"type": "json_schema",
            # "schema": ...} into a json_schema envelope; the inbound parse
            # rejects the non-canonical key, so v1 serves it
            "lm_studio",
            {
                "model": _m("lm_studio"),
                "response_format": {
                    "type": "json_schema",
                    "schema": {"type": "object"},
                },
                "messages": _USER,
            },
            "response_format",
        ),
    }
)


def _v2(provider: str, case: dict):
    return translate_chat_request(copy.deepcopy(case), provider, build_real_deps())


def _norm(body: dict) -> str:
    return json.dumps(body, sort_keys=True, default=str)


def _request_rows():
    return sorted(
        (provider, name) for provider in PROVIDERS for name in corpus_for(provider)
    )


def test_registered_providers_have_differential_coverage() -> None:
    """Green-but-untested registration is impossible (verifier-wave1a F3):
    every dispatch Literal member is registered in every pipeline table,
    the compat_sdk family registry equals the corpus SPECS exactly, and a
    provider outside the family must be in the dedicated-gates set below.
    Add a provider to that set ONLY in the commit that adds its
    differential corpus, naming the gate files."""
    providers = set(get_args(Provider))
    assert providers == set(pipeline._SERIALIZERS)
    assert providers == set(pipeline._RESPONSE_PARSERS)
    assert providers == set(pipeline._RESPONSE_DIALECTS)
    assert set(pipeline._RAW_GUARDS) <= providers
    assert set(compat_sdk.PROFILES) == set(SPECS)
    assert set(compat_sdk.ALLOWED) == set(SPECS)
    dedicated_gates = {
        "anthropic",  # test_differential_anthropic_{request,response,stream}
        "bedrock_converse",  # test_differential_bedrock_*
        "bedrock_invoke",  # test_differential_bedrock_*
        "openai_compat",  # test_differential_openai_*
        "vertex_ai",  # test_differential_google_*
        "gemini",  # test_differential_google_*
        "vertex_anthropic",  # test_differential_google_*
        "azure",  # test_differential_azure_*
        "azure_ai",  # test_differential_azure_ai_request + azure stream/response
        "azure_ai_anthropic",  # test_differential_azure_ai_request (Claude route)
        "xai",  # test_differential_xai_*
    }
    assert providers == dedicated_gates | set(SPECS)


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


# ---------------------------------------------------------------------------
# Supported-list mirror drift gates (critic-grok M4 pattern): the hand-copied
# allowlists in compat_sdk/params.py must track the v1 configs at HEAD, for
# EVERY chat model the model map knows (fixed name samples where the map has
# no rows: nvidia_nim's static table models, local lm_studio/llamafile).
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


def _base_family_allowed(model: str) -> frozenset:
    allowed = csp._BASE_LIST
    if model in _RF_NAME_GATED:
        return allowed - {"response_format"}
    return allowed


def _v2_allowed(provider: str, model: str, deps) -> frozenset:
    """The per-model allowed set, re-derived from the SAME csp.ALLOWED table
    serialization reads, with the three per-model narrowings the gates
    apply (together capability fork, nvidia_nim static table, the base
    list's gpt-4 name gate)."""
    if provider == "together_ai":
        allowed = (
            _base_family_allowed(model)
            if csp.supports_together_tools(model, deps)
            else _base_family_allowed(model) - csp._FUNCTION_CALLING_KEYS
        )
        return allowed
    if provider == "nvidia_nim":
        return csp.nvidia_nim_allowed(model)
    allowed = csp.ALLOWED[provider]
    if allowed == csp._BASE_LIST:
        return _base_family_allowed(model)
    return allowed


_SAMPLE_MODELS = {
    "nvidia_nim": (
        "google/recurrentgemma-2b",
        "google/gemma-2-27b-it",
        "google/gemma-2-9b-it",
        "gemma-2-9b-it",
        "nvidia/nemotron-4-340b-instruct",
        "nvidia/nemotron-4-340b-reward",
        "google/codegemma-1.1-7b",
        "meta/llama3-70b-instruct",
        "mistralai/mistral-large",
    ),
    "lm_studio": ("qwen2.5-7b-instruct-1m", "gpt-4", "gpt-3.5-turbo-16k"),
    "llamafile": ("LLaMA_CPP", "gpt-4"),
}


def _mirror_models(provider: str) -> list:
    mapped = sorted(
        key.split("/", 1)[1]
        for key, info in litellm.model_cost.items()
        if key.startswith(f"{provider}/") and info.get("mode") == "chat"
    )
    return list(_SAMPLE_MODELS.get(provider, ())) + mapped


@pytest.mark.parametrize("provider", PROVIDERS)
def test_supported_list_mirrors_track_v1_at_head(provider: str) -> None:
    deps = build_real_deps()
    models = _mirror_models(provider)
    assert len(models) >= 2, f"no models to mirror for {provider}"
    for model in models:
        supported = set(
            provider_config(provider, model).get_supported_openai_params(model)
        )
        allowed = _v2_allowed(provider, model, deps)
        # The mirror's soundness rests on a DIRECTION argument: every key v2
        # can ever SERVE is either in _MIRROR_KEYS or carries its own assert
        # below (user, reasoning_effort), so a v1 list growing a key v2 lacks
        # only widens the typed fallback (v1 serves its own new behavior).
        # Encode the subset relation so a future allowed set cannot silently
        # leave that argument (critic-wave1a N6).
        assert allowed <= set(_MIRROR_KEYS) | {"user", "reasoning_effort"}, (
            provider,
            model,
        )
        for key in _MIRROR_KEYS:
            assert (key in allowed) == (key in supported), (provider, model, key)
        if SPECS[provider]["user"]:
            assert "user" in supported, (provider, model)
        if provider == "cerebras":
            assert csp.supports_cerebras_reasoning(model, deps) == (
                "reasoning_effort" in supported
            ), model
        else:
            assert "reasoning_effort" not in supported, (provider, model)


def test_capability_prefix_is_load_bearing() -> None:
    """Bare model-map keys answer False even for capable models; the
    {provider}/ prefix is what reaches the map row (the xai drift-gate
    trap, re-pinned here for the two capability gates this family uses)."""
    deps = build_real_deps()
    assert deps.supports_capability(
        "together_ai/Qwen/Qwen2.5-72B-Instruct-Turbo", "supports_function_calling"
    )
    assert not deps.supports_capability(
        "Qwen/Qwen2.5-72B-Instruct-Turbo", "supports_function_calling"
    )
    assert deps.supports_capability("cerebras/qwen-3-32b", "supports_reasoning")
    assert not deps.supports_capability("qwen-3-32b", "supports_reasoning")
    assert csp.supports_together_tools(
        "Qwen/Qwen2.5-72B-Instruct-Turbo", deps
    ) == litellm.utils.supports_function_calling(
        "Qwen/Qwen2.5-72B-Instruct-Turbo", custom_llm_provider="together_ai"
    )
    assert csp.supports_cerebras_reasoning(
        "qwen-3-32b", deps
    ) == litellm.supports_reasoning(model="qwen-3-32b", custom_llm_provider="cerebras")


@pytest.mark.parametrize(
    "provider", [p for p in PROVIDERS if SPECS[p]["mct"] == "rename"]
)
def test_mct_rename_matches_v1(provider: str) -> None:
    """The renamed key must be exactly what v1's map emits (max_tokens), so
    the rename flag can never silently disagree with the v1 config."""
    case = {"model": _m(provider), "max_completion_tokens": 33, "messages": _USER}
    v1 = run_v1_request_transform(provider, case)
    assert v1.get("max_tokens") == 33 and "max_completion_tokens" not in v1
    result = _v2(provider, case)
    assert result.is_ok(), result.error.summary
    assert result.ok.get("max_tokens") == 33
    assert "max_completion_tokens" not in result.ok


@pytest.mark.parametrize(
    "provider", [p for p in PROVIDERS if SPECS[p]["mct"] == "verbatim"]
)
def test_mct_verbatim_matches_v1(provider: str) -> None:
    case = {"model": _m(provider), "max_completion_tokens": 33, "messages": _USER}
    v1 = run_v1_request_transform(provider, case)
    assert v1.get("max_completion_tokens") == 33 and "max_tokens" not in v1
    result = _v2(provider, case)
    assert result.is_ok(), result.error.summary
    assert result.ok.get("max_completion_tokens") == 33
    assert "max_tokens" not in result.ok


def test_together_text_response_format_dropped_like_v1() -> None:
    case = {
        "model": _m("together_ai"),
        "response_format": {"type": "text"},
        "messages": _USER,
    }
    v1 = run_v1_request_transform("together_ai", case)
    assert "response_format" not in v1
    result = _v2("together_ai", case)
    assert result.is_ok(), result.error.summary
    assert "response_format" not in result.ok
    assert _norm(result.ok) == _norm(v1)


def test_cerebras_reasoning_effort_served_on_capable_model() -> None:
    case = {"model": "qwen-3-32b", "reasoning_effort": "low", "messages": _USER}
    v1 = run_v1_request_transform("cerebras", case)
    assert v1.get("reasoning_effort") == "low"
    result = _v2("cerebras", case)
    assert result.is_ok(), result.error.summary
    assert _norm(result.ok) == _norm(v1)


def test_nvidia_gemma_arm_serves_its_reduced_list() -> None:
    case = {
        "model": "google/gemma-2-9b-it",
        "temperature": 0.5,
        "top_p": 0.9,
        "max_tokens": 32,
        "stop": ["END"],
        "messages": _USER,
    }
    result = _v2("nvidia_nim", case)
    assert result.is_ok(), result.error.summary
    assert _norm(result.ok) == _norm(run_v1_request_transform("nvidia_nim", case))
