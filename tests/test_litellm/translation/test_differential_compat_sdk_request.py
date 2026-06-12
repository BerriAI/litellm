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
    return SPECS[provider].model


# Rows where v1 RAISES UnsupportedParamsError (the supported-list gate); v2
# must be a typed fallback so v1 serves its own raise (or its drop under
# drop_params). The reason fragment is asserted on the v2 error; the raise
# is asserted on v1 in-process.
V1_RAISES = {}
for _p in PROVIDERS:
    _spec = SPECS[_p]
    if _spec.mct == "raise":
        V1_RAISES[f"{_p}:max_completion_tokens"] = (
            _p,
            {"model": _m(_p), "max_completion_tokens": 128, "messages": _USER},
            "max_completion_tokens",
        )
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
    if not _spec.top_p:
        V1_RAISES[f"{_p}:top_p"] = (
            _p,
            {"model": _m(_p), "top_p": 0.9, "messages": _USER},
            "top_p",
        )
    if not _spec.temperature:
        V1_RAISES[f"{_p}:temperature"] = (
            _p,
            {"model": _m(_p), "temperature": 0.4, "messages": _USER},
            "temperature",
        )
    if not _spec.stop:
        V1_RAISES[f"{_p}:stop"] = (
            _p,
            {"model": _m(_p), "stop": ["END"], "messages": _USER},
            "stop",
        )
    if not _spec.max_tokens:
        V1_RAISES[f"{_p}:max_tokens"] = (
            _p,
            {"model": _m(_p), "max_tokens": 16, "messages": _USER},
            "max_tokens",
        )
    # reasoning_effort raises everywhere except the providers whose ALLOWED
    # carries it (cerebras capability-gated on a model the corpus row lacks;
    # inception unconditional — its served row is pinned below)
    if "reasoning_effort" not in csp.ALLOWED[_p]:
        V1_RAISES[f"{_p}:reasoning_effort"] = (
            _p,
            {"model": _m(_p), "reasoning_effort": "high", "messages": _USER},
            "reasoning_effort",
        )
    # cerebras' capability-narrowed raise keeps its explicit row below;
    # inception's unconditional serve is pinned by its own test below

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

# Rows where v1 CRASHES with a bare ValueError BEFORE any param gate:
# together's non-fc fork runs list.remove("response_format") on a base list
# that already dropped the key for these model names, inside the
# get_supported_openai_params call _check_valid_arg makes on EVERY request
# (verifier-wave1a F2). v2 must fall back typed on every shape, plain text
# included, so v1 raises its own error instead of v2 serving what v1 crashes
# on.
V1_RAISES_VALUE_ERROR = {
    "together_ai:plain_text_on_gpt4_name": (
        "together_ai",
        {"model": "gpt-4", "messages": _USER},
        "ValueError",
    ),
    "together_ai:plain_text_on_gpt35_16k_name": (
        "together_ai",
        {"model": "gpt-3.5-turbo-16k", "messages": _USER},
        "ValueError",
    ),
    "together_ai:temperature_only_on_gpt4_name": (
        "together_ai",
        {"model": "gpt-4", "temperature": 0.2, "messages": _USER},
        "ValueError",
    ),
    "together_ai:response_format_on_gpt4_name": (
        "together_ai",
        {
            "model": "gpt-4",
            "response_format": {"type": "json_object"},
            "messages": _USER,
        },
        "ValueError",
    ),
}

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
    if not SPECS[_p].user:
        EXPECTED_FALLBACKS[f"{_p}:user_model_list_gate"] = (
            _p,
            {"model": _m(_p), "user": "u-1", "messages": _USER},
            "user",
        )

EXPECTED_FALLBACKS.update(
    {
        # wave-1b per-provider guard arms (each names the v1 path)
        "dashscope:cache_control_preserved": (
            "dashscope",
            {
                "model": _m("dashscope"),
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
        "zai:cache_control_preserved": (
            "zai",
            {
                "model": _m("zai"),
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
        "docker_model_runner:content_list_flatten": (
            "docker_model_runner",
            {
                "model": _m("docker_model_runner"),
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
        "publicai:content_list_flatten": (
            "publicai",
            {
                "model": _m("publicai"),
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
        "zai:thinking_verbatim_copy": (
            "zai",
            {
                "model": "glm-5",
                "thinking": {"type": "enabled"},
                "messages": _USER,
            },
            "thinking on zai",
        ),
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


def _v2(provider: str, case: dict[str, object]):
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
    each family registry equals its corpus SPECS exactly, and a provider
    outside the families must be in the dedicated-gates set below. Add a
    provider to that set ONLY in the commit that adds its differential
    corpus, naming the gate files."""
    from litellm.translation.providers import compat_httpx

    from ._compat_httpx_corpus import SPECS as HTTPX_SPECS

    providers = set(get_args(Provider))
    assert providers == set(pipeline._SERIALIZERS)
    assert providers == set(pipeline._RESPONSE_PARSERS)
    assert providers == set(pipeline._RESPONSE_DIALECTS)
    assert set(pipeline._RAW_GUARDS) <= providers
    assert set(compat_sdk.PROFILES) == set(SPECS)
    assert set(compat_sdk.ALLOWED) == set(SPECS)
    assert set(compat_sdk.GUARDS) == set(SPECS)
    assert set(compat_httpx.PROFILES) == set(HTTPX_SPECS)
    assert set(compat_httpx.ALLOWED) == set(HTTPX_SPECS)
    assert set(compat_httpx.PARSERS) == set(HTTPX_SPECS)
    assert set(compat_httpx.GUARDS) == set(HTTPX_SPECS)
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
    assert providers == dedicated_gates | set(SPECS) | set(HTTPX_SPECS)


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


@pytest.mark.parametrize("name", sorted(V1_RAISES_VALUE_ERROR))
def test_v1_value_error_rows_fall_back_typed(name: str) -> None:
    """The together gpt-4-name corner: v1 crashes with a bare ValueError on
    EVERY request for these names (not UnsupportedParamsError), asserted
    in-process; v2 serving any of them would be a parity break."""
    provider, case, reason_fragment = V1_RAISES_VALUE_ERROR[name]
    result = _v2(provider, case)
    assert result.is_error(), f"{name} unexpectedly translated: {result.ok!r}"
    assert reason_fragment in result.error.summary, result.error.summary
    with pytest.raises(ValueError) as excinfo:
        run_v1_request_transform(provider, case)
    assert type(excinfo.value) is ValueError, excinfo.value


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


def _base_family_allowed(model: str) -> frozenset[str]:
    allowed = csp._BASE_LIST
    if model in _RF_NAME_GATED:
        return allowed - {"response_format"}
    return allowed


def _v2_allowed(provider: str, model: str, deps) -> frozenset[str]:
    """The per-model allowed set, re-derived from the SAME csp.ALLOWED table
    serialization reads, with the per-model narrowings the gates apply
    (together capability fork, nvidia_nim static table, the JSON-registry
    function-calling fork, the base list's gpt-4 name gate)."""
    if provider == "together_ai":
        allowed = (
            _base_family_allowed(model)
            if csp.supports_together_tools(model, deps)
            else _base_family_allowed(model) - csp._FUNCTION_CALLING_KEYS
        )
        return allowed
    if provider == "nvidia_nim":
        return csp.nvidia_nim_allowed(model)
    if provider in csp.JSON_REGISTRY_PROVIDERS:
        allowed = _base_family_allowed(model)
        if not csp.supports_json_provider_tools(model, deps, provider):
            allowed = allowed - csp._JSON_TOOL_KEYS
        return allowed
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
    # wave-1b providers with no (or too few) model-map chat rows: fixed name
    # samples; "gpt-4" exercises the base list's response_format name gate
    # for the configs that inherit it
    "ai21_chat": ("jamba-large-1.7", "jamba-mini-1.7"),
    "docker_model_runner": ("ai/llama3.1", "ai/smollm2"),
    "empower": ("empower-functions", "empower-functions-small"),
    "galadriel": ("llama3.1", "llama3.1-70b"),
    "github": ("Llama-3.2-90B-Vision-Instruct", "Meta-Llama-3.1-405B-Instruct"),
    "inception": ("mercury-2", "mercury-coder"),
    "dashscope": ("qwen-flash", "gpt-4"),
    "vercel_ai_gateway": ("openai/gpt-4o", "gpt-4"),
    "meta_llama": ("Llama-4-Maverick-17B-128E-Instruct-FP8", "gpt-4"),
    "helicone": ("llama-3.3-70b", "gpt-4"),
    "xiaomi_mimo": ("mimo-7b", "gpt-4"),
    "scaleway": ("llama-3.3-70b-instruct", "gpt-4"),
    "synthetic": ("deepseek-v3", "gpt-4"),
    "apertis": ("apertis-large", "gpt-4"),
    "nano-gpt": ("llama-3.3-70b-instruct", "gpt-4"),
    "poe": ("claude-sonnet-4", "gpt-4"),
    "chutes": ("deepseek-ai/DeepSeek-V3", "gpt-4"),
    "assemblyai": ("assembly-best", "gpt-4"),
    "charity_engine": ("llama-3.1-8b", "gpt-4"),
    "neosantara": ("nusantara-base", "gpt-4"),
    "tensormesh": ("tm-llama-3.1-8b", "gpt-4"),
    "parasail": ("parasail-llama-33-70b", "gpt-4"),
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
        if SPECS[provider].user:
            assert "user" in supported, (provider, model)
        if provider == "cerebras":
            assert csp.supports_cerebras_reasoning(model, deps) == (
                "reasoning_effort" in supported
            ), model
        elif provider == "inception":
            # unconditional in InceptionChatConfig's static list
            assert "reasoning_effort" in supported, model
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


@pytest.mark.parametrize("provider", [p for p in PROVIDERS if SPECS[p].mct == "rename"])
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


def test_top_k_fallback_reason_matches_v1_silent_drop() -> None:
    """v1 silently drops top_k WITHOUT drop_params (it is not an OpenAI
    param, so _check_valid_arg never sees it) — the fallback reason must say
    so instead of claiming an UnsupportedParamsError raise (verifier-wave1a
    F6). Pinned in-process: v1 serves the request with top_k gone."""
    case = {"model": _m("lambda_ai"), "top_k": 3, "messages": _USER}
    result = _v2("lambda_ai", case)
    assert result.is_error()
    assert "silently drops it" in result.error.summary, result.error.summary
    assert "UnsupportedParamsError" not in result.error.summary
    v1 = run_v1_request_transform("lambda_ai", case)
    assert "top_k" not in v1


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


# ---------------------------------------------------------------------------
# wave-1b pins and canaries
# ---------------------------------------------------------------------------


def test_meta_llama_non_json_schema_response_format_dropped_like_v1() -> None:
    """LlamaAPIConfig's map POPS response_format unless type == json_schema;
    json_object is silently dropped on both sides, json_schema passes."""
    dropped = {
        "model": _m("meta_llama"),
        "response_format": {"type": "json_object"},
        "messages": _USER,
    }
    v1 = run_v1_request_transform("meta_llama", dropped)
    assert "response_format" not in v1
    result = _v2("meta_llama", dropped)
    assert result.is_ok(), result.error.summary
    assert "response_format" not in result.ok
    assert _norm(result.ok) == _norm(v1)
    kept = {
        "model": _m("meta_llama"),
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "a", "schema": {"type": "object"}, "strict": True},
        },
        "messages": _USER,
    }
    v1_kept = run_v1_request_transform("meta_llama", kept)
    assert "response_format" in v1_kept
    result_kept = _v2("meta_llama", kept)
    assert result_kept.is_ok(), result_kept.error.summary
    assert _norm(result_kept.ok) == _norm(v1_kept)


def test_inception_reasoning_effort_served_unconditionally() -> None:
    case = {"model": "mercury-2", "reasoning_effort": "low", "messages": _USER}
    v1 = run_v1_request_transform("inception", case)
    assert v1.get("reasoning_effort") == "low"
    result = _v2("inception", case)
    assert result.is_ok(), result.error.summary
    assert _norm(result.ok) == _norm(v1)


def test_publicai_tools_served_on_fc_capable_model() -> None:
    """The dynamic JSON config's function-calling fork serves tools when the
    model map carries supports_function_calling for {slug}/{model}."""
    case = {
        "model": "aisingapore/Gemma-SEA-LION-v4-27B-IT",
        "tools": _TOOLS,
        "messages": _USER,
    }
    v1 = run_v1_request_transform("publicai", case)
    assert v1.get("tools")
    result = _v2("publicai", case)
    assert result.is_ok(), result.error.summary
    assert _norm(result.ok) == _norm(v1)


def test_json_rename_set_mirrors_providers_json_at_head() -> None:
    """The _JSON_RENAME frozenset in compat_sdk/serialize.py must track
    providers.json param_mappings (the mapping arm runs FIRST in the dynamic
    config's map), and the registered JSON cohort must be exactly the enum
    members of the registry at HEAD."""
    import json as jsonlib
    from pathlib import Path

    import litellm as litellm_module
    from litellm.types.utils import LlmProviders

    from litellm.translation.providers.compat_sdk.serialize import _JSON_RENAME

    registry_path = (
        Path(litellm_module.__file__).parent / "llms" / "openai_like" / "providers.json"
    )
    registry = jsonlib.loads(registry_path.read_text())
    renames = {
        slug
        for slug, config in registry.items()
        if config.get("param_mappings", {}).get("max_completion_tokens") == "max_tokens"
    }
    enum_names = {provider.value for provider in LlmProviders}
    enum_members = {slug for slug in registry if slug in enum_names}
    assert set(csp.JSON_REGISTRY_PROVIDERS) == enum_members
    assert _JSON_RENAME == renames & enum_members
    for slug in csp.JSON_REGISTRY_PROVIDERS:
        # no JSON provider carries the (currently dead) temperature
        # constraints arm or a chat-relevant special_handling beyond
        # publicai's flatten (guard-pinned) and parasail's
        # force_store_false (Responses-API-only, dormant on chat)
        constraints = registry[slug].get("constraints", {})
        assert not constraints, (slug, constraints)
        special = dict(registry[slug].get("special_handling", {}))
        special.pop("force_store_false", None)
        if slug == "publicai":
            assert special == {"convert_content_list_to_string": True}
        else:
            assert special == {}, (slug, special)


def test_json_non_enum_providers_stay_dropped() -> None:
    """The 7 JSON-registry providers WITHOUT LlmProviders enum membership
    dispatch through v1's generic openai fallback arms (provider_config is
    None at param time, OpenAIConfig() at transform time) — their JSON param
    gates are dead, so they are DROPPED, not registered. If this canary
    fails, upstream gave them enum membership: re-evaluate porting them as
    ordinary JSON rows."""
    from litellm.types.utils import LlmProviders

    non_enum = (
        "veniceai",
        "abliteration",
        "llamagate",
        "gmi",
        "sarvam",
        "aihubmix",
        "crusoe",
    )
    enum_names = {provider.value for provider in LlmProviders}
    for slug in non_enum:
        assert slug not in enum_names, f"{slug} gained enum membership; re-evaluate"
        assert slug not in get_args(Provider)
        assert slug not in pipeline._SERIALIZERS
        assert slug not in pipeline._RESPONSE_PARSERS


def test_aiml_drop_canary() -> None:
    """aiml is DROPPED: AIMLChatConfig is unregistered in the provider
    config map at HEAD, so v1 serves aiml through the generic fallback stack
    (openai supported list + the bare OpenAILikeChatConfig map arm — mct
    RENAMES today). If the config resolution half fails, upstream registered
    the config (which has NO rename: mct would flip to verbatim) —
    re-evaluate aiml as an ordinary compat_sdk row."""
    from litellm.types.utils import LlmProviders
    from litellm.utils import ProviderConfigManager

    from litellm.utils import get_optional_params

    resolved = ProviderConfigManager.get_provider_chat_config(
        model="llama-x", provider=LlmProviders("aiml")
    )
    assert resolved is None, "AIMLChatConfig got registered; re-evaluate the drop"
    # the transform side substitutes OpenAIConfig() (openai.py no-config
    # fallback), so only the param chain is probed here: the OpenAILike
    # else-arm renames mct — AIMLChatConfig (OpenAIGPT-based) would NOT
    optional = get_optional_params(
        model="llama-x", custom_llm_provider="aiml", max_completion_tokens=9
    )
    assert optional.get("max_tokens") == 9
    assert "max_completion_tokens" not in optional
    assert "aiml" not in get_args(Provider)
    assert "aiml" not in pipeline._SERIALIZERS


def test_vercel_ai_gateway_routing_facts() -> None:
    """The dedicated vercel elif in completion() is dead code: compat-list
    membership matches the big SDK elif first. The always-injected
    ``extra_body`` is empty (popped by the invoker exactly like the SDK
    spread) unless the non-IR ``providerOptions`` rides — which the inbound
    boundary rejects typed."""
    import litellm as litellm_module

    assert "vercel_ai_gateway" in litellm_module.openai_compatible_providers
    v1 = run_v1_request_transform(
        "vercel_ai_gateway",
        {"model": _m("vercel_ai_gateway"), "messages": _USER},
    )
    assert "extra_body" not in v1
    result = _v2(
        "vercel_ai_gateway",
        {
            "model": _m("vercel_ai_gateway"),
            "providerOptions": {"gateway": {"order": ["x"]}},
            "messages": _USER,
        },
    )
    assert result.is_error()
    assert "providerOptions" in result.error.summary
