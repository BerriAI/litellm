"""Differential parity for the huggingface request path — the api_base
(dedicated endpoint) ROUTE ONLY (httpx dedicated elif, main.py:3185).

v1's transform forks on ``litellm_params["api_base"]``: SET means a
VERBATIM ``{model, messages, **optional_params}`` body (the route v2
ports, ``deps.api_base`` carrying the value); UNSET means the router
route, which v2 falls back on WHOLE — its 3-segment names fetch the HF
provider mapping over HTTP inside the transform, 2-segment names run the
base transforms + router URL synthesis, and 1-segment names CRASH v1
(ValueError on the split; all three pinned in-process below).
"""

import copy
import json

import pytest

from litellm.translation.dispatch import NEVER_PORT
from litellm.translation.engine import pipeline
from litellm.translation.engine.pipeline import translate_chat_request

from ._own_module_corpus import run_v1_request_transform
from .conftest import build_real_deps

MODEL = "meta-llama/Llama-3.3-70B-Instruct"
API_BASE = "https://my-tgi.example/v1"
_LP = {"api_base": API_BASE}
_USER = [{"role": "user", "content": "Hello, world"}]

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}

CASES: dict[str, dict] = {
    "text": {"model": MODEL, "messages": _USER},
    "sampling": {
        "model": MODEL,
        "max_tokens": 64,
        "temperature": 0.5,
        "top_p": 0.9,
        "messages": [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hi"},
        ],
    },
    "stream_true": {"model": MODEL, "stream": True, "messages": _USER},
    "stop_list": {"model": MODEL, "stop": ["END"], "messages": _USER},
    # mct passes VERBATIM (base list, no rename anywhere)
    "max_completion_tokens_verbatim": {
        "model": MODEL,
        "max_completion_tokens": 128,
        "messages": _USER,
    },
    "temperature_int_stays_int": {"model": MODEL, "temperature": 1, "messages": _USER},
    # top_k: the non-compat TOP-LEVEL passthrough (v1 serves; pinned below)
    "top_k_top_level": {"model": MODEL, "top_k": 5, "messages": _USER},
    "tools_auto": {
        "model": MODEL,
        "tools": [WEATHER_TOOL],
        "tool_choice": "auto",
        "messages": [{"role": "user", "content": "Weather in Paris?"}],
    },
    "tool_choice_specific": {
        "model": MODEL,
        "tools": [WEATHER_TOOL],
        "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
        "messages": [{"role": "user", "content": "Weather?"}],
    },
    "tool_call_compact_roundtrip": {
        "model": MODEL,
        "tools": [WEATHER_TOOL],
        "messages": [
            {"role": "user", "content": "w?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city":"Paris"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
        ],
    },
    "parallel_tool_calls_false": {
        "model": MODEL,
        "tools": [WEATHER_TOOL],
        "parallel_tool_calls": False,
        "messages": [{"role": "user", "content": "Weather?"}],
    },
    "response_format_json_object": {
        "model": MODEL,
        "response_format": {"type": "json_object"},
        "messages": _USER,
    },
    "response_format_json_schema_strict": {
        "model": MODEL,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "a", "schema": {"type": "object"}, "strict": True},
        },
        "messages": _USER,
    },
    # messages ride VERBATIM on the api_base arm (no transforms at all)
    "content_list_verbatim": {
        "model": MODEL,
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
}

# name -> (case, v2 reason fragment); generator emits FALLBACK (v1 serves it)
EXPECTED_FALLBACKS: dict[str, tuple[dict, str]] = {
    "user": ({"model": MODEL, "user": "u1", "messages": _USER}, "user"),
    # max_retries is NOT popped on the api_base arm: v1 serves it INTO the
    # body (a v1 oddity — the pop lives on the router arm only)
    "max_retries": (
        {"model": MODEL, "max_retries": 3, "messages": _USER},
        "max_retries",
    ),
    # v1 forwards message name VERBATIM on this arm
    "message_name": (
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": "hi", "name": "alice"}],
        },
        "name",
    ),
    "stream_false": ({"model": MODEL, "stream": False, "messages": _USER}, "stream"),
}

_ROUTER_FALLBACK_REASON = "router route"


def _v2(case: dict, api_base: str | None = API_BASE):
    return translate_chat_request(
        copy.deepcopy(case), "huggingface", build_real_deps(api_base=api_base)
    )


def _v1(case: dict) -> dict:
    return run_v1_request_transform("huggingface", case, litellm_params=dict(_LP))


def _norm(body: dict) -> str:
    return json.dumps(body, sort_keys=True, default=str)


@pytest.mark.parametrize("name", sorted(CASES))
def test_v2_request_matches_v1(name: str) -> None:
    case = CASES[name]
    result = _v2(case)
    assert result.is_ok(), result.error.summary
    assert _norm(result.ok) == _norm(_v1(case))


@pytest.mark.parametrize("name", sorted(EXPECTED_FALLBACKS))
def test_expected_fallbacks_are_typed(name: str) -> None:
    case, reason = EXPECTED_FALLBACKS[name]
    result = _v2(case)
    assert result.is_error(), f"{name} unexpectedly served"
    assert reason in result.error.summary, result.error.summary


def test_router_route_falls_back_whole() -> None:
    """deps.api_base None -> EVERY huggingface request falls back typed,
    naming the router route. The three v1 router shapes are pinned
    in-process: 2-segment names SERVE locally (no fetch — base transforms
    + the router URL), 1-segment names CRASH (ValueError on the split),
    and 3-segment names would fetch the provider mapping over HTTP (NOT
    exercised here — that is the in-transform I/O the port refuses)."""
    result = _v2(CASES["text"], api_base=None)
    assert result.is_error()
    assert _ROUTER_FALLBACK_REASON in result.error.summary
    # 2-segment: v1 serves WITHOUT fetching (no network in this test)
    v1 = run_v1_request_transform("huggingface", CASES["text"], litellm_params={})
    assert v1["model"] == MODEL
    # 1-segment: v1 crashes on the split
    with pytest.raises(ValueError):
        run_v1_request_transform(
            "huggingface",
            {"model": "gpt2", "messages": [{"role": "user", "content": "hi"}]},
            litellm_params={},
        )


def test_api_base_body_is_verbatim() -> None:
    """The api_base arm: model verbatim, messages verbatim, no renames —
    v1's ChatCompletionRequest passthrough."""
    v1 = _v1(CASES["content_list_verbatim"])
    assert v1["messages"] == CASES["content_list_verbatim"]["messages"]
    assert v1["model"] == MODEL
    v1_mct = _v1(CASES["max_completion_tokens_verbatim"])
    assert v1_mct["max_completion_tokens"] == 128


def test_top_k_v1_serves_top_level() -> None:
    """huggingface is NOT in openai_compatible_providers: top_k rides the
    top-level passthrough into the verbatim body (the api_base arm IS the
    wire body — hh posts the transform output for this provider, no
    extra_body crossing exists)."""
    v1 = _v1(CASES["top_k_top_level"])
    assert v1["top_k"] == 5
    result = _v2(CASES["top_k_top_level"])
    assert result.is_ok()
    assert result.ok["top_k"] == 5


def test_max_retries_and_user_v1_behavior() -> None:
    case, _ = EXPECTED_FALLBACKS["max_retries"]
    assert _v1(case)["max_retries"] == 3  # NOT popped on the api_base arm
    case, _ = EXPECTED_FALLBACKS["user"]
    assert "user" not in _v1(case)  # silently dropped upstream


def test_supported_list_mirror() -> None:
    """The plain OpenAI base list (no override) — fixed name samples, with
    the base list's gpt-4/gpt-3.5-turbo-16k response_format name gate."""
    from litellm.translation.providers.huggingface.params import _HUGGINGFACE_LIST

    from ._own_module_corpus import provider_config

    mirror_keys = (
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
    assert _HUGGINGFACE_LIST <= set(mirror_keys) | {"top_k"}
    for model in (MODEL, "tgi", "gpt-4"):
        supported = set(
            provider_config("huggingface", model).get_supported_openai_params(model)
        )
        allowed = (
            _HUGGINGFACE_LIST - {"response_format"}
            if model in ("gpt-4", "gpt-3.5-turbo-16k")
            else _HUGGINGFACE_LIST
        )
        for key in mirror_keys:
            assert (key in allowed) == (key in supported), (model, key)


def test_registration_facts() -> None:
    assert "huggingface" in pipeline._SERIALIZERS
    assert "huggingface" in pipeline._RESPONSE_PARSERS
    assert "huggingface" in pipeline._RAW_GUARDS
    assert pipeline.response_dialect("huggingface") == "openai"
    assert "huggingface" not in NEVER_PORT
