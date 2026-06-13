"""Differential parity for the fireworks_ai request path (httpx dedicated
elif, main.py:2198; transform LIVE; the WIRE model differs from the request
model — the accounts/fireworks/models/ rewrite).

Two-sided: every served row is byte-identical (normalized JSON) between v1
in-process at HEAD and v2; every fallback row asserts BOTH the typed v2
error and v1's own behavior. Dossier-drift pins recorded in
wave2b-alpha-port.md: (1) the supported list's top_k mention is DEAD at
runtime — top_k rides the extra_body crossing (fireworks IS a compat
provider), wire-proven; (2) v1's tools capability is get_provider_info's
default-TRUE + a hyphen-boundary map scan, NOT the map flag — the v2 gate
is strictly narrower (one-direction mirror below), so tools serve only on
explicitly-flagged rows and fall back elsewhere with an honest drift note;
(3) the #transform=inline vision gate is a LITERAL substring check — VL
models without "vision" in the name get the suffix.
"""

import copy
import json

import pytest

from litellm.exceptions import UnsupportedParamsError

from litellm.translation.dispatch import NEVER_PORT
from litellm.translation.engine import pipeline
from litellm.translation.engine.pipeline import translate_chat_request
from litellm.utils import get_optional_params

from ._own_module_corpus import capture_v1_wire_body, run_v1_request_transform
from .conftest import build_real_deps

# explicitly map-flagged: fc + tool_choice + reasoning all True in BOTH v1
# and the deps read (the served-tools corpus model)
FLAGGED = "accounts/fireworks/models/deepseek-v3p2"
# bare name: the model-rewrite row (no explicit map flags -> tools fall back)
BARE = "llama-v3p3-70b-instruct"
# vision-NAMED model (the literal substring gate)
VISION = "accounts/fireworks/models/llama-v3p2-11b-vision-instruct"
# explicit map fc=False: v1 RAISES on tools
NO_FC = "accounts/fireworks/models/deepseek-coder-v2-instruct"
_USER = [{"role": "user", "content": "Hello, world"}]

CLEAN_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "nested": {"type": "object", "strict": True, "properties": {}},
            },
            "required": ["city"],
        },
    },
}

CASES: dict[str, dict] = {
    # bare model name -> accounts/fireworks/models/ rewrite
    "text_bare_model_rewritten": {"model": BARE, "messages": _USER},
    "text_accounts_model_kept": {"model": FLAGGED, "messages": _USER},
    "text_hash_model_kept": {"model": "my-model#deployment", "messages": _USER},
    "system_and_sampling": {
        "model": BARE,
        "max_tokens": 64,
        "temperature": 0.5,
        "top_p": 0.9,
        "messages": [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hi"},
        ],
    },
    "stream_true": {"model": BARE, "stream": True, "messages": _USER},
    "stop_list": {"model": BARE, "stop": ["END"], "messages": _USER},
    # fireworks HAS the rename arm (unlike its wave-2b siblings)
    "max_completion_tokens_renamed": {
        "model": BARE,
        "max_completion_tokens": 128,
        "messages": _USER,
    },
    "temperature_int_stays_int": {"model": BARE, "temperature": 1, "messages": _USER},
    "user_verbatim": {"model": BARE, "user": "u1", "messages": _USER},
    "tools_auto_on_flagged_model": {
        "model": FLAGGED,
        "tools": [CLEAN_TOOL],
        "tool_choice": "auto",
        "messages": [{"role": "user", "content": "Weather in Paris?"}],
    },
    # the required -> any rewrite
    "tool_choice_required_to_any": {
        "model": FLAGGED,
        "tools": [CLEAN_TOOL],
        "tool_choice": "required",
        "messages": [{"role": "user", "content": "Weather in Paris?"}],
    },
    "tool_choice_specific": {
        "model": FLAGGED,
        "tools": [CLEAN_TOOL],
        "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
        "messages": [{"role": "user", "content": "Weather in Paris?"}],
    },
    "tool_call_compact_roundtrip": {
        "model": FLAGGED,
        "tools": [CLEAN_TOOL],
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
        "model": FLAGGED,
        "tools": [CLEAN_TOOL],
        "parallel_tool_calls": False,
        "messages": [{"role": "user", "content": "Weather?"}],
    },
    "response_format_json_object": {
        "model": BARE,
        "response_format": {"type": "json_object"},
        "messages": _USER,
    },
    "response_format_json_schema_strict": {
        "model": BARE,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "a", "schema": {"type": "object"}, "strict": True},
        },
        "messages": _USER,
    },
    "reasoning_effort_on_flagged_model": {
        "model": FLAGGED,
        "reasoning_effort": "high",
        "messages": _USER,
    },
    # the literal "vision" substring gate: non-vision-NAMED models get the
    # suffix on https urls; data: urls exempt
    "image_inline_suffix_on_non_vision_name": {
        "model": BARE,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "look"},
                    {"type": "image_url", "image_url": {"url": "https://x/i.png"}},
                ],
            }
        ],
    },
    "image_data_url_exempt": {
        "model": BARE,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "look"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,AAAA"},
                    },
                ],
            }
        ],
    },
    "image_no_suffix_on_vision_named_model": {
        "model": VISION,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "look"},
                    {"type": "image_url", "image_url": {"url": "https://x/i.png"}},
                ],
            }
        ],
    },
    # v1 strips cache_control recursively == the IR drop, byte-identical
    "cache_control_stripped": {
        "model": BARE,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "a",
                        "cache_control": {"type": "ephemeral"},
                    },
                    {"type": "text", "text": "b"},
                ],
                "cache_control": {"type": "ephemeral"},
            }
        ],
    },
}

# name -> (case, v2 reason fragment); generator emits FALLBACK (v1 serves it)
EXPECTED_FALLBACKS: dict[str, tuple[dict, str]] = {
    "response_format_with_tools": (
        {
            "model": FLAGGED,
            "tools": [CLEAN_TOOL],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "a", "schema": {"type": "object"}},
            },
            "messages": _USER,
        },
        "json_mode",
    ),
    "legacy_defs_tool": (
        {
            "model": FLAGGED,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "f",
                        "parameters": {
                            "type": "object",
                            "properties": {"a": {"$ref": "#/definitions/A"}},
                            "definitions": {"A": {"type": "string"}},
                        },
                    },
                }
            ],
            "messages": _USER,
        },
        "schema inlining",
    ),
    # the capability drift: tools on a row WITHOUT explicit map flags — v1
    # SERVES (default-true), v2's narrower read falls back with the honest
    # drift note
    "tools_on_unflagged_model_drift": (
        {
            "model": BARE,
            "tools": [CLEAN_TOOL],
            "messages": [{"role": "user", "content": "Weather?"}],
        },
        "strictly narrower",
    ),
    "top_k": ({"model": BARE, "top_k": 5, "messages": _USER}, "extra_body"),
    "n": ({"model": BARE, "n": 2, "messages": _USER}, "n"),
    "prompt_truncate_length": (
        {"model": BARE, "prompt_truncate_length": 512, "messages": _USER},
        "prompt_truncate_length",
    ),
    "provider_specific_fields_message": (
        {
            "model": BARE,
            "messages": [
                {"role": "user", "content": "q"},
                {
                    "role": "assistant",
                    "content": "a",
                    "provider_specific_fields": {"x": 1},
                },
            ],
        },
        "provider_specific_fields",
    ),
    "assistant_thinking_blocks": (
        {
            "model": BARE,
            "messages": [
                {"role": "user", "content": "q"},
                {
                    "role": "assistant",
                    "content": "a",
                    "thinking_blocks": [
                        {"type": "thinking", "thinking": "hmm", "signature": "s"}
                    ],
                },
            ],
        },
        "thinking_blocks",
    ),
    "pdf_file_part": (
        {
            "model": BARE,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "read"},
                        {"type": "file", "file": {"file_id": "https://x/d.pdf"}},
                    ],
                }
            ],
        },
        "not yet supported",
    ),
    "stream_false": ({"model": BARE, "stream": False, "messages": _USER}, "stream"),
}

# name -> (case, v2 reason fragment); v1 raises UnsupportedParamsError
V1_RAISES: dict[str, tuple[dict, str]] = {
    "tools_on_explicit_no_fc_model": (
        {
            "model": NO_FC,
            "tools": [CLEAN_TOOL],
            "messages": [{"role": "user", "content": "Weather?"}],
        },
        "strictly narrower",
    ),
    "reasoning_effort_on_non_reasoning_model": (
        {"model": BARE, "reasoning_effort": "high", "messages": _USER},
        "strictly narrower",
    ),
}


def _v2(case: dict):
    return translate_chat_request(
        copy.deepcopy(case), "fireworks_ai", build_real_deps()
    )


def _norm(body: dict) -> str:
    return json.dumps(body, sort_keys=True, default=str)


@pytest.mark.parametrize("name", sorted(CASES))
def test_v2_request_matches_v1(name: str) -> None:
    case = CASES[name]
    result = _v2(case)
    assert result.is_ok(), result.error.summary
    assert _norm(result.ok) == _norm(run_v1_request_transform("fireworks_ai", case))


@pytest.mark.parametrize("name", sorted(EXPECTED_FALLBACKS))
def test_expected_fallbacks_are_typed(name: str) -> None:
    case, reason = EXPECTED_FALLBACKS[name]
    result = _v2(case)
    assert result.is_error(), f"{name} unexpectedly served"
    assert reason in result.error.summary, result.error.summary


@pytest.mark.parametrize("name", sorted(V1_RAISES))
def test_v1_raise_rows_are_two_sided(name: str) -> None:
    case, reason = V1_RAISES[name]
    result = _v2(case)
    assert result.is_error(), f"{name} unexpectedly served"
    assert reason in result.error.summary, result.error.summary
    with pytest.raises(UnsupportedParamsError):
        run_v1_request_transform("fireworks_ai", case)


def test_model_rewrite_pins() -> None:
    assert (
        _v2(CASES["text_bare_model_rewritten"]).ok["model"]
        == f"accounts/fireworks/models/{BARE}"
    )
    assert _v2(CASES["text_accounts_model_kept"]).ok["model"] == FLAGGED
    assert _v2(CASES["text_hash_model_kept"]).ok["model"] == "my-model#deployment"


def test_required_to_any_and_strict_strip_pins() -> None:
    body = _v2(CASES["tool_choice_required_to_any"]).ok
    assert body["tool_choice"] == "any"
    function = body["tools"][0]["function"]
    assert "strict" not in function
    # deeper strict keys ride VERBATIM (contrast hosted_vllm)
    assert function["parameters"]["properties"]["nested"]["strict"] is True


def test_image_suffix_pins() -> None:
    served = _v2(CASES["image_inline_suffix_on_non_vision_name"]).ok
    assert (
        served["messages"][0]["content"][1]["image_url"]["url"]
        == "https://x/i.png#transform=inline"
    )
    data_url = _v2(CASES["image_data_url_exempt"]).ok
    assert (
        data_url["messages"][0]["content"][1]["image_url"]["url"]
        == "data:image/png;base64,AAAA"
    )
    vision = _v2(CASES["image_no_suffix_on_vision_named_model"]).ok
    assert vision["messages"][0]["content"][1]["image_url"]["url"] == "https://x/i.png"


def test_rf_with_tools_v1_serves_json_mode_machinery() -> None:
    case, _ = EXPECTED_FALLBACKS["response_format_with_tools"]
    request = copy.deepcopy(case)
    optional = get_optional_params(
        model=request.pop("model"),
        custom_llm_provider="fireworks_ai",
        stream=None,
        **{k: v for k, v in request.items() if k != "messages"},
    )
    assert optional.get("json_mode") is True
    assert "response_format" not in optional


def test_legacy_defs_v1_serves_its_inlining() -> None:
    case, _ = EXPECTED_FALLBACKS["legacy_defs_tool"]
    v1 = run_v1_request_transform("fireworks_ai", case)
    parameters = v1["tools"][0]["function"]["parameters"]
    assert parameters["properties"]["a"] == {"type": "string"}  # $ref inlined
    assert "definitions" not in parameters


def test_capability_drift_v1_serves_tools_on_unflagged_model() -> None:
    case, _ = EXPECTED_FALLBACKS["tools_on_unflagged_model_drift"]
    v1 = run_v1_request_transform("fireworks_ai", case)
    assert v1["tools"]  # v1's default-true gate SERVES the tools


def test_psf_and_thinking_blocks_v1_serves_the_pops() -> None:
    for name in ("provider_specific_fields_message", "assistant_thinking_blocks"):
        case, _ = EXPECTED_FALLBACKS[name]
        v1 = run_v1_request_transform("fireworks_ai", case)
        assert v1["messages"][1] == {"role": "assistant", "content": "a"}, name


def test_pdf_file_v1_serves_the_image_url_migration() -> None:
    case, _ = EXPECTED_FALLBACKS["pdf_file_part"]
    v1 = run_v1_request_transform("fireworks_ai", case)
    migrated = v1["messages"][0]["content"][1]
    assert migrated["type"] == "image_url"
    assert migrated["image_url"]["url"].startswith("https://x/d.pdf")


def test_n_and_truncate_v1_serves_them() -> None:
    """n is in v1's supported list (top-level serve); prompt_truncate_length
    is a NON-OpenAI param despite the list mention — it rides the extra_body
    crossing like top_k (probed; the list mention is dead at runtime)."""
    case, _ = EXPECTED_FALLBACKS["n"]
    assert run_v1_request_transform("fireworks_ai", case)["n"] == 2
    packed = get_optional_params(
        model=BARE,
        custom_llm_provider="fireworks_ai",
        stream=None,
        prompt_truncate_length=512,
    )
    assert packed["extra_body"] == {"prompt_truncate_length": 512}


def test_top_k_fallback_reason_matches_the_wire_proven_extra_body_merge() -> None:
    """The supported list's top_k mention is DEAD: top_k never reaches
    _check_valid_arg (not an OpenAI param) and rides the extra_body packing
    (fireworks IS a compat provider); hh merges it onto the wire."""
    case, _ = EXPECTED_FALLBACKS["top_k"]
    packed = get_optional_params(
        model=BARE, custom_llm_provider="fireworks_ai", stream=None, top_k=5
    )
    assert packed["extra_body"] == {"top_k": 5}
    assert "top_k" not in run_v1_request_transform("fireworks_ai", copy.deepcopy(case))
    wire = capture_v1_wire_body(f"fireworks_ai/{BARE}", top_k=5)
    assert wire["top_k"] == 5
    assert "extra_body" not in wire


def test_capability_mirror_is_one_direction() -> None:
    """The soundness pin for the drift note: over EVERY fireworks chat map
    row, the v2 capability read NEVER answers True where v1 answers False
    (v2 may only be narrower — fallback-safe), and the explicitly-flagged
    serve set is non-trivial. If a v1-False/v2-True row ever appears, v2
    would serve what v1 raises on: re-derive the gate."""
    import litellm
    from litellm.utils import (
        supports_function_calling,
        supports_reasoning,
        supports_tool_choice,
    )

    from litellm.translation.providers.fireworks_ai.params import (
        supports_fireworks_reasoning,
        supports_fireworks_tool_choice,
        supports_fireworks_tools,
    )

    deps = build_real_deps()
    rows = [
        key.split("/", 1)[1]
        for key, info in litellm.model_cost.items()
        if key.startswith("fireworks_ai/")
        and isinstance(info, dict)
        and info.get("mode") == "chat"
    ]
    assert len(rows) > 100
    served_fc = 0
    for model in rows:
        if supports_fireworks_tools(model, deps):
            served_fc += 1
            assert supports_function_calling(model, "fireworks_ai"), model
        if supports_fireworks_tool_choice(model, deps):
            assert supports_tool_choice(
                model=model, custom_llm_provider="fireworks_ai"
            ), model
        if supports_fireworks_reasoning(model, deps):
            assert supports_reasoning(model, "fireworks_ai"), model
    assert served_fc >= 10  # the explicit-flag serve set stays non-trivial


def test_supported_base_list_mirror() -> None:
    """The static half of v1's list: every key v2's base allowed set carries
    must be in v1's list for every model (user included — fireworks lists it
    explicitly), and the list carries the keys v2 deliberately leaves to the
    inbound fallback (n, logprobs, penalties, top_k, the two provider-native
    params) so those fallbacks stay the v1-serves kind."""
    from litellm.translation.providers.fireworks_ai.params import _FIREWORKS_BASE

    from ._own_module_corpus import provider_config

    for model in (BARE, FLAGGED, NO_FC):
        supported = set(
            provider_config("fireworks_ai", model).get_supported_openai_params(model)
        )
        assert _FIREWORKS_BASE <= supported, model
        assert {
            "n",
            "logprobs",
            "frequency_penalty",
            "presence_penalty",
            "top_k",
            "prompt_truncate_length",
            "context_length_exceeded_behavior",
        } <= supported, model


def test_registration_facts() -> None:
    assert "fireworks_ai" in pipeline._SERIALIZERS
    assert "fireworks_ai" in pipeline._RESPONSE_PARSERS
    assert "fireworks_ai" in pipeline._RAW_GUARDS
    assert pipeline.response_dialect("fireworks_ai") == "openai"
    assert "fireworks_ai" not in NEVER_PORT
