"""Differential parity for the snowflake request path (httpx dedicated
elif, main.py:4286; transform LIVE; a genuine wire mapping — tool_spec /
tool_choice objects — and the one wave-2b provider whose body ALWAYS
carries ``stream``).

Two-sided: every served row is byte-identical (normalized JSON) between v1
in-process at HEAD (get_optional_params + SnowflakeConfig.transform_request)
and v2; every fallback row asserts BOTH the typed v2 error and v1's own
behavior. Dossier confirmations: auth is HEADER-ONLY envelope
(researcher-4's correction stands), mct/stop/n/seed RAISE, and — beyond the
dossier — top_k rides the TOP-LEVEL passthrough (snowflake is NOT a compat
provider) so v1 SERVES it, wire-proven below.
"""

import copy
import json

import pytest

from litellm.exceptions import UnsupportedParamsError

from litellm.translation.dispatch import NEVER_PORT
from litellm.translation.engine import pipeline
from litellm.translation.engine.pipeline import translate_chat_request

from ._own_module_corpus import run_v1_request_transform
from .conftest import build_real_deps

MODEL = "claude-3-5-sonnet"
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
NO_PARAMS_TOOL = {"type": "function", "function": {"name": "ping"}}

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
    # the snowflake-only serve: stream is ALWAYS a body key, so an explicit
    # false is byte-identical to absent — v1 emits {"stream": false} both
    # ways and v2 serves it (the family guard arm is deliberately absent)
    "stream_false_served": {"model": MODEL, "stream": False, "messages": _USER},
    "temperature_int_stays_int": {"model": MODEL, "temperature": 1, "messages": _USER},
    # top_k: the non-compat top-level passthrough (v1 serves; wire-proven)
    "top_k_top_level": {"model": MODEL, "top_k": 5, "messages": _USER},
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
    # the tool_spec mapping (description kept; missing parameters default)
    "tools_to_tool_spec": {
        "model": MODEL,
        "tools": [WEATHER_TOOL, NO_PARAMS_TOOL],
        "messages": [{"role": "user", "content": "Weather in Paris?"}],
    },
    "tool_choice_auto": {
        "model": MODEL,
        "tools": [WEATHER_TOOL],
        "tool_choice": "auto",
        "messages": [{"role": "user", "content": "Weather?"}],
    },
    "tool_choice_required_to_any": {
        "model": MODEL,
        "tools": [WEATHER_TOOL],
        "tool_choice": "required",
        "messages": [{"role": "user", "content": "Weather?"}],
    },
    "tool_choice_none_object": {
        "model": MODEL,
        "tools": [WEATHER_TOOL],
        "tool_choice": "none",
        "messages": [{"role": "user", "content": "Weather?"}],
    },
    "tool_choice_specific_to_name_array": {
        "model": MODEL,
        "tools": [WEATHER_TOOL],
        "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
        "messages": [{"role": "user", "content": "Weather?"}],
    },
    # messages ride VERBATIM in v1 (no flatten, no base transforms): a
    # multi-block text list round-trips byte-identically through the IR
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
                        "id": "tooluse_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city":"Paris"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "tooluse_1", "content": "ok"},
        ],
    },
}

# name -> (case, v2 reason fragment); generator emits FALLBACK (v1 serves it)
EXPECTED_FALLBACKS: dict[str, tuple[dict, str]] = {
    "user": ({"model": MODEL, "user": "u1", "messages": _USER}, "user"),
    # v1 forwards message name VERBATIM (no strip anywhere on this chain)
    "message_name": (
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": "hi", "name": "alice"}],
        },
        "name",
    ),
}

# name -> (case, v2 reason fragment); v1 raises UnsupportedParamsError
V1_RAISES: dict[str, tuple[dict, str]] = {
    "max_completion_tokens": (
        {"model": MODEL, "max_completion_tokens": 99, "messages": _USER},
        "max_completion_tokens",
    ),
    "stop": ({"model": MODEL, "stop": ["END"], "messages": _USER}, "stop"),
    "parallel_tool_calls": (
        {
            "model": MODEL,
            "tools": [WEATHER_TOOL],
            "parallel_tool_calls": False,
            "messages": _USER,
        },
        "parallel_tool_calls",
    ),
}


def _v2(case: dict):
    return translate_chat_request(copy.deepcopy(case), "snowflake", build_real_deps())


def _norm(body: dict) -> str:
    return json.dumps(body, sort_keys=True, default=str)


@pytest.mark.parametrize("name", sorted(CASES))
def test_v2_request_matches_v1(name: str) -> None:
    case = CASES[name]
    result = _v2(case)
    assert result.is_ok(), result.error.summary
    assert _norm(result.ok) == _norm(run_v1_request_transform("snowflake", case))


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
        run_v1_request_transform("snowflake", case)


def test_stream_is_always_a_body_key() -> None:
    """The pin behind dropping the explicit-stream-false guard arm: absent
    and explicit-false produce the SAME wire byte on both sides."""
    absent = _v2(CASES["text"]).ok
    explicit = _v2(CASES["stream_false_served"]).ok
    assert absent["stream"] is False
    assert explicit["stream"] is False
    assert _norm(absent) == _norm(explicit)
    assert _v2(CASES["stream_true"]).ok["stream"] is True


def test_tool_spec_semantic_pins() -> None:
    body = _v2(CASES["tools_to_tool_spec"]).ok
    assert body["tools"][0] == {
        "tool_spec": {
            "type": "generic",
            "name": "get_weather",
            "input_schema": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
            "description": "Get weather",
        }
    }
    # missing parameters -> the default empty object schema
    assert body["tools"][1] == {
        "tool_spec": {
            "type": "generic",
            "name": "ping",
            "input_schema": {"type": "object", "properties": {}},
        }
    }


def test_tool_choice_object_pins() -> None:
    assert _v2(CASES["tool_choice_auto"]).ok["tool_choice"] == {"type": "auto"}
    assert _v2(CASES["tool_choice_required_to_any"]).ok["tool_choice"] == {
        "type": "any"
    }
    assert _v2(CASES["tool_choice_none_object"]).ok["tool_choice"] == {"type": "none"}
    assert _v2(CASES["tool_choice_specific_to_name_array"]).ok["tool_choice"] == {
        "type": "tool",
        "name": ["get_weather"],
    }


def test_top_k_wire_proven_top_level() -> None:
    """snowflake is NOT in openai_compatible_providers: top_k rides the
    top-level passthrough into the body (the wave-2b wire-prove rule).
    The mock-transport helper cannot reach this elif — main.py:4390
    OVERWRITES the caller's client with a fresh HTTPHandler (an envelope
    quirk, recorded in wave2b-alpha-port.md) — so the wire is pinned with
    respx instead, over the full completion() stack."""
    import respx as _respx
    import httpx as _httpx

    import litellm as _litellm

    v1 = run_v1_request_transform("snowflake", CASES["top_k_top_level"])
    assert v1["top_k"] == 5
    captured: dict = {}
    ok_body = {
        "id": "wire-1",
        "object": "chat.completion",
        "created": 1718000000,
        "model": MODEL,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }

    def _capture(request: _httpx.Request) -> _httpx.Response:
        captured["body"] = json.loads(request.content.decode())
        return _httpx.Response(200, json=ok_body)

    with _respx.mock(assert_all_called=True) as router:
        router.post(
            "https://acct.snowflakecomputing.com/api/v2/cortex/inference:complete"
        ).mock(side_effect=_capture)
        _litellm.completion(
            model=f"snowflake/{MODEL}",
            messages=[{"role": "user", "content": "hi"}],
            api_key="wire-test-key",
            api_base="https://acct.snowflakecomputing.com/api/v2",
            top_k=5,
        )
    wire = captured["body"]
    assert wire["top_k"] == 5
    assert wire["stream"] is False  # the always-on stream key, on the wire


def test_user_v1_drops_it_and_name_v1_forwards_it() -> None:
    case, _ = EXPECTED_FALLBACKS["user"]
    assert "user" not in run_v1_request_transform("snowflake", case)
    case, _ = EXPECTED_FALLBACKS["message_name"]
    v1 = run_v1_request_transform("snowflake", case)
    assert v1["messages"][0]["name"] == "alice"  # verbatim forward


def test_supported_list_mirror() -> None:
    """v1's small static list, row-for-row (fixed name samples — snowflake
    has its own model list, no per-model forks). top_k is in v2's allowed
    set but NOT v1's list — it is the non-compat passthrough, not a list
    member (the wire-prove test above pins the serve)."""
    from litellm.translation.providers.snowflake.params import _SNOWFLAKE_LIST

    from ._own_module_corpus import provider_config

    for model in (MODEL, "mistral-large2", "snowflake-arctic"):
        supported = set(
            provider_config("snowflake", model).get_supported_openai_params(model)
        )
        assert supported == {
            "temperature",
            "max_tokens",
            "top_p",
            "stream",
            "response_format",
            "tools",
            "tool_choice",
        }, model
        assert _SNOWFLAKE_LIST - {"top_k"} == supported, model


def test_registration_facts() -> None:
    assert "snowflake" in pipeline._SERIALIZERS
    assert "snowflake" in pipeline._RESPONSE_PARSERS
    assert "snowflake" in pipeline._RAW_GUARDS
    assert pipeline.response_dialect("snowflake") == "openai"
    assert "snowflake" not in NEVER_PORT
