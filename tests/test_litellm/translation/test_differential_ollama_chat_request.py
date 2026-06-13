"""Differential parity for the ollama_chat request path (wave 3).

v1 side, invoked the way main.py's ollama_chat elif runs:
``get_optional_params(custom_llm_provider="ollama_chat")`` then
``OllamaChatConfig.transform_request`` (httpx path, transforms LIVE). The
serializer pins (all probed in-process at HEAD): the ``{model, messages,
options, stream}`` body with ``stream`` ALWAYS present (explicit false ==
absent — the snowflake shape, NO guard arm), ``images: []`` attached to
EVERY message, message ``name`` dropped for every role, max_tokens/mct ->
``options.num_predict``, ``top_k`` riding ``options`` via the provider-
native passthrough, response_format json_object -> ``format: "json"`` /
json_schema -> the bare schema dict, reasoning_effort -> ``think`` (verbatim
for gpt-oss*, else the {low, medium, high} boolean), tool_choice silently
POPPED, tools verbatim (openai shape), assistant tool_calls munged to
``{function: {name, arguments: <parsed>}}`` (id/type dropped), and the
think-tag quirk: a string content matching v1's regex emits ``thinking``
while the content stays the FULL tagged string.

The ambient ``OllamaChatConfig`` class-attr defaults (its ``__init__``
setattrs onto the CLASS) merge into ``options`` in v1 — module state v2
cannot see; the canary below pins that the default config is empty and the
fork obligation (fall back when non-empty) is recorded in CLAUDE.md.
"""

import copy
import json

import pytest

from litellm.exceptions import UnsupportedParamsError
from litellm.llms.ollama.chat.transformation import OllamaChatConfig
from litellm.utils import get_optional_params

from litellm.translation.engine.pipeline import translate_chat_request
from litellm.translation_seam import build_translation_deps

PROVIDER = "ollama_chat"
MODEL = "llama3.1"

_U = [{"role": "user", "content": "hi"}]

CASES = {
    "plain": {"model": MODEL, "messages": _U},
    "sampling": {
        "model": MODEL,
        "messages": _U,
        "temperature": 0.5,
        "top_p": 0.9,
        "max_tokens": 100,
        "stop": ["x", "y"],
    },
    "mct_to_num_predict": {
        "model": MODEL,
        "messages": _U,
        "max_completion_tokens": 50,
    },
    "top_k_options_passthrough": {"model": MODEL, "messages": _U, "top_k": 7},
    "temperature_int_stays_int": {"model": MODEL, "messages": _U, "temperature": 1},
    "stream_true": {"model": MODEL, "messages": _U, "stream": True},
    "stream_false": {"model": MODEL, "messages": _U, "stream": False},
    "tools_verbatim": {
        "model": MODEL,
        "messages": _U,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "d",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
    },
    "tools_strict_rides": {
        "model": MODEL,
        "messages": _U,
        "tools": [
            {
                "type": "function",
                "function": {"name": "f", "parameters": {}, "strict": True},
            }
        ],
    },
    "tool_choice_popped": {
        "model": MODEL,
        "messages": _U,
        "tools": [{"type": "function", "function": {"name": "f"}}],
        "tool_choice": "required",
    },
    "rf_json_object": {
        "model": MODEL,
        "messages": _U,
        "response_format": {"type": "json_object"},
    },
    "rf_json_schema_bare_schema": {
        "model": MODEL,
        "messages": _U,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "n",
                "strict": True,
                "schema": {"type": "object", "properties": {"a": {"type": "string"}}},
            },
        },
    },
    "rf_text_dropped": {
        "model": MODEL,
        "messages": _U,
        "response_format": {"type": "text"},
    },
    "reasoning_effort_low_to_think_true": {
        "model": MODEL,
        "messages": _U,
        "reasoning_effort": "low",
    },
    "reasoning_effort_minimal_to_think_false": {
        "model": MODEL,
        "messages": _U,
        "reasoning_effort": "minimal",
    },
    "reasoning_effort_none_string_to_think_false": {
        "model": MODEL,
        "messages": _U,
        "reasoning_effort": "none",
    },
    "reasoning_effort_gptoss_verbatim": {
        "model": "gpt-oss:20b",
        "messages": _U,
        "reasoning_effort": "low",
    },
    "system_rides_in_messages": {
        "model": MODEL,
        "messages": [{"role": "system", "content": "be nice"}, *_U],
    },
    "multi_text_list_flattens": {
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
    "image_data_url_stripped_to_base64": {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "look"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,iVBOR"},
                    },
                ],
            }
        ],
    },
    "image_http_url_verbatim": {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "look"},
                    {"type": "image_url", "image_url": {"url": "https://x.test/a.png"}},
                ],
            }
        ],
    },
    "tool_roundtrip_id_type_dropped": {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "f", "arguments": '{"a": 1}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "result"},
            {"role": "user", "content": "ok"},
        ],
    },
    "message_name_dropped_every_role": {
        "model": MODEL,
        "messages": [{"role": "user", "content": "hi", "name": "bob"}],
    },
    "assistant_think_tags_full_content_plus_thinking": {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "<think>hmm</think>yes"},
            {"role": "user", "content": "more"},
        ],
    },
    "user_think_tags_same_quirk": {
        "model": MODEL,
        "messages": [{"role": "user", "content": "<think>x</think>y"}],
    },
    "empty_assistant_content_rides": {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": ""},
            {"role": "user", "content": "r"},
        ],
    },
}

# v1 RAISES UnsupportedParamsError (outside the supported list, probed);
# v2 must be a typed fallback naming the surface.
V1_RAISES = {
    "n": ({"model": MODEL, "messages": _U, "n": 2}, "n"),
    "logprobs": ({"model": MODEL, "messages": _U, "logprobs": True}, "logprobs"),
    "presence_penalty": (
        {"model": MODEL, "messages": _U, "presence_penalty": 0.1},
        "presence_penalty",
    ),
    "parallel_tool_calls": (
        {"model": MODEL, "messages": _U, "parallel_tool_calls": True},
        "parallel_tool_calls",
    ),
    "thinking": (
        {"model": MODEL, "messages": _U, "thinking": {"type": "enabled"}},
        "thinking",
    ),
}

# v1 SERVES these (silent drop / passthrough / local raise the inbound
# boundary reproduces); v2 falls back typed so v1 keeps serving.
V1_SERVES_FALLBACKS = {
    "user_silent_drop": ({"model": MODEL, "messages": _U, "user": "u1"}, "user"),
    "seed_parse_level": ({"model": MODEL, "messages": _U, "seed": 42}, "seed"),
    "frequency_penalty_to_repeat_penalty": (
        {"model": MODEL, "messages": _U, "frequency_penalty": 0.2},
        "frequency_penalty",
    ),
    "functions_to_tools_verbatim": (
        {"model": MODEL, "messages": _U, "functions": [{"name": "g"}]},
        "functions",
    ),
    "mirostat_native_passthrough": (
        {"model": MODEL, "messages": _U, "mirostat": 1},
        "mirostat",
    ),
    "keep_alive_passthrough": (
        {"model": MODEL, "messages": _U, "keep_alive": "5m"},
        "keep_alive",
    ),
    "function_name_near_dormant_synth_arm": (
        {"model": MODEL, "messages": _U, "function_name": "f"},
        "function_name",
    ),
    "string_stop_rides_verbatim": (
        {"model": MODEL, "messages": _U, "stop": "end"},
        "stop",
    ),
    "both_max_tokens_keys": (
        {"model": MODEL, "messages": _U, "max_tokens": 100, "max_completion_tokens": 50},
        "max_tokens",
    ),
    "assistant_reasoning_content_to_thinking": (
        {
            "model": MODEL,
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "ans", "reasoning_content": "hmm"},
                {"role": "user", "content": "more"},
            ],
        },
        "reasoning_content",
    ),
    "tool_cache_control_verbatim": (
        {
            "model": MODEL,
            "messages": _U,
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "f", "parameters": {}},
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        },
        "cache_control",
    ),
    "mid_conversation_system": (
        {
            "model": MODEL,
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "system", "content": "late"},
                {"role": "user", "content": "more"},
            ],
        },
        "system",
    ),
    "consecutive_user_turns": (
        {
            "model": MODEL,
            "messages": [
                {"role": "user", "content": "a"},
                {"role": "user", "content": "b"},
            ],
        },
        "consecutive",
    ),
}

# v1 CRASHES with a raw exception (not UnsupportedParamsError); v2 must be
# a typed fallback so v1 serves its own raise.
V1_RAISES_RAW = {
    "blank_tool_arguments_json_decode_error": (
        {
            "model": MODEL,
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "c",
                            "type": "function",
                            "function": {"name": "f", "arguments": "not-json"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "c", "content": "r"},
            ],
        },
        "JSON repair",
    ),
}


def run_v1_request_transform(case: dict) -> dict:
    """May RAISE — that IS the pinned v1 behavior for the raise rows."""
    request = copy.deepcopy(case)
    model = request.pop("model")
    messages = request.pop("messages")
    optional_params = get_optional_params(
        model=model,
        custom_llm_provider=PROVIDER,
        messages=copy.deepcopy(messages),
        stream=request.pop("stream", None),
        **request,
    )
    optional_params.pop("extra_body", None)
    return OllamaChatConfig().transform_request(
        model=model,
        messages=messages,
        optional_params=optional_params,
        litellm_params={},
        headers={},
    )


def _v2(case: dict):
    return translate_chat_request(
        copy.deepcopy(case), PROVIDER, build_translation_deps()
    )


def _norm(body: dict) -> str:
    return json.dumps(body, sort_keys=True, default=str)


@pytest.mark.parametrize("name", sorted(CASES))
def test_v2_request_matches_v1(name: str) -> None:
    case = CASES[name]
    result = _v2(case)
    assert result.is_ok(), result.error.summary
    assert _norm(result.ok) == _norm(run_v1_request_transform(case))


@pytest.mark.parametrize("name", sorted(V1_RAISES))
def test_v1_raise_rows_fall_back_typed(name: str) -> None:
    case, fragment = V1_RAISES[name]
    result = _v2(case)
    assert result.is_error(), f"{name} unexpectedly translated: {result.ok!r}"
    assert fragment in result.error.summary, result.error.summary
    with pytest.raises(UnsupportedParamsError):
        run_v1_request_transform(case)


@pytest.mark.parametrize("name", sorted(V1_SERVES_FALLBACKS))
def test_v1_serves_fallback_rows(name: str) -> None:
    case, fragment = V1_SERVES_FALLBACKS[name]
    result = _v2(case)
    assert result.is_error(), f"{name} unexpectedly translated: {result.ok!r}"
    assert fragment in result.error.summary, result.error.summary
    # v1 serves the same request without raising.
    run_v1_request_transform(case)


@pytest.mark.parametrize("name", sorted(V1_RAISES_RAW))
def test_v1_raw_raise_rows_fall_back_typed(name: str) -> None:
    case, fragment = V1_RAISES_RAW[name]
    result = _v2(case)
    assert result.is_error(), f"{name} unexpectedly translated: {result.ok!r}"
    assert fragment in result.error.summary, result.error.summary
    with pytest.raises(json.JSONDecodeError):
        run_v1_request_transform(case)


def test_stream_false_equals_absent_on_the_wire() -> None:
    """The snowflake shape: v1's transform pops stream with default False,
    so the body ALWAYS carries the key — explicit false and absent are the
    same wire byte and the guard deliberately has no stream:false arm."""
    absent = run_v1_request_transform(CASES["plain"])
    explicit = run_v1_request_transform(CASES["stream_false"])
    assert absent["stream"] is False
    assert _norm(absent) == _norm(explicit)
    result = _v2(CASES["stream_false"])
    assert result.is_ok() and result.ok["stream"] is False


def test_images_list_attached_to_every_message() -> None:
    v1 = run_v1_request_transform(CASES["system_rides_in_messages"])
    result = _v2(CASES["system_rides_in_messages"])
    assert result.is_ok()
    for body in (v1, result.ok):
        assert all(message["images"] == [] for message in body["messages"])


def test_think_tag_request_quirk_keeps_full_content() -> None:
    """The probe-found quirk a code-reading port would invert: v1 reads the
    regex result for ``thinking`` but flattens the ORIGINAL content, so the
    tagged bytes stay on the wire alongside the extracted thinking."""
    v1 = run_v1_request_transform(
        CASES["assistant_think_tags_full_content_plus_thinking"]
    )
    assistant = v1["messages"][1]
    assert assistant["thinking"] == "hmm"
    assert assistant["content"] == "<think>hmm</think>yes"
    result = _v2(CASES["assistant_think_tags_full_content_plus_thinking"])
    assert result.is_ok()
    assert _norm(result.ok) == _norm(v1)


def test_think_regex_mirror_matches_v1_source() -> None:
    """The serializer's THINK_TAG_RE is the one in-package mirror of v1's
    ``_parse_content_for_reasoning`` regex; drift here is a silent
    divergence, so pin the pattern against representative inputs through
    v1's live helper."""
    from litellm.litellm_core_utils.prompt_templates.common_utils import (
        _parse_content_for_reasoning,
    )

    from litellm.translation.providers.ollama_chat.serialize import THINK_TAG_RE

    samples = (
        "<think>a</think>b",
        "<thinking>a</thinking>b",
        "<budget:thinking>a</budget:thinking>b",
        "<think>multi\nline</think>rest",
        "pre<think>a</think>b",  # no match: v1 anchors at the start
        "<think>unclosed",
        "<think>a</thinking>b",  # mismatched tags still match v1's regex
        "",
        "plain",
    )
    for text in samples:
        v1_reasoning, _ = _parse_content_for_reasoning(text)
        matched = THINK_TAG_RE.match(text) if text else None
        v2_reasoning = matched.group(1) if matched else None
        assert v2_reasoning == v1_reasoning, text


def test_ambient_class_config_canary() -> None:
    """v1 merges ``OllamaChatConfig.get_config()`` CLASS-ATTR state into
    ``options`` (the groq class-attr precedent) — ambient module state v2
    cannot see. The default must be empty (every differential row relies on
    it) and the fork obligation (fall back when non-empty) is recorded in
    CLAUDE.md. If this canary goes red, an earlier test leaked class state
    or upstream grew defaults — either way the corpus assumption broke."""
    assert OllamaChatConfig.get_config() == {}
