"""Differential parity for the cohere v2 chat request path (wave-2b-beta).

v2 is the DEFAULT cohere route at HEAD (``CohereModelInfo.get_cohere_route``:
"v1" only when the model carries "v1/"), and BOTH provider names —
"cohere" and "cohere_chat" — resolve to ``CohereV2ChatConfig`` and the same
main.py elif, so both dispatch names run every row here.

v1 side, invoked the way main.py's cohere elif runs: ``get_optional_params``
(NOTE the drift probe: its cohere arm calls the LEGACY
``CohereChatConfig().map_openai_params``, not the v2 config's — the two maps
are byte-equivalent over the shared supported list, re-verified in-process
at HEAD) with completion()'s ``stream=None`` default, then
``CohereV2ChatConfig.transform_request`` (the inherited OpenAI GPT
assembly). The serializer deltas pinned IDENTICAL: top_p -> p,
stop -> stop_sequences, max_completion_tokens -> max_tokens, tool_choice
silently DROPPED (supported list, no map arm), and ``top_k`` emitted
verbatim top-level (the generic non-openai passthrough places it in
optional_params for cohere — wire-proven here, NOT the extra_body arm).
"""

import copy
import json

import pytest

from litellm.exceptions import UnsupportedParamsError
from litellm.llms.cohere.chat.v2_transformation import CohereV2ChatConfig
from litellm.utils import get_optional_params

from litellm.translation.engine.pipeline import translate_chat_request
from litellm.translation_seam import build_translation_deps

PROVIDERS = ("cohere", "cohere_chat")

_BASE_MESSAGES = [{"role": "user", "content": "hi"}]

CASES = {
    "plain": {"model": "command-r", "messages": _BASE_MESSAGES},
    "sampling": {
        "model": "command-r",
        "messages": _BASE_MESSAGES,
        "temperature": 0.5,
        "top_p": 0.9,
        "max_tokens": 100,
        "stop": ["x", "y"],
    },
    "mct_renamed": {
        "model": "command-r-plus",
        "messages": _BASE_MESSAGES,
        "max_completion_tokens": 50,
    },
    "tools": {
        "model": "command-r",
        "messages": _BASE_MESSAGES,
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
        "tool_choice": "auto",
    },
    "tool_choice_required_dropped": {
        "model": "command-r",
        "messages": _BASE_MESSAGES,
        "tools": [{"type": "function", "function": {"name": "f", "parameters": {}}}],
        "tool_choice": "required",
    },
    "top_k_wire_proven": {
        "model": "command-r",
        "messages": _BASE_MESSAGES,
        "top_k": 7,
    },
    "stream_true": {"model": "command-r", "messages": _BASE_MESSAGES, "stream": True},
    "system_history": {
        "model": "command-r",
        "messages": [
            {"role": "system", "content": "be nice"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "again"},
        ],
    },
}

# v1 RAISES UnsupportedParamsError (outside the supported list), probed
# in-process at HEAD; v2 must be a typed fallback naming the raise.
V1_RAISES = {
    "response_format": (
        {
            "model": "command-r",
            "messages": _BASE_MESSAGES,
            "response_format": {"type": "json_object"},
        },
        "response_format",
    ),
    "parallel_tool_calls": (
        {
            "model": "command-r",
            "messages": _BASE_MESSAGES,
            "parallel_tool_calls": True,
        },
        "parallel_tool_calls",
    ),
    "thinking": (
        {
            "model": "command-r",
            "messages": _BASE_MESSAGES,
            "thinking": {"type": "enabled", "budget_tokens": 1024},
        },
        "thinking",
    ),
    "reasoning_effort": (
        {
            "model": "command-r",
            "messages": _BASE_MESSAGES,
            "reasoning_effort": "high",
        },
        "reasoning_effort",
    ),
}

# v1 SERVES these (drop / passthrough / envelope rewrite); v2 falls back
# typed so v1 keeps serving its own behavior.
V1_SERVES_FALLBACKS = {
    "user_silent_drop": (
        {"model": "command-r", "messages": _BASE_MESSAGES, "user": "u1"},
        "user",
    ),
    "n_parse_level": (
        {"model": "command-r", "messages": _BASE_MESSAGES, "n": 2},
        "n",
    ),
    "seed_parse_level": (
        {"model": "command-r", "messages": _BASE_MESSAGES, "seed": 42},
        "seed",
    ),
    "frequency_penalty_parse_level": (
        {"model": "command-r", "messages": _BASE_MESSAGES, "frequency_penalty": 0.1},
        "frequency_penalty",
    ),
    "presence_penalty_parse_level": (
        {"model": "command-r", "messages": _BASE_MESSAGES, "presence_penalty": 0.2},
        "presence_penalty",
    ),
    "v1_route_predicate": (
        {"model": "v1/command-r", "messages": _BASE_MESSAGES},
        "v1",
    ),
    "v2_prefix_envelope_strip": (
        {"model": "v2/command-r", "messages": _BASE_MESSAGES},
        "v2/",
    ),
    "explicit_stream_false": (
        {"model": "command-r", "messages": _BASE_MESSAGES, "stream": False},
        "stream",
    ),
    "message_name_forwarded": (
        {
            "model": "command-r",
            "messages": [{"role": "user", "content": "hi", "name": "bob"}],
        },
        "name",
    ),
}


def run_v1_request_transform(provider: str, case: dict) -> dict:
    """May RAISE UnsupportedParamsError: that IS the pinned v1 behavior for
    the raise rows."""
    request = copy.deepcopy(case)
    model = request.pop("model")
    messages = request.pop("messages")
    optional_params = get_optional_params(
        model=model,
        custom_llm_provider=provider,
        messages=copy.deepcopy(messages),
        stream=request.pop("stream", None),
        **request,
    )
    optional_params.pop("extra_body", None)
    return CohereV2ChatConfig().transform_request(
        model=model,
        messages=messages,
        optional_params=optional_params,
        litellm_params={},
        headers={},
    )


def _v2(provider: str, case: dict):
    return translate_chat_request(
        copy.deepcopy(case), provider, build_translation_deps()
    )


def _norm(body: dict) -> str:
    return json.dumps(body, sort_keys=True, default=str)


def _rows(table: dict):
    return sorted((provider, name) for provider in PROVIDERS for name in table)


@pytest.mark.parametrize("provider,name", _rows(CASES))
def test_v2_request_matches_v1(provider: str, name: str) -> None:
    case = CASES[name]
    result = _v2(provider, case)
    assert result.is_ok(), result.error.summary
    assert _norm(result.ok) == _norm(run_v1_request_transform(provider, case))


@pytest.mark.parametrize("provider,name", _rows(V1_RAISES))
def test_v1_raise_rows_fall_back_typed(provider: str, name: str) -> None:
    case, fragment = V1_RAISES[name]
    result = _v2(provider, case)
    assert result.is_error(), f"{name} unexpectedly translated: {result.ok!r}"
    assert fragment in result.error.summary, result.error.summary
    with pytest.raises(UnsupportedParamsError):
        run_v1_request_transform(provider, case)


@pytest.mark.parametrize("provider,name", _rows(V1_SERVES_FALLBACKS))
def test_v1_serves_fallback_rows(provider: str, name: str) -> None:
    case, fragment = V1_SERVES_FALLBACKS[name]
    result = _v2(provider, case)
    assert result.is_error(), f"{name} unexpectedly translated: {result.ok!r}"
    assert fragment in result.error.summary, result.error.summary
    # v1 serves the same request without raising (drop/passthrough/envelope).
    run_v1_request_transform(provider, case)


def test_top_k_rides_the_wire_top_level() -> None:
    """The wire-prove rule (longtail guidance): cohere's top_k is NOT the
    extra_body arm — v1's generic passthrough places it top-level in
    optional_params and the GPT assembly emits it verbatim."""
    case = CASES["top_k_wire_proven"]
    v1_body = run_v1_request_transform("cohere_chat", case)
    assert v1_body["top_k"] == 7
    assert "extra_body" not in v1_body
    result = _v2("cohere_chat", case)
    assert result.is_ok() and result.ok["top_k"] == 7


def test_tool_choice_never_reaches_the_wire() -> None:
    case = CASES["tool_choice_required_dropped"]
    v1_body = run_v1_request_transform("cohere", case)
    assert "tool_choice" not in v1_body
    result = _v2("cohere", case)
    assert result.is_ok() and "tool_choice" not in result.ok


def test_both_provider_names_share_one_config() -> None:
    from litellm.types.utils import LlmProviders
    from litellm.utils import ProviderConfigManager

    for provider in PROVIDERS:
        config = ProviderConfigManager.get_provider_chat_config(
            model="command-r", provider=LlmProviders(provider)
        )
        assert type(config).__name__ == "CohereV2ChatConfig"
