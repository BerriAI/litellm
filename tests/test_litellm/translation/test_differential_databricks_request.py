"""Differential parity for the databricks request path (wave 3).

v1 side, invoked the way main.py's databricks elif runs:
``get_optional_params(custom_llm_provider="databricks")`` then
``DatabricksConfig().transform_request`` (probed in-process at HEAD). The
central structural fact is the ``"claude" in model`` SUBSTRING fork (DB-R1):
every served case is parameterized claude x non-claude so both arms are
pinned. The serializer pins: the ``{model, messages, stream}`` body with
``stream`` ALWAYS present (default false), mct -> ``max_tokens`` (ALWAYS
renamed, v1 never re-emits max_completion_tokens), ``top_k`` verbatim
top-level (researcher-5 said unsupported; the HEAD probe REFUTES it, both
arms), claude tools via the openai->anthropic->databricks round-trip (drops
function-level ``strict``/``$schema``/cache_control, repositions description
after parameters) vs non-claude tools verbatim, the thinking max-bump
(``budget_tokens + DEFAULT_MAX_TOKENS`` only when the caller gave no
max_tokens), and reasoning_effort verbatim on NON-claude WITH max_tokens.

The fallback rows (v1 SERVES its own behavior, v2 falls back typed so the
seam keeps serving v1): claude response_format (the json_tool_call machinery;
DB-R2 json_object is SILENTLY DROPPED) and claude reasoning_effort (the
thinking machinery). The raise rows: parallel_tool_calls
(UnsupportedParamsError) and the DB-R3 raw ``KeyError('thinking')`` crash on
NON-claude reasoning_effort WITHOUT max_tokens.
"""

import copy
import json

import pytest

from litellm.exceptions import UnsupportedParamsError
from litellm.llms.databricks.chat.transformation import DatabricksConfig
from litellm.utils import get_optional_params

from litellm.translation.engine.pipeline import translate_chat_request
from litellm.translation_seam import build_translation_deps

PROVIDER = "databricks"
CLAUDE = "databricks-claude-3-7-sonnet"
NONCLAUDE = "databricks-dbrx-instruct"

_U = [{"role": "user", "content": "hi"}]

_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "f",
            "description": "d",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {"a": {"type": "string"}},
                "$schema": "http://json-schema.org/draft-07/schema#",
            },
        },
    }
]

# Served shapes, each run on BOTH model arms (the "claude" substring fork).
# (name, request-without-model)
_SHARED_CASES = {
    "plain": {"messages": _U},
    "sampling": {
        "messages": _U,
        "temperature": 0.5,
        "top_p": 0.9,
        "stop": ["x", "y"],
    },
    "mct_to_max_tokens": {"messages": _U, "max_completion_tokens": 50},
    "max_tokens_verbatim": {"messages": _U, "max_tokens": 80},
    "top_k_top_level": {"messages": _U, "top_k": 7},
    "stream_true": {"messages": _U, "stream": True},
    "stream_false_always_present": {"messages": _U, "stream": False},
    "tool_choice_auto": {
        "messages": _U,
        "tools": copy.deepcopy(_TOOL),
        "tool_choice": "auto",
    },
    "tool_choice_required": {
        "messages": _U,
        "tools": copy.deepcopy(_TOOL),
        "tool_choice": "required",
    },
    "tool_choice_specific": {
        "messages": _U,
        "tools": copy.deepcopy(_TOOL),
        "tool_choice": {"type": "function", "function": {"name": "f"}},
    },
    "thinking_budget_bumps_max_tokens": {
        "messages": _U,
        "thinking": {"type": "enabled", "budget_tokens": 1024},
    },
    "thinking_budget_with_caller_max_tokens_no_bump": {
        "messages": _U,
        "thinking": {"type": "enabled", "budget_tokens": 1024},
        "max_tokens": 100,
    },
    "thinking_no_budget_no_bump": {
        "messages": _U,
        "thinking": {"type": "enabled"},
    },
}

# These differ between arms: tools (claude round-trip drops strict/$schema),
# response_format (non-claude serves; claude falls back), reasoning_effort
# (non-claude+max_tokens serves; claude falls back). Keyed by arm.
_NONCLAUDE_ONLY_CASES = {
    "tools_verbatim_nonclaude": {"messages": _U, "tools": copy.deepcopy(_TOOL)},
    "response_format_json_object_nonclaude": {
        "messages": _U,
        "response_format": {"type": "json_object"},
    },
    "response_format_json_schema_nonclaude": {
        "messages": _U,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "n",
                "schema": {"type": "object", "properties": {"a": {"type": "string"}}},
            },
        },
    },
    "reasoning_effort_with_max_tokens_nonclaude": {
        "messages": _U,
        "reasoning_effort": "low",
        "max_tokens": 100,
    },
}

_CLAUDE_ONLY_CASES = {
    "tools_roundtrip_drops_strict_and_schema_claude": {
        "messages": _U,
        "tools": copy.deepcopy(_TOOL),
    },
}

# v1 RAISES UnsupportedParamsError; v2 falls back typed naming the surface.
_V1_RAISES = {
    "parallel_tool_calls": (
        {"model": NONCLAUDE, "messages": _U, "parallel_tool_calls": True},
        "parallel_tool_calls",
    ),
}

# v1 CRASHES with a raw KeyError (NOT UnsupportedParamsError); v2 falls back
# typed so v1 serves its own crash (DB-R3).
_V1_RAISES_RAW = {
    "nonclaude_reasoning_effort_without_max_tokens_keyerror": (
        {"model": NONCLAUDE, "messages": _U, "reasoning_effort": "low"},
        "reasoning_effort",
    ),
}

# v1 SERVES (silent drop / unported machinery); v2 falls back typed so v1
# keeps serving. Each names the v1 path.
_V1_SERVES_FALLBACKS = {
    "claude_response_format_json_object_silently_dropped": (
        {"model": CLAUDE, "messages": _U, "response_format": {"type": "json_object"}},
        "response_format",
    ),
    "claude_response_format_json_schema_machinery": (
        {
            "model": CLAUDE,
            "messages": _U,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "n", "schema": {"type": "object"}},
            },
        },
        "response_format",
    ),
    "claude_reasoning_effort_thinking_machinery": (
        {"model": CLAUDE, "messages": _U, "reasoning_effort": "low"},
        "reasoning_effort",
    ),
    "user_silent_drop": (
        {"model": NONCLAUDE, "messages": _U, "user": "u1"},
        "user",
    ),
    "n_parse_level_unknown": (
        {"model": NONCLAUDE, "messages": _U, "n": 2},
        "not yet supported by translation v2: n",
    ),
}


def run_v1_request_transform(case: dict) -> dict:
    """May RAISE (UnsupportedParamsError / KeyError) — that IS the pinned v1
    behavior for the raise rows."""
    request = copy.deepcopy(case)
    model = request.pop("model")
    messages = request.pop("messages")
    optional_params = get_optional_params(
        model=model,
        custom_llm_provider=PROVIDER,
        messages=copy.deepcopy(messages),
        **request,
    )
    optional_params.pop("extra_body", None)
    return DatabricksConfig().transform_request(
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


def _shared_params() -> list[tuple[str, str]]:
    return [
        (name, model)
        for name in sorted(_SHARED_CASES)
        for model in (CLAUDE, NONCLAUDE)
    ]


@pytest.mark.parametrize("name,model", _shared_params())
def test_v2_shared_request_matches_v1_both_arms(name: str, model: str) -> None:
    case = {"model": model, **_SHARED_CASES[name]}
    result = _v2(case)
    assert result.is_ok(), f"{name}/{model}: {result.error.summary}"
    assert _norm(result.ok) == _norm(run_v1_request_transform(case))


@pytest.mark.parametrize("name", sorted(_NONCLAUDE_ONLY_CASES))
def test_v2_nonclaude_request_matches_v1(name: str) -> None:
    case = {"model": NONCLAUDE, **_NONCLAUDE_ONLY_CASES[name]}
    result = _v2(case)
    assert result.is_ok(), f"{name}: {result.error.summary}"
    assert _norm(result.ok) == _norm(run_v1_request_transform(case))


@pytest.mark.parametrize("name", sorted(_CLAUDE_ONLY_CASES))
def test_v2_claude_request_matches_v1(name: str) -> None:
    case = {"model": CLAUDE, **_CLAUDE_ONLY_CASES[name]}
    result = _v2(case)
    assert result.is_ok(), f"{name}: {result.error.summary}"
    assert _norm(result.ok) == _norm(run_v1_request_transform(case))


@pytest.mark.parametrize("name", sorted(_V1_RAISES))
def test_v1_raise_rows_fall_back_typed(name: str) -> None:
    case, fragment = _V1_RAISES[name]
    result = _v2(case)
    assert result.is_error(), f"{name} unexpectedly translated: {result.ok!r}"
    assert fragment in result.error.summary, result.error.summary
    with pytest.raises(UnsupportedParamsError):
        run_v1_request_transform(case)


@pytest.mark.parametrize("name", sorted(_V1_RAISES_RAW))
def test_v1_raw_keyerror_rows_fall_back_typed(name: str) -> None:
    case, fragment = _V1_RAISES_RAW[name]
    result = _v2(case)
    assert result.is_error(), f"{name} unexpectedly translated: {result.ok!r}"
    assert fragment in result.error.summary, result.error.summary
    with pytest.raises(KeyError):
        run_v1_request_transform(case)


@pytest.mark.parametrize("name", sorted(_V1_SERVES_FALLBACKS))
def test_v1_serves_fallback_rows(name: str) -> None:
    case, fragment = _V1_SERVES_FALLBACKS[name]
    result = _v2(case)
    assert result.is_error(), f"{name} unexpectedly translated: {result.ok!r}"
    assert fragment in result.error.summary, result.error.summary
    # v1 serves the same request without raising.
    run_v1_request_transform(case)


def test_claude_json_object_is_silently_dropped_by_v1() -> None:
    """DB-R2: the probe-found trap a code-reading port would get wrong. v1
    does NOT raise on claude json_object — it silently drops it ENTIRELY (the
    body carries no response_format), so v2 must fall back (not match the drop
    and not serve the format)."""
    case = {
        "model": CLAUDE,
        "messages": _U,
        "response_format": {"type": "json_object"},
    }
    v1 = run_v1_request_transform(case)
    assert "response_format" not in v1
    assert _v2(case).is_error()


def test_claude_tools_roundtrip_drops_strict_and_schema() -> None:
    """The claude substring fork runs openai->anthropic->databricks: the
    function-level strict and the parameters $schema are DROPPED, while
    additionalProperties/properties/required/type survive."""
    case = {"model": CLAUDE, "messages": _U, "tools": copy.deepcopy(_TOOL)}
    v1 = run_v1_request_transform(case)
    function = v1["tools"][0]["function"]
    assert "strict" not in v1["tools"][0] and "strict" not in function
    assert "$schema" not in function["parameters"]
    assert function["parameters"]["properties"] == {"a": {"type": "string"}}
    result = _v2(case)
    assert result.is_ok()
    assert _norm(result.ok) == _norm(v1)


def test_nonclaude_tools_keep_strict_and_schema_verbatim() -> None:
    """The non-claude arm sends tools VERBATIM (v1's ``if "claude" not in
    model: return tools``) — the discriminator the claude arm pins against."""
    case = {"model": NONCLAUDE, "messages": _U, "tools": copy.deepcopy(_TOOL)}
    v1 = run_v1_request_transform(case)
    assert v1["tools"][0]["function"]["strict"] is True
    assert "$schema" in v1["tools"][0]["function"]["parameters"]
    result = _v2(case)
    assert result.is_ok()
    assert _norm(result.ok) == _norm(v1)


def test_thinking_max_token_bump_arithmetic() -> None:
    """v1 sets max_tokens = budget_tokens + DEFAULT_MAX_TOKENS only when the
    caller gave none; a caller value is never adjusted (probed both arms)."""
    from litellm.constants import DEFAULT_MAX_TOKENS

    for model in (CLAUDE, NONCLAUDE):
        bumped = run_v1_request_transform(
            {
                "model": model,
                "messages": _U,
                "thinking": {"type": "enabled", "budget_tokens": 1024},
            }
        )
        assert bumped["max_tokens"] == 1024 + DEFAULT_MAX_TOKENS
        unbumped = run_v1_request_transform(
            {
                "model": model,
                "messages": _U,
                "thinking": {"type": "enabled", "budget_tokens": 1024},
                "max_tokens": 100,
            }
        )
        assert unbumped["max_tokens"] == 100


def test_anthropic_input_schema_key_mirror_matches_v1() -> None:
    """The claude tool filter keeps exactly AnthropicInputSchema's keys; drift
    here silently changes which schema keys survive the round-trip."""
    from litellm.types.llms.anthropic import AnthropicInputSchema

    from litellm.translation.providers.databricks.tools import (
        _ANTHROPIC_INPUT_SCHEMA_KEYS,
    )

    assert _ANTHROPIC_INPUT_SCHEMA_KEYS == frozenset(
        AnthropicInputSchema.__annotations__
    )
