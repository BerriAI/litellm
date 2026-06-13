"""Differential parity for the github_copilot request path (wave 3).

v1 side, invoked the way main.py's big openai elif runs for an
``openai_compatible_providers`` member:
``get_optional_params(custom_llm_provider="github_copilot")`` (OpenAIConfig's
model-shape fork) then ``GithubCopilotConfig().transform_request``
(OpenAIConfig.transform_request, which calls ``_transform_messages`` before
the base five-touch assembly). The serializer pins (all probed in-process at
HEAD with a SEEDED fake token dir — NO live OAuth):

- the system->assistant role rewrite: EVERY system message's ``role`` becomes
  ``assistant`` (content untouched) when the ambient
  ``litellm.disable_copilot_system_to_assistant`` flag is False (default), and
  rides unchanged when it is True — both states pinned two-sided below;
- max_completion_tokens passes VERBATIM (OpenAIConfig does NOT rename it);
- the body is otherwise the plain openai_compat assembly.

OAUTH SAFETY (researcher-5 GC-R1): a capability read on a github_copilot/ key
with no token cache does LIVE interactive OAuth (device-code prompt + polling
github.com). This module seeds ``GITHUB_COPILOT_TOKEN_DIR`` at a fake temp dir
and points the device-code/access-token/api-key URLs at an unroutable host
BEFORE importing anything that resolves the provider; the guard test asserts
the seed is in place and that no transform path opens a socket.
"""

import copy
import json
import os
import socket
import tempfile

import pytest

_TOKEN_DIR = tempfile.mkdtemp(prefix="gcp_tok_")
with open(os.path.join(_TOKEN_DIR, "api-key.json"), "w") as _f:
    json.dump(
        {
            "token": "fake-copilot-key",
            "expires_at": 9999999999,
            "endpoints": {"api": "https://api.fake-copilot.test"},
        },
        _f,
    )
with open(os.path.join(_TOKEN_DIR, "access-token"), "w") as _f:
    _f.write("fake-access-token")

# DI/seed only (NO monkeypatching): every env points at the fake dir / an
# unroutable host, so even a misfire cannot reach the real device flow.
os.environ.setdefault("GITHUB_COPILOT_TOKEN_DIR", _TOKEN_DIR)
os.environ.setdefault("GITHUB_COPILOT_DEVICE_CODE_URL", "http://127.0.0.1:9")
os.environ.setdefault("GITHUB_COPILOT_ACCESS_TOKEN_URL", "http://127.0.0.1:9")
os.environ.setdefault("GITHUB_COPILOT_API_KEY_URL", "http://127.0.0.1:9")

import litellm  # noqa: E402
from litellm.exceptions import UnsupportedParamsError  # noqa: E402
from litellm.llms.github_copilot.chat.transformation import (  # noqa: E402
    GithubCopilotConfig,
)
from litellm.utils import get_optional_params  # noqa: E402

from litellm.translation.engine.pipeline import translate_chat_request  # noqa: E402
from litellm.translation_seam import build_translation_deps  # noqa: E402

PROVIDER = "github_copilot"
MODEL = "gpt-4o"
CLAUDE = "claude-sonnet-4.5"

_U = [{"role": "user", "content": "hi"}]

CASES = {
    "plain": {"model": MODEL, "messages": _U},
    "system_to_assistant": {
        "model": MODEL,
        "messages": [{"role": "system", "content": "be brief"}, *_U],
    },
    "sampling": {
        "model": MODEL,
        "messages": _U,
        "temperature": 0.5,
        "top_p": 0.9,
        "max_tokens": 100,
        "stop": ["x", "y"],
    },
    "mct_verbatim_no_rename": {
        "model": MODEL,
        "messages": _U,
        "max_completion_tokens": 50,
    },
    "temperature_int_stays_int": {"model": MODEL, "messages": _U, "temperature": 1},
    "stream_true": {"model": MODEL, "messages": _U, "stream": True},
    "tools_auto": {
        "model": MODEL,
        "messages": [{"role": "user", "content": "w?"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "f",
                    "description": "d",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        "tool_choice": "auto",
    },
    "tool_choice_specific": {
        "model": MODEL,
        "messages": [{"role": "user", "content": "w?"}],
        "tools": [{"type": "function", "function": {"name": "f"}}],
        "tool_choice": {"type": "function", "function": {"name": "f"}},
    },
    "tool_call_compact_roundtrip": {
        "model": MODEL,
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
                            "name": "f",
                            "arguments": '{"city":"Paris"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
            {"role": "user", "content": "thanks"},
        ],
    },
    "claude_plain_serves": {"model": CLAUDE, "messages": _U},
    "response_format_json_schema": {
        "model": MODEL,
        "messages": _U,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "n",
                "schema": {"type": "object", "properties": {"a": {"type": "string"}}},
            },
        },
    },
}

# v1 RAISES UnsupportedParamsError (the model-shape / claude-DEAD-param gates,
# probed); v2 must be a typed fallback naming the surface.
V1_RAISES = {
    "gpt5_temperature_not_one": (
        {"model": "gpt-5", "messages": _U, "temperature": 0.5},
        "gpt-5",
    ),
    "claude_reasoning_effort": (
        {"model": CLAUDE, "messages": _U, "reasoning_effort": "high"},
        "reasoning_effort",
    ),
    "claude_thinking": (
        {"model": CLAUDE, "messages": _U, "thinking": {"type": "enabled"}},
        "thinking",
    ),
}

# v1 SERVES these (silent drop / verbatim passthrough / model-list gate); v2
# falls back typed so v1 keeps serving.
V1_SERVES_FALLBACKS = {
    "user_gpt4o_served_by_v1": ({"model": MODEL, "messages": _U, "user": "u1"}, "user"),
    "user_claude_dropped_by_v1": (
        {"model": CLAUDE, "messages": _U, "user": "u1"},
        "user",
    ),
    "top_k_not_a_chat_param": ({"model": MODEL, "messages": _U, "top_k": 7}, "top_k"),
    "seed_parse_level": ({"model": MODEL, "messages": _U, "seed": 42}, "seed"),
    "n_parse_level": ({"model": MODEL, "messages": _U, "n": 2}, "field"),
    "message_name_forwarded": (
        {"model": MODEL, "messages": [{"role": "user", "content": "hi", "name": "bob"}]},
        "name",
    ),
    "explicit_stream_false": (
        {"model": MODEL, "messages": _U, "stream": False},
        "stream",
    ),
    "string_stop": ({"model": MODEL, "messages": _U, "stop": "end"}, "stop"),
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
    return GithubCopilotConfig().transform_request(
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


def test_system_to_assistant_both_flag_states_match_v1() -> None:
    """GC-R4: the deps threading must be byte-equal for BOTH flag values. v1
    reads ``litellm.disable_copilot_system_to_assistant``; v2 reads the same
    value off ``TranslationDeps``. Toggle it in-process (DI, no monkeypatch)
    and assert the multi-system bodies match in both states."""
    case = {
        "model": MODEL,
        "messages": [{"role": "system", "content": "be brief"}, *_U],
    }
    original = litellm.disable_copilot_system_to_assistant
    try:
        litellm.disable_copilot_system_to_assistant = False
        v1_on = run_v1_request_transform(case)
        v2_on = _v2(case)
        assert v2_on.is_ok(), v2_on.error.summary
        assert v1_on["messages"][0]["role"] == "assistant"
        assert _norm(v2_on.ok) == _norm(v1_on)

        litellm.disable_copilot_system_to_assistant = True
        v1_off = run_v1_request_transform(case)
        v2_off = _v2(case)
        assert v2_off.is_ok(), v2_off.error.summary
        assert v1_off["messages"][0]["role"] == "system"
        assert _norm(v2_off.ok) == _norm(v1_off)
    finally:
        litellm.disable_copilot_system_to_assistant = original


def test_oauth_seed_is_in_place_and_no_socket_is_opened() -> None:
    """GC-R1 guard: every characterization path here must run with the token
    dir seeded and must NEVER reach the live device flow. Assert the seed env
    is set and that serializing a github_copilot request opens no socket (a
    real OAuth attempt would connect to the device-code URL)."""
    assert os.environ.get("GITHUB_COPILOT_TOKEN_DIR") == _TOKEN_DIR
    assert os.path.exists(os.path.join(_TOKEN_DIR, "api-key.json"))

    real_connect = socket.socket.connect
    connections: list[object] = []

    def _record(self, address):
        connections.append(address)
        raise AssertionError(f"transform path opened a socket to {address!r}")

    socket.socket.connect = _record  # type: ignore[method-assign]
    try:
        for case in CASES.values():
            run_v1_request_transform(case)
            _v2(case)
    finally:
        socket.socket.connect = real_connect  # type: ignore[method-assign]
    assert connections == []


def test_codex_family_stays_v1_by_responses_bridge() -> None:
    """The codex family bridges to the Responses API ABOVE the chat seam and
    never reaches this module. Pin that ``responses_api_bridge_check`` still
    returns mode "responses" for them, so an upstream flip to chat mode goes
    RED here and demands new chat rows (researcher-5 §1.2)."""
    from litellm.main import responses_api_bridge_check

    for model in ("gpt-5.3-codex", "gpt-5.1-codex-max"):
        model_info, _ = responses_api_bridge_check(model, PROVIDER)
        assert model_info.get("mode") == "responses", (model, model_info)
