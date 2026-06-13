"""Differential parity for the watsonx chat request path (wave-2b-beta).

v1 side, invoked the way ``WatsonXChatHandler`` -> ``OpenAILikeChatHandler``
runs (NOT base_llm_http_handler): ``get_optional_params("watsonx")``, then
``_get_api_params`` (auth-key pops) + ``_prepare_payload`` (model_id +
project_id-or-space_id injection) merged into optional_params, then the
openai_like body assembly ``{"model": model_id, "messages":
<base-transformed>, **optional_params}`` with the UNCONDITIONAL
``stream: <bool>`` re-add. Auth (Authorization passthrough -> token ->
ZenApiKey -> the IAM-token network POST) is envelope and never appears
here; project/space ids ride ``TranslationDeps`` (the future seam fork
must run v1's resolution chain — deps.py docstring).
"""

import copy
import dataclasses
import json

import pytest

from litellm.exceptions import UnsupportedParamsError
from litellm.llms.watsonx.chat.transformation import IBMWatsonXChatConfig
from litellm.llms.watsonx.common_utils import _get_api_params
from litellm.utils import get_optional_params

from litellm.translation.engine.pipeline import translate_chat_request
from litellm.translation_seam import build_translation_deps

MODEL = "ibm/granite-3-8b-instruct"
_U = [{"role": "user", "content": "hi"}]
_CONFIG = IBMWatsonXChatConfig()

CASES = {
    "plain": {"model": MODEL, "messages": _U},
    "sampling": {
        "model": MODEL,
        "messages": _U,
        "temperature": 0.5,
        "top_p": 0.9,
        "max_tokens": 10,
        "stop": ["x"],
    },
    "stream_true": {"model": MODEL, "messages": _U, "stream": True},
    "explicit_stream_false": {"model": MODEL, "messages": _U, "stream": False},
    "reasoning_effort": {
        "model": MODEL,
        "messages": _U,
        "reasoning_effort": "high",
    },
    "response_format_json_object": {
        "model": MODEL,
        "messages": _U,
        "response_format": {"type": "json_object"},
    },
    "response_format_json_schema": {
        "model": MODEL,
        "messages": _U,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "s",
                "schema": {"type": "object", "properties": {}},
                "strict": True,
            },
        },
    },
    "tools_strict_and_additional_properties_stripped": {
        "model": MODEL,
        "messages": _U,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "f",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "a": {"type": "string", "strict": True},
                            "b": {"type": "object", "additionalProperties": True},
                        },
                    },
                },
            }
        ],
        "tool_choice": "auto",
    },
    "tool_choice_required_to_option": {
        "model": MODEL,
        "messages": _U,
        "tools": [{"type": "function", "function": {"name": "f", "parameters": {}}}],
        "tool_choice": "required",
    },
    "tool_choice_dict_rides_verbatim": {
        "model": MODEL,
        "messages": _U,
        "tools": [{"type": "function", "function": {"name": "f", "parameters": {}}}],
        "tool_choice": {"type": "function", "function": {"name": "f"}},
    },
    "tool_history": {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "f", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "res"},
        ],
    },
}

V1_RAISES = {
    "max_completion_tokens": (
        {"model": MODEL, "messages": _U, "max_completion_tokens": 5},
        "max_completion_tokens",
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

V1_SERVES_FALLBACKS = {
    "user_silent_drop": (
        {"model": MODEL, "messages": _U, "user": "u1"},
        "user",
    ),
    "seed_parse_level": (
        {"model": MODEL, "messages": _U, "seed": 42},
        "seed",
    ),
    "n_parse_level": ({"model": MODEL, "messages": _U, "n": 2}, "n"),
    "frequency_penalty_parse_level": (
        {"model": MODEL, "messages": _U, "frequency_penalty": 0.1},
        "frequency_penalty",
    ),
    "logprobs_parse_level": (
        {"model": MODEL, "messages": _U, "logprobs": True},
        "logprobs",
    ),
    "message_name_forwarded": (
        {"model": MODEL, "messages": [{"role": "user", "content": "hi", "name": "b"}]},
        "name",
    ),
}


def _deps(project_id: str | None = "proj-1", space_id: str | None = None):
    return dataclasses.replace(
        build_translation_deps(),
        watsonx_project_id=project_id,
        watsonx_space_id=space_id,
    )


def run_v1_request_transform(
    case: dict, project_id: str | None = "proj-1", space_id: str | None = None
) -> dict:
    """May RAISE (UnsupportedParamsError, the legacy-text ValueError, or
    WatsonXAIError 401): each raise IS pinned v1 behavior."""
    request = copy.deepcopy(case)
    model = request.pop("model")
    messages = request.pop("messages")
    optional_params = get_optional_params(
        model=model,
        custom_llm_provider="watsonx",
        messages=copy.deepcopy(messages),
        stream=request.pop("stream", None),
        **request,
    )
    optional_params["project_id"] = project_id
    optional_params["space_id"] = space_id
    api_params = _get_api_params(params=optional_params, model=model)
    payload = _CONFIG._prepare_payload(model=model, api_params=api_params)
    optional_params.update(payload)
    stream = optional_params.pop("stream", None) or False
    extra_body = optional_params.pop("extra_body", {})
    optional_params.pop("json_mode", None)
    optional_params.pop("max_retries", None)
    optional_params["stream"] = stream
    messages = _CONFIG._transform_messages(messages=messages, model=model)
    return {
        "model": payload.get("model_id") or "",
        "messages": messages,
        **optional_params,
        **extra_body,
    }


def _v2(case: dict, deps=None):
    return translate_chat_request(
        copy.deepcopy(case), "watsonx", deps if deps is not None else _deps()
    )


def _norm(body: dict) -> str:
    return json.dumps(body, sort_keys=True, default=str)


@pytest.mark.parametrize("name", sorted(CASES))
def test_v2_request_matches_v1(name: str) -> None:
    case = CASES[name]
    result = _v2(case)
    assert result.is_ok(), result.error.summary
    assert _norm(result.ok) == _norm(run_v1_request_transform(case))


def test_space_id_arm_matches_v1() -> None:
    result = _v2(CASES["plain"], deps=_deps(project_id=None, space_id="space-9"))
    assert result.is_ok(), result.error.summary
    v1 = run_v1_request_transform(CASES["plain"], project_id=None, space_id="space-9")
    assert _norm(result.ok) == _norm(v1)
    assert result.ok["space_id"] == "space-9" and "project_id" not in result.ok


@pytest.mark.parametrize("name", sorted(V1_RAISES))
def test_v1_raise_rows_fall_back_typed(name: str) -> None:
    case, fragment = V1_RAISES[name]
    result = _v2(case)
    assert result.is_error(), f"{name} unexpectedly translated: {result.ok!r}"
    assert fragment in result.error.summary, result.error.summary
    with pytest.raises(UnsupportedParamsError):
        run_v1_request_transform(case)


def test_top_k_falls_back_where_v1_raises_the_legacy_value_error() -> None:
    """top_k hits the watsonx-only legacy-text arm in get_optional_params —
    a bare ValueError naming the watsonx_text provider, NOT
    UnsupportedParamsError (pinned exact type)."""
    case = {"model": MODEL, "messages": _U, "top_k": 5}
    result = _v2(case)
    assert result.is_error()
    assert "watsonx_text" in result.error.summary
    with pytest.raises(ValueError, match="watsonx_text"):
        run_v1_request_transform(case)


def test_missing_project_and_space_fall_back_where_v1_raises_401() -> None:
    result = _v2(CASES["plain"], deps=_deps(project_id=None, space_id=None))
    assert result.is_error()
    assert "WatsonXAIError" in result.error.summary
    with pytest.raises(Exception, match="project_id and space_id"):
        run_v1_request_transform(CASES["plain"], project_id=None, space_id=None)


def test_deployment_model_falls_back() -> None:
    """v1 routes deployment/ models to deployment URLs with NO
    model_id/project_id payload and pops api_version into the URL —
    envelope behavior v2 does not reproduce; v1 serves."""
    case = {"model": "deployment/dep-1", "messages": _U}
    result = _v2(case)
    assert result.is_error()
    assert "deployment" in result.error.summary
    v1_body = run_v1_request_transform(case)
    assert v1_body["model"] == "" and "model_id" not in v1_body


@pytest.mark.parametrize("name", sorted(V1_SERVES_FALLBACKS))
def test_v1_serves_fallback_rows(name: str) -> None:
    case, fragment = V1_SERVES_FALLBACKS[name]
    result = _v2(case)
    assert result.is_error(), f"{name} unexpectedly translated: {result.ok!r}"
    assert fragment in result.error.summary, result.error.summary
    run_v1_request_transform(case)


def test_stream_key_always_on_the_wire() -> None:
    """The openai_like handler re-adds ``stream or False`` unconditionally:
    absent and explicit-false both serialize ``false`` — v2 serves BOTH
    (no stream:false guard arm; the IR collapse is invisible here)."""
    for case in (CASES["plain"], CASES["explicit_stream_false"]):
        v1_body = run_v1_request_transform(case)
        result = _v2(case)
        assert result.is_ok()
        assert v1_body["stream"] is False and result.ok["stream"] is False
