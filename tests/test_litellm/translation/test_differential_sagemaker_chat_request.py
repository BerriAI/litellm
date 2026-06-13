"""Differential parity for the sagemaker_chat request path (wave-2b-beta).

v1 side: ``get_optional_params("sagemaker_chat")`` (the BASE OpenAI GPT
list — the widest wave-2b surface) + the base ``transform_request`` (no
overrides). SigV4 signs the assembled body AFTER transform (envelope, the
bedrock sign-after-body precedent) and the endpoint name doubles as the
model field. sagemaker_nova shares the main.py branch but carries its own
config and is deliberately NOT registered (v1 fallback).
"""

import copy
import json

import pytest

from litellm.exceptions import UnsupportedParamsError
from litellm.llms.sagemaker.chat.transformation import SagemakerChatConfig
from litellm.utils import get_optional_params

from litellm.translation.engine.pipeline import translate_chat_request
from litellm.translation_seam import build_translation_deps

MODEL = "my-endpoint"
_U = [{"role": "user", "content": "hi"}]

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
    "mct_verbatim_no_rename": {
        "model": MODEL,
        "messages": _U,
        "max_completion_tokens": 50,
    },
    "top_k_wire_proven": {"model": MODEL, "messages": _U, "top_k": 7},
    "stream_true": {"model": MODEL, "messages": _U, "stream": True},
    "tools": {
        "model": MODEL,
        "messages": _U,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        "tool_choice": "auto",
    },
    "response_format_json_object": {
        "model": MODEL,
        "messages": _U,
        "response_format": {"type": "json_object"},
    },
    "parallel_tool_calls": {
        "model": MODEL,
        "messages": _U,
        "tools": [{"type": "function", "function": {"name": "f", "parameters": {}}}],
        "parallel_tool_calls": True,
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
    "thinking": (
        {"model": MODEL, "messages": _U, "thinking": {"type": "enabled"}},
        "thinking",
    ),
    "reasoning_effort": (
        {"model": MODEL, "messages": _U, "reasoning_effort": "high"},
        "reasoning_effort",
    ),
    "response_format_on_gpt4_named_endpoint": (
        {
            "model": "gpt-4",
            "messages": _U,
            "response_format": {"type": "json_object"},
        },
        "gpt-4",
    ),
}

V1_SERVES_FALLBACKS = {
    "user_silent_drop": ({"model": MODEL, "messages": _U, "user": "u1"}, "user"),
    "seed_parse_level": ({"model": MODEL, "messages": _U, "seed": 42}, "seed"),
    "n_parse_level": ({"model": MODEL, "messages": _U, "n": 2}, "n"),
    "logit_bias_parse_level": (
        {"model": MODEL, "messages": _U, "logit_bias": {"1": 1}},
        "logit_bias",
    ),
    "aws_kwarg_rides_v1_body": (
        # v1 places aws_* kwargs in optional_params and the BODY carries
        # them (wire-probed); v2 rejects the unknown inbound key
        {"model": MODEL, "messages": _U, "aws_region_name": "us-east-1"},
        "aws_region_name",
    ),
    "explicit_stream_false": (
        {"model": MODEL, "messages": _U, "stream": False},
        "stream",
    ),
    "message_name_forwarded": (
        {"model": MODEL, "messages": [{"role": "user", "content": "hi", "name": "b"}]},
        "name",
    ),
}


def run_v1_request_transform(case: dict) -> dict:
    request = copy.deepcopy(case)
    model = request.pop("model")
    messages = request.pop("messages")
    optional_params = get_optional_params(
        model=model,
        custom_llm_provider="sagemaker_chat",
        messages=copy.deepcopy(messages),
        stream=request.pop("stream", None),
        **request,
    )
    optional_params.pop("extra_body", None)
    return SagemakerChatConfig().transform_request(
        model=model,
        messages=messages,
        optional_params=optional_params,
        litellm_params={},
        headers={},
    )


def _v2(case: dict):
    return translate_chat_request(
        copy.deepcopy(case), "sagemaker_chat", build_translation_deps()
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
    run_v1_request_transform(case)


def test_mct_passes_verbatim_no_rename() -> None:
    v1_body = run_v1_request_transform(CASES["mct_verbatim_no_rename"])
    result = _v2(CASES["mct_verbatim_no_rename"])
    assert result.is_ok()
    assert v1_body["max_completion_tokens"] == 50
    assert result.ok["max_completion_tokens"] == 50
    assert "max_tokens" not in result.ok


def test_top_k_rides_the_wire_top_level() -> None:
    v1_body = run_v1_request_transform(CASES["top_k_wire_proven"])
    assert v1_body["top_k"] == 7 and "extra_body" not in v1_body
    result = _v2(CASES["top_k_wire_proven"])
    assert result.is_ok() and result.ok["top_k"] == 7


def test_aws_kwargs_reach_v1_wire_body() -> None:
    """The quirk behind the aws fallback row: v1's body literally carries
    aws_region_name (SigV4 then signs it); v2's parse rejects the unknown
    key so v1 keeps serving those requests."""
    v1_body = run_v1_request_transform(
        V1_SERVES_FALLBACKS["aws_kwarg_rides_v1_body"][0]
    )
    assert v1_body["aws_region_name"] == "us-east-1"


def test_sagemaker_nova_stays_unregistered() -> None:
    from litellm.translation.engine import pipeline

    assert "sagemaker_nova" not in pipeline._SERIALIZERS
    result = translate_chat_request(
        copy.deepcopy(CASES["plain"]), "sagemaker_nova", build_translation_deps()  # type: ignore[arg-type]
    )
    assert result.is_error()
