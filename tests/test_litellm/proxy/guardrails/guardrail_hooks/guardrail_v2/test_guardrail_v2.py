import asyncio
import json

import pytest

from litellm.exceptions import GuardrailRaisedException
from litellm.proxy.guardrails.guardrail_hooks.guardrail_v2.guardrail_v2 import (
    GuardrailV2,
)


def _guardrail(verdict, capture=None, event_hook="pre_call"):
    """A GuardrailV2 whose engine call is dependency-injected to return a canned
    verdict, so the test runs without the compiled bridge."""

    def fake_apply(request_json: str) -> str:
        if capture is not None:
            capture["request"] = json.loads(request_json)
        return json.dumps({"verdict": verdict})

    guardrail = GuardrailV2(
        guardrail_type="openai_moderation",
        params={"api_key": "sk-test"},
        guardrail_name="t",
        event_hook=event_hook,
    )
    guardrail._apply_guardrail_fn = fake_apply
    return guardrail


def _apply(guardrail, inputs):
    return asyncio.run(
        guardrail.apply_guardrail(inputs, request_data={}, input_type="request")
    )


def test_block_verdict_raises():
    guardrail = _guardrail({"action": "block", "violation_message": "nope"})
    with pytest.raises(GuardrailRaisedException) as exc:
        _apply(guardrail, {"texts": ["bad"]})
    assert "nope" in str(exc.value)


def test_mask_verdict_replaces_texts():
    guardrail = _guardrail({"action": "mask", "texts": ["<EMAIL_ADDRESS>"]})
    out = _apply(guardrail, {"texts": ["jane@example.com"]})
    assert out["texts"] == ["<EMAIL_ADDRESS>"]


def test_pass_verdict_leaves_inputs_unchanged():
    guardrail = _guardrail({"action": "pass"})
    out = _apply(guardrail, {"texts": ["hello"]})
    assert out["texts"] == ["hello"]


def test_request_carries_batch_params_and_inputs():
    capture = {}
    guardrail = _guardrail({"action": "pass"}, capture=capture)
    _apply(guardrail, {"texts": ["hi"]})
    request = capture["request"]
    # The request now carries the whole guardrail set (here just self).
    assert request["guardrails"][0]["guardrail_type"] == "openai_moderation"
    assert request["guardrails"][0]["params"] == {"api_key": "sk-test"}
    assert request["input"]["texts"] == ["hi"]
    assert request["input_type"] == "request"


def test_batch_marker_does_not_leak_into_request_body():
    # The dedup marker must live in litellm metadata, never as a top-level request
    # key, or it gets forwarded to the provider as an unknown parameter.
    guardrail = _guardrail({"action": "pass"})
    rd: dict = {}
    asyncio.run(
        guardrail.apply_guardrail(
            {"texts": ["hi"]}, request_data=rd, input_type="request"
        )
    )
    assert all(not k.startswith("_litellm_rust_guardrail_batch") for k in rd)
    # Marker is scoped by hook phase (event_hook), not just input_type.
    assert (
        rd.get("metadata", {}).get("_litellm_rust_guardrail_batch_request_pre_call")
        is True
    )


def test_pre_call_marker_does_not_suppress_during_call_guardrail():
    """A pre_call guardrail marking the request must not cause a during_call
    guardrail to be skipped; that would bypass the later-phase check."""
    calls = {"during": 0}

    pre = _guardrail({"action": "pass"}, event_hook="pre_call")

    during = GuardrailV2(
        guardrail_type="openai_moderation",
        params={"api_key": "sk-test"},
        guardrail_name="t-during",
        event_hook="during_call",
    )

    def during_apply(_request_json: str) -> str:
        calls["during"] += 1
        return json.dumps({"verdict": {"action": "pass"}})

    during._apply_guardrail_fn = during_apply

    rd: dict = {}
    asyncio.run(
        pre.apply_guardrail({"texts": ["hi"]}, request_data=rd, input_type="request")
    )
    asyncio.run(
        during.apply_guardrail({"texts": ["hi"]}, request_data=rd, input_type="request")
    )
    assert (
        calls["during"] == 1
    ), "during_call guardrail was skipped by the pre_call marker"


def test_streaming_overrides_are_forwarded():
    guardrail = GuardrailV2(
        guardrail_type="openai_moderation",
        params={},
        streaming_end_of_stream_only=True,
        streaming_sampling_rate=9,
        guardrail_name="t",
    )
    assert guardrail.streaming_end_of_stream_only is True
    assert guardrail.streaming_sampling_rate == 9


def test_streaming_defaults_match_python_guardrail():
    guardrail = GuardrailV2(
        guardrail_type="openai_moderation", params={}, guardrail_name="t"
    )
    assert guardrail.streaming_end_of_stream_only is False
    assert guardrail.streaming_sampling_rate == 5
