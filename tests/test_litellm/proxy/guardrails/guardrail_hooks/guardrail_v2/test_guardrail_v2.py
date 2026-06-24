import asyncio
import json

import pytest

from litellm.exceptions import GuardrailRaisedException
from litellm.proxy.guardrails.guardrail_hooks.guardrail_v2.guardrail_v2 import (
    GuardrailV2,
)


def _guardrail(verdict, capture=None):
    """A GuardrailV2 whose engine call is dependency-injected to return a canned
    verdict, so the test runs without the compiled bridge."""

    def fake_apply(request_json: str) -> str:
        if capture is not None:
            capture["request"] = json.loads(request_json)
        return json.dumps({"verdict": verdict})

    return GuardrailV2(
        guardrail_type="openai_moderation",
        params={"api_key": "sk-test"},
        apply_guardrail_fn=fake_apply,
        guardrail_name="t",
    )


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


def test_request_carries_type_params_and_inputs():
    capture = {}
    guardrail = _guardrail({"action": "pass"}, capture=capture)
    _apply(guardrail, {"texts": ["hi"]})
    request = capture["request"]
    assert request["guardrail_type"] == "openai_moderation"
    assert request["params"] == {"api_key": "sk-test"}
    assert request["input"]["texts"] == ["hi"]
    assert request["input_type"] == "request"


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
