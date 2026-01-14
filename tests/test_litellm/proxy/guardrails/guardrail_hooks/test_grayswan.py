from typing import Optional

import pytest
from fastapi import HTTPException

from litellm.integrations.custom_guardrail import ModifyResponseException
from litellm.proxy.guardrails.guardrail_hooks.grayswan.grayswan import (
    GraySwanGuardrail,
    GraySwanGuardrailAPIError,
)
from litellm.types.guardrails import GuardrailEventHooks


@pytest.fixture
def grayswan_guardrail() -> GraySwanGuardrail:
    return GraySwanGuardrail(
        guardrail_name="grayswan-test",
        api_key="test-key",
        on_flagged_action="monitor",
        violation_threshold=0.5,
        categories={"safety": "general policy"},
        reasoning_mode="hybrid",
        policy_id="default-policy",
        event_hook=GuardrailEventHooks.pre_call,
    )


def test_prepare_payload_uses_dynamic_overrides(
    grayswan_guardrail: GraySwanGuardrail,
) -> None:
    messages = [{"role": "user", "content": "hello"}]
    dynamic_body = {
        "categories": {"custom": "override"},
        "policy_id": "dynamic-policy",
        "reasoning_mode": "thinking",
    }

    payload = grayswan_guardrail._prepare_payload(messages, dynamic_body)

    assert payload["messages"] == messages
    assert payload["categories"] == {"custom": "override"}
    assert payload["policy_id"] == "dynamic-policy"
    assert payload["reasoning_mode"] == "thinking"


def test_prepare_payload_falls_back_to_guardrail_defaults(
    grayswan_guardrail: GraySwanGuardrail,
) -> None:
    messages = [{"role": "user", "content": "hello"}]

    payload = grayswan_guardrail._prepare_payload(messages, {})

    assert payload["categories"] == {"safety": "general policy"}
    assert payload["policy_id"] == "default-policy"
    assert payload["reasoning_mode"] == "hybrid"


def test_process_response_does_not_block_under_threshold(
    grayswan_guardrail: GraySwanGuardrail,
) -> None:
    grayswan_guardrail._process_grayswan_response(
        {"violation": 0.3, "violated_rules": []}
    )


def test_process_response_blocks_when_threshold_exceeded() -> None:
    guardrail = GraySwanGuardrail(
        guardrail_name="grayswan-block",
        api_key="test-key",
        on_flagged_action="block",
        violation_threshold=0.2,
        event_hook=GuardrailEventHooks.pre_call,
    )

    # Test block mode with input violation (pre_call)
    with pytest.raises(HTTPException) as exc:
        guardrail._process_grayswan_response(
            {"violation": 0.5, "violated_rules": [1]},
            hook_type=GuardrailEventHooks.pre_call,
        )

    assert exc.value.status_code == 400
    assert exc.value.detail["violation"] == 0.5
    assert exc.value.detail["violation_location"] == "input"

    # Test block mode with output violation (post_call)
    with pytest.raises(HTTPException) as exc:
        guardrail._process_grayswan_response(
            {"violation": 0.5, "violated_rules": [1]},
            hook_type=GuardrailEventHooks.post_call,
        )

    assert exc.value.status_code == 400
    assert exc.value.detail["violation"] == 0.5
    assert exc.value.detail["violation_location"] == "output"


class _DummyResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class _DummyClient:
    def __init__(self, payload: dict):
        self.payload = payload
        self.calls: list[dict] = []

    async def post(self, *, url: str, headers: dict, json: dict, timeout: float):
        self.calls.append(
            {"url": url, "headers": headers, "json": json, "timeout": timeout}
        )
        return _DummyResponse(self.payload)


@pytest.mark.asyncio
async def test_run_guardrail_posts_payload(
    monkeypatch, grayswan_guardrail: GraySwanGuardrail
) -> None:
    dummy_client = _DummyClient({"violation": 0.1})
    grayswan_guardrail.async_handler = dummy_client

    captured = {}

    def fake_process(
        response_json: dict,
        data: Optional[dict] = None,
        hook_type: Optional[GuardrailEventHooks] = None,
    ) -> None:
        captured["response"] = response_json

    monkeypatch.setattr(grayswan_guardrail, "_process_grayswan_response", fake_process)

    payload = {"messages": [{"role": "user", "content": "test"}]}

    await grayswan_guardrail.run_grayswan_guardrail(payload)

    assert dummy_client.calls[0]["json"] == payload
    assert captured["response"] == {"violation": 0.1}


@pytest.mark.asyncio
async def test_run_guardrail_raises_api_error(
    grayswan_guardrail: GraySwanGuardrail,
) -> None:
    class _FailingClient:
        async def post(self, **_kwargs):
            raise RuntimeError("boom")

    grayswan_guardrail.async_handler = _FailingClient()

    payload = {"messages": [{"role": "user", "content": "test"}]}

    with pytest.raises(GraySwanGuardrailAPIError):
        await grayswan_guardrail.run_grayswan_guardrail(payload)


def test_process_response_passthrough_raises_exception_in_pre_call() -> None:
    """Test that passthrough mode raises ModifyResponseException in pre_call hook."""
    guardrail = GraySwanGuardrail(
        guardrail_name="grayswan-passthrough",
        api_key="test-key",
        on_flagged_action="passthrough",
        violation_threshold=0.2,
        event_hook=GuardrailEventHooks.pre_call,
    )

    data = {"messages": [{"role": "user", "content": "test"}], "model": "gpt-4"}
    response_json = {
        "violation": 0.8,
        "violated_rules": [1, 2],
        "mutation": True,
        "ipi": False,
    }

    # Should raise ModifyResponseException
    with pytest.raises(ModifyResponseException) as exc:
        guardrail._process_grayswan_response(
            response_json, data, GuardrailEventHooks.pre_call
        )

    assert "Gray Swan Cygnal Guardrail" in exc.value.message
    assert exc.value.model == "gpt-4"
    assert exc.value.detection_info["violation_score"] == 0.8
    assert exc.value.detection_info["violated_rules"] == [1, 2]


def test_process_response_passthrough_raises_exception_in_during_call() -> None:
    """Test that passthrough mode raises ModifyResponseException in during_call hook."""
    guardrail = GraySwanGuardrail(
        guardrail_name="grayswan-passthrough",
        api_key="test-key",
        on_flagged_action="passthrough",
        violation_threshold=0.2,
        event_hook=GuardrailEventHooks.during_call,
    )

    data = {"messages": [{"role": "user", "content": "test"}], "model": "gpt-4"}
    response_json = {
        "violation": 0.8,
        "violated_rules": [1, 2],
        "mutation": True,
        "ipi": False,
    }

    # Should raise ModifyResponseException
    with pytest.raises(ModifyResponseException) as exc:
        guardrail._process_grayswan_response(
            response_json, data, GuardrailEventHooks.during_call
        )

    assert "Gray Swan Cygnal Guardrail" in exc.value.message
    assert exc.value.model == "gpt-4"


def test_process_response_passthrough_stores_detection_info_in_post_call() -> None:
    """Test that passthrough mode stores detection info in post_call hook (not exception)."""
    guardrail = GraySwanGuardrail(
        guardrail_name="grayswan-passthrough",
        api_key="test-key",
        on_flagged_action="passthrough",
        violation_threshold=0.2,
        event_hook=GuardrailEventHooks.post_call,
    )

    data = {"messages": [{"role": "user", "content": "test"}]}
    response_json = {
        "violation": 0.8,
        "violated_rules": [1, 2],
        "mutation": True,
        "ipi": False,
    }

    # Should NOT raise an exception in post_call
    guardrail._process_grayswan_response(
        response_json, data, GuardrailEventHooks.post_call
    )

    # Verify detection info was stored in metadata
    assert "metadata" in data
    assert "guardrail_detections" in data["metadata"]
    assert len(data["metadata"]["guardrail_detections"]) == 1

    detection = data["metadata"]["guardrail_detections"][0]
    assert detection["guardrail"] == "grayswan"
    assert detection["flagged"] is True
    assert detection["violation_score"] == 0.8
    assert detection["violated_rules"] == [1, 2]
    assert detection["mutation"] is True
    assert detection["ipi"] is False


def test_process_response_passthrough_does_not_raise_if_under_threshold() -> None:
    """Test that passthrough mode doesn't raise exception if violation is under threshold."""
    guardrail = GraySwanGuardrail(
        guardrail_name="grayswan-passthrough",
        api_key="test-key",
        on_flagged_action="passthrough",
        violation_threshold=0.5,
        event_hook=GuardrailEventHooks.pre_call,
    )

    data = {"messages": [{"role": "user", "content": "test"}], "model": "gpt-4"}
    response_json = {
        "violation": 0.3,
        "violated_rules": [],
    }

    # Should not raise an exception since under threshold
    guardrail._process_grayswan_response(
        response_json, data, GuardrailEventHooks.pre_call
    )

    # Should not have any detection info since it didn't exceed threshold
    assert "guardrail_detections" not in data.get("metadata", {})


def test_format_violation_message() -> None:
    """Test that violation message is formatted correctly for input violations."""
    guardrail = GraySwanGuardrail(
        guardrail_name="grayswan-passthrough",
        api_key="test-key",
        on_flagged_action="passthrough",
        violation_threshold=0.5,
        event_hook=GuardrailEventHooks.pre_call,
    )

    detections = [
        {
            "guardrail": "grayswan",
            "flagged": True,
            "violation_score": 0.85,
            "violated_rules": [1, 3, 5],
            "mutation": True,
            "ipi": False,
        }
    ]

    # Test input violation message (pre_call/during_call)
    message = guardrail._format_violation_message(detections, is_output=False)

    assert "Sorry I can't help with that" in message
    assert "Gray Swan Cygnal Guardrail" in message
    assert "the input query has a violation score of 0.85" in message
    assert "violating the rule(s): 1, 3, 5" in message
    assert "Mutation effort to make the harmful intention disguised was DETECTED" in message
    # IPI should not be in message since it's False
    assert "Indirect Prompt Injection was DETECTED" not in message

    # Test output violation message (post_call)
    message = guardrail._format_violation_message(detections, is_output=True)

    assert "Sorry I can't help with that" in message
    assert "Gray Swan Cygnal Guardrail" in message
    assert "the model response has a violation score of 0.85" in message
    assert "violating the rule(s): 1, 3, 5" in message
    assert "Mutation effort to make the harmful intention disguised was DETECTED" in message
