import pytest
from fastapi import HTTPException

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


def test_prepare_payload_uses_dynamic_overrides(grayswan_guardrail: GraySwanGuardrail) -> None:
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


def test_prepare_payload_falls_back_to_guardrail_defaults(grayswan_guardrail: GraySwanGuardrail) -> None:
    messages = [{"role": "user", "content": "hello"}]

    payload = grayswan_guardrail._prepare_payload(messages, {})

    assert payload["categories"] == {"safety": "general policy"}
    assert payload["policy_id"] == "default-policy"
    assert payload["reasoning_mode"] == "hybrid"


def test_process_response_does_not_block_under_threshold(grayswan_guardrail: GraySwanGuardrail) -> None:
    grayswan_guardrail._process_grayswan_response({"violation": 0.3, "violated_rules": []})


def test_process_response_blocks_when_threshold_exceeded() -> None:
    guardrail = GraySwanGuardrail(
        guardrail_name="grayswan-block",
        api_key="test-key",
        on_flagged_action="block",
        violation_threshold=0.2,
        event_hook=GuardrailEventHooks.pre_call,
    )

    with pytest.raises(HTTPException) as exc:
        guardrail._process_grayswan_response({"violation": 0.5, "violated_rules": [1]})

    assert exc.value.status_code == 400
    assert exc.value.detail["violation"] == 0.5


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
        self.calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        return _DummyResponse(self.payload)


@pytest.mark.asyncio
async def test_run_guardrail_posts_payload(monkeypatch, grayswan_guardrail: GraySwanGuardrail) -> None:
    dummy_client = _DummyClient({"violation": 0.1})
    grayswan_guardrail.async_handler = dummy_client

    captured = {}

    def fake_process(response_json: dict) -> None:
        captured["response"] = response_json

    monkeypatch.setattr(grayswan_guardrail, "_process_grayswan_response", fake_process)

    payload = {"messages": [{"role": "user", "content": "test"}]}

    await grayswan_guardrail.run_grayswan_guardrail(payload)

    assert dummy_client.calls[0]["json"] == payload
    assert captured["response"] == {"violation": 0.1}


@pytest.mark.asyncio
async def test_run_guardrail_raises_api_error(grayswan_guardrail: GraySwanGuardrail) -> None:
    class _FailingClient:
        async def post(self, **_kwargs):
            raise RuntimeError("boom")

    grayswan_guardrail.async_handler = _FailingClient()

    payload = {"messages": [{"role": "user", "content": "test"}]}

    with pytest.raises(GraySwanGuardrailAPIError):
        await grayswan_guardrail.run_grayswan_guardrail(payload)
