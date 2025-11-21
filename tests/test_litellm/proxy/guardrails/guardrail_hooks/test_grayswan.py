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


def test_prepare_payload_uses_dynamic_overrides(
    grayswan_guardrail: GraySwanGuardrail,
) -> None:
    messages = [{"role": "user", "content": "hello"}]
    dynamic_body = {
        "categories": {"custom": "override"},
        "policy_id": "dynamic-policy",
        "reasoning_mode": "thinking",
    }

    payload = grayswan_guardrail._prepare_payload(messages, dynamic_body, None)

    assert payload["messages"] == messages
    assert payload["categories"] == {"custom": "override"}
    assert payload["policy_id"] == "dynamic-policy"
    assert payload["reasoning_mode"] == "thinking"


def test_prepare_payload_falls_back_to_guardrail_defaults(
    grayswan_guardrail: GraySwanGuardrail,
) -> None:
    messages = [{"role": "user", "content": "hello"}]

    payload = grayswan_guardrail._prepare_payload(messages, {}, None)

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

    def fake_process(response_json: dict) -> None:
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


def test_get_guardrail_custom_headers_with_valid_json(
    grayswan_guardrail: GraySwanGuardrail,
) -> None:
    """Test extraction of custom headers from request data with valid JSON."""
    import json

    request_data = {
        "proxy_server_request": {
            "headers": {
                "x-litellm-guardrail-grayswan-test": json.dumps(
                    {"policy_id": "header-policy-123", "reasoning_mode": "thinking"}
                )
            }
        }
    }

    custom_headers = grayswan_guardrail.get_guardrail_custom_headers(request_data)

    assert custom_headers == {
        "policy_id": "header-policy-123",
        "reasoning_mode": "thinking",
    }


def test_get_guardrail_custom_headers_with_dict_value(
    grayswan_guardrail: GraySwanGuardrail,
) -> None:
    """Test extraction of custom headers when header value is already a dict."""
    request_data = {
        "proxy_server_request": {
            "headers": {
                "x-litellm-guardrail-grayswan-test": {"policy_id": "header-policy-456"}
            }
        }
    }

    custom_headers = grayswan_guardrail.get_guardrail_custom_headers(request_data)

    assert custom_headers == {"policy_id": "header-policy-456"}


def test_get_guardrail_custom_headers_with_invalid_json(
    grayswan_guardrail: GraySwanGuardrail,
) -> None:
    """Test that invalid JSON in header gracefully returns empty dict."""
    request_data = {
        "proxy_server_request": {
            "headers": {"x-litellm-guardrail-grayswan-test": "invalid json {"}
        }
    }

    custom_headers = grayswan_guardrail.get_guardrail_custom_headers(request_data)

    assert custom_headers == {}


def test_get_guardrail_custom_headers_case_insensitive(
    grayswan_guardrail: GraySwanGuardrail,
) -> None:
    """Test that header matching is case-insensitive."""
    import json

    request_data = {
        "proxy_server_request": {
            "headers": {
                "X-LiteLLM-Guardrail-GraySwan-Test": json.dumps(
                    {"policy_id": "case-test"}
                )
            }
        }
    }

    custom_headers = grayswan_guardrail.get_guardrail_custom_headers(request_data)

    assert custom_headers == {"policy_id": "case-test"}


def test_get_guardrail_custom_headers_missing_header(
    grayswan_guardrail: GraySwanGuardrail,
) -> None:
    """Test that missing header returns empty dict."""
    request_data = {"proxy_server_request": {"headers": {}}}

    custom_headers = grayswan_guardrail.get_guardrail_custom_headers(request_data)

    assert custom_headers == {}


def test_prepare_payload_ignores_custom_headers_for_config_params(
    grayswan_guardrail: GraySwanGuardrail,
) -> None:
    """Test that custom headers are NOT used for config params like policy_id and reasoning_mode."""
    messages = [{"role": "user", "content": "hello"}]
    custom_headers = {"policy_id": "header-policy", "reasoning_mode": "thinking"}
    dynamic_body = {"policy_id": "dynamic-policy", "reasoning_mode": "hybrid"}

    payload = grayswan_guardrail._prepare_payload(
        messages, dynamic_body, custom_headers
    )

    # Custom headers should be ignored - dynamic_body takes priority
    assert (
        payload["policy_id"] == "dynamic-policy"
    )  # Dynamic body wins, not custom header
    assert payload["reasoning_mode"] == "hybrid"  # Dynamic body wins, not custom header


def test_prepare_payload_priority_dynamic_body_over_config(
    grayswan_guardrail: GraySwanGuardrail,
) -> None:
    """Test that dynamic_body takes priority over config."""
    messages = [{"role": "user", "content": "hello"}]
    custom_headers = {}
    dynamic_body = {"policy_id": "dynamic-policy"}

    payload = grayswan_guardrail._prepare_payload(
        messages, dynamic_body, custom_headers
    )

    assert payload["policy_id"] == "dynamic-policy"  # Dynamic body wins
    assert payload["reasoning_mode"] == "hybrid"  # Falls back to config


def test_prepare_payload_priority_config_fallback(
    grayswan_guardrail: GraySwanGuardrail,
) -> None:
    """Test that config values are used when dynamic_body is empty."""
    messages = [{"role": "user", "content": "hello"}]
    custom_headers = {}
    dynamic_body = {}

    payload = grayswan_guardrail._prepare_payload(
        messages, dynamic_body, custom_headers
    )

    assert payload["policy_id"] == "default-policy"  # Config value
    assert payload["reasoning_mode"] == "hybrid"  # Config value


@pytest.mark.asyncio
async def test_run_guardrail_with_custom_headers(
    monkeypatch, grayswan_guardrail: GraySwanGuardrail
) -> None:
    """Test that custom HTTP headers are passed to the API request."""
    dummy_client = _DummyClient({"violation": 0.1})
    grayswan_guardrail.async_handler = dummy_client

    def fake_process(response_json: dict) -> None:
        pass

    monkeypatch.setattr(grayswan_guardrail, "_process_grayswan_response", fake_process)

    payload = {"messages": [{"role": "user", "content": "test"}]}
    custom_headers = {
        "custom-header": "test-value",
    }

    await grayswan_guardrail.run_grayswan_guardrail(payload, custom_headers)

    # Verify custom HTTP header was included
    assert "custom-header" in dummy_client.calls[0]["headers"]
    assert dummy_client.calls[0]["headers"]["custom-header"] == "test-value"
