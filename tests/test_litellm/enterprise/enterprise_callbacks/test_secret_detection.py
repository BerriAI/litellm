"""Tests for the hide-secrets guardrail (LIT-3548).

Covers the three defects from the ticket:
- ``apply_guardrail`` (the UI test playground path) must redact, not echo.
- Guardrail runs must record ``standard_logging_guardrail_information`` so
  Spend Logs / the guardrails monitor show activity, with hits ("mask" +
  masked_entity_count) distinguishable from clean requests ("allow").
- Defining ``apply_guardrail`` must NOT reroute proxied traffic off the
  native ``async_pre_call_hook`` (per-key opt-out and ``data["prompt"]``
  handling live only on the native path).
"""

import pytest

from litellm_enterprise.enterprise_callbacks.secret_detection import (
    _ENTERPRISE_SecretDetection,
)
from litellm.caching.caching import DualCache
from litellm.proxy._types import UserAPIKeyAuth

AWS_KEY = "AKIAIOSFODNN7EXAMPLE"


def _guardrail() -> _ENTERPRISE_SecretDetection:
    return _ENTERPRISE_SecretDetection(
        guardrail_name="hide-secrets", event_hook="pre_call", default_on=True
    )


def _recorded(request_data: dict) -> dict:
    entries = request_data["metadata"]["standard_logging_guardrail_information"]
    assert len(entries) == 1
    return entries[0]


@pytest.mark.asyncio
async def test_apply_guardrail_redacts_secrets():
    """Playground path: the returned texts must carry [REDACTED], not the secret."""
    guardrail = _guardrail()
    request_data: dict = {"metadata": {}}

    result = await guardrail.apply_guardrail(
        inputs={"texts": [f"my key is {AWS_KEY}, keep it safe"]},
        request_data=request_data,
        input_type="request",
    )

    assert result["texts"] == ["my key is [REDACTED], keep it safe"]

    recorded = _recorded(request_data)
    assert recorded["guardrail_status"] == "success"
    assert recorded["guardrail_response"] == "mask"
    assert recorded["guardrail_provider"] == "hide-secrets"
    assert recorded["masked_entity_count"] == {"AWS Access Key": 1}


@pytest.mark.asyncio
async def test_apply_guardrail_clean_text_records_allow():
    guardrail = _guardrail()
    request_data: dict = {"metadata": {}}

    result = await guardrail.apply_guardrail(
        inputs={"texts": ["nothing sensitive here"]},
        request_data=request_data,
        input_type="request",
    )

    assert result["texts"] == ["nothing sensitive here"]

    recorded = _recorded(request_data)
    assert recorded["guardrail_status"] == "success"
    assert recorded["guardrail_response"] == "allow"
    assert recorded["masked_entity_count"] == {}


@pytest.mark.asyncio
async def test_pre_call_hook_records_mask_with_entity_count():
    """Live-traffic path: a redaction must be visible in spend-log telemetry."""
    guardrail = _guardrail()
    data = {
        "messages": [{"role": "user", "content": f"use {AWS_KEY} for auth"}],
        "metadata": {},
    }

    await guardrail.async_pre_call_hook(
        user_api_key_dict=UserAPIKeyAuth(),
        cache=DualCache(),
        data=data,
        call_type="completion",
    )

    assert data["messages"][0]["content"] == "use [REDACTED] for auth"

    recorded = _recorded(data)
    assert recorded["guardrail_status"] == "success"
    assert recorded["guardrail_response"] == "mask"
    assert recorded["guardrail_provider"] == "hide-secrets"
    assert recorded["masked_entity_count"] == {"AWS Access Key": 1}


@pytest.mark.asyncio
async def test_pre_call_hook_clean_request_records_allow():
    """A request with no secrets must be distinguishable from a redacted one."""
    guardrail = _guardrail()
    data = {
        "messages": [{"role": "user", "content": "what's the weather"}],
        "metadata": {},
    }

    await guardrail.async_pre_call_hook(
        user_api_key_dict=UserAPIKeyAuth(),
        cache=DualCache(),
        data=data,
        call_type="completion",
    )

    recorded = _recorded(data)
    assert recorded["guardrail_status"] == "success"
    assert recorded["guardrail_response"] == "allow"
    assert recorded["masked_entity_count"] == {}


@pytest.mark.asyncio
async def test_pre_call_hook_opt_out_records_nothing():
    """A key with permissions={"hide_secrets": False} skips redaction, so no
    telemetry is recorded: every reader of a recorded entry (guardrail usage
    tracking, compliance checks, the spend-log viewer) counts it as a run."""
    guardrail = _guardrail()
    content = f"my key is {AWS_KEY}"
    data = {"messages": [{"role": "user", "content": content}], "metadata": {}}

    await guardrail.async_pre_call_hook(
        user_api_key_dict=UserAPIKeyAuth(permissions={"hide_secrets": False}),
        cache=DualCache(),
        data=data,
        call_type="completion",
    )

    assert data["messages"][0]["content"] == content  # untouched
    assert "standard_logging_guardrail_information" not in data["metadata"]


@pytest.mark.asyncio
async def test_pre_call_hook_still_redacts_text_completion_prompt():
    """data["prompt"] (str and list) is a native-hook-only surface; it must
    keep redacting now that the class also implements apply_guardrail."""
    guardrail = _guardrail()
    data = {"prompt": f"key {AWS_KEY} end", "metadata": {}}
    await guardrail.async_pre_call_hook(
        user_api_key_dict=UserAPIKeyAuth(),
        cache=DualCache(),
        data=data,
        call_type="completion",
    )
    assert data["prompt"] == "key [REDACTED] end"

    guardrail = _guardrail()
    data = {"prompt": [f"key {AWS_KEY}", "clean"], "metadata": {}}
    await guardrail.async_pre_call_hook(
        user_api_key_dict=UserAPIKeyAuth(),
        cache=DualCache(),
        data=data,
        call_type="completion",
    )
    assert data["prompt"] == ["key [REDACTED]", "clean"]


def test_proxied_traffic_stays_on_native_hooks():
    """Implementing apply_guardrail must not reroute proxied requests onto the
    unified path: that path skips ``should_run_check`` (per-key opt-out) and
    never sees ``data["prompt"]``."""
    guardrail = _guardrail()
    assert guardrail.uses_apply_guardrail_interface() is True
    assert guardrail._deployment_pre_call_target() is guardrail


@pytest.mark.asyncio
async def test_apply_guardrail_without_texts_records_nothing():
    """No inputs means nothing was inspected, so no "allow" row is recorded."""
    guardrail = _guardrail()
    request_data: dict = {"metadata": {}}

    result = await guardrail.apply_guardrail(
        inputs={"texts": []}, request_data=request_data, input_type="request"
    )

    assert result == {"texts": []}
    assert "standard_logging_guardrail_information" not in request_data["metadata"]


@pytest.mark.asyncio
async def test_legacy_nameless_instance_records_nothing():
    """``litellm_settings.callbacks: ["hide_secrets"]`` builds an arg-less
    instance with no guardrail_name. It still redacts, but recording a nameless
    entry would flip every spend row's guardrail status with nothing to join on."""
    guardrail = _ENTERPRISE_SecretDetection()
    data = {
        "messages": [{"role": "user", "content": f"use {AWS_KEY} for auth"}],
        "metadata": {},
    }

    await guardrail.async_pre_call_hook(
        user_api_key_dict=UserAPIKeyAuth(),
        cache=DualCache(),
        data=data,
        call_type="completion",
    )

    assert data["messages"][0]["content"] == "use [REDACTED] for auth"
    assert "standard_logging_guardrail_information" not in data["metadata"]
