"""Tests for ``apply_guardrail`` fail-open/fail-closed behavior, missing secrets, and helpers."""

import sys
from unittest.mock import Mock

import pytest
from fastapi import HTTPException


@pytest.mark.asyncio
async def test_apply_guardrail_missing_app_id_fail_closed_returns_500(guardrail_and_client, make_request_data):
    """Missing app_id follows the fail_open pattern: fail_open=False → HTTP 500."""
    guardrail, _ = guardrail_and_client
    with pytest.raises(HTTPException) as exc:
        await guardrail.apply_guardrail(
            inputs={"texts": ["hi"]},
            request_data=make_request_data(metadata={}),
            input_type="request",
        )
    assert exc.value.status_code == 500
    assert "Error in Alice WonderFence Guardrail" in exc.value.detail["error"]
    assert "alice_wonderfence_app_id" in exc.value.detail["exception"]


@pytest.mark.asyncio
async def test_apply_guardrail_missing_api_key_fail_closed_returns_500(monkeypatch, make_guardrail, make_request_data):
    """Missing api_key follows the fail_open pattern: fail_open=False → HTTP 500."""
    monkeypatch.delenv("ALICE_API_KEY", raising=False)
    guardrail, _ = make_guardrail(api_key=None)
    with pytest.raises(HTTPException) as exc:
        await guardrail.apply_guardrail(
            inputs={"texts": ["hi"]},
            request_data=make_request_data(),
            input_type="request",
        )
    assert exc.value.status_code == 500
    assert "Error in Alice WonderFence Guardrail" in exc.value.detail["error"]
    assert "alice_wonderfence_api_key" in exc.value.detail["exception"]


@pytest.mark.asyncio
async def test_apply_guardrail_missing_app_id_fail_open_returns_500(make_guardrail, make_request_data):
    """Missing app_id is a config error: never fail-open, even with fail_open=True."""
    guardrail, _ = make_guardrail(fail_open=True)
    with pytest.raises(HTTPException) as exc:
        await guardrail.apply_guardrail(
            inputs={"texts": ["hi"]},
            request_data=make_request_data(metadata={}),
            input_type="request",
        )
    assert exc.value.status_code == 500
    assert "alice_wonderfence_app_id" in exc.value.detail["exception"]


@pytest.mark.asyncio
async def test_apply_guardrail_missing_api_key_fail_open_returns_500(monkeypatch, make_guardrail, make_request_data):
    """Missing api_key is a config error: never fail-open, even with fail_open=True."""
    monkeypatch.delenv("ALICE_API_KEY", raising=False)
    guardrail, _ = make_guardrail(api_key=None, fail_open=True)
    with pytest.raises(HTTPException) as exc:
        await guardrail.apply_guardrail(
            inputs={"texts": ["hi"]},
            request_data=make_request_data(),
            input_type="request",
        )
    assert exc.value.status_code == 500
    assert "alice_wonderfence_api_key" in exc.value.detail["exception"]


@pytest.mark.asyncio
async def test_apply_guardrail_fail_open_swallows_transport_error(make_guardrail, make_request_data):
    guardrail, client = make_guardrail(fail_open=True)
    guardrail._client_cache["default-api-key"] = client
    client.evaluate_prompt.side_effect = RuntimeError("network down")

    inputs = {"texts": ["original"]}
    out = await guardrail.apply_guardrail(
        inputs=inputs,
        request_data=make_request_data(),
        input_type="request",
    )
    assert out["texts"] == ["original"]


@pytest.mark.asyncio
async def test_apply_guardrail_fail_closed_returns_500(guardrail_and_client, make_request_data):
    guardrail, client = guardrail_and_client
    client.evaluate_prompt.side_effect = RuntimeError("network down")

    with pytest.raises(HTTPException) as exc:
        await guardrail.apply_guardrail(
            inputs={"texts": ["hi"]},
            request_data=make_request_data(),
            input_type="request",
        )
    assert exc.value.status_code == 500
    assert "Error in Alice WonderFence Guardrail" in exc.value.detail["error"]


@pytest.mark.asyncio
async def test_malformed_override_does_not_fail_open(make_guardrail, make_request_data):
    """A non-string request-metadata app_id override must not slip through under
    fail_open: it resolves to a config error (500), not a swallowed exception
    that skips scanning. The SDK is never called with a malformed value."""
    guardrail, client = make_guardrail(fail_open=True, allow_request_metadata_override=True)
    guardrail._client_cache["default-api-key"] = client

    with pytest.raises(HTTPException) as exc:
        await guardrail.apply_guardrail(
            inputs={"texts": ["hi"]},
            request_data=make_request_data(metadata={"alice_wonderfence_app_id": ["not", "a", "string"]}),
            input_type="request",
        )
    assert exc.value.status_code == 500
    assert "alice_wonderfence_app_id" in exc.value.detail["exception"]
    client.evaluate_prompt.assert_not_awaited()


def test_get_config_model(make_guardrail):
    from litellm.types.proxy.guardrails.guardrail_hooks.alice_wonderfence import (
        WonderFenceGuardrailConfigModel,
    )

    guardrail, _ = make_guardrail()
    assert guardrail.get_config_model() is WonderFenceGuardrailConfigModel


def test_build_analysis_context_falls_back_to_slash_split(monkeypatch, make_guardrail):
    """When ``litellm.get_llm_provider`` raises, fall back to ``provider/model`` split."""
    import litellm

    from litellm.proxy.guardrails.guardrail_hooks.alice_wonderfence.processing import (
        build_analysis_context,
    )

    guardrail, _ = make_guardrail()

    def boom(model):
        raise ValueError("unknown provider")

    monkeypatch.setattr(litellm, "get_llm_provider", boom)
    build_analysis_context({"model": "myorg/custom-llm"}, guardrail.platform, guardrail._AnalysisContext)

    AnalysisContext = sys.modules["wonderfence_sdk.models"].AnalysisContext
    kwargs = AnalysisContext.call_args.kwargs
    assert kwargs["provider"] == "myorg"
    assert kwargs["model_name"] == "custom-llm"
