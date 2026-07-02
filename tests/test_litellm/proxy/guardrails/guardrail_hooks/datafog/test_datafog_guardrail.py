"""
Tests for the DataFog PII Guardrail.

All PII values below are synthetic test fixtures (documentation-reserved
domains, test card numbers, invalid SSN ranges).
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../../"))  # Adds the parent directory to the system path

pytest.importorskip("datafog")

from fastapi import HTTPException

from litellm.proxy.guardrails.guardrail_hooks.datafog.datafog import DataFogGuardrail

EMAIL = "jane.doe@example.com"
CARD = "4242 4242 4242 4242"
SSN = "856-45-6789"
DE_TAX_ID = "12345678901"


def _chat_data(content) -> dict:
    return {"messages": [{"role": "user", "content": content}]}


def _model_response(text: str):
    import litellm

    resp = litellm.ModelResponse()
    resp.choices[0].message.content = text
    return resp


@pytest.mark.asyncio
async def test_pre_call_redacts_email():
    guardrail = DataFogGuardrail(guardrail_name="datafog-pii", default_on=True)
    data = await guardrail.async_pre_call_hook(
        user_api_key_dict=None,
        cache=None,
        data=_chat_data(f"email the report to {EMAIL} please"),
        call_type="completion",
    )
    content = data["messages"][0]["content"]
    assert EMAIL not in content
    assert "[EMAIL_1]" in content


@pytest.mark.asyncio
async def test_pre_call_clean_message_unchanged():
    guardrail = DataFogGuardrail(guardrail_name="datafog-pii", default_on=True)
    data = await guardrail.async_pre_call_hook(
        user_api_key_dict=None,
        cache=None,
        data=_chat_data("summarize this design doc"),
        call_type="completion",
    )
    assert data["messages"][0]["content"] == "summarize this design doc"


@pytest.mark.asyncio
async def test_pre_call_redacts_content_parts_and_skips_images():
    guardrail = DataFogGuardrail(guardrail_name="datafog-pii", default_on=True)
    data = await guardrail.async_pre_call_hook(
        user_api_key_dict=None,
        cache=None,
        data=_chat_data(
            [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,x"}},
                {"type": "text", "text": f"ssn is {SSN}"},
            ]
        ),
        call_type="completion",
    )
    parts = data["messages"][0]["content"]
    assert parts[0]["type"] == "image_url"
    assert SSN not in parts[1]["text"]


@pytest.mark.asyncio
async def test_block_raises_http_400_without_echoing_pii():
    guardrail = DataFogGuardrail(guardrail_name="datafog-pii", default_on=True, datafog_action="block")
    with pytest.raises(HTTPException) as exc:
        await guardrail.async_pre_call_hook(
            user_api_key_dict=None,
            cache=None,
            data=_chat_data(f"send {CARD} to billing"),
            call_type="completion",
        )
    assert exc.value.status_code == 400
    detail = str(exc.value.detail)
    assert "CREDIT_CARD" in detail
    assert CARD not in detail


@pytest.mark.asyncio
async def test_during_call_blocks_when_action_is_block():
    guardrail = DataFogGuardrail(guardrail_name="datafog-pii", default_on=True, datafog_action="block")
    with pytest.raises(HTTPException):
        await guardrail.async_moderation_hook(
            data=_chat_data(f"reach me at {EMAIL}"),
            user_api_key_dict=None,
            call_type="completion",
        )


@pytest.mark.asyncio
async def test_during_call_noop_when_action_is_redact():
    # during_call cannot modify content mid-flight; redact mode is a no-op.
    guardrail = DataFogGuardrail(guardrail_name="datafog-pii", default_on=True, datafog_action="redact")
    data = _chat_data(f"reach me at {EMAIL}")
    result = await guardrail.async_moderation_hook(data=data, user_api_key_dict=None, call_type="completion")
    assert result == data


@pytest.mark.asyncio
async def test_post_call_redacts_model_response():
    guardrail = DataFogGuardrail(guardrail_name="datafog-pii", default_on=True)
    response = _model_response(f"the customer is reachable at {EMAIL}")
    await guardrail.async_post_call_success_hook(data={}, user_api_key_dict=None, response=response)
    assert EMAIL not in response.choices[0].message.content


@pytest.mark.asyncio
async def test_noisy_entities_off_by_default():
    guardrail = DataFogGuardrail(guardrail_name="datafog-pii", default_on=True)
    data = await guardrail.async_pre_call_hook(
        user_api_key_dict=None,
        cache=None,
        data=_chat_data("ping 192.168.1.1 about build 2020-01-02"),
        call_type="completion",
    )
    assert data["messages"][0]["content"] == "ping 192.168.1.1 about build 2020-01-02"


@pytest.mark.asyncio
async def test_entity_types_override():
    guardrail = DataFogGuardrail(guardrail_name="datafog-pii", default_on=True, datafog_entity_types=["IP_ADDRESS"])
    data = await guardrail.async_pre_call_hook(
        user_api_key_dict=None,
        cache=None,
        data=_chat_data("ping 192.168.1.1"),
        call_type="completion",
    )
    assert "192.168.1.1" not in data["messages"][0]["content"]


@pytest.mark.asyncio
async def test_german_locale_entities():
    guardrail = DataFogGuardrail(
        guardrail_name="datafog-pii",
        default_on=True,
        datafog_entity_types=["DE_TAX_ID"],
        datafog_locales=["de"],
    )
    data = await guardrail.async_pre_call_hook(
        user_api_key_dict=None,
        cache=None,
        data=_chat_data(f"Steuer-ID {DE_TAX_ID} liegt vor."),
        call_type="completion",
    )
    assert DE_TAX_ID not in data["messages"][0]["content"]


@pytest.mark.asyncio
async def test_fail_open_passes_data_through_on_engine_error(monkeypatch):
    guardrail = DataFogGuardrail(guardrail_name="datafog-pii", default_on=True)
    monkeypatch.setattr(
        "litellm.proxy.guardrails.guardrail_hooks.datafog.datafog._redact_text",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    original = _chat_data(f"reach me at {EMAIL}")
    data = await guardrail.async_pre_call_hook(
        user_api_key_dict=None, cache=None, data=original, call_type="completion"
    )
    assert data["messages"][0]["content"] == f"reach me at {EMAIL}"


@pytest.mark.asyncio
async def test_fail_closed_raises_without_pii_or_cause_chain(monkeypatch):
    guardrail = DataFogGuardrail(guardrail_name="datafog-pii", default_on=True, datafog_fail_policy="closed")
    monkeypatch.setattr(
        "litellm.proxy.guardrails.guardrail_hooks.datafog.datafog._redact_text",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError(f"parser choked on: reach me at {EMAIL}")),
    )
    with pytest.raises(RuntimeError, match="datafog_fail_policy") as exc:
        await guardrail.async_pre_call_hook(
            user_api_key_dict=None,
            cache=None,
            data=_chat_data(f"reach me at {EMAIL}"),
            call_type="completion",
        )
    assert exc.value.__cause__ is None
    assert EMAIL not in str(exc.value)


def test_invalid_config_rejected():
    with pytest.raises(ValueError):
        DataFogGuardrail(guardrail_name="datafog-pii", default_on=True, datafog_action="explode")
    with pytest.raises(ValueError):
        DataFogGuardrail(guardrail_name="datafog-pii", default_on=True, datafog_fail_policy="maybe")


def test_registry_registration():
    from litellm.proxy.guardrails.guardrail_hooks.datafog import (
        guardrail_class_registry,
        guardrail_initializer_registry,
    )
    from litellm.types.guardrails import SupportedGuardrailIntegrations

    assert SupportedGuardrailIntegrations.DATAFOG.value in guardrail_initializer_registry
    assert guardrail_class_registry[SupportedGuardrailIntegrations.DATAFOG.value] is (DataFogGuardrail)
