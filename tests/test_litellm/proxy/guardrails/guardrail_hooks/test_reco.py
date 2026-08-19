"""
Tests for the Reco guardrail integration.
"""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from litellm.exceptions import GuardrailRaisedException
from litellm.proxy.guardrails.guardrail_hooks.reco import (
    RecoGuardrail,
    initialize_guardrail,
)
from litellm.types.guardrails import GuardrailEventHooks, LitellmParams
from litellm.types.proxy.guardrails.guardrail_hooks.reco import (
    RecoConfigModel,
    RecoOptionalParams,
)


@pytest.fixture
def reco_guardrail():
    return RecoGuardrail(
        reco_tenant_id="11111111-1111-1111-1111-111111111111",
        api_base="https://edge1.us.reco.ai",
        guardrail_name="my-reco-guardrail",
        event_hook="pre_call",
        default_on=True,
    )


class TestRecoGuardrailConfiguration:
    def test_tenant_id_forwarded_as_header(self, reco_guardrail):
        assert reco_guardrail.headers == {"X-Reco-Tenant-Id": "11111111-1111-1111-1111-111111111111"}

    def test_tenant_header_merged_with_existing_headers(self):
        guardrail = RecoGuardrail(
            reco_tenant_id="11111111-1111-1111-1111-111111111111",
            api_base="https://edge1.us.reco.ai",
            headers={"X-Custom": "value"},
        )
        assert guardrail.headers == {
            "X-Custom": "value",
            "X-Reco-Tenant-Id": "11111111-1111-1111-1111-111111111111",
        }

    def test_unreachable_fallback_and_fail_on_error_are_hardcoded(self, reco_guardrail):
        assert reco_guardrail.unreachable_fallback == "fail_open"
        assert reco_guardrail.fail_on_error is False

    def test_missing_reco_tenant_id_raises(self):
        with pytest.raises(ValueError, match="reco_tenant_id"):
            RecoGuardrail(reco_tenant_id=None, api_base="https://edge1.us.reco.ai")

    def test_missing_api_base_raises(self):
        with pytest.raises(ValueError, match="api_base"):
            RecoGuardrail(reco_tenant_id="11111111-1111-1111-1111-111111111111", api_base=None)

    def test_non_uuid_reco_tenant_id_raises(self):
        with pytest.raises(ValueError, match="UUID"):
            RecoGuardrail(reco_tenant_id="tenant-123", api_base="https://edge1.us.reco.ai")

    def test_optional_params_rejects_non_uuid_tenant_id(self):
        with pytest.raises(ValidationError, match="UUID"):
            RecoOptionalParams(reco_tenant_id="tenant-123", api_base="https://edge1.us.reco.ai")

    def test_get_supported_event_hooks_returns_pre_call_only(self):
        assert RecoGuardrail.get_supported_event_hooks() == [GuardrailEventHooks.pre_call]

    def test_get_config_model_returns_reco_config_model(self):
        assert RecoGuardrail.get_config_model() is RecoConfigModel

    def test_config_model_ui_friendly_name(self):
        assert RecoConfigModel.ui_friendly_name() == "Reco"


class TestRecoGuardrailInitializer:
    def test_initialize_guardrail_wires_optional_params(self):
        litellm_params = LitellmParams(
            guardrail="reco",
            mode="pre_call",
            optional_params={
                "reco_tenant_id": "22222222-2222-2222-2222-222222222222",
                "api_base": "https://edge2.eu.reco.ai",
            },
        )

        guardrail = initialize_guardrail(litellm_params, {"guardrail_name": "prod-reco"})

        assert isinstance(guardrail, RecoGuardrail)
        assert guardrail.guardrail_name == "prod-reco"
        assert guardrail.headers == {"X-Reco-Tenant-Id": "22222222-2222-2222-2222-222222222222"}
        assert guardrail.api_base == "https://edge2.eu.reco.ai/beta/litellm_basic_guardrail_api"


class TestRecoGuardrailBlocking:
    @pytest.mark.asyncio
    async def test_action_blocked_raises_exception_with_configured_guardrail_name(self, reco_guardrail):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "action": "BLOCKED",
            "blocked_reason": "Sensitive data detected",
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(reco_guardrail.async_handler, "post", return_value=mock_response) as mock_post:
            with pytest.raises(GuardrailRaisedException) as exc_info:
                await reco_guardrail.apply_guardrail(
                    inputs={"texts": ["some sensitive prompt"]},
                    request_data={"metadata": {}},
                    input_type="request",
                )

        assert exc_info.value.guardrail_name == "my-reco-guardrail"
        assert str(exc_info.value) == "Sensitive data detected"

        _, call_kwargs = mock_post.call_args
        assert call_kwargs["headers"]["X-Reco-Tenant-Id"] == "11111111-1111-1111-1111-111111111111"

    @pytest.mark.asyncio
    async def test_action_none_allows_content(self, reco_guardrail):
        mock_response = MagicMock()
        mock_response.json.return_value = {"action": "NONE", "texts": ["hello"]}
        mock_response.raise_for_status = MagicMock()

        with patch.object(reco_guardrail.async_handler, "post", return_value=mock_response):
            result = await reco_guardrail.apply_guardrail(
                inputs={"texts": ["hello"]},
                request_data={"metadata": {}},
                input_type="request",
            )

        assert result["texts"] == ["hello"]
