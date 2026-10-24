"""
Tests for the Reco guardrail integration.
"""

from collections.abc import Mapping
from typing import Final
from unittest.mock import MagicMock, patch

import httpx
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
from litellm.types.utils import GenericGuardrailAPIInputs


@pytest.fixture
def reco_guardrail() -> RecoGuardrail:
    return RecoGuardrail(
        reco_tenant_id="11111111-1111-1111-1111-111111111111",
        api_base="https://edge1.us.reco.ai",
        guardrail_name="my-reco-guardrail",
        event_hook="pre_call",
        default_on=True,
    )


class TestRecoGuardrailConfiguration:
    def test_tenant_id_forwarded_as_header(self, reco_guardrail: RecoGuardrail) -> None:
        assert reco_guardrail.headers == {"X-Reco-Tenant-Id": "11111111-1111-1111-1111-111111111111"}

    def test_tenant_header_merged_with_existing_headers(self) -> None:
        guardrail: Final = RecoGuardrail(
            reco_tenant_id="11111111-1111-1111-1111-111111111111",
            api_base="https://edge1.us.reco.ai",
            headers={"X-Custom": "value"},
        )
        assert guardrail.headers == {
            "X-Custom": "value",
            "X-Reco-Tenant-Id": "11111111-1111-1111-1111-111111111111",
        }

    def test_missing_reco_tenant_id_raises(self) -> None:
        with pytest.raises(ValueError, match="reco_tenant_id"):
            RecoGuardrail(reco_tenant_id=None, api_base="https://edge1.us.reco.ai")

    def test_missing_api_base_raises(self) -> None:
        with pytest.raises(ValueError, match="api_base"):
            RecoGuardrail(reco_tenant_id="11111111-1111-1111-1111-111111111111", api_base=None)

    def test_non_uuid_reco_tenant_id_raises(self) -> None:
        with pytest.raises(ValueError, match="UUID"):
            RecoGuardrail(reco_tenant_id="tenant-123", api_base="https://edge1.us.reco.ai")

    def test_optional_params_rejects_non_uuid_tenant_id(self) -> None:
        with pytest.raises(ValidationError, match="UUID"):
            RecoOptionalParams(reco_tenant_id="tenant-123", api_base="https://edge1.us.reco.ai")

    def test_get_supported_event_hooks_returns_pre_call_only(self) -> None:
        assert RecoGuardrail.get_supported_event_hooks() == [GuardrailEventHooks.pre_call]

    def test_get_config_model_returns_reco_config_model(self) -> None:
        assert RecoGuardrail.get_config_model() is RecoConfigModel

    def test_config_model_ui_friendly_name(self) -> None:
        assert RecoConfigModel.ui_friendly_name() == "Reco"

    def test_litellm_params_stay_assignable_for_registry_secret_resolution(self) -> None:
        litellm_params: Final = LitellmParams(guardrail="reco", mode="pre_call")

        litellm_params.api_base = "https://edge1.us.reco.ai"

        assert litellm_params.api_base == "https://edge1.us.reco.ai"


class TestRecoGuardrailInitializer:
    def test_initialize_guardrail_wires_optional_params(self) -> None:
        litellm_params: Final = LitellmParams(
            guardrail="reco",
            mode="pre_call",
            optional_params={
                "reco_tenant_id": "22222222-2222-2222-2222-222222222222",
                "api_base": "https://edge2.eu.reco.ai",
            },
        )

        guardrail: Final = initialize_guardrail(litellm_params, {"guardrail_name": "prod-reco"})

        assert isinstance(guardrail, RecoGuardrail)
        assert guardrail.guardrail_name == "prod-reco"
        assert guardrail.headers == {"X-Reco-Tenant-Id": "22222222-2222-2222-2222-222222222222"}
        assert guardrail.api_base == "https://edge2.eu.reco.ai/beta/litellm_basic_guardrail_api"

    def test_initialize_guardrail_reads_plain_dict_optional_params(self) -> None:
        """Guardrails reconciled from the DB can carry `optional_params` as a
        plain dict rather than a pydantic object; attribute-only access would
        silently read None for every field."""
        litellm_params: Final = LitellmParams.model_construct(
            guardrail="reco",
            optional_params={
                "reco_tenant_id": "33333333-3333-3333-3333-333333333333",
                "api_base": "https://edge3.eu.reco.ai",
            },
            mode="pre_call",
            default_on=True,
        )

        guardrail: Final = initialize_guardrail(litellm_params, {"guardrail_name": "prod-reco"})

        assert guardrail.headers == {"X-Reco-Tenant-Id": "33333333-3333-3333-3333-333333333333"}
        assert guardrail.api_base == "https://edge3.eu.reco.ai/beta/litellm_basic_guardrail_api"

    def test_initialize_guardrail_resolves_env_var_reference(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RECO_TEST_TENANT_ID", "44444444-4444-4444-4444-444444444444")
        litellm_params: Final = LitellmParams(
            guardrail="reco",
            optional_params={
                "reco_tenant_id": "os.environ/RECO_TEST_TENANT_ID",
                "api_base": "https://edge4.eu.reco.ai",
            },
            mode="pre_call",
            default_on=True,
        )

        guardrail: Final = initialize_guardrail(litellm_params, {"guardrail_name": "prod-reco"})

        assert guardrail.headers == {"X-Reco-Tenant-Id": "44444444-4444-4444-4444-444444444444"}


def _litellm_params(top_level: Mapping[str, object], nested: Mapping[str, object]) -> LitellmParams:
    return LitellmParams(
        guardrail="reco",
        mode="pre_call",
        optional_params={
            "reco_tenant_id": "22222222-2222-2222-2222-222222222222",
            "api_base": "https://edge2.eu.reco.ai",
            **nested,
        },
        **top_level,
    )


async def _apply_with_reco_unreachable(guardrail: RecoGuardrail) -> GenericGuardrailAPIInputs:
    with patch.object(guardrail.async_handler, "post", side_effect=httpx.ConnectError("connection refused")):
        return await guardrail.apply_guardrail(
            inputs={"texts": ["hello"]},
            request_data={"metadata": {}},
            input_type="request",
        )


class TestRecoGuardrailFailureModes:
    @pytest.mark.asyncio
    async def test_unreachable_reco_blocks_request_by_default(self, reco_guardrail: RecoGuardrail) -> None:
        with pytest.raises(Exception, match="connection refused"):
            await _apply_with_reco_unreachable(reco_guardrail)

    @pytest.mark.asyncio
    async def test_fail_open_lets_request_through_when_reco_unreachable(self) -> None:
        guardrail: Final = RecoGuardrail(
            reco_tenant_id="11111111-1111-1111-1111-111111111111",
            api_base="https://edge1.us.reco.ai",
            fail_on_error=False,
        )

        result: Final = await _apply_with_reco_unreachable(guardrail)

        assert result["texts"] == ["hello"]

    @pytest.mark.asyncio
    async def test_fail_closed_blocks_request_when_reco_unreachable(self) -> None:
        guardrail: Final = RecoGuardrail(
            reco_tenant_id="11111111-1111-1111-1111-111111111111",
            api_base="https://edge1.us.reco.ai",
            unreachable_fallback="fail_closed",
            fail_on_error=True,
        )

        with pytest.raises(Exception, match="connection refused"):
            await _apply_with_reco_unreachable(guardrail)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("top_level", "nested"),
        [
            pytest.param({}, {}, id="defaults"),
            pytest.param({}, {"unreachable_fallback": "fail_closed", "fail_on_error": True}, id="optional-params"),
            pytest.param({"unreachable_fallback": "fail_closed", "fail_on_error": True}, {}, id="top-level"),
        ],
    )
    async def test_initialize_guardrail_fails_closed_when_configured(
        self, top_level: Mapping[str, object], nested: Mapping[str, object]
    ) -> None:
        guardrail: Final = initialize_guardrail(_litellm_params(top_level, nested), {"guardrail_name": "prod-reco"})

        with pytest.raises(Exception, match="connection refused"):
            await _apply_with_reco_unreachable(guardrail)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("top_level", "nested"),
        [
            pytest.param({}, {"fail_on_error": False}, id="optional-params"),
            pytest.param({"fail_on_error": False}, {}, id="top-level"),
            pytest.param(
                {"unreachable_fallback": "fail_closed", "fail_on_error": True},
                {"fail_on_error": False},
                id="optional-params-wins-over-top-level",
            ),
        ],
    )
    async def test_initialize_guardrail_fails_open_when_configured(
        self, top_level: Mapping[str, object], nested: Mapping[str, object]
    ) -> None:
        guardrail: Final = initialize_guardrail(_litellm_params(top_level, nested), {"guardrail_name": "prod-reco"})

        result: Final = await _apply_with_reco_unreachable(guardrail)

        assert result["texts"] == ["hello"]

    @pytest.mark.asyncio
    async def test_initialize_guardrail_keeps_top_level_fail_open_when_nested_params_are_a_model(self) -> None:
        nested: Final = RecoOptionalParams(
            reco_tenant_id="22222222-2222-2222-2222-222222222222", api_base="https://edge2.eu.reco.ai"
        )
        litellm_params: Final = LitellmParams.model_construct(
            guardrail="reco", mode="pre_call", optional_params=nested, fail_on_error=False
        )

        guardrail: Final = initialize_guardrail(litellm_params, {"guardrail_name": "prod-reco"})
        result: Final = await _apply_with_reco_unreachable(guardrail)

        assert result["texts"] == ["hello"]


class TestRecoGuardrailBlocking:
    @pytest.mark.asyncio
    async def test_action_blocked_raises_exception_with_configured_guardrail_name(
        self, reco_guardrail: RecoGuardrail
    ) -> None:
        mock_response: Final = MagicMock()
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
        assert exc_info.value.blocked_content is True

        _, call_kwargs = mock_post.call_args
        assert call_kwargs["headers"]["X-Reco-Tenant-Id"] == "11111111-1111-1111-1111-111111111111"

    @pytest.mark.asyncio
    async def test_action_none_allows_content(self, reco_guardrail: RecoGuardrail) -> None:
        mock_response: Final = MagicMock()
        mock_response.json.return_value = {"action": "NONE", "texts": ["hello"]}
        mock_response.raise_for_status = MagicMock()

        with patch.object(reco_guardrail.async_handler, "post", return_value=mock_response):
            result: Final = await reco_guardrail.apply_guardrail(
                inputs={"texts": ["hello"]},
                request_data={"metadata": {}},
                input_type="request",
            )

        assert result["texts"] == ["hello"]
