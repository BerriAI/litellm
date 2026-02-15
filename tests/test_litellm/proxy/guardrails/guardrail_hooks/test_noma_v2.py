import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.proxy.guardrails.guardrail_hooks.noma import NomaV2Guardrail
from litellm.proxy.guardrails.guardrail_hooks.noma.noma import NomaBlockedMessage


@pytest.fixture
def noma_v2_guardrail():
    return NomaV2Guardrail(
        api_key="test-api-key",
        api_base="https://api.test.noma.security/",
        application_id="test-app",
        guardrail_name="test-noma-v2-guardrail",
        event_hook="pre_call",
        default_on=True,
    )


class TestNomaV2Configuration:
    @pytest.mark.asyncio
    async def test_provider_specific_params_include_noma_v2_fields(self):
        from litellm.proxy.guardrails.guardrail_endpoints import (
            get_provider_specific_params,
        )

        provider_params = await get_provider_specific_params()
        assert "noma_v2" in provider_params

        noma_v2_params = provider_params["noma_v2"]
        assert noma_v2_params["ui_friendly_name"] == "Noma Security v2"
        assert "api_key" in noma_v2_params
        assert "api_base" in noma_v2_params
        assert "application_id" in noma_v2_params
        assert "client_id" in noma_v2_params
        assert "client_secret" in noma_v2_params
        assert "use_v2" not in noma_v2_params

    def test_init_requires_auth(self):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(
                ValueError,
                match="Noma v2 guardrail requires either api_key or client_id\\+client_secret",
            ):
                NomaV2Guardrail()

    @pytest.mark.asyncio
    async def test_api_key_auth_path(self, noma_v2_guardrail):
        assert (
            await noma_v2_guardrail._get_authorization_header() == "Bearer test-api-key"
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = '{"action":"NONE"}'
        mock_response.json.return_value = {
            "action": "NONE",
        }
        mock_response.raise_for_status = MagicMock()
        mock_post = AsyncMock(return_value=mock_response)

        with patch.object(noma_v2_guardrail.async_handler, "post", mock_post):
            await noma_v2_guardrail._call_noma_scan(
                payload={"inputs": {"texts": []}},
                request_data={"litellm_call_id": "test-call-id"},
                logging_obj=None,
            )

        call_kwargs = mock_post.call_args.kwargs
        assert call_kwargs["headers"]["Authorization"] == "Bearer test-api-key"
        assert call_kwargs["headers"]["X-Noma-Request-ID"] == "test-call-id"

    def test_build_scan_payload_sends_raw_available_data(self, noma_v2_guardrail):
        inputs = {
            "texts": ["hello"],
            "images": ["https://example.com/image.png"],
            "structured_messages": [{"role": "user", "content": "hello"}],
            "tool_calls": [{"id": "tool-1"}],
            "model": "gpt-4o-mini",
        }
        request_data = {
            "messages": [{"role": "user", "content": "hello"}],
            "metadata": {"headers": {"x-noma-application-id": "header-app"}},
            "litellm_metadata": {"user_api_key_alias": "litellm-alias"},
            "litellm_call_id": "call-id-1",
        }
        dynamic_params = {"application_id": "dynamic-app", "future_field": {"a": 1}}

        payload = noma_v2_guardrail._build_scan_payload(
            inputs=inputs,
            request_data=request_data,
            input_type="request",
            logging_obj=None,
            dynamic_params=dynamic_params,
        )

        assert payload["inputs"] == inputs
        assert payload["request_data"] == request_data
        assert payload["input_type"] == "request"
        assert payload["dynamic_params"] == dynamic_params
        assert payload["x-noma-context"]["applicationId"] == "dynamic-app"
        assert "input" not in payload

    def test_build_scan_payload_passes_model_call_details_as_is(
        self, noma_v2_guardrail
    ):
        class _LoggingObj:
            def __init__(self) -> None:
                self.model_call_details = {
                    "model": "gpt-4.1-mini",
                    "messages": [{"role": "user", "content": "hello"}],
                    "stream": False,
                    "call_type": "acompletion",
                    "litellm_call_id": "call-id-123",
                    "function_id": "fn-id-456",
                    "litellm_trace_id": "trace-id-789",
                    "api_key": "included-as-is",
                }

        request_data = {"litellm_logging_obj": "<Logging object>"}
        payload = noma_v2_guardrail._build_scan_payload(
            inputs={"texts": ["hello"]},
            request_data=request_data,
            input_type="request",
            logging_obj=_LoggingObj(),
            dynamic_params={},
        )

        assert payload["request_data"]["litellm_logging_obj"] == {
            "model": "gpt-4.1-mini",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": False,
            "call_type": "acompletion",
            "litellm_call_id": "call-id-123",
            "function_id": "fn-id-456",
            "litellm_trace_id": "trace-id-789",
            "api_key": "included-as-is",
        }
        assert "logging_obj" not in payload
        assert request_data["litellm_logging_obj"] == "<Logging object>"

    @pytest.mark.asyncio
    async def test_oauth_path_and_token_reuse(self):
        guardrail = NomaV2Guardrail(
            api_base="https://api.test.noma.security/",
            client_id="test-client-id",
            client_secret="test-client-secret",
            guardrail_name="test-noma-v2-guardrail",
            event_hook="pre_call",
            default_on=True,
        )

        token_response = MagicMock()
        token_response.status_code = 200
        token_response.json.return_value = {
            "accessToken": "oauth-token",
            "expiresIn": 3600,
        }
        token_response.raise_for_status = MagicMock()
        token_post = AsyncMock(return_value=token_response)

        with patch.object(guardrail.async_handler, "post", token_post):
            auth_1 = await guardrail._get_authorization_header()
            auth_2 = await guardrail._get_authorization_header()

        assert auth_1 == "Bearer oauth-token"
        assert auth_2 == "Bearer oauth-token"
        assert token_post.call_count == 1

    @pytest.mark.asyncio
    async def test_call_noma_scan_sanitizes_response_model_dump_object(
        self, noma_v2_guardrail
    ):
        import json

        class _FakeModelResponse:
            def model_dump(self):
                return {"id": "resp-1", "content": "ok"}

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = '{"action":"NONE"}'
        mock_response.json.return_value = {"action": "NONE"}
        mock_response.raise_for_status = MagicMock()
        mock_post = AsyncMock(return_value=mock_response)

        payload = {
            "inputs": {"texts": ["hello"]},
            "request_data": {"response": _FakeModelResponse()},
            "input_type": "response",
            "dynamic_params": {},
        }

        with patch.object(noma_v2_guardrail.async_handler, "post", mock_post):
            await noma_v2_guardrail._call_noma_scan(
                payload=payload,
                request_data={"litellm_call_id": "test-call-id"},
                logging_obj=None,
            )

        sent_payload = mock_post.call_args.kwargs["json"]
        json.dumps(sent_payload)
        assert sent_payload["request_data"]["response"]["id"] == "resp-1"


class TestNomaV2ActionBehavior:
    @pytest.mark.asyncio
    async def test_native_action_none(self, noma_v2_guardrail):
        inputs = {"texts": ["hello"]}
        with patch.object(
            noma_v2_guardrail,
            "_call_noma_scan",
            AsyncMock(
                return_value={
                    "action": "NONE",
                }
            ),
        ):
            result = await noma_v2_guardrail.apply_guardrail(
                inputs=inputs,
                request_data={"metadata": {}},
                input_type="request",
            )

        assert result == inputs

    @pytest.mark.asyncio
    async def test_native_action_guardrail_intervened_updates_supported_fields(
        self, noma_v2_guardrail
    ):
        inputs = {
            "texts": ["Name: Jane"],
            "images": ["https://old.example/image.png"],
            "tools": [{"type": "function", "function": {"name": "old_tool"}}],
        }
        with patch.object(
            noma_v2_guardrail,
            "_call_noma_scan",
            AsyncMock(
                return_value={
                    "action": "GUARDRAIL_INTERVENED",
                    "texts": ["Name: *******"],
                    "images": ["https://new.example/image.png"],
                    "tools": [{"type": "function", "function": {"name": "new_tool"}}],
                }
            ),
        ):
            result = await noma_v2_guardrail.apply_guardrail(
                inputs=inputs,
                request_data={"metadata": {}},
                input_type="request",
            )

        assert result["texts"] == ["Name: *******"]
        assert result["images"] == ["https://new.example/image.png"]
        assert result["tools"] == [
            {"type": "function", "function": {"name": "new_tool"}}
        ]

    @pytest.mark.asyncio
    async def test_native_action_blocked(self, noma_v2_guardrail):
        inputs = {"texts": ["bad"]}
        with patch.object(
            noma_v2_guardrail,
            "_call_noma_scan",
            AsyncMock(
                return_value={
                    "action": "BLOCKED",
                    "blocked_reason": "blocked by policy",
                }
            ),
        ):
            with pytest.raises(NomaBlockedMessage) as exc_info:
                await noma_v2_guardrail.apply_guardrail(
                    inputs=inputs,
                    request_data={"metadata": {}},
                    input_type="request",
                )
        assert exc_info.value.detail["details"]["blocked_reason"] == "blocked by policy"

    @pytest.mark.asyncio
    async def test_intervened_without_modifications_returns_original_inputs(
        self, noma_v2_guardrail
    ):
        inputs = {"texts": ["Name: Jane"]}
        with patch.object(
            noma_v2_guardrail,
            "_call_noma_scan",
            AsyncMock(
                return_value={
                    "action": "GUARDRAIL_INTERVENED",
                }
            ),
        ):
            result = await noma_v2_guardrail.apply_guardrail(
                inputs=inputs,
                request_data={"metadata": {}},
                input_type="request",
            )
        assert result == inputs

    @pytest.mark.asyncio
    async def test_fail_open_on_technical_scan_failure(self, noma_v2_guardrail):
        inputs = {"texts": ["hello"]}
        with patch.object(
            noma_v2_guardrail,
            "_call_noma_scan",
            AsyncMock(side_effect=Exception("network error")),
        ):
            result = await noma_v2_guardrail.apply_guardrail(
                inputs=inputs,
                request_data={"metadata": {}},
                input_type="request",
            )

        assert result == inputs


class TestNomaV2ApplicationIdResolution:
    def test_application_id_precedence(self, noma_v2_guardrail, monkeypatch):
        monkeypatch.delenv("NOMA_APPLICATION_ID", raising=False)
        request_data = {
            "metadata": {
                "headers": {"x-noma-application-id": "header-app"},
                "user_api_key_alias": "metadata-alias",
            },
            "litellm_metadata": {"user_api_key_alias": "litellm-alias"},
        }

        assert (
            noma_v2_guardrail._build_noma_context(
                request_data=request_data,
                logging_obj=None,
                dynamic_params={"application_id": "dynamic-app"},
            )["applicationId"]
            == "dynamic-app"
        )

        assert (
            noma_v2_guardrail._build_noma_context(
                request_data=request_data,
                logging_obj=None,
                dynamic_params={},
            )["applicationId"]
            == "header-app"
        )

        request_data_no_header = {
            "metadata": {},
            "litellm_metadata": {"user_api_key_alias": "litellm-alias"},
        }
        assert (
            noma_v2_guardrail._build_noma_context(
                request_data=request_data_no_header,
                logging_obj=None,
                dynamic_params={},
            )["applicationId"]
            == "test-app"
        )

        guardrail_no_config = NomaV2Guardrail(
            api_key="test-api-key",
            application_id=None,
            guardrail_name="test-noma-v2-guardrail",
            event_hook="pre_call",
            default_on=True,
        )
        request_data_alias_only = {
            "metadata": {},
            "litellm_metadata": {"user_api_key_alias": "litellm-alias"},
        }
        assert (
            guardrail_no_config._build_noma_context(
                request_data=request_data_alias_only,
                logging_obj=None,
                dynamic_params={},
            )["applicationId"]
            == "litellm-alias"
        )

        assert (
            guardrail_no_config._build_noma_context(
                request_data={"metadata": {}, "litellm_metadata": {}},
                logging_obj=None,
                dynamic_params={},
            )["applicationId"]
            == "litellm"
        )
