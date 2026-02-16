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
        monitor_mode=False,
        block_failures=False,
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
        assert "monitor_mode" in noma_v2_params
        assert "block_failures" in noma_v2_params
        assert "client_id" not in noma_v2_params
        assert "client_secret" not in noma_v2_params
        assert "use_v2" not in noma_v2_params

    def test_init_requires_auth(self):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(
                ValueError,
                match="Noma v2 guardrail requires api_key",
            ):
                NomaV2Guardrail()

    def test_init_defaults_monitor_and_block_failures(self):
        with patch.dict(os.environ, {"NOMA_API_KEY": "test-api-key"}, clear=True):
            guardrail = NomaV2Guardrail()

        assert guardrail.monitor_mode is False
        assert guardrail.block_failures is False

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
            )

        call_kwargs = mock_post.call_args.kwargs
        assert call_kwargs["headers"]["Authorization"] == "Bearer test-api-key"

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
        payload = noma_v2_guardrail._build_scan_payload(
            inputs=inputs,
            request_data=request_data,
            input_type="request",
            logging_obj=None,
            application_id="dynamic-app",
        )

        assert payload["inputs"] == inputs
        assert payload["request_data"] == request_data
        assert payload["input_type"] == "request"
        assert payload["monitor_mode"] is False
        assert payload["application_id"] == "dynamic-app"
        assert "dynamic_params" not in payload
        assert "x-noma-context" not in payload
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
            application_id="test-app",
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
            "application_id": "test-app",
        }

        with patch.object(noma_v2_guardrail.async_handler, "post", mock_post):
            await noma_v2_guardrail._call_noma_scan(payload=payload)

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

    @pytest.mark.asyncio
    async def test_fail_closed_on_technical_scan_failure_when_block_failures_true(self):
        guardrail = NomaV2Guardrail(
            api_key="test-api-key",
            block_failures=True,
            guardrail_name="test-noma-v2-guardrail",
            event_hook="pre_call",
            default_on=True,
        )
        with patch.object(
            guardrail,
            "_call_noma_scan",
            AsyncMock(side_effect=Exception("network error")),
        ):
            with pytest.raises(Exception, match="network error"):
                await guardrail.apply_guardrail(
                    inputs={"texts": ["hello"]},
                    request_data={"metadata": {}},
                    input_type="request",
                )

    @pytest.mark.asyncio
    async def test_monitor_mode_ignores_block_action(self):
        guardrail = NomaV2Guardrail(
            api_key="test-api-key",
            monitor_mode=True,
            guardrail_name="test-noma-v2-guardrail",
            event_hook="pre_call",
            default_on=True,
        )
        call_mock = AsyncMock(return_value={"action": "BLOCKED"})
        with patch.object(guardrail, "_call_noma_scan", call_mock):
            result = await guardrail.apply_guardrail(
                inputs={"texts": ["hello"]},
                request_data={"metadata": {}},
                input_type="request",
            )

        payload = call_mock.call_args.kwargs["payload"]
        assert payload["monitor_mode"] is True
        assert result == {"texts": ["hello"]}


class TestNomaV2ApplicationIdResolution:
    @pytest.mark.asyncio
    async def test_apply_guardrail_uses_dynamic_application_id(self, noma_v2_guardrail):
        call_mock = AsyncMock(return_value={"action": "NONE"})
        with patch.object(
            noma_v2_guardrail,
            "get_guardrail_dynamic_request_body_params",
            return_value={"application_id": "dynamic-app"},
        ):
            with patch.object(noma_v2_guardrail, "_call_noma_scan", call_mock):
                await noma_v2_guardrail.apply_guardrail(
                    inputs={"texts": ["hello"]},
                    request_data={"metadata": {}},
                    input_type="request",
                )

        payload = call_mock.call_args.kwargs["payload"]
        assert payload["application_id"] == "dynamic-app"

    @pytest.mark.asyncio
    async def test_apply_guardrail_uses_configured_application_id(
        self, noma_v2_guardrail
    ):
        call_mock = AsyncMock(return_value={"action": "NONE"})
        with patch.object(
            noma_v2_guardrail,
            "get_guardrail_dynamic_request_body_params",
            return_value={},
        ):
            with patch.object(noma_v2_guardrail, "_call_noma_scan", call_mock):
                await noma_v2_guardrail.apply_guardrail(
                    inputs={"texts": ["hello"]},
                    request_data={"metadata": {}},
                    input_type="request",
                )

        payload = call_mock.call_args.kwargs["payload"]
        assert payload["application_id"] == "test-app"

    @pytest.mark.asyncio
    async def test_apply_guardrail_omits_application_id_when_not_explicit(self):
        guardrail_no_config = NomaV2Guardrail(
            api_key="test-api-key",
            application_id=None,
            guardrail_name="test-noma-v2-guardrail",
            event_hook="pre_call",
            default_on=True,
        )

        call_mock = AsyncMock(return_value={"action": "NONE"})
        with patch.object(
            guardrail_no_config,
            "get_guardrail_dynamic_request_body_params",
            return_value={},
        ):
            with patch.object(guardrail_no_config, "_call_noma_scan", call_mock):
                await guardrail_no_config.apply_guardrail(
                    inputs={"texts": ["hello"]},
                    request_data={"metadata": {}},
                    input_type="request",
                )

        payload = call_mock.call_args.kwargs["payload"]
        assert "application_id" not in payload
