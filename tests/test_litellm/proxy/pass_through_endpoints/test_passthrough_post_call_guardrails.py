"""
Tests for post-call guardrail invocation on pass-through endpoints.

Verifies that apply_guardrail(input_type="response") is called for
non-streaming pass-through responses. Addresses issue #20270.
"""

import json
import sys
from contextlib import ExitStack
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    ModifyResponseException,
)

_PT_MOD = "litellm.proxy.pass_through_endpoints.pass_through_endpoints"
_COLLECT = "litellm.proxy.pass_through_endpoints.passthrough_guardrails.PassthroughGuardrailHandler.collect_guardrails"

_GEMINI_RESPONSE = {
    "candidates": [
        {
            "content": {
                "role": "model",
                "parts": [{"text": "Hello"}],
            }
        }
    ]
}


def _make_user_api_key_dict(**overrides):
    d = MagicMock()
    d.api_key = "sk-test"
    d.user_id = "user-1"
    d.team_id = "team-1"
    d.org_id = None
    d.request_route = "/vertex_ai/v1/projects/p/locations/l/publishers/google/models/gemini:generateContent"
    for k, v in overrides.items():
        setattr(d, k, v)
    return d


def _make_httpx_response(body: dict, status_code: int = 200) -> httpx.Response:
    content = json.dumps(body).encode("utf-8")
    return httpx.Response(
        status_code=status_code,
        headers={"content-type": "application/json"},
        content=content,
        request=httpx.Request("POST", "https://example.com/v1/generateContent"),
    )


def _make_mock_request():
    mock_request = MagicMock()
    mock_request.method = "POST"
    mock_request.query_params = {}
    mock_request.headers = MagicMock()
    mock_request.headers.copy.return_value = {}
    return mock_request


def _ensure_proxy_server_mock():
    """Insert a mock proxy_server module if the real one can't import."""
    key = "litellm.proxy.proxy_server"
    if key not in sys.modules:
        mock_mod = MagicMock()
        mock_mod.proxy_logging_obj = MagicMock()
        sys.modules[key] = mock_mod
    import litellm.proxy

    if not hasattr(litellm.proxy, "proxy_server"):
        litellm.proxy.proxy_server = sys.modules[key]


_ensure_proxy_server_mock()

from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
    pass_through_request,
)


def _common_patches(mock_proxy_logging, mock_response):
    """Return a combined context manager for the patches shared by all tests."""
    mock_async_client = AsyncMock()
    mock_async_client_obj = MagicMock()
    mock_async_client_obj.client = mock_async_client

    mock_pt_logging = MagicMock()
    mock_pt_logging.pass_through_async_success_handler = AsyncMock()

    patches = [
        patch(
            f"{_PT_MOD}.HttpPassThroughEndpointHelpers.non_streaming_http_request_handler",
            new_callable=AsyncMock,
            return_value=mock_response,
        ),
        patch(f"{_PT_MOD}._is_streaming_response", return_value=False),
        patch("litellm.proxy.proxy_server.proxy_logging_obj", mock_proxy_logging),
        patch(f"{_PT_MOD}.pass_through_endpoint_logging", mock_pt_logging),
        patch(f"{_PT_MOD}.get_async_httpx_client", return_value=mock_async_client_obj),
        patch(f"{_PT_MOD}._read_request_body", new_callable=AsyncMock, return_value={}),
        patch(f"{_PT_MOD}._safe_get_request_headers", return_value={}),
    ]

    stack = ExitStack()
    for p in patches:
        stack.enter_context(p)
    return stack


@pytest.mark.asyncio
class TestPassthroughPostCallGuardrails:

    @patch(_COLLECT, return_value=["rubrik"])
    async def test_post_call_success_hook_called_when_guardrails_configured(
        self,
        mock_collect,
    ):
        """post_call_success_hook should fire when guardrails are configured."""
        mock_response = _make_httpx_response(_GEMINI_RESPONSE)

        mock_proxy_logging = MagicMock()
        mock_proxy_logging.pre_call_hook = AsyncMock(return_value={})
        mock_proxy_logging.post_call_success_hook = AsyncMock(
            return_value=_GEMINI_RESPONSE
        )

        with _common_patches(mock_proxy_logging, mock_response):
            await pass_through_request(
                request=_make_mock_request(),
                target="https://example.com/v1/generateContent",
                custom_headers={"Content-Type": "application/json"},
                user_api_key_dict=_make_user_api_key_dict(),
                stream=False,
            )

        mock_proxy_logging.post_call_success_hook.assert_awaited_once()
        call_kwargs = mock_proxy_logging.post_call_success_hook.call_args
        assert call_kwargs.kwargs["response"] == _GEMINI_RESPONSE

    @patch(_COLLECT, return_value=[])
    async def test_post_call_success_hook_skipped_when_no_guardrails(
        self,
        mock_collect,
    ):
        """post_call_success_hook should NOT fire when no guardrails are configured."""
        mock_response = _make_httpx_response(_GEMINI_RESPONSE)

        mock_proxy_logging = MagicMock()
        mock_proxy_logging.pre_call_hook = AsyncMock(return_value={})
        mock_proxy_logging.post_call_success_hook = AsyncMock()

        with _common_patches(mock_proxy_logging, mock_response):
            result = await pass_through_request(
                request=_make_mock_request(),
                target="https://example.com/v1/generateContent",
                custom_headers={"Content-Type": "application/json"},
                user_api_key_dict=_make_user_api_key_dict(),
                stream=False,
            )

        mock_proxy_logging.post_call_success_hook.assert_not_awaited()
        assert result.status_code == 200

    @patch(_COLLECT, return_value=["rubrik"])
    async def test_modify_response_exception_returns_error(
        self,
        mock_collect,
    ):
        """ModifyResponseException from guardrail should return 200 with provider-agnostic error."""
        response_body = {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [
                            {"functionCall": {"name": "dangerous_tool", "args": {}}}
                        ],
                    }
                }
            ]
        }
        mock_response = _make_httpx_response(response_body)

        mock_proxy_logging = MagicMock()
        mock_proxy_logging.pre_call_hook = AsyncMock(return_value={})
        mock_proxy_logging.post_call_success_hook = AsyncMock(
            side_effect=ModifyResponseException(
                message="Tool dangerous_tool blocked by policy",
                model="gemini-2.0-flash",
                request_data={},
                guardrail_name="rubrik",
            )
        )
        mock_proxy_logging.post_call_failure_hook = AsyncMock()

        with _common_patches(mock_proxy_logging, mock_response):
            result = await pass_through_request(
                request=_make_mock_request(),
                target="https://example.com/v1/generateContent",
                custom_headers={"Content-Type": "application/json"},
                user_api_key_dict=_make_user_api_key_dict(),
                stream=False,
            )

        mock_proxy_logging.post_call_failure_hook.assert_awaited_once()
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["error"]["type"] == "content_filter"
        assert body["error"]["message"] == "Tool dangerous_tool blocked by policy"
        assert body["error"]["guardrail_name"] == "rubrik"
        assert body["error"]["model"] == "gemini-2.0-flash"


@pytest.mark.asyncio
class TestUnifiedGuardrailCallTypeResolution:

    async def test_pass_through_call_type_resolved_from_logging_obj(self):
        """Unified guardrail should resolve call_type from logging_obj for pass-through."""
        from litellm.proxy.guardrails.guardrail_hooks.unified_guardrail.unified_guardrail import (
            UnifiedLLMGuardrails,
        )

        unified = UnifiedLLMGuardrails()

        mock_guardrail = MagicMock(spec=CustomGuardrail)
        mock_guardrail.guardrail_name = "test-guardrail"
        mock_guardrail.should_run_guardrail.return_value = True

        mock_logging_obj = MagicMock()
        mock_logging_obj.call_type = "pass_through_endpoint"

        user_api_key_dict = _make_user_api_key_dict()

        data = {
            "guardrail_to_apply": mock_guardrail,
            "litellm_logging_obj": mock_logging_obj,
        }

        response_body = {"candidates": [{"content": {"parts": [{"text": "hello"}]}}]}

        with patch(
            "litellm.proxy.guardrails.guardrail_hooks.unified_guardrail.unified_guardrail.load_guardrail_translation_mappings"
        ) as mock_load:
            mock_handler_instance = AsyncMock()
            mock_handler_instance.process_output_response = AsyncMock(
                return_value=response_body
            )
            mock_handler_class = MagicMock(return_value=mock_handler_instance)

            from litellm.types.utils import CallTypes

            mock_load.return_value = {CallTypes.pass_through: mock_handler_class}

            result = await unified.async_post_call_success_hook(
                data=data,
                user_api_key_dict=user_api_key_dict,
                response=response_body,
            )

        mock_handler_instance.process_output_response.assert_awaited_once()


def test_modify_response_exception_importable_from_both_paths():
    """ModifyResponseException re-export from custom_guardrail must stay in sync."""
    from litellm.exceptions import ModifyResponseException as FromExceptions
    from litellm.integrations.custom_guardrail import (
        ModifyResponseException as FromGuardrail,
    )

    assert FromExceptions is FromGuardrail
