"""
Tests for post-call guardrail invocation on pass-through endpoints.

Verifies that apply_guardrail(input_type="response") is called for
non-streaming pass-through responses. Addresses issue #20270.

Also verifies that post-call guardrails enforce (block / rewrite) no matter
how they are attached — endpoint-level config, ``default_on: true``, or a
per-request ``guardrails`` body param. Addresses issue #32201.
"""

import json
from contextlib import ExitStack
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import HTTPException

import litellm
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    ModifyResponseException,
)
from litellm.proxy._types import ProxyException

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
        mock_proxy_logging.post_call_response_headers_hook = AsyncMock(return_value={})

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
        mock_proxy_logging.post_call_response_headers_hook = AsyncMock(return_value={})

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

    @patch(_COLLECT, return_value=["rubrik"])
    async def test_deny_forwards_guardrail_logging_info_to_failure_hook(
        self,
        mock_collect,
    ):
        """A post-call guardrail deny (non-ModifyResponseException) records its
        standard_logging_guardrail_information on the hook_data dict; the failure
        handler must forward that info to post_call_failure_hook so downstream
        loggers (e.g. the otel guardrail span) still see it. Regression for the
        block path dropping it."""
        mock_response = _make_httpx_response(_GEMINI_RESPONSE)

        def _block(*, data, user_api_key_dict, response):
            metadata = data.setdefault("metadata", {})
            metadata.setdefault("standard_logging_guardrail_information", []).append(
                {"guardrail_name": "rubrik", "guardrail_status": "guardrail_intervened"}
            )
            raise HTTPException(status_code=400, detail={"error": "blocked"})

        captured = {}

        async def _capture_failure(**kwargs):
            captured.update(kwargs)

        mock_proxy_logging = MagicMock()
        mock_proxy_logging.pre_call_hook = AsyncMock(return_value={})
        mock_proxy_logging.post_call_success_hook = AsyncMock(side_effect=_block)
        mock_proxy_logging.post_call_failure_hook = AsyncMock(
            side_effect=_capture_failure
        )

        with _common_patches(mock_proxy_logging, mock_response):
            with pytest.raises(ProxyException):
                await pass_through_request(
                    request=_make_mock_request(),
                    target="https://example.com/v1/generateContent",
                    custom_headers={"Content-Type": "application/json"},
                    user_api_key_dict=_make_user_api_key_dict(),
                    stream=False,
                )

        mock_proxy_logging.post_call_failure_hook.assert_awaited_once()
        entries = captured["request_data"]["metadata"][
            "standard_logging_guardrail_information"
        ]
        assert any(e.get("guardrail_name") == "rubrik" for e in entries)


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

        mock_handler_instance = AsyncMock()
        mock_handler_instance.process_output_response = AsyncMock(
            return_value=response_body
        )
        mock_handler_class = MagicMock(return_value=mock_handler_instance)

        from litellm.proxy.guardrails.guardrail_hooks.unified_guardrail import (
            unified_guardrail as unified_guardrail_module,
        )
        from litellm.types.utils import CallTypes

        with patch.object(
            unified_guardrail_module,
            "endpoint_guardrail_translation_mappings",
            {CallTypes.pass_through: mock_handler_class},
        ):
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


# ---------------------------------------------------------------------------
# Issue #32201: post_call guardrails attached to a pass-through endpoint are
# consulted but never enforce.
#
# These tests run the REAL dispatch (real ProxyLogging + real
# ToolPermissionGuardrail registered in litellm.callbacks) so a regression in
# any layer — the gate in pass_through_request, should_run_guardrail, or the
# guardrail's dict-response handling — fails them.
# ---------------------------------------------------------------------------

_ANTHROPIC_DENIED_TOOL_USE_RESPONSE = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "model": "claude-x",
    "stop_reason": "tool_use",
    "content": [
        {
            "type": "tool_use",
            "id": "t1",
            "name": "Bash",
            "input": {"command": "rm -rf /"},
        }
    ],
}

_ANTHROPIC_SAFE_RESPONSE = {
    "id": "msg_2",
    "type": "message",
    "role": "assistant",
    "model": "claude-x",
    "stop_reason": "tool_use",
    "content": [
        {
            "type": "tool_use",
            "id": "t2",
            "name": "Bash",
            "input": {"command": "ls -la"},
        }
    ],
}

_TOOL_FIREWALL_RULES = [
    {
        "id": "allow_safe_bash",
        "tool_name": "Bash",
        "decision": "allow",
        "allowed_param_patterns": {
            "command": r"^(?!.*(rm\s+-rf|terraform\s+destroy|kubectl\s+delete)).*$"
        },
    }
]


def _make_tool_firewall_guardrail(default_on: bool):
    from litellm.proxy.guardrails.guardrail_hooks.tool_permission import (
        ToolPermissionGuardrail,
    )

    return ToolPermissionGuardrail(
        guardrail_name="tool-firewall",
        event_hook="post_call",
        rules=_TOOL_FIREWALL_RULES,
        default_action="deny",
        on_disallowed_action="block",
        default_on=default_on,
    )


def _make_real_dispatch_user_api_key_dict():
    d = _make_user_api_key_dict(request_route="/anthropic/v1/messages")
    # collect_guardrails reads these when an endpoint-level config is present
    d.metadata = {}
    d.team_metadata = {}
    return d


def _real_dispatch_patches(upstream_body: dict, request_body: dict):
    """Patches for end-to-end pass_through_request runs with a REAL ProxyLogging."""
    from litellm.caching.dual_cache import DualCache
    from litellm.proxy.utils import ProxyLogging

    proxy_logging_obj = ProxyLogging(user_api_key_cache=DualCache())

    mock_response = _make_httpx_response(upstream_body)
    mock_async_client_obj = MagicMock()
    mock_async_client_obj.client = AsyncMock()
    mock_pt_logging = MagicMock()
    mock_pt_logging.pass_through_async_success_handler = AsyncMock()

    patches = [
        patch(
            f"{_PT_MOD}.HttpPassThroughEndpointHelpers.non_streaming_http_request_handler",
            new_callable=AsyncMock,
            return_value=mock_response,
        ),
        patch(f"{_PT_MOD}._is_streaming_response", return_value=False),
        patch(  # test-quality-ok: inject the real ProxyLogging that pass_through_request imports at call time
            "litellm.proxy.proxy_server.proxy_logging_obj",
            proxy_logging_obj,
        ),
        patch(f"{_PT_MOD}.pass_through_endpoint_logging", mock_pt_logging),
        patch(f"{_PT_MOD}.get_async_httpx_client", return_value=mock_async_client_obj),
        patch(
            f"{_PT_MOD}._read_request_body",
            new_callable=AsyncMock,
            return_value=request_body,
        ),
        patch(f"{_PT_MOD}._safe_get_request_headers", return_value={}),
    ]

    stack = ExitStack()
    for p in patches:
        stack.enter_context(p)
    return stack


_PASSTHROUGH_REQUEST_BODY = {
    "model": "claude-x",
    "max_tokens": 64,
    "messages": [{"role": "user", "content": "go"}],
}


@pytest.mark.asyncio
class TestPostCallGuardrailEnforcement:
    """A post_call guardrail attached to a pass-through endpoint must enforce."""

    async def test_endpoint_attached_guardrail_blocks_denied_tool_use(self):
        """Issue #32201 primary repro: guardrail attached via the endpoint's
        `guardrails` config blocks an upstream response with a denied tool_use
        instead of relaying it with HTTP 200."""
        guardrail = _make_tool_firewall_guardrail(default_on=False)
        litellm.logging_callback_manager.add_litellm_callback(guardrail)

        with _real_dispatch_patches(
            _ANTHROPIC_DENIED_TOOL_USE_RESPONSE, dict(_PASSTHROUGH_REQUEST_BODY)
        ):
            with pytest.raises(ProxyException) as exc_info:
                await pass_through_request(
                    request=_make_mock_request(),
                    target="https://example.com/v1/messages",
                    custom_headers={"Content-Type": "application/json"},
                    user_api_key_dict=_make_real_dispatch_user_api_key_dict(),
                    stream=False,
                    guardrails_config={"tool-firewall": None},
                )

        assert str(exc_info.value.code) == "400"
        assert "tool-firewall" in str(exc_info.value.message)

    async def test_default_on_guardrail_blocks_without_endpoint_config(self):
        """A `default_on: true` post_call guardrail must enforce on
        pass-through routes that have no endpoint-level guardrails config
        (e.g. the provider-native /anthropic/* passthrough)."""
        guardrail = _make_tool_firewall_guardrail(default_on=True)
        litellm.logging_callback_manager.add_litellm_callback(guardrail)

        with _real_dispatch_patches(
            _ANTHROPIC_DENIED_TOOL_USE_RESPONSE, dict(_PASSTHROUGH_REQUEST_BODY)
        ):
            with pytest.raises(ProxyException) as exc_info:
                await pass_through_request(
                    request=_make_mock_request(),
                    target="https://example.com/v1/messages",
                    custom_headers={"Content-Type": "application/json"},
                    user_api_key_dict=_make_real_dispatch_user_api_key_dict(),
                    stream=False,
                )

        assert str(exc_info.value.code) == "400"

    async def test_request_body_guardrails_param_enforced_post_call(self):
        """A per-request `guardrails` body param (honored by pre_call today)
        must also attach the guardrail for post-call enforcement."""
        guardrail = _make_tool_firewall_guardrail(default_on=False)
        litellm.logging_callback_manager.add_litellm_callback(guardrail)

        body = dict(_PASSTHROUGH_REQUEST_BODY)
        body["guardrails"] = ["tool-firewall"]

        with _real_dispatch_patches(_ANTHROPIC_DENIED_TOOL_USE_RESPONSE, body):
            with pytest.raises(ProxyException) as exc_info:
                await pass_through_request(
                    request=_make_mock_request(),
                    target="https://example.com/v1/messages",
                    custom_headers={"Content-Type": "application/json"},
                    user_api_key_dict=_make_real_dispatch_user_api_key_dict(),
                    stream=False,
                )

        assert str(exc_info.value.code) == "400"

    async def test_passing_response_is_relayed_unchanged(self):
        """An upstream response whose tool_use passes the rules must be
        relayed unchanged with the upstream status code."""
        guardrail = _make_tool_firewall_guardrail(default_on=True)
        litellm.logging_callback_manager.add_litellm_callback(guardrail)

        with _real_dispatch_patches(
            _ANTHROPIC_SAFE_RESPONSE, dict(_PASSTHROUGH_REQUEST_BODY)
        ):
            result = await pass_through_request(
                request=_make_mock_request(),
                target="https://example.com/v1/messages",
                custom_headers={"Content-Type": "application/json"},
                user_api_key_dict=_make_real_dispatch_user_api_key_dict(),
                stream=False,
            )

        assert result.status_code == 200
        assert json.loads(bytes(result.body)) == _ANTHROPIC_SAFE_RESPONSE

    async def test_unattached_guardrail_skips_post_call_hook(self):
        """With a registered but unattached (non-default_on) guardrail, the
        post-call hook must not run: pass-through stays opt-in for guardrails
        and non-guardrail callbacks must not start firing on plain traffic."""
        guardrail = _make_tool_firewall_guardrail(default_on=False)
        litellm.logging_callback_manager.add_litellm_callback(guardrail)

        mock_proxy_logging = MagicMock()
        mock_proxy_logging.pre_call_hook = AsyncMock(
            return_value=dict(_PASSTHROUGH_REQUEST_BODY)
        )
        mock_proxy_logging.post_call_success_hook = AsyncMock()
        mock_proxy_logging.post_call_response_headers_hook = AsyncMock(return_value={})

        mock_response = _make_httpx_response(_ANTHROPIC_DENIED_TOOL_USE_RESPONSE)
        with _common_patches(mock_proxy_logging, mock_response):
            result = await pass_through_request(
                request=_make_mock_request(),
                target="https://example.com/v1/messages",
                custom_headers={"Content-Type": "application/json"},
                user_api_key_dict=_make_real_dispatch_user_api_key_dict(),
                stream=False,
            )

        mock_proxy_logging.post_call_success_hook.assert_not_awaited()
        assert result.status_code == 200
