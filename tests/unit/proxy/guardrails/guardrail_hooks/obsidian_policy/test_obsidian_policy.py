import json
from collections.abc import Callable

import httpx
import pytest
from fastapi import HTTPException
from mcp.types import TextContent

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.guardrails.guardrail_hooks.obsidian_policy import (
    ObsidianPolicyGuardrail,
    guardrail_class_registry,
    guardrail_initializer_registry,
    resolve_settings,
)
from litellm.proxy.guardrails.guardrail_hooks.obsidian_policy.client import ObsidianClient, ObsidianSettings
from litellm.types.guardrails import GuardrailEventHooks, LitellmParams, SupportedGuardrailIntegrations
from litellm.types.llms.base import HiddenParams
from litellm.types.mcp import MCPPostCallResponseObject

Handler = Callable[[httpx.Request], httpx.Response]


def _guardrail(
    handler: Handler, event_hook=(GuardrailEventHooks.pre_mcp_call, GuardrailEventHooks.post_mcp_call)
) -> ObsidianPolicyGuardrail:
    settings = ObsidianSettings(
        endpoint="https://obsidian.example.test/enforcement",
        gateway_token="gw-token",
        client_platform="acme_gateway",
        timeout_seconds=1.0,
        fail_open=False,
    )
    client = ObsidianClient(settings, httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    return ObsidianPolicyGuardrail(
        client=client, guardrail_name="obsidian", event_hook=list(event_hook), default_on=True
    )


def _verdict(decision: str, **extra) -> httpx.Response:
    return httpx.Response(200, json={"decision": decision, "evaluation_id": "eval-9", **extra})


def _pre_data() -> dict:
    return {
        "mcp_tool_name": "delete_repository",
        "mcp_arguments": {"repo": "acme/app"},
        "mcp_server_name": "github",
        "metadata": {},
    }


def _auth() -> UserAPIKeyAuth:
    return UserAPIKeyAuth(user_id="u-1", user_email="a@example.test")


def _post_kwargs() -> dict:
    return {
        "litellm_call_id": "call-1",
        "litellm_params": {
            "metadata": {"user_api_key_user_id": "u-2", "user_api_key_user_email": "b@example.test"},
        },
        "mcp_tool_call_metadata": {
            "name": "get_secret",
            "arguments": {"key": "db"},
            "mcp_server_name": "vault",
            "mcp_session_id": "mcp-sess-1",
        },
    }


def _post_response(text: str = "hunter2") -> MCPPostCallResponseObject:
    return MCPPostCallResponseObject(
        mcp_tool_call_response=[TextContent(type="text", text=text)], hidden_params=HiddenParams()
    )


async def test_pre_call_allows_and_returns_data_unchanged():
    data = _pre_data()
    result = await _guardrail(lambda request: _verdict("pass")).async_pre_call_hook(
        _auth(), None, data, "call_mcp_tool"
    )
    assert result is data


async def test_pre_call_sends_tool_server_arguments_and_caller():
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return _verdict("pass")

    await _guardrail(handler).async_pre_call_hook(_auth(), None, _pre_data(), "call_mcp_tool")

    body = json.loads(captured[0].content)
    assert body["stage"] == "PreToolUse"
    assert body["target_service"] == "github"
    assert body["action_param"] == {"tool_name": "delete_repository", "args": {"repo": "acme/app"}}
    assert body["user"] == {"id": "u-1", "email": "a@example.test"}


async def test_pre_call_block_raises_403_with_evaluation_id_and_message():
    handler = lambda request: _verdict(  # noqa: E731
        "block", message="No repo deletion", matched_policies=[{"id": "p1", "name": "Repo deletion"}]
    )
    with pytest.raises(HTTPException) as raised:
        await _guardrail(handler).async_pre_call_hook(_auth(), None, _pre_data(), "call_mcp_tool")

    assert raised.value.status_code == 403
    assert raised.value.detail["evaluation_id"] == "eval-9"
    assert raised.value.detail["message"] == "No repo deletion"
    assert raised.value.detail["stage"] == "PreToolUse"
    assert raised.value.detail["matched_policies"] == [{"id": "p1", "name": "Repo deletion"}]


async def test_pre_call_fails_closed_when_obsidian_is_unreachable():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(HTTPException) as raised:
        await _guardrail(handler).async_pre_call_hook(_auth(), None, _pre_data(), "call_mcp_tool")
    assert raised.value.status_code == 403


async def test_pre_call_ignores_non_mcp_calls_without_contacting_obsidian():
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return _verdict("block")

    data = {"messages": []}
    result = await _guardrail(handler).async_pre_call_hook(_auth(), None, data, "completion")
    assert result is data
    assert calls == []


async def test_pre_call_does_nothing_when_only_post_stage_is_configured():
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return _verdict("block")

    guardrail = _guardrail(handler, event_hook=(GuardrailEventHooks.post_mcp_call,))
    data = _pre_data()
    assert await guardrail.async_pre_call_hook(_auth(), None, data, "call_mcp_tool") is data
    assert calls == []


async def test_post_call_allows_by_returning_none():
    result = await _guardrail(lambda request: _verdict("pass")).async_post_mcp_tool_call_hook(
        _post_kwargs(), _post_response(), None, None
    )
    assert result is None


async def test_post_call_sends_tool_call_result_and_caller():
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return _verdict("pass")

    await _guardrail(handler).async_post_mcp_tool_call_hook(_post_kwargs(), _post_response("s3cret"), None, None)

    body = json.loads(captured[0].content)
    assert body["stage"] == "PostToolUse"
    assert body["target_service"] == "vault"
    assert body["session_id"] == "mcp-sess-1"
    assert body["action_param"]["tool_name"] == "get_secret"
    assert body["action_param"]["args"] == {"key": "db"}
    assert body["action_param"]["tool_response"]["content"][0]["text"] == "s3cret"
    assert body["user"] == {"id": "u-2", "email": "b@example.test"}


async def test_post_call_block_replaces_result_so_the_original_is_never_returned():
    handler = lambda request: _verdict("block", message="Secret in output")  # noqa: E731
    original = _post_response("hunter2")

    replacement = await _guardrail(handler).async_post_mcp_tool_call_hook(_post_kwargs(), original, None, None)

    assert replacement is not None
    payload = json.loads(replacement.mcp_tool_call_response[0].text)
    assert payload["evaluation_id"] == "eval-9"
    assert payload["stage"] == "PostToolUse"
    assert "hunter2" not in json.dumps(payload)
    assert len(replacement.mcp_tool_call_response) == 1


async def test_post_call_fails_closed_when_obsidian_is_unreachable():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    replacement = await _guardrail(handler).async_post_mcp_tool_call_hook(_post_kwargs(), _post_response(), None, None)
    assert replacement is not None
    assert "hunter2" not in replacement.mcp_tool_call_response[0].text


async def test_post_call_does_nothing_when_only_pre_stage_is_configured():
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return _verdict("block")

    guardrail = _guardrail(handler, event_hook=(GuardrailEventHooks.pre_mcp_call,))
    assert await guardrail.async_post_mcp_tool_call_hook(_post_kwargs(), _post_response(), None, None) is None
    assert calls == []


def test_supported_hooks_are_the_two_mcp_stages():
    assert set(ObsidianPolicyGuardrail.get_supported_event_hooks()) == {
        GuardrailEventHooks.pre_mcp_call,
        GuardrailEventHooks.post_mcp_call,
    }


def test_guardrail_is_registered_under_obsidian_policy():
    key = SupportedGuardrailIntegrations.OBSIDIAN_POLICY.value
    assert guardrail_class_registry[key] is ObsidianPolicyGuardrail
    assert key in guardrail_initializer_registry


def _params(**overrides) -> LitellmParams:
    return LitellmParams(guardrail="obsidian_policy", mode="pre_mcp_call", **overrides)


def test_settings_require_a_gateway_token(monkeypatch):
    monkeypatch.delenv("OBSIDIAN_GATEWAY_TOKEN", raising=False)
    with pytest.raises(ValueError, match="OBSIDIAN_GATEWAY_TOKEN"):
        resolve_settings(_params())


def test_settings_read_token_and_overrides_from_environment(monkeypatch):
    monkeypatch.setenv("OBSIDIAN_GATEWAY_TOKEN", "env-token")
    monkeypatch.setenv("OBSIDIAN_CLIENT_PLATFORM", "env_platform")
    monkeypatch.setenv("OBSIDIAN_ENDPOINT", "https://env.example.test/enforce")
    settings = resolve_settings(_params())
    assert settings.gateway_token == "env-token"
    assert settings.client_platform == "env_platform"
    assert settings.endpoint == "https://env.example.test/enforce"


def test_explicit_params_win_over_environment(monkeypatch):
    monkeypatch.setenv("OBSIDIAN_GATEWAY_TOKEN", "env-token")
    settings = resolve_settings(_params(api_key="cfg-token", client_platform="cfg_platform", timeout=7.0))
    assert settings.gateway_token == "cfg-token"
    assert settings.client_platform == "cfg_platform"
    assert settings.timeout_seconds == 7.0


def test_reserved_client_platform_is_rejected_at_startup():
    with pytest.raises(ValueError, match="reserved"):
        resolve_settings(_params(api_key="t", client_platform="Cursor"))


@pytest.mark.parametrize("fallback, fail_open", [("fail_closed", False), ("fail_open", True)])
def test_unreachable_fallback_maps_to_fail_open(fallback, fail_open):
    assert resolve_settings(_params(api_key="t", unreachable_fallback=fallback)).fail_open is fail_open


def test_fails_closed_by_default():
    assert resolve_settings(_params(api_key="t")).fail_open is False
