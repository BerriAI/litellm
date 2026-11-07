import json
from collections.abc import Callable

import httpx
import pytest
from fastapi import HTTPException
from types import SimpleNamespace

from mcp.types import CallToolResult, ImageContent, TextContent

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.guardrails.guardrail_hooks.obsidian_policy import (
    ObsidianPolicyGuardrail,
    build_guardrail,
    guardrail_class_registry,
    guardrail_initializer_registry,
    resolve_settings,
)
from litellm.proxy.guardrails.guardrail_hooks.obsidian_policy.client import (
    InvalidSettings,
    ObsidianClient,
    ObsidianSettings,
)
from litellm.proxy.mcp_server.operations import _http_detail_message
from litellm.types.guardrails import GuardrailEventHooks, LitellmParams, SupportedGuardrailIntegrations
from litellm.types.llms.base import HiddenParams
from litellm.types.mcp import MCPPostCallResponseObject

Handler = Callable[[httpx.Request], httpx.Response]


def _guardrail(
    handler: Handler,
    event_hook=(GuardrailEventHooks.pre_mcp_call, GuardrailEventHooks.post_mcp_call),
    fail_open: bool = False,
    default_on: bool = True,
) -> ObsidianPolicyGuardrail:
    settings = ObsidianSettings(
        endpoint="https://obsidian.example.test/enforcement",
        gateway_token="gw-token",
        client_platform="acme_gateway",
        timeout_seconds=1.0,
        fail_open=fail_open,
    )
    client = ObsidianClient(settings, httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    return ObsidianPolicyGuardrail(
        client=client, guardrail_name="obsidian", event_hook=list(event_hook), default_on=default_on
    )


def _recorder(verdict: httpx.Response) -> tuple[list[httpx.Request], Handler]:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return verdict

    return captured, handler


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


async def test_pre_call_block_reaches_mcp_clients_with_policy_message_and_evaluation_id():
    handler = lambda request: _verdict("block", message="No repo deletion")  # noqa: E731
    with pytest.raises(HTTPException) as raised:
        await _guardrail(handler).async_pre_call_hook(_auth(), None, _pre_data(), "call_mcp_tool")

    shown_to_mcp_client = _http_detail_message(raised.value.detail)
    assert shown_to_mcp_client == "Blocked by Obsidian policy: No repo deletion (evaluation_id eval-9)"


async def test_pre_call_fails_closed_when_obsidian_is_unreachable():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(HTTPException) as raised:
        await _guardrail(handler).async_pre_call_hook(_auth(), None, _pre_data(), "call_mcp_tool")
    assert raised.value.status_code == 403
    assert _http_detail_message(raised.value.detail) == (
        "Obsidian policy check unavailable: Obsidian request failed: ConnectError"
    ), "an outage must not read as a policy block"


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


def test_settings_without_a_gateway_token_are_invalid(monkeypatch):
    monkeypatch.delenv("OBSIDIAN_GATEWAY_TOKEN", raising=False)
    settings = resolve_settings(_params())
    assert isinstance(settings, InvalidSettings)
    assert "OBSIDIAN_GATEWAY_TOKEN" in settings.reason


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


def test_reserved_client_platform_is_invalid():
    settings = resolve_settings(_params(api_key="t", client_platform="Cursor"))
    assert isinstance(settings, InvalidSettings)
    assert "reserved" in settings.reason


def _never_called(request: httpx.Request) -> httpx.Response:
    raise AssertionError("a misconfigured guardrail must not contact Obsidian")


_MISCONFIGURATIONS = pytest.mark.parametrize(
    "overrides",
    [
        {"api_key": None},
        {"client_platform": "cursor"},
        {"mode": "pre_mpc_call"},
        {"mode": "pre_call"},
    ],
    ids=["no-token", "reserved-platform", "unknown-mode", "non-mcp-mode"],
)


def _built(monkeypatch, **overrides) -> ObsidianPolicyGuardrail:
    monkeypatch.delenv("OBSIDIAN_GATEWAY_TOKEN", raising=False)
    params = {"api_key": "t", "mode": ["pre_mcp_call", "post_mcp_call"], "default_on": True, **overrides}
    http = httpx.AsyncClient(transport=httpx.MockTransport(_never_called))
    return build_guardrail(LitellmParams(guardrail="obsidian_policy", **params), "obsidian", http)


@_MISCONFIGURATIONS
async def test_misconfigured_guardrail_still_denies_tool_calls_instead_of_being_skipped(monkeypatch, overrides):
    guardrail = _built(monkeypatch, **overrides)

    with pytest.raises(HTTPException) as raised:
        await guardrail.async_pre_call_hook(_auth(), None, _pre_data(), "call_mcp_tool")
    assert _http_detail_message(raised.value.detail) == (
        "Obsidian policy check unavailable: the guardrail configuration is invalid, see the proxy log"
    )
    withheld = await guardrail.async_post_mcp_tool_call_hook(_post_kwargs(), _post_response("hunter2"), None, None)
    assert withheld is not None and "hunter2" not in withheld.mcp_tool_call_response[0].text


@_MISCONFIGURATIONS
async def test_misconfigured_guardrail_allows_tool_calls_when_set_to_fail_open(monkeypatch, overrides):
    guardrail = _built(monkeypatch, unreachable_fallback="fail_open", **overrides)

    data = _pre_data()
    assert await guardrail.async_pre_call_hook(_auth(), None, data, "call_mcp_tool") is data
    assert await guardrail.async_post_mcp_tool_call_hook(_post_kwargs(), _post_response(), None, None) is None


async def test_valid_configuration_builds_a_guardrail_that_consults_obsidian(monkeypatch):
    captured, handler = _recorder(_verdict("pass"))
    params = LitellmParams(guardrail="obsidian_policy", mode="pre_mcp_call", api_key="t", default_on=True)
    guardrail = build_guardrail(params, "obsidian", httpx.AsyncClient(transport=httpx.MockTransport(handler)))

    data = _pre_data()
    assert await guardrail.async_pre_call_hook(_auth(), None, data, "call_mcp_tool") is data
    assert captured[0].headers["Authorization"] == "Bearer t"


@pytest.mark.parametrize("fallback, fail_open", [("fail_closed", False), ("fail_open", True)])
def test_unreachable_fallback_maps_to_fail_open(fallback, fail_open):
    assert resolve_settings(_params(api_key="t", unreachable_fallback=fallback)).fail_open is fail_open


def test_fails_closed_by_default():
    assert resolve_settings(_params(api_key="t")).fail_open is False


async def test_pre_call_without_an_auth_object_still_evaluates_and_omits_the_user():
    captured, handler = _recorder(_verdict("pass"))
    data = _pre_data()
    assert await _guardrail(handler).async_pre_call_hook(None, None, data, "call_mcp_tool") is data
    assert "user" not in json.loads(captured[0].content)


def _recorded_call(**overrides) -> dict:
    return {"name": "get_secret", "arguments": {"key": "db"}, "mcp_session_id": "sess-9", **overrides}


@pytest.mark.parametrize(
    "recorded, expected_service",
    [
        (_recorded_call(mcp_server_name="vault"), "vault"),
        (_recorded_call(namespaced_tool_name="gh/get_secret"), "gh"),
    ],
)
async def test_both_stages_report_the_same_session_and_service(recorded, expected_service):
    captured, handler = _recorder(_verdict("pass"))
    guardrail = _guardrail(handler)
    logging_obj = SimpleNamespace(litellm_call_id="call-1", model_call_details={"mcp_tool_call_metadata": recorded})

    pre = {**_pre_data(), "mcp_server_name": "gh-alias", "litellm_logging_obj": logging_obj}
    await guardrail.async_pre_call_hook(_auth(), None, pre, "call_mcp_tool")
    post = {**_post_kwargs(), "mcp_tool_call_metadata": recorded}
    await guardrail.async_post_mcp_tool_call_hook(post, _post_response(), None, None)

    pre_body, post_body = (json.loads(request.content) for request in captured)
    assert pre_body["session_id"] == post_body["session_id"] == "sess-9"
    assert pre_body["target_service"] == post_body["target_service"] == expected_service


async def test_post_call_denies_on_a_block_even_when_the_recorded_call_is_malformed():
    kwargs = {**_post_kwargs(), "mcp_tool_call_metadata": {"name": 42, "arguments": ["not", "a", "mapping"]}}
    replacement = await _guardrail(lambda request: _verdict("block")).async_post_mcp_tool_call_hook(
        kwargs, _post_response(), None, None
    )
    assert replacement is not None
    assert "hunter2" not in replacement.mcp_tool_call_response[0].text


@pytest.mark.parametrize("fail_open", [False, True])
async def test_post_call_turns_an_internal_error_into_a_fail_mode_decision(fail_open):
    guardrail = _guardrail(lambda request: _verdict("pass"), fail_open=fail_open)
    replacement = await guardrail.async_post_mcp_tool_call_hook(_post_kwargs(), object(), None, None)
    if fail_open:
        assert replacement is None
    else:
        assert replacement is not None
        assert "guardrail raised" in replacement.mcp_tool_call_response[0].text


async def test_post_call_strips_binary_content_before_sending_it_to_obsidian():
    captured, handler = _recorder(_verdict("pass"))
    response = MCPPostCallResponseObject(
        mcp_tool_call_response=[ImageContent(type="image", data="A" * 400, mimeType="image/png")],
        hidden_params=HiddenParams(),
    )
    await _guardrail(handler).async_post_mcp_tool_call_hook(_post_kwargs(), response, None, None)
    item = json.loads(captured[0].content)["action_param"]["tool_response"]["content"][0]
    assert "data" not in item
    assert item["data_bytes"] == 400
    assert item["mimeType"] == "image/png"


@pytest.mark.parametrize("decision, is_error", [("block", True), ("pass", False)])
async def test_post_call_block_marks_the_real_tool_result_as_an_error(decision, is_error):
    original = CallToolResult(content=[TextContent(type="text", text="hunter2")])
    kwargs = {**_post_kwargs(), "original_response": original}
    await _guardrail(lambda request: _verdict(decision)).async_post_mcp_tool_call_hook(
        kwargs, _post_response(), None, None
    )
    assert original.is_error is is_error


@pytest.mark.parametrize(
    "metadata, runs",
    [
        ({"user_api_key_metadata": {"guardrails": ["obsidian"]}}, True),
        ({"user_api_key_team_metadata": {"guardrails": ["obsidian"]}}, True),
        ({"guardrails": ["obsidian"]}, True),
        ({"user_api_key_metadata": {"guardrails": ["other"]}}, False),
        ({}, False),
    ],
)
async def test_post_call_honours_key_and_team_attached_guardrails_when_not_default_on(metadata, runs):
    captured, handler = _recorder(_verdict("pass"))
    kwargs = {**_post_kwargs(), "litellm_params": {"metadata": metadata}}
    await _guardrail(handler, default_on=False).async_post_mcp_tool_call_hook(kwargs, _post_response(), None, None)
    assert bool(captured) is runs
