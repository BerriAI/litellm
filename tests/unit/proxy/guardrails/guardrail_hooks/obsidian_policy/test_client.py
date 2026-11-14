import json
from collections.abc import Callable

import httpx
import pytest

from litellm.proxy.guardrails.guardrail_hooks.obsidian_policy.client import (
    ObsidianClient,
    ObsidianSettings,
    ToolEvent,
    build_event,
    validate_client_platform,
)

ENDPOINT = "https://obsidian.example.test/enforcement"
Handler = Callable[[httpx.Request], httpx.Response]


def _client(handler: Handler, fail_open: bool = False) -> ObsidianClient:
    settings = ObsidianSettings(
        endpoint=ENDPOINT,
        gateway_token="gw-token",
        client_platform="acme_gateway",
        timeout_seconds=1.0,
        fail_open=fail_open,
    )
    return ObsidianClient(settings, httpx.AsyncClient(transport=httpx.MockTransport(handler)))


def _event(stage="PreToolUse", **overrides) -> ToolEvent:
    return ToolEvent(
        stage=stage,
        server_name="github",
        tool_name="delete_repository",
        arguments={"repo": "acme/app"},
        session_id="session-1",
        **overrides,
    )


def _verdict(decision: str, evaluation_id: str | None = "eval-1", **extra) -> httpx.Response:
    return httpx.Response(200, json={"decision": decision, "evaluation_id": evaluation_id, **extra})


async def test_pass_with_evaluation_id_allows():
    decision = await _client(lambda request: _verdict("pass")).evaluate(_event())
    assert decision.allowed is True
    assert decision.kind == "pass"
    assert decision.evaluation_id == "eval-1"
    assert decision.via_fail_mode is False


async def test_warn_allows_and_keeps_matched_policies():
    handler = lambda request: _verdict(  # noqa: E731
        "warn", matched_policies=[{"id": "p1", "name": "Repo deletion", "action": "warn"}]
    )
    decision = await _client(handler).evaluate(_event())
    assert decision.allowed is True
    assert decision.kind == "warn"
    assert [policy.name for policy in decision.matched_policies] == ["Repo deletion"]


async def test_block_denies_with_policy_message():
    handler = lambda request: _verdict("block", message="Repository deletion is not allowed")  # noqa: E731
    decision = await _client(handler).evaluate(_event())
    assert decision.allowed is False
    assert decision.kind == "block"
    assert decision.message == "Repository deletion is not allowed"
    assert decision.evaluation_id == "eval-1"


@pytest.mark.parametrize("fail_open", [True, False])
async def test_unrecognised_decision_blocks_even_when_fail_open(fail_open):
    decision = await _client(lambda request: _verdict("maybe"), fail_open=fail_open).evaluate(_event())
    assert decision.allowed is False
    assert decision.kind == "block"
    assert decision.via_fail_mode is False


@pytest.mark.parametrize("fail_open", [True, False])
async def test_pass_without_evaluation_id_follows_fail_mode(fail_open):
    handler = lambda request: _verdict("pass", evaluation_id=None)  # noqa: E731
    decision = await _client(handler, fail_open=fail_open).evaluate(_event())
    assert decision.allowed is fail_open
    assert decision.kind == "not_evaluated"
    assert decision.via_fail_mode is True


@pytest.mark.parametrize("fail_open", [True, False])
async def test_timeout_follows_fail_mode_and_is_not_retried(fail_open):
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        raise httpx.ReadTimeout("slow", request=request)

    decision = await _client(handler, fail_open=fail_open).evaluate(_event())
    assert decision.allowed is fail_open
    assert decision.kind == "error"
    assert decision.via_fail_mode is True
    assert len(calls) == 1


async def test_server_error_is_retried_once_then_succeeds():
    responses = iter([httpx.Response(503), _verdict("pass")])
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return next(responses)

    decision = await _client(handler).evaluate(_event())
    assert decision.allowed is True
    assert len(calls) == 2


async def test_persistent_server_error_gives_up_after_one_retry():
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(502)

    decision = await _client(handler).evaluate(_event())
    assert decision.allowed is False
    assert decision.kind == "error"
    assert "HTTP 502" in (decision.message or "")
    assert len(calls) == 2


async def test_connection_error_is_retried_once():
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        raise httpx.ConnectError("refused", request=request)

    decision = await _client(handler).evaluate(_event())
    assert decision.allowed is False
    assert decision.kind == "error"
    assert len(calls) == 2


async def test_client_error_is_not_retried_and_reports_obsidian_detail():
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(
            400,
            json={
                "error": {"code": "unregistered_runtime_extension", "message": "namespace is not registered"},
                "event_id": "e-1",
            },
        )

    decision = await _client(handler).evaluate(_event())
    assert decision.allowed is False
    assert "HTTP 400" in (decision.message or "")
    assert "unregistered_runtime_extension" in (decision.message or "")
    assert "namespace is not registered" in (decision.message or "")
    assert len(calls) == 1


async def test_non_json_response_is_an_error():
    decision = await _client(lambda request: httpx.Response(200, content=b"<html>")).evaluate(_event())
    assert decision.allowed is False
    assert decision.kind == "error"


async def test_request_carries_token_and_event_body_without_leaking_token_into_it():
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return _verdict("pass")

    await _client(handler).evaluate(_event(user_id="u-1", user_email="a@example.test"))

    request = captured[0]
    body = json.loads(request.content)
    assert str(request.url) == ENDPOINT
    assert request.headers["authorization"] == "Bearer gw-token"
    assert "gw-token" not in request.content.decode()
    assert body["stage"] == "PreToolUse"
    assert body["client_platform"] == "acme_gateway"
    assert body["target_service"] == "github"
    assert body["session_id"] == "session-1"
    assert body["action_param"] == {"tool_name": "delete_repository", "args": {"repo": "acme/app"}}
    assert body["user"] == {"id": "u-1", "email": "a@example.test"}


def test_pre_event_has_no_tool_response_and_post_event_does():
    pre = build_event(_event("PreToolUse", tool_response={"content": []}), "p")
    post = build_event(_event("PostToolUse", tool_response={"content": [{"type": "text", "text": "x"}]}), "p")
    assert "tool_response" not in pre["action_param"]
    assert post["action_param"]["tool_response"] == {"content": [{"type": "text", "text": "x"}]}


@pytest.mark.parametrize("response", [None, ""])
def test_post_event_sends_empty_object_when_tool_returned_nothing(response):
    event = build_event(_event("PostToolUse", tool_response=response), "p")
    assert event["action_param"]["tool_response"] == {}


def test_event_omits_user_when_unknown_and_generates_distinct_ids():
    first = build_event(_event(), "p")
    second = build_event(_event(), "p")
    assert "user" not in first
    assert first["event_id"] != second["event_id"]


@pytest.mark.parametrize("name", ["claudecode", "Cursor", "mcp-gateway", "MCP_GATEWAY", "copilot_studio", "n8n"])
def test_reserved_client_platforms_are_rejected_regardless_of_case_and_dashes(name):
    assert validate_client_platform(name) is not None


@pytest.mark.parametrize("name", ["", "has space", "x" * 65, "semi;colon", "acme_gateway\n"])
def test_malformed_client_platforms_are_rejected(name):
    assert validate_client_platform(name) is not None


@pytest.mark.parametrize("name", ["litellm_gateway", "acme-gw.1", "A" * 64])
def test_ordinary_client_platforms_are_accepted(name):
    assert validate_client_platform(name) is None
