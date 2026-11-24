"""Behavior pins for ``proxy_server.py`` queue routes.

Pins (PR2):
    - POST /queue/chat/completions
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import AbstractContextManager
from datetime import datetime, timezone
from queue import SimpleQueue
from typing import Final
from unittest.mock import AsyncMock, MagicMock, patch

import litellm
import pytest
from fastapi import Request, Response
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient
from starlette.requests import ClientDisconnect
from starlette.types import Message

from litellm import Router
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy import proxy_server
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.spend_tracking.spend_tracking_utils import get_logging_payload
from litellm.proxy.utils import ProxyLogging

from .conftest import normalize  # type: ignore[import-not-found]

HAPPY_RESPONSE = {
    "id": "chatcmpl-queue",
    "object": "chat.completion",
    "created": 0,
    "model": "gpt-4",
    "choices": [
        {
            "index": 0,
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": "queued reply"},
        }
    ],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    "priority": 0,
}


@pytest.fixture
def patched_queue(monkeypatch):
    router = MagicMock()
    router.health.return_value.split_priority.return_value = (0, {})
    router.schedule_acompletion = AsyncMock(return_value=dict(HAPPY_RESPONSE))
    monkeypatch.setattr(proxy_server, "llm_router", router)
    monkeypatch.setattr(
        proxy_server,
        "proxy_logging_obj",
        MagicMock(post_call_failure_hook=AsyncMock()),
    )
    return router


@pytest.fixture
def queue_no_router(monkeypatch):
    monkeypatch.setattr(proxy_server, "llm_router", None)
    monkeypatch.setattr(
        proxy_server,
        "proxy_logging_obj",
        MagicMock(post_call_failure_hook=AsyncMock()),
    )
    yield


def test_queue_chat_completions_happy(client, auth_as, patched_queue):
    """Pins ``POST /queue/chat/completions`` (happy)."""
    payload = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "hi"}],
        "priority": 0,
    }
    with auth_as():
        response = client.post("/queue/chat/completions", json=payload)
    assert response.status_code == 200
    assert normalize(response.json()) == {
        "id": "<VOLATILE>",
        "object": "chat.completion",
        "created": "<VOLATILE>",
        "model": "gpt-4",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "queued reply"},
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        "priority": 0,
    }


def test_queue_chat_completions_drops_client_sent_stream_usage_flags(
    client: TestClient, auth_as: Callable[..., AbstractContextManager[None]], patched_queue: MagicMock
) -> None:
    payload: Final = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "hi"}],
        "priority": 0,
        "_litellm_strip_stream_usage": True,
        "_litellm_stream_usage_injected": True,
    }
    with auth_as():
        response: Final = client.post("/queue/chat/completions", json=payload)
    assert response.status_code == 200
    forwarded: Final = patched_queue.schedule_acompletion.call_args.kwargs
    assert "_litellm_strip_stream_usage" not in forwarded
    assert "_litellm_stream_usage_injected" not in forwarded
    assert forwarded["messages"] == [{"role": "user", "content": "hi"}]


def test_queue_chat_completions_no_router_error(client, auth_as, queue_no_router):
    """Pins ``POST /queue/chat/completions`` (error: no llm_router)."""
    payload = {"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]}
    with auth_as():
        response = client.post("/queue/chat/completions", json=payload)
    assert response.status_code == 500
    assert len(response.content) > 0


def _queue_router(
    monkeypatch: pytest.MonkeyPatch, default_priority: int | None, mock_response: str = "queued"
) -> Router:
    router: Final = Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {"model": "openai/gpt-4", "api_key": "sk-queue-test", "mock_response": mock_response},
            }
        ],
        default_priority=default_priority,
    )
    monkeypatch.setattr(proxy_server, "llm_router", router)
    monkeypatch.setattr(proxy_server, "proxy_logging_obj", MagicMock(post_call_failure_hook=AsyncMock()))
    return router


class _QueueRequestLogger(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.requests: Final = SimpleQueue[dict[str, object]]()

    def log_pre_api_call(self, model: str, messages: object, kwargs: dict[str, object]) -> None:
        self.requests.put(kwargs)


@pytest.mark.parametrize("storage_mode", [None, "latest_turn"])
@pytest.mark.parametrize(
    ("key_setting", "team_setting", "stored"),
    [(None, False, False), (True, False, False), (False, None, False), (None, None, True), (True, True, True)],
)
def test_queued_requests_apply_authenticated_prompt_storage_policy(
    client: TestClient,
    auth_as: Callable[..., AbstractContextManager[None]],
    monkeypatch: pytest.MonkeyPatch,
    storage_mode: str | None,
    key_setting: bool | None,
    team_setting: bool | None,
    stored: bool,
) -> None:
    _queue_router(monkeypatch, default_priority=0)
    logger: Final = _QueueRequestLogger()
    monkeypatch.setattr(litellm, "callbacks", [logger])
    monkeypatch.setattr(
        proxy_server,
        "general_settings",
        {"store_prompts_in_spend_logs": True, "spend_logs_prompt_storage": storage_mode},
    )
    with auth_as(
        metadata={} if key_setting is None else {"store_prompts_in_spend_logs": key_setting},
        team_metadata={} if team_setting is None else {"store_prompts_in_spend_logs": team_setting},
    ):
        response: Final = client.post(
            "/queue/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "queue-private-prompt"}],
                "metadata": {
                    "user_api_key_metadata": {"store_prompts_in_spend_logs": not stored},
                    "user_api_key_team_metadata": {"store_prompts_in_spend_logs": not stored},
                },
            },
        )
    assert response.status_code == 200, response.text
    assert response.json()["choices"][0]["message"]["content"] == "queued"
    now: Final = datetime(2026, 1, 1, tzinfo=timezone.utc)
    payload: Final = get_logging_payload(
        kwargs=logger.requests.get_nowait(),
        response_obj=litellm.ModelResponse.model_validate(response.json()),
        start_time=now,
        end_time=now,
    )
    assert ("queue-private-prompt" in payload["proxy_server_request"]) is stored
    assert ("queued" in payload["response"]) is stored


@pytest.mark.parametrize(
    ("default_priority", "priority_fields", "expected_priority"),
    [
        (3, {"priority": "high", "drop_params": True}, 3),
        (3, {"priority": "1"}, 1),
        (None, {"priority": 1.0}, 1),
        (3, {"priority": 1}, 1),
        (3, {}, 3),
    ],
    ids=("dropped-word-takes-default", "digit-string", "integral-float", "explicit", "absent-takes-default"),
)
def test_queue_chat_completions_reports_the_priority_the_router_queued_at(
    client,
    auth_as,
    monkeypatch: pytest.MonkeyPatch,
    default_priority: int | None,
    priority_fields: dict[str, object],
    expected_priority: int,
) -> None:
    router: Final = _queue_router(monkeypatch, default_priority)
    payload: Final = {"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}], **priority_fields}
    with patch.object(router.scheduler, "add_request", wraps=router.scheduler.add_request) as add_request:
        with auth_as():
            response: Final = client.post("/queue/chat/completions", json=payload)
    assert response.status_code == 200, response.text
    assert [call.kwargs["request"].priority for call in add_request.await_args_list] == [expected_priority]
    assert response.headers.get("x-litellm-priority") == str(expected_priority)


@pytest.mark.parametrize(
    ("default_priority", "priority_fields", "message"),
    [
        (None, {}, "priority is required"),
        (3, {"priority": "high"}, "priority must be an integer"),
        (None, {"priority": "high", "drop_params": True}, "priority must be an integer, got str 'high'"),
    ],
    ids=("no-priority-and-no-default", "malformed", "dropped-word-and-no-default"),
)
def test_queue_chat_completions_rejects_a_request_it_cannot_queue_as_an_invalid_request(
    client,
    auth_as,
    monkeypatch: pytest.MonkeyPatch,
    default_priority: int | None,
    priority_fields: dict[str, object],
    message: str,
) -> None:
    router: Final = _queue_router(monkeypatch, default_priority)
    payload: Final = {"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}], **priority_fields}
    with patch.object(router.scheduler, "add_request", wraps=router.scheduler.add_request) as add_request:
        with auth_as():
            response: Final = client.post("/queue/chat/completions", json=payload)
    assert response.status_code == 400, response.text
    error: Final = response.json()["error"]
    assert (error["type"], error["param"]) == ("invalid_request_error", "priority")
    assert message in error["message"]
    assert add_request.await_count == 0


@pytest.mark.parametrize(
    ("default_priority", "priority_header", "priority_fields", "expected_priority"),
    [
        (5, "0", {"priority": 5}, 0),
        (5, "0", {}, 0),
        (None, "high", {"priority": 1, "drop_params": True}, 1),
        (None, "", {"priority": 1}, 1),
    ],
    ids=("header-beats-body", "header-only", "drop-params-keeps-body", "empty-header-keeps-body"),
)
def test_queue_chat_completions_queues_at_the_priority_header(
    client,
    auth_as,
    monkeypatch: pytest.MonkeyPatch,
    default_priority: int | None,
    priority_header: str,
    priority_fields: dict[str, object],
    expected_priority: int,
) -> None:
    router: Final = _queue_router(monkeypatch, default_priority=default_priority)
    payload: Final = {"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}], **priority_fields}
    with patch.object(router.scheduler, "add_request", wraps=router.scheduler.add_request) as add_request:
        with auth_as():
            response: Final = client.post(
                "/queue/chat/completions", json=payload, headers={"x-litellm-priority": priority_header}
            )
    assert response.status_code == 200, response.text
    assert [call.kwargs["request"].priority for call in add_request.await_args_list] == [expected_priority]
    assert response.headers.get("x-litellm-priority") == str(expected_priority)


def test_queue_chat_completions_rejects_a_malformed_priority_header_like_a_malformed_body_value(
    client, auth_as, monkeypatch: pytest.MonkeyPatch
) -> None:
    router: Final = _queue_router(monkeypatch, default_priority=5)
    payload: Final = {"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}], "priority": 1}
    with patch.object(router.scheduler, "add_request", wraps=router.scheduler.add_request) as add_request:
        with auth_as():
            response: Final = client.post(
                "/queue/chat/completions", json=payload, headers={"x-litellm-priority": "high"}
            )
    assert response.status_code == 400, response.text
    error: Final = response.json()["error"]
    assert (error["type"], error["param"]) == ("invalid_request_error", "priority")
    assert "priority must be an integer" in error["message"]
    assert add_request.await_count == 0


def test_streamed_queue_responses_report_the_priority_the_router_queued_at(
    client, auth_as, monkeypatch: pytest.MonkeyPatch
) -> None:
    _queue_router(monkeypatch, default_priority=3)
    payload: Final = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "hi"}],
        "priority": "1",
        "stream": True,
    }
    with auth_as():
        response: Final = client.post("/queue/chat/completions", json=payload)
    assert response.status_code == 200, response.text
    assert response.headers.get("x-litellm-priority") == "1"


def test_a_provider_bad_request_on_the_queue_route_keeps_its_own_message(
    client, auth_as, monkeypatch: pytest.MonkeyPatch
) -> None:
    _queue_router(monkeypatch, default_priority=3, mock_response="litellm.ContextWindowExceededError")
    payload: Final = {"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]}
    with auth_as():
        response: Final = client.post("/queue/chat/completions", json=payload)
    assert response.status_code == 400, response.text
    error: Final = response.json()["error"]
    assert error["type"] == "invalid_request_error"
    assert "context window" in error["message"].lower()
    assert "Authentication Error" not in error["message"]


@pytest.mark.parametrize(
    "body_tags,caller_tag", [(None, "header-tag"), (["body-tag"], "body-tag")], ids=["header_tags", "body_tags"]
)
def test_queued_chat_carries_key_team_and_caller_tags_and_the_budget_reservation(
    client: TestClient,
    auth_as: Callable[..., AbstractContextManager[None]],
    patched_queue: MagicMock,
    body_tags: list[str] | None,
    caller_tag: str,
) -> None:
    reservation: Final = {"reserved_cost": 0.01, "callback_bound": False}
    payload: Final = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "hi"}],
        "metadata": {"tags": ["sent"]},
        **({"tags": body_tags} if body_tags is not None else {}),
    }
    with auth_as(metadata={"tags": ["key-tag"]}, team_metadata={"tags": ["team-tag"]}, budget_reservation=reservation):
        response: Final = client.post("/queue/chat/completions", json=payload, headers={"x-litellm-tags": "header-tag"})
    assert response.status_code == 200, response.text
    metadata: Final = patched_queue.schedule_acompletion.call_args.kwargs["metadata"]
    assert metadata["tags"] == ["sent", "key-tag", "team-tag", caller_tag]
    assert tuple(metadata["inherited_tags"]) == ("key-tag", "team-tag")
    assert metadata["user_api_key_budget_reservation"] == reservation


@pytest.mark.parametrize(
    ("key_metadata", "litellm_metadata"),
    [({"tags": ["key-tag"]}, {}), ({}, {"tags": ["key-tag"]})],
    ids=["key_tags_with_empty_bucket", "bucket_tags"],
)
def test_a_queued_call_with_a_litellm_metadata_bucket_still_routes_by_its_tags(
    client: TestClient,
    auth_as: Callable[..., AbstractContextManager[None]],
    monkeypatch: pytest.MonkeyPatch,
    key_metadata: dict[str, object],
    litellm_metadata: dict[str, object],
) -> None:
    deployments: Final = [
        {
            "model_name": "gpt-4",
            "litellm_params": {
                "model": "openai/gpt-4",
                "api_key": "sk-queue-test",
                "mock_response": reply,
                "tags": [reply],
            },
        }
        for reply in ("key-tag", "default")
    ]
    monkeypatch.setattr(
        proxy_server, "llm_router", Router(model_list=deployments, default_priority=0, enable_tag_filtering=True)
    )
    monkeypatch.setattr(proxy_server, "proxy_logging_obj", MagicMock(post_call_failure_hook=AsyncMock()))
    with auth_as(metadata=key_metadata):
        response: Final = client.post(
            "/queue/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "hi"}],
                "litellm_metadata": litellm_metadata,
            },
        )
    assert response.status_code == 200, response.text
    assert response.json()["choices"][0]["message"]["content"] == "key-tag"


def test_a_queued_spend_row_ignores_key_fields_forged_in_the_litellm_metadata_bucket(
    client: TestClient, auth_as: Callable[..., AbstractContextManager[None]], monkeypatch: pytest.MonkeyPatch
) -> None:
    _queue_router(monkeypatch, default_priority=0)
    logger: Final = _QueueRequestLogger()
    monkeypatch.setattr(litellm, "callbacks", [logger])
    with auth_as(team_id="real-team"):
        response: Final = client.post(
            "/queue/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "hi"}],
                "litellm_metadata": {"user_api_key_team_id": "forged-team", "user_api_key_org_id": "forged-org"},
            },
        )
    assert response.status_code == 200, response.text
    now: Final = datetime(2026, 1, 1, tzinfo=timezone.utc)
    payload: Final = get_logging_payload(
        kwargs=logger.requests.get_nowait(),
        response_obj=litellm.ModelResponse.model_validate(response.json()),
        start_time=now,
        end_time=now,
    )
    assert payload["team_id"] == "real-team"
    assert payload["organization_id"] != "forged-org"


def test_a_queued_spend_row_stores_the_trace_id_the_caller_sent(
    client: TestClient, auth_as: Callable[..., AbstractContextManager[None]], monkeypatch: pytest.MonkeyPatch
) -> None:
    _queue_router(monkeypatch, default_priority=0)
    logger: Final = _QueueRequestLogger()
    monkeypatch.setattr(litellm, "callbacks", [logger])
    with auth_as():
        response: Final = client.post(
            "/queue/chat/completions",
            json={"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]},
            headers={"x-litellm-trace-id": "queue-trace-1"},
        )
    assert response.status_code == 200, response.text
    now: Final = datetime(2026, 1, 1, tzinfo=timezone.utc)
    payload: Final = get_logging_payload(
        kwargs=logger.requests.get_nowait(),
        response_obj=litellm.ModelResponse.model_validate(response.json()),
        start_time=now,
        end_time=now,
    )
    assert payload["session_id"] == "queue-trace-1"


def test_a_queued_stream_error_carries_the_call_id_in_its_header_and_body(
    client: TestClient, auth_as: Callable[..., AbstractContextManager[None]], monkeypatch: pytest.MonkeyPatch
) -> None:
    async def provider_rate_limited() -> AsyncGenerator[litellm.ModelResponseStream, None]:
        raise litellm.RateLimitError(message="slow down", llm_provider="openai", model="gpt-4")
        yield litellm.ModelResponseStream()

    router: Final = MagicMock()
    router.health.return_value.split_priority.return_value = (0, {})
    router.health.return_value.priority_header_fields.return_value = {}
    router.registry.return_value.deployments.return_value = []
    router.schedule_acompletion = AsyncMock(return_value=provider_rate_limited())
    monkeypatch.setattr(proxy_server, "llm_router", router)
    monkeypatch.setattr(proxy_server, "proxy_logging_obj", ProxyLogging(user_api_key_cache=MagicMock()))
    monkeypatch.setattr(proxy_server, "general_settings", {"include_call_id_in_error_body": True})
    with auth_as():
        response: Final = client.post(
            "/queue/chat/completions",
            json={"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}], "stream": True},
            headers={"x-litellm-call-id": "queue-call-1"},
        )
    assert response.status_code == 429, response.text
    assert response.headers.get("x-litellm-call-id") == "queue-call-1"
    assert response.json()["error"]["litellm_call_id"] == "queue-call-1"


@pytest.mark.parametrize("stream", [True, False])
def test_a_queued_call_the_provider_rejects_carries_the_call_id_in_its_header_and_body(
    client: TestClient,
    auth_as: Callable[..., AbstractContextManager[None]],
    monkeypatch: pytest.MonkeyPatch,
    stream: bool,
) -> None:
    router: Final = MagicMock()
    router.health.return_value.split_priority.return_value = (0, {})
    router.health.return_value.priority_header_fields.return_value = {}
    router.registry.return_value.deployments.return_value = []
    router.schedule_acompletion = AsyncMock(
        side_effect=litellm.NotFoundError(message="no such model", llm_provider="gemini", model="gemini-missing")
    )
    monkeypatch.setattr(proxy_server, "llm_router", router)
    monkeypatch.setattr(proxy_server, "proxy_logging_obj", ProxyLogging(user_api_key_cache=MagicMock()))
    monkeypatch.setattr(proxy_server, "general_settings", {"include_call_id_in_error_body": True})
    with auth_as():
        response: Final = client.post(
            "/queue/chat/completions",
            json={"model": "gemini-missing", "messages": [{"role": "user", "content": "hi"}], "stream": stream},
            headers={"x-litellm-call-id": "queue-call-2"},
        )
    assert response.status_code == 404, response.text
    assert response.headers.get("x-litellm-call-id") == "queue-call-2"
    assert response.json()["error"]["litellm_call_id"] == "queue-call-2"


class _QueueSuccessLogger(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.responses: Final = SimpleQueue[object]()

    async def async_log_success_event(
        self, kwargs: dict[str, object], response_obj: object, start_time: object, end_time: object
    ) -> None:
        self.responses.put(response_obj)


_QUEUED_STREAM_SCOPE: Final = {
    "type": "http",
    "asgi": {"version": "3.0", "spec_version": "2.4"},
    "http_version": "1.1",
    "method": "POST",
    "scheme": "http",
    "server": ("testserver", 80),
    "path": "/queue/chat/completions",
    "root_path": "",
    "headers": [(b"content-type", b"application/json")],
    "query_string": b"",
}


def _queued_stream_request(after_body: Callable[[], Awaitable[Message]]) -> Request:
    body: Final = json.dumps({"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}], "stream": True})
    body_read: Final = asyncio.Event()

    async def receive() -> Message:
        if body_read.is_set():
            return await after_body()
        body_read.set()
        return {"type": "http.request", "body": body.encode(), "more_body": False}

    return Request(_QUEUED_STREAM_SCOPE, receive)


async def _still_connected() -> Message:
    await asyncio.Event().wait()
    return {"type": "http.disconnect"}


async def _already_gone() -> Message:
    return {"type": "http.disconnect"}


async def test_a_queued_stream_the_client_abandons_bills_the_reply_it_already_streamed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _queue_router(monkeypatch, default_priority=0, mock_response="queued partial reply")
    monkeypatch.setattr(proxy_server, "proxy_logging_obj", ProxyLogging(user_api_key_cache=MagicMock()))
    logger: Final = _QueueSuccessLogger()
    monkeypatch.setattr(litellm, "callbacks", [logger])
    request: Final = _queued_stream_request(_still_connected)
    response: Final = await proxy_server.async_queue_request(
        request=request, fastapi_response=Response(), user_api_key_dict=UserAPIKeyAuth(api_key="sk-queue")
    )
    assert isinstance(response, StreamingResponse)
    delivered: Final = SimpleQueue[bytes]()

    async def send_until_the_client_hangs_up(message: Message) -> None:
        if message["type"] != "http.response.body":
            return
        if delivered.qsize() == 2:
            raise OSError("client hung up")
        delivered.put(bytes(message["body"]))

    with pytest.raises(ClientDisconnect):
        await response(_QUEUED_STREAM_SCOPE, request.receive, send_until_the_client_hangs_up)

    billed: Final = logger.responses.get_nowait()
    assert isinstance(billed, litellm.ModelResponse)
    billed_reply: Final = billed.choices[0].message.content
    assert isinstance(billed_reply, str)
    assert billed_reply
    assert "queued partial reply".startswith(billed_reply)
    assert billed_reply != "queued partial reply"


async def test_a_queued_stream_the_client_leaves_before_its_first_chunk_gives_back_its_budget_hold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider_closed: Final = asyncio.Event()

    async def provider_still_thinking() -> AsyncGenerator[litellm.ModelResponseStream, None]:
        try:
            await asyncio.Event().wait()
            yield litellm.ModelResponseStream()
        finally:
            provider_closed.set()

    router: Final = MagicMock()
    router.health.return_value.split_priority.return_value = (0, {})
    router.health.return_value.priority_header_fields.return_value = {}
    router.registry.return_value.deployments.return_value = []
    router.schedule_acompletion = AsyncMock(return_value=provider_still_thinking())
    monkeypatch.setattr(proxy_server, "llm_router", router)
    monkeypatch.setattr(proxy_server, "proxy_logging_obj", ProxyLogging(user_api_key_cache=MagicMock()))
    auth: Final = UserAPIKeyAuth(
        api_key="sk-queue", budget_reservation={"reserved_cost": 0.01, "input_cost": 0.0, "entries": []}
    )

    response: Final = await proxy_server.async_queue_request(
        request=_queued_stream_request(_already_gone), fastapi_response=Response(), user_api_key_dict=auth
    )

    assert auth.budget_reservation is not None
    assert auth.budget_reservation.get("finalized") is True
    assert provider_closed.is_set()
    assert not isinstance(response, StreamingResponse)


@pytest.mark.parametrize("stream", [False, True])
async def test_a_queued_call_cancelled_before_it_answers_gives_back_its_budget_hold(
    monkeypatch: pytest.MonkeyPatch, stream: bool
) -> None:
    waiting: Final = asyncio.Event()

    async def wait_for_a_worker(**_: object) -> object:
        waiting.set()
        await asyncio.Event().wait()
        return dict(HAPPY_RESPONSE)

    router: Final = MagicMock()
    router.health.return_value.split_priority.return_value = (0, {})
    router.health.return_value.priority_header_fields.return_value = {}
    router.registry.return_value.deployments.return_value = []
    router.schedule_acompletion = wait_for_a_worker
    monkeypatch.setattr(proxy_server, "llm_router", router)
    monkeypatch.setattr(proxy_server, "proxy_logging_obj", ProxyLogging(user_api_key_cache=MagicMock()))
    auth: Final = UserAPIKeyAuth(
        api_key="sk-queue", budget_reservation={"reserved_cost": 0.01, "input_cost": 0.0, "entries": []}
    )
    body: Final = json.dumps({"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}], "stream": stream})

    async def receive() -> Message:
        return {"type": "http.request", "body": body.encode(), "more_body": False}

    queued: Final = asyncio.ensure_future(
        proxy_server.async_queue_request(
            request=Request(_QUEUED_STREAM_SCOPE, receive), fastapi_response=Response(), user_api_key_dict=auth
        )
    )
    await waiting.wait()
    queued.cancel()
    with pytest.raises(asyncio.CancelledError):
        await queued

    assert auth.budget_reservation is not None
    assert auth.budget_reservation.get("finalized") is True
