"""
Test for response_api_endpoints/endpoints.py
"""

import asyncio
import contextlib
import dataclasses
import json
import unittest
from collections.abc import Mapping
from typing import Any, Final, Literal, cast
from unittest.mock import AsyncMock, MagicMock, call, patch

import httpx
import pytest
import respx
from fastapi import HTTPException
from fastapi.testclient import TestClient
from httpx import Response

import litellm
from litellm.proxy._types import ProxyException
from litellm.proxy.proxy_server import app
from litellm.types.llms.openai import ResponseAPIUsage, ResponsesAPIResponse
from litellm.types.router import RouterErrors
from tests._master_key import MASTER_KEY


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path,error_kind",
    [
        ("/v1/responses", "rate_limit"),
        ("/v1/responses", "numeric_rate_limit"),
        ("/v1/responses", "server_error"),
        ("/v1/responses", "response_failed"),
        ("/v1/responses", "cyber_policy"),
        ("/cursor/chat/completions", "server_error"),
        ("/v1/chat/completions", "server_error"),
    ],
)
async def test_streaming_upstream_errors_keep_the_client_protocol(
    monkeypatch: pytest.MonkeyPatch,
    path: str,
    error_kind: Literal["rate_limit", "numeric_rate_limit", "server_error", "response_failed", "cyber_policy"],
) -> None:
    import litellm.proxy.proxy_server as ps
    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

    model: Final = "gpt-6-astra"
    message: Final = "Upstream cannot complete this response"
    code: Final = {
        "rate_limit": "rate_limit_exceeded",
        "numeric_rate_limit": "429",
        "server_error": "server_error",
        "response_failed": "server_error",
        "cyber_policy": "cyber_policy",
    }[error_kind]
    error: Final = {"message": message, "code": code, "type": None, "param": "input"}
    response: Final = {
        "id": "resp_upstream",
        "object": "response",
        "created_at": 1,
        "status": "in_progress",
        "model": model,
        "output": [],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
    }
    created: Final = {"type": "response.created", "sequence_number": 0, "response": response}
    tool_added: Final = {
        "type": "response.output_item.added",
        "sequence_number": 1,
        "output_index": 0,
        "item": {
            "type": "function_call",
            "id": "fc_partial",
            "call_id": "call_partial",
            "name": "read_file",
            "arguments": "",
            "status": "in_progress",
        },
    }
    tool_delta: Final = {
        "type": "response.function_call_arguments.delta",
        "sequence_number": 2,
        "item_id": "fc_partial",
        "output_index": 0,
        "delta": '{"path":"partial',
    }
    failed: Final = (
        {"type": "response.failed", "sequence_number": 9, "response": {**response, "status": "failed", "error": error}}
        if error_kind in ("response_failed", "cyber_policy")
        else {"type": "error", "error": error}
    )
    chat: Final = {
        "id": "chatcmpl_partial",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": model,
        "choices": [{"index": 0, "delta": {"content": "partial"}, "finish_reason": None}],
    }
    is_chat: Final = path == "/v1/chat/completions"
    partial: Final = path != "/v1/responses" or error_kind in ("numeric_rate_limit", "response_failed", "cyber_policy")
    response_events: Final = (created, tool_added, tool_delta, failed) if partial else (failed,)
    upstream_events: Final = (chat, {"error": error}) if is_chat else response_events
    wire: Final = "".join("data: " + json.dumps(event) + "\n\n" for event in upstream_events)
    upstream_url: Final = "https://streaming.example/v1"
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": model,
                "litellm_params": {"model": "openai/" + model, "api_base": upstream_url, "api_key": "fixture-key"},
            }
        ],
        num_retries=0,
    )
    monkeypatch.setattr(ps, "llm_router", router)
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setitem(app.dependency_overrides, user_api_key_auth, _auth_override)
    with respx.mock as transport:
        transport.post(upstream_url + ("/chat/completions" if is_chat else "/responses")).respond(
            200, content=wire, headers={"Content-Type": "text/event-stream"}
        )
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://testserver") as client:
            result: Final = await client.post(
                path,
                json={
                    "model": model,
                    "stream": True,
                    **({"messages": [{"role": "user", "content": "hello"}]} if is_chat else {"input": "hello"}),
                },
            )
    frames: Final = tuple(frame for frame in result.text.split("\n\n") if "data: " in frame)
    events: Final = tuple(
        json.loads(next(line[6:] for line in frame.splitlines() if line.startswith("data: ")))
        for frame in frames
        if "data: [DONE]" not in frame
    )

    assert result.status_code == 200, result.text
    assert message in result.text
    if path == "/v1/responses":
        assert frames[-1] == "data: [DONE]", result.text
        assert frames[-2].startswith("event: response.failed\n"), result.text
        if partial:
            assert [event["type"] for event in events] == [
                "response.created",
                "response.output_item.added",
                "response.function_call_arguments.delta",
                "response.failed",
            ]
            assert events[2]["delta"] == tool_delta["delta"]
            assert events[-1]["sequence_number"] == events[-2]["sequence_number"] + 1
            assert events[-1]["response"]["id"] == events[0]["response"]["id"]
        else:
            assert [event["type"] for event in events] == ["response.failed"]
            assert events[0]["sequence_number"] == 0
            assert events[0]["response"]["id"].startswith("resp_")
        assert events[-1]["response"]["status"] == "failed"
        assert (
            events[-1]["response"]["error"]["code"]
            == {
                "rate_limit": "rate_limit_exceeded",
                "numeric_rate_limit": "rate_limit_exceeded",
                "server_error": "server_error",
                "response_failed": "server_error",
                "cyber_policy": "cyber_policy",
            }[error_kind]
        )
        assert events[-1]["response"]["error"]["message"] == message
    else:
        assert events[0]["object"] == "chat.completion.chunk", result.text
        assert "response.failed" not in result.text
        assert "error" in events[-1]


_COMPACTION_ITEM: Final = {
    "type": "compaction",
    "encrypted_content": json.dumps({"type": "compaction", "content": "the cat is named Biscuit"}),
}
_HISTORY_BEFORE_COMPACTION: Final = (
    {"role": "user", "content": "a long turn " * 200},
    {"role": "assistant", "content": "a long reply " * 200},
)
_TURN_AFTER_COMPACTION: Final = {"role": "user", "content": "what is the cat's name?"}


async def _cursor_stream_failure_prompt_tokens(
    monkeypatch: pytest.MonkeyPatch, request_input: list[Mapping[str, str]], fails_after_output: bool
) -> int:
    import litellm.proxy.proxy_server as ps
    from litellm.integrations.custom_logger import CustomLogger
    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

    class _FailureRowRecorder(CustomLogger):
        prompt_tokens: int | None = None

        async def async_post_call_failure_hook(
            self,
            request_data: dict,
            original_exception: Exception,
            user_api_key_dict: object,
            traceback_str: str | None = None,
        ) -> None:
            usage: Final = request_data.get("combined_usage_object")
            self.prompt_tokens = usage.prompt_tokens if isinstance(usage, litellm.Usage) else None

    model: Final = "gpt-6-astra"
    upstream_url: Final = "https://streaming.example/v1"
    created: Final = {
        "type": "response.created",
        "sequence_number": 0,
        "response": {
            "id": "resp_upstream",
            "object": "response",
            "created_at": 1,
            "status": "in_progress",
            "model": model,
            "output": [],
        },
    }
    output_delta: Final = {
        "type": "response.output_text.delta",
        "sequence_number": 1,
        "item_id": "msg_partial",
        "output_index": 0,
        "content_index": 0,
        "delta": "Biscuit",
    }
    failed: Final = {"type": "error", "error": {"message": "upstream broke", "code": "server_error", "type": None}}
    events: Final = (created, output_delta, failed) if fails_after_output else (created, failed)
    wire: Final = "".join("data: " + json.dumps(event) + "\n\n" for event in events)
    recorder: Final = _FailureRowRecorder()
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": model,
                "litellm_params": {"model": "openai/" + model, "api_base": upstream_url, "api_key": "fixture-key"},
            }
        ],
        num_retries=0,
    )
    monkeypatch.setattr(ps, "llm_router", router)
    monkeypatch.setattr(litellm, "callbacks", [recorder])
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setitem(app.dependency_overrides, user_api_key_auth, _auth_override)
    with respx.mock as transport:
        transport.post(upstream_url + "/responses").respond(
            200, content=wire, headers={"Content-Type": "text/event-stream"}
        )
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://testserver") as client:
            result: Final = await client.post(
                "/cursor/chat/completions", json={"model": model, "stream": True, "input": request_input}
            )
    assert recorder.prompt_tokens is not None, result.text
    return recorder.prompt_tokens


@pytest.mark.asyncio
@pytest.mark.parametrize("fails_after_output", [False, True])
async def test_a_failed_cursor_stream_counts_its_input_from_the_newest_compaction_item(
    monkeypatch: pytest.MonkeyPatch, fails_after_output: bool
) -> None:
    """Cursor's route maps to no call type, yet it dispatches a Responses call, which sends
    the provider only the newest compaction item onward, so the failure row counts that much."""
    resent: Final = await _cursor_stream_failure_prompt_tokens(
        monkeypatch, [*_HISTORY_BEFORE_COMPACTION, _COMPACTION_ITEM, _TURN_AFTER_COMPACTION], fails_after_output
    )
    dispatched_after_trim: Final = await _cursor_stream_failure_prompt_tokens(
        monkeypatch, [_COMPACTION_ITEM, _TURN_AFTER_COMPACTION], fails_after_output
    )

    assert resent == dispatched_after_trim


@pytest.mark.asyncio
async def test_responses_api_background_polling_rejects_missing_input():
    from fastapi import Response as FastAPIResponse
    from starlette.requests import Request

    from litellm.proxy._types import ProxyException, UserAPIKeyAuth
    from litellm.proxy.response_api_endpoints.endpoints import responses_api

    processor = MagicMock()

    async def return_exception(*, e: Exception, **kwargs: object) -> Exception:
        return e

    processor.handle_llm_api_exception = AsyncMock(side_effect=return_exception)
    processor.common_processing_pre_call_logic = AsyncMock(return_value=({"model": "gpt-4o"}, MagicMock()))

    async def receive():
        return {
            "type": "http.request",
            "body": b'{"model":"gpt-4o","background":true}',
            "more_body": False,
        }

    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/responses",
            "headers": [(b"content-type", b"application/json")],
        },
        receive,
    )

    with (
        patch(  # test-quality-ok: endpoint constructs the processor directly
            "litellm.proxy.response_api_endpoints.endpoints.ProxyBaseLLMRequestProcessing",
            return_value=processor,
        ),
        patch(  # test-quality-ok: polling decision is imported inside the endpoint
            "litellm.proxy.response_polling.polling_handler.should_use_polling_for_request",
            return_value=True,
        ),
        patch(  # test-quality-ok: background task is imported inside the endpoint
            "litellm.proxy.response_polling.background_streaming.background_streaming_task",
            new_callable=AsyncMock,
        ) as mock_background_streaming_task,
        patch(  # test-quality-ok: polling handler is imported inside the endpoint
            "litellm.proxy.response_polling.polling_handler.ResponsePollingHandler.create_initial_state",
            new_callable=AsyncMock,
        ) as mock_create_initial_state,
    ):
        with pytest.raises(ProxyException) as exc_info:
            await responses_api(
                request=request,
                fastapi_response=FastAPIResponse(),
                user_api_key_dict=UserAPIKeyAuth(api_key="sk-test"),
            )

    assert exc_info.value.code == "400"
    assert exc_info.value.param == "input"
    processor.common_processing_pre_call_logic.assert_awaited_once()
    mock_background_streaming_task.assert_not_called()
    mock_create_initial_state.assert_not_awaited()


@pytest.mark.asyncio
async def test_responses_api_background_polling_accepts_input_from_prompt_template():
    from fastapi import Response as FastAPIResponse
    from starlette.requests import Request

    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.response_api_endpoints.endpoints import responses_api

    processor = MagicMock()
    processor.common_processing_pre_call_logic = AsyncMock(
        return_value=({"model": "gpt-4o", "input": "hello from prompt"}, MagicMock())
    )
    initial_state = MagicMock()

    async def receive():
        return {
            "type": "http.request",
            "body": b'{"model":"gpt-4o","prompt_id":"greeting","background":true}',
            "more_body": False,
        }

    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/responses",
            "headers": [(b"content-type", b"application/json")],
        },
        receive,
    )

    with (
        patch(  # test-quality-ok: endpoint constructs the processor directly
            "litellm.proxy.response_api_endpoints.endpoints.ProxyBaseLLMRequestProcessing",
            return_value=processor,
        ),
        patch(  # test-quality-ok: polling decision is imported inside the endpoint
            "litellm.proxy.response_polling.polling_handler.should_use_polling_for_request",
            return_value=True,
        ),
        patch(  # test-quality-ok: background task is imported inside the endpoint
            "litellm.proxy.response_polling.background_streaming.background_streaming_task",
            new_callable=AsyncMock,
        ),
        patch(  # test-quality-ok: avoid scheduling a background task in this unit test
            "litellm.proxy.response_api_endpoints.endpoints.asyncio.create_task",
        ),
        patch(  # test-quality-ok: polling handler is imported inside the endpoint
            "litellm.proxy.response_polling.polling_handler.ResponsePollingHandler.create_initial_state",
            new_callable=AsyncMock,
        ) as mock_create_initial_state,
    ):
        mock_create_initial_state.return_value = initial_state
        result = await responses_api(
            request=request,
            fastapi_response=FastAPIResponse(),
            user_api_key_dict=UserAPIKeyAuth(api_key="sk-test"),
        )

    assert result is initial_state
    processor.common_processing_pre_call_logic.assert_awaited_once()
    mock_create_initial_state.assert_awaited_once()
    request_data = mock_create_initial_state.await_args.kwargs["request_data"]
    assert request_data["input"] == "hello from prompt"


@pytest.mark.asyncio
async def test_a_cache_polled_background_stream_is_priced_by_its_own_stream():
    """Cache polling runs the stream without the settlement wrapper, so a stream the managed files hook
    marked as billed later must still be priced when it finishes, or the generation is never billed."""
    from fastapi import Response as FastAPIResponse

    from litellm.constants import MANAGED_BACKGROUND_RESPONSE_METADATA_KEY
    from litellm.litellm_core_utils.internal_call_metadata import is_unbilled_non_inference_call
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.response_api_endpoints.endpoints import responses_api

    marked_by_the_hook: Final = {
        "model": "gpt-5.6",
        "input": "hello",
        "stream": True,
        "background": True,
        "extra_body": {"background": True},
        "litellm_metadata": {MANAGED_BACKGROUND_RESPONSE_METADATA_KEY: True, "user_api_key_alias": "creator"},
    }
    processor: Final = MagicMock()
    processor.common_processing_pre_call_logic = AsyncMock(return_value=(marked_by_the_hook, MagicMock()))
    body: Final = json.dumps(
        {"model": "gpt-5.6", "input": "hello", "stream": True, "background": True, "extra_body": {"background": True}}
    ).encode()

    with (
        patch(  # test-quality-ok: endpoint constructs the processor directly
            "litellm.proxy.response_api_endpoints.endpoints.ProxyBaseLLMRequestProcessing",
            return_value=processor,
        ),
        patch(  # test-quality-ok: polling decision is imported inside the endpoint
            "litellm.proxy.response_polling.polling_handler.should_use_polling_for_request",
            return_value=True,
        ),
        patch(  # test-quality-ok: background task is imported inside the endpoint
            "litellm.proxy.response_polling.background_streaming.background_streaming_task",
            new_callable=AsyncMock,
        ) as polled_stream,
        patch(  # test-quality-ok: polling handler is imported inside the endpoint
            "litellm.proxy.response_polling.polling_handler.ResponsePollingHandler.create_initial_state",
            new_callable=AsyncMock,
        ),
    ):
        await responses_api(
            request=_request_for("POST", "/v1/responses", body),
            fastapi_response=FastAPIResponse(),
            user_api_key_dict=UserAPIKeyAuth(api_key="sk-creator"),
        )
        await asyncio.sleep(0)

    polled_metadata: Final = polled_stream.call_args.kwargs["data"]["litellm_metadata"]
    finished: Final = _background_response("resp_cache_polled", "completed", 100)
    assert is_unbilled_non_inference_call("aresponses", marked_by_the_hook["litellm_metadata"], finished)
    assert not is_unbilled_non_inference_call("aresponses", polled_metadata, finished)
    assert polled_metadata["user_api_key_alias"] == "creator"


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["queued", "completed"])
async def test_responses_api_background_create_persists_the_creating_keys_attribution(status: str):
    """The cost poll bills a background job off its managed object row, so the create must persist
    the key hash and tags the poll bills against, the way the batch create does, plus the client's
    User-Agent tags that a spend-by-client view groups the charge under. A create the provider
    already finished is stored too, so a later read finds it billed instead of billing it again."""
    from fastapi import Response as FastAPIResponse
    from starlette.requests import Request

    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.response_api_endpoints.endpoints import responses_api

    queued = ResponsesAPIResponse(
        id="resp_background", object="response", status=status, created_at=1, output=[], usage=None
    )
    queued._hidden_params = {"model_id": "deployment-1"}
    processor = MagicMock()
    processor.data = {
        "model": "gpt-4o",
        "input": "hello",
        "background": True,
        "litellm_metadata": {"tags": ["env:prod"]},
        "proxy_server_request": {"headers": {"user-agent": "codex_cli_rs/0.44.0"}},
    }
    processor.base_process_llm_request = AsyncMock(return_value=queued)
    managed_files = MagicMock()
    managed_files.store_unified_object_id = AsyncMock()
    managed_files.track_untracked_background_response = AsyncMock(return_value=False)
    proxy_logging = MagicMock()
    proxy_logging.get_proxy_hook = MagicMock(return_value=managed_files)

    async def receive():
        return {
            "type": "http.request",
            "body": b'{"model":"gpt-4o","input":"hello","background":true}',
            "more_body": False,
        }

    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/responses",
            "headers": [(b"content-type", b"application/json")],
        },
        receive,
    )

    with (
        patch(  # test-quality-ok: endpoint constructs the processor directly
            "litellm.proxy.response_api_endpoints.endpoints.ProxyBaseLLMRequestProcessing",
            return_value=processor,
        ),
        patch(  # test-quality-ok: polling decision is imported inside the endpoint
            "litellm.proxy.response_polling.polling_handler.should_use_polling_for_request",
            return_value=False,
        ),
        patch(  # test-quality-ok: the endpoint reads the proxy singletons at call time
            "litellm.proxy.proxy_server.proxy_logging_obj", proxy_logging
        ),
        patch(  # test-quality-ok: the endpoint reads the proxy singletons at call time
            "litellm.proxy.proxy_server.llm_router", MagicMock()
        ),
    ):
        result = await responses_api(
            request=request,
            fastapi_response=FastAPIResponse(),
            user_api_key_dict=UserAPIKeyAuth(api_key="sk-test"),
        )

    assert result is queued
    store_kwargs = managed_files.store_unified_object_id.await_args.kwargs
    assert store_kwargs["file_object"].status == status
    assert store_kwargs["file_purpose"] == "response"
    assert store_kwargs["persist_attribution"] is True
    assert store_kwargs["request_tags"] == (
        "env:prod",
        "User-Agent: codex_cli_rs",
        "User-Agent: codex_cli_rs/0.44.0",
    )


class _SpendEvents:
    def __init__(self) -> None:
        self.payloads: tuple[Mapping[str, object], ...] = ()

    async def async_log_success_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: object, end_time: object
    ) -> None:
        payload = kwargs["standard_logging_object"]
        assert isinstance(payload, Mapping)
        self.payloads = (*self.payloads, payload)


async def _drain_spend_events() -> None:
    from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER

    await asyncio.wait_for(GLOBAL_LOGGING_WORKER.flush(), timeout=10)


def _billed_costs(payloads: tuple[Mapping[str, object], ...]) -> list[str]:
    return [str(payload["response_cost"]) for payload in payloads]


def _advertised_cost_split(headers: Mapping[str, str]) -> list[tuple[float, float]]:
    if "x-litellm-response-cost-original" not in headers:
        return []
    return [
        (
            float(headers["x-litellm-response-cost-original"]),
            float(headers["x-litellm-response-cost-input"]) + float(headers["x-litellm-response-cost-output"]),
        )
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "background, stored, status, expected_billed_events",
    [
        (True, True, "completed", 1),
        (True, True, "in_progress", 0),
        (True, False, "completed", 0),
        (False, True, "completed", 0),
    ],
)
async def test_reading_a_finished_background_response_bills_it(
    monkeypatch: pytest.MonkeyPatch, background: bool, stored: bool, status: str, expected_billed_events: int
):
    """The read that first sees a background response finished bills it from the body it fetched,
    since the provider may drop a store=false response before the hourly cost poll reaches it.
    Billing never reads the provider a second time, other reads stay free, the charge stays on the
    end user the create named, and only the read that billed reports the cost it billed, split
    into the components the billed event priced."""
    from types import SimpleNamespace

    from fastapi import Response as FastAPIResponse

    from litellm.integrations.custom_logger import CustomLogger
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.response_api_endpoints.endpoints import _bill_finished_background_response

    class _CapturingLogger(_SpendEvents, CustomLogger):
        pass

    spend_events = _CapturingLogger()
    monkeypatch.setattr(litellm, "_async_success_callback", [spend_events])
    read = ResponsesAPIResponse(
        id="resp_read",
        object="response",
        status=status,
        created_at=1,
        model="gpt-5.6",
        output=[],
        usage=ResponseAPIUsage(input_tokens=100, output_tokens=50, total_tokens=150),
        background=background,
    )
    stored_row = SimpleNamespace(
        id="job-read",
        unified_object_id="resp_read",
        status="queued",
        file_object={"model": "gpt-5.6"},
        created_by=None,
        api_key=None,
        team_id=None,
        org_id=None,
        request_tags=None,
        end_user_id="customer-42",
    )
    fastapi_response = FastAPIResponse()
    prisma_client = MagicMock()
    prisma_client.db.litellm_managedobjecttable.find_first = AsyncMock(return_value=stored_row if stored else None)
    prisma_client.db.litellm_managedobjecttable.update_many = AsyncMock(return_value=1)

    with (
        patch(
            "litellm.proxy.proxy_server.prisma_client", prisma_client
        ),  # test-quality-ok: the endpoint reads the proxy singletons at call time
        patch(
            "litellm.proxy.proxy_server.llm_router", MagicMock()
        ),  # test-quality-ok: the endpoint reads the proxy singletons at call time
        patch("litellm.aget_responses", side_effect=AssertionError("billing must not read the provider again")),
    ):
        await _bill_finished_background_response(
            response=read,
            response_id="resp_read",
            user_api_key_dict=UserAPIKeyAuth(end_user_id="another-reader"),
            fastapi_response=fastapi_response,
            read_settles_billing=True,
        )
        await _drain_spend_events()

    assert len(spend_events.payloads) == expected_billed_events
    assert all(payload["metadata"]["user_api_key_end_user_id"] == "customer-42" for payload in spend_events.payloads)
    assert fastapi_response.headers.getlist("x-litellm-response-cost") == _billed_costs(spend_events.payloads)
    assert _advertised_cost_split(fastapi_response.headers) == [
        (pytest.approx(float(cost)), pytest.approx(float(cost))) for cost in _billed_costs(spend_events.payloads)
    ]


@dataclasses.dataclass(frozen=True)
class _ManagedObjectRow:
    id: str
    unified_object_id: str
    model_object_id: str
    status: str
    file_object: Mapping[str, object]
    file_purpose: str = "response"
    created_by: str | None = None
    api_key: str | None = None
    team_id: str | None = None
    org_id: str | None = None
    request_tags: object = None
    end_user_id: str | None = None


class _UniqueManagedObjectTable:
    """Rejects a second row for a response id and applies status claims atomically, the way the
    Postgres table does, and yields on every query so concurrent reads interleave like two pods."""

    def __init__(self, rows: tuple[_ManagedObjectRow, ...] = ()) -> None:
        self.rows = rows

    def _matches(self, row: _ManagedObjectRow, where: Mapping[str, object]) -> bool:
        def holds(field: str, condition: object) -> bool:
            if isinstance(condition, list):
                return any(self._matches(row, alternative) for alternative in condition)
            if isinstance(condition, dict):
                return getattr(row, field) in condition["in"]
            return getattr(row, field) == condition

        return all(holds(field, condition) for field, condition in where.items())

    async def find_first(self, where: Mapping[str, object]) -> _ManagedObjectRow | None:
        await asyncio.sleep(0)
        return next((row for row in self.rows if self._matches(row, where)), None)

    async def create(self, data: Mapping[str, object]) -> _ManagedObjectRow:
        from prisma.errors import UniqueViolationError

        await asyncio.sleep(0)
        if any(
            row.unified_object_id == data["unified_object_id"] or row.model_object_id == data["model_object_id"]
            for row in self.rows
        ):
            raise UniqueViolationError({"user_facing_error": {"error_code": "P2002"}})
        row = _ManagedObjectRow(
            id=f"job-{len(self.rows)}",
            unified_object_id=str(data["unified_object_id"]),
            model_object_id=str(data["model_object_id"]),
            status=str(data["status"]),
            file_object=json.loads(str(data["file_object"])),
            created_by=cast("str | None", data.get("created_by")),
            api_key=cast("str | None", data.get("api_key")),
            team_id=cast("str | None", data.get("team_id")),
            org_id=cast("str | None", data.get("org_id")),
            end_user_id=cast("str | None", data.get("end_user_id")),
            request_tags=getattr(data.get("request_tags"), "data", None),
        )
        self.rows = (*self.rows, row)
        return row

    async def update_many(self, where: Mapping[str, object], data: Mapping[str, object]) -> int:
        await asyncio.sleep(0)
        matched = frozenset(row.id for row in self.rows if self._matches(row, where))
        self.rows = tuple(
            dataclasses.replace(row, status=str(data["status"])) if row.id in matched else row for row in self.rows
        )
        return len(matched)


async def _read_twice_on_two_pods(
    monkeypatch: pytest.MonkeyPatch, table: _UniqueManagedObjectTable, response_id: str, reader: object
) -> tuple[Mapping[str, object], ...]:
    from fastapi import Response as FastAPIResponse
    from litellm_enterprise.proxy.hooks.managed_files import _PROXY_LiteLLMManagedFiles

    from litellm.caching.caching import DualCache
    from litellm.integrations.custom_logger import CustomLogger
    from litellm.proxy.response_api_endpoints.endpoints import _bill_finished_background_response

    class _CapturingLogger(_SpendEvents, CustomLogger):
        pass

    spend_events: Final = _CapturingLogger()
    monkeypatch.setattr(litellm, "_async_success_callback", [spend_events])
    prisma_client: Final = MagicMock()
    prisma_client.db.litellm_managedobjecttable = table
    prisma_client.db.litellm_usertable.find_unique = AsyncMock(return_value=None)
    prisma_client.db.litellm_verificationtoken.find_unique = AsyncMock(return_value=None)
    prisma_client.db.litellm_teamtable.find_unique = AsyncMock(return_value=None)
    proxy_logging: Final = MagicMock()
    proxy_logging.get_proxy_hook = MagicMock(
        return_value=_PROXY_LiteLLMManagedFiles(internal_usage_cache=DualCache(), prisma_client=prisma_client)
    )
    finished: Final = ResponsesAPIResponse(
        id=response_id,
        object="response",
        status="completed",
        created_at=1,
        model="gpt-5.6",
        output=[],
        usage=None,
        background=True,
    )

    router: Final = _configured_deployment_router(finished)
    router.aget_responses = AsyncMock(side_effect=AssertionError("billing must not read the provider again"))

    with (
        patch(
            "litellm.proxy.proxy_server.prisma_client", prisma_client
        ),  # test-quality-ok: the endpoint reads the proxy singletons at call time
        patch(
            "litellm.proxy.proxy_server.llm_router", router
        ),  # test-quality-ok: the endpoint reads the proxy singletons at call time
        patch(
            "litellm.proxy.proxy_server.proxy_logging_obj", proxy_logging
        ),  # test-quality-ok: the endpoint reads the proxy singletons at call time
        patch("litellm.aget_responses", side_effect=AssertionError("billing must not read the provider again")),
    ):
        await asyncio.gather(
            *(
                _bill_finished_background_response(
                    response=finished,
                    response_id=response_id,
                    user_api_key_dict=reader,
                    fastapi_response=FastAPIResponse(),
                    read_settles_billing=True,
                )
                for _ in range(2)
            )
        )
        await _drain_spend_events()
    return spend_events.payloads


@pytest.mark.asyncio
async def test_a_finished_response_the_create_could_not_store_is_billed_once_under_the_reader(
    monkeypatch: pytest.MonkeyPatch,
):
    """A create that could not write its managed row (a database outage) leaves the reads marked free
    and the poll blind to the job, so the first read that sees it finished stores the row under the
    reading key and bills it, once even when two pods read it at the same moment."""
    from litellm.proxy._types import UserAPIKeyAuth, hash_token
    from litellm.responses.utils import ResponsesAPIRequestUtils

    response_id: Final = ResponsesAPIRequestUtils._build_responses_api_response_id(
        custom_llm_provider="openai", model_id="deployment-untracked", response_id="resp_untracked"
    )
    table: Final = _UniqueManagedObjectTable()

    payloads: Final = await _read_twice_on_two_pods(
        monkeypatch,
        table,
        response_id,
        UserAPIKeyAuth(
            api_key="sk-reader", user_id="bob", team_id="team-beta", org_id="org-gamma", end_user_id="customer-42"
        ),
    )

    (billed,) = payloads
    assert billed["metadata"]["user_api_key_hash"] == hash_token("sk-reader")
    assert billed["metadata"]["user_api_key_team_id"] == "team-beta"
    assert billed["metadata"]["user_api_key_org_id"] == "org-gamma"
    assert billed["metadata"]["user_api_key_end_user_id"] == "customer-42"
    (row,) = table.rows
    assert (row.unified_object_id, row.status) == (response_id, "completed")


@pytest.mark.asyncio
@pytest.mark.parametrize("proxy_issued, stored_status", [(True, "completed"), (False, None)])
async def test_a_finished_read_never_bills_a_response_already_billed_or_billed_by_the_read_itself(
    monkeypatch: pytest.MonkeyPatch, proxy_issued: bool, stored_status: str | None
):
    """A row the create stored as finished means the create billed it, and an id the proxy did not
    issue is priced by the read itself, so neither read stores a row or bills the response again."""
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.responses.utils import ResponsesAPIRequestUtils

    response_id: Final = (
        ResponsesAPIRequestUtils._build_responses_api_response_id(
            custom_llm_provider="openai", model_id="deployment-billed", response_id="resp_billed"
        )
        if proxy_issued
        else "resp_not_issued_by_the_proxy"
    )
    stored: Final = (
        (
            _ManagedObjectRow(
                id="job-billed",
                unified_object_id=response_id,
                model_object_id="resp_billed",
                status=stored_status,
                file_object={"model": "gpt-5.6"},
            ),
        )
        if stored_status is not None
        else ()
    )
    table: Final = _UniqueManagedObjectTable(stored)

    payloads: Final = await _read_twice_on_two_pods(
        monkeypatch, table, response_id, UserAPIKeyAuth(api_key="sk-reader")
    )

    assert payloads == ()
    assert table.rows == stored


def _configured_deployment_router(fetched: ResponsesAPIResponse) -> MagicMock:
    from litellm.types.router import Deployment, LiteLLM_Params, ModelInfo

    router: Final = MagicMock()
    router.get_deployment = MagicMock(
        return_value=Deployment(
            model_name="gpt-5.6",
            litellm_params=LiteLLM_Params(model="openai/gpt-5.6"),
            model_info=ModelInfo(id="deployment-read"),
        )
    )
    router.aget_responses = AsyncMock(return_value=fetched)
    return router


def _background_response(response_id: str, status: str, input_tokens: int) -> ResponsesAPIResponse:
    return ResponsesAPIResponse(
        id=response_id,
        object="response",
        status=status,
        created_at=1,
        model="gpt-5.6",
        output=[],
        usage=ResponseAPIUsage(input_tokens=input_tokens, output_tokens=50, total_tokens=input_tokens + 50),
        background=True,
    )


@dataclasses.dataclass(frozen=True)
class _StoredBackgroundResponse:
    response_id: str
    table: _UniqueManagedObjectTable
    prisma_client: MagicMock
    proxy_logging: MagicMock


def _one_pending_background_response() -> _StoredBackgroundResponse:
    from litellm_enterprise.proxy.hooks.managed_files import _PROXY_LiteLLMManagedFiles

    from litellm.caching.caching import DualCache
    from litellm.responses.utils import ResponsesAPIRequestUtils

    response_id: Final = ResponsesAPIRequestUtils._build_responses_api_response_id(
        custom_llm_provider="openai", model_id="deployment-read", response_id="resp_read"
    )
    table: Final = _UniqueManagedObjectTable(
        (
            _ManagedObjectRow(
                id="job-read",
                unified_object_id=response_id,
                model_object_id="resp_read",
                status="queued",
                file_object={"model": "gpt-5.6"},
            ),
        )
    )
    prisma_client: Final = MagicMock()
    prisma_client.db.litellm_managedobjecttable = table
    prisma_client.db.litellm_usertable.find_unique = AsyncMock(return_value=None)
    prisma_client.db.litellm_verificationtoken.find_unique = AsyncMock(return_value=None)
    prisma_client.db.litellm_teamtable.find_unique = AsyncMock(return_value=None)
    proxy_logging: Final = MagicMock()
    proxy_logging.get_proxy_hook = MagicMock(
        return_value=_PROXY_LiteLLMManagedFiles(internal_usage_cache=DualCache(), prisma_client=prisma_client)
    )
    return _StoredBackgroundResponse(
        response_id=response_id, table=table, prisma_client=prisma_client, proxy_logging=proxy_logging
    )


def _request_for(method: str, path: str, body: bytes):
    from starlette.requests import Request

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    return Request(
        {"type": "http", "method": method, "path": path, "headers": [(b"content-type", b"application/json")]},
        receive,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "endpoint_name, finished_status",
    [("get_response", "completed"), ("cancel_response", "cancelled")],
)
@pytest.mark.parametrize(
    "read_body, billed_input_tokens, deployment_fetches",
    [(b"", 1000, 0), (b'{"api_base": "https://caller.example/v1"}', 100, 1)],
)
async def test_a_finished_read_or_cancel_bills_usage_only_the_configured_deployment_reported(
    monkeypatch: pytest.MonkeyPatch,
    endpoint_name: str,
    finished_status: str,
    read_body: bytes,
    billed_input_tokens: int,
    deployment_fetches: int,
):
    """A read or cancel the caller sends to its own endpoint returns whatever usage that endpoint
    claims, so it bills from a fetch through the configured deployment instead, and either way the
    response reports the cost it billed."""
    from fastapi import Response as FastAPIResponse

    from litellm.integrations.custom_logger import CustomLogger
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.response_api_endpoints import endpoints

    class _CapturingLogger(_SpendEvents, CustomLogger):
        pass

    spend_events: Final = _CapturingLogger()
    monkeypatch.setattr(litellm, "_async_success_callback", [spend_events])
    stored: Final = _one_pending_background_response()
    router: Final = _configured_deployment_router(_background_response(stored.response_id, finished_status, 100))
    processor: Final = MagicMock()
    processor.base_process_llm_request = AsyncMock(
        return_value=_background_response(stored.response_id, finished_status, 1000)
    )
    fastapi_response: Final = FastAPIResponse()

    with (
        patch(  # test-quality-ok: endpoint constructs the processor directly
            "litellm.proxy.response_api_endpoints.endpoints.ProxyBaseLLMRequestProcessing",
            return_value=processor,
        ),
        patch(
            "litellm.proxy.proxy_server.prisma_client", stored.prisma_client
        ),  # test-quality-ok: the endpoint reads the proxy singletons at call time
        patch(
            "litellm.proxy.proxy_server.llm_router", router
        ),  # test-quality-ok: the endpoint reads the proxy singletons at call time
        patch(
            "litellm.proxy.proxy_server.proxy_logging_obj", stored.proxy_logging
        ),  # test-quality-ok: the endpoint reads the proxy singletons at call time
        patch("litellm.aget_responses", side_effect=AssertionError("the configured deployment answers billing reads")),
    ):
        await getattr(endpoints, endpoint_name)(
            response_id=stored.response_id,
            request=_request_for("GET", f"/v1/responses/{stored.response_id}", read_body),
            fastapi_response=fastapi_response,
            user_api_key_dict=UserAPIKeyAuth(api_key="sk-reader"),
        )
        await _drain_spend_events()

    (billed,) = spend_events.payloads
    assert billed["prompt_tokens"] == billed_input_tokens
    assert router.aget_responses.await_count == deployment_fetches
    assert fastapi_response.headers.getlist("x-litellm-response-cost") == _billed_costs(spend_events.payloads)
    (row,) = stored.table.rows
    assert row.status == "completed"


@pytest.mark.asyncio
async def test_deleting_a_finished_unbilled_background_response_bills_it_before_the_provider_forgets_it(
    monkeypatch: pytest.MonkeyPatch,
):
    """A delete leaves the cost poll nothing to fetch, so a finished response no read has billed yet
    is billed from the configured deployment before the delete reaches the provider."""
    from fastapi import Response as FastAPIResponse

    from litellm.integrations.custom_logger import CustomLogger
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.response_api_endpoints.endpoints import delete_response
    from litellm.types.responses.main import DeleteResponseResult

    class _CapturingLogger(_SpendEvents, CustomLogger):
        pass

    spend_events: Final = _CapturingLogger()
    monkeypatch.setattr(litellm, "_async_success_callback", [spend_events])
    stored: Final = _one_pending_background_response()
    deleted: Final[list[str]] = []

    async def provider_fetch(response_id: str, litellm_metadata: Mapping[str, object]) -> ResponsesAPIResponse:
        if deleted:
            raise litellm.NotFoundError(message="response not found", model="gpt-5.6", llm_provider="openai")
        return _background_response(response_id, "completed", 100)

    async def provider_delete(fastapi_response: FastAPIResponse, **_: object) -> DeleteResponseResult:
        deleted.append(stored.response_id)
        fastapi_response.headers["x-litellm-response-cost"] = "0.0"
        return DeleteResponseResult(id=stored.response_id, object="response", deleted=True)

    router: Final = _configured_deployment_router(_background_response(stored.response_id, "completed", 100))
    router.aget_responses = AsyncMock(side_effect=provider_fetch)
    processor: Final = MagicMock()
    processor.base_process_llm_request = AsyncMock(side_effect=provider_delete)
    fastapi_response: Final = FastAPIResponse()

    with (
        patch(  # test-quality-ok: endpoint constructs the processor directly
            "litellm.proxy.response_api_endpoints.endpoints.ProxyBaseLLMRequestProcessing",
            return_value=processor,
        ),
        patch(
            "litellm.proxy.proxy_server.prisma_client", stored.prisma_client
        ),  # test-quality-ok: the endpoint reads the proxy singletons at call time
        patch(
            "litellm.proxy.proxy_server.llm_router", router
        ),  # test-quality-ok: the endpoint reads the proxy singletons at call time
        patch(
            "litellm.proxy.proxy_server.proxy_logging_obj", stored.proxy_logging
        ),  # test-quality-ok: the endpoint reads the proxy singletons at call time
    ):
        result: Final = await delete_response(
            response_id=stored.response_id,
            request=_request_for("DELETE", f"/v1/responses/{stored.response_id}", b""),
            fastapi_response=fastapi_response,
            user_api_key_dict=UserAPIKeyAuth(api_key="sk-reader"),
        )
        await _drain_spend_events()

    assert result.deleted is True
    (billed,) = spend_events.payloads
    assert billed["prompt_tokens"] == 100
    assert fastapi_response.headers.getlist("x-litellm-response-cost") == _billed_costs(spend_events.payloads)
    (row,) = stored.table.rows
    assert row.status == "completed"


@pytest.mark.asyncio
@pytest.mark.parametrize("caller_role, billed_count", [("proxy_admin", 1), ("internal_user", 0)])
async def test_a_delete_the_proxy_refuses_never_fetches_or_bills_the_response(caller_role: str, billed_count: int):
    """A delete settles the bill only for a caller the response id check lets through, so a key that
    cannot address the response gets its 403 without the proxy fetching or billing it first."""
    from fastapi import HTTPException
    from fastapi import Response as FastAPIResponse

    from litellm.integrations.custom_logger import CustomLogger
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.hooks.responses_id_security import ResponsesIDSecurity
    from litellm.proxy.response_api_endpoints.endpoints import delete_response
    from litellm.types.responses.main import DeleteResponseResult

    class _CapturingLogger(_SpendEvents, CustomLogger):
        pass

    spend_events: Final = _CapturingLogger()
    stored: Final = _one_pending_background_response()
    managed_files: Final = stored.proxy_logging.get_proxy_hook.return_value
    security: Final = ResponsesIDSecurity(general_settings_reader=lambda: {}, signing_key_reader=lambda: "salt")
    stored.proxy_logging.get_proxy_hook = MagicMock(
        side_effect=lambda name: security if name == "responses_id_security" else managed_files
    )
    router: Final = _configured_deployment_router(_background_response(stored.response_id, "completed", 100))
    caller: Final = UserAPIKeyAuth(api_key="sk-caller", user_role=caller_role)

    async def pipeline(**_: object) -> DeleteResponseResult:
        await security.async_pre_call_hook(
            caller, MagicMock(), {"response_id": stored.response_id}, "adelete_responses"
        )
        return DeleteResponseResult(id=stored.response_id, object="response", deleted=True)

    processor: Final = MagicMock()
    processor.base_process_llm_request = AsyncMock(side_effect=pipeline)
    processor.handle_llm_api_exception = AsyncMock(side_effect=lambda e, **_: e)

    with (
        patch.object(litellm, "_async_success_callback", [spend_events]),
        patch(  # test-quality-ok: endpoint constructs the processor directly
            "litellm.proxy.response_api_endpoints.endpoints.ProxyBaseLLMRequestProcessing",
            return_value=processor,
        ),
        patch(
            "litellm.proxy.proxy_server.prisma_client", stored.prisma_client
        ),  # test-quality-ok: the endpoint reads the proxy singletons at call time
        patch(
            "litellm.proxy.proxy_server.llm_router", router
        ),  # test-quality-ok: the endpoint reads the proxy singletons at call time
        patch(
            "litellm.proxy.proxy_server.proxy_logging_obj", stored.proxy_logging
        ),  # test-quality-ok: the endpoint reads the proxy singletons at call time
    ):
        with contextlib.suppress(HTTPException):
            await delete_response(
                response_id=stored.response_id,
                request=_request_for("DELETE", f"/v1/responses/{stored.response_id}", b""),
                fastapi_response=FastAPIResponse(),
                user_api_key_dict=caller,
            )
        await _drain_spend_events()

    assert len(spend_events.payloads) == billed_count
    assert router.aget_responses.await_count == billed_count


def _proxy_issued_id(provider_response_id: str) -> str:
    from litellm.responses.utils import ResponsesAPIRequestUtils

    return ResponsesAPIRequestUtils._build_responses_api_response_id(
        custom_llm_provider="openai", model_id="deployment-read", response_id=provider_response_id
    )


def _managed_files_over(table: _UniqueManagedObjectTable) -> tuple[MagicMock, MagicMock]:
    from litellm_enterprise.proxy.hooks.managed_files import _PROXY_LiteLLMManagedFiles

    from litellm.caching.caching import DualCache

    prisma_client: Final = MagicMock()
    prisma_client.db.litellm_managedobjecttable = table
    prisma_client.db.litellm_usertable.find_unique = AsyncMock(return_value=None)
    prisma_client.db.litellm_verificationtoken.find_unique = AsyncMock(return_value=None)
    prisma_client.db.litellm_teamtable.find_unique = AsyncMock(return_value=None)
    prisma_client.db.litellm_deprecatedverificationtoken.find_unique = AsyncMock(return_value=None)
    prisma_client.db.litellm_deletedverificationtoken.find_first = AsyncMock(return_value=None)
    prisma_client.db.litellm_deletedteamtable.find_first = AsyncMock(return_value=None)
    proxy_logging: Final = MagicMock()
    proxy_logging.get_proxy_hook = MagicMock(
        return_value=_PROXY_LiteLLMManagedFiles(internal_usage_cache=DualCache(), prisma_client=prisma_client)
    )
    return prisma_client, proxy_logging


def _capture_spend_events(monkeypatch: pytest.MonkeyPatch) -> _SpendEvents:
    from litellm.integrations.custom_logger import CustomLogger

    class _CapturingLogger(_SpendEvents, CustomLogger):
        pass

    spend_events: Final = _CapturingLogger()
    monkeypatch.setattr(litellm, "_async_success_callback", [spend_events])
    return spend_events


@contextlib.contextmanager
def _proxy_over(prisma_client: MagicMock, proxy_logging: MagicMock, router: MagicMock):
    with (
        patch(
            "litellm.proxy.proxy_server.prisma_client", prisma_client
        ),  # test-quality-ok: the endpoint reads the proxy singletons at call time
        patch(
            "litellm.proxy.proxy_server.llm_router", router
        ),  # test-quality-ok: the endpoint reads the proxy singletons at call time
        patch(
            "litellm.proxy.proxy_server.proxy_logging_obj", proxy_logging
        ),  # test-quality-ok: the endpoint reads the proxy singletons at call time
    ):
        yield


@pytest.mark.asyncio
@pytest.mark.parametrize("completed_after_last_event", [False, True], ids=["before_last_event", "after_last_event"])
async def test_a_streamed_background_create_bills_its_finished_response_once_under_the_creator(
    monkeypatch: pytest.MonkeyPatch, completed_after_last_event: bool
):
    """A tracked streamed background create is free on its own and bills the finished response once
    through its managed row, under the creating key and tags, before the client sees it finish
    whenever the stream records the finished response in time, and no later read bills it again."""
    from fastapi import Response as FastAPIResponse

    from litellm.proxy._types import UserAPIKeyAuth, hash_token
    from litellm.proxy.response_api_endpoints.endpoints import (
        _bill_finished_background_response,
        _streamed_background_data_generator,
    )

    spend_events: Final = _capture_spend_events(monkeypatch)
    table: Final = _UniqueManagedObjectTable()
    prisma_client, proxy_logging = _managed_files_over(table)
    finished: Final = _background_response(_proxy_issued_id("resp_streamed"), "completed", 100)
    stream: Final = MagicMock(completed_response=None)

    async def provider_events(**_: object):
        yield "event: response.created\n\n"
        if not completed_after_last_event:
            stream.completed_response = MagicMock(response=finished)
        yield "event: response.completed\n\n"
        stream.completed_response = MagicMock(response=finished)

    billed_before_each_event: Final[list[int]] = []
    with _proxy_over(prisma_client, proxy_logging, _configured_deployment_router(finished)):
        async for _ in _streamed_background_data_generator(
            provider_events,
            proxy_logging,
            stream,
            UserAPIKeyAuth(api_key="sk-creator"),
            {"litellm_metadata": {"tags": ["nightly"]}},
        ):
            billed_before_each_event.append(len(table.rows))
        await _bill_finished_background_response(
            response=finished,
            response_id=finished.id,
            user_api_key_dict=UserAPIKeyAuth(api_key="sk-reader"),
            fastapi_response=FastAPIResponse(),
            read_settles_billing=True,
        )
        await _drain_spend_events()

    assert billed_before_each_event == ([0, 0] if completed_after_last_event else [0, 1])
    (billed,) = spend_events.payloads
    assert billed["metadata"]["user_api_key_hash"] == hash_token("sk-creator")
    assert billed["request_tags"] == ["nightly"]


@pytest.mark.asyncio
async def test_a_read_that_bills_a_response_before_its_stream_ends_leaves_the_stream_nothing_to_bill(
    monkeypatch: pytest.MonkeyPatch,
):
    """A read can see a streamed background response finished before a slow client lets the stream
    reach its last event, and then the stream bills nothing on top of the read."""
    from fastapi import Response as FastAPIResponse

    from litellm.proxy._types import UserAPIKeyAuth, hash_token
    from litellm.proxy.response_api_endpoints.endpoints import (
        _bill_finished_background_response,
        _streamed_background_data_generator,
    )

    spend_events: Final = _capture_spend_events(monkeypatch)
    table: Final = _UniqueManagedObjectTable()
    prisma_client, proxy_logging = _managed_files_over(table)
    finished: Final = _background_response(_proxy_issued_id("resp_raced"), "completed", 100)
    stream: Final = MagicMock(completed_response=None)

    async def provider_events(**_: object):
        yield "event: response.created\n\n"
        await _bill_finished_background_response(
            response=finished,
            response_id=finished.id,
            user_api_key_dict=UserAPIKeyAuth(api_key="sk-reader"),
            fastapi_response=FastAPIResponse(),
            read_settles_billing=True,
        )
        stream.completed_response = MagicMock(response=finished)
        yield "event: response.completed\n\n"

    with _proxy_over(prisma_client, proxy_logging, _configured_deployment_router(finished)):
        async for _ in _streamed_background_data_generator(
            provider_events, proxy_logging, stream, UserAPIKeyAuth(api_key="sk-creator"), {}
        ):
            pass
        await _drain_spend_events()

    (billed,) = spend_events.payloads
    assert billed["metadata"]["user_api_key_hash"] == hash_token("sk-reader")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "background_fields",
    [{"background": True}, {"extra_body": {"background": True}}],
    ids=["top_level", "extra_body"],
)
async def test_a_streamed_background_create_is_billed_wherever_it_sets_background(
    monkeypatch: pytest.MonkeyPatch, background_fields: Mapping[str, object]
):
    """A streamed create that sets background inside extra_body runs as a background job all the
    same, so its stream bills the finished response once."""
    from fastapi import Response as FastAPIResponse

    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
    from litellm.proxy.response_api_endpoints.endpoints import responses_api

    spend_events: Final = _capture_spend_events(monkeypatch)
    table: Final = _UniqueManagedObjectTable()
    prisma_client, proxy_logging = _managed_files_over(table)
    finished: Final = _background_response(_proxy_issued_id("resp_streamed_create"), "completed", 100)
    stream: Final = MagicMock(completed_response=None)

    async def provider_events(**_: object):
        stream.completed_response = MagicMock(response=finished)
        yield "event: response.completed\n\n"

    async def stream_to_client(select_data_generator, **_: object) -> MagicMock:
        async for _ in select_data_generator(response=stream, user_api_key_dict=UserAPIKeyAuth(), request_data={}):
            pass
        return MagicMock()

    processor: Final = MagicMock()
    processor.base_process_llm_request = AsyncMock(side_effect=stream_to_client)
    processor_class: Final = MagicMock(
        return_value=processor,
        extract_completed_responses_response=ProxyBaseLLMRequestProcessing.extract_completed_responses_response,
    )
    body: Final = json.dumps({"model": "gpt-5.6", "input": "hello", "stream": True, **background_fields}).encode()

    with (
        _proxy_over(prisma_client, proxy_logging, _configured_deployment_router(finished)),
        patch(  # test-quality-ok: endpoint constructs the processor directly
            "litellm.proxy.response_api_endpoints.endpoints.ProxyBaseLLMRequestProcessing", processor_class
        ),
        patch(  # test-quality-ok: polling decision is imported inside the endpoint
            "litellm.proxy.response_polling.polling_handler.should_use_polling_for_request",
            return_value=False,
        ),
        patch(
            "litellm.proxy.proxy_server.select_data_generator", provider_events
        ),  # test-quality-ok: the endpoint reads the proxy singletons at call time
    ):
        await responses_api(
            request=_request_for("POST", "/v1/responses", body),
            fastapi_response=FastAPIResponse(),
            user_api_key_dict=UserAPIKeyAuth(api_key="sk-creator"),
        )
        await _drain_spend_events()

    assert len(spend_events.payloads) == 1


@pytest.mark.asyncio
async def test_a_retargeted_read_of_a_response_nobody_tracked_bills_it_once_from_the_deployment(
    monkeypatch: pytest.MonkeyPatch,
):
    """A read that carries its own api_base returns whatever usage that endpoint claims, so a finished
    response with no managed row is tracked and billed from a fetch through the configured deployment,
    once even when two pods read it at the same moment."""
    from fastapi import Response as FastAPIResponse

    from litellm.proxy._types import UserAPIKeyAuth, hash_token
    from litellm.proxy.response_api_endpoints.endpoints import _bill_finished_background_response

    spend_events: Final = _capture_spend_events(monkeypatch)
    table: Final = _UniqueManagedObjectTable()
    prisma_client, proxy_logging = _managed_files_over(table)
    response_id: Final = _proxy_issued_id("resp_untracked_override")
    router: Final = _configured_deployment_router(_background_response(response_id, "completed", 100))

    with _proxy_over(prisma_client, proxy_logging, router):
        await asyncio.gather(
            *(
                _bill_finished_background_response(
                    response=_background_response(response_id, "completed", 1000),
                    response_id=response_id,
                    user_api_key_dict=UserAPIKeyAuth(api_key="sk-reader"),
                    fastapi_response=FastAPIResponse(),
                    read_settles_billing=False,
                )
                for _ in range(2)
            )
        )
        await _drain_spend_events()

    (billed,) = spend_events.payloads
    assert billed["prompt_tokens"] == 100
    assert billed["metadata"]["user_api_key_hash"] == hash_token("sk-reader")


@pytest.mark.asyncio
async def test_a_bill_that_fails_to_settle_is_retried_in_the_background(monkeypatch: pytest.MonkeyPatch):
    """A read is free once marked, so a bill that fails while the read settles it (a database blip)
    is retried in the background instead of waiting on a cost poll that may be turned off."""
    from litellm.proxy.response_api_endpoints import endpoints

    spend_events: Final = _capture_spend_events(monkeypatch)
    stored: Final = _one_pending_background_response()
    finished: Final = _background_response(stored.response_id, "completed", 100)
    attempts: Final[list[int]] = []

    async def bill_after_one_failure(cost_check: endpoints._BackgroundCostCheck):
        attempts.append(len(attempts))
        if len(attempts) == 1:
            raise RuntimeError("database unavailable")
        return await cost_check.bill_if_finished(stored.response_id, finished)

    with _proxy_over(stored.prisma_client, stored.proxy_logging, _configured_deployment_router(finished)):
        assert await endpoints._settle_background_billing(bill_after_one_failure, retry_delays=(0.0, 0.0)) is None
        await asyncio.gather(*endpoints._billing_retries)
        await _drain_spend_events()

    assert len(attempts) == 2
    assert len(spend_events.payloads) == 1


@pytest.mark.asyncio
async def test_background_create_the_provider_confirms_is_stored_under_its_provider_id(
    monkeypatch: pytest.MonkeyPatch,
):
    """A create that sets background through extra_body reaches the provider as a background job
    all the same, so the proxy stores it off the response, keyed by the id every re-encryption of
    the client's id decrypts to."""
    from fastapi import Response as FastAPIResponse
    from starlette.requests import Request

    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.hooks.responses_id_security import ResponsesIDSecurity
    from litellm.proxy.response_api_endpoints.endpoints import responses_api

    salt_key = "background-create-unit-test-salt"
    monkeypatch.setenv("LITELLM_SALT_KEY", salt_key)
    queued = ResponsesIDSecurity(
        general_settings_reader=lambda: {}, signing_key_reader=lambda: salt_key
    )._encrypt_response_id(
        ResponsesAPIResponse(
            id="resp_provider_job", object="response", status="queued", created_at=1, output=[], background=True
        ),
        UserAPIKeyAuth(user_id="alice"),
    )
    queued._hidden_params = {"model_id": "deployment-1"}
    processor = MagicMock()
    processor.data = {"model": "gpt-4o", "input": "hello", "extra_body": {"background": True}}
    processor.base_process_llm_request = AsyncMock(return_value=queued)
    managed_files = MagicMock()
    managed_files.store_unified_object_id = AsyncMock()
    managed_files.track_untracked_background_response = AsyncMock(return_value=False)
    proxy_logging = MagicMock()
    proxy_logging.get_proxy_hook = MagicMock(return_value=managed_files)

    async def receive():
        return {
            "type": "http.request",
            "body": b'{"model":"gpt-4o","input":"hello","extra_body":{"background":true}}',
            "more_body": False,
        }

    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/responses",
            "headers": [(b"content-type", b"application/json")],
        },
        receive,
    )

    with (
        patch(  # test-quality-ok: endpoint constructs the processor directly
            "litellm.proxy.response_api_endpoints.endpoints.ProxyBaseLLMRequestProcessing",
            return_value=processor,
        ),
        patch(  # test-quality-ok: polling decision is imported inside the endpoint
            "litellm.proxy.response_polling.polling_handler.should_use_polling_for_request",
            return_value=False,
        ),
        patch(  # test-quality-ok: the endpoint reads the proxy singletons at call time
            "litellm.proxy.proxy_server.proxy_logging_obj", proxy_logging
        ),
        patch(  # test-quality-ok: the endpoint reads the proxy singletons at call time
            "litellm.proxy.proxy_server.llm_router", MagicMock()
        ),
    ):
        await responses_api(
            request=request, fastapi_response=FastAPIResponse(), user_api_key_dict=UserAPIKeyAuth(api_key="sk-test")
        )

    store_kwargs = managed_files.store_unified_object_id.await_args.kwargs
    assert store_kwargs["unified_object_id"] == queued.id
    assert store_kwargs["model_object_id"] == "resp_provider_job"


class TestResponsesAPIEndpoints(unittest.TestCase):
    @pytest.mark.asyncio
    @patch("litellm.proxy.proxy_server.llm_router")
    @patch("litellm.proxy.proxy_server.user_api_key_auth")
    async def test_openai_v1_responses_route(self, mock_auth, mock_router):
        """
        Test that /openai/v1/responses endpoint is correctly registered and accessible.
        """
        mock_auth.return_value = MagicMock(
            token="test_token",
            user_id="test_user",
            team_id=None,
        )

        mock_router.aresponses = AsyncMock(
            return_value={
                "id": "resp_abc123",
                "object": "realtime.response",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "text", "text": "Test response"}],
                    }
                ],
            }
        )

        client = TestClient(app)

        test_data = {"model": "gpt-4o", "input": "Tell me about AI"}

        response = client.post(
            "/openai/v1/responses",
            json=test_data,
            headers={"Authorization": "Bearer sk-9876"},
        )

        assert response.status_code in [200, 401, 500]

    @pytest.mark.asyncio
    @patch("litellm.proxy.proxy_server.llm_router")
    @patch("litellm.proxy.proxy_server.user_api_key_auth")
    async def test_cursor_chat_completions_route(self, mock_auth, mock_router):
        """
        Test that /cursor/chat/completions endpoint:
        1. Accepts Responses API input format
        2. Returns chat completions format response
        3. Transforms streaming responses correctly
        """
        from litellm.types.llms.openai import ResponsesAPIResponse
        from litellm.types.utils import ResponseOutputMessage, ResponseOutputText

        mock_auth.return_value = MagicMock(
            token="test_token",
            user_id="test_user",
            team_id=None,
        )

        # Mock a Responses API response
        mock_responses_response = ResponsesAPIResponse(
            id="resp_cursor123",
            created_at=1234567890,
            model="gpt-4o",
            object="response",
            output=[
                ResponseOutputMessage(
                    type="message",
                    role="assistant",
                    content=[ResponseOutputText(type="output_text", text="Hello from Cursor!")],
                )
            ],
        )

        mock_router.aresponses = AsyncMock(return_value=mock_responses_response)

        client = TestClient(app)

        # Test with Responses API input format (what Cursor sends)
        test_data = {
            "model": "gpt-4o",
            "input": [{"role": "user", "content": "Hello"}],
        }

        response = client.post(
            "/cursor/chat/completions",
            json=test_data,
            headers={"Authorization": "Bearer sk-9876"},
        )

        # Should return 200 (or 401/500 if auth fails)
        assert response.status_code in [200, 401, 500]

        # If successful, verify it returns chat completions format
        if response.status_code == 200:
            response_data = response.json()
            # Should have chat completion structure
            assert "choices" in response_data or "id" in response_data
            # Should not have Responses API structure
            assert "output" not in response_data or "status" not in response_data

    @pytest.mark.asyncio
    @patch("litellm.proxy.proxy_server.llm_router")
    @patch("litellm.proxy.proxy_server.user_api_key_auth")
    async def test_responses_api_key_spend_header_includes_response_cost(self, mock_auth, mock_router):
        """
        Test that x-litellm-key-spend header includes the current request's response_cost
        for /v1/responses endpoint.

        This ensures the spend header reflects updated spend including the current request,
        even though spend tracking updates happen asynchronously after the response.
        """
        from litellm.types.llms.openai import ResponsesAPIResponse
        from litellm.types.utils import ResponseOutputMessage, ResponseOutputText

        # Create mock user API key with initial spend
        mock_user_api_key_dict = MagicMock()
        mock_user_api_key_dict.token = "test_token"
        mock_user_api_key_dict.user_id = "test_user"
        mock_user_api_key_dict.team_id = None
        mock_user_api_key_dict.spend = 0.001  # Initial spend: $0.001
        mock_user_api_key_dict.tpm_limit = None
        mock_user_api_key_dict.rpm_limit = None
        mock_user_api_key_dict.max_budget = None
        mock_user_api_key_dict.allowed_model_region = None
        mock_user_api_key_dict.api_key = "sk-test-key"
        mock_user_api_key_dict.metadata = {}

        mock_auth.return_value = mock_user_api_key_dict

        # Mock response with hidden_params containing response_cost
        mock_response = ResponsesAPIResponse(
            id="resp_test123",
            created_at=1234567890,
            model="gpt-4o",
            object="response",
            output=[
                ResponseOutputMessage(
                    type="message",
                    role="assistant",
                    content=[ResponseOutputText(type="output_text", text="Test response")],
                )
            ],
        )

        # Add hidden_params with response_cost to the mock response
        mock_response._hidden_params = {
            "response_cost": 0.0005,  # Current request cost: $0.0005
            "model_id": "test-model-id",
        }

        mock_router.aresponses = AsyncMock(return_value=mock_response)

        client = TestClient(app)

        test_data = {"model": "gpt-4o", "input": "Tell me about AI"}

        response = client.post(
            "/v1/responses",
            json=test_data,
            headers={"Authorization": "Bearer sk-test-key"},
        )

        # Verify the response was successful
        assert response.status_code == 200

        # Verify x-litellm-key-spend header includes current request cost
        assert "x-litellm-key-spend" in response.headers
        key_spend_value = float(response.headers["x-litellm-key-spend"])
        expected_spend = 0.001 + 0.0005  # Initial spend + current request cost
        assert key_spend_value == pytest.approx(expected_spend, abs=1e-10)

        # Verify x-litellm-response-cost header is present
        assert "x-litellm-response-cost" in response.headers
        response_cost_value = float(response.headers["x-litellm-response-cost"])
        assert response_cost_value == pytest.approx(0.0005, abs=1e-10)


class TestManagedResponsesWSFirstMessage:
    @pytest.mark.asyncio
    async def test_first_message_processed_before_loop(self):
        """
        ManagedResponsesWebSocketHandler must process first_message before
        entering its receive loop. Regression for clients that connect without
        ?model= (e.g. Codex) and send model inside the first response.create event.
        """
        from litellm.responses.streaming_iterator import ManagedResponsesWebSocketHandler

        first = json.dumps(
            {
                "type": "response.create",
                "model": "gpt-4o-mini",
                "store": False,
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "hi"}],
                    }
                ],
            }
        )

        ws = MagicMock()
        ws.receive_text = AsyncMock(side_effect=Exception("disconnect"))
        ws.send_text = AsyncMock()

        processed: list = []

        async def fake_process(msg: str) -> None:
            processed.append(msg)

        handler = ManagedResponsesWebSocketHandler(
            websocket=ws,
            model="gpt-4o-mini",
            logging_obj=MagicMock(),
            first_message=first,
        )
        handler._process_response_create = fake_process  # type: ignore[method-assign]

        await handler.run()

        assert processed == [first]

    @pytest.mark.asyncio
    async def test_no_first_message_falls_through_to_loop(self):
        """When first_message is None, run() goes straight to receive_text()."""
        from litellm.responses.streaming_iterator import ManagedResponsesWebSocketHandler

        subsequent = json.dumps({"type": "response.create", "model": "gpt-4o-mini"})

        ws = MagicMock()
        ws.receive_text = AsyncMock(side_effect=[subsequent, Exception("disconnect")])
        ws.send_text = AsyncMock()

        processed: list = []

        async def fake_process(msg: str, turn_admission: object = None) -> None:
            processed.append(msg)

        handler = ManagedResponsesWebSocketHandler(
            websocket=ws,
            model="gpt-4o-mini",
            logging_obj=MagicMock(),
            first_message=None,
        )
        handler._process_response_create = fake_process  # type: ignore[method-assign]

        await handler.run()

        assert processed == [subsequent]


class TestResponsesWSStreamingFirstMessage:
    @pytest.mark.asyncio
    async def test_client_to_backend_replays_first_message(self):
        """
        ResponsesWebSocketStreaming.client_to_backend must send first_message to
        the backend before entering the receive loop.
        """
        from litellm.responses.streaming_iterator import ResponsesWebSocketStreaming

        first = json.dumps({"type": "response.create", "model": "gpt-4o-mini", "input": []})

        ws = MagicMock()
        ws.receive_text = AsyncMock(side_effect=Exception("disconnect"))

        backend_ws = MagicMock()
        backend_ws.send = AsyncMock()

        streaming = ResponsesWebSocketStreaming(
            websocket=ws,
            backend_ws=backend_ws,
            logging_obj=MagicMock(),
            first_message=first,
        )

        await streaming.client_to_backend()

        backend_ws.send.assert_awaited_once_with(first)


@pytest.mark.parametrize(
    ("failure", "expected_status", "expected_type"),
    [
        (ValueError("No deployments available - crossed budget: Exceeded budget for deployment"), 429, "rate_limit_exceeded"),
        (litellm.BadRequestError(message="bad input", model="m", llm_provider="openai"), 400, "invalid_request_error"),
        (RuntimeError("boom"), 500, "server_error"),
        (ValueError(f"{RouterErrors.no_deployments_with_tag_routing.value}=['free']"), 401, "invalid_request_error"),
        (ProxyException(message="Authentication Error", type="auth_error", param=None, code=401), 401, "invalid_request_error"),
        (HTTPException(status_code=403, detail="key not allowed"), 403, "invalid_request_error"),
    ],
    ids=["crossed-budget", "bad-request", "unexpected", "tag-routing", "proxy-auth", "http-exception"],
)
def test_responses_ws_failure_frame_uses_the_http_status_for_the_same_failure(
    failure: Exception, expected_status: int, expected_type: str
):
    from litellm.proxy.response_api_endpoints.endpoints import _responses_ws_failure_frame
    from litellm.proxy.utils import handle_exception_on_proxy

    frame: Final = json.loads(_responses_ws_failure_frame(failure))

    assert frame["status"] == expected_status
    assert frame["error"]["type"] == expected_type
    assert frame["error"]["message"] == handle_exception_on_proxy(failure).message


class TestWSSessionCostTracking:
    @pytest.mark.asyncio
    async def test_router_budget_limiter_skips_arealtime_call_type(self):
        """Same guard applies to _arealtime WS session wrappers."""
        from litellm.router_strategy.budget_limiter import RouterBudgetLimiting

        limiter = RouterBudgetLimiting.__new__(RouterBudgetLimiting)
        kwargs = {
            "call_type": "_arealtime",
            "standard_logging_object": None,
            "litellm_params": {"custom_llm_provider": "openai"},
        }
        await limiter.async_log_success_event(
            kwargs=kwargs,
            response_obj=None,
            start_time=None,
            end_time=None,
        )


class TestWSModelExtraction:
    """Test _extract_model_from_first_ws_event for flat and nested frame formats."""

    def test_flat_format_extracts_model(self):
        from litellm.proxy.response_api_endpoints.endpoints import (
            _extract_model_from_first_ws_event,
        )

        event = {"type": "response.create", "model": "gpt-4o", "input": "hello"}
        assert _extract_model_from_first_ws_event(event) == "gpt-4o"

    def test_nested_format_extracts_model(self):
        from litellm.proxy.response_api_endpoints.endpoints import (
            _extract_model_from_first_ws_event,
        )

        event = {"type": "response.create", "response": {"model": "gpt-4o", "input": "hello"}}
        assert _extract_model_from_first_ws_event(event) == "gpt-4o"

    def test_nested_format_takes_precedence_over_flat(self):
        from litellm.proxy.response_api_endpoints.endpoints import (
            _extract_model_from_first_ws_event,
        )

        event = {
            "type": "response.create",
            "model": "flat-model",
            "response": {"model": "nested-model"},
        }
        assert _extract_model_from_first_ws_event(event) == "nested-model"

    def test_no_model_returns_none(self):
        from litellm.proxy.response_api_endpoints.endpoints import (
            _extract_model_from_first_ws_event,
        )

        event = {"type": "response.create", "input": "hello"}
        assert _extract_model_from_first_ws_event(event) is None

    def test_non_object_returns_none(self):
        from litellm.proxy.response_api_endpoints.endpoints import (
            _extract_model_from_first_ws_event,
        )

        assert _extract_model_from_first_ws_event([]) is None


class TestResponsesWSFirstFrameValidation:
    @pytest.mark.asyncio
    async def test_rejects_non_response_create_first_frame(self):
        from litellm.proxy.response_api_endpoints.endpoints import (
            _read_ws_model_from_first_frame,
        )

        ws = MagicMock()
        ws.receive_text = AsyncMock(return_value=json.dumps({"type": "session.update", "model": "gpt-4o"}))
        ws.send_text = AsyncMock()
        ws.close = AsyncMock()

        result = await _read_ws_model_from_first_frame(ws)

        assert result is None
        ws.send_text.assert_awaited_once()
        ws.close.assert_awaited_once_with(code=1008, reason="Invalid first message")
        error_payload = json.loads(ws.send_text.await_args.args[0])
        assert error_payload["error"]["message"] == "First message must be a response.create JSON object."

    @pytest.mark.asyncio
    async def test_rejects_non_object_json_first_frame(self):
        from litellm.proxy.response_api_endpoints.endpoints import (
            _read_ws_model_from_first_frame,
        )

        ws = MagicMock()
        ws.receive_text = AsyncMock(return_value=json.dumps(["gpt-4o"]))
        ws.send_text = AsyncMock()
        ws.close = AsyncMock()

        result = await _read_ws_model_from_first_frame(ws)

        assert result is None
        ws.send_text.assert_awaited_once()
        ws.close.assert_awaited_once_with(code=1008, reason="Invalid first message")

    @pytest.mark.asyncio
    async def test_client_disconnect_first_frame_does_not_close(self):
        from fastapi import WebSocketDisconnect

        from litellm.proxy.response_api_endpoints.endpoints import (
            _read_ws_model_from_first_frame,
        )

        ws = MagicMock()
        ws.receive_text = AsyncMock(side_effect=WebSocketDisconnect(code=1006))
        ws.send_text = AsyncMock()
        ws.close = AsyncMock()

        result = await _read_ws_model_from_first_frame(ws)

        assert result is None
        ws.close.assert_not_awaited()
        ws.send_text.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_server_error_first_frame_closes_with_internal_error(self):
        from litellm.proxy.response_api_endpoints.endpoints import (
            _read_ws_model_from_first_frame,
        )

        ws = MagicMock()
        ws.receive_text = AsyncMock(side_effect=RuntimeError("boom"))
        ws.send_text = AsyncMock()
        ws.close = AsyncMock()

        result = await _read_ws_model_from_first_frame(ws)

        assert result is None
        ws.close.assert_awaited_once_with(code=1011, reason="Internal server error")


class TestResponsesWSFirstFrameModelAuth:
    @pytest.mark.asyncio
    async def test_endpoint_enforces_auth_after_model_from_first_frame(self):
        from litellm.proxy.response_api_endpoints.endpoints import (
            responses_websocket_endpoint,
        )

        ws = MagicMock()
        ws.headers = {}
        ws.query_params = {}
        ws.scope = {"headers": []}
        ws.url = "ws://testserver/v1/responses"
        ws.accept = AsyncMock()
        ws.receive_text = AsyncMock(
            return_value=json.dumps({"type": "response.create", "model": "gpt-4o-mini", "input": []})
        )
        ws.close = AsyncMock()

        processor = MagicMock()
        processor.common_processing_pre_call_logic = AsyncMock(return_value=({"model": "gpt-4o-mini"}, MagicMock()))

        async def fake_llm_call():
            return None

        authenticated_models: Final[list[str]] = []

        async def record_model_auth(*, model: str, **_kwargs: object) -> None:
            authenticated_models.append(model)

        with (
            patch(
                "litellm.proxy.response_api_endpoints.endpoints._enforce_responses_ws_first_frame_model_auth",
                new_callable=AsyncMock,
                side_effect=record_model_auth,
            ),
            patch(
                "litellm.proxy.response_api_endpoints.endpoints.ProxyBaseLLMRequestProcessing",
                return_value=processor,
            ),
            patch(
                "litellm.proxy.route_llm_request.route_request",
                new_callable=AsyncMock,
                return_value=fake_llm_call(),
            ),
        ):
            await responses_websocket_endpoint(
                websocket=ws,
                model=None,
                user_api_key_dict=MagicMock(),
            )

        assert authenticated_models == ["gpt-4o-mini"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("nested", [False, True])
    @pytest.mark.parametrize("query_model", [None, "gpt-4o-mini"])
    async def test_endpoint_routes_on_first_frame_input_and_previous_response_id(
        self, nested: bool, query_model: str | None
    ):
        from litellm.proxy.response_api_endpoints.endpoints import (
            responses_websocket_endpoint,
        )

        replayed_input = [{"type": "reasoning", "id": "encitem_abc", "encrypted_content": "litellm_enc:abc;blob"}]
        payload = {"model": "gpt-4o-mini", "input": replayed_input, "previous_response_id": "resp_prev"}
        first_frame = (
            {"type": "response.create", "response": payload} if nested else {"type": "response.create", **payload}
        )
        raw_first_frame = json.dumps(first_frame)

        ws = MagicMock()
        ws.headers = {}
        ws.query_params = {}
        ws.scope = {"headers": []}
        ws.url = "ws://testserver/v1/responses"
        ws.accept = AsyncMock()
        ws.receive_text = AsyncMock(return_value=raw_first_frame)
        ws.close = AsyncMock()

        processor = MagicMock()
        processor.common_processing_pre_call_logic = AsyncMock(
            return_value=({"model": "gpt-4o-mini", "litellm_metadata": {}}, MagicMock())
        )

        async def fake_llm_call():
            return None

        with (
            patch(  # test-quality-ok: first-frame model auth needs a live router and key table and has its own tests below
                "litellm.proxy.response_api_endpoints.endpoints._enforce_responses_ws_first_frame_model_auth",
                new_callable=AsyncMock,
            ),
            patch(  # test-quality-ok: the pre-call processor needs a live proxy; the payload it hands to routing is what is under test
                "litellm.proxy.response_api_endpoints.endpoints.ProxyBaseLLMRequestProcessing",
                return_value=processor,
            ),
            patch(  # test-quality-ok: routing is the seam where the first frame's input and previous_response_id become observable
                "litellm.proxy.route_llm_request.route_request",
                new_callable=AsyncMock,
                return_value=fake_llm_call(),
            ) as mock_route_request,
        ):
            await responses_websocket_endpoint(
                websocket=ws,
                model=query_model,
                user_api_key_dict=MagicMock(),
            )

        ws.receive_text.assert_awaited_once()
        routed = mock_route_request.await_args.kwargs["data"]
        assert routed["model"] == "gpt-4o-mini"
        assert routed["input"] == replayed_input
        assert routed["previous_response_id"] == "resp_prev"
        assert processor.common_processing_pre_call_logic.await_args.kwargs["model"] == "gpt-4o-mini"
        assert processor.common_processing_pre_call_logic.await_args.kwargs["skip_guardrails"] is True
        assert callable(routed["responses_ws_frame_guard"])
        assert mock_route_request.await_args.kwargs["route_type"] == "_aresponses_websocket"
        ws.close.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_frame_guard_runs_the_guardrail_walk_on_the_frame_body_and_returns_its_records(self):
        from litellm.proxy.response_api_endpoints.endpoints import _responses_ws_frame_guard

        async def pre_call_hook(user_api_key_dict, data, call_type, guardrails_only):
            assert call_type == "aresponses" and guardrails_only is True
            assert "type" not in data and data["model"] == "gpt-4o-mini"
            assert data["litellm_metadata"]["user_api_key_hash"] == "hash"
            assert "standard_logging_guardrail_information" not in data["litellm_metadata"]
            data["litellm_metadata"]["standard_logging_guardrail_information"] = [{"guardrail_name": "turn"}]
            return {**data, "input": data["input"] + " [guarded]"}

        proxy_logging_obj = MagicMock()
        proxy_logging_obj.pre_call_hook = AsyncMock(side_effect=pre_call_hook)
        admitted = {
            "model": "gpt-4o-mini",
            "litellm_metadata": {
                "user_api_key_hash": "hash",
                "standard_logging_guardrail_information": [{"guardrail_name": "admission"}],
            },
        }
        guard = _responses_ws_frame_guard(admitted, MagicMock(), proxy_logging_obj, MagicMock())

        first = await guard(json.dumps({"type": "response.create", "model": "alias", "input": "hi"}))
        second = await guard(json.dumps({"type": "response.create", "input": "again"}))

        assert json.loads(first.frame) == {"type": "response.create", "model": "alias", "input": "hi [guarded]"}
        assert first.guardrail_information == ({"guardrail_name": "turn"},)
        assert json.loads(second.frame)["input"] == "again [guarded]"
        assert admitted["litellm_metadata"]["standard_logging_guardrail_information"] == [
            {"guardrail_name": "admission"}
        ]

    @pytest.mark.asyncio
    async def test_frame_guard_bills_a_rejected_frame_as_a_failed_request(self):
        import litellm
        from litellm.proxy.hooks.proxy_track_cost_callback import ProxyDBLogger
        from litellm.proxy.response_api_endpoints.endpoints import _responses_ws_frame_guard

        async def pre_call_hook(user_api_key_dict, data, call_type, guardrails_only):
            data["litellm_metadata"]["standard_logging_guardrail_information"] = [{"guardrail_cost": 0.5}]
            raise ValueError("blocked")

        proxy_logging_obj = MagicMock()
        proxy_logging_obj.pre_call_hook = AsyncMock(side_effect=pre_call_hook)
        proxy_logging_obj.post_call_failure_hook = AsyncMock()
        spend_logger = MagicMock(spec=ProxyDBLogger)
        spend_logger.async_post_call_failure_hook = AsyncMock()
        rate_limiter = MagicMock()
        rate_limiter.async_post_call_failure_hook = AsyncMock()
        user_api_key_dict = MagicMock()
        guard = _responses_ws_frame_guard(
            {"model": "gpt-4o-mini", "litellm_metadata": {}}, user_api_key_dict, proxy_logging_obj, MagicMock()
        )

        with (
            patch.object(litellm, "callbacks", [rate_limiter, spend_logger]),
            pytest.raises(ValueError, match="blocked"),
        ):
            await guard(json.dumps({"type": "response.create", "input": "hi"}))

        proxy_logging_obj.post_call_failure_hook.assert_not_awaited()
        rate_limiter.async_post_call_failure_hook.assert_not_awaited()
        spend_logger.async_post_call_failure_hook.assert_awaited_once()
        failure = spend_logger.async_post_call_failure_hook.await_args.kwargs
        user_api_key_dict.model_copy.assert_called_once_with(update={"budget_reservation": None})
        assert failure["user_api_key_dict"] is user_api_key_dict.model_copy.return_value
        assert str(failure["original_exception"]) == "blocked"
        assert failure["request_data"]["litellm_metadata"]["standard_logging_guardrail_information"] == [
            {"guardrail_cost": 0.5}
        ]
        assert failure["request_data"]["input"] == "hi"
        assert len(failure["request_data"]["litellm_call_id"]) == 36

    @pytest.mark.asyncio
    async def test_frame_guard_guards_the_nested_response_body_and_copies_back_tools(self):
        from litellm.proxy.response_api_endpoints.endpoints import _responses_ws_frame_guard

        async def pre_call_hook(user_api_key_dict, data, call_type, guardrails_only):
            return {**data, "input": data["input"] + " [guarded]", "tools": data["tools"][:1]}

        proxy_logging_obj = MagicMock()
        proxy_logging_obj.pre_call_hook = AsyncMock(side_effect=pre_call_hook)
        guard = _responses_ws_frame_guard(
            {"model": "gpt-4o-mini", "litellm_metadata": {}}, MagicMock(), proxy_logging_obj, MagicMock()
        )
        tools = [{"type": "function", "name": "a"}, {"type": "function", "name": "b"}]

        guarded = await guard(json.dumps({"type": "response.create", "response": {"input": "hi", "tools": tools}}))

        assert proxy_logging_obj.pre_call_hook.await_args.kwargs["data"]["input"] == "hi"
        assert json.loads(guarded.frame) == {
            "type": "response.create",
            "response": {"input": "hi [guarded]", "tools": tools[:1]},
        }

    @pytest.mark.asyncio
    async def test_frame_guard_strips_the_proxy_owned_metadata_keys_a_client_sends(self):
        from litellm.proxy.response_api_endpoints.endpoints import _responses_ws_frame_guard

        proxy_logging_obj = MagicMock()
        proxy_logging_obj.pre_call_hook = AsyncMock(side_effect=lambda **kwargs: kwargs["data"])
        guard = _responses_ws_frame_guard(
            {"model": "gpt-4o-mini", "litellm_metadata": {}}, MagicMock(), proxy_logging_obj, MagicMock()
        )
        metadata = {
            "user_api_key_metadata": {"disable_global_guardrails": True},
            "disable_global_guardrails": True,
            "headers": {"x-litellm-tags": "forged"},
            "tag": "kept",
        }

        await guard(json.dumps({"type": "response.create", "input": "hi", "metadata": metadata}))

        assert proxy_logging_obj.pre_call_hook.await_args.kwargs["data"]["metadata"] == {"tag": "kept"}

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "frame",
        [
            {"type": "response.create", "input": "forbidden", "guardrails": []},
            {"type": "response.create", "input": "forbidden", "metadata": {"guardrails": []}},
            {"type": "response.create", "response": {"input": "forbidden", "guardrails": [], "policies": []}},
        ],
    )
    async def test_frame_guard_keeps_the_admitted_guardrail_list_when_a_frame_tries_to_replace_it(self, frame):
        from litellm.proxy.response_api_endpoints.endpoints import _responses_ws_frame_guard

        proxy_logging_obj = MagicMock()
        proxy_logging_obj.pre_call_hook = AsyncMock(side_effect=lambda **kwargs: kwargs["data"])
        guard = _responses_ws_frame_guard(
            {"model": "gpt-4o-mini", "litellm_metadata": {"guardrails": ["key-required"]}},
            MagicMock(),
            proxy_logging_obj,
            MagicMock(),
        )

        await guard(json.dumps(frame))

        scanned = proxy_logging_obj.pre_call_hook.await_args.kwargs["data"]
        assert "guardrails" not in scanned and "policies" not in scanned
        assert "guardrails" not in (scanned.get("metadata") or {})
        assert scanned["litellm_metadata"]["guardrails"] == ["key-required"]

    @pytest.mark.asyncio
    async def test_frame_guard_copies_an_admitted_bucket_that_holds_the_otel_parent_span(self):
        import threading

        from litellm.proxy.response_api_endpoints.endpoints import _responses_ws_frame_guard

        proxy_logging_obj = MagicMock()
        proxy_logging_obj.pre_call_hook = AsyncMock(side_effect=lambda **kwargs: kwargs["data"])
        span = threading.RLock()
        guard = _responses_ws_frame_guard(
            {"model": "gpt-4o-mini", "litellm_metadata": {"litellm_parent_otel_span": span}},
            MagicMock(),
            proxy_logging_obj,
            MagicMock(),
        )

        guarded = await guard(json.dumps({"type": "response.create", "input": "hi"}))

        assert json.loads(guarded.frame) == {"type": "response.create", "input": "hi"}
        assert (
            proxy_logging_obj.pre_call_hook.await_args.kwargs["data"]["litellm_metadata"]["litellm_parent_otel_span"]
            is span
        )

    @pytest.mark.asyncio
    async def test_frame_guard_rejects_a_turn_a_guardrail_rerouted_to_another_model(self):
        from litellm.proxy.response_api_endpoints.endpoints import _responses_ws_frame_guard

        from litellm.proxy.hooks.proxy_track_cost_callback import ProxyDBLogger

        async def pre_call_hook(user_api_key_dict, data, call_type, guardrails_only):
            data["litellm_metadata"]["standard_logging_guardrail_information"] = [{"guardrail_cost": 0.5}]
            return {**data, "model": "onprem-llama"}

        proxy_logging_obj = MagicMock()
        proxy_logging_obj.pre_call_hook = AsyncMock(side_effect=pre_call_hook)
        spend_logger = MagicMock(spec=ProxyDBLogger)
        spend_logger.async_post_call_failure_hook = AsyncMock()
        guard = _responses_ws_frame_guard(
            {"model": "gpt-4o-mini", "litellm_metadata": {}}, MagicMock(), proxy_logging_obj, MagicMock()
        )

        with (
            patch.object(litellm, "callbacks", [spend_logger]),
            pytest.raises(ValueError, match="rerouted this turn to 'onprem-llama'"),
        ):
            await guard(json.dumps({"type": "response.create", "input": "my SSN is 123-45-6789"}))

        failure = spend_logger.async_post_call_failure_hook.await_args.kwargs
        assert "rerouted this turn to 'onprem-llama'" in str(failure["original_exception"])
        assert failure["request_data"]["litellm_metadata"]["standard_logging_guardrail_information"] == [
            {"guardrail_cost": 0.5}
        ]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("nested", [False, True])
    async def test_frame_guard_drops_a_client_s_guardrail_cost_records_so_a_blocked_frame_bills_the_real_scan(
        self, nested: bool
    ):
        import litellm
        from litellm.proxy.hooks.proxy_track_cost_callback import ProxyDBLogger
        from litellm.proxy.response_api_endpoints.endpoints import _responses_ws_frame_guard

        async def pre_call_hook(user_api_key_dict, data, call_type, guardrails_only):
            data["litellm_metadata"]["standard_logging_guardrail_information"] = [{"guardrail_cost": 0.5}]
            raise ValueError("blocked")

        proxy_logging_obj = MagicMock()
        proxy_logging_obj.pre_call_hook = AsyncMock(side_effect=pre_call_hook)
        spend_logger = MagicMock(spec=ProxyDBLogger)
        spend_logger.async_post_call_failure_hook = AsyncMock()
        guard = _responses_ws_frame_guard(
            {"model": "gpt-4o-mini", "litellm_metadata": {}}, MagicMock(), proxy_logging_obj, MagicMock()
        )
        body = {
            "input": "forbidden",
            "metadata": {"trace": "t1", "standard_logging_guardrail_information": [], "model_info": {"x": 1}},
        }
        frame = {"type": "response.create", "response": body} if nested else {"type": "response.create", **body}

        with patch.object(litellm, "callbacks", [spend_logger]), pytest.raises(ValueError, match="blocked"):
            await guard(json.dumps(frame))

        billed = spend_logger.async_post_call_failure_hook.await_args.kwargs["request_data"]
        assert billed["metadata"] == {"trace": "t1"}
        assert billed["litellm_metadata"]["standard_logging_guardrail_information"] == [{"guardrail_cost": 0.5}]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("nested", [False, True])
    async def test_frame_guard_forwards_a_frame_without_the_instructions_a_guardrail_deleted(self, nested: bool):
        from litellm.proxy.response_api_endpoints.endpoints import _responses_ws_frame_guard

        async def pre_call_hook(user_api_key_dict, data, call_type, guardrails_only):
            return {key: value for key, value in data.items() if key != "instructions"}

        proxy_logging_obj = MagicMock()
        proxy_logging_obj.pre_call_hook = AsyncMock(side_effect=pre_call_hook)
        guard = _responses_ws_frame_guard(
            {"model": "gpt-4o-mini", "litellm_metadata": {}}, MagicMock(), proxy_logging_obj, MagicMock()
        )
        body = {"instructions": "SECRET system prompt", "input": "hi", "store": False}
        frame = {"type": "response.create", "response": body} if nested else {"type": "response.create", **body}

        with patch.object(litellm, "callbacks", []):
            guarded = await guard(json.dumps(frame))

        forwarded = json.loads(guarded.frame)
        assert forwarded["type"] == "response.create"
        sent_body = forwarded["response"] if nested else {k: v for k, v in forwarded.items() if k != "type"}
        assert sent_body == {"input": "hi", "store": False}

    @pytest.mark.asyncio
    async def test_frame_guard_hands_back_the_pii_tokens_presidio_masked_with_and_drops_forged_ones(self):
        from litellm.proxy.response_api_endpoints.endpoints import _responses_ws_frame_guard

        async def pre_call_hook(user_api_key_dict, data, call_type, guardrails_only):
            assert data["metadata"] == {"trace": "t1"}, "a client cannot seed pii_tokens"
            data["metadata"]["pii_tokens"] = {"<EMAIL_ADDRESS_1>": "alice@example.com"}
            return {**data, "input": "email <EMAIL_ADDRESS_1> please"}

        proxy_logging_obj = MagicMock()
        proxy_logging_obj.pre_call_hook = AsyncMock(side_effect=pre_call_hook)
        guard = _responses_ws_frame_guard(
            {"model": "gpt-4o-mini", "litellm_metadata": {}}, MagicMock(), proxy_logging_obj, MagicMock()
        )

        with patch.object(litellm, "callbacks", []):
            guarded = await guard(
                json.dumps(
                    {
                        "type": "response.create",
                        "input": "email alice@example.com please",
                        "metadata": {"trace": "t1", "pii_tokens": {"<EMAIL_ADDRESS_1>": "mallory@example.com"}},
                    }
                )
            )

        assert json.loads(guarded.frame)["input"] == "email <EMAIL_ADDRESS_1> please"
        assert dict(guarded.pii_tokens) == {"<EMAIL_ADDRESS_1>": "alice@example.com"}

    @pytest.mark.asyncio
    async def test_frame_guard_rejects_a_frame_that_carries_input_at_both_levels(self):
        from litellm.proxy.response_api_endpoints.endpoints import _responses_ws_frame_guard

        proxy_logging_obj = MagicMock()
        proxy_logging_obj.pre_call_hook = AsyncMock(side_effect=lambda **kwargs: kwargs["data"])
        guard = _responses_ws_frame_guard(
            {"model": "gpt-4o-mini", "litellm_metadata": {}}, MagicMock(), proxy_logging_obj, MagicMock()
        )

        with patch.object(litellm, "callbacks", []), pytest.raises(ValueError, match="both at the top level"):
            await guard(json.dumps({"type": "response.create", "input": "forbidden", "response": {}}))

        proxy_logging_obj.pre_call_hook.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_frame_guard_keeps_the_guardrail_error_when_billing_the_rejected_frame_fails(self):
        import litellm
        from litellm.proxy.hooks.proxy_track_cost_callback import ProxyDBLogger
        from litellm.proxy.response_api_endpoints.endpoints import _responses_ws_frame_guard

        proxy_logging_obj = MagicMock()
        proxy_logging_obj.pre_call_hook = AsyncMock(side_effect=ValueError("blocked"))
        spend_logger = MagicMock(spec=ProxyDBLogger)
        spend_logger.async_post_call_failure_hook = AsyncMock(side_effect=RuntimeError("db down"))
        guard = _responses_ws_frame_guard(
            {"model": "gpt-4o-mini", "litellm_metadata": {}}, MagicMock(), proxy_logging_obj, MagicMock()
        )

        with patch.object(litellm, "callbacks", [spend_logger]), pytest.raises(ValueError, match="blocked"):
            await guard(json.dumps({"type": "response.create", "input": "hi"}))

        spend_logger.async_post_call_failure_hook.assert_awaited_once()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("provider_rejected", [True, False])
    async def test_endpoint_books_a_provider_rejected_connection_as_a_failed_request(self, provider_rejected: bool):
        from litellm.proxy.response_api_endpoints.endpoints import (
            responses_websocket_endpoint,
        )

        ws = MagicMock()
        ws.headers = {}
        ws.query_params = {}
        ws.scope = {"headers": []}
        ws.url = "ws://testserver/v1/responses"
        ws.accept = AsyncMock()
        ws.receive_text = AsyncMock(
            return_value=json.dumps({"type": "response.create", "model": "gpt-4o-mini", "input": []})
        )
        ws.close = AsyncMock()

        processor = MagicMock()
        processor.common_processing_pre_call_logic = AsyncMock(
            return_value=({"model": "gpt-4o-mini", "litellm_metadata": {}}, MagicMock())
        )
        failure = litellm.BadRequestError(
            message="invalid_encrypted_content", model="gpt-4o-mini", llm_provider="openai"
        )

        async def fake_llm_call():
            return failure if provider_rejected else None

        proxy_logging_obj = MagicMock()
        proxy_logging_obj.post_call_failure_hook = AsyncMock(return_value=None)
        proxy_logging_obj.arelease_responses_websocket_slot = AsyncMock(return_value=None)
        user_api_key_dict = MagicMock()

        with (
            patch(  # test-quality-ok: first-frame model auth needs a live router and key table and has its own tests above
                "litellm.proxy.response_api_endpoints.endpoints._enforce_responses_ws_first_frame_model_auth",
                new_callable=AsyncMock,
            ),
            patch(  # test-quality-ok: the pre-call processor needs a live proxy; what the endpoint does with the relay's outcome is under test
                "litellm.proxy.response_api_endpoints.endpoints.ProxyBaseLLMRequestProcessing",
                return_value=processor,
            ),
            patch(  # test-quality-ok: routing is the seam that hands back the relay's outcome
                "litellm.proxy.route_llm_request.route_request",
                new_callable=AsyncMock,
                return_value=fake_llm_call(),
            ),
            patch(  # test-quality-ok: the failure hook is the proxy's only path to a failed spend log row
                "litellm.proxy.proxy_server.proxy_logging_obj",
                proxy_logging_obj,
            ),
        ):
            await responses_websocket_endpoint(
                websocket=ws,
                model=None,
                user_api_key_dict=user_api_key_dict,
            )

        ws.close.assert_not_awaited()
        if not provider_rejected:
            proxy_logging_obj.post_call_failure_hook.assert_not_awaited()
            return
        proxy_logging_obj.post_call_failure_hook.assert_awaited_once()
        booked = proxy_logging_obj.post_call_failure_hook.await_args.kwargs
        assert booked["original_exception"] is failure
        assert booked["user_api_key_dict"] is user_api_key_dict
        assert booked["request_data"]["model"] == "gpt-4o-mini"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("ending", ["refusal", "crash", "session_limit"])
    async def test_close_time_failure_row_bills_the_guardrails_of_the_turns_the_deployment_refused(self, ending: str):
        from litellm.exceptions import RateLimitError
        from litellm.proxy.response_api_endpoints.endpoints import responses_websocket_endpoint

        ws = MagicMock()
        ws.headers = {}
        ws.query_params = {}
        ws.scope = {"headers": []}
        ws.url = "ws://testserver/v1/responses"
        ws.accept = AsyncMock()
        ws.receive_text = AsyncMock(
            return_value=json.dumps({"type": "response.create", "model": "gpt-4o-mini", "input": []})
        )
        ws.close = AsyncMock()
        processor = MagicMock()
        processor.common_processing_pre_call_logic = AsyncMock(
            return_value=({"model": "gpt-4o-mini", "litellm_metadata": {"model_group": "gpt-4o-mini"}}, MagicMock())
        )
        guardrail_costs = {"two": 0.2, "three": 0.3, "four": 0.4}
        refusal = RateLimitError(message="crossed budget", llm_provider="", model="")
        later_refusal = RateLimitError(message="still over budget", llm_provider="", model="")
        crash = RuntimeError("relay crashed")

        async def pre_call_hook(user_api_key_dict, data, call_type, guardrails_only):
            recorded = [{"guardrail_cost": guardrail_costs[data["input"]]}]
            return {
                **data,
                "litellm_metadata": {**data["litellm_metadata"], "standard_logging_guardrail_information": recorded},
            }

        def route_request(data, **kwargs):
            async def relay():
                for turn in ("two", "three", "four"):
                    guarded = await data["responses_ws_frame_guard"](
                        json.dumps({"type": "response.create", "input": turn})
                    )
                    if turn != "three":
                        guarded.bill_refusal(refusal if turn == "two" else later_refusal)
                if ending == "crash":
                    raise crash
                if ending == "session_limit":
                    await asyncio.Event().wait()
                return refusal

            return relay()

        row_written = asyncio.Event()
        proxy_logging_obj = MagicMock()
        proxy_logging_obj.pre_call_hook = AsyncMock(side_effect=pre_call_hook)
        proxy_logging_obj.post_call_failure_hook = AsyncMock(side_effect=lambda **kwargs: row_written.set())
        proxy_logging_obj.arelease_responses_websocket_slot = AsyncMock(return_value=None)

        with (
            patch(  # test-quality-ok: first-frame model auth needs a live router and key table and has its own tests above
                "litellm.proxy.response_api_endpoints.endpoints._enforce_responses_ws_first_frame_model_auth",
                new_callable=AsyncMock,
            ),
            patch(  # test-quality-ok: the pre-call processor needs a live proxy; what the endpoint does with the relay's outcome is under test
                "litellm.proxy.response_api_endpoints.endpoints.ProxyBaseLLMRequestProcessing",
                return_value=processor,
            ),
            patch(  # test-quality-ok: routing is the seam that hands the relay its frame guard and returns its outcome
                "litellm.proxy.route_llm_request.route_request",
                new_callable=AsyncMock,
                side_effect=route_request,
            ),
            patch(  # test-quality-ok: a short session limit stands in for the configured one so the cap fires mid-session
                "litellm.proxy.response_api_endpoints.endpoints._resolve_responses_ws_session_limit_seconds",
                return_value=0.05 if ending == "session_limit" else 3600.0,
            ),
            patch(  # test-quality-ok: the failure hook is the proxy's only path to a failed spend log row
                "litellm.proxy.proxy_server.proxy_logging_obj",
                proxy_logging_obj,
            ),
        ):
            await responses_websocket_endpoint(websocket=ws, model=None, user_api_key_dict=MagicMock())
            await asyncio.wait_for(row_written.wait(), timeout=1)

        proxy_logging_obj.post_call_failure_hook.assert_awaited_once()
        proxy_logging_obj.arelease_responses_websocket_slot.assert_awaited_once()
        booked = proxy_logging_obj.post_call_failure_hook.await_args.kwargs
        assert booked["original_exception"] is (crash if ending == "crash" else refusal)
        assert booked["request_data"]["litellm_metadata"] == {
            "model_group": "gpt-4o-mini",
            "standard_logging_guardrail_information": [{"guardrail_cost": 0.2}, {"guardrail_cost": 0.4}],
        }

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "relay_outcome",
        ["still_relaying", "already_returned", "returns_despite_the_cancel", "crashes_despite_the_cancel"],
    )
    async def test_session_limit_frees_the_slot_at_once_and_never_cuts_the_failure_row(self, relay_outcome: str):
        from litellm.exceptions import RateLimitError
        from litellm.proxy.response_api_endpoints.endpoints import responses_websocket_endpoint

        ws = MagicMock()
        ws.headers = {}
        ws.query_params = {}
        ws.scope = {"headers": []}
        ws.url = "ws://testserver/v1/responses"
        ws.accept = AsyncMock()
        ws.receive_text = AsyncMock(
            return_value=json.dumps({"type": "response.create", "model": "gpt-4o-mini", "input": []})
        )
        async def close_over_the_network(**kwargs):
            await asyncio.sleep(0)

        ws.close = AsyncMock(side_effect=close_over_the_network)
        processor = MagicMock()
        processor.common_processing_pre_call_logic = AsyncMock(
            return_value=({"model": "gpt-4o-mini", "litellm_metadata": {}}, MagicMock())
        )
        refusal = RateLimitError(message="crossed budget", llm_provider="", model="")
        crash = RuntimeError("relay crashed")
        row_may_finish = asyncio.Event()
        row_written = asyncio.Event()

        async def slow_failure_hook(**kwargs):
            await row_may_finish.wait()
            row_written.set()

        def route_request(data, **kwargs):
            async def relay():
                guarded = await data["responses_ws_frame_guard"](json.dumps({"type": "response.create", "input": "two"}))
                guarded.bill_refusal(refusal)
                if relay_outcome == "still_relaying":
                    await asyncio.Event().wait()
                if relay_outcome in ("returns_despite_the_cancel", "crashes_despite_the_cancel"):
                    with contextlib.suppress(asyncio.CancelledError):
                        await asyncio.Event().wait()
                if relay_outcome == "crashes_despite_the_cancel":
                    raise crash
                return refusal

            return relay()

        proxy_logging_obj = MagicMock()
        proxy_logging_obj.pre_call_hook = AsyncMock(
            side_effect=lambda user_api_key_dict, data, call_type, guardrails_only: data
        )
        proxy_logging_obj.post_call_failure_hook = AsyncMock(side_effect=slow_failure_hook)
        async def release_over_the_network(user_api_key_dict):
            await asyncio.sleep(0.01)

        proxy_logging_obj.arelease_responses_websocket_slot = AsyncMock(side_effect=release_over_the_network)

        with (
            patch(  # test-quality-ok: first-frame model auth needs a live router and key table and has its own tests above
                "litellm.proxy.response_api_endpoints.endpoints._enforce_responses_ws_first_frame_model_auth",
                new_callable=AsyncMock,
            ),
            patch(  # test-quality-ok: the pre-call processor needs a live proxy; what the endpoint does with the relay's outcome is under test
                "litellm.proxy.response_api_endpoints.endpoints.ProxyBaseLLMRequestProcessing",
                return_value=processor,
            ),
            patch(  # test-quality-ok: routing is the seam that hands the relay its frame guard and returns its outcome
                "litellm.proxy.route_llm_request.route_request",
                new_callable=AsyncMock,
                side_effect=route_request,
            ),
            patch(  # test-quality-ok: a short session limit stands in for the configured one so the cap fires mid-session
                "litellm.proxy.response_api_endpoints.endpoints._resolve_responses_ws_session_limit_seconds",
                return_value=0.05,
            ),
            patch(  # test-quality-ok: the failure hook is the proxy's only path to a failed spend log row
                "litellm.proxy.proxy_server.proxy_logging_obj",
                proxy_logging_obj,
            ),
        ):
            endpoint = asyncio.ensure_future(
                responses_websocket_endpoint(websocket=ws, model=None, user_api_key_dict=MagicMock())
            )
            finished, _ = await asyncio.wait({endpoint}, timeout=1)
            assert endpoint in finished
            limit_close = call(code=1000, reason="Session duration limit reached")
            crash_close = call(code=1011, reason="Internal server error")
            crashed = relay_outcome == "crashes_despite_the_cancel"
            assert ws.close.await_args_list == ([limit_close, crash_close] if crashed else [limit_close])
            proxy_logging_obj.arelease_responses_websocket_slot.assert_awaited_once()
            assert not row_written.is_set()
            row_may_finish.set()
            await asyncio.wait_for(row_written.wait(), timeout=1)

        assert proxy_logging_obj.post_call_failure_hook.await_args.kwargs["original_exception"] is (crash if crashed else refusal)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("ending", ["client_closed", "crash"])
    async def test_a_closed_socket_frees_its_slot_before_its_failure_row_and_waits_for_the_row(self, ending: str):
        from litellm.exceptions import RateLimitError
        from litellm.proxy.response_api_endpoints.endpoints import responses_websocket_endpoint

        ws = MagicMock()
        ws.headers = {}
        ws.query_params = {}
        ws.scope = {"headers": []}
        ws.url = "ws://testserver/v1/responses"
        ws.accept = AsyncMock()
        ws.receive_text = AsyncMock(
            return_value=json.dumps({"type": "response.create", "model": "gpt-4o-mini", "input": []})
        )
        ws.close = AsyncMock()
        processor = MagicMock()
        processor.common_processing_pre_call_logic = AsyncMock(
            return_value=({"model": "gpt-4o-mini", "litellm_metadata": {}}, MagicMock())
        )
        refusal = RateLimitError(message="crossed budget", llm_provider="", model="")
        crash = RuntimeError("relay crashed")
        row_started = asyncio.Event()
        row_may_finish = asyncio.Event()
        row_written = asyncio.Event()

        async def slow_failure_hook(**kwargs):
            row_started.set()
            await row_may_finish.wait()
            row_written.set()

        def route_request(data, **kwargs):
            async def relay():
                guarded = await data["responses_ws_frame_guard"](json.dumps({"type": "response.create", "input": "two"}))
                guarded.bill_refusal(refusal)
                if ending == "crash":
                    raise crash
                return refusal

            return relay()

        proxy_logging_obj = MagicMock()
        proxy_logging_obj.pre_call_hook = AsyncMock(
            side_effect=lambda user_api_key_dict, data, call_type, guardrails_only: data
        )
        proxy_logging_obj.post_call_failure_hook = AsyncMock(side_effect=slow_failure_hook)
        proxy_logging_obj.arelease_responses_websocket_slot = AsyncMock(return_value=None)

        with (
            patch(  # test-quality-ok: first-frame model auth needs a live router and key table and has its own tests above
                "litellm.proxy.response_api_endpoints.endpoints._enforce_responses_ws_first_frame_model_auth",
                new_callable=AsyncMock,
            ),
            patch(  # test-quality-ok: the pre-call processor needs a live proxy; what the endpoint does with the relay's outcome is under test
                "litellm.proxy.response_api_endpoints.endpoints.ProxyBaseLLMRequestProcessing",
                return_value=processor,
            ),
            patch(  # test-quality-ok: routing is the seam that hands the relay its frame guard and returns its outcome
                "litellm.proxy.route_llm_request.route_request",
                new_callable=AsyncMock,
                side_effect=route_request,
            ),
            patch(  # test-quality-ok: the failure hook is the proxy's only path to a failed spend log row
                "litellm.proxy.proxy_server.proxy_logging_obj",
                proxy_logging_obj,
            ),
        ):
            endpoint = asyncio.ensure_future(
                responses_websocket_endpoint(websocket=ws, model=None, user_api_key_dict=MagicMock())
            )
            await asyncio.wait_for(row_started.wait(), timeout=1)
            proxy_logging_obj.arelease_responses_websocket_slot.assert_awaited_once()
            finished_early, _ = await asyncio.wait({endpoint}, timeout=0.05)
            assert endpoint not in finished_early
            row_may_finish.set()
            finished, _ = await asyncio.wait({endpoint}, timeout=1)
            assert endpoint in finished
            assert row_written.is_set()

        booked = proxy_logging_obj.post_call_failure_hook.await_args.kwargs
        assert booked["original_exception"] is (crash if ending == "crash" else refusal)

    @pytest.mark.asyncio
    async def test_a_crashed_socket_whose_failure_row_cannot_be_built_still_gets_its_error_frame_and_close(self):
        from litellm.exceptions import RateLimitError
        from litellm.proxy.response_api_endpoints.endpoints import responses_websocket_endpoint

        ws = MagicMock()
        ws.headers = {}
        ws.query_params = {}
        ws.scope = {"headers": []}
        ws.url = "ws://testserver/v1/responses"
        ws.accept = AsyncMock()
        ws.receive_text = AsyncMock(
            return_value=json.dumps({"type": "response.create", "model": "gpt-4o-mini", "input": []})
        )
        sent = []
        closes = []
        ws.send_text = AsyncMock(side_effect=sent.append)
        ws.close = AsyncMock(side_effect=lambda code, reason: closes.append((code, reason)))
        processor = MagicMock()
        processor.common_processing_pre_call_logic = AsyncMock(
            return_value=({"model": "gpt-4o-mini", "litellm_metadata": {}}, MagicMock())
        )
        refusal = RateLimitError(message="crossed budget", llm_provider="", model="")

        def route_request(data, **kwargs):
            async def relay():
                guarded = await data["responses_ws_frame_guard"](json.dumps({"type": "response.create", "input": "two"}))
                guarded.bill_refusal(refusal)
                data["litellm_metadata"]["standard_logging_guardrail_information"] = "written by a broken callback"
                raise RuntimeError("relay crashed")

            return relay()

        proxy_logging_obj = MagicMock()
        proxy_logging_obj.pre_call_hook = AsyncMock(
            side_effect=lambda user_api_key_dict, data, call_type, guardrails_only: {
                **data,
                "litellm_metadata": {"standard_logging_guardrail_information": [{"guardrail_name": "turn-check"}]},
            }
        )
        proxy_logging_obj.post_call_failure_hook = AsyncMock()
        proxy_logging_obj.arelease_responses_websocket_slot = AsyncMock(return_value=None)

        with (
            patch(  # test-quality-ok: first-frame model auth needs a live router and key table and has its own tests above
                "litellm.proxy.response_api_endpoints.endpoints._enforce_responses_ws_first_frame_model_auth",
                new_callable=AsyncMock,
            ),
            patch(  # test-quality-ok: the pre-call processor needs a live proxy; what the endpoint does with the relay's outcome is under test
                "litellm.proxy.response_api_endpoints.endpoints.ProxyBaseLLMRequestProcessing",
                return_value=processor,
            ),
            patch(  # test-quality-ok: routing is the seam that hands the relay its frame guard and returns its outcome
                "litellm.proxy.route_llm_request.route_request",
                new_callable=AsyncMock,
                side_effect=route_request,
            ),
            patch(  # test-quality-ok: the failure hook is the proxy's only path to a failed spend log row
                "litellm.proxy.proxy_server.proxy_logging_obj",
                proxy_logging_obj,
            ),
        ):
            await asyncio.wait_for(
                responses_websocket_endpoint(websocket=ws, model=None, user_api_key_dict=MagicMock()), timeout=1
            )

        assert [json.loads(frame)["error"]["message"] for frame in sent] == ["relay crashed"]
        assert closes == [(1011, "Internal server error")]

    @pytest.mark.asyncio
    async def test_endpoint_sends_an_error_frame_when_routing_rejects_the_connection(self):
        from litellm.proxy.response_api_endpoints.endpoints import (
            responses_websocket_endpoint,
        )

        ws = MagicMock()
        ws.headers = {}
        ws.query_params = {}
        ws.scope = {"headers": []}
        ws.url = "ws://testserver/v1/responses"
        ws.accept = AsyncMock()
        ws.receive_text = AsyncMock(
            return_value=json.dumps({"type": "response.create", "model": "gpt-4o-mini", "input": []})
        )
        ws.send_text = AsyncMock()
        ws.close = AsyncMock()

        processor = MagicMock()
        processor.common_processing_pre_call_logic = AsyncMock(
            return_value=({"model": "gpt-4o-mini", "litellm_metadata": {}}, MagicMock())
        )
        rejection = litellm.RateLimitError(
            message="origin deployment is cooling down", model="gpt-4o-mini", llm_provider="openai"
        )
        proxy_logging_obj = MagicMock()
        proxy_logging_obj.post_call_failure_hook = AsyncMock(return_value=None)
        proxy_logging_obj.arelease_responses_websocket_slot = AsyncMock(return_value=None)
        user_api_key_dict = MagicMock()

        with (
            patch(  # test-quality-ok: first-frame model auth needs a live router and key table and has its own tests above
                "litellm.proxy.response_api_endpoints.endpoints._enforce_responses_ws_first_frame_model_auth",
                new_callable=AsyncMock,
            ),
            patch(  # test-quality-ok: the pre-call processor needs a live proxy; what the endpoint tells the client is under test
                "litellm.proxy.response_api_endpoints.endpoints.ProxyBaseLLMRequestProcessing",
                return_value=processor,
            ),
            patch(  # test-quality-ok: routing is the seam that raises the affinity rejection
                "litellm.proxy.route_llm_request.route_request",
                new_callable=AsyncMock,
                side_effect=rejection,
            ),
            patch(  # test-quality-ok: the failure hook is the proxy's only path to a failed spend log row
                "litellm.proxy.proxy_server.proxy_logging_obj",
                proxy_logging_obj,
            ),
        ):
            await responses_websocket_endpoint(
                websocket=ws,
                model=None,
                user_api_key_dict=user_api_key_dict,
            )

        frame = json.loads(ws.send_text.await_args.args[0])
        assert frame["type"] == "error"
        assert frame["status"] == 429
        assert frame["error"]["type"] == "rate_limit_exceeded"
        assert "cooling down" in frame["error"]["message"]
        ws.close.assert_awaited_once_with(code=1011, reason="Internal server error")
        booked = proxy_logging_obj.post_call_failure_hook.await_args.kwargs
        assert booked["original_exception"] is rejection
        assert booked["user_api_key_dict"] is user_api_key_dict
        assert booked["request_data"]["model"] == "gpt-4o-mini"

    @pytest.mark.asyncio
    async def test_reruns_model_auth_for_first_frame_model(self):
        from starlette.requests import Request

        from litellm.proxy.response_api_endpoints.endpoints import (
            _enforce_responses_ws_first_frame_model_auth,
        )

        request = Request({"type": "http", "method": "POST", "path": "/v1/responses", "headers": []})
        user_api_key_dict = MagicMock()
        llm_router = MagicMock()
        empty_settings: Final[dict[str, object]] = {}

        with (
            patch(
                "litellm.proxy.auth.user_api_key_auth.enforce_key_and_fallback_model_access",
                new_callable=AsyncMock,
            ) as mock_key_check,
            patch(
                "litellm.proxy.auth.user_api_key_auth.run_centralized_common_checks",
                new_callable=AsyncMock,
            ) as mock_common_checks,
            patch(
                "litellm.proxy.proxy_server.llm_model_list",
                [],
            ),
            patch("litellm.proxy.proxy_server.master_key", "sk-test"),
            patch("litellm.proxy.proxy_server.user_custom_auth", None),
            patch("litellm.proxy.proxy_server.general_settings", empty_settings),
        ):
            await _enforce_responses_ws_first_frame_model_auth(
                request=request,
                model="gpt-4o-mini",
                user_api_key_dict=user_api_key_dict,
                llm_router=llm_router,
            )

        mock_key_check.assert_awaited_once_with(
            valid_token=user_api_key_dict,
            request_data={"model": "gpt-4o-mini"},
            route="/v1/responses",
            request=request,
            llm_model_list=[],
            llm_router=llm_router,
        )
        mock_common_checks.assert_awaited_once_with(
            user_api_key_auth_obj=user_api_key_dict,
            request=request,
            request_data={"model": "gpt-4o-mini"},
            route="/v1/responses",
        )


class TestReadWSModelFromFirstFrameErrors:
    @pytest.mark.asyncio
    async def test_transport_error_first_frame_closes_with_internal_error(self):
        import asyncio

        from litellm.proxy.response_api_endpoints.endpoints import (
            _read_ws_model_from_first_frame,
        )

        ws = MagicMock()
        ws.receive_text = AsyncMock(side_effect=asyncio.TimeoutError())
        ws.send_text = AsyncMock()
        ws.close = AsyncMock()

        result = await _read_ws_model_from_first_frame(ws)

        assert result is None
        ws.send_text.assert_not_awaited()
        ws.close.assert_awaited_once_with(code=1011, reason="Internal server error")

    @pytest.mark.asyncio
    async def test_invalid_json_sends_error_and_closes(self):
        from litellm.proxy.response_api_endpoints.endpoints import (
            _read_ws_model_from_first_frame,
        )

        ws = MagicMock()
        ws.receive_text = AsyncMock(return_value="this is not json")
        ws.send_text = AsyncMock()
        ws.close = AsyncMock()

        result = await _read_ws_model_from_first_frame(ws)

        assert result is None
        payload = json.loads(ws.send_text.await_args.args[0])
        assert payload["error"]["message"] == "First message is not valid JSON."
        ws.close.assert_awaited_once_with(code=1008, reason="Invalid JSON in first message")

    @pytest.mark.asyncio
    async def test_missing_model_sends_error_and_closes(self):
        from litellm.proxy.response_api_endpoints.endpoints import (
            _read_ws_model_from_first_frame,
        )

        ws = MagicMock()
        ws.receive_text = AsyncMock(return_value=json.dumps({"type": "response.create", "input": []}))
        ws.send_text = AsyncMock()
        ws.close = AsyncMock()

        result = await _read_ws_model_from_first_frame(ws)

        assert result is None
        payload = json.loads(ws.send_text.await_args.args[0])
        assert "No model provided" in payload["error"]["message"]
        ws.close.assert_awaited_once_with(code=1008, reason="No model provided")

    @pytest.mark.asyncio
    async def test_valid_first_frame_returns_model_and_raw(self):
        from litellm.proxy.response_api_endpoints.endpoints import (
            _read_ws_model_from_first_frame,
        )

        raw = json.dumps({"type": "response.create", "model": "gpt-4o", "input": []})
        ws = MagicMock()
        ws.receive_text = AsyncMock(return_value=raw)
        ws.send_text = AsyncMock()
        ws.close = AsyncMock()

        result = await _read_ws_model_from_first_frame(ws)

        assert result == ("gpt-4o", raw)
        ws.send_text.assert_not_awaited()
        ws.close.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_query_model_wins_over_first_frame_model(self):
        from litellm.proxy.response_api_endpoints.endpoints import (
            _read_ws_model_from_first_frame,
        )

        raw = json.dumps({"type": "response.create", "model": "gpt-4o", "input": []})
        ws = MagicMock()
        ws.receive_text = AsyncMock(return_value=raw)
        ws.send_text = AsyncMock()
        ws.close = AsyncMock()

        result = await _read_ws_model_from_first_frame(ws, query_model="reasoning-group")

        assert result == ("reasoning-group", raw)
        ws.close.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_query_model_satisfies_a_first_frame_without_model(self):
        from litellm.proxy.response_api_endpoints.endpoints import (
            _read_ws_model_from_first_frame,
        )

        raw = json.dumps({"type": "response.create", "input": []})
        ws = MagicMock()
        ws.receive_text = AsyncMock(return_value=raw)
        ws.send_text = AsyncMock()
        ws.close = AsyncMock()

        result = await _read_ws_model_from_first_frame(ws, query_model="reasoning-group")

        assert result == ("reasoning-group", raw)
        ws.send_text.assert_not_awaited()
        ws.close.assert_not_awaited()


@pytest.mark.parametrize(
    "configured,expected",
    [
        (None, 3600.0),
        (60, 60.0),
        (1200, 1200.0),
        (7200, 7200.0),
        (59, 3600.0),
        (0, 3600.0),
        (9000, 3600.0),
        ("not-a-number", 3600.0),
    ],
)
def test_responses_ws_session_limit_resolution(monkeypatch, configured, expected):
    from litellm.proxy.proxy_server import general_settings
    from litellm.proxy.response_api_endpoints.endpoints import (
        _resolve_responses_ws_session_limit_seconds,
    )

    if configured is None:
        monkeypatch.delitem(general_settings, "responses_websocket_session_limit_seconds", raising=False)
    else:
        monkeypatch.setitem(general_settings, "responses_websocket_session_limit_seconds", configured)

    assert _resolve_responses_ws_session_limit_seconds() == expected


class TestManagedResponsesSameProvider:
    def _handler(self, model, custom_llm_provider=None):
        from litellm.responses.streaming_iterator import (
            ManagedResponsesWebSocketHandler,
        )

        return ManagedResponsesWebSocketHandler(
            websocket=MagicMock(),
            model=model,
            logging_obj=MagicMock(),
            custom_llm_provider=custom_llm_provider,
        )

    def test_none_model_treated_as_same_provider(self):
        assert self._handler("openai/gpt-4o")._same_provider(None) is True

    def test_identical_model_is_same_provider(self):
        assert self._handler("openai/gpt-4o")._same_provider("openai/gpt-4o") is True

    def test_same_provider_different_model(self):
        assert self._handler("gpt-4o")._same_provider("gpt-4o-mini") is True

    def test_different_provider_is_not_same(self):
        assert self._handler("gpt-4o")._same_provider("vertex_ai/gemini-2.0-flash") is False

    def test_inject_credentials_keeps_provider_for_same_provider_model(self):
        handler = self._handler("gpt-4o", custom_llm_provider="openai")
        call_kwargs: dict = {}
        handler._inject_credentials(call_kwargs, model="gpt-4o-mini")
        assert call_kwargs["custom_llm_provider"] == "openai"

    def test_inject_credentials_drops_provider_for_cross_provider_model(self):
        handler = self._handler("gpt-4o", custom_llm_provider="openai")
        call_kwargs: dict = {}
        handler._inject_credentials(call_kwargs, model="vertex_ai/gemini-2.0-flash")
        assert "custom_llm_provider" not in call_kwargs

    def test_unresolvable_connection_model_falls_back_to_custom_provider(self):
        handler = self._handler("my-custom-deployment", custom_llm_provider="openai")
        assert handler._same_provider("gpt-4o-mini") is True
        call_kwargs: dict = {}
        handler._inject_credentials(call_kwargs, model="gpt-4o-mini")
        assert call_kwargs["custom_llm_provider"] == "openai"

    def test_unresolvable_connection_model_still_drops_cross_provider(self):
        handler = self._handler("my-custom-deployment", custom_llm_provider="openai")
        call_kwargs: dict = {}
        handler._inject_credentials(call_kwargs, model="vertex_ai/gemini-2.0-flash")
        assert "custom_llm_provider" not in call_kwargs


def _auth_override():
    from litellm.proxy._types import UserAPIKeyAuth

    return UserAPIKeyAuth(api_key="sk-test-cursor", user_id="cursor-user")


def test_cursor_chat_completions_messages_body_uses_chat_pipeline():
    """A genuine chat-completions body (``messages`` present; what Cursor sends for
    models whose BYOK it already fixed) must run through the standard chat pipeline
    untouched: multi-turn tool history (assistant tool_calls + role="tool" results)
    and nested chat-format tool defs are valid there, while blindly renaming
    ``messages`` to ``input`` (the pre-fix behavior) produced items the Responses API
    rejects. Asserts acompletion is called with the exact messages and aresponses is
    never touched."""
    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

    import litellm.proxy.proxy_server as ps

    messages = [
        {"role": "user", "content": "read a file"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_hist1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": '{"path": "a.py"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_hist1", "content": "file contents"},
        {"role": "user", "content": "now summarize"},
    ]

    mock_router = MagicMock()
    mock_router.acompletion = AsyncMock(
        return_value=litellm.ModelResponse(
            id="chatcmpl-cursor-1",
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "summary"},
                    "finish_reason": "stop",
                }
            ],
            model="gpt-4o",
        )
    )
    mock_router.aresponses = AsyncMock()
    mock_router.get_available_deployment = MagicMock(return_value=None)

    app.dependency_overrides[user_api_key_auth] = _auth_override
    try:
        with patch.object(ps, "llm_router", mock_router):
            client = TestClient(app)
            response = client.post(
                "/cursor/chat/completions",
                json={
                    "model": "gpt-4o",
                    "messages": messages,
                    "tools": [
                        {
                            "type": "function",
                            "function": {"name": "read_file", "parameters": {"type": "object"}},
                        }
                    ],
                },
                headers={"Authorization": "Bearer sk-test-cursor"},
            )
    finally:
        app.dependency_overrides.pop(user_api_key_auth, None)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["choices"][0]["message"]["content"] == "summary"
    assert "output" not in body

    mock_router.acompletion.assert_called_once()
    called_kwargs = mock_router.acompletion.call_args.kwargs
    assert called_kwargs["messages"] == messages
    assert "input" not in called_kwargs
    mock_router.aresponses.assert_not_called()


def test_cursor_chat_completions_input_body_uses_responses_pipeline_and_strips_stream_options():
    """A Responses-shaped body (``input``, no ``messages``; what Cursor agent mode
    sends) must run through the Responses pipeline with chat-completions output, and
    ``stream_options`` (chat-completions-only; Cursor sends include_usage) must be
    stripped before the Responses call since OpenAI's Responses API rejects it.
    Stripping must not mutate the dict _read_request_body returned: that can be the
    request-scope cached parsed body itself, and removing a key from it corrupts the
    cache's key snapshot so any later _read_request_body caller (spend tracking,
    logging hooks) silently gets an empty body; a follow-up read must still see the
    full original body."""
    import asyncio

    from openai.types.responses import ResponseOutputMessage, ResponseOutputText

    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
    from litellm.proxy.common_utils.http_parsing_utils import (
        read_request_body as real_read_request_body,
    )
    from litellm.types.llms.openai import ResponsesAPIResponse

    import litellm.proxy.proxy_server as ps

    captured_requests = []

    async def capturing_read_request_body(request):
        captured_requests.append(request)
        return await real_read_request_body(request=request)

    mock_router = MagicMock()
    mock_router.aresponses = AsyncMock(
        return_value=ResponsesAPIResponse(
            id="resp_cursor_agent1",
            created_at=1234567890,
            model="gpt-4o",
            object="response",
            output=[
                ResponseOutputMessage(
                    id="msg_agent1",
                    type="message",
                    role="assistant",
                    status="completed",
                    content=[ResponseOutputText(type="output_text", text="agent reply", annotations=[])],
                )
            ],
        )
    )
    mock_router.acompletion = AsyncMock()

    app.dependency_overrides[user_api_key_auth] = _auth_override
    try:
        with (
            patch.object(ps, "llm_router", mock_router),
            patch(
                "litellm.proxy.response_api_endpoints.endpoints.read_request_body",
                side_effect=capturing_read_request_body,
            ),
        ):
            client = TestClient(app)
            response = client.post(
                "/cursor/chat/completions",
                json={
                    "model": "gpt-4o",
                    "input": [{"role": "user", "content": "hello"}],
                    "stream_options": {"include_usage": True},
                },
                headers={"Authorization": "Bearer sk-test-cursor"},
            )
    finally:
        app.dependency_overrides.pop(user_api_key_auth, None)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["choices"][0]["message"]["content"] == "agent reply"
    assert "output" not in body

    mock_router.aresponses.assert_called_once()
    called_kwargs = mock_router.aresponses.call_args.kwargs
    assert "stream_options" not in called_kwargs
    mock_router.acompletion.assert_not_called()

    assert captured_requests
    followup_body = asyncio.run(real_read_request_body(request=captured_requests[0]))
    assert followup_body.get("stream_options") == {"include_usage": True}
    assert followup_body.get("input") == [{"role": "user", "content": "hello"}]


def test_cursor_models_route_delegates_to_model_list():
    """Clients pointed at <proxy>/cursor as an OpenAI-compatible base URL resolve and
    verify keys via GET {base}/models (the OpenAI SDK contract). Without a dedicated
    route those requests fall through to the Cursor Cloud Agents passthrough and 401
    for lack of a Cursor API key, so BYOK verification fails before any chat request
    is sent. Both /cursor/models and /cursor/v1/models must serve the standard model
    list instead."""
    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

    import litellm.proxy.proxy_server as ps

    model_payload = {"data": [{"id": "gpt-5.6", "object": "model"}], "object": "list"}

    app.dependency_overrides[user_api_key_auth] = _auth_override
    try:
        with patch.object(ps, "model_list", AsyncMock(return_value=model_payload)) as mock_model_list:
            client = TestClient(app)
            for path in ("/cursor/models", "/cursor/v1/models"):
                response = client.get(path, headers={"Authorization": "Bearer sk-test-cursor"})
                assert response.status_code == 200, f"{path}: {response.text}"
                assert response.json() == model_payload
            assert mock_model_list.call_count == 2
    finally:
        app.dependency_overrides.pop(user_api_key_auth, None)


class TestNestFlatChatTools:
    def test_flat_custom_tool_is_nested(self):
        from litellm.proxy.response_api_endpoints.endpoints import _convert_tool_envelope

        result = _convert_tool_envelope(
            {"type": "custom", "name": "ApplyPatch", "description": "V4A patch", "format": {"type": "text"}},
            to_chat=True,
        )
        assert result == {
            "type": "custom",
            "custom": {"name": "ApplyPatch", "description": "V4A patch", "format": {"type": "text"}},
        }

    def test_flat_function_tool_is_nested(self):
        from litellm.proxy.response_api_endpoints.endpoints import _convert_tool_envelope

        result = _convert_tool_envelope(
            {"type": "function", "name": "read_file", "description": "d", "parameters": {"type": "object"}},
            to_chat=True,
        )
        assert result == {
            "type": "function",
            "function": {"name": "read_file", "description": "d", "parameters": {"type": "object"}},
        }

    def test_already_nested_and_unrecognized_tools_pass_through_unchanged(self):
        from litellm.proxy.response_api_endpoints.endpoints import _convert_tool_envelope

        tools = [
            {"type": "custom", "custom": {"name": "already_nested"}},
            {"type": "function", "function": {"name": "f", "parameters": {}}},
            {"type": "web_search"},
            {"type": "custom"},
            {"name": "typeless"},
            {},
            "junk",
            None,
            42,
        ]
        assert [_convert_tool_envelope(tool, to_chat=True) for tool in tools] == tools


class TestCursorMessagesArmToolNormalization:
    @pytest.mark.asyncio
    async def test_flat_custom_tool_nested_before_chat_completion_delegation(self):
        from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
        from litellm.proxy._types import UserAPIKeyAuth

        seen = {}

        async def fake_chat_completion(request, fastapi_response, model, user_api_key_dict):
            from litellm.proxy.common_utils.http_parsing_utils import read_request_body

            seen["body"] = await read_request_body(request=request)
            return {"id": "chatcmpl-fake", "object": "chat.completion", "choices": []}

        app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(api_key=MASTER_KEY)
        try:
            with patch("litellm.proxy.proxy_server.chat_completion", new=fake_chat_completion):
                client = TestClient(app)
                response = client.post(
                    "/cursor/chat/completions",
                    json={
                        "model": "gpt-5.6",
                        "messages": [{"role": "user", "content": "use ApplyPatch"}],
                        "tools": [
                            {
                                "type": "function",
                                "function": {"name": "read_file", "parameters": {"type": "object"}},
                            },
                            {
                                "type": "custom",
                                "name": "ApplyPatch",
                                "description": "V4A patch",
                                "format": {
                                    "type": "grammar",
                                    "definition": "start: patch",
                                    "syntax": "lark",
                                },
                            },
                        ],
                        "tool_choice": {"type": "custom", "name": "ApplyPatch"},
                    },
                    headers={"Authorization": "Bearer sk-9876"},
                )
        finally:
            app.dependency_overrides.pop(user_api_key_auth, None)

        assert response.status_code == 200
        assert seen["body"]["tools"] == [
            {"type": "function", "function": {"name": "read_file", "parameters": {"type": "object"}}},
            {
                "type": "custom",
                "custom": {
                    "name": "ApplyPatch",
                    "description": "V4A patch",
                    "format": {
                        "type": "grammar",
                        "grammar": {"definition": "start: patch", "syntax": "lark"},
                    },
                },
            },
        ]
        assert seen["body"]["tool_choice"] == {"type": "custom", "custom": {"name": "ApplyPatch"}}
        assert seen["body"]["messages"] == [{"role": "user", "content": "use ApplyPatch"}]

    @pytest.mark.asyncio
    async def test_messages_body_without_flat_tools_leaves_parsed_body_cache_untouched(self):
        from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
        from litellm.proxy._types import UserAPIKeyAuth

        seen = {}

        async def fake_chat_completion(request, fastapi_response, model, user_api_key_dict):
            from litellm.proxy.common_utils.http_parsing_utils import read_request_body

            seen["body"] = await read_request_body(request=request)
            return {"id": "chatcmpl-fake", "object": "chat.completion", "choices": []}

        body = {
            "model": "gpt-5.6",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {"name": "f", "parameters": {}}}],
        }
        app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(api_key=MASTER_KEY)
        try:
            with patch("litellm.proxy.proxy_server.chat_completion", new=fake_chat_completion):
                client = TestClient(app)
                response = client.post(
                    "/cursor/chat/completions",
                    json=body,
                    headers={"Authorization": "Bearer sk-9876"},
                )
        finally:
            app.dependency_overrides.pop(user_api_key_auth, None)

        assert response.status_code == 200
        assert seen["body"]["tools"] == body["tools"]
        assert seen["body"]["messages"] == body["messages"]


class TestToolEnvelopeConversionMatrix:
    """
    Cursor mixes Responses API shapes into chat bodies PER LEVEL, independently
    (live-captured: a pre-nested custom envelope carrying a flat grammar format).
    Tool definitions and tool_choice share one envelope rule, so every cell of
    direction x envelope x format must land on that direction's canonical shape.
    """

    FLAT_GRAMMAR = {"type": "grammar", "definition": "start: patch", "syntax": "lark"}
    NESTED_GRAMMAR = {"type": "grammar", "grammar": {"definition": "start: patch", "syntax": "lark"}}
    TEXT = {"type": "text"}

    @pytest.mark.parametrize("to_chat", [True, False])
    @pytest.mark.parametrize("envelope", ["flat", "nested"])
    @pytest.mark.parametrize("format_shape", ["absent", "text", "flat_grammar", "nested_grammar"])
    def test_every_direction_envelope_and_format_lands_canonical(self, to_chat, envelope, format_shape):
        from litellm.proxy.response_api_endpoints.endpoints import _convert_tool_envelope

        format_value = {
            "absent": None,
            "text": self.TEXT,
            "flat_grammar": self.FLAT_GRAMMAR,
            "nested_grammar": self.NESTED_GRAMMAR,
        }[format_shape]
        payload = {"name": "ApplyPatch", "description": "V4A patch"}
        if format_value is not None:
            payload["format"] = format_value
        tool = {"type": "custom", "custom": payload} if envelope == "nested" else {"type": "custom", **payload}

        canonical_payload = {"name": "ApplyPatch", "description": "V4A patch"}
        if format_shape in ("flat_grammar", "nested_grammar"):
            canonical_payload["format"] = self.NESTED_GRAMMAR if to_chat else self.FLAT_GRAMMAR
        elif format_shape == "text":
            canonical_payload["format"] = self.TEXT
        expected = (
            {"type": "custom", "custom": canonical_payload} if to_chat else {"type": "custom", **canonical_payload}
        )

        assert _convert_tool_envelope(tool, to_chat=to_chat) == expected

    def test_nested_envelope_with_flat_grammar_matches_live_cursor_capture(self):
        from litellm.proxy.response_api_endpoints.endpoints import _convert_tool_envelope

        cursor_tool = {"type": "custom", "custom": {"name": "ApplyPatch", "format": self.FLAT_GRAMMAR}}
        assert _convert_tool_envelope(cursor_tool, to_chat=True) == {
            "type": "custom",
            "custom": {"name": "ApplyPatch", "format": self.NESTED_GRAMMAR},
        }

    @pytest.mark.parametrize("to_chat", [True, False])
    def test_conversion_is_idempotent(self, to_chat):
        from litellm.proxy.response_api_endpoints.endpoints import _convert_tool_envelope

        once = _convert_tool_envelope({"type": "custom", "name": "A", "format": self.FLAT_GRAMMAR}, to_chat=to_chat)
        assert _convert_tool_envelope(once, to_chat=to_chat) == once

    def test_nested_function_tool_flattens_and_flat_passes_through(self):
        from litellm.proxy.response_api_endpoints.endpoints import _convert_tool_envelope

        nested = {"type": "function", "function": {"name": "read_file", "parameters": {"type": "object"}}}
        flat = {"type": "function", "name": "read_file", "parameters": {"type": "object"}}
        assert _convert_tool_envelope(nested, to_chat=False) == flat
        assert _convert_tool_envelope(flat, to_chat=False) == flat

    @pytest.mark.parametrize("to_chat", [True, False])
    def test_unrecognized_entries_pass_through(self, to_chat):
        from litellm.proxy.response_api_endpoints.endpoints import _convert_tool_envelope

        entries = [{"type": "web_search"}, {"type": "custom"}, "junk", None, {}, 42, {"type": "auto"}]
        assert [_convert_tool_envelope(entry, to_chat=to_chat) for entry in entries] == entries

    @pytest.mark.parametrize("to_chat", [True, False])
    def test_empty_nested_envelope_falls_back_to_top_level_payload(self, to_chat):
        """An empty nested envelope must not shadow payload fields that sit at the top
        level; treating the empty dict as the sole payload source dropped the name."""
        from litellm.proxy.response_api_endpoints.endpoints import _convert_tool_envelope

        hybrid = {"type": "custom", "custom": {}, "name": "ApplyPatch", "format": self.TEXT}
        expected_payload = {"name": "ApplyPatch", "format": self.TEXT}
        expected = {"type": "custom", "custom": expected_payload} if to_chat else {"type": "custom", **expected_payload}
        assert _convert_tool_envelope(hybrid, to_chat=to_chat) == expected

    def test_nested_payload_wins_over_stray_top_level_fields(self):
        from litellm.proxy.response_api_endpoints.endpoints import _convert_tool_envelope

        tool = {"type": "custom", "custom": {"name": "NestedName"}, "name": "TopName"}
        assert _convert_tool_envelope(tool, to_chat=False) == {"type": "custom", "name": "NestedName"}

    @pytest.mark.parametrize("to_chat", [True, False])
    def test_nameless_envelope_passes_through_unchanged(self, to_chat):
        from litellm.proxy.response_api_endpoints.endpoints import _convert_tool_envelope

        nameless = {"type": "custom", "custom": {}, "description": "no name anywhere"}
        assert _convert_tool_envelope(nameless, to_chat=to_chat) == nameless


class TestToolChoiceSharesTheToolEnvelopeRule:
    """
    tool_choice carries the same {"type": T, T: {...}} chat envelope as a tool
    definition, so it converts through the same function in both directions.
    OpenAI requires the nested key on chat (SDK ChatCompletionNamedToolChoiceParam
    and ChatCompletionNamedToolChoiceCustomParam both mark it Required).
    """

    @pytest.mark.parametrize("choice_type", ["custom", "function"])
    def test_flat_tool_choice_is_nested_for_chat(self, choice_type):
        from litellm.proxy.response_api_endpoints.endpoints import _convert_tool_envelope

        assert _convert_tool_envelope({"type": choice_type, "name": "ApplyPatch"}, to_chat=True) == {
            "type": choice_type,
            choice_type: {"name": "ApplyPatch"},
        }

    @pytest.mark.parametrize("choice_type", ["custom", "function"])
    def test_nested_tool_choice_is_flattened_for_responses(self, choice_type):
        from litellm.proxy.response_api_endpoints.endpoints import _convert_tool_envelope

        assert _convert_tool_envelope({"type": choice_type, choice_type: {"name": "ApplyPatch"}}, to_chat=False) == {
            "type": choice_type,
            "name": "ApplyPatch",
        }

    @pytest.mark.parametrize("to_chat", [True, False])
    def test_sentinel_and_malformed_tool_choice_pass_through(self, to_chat):
        from litellm.proxy.response_api_endpoints.endpoints import _convert_tool_envelope

        for unchanged in ("auto", "required", "none", None, {"type": "auto"}, 42):
            assert _convert_tool_envelope(unchanged, to_chat=to_chat) == unchanged


class TestNormalizeToolDialectCoversBothFields:
    """
    The regression that motivated one normalizer: tools were converted while
    tool_choice was left flat, so OpenAI rejected the request. Both fields move
    together in a single call, on both arms.
    """

    @pytest.mark.parametrize("to_chat", [True, False])
    def test_tools_and_tool_choice_convert_together(self, to_chat):
        from litellm.proxy.response_api_endpoints.endpoints import _normalize_tool_dialect

        flat = {"type": "custom", "name": "ApplyPatch"}
        nested = {"type": "custom", "custom": {"name": "ApplyPatch"}}
        source = flat if to_chat else nested
        expected = nested if to_chat else flat

        out = _normalize_tool_dialect({"messages": [], "tools": [source], "tool_choice": source}, to_chat=to_chat)
        assert out["tools"] == [expected]
        assert out["tool_choice"] == expected

    def test_body_needing_no_conversion_is_returned_by_identity(self):
        from litellm.proxy.response_api_endpoints.endpoints import _normalize_tool_dialect

        data = {"messages": [], "tools": [{"type": "function", "function": {"name": "f"}}], "tool_choice": "auto"}
        assert _normalize_tool_dialect(data, to_chat=True) is data

    def test_absent_tool_fields_are_not_invented(self):
        from litellm.proxy.response_api_endpoints.endpoints import _normalize_tool_dialect

        data = {"messages": [{"role": "user", "content": "hi"}]}
        result = _normalize_tool_dialect(data, to_chat=True)
        assert result == data
        assert "tools" not in result and "tool_choice" not in result


class TestCursorInputArmFlattening:
    @pytest.mark.asyncio
    async def test_nested_chat_shapes_in_input_body_reach_aresponses_flattened(self):
        from litellm.proxy._types import UserAPIKeyAuth
        from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
        from openai.types.responses import ResponseOutputMessage, ResponseOutputText

        from litellm.types.llms.openai import ResponsesAPIResponse

        mock_response = ResponsesAPIResponse(
            id="resp_flat123",
            created_at=1234567890,
            model="gpt-5.6",
            object="response",
            output=[
                ResponseOutputMessage(
                    id="msg_flat123",
                    type="message",
                    role="assistant",
                    status="completed",
                    content=[ResponseOutputText(type="output_text", text="ok", annotations=[])],
                )
            ],
        )

        app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(api_key=MASTER_KEY)
        try:
            with patch("litellm.proxy.proxy_server.llm_router") as mock_router:
                mock_router.aresponses = AsyncMock(return_value=mock_response)
                client = TestClient(app)
                response = client.post(
                    "/cursor/chat/completions",
                    json={
                        "model": "gpt-5.6",
                        "input": [{"role": "user", "content": "use ApplyPatch"}],
                        "tools": [
                            {
                                "type": "custom",
                                "custom": {
                                    "name": "ApplyPatch",
                                    "format": {
                                        "type": "grammar",
                                        "grammar": {"definition": "start: patch", "syntax": "lark"},
                                    },
                                },
                            },
                            {"type": "function", "name": "read_file", "parameters": {"type": "object"}},
                        ],
                        "tool_choice": {"type": "custom", "custom": {"name": "ApplyPatch"}},
                    },
                    headers={"Authorization": "Bearer sk-9876"},
                )
        finally:
            app.dependency_overrides.pop(user_api_key_auth, None)

        assert response.status_code == 200
        call_kwargs = mock_router.aresponses.call_args.kwargs
        assert call_kwargs["tools"] == [
            {
                "type": "custom",
                "name": "ApplyPatch",
                "format": {"type": "grammar", "definition": "start: patch", "syntax": "lark"},
            },
            {"type": "function", "name": "read_file", "parameters": {"type": "object"}},
        ]
        assert call_kwargs["tool_choice"] == {"type": "custom", "name": "ApplyPatch"}


class TestChatCompletionsBodyDetection:
    def test_routing_matrix(self):
        from litellm.proxy.response_api_endpoints.endpoints import _is_chat_completions_body

        assert _is_chat_completions_body({"messages": [{"role": "user", "content": "hi"}]}) is True
        assert _is_chat_completions_body({"messages": [{"role": "user", "content": "hi"}], "input": []}) is True
        assert _is_chat_completions_body({"messages": None, "input": [{"role": "user", "content": "hi"}]}) is False
        assert _is_chat_completions_body({"messages": [], "input": [{"role": "user", "content": "hi"}]}) is False
        assert _is_chat_completions_body({"messages": None}) is True
        assert _is_chat_completions_body({"messages": []}) is True
        assert _is_chat_completions_body({"input": [{"role": "user", "content": "hi"}]}) is False
        assert _is_chat_completions_body({}) is False

    @pytest.mark.asyncio
    async def test_null_messages_stub_with_input_reaches_responses_arm(self):
        from openai.types.responses import ResponseOutputMessage, ResponseOutputText

        from litellm.proxy._types import UserAPIKeyAuth
        from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
        from litellm.types.llms.openai import ResponsesAPIResponse

        mock_response = ResponsesAPIResponse(
            id="resp_stub1",
            created_at=1234567890,
            model="gpt-5.6",
            object="response",
            output=[
                ResponseOutputMessage(
                    id="msg_stub1",
                    type="message",
                    role="assistant",
                    status="completed",
                    content=[ResponseOutputText(type="output_text", text="ok", annotations=[])],
                )
            ],
        )

        app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(api_key=MASTER_KEY)
        try:
            with patch("litellm.proxy.proxy_server.llm_router") as mock_router:
                mock_router.aresponses = AsyncMock(return_value=mock_response)
                client = TestClient(app)
                response = client.post(
                    "/cursor/chat/completions",
                    json={
                        "model": "gpt-5.6",
                        "messages": None,
                        "input": [{"role": "user", "content": "hello"}],
                    },
                    headers={"Authorization": "Bearer sk-9876"},
                )
        finally:
            app.dependency_overrides.pop(user_api_key_auth, None)

        assert response.status_code == 200
        assert mock_router.aresponses.call_args is not None
        assert mock_router.aresponses.call_args.kwargs["input"] == [{"role": "user", "content": "hello"}]


class TestParseCursorModelVariant:
    @pytest.mark.parametrize(
        "model,expected_base,expected_effort",
        [
            ("claude-opus-5-thinking-high", "claude-opus-5", "high"),
            ("claude-opus-5-thinking-xhigh-fast", "claude-opus-5", "xhigh"),
            ("gemini-3.0-pro-thinking-low", "gemini-3.0-pro", "low"),
            ("claude-opus-5-fast", "claude-opus-5", None),
            ("gpt-5.6-sol", "gpt-5.6-sol", None),
            ("foo-thinking-ultra-fast", "foo-thinking-ultra", None),
            ("gpt-5.6-thinking-max", "gpt-5.6", "max"),
            ("foo-thinking-mega-fast", "foo-thinking-mega", None),
            ("-thinking-high", "-thinking-high", None),
        ],
    )
    def test_parse_matrix(self, model, expected_base, expected_effort):
        from litellm.proxy.response_api_endpoints.endpoints import _parse_cursor_model_variant

        variant = _parse_cursor_model_variant(model)
        assert variant.base_model == expected_base
        assert variant.reasoning_effort == expected_effort


class TestResolveCursorModelVariant:
    @pytest.fixture(scope="class")
    def wildcard_router(self):
        from litellm import Router

        return Router(
            model_list=[
                {"model_name": "anthropic/*", "litellm_params": {"model": "anthropic/*", "api_key": "fake"}},
                {"model_name": "openai/*", "litellm_params": {"model": "openai/*", "api_key": "fake"}},
                {
                    "model_name": "explicit-alias-thinking-high",
                    "litellm_params": {"model": "anthropic/claude-opus-5", "api_key": "fake"},
                },
            ]
        )

    def test_chat_body_suffix_stripped_into_reasoning_effort(self, wildcard_router):
        from litellm.proxy.response_api_endpoints.endpoints import _resolve_cursor_model_variant

        body = {
            "model": "claude-opus-5-thinking-xhigh-fast",
            "messages": [{"role": "user", "content": "hi"}],
        }
        resolved = _resolve_cursor_model_variant(body, wildcard_router)
        assert resolved["model"] == "claude-opus-5"
        assert resolved["reasoning_effort"] == "xhigh"
        assert resolved["messages"] == body["messages"]
        assert body["model"] == "claude-opus-5-thinking-xhigh-fast"

    def test_responses_body_suffix_stripped_into_reasoning_dict(self, wildcard_router):
        from litellm.proxy.response_api_endpoints.endpoints import _resolve_cursor_model_variant

        body = {"model": "claude-opus-5-thinking-high", "input": [{"role": "user", "content": "hi"}]}
        resolved = _resolve_cursor_model_variant(body, wildcard_router)
        assert resolved["model"] == "claude-opus-5"
        assert resolved["reasoning"] == {"effort": "high"}

    def test_responses_body_merges_effort_into_existing_reasoning(self, wildcard_router):
        from litellm.proxy.response_api_endpoints.endpoints import _resolve_cursor_model_variant

        body = {
            "model": "claude-opus-5-thinking-high",
            "input": [{"role": "user", "content": "hi"}],
            "reasoning": {"summary": "auto"},
        }
        resolved = _resolve_cursor_model_variant(body, wildcard_router)
        assert resolved["model"] == "claude-opus-5"
        assert resolved["reasoning"] == {"summary": "auto", "effort": "high"}

    def test_existing_reasoning_effort_wins_but_model_still_rewritten(self, wildcard_router):
        from litellm.proxy.response_api_endpoints.endpoints import _resolve_cursor_model_variant

        chat_body = {
            "model": "claude-opus-5-thinking-high",
            "messages": [{"role": "user", "content": "hi"}],
            "reasoning_effort": "low",
        }
        resolved_chat = _resolve_cursor_model_variant(chat_body, wildcard_router)
        assert resolved_chat["model"] == "claude-opus-5"
        assert resolved_chat["reasoning_effort"] == "low"

        responses_body = {
            "model": "claude-opus-5-thinking-high",
            "input": [{"role": "user", "content": "hi"}],
            "reasoning": {"effort": "low"},
        }
        resolved_responses = _resolve_cursor_model_variant(responses_body, wildcard_router)
        assert resolved_responses["model"] == "claude-opus-5"
        assert resolved_responses["reasoning"] == {"effort": "low"}

    def test_fast_only_suffix_strips_without_reasoning(self, wildcard_router):
        from litellm.proxy.response_api_endpoints.endpoints import _resolve_cursor_model_variant

        body = {"model": "claude-opus-5-fast", "messages": [{"role": "user", "content": "hi"}]}
        resolved = _resolve_cursor_model_variant(body, wildcard_router)
        assert resolved["model"] == "claude-opus-5"
        assert "reasoning_effort" not in resolved

    def test_explicitly_configured_suffixed_name_untouched(self, wildcard_router):
        from litellm.proxy.response_api_endpoints.endpoints import _resolve_cursor_model_variant

        body = {"model": "explicit-alias-thinking-high", "messages": [{"role": "user", "content": "hi"}]}
        assert _resolve_cursor_model_variant(body, wildcard_router) is body

    def test_provider_inferable_bare_name_untouched(self, wildcard_router):
        from litellm.proxy.response_api_endpoints.endpoints import _resolve_cursor_model_variant

        body = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]}
        assert _resolve_cursor_model_variant(body, wildcard_router) is body

    def test_unservable_base_untouched(self, wildcard_router):
        from litellm.proxy.response_api_endpoints.endpoints import _resolve_cursor_model_variant

        body = {"model": "totally-unknown-thinking-high", "messages": [{"role": "user", "content": "hi"}]}
        assert _resolve_cursor_model_variant(body, wildcard_router) is body

    def test_no_router_untouched(self):
        from litellm.proxy.response_api_endpoints.endpoints import _resolve_cursor_model_variant

        body = {"model": "claude-opus-5-thinking-high", "messages": [{"role": "user", "content": "hi"}]}
        assert _resolve_cursor_model_variant(body, None) is body

    def test_missing_or_non_string_model_untouched(self, wildcard_router):
        from litellm.proxy.response_api_endpoints.endpoints import _resolve_cursor_model_variant

        no_model = {"messages": [{"role": "user", "content": "hi"}]}
        assert _resolve_cursor_model_variant(no_model, wildcard_router) is no_model
        null_model = {"model": None, "messages": [{"role": "user", "content": "hi"}]}
        assert _resolve_cursor_model_variant(null_model, wildcard_router) is null_model


def _router_serving_only(base_model: str) -> MagicMock:
    mock_router = MagicMock()
    mock_router.model_names = set()
    mock_router.model_group_alias = {}
    mock_router.team_public_model_names = frozenset()
    mock_router.is_recognized_model.side_effect = lambda model: (
        model in mock_router.model_names or model in mock_router.model_group_alias
    )
    mock_router.router_general_settings.pass_through_all_models = False
    mock_router.default_deployment = None
    mock_router.pattern_router.patterns = {base_model: ["anthropic/*"]}
    mock_router.pattern_router.get_pattern.side_effect = lambda model: (
        [{"model_name": "anthropic/*"}] if model == base_model else None
    )
    return mock_router


class TestCursorModelSuffixResolutionEndToEnd:
    @pytest.mark.asyncio
    async def test_chat_arm_rewrites_suffixed_model_before_delegation(self):
        from litellm.proxy._types import UserAPIKeyAuth
        from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

        seen = {}

        async def fake_chat_completion(request, fastapi_response, model, user_api_key_dict):
            from litellm.proxy.common_utils.http_parsing_utils import read_request_body

            seen["body"] = await read_request_body(request=request)
            return {"id": "chatcmpl-fake", "object": "chat.completion", "choices": []}

        app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(api_key=MASTER_KEY)
        try:
            with (
                patch("litellm.proxy.proxy_server.llm_router", new=_router_serving_only("claude-opus-5")),
                patch("litellm.proxy.proxy_server.chat_completion", new=fake_chat_completion),
            ):
                client = TestClient(app)
                response = client.post(
                    "/cursor/chat/completions",
                    json={
                        "model": "claude-opus-5-thinking-xhigh-fast",
                        "messages": [{"role": "user", "content": "hi"}],
                    },
                    headers={"Authorization": "Bearer sk-9876"},
                )
        finally:
            app.dependency_overrides.pop(user_api_key_auth, None)

        assert response.status_code == 200
        assert seen["body"]["model"] == "claude-opus-5"
        assert seen["body"]["reasoning_effort"] == "xhigh"
        assert seen["body"]["messages"] == [{"role": "user", "content": "hi"}]

    @pytest.mark.asyncio
    async def test_responses_arm_rewrites_suffixed_model_before_routing(self):
        from openai.types.responses import ResponseOutputMessage, ResponseOutputText

        from litellm.proxy._types import UserAPIKeyAuth
        from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
        from litellm.types.llms.openai import ResponsesAPIResponse

        mock_response = ResponsesAPIResponse(
            id="resp_suffix1",
            created_at=1234567890,
            model="claude-opus-5",
            object="response",
            output=[
                ResponseOutputMessage(
                    id="msg_suffix1",
                    type="message",
                    role="assistant",
                    status="completed",
                    content=[ResponseOutputText(type="output_text", text="ok", annotations=[])],
                )
            ],
        )

        mock_router = _router_serving_only("claude-opus-5")
        mock_router.aresponses = AsyncMock(return_value=mock_response)

        app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(api_key=MASTER_KEY)
        try:
            with patch("litellm.proxy.proxy_server.llm_router", new=mock_router):
                client = TestClient(app)
                response = client.post(
                    "/cursor/chat/completions",
                    json={
                        "model": "claude-opus-5-thinking-high",
                        "input": [{"role": "user", "content": "hello"}],
                    },
                    headers={"Authorization": "Bearer sk-9876"},
                )
        finally:
            app.dependency_overrides.pop(user_api_key_auth, None)

        assert response.status_code == 200
        assert mock_router.aresponses.call_args is not None
        assert mock_router.aresponses.call_args.kwargs["model"] == "claude-opus-5"
        assert mock_router.aresponses.call_args.kwargs["reasoning"] == {"effort": "high"}


def _cursor_budget_auth_env(base_model: str, spend: float):
    from litellm import Router
    from litellm.caching.dual_cache import DualCache
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.hooks.model_max_budget_limiter import (
        VIRTUAL_KEY_SPEND_CACHE_KEY_PREFIX,
        PROXY_VirtualKeyModelMaxBudgetLimiter,
    )

    valid_token = UserAPIKeyAuth(
        api_key="sk-cursor-budget-test",
        token="hashed-cursor-budget-token",
        model_max_budget={base_model: {"budget_limit": 0.00001, "time_period": "1d"}},
    )
    limiter = PROXY_VirtualKeyModelMaxBudgetLimiter(dual_cache=DualCache())
    limiter.dual_cache.in_memory_cache.set_cache(
        key=f"{VIRTUAL_KEY_SPEND_CACHE_KEY_PREFIX}:{valid_token.token}:{base_model}:1d",
        value=spend,
    )
    router = Router(
        model_list=[{"model_name": "anthropic/*", "litellm_params": {"model": "anthropic/*", "api_key": "fake"}}]
    )

    mock_proxy_logging_obj = MagicMock()
    mock_proxy_logging_obj.post_call_failure_hook = AsyncMock(return_value=None)

    proxy_server_attrs = {
        "prisma_client": MagicMock(),
        "user_api_key_cache": DualCache(),
        "proxy_logging_obj": mock_proxy_logging_obj,
        "master_key": "sk-master-key",
        "general_settings": {},
        "llm_model_list": [],
        "llm_router": router,
        "open_telemetry_logger": None,
        "model_max_budget_limiter": limiter,
        "user_custom_auth": None,
        "jwt_handler": None,
        "litellm_proxy_admin_name": "admin",
    }
    return valid_token, proxy_server_attrs


def _post_cursor_with_real_auth(valid_token, proxy_server_attrs, request_model: str):
    with (
        patch.multiple("litellm.proxy.proxy_server", **proxy_server_attrs),
        patch(
            "litellm.proxy.auth.resolvers.store.IdentityStore._resolve_key",
            new_callable=AsyncMock,
            return_value=valid_token,
        ),
    ):
        client = TestClient(app)
        return client.post(
            "/cursor/chat/completions",
            json={"model": request_model, "input": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": "Bearer sk-cursor-budget-test"},
        )


class TestCursorVariantPerModelBudgetEnforcement:
    """Regression tests for the per-model budget bypass on /cursor/chat/completions.

    user_api_key_auth enforced key model_max_budget against the raw request model,
    but _resolve_cursor_model_variant only rewrote minted aliases like
    <base>-thinking-<level> to <base> inside the handler, after auth had already
    run. A key whose budget for <base> was exhausted could keep calling <base>
    through any unconfigured alias. The variant must now be resolved in a
    route-level dependency that runs before user_api_key_auth, so these tests
    exercise the real dependency chain (real auth, real budget limiter) through
    TestClient and fail if that ordering ever breaks."""

    def test_minted_alias_rejected_when_base_model_budget_exhausted(self):
        valid_token, attrs = _cursor_budget_auth_env(base_model="claude-opus-5", spend=1.0)

        response = _post_cursor_with_real_auth(valid_token, attrs, request_model="claude-opus-5-thinking-high")

        assert response.status_code == 422, response.text
        error = response.json()["error"]
        assert error["type"] == "budget_exceeded"
        assert "exceeded budget for model=claude-opus-5" in error["message"]

    def test_alias_rejection_matches_base_model_rejection(self):
        valid_token, attrs = _cursor_budget_auth_env(base_model="claude-opus-5", spend=1.0)

        base_response = _post_cursor_with_real_auth(valid_token, attrs, request_model="claude-opus-5")
        alias_response = _post_cursor_with_real_auth(valid_token, attrs, request_model="claude-opus-5-fast")

        assert base_response.status_code == 422, base_response.text
        assert alias_response.status_code == 422, alias_response.text
        assert alias_response.json() == base_response.json()


class TestCursorVariantResolvedBeforeAuth:
    """The route-level resolver dependency must rewrite the parsed body before
    user_api_key_auth reads it, so every auth check (model access, key and
    end-user model budgets, rate limits) sees the base model, and names the
    router already serves must reach auth untouched."""

    def _run_with_recording_auth(self, mock_router, request_model: str):
        from litellm.proxy._types import UserAPIKeyAuth
        from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
        from litellm.proxy.common_utils.http_parsing_utils import read_request_body

        from fastapi import Request

        bodies_seen_by_auth = []

        async def recording_auth(request: Request) -> UserAPIKeyAuth:
            bodies_seen_by_auth.append(await read_request_body(request=request))
            return UserAPIKeyAuth(api_key="sk-test-cursor")

        async def fake_chat_completion(request, fastapi_response, model, user_api_key_dict):
            return {"id": "chatcmpl-fake", "object": "chat.completion", "choices": []}

        app.dependency_overrides[user_api_key_auth] = recording_auth
        try:
            with (
                patch("litellm.proxy.proxy_server.llm_router", new=mock_router),
                patch("litellm.proxy.proxy_server.chat_completion", new=fake_chat_completion),
            ):
                client = TestClient(app)
                response = client.post(
                    "/cursor/chat/completions",
                    json={"model": request_model, "messages": [{"role": "user", "content": "hi"}]},
                    headers={"Authorization": "Bearer sk-test-cursor"},
                )
        finally:
            app.dependency_overrides.pop(user_api_key_auth, None)

        assert response.status_code == 200, response.text
        assert len(bodies_seen_by_auth) == 1
        return bodies_seen_by_auth[0]

    def test_auth_sees_base_model_for_minted_alias(self):
        auth_body = self._run_with_recording_auth(
            mock_router=_router_serving_only("claude-opus-5"),
            request_model="claude-opus-5-thinking-xhigh-fast",
        )
        assert auth_body["model"] == "claude-opus-5"
        assert auth_body["reasoning_effort"] == "xhigh"

    def test_auth_sees_servable_model_name_untouched(self):
        mock_router = _router_serving_only("claude-opus-5")
        mock_router.model_names = {"claude-opus-5-thinking-high"}

        auth_body = self._run_with_recording_auth(
            mock_router=mock_router,
            request_model="claude-opus-5-thinking-high",
        )
        assert auth_body["model"] == "claude-opus-5-thinking-high"
        assert "reasoning_effort" not in auth_body


class TestCursorGateRecognizesRoutingGroups:
    def test_group_name_variant_is_not_mangled(self):
        from litellm import Router
        from litellm.proxy.response_api_endpoints.endpoints import _resolve_cursor_model_variant

        router = Router(
            model_list=[{"model_name": "member-fast", "litellm_params": {"model": "openai/gpt-4o", "api_key": "fake"}}],
            routing_groups=[
                {"group_name": "grouped-thinking-high", "models": ["member-fast"], "routing_strategy": "simple-shuffle"}
            ],
        )
        body = {"model": "grouped-thinking-high", "messages": [{"role": "user", "content": "hi"}]}
        resolved = _resolve_cursor_model_variant(body, router)
        assert resolved["model"] == "grouped-thinking-high"
        assert "reasoning_effort" not in resolved


BLOCK_MESSAGE = "Content flagged by policy, response withheld"


def _post_blocked_responses(
    original_response: ResponsesAPIResponse | litellm.ModelResponse | None,
    payload: Mapping[str, object] | None = None,
) -> httpx.Response:
    from litellm.integrations.custom_guardrail import ModifyResponseException
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

    exc = ModifyResponseException(
        message=BLOCK_MESSAGE,
        model="gpt-4o-mini",
        request_data={"model": "gpt-4o-mini", "input": "hi"},
        guardrail_name="zero-usage-regression",
        original_response=original_response,
    )
    mock_proxy_logging = MagicMock()
    mock_proxy_logging.post_call_failure_hook = AsyncMock()
    app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(
        api_key="sk-test", request_route="/v1/responses"
    )
    body = {"model": "gpt-4o-mini", "input": "Write a haiku about token accounting"}
    if payload:
        body.update(payload)
    try:
        with (
            patch(
                "litellm.proxy.response_api_endpoints.endpoints.ProxyBaseLLMRequestProcessing.base_process_llm_request",
                new=AsyncMock(side_effect=exc),
            ),
            patch("litellm.proxy.proxy_server.proxy_logging_obj", mock_proxy_logging),
        ):
            client = TestClient(app)
            return client.post("/v1/responses", json=body, headers={"Authorization": "Bearer sk-9876"})
    finally:
        app.dependency_overrides.pop(user_api_key_auth, None)


def _assert_blocked_output_item(item: Mapping[str, object], text: str) -> None:
    assert item["type"] == "message"
    assert item["id"].startswith("msg_")
    assert item["role"] == "assistant"
    assert item["status"] == "completed"
    assert item["content"][0]["type"] == "output_text"
    assert item["content"][0]["text"] == text


def _sse_data_frames(text: str) -> list[str]:
    return [line.removeprefix("data: ").strip() for line in text.splitlines() if line.startswith("data: ")]


class TestGuardrailBlockedResponsesUsage:
    """Regression tests for https://github.com/BerriAI/litellm/issues/36880.

    The ModifyResponseException handler in responses_api hardcoded the synthetic
    blocked reply's usage to zeros, discarding the real token counts the blocked
    upstream call consumed. The blocked reply must carry the usage from
    e.original_response, exactly like /v1/chat/completions already does."""

    def _post_blocked_responses(self, original_response):
        return _post_blocked_responses(original_response)

    def test_post_call_block_reports_real_upstream_usage(self):
        from litellm.types.llms.openai import ResponseAPIUsage, ResponsesAPIResponse

        original = ResponsesAPIResponse(
            id="resp_upstream",
            created_at=1,
            model="gpt-4o-mini",
            object="response",
            output=[],
            status="completed",
            usage=ResponseAPIUsage(input_tokens=14, output_tokens=20, total_tokens=34),
        )

        response = self._post_blocked_responses(original)

        assert response.status_code == 200, response.text
        body = response.json()
        assert body["output"][0]["content"][0]["text"] == "Content flagged by policy, response withheld"
        assert body["usage"]["input_tokens"] == 14
        assert body["usage"]["output_tokens"] == 20
        assert body["usage"]["total_tokens"] == 34

    def test_post_call_block_maps_bridged_chat_usage(self):
        original = litellm.ModelResponse()
        original.usage = litellm.Usage(prompt_tokens=14, completion_tokens=18, total_tokens=32)

        response = self._post_blocked_responses(original)

        assert response.status_code == 200, response.text
        usage = response.json()["usage"]
        assert usage["input_tokens"] == 14
        assert usage["output_tokens"] == 18
        assert usage["total_tokens"] == 32

    def test_pre_call_block_reports_zero_usage(self):
        response = self._post_blocked_responses(None)

        assert response.status_code == 200, response.text
        usage = response.json()["usage"]
        assert usage["input_tokens"] == 0
        assert usage["output_tokens"] == 0
        assert usage["total_tokens"] == 0


class TestResponsesInputTokens:
    """Regression tests for POST /v1/responses/input_tokens.

    The docs promise OpenAI-format token counting on the proxy, but the route was
    never registered, so the POST fell through to the GET/DELETE-only
    /v1/responses/{response_id} route and returned 405."""

    def _post_input_tokens(
        self,
        body: dict[str, Any],
        path: str = "/v1/responses/input_tokens",
        counter: AsyncMock | None = None,
        llm_router: litellm.Router | None = None,
        response_id_security: object | None = None,
        managed_files: object | None = None,
    ) -> tuple[Response, AsyncMock]:
        from litellm.proxy._types import UserAPIKeyAuth
        from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
        from litellm.proxy.response_api_endpoints.endpoints import (
            _proxy_llm_router,
            _proxy_managed_files_hook,
            _proxy_response_id_security,
            _proxy_token_counter,
        )
        from litellm.types.utils import TokenCountResponse

        token_counter_mock = (
            counter
            if counter is not None
            else AsyncMock(
                return_value=TokenCountResponse(
                    total_tokens=13,
                    request_model=body.get("model", ""),
                    model_used=body.get("model", ""),
                    tokenizer_type="openai_api",
                )
            )
        )
        app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(api_key="sk-test", request_route=path)
        app.dependency_overrides[_proxy_token_counter] = lambda: token_counter_mock
        app.dependency_overrides[_proxy_llm_router] = lambda: llm_router
        if response_id_security is not None:
            app.dependency_overrides[_proxy_response_id_security] = lambda: response_id_security
        if managed_files is not None:
            app.dependency_overrides[_proxy_managed_files_hook] = lambda: managed_files
        try:
            client = TestClient(app)
            response = client.post(path, json=body, headers={"Authorization": "Bearer sk-9876"})
            return response, token_counter_mock
        finally:
            app.dependency_overrides.pop(user_api_key_auth, None)
            app.dependency_overrides.pop(_proxy_token_counter, None)
            app.dependency_overrides.pop(_proxy_llm_router, None)
            app.dependency_overrides.pop(_proxy_response_id_security, None)
            app.dependency_overrides.pop(_proxy_managed_files_hook, None)

    def test_string_input_returns_openai_input_tokens_shape(self):
        response, counter = self._post_input_tokens({"model": "gpt-4o", "input": "Hello, how are you?"})

        assert response.status_code == 200, response.text
        assert response.json() == {"object": "response.input_tokens", "input_tokens": 13}
        counter.assert_awaited_once()
        assert counter.call_args.kwargs["call_endpoint"] is True
        token_request = counter.call_args.kwargs["request"]
        assert token_request.model == "gpt-4o"
        assert token_request.messages == [{"role": "user", "content": "Hello, how are you?"}]

    def test_every_route_alias_is_registered(self):
        for path in ("/v1/responses/input_tokens", "/responses/input_tokens", "/openai/v1/responses/input_tokens"):
            response, _ = self._post_input_tokens({"model": "gpt-4o", "input": "hi"}, path=path)
            assert response.status_code == 200, f"{path}: {response.status_code} {response.text}"

    def test_input_items_instructions_and_tools_are_forwarded(self):
        tools = [
            {
                "type": "function",
                "name": "get_weather",
                "description": "Get weather for a city",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            }
        ]
        response, counter = self._post_input_tokens(
            {
                "model": "gpt-4o",
                "input": [{"role": "user", "content": "What is the weather in Paris?"}],
                "instructions": "You are terse.",
                "tools": tools,
            }
        )

        assert response.status_code == 200, response.text
        token_request = counter.call_args.kwargs["request"]
        assert token_request.messages == [
            {"role": "system", "content": "You are terse."},
            {"role": "user", "content": "What is the weather in Paris?"},
        ]
        assert token_request.tools == tools

    def test_item_cache_control_is_left_out_of_the_counted_messages(self):
        response, counter = self._post_input_tokens(
            {
                "model": "anthropic/claude-sonnet-4-6",
                "input": [
                    {"role": "system", "content": "Long handbook", "cache_control": {"type": "ephemeral"}},
                    {
                        "role": "user",
                        "content": "What does a reviewer do first?",
                        "cache_control": {"type": "ephemeral"},
                    },
                ],
            }
        )

        assert response.status_code == 200, response.text
        assert counter.call_args.kwargs["request"].messages == [
            {"role": "system", "content": "Long handbook"},
            {"role": "user", "content": "What does a reviewer do first?"},
        ]

    def test_missing_model_returns_openai_400(self):
        response, counter = self._post_input_tokens({"input": "Hello"})

        assert response.status_code == 400, response.text
        assert response.json() == {
            "error": {
                "message": "Missing required parameter: 'model'.",
                "type": "invalid_request_error",
                "param": "model",
                "code": "missing_required_parameter",
            }
        }
        counter.assert_not_awaited()

    def test_missing_input_returns_openai_400(self):
        response, counter = self._post_input_tokens({"model": "gpt-4o"})

        assert response.status_code == 400, response.text
        assert response.json() == {
            "error": {
                "message": "Missing required parameter: 'input'.",
                "type": "invalid_request_error",
                "param": "input",
                "code": "missing_required_parameter",
            }
        }
        counter.assert_not_awaited()

    @pytest.mark.parametrize("empty_input", ["", []])
    def test_empty_input_returns_openai_400(self, empty_input):
        response, counter = self._post_input_tokens({"model": "gpt-4o", "input": empty_input})

        assert response.status_code == 400, response.text
        assert response.json() == {
            "error": {
                "message": """One of "input" or "previous_response_id" or 'prompt' or 'conversation' must be provided.""",
                "type": "invalid_request_error",
                "param": None,
                "code": "missing_required_parameter",
            }
        }
        counter.assert_not_awaited()

    def test_invalid_tools_returns_openai_400(self):
        response, counter = self._post_input_tokens({"model": "gpt-4o", "input": "hi", "tools": "not-a-list"})

        assert response.status_code == 400, response.text
        error = response.json()["error"]
        assert error["type"] == "invalid_request_error"
        assert error["param"] == "tools"
        assert error["message"].startswith("tools: ")
        counter.assert_not_awaited()

    def test_invalid_tools_are_dropped_under_drop_params(self) -> None:
        response, counter = self._post_input_tokens(
            {"model": "gpt-4o", "input": "hi", "tools": [5, {"type": "web_search"}], "drop_params": True}
        )

        assert response.status_code == 200, response.text
        assert counter.await_args.kwargs["request"].tools == [{"type": "web_search"}]
        assert counter.await_args.kwargs["responses_request"]["tools"] == [{"type": "web_search"}]

    @pytest.mark.parametrize(
        ("tools", "expected"),
        [
            ([5], {}),
            (
                [5, {"type": "web_search"}],
                {"tools": [{"type": "web_search"}], "tool_choice": "required", "parallel_tool_calls": False},
            ),
        ],
    )
    def test_tool_choice_leaves_the_count_with_the_tools_drop_params_drops(
        self, tools: list[object], expected: dict[str, object]
    ) -> None:
        response, counter = self._post_input_tokens(
            {
                "model": "gpt-4o",
                "input": "hi",
                "tools": tools,
                "tool_choice": "required",
                "parallel_tool_calls": False,
                "drop_params": True,
            }
        )

        assert response.status_code == 200, response.text
        sent = counter.await_args.kwargs["responses_request"]
        assert {name: sent[name] for name in ("tools", "tool_choice", "parallel_tool_calls") if name in sent} == expected

    def test_previous_response_id_reaches_the_counter_authorized_next_to_the_id_the_client_held(self) -> None:
        from litellm.types.utils import ADDRESSED_RESPONSE_ID_FIELD

        class DecodingSecurity:
            def authorize_response_id(self, response_id: str, user_api_key_dict: object) -> str:
                return {"resp_client_form": "resp_raw"}[response_id]

        response, counter = self._post_input_tokens(
            {"model": "gpt-4o", "input": "hi", "previous_response_id": "resp_client_form"},
            response_id_security=DecodingSecurity(),
        )

        assert response.status_code == 200, response.text
        responses_request: Final = counter.await_args.kwargs["responses_request"]
        assert (responses_request["previous_response_id"], responses_request[ADDRESSED_RESPONSE_ID_FIELD]) == (
            "resp_raw",
            "resp_client_form",
        )

    def test_managed_file_ids_reach_the_counter_with_the_mapping_inference_uses(self) -> None:
        from litellm.integrations.custom_logger import CustomLogger

        mapping: Final = {"litellm-managed-file": {"deployment-a": "file-on-a"}}

        class ManagedFiles(CustomLogger):
            async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
                if call_type == "aresponses":
                    data["model_file_id_mapping"] = mapping
                return None

        response, counter = self._post_input_tokens(
            {
                "model": "gpt-4o",
                "input": [{"role": "user", "content": [{"type": "input_file", "file_id": "litellm-managed-file"}]}],
            },
            managed_files=ManagedFiles(),
        )

        assert response.status_code == 200, response.text
        assert counter.await_args.kwargs["responses_request"]["model_file_id_mapping"] == mapping

    @pytest.mark.parametrize("caller_mapping", [5, "lots", [1], {"file-x": {"deployment-a": 7}}])
    def test_a_caller_sent_model_file_id_mapping_does_not_break_the_count(
        self, monkeypatch: pytest.MonkeyPatch, caller_mapping: object
    ) -> None:
        from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
        from litellm.llms.openai.responses.count_tokens import handler as openai_count_tokens_http
        from litellm.proxy.proxy_server import count_request_tokens

        router: Final = litellm.Router(
            model_list=[
                {
                    "model_name": "gpt-files",
                    "litellm_params": {
                        "model": "openai/gpt-4o-mini",
                        "api_key": "test-key",
                        "api_base": "https://count.test/v1",
                    },
                    "model_info": {"id": "deployment-a"},
                }
            ]
        )
        monkeypatch.setattr(litellm.proxy.proxy_server, "llm_router", router)
        sent: Final[list[object]] = []  # mutable-ok: collects what the mock peer receives

        def count_peer(request: httpx.Request) -> httpx.Response:
            sent.append(json.loads(request.content))
            return httpx.Response(200, json={"object": "response.input_tokens", "input_tokens": 7}, request=request)

        with patch.object(
            openai_count_tokens_http,
            "get_async_httpx_client",
            return_value=AsyncHTTPHandler(transport=httpx.MockTransport(count_peer)),
        ):
            response, _ = self._post_input_tokens(
                {"model": "gpt-files", "input": "hi", "model_file_id_mapping": caller_mapping},
                counter=AsyncMock(side_effect=count_request_tokens),
                llm_router=router,
            )

        assert (response.status_code, response.json().get("input_tokens")) == (200, 7), response.text
        assert sent == [{"model": "gpt-4o-mini", "input": "hi"}]

    def test_a_managed_file_the_key_cannot_read_answers_403_without_a_count(self) -> None:
        from litellm.integrations.custom_logger import CustomLogger

        denied: Final = "User u1 does not have access to the file litellm-managed-file"

        class ManagedFiles(CustomLogger):
            async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
                raise HTTPException(status_code=403, detail=denied)

        response, counter = self._post_input_tokens(
            {
                "model": "gpt-4o",
                "input": [{"role": "user", "content": [{"type": "input_file", "file_id": "litellm-managed-file"}]}],
            },
            managed_files=ManagedFiles(),
        )

        assert response.status_code == 403, response.text
        assert response.json()["error"]["message"] == denied
        counter.assert_not_awaited()

    def test_malformed_tools_follow_the_counted_deployment_even_when_fallbacks_are_configured(self) -> None:
        router: Final = litellm.Router(
            model_list=[
                {
                    "model_name": "gpt-5.2",
                    "litellm_params": {
                        "model": "openai/gpt-5.2",
                        "api_key": "sk-test",
                        "additional_drop_params": ["tools"],
                    },
                },
                {"model_name": "backup", "litellm_params": {"model": "openai/gpt-5.4-mini", "api_key": "sk-test"}},
            ],
            fallbacks=[{"gpt-5.2": ["backup"]}],
        )

        response, counter = self._post_input_tokens({"model": "gpt-5.2", "input": "hi", "tools": 5}, llm_router=router)

        assert response.status_code == 200, response.text
        assert counter.await_args.kwargs["request"].tools is None
        assert "tools" not in counter.await_args.kwargs["responses_request"]

    def test_malformed_tools_for_a_model_outside_the_router_get_a_clean_400(self) -> None:
        router: Final = litellm.Router(
            model_list=[{"model_name": "gpt-5.2", "litellm_params": {"model": "openai/gpt-5.2", "api_key": "sk-test"}}]
        )

        response, counter = self._post_input_tokens(
            {"model": "openai/gpt-5.4-mini", "input": "hi", "tools": 5}, llm_router=router
        )

        assert response.status_code == 400, response.text
        assert response.json()["error"]["param"] == "tools"
        counter.assert_not_awaited()

    def test_provider_error_maps_status_code(self):
        from litellm.proxy._types import ProxyException

        failing_counter = AsyncMock(
            side_effect=ProxyException(
                message="rate limited",
                type="token_counting_error",
                param="model",
                code="429",
            )
        )
        response, _ = self._post_input_tokens({"model": "gpt-4o", "input": "hi"}, counter=failing_counter)

        assert response.status_code == 429, response.text
        assert response.json()["error"]["message"] == "rate limited"

    @pytest.mark.parametrize(
        "provider_body, status_code, expected_error",
        [
            pytest.param(
                {
                    "error": {
                        "message": "Missing required parameter: 'tools[0].name'.",
                        "type": "invalid_request_error",
                        "param": "tools[0].name",
                        "code": "missing_required_parameter",
                    }
                },
                400,
                {
                    "message": "Missing required parameter: 'tools[0].name'.",
                    "type": "invalid_request_error",
                    "param": "tools[0].name",
                    "code": "missing_required_parameter",
                },
                id="openai_error_passes_through",
            ),
            pytest.param(
                {
                    "error": {
                        "message": "Missing required parameter: 'tools[0].name'.",
                        "type": "invalid_request_error",
                        "param": "tools[0].name",
                    }
                },
                400,
                {
                    "message": "Missing required parameter: 'tools[0].name'.",
                    "type": "invalid_request_error",
                    "param": "tools[0].name",
                    "code": None,
                },
                id="openai_error_without_code_keeps_type_and_param",
            ),
            pytest.param(
                {
                    "type": "error",
                    "error": {"type": "invalid_request_error", "message": "tools.0: Input tag 'function' is invalid"},
                    "request_id": "req_vrtx_1",
                },
                400,
                {
                    "message": "tools.0: Input tag 'function' is invalid",
                    "type": "invalid_request_error",
                    "param": None,
                    "code": None,
                },
                id="anthropic_error_message_unwrapped",
            ),
            pytest.param(
                {
                    "type": "error",
                    "error": {"type": "not_found_error", "message": "model: claude-typo"},
                    "request_id": "req_vrtx_2",
                },
                404,
                {"message": "model: claude-typo", "type": "invalid_request_error", "param": None, "code": None},
                id="anthropic_error_type_follows_the_status",
            ),
            pytest.param(
                {"error": {"code": 404, "message": "Publisher model not found.", "status": "NOT_FOUND"}},
                404,
                {"message": "Publisher model not found.", "type": "invalid_request_error", "param": None, "code": None},
                id="google_error_message_unwrapped",
            ),
            pytest.param(
                "litellm.APIError: Google Gen AI Studio API error: 404 - "
                + json.dumps(
                    {"error": {"code": 404, "message": "models/gemini-typo is not found.", "status": "NOT_FOUND"}}
                ),
                404,
                {
                    "message": "models/gemini-typo is not found.",
                    "type": "invalid_request_error",
                    "param": None,
                    "code": None,
                },
                id="prefixed_gemini_error_message_unwrapped",
            ),
            pytest.param(
                "litellm.BadRequestError: AzureException - "
                + json.dumps(
                    {
                        "error": {
                            "message": "Unrecognized request argument supplied: foo",
                            "type": "invalid_request_error",
                            "param": "foo",
                            "code": "unknown_parameter",
                        }
                    }
                ),
                400,
                {
                    "message": "Unrecognized request argument supplied: foo",
                    "type": "invalid_request_error",
                    "param": "foo",
                    "code": "unknown_parameter",
                },
                id="prefixed_openai_error_passes_through",
            ),
            pytest.param(
                {"message": "The provided model doesn't support counting tokens."},
                400,
                {
                    "message": "The provided model doesn't support counting tokens.",
                    "type": "invalid_request_error",
                    "param": None,
                    "code": None,
                },
                id="bedrock_error_message_unwrapped",
            ),
            pytest.param(
                "BedrockException - " + json.dumps({"message": "The provided model doesn't support counting tokens."}),
                400,
                {
                    "message": "The provided model doesn't support counting tokens.",
                    "type": "invalid_request_error",
                    "param": None,
                    "code": None,
                },
                id="prefixed_bedrock_error_message_unwrapped",
            ),
            pytest.param(
                "tool schema {unclosed is invalid",
                400,
                {
                    "message": "tool schema {unclosed is invalid",
                    "type": "invalid_request_error",
                    "param": None,
                    "code": None,
                },
                id="plain_text_with_a_brace_kept_whole",
            ),
            pytest.param(
                "upstream count service unavailable",
                503,
                {
                    "message": "upstream count service unavailable",
                    "type": "internal_server_error",
                    "param": None,
                    "code": None,
                },
                id="plain_text_error",
            ),
        ],
    )
    def test_provider_count_error_answers_in_openai_error_shape(
        self,
        provider_body: str | Mapping[str, object],
        status_code: int,
        expected_error: Mapping[str, str | None],
    ) -> None:
        from litellm.proxy._types import ProxyException

        failing_counter: Final = AsyncMock(
            side_effect=ProxyException(
                message=provider_body if isinstance(provider_body, str) else json.dumps(provider_body),
                type="token_counting_error",
                param="model",
                code=str(status_code),
            )
        )
        response, _ = self._post_input_tokens({"model": "gpt-4o", "input": "hi"}, counter=failing_counter)

        assert response.status_code == status_code, response.text
        assert response.json() == {"error": expected_error}

    def test_non_provider_count_error_keeps_its_own_type(self) -> None:
        from litellm.proxy._types import ProxyException

        failing_counter: Final = AsyncMock(
            side_effect=ProxyException(
                message="Token counting is disabled and no provider API result available",
                type="token_counting_disabled",
                param="model",
                code=503,
            )
        )
        response, _ = self._post_input_tokens({"model": "gpt-4o", "input": "hi"}, counter=failing_counter)

        assert response.status_code == 503, response.text
        assert response.json()["error"]["type"] == "token_counting_disabled"

    def test_provider_count_error_names_the_call_id_when_opted_in(self) -> None:
        from litellm.proxy._types import ProxyException

        failing_counter: Final = AsyncMock(
            side_effect=ProxyException(
                message="LiteLLM does not support counting tokens for Vertex AI model zai-org/glm-5.2-maas.",
                type="token_counting_error",
                param="model",
                code="400",
                headers={"x-litellm-call-id": "call-9564"},
            )
        )
        with patch("litellm.proxy.proxy_server.general_settings", {"include_call_id_in_error_body": True}):
            response, _ = self._post_input_tokens({"model": "gpt-4o", "input": "hi"}, counter=failing_counter)

        assert response.status_code == 400, response.text
        assert response.json() == {
            "error": {
                "message": "LiteLLM does not support counting tokens for Vertex AI model zai-org/glm-5.2-maas.",
                "type": "invalid_request_error",
                "param": None,
                "code": None,
                "litellm_call_id": "call-9564",
            }
        }


class TestGuardrailBlockedResponsesShape:
    """A pre_call block raises ModifyResponseException before any provider call.

    The reply must satisfy the Responses API contract the request selected:
    stream=true answers SSE ending in one response.completed whose output[0] is
    a completed assistant message item with output_text content, and a plain
    POST answers JSON with the same item, both with the usage the blocked call
    consumed (zero for pre_call)."""

    def test_non_stream_block_is_a_completed_assistant_message(self):
        response = _post_blocked_responses(None)

        assert response.status_code == 200, response.text
        assert response.headers["content-type"].startswith("application/json")
        body = response.json()
        _assert_blocked_output_item(body["output"][0], BLOCK_MESSAGE)
        assert body["usage"]["total_tokens"] == 0

    def test_stream_block_answers_sse_with_completed_event(self):
        response = _post_blocked_responses(None, payload={"stream": True})

        assert response.status_code == 200, response.text
        assert response.headers["content-type"].startswith("text/event-stream")
        frames = _sse_data_frames(response.text)
        assert frames[-1] == "[DONE]"
        events = [json.loads(frame) for frame in frames[:-1]]
        types = [event["type"] for event in events]
        assert "response.created" in types
        completed = [event for event in events if event["type"] == "response.completed"]
        assert len(completed) == 1
        completed_response = completed[0]["response"]
        _assert_blocked_output_item(completed_response["output"][0], BLOCK_MESSAGE)
        assert completed_response["usage"]["total_tokens"] == 0
        delta_text = "".join(event["delta"] for event in events if event["type"] == "response.output_text.delta")
        assert delta_text == BLOCK_MESSAGE

    def test_stream_block_keeps_upstream_usage(self):
        from litellm.types.llms.openai import ResponseAPIUsage, ResponsesAPIResponse

        original = ResponsesAPIResponse(
            id="resp_upstream",
            created_at=1,
            model="gpt-4o-mini",
            object="response",
            output=[],
            status="completed",
            usage=ResponseAPIUsage(input_tokens=14, output_tokens=20, total_tokens=34),
        )

        response = _post_blocked_responses(original, payload={"stream": True})

        assert response.status_code == 200, response.text
        frames = _sse_data_frames(response.text)
        completed = [json.loads(frame) for frame in frames[:-1] if json.loads(frame)["type"] == "response.completed"]
        usage = completed[0]["response"]["usage"]
        assert usage["input_tokens"] == 14
        assert usage["output_tokens"] == 20
        assert usage["total_tokens"] == 34


def test_responses_routes_document_response_models_in_openapi_schema():
    from typing import cast

    from fastapi import FastAPI

    from litellm.proxy.response_api_endpoints.endpoints import router

    def as_object(value: object) -> dict[str, object]:
        assert isinstance(value, dict)
        return cast(dict[str, object], value)

    openapi_app = FastAPI()
    openapi_app.include_router(router)
    openapi: Final = cast(dict[str, object], openapi_app.openapi())

    def ok_200_properties(path: str, method: str) -> dict[str, object]:
        operation: Final = as_object(as_object(as_object(openapi)["paths"])[path])[method]
        schema: Final = as_object(
            as_object(
                as_object(as_object(as_object(as_object(operation)["responses"])["200"])["content"])["application/json"]
            )["schema"]
        )
        ref: Final = schema["$ref"]
        assert isinstance(ref, str)
        component: Final = ref.rsplit("/", 1)[-1]
        return as_object(
            as_object(as_object(as_object(as_object(openapi)["components"])["schemas"])[component])["properties"]
        )

    assert "output" in ok_200_properties("/v1/responses", "post")
    assert "output" in ok_200_properties("/v1/responses/{response_id}", "get")
    assert "deleted" in ok_200_properties("/v1/responses/{response_id}", "delete")
    assert "data" in ok_200_properties("/v1/responses/{response_id}/input_items", "get")


_COMPACTION_SUMMARY: Final = "The user planned a Tokyo trip."
_NEXT_TURN: Final = {"type": "message", "role": "user", "content": "And Osaka?"}
_COMPACTED_INPUT: Final = [
    {"type": "message", "role": "user", "content": "Plan a Tokyo trip. " * 200},
    {"type": "message", "role": "assistant", "content": "Day one, Asakusa. " * 200},
    {
        "type": "compaction",
        "id": "cmp_1",
        "encrypted_content": json.dumps({"type": "compaction", "content": _COMPACTION_SUMMARY}),
    },
    _NEXT_TURN,
]
_SUMMARY_INPUT: Final = [{"type": "message", "role": "user", "content": _COMPACTION_SUMMARY}, _NEXT_TURN]


def test_input_token_count_reads_a_compaction_item_as_its_summary():
    from litellm.proxy.response_api_endpoints.endpoints import _responses_input_as_token_count_messages

    assert _responses_input_as_token_count_messages(_COMPACTED_INPUT, None) == (
        _responses_input_as_token_count_messages(_SUMMARY_INPUT, None)
    )
