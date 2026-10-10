"""
Tests for AI Usage Chat module.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.proxy.management_endpoints.usage_endpoints.ai_usage_chat import (
    TOOL_HANDLERS,
    TOOLS_ADMIN,
    TOOLS_BASE,
    _build_system_prompt,
    _summarise_entity_data,
    _summarise_usage_data,
    stream_usage_ai_chat,
)
from litellm.utils import ModelResponse, ModelResponseStream

SAMPLE_AGGREGATED_RESPONSE = {
    "results": [
        {
            "date": "2025-01-15",
            "metrics": {
                "spend": 50.25,
                "prompt_tokens": 20000,
                "completion_tokens": 10000,
                "total_tokens": 30000,
                "api_requests": 500,
                "successful_requests": 480,
                "failed_requests": 20,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
            },
            "breakdown": {
                "models": {
                    "gpt-4": {
                        "metrics": {
                            "spend": 40.0,
                            "api_requests": 300,
                            "total_tokens": 25000,
                        },
                        "metadata": {},
                        "api_key_breakdown": {},
                    },
                },
                "providers": {
                    "openai": {
                        "metrics": {"spend": 50.25, "api_requests": 500},
                        "metadata": {},
                        "api_key_breakdown": {},
                    },
                },
                "api_keys": {
                    "sk-test123": {
                        "metrics": {"spend": 50.25},
                        "metadata": {"key_alias": "Production Key"},
                    },
                },
                "model_groups": {},
                "mcp_servers": {},
                "entities": {},
            },
        },
    ],
    "metadata": {
        "total_spend": 50.25,
        "total_api_requests": 500,
        "total_successful_requests": 480,
        "total_failed_requests": 20,
        "total_tokens": 30000,
    },
}

SAMPLE_TEAM_RESPONSE = {
    "results": [
        {
            "date": "2025-01-15",
            "metrics": {"spend": 100.0, "api_requests": 1000, "total_tokens": 50000},
            "breakdown": {
                "entities": {
                    "team-1": {
                        "metrics": {
                            "spend": 60.0,
                            "api_requests": 600,
                            "total_tokens": 30000,
                        },
                        "metadata": {"alias": "Engineering"},
                        "api_key_breakdown": {},
                    },
                    "team-2": {
                        "metrics": {
                            "spend": 40.0,
                            "api_requests": 400,
                            "total_tokens": 20000,
                        },
                        "metadata": {"alias": "Marketing"},
                        "api_key_breakdown": {},
                    },
                },
                "models": {},
                "providers": {},
                "api_keys": {},
                "model_groups": {},
                "mcp_servers": {},
            },
        },
    ],
    "metadata": {"total_spend": 100.0, "total_api_requests": 1000},
}


class TestToolSchemas:
    def test_admin_tools_include_all(self):
        assert len(TOOLS_ADMIN) == 3
        names = {t["function"]["name"] for t in TOOLS_ADMIN}
        assert "get_usage_data" in names
        assert "get_team_usage_data" in names
        assert "get_tag_usage_data" in names

    def test_base_tools_restricted_to_usage_only(self):
        assert len(TOOLS_BASE) == 1
        assert TOOLS_BASE[0]["function"]["name"] == "get_usage_data"

    def test_admin_prompt_mentions_all_tools(self):
        prompt = _build_system_prompt(is_admin=True)
        assert "get_usage_data" in prompt
        assert "get_team_usage_data" in prompt
        assert "get_tag_usage_data" in prompt

    def test_non_admin_prompt_only_mentions_usage_tool(self):
        prompt = _build_system_prompt(is_admin=False)
        assert "get_usage_data" in prompt
        assert "get_team_usage_data" not in prompt
        assert "get_tag_usage_data" not in prompt

    def test_system_prompt_includes_todays_date(self):
        from datetime import date

        prompt = _build_system_prompt(is_admin=True)
        assert date.today().isoformat() in prompt


class TestSummariseUsageData:
    def test_summarise_includes_totals(self):
        summary = _summarise_usage_data(SAMPLE_AGGREGATED_RESPONSE)
        assert "$50.25" in summary
        assert "500" in summary

    def test_summarise_includes_models(self):
        summary = _summarise_usage_data(SAMPLE_AGGREGATED_RESPONSE)
        assert "gpt-4" in summary

    def test_summarise_includes_providers(self):
        summary = _summarise_usage_data(SAMPLE_AGGREGATED_RESPONSE)
        assert "openai" in summary

    def test_summarise_handles_empty_data(self):
        empty = {"results": [], "metadata": {}}
        summary = _summarise_usage_data(empty)
        assert "no data" in summary.lower()


class TestSummariseEntityData:
    def test_team_summary_includes_teams(self):
        summary = _summarise_entity_data(SAMPLE_TEAM_RESPONSE, "Team")
        assert "Engineering" in summary
        assert "Marketing" in summary
        assert "$60.0" in summary
        assert "$40.0" in summary

    def test_team_summary_empty(self):
        empty = {"results": [], "metadata": {}}
        summary = _summarise_entity_data(empty, "Team")
        assert "No Team usage data" in summary


class FakeUsageCompletion:
    def __init__(self, response, stream):
        self.response = response
        self.stream_response = stream

    async def complete(self, model, messages, tools):
        return self.response

    async def stream(self, model, messages):
        async for chunk in self.stream_response:
            yield chunk


class TestStreamUsageAiChat:
    @pytest.mark.asyncio
    async def test_stream_emits_status_events(self):
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_123"
        mock_tool_call.function.name = "get_usage_data"
        mock_tool_call.function.arguments = json.dumps(
            {
                "start_date": "2025-01-01",
                "end_date": "2025-01-31",
            }
        )

        mock_first_response = MagicMock()
        mock_first_response.choices = [MagicMock()]
        mock_first_response.choices[0].message.tool_calls = [mock_tool_call]
        mock_first_response.choices[0].message.model_dump.return_value = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {
                        "name": "get_usage_data",
                        "arguments": '{"start_date":"2025-01-01","end_date":"2025-01-31"}',
                    },
                }
            ],
        }

        async def mock_stream():
            chunk = ModelResponseStream(
                model="test-model",
                choices=[{"index": 0, "delta": {"content": "Total spend is $50.25"}, "finish_reason": "stop"}],
            )
            yield chunk

        with (
            patch(
                "litellm.proxy.management_endpoints.usage_endpoints.ai_usage_chat._fetch_usage_data",
                new_callable=AsyncMock,
            ) as mock_fetch,
        ):
            completion = FakeUsageCompletion(
                ModelResponse(choices=[{"message": mock_first_response.choices[0].message.model_dump()}]), mock_stream()
            )
            mock_fetch.return_value = SAMPLE_AGGREGATED_RESPONSE

            events = []
            async for event in stream_usage_ai_chat(
                completion=completion,
                messages=[{"role": "user", "content": "What is my total spend?"}],
                model="gpt-4o-mini",
                user_id="user-123",
                is_admin=True,
            ):
                events.append(json.loads(event.replace("data: ", "").strip()))

            status_events = [e for e in events if e["type"] == "status"]
            tool_call_events = [e for e in events if e["type"] == "tool_call"]
            chunk_events = [e for e in events if e["type"] == "chunk"]
            done_events = [e for e in events if e["type"] == "done"]

            assert len(status_events) >= 1
            assert "Thinking" in status_events[0]["message"]
            assert len(tool_call_events) >= 1
            assert tool_call_events[0]["tool_name"] == "get_usage_data"
            assert tool_call_events[0]["status"] in ("running", "complete")
            assert len(chunk_events) >= 1
            assert len(done_events) == 1

    @pytest.mark.asyncio
    async def test_stream_handles_team_tool(self):
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_team"
        mock_tool_call.function.name = "get_team_usage_data"
        mock_tool_call.function.arguments = json.dumps(
            {
                "start_date": "2025-01-01",
                "end_date": "2025-01-31",
            }
        )

        mock_first_response = MagicMock()
        mock_first_response.choices = [MagicMock()]
        mock_first_response.choices[0].message.tool_calls = [mock_tool_call]
        mock_first_response.choices[0].message.model_dump.return_value = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_team",
                    "type": "function",
                    "function": {
                        "name": "get_team_usage_data",
                        "arguments": '{"start_date":"2025-01-01","end_date":"2025-01-31"}',
                    },
                }
            ],
        }

        async def mock_stream():
            chunk = ModelResponseStream(
                model="test-model",
                choices=[{"index": 0, "delta": {"content": "Engineering is the top team."}, "finish_reason": "stop"}],
            )
            yield chunk

        with (
            patch(
                "litellm.proxy.management_endpoints.usage_endpoints.ai_usage_chat._fetch_team_usage_data",
                new_callable=AsyncMock,
            ) as mock_fetch,
        ):
            completion = FakeUsageCompletion(
                ModelResponse(choices=[{"message": mock_first_response.choices[0].message.model_dump()}]), mock_stream()
            )
            mock_fetch.return_value = SAMPLE_TEAM_RESPONSE

            events = []
            async for event in stream_usage_ai_chat(
                completion=completion,
                messages=[{"role": "user", "content": "Which team spends the most?"}],
                model="gpt-4o-mini",
                is_admin=True,
            ):
                events.append(json.loads(event.replace("data: ", "").strip()))

            chunk_events = [e for e in events if e["type"] == "chunk"]
            assert len(chunk_events) >= 1
            assert "Engineering" in chunk_events[0]["content"]

    @pytest.mark.asyncio
    async def test_stream_handles_error(self):
        completion = MagicMock()
        completion.complete = AsyncMock(side_effect=Exception("LLM error"))
        with patch.dict(TOOL_HANDLERS, {}):
            events = []
            async for event in stream_usage_ai_chat(
                completion=completion,
                messages=[{"role": "user", "content": "test"}],
            ):
                events.append(json.loads(event.replace("data: ", "").strip()))

            error_events = [e for e in events if e["type"] == "error"]
            assert len(error_events) == 1
            assert "internal error" in error_events[0]["message"].lower()

    @pytest.mark.asyncio
    async def test_non_admin_enforces_user_id(self):
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_456"
        mock_tool_call.function.name = "get_usage_data"
        mock_tool_call.function.arguments = json.dumps(
            {
                "start_date": "2025-01-01",
                "end_date": "2025-01-31",
                "user_id": "other-user",
            }
        )

        mock_first_response = MagicMock()
        mock_first_response.choices = [MagicMock()]
        mock_first_response.choices[0].message.tool_calls = [mock_tool_call]
        mock_first_response.choices[0].message.model_dump.return_value = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_456",
                    "type": "function",
                    "function": {
                        "name": "get_usage_data",
                        "arguments": '{"start_date":"2025-01-01","end_date":"2025-01-31","user_id":"other-user"}',
                    },
                }
            ],
        }

        async def mock_stream():
            chunk = ModelResponseStream(
                model="test-model", choices=[{"index": 0, "delta": {"content": "Data."}, "finish_reason": "stop"}]
            )
            yield chunk

        mock_fetch = AsyncMock(return_value=SAMPLE_AGGREGATED_RESPONSE)

        with (
            patch.dict(
                "litellm.proxy.management_endpoints.usage_endpoints.ai_usage_chat.TOOL_HANDLERS",
                {
                    "get_usage_data": {
                        "fetch": mock_fetch,
                        "summarise": _summarise_usage_data,
                        "label": "global usage data",
                    }
                },
            ),
        ):
            completion = FakeUsageCompletion(
                ModelResponse(choices=[{"message": mock_first_response.choices[0].message.model_dump()}]), mock_stream()
            )

            events = []
            async for event in stream_usage_ai_chat(
                completion=completion,
                messages=[{"role": "user", "content": "Show data"}],
                model="gpt-4o-mini",
                user_id="my-user-id",
                is_admin=False,
            ):
                events.append(event)

            mock_fetch.assert_called_once_with(
                start_date="2025-01-01",
                end_date="2025-01-31",
                user_id="my-user-id",
            )


class TestUsageAiChatServiceAccountGuard:
    """
    Security regression: a non-admin caller with user_id=None (service-account
    key) must be rejected at the endpoint boundary, before any tool dispatch.
    """

    @pytest.mark.asyncio
    async def test_non_admin_with_user_id_none_is_rejected(self):
        from fastapi import HTTPException

        from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
        from litellm.proxy.management_endpoints.usage_endpoints.endpoints import (
            ChatMessage,
            UsageAIChatRequest,
            usage_ai_chat,
        )

        service_account_key = UserAPIKeyAuth(
            user_id=None,
            user_role=LitellmUserRoles.INTERNAL_USER,
        )
        request = MagicMock()
        body = UsageAIChatRequest(
            messages=[ChatMessage(role="user", content="hi")],
            model="gpt-4o-mini",
        )

        with pytest.raises(HTTPException) as exc_info:
            await usage_ai_chat(
                data=body,
                request=request,
                user_api_key_dict=service_account_key,
            )

        assert exc_info.value.status_code == 403
        assert "Service-account keys" in str(exc_info.value.detail)

    def test_resolve_fetch_kwargs_tripwire_fires_on_none_user_id(self):
        """
        Defense-in-depth: if a future endpoint forgets the entry guard and
        a non-admin caller with user_id=None reaches _resolve_fetch_kwargs,
        the tripwire must fire rather than issuing an unscoped query.
        """
        from litellm.proxy.management_endpoints.usage_endpoints.ai_usage_chat import (
            _resolve_fetch_kwargs,
        )

        with pytest.raises(ValueError, match="Non-admin caller has user_id=None; refusing to issue an") as exc_info:
            _resolve_fetch_kwargs(
                fn_name="get_usage_data",
                fn_args={"start_date": "2025-01-01", "end_date": "2025-01-31"},
                user_id=None,
                is_admin=False,
            )
        assert "Endpoint-level guard missing" in str(exc_info.value)


class TestUsageAiChatKeepalive:
    async def _collect_endpoint_body(self, monkeypatch, interval, delay=0.3) -> tuple[list[bytes], dict]:
        import asyncio

        from fastapi.responses import StreamingResponse

        import litellm
        from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
        from litellm.proxy.management_endpoints.usage_endpoints.endpoints import (
            ChatMessage,
            UsageAIChatRequest,
            usage_ai_chat,
        )

        monkeypatch.setattr(litellm, "sse_keepalive_ping_interval_seconds", interval)

        async def slow_acompletion(*args):
            await asyncio.sleep(delay)
            response = ModelResponse(choices=[{"message": {"role": "assistant", "content": "Total spend is $50.25"}}])
            return response

        completion = MagicMock()
        completion.complete = AsyncMock(side_effect=slow_acompletion)
        with patch(
            "litellm.proxy.management_endpoints.usage_endpoints.endpoints.ProxyUsageChatCompletion",
            return_value=completion,
        ):
            response = await usage_ai_chat(
                data=UsageAIChatRequest(messages=[ChatMessage(role="user", content="hi")], model="gpt-4o-mini"),
                request=MagicMock(),
                user_api_key_dict=UserAPIKeyAuth(user_id="admin", user_role=LitellmUserRoles.PROXY_ADMIN),
            )
            assert isinstance(response, StreamingResponse)
            chunks = [chunk if isinstance(chunk, bytes) else chunk.encode() async for chunk in response.body_iterator]
        return chunks, dict(response.headers)

    @pytest.mark.asyncio
    async def test_endpoint_pings_while_the_planning_completion_is_still_running(self, monkeypatch):
        chunks, headers = await self._collect_endpoint_body(monkeypatch, interval=0.05)

        assert headers["content-type"].startswith("text/event-stream")
        assert headers["cache-control"] == "no-cache"
        assert headers["x-accel-buffering"] == "no"
        assert chunks[0].startswith(b'data: {"type": "status"')
        assert chunks[1] == b": ping\n\n"
        assert chunks.count(b": ping\n\n") >= 3
        assert b'"content": "Total spend is $50.25"' in b"".join(chunks)
        assert chunks[-1] == b'data: {"type": "done"}\n\n'

    @pytest.mark.asyncio
    async def test_endpoint_stream_is_untouched_while_keepalives_are_unconfigured(self, monkeypatch):
        chunks, _ = await self._collect_endpoint_body(monkeypatch, interval=None, delay=0.15)

        assert b": ping\n\n" not in chunks
        assert chunks[0].startswith(b'data: {"type": "status"')
        assert chunks[-1] == b'data: {"type": "done"}\n\n'


class TestProxyUsageChatCompletion:
    @pytest.mark.asyncio
    async def test_planning_and_streaming_use_authenticated_proxy_processing(self):
        from fastapi import Request
        from fastapi.responses import StreamingResponse

        from litellm.proxy._types import UserAPIKeyAuth
        from litellm.proxy.management_endpoints.usage_endpoints.endpoints import ProxyUsageChatCompletion
        from litellm.utils import ModelResponse

        auth = UserAPIKeyAuth(user_id="caller", team_id="team", models=["configured-alias"])
        request = Request({"type": "http", "method": "POST", "path": "/usage/ai/chat", "headers": []})
        completion = ProxyUsageChatCompletion(request=request, auth=auth)
        messages = [{"role": "user", "content": "spend"}]
        response = ModelResponse(choices=[{"message": {"role": "assistant", "content": "planning"}}])

        async def stream():
            yield 'data: {"choices":[{"index":0,"delta":{"content":"answer"}}]}\n\n'
            yield "data: [DONE]\n\n"

        checks = AsyncMock()
        process = AsyncMock(side_effect=[response, StreamingResponse(stream())])
        with (
            patch("litellm.proxy.auth.user_api_key_auth.run_centralized_common_checks", checks),
            patch("litellm.proxy.common_request_processing.ProxyBaseLLMRequestProcessing") as processor,
        ):
            processor.return_value.base_process_llm_request = process
            processor.return_value.data = {}
            assert await completion.complete("configured-alias", messages, []) is response
            chunks = [chunk async for chunk in completion.stream("configured-alias", messages)]

        assert chunks[0].choices[0].delta.content == "answer"
        assert checks.await_count == 2
        assert process.await_count == 2
        for call in checks.await_args_list:
            assert call.kwargs["route"] == "/chat/completions"
            assert call.kwargs["user_api_key_auth_obj"].user_id == auth.user_id
            assert call.kwargs["user_api_key_auth_obj"].team_id == auth.team_id
            assert call.kwargs["user_api_key_auth_obj"] is not auth
            assert call.kwargs["request"].scope["path"] == "/chat/completions"
            assert call.kwargs["request_data"]["drop_params"] is True
        assert processor.call_args_list[0].kwargs["data"]["model"] == "configured-alias"
        assert "tools" in processor.call_args_list[0].kwargs["data"]
        assert processor.call_args_list[1].kwargs["data"]["tools"]
        assert processor.call_args_list[1].kwargs["data"]["stream"] is True

    @pytest.mark.asyncio
    async def test_access_denial_never_reaches_provider(self):
        from fastapi import Request

        from litellm.proxy._types import UserAPIKeyAuth
        from litellm.proxy.management_endpoints.usage_endpoints.endpoints import ProxyUsageChatCompletion

        completion = ProxyUsageChatCompletion(
            request=Request({"type": "http", "method": "POST", "path": "/usage/ai/chat", "headers": []}),
            auth=UserAPIKeyAuth(user_id="caller", models=["allowed"]),
        )
        from litellm.proxy._types import ModelAccessDeniedProxyException

        with (
            patch("litellm.proxy.common_request_processing.ProxyBaseLLMRequestProcessing") as processor,
            patch("litellm.proxy.proxy_server.proxy_logging_obj") as logging,
        ):
            logging.post_call_failure_hook = AsyncMock()
            processor.return_value.base_process_llm_request = AsyncMock()
            with pytest.raises(ModelAccessDeniedProxyException):
                await completion.complete("denied", [{"role": "user", "content": "hi"}], [])
            processor.return_value.base_process_llm_request.assert_not_awaited()


@pytest.mark.asyncio
async def test_final_stream_executes_followup_tool_and_emits_answer():
    from litellm.proxy.management_endpoints.usage_endpoints.ai_usage_chat import _stream_final_response

    histories = []

    class Completion:
        async def stream(self, model, messages):
            histories.append(tuple(messages))
            if len(histories) == 1:
                yield ModelResponseStream(
                    model="test-model",
                    choices=[
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "followup",
                                        "type": "function",
                                        "function": {
                                            "name": "get_usage_data",
                                            "arguments": json.dumps(
                                                {
                                                    "start_date": "2025-01-01",
                                                    "end_date": "2025-01-31",
                                                    "user_id": "other-user",
                                                }
                                            ),
                                        },
                                    }
                                ]
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                )
            else:
                yield ModelResponseStream(
                    model="test-model",
                    choices=[{"index": 0, "delta": {"content": "Final answer"}, "finish_reason": "stop"}],
                )

    fetch = AsyncMock(return_value=SAMPLE_AGGREGATED_RESPONSE)
    with patch.dict(
        TOOL_HANDLERS,
        {"get_usage_data": {"fetch": fetch, "summarise": _summarise_usage_data, "label": "global usage data"}},
    ):
        events = [
            json.loads(event.removeprefix("data: "))
            async for event in _stream_final_response(
                "alias", [{"role": "user", "content": "spend"}], Completion(), "own-user", False
            )
        ]
    fetch.assert_awaited_once_with(start_date="2025-01-01", end_date="2025-01-31", user_id="own-user")
    assert histories[1][-1]["role"] == "tool"
    assert histories[1][-1]["tool_call_id"] == "followup"
    assert {"type": "chunk", "content": "Final answer"} in events


@pytest.mark.asyncio
async def test_empty_final_stream_is_not_success():
    from fastapi import HTTPException

    from litellm.proxy.management_endpoints.usage_endpoints.ai_usage_chat import _stream_final_response

    class Completion:
        async def stream(self, model, messages):
            yield ModelResponseStream(model="test-model", choices=[{"index": 0, "delta": {}, "finish_reason": "stop"}])

    with pytest.raises(HTTPException):
        _ = [event async for event in _stream_final_response("alias", [], Completion())]


@pytest.mark.asyncio
async def test_empty_planning_answer_emits_error_not_done():
    class Completion:
        async def complete(self, model, messages, tools):
            return ModelResponse(choices=[{"message": {"role": "assistant", "content": None}}])

    events = [
        json.loads(event.removeprefix("data: "))
        async for event in stream_usage_ai_chat([{"role": "user", "content": "spend"}], completion=Completion())
    ]
    assert events[-1]["type"] == "error"
    assert not any(event["type"] == "done" for event in events)


@pytest.mark.asyncio
async def test_adapter_close_closes_body_and_unstarted_upstream():
    from fastapi import Request

    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.common_request_processing import _UpstreamClosingStreamingResponse
    from litellm.proxy.management_endpoints.usage_endpoints.endpoints import ProxyUsageChatCompletion

    closed = []

    async def body():
        try:
            yield 'data: {"choices":[{"index":0,"delta":{"content":"answer"}}]}\n\n'
        finally:
            closed.append("body")

    class Upstream:
        async def aclose(self):
            closed.append("upstream")

    response = _UpstreamClosingStreamingResponse(body(), upstream_generator=Upstream())
    with (
        patch("litellm.proxy.auth.user_api_key_auth.run_centralized_common_checks", AsyncMock()),
        patch("litellm.proxy.common_request_processing.ProxyBaseLLMRequestProcessing") as processor,
    ):
        processor.return_value.base_process_llm_request = AsyncMock(return_value=response)
        completion = ProxyUsageChatCompletion(
            Request({"type": "http", "method": "POST", "path": "/usage/ai/chat", "headers": []}),
            UserAPIKeyAuth(user_id="caller", models=["alias"]),
        )
        stream = completion.stream("alias", [])
        assert (await anext(stream)).choices[0].delta.content == "answer"
        await stream.aclose()
    assert closed == ["body", "upstream"]


@pytest.mark.asyncio
async def test_final_response_close_closes_completion_stream():
    from litellm.proxy.management_endpoints.usage_endpoints.ai_usage_chat import _stream_final_response

    closed = []

    class Completion:
        async def stream(self, model, messages):
            try:
                yield ModelResponseStream(model="test", choices=[{"index": 0, "delta": {"content": "answer"}}])
            finally:
                closed.append(True)

    stream = _stream_final_response("alias", [], Completion())
    await anext(stream)
    assert "answer" in await anext(stream)
    await stream.aclose()
    assert closed == [True]


@pytest.mark.asyncio
async def test_followup_tool_limit_rejects_another_round():
    from fastapi import HTTPException

    from litellm.proxy.management_endpoints.usage_endpoints.ai_usage_chat import _stream_final_response

    class Completion:
        async def stream(self, model, messages):
            yield ModelResponseStream(
                model="test",
                choices=[
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "tool",
                                    "type": "function",
                                    "function": {"name": "get_usage_data", "arguments": "{}"},
                                }
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
            )

    with pytest.raises(HTTPException, match="tool-call limit"):
        _ = [event async for event in _stream_final_response("alias", [], Completion(), remaining_tool_rounds=0)]


@pytest.mark.asyncio
async def test_fragmented_followup_arguments_preserve_tool_id_and_user_scope():
    from litellm.proxy.management_endpoints.usage_endpoints.ai_usage_chat import _stream_final_response

    history = []

    class Completion:
        async def stream(self, model, messages):
            history.append(tuple(messages))
            if len(history) == 1:
                for delta in (
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "fragmented",
                                "type": "function",
                                "function": {"name": "get_usage_data", "arguments": '{"start_date":"2025-01-01",'},
                            }
                        ]
                    },
                    {
                        "tool_calls": [
                            {"index": 0, "function": {"arguments": '"end_date":"2025-01-31","user_id":"other-user"}'}}
                        ]
                    },
                ):
                    yield ModelResponseStream(model="test", choices=[{"index": 0, "delta": delta}])
            else:
                yield ModelResponseStream(
                    model="test", choices=[{"index": 0, "delta": {"content": "Scoped answer"}, "finish_reason": "stop"}]
                )

    fetch = AsyncMock(return_value=SAMPLE_AGGREGATED_RESPONSE)
    with patch.dict(
        TOOL_HANDLERS, {"get_usage_data": {"fetch": fetch, "summarise": _summarise_usage_data, "label": "usage"}}
    ):
        events = [
            json.loads(event.removeprefix("data: "))
            async for event in _stream_final_response("alias", [], Completion(), user_id="own-user")
        ]
    assert history[-1][-1]["tool_call_id"] == "fragmented"
    assert events[-1] == {"type": "chunk", "content": "Scoped answer"}
    fetch.assert_awaited_once_with(start_date="2025-01-01", end_date="2025-01-31", user_id="own-user")


@pytest.mark.asyncio
async def test_planning_cancellation_releases_inner_reservation():
    import asyncio

    from fastapi import Request

    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.management_endpoints.usage_endpoints.endpoints import ProxyUsageChatCompletion

    reservations = []

    async def checks(user_api_key_auth_obj, request, **kwargs):
        user_api_key_auth_obj.budget_reservation = {"reserved_cost": 1.0}
        request.state.budget_reservation = user_api_key_auth_obj.budget_reservation
        reservations.append(user_api_key_auth_obj.budget_reservation)

    release = AsyncMock()
    request = Request({"type": "http", "method": "POST", "path": "/usage/ai/chat", "headers": []})
    completion = ProxyUsageChatCompletion(request, UserAPIKeyAuth(user_id="caller", models=["alias"]))
    with (
        patch("litellm.proxy.auth.user_api_key_auth.run_centralized_common_checks", checks),
        patch("litellm.proxy.common_request_processing.ProxyBaseLLMRequestProcessing") as processor,
        patch("litellm.proxy.spend_tracking.budget_reservation.release_budget_reservation_on_cancel", release),
    ):
        processor.return_value.base_process_llm_request = AsyncMock(side_effect=asyncio.CancelledError())
        with pytest.raises(asyncio.CancelledError):
            await completion.complete("alias", [], [])
    assert reservations == [{"reserved_cost": 1.0}]
    release.assert_awaited_once_with(reservations[0])


@pytest.mark.asyncio
@pytest.mark.parametrize("followup_round", [False, True])
async def test_outer_chat_close_closes_provider_before_returning(followup_round):
    closed = []
    rounds = []
    tool_message = {
        "role": "assistant",
        "tool_calls": [{"id": "unknown", "type": "function", "function": {"name": "unknown", "arguments": "{}"}}],
    }

    class Completion:
        async def complete(self, model, messages, tools):
            return ModelResponse(choices=[{"message": tool_message}])

        async def stream(self, model, messages):
            rounds.append(True)
            try:
                if followup_round and len(rounds) == 1:
                    yield ModelResponseStream(
                        model="test",
                        choices=[
                            {
                                "index": 0,
                                "delta": {"tool_calls": [{"index": 0, **tool_message["tool_calls"][0]}]},
                                "finish_reason": "tool_calls",
                            }
                        ],
                    )
                else:
                    yield ModelResponseStream(model="test", choices=[{"index": 0, "delta": {"content": "answer"}}])
            finally:
                closed.append(len(rounds))

    stream = stream_usage_ai_chat([{"role": "user", "content": "spend"}], completion=Completion())
    async for event in stream:
        if json.loads(event.removeprefix("data: "))["type"] == "chunk":
            break
    await stream.aclose()
    assert closed == ([1, 2] if followup_round else [1])
