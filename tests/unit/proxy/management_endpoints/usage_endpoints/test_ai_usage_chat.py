"""
Tests for AI Usage Chat module.
"""

import json
from collections.abc import AsyncIterator
from typing import Final
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from litellm.proxy.management_endpoints.usage_endpoints.ai_usage_chat import (
    TOOL_HANDLERS,
    TOOLS_ADMIN,
    TOOLS_BASE,
    _build_system_prompt,
    _summarise_entity_data,
    _summarise_usage_data,
    stream_usage_ai_chat,
)


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
            chunk = MagicMock()
            chunk.choices = [MagicMock()]
            chunk.choices[0].delta.content = "Total spend is $50.25"
            yield chunk

        with (
            patch(
                "litellm.proxy.management_endpoints.usage_endpoints.ai_usage_chat.litellm"
            ) as mock_litellm,
            patch(
                "litellm.proxy.management_endpoints.usage_endpoints.ai_usage_chat._fetch_usage_data",
                new_callable=AsyncMock,
            ) as mock_fetch,
        ):
            mock_litellm.acompletion = AsyncMock(
                side_effect=[
                    mock_first_response,
                    mock_stream(),
                ]
            )
            mock_fetch.return_value = SAMPLE_AGGREGATED_RESPONSE

            events = []
            async for event in stream_usage_ai_chat(
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
            chunk = MagicMock()
            chunk.choices = [MagicMock()]
            chunk.choices[0].delta.content = "Engineering is the top team."
            yield chunk

        with (
            patch(
                "litellm.proxy.management_endpoints.usage_endpoints.ai_usage_chat.litellm"
            ) as mock_litellm,
            patch(
                "litellm.proxy.management_endpoints.usage_endpoints.ai_usage_chat._fetch_team_usage_data",
                new_callable=AsyncMock,
            ) as mock_fetch,
        ):
            mock_litellm.acompletion = AsyncMock(
                side_effect=[
                    mock_first_response,
                    mock_stream(),
                ]
            )
            mock_fetch.return_value = SAMPLE_TEAM_RESPONSE

            events = []
            async for event in stream_usage_ai_chat(
                messages=[{"role": "user", "content": "Which team spends the most?"}],
                model="gpt-4o-mini",
                is_admin=True,
            ):
                events.append(json.loads(event.replace("data: ", "").strip()))

            chunk_events = [e for e in events if e["type"] == "chunk"]
            assert len(chunk_events) >= 1
            assert "Engineering" in chunk_events[0]["content"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "reason",
        [
            "end_date must be on or after start_date",
            "start_date and end_date must be valid YYYY-MM-DD dates",
        ],
        ids=["reversed", "invalid-format"],
    )
    @pytest.mark.parametrize(
        "tool_name", ["get_usage_data", "get_team_usage_data", "get_tag_usage_data"]
    )
    async def test_invalid_tool_dates_are_reported_to_the_model(
        self, tool_name: str, reason: str
    ) -> None:
        fetch: Final = AsyncMock(
            side_effect=HTTPException(status_code=400, detail={"error": reason})
        )

        tool_message, statuses = await _run_one_tool(
            tool_name, "2025-01-31", "2025-01-01", fetch
        )

        fetch.assert_awaited_once()
        assert tool_message == f"Invalid date range: {reason}"
        assert statuses == ["running", "complete"]

    @pytest.mark.asyncio
    async def test_tool_dates_the_query_layer_accepts_are_queried_as_sent(
        self,
    ) -> None:
        fetch: Final = AsyncMock(return_value=SAMPLE_AGGREGATED_RESPONSE)

        tool_message, statuses = await _run_one_tool(
            "get_usage_data", "2025/01/01", "2025-01-31T00:00:00", fetch
        )

        fetch.assert_awaited_once_with(
            start_date="2025/01/01", end_date="2025-01-31T00:00:00"
        )
        assert tool_message.startswith("Total Spend: $")
        assert statuses == ["running", "complete"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "error",
        [
            HTTPException(status_code=500, detail={"error": "db down"}),
            HTTPException(status_code=400, detail="page and page_size must be >= 1"),
        ],
        ids=["server-error", "other-400"],
    )
    async def test_other_tool_errors_stay_a_generic_fetch_error(
        self, error: HTTPException
    ) -> None:
        tool_message, statuses = await _run_one_tool(
            "get_team_usage_data",
            "2025-01-01",
            "2025-01-31",
            AsyncMock(side_effect=error),
        )

        assert tool_message == "Error fetching team usage data. Please try again."
        assert statuses == ["running", "error"]

    @pytest.mark.asyncio
    async def test_stream_handles_error(self):
        with patch(
            "litellm.proxy.management_endpoints.usage_endpoints.ai_usage_chat.litellm"
        ) as mock_litellm:
            mock_litellm.acompletion = AsyncMock(side_effect=Exception("LLM error"))

            events = []
            async for event in stream_usage_ai_chat(
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
            chunk = MagicMock()
            chunk.choices = [MagicMock()]
            chunk.choices[0].delta.content = "Data."
            yield chunk

        mock_fetch = AsyncMock(return_value=SAMPLE_AGGREGATED_RESPONSE)

        with (
            patch(
                "litellm.proxy.management_endpoints.usage_endpoints.ai_usage_chat.litellm"
            ) as mock_litellm,
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
            mock_litellm.acompletion = AsyncMock(
                side_effect=[
                    mock_first_response,
                    mock_stream(),
                ]
            )

            events = []
            async for event in stream_usage_ai_chat(
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

        with pytest.raises(ValueError, match='Non-admin caller has user_id=None; refusing to issue an') as exc_info:
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

        import litellm
        from fastapi.responses import StreamingResponse
        from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
        from litellm.proxy.management_endpoints.usage_endpoints.endpoints import (
            ChatMessage,
            UsageAIChatRequest,
            usage_ai_chat,
        )

        monkeypatch.setattr(litellm, "sse_keepalive_ping_interval_seconds", interval)

        async def slow_acompletion(**kwargs):
            await asyncio.sleep(delay)
            response = MagicMock()
            response.choices = [MagicMock()]
            response.choices[0].message.tool_calls = None
            response.choices[0].message.content = "Total spend is $50.25"
            return response

        with patch(  # test-quality-ok: the stream calls the module-level litellm.acompletion directly; no injection seam
            "litellm.proxy.management_endpoints.usage_endpoints.ai_usage_chat.litellm.acompletion",
            new=AsyncMock(side_effect=slow_acompletion),
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


async def _run_one_tool(
    tool_name: str, start_date: str, end_date: str, fetch: AsyncMock
) -> tuple[str, list[object]]:
    """Runs one admin chat turn whose model calls `tool_name` with these dates; returns the
    tool message the follow-up model call carried and the tool statuses streamed."""
    arguments: Final = json.dumps({"start_date": start_date, "end_date": end_date})
    mock_tool_call: Final = MagicMock()
    mock_tool_call.id = "call_dates"
    mock_tool_call.function.name = tool_name
    mock_tool_call.function.arguments = arguments

    mock_first_response: Final = MagicMock()
    mock_first_response.choices = [MagicMock()]
    mock_first_response.choices[0].message.tool_calls = [mock_tool_call]
    mock_first_response.choices[0].message.model_dump.return_value = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_dates",
                "type": "function",
                "function": {"name": tool_name, "arguments": arguments},
            }
        ],
    }

    async def mock_stream() -> AsyncIterator[MagicMock]:
        chunk: Final = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta.content = "Done."
        yield chunk

    with (
        patch(
            "litellm.proxy.management_endpoints.usage_endpoints.ai_usage_chat.litellm"
        ) as mock_litellm,
        patch.dict(TOOL_HANDLERS[tool_name], {"fetch": fetch}),
    ):
        mock_litellm.acompletion = AsyncMock(
            side_effect=[mock_first_response, mock_stream()]
        )
        events: Final[tuple[dict[str, object], ...]] = tuple(
            [
                json.loads(event.replace("data: ", "").strip())
                async for event in stream_usage_ai_chat(
                    messages=[{"role": "user", "content": "How much did we spend?"}],
                    model="gpt-4o-mini",
                    is_admin=True,
                )
            ]
        )
        final_messages: Final = mock_litellm.acompletion.call_args_list[1].kwargs[
            "messages"
        ]

    tool_messages: Final = [m for m in final_messages if m["role"] == "tool"]
    assert len(tool_messages) == 1, tool_messages
    assert tool_messages[0]["tool_call_id"] == "call_dates"
    return str(tool_messages[0]["content"]), [
        e["status"] for e in events if e["type"] == "tool_call"
    ]
