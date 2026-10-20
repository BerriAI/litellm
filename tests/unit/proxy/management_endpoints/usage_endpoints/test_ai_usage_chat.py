"""
Tests for AI Usage Chat module.
"""

import json
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Final
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import respx
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

if TYPE_CHECKING:
    from litellm.router import Router


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
            patch("litellm.proxy.management_endpoints.usage_endpoints.ai_usage_chat.litellm") as mock_litellm,
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
            patch("litellm.proxy.management_endpoints.usage_endpoints.ai_usage_chat.litellm") as mock_litellm,
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
    @pytest.mark.parametrize("tool_name", ["get_usage_data", "get_team_usage_data", "get_tag_usage_data"])
    async def test_invalid_tool_dates_are_reported_to_the_model(self, tool_name: str, reason: str) -> None:
        fetch: Final = AsyncMock(side_effect=HTTPException(status_code=400, detail={"error": reason}))

        tool_message, statuses = await _run_one_tool(tool_name, "2025-01-31", "2025-01-01", fetch)

        fetch.assert_awaited_once()
        assert tool_message == f"Invalid date range: {reason}"
        assert statuses == ["running", "complete"]

    @pytest.mark.asyncio
    async def test_tool_dates_the_query_layer_accepts_are_queried_as_sent(
        self,
    ) -> None:
        fetch: Final = AsyncMock(return_value=SAMPLE_AGGREGATED_RESPONSE)

        tool_message, statuses = await _run_one_tool("get_usage_data", "2025/01/01", "2025-01-31T00:00:00", fetch)

        fetch.assert_awaited_once_with(start_date="2025/01/01", end_date="2025-01-31T00:00:00")
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
    async def test_other_tool_errors_stay_a_generic_fetch_error(self, error: HTTPException) -> None:
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
        with patch("litellm.proxy.management_endpoints.usage_endpoints.ai_usage_chat.litellm") as mock_litellm:
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
            patch("litellm.proxy.management_endpoints.usage_endpoints.ai_usage_chat.litellm") as mock_litellm,
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

        with pytest.raises(ValueError, match="Non-admin caller has user_id=None; refusing to issue an") as exc_info:
            _resolve_fetch_kwargs(
                fn_name="get_usage_data",
                fn_args={"start_date": "2025-01-01", "end_date": "2025-01-31"},
                user_id=None,
                is_admin=False,
            )
        assert "Endpoint-level guard missing" in str(exc_info.value)


class TestUsageAiChatCompletionRouting:
    @staticmethod
    def _plain_content_response() -> MagicMock:
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.tool_calls = None
        mock_response.choices[0].message.content = "Hello!"
        return mock_response

    @pytest.mark.asyncio
    async def test_stream_usage_ai_chat_passes_drop_params(self):
        mock_acompletion = AsyncMock(return_value=self._plain_content_response())

        with (
            patch(  # test-quality-ok: ai_usage_chat reads the proxy_server.llm_router module global; no injection seam
                "litellm.proxy.proxy_server.llm_router", None
            ),
            patch(  # test-quality-ok: the stream calls the module-level litellm.acompletion directly; no injection seam
                "litellm.proxy.management_endpoints.usage_endpoints.ai_usage_chat.litellm.acompletion",
                new=mock_acompletion,
            ),
        ):
            events = []
            async for event in stream_usage_ai_chat(
                messages=[{"role": "user", "content": "hi"}],
                model="gpt-4o-mini",
                is_admin=True,
            ):
                events.append(json.loads(event.replace("data: ", "").strip()))

            mock_acompletion.assert_awaited_once()
            assert mock_acompletion.await_args.kwargs["drop_params"] is True
            assert mock_acompletion.await_args.kwargs["model"] == "gpt-4o-mini"

            event_types = [e["type"] for e in events]
            assert "chunk" in event_types
            assert event_types[-1] == "done"
            assert "error" not in event_types

    @pytest.mark.asyncio
    async def test_stream_usage_ai_chat_uses_router_for_configured_model(self):
        mock_router = MagicMock()
        mock_router.is_recognized_model.return_value = True
        mock_router.acompletion = AsyncMock(return_value=self._plain_content_response())
        mock_acompletion = AsyncMock(return_value=self._plain_content_response())

        with (
            patch(  # test-quality-ok: ai_usage_chat reads the proxy_server.llm_router module global; no injection seam
                "litellm.proxy.proxy_server.llm_router", mock_router
            ),
            patch(  # test-quality-ok: the stream calls the module-level litellm.acompletion directly; no injection seam
                "litellm.proxy.management_endpoints.usage_endpoints.ai_usage_chat.litellm.acompletion",
                new=mock_acompletion,
            ),
        ):
            events = []
            async for event in stream_usage_ai_chat(
                messages=[{"role": "user", "content": "hi"}],
                model="my-alias",
                is_admin=True,
            ):
                events.append(json.loads(event.replace("data: ", "").strip()))

            mock_router.acompletion.assert_awaited_once()
            assert mock_router.acompletion.await_args.kwargs["model"] == "my-alias"
            assert mock_router.acompletion.await_args.kwargs["drop_params"] is True
            mock_acompletion.assert_not_called()

            event_types = [e["type"] for e in events]
            assert "chunk" in event_types
            assert "error" not in event_types

    @pytest.mark.asyncio
    async def test_stream_usage_ai_chat_falls_back_to_litellm_for_unknown_model(self):
        mock_router = MagicMock()
        mock_router.is_recognized_model.return_value = False
        mock_router.router_general_settings.pass_through_all_models = False
        mock_router.default_deployment = None
        mock_router.pattern_router.route.return_value = None
        mock_router.acompletion = AsyncMock(return_value=self._plain_content_response())
        mock_acompletion = AsyncMock(return_value=self._plain_content_response())

        with (
            patch(  # test-quality-ok: ai_usage_chat reads the proxy_server.llm_router module global; no injection seam
                "litellm.proxy.proxy_server.llm_router", mock_router
            ),
            patch(  # test-quality-ok: the stream calls the module-level litellm.acompletion directly; no injection seam
                "litellm.proxy.management_endpoints.usage_endpoints.ai_usage_chat.litellm.acompletion",
                new=mock_acompletion,
            ),
        ):
            events = []
            async for event in stream_usage_ai_chat(
                messages=[{"role": "user", "content": "hi"}],
                model="gpt-4o-mini",
                is_admin=True,
            ):
                events.append(json.loads(event.replace("data: ", "").strip()))

            mock_acompletion.assert_awaited_once()
            assert mock_acompletion.await_args.kwargs["model"] == "gpt-4o-mini"
            mock_router.acompletion.assert_not_called()

            event_types = [e["type"] for e in events]
            assert "chunk" in event_types
            assert "error" not in event_types

    @pytest.mark.asyncio
    async def test_stream_usage_ai_chat_uses_router_when_wildcard_pattern_exists(self):
        from litellm.router import Router

        router: Final = Router(
            model_list=[
                {
                    "model_name": "openai/*",
                    "litellm_params": {"model": "openai/*", "api_key": "sk-dummy"},
                }
            ]
        )
        router.acompletion = AsyncMock(return_value=self._plain_content_response())
        mock_acompletion = AsyncMock(return_value=self._plain_content_response())

        with (
            patch(  # test-quality-ok: ai_usage_chat reads the proxy_server.llm_router module global; no injection seam
                "litellm.proxy.proxy_server.llm_router", router
            ),
            patch(  # test-quality-ok: the stream calls the module-level litellm.acompletion directly; no injection seam
                "litellm.proxy.management_endpoints.usage_endpoints.ai_usage_chat.litellm.acompletion",
                new=mock_acompletion,
            ),
        ):
            events = []
            async for event in stream_usage_ai_chat(
                messages=[{"role": "user", "content": "hi"}],
                model="openai/gpt-4o-mini",
                is_admin=True,
            ):
                events.append(json.loads(event.replace("data: ", "").strip()))

            router.acompletion.assert_awaited_once()
            assert router.acompletion.await_args.kwargs["model"] == "openai/gpt-4o-mini"
            assert router.acompletion.await_args.kwargs["drop_params"] is True
            mock_acompletion.assert_not_called()

            event_types = [e["type"] for e in events]
            assert "chunk" in event_types
            assert "error" not in event_types

    @pytest.mark.asyncio
    async def test_stream_usage_ai_chat_falls_back_when_wildcard_does_not_match(self):
        from litellm.router import Router

        router: Final = Router(
            model_list=[
                {
                    "model_name": "anthropic/*",
                    "litellm_params": {
                        "model": "anthropic/claude-3-haiku-20240307",
                        "api_key": "sk-dummy",
                    },
                }
            ]
        )
        router.acompletion = AsyncMock(return_value=self._plain_content_response())
        mock_acompletion = AsyncMock(return_value=self._plain_content_response())

        with (
            patch(  # test-quality-ok: ai_usage_chat reads the proxy_server.llm_router module global; no injection seam
                "litellm.proxy.proxy_server.llm_router", router
            ),
            patch(  # test-quality-ok: the stream calls the module-level litellm.acompletion directly; no injection seam
                "litellm.proxy.management_endpoints.usage_endpoints.ai_usage_chat.litellm.acompletion",
                new=mock_acompletion,
            ),
        ):
            events = []
            async for event in stream_usage_ai_chat(
                messages=[{"role": "user", "content": "hi"}],
                model="gpt-4o-mini",
                is_admin=True,
            ):
                events.append(json.loads(event.replace("data: ", "").strip()))

            mock_acompletion.assert_awaited_once()
            assert mock_acompletion.await_args.kwargs["model"] == "gpt-4o-mini"
            router.acompletion.assert_not_called()

            event_types = [e["type"] for e in events]
            assert "chunk" in event_types
            assert "error" not in event_types

    @pytest.mark.asyncio
    async def test_stream_usage_ai_chat_falls_back_when_pass_through_all_models(self):
        from litellm.router import Router
        from litellm.types.router import RouterGeneralSettings

        router: Final = Router(
            model_list=[
                {
                    "model_name": "openai/*",
                    "litellm_params": {"model": "openai/*", "api_key": "sk-dummy"},
                }
            ],
            router_general_settings=RouterGeneralSettings(pass_through_all_models=True),
        )
        router.acompletion = AsyncMock(return_value=self._plain_content_response())
        mock_acompletion = AsyncMock(return_value=self._plain_content_response())

        with (
            patch(  # test-quality-ok: ai_usage_chat reads the proxy_server.llm_router module global; no injection seam
                "litellm.proxy.proxy_server.llm_router", router
            ),
            patch(  # test-quality-ok: the stream calls the module-level litellm.acompletion directly; no injection seam
                "litellm.proxy.management_endpoints.usage_endpoints.ai_usage_chat.litellm.acompletion",
                new=mock_acompletion,
            ),
        ):
            events = []
            async for event in stream_usage_ai_chat(
                messages=[{"role": "user", "content": "hi"}],
                model="gpt-4o-mini",
                is_admin=True,
            ):
                events.append(json.loads(event.replace("data: ", "").strip()))

            mock_acompletion.assert_awaited_once()
            assert mock_acompletion.await_args.kwargs["model"] == "gpt-4o-mini"
            router.acompletion.assert_not_called()

            event_types = [e["type"] for e in events]
            assert "chunk" in event_types
            assert "error" not in event_types

    @pytest.mark.asyncio
    async def test_stream_usage_ai_chat_uses_router_when_default_deployment_exists(self):
        mock_router = MagicMock()
        mock_router.is_recognized_model.return_value = False
        mock_router.router_general_settings.pass_through_all_models = False
        mock_router.default_deployment = {"model_name": "default", "litellm_params": {}}
        mock_router.acompletion = AsyncMock(return_value=self._plain_content_response())
        mock_acompletion = AsyncMock(return_value=self._plain_content_response())

        with (
            patch(  # test-quality-ok: ai_usage_chat reads the proxy_server.llm_router module global; no injection seam
                "litellm.proxy.proxy_server.llm_router", mock_router
            ),
            patch(  # test-quality-ok: the stream calls the module-level litellm.acompletion directly; no injection seam
                "litellm.proxy.management_endpoints.usage_endpoints.ai_usage_chat.litellm.acompletion",
                new=mock_acompletion,
            ),
        ):
            events = []
            async for event in stream_usage_ai_chat(
                messages=[{"role": "user", "content": "hi"}],
                model="gpt-5.6-terra",
                is_admin=True,
            ):
                events.append(json.loads(event.replace("data: ", "").strip()))

            mock_router.acompletion.assert_awaited_once()
            assert mock_router.acompletion.await_args.kwargs["model"] == "gpt-5.6-terra"
            assert mock_router.acompletion.await_args.kwargs["drop_params"] is True
            mock_acompletion.assert_not_called()

            event_types = [e["type"] for e in events]
            assert "chunk" in event_types
            assert "error" not in event_types

    @pytest.mark.asyncio
    async def test_stream_usage_ai_chat_streams_final_response_through_router(self):
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_789"
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
                    "id": "call_789",
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

        mock_router = MagicMock()
        mock_router.is_recognized_model.return_value = True
        mock_router.acompletion = AsyncMock(side_effect=[mock_first_response, mock_stream()])

        with (
            patch(  # test-quality-ok: ai_usage_chat reads the proxy_server.llm_router module global; no injection seam
                "litellm.proxy.proxy_server.llm_router", mock_router
            ),
            patch(  # test-quality-ok: the stream calls the module-level _fetch_usage_data directly; no injection seam
                "litellm.proxy.management_endpoints.usage_endpoints.ai_usage_chat._fetch_usage_data",
                new_callable=AsyncMock,
            ) as mock_fetch,
        ):
            mock_fetch.return_value = SAMPLE_AGGREGATED_RESPONSE

            events = []
            async for event in stream_usage_ai_chat(
                messages=[{"role": "user", "content": "What is my total spend?"}],
                model="my-alias",
                is_admin=True,
            ):
                events.append(json.loads(event.replace("data: ", "").strip()))

            assert mock_router.acompletion.await_count == 2
            stream_call = mock_router.acompletion.await_args_list[1]
            assert stream_call.kwargs["stream"] is True
            assert stream_call.kwargs["model"] == "my-alias"
            assert stream_call.kwargs["drop_params"] is True

            chunk_events = [e for e in events if e["type"] == "chunk"]
            assert any("$50.25" in e["content"] for e in chunk_events)
            assert events[-1]["type"] == "done"
            assert "error" not in [e["type"] for e in events]


class TestUsageAiChatProviderRequest:
    USAGE_TOOL: Final = TOOLS_ADMIN[0]

    @staticmethod
    def _router() -> "Router":
        from litellm.router import Router

        return Router(
            model_list=[
                {
                    "model_name": "gpt-5-prod",
                    "litellm_params": {"model": "openai/gpt-5.6", "api_key": "sk-gpt5-deployment"},
                },
                {
                    "model_name": "llama-on-bedrock",
                    "litellm_params": {
                        "model": "bedrock/converse/meta.llama3-8b-instruct-v1:0",
                        "aws_access_key_id": "AKIDEXAMPLE",
                        "aws_secret_access_key": "secret",
                        "aws_region_name": "us-east-1",
                    },
                },
            ]
        )

    @staticmethod
    def _mock_openai(respx_mock: respx.MockRouter) -> None:
        respx_mock.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "chatcmpl-1",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "gpt-5.6",
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                },
            )
        )
        respx_mock.post("https://api.openai.com/v1/responses").mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "resp_1",
                    "object": "response",
                    "created_at": 1,
                    "status": "completed",
                    "model": "gpt-5.6",
                    "output": [
                        {
                            "type": "message",
                            "id": "msg_1",
                            "status": "completed",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "hi", "annotations": []}],
                        }
                    ],
                    "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                },
            )
        )

    @staticmethod
    def _tool_names(sent: dict[str, object]) -> list[object]:
        return [tool.get("name") or tool["function"]["name"] for tool in sent["tools"]]

    async def _ask(self, model: str, monkeypatch: pytest.MonkeyPatch) -> None:
        import litellm
        from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
        from litellm.proxy.management_endpoints.usage_endpoints.ai_usage_chat import (
            _acompletion,
        )

        monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
        with patch(  # test-quality-ok: _acompletion reads the proxy_server.llm_router module global; no injection seam
            "litellm.proxy.proxy_server.llm_router", self._router()
        ):
            await _acompletion(
                model=model,
                messages=[{"role": "user", "content": "hi"}],
                tools=[self.USAGE_TOOL],
                user_api_key_dict=UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN, api_key="sk-test"),
            )

    @pytest.mark.asyncio
    async def test_reasoning_model_request_drops_temperature_and_keeps_tools(
        self, respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._mock_openai(respx_mock)

        await self._ask("gpt-5-prod", monkeypatch)

        sent: Final = json.loads(respx_mock.calls.last.request.content)
        assert "temperature" not in sent
        assert self._tool_names(sent) == [self.USAGE_TOOL["function"]["name"]]

    @pytest.mark.asyncio
    async def test_usage_tools_reach_a_model_litellm_lists_without_tool_support(
        self, respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        route: Final = respx_mock.post(url__regex=r"https://bedrock-runtime\.us-east-1\.amazonaws\.com/.*").mock(
            return_value=httpx.Response(
                200,
                json={
                    "output": {"message": {"role": "assistant", "content": [{"text": "hi"}]}},
                    "stopReason": "end_turn",
                    "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
                    "metrics": {"latencyMs": 1},
                },
            )
        )

        await self._ask("llama-on-bedrock", monkeypatch)

        sent: Final = json.loads(route.calls.last.request.content)
        assert [tool["toolSpec"]["name"] for tool in sent["toolConfig"]["tools"]] == [
            self.USAGE_TOOL["function"]["name"]
        ]

    @pytest.mark.asyncio
    async def test_deployment_name_is_called_with_that_deployment_credentials(
        self, respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._mock_openai(respx_mock)

        await self._ask("openai/gpt-5.6", monkeypatch)

        assert respx_mock.calls.last.request.headers["Authorization"] == "Bearer sk-gpt5-deployment"


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

        with (
            patch(  # test-quality-ok: the stream calls the module-level litellm.acompletion directly; no injection seam
                "litellm.proxy.management_endpoints.usage_endpoints.ai_usage_chat.litellm.acompletion",
                new=AsyncMock(side_effect=slow_acompletion),
            )
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


async def _run_one_tool(tool_name: str, start_date: str, end_date: str, fetch: AsyncMock) -> tuple[str, list[object]]:
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
        patch("litellm.proxy.management_endpoints.usage_endpoints.ai_usage_chat.litellm") as mock_litellm,
        patch.dict(TOOL_HANDLERS[tool_name], {"fetch": fetch}),
    ):
        mock_litellm.acompletion = AsyncMock(side_effect=[mock_first_response, mock_stream()])
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
        final_messages: Final = mock_litellm.acompletion.call_args_list[1].kwargs["messages"]

    tool_messages: Final = [m for m in final_messages if m["role"] == "tool"]
    assert len(tool_messages) == 1, tool_messages
    assert tool_messages[0]["tool_call_id"] == "call_dates"
    return str(tool_messages[0]["content"]), [e["status"] for e in events if e["type"] == "tool_call"]


class TestUsageAiChatModelAccess:
    @staticmethod
    def _router() -> "Router":
        from litellm.router import Router

        return Router(
            model_list=[
                {
                    "model_name": "team-a-secret",
                    "litellm_params": {"model": "gpt-4o", "api_key": "sk-dummy"},
                },
                {
                    "model_name": "shared-model",
                    "litellm_params": {"model": "gpt-4o-mini", "api_key": "sk-dummy"},
                },
            ]
        )

    @pytest.mark.asyncio
    async def test_key_without_model_access_is_denied(self):
        from litellm.proxy._types import ProxyException, UserAPIKeyAuth
        from litellm.proxy.management_endpoints.usage_endpoints.ai_usage_chat import (
            check_usage_ai_chat_model_access,
        )

        router: Final = self._router()
        with (
            patch(  # test-quality-ok: check reads the proxy_server.llm_router module global; no injection seam
                "litellm.proxy.proxy_server.llm_router", router
            ),
            patch(  # test-quality-ok: check reads the proxy_server.llm_model_list module global; no injection seam
                "litellm.proxy.proxy_server.llm_model_list", None
            ),
            patch(  # test-quality-ok: check reads the proxy_server.prisma_client module global; no injection seam
                "litellm.proxy.proxy_server.prisma_client", None
            ),
        ):
            with pytest.raises(ProxyException) as exc_info:
                await check_usage_ai_chat_model_access(
                    user_api_key_dict=UserAPIKeyAuth(models=["shared-model"], api_key="sk-test"),
                    model="team-a-secret",
                )

        assert exc_info.value.code == "403"

    @pytest.mark.asyncio
    async def test_key_with_model_access_is_allowed(self):
        from litellm.proxy._types import UserAPIKeyAuth
        from litellm.proxy.management_endpoints.usage_endpoints.ai_usage_chat import (
            check_usage_ai_chat_model_access,
        )

        router: Final = self._router()
        with (
            patch(  # test-quality-ok: check reads the proxy_server.llm_router module global; no injection seam
                "litellm.proxy.proxy_server.llm_router", router
            ),
            patch(  # test-quality-ok: check reads the proxy_server.llm_model_list module global; no injection seam
                "litellm.proxy.proxy_server.llm_model_list", None
            ),
            patch(  # test-quality-ok: check reads the proxy_server.prisma_client module global; no injection seam
                "litellm.proxy.proxy_server.prisma_client", None
            ),
        ):
            result: Final = await check_usage_ai_chat_model_access(
                user_api_key_dict=UserAPIKeyAuth(models=["shared-model"], api_key="sk-test"),
                model="shared-model",
            )
            assert result is None

    @pytest.mark.asyncio
    async def test_unrestricted_key_can_call_any_model(self):
        from litellm.proxy._types import UserAPIKeyAuth
        from litellm.proxy.management_endpoints.usage_endpoints.ai_usage_chat import (
            check_usage_ai_chat_model_access,
        )

        router: Final = self._router()
        with (
            patch(  # test-quality-ok: check reads the proxy_server.llm_router module global; no injection seam
                "litellm.proxy.proxy_server.llm_router", router
            ),
            patch(  # test-quality-ok: check reads the proxy_server.llm_model_list module global; no injection seam
                "litellm.proxy.proxy_server.llm_model_list", None
            ),
            patch(  # test-quality-ok: check reads the proxy_server.prisma_client module global; no injection seam
                "litellm.proxy.proxy_server.prisma_client", None
            ),
        ):
            result: Final = await check_usage_ai_chat_model_access(
                user_api_key_dict=UserAPIKeyAuth(models=[], api_key="sk-test"),
                model="team-a-secret",
            )
            assert result is None

    @pytest.mark.asyncio
    async def test_team_model_restriction_is_enforced(self):
        from litellm.proxy._types import ProxyException, UserAPIKeyAuth
        from litellm.proxy.management_endpoints.usage_endpoints.ai_usage_chat import (
            check_usage_ai_chat_model_access,
        )

        router: Final = self._router()
        with (
            patch(  # test-quality-ok: check reads the proxy_server.llm_router module global; no injection seam
                "litellm.proxy.proxy_server.llm_router", router
            ),
            patch(  # test-quality-ok: check reads the proxy_server.llm_model_list module global; no injection seam
                "litellm.proxy.proxy_server.llm_model_list", None
            ),
            patch(  # test-quality-ok: check reads the proxy_server.prisma_client module global; no injection seam
                "litellm.proxy.proxy_server.prisma_client", None
            ),
        ):
            with pytest.raises(ProxyException) as exc_info:
                await check_usage_ai_chat_model_access(
                    user_api_key_dict=UserAPIKeyAuth(
                        models=[],
                        team_models=["shared-model"],
                        team_id="t1",
                        api_key="sk-test",
                    ),
                    model="team-a-secret",
                )

        assert exc_info.value.code == "403"

    @pytest.mark.asyncio
    async def test_implicit_default_model_is_checked_against_key_models(self):
        from litellm.proxy._types import ProxyException, UserAPIKeyAuth
        from litellm.proxy.management_endpoints.usage_endpoints.ai_usage_chat import (
            check_usage_ai_chat_model_access,
        )

        router: Final = self._router()
        with (
            patch(  # test-quality-ok: check reads the proxy_server.llm_router module global; no injection seam
                "litellm.proxy.proxy_server.llm_router", router
            ),
            patch(  # test-quality-ok: check reads the proxy_server.llm_model_list module global; no injection seam
                "litellm.proxy.proxy_server.llm_model_list", None
            ),
            patch(  # test-quality-ok: check reads the proxy_server.prisma_client module global; no injection seam
                "litellm.proxy.proxy_server.prisma_client", None
            ),
        ):
            with pytest.raises(ProxyException) as exc_info:
                await check_usage_ai_chat_model_access(
                    user_api_key_dict=UserAPIKeyAuth(models=["shared-model"], api_key="sk-test"),
                    model=None,
                )

        assert exc_info.value.code == "403"


class TestUsageAiChatTeamModelRouting:
    @staticmethod
    def _team_router() -> "Router":
        from litellm.router import Router

        return Router(
            model_list=[
                {
                    "model_name": "team-a-internal",
                    "litellm_params": {
                        "model": "openai/gpt-4o",
                        "api_key": "sk-dummy",
                        "mock_response": "from the team deployment",
                    },
                    "model_info": {
                        "team_id": "team-a",
                        "team_public_model_name": "my-team-model",
                    },
                }
            ]
        )

    @pytest.mark.asyncio
    async def test_admin_reaches_a_team_public_model_name(self):
        from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
        from litellm.proxy.management_endpoints.usage_endpoints.ai_usage_chat import (
            _acompletion,
        )

        with patch(  # test-quality-ok: _acompletion reads the proxy_server.llm_router module global; no injection seam
            "litellm.proxy.proxy_server.llm_router", self._team_router()
        ):
            result: Final = await _acompletion(
                model="my-team-model",
                messages=[{"role": "user", "content": "hi"}],
                user_api_key_dict=UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN, api_key="sk-test"),
            )

        assert result.choices[0].message.content == "from the team deployment"

    @pytest.mark.asyncio
    async def test_team_key_ask_ai_spend_is_not_attributed_to_the_team(self, monkeypatch: pytest.MonkeyPatch):
        import asyncio

        import litellm
        from litellm.integrations.custom_logger import CustomLogger
        from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
        from litellm.proxy.management_endpoints.usage_endpoints.ai_usage_chat import (
            _acompletion,
        )

        logged: Final[asyncio.Queue[object]] = asyncio.Queue()

        class SpendCapture(CustomLogger):
            async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
                await logged.put(kwargs["standard_logging_object"]["metadata"]["user_api_key_team_id"])

        monkeypatch.setattr(litellm, "callbacks", [SpendCapture()])
        with patch(  # test-quality-ok: _acompletion reads the proxy_server.llm_router module global; no injection seam
            "litellm.proxy.proxy_server.llm_router", self._team_router()
        ):
            await _acompletion(
                model="my-team-model",
                messages=[{"role": "user", "content": "hi"}],
                user_api_key_dict=UserAPIKeyAuth(
                    user_role=LitellmUserRoles.PROXY_ADMIN, team_id="team-a", api_key="sk-test"
                ),
            )

        assert await asyncio.wait_for(logged.get(), timeout=10) is None

    @pytest.mark.asyncio
    async def test_team_public_model_name_without_team_id_uses_litellm(self):
        from litellm.proxy._types import UserAPIKeyAuth
        from litellm.proxy.management_endpoints.usage_endpoints.ai_usage_chat import (
            _acompletion,
        )
        from litellm.router import Router

        router: Final = Router(
            model_list=[
                {
                    "model_name": "team-a-internal",
                    "litellm_params": {"model": "gpt-4o", "api_key": "sk-dummy"},
                    "model_info": {
                        "team_id": "team-a",
                        "team_public_model_name": "my-team-model",
                    },
                }
            ]
        )
        expected_response: Final = MagicMock()
        router.acompletion = AsyncMock(return_value=MagicMock())
        mock_litellm_acompletion = AsyncMock(return_value=expected_response)

        with (
            patch(  # test-quality-ok: _acompletion reads the proxy_server.llm_router module global; no injection seam
                "litellm.proxy.proxy_server.llm_router", router
            ),
            patch(  # test-quality-ok: _acompletion calls the module-level litellm.acompletion directly; no injection seam
                "litellm.proxy.management_endpoints.usage_endpoints.ai_usage_chat.litellm.acompletion",
                new=mock_litellm_acompletion,
            ),
        ):
            result: Final = await _acompletion(
                model="my-team-model",
                messages=[{"role": "user", "content": "hi"}],
                user_api_key_dict=UserAPIKeyAuth(),
            )

        assert result is expected_response
        mock_litellm_acompletion.assert_awaited_once()
        router.acompletion.assert_not_called()

    @pytest.mark.asyncio
    async def test_key_alias_resolves_to_router_model(self):
        from litellm.proxy._types import UserAPIKeyAuth
        from litellm.proxy.management_endpoints.usage_endpoints.ai_usage_chat import (
            _acompletion,
        )
        from litellm.router import Router

        router: Final = Router(
            model_list=[
                {
                    "model_name": "gpt-4o-mini",
                    "litellm_params": {"model": "gpt-4o-mini", "api_key": "sk-dummy"},
                }
            ]
        )
        expected_response: Final = MagicMock()
        router.acompletion = AsyncMock(return_value=expected_response)
        mock_litellm_acompletion = AsyncMock(return_value=MagicMock())

        with (
            patch(  # test-quality-ok: _acompletion reads the proxy_server.llm_router module global; no injection seam
                "litellm.proxy.proxy_server.llm_router", router
            ),
            patch(  # test-quality-ok: _acompletion calls the module-level litellm.acompletion directly; no injection seam
                "litellm.proxy.management_endpoints.usage_endpoints.ai_usage_chat.litellm.acompletion",
                new=mock_litellm_acompletion,
            ),
        ):
            result: Final = await _acompletion(
                model="my-alias",
                messages=[{"role": "user", "content": "hi"}],
                user_api_key_dict=UserAPIKeyAuth(aliases={"my-alias": "gpt-4o-mini"}),
            )

        assert result is expected_response
        router.acompletion.assert_awaited_once()
        assert router.acompletion.await_args.kwargs["model"] == "gpt-4o-mini"
        mock_litellm_acompletion.assert_not_called()

    @pytest.mark.asyncio
    async def test_router_calls_receive_sanitized_caller_auth_for_access_group_filtering(self):
        from litellm.proxy._types import UserAPIKeyAuth
        from litellm.proxy.management_endpoints.usage_endpoints.ai_usage_chat import (
            _acompletion,
        )
        from litellm.router import Router

        router: Final = Router(
            model_list=[
                {
                    "model_name": "shared-model",
                    "litellm_params": {"model": "gpt-4o", "api_key": "sk-dummy"},
                    "model_info": {"access_groups": ["AG-A"]},
                },
                {
                    "model_name": "shared-model",
                    "litellm_params": {"model": "gpt-4o-mini", "api_key": "sk-dummy"},
                    "model_info": {"access_groups": ["AG-B"]},
                },
            ]
        )
        expected_response: Final = MagicMock()
        router.acompletion = AsyncMock(return_value=expected_response)
        mock_litellm_acompletion = AsyncMock(return_value=MagicMock())

        with (
            patch(  # test-quality-ok: _acompletion reads the proxy_server.llm_router module global; no injection seam
                "litellm.proxy.proxy_server.llm_router", router
            ),
            patch(  # test-quality-ok: _acompletion calls the module-level litellm.acompletion directly; no injection seam
                "litellm.proxy.management_endpoints.usage_endpoints.ai_usage_chat.litellm.acompletion",
                new=mock_litellm_acompletion,
            ),
        ):
            result: Final = await _acompletion(
                model="shared-model",
                messages=[{"role": "user", "content": "hi"}],
                user_api_key_dict=UserAPIKeyAuth(models=["AG-A"], api_key="sk-test"),
            )

        assert result is expected_response
        router.acompletion.assert_awaited_once()
        forwarded_auth: Final = router.acompletion.await_args.kwargs["metadata"]["user_api_key_auth"]
        assert forwarded_auth.models == ["AG-A"]
        mock_litellm_acompletion.assert_not_called()

    @pytest.mark.asyncio
    async def test_team_alias_routes_default_model_to_the_aliased_team_deployment(self):
        from litellm.proxy._types import UserAPIKeyAuth
        from litellm.proxy.management_endpoints.usage_endpoints.ai_usage_chat import (
            DEFAULT_COMPETITOR_DISCOVERY_MODEL,
            check_usage_ai_chat_model_access,
        )
        from litellm.router import Router

        router: Final = Router(
            model_list=[
                {
                    "model_name": DEFAULT_COMPETITOR_DISCOVERY_MODEL,
                    "litellm_params": {
                        "model": "openai/gpt-4o-mini",
                        "api_key": "sk-admin",
                        "mock_response": "from the admin deployment",
                    },
                },
                {
                    "model_name": "team-gpt",
                    "litellm_params": {
                        "model": "openai/gpt-4o-mini",
                        "api_key": "sk-team",
                        "mock_response": "from the team deployment",
                    },
                },
            ]
        )
        team_key: Final = UserAPIKeyAuth(
            api_key="sk-test",
            team_id="t1",
            team_models=["team-gpt"],
            team_model_aliases={DEFAULT_COMPETITOR_DISCOVERY_MODEL: "team-gpt"},
        )

        with (
            patch(  # test-quality-ok: Ask AI reads the proxy_server.llm_router module global; no injection seam
                "litellm.proxy.proxy_server.llm_router", router
            ),
            patch(  # test-quality-ok: the check reads the proxy_server.llm_model_list module global; no injection seam
                "litellm.proxy.proxy_server.llm_model_list", router.model_list
            ),
            patch(  # test-quality-ok: the check reads the proxy_server.prisma_client module global; no injection seam
                "litellm.proxy.proxy_server.prisma_client", None
            ),
        ):
            await check_usage_ai_chat_model_access(user_api_key_dict=team_key, model=None)
            events: Final = [
                json.loads(raw.removeprefix("data: "))
                async for raw in stream_usage_ai_chat(
                    messages=[{"role": "user", "content": "hi"}],
                    model=None,
                    is_admin=True,
                    user_api_key_dict=team_key,
                )
            ]

        assert {"type": "chunk", "content": "from the team deployment"} in events
        assert {"type": "chunk", "content": "from the admin deployment"} not in events


class TestUsageAiChatEndpointAccessCheck:
    @pytest.mark.asyncio
    async def test_endpoint_denies_key_without_model_access(self):
        from litellm.proxy._types import LitellmUserRoles, ProxyException, UserAPIKeyAuth
        from litellm.proxy.management_endpoints.usage_endpoints.endpoints import (
            ChatMessage,
            UsageAIChatRequest,
            usage_ai_chat,
        )
        from litellm.router import Router

        router: Final = Router(
            model_list=[
                {
                    "model_name": "team-a-secret",
                    "litellm_params": {"model": "gpt-4o", "api_key": "sk-dummy"},
                },
                {
                    "model_name": "shared-model",
                    "litellm_params": {"model": "gpt-4o-mini", "api_key": "sk-dummy"},
                },
            ]
        )
        mock_stream = MagicMock()
        restricted_key: Final = UserAPIKeyAuth(
            user_id="user-1",
            user_role=LitellmUserRoles.INTERNAL_USER,
            models=["shared-model"],
            api_key="sk-test",
        )
        body: Final = UsageAIChatRequest(
            messages=[ChatMessage(role="user", content="hi")],
            model="team-a-secret",
        )

        with (
            patch(  # test-quality-ok: the check reads the proxy_server.llm_router module global; no injection seam
                "litellm.proxy.proxy_server.llm_router", router
            ),
            patch(  # test-quality-ok: the check reads the proxy_server.llm_model_list module global; no injection seam
                "litellm.proxy.proxy_server.llm_model_list", None
            ),
            patch(  # test-quality-ok: the check reads the proxy_server.prisma_client module global; no injection seam
                "litellm.proxy.proxy_server.prisma_client", None
            ),
            patch(  # test-quality-ok: endpoint lazy-imports stream_usage_ai_chat from the module; no injection seam
                "litellm.proxy.management_endpoints.usage_endpoints.ai_usage_chat.stream_usage_ai_chat",
                new=mock_stream,
            ),
            pytest.raises(ProxyException) as exc_info,
        ):
            await usage_ai_chat(
                data=body,
                request=MagicMock(),
                user_api_key_dict=restricted_key,
            )

        assert exc_info.value.code == "403"
        mock_stream.assert_not_called()
