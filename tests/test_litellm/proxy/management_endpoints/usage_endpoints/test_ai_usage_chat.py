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

        mock_second_response = MagicMock()
        mock_second_response.choices = [MagicMock()]
        mock_second_response.choices[0].message.tool_calls = None
        mock_second_response.choices[0].message.content = "Total spend is $50.25"

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
                    mock_second_response,
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

        mock_second_response = MagicMock()
        mock_second_response.choices = [MagicMock()]
        mock_second_response.choices[0].message.tool_calls = None
        mock_second_response.choices[0].message.content = "Engineering is the top team."

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
                    mock_second_response,
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

        mock_second_response = MagicMock()
        mock_second_response.choices = [MagicMock()]
        mock_second_response.choices[0].message.tool_calls = None
        mock_second_response.choices[0].message.content = "Data."

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
                    mock_second_response,
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

        with pytest.raises(ValueError) as exc_info:
            _resolve_fetch_kwargs(
                fn_name="get_usage_data",
                fn_args={"start_date": "2025-01-01", "end_date": "2025-01-31"},
                user_id=None,
                is_admin=False,
            )
        assert "Endpoint-level guard missing" in str(exc_info.value)


# ---------------------------------------------------------------------------
# LIT-3145 regression tests: system-prompt date-default + multi-turn tool loop
# ---------------------------------------------------------------------------


class TestLit3145SystemPromptDateDefault:
    """Pin the date-default guidance added to the system prompt."""

    def test_admin_prompt_tells_model_to_default_to_last_30_days(self):
        prompt = _build_system_prompt(is_admin=True)
        assert "last 30 days" in prompt

    def test_non_admin_prompt_tells_model_to_default_to_last_30_days(self):
        prompt = _build_system_prompt(is_admin=False)
        assert "last 30 days" in prompt

    def test_prompt_forbids_asking_for_date_range_clarification(self):
        prompt = _build_system_prompt(is_admin=True)
        # Must explicitly tell the model NOT to ask for clarification, otherwise
        # we regress the LIT-3145 UX where the assistant replied with
        # "what date range?" on a fresh turn.
        assert "Do NOT ask" in prompt
        assert "date range" in prompt

    def test_prompt_instructs_model_to_state_chosen_range(self):
        prompt = _build_system_prompt(is_admin=True)
        # The model should report which range it chose so the user can sanity-check.
        assert "which range you used" in prompt


class TestLit3145MultiTurnToolLoop:
    """Multi-round tool loop: model can chain follow-up tool calls in one turn."""

    def _make_tool_call_choice(self, tool_name, args, call_id="call_x"):
        tc = MagicMock()
        tc.id = call_id
        tc.function.name = tool_name
        tc.function.arguments = json.dumps(args)

        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.tool_calls = [tc]
        resp.choices[0].message.model_dump.return_value = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": json.dumps(args),
                    },
                }
            ],
        }
        return resp

    def _make_final_choice(self, content):
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.tool_calls = None
        resp.choices[0].message.content = content
        return resp

    @pytest.mark.asyncio
    async def test_two_tool_rounds_in_a_single_user_turn(self):
        """First round: get_usage_data. Second round (post-results): get_team_usage_data.
        Third round: final natural-language answer. All in ONE user message.
        """
        round1 = self._make_tool_call_choice(
            "get_usage_data",
            {"start_date": "2026-04-29", "end_date": "2026-05-28"},
            call_id="c1",
        )
        round2 = self._make_tool_call_choice(
            "get_team_usage_data",
            {"start_date": "2026-04-29", "end_date": "2026-05-28"},
            call_id="c2",
        )
        round3 = self._make_final_choice(
            "Showing usage for 2026-04-29 -> 2026-05-28. Top team is Engineering."
        )

        with (
            patch(
                "litellm.proxy.management_endpoints.usage_endpoints.ai_usage_chat.litellm"
            ) as mock_litellm,
            patch(
                "litellm.proxy.management_endpoints.usage_endpoints.ai_usage_chat._fetch_usage_data",
                new_callable=AsyncMock,
            ) as mock_fetch_usage,
            patch(
                "litellm.proxy.management_endpoints.usage_endpoints.ai_usage_chat._fetch_team_usage_data",
                new_callable=AsyncMock,
            ) as mock_fetch_team,
        ):
            mock_litellm.acompletion = AsyncMock(side_effect=[round1, round2, round3])
            mock_fetch_usage.return_value = SAMPLE_AGGREGATED_RESPONSE
            mock_fetch_team.return_value = SAMPLE_TEAM_RESPONSE

            events = []
            async for event in stream_usage_ai_chat(
                messages=[
                    {
                        "role": "user",
                        "content": "Which team spent the most last month?",
                    }
                ],
                model="gpt-4o-mini",
                user_id="admin-1",
                is_admin=True,
            ):
                events.append(json.loads(event.replace("data: ", "").strip()))

            tool_call_events = [e for e in events if e["type"] == "tool_call"]
            tool_names_run = {
                e["tool_name"] for e in tool_call_events if e.get("status") == "running"
            }
            # Critical assertion: both tool calls happened in a single user turn.
            assert "get_usage_data" in tool_names_run
            assert "get_team_usage_data" in tool_names_run

            chunks = [e for e in events if e["type"] == "chunk"]
            joined = "".join(c["content"] for c in chunks)
            assert "Engineering" in joined

            done_events = [e for e in events if e["type"] == "done"]
            assert len(done_events) == 1

    @pytest.mark.asyncio
    async def test_round_cap_enforced_no_runaway_tool_calls(self):
        """A model that keeps requesting tool calls forever must be stopped by
        MAX_TOOL_ROUNDS. After the cap we force a final natural-language reply
        via _stream_final_response (no more tools)."""
        # Three identical tool-call responses in a row.
        rounds = [
            self._make_tool_call_choice(
                "get_usage_data",
                {"start_date": "2026-04-29", "end_date": "2026-05-28"},
                call_id=f"c{i}",
            )
            for i in range(5)  # more than MAX_TOOL_ROUNDS
        ]

        async def mock_final_stream():
            chunk = MagicMock()
            chunk.choices = [MagicMock()]
            chunk.choices[0].delta.content = "Final fallback answer."
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
            # Sequence: MAX_TOOL_ROUNDS tool-call responses, then a streamed final.
            from litellm.proxy.management_endpoints.usage_endpoints.ai_usage_chat import (
                MAX_TOOL_ROUNDS,
            )

            mock_litellm.acompletion = AsyncMock(
                side_effect=rounds[:MAX_TOOL_ROUNDS] + [mock_final_stream()]
            )
            mock_fetch.return_value = SAMPLE_AGGREGATED_RESPONSE

            events = []
            async for event in stream_usage_ai_chat(
                messages=[{"role": "user", "content": "summarise usage"}],
                model="gpt-4o-mini",
                user_id="admin-1",
                is_admin=True,
            ):
                events.append(json.loads(event.replace("data: ", "").strip()))

            running_tool_calls = [
                e
                for e in events
                if e["type"] == "tool_call" and e.get("status") == "running"
            ]
            # Exactly MAX_TOOL_ROUNDS tool calls — not 4, not 5.
            assert (
                len(running_tool_calls) == MAX_TOOL_ROUNDS
            ), f"expected {MAX_TOOL_ROUNDS} tool calls, got {len(running_tool_calls)}"

            # And the final synth chunk does land.
            chunks = [e for e in events if e["type"] == "chunk"]
            assert any("Final fallback answer." in c["content"] for c in chunks)

            # Total acompletion calls = MAX_TOOL_ROUNDS + 1 (the final synth).
            assert mock_litellm.acompletion.await_count == MAX_TOOL_ROUNDS + 1

    @pytest.mark.asyncio
    async def test_model_can_answer_without_any_tool_call_on_first_round(self):
        """If the model responds with content (no tool_calls) on round 1 we
        emit that content and stop — no fallback _stream_final_response call.
        This is a regression guard for the single-round happy path so we don't
        accidentally call acompletion a second time when the first response
        already had the answer.
        """
        round1 = self._make_final_choice("Hello! How can I help with your usage?")

        with patch(
            "litellm.proxy.management_endpoints.usage_endpoints.ai_usage_chat.litellm"
        ) as mock_litellm:
            mock_litellm.acompletion = AsyncMock(side_effect=[round1])

            events = []
            async for event in stream_usage_ai_chat(
                messages=[{"role": "user", "content": "hi"}],
                model="gpt-4o-mini",
                user_id="u-1",
                is_admin=True,
            ):
                events.append(json.loads(event.replace("data: ", "").strip()))

            chunks = [e for e in events if e["type"] == "chunk"]
            done = [e for e in events if e["type"] == "done"]
            assert len(chunks) == 1
            assert chunks[0]["content"] == "Hello! How can I help with your usage?"
            assert len(done) == 1
            # Exactly one model call -- no second-round fallback streaming.
            assert mock_litellm.acompletion.await_count == 1
