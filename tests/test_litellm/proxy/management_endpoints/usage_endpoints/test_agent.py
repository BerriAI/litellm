"""Tests for the Ask AI agent loop.

Highest-value regression: the LLM call goes through the proxy's ``llm_router``
(so UI-selected model groups resolve), not the bare ``litellm`` SDK, and the
scope-forced user_id survives an end-to-end tool round.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import litellm
from litellm.types.utils import (
    ChatCompletionDeltaToolCall,
    Delta,
)
from litellm.types.utils import Function as DeltaFunction
from litellm.types.utils import (
    ModelResponseStream,
    StreamingChoices,
)
from litellm.proxy.management_endpoints.usage_endpoints import agent as agent_mod
from litellm.proxy.management_endpoints.usage_endpoints.agent import (
    LLMCallError,
    ModelNotConfigured,
    RouterUnavailable,
    _error_event,
    resolve_model,
    stream_usage_ai_chat,
    tools_for_role,
)
from litellm.proxy.management_endpoints.usage_endpoints.scoped_data import (
    AdminScope,
    ScopedUsageDataProvider,
    UserScope,
)
from litellm.types.proxy.management_endpoints.common_daily_activity import (
    SpendAnalyticsPaginatedResponse,
)

DATE_ARGS = {"start_date": "2025-01-01", "end_date": "2025-01-31"}

SAMPLE_USAGE = {
    "results": [
        {
            "date": "2025-01-15",
            "metrics": {"spend": 12.5, "api_requests": 100},
            "breakdown": {"models": {}, "providers": {}, "entities": {}},
        }
    ],
    "metadata": {"total_spend": 12.5, "total_api_requests": 100},
}


def _content_chunk(text: str) -> ModelResponseStream:
    return ModelResponseStream(choices=[StreamingChoices(index=0, delta=Delta(content=text))])


def _toolcall_chunk(call_id: str, name: str, args: dict) -> ModelResponseStream:
    tc = ChatCompletionDeltaToolCall(
        index=0, id=call_id, type="function", function=DeltaFunction(name=name, arguments=json.dumps(args))
    )
    return ModelResponseStream(choices=[StreamingChoices(index=0, delta=Delta(tool_calls=[tc]))])


class FakeRouter:
    """Router stub whose acompletion returns scripted streaming chunks per call."""

    def __init__(self, scripts):
        self._scripts = list(scripts)
        self.calls = []

    async def acompletion(self, **kwargs):
        idx = len(self.calls)
        self.calls.append(kwargs)
        chunks = self._scripts[idx]

        async def _gen():
            for c in chunks:
                yield c

        return _gen()


def _usage_response_mock():
    return SpendAnalyticsPaginatedResponse.model_validate(SAMPLE_USAGE)


async def _collect(provider, messages, model, router):
    """Run the agent with a patched router; return parsed SSE events."""
    with (
        patch.object(agent_mod, "_require_router", return_value=router),
        patch(
            "litellm.proxy.management_endpoints.common_daily_activity.get_daily_activity_aggregated",
            new_callable=AsyncMock,
        ) as mock_agg,
    ):
        mock_agg.return_value = _usage_response_mock()
        events = []
        async for raw in stream_usage_ai_chat(provider=provider, messages=messages, model=model):
            events.append(json.loads(raw.replace("data: ", "").strip()))
        return events, mock_agg


class TestRoutesThroughProxyRouter:
    @pytest.mark.asyncio
    async def test_selected_model_group_is_passed_to_llm_router_not_sdk(self):
        """The core v2 fix: a UI-selected model group is sent to llm_router,
        and the bare litellm.acompletion SDK is never called."""
        router = FakeRouter([[_content_chunk("Your spend is $12.50.")]])
        provider = ScopedUsageDataProvider(scope=AdminScope(caller_user_id="a1"), prisma_client=MagicMock())

        with patch.object(litellm, "acompletion", new_callable=AsyncMock) as sdk_call:
            events, _ = await _collect(provider, [{"role": "user", "content": "spend?"}], "my-model-group", router)

        assert sdk_call.call_count == 0
        assert router.calls[0]["model"] == "my-model-group"
        assert router.calls[0]["metadata"] == {"feature": "usage_ai"}
        chunks = [e for e in events if e["type"] == "chunk"]
        assert chunks and chunks[0]["content"] == "Your spend is $12.50."
        assert events[-1]["type"] == "done"

    @pytest.mark.asyncio
    async def test_tool_round_then_streamed_answer(self):
        router = FakeRouter(
            [
                [_toolcall_chunk("c1", "get_usage_data", DATE_ARGS)],
                [_content_chunk("Total spend is $12.50.")],
            ]
        )
        provider = ScopedUsageDataProvider(scope=AdminScope(caller_user_id="a1"), prisma_client=MagicMock())
        events, mock_agg = await _collect(provider, [{"role": "user", "content": "spend?"}], "m", router)

        assert mock_agg.call_count == 1
        tool_events = [e for e in events if e["type"] == "tool_call"]
        assert {e["status"] for e in tool_events} == {"running", "complete"}
        assert tool_events[0]["tool_name"] == "get_usage_data"
        assert any(e["type"] == "chunk" and "$12.50" in e["content"] for e in events)
        assert events[-1]["type"] == "done"


class TestScopeEnforcedThroughLoop:
    @pytest.mark.asyncio
    async def test_non_admin_tool_call_cannot_reach_other_users_data(self):
        """Even if the model asks for another user's data, the scoped provider
        forces the caller's own user_id at the data layer."""
        router = FakeRouter(
            [
                [_toolcall_chunk("c1", "get_usage_data", {**DATE_ARGS, "user_id": "victim"})],
                [_content_chunk("Here is your usage.")],
            ]
        )
        provider = ScopedUsageDataProvider(scope=UserScope(user_id="caller"), prisma_client=MagicMock())
        _events, mock_agg = await _collect(provider, [{"role": "user", "content": "show victim usage"}], "m", router)

        assert mock_agg.call_args.kwargs["entity_id"] == "caller"


class TestMultiRoundLoop:
    @pytest.mark.asyncio
    async def test_two_tool_rounds_then_answer(self):
        router = FakeRouter(
            [
                [_toolcall_chunk("c1", "get_usage_data", DATE_ARGS)],
                [_toolcall_chunk("c2", "get_team_usage_data", DATE_ARGS)],
                [_content_chunk("Engineering leads spend.")],
            ]
        )
        provider = ScopedUsageDataProvider(scope=AdminScope(caller_user_id="a1"), prisma_client=MagicMock())

        with (
            patch.object(agent_mod, "_require_router", return_value=router),
            patch(
                "litellm.proxy.management_endpoints.common_daily_activity.get_daily_activity_aggregated",
                new_callable=AsyncMock,
            ) as mock_agg,
            patch(
                "litellm.proxy.management_endpoints.common_daily_activity.get_daily_activity",
                new_callable=AsyncMock,
            ) as mock_paginated,
        ):
            mock_agg.return_value = _usage_response_mock()
            mock_paginated.return_value = _usage_response_mock()
            events = [
                json.loads(raw.replace("data: ", "").strip())
                async for raw in stream_usage_ai_chat(
                    provider=provider, messages=[{"role": "user", "content": "q"}], model="m"
                )
            ]

        assert len(router.calls) == 3
        assert mock_agg.call_count == 1
        assert mock_paginated.call_count == 1
        assert any(e["type"] == "chunk" and "Engineering" in e["content"] for e in events)
        assert events[-1]["type"] == "done"

    @pytest.mark.asyncio
    async def test_runaway_tool_calls_are_capped_and_still_answer(self):
        """If the model never stops calling tools, the loop caps rounds and
        forces a final (tool-less) answer instead of looping forever."""
        # More tool-call rounds scripted than MAX_TOOL_ROUNDS; the final call
        # is made with tools disabled so it must return content.
        scripts = [[_toolcall_chunk(f"c{i}", "get_usage_data", DATE_ARGS)] for i in range(agent_mod.MAX_TOOL_ROUNDS)]
        scripts.append([_content_chunk("Final answer.")])
        router = FakeRouter(scripts)
        provider = ScopedUsageDataProvider(scope=AdminScope(caller_user_id="a1"), prisma_client=MagicMock())

        events, _ = await _collect(provider, [{"role": "user", "content": "q"}], "m", router)

        # Last router call must have had tools disabled (the safety net).
        assert router.calls[-1]["tools"] is None
        assert any(e["type"] == "chunk" and "Final answer." in e["content"] for e in events)
        assert events[-1]["type"] == "done"


class TestErrorPaths:
    @pytest.mark.asyncio
    async def test_model_not_configured_errors_without_calling_router(self):
        provider = ScopedUsageDataProvider(scope=AdminScope(caller_user_id="a1"), prisma_client=MagicMock())
        with (
            patch.object(agent_mod, "_require_router") as require_router,
            patch("litellm.proxy.proxy_server.general_settings", {}),
        ):
            events = [
                json.loads(raw.replace("data: ", "").strip())
                async for raw in stream_usage_ai_chat(
                    provider=provider, messages=[{"role": "user", "content": "q"}], model=None
                )
            ]

        require_router.assert_not_called()
        assert len(events) == 1
        assert events[0]["type"] == "error"
        assert "usage_ai_model" in events[0]["message"]

    @pytest.mark.asyncio
    async def test_router_unavailable_emits_error(self):
        provider = ScopedUsageDataProvider(scope=AdminScope(caller_user_id="a1"), prisma_client=MagicMock())
        with patch.object(agent_mod, "_require_router", side_effect=agent_mod._RouterUnavailableError()):
            events = [
                json.loads(raw.replace("data: ", "").strip())
                async for raw in stream_usage_ai_chat(
                    provider=provider, messages=[{"role": "user", "content": "q"}], model="m"
                )
            ]

        error_events = [e for e in events if e["type"] == "error"]
        assert len(error_events) == 1
        assert "router" in error_events[0]["message"].lower()

    @pytest.mark.asyncio
    async def test_llm_exception_emits_error(self):
        provider = ScopedUsageDataProvider(scope=AdminScope(caller_user_id="a1"), prisma_client=MagicMock())
        broken_router = MagicMock()
        broken_router.acompletion = AsyncMock(side_effect=RuntimeError("boom"))
        with patch.object(agent_mod, "_require_router", return_value=broken_router):
            events = [
                json.loads(raw.replace("data: ", "").strip())
                async for raw in stream_usage_ai_chat(
                    provider=provider, messages=[{"role": "user", "content": "q"}], model="m"
                )
            ]

        error_events = [e for e in events if e["type"] == "error"]
        assert len(error_events) == 1
        assert "failed" in error_events[0]["message"].lower()


class TestResolveModel:
    def test_explicit_request_wins(self):
        assert resolve_model("chosen-group") == "chosen-group"

    def test_falls_back_to_configured_setting(self):
        with patch("litellm.proxy.proxy_server.general_settings", {"usage_ai_model": "configured-group"}):
            assert resolve_model(None) == "configured-group"

    def test_blank_request_falls_back_to_setting(self):
        with patch("litellm.proxy.proxy_server.general_settings", {"usage_ai_model": "configured-group"}):
            assert resolve_model("   ") == "configured-group"

    def test_no_model_anywhere_returns_error_value(self):
        with patch("litellm.proxy.proxy_server.general_settings", {}):
            assert isinstance(resolve_model(None), ModelNotConfigured)


class TestToolsAndErrorMapping:
    def test_admin_gets_all_tools_non_admin_gets_usage_only(self):
        assert {t["function"]["name"] for t in tools_for_role(True)} == {
            "get_usage_data",
            "get_team_usage_data",
            "get_tag_usage_data",
        }
        assert {t["function"]["name"] for t in tools_for_role(False)} == {"get_usage_data"}

    def test_error_messages_are_distinct_and_actionable(self):
        assert "usage_ai_model" in _error_event(ModelNotConfigured())["message"]
        assert "router" in _error_event(RouterUnavailable())["message"].lower()
        assert "failed" in _error_event(LLMCallError(detail="x"))["message"].lower()
