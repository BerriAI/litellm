"""
Integration tests for WebSearch interception with the Responses API.

Tests that the websearch_interception callback intercepts litellm_web_search
tool calls returned by /v1/responses, executes the search server-side, and
builds a Responses-format follow-up request.
"""

import json
from types import SimpleNamespace
from typing import Final
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import respx

import litellm
from litellm.integrations.websearch_interception.handler import (
    WebSearchInterceptionLogger,
)
from litellm.types.integrations.custom_logger import (
    RESPONSES_AGENTIC_SURFACE,
)
from litellm.types.utils import CallTypes, LlmProviders


def _responses_output_with_web_search(call_id: str = "fc_1", query: str = "latest ai news"):
    return SimpleNamespace(
        output=[
            SimpleNamespace(
                type="function_call",
                name="litellm_web_search",
                call_id=call_id,
                arguments='{"query": "%s"}' % query,
            )
        ]
    )


@pytest.mark.asyncio
async def test_responses_hook_detects_function_call():
    """async_should_run_responses_agentic_loop detects a litellm_web_search function_call."""
    logger = WebSearchInterceptionLogger(enabled_providers=[LlmProviders.OPENAI])

    should_run, tools_dict = await logger.async_should_run_responses_agentic_loop(
        response=_responses_output_with_web_search(),
        model="gpt-4o",
        messages=[{"role": "user", "content": "What's the latest AI news?"}],
        tools=[{"type": "function", "name": "litellm_web_search"}],
        stream=False,
        custom_llm_provider="openai",
        kwargs={},
    )

    assert should_run is True
    assert tools_dict["response_format"] == "responses"
    assert len(tools_dict["tool_calls"]) == 1
    assert tools_dict["tool_calls"][0]["name"] == "litellm_web_search"
    assert tools_dict["tool_calls"][0]["call_id"] == "fc_1"
    assert tools_dict["tool_calls"][0]["input"] == {"query": "latest ai news"}


@pytest.mark.asyncio
async def test_responses_hook_not_triggered_without_tool():
    """No web search tool in the request -> hook must not run."""
    logger = WebSearchInterceptionLogger(enabled_providers=[LlmProviders.OPENAI])

    should_run, tools_dict = await logger.async_should_run_responses_agentic_loop(
        response=_responses_output_with_web_search(),
        model="gpt-4o",
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "name": "get_weather"}],
        stream=False,
        custom_llm_provider="openai",
        kwargs={},
    )

    assert should_run is False
    assert tools_dict == {}


@pytest.mark.asyncio
async def test_responses_hook_not_triggered_for_disabled_provider():
    """Provider not in enabled_providers -> hook must not run."""
    logger = WebSearchInterceptionLogger(enabled_providers=[LlmProviders.BEDROCK])

    should_run, tools_dict = await logger.async_should_run_responses_agentic_loop(
        response=_responses_output_with_web_search(),
        model="gpt-4o",
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "name": "litellm_web_search"}],
        stream=False,
        custom_llm_provider="openai",
        kwargs={},
    )

    assert should_run is False
    assert tools_dict == {}


@pytest.mark.asyncio
async def test_responses_hook_ignores_non_websearch_function_call():
    """A function_call for a different tool must not be intercepted."""
    logger = WebSearchInterceptionLogger(enabled_providers=[LlmProviders.OPENAI])
    response = SimpleNamespace(
        output=[SimpleNamespace(type="function_call", name="get_weather", call_id="c1", arguments="{}")]
    )

    should_run, tools_dict = await logger.async_should_run_responses_agentic_loop(
        response=response,
        model="gpt-4o",
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "name": "litellm_web_search"}],
        stream=False,
        custom_llm_provider="openai",
        kwargs={},
    )

    assert should_run is False
    assert tools_dict == {}


@pytest.mark.asyncio
async def test_responses_hook_ignores_bare_web_search_function_call():
    logger = WebSearchInterceptionLogger(enabled_providers=[LlmProviders.OPENAI])
    response = SimpleNamespace(
        output=[SimpleNamespace(type="function_call", name="web_search", call_id="c1", arguments="{}")]
    )

    should_run, tools_dict = await logger.async_should_run_responses_agentic_loop(
        response=response,
        model="gpt-4o",
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "name": "litellm_web_search"}],
        stream=False,
        custom_llm_provider="openai",
        kwargs={},
    )

    assert should_run is False
    assert tools_dict == {}


@pytest.mark.asyncio
async def test_surface_marker_routes_should_run_to_responses_branch():
    """async_should_run_agentic_loop must dispatch to the responses branch when the
    surface marker says responses.

    Without the marker the default anthropic branch runs and never detects the
    Responses-format function_call, so interception silently no-ops on /v1/responses.
    """
    logger = WebSearchInterceptionLogger(enabled_providers=[LlmProviders.OPENAI])

    should_run, tools_dict = await logger.async_should_run_agentic_loop(
        response=_responses_output_with_web_search(),
        model="gpt-4o",
        messages=[{"role": "user", "content": "What's the latest AI news?"}],
        tools=[{"type": "function", "name": "litellm_web_search"}],
        stream=False,
        custom_llm_provider="openai",
        kwargs={"_agentic_loop_api_surface": RESPONSES_AGENTIC_SURFACE},
    )

    assert should_run is True
    assert tools_dict["response_format"] == "responses"


@pytest.mark.asyncio
async def test_default_branch_does_not_detect_responses_output():
    """Regression guard: the default (anthropic) branch must not detect a
    Responses-format function_call, proving the responses branch is required.
    """
    logger = WebSearchInterceptionLogger(enabled_providers=[LlmProviders.OPENAI])

    should_run, tools_dict = await logger.async_should_run_agentic_loop(
        response=_responses_output_with_web_search(),
        model="gpt-4o",
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "name": "litellm_web_search"}],
        stream=False,
        custom_llm_provider="openai",
        kwargs={},
    )

    assert should_run is False


@pytest.mark.asyncio
async def test_build_responses_plan_produces_responses_input():
    """async_build_responses_agentic_loop_plan builds a Responses-format follow-up:
    the user input followed by function_call + function_call_output items, with
    the web search tool preserved and tool_choice stripped.
    """
    logger = WebSearchInterceptionLogger(enabled_providers=[LlmProviders.OPENAI])

    tools_dict = {
        "tool_calls": [
            {
                "id": "fc_1",
                "call_id": "fc_1",
                "type": "function_call",
                "name": "litellm_web_search",
                "arguments": '{"query": "latest ai news"}',
                "input": {"query": "latest ai news"},
            }
        ],
        "tool_type": "websearch",
        "provider": "openai",
        "response_format": "responses",
    }

    with patch.object(
        logger,
        "_execute_search",
        new=AsyncMock(return_value=("OpenAI shipped a new model", None)),
    ):
        plan = await logger.async_build_responses_agentic_loop_plan(
            tools=tools_dict,
            model="gpt-4o",
            messages=[{"role": "user", "content": "What's the latest AI news?"}],
            response=_responses_output_with_web_search(),
            optional_params={
                "tools": [{"type": "function", "name": "litellm_web_search"}],
                "tool_choice": {"type": "function", "name": "litellm_web_search"},
            },
            logging_obj=MagicMock(),
            stream=False,
            kwargs={
                "custom_llm_provider": "openai",
                "_agentic_loop_api_surface": RESPONSES_AGENTIC_SURFACE,
            },
        )

    assert plan.run_agentic_loop is True
    patch_obj = plan.request_patch
    assert patch_obj is not None
    input_items = patch_obj.messages
    assert input_items is not None

    assert input_items[0] == {"role": "user", "content": "What's the latest AI news?"}
    assert input_items[1] == {
        "type": "function_call",
        "call_id": "fc_1",
        "name": "litellm_web_search",
        "arguments": '{"query": "latest ai news"}',
    }
    assert input_items[2] == {
        "type": "function_call_output",
        "call_id": "fc_1",
        "output": "OpenAI shipped a new model",
    }

    assert patch_obj.tools == [{"type": "function", "name": "litellm_web_search"}]
    assert patch_obj.optional_params["tool_choice"] == "auto"
    assert "_agentic_loop_api_surface" not in patch_obj.kwargs
    assert patch_obj.model == "openai/gpt-4o"


@pytest.mark.asyncio
async def test_deployment_hook_converts_native_responses_web_search_tool():
    """async_pre_call_deployment_hook converts a native Responses web_search tool
    into the flat litellm_web_search function tool (Responses shape, not the
    nested Chat Completions {"function": {...}} wrapper).
    """
    logger = WebSearchInterceptionLogger(enabled_providers=[LlmProviders.OPENAI])

    result = await logger.async_pre_call_deployment_hook(
        kwargs={
            "model": "gpt-4o",
            "custom_llm_provider": "openai",
            "tools": [{"type": "web_search"}],
        },
        call_type=CallTypes.aresponses,
    )

    assert result is not None
    converted_tools = result["tools"]
    assert len(converted_tools) == 1
    tool = converted_tools[0]
    assert tool["type"] == "function"
    assert tool["name"] == "litellm_web_search"
    assert "function" not in tool
    assert tool["parameters"]["required"] == ["query"]


@pytest.mark.asyncio
async def test_deployment_hook_responses_returns_none_without_web_search():
    """No web search tool in a responses request -> deployment hook makes no change."""
    logger = WebSearchInterceptionLogger(enabled_providers=[LlmProviders.OPENAI])

    result = await logger.async_pre_call_deployment_hook(
        kwargs={
            "model": "gpt-4o",
            "custom_llm_provider": "openai",
            "tools": [{"type": "function", "name": "get_weather"}],
        },
        call_type=CallTypes.aresponses,
    )

    assert result is None


@pytest.mark.asyncio
async def test_deployment_hook_responses_converts_stream_to_non_stream():
    """Streaming responses requests are converted to non-streaming so the agentic
    loop can run, and flagged for re-wrapping afterwards.
    """
    logger = WebSearchInterceptionLogger(enabled_providers=[LlmProviders.OPENAI])

    result = await logger.async_pre_call_deployment_hook(
        kwargs={
            "model": "gpt-4o",
            "custom_llm_provider": "openai",
            "tools": [{"type": "web_search_preview"}],
            "stream": True,
        },
        call_type=CallTypes.aresponses,
    )

    assert result is not None
    assert result["stream"] is False
    assert result["_websearch_interception_converted_stream"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("search_type", ["web_search", "web_search_preview", "web_search_preview_2025_03_11"])
async def test_forced_responses_search_names_the_converted_function(search_type: str, stream: bool) -> None:
    logger: Final = WebSearchInterceptionLogger(enabled_providers=[LlmProviders.BEDROCK])
    choice: Final = {"type": search_type}
    result: Final = await logger.async_pre_call_deployment_hook(
        kwargs={
            "model": "us.openai.gpt-6.1-sol",
            "custom_llm_provider": "bedrock",
            "tools": [{"type": search_type}],
            "tool_choice": choice,
            "stream": stream,
        },
        call_type=CallTypes.aresponses,
    )

    assert result is not None
    assert result["tool_choice"] == {"type": "function", "name": result["tools"][0]["name"]}
    assert result["stream"] is False
    assert choice == {"type": search_type}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "choice",
    [
        None,
        "auto",
        "none",
        "required",
        {"type": "function", "name": "get_weather"},
        {"type": "function", "name": "litellm_web_search"},
    ],
)
async def test_responses_search_conversion_preserves_other_tool_choices(choice: object) -> None:
    logger: Final = WebSearchInterceptionLogger(enabled_providers=[LlmProviders.BEDROCK])
    weather: Final = {"type": "function", "name": "get_weather", "parameters": {"type": "object"}}
    result: Final = await logger.async_pre_call_deployment_hook(
        kwargs={
            "model": "us.openai.gpt-6.1-sol",
            "custom_llm_provider": "bedrock",
            "tools": [{"type": "web_search"}, weather],
            "tool_choice": choice,
        },
        call_type=CallTypes.aresponses,
    )

    assert result is not None
    assert result["tool_choice"] == choice
    assert result["tools"][1] is weather


@pytest.mark.asyncio
async def test_forced_responses_search_executes_once_then_allows_the_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.proxy import proxy_server

    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setenv("TAVILY_API_BASE", "https://search.example.test")
    monkeypatch.setattr(litellm, "callbacks", [WebSearchInterceptionLogger(enabled_providers=["bedrock"])])
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "search-canary",
                "litellm_params": {
                    "model": "bedrock/us.openai.gpt-6.1-sol",
                    "api_key": "synthetic-bedrock-bearer",
                    "aws_region_name": "us-east-1",
                },
            }
        ],
        search_tools=[
            {
                "search_tool_name": "fixture-search",
                "litellm_params": {
                    "search_provider": "tavily",
                    "api_key": "synthetic-search-key",
                    "api_base": "https://search.example.test",
                },
            }
        ],
        num_retries=0,
    )
    monkeypatch.setattr(proxy_server, "llm_router", router)
    common: Final = {
        "object": "response",
        "created_at": 1,
        "status": "completed",
        "model": "us.openai.gpt-6.1-sol",
        "usage": {"input_tokens": 11, "output_tokens": 5, "total_tokens": 16},
    }
    with respx.mock as upstream:
        search: Final = upstream.post("https://search.example.test/search").respond(
            200,
            json={
                "results": [
                    {
                        "title": "Synthetic search",
                        "url": "https://example.test/search",
                        "content": "Fixture evidence",
                    }
                ]
            },
        )
        responses_route: Final = upstream.post(
            "https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1/responses"
        ).mock(
            side_effect=[
                httpx.Response(
                    200,
                    json={
                        **common,
                        "id": "resp_search",
                        "output": [
                            {
                                "type": "function_call",
                                "id": "fc_search",
                                "call_id": "call_search",
                                "name": "litellm_web_search",
                                "arguments": '{"query":"synthetic query"}',
                                "status": "completed",
                            }
                        ],
                    },
                ),
                httpx.Response(
                    200,
                    json={
                        **common,
                        "id": "resp_answer",
                        "output": [
                            {
                                "type": "message",
                                "id": "msg_answer",
                                "role": "assistant",
                                "status": "completed",
                                "content": [{"type": "output_text", "text": "Answer from search", "annotations": []}],
                            }
                        ],
                    },
                ),
            ]
        )
        result: Final = await router.aresponses(
            model="search-canary",
            input="Search for synthetic evidence",
            max_output_tokens=100,
            tools=[{"type": "web_search"}],
            tool_choice={"type": "web_search"},
        )

    assert result.output[0].content[0].text == "Answer from search"
    assert len(responses_route.calls) == 2
    first: Final = json.loads(responses_route.calls[0].request.content)
    followup: Final = json.loads(responses_route.calls[1].request.content)
    assert first["tool_choice"] == {"type": "function", "name": first["tools"][0]["name"]}
    assert followup["tool_choice"] == "auto"
    assert followup["input"][-1] == {
        "type": "function_call_output",
        "call_id": "call_search",
        "output": "Title: Synthetic search\nURL: https://example.test/search\nSnippet: Fixture evidence",
    }
    assert len(search.calls) == 1
    assert json.loads(search.calls[0].request.content)["query"] == "synthetic query"
