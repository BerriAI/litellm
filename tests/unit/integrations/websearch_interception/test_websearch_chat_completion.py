"""
Integration tests for WebSearch interception with chat completions API.

Tests the end-to-end flow of websearch_interception callback with
litellm.acompletion() for transparent server-side web search execution.
"""

import datetime
import json
from typing import Final
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import respx

import litellm
from litellm.integrations.websearch_interception.handler import (
    WebSearchInterceptionLogger,
)
from litellm.integrations.websearch_interception.tools import get_litellm_web_search_tool_openai
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.types.utils import CallTypes, LlmProviders, ModelResponse

from .conftest import FINAL_ANSWER, offered_tool_names, request_bodies


@pytest.fixture
def mock_search_response():
    """Mock search response from litellm.asearch()"""
    mock_response = MagicMock()
    mock_response.results = [
        MagicMock(
            title="Weather in San Francisco",
            url="https://weather.com/sf",
            snippet="Current weather: 65°F, partly cloudy",
        )
    ]
    return mock_response


@pytest.fixture
def websearch_logger():
    """Create a WebSearchInterceptionLogger instance"""
    return WebSearchInterceptionLogger(enabled_providers=[LlmProviders.OPENAI, LlmProviders.MINIMAX])


@pytest.mark.asyncio
async def test_websearch_chat_completion_hook_detection():
    """Test that websearch hook correctly detects tool calls in response."""
    from litellm.types.utils import (
        ChatCompletionMessageToolCall,
        Choices,
        Function,
        Message,
    )

    websearch_logger = WebSearchInterceptionLogger(enabled_providers=[LlmProviders.OPENAI])

    # Mock response with litellm_web_search tool call
    mock_response = ModelResponse(
        id="test-123",
        choices=[
            Choices(
                finish_reason="tool_calls",
                index=0,
                message=Message(
                    role="assistant",
                    content=None,
                    tool_calls=[
                        ChatCompletionMessageToolCall(
                            id="call_123",
                            type="function",
                            function=Function(
                                name="litellm_web_search",
                                arguments='{"query": "weather in SF"}',
                            ),
                        )
                    ],
                ),
            )
        ],
        model="gpt-4o",
        object="chat.completion",
        created=1234567890,
    )

    # Test should_run_chat_completion_agentic_loop
    should_run, tools_dict = await websearch_logger.async_should_run_chat_completion_agentic_loop(
        response=mock_response,
        model="gpt-4o",
        messages=[{"role": "user", "content": "What's the weather?"}],
        tools=[
            {
                "type": "function",
                "function": {"name": "litellm_web_search"},
            }
        ],
        stream=False,
        custom_llm_provider="openai",
        kwargs={},
    )

    # Verify hook detected the tool call
    assert should_run is True
    assert "tool_calls" in tools_dict
    assert len(tools_dict["tool_calls"]) == 1
    assert tools_dict["tool_calls"][0]["name"] == "litellm_web_search"
    assert tools_dict["response_format"] == "openai"


@pytest.mark.asyncio
async def test_websearch_not_triggered_without_tool():
    """Test that websearch hook is NOT triggered when no web search tool in request."""
    from litellm.types.utils import Choices, Message

    websearch_logger = WebSearchInterceptionLogger(enabled_providers=[LlmProviders.OPENAI])

    mock_response = ModelResponse(
        id="test-123",
        choices=[
            Choices(
                finish_reason="stop",
                index=0,
                message=Message(
                    role="assistant",
                    content="Here's the answer",
                    tool_calls=None,
                ),
            )
        ],
        model="gpt-4o",
        object="chat.completion",
        created=1234567890,
    )

    # Test without web search tool
    should_run, tools_dict = await websearch_logger.async_should_run_chat_completion_agentic_loop(
        response=mock_response,
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello"}],
        tools=[
            {
                "type": "function",
                "function": {"name": "some_other_tool"},
            }
        ],
        stream=False,
        custom_llm_provider="openai",
        kwargs={},
    )

    # Verify hook did NOT trigger
    assert should_run is False
    assert tools_dict == {}


@pytest.mark.asyncio
async def test_websearch_not_triggered_for_disabled_provider():
    """Test that websearch hook is NOT triggered for providers not in enabled_providers."""
    from litellm.types.utils import (
        ChatCompletionMessageToolCall,
        Choices,
        Function,
        Message,
    )

    # Only enable bedrock
    websearch_logger = WebSearchInterceptionLogger(enabled_providers=[LlmProviders.BEDROCK])

    mock_response = ModelResponse(
        id="test-123",
        choices=[
            Choices(
                finish_reason="tool_calls",
                index=0,
                message=Message(
                    role="assistant",
                    content=None,
                    tool_calls=[
                        ChatCompletionMessageToolCall(
                            id="call_123",
                            type="function",
                            function=Function(
                                name="litellm_web_search",
                                arguments='{"query": "test"}',
                            ),
                        )
                    ],
                ),
            )
        ],
        model="gpt-4o",
        object="chat.completion",
        created=1234567890,
    )

    # Test with OpenAI provider (not enabled)
    should_run, tools_dict = await websearch_logger.async_should_run_chat_completion_agentic_loop(
        response=mock_response,
        model="gpt-4o",
        messages=[{"role": "user", "content": "test"}],
        tools=[
            {
                "type": "function",
                "function": {"name": "litellm_web_search"},
            }
        ],
        stream=False,
        custom_llm_provider="openai",  # Not in enabled_providers
        kwargs={},
    )

    # Verify hook did NOT trigger
    assert should_run is False
    assert tools_dict == {}


@pytest.mark.asyncio
async def test_websearch_json_serialization_fix():
    """Test that tool call arguments are properly JSON serialized.

    Regression test for the bug where arguments were converted to Python
    string representation instead of proper JSON, causing providers like
    MiniMax to reject requests with 'invalid function arguments json string'.
    """
    from litellm.integrations.websearch_interception.transformation import (
        WebSearchTransformation,
    )

    # Mock tool calls with dict input
    tool_calls = [
        {
            "id": "call_123",
            "name": "litellm_web_search",
            "input": {"query": "weather in SF"},  # Dict input
        }
    ]

    search_results = ["Weather: 65°F, partly cloudy"]

    # Transform to OpenAI format
    assistant_message, tool_messages = WebSearchTransformation.transform_response(
        tool_calls=tool_calls,
        search_results=search_results,
        response_format="openai",
    )

    # Verify arguments are properly JSON serialized
    import json

    arguments_str = assistant_message["tool_calls"][0]["function"]["arguments"]

    # Should be valid JSON
    parsed_args = json.loads(arguments_str)
    assert parsed_args == {"query": "weather in SF"}

    # Should NOT be Python string representation like "{'query': 'weather in SF'}"
    assert arguments_str == '{"query": "weather in SF"}'
    assert arguments_str != "{'query': 'weather in SF'}"


@pytest.mark.asyncio
async def test_maybe_run_chat_completion_agentic_loop_calls_chat_completion_hook():
    """Regression test: maybe_run_chat_completion_agentic_loop must call
    async_should_run_chat_completion_agentic_loop, not async_should_run_agentic_loop.

    Before the fix, the function used the wrong gate check and wrong hook,
    causing WebSearchInterceptionLogger to never intercept chat completion requests
    even when the LLM returned a litellm_web_search tool call.
    """
    from litellm.litellm_core_utils.chat_completion_agentic_loop import (
        maybe_run_chat_completion_agentic_loop,
    )
    from litellm.types.utils import (
        ChatCompletionMessageToolCall,
        Choices,
        Function,
        Message,
    )

    mock_response = ModelResponse(
        id="test-regression-123",
        choices=[
            Choices(
                finish_reason="tool_calls",
                index=0,
                message=Message(
                    role="assistant",
                    content=None,
                    tool_calls=[
                        ChatCompletionMessageToolCall(
                            id="call_abc",
                            type="function",
                            function=Function(
                                name="litellm_web_search",
                                arguments='{"query": "latest news"}',
                            ),
                        )
                    ],
                ),
            )
        ],
        model="gpt-4o",
        object="chat.completion",
        created=1234567890,
    )

    sentinel = ModelResponse(
        id="sentinel-final",
        choices=[
            Choices(
                finish_reason="stop",
                index=0,
                message=Message(role="assistant", content="Here is the news."),
            )
        ],
        model="gpt-4o",
        object="chat.completion",
        created=1234567890,
    )

    websearch_logger = WebSearchInterceptionLogger(enabled_providers=[LlmProviders.OPENAI])

    chat_completion_hook_called = False

    async def fake_should_run_chat_completion(response, model, messages, tools, stream, custom_llm_provider, kwargs):
        nonlocal chat_completion_hook_called
        chat_completion_hook_called = True
        return True, {
            "tool_calls": [{"id": "call_abc", "name": "litellm_web_search", "input": {"query": "latest news"}}],
            "tool_type": "websearch",
            "provider": "openai",
            "response_format": "openai",
        }

    async def fake_build_plan(tools, model, messages, response, optional_params, logging_obj, stream, kwargs):
        from litellm.types.integrations.custom_logger import AgenticLoopPlan

        return AgenticLoopPlan(run_agentic_loop=False, response_override=sentinel)

    websearch_logger.async_should_run_chat_completion_agentic_loop = fake_should_run_chat_completion
    websearch_logger.async_build_chat_completion_agentic_loop_plan = fake_build_plan

    import litellm as _litellm

    original_callbacks = _litellm.callbacks[:]
    _litellm.callbacks = [websearch_logger]

    mock_logging_obj = MagicMock()
    mock_logging_obj.dynamic_success_callbacks = None

    try:
        result = await maybe_run_chat_completion_agentic_loop(
            response=mock_response,
            model="gpt-4o",
            messages=[{"role": "user", "content": "Latest news?"}],
            optional_params={
                "tools": [
                    {
                        "type": "function",
                        "function": {"name": "litellm_web_search"},
                    }
                ]
            },
            kwargs={},
            logging_obj=mock_logging_obj,
            custom_llm_provider="openai",
            stream=False,
        )
    finally:
        _litellm.callbacks = original_callbacks

    assert chat_completion_hook_called, (
        "async_should_run_chat_completion_agentic_loop was never called; "
        "maybe_run_chat_completion_agentic_loop used the wrong hook"
    )
    assert result is sentinel, "Expected agentic loop to return sentinel final response"


@pytest.mark.asyncio
async def test_execute_chat_completion_agentic_loop_strips_tool_choice():
    """Regression: _execute_chat_completion_agentic_loop must not forward tool_choice
    from the original request into the follow-up synthesis call.

    When the original request forces tool_choice to litellm_web_search, merging
    optional_params into the follow-up params without explicit removal causes the
    model to call the search tool again instead of synthesizing an answer.
    """

    websearch_logger = WebSearchInterceptionLogger(enabled_providers=[LlmProviders.OPENAI])

    captured_kwargs: dict = {}

    async def fake_acompletion(**kwargs):
        captured_kwargs.update(kwargs)
        return ModelResponse(id="followup", model="gpt-4o", object="chat.completion")

    async def fake_search(query):
        return ("Bitcoin price is $60,000", None)

    with patch.object(websearch_logger, "_execute_search", side_effect=fake_search):
        with patch("litellm.acompletion", side_effect=fake_acompletion):
            await websearch_logger._execute_chat_completion_agentic_loop(
                model="gpt-4o",
                messages=[{"role": "user", "content": "What is Bitcoin price?"}],
                tool_calls=[
                    {
                        "id": "call_1",
                        "name": "litellm_web_search",
                        "input": {"query": "bitcoin price"},
                    }
                ],
                optional_params={
                    "tools": [{"type": "function", "function": {"name": "litellm_web_search"}}],
                    "tool_choice": {"type": "function", "function": {"name": "litellm_web_search"}},
                    "max_tokens": 512,
                },
                logging_obj=MagicMock(),
                stream=False,
                kwargs={},
            )

    assert "tool_choice" not in captured_kwargs, (
        "tool_choice must not appear in follow-up acompletion kwargs; "
        "it would force the model to call the search tool again instead of synthesizing"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["auto", "required"])
async def test_execute_chat_completion_agentic_loop_keeps_an_allowed_tools_choice_without_forcing_another_call(
    mode: str,
    perplexity_search: respx.Route,
    respx_mock: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    litellm.in_memory_llm_clients_cache.flush_cache()
    followup: Final = respx_mock.post("https://api.openai.com/v1/chat/completions").respond(
        json={
            "id": "followup",
            "object": "chat.completion",
            "created": 0,
            "model": "gpt-4o",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "done"}, "finish_reason": "stop"}],
        }
    )
    tool_choice: Final = {
        "type": "allowed_tools",
        "allowed_tools": {
            "mode": mode,
            "tools": [{"type": "function", "function": {"name": "litellm_web_search"}}],
        },
    }

    await WebSearchInterceptionLogger(enabled_providers=[LlmProviders.OPENAI])._execute_chat_completion_agentic_loop(
        model="gpt-4o",
        messages=[{"role": "user", "content": "What is Bitcoin price?"}],
        tool_calls=[{"id": "call_1", "name": "litellm_web_search", "input": {"query": "bitcoin price"}}],
        optional_params={
            "tools": [
                {"type": "function", "function": {"name": "litellm_web_search"}},
                {"type": "function", "function": {"name": "get_time"}},
            ],
            "tool_choice": tool_choice,
        },
        logging_obj=MagicMock(),
        stream=False,
        kwargs={},
    )

    assert perplexity_search.call_count == 1
    assert json.loads(followup.calls.last.request.content)["tool_choice"] == {
        "type": "allowed_tools",
        "allowed_tools": {"mode": "auto", "tools": [{"type": "function", "function": {"name": "litellm_web_search"}}]},
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool_choice", "followup_tool_choice"),
    [
        (
            {
                "type": "allowed_tools",
                "allowed_tools": {"mode": "required", "tools": [{"type": "function", "function": {"name": "get_time"}}]},
            },
            {
                "type": "allowed_tools",
                "allowed_tools": {"mode": "auto", "tools": [{"type": "function", "function": {"name": "get_time"}}]},
            },
        ),
        ({"type": "function", "function": {"name": "litellm_web_search"}}, None),
        ("required", None),
    ],
)
async def test_chat_followup_plan_keeps_only_an_allowed_tools_choice(
    tool_choice: object, followup_tool_choice: object, perplexity_search: respx.Route
) -> None:
    plan: Final = await WebSearchInterceptionLogger().async_build_chat_completion_agentic_loop_plan(
        tools={
            "tool_calls": [{"id": "call_abc", "name": "litellm_web_search", "input": {"query": "latest ai news"}}],
            "tool_type": "websearch",
            "provider": "openai",
            "response_format": "openai",
        },
        model="gpt-4o",
        messages=[{"role": "user", "content": "What's the latest AI news?"}],
        response=MagicMock(),
        optional_params={"tool_choice": tool_choice, "max_tokens": 512},
        logging_obj=None,
        stream=False,
        kwargs={"custom_llm_provider": "openai"},
    )

    assert plan.request_patch is not None
    assert plan.request_patch.optional_params.get("tool_choice") == followup_tool_choice


if __name__ == "__main__":
    # Run with: pytest test_websearch_chat_completion.py -v -s
    pytest.main([__file__, "-v", "-s"])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "requested_model,handler_model,custom_llm_provider",
    [
        ("openrouter/openrouter/free", "openrouter/free", "openrouter"),
        ("bedrock/converse_like/my-endpoint-model", "my-endpoint-model", "bedrock"),
    ],
)
async def test_chat_followup_reuses_the_model_the_caller_requested(
    requested_model: str, handler_model: str, custom_llm_provider: str, perplexity_search: respx.Route
) -> None:
    """The provider handler sees the model after its routing prefixes are stripped, which
    re-qualifying with the provider alone cannot restore."""
    logging_obj: Final = Logging(
        model=requested_model,
        messages=[],
        stream=False,
        call_type=CallTypes.completion.value,
        start_time=datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc),
        litellm_call_id="call-chat-followup",
        function_id="fn-chat-followup",
    )
    litellm.completion(
        model=requested_model,
        messages=[{"role": "user", "content": "What's the latest AI news?"}],
        mock_response="searching",
        litellm_logging_obj=logging_obj,
    )
    logger: Final = WebSearchInterceptionLogger()

    plan: Final = await logger.async_build_chat_completion_agentic_loop_plan(
        tools={
            "tool_calls": [{"id": "call_abc", "name": "litellm_web_search", "input": {"query": "latest ai news"}}],
            "tool_type": "websearch",
            "provider": custom_llm_provider,
            "response_format": "openai",
        },
        model=handler_model,
        messages=[{"role": "user", "content": "What's the latest AI news?"}],
        response=MagicMock(),
        optional_params={},
        logging_obj=logging_obj,
        stream=False,
        kwargs={"custom_llm_provider": custom_llm_provider},
    )

    assert plan.request_patch is not None
    assert plan.request_patch.model == requested_model
    assert perplexity_search.call_count == 1


WEATHER_FUNCTION_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
}
SEARCH_TOOL_NAME = "litellm_web_search"
NEWS_QUESTION = [{"role": "user", "content": "What's the latest AI news?"}]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("feature_max_loops", "deployment_max_loops", "searches"),
    [(None, 4, 4), (4, None, 3), (1, None, 3)],
    ids=["deployment cap above the default", "feature cap above the default", "feature cap below the default"],
)
async def test_anthropic_chat_turn_stops_at_the_deployment_cap_with_the_model_answer(
    feature_max_loops: int | None,
    deployment_max_loops: int | None,
    searches: int,
    monkeypatch: pytest.MonkeyPatch,
    anthropic_model: respx.Route,
    perplexity_search: respx.Route,
):
    monkeypatch.setattr(
        litellm,
        "callbacks",
        [WebSearchInterceptionLogger(enabled_providers=["anthropic"], max_agentic_loops=feature_max_loops)],
    )

    response = await litellm.acompletion(
        model="anthropic/claude-sonnet-5-5",
        messages=NEWS_QUESTION,
        tools=[get_litellm_web_search_tool_openai(), WEATHER_FUNCTION_TOOL],
        max_agentic_loops=deployment_max_loops,
    )

    assert response.choices[0].message.content == FINAL_ANSWER
    assert not response.choices[0].message.tool_calls
    assert [offered_tool_names(body) for body in request_bodies(anthropic_model)] == [
        *[[SEARCH_TOOL_NAME, "get_weather"]] * (searches + 1),
        ["get_weather"],
    ]
    assert perplexity_search.call_count == searches


@pytest.mark.asyncio
async def test_anthropic_chat_last_follow_up_blocks_tool_calls_when_search_was_the_only_tool(
    monkeypatch, anthropic_model, perplexity_search
):
    """Claude can call a tool it saw earlier in the history even when the request offers none, so the call is blocked."""
    monkeypatch.setattr(
        litellm, "callbacks", [WebSearchInterceptionLogger(enabled_providers=["anthropic"])]
    )

    response = await litellm.acompletion(
        max_agentic_loops=1,
        model="anthropic/claude-sonnet-5-5",
        messages=NEWS_QUESTION,
        tools=[get_litellm_web_search_tool_openai()],
    )

    last_request = request_bodies(anthropic_model)[-1]
    assert response.choices[0].message.content == FINAL_ANSWER
    assert SEARCH_TOOL_NAME not in offered_tool_names(last_request)
    assert last_request["tool_choice"] == {"type": "none"}


@pytest.mark.asyncio
async def test_openai_chat_turn_at_the_cap_ends_with_the_model_answer(
    monkeypatch, openai_chat_model, perplexity_search
):
    """reasoning_effort none keeps the request on /v1/chat/completions, where the loop runs inside the OpenAI handler."""
    monkeypatch.setattr(
        litellm, "callbacks", [WebSearchInterceptionLogger(enabled_providers=["openai"])]
    )

    response = await litellm.acompletion(
        max_agentic_loops=2,
        model="openai/gpt-5.6",
        messages=NEWS_QUESTION,
        tools=[get_litellm_web_search_tool_openai(), WEATHER_FUNCTION_TOOL],
        reasoning_effort="none",
        api_key="test-openai-key-capped-chat",
    )

    assert response.choices[0].message.content == FINAL_ANSWER
    assert [offered_tool_names(body) for body in request_bodies(openai_chat_model)] == [
        *[[SEARCH_TOOL_NAME, "get_weather"]] * 3,
        ["get_weather"],
    ]
    assert perplexity_search.call_count == 2


@pytest.mark.asyncio
async def test_openai_chat_bridged_to_responses_at_the_cap_ends_with_the_model_answer(
    monkeypatch, openai_responses_model, perplexity_search
):
    monkeypatch.setattr(
        litellm, "callbacks", [WebSearchInterceptionLogger(enabled_providers=["openai"])]
    )

    response = await litellm.acompletion(
        max_agentic_loops=2,
        model="openai/gpt-5.6",
        messages=NEWS_QUESTION,
        tools=[get_litellm_web_search_tool_openai(), WEATHER_FUNCTION_TOOL],
        api_key="test-openai-key-bridged-chat",
    )

    assert response.choices[0].message.content == FINAL_ANSWER
    assert [offered_tool_names(body) for body in request_bodies(openai_responses_model)] == [
        *[[SEARCH_TOOL_NAME, "get_weather"]] * 3,
        ["get_weather"],
    ]
    assert perplexity_search.call_count == 2


@pytest.mark.asyncio
async def test_openai_chat_last_follow_up_sends_no_tools_when_search_was_the_only_one(
    monkeypatch, openai_chat_model, perplexity_search
):
    monkeypatch.setattr(
        litellm, "callbacks", [WebSearchInterceptionLogger(enabled_providers=["openai"])]
    )

    response = await litellm.acompletion(
        max_agentic_loops=1,
        model="openai/gpt-5.6",
        messages=NEWS_QUESTION,
        tools=[get_litellm_web_search_tool_openai()],
        tool_choice="required",
        reasoning_effort="none",
        api_key="test-openai-key-search-only-chat",
    )

    last_request = request_bodies(openai_chat_model)[-1]
    assert response.choices[0].message.content == FINAL_ANSWER
    assert "tools" not in last_request
    assert "tool_choice" not in last_request
    assert len(request_bodies(openai_chat_model)) == 3


@pytest.mark.asyncio
async def test_a_search_call_without_a_query_runs_no_search_and_says_so(perplexity_search):
    plan = await WebSearchInterceptionLogger().async_build_chat_completion_agentic_loop_plan(
        tools={
            "tool_calls": [
                {"id": "call_with_query", "name": SEARCH_TOOL_NAME, "input": {"query": "latest ai news"}},
                {"id": "call_without_query", "name": SEARCH_TOOL_NAME, "input": {}},
            ],
            "response_format": "openai",
        },
        model="gpt-5.6",
        messages=NEWS_QUESTION,
        response=MagicMock(),
        optional_params={},
        logging_obj=None,
        stream=False,
        kwargs={"custom_llm_provider": "openai"},
    )

    assert plan.request_patch is not None
    tool_results = {
        message["tool_call_id"]: message["content"]
        for message in plan.request_patch.messages
        if message.get("role") == "tool"
    }
    assert tool_results["call_without_query"] == "No search query provided"
    assert "AI news" in tool_results["call_with_query"]
    assert perplexity_search.call_count == 1
