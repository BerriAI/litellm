"""
Unit tests for WebSearchInterceptionLogger on the OpenAI Responses API path.

Covers the call_type=="aresponses" branch of async_pre_call_deployment_hook,
the new Responses-API output parser, the should-run hook, and the agentic
loop that rebuilds the input chain with function_call_output items.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import litellm
from litellm.integrations.websearch_interception.handler import (
    WebSearchInterceptionLogger,
)
from litellm.integrations.websearch_interception.tools import (
    get_litellm_web_search_tool_responses_api,
    is_web_search_tool_responses_api,
)
from litellm.integrations.websearch_interception.transformation import (
    WebSearchTransformation,
)
from litellm.types.utils import LlmProviders


@pytest.fixture
def logger() -> WebSearchInterceptionLogger:
    return WebSearchInterceptionLogger(
        enabled_providers=[LlmProviders.BEDROCK_MANTLE, LlmProviders.OPENAI],
        search_tool_name="tavily-search",
    )


def test_is_web_search_tool_responses_api_detects_codex_shape() -> None:
    assert is_web_search_tool_responses_api({"type": "web_search"})
    assert is_web_search_tool_responses_api({"type": "web_search_preview"})
    assert is_web_search_tool_responses_api({"type": "web_search_20250305"})
    assert is_web_search_tool_responses_api(
        {"type": "function", "name": "litellm_web_search"}
    )
    assert not is_web_search_tool_responses_api({"type": "function", "name": "foo"})
    assert not is_web_search_tool_responses_api({"type": "image_generation"})
    assert not is_web_search_tool_responses_api({})


def test_get_litellm_web_search_tool_responses_api_is_flat() -> None:
    tool = get_litellm_web_search_tool_responses_api()
    # Flat shape — no nested ``function`` key (which would be Chat-Completions style).
    assert tool["type"] == "function"
    assert tool["name"] == "litellm_web_search"
    assert "parameters" in tool
    assert "function" not in tool


def test_responses_api_response_parser_finds_function_call() -> None:
    response = {
        "output": [
            {"type": "reasoning", "id": "rs_1", "summary": []},
            {
                "type": "function_call",
                "call_id": "call_abc",
                "name": "litellm_web_search",
                "arguments": json.dumps({"query": "weather hong kong"}),
            },
        ]
    }
    should, calls = WebSearchTransformation.transform_request(
        response=response, stream=False, response_format="responses"
    )
    assert should is True
    assert len(calls) == 1
    assert calls[0]["call_id"] == "call_abc"
    assert calls[0]["input"]["query"] == "weather hong kong"


def test_responses_api_response_parser_ignores_other_function_calls() -> None:
    response = {
        "output": [
            {
                "type": "function_call",
                "call_id": "call_xyz",
                "name": "calculator",
                "arguments": json.dumps({"x": 1}),
            }
        ]
    }
    should, calls = WebSearchTransformation.transform_request(
        response=response, stream=False, response_format="responses"
    )
    assert should is False
    assert calls == []


@pytest.mark.asyncio
async def test_pre_call_hook_aresponses_uses_flat_shape(
    logger: WebSearchInterceptionLogger,
) -> None:
    kwargs = {
        "model": "openai.gpt-5.5",
        "custom_llm_provider": "bedrock_mantle",
        "tools": [{"type": "web_search"}],
        "stream": True,
    }
    out = await logger.async_pre_call_deployment_hook(kwargs, call_type="aresponses")
    assert out is not None
    converted = out["tools"][0]
    # Flat — no nested ``function`` key.
    assert converted["type"] == "function"
    assert converted["name"] == "litellm_web_search"
    assert "function" not in converted
    # Streaming converted to non-streaming for interception.
    assert out["stream"] is False
    assert out["_websearch_interception_converted_stream"] is True


@pytest.mark.asyncio
async def test_pre_call_hook_aresponses_skips_disabled_provider(
    logger: WebSearchInterceptionLogger,
) -> None:
    # vertex_ai isn't in the logger's enabled list.
    kwargs = {
        "model": "publishers/google/models/gemini-1.5-pro",
        "custom_llm_provider": "vertex_ai",
        "tools": [{"type": "web_search"}],
    }
    out = await logger.async_pre_call_deployment_hook(kwargs, call_type="aresponses")
    assert out is None


@pytest.mark.asyncio
async def test_should_run_responses_api_hook_returns_tool_calls(
    logger: WebSearchInterceptionLogger,
) -> None:
    response = {
        "output": [
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "litellm_web_search",
                "arguments": json.dumps({"query": "btc price"}),
            }
        ]
    }
    should, hook_tools = await logger.async_should_run_responses_api_agentic_loop(
        response=response,
        model="openai.gpt-5.5",
        input="hi",
        tools=[{"type": "web_search"}],
        stream=False,
        custom_llm_provider="bedrock_mantle",
        kwargs={},
    )
    assert should is True
    assert hook_tools["response_format"] == "responses"
    assert len(hook_tools["tool_calls"]) == 1


@pytest.mark.asyncio
async def test_run_responses_api_hook_preserves_assistant_turn_then_appends_outputs(
    logger: WebSearchInterceptionLogger,
) -> None:
    """The follow-up ``input`` must carry the user message, then the entire
    first-turn assistant ``output`` (reasoning + function_call), then the
    paired ``function_call_output`` items. Strict providers (OpenAI-native)
    400 if a ``function_call_output`` is not preceded by its matching
    ``function_call`` from the same conversation turn — see
    https://platform.openai.com/docs/api-reference/responses."""

    fake_search = MagicMock()
    fake_search.results = [
        MagicMock(title="t", url="https://example.com", snippet="snippet"),
    ]

    captured = {}

    async def fake_aresponses(**kwargs):
        captured.update(kwargs)
        return {"id": "resp_final", "output": [{"type": "message"}]}

    initial_response = {
        "id": "resp_initial",
        "output": [
            {"type": "reasoning", "id": "rs_1", "summary": []},
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "litellm_web_search",
                "arguments": '{"query": "btc price"}',
            },
        ],
    }

    with (
        patch.object(litellm, "asearch", new=AsyncMock(return_value=fake_search)),
        patch.object(litellm, "aresponses", new=AsyncMock(side_effect=fake_aresponses)),
    ):
        await logger.async_run_responses_api_agentic_loop(
            tools={
                "tool_calls": [
                    {
                        "id": "call_1",
                        "call_id": "call_1",
                        "type": "function_call",
                        "name": "litellm_web_search",
                        "input": {"query": "btc price"},
                        "arguments": {"query": "btc price"},
                    }
                ],
                "response_format": "responses",
            },
            model="openai.gpt-5.5",
            input="What's the BTC price?",
            response=initial_response,
            response_api_optional_request_params={"tools": [{"type": "function"}]},
            litellm_params={},
            logging_obj=MagicMock(),
            stream=False,
            kwargs={},
        )

    follow_up_input = captured["input"]
    assert follow_up_input[0] == {"role": "user", "content": "What's the BTC price?"}
    assert follow_up_input[1]["type"] == "reasoning"
    assert follow_up_input[2]["type"] == "function_call"
    assert follow_up_input[2]["call_id"] == "call_1"
    assert follow_up_input[2]["name"] == "litellm_web_search"
    assert follow_up_input[3]["type"] == "function_call_output"
    assert follow_up_input[3]["call_id"] == "call_1"
    assert "snippet" in follow_up_input[3]["output"]


@pytest.mark.asyncio
async def test_run_responses_api_hook_aborts_on_max_loops(
    logger: WebSearchInterceptionLogger,
) -> None:
    """Depth guard prevents a model from inducing unbounded recursion by
    keeping it returning the same ``litellm_web_search`` tool call."""

    asearch_mock = AsyncMock()
    aresponses_mock = AsyncMock()

    with (
        patch.object(litellm, "asearch", new=asearch_mock),
        patch.object(litellm, "aresponses", new=aresponses_mock),
    ):
        with pytest.raises(ValueError, match="exceeded max_agentic_loops"):
            await logger.async_run_responses_api_agentic_loop(
                tools={
                    "tool_calls": [
                        {
                            "id": "c",
                            "call_id": "c",
                            "type": "function_call",
                            "name": "litellm_web_search",
                            "input": {"query": "q"},
                            "arguments": {"query": "q"},
                        }
                    ],
                    "response_format": "responses",
                },
                model="openai.gpt-5.5",
                input="hi",
                response={"id": "r", "output": []},
                response_api_optional_request_params={},
                litellm_params={},
                logging_obj=MagicMock(),
                stream=False,
                kwargs={
                    "_agentic_loop_depth": 3,
                    "max_agentic_loops": 3,
                    "_agentic_loop_fingerprints": [],
                },
            )

    # Guard MUST run before any work — capped-out clients shouldn't burn
    # parallel Tavily calls per iteration.
    asearch_mock.assert_not_called()
    aresponses_mock.assert_not_called()


@pytest.mark.asyncio
async def test_run_responses_api_hook_propagates_max_agentic_loops(
    logger: WebSearchInterceptionLogger,
) -> None:
    """Re-entrant call must carry ``max_agentic_loops`` forward; otherwise a
    deployment-configured cap silently resets to the default on each hop."""

    fake_search = MagicMock()
    fake_search.results = [MagicMock(title="t", url="u", snippet="s")]
    captured = {}

    async def fake_aresponses(**kwargs):
        captured.update(kwargs)
        return {"id": "r2", "output": []}

    with (
        patch.object(litellm, "asearch", new=AsyncMock(return_value=fake_search)),
        patch.object(litellm, "aresponses", new=AsyncMock(side_effect=fake_aresponses)),
    ):
        await logger.async_run_responses_api_agentic_loop(
            tools={
                "tool_calls": [
                    {
                        "id": "c",
                        "call_id": "c",
                        "type": "function_call",
                        "name": "litellm_web_search",
                        "input": {"query": "q"},
                        "arguments": {"query": "q"},
                    }
                ],
                "response_format": "responses",
            },
            model="openai.gpt-5.5",
            input="hi",
            response={"id": "r1", "output": []},
            response_api_optional_request_params={},
            litellm_params={},
            logging_obj=MagicMock(),
            stream=False,
            kwargs={
                "_agentic_loop_depth": 1,
                "max_agentic_loops": 7,
                "_agentic_loop_fingerprints": [],
            },
        )

    assert captured["max_agentic_loops"] == 7
    assert captured["_agentic_loop_depth"] == 2
    assert len(captured["_agentic_loop_fingerprints"]) == 1


@pytest.mark.asyncio
async def test_run_responses_api_hook_propagates_caller_attribution(
    logger: WebSearchInterceptionLogger,
) -> None:
    """Internal follow-up call must carry caller-attribution fields
    (``metadata``, ``litellm_metadata``, ``user``, ``user_api_key_*``) so
    proxy budget / spend logging accounts the call against the original
    API key and team rather than an empty owner."""

    fake_search = MagicMock()
    fake_search.results = [MagicMock(title="t", url="u", snippet="s")]
    captured = {}

    async def fake_aresponses(**kwargs):
        captured.update(kwargs)
        return {"id": "r2", "output": []}

    with (
        patch.object(litellm, "asearch", new=AsyncMock(return_value=fake_search)),
        patch.object(litellm, "aresponses", new=AsyncMock(side_effect=fake_aresponses)),
    ):
        await logger.async_run_responses_api_agentic_loop(
            tools={
                "tool_calls": [
                    {
                        "id": "c",
                        "call_id": "c",
                        "type": "function_call",
                        "name": "litellm_web_search",
                        "input": {"query": "q"},
                        "arguments": {"query": "q"},
                    }
                ],
                "response_format": "responses",
            },
            model="openai.gpt-5.5",
            input="hi",
            response={"id": "r1", "output": []},
            response_api_optional_request_params={},
            litellm_params={
                "metadata": {"trace_id": "abc"},
                "litellm_metadata": {"user_api_key_alias": "qmachu"},
                "user": "u-1",
                "user_api_key": "sk-test",
                "user_api_key_user_id": "u-1",
                "user_api_key_team_id": "t-1",
                "user_api_key_team_alias": "team-a",
                "user_api_key_org_id": "o-1",
                "proxy_server_request": {"arrival_time": "now"},
            },
            logging_obj=MagicMock(),
            stream=False,
            kwargs={},
        )

    assert captured["metadata"] == {"trace_id": "abc"}
    assert captured["litellm_metadata"] == {"user_api_key_alias": "qmachu"}
    assert captured["user"] == "u-1"
    assert captured["user_api_key"] == "sk-test"
    assert captured["user_api_key_team_id"] == "t-1"
    assert captured["user_api_key_team_alias"] == "team-a"
    assert captured["user_api_key_org_id"] == "o-1"
    assert captured["proxy_server_request"] == {"arrival_time": "now"}


@pytest.mark.asyncio
async def test_run_responses_api_hook_aborts_on_repeated_fingerprint(
    logger: WebSearchInterceptionLogger,
) -> None:
    """Cycle break: same tool_calls fingerprint twice in a row aborts."""

    fake_search = MagicMock()
    fake_search.results = [MagicMock(title="t", url="u", snippet="s")]

    tool_calls = [
        {
            "id": "c",
            "call_id": "c",
            "type": "function_call",
            "name": "litellm_web_search",
            "input": {"query": "q"},
            "arguments": {"query": "q"},
        }
    ]
    import json as _json

    seen_fingerprint = _json.dumps(tool_calls, sort_keys=True, default=str)

    with (
        patch.object(litellm, "asearch", new=AsyncMock(return_value=fake_search)),
        patch.object(litellm, "aresponses", new=AsyncMock()),
    ):
        with pytest.raises(ValueError, match="repeated tool-call fingerprint"):
            await logger.async_run_responses_api_agentic_loop(
                tools={
                    "tool_calls": tool_calls,
                    "response_format": "responses",
                },
                model="openai.gpt-5.5",
                input="hi",
                response={"id": "r", "output": []},
                response_api_optional_request_params={},
                litellm_params={},
                logging_obj=MagicMock(),
                stream=False,
                kwargs={
                    "_agentic_loop_depth": 0,
                    "max_agentic_loops": 5,
                    "_agentic_loop_fingerprints": [seen_fingerprint],
                },
            )
