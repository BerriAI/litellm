"""
Unit tests for WebSearchInterceptionLogger on the OpenAI Responses API path.

The Responses surface rides the shared agentic-loop mechanism
(``async_should_run_agentic_loop`` / ``async_build_agentic_loop_plan`` ->
``_execute_responses_agentic_plan``), gated by the ``_agentic_loop_api_surface``
== ``responses`` marker, the same way ``code_interpreter_interception`` does.
These tests cover tool detection, the deployment-hook tool conversion (sync and
async call types), surface routing, plan construction, search-tool auth
forwarding, and end-to-end dispatch through the shared mechanism.
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
from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
from litellm.types.integrations.custom_logger import RESPONSES_AGENTIC_SURFACE
from litellm.types.utils import LlmProviders


@pytest.fixture
def logger() -> WebSearchInterceptionLogger:
    return WebSearchInterceptionLogger(
        enabled_providers=[LlmProviders.BEDROCK_MANTLE, LlmProviders.OPENAI],
        search_tool_name="tavily-search",
    )


def _responses_with_search_call(query: str = "btc price") -> dict:
    return {
        "id": "resp_initial",
        "output": [
            {"type": "reasoning", "id": "rs_1", "summary": []},
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "litellm_web_search",
                "arguments": json.dumps({"query": query}),
            },
        ],
    }


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
    assert tool["type"] == "function"
    assert tool["name"] == "litellm_web_search"
    assert "parameters" in tool
    assert "function" not in tool


def test_responses_api_response_parser_finds_function_call() -> None:
    should, calls = WebSearchTransformation.transform_request(
        response=_responses_with_search_call("weather hong kong"),
        stream=False,
        response_format="responses",
    )
    assert should is True
    assert len(calls) == 1
    assert calls[0]["call_id"] == "call_1"
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
@pytest.mark.parametrize("call_type", ["aresponses", "responses"])
async def test_pre_call_hook_responses_uses_flat_shape(
    logger: WebSearchInterceptionLogger, call_type: str
) -> None:
    # Both the async (``aresponses``) and sync (``responses``) Responses call
    # types must convert the server-hosted tool to the flat function shape.
    kwargs = {
        "model": "openai.gpt-5.5",
        "custom_llm_provider": "bedrock_mantle",
        "tools": [{"type": "web_search"}],
        "stream": True,
    }
    out = await logger.async_pre_call_deployment_hook(kwargs, call_type=call_type)
    assert out is not None
    converted = out["tools"][0]
    assert converted["type"] == "function"
    assert converted["name"] == "litellm_web_search"
    assert "function" not in converted
    assert out["stream"] is False
    assert out["_websearch_interception_converted_stream"] is True


@pytest.mark.asyncio
async def test_pre_call_hook_responses_skips_disabled_provider(
    logger: WebSearchInterceptionLogger,
) -> None:
    kwargs = {
        "model": "publishers/google/models/gemini-1.5-pro",
        "custom_llm_provider": "vertex_ai",
        "tools": [{"type": "web_search"}],
    }
    out = await logger.async_pre_call_deployment_hook(kwargs, call_type="aresponses")
    assert out is None


@pytest.mark.asyncio
async def test_should_run_agentic_loop_routes_responses_surface(
    logger: WebSearchInterceptionLogger,
) -> None:
    # With the responses surface marker set, the shared hook must dispatch to
    # the Responses detection (which reads ``output[].function_call``), not the
    # Anthropic detection (which reads ``content[].tool_use``).
    should, hook_tools = await logger.async_should_run_agentic_loop(
        response=_responses_with_search_call(),
        model="openai.gpt-5.5",
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"type": "web_search"}],
        stream=False,
        custom_llm_provider="bedrock_mantle",
        kwargs={"_agentic_loop_api_surface": RESPONSES_AGENTIC_SURFACE},
    )
    assert should is True
    assert len(hook_tools["tool_calls"]) == 1


@pytest.mark.asyncio
async def test_build_plan_responses_preserves_assistant_turn_then_appends_outputs(
    logger: WebSearchInterceptionLogger,
) -> None:
    # The plan's follow-up ``input`` must carry the user message, then the
    # entire first-turn assistant ``output`` (reasoning + function_call), then
    # the paired ``function_call_output`` items. Strict providers 400 if a
    # ``function_call_output`` is not preceded by its matching ``function_call``.
    fake_search = MagicMock()
    fake_search.results = [MagicMock(title="t", url="https://example.com", snippet="s")]

    with patch.object(litellm, "asearch", new=AsyncMock(return_value=fake_search)):
        plan = await logger.async_build_agentic_loop_plan(
            tools={
                "tool_calls": [
                    {"call_id": "call_1", "input": {"query": "btc price"}},
                ],
                "response_format": "responses",
            },
            model="openai.gpt-5.5",
            messages=[{"role": "user", "content": "what is btc?"}],
            response=_responses_with_search_call(),
            anthropic_messages_provider_config=MagicMock(),
            anthropic_messages_optional_request_params={"tools": []},
            logging_obj=MagicMock(),
            stream=False,
            kwargs={
                "_agentic_loop_api_surface": RESPONSES_AGENTIC_SURFACE,
                "custom_llm_provider": "bedrock_mantle",
            },
        )

    assert plan.run_agentic_loop is True
    # The plan carries the bare backend id; the shared executor
    # (_execute_responses_agentic_plan) re-prefixes provider/model for the
    # follow-up call — covered by test_generic_dispatch_runs_websearch_on_responses_surface.
    assert plan.request_patch.model == "openai.gpt-5.5"

    items = plan.request_patch.messages
    assert items[0] == {"role": "user", "content": "what is btc?"}
    types = [it.get("type") for it in items if isinstance(it, dict) and "type" in it]
    assert "reasoning" in types
    assert "function_call" in types
    outputs = [it for it in items if it.get("type") == "function_call_output"]
    assert len(outputs) == 1
    assert outputs[0]["call_id"] == "call_1"


@pytest.mark.asyncio
async def test_build_plan_responses_forwards_auth_to_search(
    logger: WebSearchInterceptionLogger,
) -> None:
    # Regression guard: the search must be executed WITH the caller's kwargs so
    # per-key/per-team search-tool authorization is enforced. Dropping kwargs
    # (as an earlier revision did) silently skips the ACL check.
    captured: dict = {}

    async def fake_execute_search(query, kwargs=None):
        captured["kwargs"] = kwargs
        return ("result text", None)

    hook_kwargs = {
        "_agentic_loop_api_surface": RESPONSES_AGENTIC_SURFACE,
        "custom_llm_provider": "bedrock_mantle",
        "litellm_metadata": {"user_api_key_auth": {"api_key": "sk-sentinel"}},
    }
    with patch.object(logger, "_execute_search", side_effect=fake_execute_search):
        await logger.async_build_agentic_loop_plan(
            tools={
                "tool_calls": [{"call_id": "call_1", "input": {"query": "q"}}],
                "response_format": "responses",
            },
            model="openai.gpt-5.5",
            messages=[{"role": "user", "content": "hi"}],
            response=_responses_with_search_call(),
            anthropic_messages_provider_config=MagicMock(),
            anthropic_messages_optional_request_params={"tools": []},
            logging_obj=MagicMock(),
            stream=False,
            kwargs=hook_kwargs,
        )

    assert captured["kwargs"] is not None
    assert (
        captured["kwargs"].get("litellm_metadata", {}).get("user_api_key_auth")
        == {"api_key": "sk-sentinel"}
    )


@pytest.mark.asyncio
async def test_generic_dispatch_runs_websearch_on_responses_surface() -> None:
    # End-to-end: the shared ``_call_agentic_completion_hooks`` with
    # api_surface="responses" must route to the websearch plan and re-run the
    # model via ``litellm.aresponses`` with the patched input chain — proving
    # the surface marker + plan mechanism replaces the old bespoke dispatcher.
    logger = WebSearchInterceptionLogger(
        enabled_providers=[LlmProviders.BEDROCK_MANTLE],
        search_tool_name="tavily-search",
    )
    handler = BaseLLMHTTPHandler()

    captured: dict = {}

    async def fake_aresponses(**kwargs):
        captured.update(kwargs)
        return {"id": "resp_final", "output": [{"type": "message"}]}

    async def fake_execute_search(query, kwargs=None):
        return (f"results for {query}", None)

    logging_obj = MagicMock()
    logging_obj.dynamic_success_callbacks = []
    logging_obj.litellm_call_id = "call-xyz"

    original_callbacks = litellm.callbacks
    litellm.callbacks = [logger]
    try:
        with (
            patch.object(logger, "_execute_search", side_effect=fake_execute_search),
            patch.object(
                litellm, "aresponses", new=AsyncMock(side_effect=fake_aresponses)
            ),
        ):
            result = await handler._call_agentic_completion_hooks(
                response=_responses_with_search_call(),
                model="openai.gpt-5.5",
                messages=[{"role": "user", "content": "what is btc?"}],
                anthropic_messages_provider_config=MagicMock(),
                anthropic_messages_optional_request_params={
                    "tools": [{"type": "function", "name": "litellm_web_search"}]
                },
                logging_obj=logging_obj,
                stream=False,
                custom_llm_provider="bedrock_mantle",
                kwargs={"custom_llm_provider": "bedrock_mantle"},
                api_surface="responses",
            )
    finally:
        litellm.callbacks = original_callbacks

    # The follow-up model call ran, carrying the rebuilt input chain.
    assert captured, "expected a follow-up litellm.aresponses call"
    assert captured["model"] == "bedrock_mantle/openai.gpt-5.5"
    followup_outputs = [
        it
        for it in captured["input"]
        if isinstance(it, dict) and it.get("type") == "function_call_output"
    ]
    assert len(followup_outputs) == 1
    assert result == {"id": "resp_final", "output": [{"type": "message"}]}


@pytest.mark.asyncio
async def test_build_plan_responses_coerces_search_failure(
    logger: WebSearchInterceptionLogger,
) -> None:
    # A failed search must not abort the loop: it is coerced to text and the
    # function_call_output still pairs with its call_id.
    async def boom(query, kwargs=None):
        raise RuntimeError("tavily down")

    with patch.object(logger, "_execute_search", side_effect=boom):
        plan = await logger.async_build_agentic_loop_plan(
            tools={"tool_calls": [{"call_id": "call_1", "input": {"query": "q"}}]},
            model="openai.gpt-5.5",
            messages=[{"role": "user", "content": "hi"}],
            response=_responses_with_search_call(),
            anthropic_messages_provider_config=MagicMock(),
            anthropic_messages_optional_request_params={"tools": []},
            logging_obj=MagicMock(),
            stream=False,
            kwargs={"_agentic_loop_api_surface": RESPONSES_AGENTIC_SURFACE},
        )

    outputs = [
        it for it in plan.request_patch.messages if it.get("type") == "function_call_output"
    ]
    assert len(outputs) == 1
    assert outputs[0]["call_id"] == "call_1"
    assert "Search failed" in outputs[0]["output"]


@pytest.mark.asyncio
async def test_build_plan_responses_missing_query_still_pairs_output(
    logger: WebSearchInterceptionLogger,
) -> None:
    # A tool_call with no query falls to the empty-result branch but still
    # emits a paired function_call_output so strict providers accept the input.
    with patch.object(
        logger, "_execute_search", new=AsyncMock(side_effect=AssertionError("should not run"))
    ):
        plan = await logger.async_build_agentic_loop_plan(
            tools={"tool_calls": [{"call_id": "call_1", "input": {}}]},
            model="openai.gpt-5.5",
            messages=[{"role": "user", "content": "hi"}],
            response=_responses_with_search_call(),
            anthropic_messages_provider_config=MagicMock(),
            anthropic_messages_optional_request_params={"tools": []},
            logging_obj=MagicMock(),
            stream=False,
            kwargs={"_agentic_loop_api_surface": RESPONSES_AGENTIC_SURFACE},
        )

    outputs = [
        it for it in plan.request_patch.messages if it.get("type") == "function_call_output"
    ]
    assert len(outputs) == 1
    assert outputs[0]["call_id"] == "call_1"


def test_dump_output_item_variants() -> None:
    dump = WebSearchInterceptionLogger._dump_output_item

    # dict passes through unchanged
    assert dump({"type": "message", "x": 1}) == {"type": "message", "x": 1}

    # object exposing model_dump()
    class WithModelDump:
        def model_dump(self):
            return {"type": "reasoning", "id": "rs_1"}

    assert dump(WithModelDump()) == {"type": "reasoning", "id": "rs_1"}

    # plain object -> __dict__ fallback
    class Plain:
        def __init__(self):
            self.type = "function_call"
            self.call_id = "c1"

    assert dump(Plain()) == {"type": "function_call", "call_id": "c1"}
