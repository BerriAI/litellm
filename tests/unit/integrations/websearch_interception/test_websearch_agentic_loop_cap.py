"""
Unit tests for what an intercepted request returns once a safety rail refuses
another agentic loop.

The web search interception loop injects an internal tool (litellm_web_search)
that the client never declared. When the loop cap or the repeated-fingerprint
guard trips, the turn has to end with a terminal response: leaking that internal
tool_use block leaves the client holding a tool call it cannot answer.

Also covers the max_agentic_loops knob on websearch_interception_params, from
config.yaml through to the settings the loop actually reads.
"""

import asyncio
import json
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import suppress
from datetime import datetime, timezone
from types import MappingProxyType
from typing import ClassVar, Final, TypedDict
from unittest.mock import MagicMock, patch

import httpx
import pytest
import respx
from pydantic import TypeAdapter
from typing_extensions import ReadOnly

import litellm
from litellm.exceptions import AuthenticationError, RateLimitError
from litellm.integrations.custom_logger import CustomLogger
from litellm.integrations.websearch_interception.handler import (
    WEBSEARCH_EMIT_NATIVE_BLOCKS_KEY,
    WebSearchInterceptionLogger,
)
from litellm.integrations.websearch_interception.tools import (
    get_litellm_web_search_tool,
    get_litellm_web_search_tool_openai,
)
from litellm.litellm_core_utils.agentic_followup_failure import is_agentic_followup_failure
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from litellm.litellm_core_utils.model_cost_table import changed_in_place
from litellm.litellm_core_utils.agentic_loop_settings import DEFAULT_MAX_AGENTIC_LOOPS
from litellm.llms.anthropic.messages.native.fake_stream_iterator import (
    FakeAnthropicMessagesStreamIterator,
)
from litellm.llms.base_llm.anthropic_messages.transformation import BaseAnthropicMessagesConfig
from litellm.llms.base_llm.search.transformation import SearchResponse, SearchResult
from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
from litellm.proxy.spend_tracking.cost_finalization import prepare_response_cost
from litellm.secret_managers.main import get_secret
from litellm.types.llms.openai import ResponsesAPIResponse
from litellm.types.utils import CallTypes
from litellm.types.integrations.custom_logger import (
    AgenticLoopPlan,
    AgenticLoopRequestPatch,
    AgenticLoopSafetyError,
)
from litellm.types.utils import ChatCompletionMessageToolCall, Choices, Function, Message, ModelResponse

from .conftest import (
    FINAL_ANSWER,
    OPENAI_CHAT_URL,
    SEARCH_LIMIT_ERROR_CODE,
    RecordingUpstream,
    offered_tool_names,
)

INTERNAL_TOOL_NAME = "litellm_web_search"
CODE_EXECUTION_TOOL_NAME = "litellm_code_execution"


@pytest.fixture(autouse=True)
def only_the_callbacks_these_tests_register(monkeypatch):
    """
    These tests drive the hooks with a callback of their own on the logging
    object, so a logger another test left on litellm.callbacks would join the
    run and change what the hooks do.
    """
    monkeypatch.setattr(litellm, "callbacks", [])


def _internal_tool_use_block(block_id: str = "toolu_internal_1") -> dict:
    return {
        "id": block_id,
        "type": "tool_use",
        "name": INTERNAL_TOOL_NAME,
        "input": {"query": "who won the world cup"},
    }


def _native_search_blocks(index: int = 1) -> list[dict]:
    return [
        {
            "type": "server_tool_use",
            "id": f"srvtoolu_{index}",
            "name": "web_search",
            "input": {"query": "who won the world cup"},
        },
        {
            "type": "web_search_tool_result",
            "tool_use_id": f"srvtoolu_{index}",
            "content": [{"type": "web_search_result", "url": "https://example.com", "title": "Result"}],
        },
    ]


def _response_asking_for_another_search(block_id: str = "toolu_internal_1") -> dict:
    return {
        "id": "msg_123",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-5",
        "content": [
            *_native_search_blocks(index=1),
            {"type": "text", "text": "Let me check one more source."},
            _internal_tool_use_block(block_id),
        ],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }


def _block_types(response: dict) -> list[str]:
    return [block["type"] for block in response["content"]]


def _tool_use_names(response: dict) -> list[str]:
    return [block.get("name") for block in response["content"] if block.get("type") == "tool_use"]


def _responses_turn(*function_calls: tuple[str, str]) -> ResponsesAPIResponse:
    return ResponsesAPIResponse(
        id="resp_1",
        created_at=0,
        status="completed",
        model="claude-sonnet-4-5",
        object="response",
        output=[
            {
                "type": "message",
                "id": "msg_1",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Let me check one more source.", "annotations": []}],
            },
            *(
                {"type": "function_call", "id": f"fc_{call_id}", "call_id": call_id, "name": name, "arguments": "{}"}
                for call_id, name in function_calls
            ),
        ],
    )


class _InterceptingCallback(CustomLogger):
    """
    Stands in for the websearch interceptor: asks for another loop whenever the
    response carries an internal web search tool_use block, and injects the
    native block pair on the way back out.
    """

    def __init__(self):
        self.plan_calls = 0
        self.post_hook_calls = 0

    async def async_should_run_agentic_loop(
        self, response, model, messages, tools, stream, custom_llm_provider, kwargs
    ):
        if not isinstance(response, dict):
            return True, {"tool_calls": [_internal_tool_use_block()]}
        tool_calls = [
            block
            for block in response.get("content", [])
            if block.get("type") == "tool_use" and block.get("name") == INTERNAL_TOOL_NAME
        ]
        if not tool_calls:
            return False, {}
        return True, {"tool_calls": tool_calls, "tool_type": "websearch"}

    async def async_build_agentic_loop_plan(
        self,
        tools,
        model,
        messages,
        response,
        anthropic_messages_provider_config,
        anthropic_messages_optional_request_params,
        logging_obj,
        stream,
        kwargs,
    ):
        self.plan_calls += 1
        return AgenticLoopPlan(
            run_agentic_loop=True,
            request_patch=AgenticLoopRequestPatch(
                messages=[{"role": "user", "content": "here are the search results"}],
                max_tokens=1024,
            ),
        )

    async def async_post_agentic_loop_response_hook(self, response, plan, kwargs):
        self.post_hook_calls += 1
        if isinstance(response, dict):
            response["content"] = [*_native_search_blocks(index=2), *response.get("content", [])]
        return response


class _ResponsesInterceptingCallback(_InterceptingCallback):
    async def async_should_run_agentic_loop(
        self, response, model, messages, tools, stream, custom_llm_provider, kwargs
    ):
        calls = [
            {"id": item.call_id, "call_id": item.call_id, "name": item.name}
            for item in response.output
            if item.type == "function_call" and item.name == INTERNAL_TOOL_NAME
        ]
        return bool(calls), {"tool_calls": calls}


def _logging_obj(*callbacks: CustomLogger, converted_stream: bool = False) -> MagicMock:
    logging_obj = MagicMock()
    logging_obj.model_call_details = {"websearch_interception_converted_stream": converted_stream}
    logging_obj.dynamic_success_callbacks = list(callbacks)
    logging_obj.litellm_call_id = "call-abc"
    return logging_obj


class _CodeExecutionGate(CustomLogger):
    async def async_should_run_agentic_loop(
        self, response, model, messages, tools, stream, custom_llm_provider, kwargs
    ):
        calls = (
            [
                {"id": block["id"], "name": block["name"]}
                for block in response["content"]
                if block.get("type") == "tool_use" and block.get("name") == CODE_EXECUTION_TOOL_NAME
            ]
            if isinstance(response, dict)
            else [
                {"call_id": item.call_id, "name": item.name}
                for item in response.output
                if item.type == "function_call" and item.name == CODE_EXECUTION_TOOL_NAME
            ]
        )
        return bool(calls), {"tool_calls": calls}


async def _run_hooks(
    handler: BaseLLMHTTPHandler,
    callback: CustomLogger,
    kwargs: dict,
    response: object = None,
    stream: bool = False,
    converted_stream: bool = False,
    api_surface: str = "anthropic_messages",
):
    return await handler._call_agentic_completion_hooks(
        response=_response_asking_for_another_search() if response is None else response,
        model="claude-sonnet-4-5",
        messages=[{"role": "user", "content": "who won the world cup"}],
        anthropic_messages_provider_config=MagicMock(),
        anthropic_messages_optional_request_params={},
        logging_obj=_logging_obj(callback, converted_stream=converted_stream),
        stream=stream,
        custom_llm_provider="anthropic",
        kwargs=kwargs,
        api_surface=api_surface,
    )


class TestCappedLoopReturnsTerminalResponse:
    def setup_method(self):
        self.handler = BaseLLMHTTPHandler()
        self.callback = _InterceptingCallback()

    @pytest.mark.asyncio
    async def test_internal_tool_use_block_is_dropped(self):
        result = await _run_hooks(
            self.handler,
            self.callback,
            kwargs={"_agentic_loop_depth": 3, "max_agentic_loops": 3},
        )

        assert isinstance(result, dict)
        assert INTERNAL_TOOL_NAME not in _tool_use_names(result)

    @pytest.mark.asyncio
    async def test_stop_reason_is_closed_out(self):
        result = await _run_hooks(
            self.handler,
            self.callback,
            kwargs={"_agentic_loop_depth": 3, "max_agentic_loops": 3},
        )

        assert result["stop_reason"] == "end_turn"

    @pytest.mark.asyncio
    async def test_native_blocks_and_text_survive(self):
        result = await _run_hooks(
            self.handler,
            self.callback,
            kwargs={"_agentic_loop_depth": 3, "max_agentic_loops": 3},
        )

        assert _block_types(result) == ["server_tool_use", "web_search_tool_result", "text"]

    @pytest.mark.asyncio
    async def test_turn_carrying_only_the_refused_call_still_ends_cleanly(self):
        """
        The refused call can be every block the model produced, which leaves the
        turn with no content once it is dropped. That still has to come back as a
        finished turn rather than as the leaked call, so the client stops instead
        of waiting on a tool it cannot run, and the rest of the message survives
        so the request is still billed and traceable.

        An empty turn renders as nothing, which is the ceiling being set too low
        for the question rather than a malformed response.
        """
        nothing_but_the_refused_call = {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-5",
            "content": [_internal_tool_use_block()],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }

        result = await _run_hooks(
            self.handler,
            self.callback,
            kwargs={"_agentic_loop_depth": 3, "max_agentic_loops": 3},
            response=nothing_but_the_refused_call,
        )

        assert result["content"] == []
        assert result["stop_reason"] == "end_turn"
        assert result["usage"] == {"input_tokens": 10, "output_tokens": 5}
        assert result["id"] == "msg_123"

    @pytest.mark.asyncio
    async def test_no_follow_up_model_call_is_planned(self):
        """
        The rail has to end the turn without planning another model call, and it
        has to end it by returning rather than by raising, which is the half that
        the caller's response depends on.
        """
        result = await _run_hooks(
            self.handler,
            self.callback,
            kwargs={"_agentic_loop_depth": 3, "max_agentic_loops": 3},
        )

        assert self.callback.plan_calls == 0
        assert result["stop_reason"] == "end_turn"

    @pytest.mark.asyncio
    async def test_original_response_is_not_mutated(self):
        response = _response_asking_for_another_search()

        await _run_hooks(
            self.handler,
            self.callback,
            kwargs={"_agentic_loop_depth": 3, "max_agentic_loops": 3},
            response=response,
        )

        assert response["stop_reason"] == "tool_use"
        assert INTERNAL_TOOL_NAME in _tool_use_names(response)

    @pytest.mark.asyncio
    async def test_repeated_fingerprint_guard_is_terminal_too(self):
        tool_calls = {"tool_calls": [_internal_tool_use_block()], "tool_type": "websearch"}
        seen = json.dumps(tool_calls, sort_keys=True, default=str)

        result = await _run_hooks(
            self.handler,
            self.callback,
            kwargs={"_agentic_loop_depth": 0, "max_agentic_loops": 3, "_agentic_loop_fingerprints": [seen]},
        )

        assert self.callback.plan_calls == 0
        assert INTERNAL_TOOL_NAME not in _tool_use_names(result)
        assert result["stop_reason"] == "end_turn"

    @pytest.mark.asyncio
    async def test_client_declared_tool_use_is_left_alone(self):
        response = _response_asking_for_another_search()
        client_tool_use = {"id": "toolu_client_1", "type": "tool_use", "name": "get_weather", "input": {}}
        response["content"].append(client_tool_use)

        result = await _run_hooks(
            self.handler,
            self.callback,
            kwargs={"_agentic_loop_depth": 3, "max_agentic_loops": 3},
            response=response,
        )

        assert _tool_use_names(result) == ["get_weather"]
        assert result["stop_reason"] == "tool_use"

    @pytest.mark.asyncio
    async def test_every_hooks_internal_call_is_dropped_and_the_clients_survives(self):
        response = _response_asking_for_another_search()
        response["content"] += [
            {"id": "toolu_code_1", "type": "tool_use", "name": CODE_EXECUTION_TOOL_NAME, "input": {}},
            {"id": "toolu_client_1", "type": "tool_use", "name": "get_weather", "input": {}},
        ]

        result = await self.handler._call_agentic_completion_hooks(
            response=response,
            model="claude-sonnet-4-5",
            messages=[{"role": "user", "content": "who won the world cup"}],
            anthropic_messages_provider_config=MagicMock(),
            anthropic_messages_optional_request_params={},
            logging_obj=_logging_obj(self.callback, _CodeExecutionGate()),
            stream=False,
            custom_llm_provider="anthropic",
            kwargs={"_agentic_loop_depth": 3, "max_agentic_loops": 3},
        )

        assert _tool_use_names(result) == ["get_weather"]
        assert self.callback.plan_calls == 0

    @pytest.mark.asyncio
    async def test_responses_surface_drops_every_hooks_internal_call_and_keeps_the_clients(self):
        result = await self.handler._call_agentic_completion_hooks(
            response=_responses_turn(
                ("call_search", INTERNAL_TOOL_NAME), ("call_code", CODE_EXECUTION_TOOL_NAME), ("call_client", "lookup")
            ),
            model="claude-sonnet-4-5",
            messages=[{"role": "user", "content": "who won the world cup"}],
            anthropic_messages_provider_config=MagicMock(),
            anthropic_messages_optional_request_params={},
            logging_obj=_logging_obj(_ResponsesInterceptingCallback(), _CodeExecutionGate()),
            stream=False,
            custom_llm_provider="anthropic",
            kwargs={"_agentic_loop_depth": 3, "max_agentic_loops": 3},
            api_surface="responses",
        )

        assert [getattr(item, "call_id", None) for item in result.output] == [None, "call_client"]

    def test_only_the_refused_tool_calls_are_dropped(self):
        """
        A block is matched on the id the rail refused, not on the tool name, so a
        second block sharing that name survives when the rail never listed it. A
        callback that picks its tool calls out by name hands both over and both
        go, which is its own call to make; this is about not widening it here.
        """
        response = _response_asking_for_another_search()
        response["content"].append(
            {"id": "toolu_client_1", "type": "tool_use", "name": INTERNAL_TOOL_NAME, "input": {}}
        )

        result = BaseLLMHTTPHandler._finalize_refused_agentic_response(
            response=response,
            tool_calls={"tool_calls": [_internal_tool_use_block()]},
        )

        assert [block["id"] for block in result["content"] if block.get("type") == "tool_use"] == ["toolu_client_1"]
        assert result["stop_reason"] == "tool_use"

    def test_tool_calls_without_ids_still_match_by_name(self):
        """
        Not every callback shape carries ids on its tool calls, so the name is
        still what decides when the rail refused a call that has no id.
        """
        result = BaseLLMHTTPHandler._finalize_refused_agentic_response(
            response=_response_asking_for_another_search(),
            tool_calls={"tool_calls": [{"name": INTERNAL_TOOL_NAME, "input": {}}]},
        )

        assert _tool_use_names(result) == []
        assert result["stop_reason"] == "end_turn"

    def test_a_name_only_refusal_keeps_the_native_search_blocks_sharing_that_name(self):
        response = _response_asking_for_another_search()
        response["content"][-1] = {"type": "tool_use", "name": "web_search", "input": {}}

        result = BaseLLMHTTPHandler._finalize_refused_agentic_response(
            response=response,
            tool_calls={"tool_calls": [{"name": "web_search", "input": {}}]},
        )

        assert _block_types(result) == ["server_tool_use", "web_search_tool_result", "text"]
        assert result["stop_reason"] == "end_turn"

    @pytest.mark.asyncio
    async def test_streaming_caller_is_left_to_its_existing_behavior(self):
        """
        A streaming caller has already sent the original message to the client, so
        a finalized turn would land as a second message rather than replace the
        first. The rail keeps raising there and the caller handles it as before.
        """
        with pytest.raises(AgenticLoopSafetyError):
            await _run_hooks(
                self.handler,
                self.callback,
                kwargs={"_agentic_loop_depth": 3, "max_agentic_loops": 3},
                stream=True,
            )

        assert self.callback.plan_calls == 0

    @pytest.mark.asyncio
    async def test_responses_surface_drops_the_refused_function_call_and_keeps_the_answer(self):
        response = _responses_turn(("call_search", INTERNAL_TOOL_NAME), ("call_client", "lookup"))
        callback = _ResponsesInterceptingCallback()

        result = await _run_hooks(
            self.handler,
            callback,
            kwargs={"_agentic_loop_depth": 3, "max_agentic_loops": 3},
            response=response,
            api_surface="responses",
        )

        assert isinstance(result, ResponsesAPIResponse)
        assert [(item.type, getattr(item, "name", None)) for item in result.output] == [
            ("message", None),
            ("function_call", "lookup"),
        ]
        assert callback.plan_calls == 0

    def test_responses_refusal_by_call_id_alone_keeps_the_calls_it_did_not_list(self):
        response = _responses_turn(("call_search", INTERNAL_TOOL_NAME), ("call_unlisted", INTERNAL_TOOL_NAME))

        result = BaseLLMHTTPHandler._finalize_refused_agentic_response(
            response=response,
            tool_calls={"tool_calls": [{"call_id": "call_search", "name": INTERNAL_TOOL_NAME}]},
        )

        assert [getattr(item, "call_id", None) for item in result.output] == [None, "call_unlisted"]

    @pytest.mark.asyncio
    async def test_non_dict_response_is_returned_untouched(self):
        response = MagicMock()

        result = await _run_hooks(
            self.handler,
            self.callback,
            kwargs={"_agentic_loop_depth": 3, "max_agentic_loops": 3},
            response=response,
        )

        assert result is response

    @pytest.mark.asyncio
    async def test_converted_stream_gets_a_terminal_fake_stream(self):
        """
        A converted stream is wrapped back into an Anthropic SSE stream here, the
        same as every other return in this function, so a streaming client gets a
        terminal stream rather than a bare dict. The interceptor turns the client's
        stream into a non-streaming upstream call, so stream is False on this path
        and the converted flag on the logging object is what marks it.
        """
        result = await _run_hooks(
            self.handler,
            self.callback,
            kwargs={"_agentic_loop_depth": 3, "max_agentic_loops": 3},
            converted_stream=True,
        )

        assert isinstance(result, FakeAnthropicMessagesStreamIterator)
        assert result.response["stop_reason"] == "end_turn"
        assert INTERNAL_TOOL_NAME not in _tool_use_names(result.response)

    def test_rails_cannot_trip_in_the_outermost_frame(self):
        """
        Backs the invariant the test above relies on: at depth 0 the fingerprint set
        is empty and the ceiling is at least 1, so neither rail can refuse.
        """
        depth, max_loops, fingerprints = BaseLLMHTTPHandler._get_agentic_loop_settings(kwargs={})

        assert depth == 0
        assert fingerprints == []
        assert max_loops >= 1

        depth, max_loops, fingerprints = BaseLLMHTTPHandler._get_agentic_loop_settings(
            kwargs={"max_agentic_loops": 1}
        )

        assert max_loops == 1
        assert BaseLLMHTTPHandler._check_agentic_loop_safety(
            tool_calls={"tool_calls": [_internal_tool_use_block()]},
            fingerprints=fingerprints,
            depth=depth,
            max_loops=max_loops,
            model="claude-sonnet-4-5",
        )

    def test_safety_error_is_still_a_value_error(self):
        assert issubclass(AgenticLoopSafetyError, ValueError)

    def test_safety_error_type_names_the_rail(self):
        with pytest.raises(AgenticLoopSafetyError, match="max_agentic_loops"):
            BaseLLMHTTPHandler._check_agentic_loop_safety(
                tool_calls={"tool_calls": [_internal_tool_use_block()]},
                fingerprints=[],
                depth=3,
                max_loops=3,
                model="claude-sonnet-4-5",
            )


class TestOuterFramePostHookStillRuns:
    """
    The cap used to raise through the parent frame's await, which skipped the
    parent's post-loop hook. The parent now gets its terminal response back and
    finishes normally, so the blocks it was going to inject still land.
    """

    @pytest.mark.asyncio
    async def test_parent_frame_injects_its_blocks_after_the_cap_trips(self, monkeypatch):
        handler = BaseLLMHTTPHandler()
        callback = _InterceptingCallback()

        async def fake_acreate(**call_kwargs):
            return await handler._call_agentic_completion_hooks(
                response=_response_asking_for_another_search(block_id="toolu_internal_2"),
                model=call_kwargs["model"],
                messages=call_kwargs["messages"],
                anthropic_messages_provider_config=MagicMock(),
                anthropic_messages_optional_request_params={},
                logging_obj=_logging_obj(callback),
                stream=False,
                custom_llm_provider="anthropic",
                kwargs={
                    key: call_kwargs[key]
                    for key in ("_agentic_loop_depth", "max_agentic_loops", "_agentic_loop_fingerprints")
                    if key in call_kwargs
                },
            )

        monkeypatch.setattr("litellm.anthropic_interface.messages.acreate", fake_acreate)

        result = await _run_hooks(
            handler,
            callback,
            kwargs={"_agentic_loop_depth": 0, "max_agentic_loops": 1},
        )

        assert callback.plan_calls == 1
        assert callback.post_hook_calls == 1
        assert _block_types(result)[:2] == ["server_tool_use", "web_search_tool_result"]
        assert INTERNAL_TOOL_NAME not in _tool_use_names(result)
        assert result["stop_reason"] == "end_turn"


def _response_asking_for_searches(*queries: str) -> dict:
    return {
        "id": "msg_123",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-5",
        "content": [
            {"id": f"toolu_internal_{index}", "type": "tool_use", "name": INTERNAL_TOOL_NAME, "input": {"query": query}}
            for index, query in enumerate(queries, start=1)
        ],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }


def _request_logging_obj(call_type: str, *callbacks: CustomLogger) -> Logging:
    logging_obj = Logging(
        model="claude-sonnet-4-5",
        messages=[{"role": "user", "content": "who won the world cup"}],
        stream=False,
        call_type=call_type,
        start_time=time.time(),
        litellm_call_id="call-abc",
        function_id="f",
        dynamic_success_callbacks=list(callbacks),
    )
    logging_obj.update_environment_variables(
        model="claude-sonnet-4-5", user="", optional_params={}, litellm_params={}, custom_llm_provider="anthropic"
    )
    return logging_obj


def _assert_first_call_billed(logging_obj: Logging, *, input_tokens: int, output_tokens: int) -> None:
    price = litellm.model_cost["claude-sonnet-4-5"]
    assert logging_obj.model_call_details["combined_usage_object"].total_tokens == input_tokens + output_tokens
    assert logging_obj.model_call_details["response_cost"] == pytest.approx(
        input_tokens * price["input_cost_per_token"] + output_tokens * price["output_cost_per_token"]
    )


class TestFailedFollowUpCallReachesTheClient:
    def setup_method(self):
        self.handler = BaseLLMHTTPHandler()
        self.logger = WebSearchInterceptionLogger(enabled_providers=["anthropic"])

    @staticmethod
    async def _rate_limited_follow_up(**call_kwargs):
        raise RateLimitError(
            "UserByModelByMinuteUncachedInputTokens", llm_provider="anthropic", model="claude-sonnet-4-5"
        )

    @staticmethod
    async def _search(query, kwargs=None, rich=None):
        found = SearchResult(title="Result", url="https://example.com", snippet="A result.", date=None)
        return ("Title: Result\nURL: https://example.com", SearchResponse(results=[found]))

    @pytest.mark.asyncio
    @pytest.mark.parametrize("converted_stream", (False, True), ids=("non_streaming", "streaming_client"))
    async def test_follow_up_rate_limit_is_raised_to_the_caller(self, monkeypatch, converted_stream):
        monkeypatch.setattr("litellm.anthropic_interface.messages.acreate", self._rate_limited_follow_up)

        with (
            patch.object(self.logger, "_execute_search", side_effect=self._search),
            pytest.raises(RateLimitError, match="UserByModelByMinuteUncachedInputTokens"),
        ):
            await self.handler._call_agentic_completion_hooks(
                response=_response_asking_for_searches("who won the world cup"),
                model="claude-sonnet-4-5",
                messages=[{"role": "user", "content": "who won the world cup"}],
                anthropic_messages_provider_config=MagicMock(),
                anthropic_messages_optional_request_params={"tools": [get_litellm_web_search_tool()]},
                logging_obj=_logging_obj(self.logger, converted_stream=converted_stream),
                stream=False,
                custom_llm_provider="anthropic",
                kwargs={"_agentic_loop_depth": 0, "max_agentic_loops": 3, WEBSEARCH_EMIT_NATIVE_BLOCKS_KEY: True},
            )

    @pytest.mark.asyncio
    async def test_follow_up_rate_limit_bills_the_first_call_on_the_failure_log(self, monkeypatch):
        monkeypatch.setattr("litellm.anthropic_interface.messages.acreate", self._rate_limited_follow_up)
        logging_obj = _request_logging_obj("anthropic_messages", self.logger)

        with (
            patch.object(self.logger, "_execute_search", side_effect=self._search),
            pytest.raises(RateLimitError),
        ):
            await self.handler._call_agentic_completion_hooks(
                response=_response_asking_for_searches("who won the world cup"),
                model="claude-sonnet-4-5",
                messages=[{"role": "user", "content": "who won the world cup"}],
                anthropic_messages_provider_config=MagicMock(),
                anthropic_messages_optional_request_params={"tools": [get_litellm_web_search_tool()]},
                logging_obj=logging_obj,
                stream=False,
                custom_llm_provider="anthropic",
                kwargs={"_agentic_loop_depth": 0, "max_agentic_loops": 3, WEBSEARCH_EMIT_NATIVE_BLOCKS_KEY: True},
            )

        _assert_first_call_billed(logging_obj, input_tokens=10, output_tokens=5)

    @pytest.mark.asyncio
    async def test_a_crashed_plan_build_is_not_retried_and_bills_the_first_call(self):
        logging_obj = _request_logging_obj("anthropic_messages", self.logger)

        with (
            patch.object(self.logger, "async_build_agentic_loop_plan", side_effect=ValueError("the plan crashed")),
            pytest.raises(ValueError, match="the plan crashed") as raised,
        ):
            await self.handler._call_agentic_completion_hooks(
                response=_response_asking_for_searches("who won the world cup"),
                model="claude-sonnet-4-5",
                messages=[{"role": "user", "content": "who won the world cup"}],
                anthropic_messages_provider_config=MagicMock(),
                anthropic_messages_optional_request_params={"tools": [get_litellm_web_search_tool()]},
                logging_obj=logging_obj,
                stream=False,
                custom_llm_provider="anthropic",
                kwargs={"_agentic_loop_depth": 0, "max_agentic_loops": 3, WEBSEARCH_EMIT_NATIVE_BLOCKS_KEY: True},
            )

        assert is_agentic_followup_failure(raised.value)
        _assert_first_call_billed(logging_obj, input_tokens=10, output_tokens=5)

    @pytest.mark.asyncio
    async def test_responses_follow_up_rate_limit_bills_the_first_call_on_the_failure_log(self, monkeypatch):
        monkeypatch.setattr(litellm, "aresponses", self._rate_limited_follow_up)
        callback = _ResponsesInterceptingCallback()
        logging_obj = _request_logging_obj("aresponses", callback)
        first_call = _responses_turn(("call_search", INTERNAL_TOOL_NAME)).model_copy(
            update={"usage": {"input_tokens": 13, "output_tokens": 6, "total_tokens": 19}}
        )

        with pytest.raises(RateLimitError):
            await self.handler._call_agentic_completion_hooks(
                response=first_call,
                model="claude-sonnet-4-5",
                messages=[{"role": "user", "content": "who won the world cup"}],
                anthropic_messages_provider_config=MagicMock(),
                anthropic_messages_optional_request_params={},
                logging_obj=logging_obj,
                stream=False,
                custom_llm_provider="anthropic",
                kwargs={"_agentic_loop_depth": 0, "max_agentic_loops": 3},
                api_surface="responses",
            )

        _assert_first_call_billed(logging_obj, input_tokens=13, output_tokens=6)

    @pytest.mark.asyncio
    async def test_responses_follow_up_rate_limit_is_raised_to_the_caller(self, monkeypatch):
        monkeypatch.setattr(litellm, "aresponses", self._rate_limited_follow_up)
        callback = _ResponsesInterceptingCallback()

        with pytest.raises(RateLimitError, match="UserByModelByMinuteUncachedInputTokens"):
            await _run_hooks(
                self.handler,
                callback,
                kwargs={"_agentic_loop_depth": 0, "max_agentic_loops": 3},
                response=_responses_turn(("call_search", INTERNAL_TOOL_NAME)),
                api_surface="responses",
            )

        assert callback.plan_calls == 1


class TestFailedSearchEndsTheTurn:
    """
    A search that failed used to come back to the client as an empty successful
    ``web_search_tool_result`` while the model was re-asked the same query until
    the loop cap tripped. When the client sent a native web search tool, the
    turn now ends after the first failed search, with Anthropic's
    ``web_search_tool_result_error`` object in the tool result and no follow-up
    model call. An iteration where some search still succeeded keeps its
    follow-up call.
    """

    def setup_method(self):
        self.handler = BaseLLMHTTPHandler()
        self.logger = WebSearchInterceptionLogger(enabled_providers=["anthropic"])
        self.followup_calls: list[dict] = []

    async def _fake_acreate(self, **call_kwargs):
        self.followup_calls.append(call_kwargs)
        return {
            "id": "msg_followup",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-5",
            "content": [{"type": "text", "text": "final answer"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 20, "output_tokens": 5},
        }

    async def _run(self, response: dict, converted_stream: bool = False):
        return await self.handler._call_agentic_completion_hooks(
            response=response,
            model="claude-sonnet-4-5",
            messages=[{"role": "user", "content": "who won the world cup"}],
            anthropic_messages_provider_config=MagicMock(),
            anthropic_messages_optional_request_params={"tools": [get_litellm_web_search_tool()]},
            logging_obj=_logging_obj(self.logger, converted_stream=converted_stream),
            stream=False,
            custom_llm_provider="anthropic",
            kwargs={"_agentic_loop_depth": 0, "max_agentic_loops": 3, WEBSEARCH_EMIT_NATIVE_BLOCKS_KEY: True},
        )

    @pytest.mark.asyncio
    async def test_all_failed_iteration_ends_the_turn_without_a_follow_up_call(self, monkeypatch):
        monkeypatch.setattr("litellm.anthropic_interface.messages.acreate", self._fake_acreate)

        with patch.object(
            self.logger,
            "_execute_search",
            side_effect=AuthenticationError("401 Unauthorized", llm_provider="tavily", model="tavily"),
        ):
            result = await self._run(_response_asking_for_searches("who won the world cup"))

        assert self.followup_calls == []
        assert result["stop_reason"] == "end_turn"
        assert INTERNAL_TOOL_NAME not in _tool_use_names(result)
        assert _block_types(result) == ["server_tool_use", "web_search_tool_result"]
        server_tool_use, tool_result = result["content"]
        assert server_tool_use["id"].startswith("srvtoolu_")
        assert server_tool_use["input"] == {"query": "who won the world cup"}
        assert tool_result["tool_use_id"] == server_tool_use["id"]
        assert tool_result["content"] == {"type": "web_search_tool_result_error", "error_code": "unavailable"}

    @pytest.mark.asyncio
    async def test_all_failed_iteration_streams_the_error_block(self, monkeypatch):
        monkeypatch.setattr("litellm.anthropic_interface.messages.acreate", self._fake_acreate)

        with patch.object(
            self.logger,
            "_execute_search",
            side_effect=AuthenticationError("401 Unauthorized", llm_provider="tavily", model="tavily"),
        ):
            result = await self._run(_response_asking_for_searches("who won the world cup"), converted_stream=True)

        assert self.followup_calls == []
        assert isinstance(result, FakeAnthropicMessagesStreamIterator)
        events = _stream_events(result.response)
        started = [event["content_block"] for event in events if event["type"] == "content_block_start"]
        assert [block["type"] for block in started] == ["server_tool_use", "web_search_tool_result"]
        assert started[1]["tool_use_id"] == started[0]["id"]
        assert started[1]["content"] == {"type": "web_search_tool_result_error", "error_code": "unavailable"}
        assert [event["delta"]["stop_reason"] for event in events if event["type"] == "message_delta"] == ["end_turn"]

    @pytest.mark.asyncio
    async def test_mixed_iteration_keeps_the_follow_up_call(self, monkeypatch):
        monkeypatch.setattr("litellm.anthropic_interface.messages.acreate", self._fake_acreate)

        async def search(query, kwargs=None, rich=None):
            if query == "fails":
                raise RateLimitError("slow down", llm_provider="tavily", model="tavily")
            found = SearchResult(title="Result", url="https://example.com", snippet="A result.", date=None)
            return ("Title: Result\nURL: https://example.com", SearchResponse(results=[found]))

        with patch.object(self.logger, "_execute_search", side_effect=search):
            result = await self._run(_response_asking_for_searches("fails", "works"))

        assert len(self.followup_calls) == 1
        tool_results = self.followup_calls[0]["messages"][-1]["content"]
        assert [block["type"] for block in tool_results] == ["tool_result", "tool_result"]
        assert tool_results[0]["content"] == "Search failed: litellm.RateLimitError: slow down"
        assert tool_results[1]["content"] == "Title: Result\nURL: https://example.com"
        assert result["stop_reason"] == "end_turn"
        assert _block_types(result) == [
            "server_tool_use",
            "web_search_tool_result",
            "server_tool_use",
            "web_search_tool_result",
            "text",
        ]
        assert result["content"][0]["input"] == {"query": "fails"}
        assert result["content"][1]["content"] == {"type": "web_search_tool_result_error", "error_code": "too_many_requests"}
        assert result["content"][2]["input"] == {"query": "works"}
        assert result["content"][3]["content"][0]["url"] == "https://example.com"


class TestMaxAgenticLoopsConfigKnob:
    def test_from_config_yaml_reads_the_knob(self):
        logger = WebSearchInterceptionLogger.from_config_yaml(
            {"enabled_providers": ["bedrock"], "max_agentic_loops": 7}
        )

        assert logger.max_agentic_loops == 7

    def test_from_config_yaml_leaves_it_unset_by_default(self):
        logger = WebSearchInterceptionLogger.from_config_yaml({"enabled_providers": ["bedrock"]})

        assert logger.max_agentic_loops is None

    @pytest.mark.parametrize("bad_value", [0, -1])
    def test_out_of_range_ceilings_are_rejected_at_config_load(self, bad_value):
        with pytest.raises(ValueError, match="max_agentic_loops"):
            WebSearchInterceptionLogger.from_config_yaml(
                {"enabled_providers": ["bedrock"], "max_agentic_loops": bad_value}
            )

    @pytest.mark.parametrize("bad_value", ["three", True, 2.5])
    def test_non_integer_ceilings_are_rejected_at_config_load(self, bad_value):
        with pytest.raises(TypeError, match="max_agentic_loops"):
            WebSearchInterceptionLogger.from_config_yaml(
                {"enabled_providers": ["bedrock"], "max_agentic_loops": bad_value}
            )

    def test_a_ceiling_spelled_as_a_string_is_read_at_config_load(self):
        """
        `max_agentic_loops: os.environ/MAX_AGENTIC_LOOPS` resolves to a string
        before it reaches the knob, so refusing "5" would break a config that
        works today.
        """
        logger = WebSearchInterceptionLogger.from_config_yaml(
            {"enabled_providers": ["bedrock"], "max_agentic_loops": "5"}
        )

        assert logger.max_agentic_loops == 5

    @pytest.mark.asyncio
    async def test_knob_reaches_the_loop_settings(self):
        logger = WebSearchInterceptionLogger.from_config_yaml(
            {"enabled_providers": ["bedrock"], "max_agentic_loops": 7}
        )
        kwargs = {
            "tools": [{"type": "web_search_20250305", "name": "web_search", "max_uses": 5}],
            "litellm_params": {"custom_llm_provider": "bedrock"},
        }

        updated = await logger.async_pre_request_hook(model="claude-sonnet-4-5", messages=[], kwargs=kwargs)

        _, max_loops, _ = BaseLLMHTTPHandler._get_agentic_loop_settings(kwargs=updated)
        assert max_loops == 7

    @pytest.mark.asyncio
    async def test_deployment_setting_wins_over_the_feature_setting(self):
        logger = WebSearchInterceptionLogger.from_config_yaml(
            {"enabled_providers": ["bedrock"], "max_agentic_loops": 7}
        )
        kwargs = {
            "tools": [{"type": "web_search_20250305", "name": "web_search", "max_uses": 5}],
            "litellm_params": {"custom_llm_provider": "bedrock"},
            "max_agentic_loops": 2,
        }

        updated = await logger.async_pre_request_hook(model="claude-sonnet-4-5", messages=[], kwargs=kwargs)

        _, max_loops, _ = BaseLLMHTTPHandler._get_agentic_loop_settings(kwargs=updated)
        assert max_loops == 2

    @pytest.mark.asyncio
    async def test_default_ceiling_applies_when_the_knob_is_unset(self):
        logger = WebSearchInterceptionLogger.from_config_yaml({"enabled_providers": ["bedrock"]})
        kwargs = {
            "tools": [{"type": "web_search_20250305", "name": "web_search", "max_uses": 5}],
            "litellm_params": {"custom_llm_provider": "bedrock"},
        }

        updated = await logger.async_pre_request_hook(model="claude-sonnet-4-5", messages=[], kwargs=kwargs)

        assert "max_agentic_loops" not in updated
        _, max_loops, _ = BaseLLMHTTPHandler._get_agentic_loop_settings(kwargs=updated)
        assert max_loops == 3


def _stream_events(response: dict) -> list[dict]:
    events: list[dict] = []
    for chunk in FakeAnthropicMessagesStreamIterator(response=response):
        for line in chunk.decode().splitlines():
            if line.startswith("data: "):
                events.append(json.loads(line[len("data: ") :]))
    return events


class TestBothCeilingKnobsAreValidated:
    """
    ``max_agentic_loops`` is settable per deployment and feature-wide, and the
    per-deployment one wins. Only the feature-wide one used to be checked, so a
    per-deployment ``0`` was swallowed by an ``or 3`` and read as the default 3,
    handing the loosest ceiling to whoever asked for the tightest.
    """

    def test_a_per_deployment_zero_is_rejected_not_read_as_the_default(self):
        with pytest.raises(ValueError, match="must be at least 1, got 0"):
            BaseLLMHTTPHandler._get_agentic_loop_settings(kwargs={"max_agentic_loops": 0})

    def test_a_per_deployment_non_integer_names_the_field_it_came_from(self):
        with pytest.raises(TypeError, match=r"litellm_params\.max_agentic_loops must be an integer"):
            BaseLLMHTTPHandler._get_agentic_loop_settings(kwargs={"max_agentic_loops": "three"})

    def test_a_per_deployment_true_is_not_read_as_a_ceiling_of_one(self):
        with pytest.raises(TypeError, match="must be an integer"):
            BaseLLMHTTPHandler._get_agentic_loop_settings(kwargs={"max_agentic_loops": True})

    def test_an_absent_ceiling_falls_back_to_the_shared_default(self):
        _, max_loops, _ = BaseLLMHTTPHandler._get_agentic_loop_settings(kwargs={})

        assert max_loops == DEFAULT_MAX_AGENTIC_LOOPS

    def test_an_explicit_none_falls_back_to_the_shared_default(self):
        _, max_loops, _ = BaseLLMHTTPHandler._get_agentic_loop_settings(kwargs={"max_agentic_loops": None})

        assert max_loops == DEFAULT_MAX_AGENTIC_LOOPS

    def test_a_valid_per_deployment_ceiling_is_passed_through(self):
        _, max_loops, _ = BaseLLMHTTPHandler._get_agentic_loop_settings(kwargs={"max_agentic_loops": 6})

        assert max_loops == 6

    @pytest.mark.parametrize("rejected", [0, -1, "three", True])
    def test_the_two_knobs_reject_the_same_values(self, rejected):
        with pytest.raises((TypeError, ValueError)):
            WebSearchInterceptionLogger(max_agentic_loops=rejected)
        with pytest.raises((TypeError, ValueError)):
            BaseLLMHTTPHandler._get_agentic_loop_settings(kwargs={"max_agentic_loops": rejected})

    def test_each_knob_names_its_own_config_field(self):
        with pytest.raises(ValueError, match=r"websearch_interception_params\.max_agentic_loops"):
            WebSearchInterceptionLogger(max_agentic_loops=0)
        with pytest.raises(ValueError, match=r"litellm_params\.max_agentic_loops"):
            BaseLLMHTTPHandler._get_agentic_loop_settings(kwargs={"max_agentic_loops": 0})


class TestACeilingThatSpellsAWholeNumberStillWorks:
    """
    The ceiling used to go through ``int(... or 3)``, which accepted anything
    ``int()`` accepted. A ceiling is routinely parameterized as
    ``max_agentic_loops: os.environ/MAX_AGENTIC_LOOPS``, and ``get_secret``
    hands that back as the string ``"5"``, so tightening the check to
    ``isinstance(int)`` would stop such a proxy from booting on upgrade.
    """

    @pytest.mark.parametrize("spelled", ["5", " 5 ", 5.0])
    def test_a_ceiling_that_spells_five_is_accepted_by_both_knobs(self, spelled):
        _, max_loops, _ = BaseLLMHTTPHandler._get_agentic_loop_settings(kwargs={"max_agentic_loops": spelled})

        assert max_loops == 5
        assert WebSearchInterceptionLogger(max_agentic_loops=spelled).max_agentic_loops == 5

    def test_an_env_var_sourced_ceiling_survives_secret_resolution(self, monkeypatch):
        monkeypatch.setenv("MAX_AGENTIC_LOOPS_UNDER_TEST", "7")
        resolved = get_secret("os.environ/MAX_AGENTIC_LOOPS_UNDER_TEST")

        assert isinstance(resolved, str)
        _, max_loops, _ = BaseLLMHTTPHandler._get_agentic_loop_settings(kwargs={"max_agentic_loops": resolved})
        assert max_loops == 7

    def test_a_spelled_zero_is_still_refused_and_reports_the_number(self):
        with pytest.raises(ValueError, match="must be at least 1, got 0"):
            BaseLLMHTTPHandler._get_agentic_loop_settings(kwargs={"max_agentic_loops": "0"})

    def test_a_word_is_still_refused(self):
        with pytest.raises(TypeError, match=r"litellm_params\.max_agentic_loops must be an integer"):
            BaseLLMHTTPHandler._get_agentic_loop_settings(kwargs={"max_agentic_loops": "three"})

    def test_a_fractional_ceiling_is_refused_rather_than_truncated(self):
        with pytest.raises(TypeError, match="must be an integer"):
            BaseLLMHTTPHandler._get_agentic_loop_settings(kwargs={"max_agentic_loops": 5.5})


class TestRebuiltStreamIsWellFormed:
    """
    A capped turn is rebuilt into SSE by FakeAnthropicMessagesStreamIterator.

    Anthropic's SDK accumulator appends on content_block_start and then indexes
    content[event.index] on content_block_delta, so a block that stops without
    ever starting shifts every later index and the accumulator raises
    IndexError. A web search turn carries server_tool_use and
    web_search_tool_result blocks, which is exactly where that used to happen.
    """

    @staticmethod
    def _capped_search_turn() -> dict:
        return {
            "id": "msg_01",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-5",
            "stop_reason": "end_turn",
            "content": [
                {
                    "type": "server_tool_use",
                    "id": "srvtoolu_01",
                    "name": "web_search",
                    "input": {"query": "on-demand H100 hourly price"},
                },
                {
                    "type": "web_search_tool_result",
                    "tool_use_id": "srvtoolu_01",
                    "content": [
                        {
                            "type": "web_search_result",
                            "url": "https://example.com/h100",
                            "title": "H100 pricing",
                        }
                    ],
                },
                {"type": "text", "text": "AWS lists the H100 at $12.29 an hour."},
            ],
            "usage": {"input_tokens": 100, "output_tokens": 20},
        }

    def test_every_content_block_stop_has_a_matching_start(self):
        events = _stream_events(self._capped_search_turn())

        started = [event["index"] for event in events if event["type"] == "content_block_start"]
        stopped = [event["index"] for event in events if event["type"] == "content_block_stop"]

        assert started == [0, 1, 2]
        assert stopped == [0, 1, 2]

    def test_no_delta_indexes_past_the_blocks_started_before_it(self):
        events = _stream_events(self._capped_search_turn())

        blocks_started = 0
        for event in events:
            if event["type"] == "content_block_start":
                blocks_started += 1
            elif event["type"] == "content_block_delta":
                assert event["index"] < blocks_started

    def test_search_blocks_reach_the_client(self):
        events = _stream_events(self._capped_search_turn())

        started_types = [
            event["content_block"]["type"] for event in events if event["type"] == "content_block_start"
        ]

        assert started_types == ["server_tool_use", "web_search_tool_result", "text"]

    def test_the_search_result_survives_the_rebuild_intact(self):
        events = _stream_events(self._capped_search_turn())

        result_block = next(
            event["content_block"]
            for event in events
            if event["type"] == "content_block_start"
            and event["content_block"]["type"] == "web_search_tool_result"
        )

        assert result_block["tool_use_id"] == "srvtoolu_01"
        assert result_block["content"][0]["url"] == "https://example.com/h100"


WEATHER_TOOL = {
    "name": "get_weather",
    "description": "Get the current weather for a city",
    "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
}


class TestACappedTurnEndsWithTheModelAnswer:
    """
    A search the cap refuses gets Anthropic's max_uses_exceeded result instead of
    running, and one last call without the search tool lets the model answer
    from the results it already has.
    """

    @pytest.fixture(autouse=True)
    def interception_capped_at_two(self, monkeypatch):
        monkeypatch.setattr(
            litellm,
            "callbacks",
            [WebSearchInterceptionLogger(enabled_providers=["anthropic", "bedrock"], max_agentic_loops=2)],
        )

    @staticmethod
    async def _ask(max_tokens: int = 1024, **request) -> dict:
        return await litellm.anthropic_interface.messages.acreate(
            model="anthropic/claude-sonnet-5-5",
            messages=[{"role": "user", "content": "What's the latest AI news?"}],
            max_tokens=max_tokens,
            **request,
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("stream", [False, True])
    async def test_every_call_of_the_turn_keeps_the_max_tokens_the_client_sent(
        self, anthropic_messages_upstream, perplexity_search, stream
    ):
        response = await self._ask(max_tokens=2048, stream=stream, tools=[get_litellm_web_search_tool()])
        if stream:
            _ = [chunk async for chunk in response]

        assert [body["max_tokens"] for body in anthropic_messages_upstream.bodies] == [2048] * 4

    @pytest.mark.asyncio
    async def test_capped_turn_ends_with_the_model_answer(self, anthropic_messages_upstream, perplexity_search):
        response = await self._ask(tools=[get_litellm_web_search_tool(), WEATHER_TOOL])

        assert [block.get("text") for block in response["content"] if block["type"] == "text"] == [FINAL_ANSWER]
        assert response["stop_reason"] == "end_turn"
        assert perplexity_search.call_count == 2

    @pytest.mark.asyncio
    async def test_the_refused_search_gets_the_limit_result_and_one_last_call_without_search(
        self, anthropic_messages_upstream, perplexity_search
    ):
        await self._ask(tools=[get_litellm_web_search_tool(), WEATHER_TOOL])

        bodies = anthropic_messages_upstream.bodies
        assert [offered_tool_names(body) for body in bodies] == [
            *[[INTERNAL_TOOL_NAME, "get_weather"]] * 3,
            ["get_weather"],
        ]
        refused_result = bodies[-1]["messages"][-1]["content"][0]
        assert refused_result["tool_use_id"] == "toolu_3"
        assert SEARCH_LIMIT_ERROR_CODE in refused_result["content"]
        assert all(SEARCH_LIMIT_ERROR_CODE not in json.dumps(body) for body in bodies[:-1])

    @pytest.mark.asyncio
    async def test_last_call_blocks_tool_calls_when_search_was_the_only_tool(self, anthropic_messages_upstream, perplexity_search):
        response = await self._ask(tools=[get_litellm_web_search_tool()])

        last_request = anthropic_messages_upstream.bodies[-1]
        assert "tools" not in last_request
        assert last_request["tool_choice"] == {"type": "none"}
        assert [block.get("text") for block in response["content"] if block["type"] == "text"] == [FINAL_ANSWER]

    @pytest.mark.asyncio
    async def test_a_forced_search_choice_relaxes_to_auto_once_search_is_gone(
        self, anthropic_messages_upstream, perplexity_search
    ):
        await self._ask(
            tools=[get_litellm_web_search_tool(), WEATHER_TOOL],
            tool_choice={"type": "tool", "name": INTERNAL_TOOL_NAME},
        )

        bodies = anthropic_messages_upstream.bodies
        assert [body.get("tool_choice") for body in bodies] == [
            *[{"type": "tool", "name": INTERNAL_TOOL_NAME}] * 3,
            {"type": "auto"},
        ]
        assert offered_tool_names(bodies[-1]) == ["get_weather"]

    @pytest.mark.asyncio
    async def test_a_choice_naming_a_client_tool_survives(self, anthropic_messages_upstream, perplexity_search):
        await self._ask(
            tools=[get_litellm_web_search_tool(), WEATHER_TOOL],
            tool_choice={"type": "tool", "name": "get_weather"},
        )

        assert anthropic_messages_upstream.bodies[-1]["tool_choice"] == {"type": "tool", "name": "get_weather"}

    @pytest.mark.asyncio
    async def test_a_choice_forcing_some_tool_lets_the_last_call_answer(
        self, anthropic_messages_upstream, perplexity_search
    ):
        response = await self._ask(tools=[get_litellm_web_search_tool(), WEATHER_TOOL], tool_choice={"type": "any"})

        assert anthropic_messages_upstream.bodies[-1]["tool_choice"] == {"type": "auto"}
        assert [block.get("text") for block in response["content"] if block["type"] == "text"] == [FINAL_ANSWER]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "tool_choice",
        [
            {"type": "any", "disable_parallel_tool_use": True},
            {"type": "tool", "name": INTERNAL_TOOL_NAME, "disable_parallel_tool_use": True},
        ],
        ids=["forcing some tool", "forcing the search"],
    )
    async def test_a_relaxed_choice_keeps_the_one_tool_call_limit(
        self, anthropic_messages_upstream, perplexity_search, tool_choice
    ):
        await self._ask(tools=[get_litellm_web_search_tool(), WEATHER_TOOL], tool_choice=tool_choice)

        assert anthropic_messages_upstream.bodies[-1]["tool_choice"] == {
            "type": "auto",
            "disable_parallel_tool_use": True,
        }

    @pytest.mark.asyncio
    async def test_a_turn_that_ends_before_the_cap_makes_no_extra_call(
        self, monkeypatch, anthropic_messages_upstream, perplexity_search
    ):
        monkeypatch.setattr(
            litellm, "callbacks", [WebSearchInterceptionLogger(enabled_providers=["anthropic"], max_agentic_loops=8)]
        )

        await self._ask(tools=[get_litellm_web_search_tool()])

        assert len(anthropic_messages_upstream.bodies) == 8
        assert all(SEARCH_LIMIT_ERROR_CODE not in json.dumps(body) for body in anthropic_messages_upstream.bodies)

    @pytest.mark.asyncio
    async def test_bedrock_still_refuses_the_search_past_the_cap(self, perplexity_search):
        handler = BaseLLMHTTPHandler()
        callback = WebSearchInterceptionLogger(enabled_providers=["bedrock"], max_agentic_loops=2)

        result = await handler._call_agentic_completion_hooks(
            response=_response_asking_for_another_search(),
            model="anthropic.claude-sonnet-5-5",
            messages=[{"role": "user", "content": "who won the world cup"}],
            anthropic_messages_provider_config=MagicMock(),
            anthropic_messages_optional_request_params={"tools": [get_litellm_web_search_tool()]},
            logging_obj=_logging_obj(callback),
            stream=False,
            custom_llm_provider="bedrock",
            kwargs={"_agentic_loop_depth": 2, "max_agentic_loops": 2},
        )

        assert result["stop_reason"] == "end_turn"
        assert INTERNAL_TOOL_NAME not in _tool_use_names(result)
        assert perplexity_search.call_count == 0

    @pytest.mark.asyncio
    async def test_bedrock_chat_still_refuses_the_search_past_the_cap(self, perplexity_search):
        callback = WebSearchInterceptionLogger(enabled_providers=["bedrock"], max_agentic_loops=2)
        response_asking_for_another_search = ModelResponse(
            choices=[
                Choices(
                    finish_reason="tool_calls",
                    index=0,
                    message=Message(
                        role="assistant",
                        content=None,
                        tool_calls=[
                            ChatCompletionMessageToolCall(
                                id="call_3",
                                type="function",
                                function=Function(name=INTERNAL_TOOL_NAME, arguments='{"query": "world cup winner"}'),
                            )
                        ],
                    ),
                )
            ],
        )

        result = await BaseLLMHTTPHandler()._call_agentic_chat_completion_hooks(
            response=response_asking_for_another_search,
            model="anthropic.claude-sonnet-5-5",
            messages=[{"role": "user", "content": "who won the world cup"}],
            optional_params={"tools": [get_litellm_web_search_tool_openai()]},
            logging_obj=_logging_obj(callback),
            stream=False,
            custom_llm_provider="bedrock",
            kwargs={"_agentic_loop_depth": 2, "max_agentic_loops": 2},
        )

        assert isinstance(result, ModelResponse)
        assert result.choices[0].finish_reason == "stop"
        assert not result.choices[0].message.tool_calls
        assert perplexity_search.call_count == 0


OPENAI_COMPATIBLE_API_BASE: Final = "https://openai-compatible.example/v1"
SEARCHES_THE_MODEL_ASKS_FOR: Final = DEFAULT_MAX_AGENTIC_LOOPS + 2
WORLD_CUP_ANSWER: Final = "Argentina won the 2022 World Cup."


def _model_calls_in_a_turn_that_reaches_the_cap(loops_allowed: int) -> int:
    return loops_allowed + 2


def _openai_reply_asking_for_another_search_until_enough_results(request: httpx.Request) -> httpx.Response:
    body: Final = json.loads(request.content)
    searches_done: Final = sum(1 for message in body["messages"] if message["role"] == "tool")
    done_searching: Final = (
        searches_done >= SEARCHES_THE_MODEL_ASKS_FOR or INTERNAL_TOOL_NAME not in offered_tool_names(body)
    )
    search_call: Final = {
        "id": f"call_{searches_done}",
        "type": "function",
        "function": {
            "name": INTERNAL_TOOL_NAME,
            "arguments": json.dumps({"query": f"world cup winner, source {searches_done}"}),
        },
    }
    message: Final = (
        {"role": "assistant", "content": WORLD_CUP_ANSWER}
        if done_searching
        else {"role": "assistant", "content": None, "tool_calls": [search_call]}
    )
    return httpx.Response(
        200,
        json={
            "id": f"chatcmpl-{searches_done}",
            "object": "chat.completion",
            "created": 0,
            "model": "gpt-5.6",
            "choices": [
                {"index": 0, "message": message, "finish_reason": "tool_calls" if "tool_calls" in message else "stop"}
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        },
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("request_cap", "loops_allowed"),
    [({"max_agentic_loops": 1}, 1), ({}, DEFAULT_MAX_AGENTIC_LOOPS)],
    ids=["max_agentic_loops=1", "default cap"],
)
async def test_openai_chat_completion_stops_searching_at_the_agentic_loop_cap(
    request_cap: dict[str, int],
    loops_allowed: int,
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
    perplexity_search: respx.Route,
):
    monkeypatch.setattr(litellm, "callbacks", [WebSearchInterceptionLogger(enabled_providers=["openai"])])
    litellm.in_memory_llm_clients_cache.flush_cache()
    openai_route: Final = respx_mock.post(f"{OPENAI_COMPATIBLE_API_BASE}/chat/completions").mock(
        side_effect=_openai_reply_asking_for_another_search_until_enough_results
    )

    response: Final = await litellm.acompletion(
        model="openai/gpt-5.6",
        messages=[{"role": "user", "content": "Who won the last World Cup?"}],
        tools=[{"type": "web_search"}],
        api_key="sk-test",
        api_base=OPENAI_COMPATIBLE_API_BASE,
        **request_cap,
    )

    assert openai_route.call_count == _model_calls_in_a_turn_that_reaches_the_cap(loops_allowed)
    assert perplexity_search.call_count == loops_allowed
    assert response.choices[0].message.content == WORLD_CUP_ANSWER
    assert {call.request.headers["authorization"] for call in openai_route.calls} == {"Bearer sk-test"}


GEMINI_GENERATE_CONTENT_URL: Final = r"^https://generativelanguage\.googleapis\.com/.+:generateContent"


def _gemini_reply(searches: bool) -> httpx.Response:
    part: Final = (
        {"functionCall": {"name": INTERNAL_TOOL_NAME, "args": {"query": "world cup winner"}}}
        if searches
        else {"text": WORLD_CUP_ANSWER}
    )
    return httpx.Response(
        200,
        json={
            "candidates": [{"content": {"role": "model", "parts": [part]}, "finishReason": "STOP", "index": 0}],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5, "totalTokenCount": 15},
        },
    )


def _gemini_reply_that_keeps_searching_until_told_to_answer(request: httpx.Request) -> httpx.Response:
    """
    Gemini 3.8 Flash called the undeclared search function at the capped call in 6 of 18
    live replays (2026-11-23) unless the refused search result told it to answer now.
    """
    content: Final = request.content.decode()
    search_offered: Final = INTERNAL_TOOL_NAME in json.dumps(json.loads(content).get("tools") or [])
    return _gemini_reply(searches=search_offered or "answer now" not in content)


def _gemini_reply_that_always_searches(request: httpx.Request) -> httpx.Response:
    return _gemini_reply(searches=True)


@pytest.fixture
def gemini_upstream(
    monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter, perplexity_search: respx.Route
) -> respx.Route:
    monkeypatch.setattr(litellm, "callbacks", [WebSearchInterceptionLogger(enabled_providers=["gemini"])])
    litellm.in_memory_llm_clients_cache.flush_cache()
    return respx_mock.post(url__regex=GEMINI_GENERATE_CONTENT_URL)


@pytest.mark.asyncio
async def test_gemini_chat_completion_answers_instead_of_searching_again_at_the_agentic_loop_cap(
    gemini_upstream: respx.Route,
):
    gemini_upstream.mock(side_effect=_gemini_reply_that_keeps_searching_until_told_to_answer)

    response: Final = await litellm.acompletion(
        model="gemini/gemini-3.8-flash",
        messages=[{"role": "user", "content": "Who won the last World Cup?"}],
        tools=[{"type": "web_search"}],
        api_key="test-gemini-key",
        max_agentic_loops=1,
    )

    assert gemini_upstream.call_count == _model_calls_in_a_turn_that_reaches_the_cap(1)
    assert response.choices[0].message.content == WORLD_CUP_ANSWER
    assert not response.choices[0].message.tool_calls


@pytest.mark.asyncio
async def test_gemini_chat_completion_never_hands_the_client_a_search_call_the_capped_call_still_made(
    gemini_upstream: respx.Route,
):
    gemini_upstream.mock(side_effect=_gemini_reply_that_always_searches)

    response: Final = await litellm.acompletion(
        model="gemini/gemini-3.8-flash",
        messages=[{"role": "user", "content": "Who won the last World Cup?"}],
        tools=[{"type": "web_search"}],
        api_key="test-gemini-key",
        max_agentic_loops=1,
    )

    assert gemini_upstream.call_count == _model_calls_in_a_turn_that_reaches_the_cap(1)
    assert response.choices[0].finish_reason == "stop"
    assert not response.choices[0].message.tool_calls


@pytest.mark.asyncio
async def test_gemini_responses_never_hands_the_client_a_search_call_the_capped_call_still_made(
    gemini_upstream: respx.Route,
):
    gemini_upstream.mock(side_effect=_gemini_reply_that_always_searches)

    response: Final = await litellm.aresponses(
        model="gemini/gemini-3.8-flash",
        input="Who won the last World Cup?",
        tools=[{"type": "web_search"}],
        api_key="test-gemini-key",
        max_agentic_loops=1,
    )

    assert gemini_upstream.call_count == _model_calls_in_a_turn_that_reaches_the_cap(1)
    assert [item.type for item in response.output if item.type == "function_call"] == []


@pytest.mark.asyncio
async def test_anthropic_messages_never_hands_the_client_a_search_call_the_capped_call_still_made(
    monkeypatch: pytest.MonkeyPatch,
    always_searching_anthropic_messages_upstream: RecordingUpstream,
    perplexity_search: respx.Route,
):
    monkeypatch.setattr(litellm, "callbacks", [WebSearchInterceptionLogger(enabled_providers=["anthropic"])])

    response: Final = await litellm.anthropic_interface.messages.acreate(
        model="anthropic/claude-sonnet-5-5",
        messages=[{"role": "user", "content": "Who won the last World Cup?"}],
        max_tokens=1024,
        tools=[get_litellm_web_search_tool()],
        max_agentic_loops=1,
    )

    assert len(always_searching_anthropic_messages_upstream.bodies) == _model_calls_in_a_turn_that_reaches_the_cap(1)
    assert _tool_use_names(response) == []


def _chat_function_tool(name: str) -> dict[str, object]:
    return {"type": "function", "function": {"name": name, "parameters": {"type": "object", "properties": {}}}}


def _chat_allowed_tools(*entries: dict[str, object]) -> dict[str, object]:
    return {"type": "allowed_tools", "allowed_tools": {"mode": "auto", "tools": list(entries)}}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("allowed_client_tools", "last_call_tools", "last_call_tool_choice"),
    [
        ((), [], None),
        (
            ("get_time",),
            ["get_time", "delete_records"],
            _chat_allowed_tools({"type": "function", "function": {"name": "get_time"}}),
        ),
    ],
    ids=["only search allowed", "a client tool allowed"],
)
async def test_openai_chat_completion_keeps_the_allowed_tools_limit_on_the_last_call_at_the_cap(
    allowed_client_tools: tuple[str, ...],
    last_call_tools: list[str],
    last_call_tool_choice: object,
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
    perplexity_search: respx.Route,
):
    monkeypatch.setattr(litellm, "callbacks", [WebSearchInterceptionLogger(enabled_providers=["openai"])])
    litellm.in_memory_llm_clients_cache.flush_cache()
    openai_route: Final = respx_mock.post(f"{OPENAI_COMPATIBLE_API_BASE}/chat/completions").mock(
        side_effect=_openai_reply_asking_for_another_search_until_enough_results
    )

    response: Final = await litellm.acompletion(
        model="openai/gpt-5.6",
        messages=[{"role": "user", "content": "Who won the last World Cup?"}],
        tools=[{"type": "web_search"}, _chat_function_tool("get_time"), _chat_function_tool("delete_records")],
        tool_choice=_chat_allowed_tools(
            {"type": "web_search"}, *({"type": "function", "function": {"name": name}} for name in allowed_client_tools)
        ),
        api_key="sk-test",
        api_base=OPENAI_COMPATIBLE_API_BASE,
        max_agentic_loops=1,
    )

    last_call: Final = json.loads(openai_route.calls.last.request.content)
    assert offered_tool_names(last_call) == last_call_tools
    assert last_call.get("tool_choice") == last_call_tool_choice
    assert response.choices[0].message.content == WORLD_CUP_ANSWER


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("client_tools", "last_call_parallel_tool_calls"),
    [((), None), ((_chat_function_tool("get_time"),), False)],
    ids=["no client tool left", "a client tool left"],
)
async def test_openai_chat_completion_last_call_at_the_cap_sends_parallel_tool_calls_only_with_tools(
    client_tools: tuple[dict[str, object], ...],
    last_call_parallel_tool_calls: bool | None,
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
    perplexity_search: respx.Route,
):
    monkeypatch.setattr(litellm, "callbacks", [WebSearchInterceptionLogger(enabled_providers=["openai"])])
    litellm.in_memory_llm_clients_cache.flush_cache()
    openai_route: Final = respx_mock.post(f"{OPENAI_COMPATIBLE_API_BASE}/chat/completions").mock(
        side_effect=_openai_reply_asking_for_another_search_until_enough_results
    )

    response: Final = await litellm.acompletion(
        model="openai/gpt-5.6",
        messages=[{"role": "user", "content": "Who won the last World Cup?"}],
        tools=[{"type": "web_search"}, *client_tools],
        parallel_tool_calls=False,
        api_key="sk-test",
        api_base=OPENAI_COMPATIBLE_API_BASE,
        max_agentic_loops=1,
    )

    last_call: Final = json.loads(openai_route.calls.last.request.content)
    assert last_call.get("parallel_tool_calls") == last_call_parallel_tool_calls
    assert response.choices[0].message.content == WORLD_CUP_ANSWER


def _round_input_tokens(searches_done: int) -> int:
    return 10 + 1000 * searches_done


def _responses_reply_asking_for_searches_until(searches_wanted: int) -> Callable[[httpx.Request], httpx.Response]:
    def reply(request: httpx.Request) -> httpx.Response:
        body: Final = json.loads(request.content)
        searches_done: Final = sum(1 for item in body["input"] if item.get("type") == "function_call_output")
        done_searching: Final = searches_done >= searches_wanted or INTERNAL_TOOL_NAME not in offered_tool_names(body)
        output_item: Final = (
            {
                "type": "message",
                "id": f"msg_{searches_done}",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": WORLD_CUP_ANSWER, "annotations": []}],
            }
            if done_searching
            else {
                "type": "function_call",
                "id": f"fc_{searches_done}",
                "call_id": f"call_{searches_done}",
                "name": INTERNAL_TOOL_NAME,
                "arguments": json.dumps({"query": f"world cup winner, source {searches_done}"}),
                "status": "completed",
            }
        )
        return httpx.Response(
            200,
            json={
                "id": f"resp_{searches_done}",
                "object": "response",
                "created_at": 0,
                "status": "completed",
                "model": "gpt-5.6",
                "output": [output_item],
                "parallel_tool_calls": True,
                "tool_choice": "auto",
                "tools": [],
                "usage": {
                    "input_tokens": _round_input_tokens(searches_done),
                    "output_tokens": 5,
                    "total_tokens": _round_input_tokens(searches_done) + 5,
                },
            },
        )

    return reply


_responses_reply_asking_for_another_search_until_enough_results: Final = _responses_reply_asking_for_searches_until(
    SEARCHES_THE_MODEL_ASKS_FOR
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("request_cap", "loops_allowed"),
    [({"max_agentic_loops": 1}, 1), ({}, DEFAULT_MAX_AGENTIC_LOOPS)],
    ids=["max_agentic_loops=1", "default cap"],
)
async def test_openai_chat_completion_bridged_to_the_responses_api_stops_searching_at_the_agentic_loop_cap(
    request_cap: dict[str, int],
    loops_allowed: int,
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
    perplexity_search: respx.Route,
):
    monkeypatch.setattr(litellm, "callbacks", [WebSearchInterceptionLogger(enabled_providers=["openai"])])
    litellm.in_memory_llm_clients_cache.flush_cache()
    responses_route: Final = respx_mock.post("https://api.openai.com/v1/responses").mock(
        side_effect=_responses_reply_asking_for_another_search_until_enough_results
    )
    chat_route: Final = respx_mock.post(OPENAI_CHAT_URL).mock(
        side_effect=_openai_reply_asking_for_another_search_until_enough_results
    )

    response: Final = await litellm.acompletion(
        model="openai/gpt-5.6",
        messages=[{"role": "user", "content": "Who won the last World Cup?"}],
        tools=[{"type": "web_search"}],
        api_key="sk-test",
        **request_cap,
    )

    model_calls: Final = (*responses_route.calls, *chat_route.calls)
    assert len(model_calls) == _model_calls_in_a_turn_that_reaches_the_cap(loops_allowed)
    assert perplexity_search.call_count == loops_allowed
    assert all("_agentic_loop_api_surface" not in json.loads(call.request.content) for call in model_calls)
    assert response.choices[0].message.content == WORLD_CUP_ANSWER


@pytest.mark.asyncio
async def test_responses_last_call_at_the_cap_does_not_force_a_client_tool(
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
    perplexity_search: respx.Route,
):
    monkeypatch.setattr(litellm, "callbacks", [WebSearchInterceptionLogger(enabled_providers=["openai"])])
    litellm.in_memory_llm_clients_cache.flush_cache()
    responses_route: Final = respx_mock.post(f"{OPENAI_COMPATIBLE_API_BASE}/responses").mock(
        side_effect=_responses_reply_asking_for_another_search_until_enough_results
    )

    response: Final = await litellm.aresponses(
        model="openai/gpt-5.6",
        input=[{"role": "user", "content": "Who won the last World Cup?"}],
        tools=[
            {"type": "web_search"},
            {"type": "function", "name": "delete_records", "parameters": {"type": "object", "properties": {}}},
        ],
        tool_choice="required",
        api_key="sk-test",
        api_base=OPENAI_COMPATIBLE_API_BASE,
        max_agentic_loops=1,
    )

    last_call: Final = json.loads(responses_route.calls.last.request.content)
    assert offered_tool_names(last_call) == ["delete_records"]
    assert last_call["tool_choice"] == "auto"
    assert response.output_text == WORLD_CUP_ANSWER


@pytest.mark.asyncio
async def test_responses_pinned_openai_chat_completion_keeps_its_search_follow_ups_on_the_responses_api(
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
    perplexity_search: respx.Route,
):
    monkeypatch.setattr(litellm, "callbacks", [WebSearchInterceptionLogger(enabled_providers=["openai"])])
    litellm.in_memory_llm_clients_cache.flush_cache()
    responses_route: Final = respx_mock.post(f"{OPENAI_COMPATIBLE_API_BASE}/responses").mock(
        side_effect=_responses_reply_asking_for_another_search_until_enough_results
    )
    chat_route: Final = respx_mock.post(f"{OPENAI_COMPATIBLE_API_BASE}/chat/completions").respond(404)

    response = await litellm.acompletion(
        model="openai/responses/responses-only-model",
        messages=[{"role": "user", "content": "Who won the last World Cup?"}],
        tools=[{"type": "web_search"}],
        api_key="sk-test",
        api_base=OPENAI_COMPATIBLE_API_BASE,
        max_agentic_loops=SEARCHES_THE_MODEL_ASKS_FOR,
    )

    assert chat_route.call_count == 0
    assert responses_route.call_count == SEARCHES_THE_MODEL_ASKS_FOR + 1
    assert response.choices[0].message.content == WORLD_CUP_ANSWER


class _LoggedRow(TypedDict):
    call_type: ReadOnly[str]
    prompt_tokens: ReadOnly[int]
    response_cost: ReadOnly[float]
    cost_breakdown: ReadOnly[Mapping[str, object] | None]


class _ModelSpendRowRecorder(CustomLogger):
    def __init__(self, rows_expected: int) -> None:
        super().__init__()
        self.prompt_tokens: Final[list[int]] = []
        self.response_costs: Final[list[float]] = []
        self.breakdown_totals: Final[list[object]] = []
        self._rows_expected: Final = rows_expected
        self.all_rows_logged: Final = asyncio.Event()
        self.streamed_reply_logged: Final = asyncio.Event()

    def _record(self, kwargs: dict[str, object]) -> None:
        row: Final = TypeAdapter(_LoggedRow).validate_python(kwargs["standard_logging_object"])
        if row["call_type"] != "acompletion":
            return
        self.prompt_tokens.append(row["prompt_tokens"])
        self.response_costs.append(row["response_cost"])
        self.breakdown_totals.append((row["cost_breakdown"] or {}).get("total_cost"))
        if kwargs.get("stream") is True:
            self.streamed_reply_logged.set()
        if len(self.prompt_tokens) >= self._rows_expected:
            self.all_rows_logged.set()

    async def async_log_success_event(
        self, kwargs: dict[str, object], response_obj: object, start_time: object, end_time: object
    ) -> None:
        self._record(kwargs)

    async def async_log_failure_event(
        self, kwargs: dict[str, object], response_obj: object, start_time: object, end_time: object
    ) -> None:
        self._record(kwargs)


def _every_round_input_tokens(loops_allowed: int) -> int:
    return sum(
        _round_input_tokens(searches_done)
        for searches_done in range(_model_calls_in_a_turn_that_reaches_the_cap(loops_allowed))
    )


_ASKS_FOR_A_WEB_SEARCH: Final = pytest.mark.parametrize(
    "search_ask",
    [
        {"tools": [{"type": "web_search"}]},
        {"web_search_options": {}, "allowed_openai_params": ["web_search_options"]},
    ],
    ids=["web_search tool", "web_search_options"],
)


@pytest.mark.asyncio
@pytest.mark.parametrize("loops_allowed", [1, 2])
@_ASKS_FOR_A_WEB_SEARCH
async def test_responses_pinned_openai_chat_completion_bills_every_search_round_it_served(
    loops_allowed: int,
    search_ask: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
    perplexity_search: respx.Route,
):
    recorder: Final = _ModelSpendRowRecorder(rows_expected=_model_calls_in_a_turn_that_reaches_the_cap(loops_allowed))
    monkeypatch.setattr(
        litellm, "callbacks", [WebSearchInterceptionLogger(enabled_providers=["openai"]), recorder]
    )
    litellm.in_memory_llm_clients_cache.flush_cache()
    responses_route: Final = respx_mock.post(f"{OPENAI_COMPATIBLE_API_BASE}/responses").mock(
        side_effect=_responses_reply_asking_for_another_search_until_enough_results
    )

    await litellm.acompletion(
        model="openai/responses/gpt-5.6",
        messages=[{"role": "user", "content": "Who won the last World Cup?"}],
        api_key="sk-test",
        api_base=OPENAI_COMPATIBLE_API_BASE,
        max_agentic_loops=loops_allowed,
        **search_ask,
    )
    with suppress(asyncio.TimeoutError):
        await asyncio.wait_for(recorder.all_rows_logged.wait(), timeout=5)
    await GLOBAL_LOGGING_WORKER.flush()

    assert perplexity_search.call_count == loops_allowed
    assert responses_route.call_count == _model_calls_in_a_turn_that_reaches_the_cap(loops_allowed)
    assert sum(recorder.prompt_tokens) == _every_round_input_tokens(loops_allowed)


@pytest.mark.asyncio
@pytest.mark.parametrize("loops_allowed", [1, 2])
@_ASKS_FOR_A_WEB_SEARCH
async def test_streamed_responses_pinned_openai_chat_completion_bills_every_search_round_once(
    loops_allowed: int,
    search_ask: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
    perplexity_search: respx.Route,
):
    recorder: Final = _ModelSpendRowRecorder(rows_expected=2)
    monkeypatch.setattr(
        litellm, "callbacks", [WebSearchInterceptionLogger(enabled_providers=["openai"]), recorder]
    )
    litellm.in_memory_llm_clients_cache.flush_cache()
    respx_mock.post(f"{OPENAI_COMPATIBLE_API_BASE}/responses").mock(
        side_effect=_responses_reply_asking_for_another_search_until_enough_results
    )

    stream: Final = await litellm.acompletion(
        model="openai/responses/gpt-5.6",
        messages=[{"role": "user", "content": "Who won the last World Cup?"}],
        api_key="sk-test",
        api_base=OPENAI_COMPATIBLE_API_BASE,
        max_agentic_loops=loops_allowed,
        stream=True,
        **search_ask,
    )
    assert isinstance(stream, litellm.CustomStreamWrapper)
    _ = [chunk async for chunk in stream]
    await asyncio.wait_for(recorder.streamed_reply_logged.wait(), timeout=5)
    await GLOBAL_LOGGING_WORKER.flush()

    assert perplexity_search.call_count == loops_allowed
    assert sum(recorder.prompt_tokens) == _every_round_input_tokens(loops_allowed)


def _responses_reply_failing_after(searches_that_succeed: int):
    def reply(request: httpx.Request) -> httpx.Response:
        request_input: Final = json.loads(request.content)["input"]
        searches_done: Final = sum(1 for item in request_input if item.get("type") == "function_call_output")
        if searches_done >= searches_that_succeed:
            return httpx.Response(400, json={"error": {"message": "rejected", "type": "invalid_request_error"}})
        return _responses_reply_asking_for_another_search_until_enough_results(request)

    return reply


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True], ids=["non_streamed", "streamed"])
@pytest.mark.parametrize("rounds_served", [1, 2])
async def test_responses_pinned_openai_chat_completion_bills_each_served_round_once_when_a_follow_up_fails(
    rounds_served: int,
    stream: bool,
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
    perplexity_search: respx.Route,
):
    recorder: Final = _ModelSpendRowRecorder(rows_expected=2 if stream else 2 * rounds_served + 1)
    monkeypatch.setattr(
        litellm, "callbacks", [WebSearchInterceptionLogger(enabled_providers=["openai"]), recorder]
    )
    litellm.in_memory_llm_clients_cache.flush_cache()
    respx_mock.post(f"{OPENAI_COMPATIBLE_API_BASE}/responses").mock(
        side_effect=_responses_reply_failing_after(rounds_served)
    )

    with pytest.raises(litellm.BadRequestError):
        await litellm.acompletion(
            model="openai/responses/gpt-5.6",
            messages=[{"role": "user", "content": "Who won the last World Cup?"}],
            tools=[{"type": "web_search"}],
            api_key="sk-test",
            api_base=OPENAI_COMPATIBLE_API_BASE,
            max_agentic_loops=SEARCHES_THE_MODEL_ASKS_FOR,
            stream=stream,
        )
    with suppress(asyncio.TimeoutError):
        await asyncio.wait_for(recorder.all_rows_logged.wait(), timeout=5)
    await GLOBAL_LOGGING_WORKER.flush()

    assert sum(recorder.prompt_tokens) == sum(_round_input_tokens(served) for served in range(rounds_served))


class _ProxyParksTheRequestSuccessLog(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.request_logging_obj: Logging | None = None

    async def async_pre_call_deployment_hook(
        self, kwargs: dict[str, object], call_type: CallTypes | None
    ) -> dict[str, object] | None:
        logging_obj: Final = kwargs.get("litellm_logging_obj")
        if self.request_logging_obj is not None or not isinstance(logging_obj, Logging):
            return None
        logging_obj.defer_async_logging = True
        self.request_logging_obj = logging_obj
        return None


@pytest.mark.asyncio
@pytest.mark.parametrize("rounds_served", [1, 2])
async def test_responses_pinned_openai_chat_completion_bills_its_parked_first_round_when_a_follow_up_fails(
    rounds_served: int,
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
    perplexity_search: respx.Route,
):
    parker: Final = _ProxyParksTheRequestSuccessLog()
    recorder: Final = _ModelSpendRowRecorder(rows_expected=1)
    monkeypatch.setattr(
        litellm, "callbacks", [WebSearchInterceptionLogger(enabled_providers=["openai"]), parker, recorder]
    )
    litellm.in_memory_llm_clients_cache.flush_cache()
    respx_mock.post(f"{OPENAI_COMPATIBLE_API_BASE}/responses").mock(
        side_effect=_responses_reply_failing_after(rounds_served)
    )

    with pytest.raises(litellm.BadRequestError):
        await litellm.acompletion(
            model="openai/responses/gpt-5.6",
            messages=[{"role": "user", "content": "Who won the last World Cup?"}],
            tools=[{"type": "web_search"}],
            api_key="sk-test",
            api_base=OPENAI_COMPATIBLE_API_BASE,
            max_agentic_loops=SEARCHES_THE_MODEL_ASKS_FOR,
        )
    with suppress(asyncio.TimeoutError):
        await asyncio.wait_for(recorder.all_rows_logged.wait(), timeout=5)
    await GLOBAL_LOGGING_WORKER.flush()

    assert parker.request_logging_obj is not None
    assert sum(recorder.prompt_tokens) == sum(_round_input_tokens(served) for served in range(rounds_served))


@pytest.mark.asyncio
@pytest.mark.parametrize("loops_allowed", [1, 2])
async def test_responses_pinned_openai_chat_completion_parks_one_row_billing_every_round_until_the_proxy_releases_it(
    loops_allowed: int,
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
    perplexity_search: respx.Route,
):
    parker: Final = _ProxyParksTheRequestSuccessLog()
    recorder: Final = _ModelSpendRowRecorder(rows_expected=_model_calls_in_a_turn_that_reaches_the_cap(loops_allowed))
    monkeypatch.setattr(
        litellm, "callbacks", [WebSearchInterceptionLogger(enabled_providers=["openai"]), parker, recorder]
    )
    litellm.in_memory_llm_clients_cache.flush_cache()
    respx_mock.post(f"{OPENAI_COMPATIBLE_API_BASE}/responses").mock(
        side_effect=_responses_reply_asking_for_another_search_until_enough_results
    )

    response: Final = await litellm.acompletion(
        model="openai/responses/gpt-5.6",
        messages=[{"role": "user", "content": "Who won the last World Cup?"}],
        tools=[{"type": "web_search"}],
        api_key="sk-test",
        api_base=OPENAI_COMPATIBLE_API_BASE,
        max_agentic_loops=loops_allowed,
    )
    await GLOBAL_LOGGING_WORKER.flush()
    rows_before_the_proxy_releases_the_log: Final = tuple(recorder.prompt_tokens)
    request_logging_obj: Final = parker.request_logging_obj
    assert request_logging_obj is not None
    prepared: Final = prepare_response_cost(request_logging_obj, response, datetime.now(timezone.utc))
    release: Final = request_logging_obj.enqueue_deferred_logging
    assert release is not None
    release()
    with suppress(asyncio.TimeoutError):
        await asyncio.wait_for(recorder.all_rows_logged.wait(), timeout=1)
    await GLOBAL_LOGGING_WORKER.flush()

    assert rows_before_the_proxy_releases_the_log == ()
    assert prepared is not None
    assert prepared.usage.prompt_tokens == _every_round_input_tokens(loops_allowed)
    assert recorder.prompt_tokens == [_every_round_input_tokens(loops_allowed)]


ROUND_INPUT_TOKENS_UNDER_THE_LONG_CONTEXT_TIER: Final = 200_000
ROUND_OUTPUT_TOKENS: Final = 5
TEST_OWNED_TIER_PRICES: Final = MappingProxyType(
    {
        "input_cost_per_token": 1e-06,
        "input_cost_per_token_above_272k_tokens": 3e-06,
        "output_cost_per_token": 2e-06,
        "output_cost_per_token_above_272k_tokens": 5e-06,
    }
)


@pytest.fixture
def gpt_5_6_at_test_owned_tier_prices(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    vendor_entry: Final = litellm.model_cost["gpt-5.6"]
    assert isinstance(vendor_entry, dict)
    unpriced: Final = {key: value for key, value in vendor_entry.items() if "cost" not in key}
    monkeypatch.setitem(litellm.model_cost, "gpt-5.6", {**unpriced, **TEST_OWNED_TIER_PRICES})
    changed_in_place()
    yield
    changed_in_place()


def _long_context_round_reply(request: httpx.Request) -> httpx.Response:
    reply: Final = _responses_reply_asking_for_another_search_until_enough_results(request)
    return httpx.Response(
        200,
        json={
            **reply.json(),
            "usage": {
                "input_tokens": ROUND_INPUT_TOKENS_UNDER_THE_LONG_CONTEXT_TIER,
                "output_tokens": ROUND_OUTPUT_TOKENS,
                "total_tokens": ROUND_INPUT_TOKENS_UNDER_THE_LONG_CONTEXT_TIER + ROUND_OUTPUT_TOKENS,
            },
        },
    )


def _gpt_5_6_price(rounds: int) -> float:
    prompt_cost, completion_cost = litellm.cost_per_token(
        model="gpt-5.6",
        custom_llm_provider="openai",
        prompt_tokens=ROUND_INPUT_TOKENS_UNDER_THE_LONG_CONTEXT_TIER * rounds,
        completion_tokens=ROUND_OUTPUT_TOKENS * rounds,
    )
    return prompt_cost + completion_cost


def _price_of_rounds_priced_one_by_one(rounds: int) -> float:
    one_round: Final = _gpt_5_6_price(1)
    assert _gpt_5_6_price(rounds) != pytest.approx(rounds * one_round)
    return rounds * one_round


@pytest.mark.asyncio
@pytest.mark.parametrize("loops_allowed", [1, 2])
@pytest.mark.usefixtures("gpt_5_6_at_test_owned_tier_prices")
async def test_responses_pinned_openai_chat_completion_prices_each_round_of_its_one_row_at_that_rounds_size(
    loops_allowed: int,
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
    perplexity_search: respx.Route,
):
    expected_cost: Final = _price_of_rounds_priced_one_by_one(_model_calls_in_a_turn_that_reaches_the_cap(loops_allowed))
    parker: Final = _ProxyParksTheRequestSuccessLog()
    recorder: Final = _ModelSpendRowRecorder(rows_expected=1)
    monkeypatch.setattr(
        litellm, "callbacks", [WebSearchInterceptionLogger(enabled_providers=["openai"]), parker, recorder]
    )
    litellm.in_memory_llm_clients_cache.flush_cache()
    respx_mock.post(f"{OPENAI_COMPATIBLE_API_BASE}/responses").mock(side_effect=_long_context_round_reply)

    response: Final = await litellm.acompletion(
        model="openai/responses/gpt-5.6",
        messages=[{"role": "user", "content": "Who won the last World Cup?"}],
        tools=[{"type": "web_search"}],
        api_key="sk-test",
        api_base=OPENAI_COMPATIBLE_API_BASE,
        max_agentic_loops=loops_allowed,
    )
    request_logging_obj: Final = parker.request_logging_obj
    assert request_logging_obj is not None
    prepared: Final = prepare_response_cost(request_logging_obj, response, datetime.now(timezone.utc))
    release: Final = request_logging_obj.enqueue_deferred_logging
    assert release is not None
    release()
    with suppress(asyncio.TimeoutError):
        await asyncio.wait_for(recorder.all_rows_logged.wait(), timeout=5)
    await GLOBAL_LOGGING_WORKER.flush()

    assert prepared is not None
    assert prepared.amount == pytest.approx(expected_cost)
    assert recorder.response_costs == [pytest.approx(expected_cost)]
    assert recorder.breakdown_totals == [pytest.approx(expected_cost)]


@pytest.mark.asyncio
@pytest.mark.parametrize("loops_allowed", [1, 2])
@pytest.mark.usefixtures("gpt_5_6_at_test_owned_tier_prices")
async def test_streamed_responses_pinned_openai_chat_completion_prices_each_round_at_that_rounds_size(
    loops_allowed: int,
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
    perplexity_search: respx.Route,
):
    expected_cost: Final = _price_of_rounds_priced_one_by_one(_model_calls_in_a_turn_that_reaches_the_cap(loops_allowed))
    recorder: Final = _ModelSpendRowRecorder(rows_expected=2)
    monkeypatch.setattr(
        litellm, "callbacks", [WebSearchInterceptionLogger(enabled_providers=["openai"]), recorder]
    )
    litellm.in_memory_llm_clients_cache.flush_cache()
    respx_mock.post(f"{OPENAI_COMPATIBLE_API_BASE}/responses").mock(side_effect=_long_context_round_reply)

    stream: Final = await litellm.acompletion(
        model="openai/responses/gpt-5.6",
        messages=[{"role": "user", "content": "Who won the last World Cup?"}],
        tools=[{"type": "web_search"}],
        api_key="sk-test",
        api_base=OPENAI_COMPATIBLE_API_BASE,
        max_agentic_loops=loops_allowed,
        stream=True,
    )
    assert isinstance(stream, litellm.CustomStreamWrapper)
    _ = [chunk async for chunk in stream]
    await asyncio.wait_for(recorder.streamed_reply_logged.wait(), timeout=5)
    await GLOBAL_LOGGING_WORKER.flush()

    assert sum(recorder.response_costs) == pytest.approx(expected_cost)
    assert recorder.breakdown_totals == pytest.approx(recorder.response_costs)


class _AnswersOnTheResponsesApi(CustomLogger):
    async def async_should_run_agentic_loop(
        self,
        response: object,
        model: str,
        messages: list[dict],
        tools: list[dict] | None,
        stream: bool,
        custom_llm_provider: str,
        kwargs: dict,
    ) -> tuple[bool, dict]:
        return kwargs.get("_agentic_loop_api_surface") == "responses", {}

    async def async_build_agentic_loop_plan(
        self,
        tools: dict,
        model: str,
        messages: list[dict],
        response: object,
        anthropic_messages_provider_config: BaseAnthropicMessagesConfig | None,
        anthropic_messages_optional_request_params: dict,
        logging_obj: Logging,
        stream: bool,
        kwargs: dict,
    ) -> AgenticLoopPlan:
        assert isinstance(response, ResponsesAPIResponse)
        return AgenticLoopPlan(
            response_override=response.model_copy(
                update={
                    "output": [
                        {
                            "type": "message",
                            "id": "msg_handled",
                            "status": "completed",
                            "role": "assistant",
                            "content": [
                                {"type": "output_text", "text": "Handled on the Responses API.", "annotations": []}
                            ],
                        }
                    ]
                }
            )
        )


@pytest.mark.asyncio
async def test_responses_pinned_openai_chat_completion_still_runs_agentic_hooks_that_only_handle_the_responses_api(
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
    perplexity_search: respx.Route,
):
    monkeypatch.setattr(
        litellm, "callbacks", [WebSearchInterceptionLogger(enabled_providers=["openai"]), _AnswersOnTheResponsesApi()]
    )
    litellm.in_memory_llm_clients_cache.flush_cache()
    respx_mock.post(f"{OPENAI_COMPATIBLE_API_BASE}/responses").mock(
        side_effect=_responses_reply_asking_for_another_search_until_enough_results
    )

    response: Final = await litellm.acompletion(
        model="openai/responses/gpt-5.6",
        messages=[{"role": "user", "content": "Who won the last World Cup?"}],
        tools=[{"type": "web_search"}],
        api_key="sk-test",
        api_base=OPENAI_COMPATIBLE_API_BASE,
    )

    assert response.choices[0].message.content == "Handled on the Responses API."


class _AnswersOnBothApis(_AnswersOnTheResponsesApi):
    async def async_should_run_chat_completion_agentic_loop(
        self,
        response: object,
        model: str,
        messages: list[dict],
        tools: list[dict] | None,
        stream: bool,
        custom_llm_provider: str,
        kwargs: dict,
    ) -> tuple[bool, dict]:
        return isinstance(response, litellm.ModelResponse), {}

    async def async_build_chat_completion_agentic_loop_plan(
        self,
        tools: dict,
        model: str,
        messages: list[dict],
        response: object,
        optional_params: dict,
        logging_obj: Logging,
        stream: bool,
        kwargs: dict,
    ) -> AgenticLoopPlan:
        assert isinstance(response, litellm.ModelResponse)
        return AgenticLoopPlan(
            response_override=litellm.ModelResponse(
                model=response.model,
                choices=[
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "Handled on the chat API."},
                    }
                ],
            )
        )


class _DeclinesOnTheChatApi(_AnswersOnTheResponsesApi):
    async def async_should_run_chat_completion_agentic_loop(
        self,
        response: object,
        model: str,
        messages: list[dict],
        tools: list[dict] | None,
        stream: bool,
        custom_llm_provider: str,
        kwargs: dict,
    ) -> tuple[bool, dict]:
        return False, {}


class _WantsChatHooksOnBridgedChat(_AnswersOnBothApis):
    wants_chat_completion_hooks_on_bridged_chat: ClassVar[bool] = True


@pytest.mark.asyncio
@pytest.mark.usefixtures("perplexity_search")
@pytest.mark.parametrize(
    ("callback", "answer"),
    [
        (_AnswersOnBothApis(), "Handled on the Responses API."),
        (_DeclinesOnTheChatApi(), "Handled on the Responses API."),
        (_WantsChatHooksOnBridgedChat(), "Handled on the chat API."),
    ],
)
async def test_responses_pinned_openai_chat_completion_asks_the_chat_hook_only_of_callbacks_that_opt_in(
    callback: CustomLogger, answer: str, monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter
):
    monkeypatch.setattr(litellm, "callbacks", [callback])
    litellm.in_memory_llm_clients_cache.flush_cache()
    respx_mock.post(f"{OPENAI_COMPATIBLE_API_BASE}/responses").mock(
        side_effect=_responses_reply_asking_for_another_search_until_enough_results
    )

    response: Final = await litellm.acompletion(
        model="openai/responses/gpt-5.6",
        messages=[{"role": "user", "content": "Who won the last World Cup?"}],
        api_key="sk-test",
        api_base=OPENAI_COMPATIBLE_API_BASE,
    )

    assert response.choices[0].message.content == answer


async def _answer_text(response: litellm.ModelResponse | litellm.CustomStreamWrapper) -> str | None:
    if isinstance(response, litellm.CustomStreamWrapper):
        return "".join([chunk.choices[0].delta.content or "" async for chunk in response if chunk.choices])
    return response.choices[0].message.content


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True], ids=["non-streamed", "streamed"])
async def test_responses_pinned_openai_chat_completion_runs_the_search_its_web_search_options_ask_for(
    stream: bool,
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
    perplexity_search: respx.Route,
):
    monkeypatch.setattr(litellm, "callbacks", [WebSearchInterceptionLogger(enabled_providers=["openai"])])
    litellm.in_memory_llm_clients_cache.flush_cache()
    respx_mock.post(f"{OPENAI_COMPATIBLE_API_BASE}/responses").mock(
        side_effect=_responses_reply_asking_for_searches_until(1)
    )

    response: Final = await litellm.acompletion(
        model="openai/responses/gpt-5.6",
        messages=[{"role": "user", "content": "Who won the last World Cup?"}],
        web_search_options={},
        allowed_openai_params=["web_search_options"],
        api_key="sk-test",
        api_base=OPENAI_COMPATIBLE_API_BASE,
        stream=stream,
    )

    assert perplexity_search.call_count == 1
    assert await _answer_text(response) == "Argentina won the 2022 World Cup."


@pytest.mark.asyncio
async def test_responses_pinned_chat_completion_with_web_search_options_and_a_web_search_tool_sends_one_search_tool(
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
    perplexity_search: respx.Route,
):
    monkeypatch.setattr(litellm, "callbacks", [WebSearchInterceptionLogger(enabled_providers=["openai"])])
    litellm.in_memory_llm_clients_cache.flush_cache()
    responses_route: Final = respx_mock.post(f"{OPENAI_COMPATIBLE_API_BASE}/responses").mock(
        side_effect=_responses_reply_asking_for_searches_until(1)
    )

    response: Final = await litellm.acompletion(
        model="openai/responses/gpt-5.6",
        messages=[{"role": "user", "content": "Who won the last World Cup?"}],
        web_search_options={},
        allowed_openai_params=["web_search_options"],
        tools=[{"type": "web_search"}],
        api_key="sk-test",
        api_base=OPENAI_COMPATIBLE_API_BASE,
    )

    first_round_tools: Final = json.loads(responses_route.calls[0].request.content)["tools"]
    assert [tool.get("name", tool.get("type")) for tool in first_round_tools] == [INTERNAL_TOOL_NAME]
    assert perplexity_search.call_count == 1
    assert await _answer_text(response) == "Argentina won the 2022 World Cup."


@pytest.mark.asyncio
async def test_responses_pinned_chat_completion_with_web_search_options_still_answers_400_when_nothing_allows_them(
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
    perplexity_search: respx.Route,
):
    monkeypatch.setattr(litellm, "callbacks", [WebSearchInterceptionLogger(enabled_providers=["openai"])])
    litellm.in_memory_llm_clients_cache.flush_cache()
    responses_route: Final = respx_mock.post(f"{OPENAI_COMPATIBLE_API_BASE}/responses").mock(
        side_effect=_responses_reply_asking_for_searches_until(1)
    )

    with pytest.raises(litellm.UnsupportedParamsError, match="web_search_options"):
        await litellm.acompletion(
            model="openai/responses/gpt-5.6",
            messages=[{"role": "user", "content": "Who won the last World Cup?"}],
            web_search_options={},
            api_key="sk-test",
            api_base=OPENAI_COMPATIBLE_API_BASE,
        )

    assert responses_route.call_count == 0
    assert perplexity_search.call_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "drop_settings",
    [
        {"drop_params": True},
        {"allowed_openai_params": ["web_search_options"], "additional_drop_params": ["web_search_options"]},
    ],
    ids=["drop_params", "additional_drop_params"],
)
async def test_responses_pinned_chat_completion_drops_web_search_options_the_deployment_drops_without_searching(
    drop_settings: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
    perplexity_search: respx.Route,
):
    monkeypatch.setattr(litellm, "callbacks", [WebSearchInterceptionLogger(enabled_providers=["openai"])])
    litellm.in_memory_llm_clients_cache.flush_cache()
    responses_route: Final = respx_mock.post(f"{OPENAI_COMPATIBLE_API_BASE}/responses").mock(
        side_effect=_responses_reply_asking_for_searches_until(0)
    )

    response: Final = await litellm.acompletion(
        model="openai/responses/gpt-5.6",
        messages=[{"role": "user", "content": "Who won the last World Cup?"}],
        web_search_options={},
        api_key="sk-test",
        api_base=OPENAI_COMPATIBLE_API_BASE,
        **drop_settings,
    )

    assert await _answer_text(response) == "Argentina won the 2022 World Cup."
    assert json.loads(responses_route.calls[0].request.content).get("tools") in (None, [])
    assert perplexity_search.call_count == 0


@pytest.mark.asyncio
async def test_chat_completion_bridged_for_a_reasoning_summary_bills_every_web_search_options_round(
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
    perplexity_search: respx.Route,
):
    recorder: Final = _ModelSpendRowRecorder(rows_expected=_model_calls_in_a_turn_that_reaches_the_cap(1))
    monkeypatch.setattr(
        litellm, "callbacks", [WebSearchInterceptionLogger(enabled_providers=["openai"]), recorder]
    )
    litellm.in_memory_llm_clients_cache.flush_cache()
    respx_mock.post(f"{OPENAI_COMPATIBLE_API_BASE}/responses").mock(
        side_effect=_responses_reply_asking_for_another_search_until_enough_results
    )

    await litellm.acompletion(
        model="openai/gpt-5.6",
        messages=[{"role": "user", "content": "Who won the last World Cup?"}],
        web_search_options={},
        allowed_openai_params=["web_search_options"],
        reasoning_effort="low",
        reasoning_summary="auto",
        max_agentic_loops=1,
        api_key="sk-test",
        api_base=OPENAI_COMPATIBLE_API_BASE,
    )
    with suppress(asyncio.TimeoutError):
        await asyncio.wait_for(recorder.all_rows_logged.wait(), timeout=5)
    await GLOBAL_LOGGING_WORKER.flush()

    assert perplexity_search.call_count == 1
    assert sum(recorder.prompt_tokens) == _every_round_input_tokens(1)


def _tool_names(tools: Sequence[Mapping[str, object]]) -> list[object]:
    return [tool["function"]["name"] if "function" in tool else tool.get("type") for tool in tools]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider", "deployment", "routes_all_openai_chat_to_responses"),
    [
        ("azure", {"model": "azure/responses/gpt-5.6"}, False),
        ("chatgpt", {"model": "chatgpt/gpt-5.6"}, False),
        ("openai", {"model": "openai/gpt-5.6", "model_info": {"mode": "responses"}}, False),
        ("openai", {"model": "openai/gpt-5.6"}, True),
    ],
    ids=["azure responses model", "chatgpt", "deployment pinned to mode responses", "route_all_chat_openai_to_responses"],
)
async def test_chat_completion_ending_in_a_responses_api_turns_web_search_options_into_the_search_tool(
    provider: str,
    deployment: dict[str, object],
    routes_all_openai_chat_to_responses: bool,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(litellm, "route_all_chat_openai_to_responses", routes_all_openai_chat_to_responses)

    converted: Final = await WebSearchInterceptionLogger(enabled_providers=[provider]).async_pre_call_deployment_hook(
        {
            **deployment,
            "custom_llm_provider": provider,
            "messages": [{"role": "user", "content": "Who won the last World Cup?"}],
            "web_search_options": {},
            "allowed_openai_params": ["web_search_options"],
        },
        CallTypes.acompletion,
    )

    assert converted is not None
    assert "web_search_options" not in converted
    assert _tool_names(converted["tools"]) == [INTERNAL_TOOL_NAME]


@pytest.mark.asyncio
async def test_chat_completion_that_stays_on_chat_keeps_its_web_search_options():
    converted: Final = await WebSearchInterceptionLogger(enabled_providers=["openai"]).async_pre_call_deployment_hook(
        {
            "model": "openai/gpt-5.6",
            "messages": [{"role": "user", "content": "Who won the last World Cup?"}],
            "web_search_options": {},
            "allowed_openai_params": ["web_search_options"],
        },
        CallTypes.acompletion,
    )

    assert converted is None
