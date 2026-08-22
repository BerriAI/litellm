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

import json
from unittest.mock import MagicMock

import pytest

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.integrations.websearch_interception.handler import (
    WebSearchInterceptionLogger,
)
from litellm.llms.anthropic.experimental_pass_through.messages.fake_stream_iterator import (
    FakeAnthropicMessagesStreamIterator,
)
from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
from litellm.types.integrations.custom_logger import (
    AgenticLoopPlan,
    AgenticLoopRequestPatch,
    AgenticLoopSafetyError,
)

INTERNAL_TOOL_NAME = "litellm_web_search"


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


def _logging_obj(callback: CustomLogger, converted_stream: bool = False) -> MagicMock:
    logging_obj = MagicMock()
    logging_obj.model_call_details = {"websearch_interception_converted_stream": converted_stream}
    logging_obj.dynamic_success_callbacks = [callback]
    logging_obj.litellm_call_id = "call-abc"
    return logging_obj


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
    async def test_responses_surface_is_left_to_its_existing_behavior(self):
        """
        The responses surface carries a pydantic model rather than the anthropic
        dict this finalizer rewrites, so it keeps raising instead of being handed
        a response that was never actually finalized.
        """
        with pytest.raises(AgenticLoopSafetyError):
            await _run_hooks(
                self.handler,
                self.callback,
                kwargs={"_agentic_loop_depth": 3, "max_agentic_loops": 3},
                api_surface="responses",
            )

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
        is empty and max_loops is clamped to at least 1, so neither rail can refuse.
        """
        depth, max_loops, fingerprints = BaseLLMHTTPHandler._get_agentic_loop_settings(kwargs={})

        assert depth == 0
        assert fingerprints == []
        assert max_loops >= 1

        depth, max_loops, fingerprints = BaseLLMHTTPHandler._get_agentic_loop_settings(
            kwargs={"max_agentic_loops": 0}
        )

        assert max_loops >= 1
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

    @pytest.mark.parametrize("bad_value", ["5", True, 2.5])
    def test_non_integer_ceilings_are_rejected_at_config_load(self, bad_value):
        with pytest.raises(TypeError, match="max_agentic_loops"):
            WebSearchInterceptionLogger.from_config_yaml(
                {"enabled_providers": ["bedrock"], "max_agentic_loops": bad_value}
            )

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
