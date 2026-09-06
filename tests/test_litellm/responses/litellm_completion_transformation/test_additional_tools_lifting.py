"""``additional_tools`` input items carry tool definitions that belong in ``tools``.

The item has no ``content``, so input-to-messages conversion drops it and the tools
never reach the provider. Codex CLI emits this shape.
"""

import json
from typing import Any

import pytest

from litellm.responses.litellm_completion_transformation.transformation import (
    LiteLLMCompletionResponsesConfig,
)

lift = LiteLLMCompletionResponsesConfig._lift_additional_tools


def _fn(name: str) -> dict[str, Any]:
    return {
        "type": "function",
        "name": name,
        "description": f"{name} tool",
        "parameters": {"type": "object", "properties": {}},
        "strict": False,
    }


def _item() -> dict[str, Any]:
    """Shaped like a real Codex 0.149 request: namespaces wrapping nested tools."""
    return {
        "type": "additional_tools",
        "role": "developer",
        "tools": [
            {"type": "namespace", "name": "functions", "description": "Local", "tools": [_fn("wait")]},
            {"type": "namespace", "name": "collaboration", "description": "Agents", "tools": [_fn("spawn_agent")]},
        ],
    }


class TestLiftAdditionalTools:
    def test_lifts_tools_and_strips_the_item(self):
        item = _item()
        kept, lifted = lift([{"role": "user", "content": "hi"}, item])
        assert kept == [{"role": "user", "content": "hi"}]
        assert lifted == tuple(item["tools"])

    def test_multiple_items_lift_in_order(self):
        first = {"type": "additional_tools", "role": "developer", "tools": [_fn("a")]}
        second = {"type": "additional_tools", "role": "developer", "tools": [_fn("b")]}
        kept, lifted = lift([first, {"role": "user", "content": "x"}, second])
        assert kept == [{"role": "user", "content": "x"}]
        assert [t["name"] for t in lifted] == ["a", "b"]

    def test_string_input_passes_through(self):
        assert lift("just a prompt") == ("just a prompt", ())

    def test_input_without_the_item_is_returned_unchanged(self):
        original = [{"role": "user", "content": "hi"}]
        kept, lifted = lift(original)
        assert kept is original
        assert lifted == ()

    def test_non_mapping_items_are_left_alone(self):
        """Input lists can carry non-mapping entries; they are not tool containers."""
        original = ["a bare string", 42, None, {"role": "user", "content": "hi"}]
        kept, lifted = lift(original)
        assert kept is original
        assert lifted == ()

    @pytest.mark.parametrize("malformed", [{}, {"tools": None}, {"tools": "not-a-list"}])
    def test_malformed_item_is_still_stripped(self, malformed):
        kept, lifted = lift([{"type": "additional_tools", **malformed}, {"role": "user", "content": "hi"}])
        assert kept == [{"role": "user", "content": "hi"}]
        assert lifted == ()


class TestAdditionalToolsThroughTheBridge:
    @staticmethod
    def _bridge(input_, tools=None):
        return LiteLLMCompletionResponsesConfig.transform_responses_api_request_to_chat_completion_request(
            model="gpt-5.6-sol",
            input=input_,
            responses_api_request={"tools": tools} if tools is not None else {},
            custom_llm_provider="bedrock",
        )

    def test_nested_tools_reach_the_chat_request(self):
        request = self._bridge([_item()], tools=[])
        assert request.get("tools"), "lifted tools must reach the chat request"
        serialized = json.dumps(request["tools"])
        assert "wait" in serialized
        assert "spawn_agent" in serialized

    def test_top_level_tools_are_preserved_alongside_lifted_ones(self):
        request = self._bridge([_item()], tools=[_fn("already_here")])
        names = json.dumps(request["tools"])
        assert "already_here" in names
        assert "spawn_agent" in names

    def test_request_without_tools_still_sends_none(self):
        """A request that genuinely has no tools must not gain ``tools: []`` —
        some providers reject an empty array."""
        request = self._bridge([{"role": "user", "content": "hi"}])
        assert "tools" not in request
        assert "tool_choice" not in request

    def test_user_messages_survive_the_lift(self):
        """Regression: the surviving input must stay a list. The input-to-messages
        conversion narrows on isinstance(input, list), so returning a tuple produced
        zero messages and Bedrock rejected the request outright."""
        request = self._bridge(
            [_item(), {"role": "user", "content": "read a file for me"}],
            tools=[],
        )
        contents = json.dumps(request["messages"])
        assert request["messages"], "the user turn must survive"
        assert "read a file for me" in contents
