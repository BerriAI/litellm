import json
from pathlib import Path

import pytest

from litellm.llms.vertex_ai.agent_engine.sse_iterator import (
    VertexAgentEngineResponseIterator,
)


class TestChunkParser:
    @pytest.fixture
    def iterator(self):
        return VertexAgentEngineResponseIterator(streaming_response=iter([]), sync_stream=True)

    def test_joins_every_text_part_of_an_event(self, iterator):
        chunk = {
            "content": {"parts": [{"text": "Part one. "}, {"text": "Part two."}], "role": "model"},
        }

        result = iterator.chunk_parser(chunk)

        assert result.choices[0].delta.content == "Part one. Part two."

    def test_tool_call_event_does_not_finish_the_stream(self, iterator):
        """Regression for #19121. Shape taken from a live :streamQuery response."""
        chunk = {
            "author": "interactive_planner_agent",
            "content": {
                "parts": [{"function_call": {"id": "call_1", "name": "plan"}, "thought_signature": "..."}],
                "role": "model",
            },
            "finish_reason": "STOP",
            "partial": False,
        }

        result = iterator.chunk_parser(chunk)

        assert result.choices[0].finish_reason is None
        assert result.choices[0].delta.content is None

    def test_text_event_still_finishes_the_stream(self, iterator):
        chunk = {
            "content": {"parts": [{"text": "Done."}], "role": "model"},
            "finish_reason": "STOP",
        }

        result = iterator.chunk_parser(chunk)

        assert result.choices[0].finish_reason == "stop"
        assert result.choices[0].delta.content == "Done."

    def test_non_stop_finish_reason_is_still_mapped(self, iterator):
        chunk = {
            "content": {"parts": [{"text": "cut off"}], "role": "model"},
            "finish_reason": "MAX_TOKENS",
        }

        assert iterator.chunk_parser(chunk).choices[0].finish_reason == "length"

    def test_a_blocked_response_is_not_reported_as_a_clean_stop(self, iterator):
        """A textless SAFETY event must still terminate, and say why."""
        chunk = {"content": {"parts": []}, "finish_reason": "SAFETY"}

        assert iterator.chunk_parser(chunk).choices[0].finish_reason == "content_filter"

    def test_reasoning_parts_are_not_streamed_as_the_answer(self, iterator):
        chunk = {
            "content": {"parts": [{"text": "Let me think.", "thought": True}, {"text": "42."}], "role": "model"},
        }

        assert iterator.chunk_parser(chunk).choices[0].delta.content == "42."

    def test_the_captured_tool_call_would_have_ended_the_stream(self, iterator):
        """Event 2 of 5 in a redacted capture; honouring its STOP loses the answer two events later."""
        body = (Path(__file__).parent / "fixtures" / "streamquery_tool_loop.jsonl").read_text()
        tool_call_event = [json.loads(line) for line in body.strip().split("\n")][1]

        assert tool_call_event["finish_reason"] == "STOP"
        assert not any("text" in part for part in tool_call_event["content"]["parts"])
        assert iterator.chunk_parser(tool_call_event).choices[0].finish_reason is None
