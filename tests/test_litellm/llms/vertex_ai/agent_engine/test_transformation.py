import json
from pathlib import Path

import httpx
import pytest

from litellm.llms.vertex_ai.agent_engine.transformation import VertexAgentEngineConfig
from litellm.types.utils import ModelResponse


def _sse_body(*events: dict) -> str:
    return "\n".join(json.dumps(event) for event in events)


def _transform(body: str) -> ModelResponse:
    config = VertexAgentEngineConfig()
    return config.transform_response(
        model="agent_engine/123456789",
        raw_response=httpx.Response(200, text=body),
        model_response=ModelResponse(),
        logging_obj=None,
        request_data={},
        messages=[{"role": "user", "content": "hi"}],
        optional_params={},
        litellm_params={},
        encoding=None,
    )


class TestExtractText:
    def test_joins_every_text_part(self):
        config = VertexAgentEngineConfig()

        text = config._extract_text_from_response(
            {"content": {"parts": [{"text": "a"}, {"function_call": {"name": "f"}}, {"text": "b"}]}}
        )

        assert text == "ab"

    def test_reasoning_parts_are_not_part_of_the_answer(self):
        config = VertexAgentEngineConfig()

        text = config._extract_text_from_response(
            {"content": {"parts": [{"text": "Let me think.", "thought": True}, {"text": "42."}]}}
        )

        assert text == "42."

    def test_internal_state_is_not_surfaced_as_content(self):
        """state_delta holds internal scratch state, not an answer."""
        config = VertexAgentEngineConfig()

        text = config._extract_text_from_response(
            {"actions": {"state_delta": {"datetime": "2026-08-20"}, "artifact_delta": {}}}
        )

        assert text == ""


class TestTransformResponse:
    def test_accumulates_text_across_events(self):
        body = _sse_body(
            {"content": {"parts": [{"text": "First agent. "}], "role": "model"}},
            {"content": {"parts": [{"functionCall": {"name": "transfer_to_agent"}}], "role": "model"}},
            {"content": {"parts": [{"text": "Second agent."}], "role": "model"}, "finish_reason": "STOP"},
        )

        result = _transform(body)

        assert result.choices[0].message.content == "First agent. Second agent."

    def test_partial_events_are_not_double_counted(self):
        """A finalised event repeats the text its partials streamed."""
        body = _sse_body(
            {"content": {"parts": [{"text": "Hel"}], "role": "model"}, "partial": True},
            {"content": {"parts": [{"text": "Hello"}], "role": "model"}, "finish_reason": "STOP"},
        )

        assert _transform(body).choices[0].message.content == "Hello"

    def test_reported_usage_is_preferred_over_an_estimate(self):
        body = _sse_body(
            {
                "content": {"parts": [{"text": "Hello"}], "role": "model"},
                "finish_reason": "STOP",
                "usage_metadata": {
                    "prompt_token_count": 100,
                    "candidates_token_count": 50,
                    "total_token_count": 150,
                },
            }
        )

        usage = _transform(body).usage

        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50
        assert usage.total_tokens == 150

    def test_usage_is_summed_over_every_model_call(self):
        """Counts from a live two-step run; keeping only the last undercuts it by a third."""
        body = _sse_body(
            {
                "content": {"parts": [{"function_call": {"name": "plan"}}], "role": "model"},
                "finish_reason": "STOP",
                "usage_metadata": {
                    "prompt_token_count": 907,
                    "candidates_token_count": 129,
                    "total_token_count": 1036,
                },
            },
            {
                "content": {"parts": [{"text": "Done."}], "role": "model"},
                "finish_reason": "STOP",
                "usage_metadata": {
                    "prompt_token_count": 1120,
                    "candidates_token_count": 225,
                    "total_token_count": 1345,
                },
            },
        )

        usage = _transform(body).usage

        assert usage.prompt_tokens == 2027
        assert usage.completion_tokens == 354
        assert usage.total_tokens == 2381

    def test_usage_falls_back_to_an_estimate_when_none_is_reported(self):
        body = _sse_body({"content": {"parts": [{"text": "Hello"}], "role": "model"}, "finish_reason": "STOP"})

        usage = _transform(body).usage

        assert (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens) == (8, 1, 9)

    def test_an_empty_usage_block_does_not_bill_zero(self):
        """An all-zero block reports nothing, so the estimate has to take over."""
        body = _sse_body(
            {"content": {"parts": [{"text": "Hello"}], "role": "model"}, "finish_reason": "STOP", "usage_metadata": {}}
        )

        assert _transform(body).usage.total_tokens == 9

    def test_a_partial_does_not_bill_the_call_it_previews(self):
        """ADK repeats the same counts on the partial and on the event that finalises it."""
        reported = {"prompt_token_count": 100, "candidates_token_count": 10, "total_token_count": 110}
        body = _sse_body(
            {"content": {"parts": [{"text": "Hel"}]}, "partial": True, "usage_metadata": reported},
            {"content": {"parts": [{"text": "Hello"}]}, "finish_reason": "STOP", "usage_metadata": reported},
        )

        assert _transform(body).usage.total_tokens == 110


class TestLiveCapture:
    """Against a redacted :streamQuery capture, so the event shapes are the API's."""

    @pytest.fixture
    def body(self):
        return (Path(__file__).parent / "fixtures" / "streamquery_tool_loop.jsonl").read_text()

    def test_the_answer_survives_the_tool_call(self, body):
        assert "research plan" in _transform(body).choices[0].message.content

    def test_the_opening_bookkeeping_event_does_not_leak(self, body):
        assert "2024-01-01" not in _transform(body).choices[0].message.content

    def test_both_model_calls_are_billed(self, body):
        assert _transform(body).usage.total_tokens == 2381

