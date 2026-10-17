# -*- coding: utf-8 -*-
"""Regression tests for #45796: OTEL tool-call parts + streamed finish reason.

Covered behavior:
- ``_transform_messages_to_otel_semantic_conventions`` emits GenAI
  ``{"type": "tool_call", "id", "name", "arguments"}`` parts for assistant
  messages carrying ``tool_calls`` — for plain dicts and for Pydantic
  message objects (which used to be skipped because ``"tool_calls" in msg``
  is False for non-dict objects).
- Plain string ``content`` still emits a text part (no regression).
- ``ChunkProcessor.build_base_response`` aggregates a stream whose chunks
  carried tool-call deltas but no explicit ``finish_reason`` to
  ``"tool_calls"`` instead of the default ``"stop"``.
"""
from pydantic import BaseModel

from litellm.integrations.opentelemetry import OpenTelemetry
from litellm.litellm_core_utils.streaming_chunk_builder_utils import ChunkProcessor


class _FakeAssistantMessage(BaseModel):
    role: str = "assistant"
    content: str | None = None
    tool_calls: list | None = None


class TestOtelToolCallParts:
    def test_dict_message_with_tool_calls_emits_genai_tool_call_parts(self):
        msg = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "SF"}'},
                }
            ],
        }
        out = OpenTelemetry()._transform_messages_to_otel_semantic_conventions([msg])[0]
        assert out["parts"] == [
            {"type": "tool_call", "id": "call_1", "name": "get_weather", "arguments": '{"city": "SF"}'}
        ]
        assert "tool_calls" not in out

    def test_pydantic_message_with_tool_calls_emits_genai_tool_call_parts(self):
        msg = _FakeAssistantMessage(
            tool_calls=[
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{}"},
                }
            ]
        )
        out = OpenTelemetry()._transform_messages_to_otel_semantic_conventions([msg])[0]
        assert out["parts"] == [
            {"type": "tool_call", "id": "call_2", "name": "lookup", "arguments": "{}"}
        ]

    def test_string_content_still_emits_text_part(self):
        out = OpenTelemetry()._transform_messages_to_otel_semantic_conventions(
            [{"role": "assistant", "content": "hello"}]
        )[0]
        assert out["parts"] == [{"type": "text", "content": "hello"}]

    def test_streamed_tool_call_without_explicit_finish_reason_aggregates_to_tool_calls(self):
        chunks = [
            {
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "mock",
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": None}}],
            },
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "id": "c1",
                                    "type": "function",
                                    "function": {"name": "f", "arguments": "{}"},
                                }
                            ]
                        },
                    }
                ]
            },
        ]
        resp = ChunkProcessor(chunks).build_base_response(chunks)
        assert resp.choices[0].finish_reason == "tool_calls"
