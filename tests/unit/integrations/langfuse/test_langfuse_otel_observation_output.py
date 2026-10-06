"""Regression tests for the langfuse_otel Responses API observation output (#44824).

The v1 path indexed ``content[0].text`` for ``message`` items, so refusal text
was dropped, multi-part messages lost everything past the first part, and an
empty content list raised IndexError. Aligned with the OTel v2 path's
``_responses_parts_text`` behavior.
"""

import json

from litellm.integrations.langfuse.langfuse_otel import LangfuseOtelLogger
from litellm.types.llms.openai import ResponsesAPIResponse


class RecordingSpan:
    def __init__(self):
        self.attributes = {}

    def set_attribute(self, key, value):
        self.attributes[key] = value


def observation_output(content) -> dict | None:
    response = ResponsesAPIResponse(
        id="resp_123",
        created_at=1700000000,
        model="gpt-5.6",
        object="response",
        status="completed",
        output=[{"type": "message", "id": "msg_1", "status": "completed", "role": "assistant", "content": content}],
    )
    span = RecordingSpan()
    LangfuseOtelLogger._set_observation_output(span=span, response_obj=response)
    raw = span.attributes.get("litellm.params.observation_output") or span.attributes.get("langfuse.observation.output")
    return json.loads(raw) if raw else None


def test_refusal_text_is_mapped() -> None:
    result = observation_output([{"type": "refusal", "refusal": "I can't help with that."}])
    assert result is not None
    assert len(result) == 1
    assert result[0]["role"] == "assistant"
    assert result[0]["refusal"] == "I can't help with that."
    assert "content" not in result[0]


def test_multi_part_output_text_is_joined() -> None:
    result = observation_output(
        [
            {"type": "output_text", "text": "Part one. ", "annotations": []},
            {"type": "output_text", "text": "Part two.", "annotations": []},
        ]
    )
    assert result is not None
    assert result[0]["content"] == "Part one. Part two."


def test_empty_content_list_does_not_raise() -> None:
    result = observation_output([])
    assert result is not None
    assert result[0]["role"] == "assistant"
    assert "content" not in result[0]


def test_mixed_text_and_refusal_parts() -> None:
    result = observation_output(
        [
            {"type": "output_text", "text": "partial answer. ", "annotations": []},
            {"type": "refusal", "refusal": "then I stopped."},
        ]
    )
    assert result is not None
    assert result[0]["content"] == "partial answer. "
    assert result[0]["refusal"] == "then I stopped."
