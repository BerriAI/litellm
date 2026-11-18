"""
Unit tests for litellm.stream_chunk_builder verifying reasoning_content aggregation.
Verifies that streaming deltas containing reasoning_content (e.g. DeepSeek-R1, Claude 3.7 Thinking)
are properly preserved and aggregated into message.reasoning_content.
"""
import pytest
from litellm import ModelResponseStream, ChoicesStream, StreamingChoicesDelta
from litellm.litellm_core_utils.streaming_chunk_builder_utils import stream_chunk_builder


def test_stream_chunk_builder_reasoning_content_aggregation():
    chunks = [
        ModelResponseStream(
            id="chatcmpl-test-01",
            model="deepseek/deepseek-reasoner",
            choices=[
                ChoicesStream(
                    delta=StreamingChoicesDelta(
                        role="assistant",
                        reasoning_content="Analyze query:\n",
                        content=None,
                    ),
                    index=0,
                )
            ],
        ),
        ModelResponseStream(
            id="chatcmpl-test-01",
            model="deepseek/deepseek-reasoner",
            choices=[
                ChoicesStream(
                    delta=StreamingChoicesDelta(
                        reasoning_content="The user wants reasoning token aggregation.",
                        content=None,
                    ),
                    index=0,
                )
            ],
        ),
        ModelResponseStream(
            id="chatcmpl-test-01",
            model="deepseek/deepseek-reasoner",
            choices=[
                ChoicesStream(
                    delta=StreamingChoicesDelta(
                        reasoning_content=None,
                        content="Conclusion reached successfully.",
                    ),
                    finish_reason="stop",
                    index=0,
                )
            ],
        ),
    ]

    response = stream_chunk_builder(chunks)
    assert response is not None
    assert response.choices[0].message.role == "assistant"
    assert response.choices[0].message.content == "Conclusion reached successfully."
    assert (
        response.choices[0].message.reasoning_content
        == "Analyze query:\nThe user wants reasoning token aggregation."
    )
