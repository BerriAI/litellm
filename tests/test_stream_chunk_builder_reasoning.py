"""
Unit tests for litellm.stream_chunk_builder verifying reasoning_content aggregation.
Covers:
1. Standard sequential reasoning-then-content happy path.
2. Mixed deltas carrying both reasoning_content and content in the same chunk.
3. Multi-choice streams with reasoning_content deltas.
"""
import pytest
from litellm import ModelResponseStream, ChoicesStream, StreamingChoicesDelta
from litellm.litellm_core_utils.streaming_chunk_builder_utils import stream_chunk_builder


def test_stream_chunk_builder_reasoning_content_aggregation():
    """验证标准单 Choice 顺序思维链与正文聚合"""
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


def test_stream_chunk_builder_reasoning_and_content_mixed_in_same_chunk():
    """验证同一 chunk 的 delta 中同时携带 reasoning_content 与 content 时的边界处理"""
    chunks = [
        ModelResponseStream(
            id="chatcmpl-test-02",
            model="deepseek/deepseek-reasoner",
            choices=[
                ChoicesStream(
                    delta=StreamingChoicesDelta(
                        role="assistant",
                        reasoning_content="Preliminary thought.",
                        content=None,
                    ),
                    index=0,
                )
            ],
        ),
        # 边界场景：过渡 chunk 同时输出 reasoning_content 和 content
        ModelResponseStream(
            id="chatcmpl-test-02",
            model="deepseek/deepseek-reasoner",
            choices=[
                ChoicesStream(
                    delta=StreamingChoicesDelta(
                        reasoning_content=" Final thought concluded.",
                        content="Direct answer starts here: ",
                    ),
                    index=0,
                )
            ],
        ),
        ModelResponseStream(
            id="chatcmpl-test-02",
            model="deepseek/deepseek-reasoner",
            choices=[
                ChoicesStream(
                    delta=StreamingChoicesDelta(
                        reasoning_content=None,
                        content="all clear.",
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
    # 验证两者各自累加，互不污染、互不丢失
    assert (
        response.choices[0].message.reasoning_content
        == "Preliminary thought. Final thought concluded."
    )
    assert response.choices[0].message.content == "Direct answer starts here: all clear."


def test_stream_chunk_builder_reasoning_multiple_choices():
    """验证多 Choices 并发流式输出时每个 choice 的 reasoning_content 独立聚合"""
    chunks = [
        ModelResponseStream(
            id="chatcmpl-test-03",
            model="deepseek/deepseek-reasoner",
            choices=[
                ChoicesStream(
                    delta=StreamingChoicesDelta(
                        role="assistant",
                        reasoning_content="Choice 0 thinking",
                        content=None,
                    ),
                    index=0,
                ),
                ChoicesStream(
                    delta=StreamingChoicesDelta(
                        role="assistant",
                        reasoning_content="Choice 1 thinking",
                        content=None,
                    ),
                    index=1,
                ),
            ],
        ),
        ModelResponseStream(
            id="chatcmpl-test-03",
            model="deepseek/deepseek-reasoner",
            choices=[
                ChoicesStream(
                    delta=StreamingChoicesDelta(
                        reasoning_content=None,
                        content="Choice 0 answer",
                    ),
                    finish_reason="stop",
                    index=0,
                ),
                ChoicesStream(
                    delta=StreamingChoicesDelta(
                        reasoning_content=None,
                        content="Choice 1 answer",
                    ),
                    finish_reason="stop",
                    index=1,
                ),
            ],
        ),
    ]

    response = stream_chunk_builder(chunks)
    assert response is not None
    assert len(response.choices) == 2
    assert response.choices[0].message.reasoning_content == "Choice 0 thinking"
    assert response.choices[0].message.content == "Choice 0 answer"
    assert response.choices[1].message.reasoning_content == "Choice 1 thinking"
    assert response.choices[1].message.content == "Choice 1 answer"
