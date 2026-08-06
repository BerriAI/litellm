"""Regression coverage for issue #30761.

OpenAI-compatible upstreams can emit streaming chunks with ``choices=[]``
(usage-only / metadata frames). Before the fix, multiple unguarded
``chunk.choices[0]`` accesses in the Anthropic pass-through adapter raised
``IndexError`` mid-stream and crashed the SSE response.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../..")))

from litellm.llms.anthropic.experimental_pass_through.adapters.streaming_iterator import (
    AnthropicStreamWrapper,
)
from litellm.llms.anthropic.experimental_pass_through.adapters.transformation import (
    LiteLLMAnthropicMessagesAdapter,
)
from litellm.types.utils import ModelResponse, ModelResponseStream, Usage


def test_should_start_new_content_block_skips_chunks_without_choices():
    """Pre-fix: ``chunk.choices[0].finish_reason`` raised ``IndexError`` on a
    usage-only frame inside ``AnthropicStreamWrapper._should_start_new_content_block``.
    """
    wrapper = AnthropicStreamWrapper(
        completion_stream=iter([]),
        model="gpt-5.4-mini-azure",
    )
    chunk = ModelResponseStream(choices=[])

    # Must not raise; an empty-choices frame is not a new content block.
    assert wrapper._should_start_new_content_block(chunk) is False


def test_translate_streaming_response_handles_chunks_without_choices():
    """Pre-fix: ``response.choices[0].finish_reason`` at
    ``transformation.py:1603`` raised ``IndexError`` on a usage-only chunk.
    """
    adapter = LiteLLMAnthropicMessagesAdapter()
    chunk = ModelResponseStream(choices=[])

    # Must not raise; the function falls through to the non-final branch.
    result = adapter.translate_streaming_openai_response_to_anthropic(
        response=chunk,
        current_content_block_index=0,
    )
    assert result is not None


def test_translate_openai_response_handles_empty_choices_on_non_streaming():
    """Pre-fix: ``response.choices[0].finish_reason`` at
    ``transformation.py:1401`` raised ``IndexError`` on a non-streaming response
    that came back without choices.
    """
    adapter = LiteLLMAnthropicMessagesAdapter()
    response = ModelResponse(choices=[], usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0))

    # Must not raise. The translator will route the missing finish_reason
    # through ``_translate_openai_finish_reason_to_anthropic(None)`` and emit
    # a structurally valid Anthropic-shaped result.
    out = adapter.translate_openai_response_to_anthropic(response=response)
    assert out is not None
