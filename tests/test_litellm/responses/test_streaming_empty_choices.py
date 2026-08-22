"""
Test for the empty-choices guard in the Responses API streaming iterator.

Providers such as DeepSeek emit a terminal streaming chunk with "choices": []
(finish/usage chunk). _get_delta_string_from_streaming_choices must return ""
for it instead of raising IndexError and killing the /v1/responses stream.
"""

from litellm.responses.litellm_completion_transformation.streaming_iterator import (
    LiteLLMCompletionStreamingIterator,
)
from litellm.types.utils import Delta, StreamingChoices


class TestEmptyChoicesGuard:
    def test_empty_choices_returns_empty_string(self):
        """A terminal chunk with choices: [] must not raise IndexError."""
        result = LiteLLMCompletionStreamingIterator._get_delta_string_from_streaming_choices(
            None, []
        )
        assert result == ""

    def test_non_empty_choices_returns_delta_content(self):
        """The normal path is unchanged: first choice's delta content is returned."""
        choice = StreamingChoices(
            index=0,
            delta=Delta(content="hello", role="assistant"),
            finish_reason=None,
        )
        result = LiteLLMCompletionStreamingIterator._get_delta_string_from_streaming_choices(
            None, [choice]
        )
        assert result == "hello"
