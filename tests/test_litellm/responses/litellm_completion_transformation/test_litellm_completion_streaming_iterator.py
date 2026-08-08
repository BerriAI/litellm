"""Regression test: the /v1/responses bridge iterator must expose aclose()
so the proxy's streaming cleanup can release the upstream provider
connection on client disconnect.
"""

from typing import Final
from unittest.mock import MagicMock

import pytest

from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.responses.litellm_completion_transformation.streaming_iterator import (
    LiteLLMCompletionStreamingIterator,
)


class _TrackingCompletionStream:
    def __init__(self):
        self.aclose_called = False

    async def aclose(self) -> None:
        self.aclose_called = True


@pytest.mark.asyncio
async def test_aclose_closes_wrapped_completion_stream():
    completion_stream: Final = _TrackingCompletionStream()
    wrapper: Final = CustomStreamWrapper(
        completion_stream=completion_stream,
        model="gpt-4o-mini",
        logging_obj=MagicMock(),
        custom_llm_provider="openai",
    )
    iterator: Final = LiteLLMCompletionStreamingIterator(
        model="gpt-4o-mini",
        litellm_custom_stream_wrapper=wrapper,
        request_input="hello",
        responses_api_request={},
    )

    await iterator.aclose()

    assert completion_stream.aclose_called
    assert wrapper.completion_stream is None
