from unittest.mock import Mock

import pytest

from litellm.llms.anthropic.experimental_pass_through.messages.agentic_streaming_iterator import (
    AgenticAnthropicStreamingIterator,
)


class AsyncBytesIterator:
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = iter(chunks)

    def __aiter__(self):
        return self

    async def __anext__(self) -> bytes:
        try:
            return next(self._chunks)
        except StopIteration:
            raise StopAsyncIteration


@pytest.mark.asyncio
async def test_agentic_streaming_iterator_records_completion_start_time_once():
    logging_obj = Mock()
    logging_obj.model_call_details = {}

    iterator = AgenticAnthropicStreamingIterator(
        completion_stream=AsyncBytesIterator([b"event: message_start\n\n", b"event: content_block_delta\n\n"]),
        http_handler=Mock(),
        model="claude-3-5-sonnet",
        messages=[],
        anthropic_messages_provider_config=Mock(),
        anthropic_messages_optional_request_params={},
        logging_obj=logging_obj,
        custom_llm_provider="anthropic",
        kwargs={},
    )

    assert await iterator.__anext__() == b"event: message_start\n\n"
    assert await iterator.__anext__() == b"event: content_block_delta\n\n"

    logging_obj._update_completion_start_time.assert_called_once()
