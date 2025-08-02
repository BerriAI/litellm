import pytest
import litellm
import time

@pytest.mark.asyncio
async def test_streaming_tool_call_filter():
    # Simulate a model that streams tool call and then a final response
    class DummyStream:
        def __init__(self):
            self.chunks = [
                litellm.ModelResponse(
                    stream=True,
                    choices=[
                        litellm.utils.StreamingChoices(
                            delta=litellm.utils.Delta(
                                tool_calls=[{"id": "1", "function": {"name": "foo", "arguments": "{}"}, "type": "function"}],
                                content=None,
                            ),
                            finish_reason="tool_calls",
                        )
                    ],
                ),
                litellm.ModelResponse(
                    stream=True,
                    choices=[
                        litellm.utils.StreamingChoices(
                            delta=litellm.utils.Delta(
                                tool_calls=None,
                                content="Final answer",
                            ),
                            finish_reason="stop",
                        )
                    ],
                ),
            ]
            self.index = 0
        def __iter__(self):
            return self
        def __next__(self):
            if self.index >= len(self.chunks):
                raise StopIteration
            chunk = self.chunks[self.index]
            self.index += 1
            return chunk
        def __aiter__(self):
            return self
        async def __anext__(self):
            if self.index >= len(self.chunks):
                raise StopAsyncIteration
            chunk = self.chunks[self.index]
            self.index += 1
            return chunk

    # Use the CustomStreamWrapper directly for test
    logging_obj = litellm.Logging(
        model="gpt-3.5-turbo",
        messages=[],
        stream=True,
        call_type="test",
        start_time=time.time(),
        litellm_call_id="dummy_id",
        function_id=None,
    )
    stream_wrapper = litellm.litellm_core_utils.streaming_handler.CustomStreamWrapper(
        completion_stream=DummyStream(),
        model="gpt-3.5-turbo",
        logging_obj=logging_obj,
        filter_tool_calls=True,
    )
    # Only count chunks with non-empty content
    results = [r for r in stream_wrapper if r.choices[0].delta.content]
    assert len(results) == 1
    assert results[0].choices[0].delta.content == "Final answer" 