"""A downgraded stream must come back as something `async for` can consume."""

import pytest

from litellm.litellm_core_utils.chat_completion_agentic_loop import (
    _wrap_response_as_fake_stream,
)
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLogging
from litellm.types.utils import Choices, Message, ModelResponse
from litellm.utils import CustomStreamWrapper


def _response() -> ModelResponse:
    return ModelResponse(
        id="chatcmpl-1",
        choices=[Choices(index=0, finish_reason="stop", message=Message(role="assistant", content="hello"))],
        model="gpt-4o",
        object="chat.completion",
    )


def _logging_obj() -> LiteLLMLogging:
    return LiteLLMLogging(
        model="gpt-4o",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        call_type="completion",
        start_time=None,
        litellm_call_id="test-call-id",
        function_id="1",
    )


def _wrap(response):
    return _wrap_response_as_fake_stream(
        response, model="gpt-4o", custom_llm_provider="openai", logging_obj=_logging_obj()
    )


class TestWrapResponseAsFakeStream:
    def test_result_is_async_iterable(self):
        """The reported crash: `async for` got a bare ModelResponseStream."""
        wrapped = _wrap(_response())

        assert hasattr(wrapped, "__aiter__"), "result must support `async for`"
        assert isinstance(wrapped, CustomStreamWrapper)

    def test_result_is_sync_iterable_too(self):
        wrapped = _wrap(_response())

        assert hasattr(wrapped, "__iter__")

    @pytest.mark.asyncio
    async def test_yields_the_original_content_as_chunks(self):
        wrapped = _wrap(_response())

        chunks = [chunk async for chunk in wrapped]

        assert chunks, "expected at least one chunk"
        assert all(getattr(c, "object", None) == "chat.completion.chunk" for c in chunks)
        text = "".join((c.choices[0].delta.content or "") for c in chunks if getattr(c, "choices", None))
        assert "hello" in text

    def test_an_already_wrapped_stream_is_passed_through(self):
        wrapped = _wrap(_response())

        assert _wrap(wrapped) is wrapped

    def test_object_without_choices_is_returned_unchanged(self):
        sentinel = object()

        assert _wrap(sentinel) is sentinel
