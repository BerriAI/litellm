"""Anthropic SSE <-> ModelResponse conversion for guardrail streaming hooks.

`/v1/messages` streams reach a guardrail's `async_post_call_streaming_iterator_hook` as raw SSE
frames rather than chunk objects, which `stream_chunk_builder` cannot assemble. These helpers let a
hook scan such a stream, and re-emit it when the guardrail rewrote the response.
"""

from __future__ import annotations

import json
from typing import Final

from litellm.llms.anthropic.pass_through.stream_assembly import (
    assemble_anthropic_sse_stream as assemble_anthropic_sse_stream,
)
from litellm.llms.anthropic.pass_through.stream_assembly import (
    is_anthropic_sse_stream as is_anthropic_sse_stream,
)
from litellm.llms.anthropic.pass_through.stream_assembly import is_raw_sse_stream as is_raw_sse_stream
from litellm.types.utils import Choices, ModelResponse


def model_response_text(response: ModelResponse) -> str:
    """Assistant text of a response, used to detect whether a guardrail rewrote it."""
    return "".join(
        choice.message.content
        for choice in response.choices
        if isinstance(choice, Choices)  # pyright: ignore[reportUnnecessaryIsInstance]  # runtime choices can be StreamingChoices
        and isinstance(choice.message.content, str)
    )


def anthropic_sse_error_frames(message: str) -> tuple[bytes, ...]:
    """Anthropic error event, for a failure discovered after the response headers were flushed.

    Once a keepalive ping has been sent a raise cannot reach the client, so the failure has to
    travel as a frame.
    """
    body: Final = json.dumps(message)
    return (
        f'event: error\ndata: {{"type": "error", "error": {{"type": "guardrail_error", '
        f'"message": {body}}}}}\n\n'.encode(),
    )


from litellm.llms.anthropic.pass_through.stream_assembly import is_sse_error_stream as is_sse_error_stream  # noqa: E402


def anthropic_sse_chunks_from_response(assembled: ModelResponse) -> tuple[bytes, ...]:
    from litellm.llms.anthropic.pass_through.adapters.transformation import (
        LiteLLMAnthropicMessagesAdapter,
    )
    from litellm.llms.anthropic.pass_through.messages.fake_stream_iterator import (
        FakeAnthropicMessagesStreamIterator,
    )

    anthropic_response: Final = LiteLLMAnthropicMessagesAdapter().translate_openai_response_to_anthropic(
        response=assembled
    )
    return tuple(FakeAnthropicMessagesStreamIterator(response=anthropic_response).chunks)
