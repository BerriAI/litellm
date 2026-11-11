import json
from typing import Final

import httpx
import pytest

from litellm.llms.langgraph.chat.sse_iterator import LangGraphSSEStreamIterator

_LINE_SEPARATED_TEXT: Final = "before\u2028between\u2029after"


def _messages_event_stream(text: str) -> httpx.Response:
    event: Final = ["messages", [[{"type": "AIMessageChunk", "content": text}, {}]]]
    return httpx.Response(200, stream=httpx.ByteStream(f"data: {json.dumps(event, ensure_ascii=False)}\n\n".encode()))


async def _streamed_contents(iterator: LangGraphSSEStreamIterator, sync_stream: bool) -> list[str | None]:
    if sync_stream:
        return [chunk.choices[0].delta.content for chunk in iterator]
    return [chunk.choices[0].delta.content async for chunk in iterator]


@pytest.mark.asyncio
@pytest.mark.parametrize("sync_stream", [True, False])
async def test_stream_keeps_unicode_line_separators_inside_a_message_chunk(sync_stream: bool) -> None:
    iterator: Final = LangGraphSSEStreamIterator(
        response=_messages_event_stream(_LINE_SEPARATED_TEXT), model="langgraph/agent"
    )

    contents: Final = await _streamed_contents(iterator, sync_stream)

    assert contents == [_LINE_SEPARATED_TEXT, None]
