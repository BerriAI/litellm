from collections.abc import AsyncIterator, Iterator
from types import MappingProxyType
from typing import Final

import pytest

from litellm.router_config import build_router_config
from litellm.router_python_host import python_host
from litellm.router_rust.core import RustCore
from litellm.router_rust.invoke.streaming import StreamAttempt, wrap_anthropic_stream


@pytest.fixture
def stream_attempt() -> Iterator[StreamAttempt]:
    core: Final = RustCore.build(build_router_config(MappingProxyType({"model_list": []})), python_host)
    yield StreamAttempt(
        core=core,
        op="anthropic_messages",
        request=MappingProxyType({"model": "claude"}),
        kwargs=MappingProxyType({"stream": True}),
        group="claude",
        order=None,
        depth=0,
        level=0,
        deployment_id=None,
        attempted_retries=0,
    )
    core.retire()


@pytest.mark.asyncio
async def test_a_deeply_nested_line_with_a_usage_key_does_not_end_the_messages_stream(stream_attempt: StreamAttempt):
    frames: Final = (
        b'event: message_start\ndata: {"type":"message_start","message":{"usage":{"input_tokens":7}}}\n\n',
        b'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,'
        b'"delta":{"type":"text_delta","text":"hi"}}\n\n',
        b'event: message_delta\ndata: {"type":"message_delta","usage":' + b"[" * 100_000 + b"]" * 100_000 + b"}\n\n",
        b'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn"},'
        b'"usage":{"output_tokens":5}}\n\n',
        b'event: message_stop\ndata: {"type":"message_stop"}\n\n',
    )

    async def upstream() -> AsyncIterator[bytes]:
        for frame in frames:
            yield frame

    relayed: Final = b"".join([chunk async for chunk in wrap_anthropic_stream(stream_attempt, upstream())])

    assert relayed == b"".join(frames)
