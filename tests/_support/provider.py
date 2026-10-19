"""The fake provider shared by every integration test that takes the `provider` fixture.

Deployments in `proxy_config.yaml` point at `PROVIDER_URL`, so one server answers for all of them. A test
queues the replies it expects with `expect` and reads what the proxy sent with `received`. Tests run one at a
time against it; the `provider` fixture checks nothing is left over between tests.
The proxy's model-info refresh is answered here.
"""

from __future__ import annotations

import asyncio
import os
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Final

from tests._support.wire import Reply, Request, Wire, wire_server

PROVIDER_PORT: Final = int(os.environ.get("INTEGRATION_PROVIDER_PORT", "8191"))
PROVIDER_URL: Final = f"http://127.0.0.1:{PROVIDER_PORT}"
MODEL_DISCOVERY: Final = ("GET", "/v1/models")
_UNQUEUED: Final = Reply(status=500, body=b'{"error": "the shared fake provider has no reply queued for this request"}')


@dataclass(slots=True)
class SharedProvider:
    wire: Wire
    replies: deque[Reply]
    last_test: str | None = field(default=None)

    def expect(self, *replies: Reply) -> None:
        self.replies.extend(replies)

    def received(self, *, include_model_discovery: bool = False) -> tuple[Request, ...]:
        requests: Final = self.wire.drain()
        if include_model_discovery:
            return requests
        return tuple(request for request in requests if (request.method, request.target) != MODEL_DISCOVERY)

    async def wait_for_requests(self, count: int) -> None:
        async with asyncio.timeout(2):
            while self.wire.received.qsize() < count:
                await asyncio.sleep(0.01)

    @contextmanager
    def checked(self, test_id: str) -> Iterator[None]:
        stray: Final = self.received()
        self.replies.clear()
        assert stray == (), (
            f"the shared fake provider got {[item.target for item in stray]} after {self.last_test} finished"
        )
        try:
            yield
        finally:
            self.last_test = test_id
            unused: Final = len(self.replies)
            unread: Final = self.received()
            self.replies.clear()
            assert unused == 0 and unread == (), (
                f"{unused} queued provider replies were never requested; "
                f"the test never read the provider requests {[item.target for item in unread]}"
            )


@contextmanager
def shared_provider(port: int = 0) -> Iterator[SharedProvider]:
    replies: Final[deque[Reply]] = deque()

    def respond(request: Request) -> Reply:
        if (request.method, request.target) == MODEL_DISCOVERY:
            return Reply(body=b'{"object":"list","data":[]}')
        return replies.popleft() if replies else _UNQUEUED

    with wire_server(respond, port=port) as wire:
        yield SharedProvider(wire, replies)
