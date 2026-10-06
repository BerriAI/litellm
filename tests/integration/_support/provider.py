"""The fake provider shared by every integration test that takes the `provider` fixture.

Deployments in `proxy_config.yaml` point at `PROVIDER_URL`, so one server answers for all of them. A test
queues the replies it expects with `expect` and reads what the proxy sent with `received`. Tests run one at a
time against it; the `provider` fixture checks nothing is left over between tests.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Final

from tests.integration._support.wire import Reply, Request, Wire, wire_server

PROVIDER_PORT: Final = 8191
PROVIDER_URL: Final = f"http://127.0.0.1:{PROVIDER_PORT}"
_UNQUEUED: Final = Reply(status=500, body=b'{"error": "the shared fake provider has no reply queued for this request"}')


@dataclass(slots=True)
class SharedProvider:
    wire: Wire
    replies: deque[Reply]
    last_test: str | None = field(default=None)

    def expect(self, *replies: Reply) -> None:
        self.replies.extend(replies)

    def received(self) -> tuple[Request, ...]:
        return self.wire.drain()


@contextmanager
def shared_provider() -> Iterator[SharedProvider]:
    replies: Final[deque[Reply]] = deque()

    def respond(request: Request) -> Reply:
        return replies.popleft() if replies else _UNQUEUED

    with wire_server(respond, port=PROVIDER_PORT) as wire:
        yield SharedProvider(wire, replies)
