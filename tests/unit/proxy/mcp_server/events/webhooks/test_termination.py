import base64
import json
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Final

import pytest

from litellm.proxy.mcp_server.events.webhooks.delivery import CallbackResponse
from litellm.proxy.mcp_server.events.webhooks.store import SubscriptionRow
from litellm.proxy.mcp_server.outbound_credentials.result import Error, Ok, Result
from litellm.types.mcp_server.events import CallbackFailure

from ..fakes import SETTINGS, FakeUpstream, runtime, server
from .memory_store import MemoryStore

KEY: Final = bytes(range(32))
SECRET: Final = "whsec_" + base64.b64encode(KEY).decode()
NOW: Final = datetime(2026, 11, 20, 12, tzinfo=timezone.utc)
ROW: Final = SubscriptionRow(
    id="sub_1",
    principal="key:1",
    key_hash="abc",
    server_id="srv-ci",
    event_name="build.failed",
    arguments={"repo": "acme/web"},
    client_url="https://client.example/hook",
    client_secret=SECRET,
    upstream_subscription_id="up_1",
    upstream_secret=SECRET,
    status="active",
)


class _Client:
    def __init__(self, *statuses: int | CallbackFailure) -> None:
        self.statuses = list(statuses) or [200]
        self.posts: list[tuple[bytes, Mapping[str, str]]] = []  # mutable-ok: records deliveries

    async def __call__(
        self, url: str, body: bytes, headers: Mapping[str, str]
    ) -> Result[CallbackResponse, CallbackFailure]:
        self.posts.append((body, headers))
        status: Final = self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]
        return Ok(CallbackResponse(status=status, body=b"")) if isinstance(status, int) else Error(status)

    def bodies(self) -> list[dict[str, object]]:
        return [json.loads(body) for body, _ in self.posts]


@pytest.mark.parametrize("notify", (True, False))
@pytest.mark.asyncio
async def test_terminating_twice_notifies_once_and_always_ends_the_upstream_subscription(notify: bool) -> None:
    from litellm.proxy.mcp_server.events.errors import forbidden
    from litellm.proxy.mcp_server.events.webhooks.termination import terminate

    store: Final = MemoryStore(rows={"sub_1": ROW})
    upstream: Final = FakeUpstream(responses={"events/unsubscribe": {}})
    client: Final = _Client()
    events_runtime: Final = runtime((server(),), upstream=upstream)
    for _ in range(2):
        await terminate(
            ROW, forbidden("Key revoked"), store, SETTINGS, events_runtime, client, lambda _id: server(), notify=notify
        )
    assert (len(client.posts), [method for method, *_ in upstream.calls], store.rows["sub_1"].status) == (
        1 if notify else 0,
        ["events/unsubscribe"],
        "terminated",
    )
