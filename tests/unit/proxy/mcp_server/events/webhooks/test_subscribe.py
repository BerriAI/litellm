import base64
import json
from collections.abc import Mapping
from typing import Final

import pytest

from litellm.proxy.mcp_server.events.access import principal_of
from litellm.proxy.mcp_server.events.config import EventsSettings
from litellm.proxy.mcp_server.events.webhooks.delivery import CallbackResponse
from litellm.proxy.mcp_server.events.webhooks.policy import subscription_id
from litellm.proxy.mcp_server.events.webhooks.signing import parse_secret
from litellm.proxy.mcp_server.events.webhooks.subscribe import (
    _prepare,  # pyright: ignore[reportPrivateUsage]  # the reservation step is the unit under test
    _Request,  # pyright: ignore[reportPrivateUsage]  # see above
    _reserve,  # pyright: ignore[reportPrivateUsage]  # see above
    _validate,
)
from litellm.proxy.mcp_server.outbound_credentials.result import Error, Ok, Result
from litellm.types.mcp_server.events import (
    CallbackFailure,
    EventsErrorCode,
    SubscribeEventsParams,
    WebhookDelivery,
)

from ..fakes import CALLER, SETTINGS, FakeUpstream, context, runtime, server
from .memory_store import MemoryStore

SECRET: Final = "whsec_" + base64.b64encode(bytes(range(32))).decode()
HOOK: Final = "https://client.example/hook"
ROTATED: Final = "whsec_" + base64.b64encode(bytes(range(1, 33))).decode()
ARGS: Final = {"repo": "acme/web"}
SID: Final = subscription_id(principal_of(CALLER) or "", HOOK, "ci-build.failed", ARGS)
UPSTREAM_OK: Final = {"id": "up_1", "refreshBefore": "2099-01-01T00:00:00Z", "cursor": "c0", "truncated": False}


def _request(store: MemoryStore) -> _Request:
    secret: Final = parse_secret(SECRET)
    assert secret is not None
    return _Request(principal=principal_of(CALLER) or "", secret=secret, store=store)


@pytest.fixture(autouse=True)
def salt_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_SALT_KEY", "synthetic-events-subscribe-key")


class _Client:
    def __init__(self, echo: bool = True, failure: CallbackFailure | None = None) -> None:
        self.echo = echo
        self.failure = failure
        self.posts: list[bytes] = []  # mutable-ok: records verification posts

    async def __call__(
        self, url: str, body: bytes, headers: Mapping[str, str]
    ) -> Result[CallbackResponse, CallbackFailure]:
        self.posts.append(body)
        if self.failure is not None:
            return Error(self.failure)
        challenge: Final = json.loads(body)["challenge"] if self.echo else "wrong"
        return Ok(CallbackResponse(status=200, body=json.dumps({"challenge": challenge}).encode()))


def _params(ttl_ms: int | None = 600_000, url: str = HOOK, secret: str = SECRET) -> SubscribeEventsParams:
    return SubscribeEventsParams(
        name="ci-build.failed", arguments=ARGS, delivery=WebhookDelivery(url=url, secret=secret), ttl_ms=ttl_ms
    )


@pytest.mark.asyncio
async def test_a_reservation_past_the_cap_backs_itself_out() -> None:
    capped: Final = EventsSettings(enabled=True, public_base_url="https://gw", max_subscriptions_per_key=1)
    store: Final = MemoryStore()
    upstream: Final = FakeUpstream(responses={"events/subscribe": UPSTREAM_OK})
    prepared: Final = await _prepare(
        context(), _params(), None, capped, runtime((server(),), upstream=upstream), _request(store), _Client()
    )
    assert isinstance(prepared, Ok)
    await store.reserve(prepared.ok.base.model_copy(update={"id": "sub_other"}))
    reserved: Final = await _reserve(prepared.ok, capped)
    assert isinstance(reserved, Error) and reserved.error.code is EventsErrorCode.RESOURCE_EXHAUSTED
    assert set(store.rows) == {"sub_other"}


@pytest.mark.parametrize("field", ("end_user_id", "agent_id"))
def test_a_request_scoped_authority_cannot_open_a_durable_subscription(field: str) -> None:
    caller: Final = CALLER.model_copy(update={field: "request-id"})
    result: Final = _validate(context(caller), _params(), SETTINGS, MemoryStore())
    assert isinstance(result, Error) and result.error.code is EventsErrorCode.FORBIDDEN
