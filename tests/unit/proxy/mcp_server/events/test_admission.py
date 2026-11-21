from collections.abc import Mapping
from typing import Final

import pytest

from litellm.proxy.mcp_server.events.access import EventGrant
from litellm.proxy.mcp_server.events.admission import admit, check_routing_headers, still_authorized
from litellm.proxy.mcp_server.events.catalog import prepare_event
from litellm.proxy.mcp_server.events.config import EventsSettings
from litellm.proxy.mcp_server.outbound_credentials.result import Error, Ok
from litellm.types.mcp_server.events import EventsErrorCode, EventType, JsonObject
from litellm.types.mcp_server.mcp_server_manager import MCPServerEventsConfig

from .fakes import SETTINGS, context, runtime, server

ARGS: Final[JsonObject] = {"repo": "acme/web"}


@pytest.mark.parametrize(
    ("headers", "accepted"),
    (
        (None, True),
        ({"mcp-name": "ci-build.failed", "mcp-param-repo": "acme/web"}, True),
        ({"mcp-name": "build.failed", "mcp-param-repo": "acme/web"}, False),
        ({"mcp-param-repo": "acme/web"}, False),
        ({"mcp-name": "ci-build.failed"}, False),
        ({"mcp-name": "ci-build.failed", "mcp-param-repo": "acme/api"}, False),
    ),
)
@pytest.mark.asyncio
async def test_routing_headers_must_match_the_body(headers: Mapping[str, str] | None, accepted: bool) -> None:
    prepared: Final = await prepare_event(context(), "ci-build.failed", runtime((server(),)))
    assert isinstance(prepared, Ok)
    failure: Final = check_routing_headers(headers, "ci-build.failed", ARGS, prepared.ok)
    assert (failure is None) is accepted
    assert failure is None or failure.code is EventsErrorCode.HEADER_MISMATCH


@pytest.mark.asyncio
async def test_a_mode_the_event_does_not_offer_is_unsupported() -> None:
    poll_only: Final = EventType(name="deploy.done", delivery=("poll",))
    admitted: Final = await admit(
        context(), "ci-deploy.done", {}, "push", None, SETTINGS, runtime((server(),), (poll_only,))
    )
    assert isinstance(admitted, Error)
    assert (admitted.error.code, admitted.error.data) == (
        EventsErrorCode.UNSUPPORTED,
        {"feature": "deliveryMode", "value": "push"},
    )


@pytest.mark.asyncio
async def test_webhook_without_a_public_url_is_unsupported() -> None:
    admitted: Final = await admit(
        context(), "ci-build.failed", ARGS, "webhook", None, EventsSettings(enabled=True), runtime((server(),))
    )
    assert isinstance(admitted, Error) and admitted.error.code is EventsErrorCode.UNSUPPORTED


@pytest.mark.asyncio
async def test_the_server_rate_limit_applies_to_events() -> None:
    admitted: Final = await admit(
        context(), "ci-build.failed", ARGS, "poll", None, SETTINGS, runtime((server(),), rate_limited=True)
    )
    assert isinstance(admitted, Error) and admitted.error.code is EventsErrorCode.RESOURCE_EXHAUSTED


@pytest.mark.parametrize(
    ("servers", "grant", "authorized"),
    (
        ((server(),), None, True),
        ((), None, False),
        ((server(),), (frozenset({"deploy.done"}),), False),
        ((server(),), (frozenset({"build.*"}),), True),
        ((server(events=MCPServerEventsConfig(enabled=True, allowed_events=("deploy.*",))),), None, False),
    ),
)
@pytest.mark.asyncio
async def test_reauthorization_follows_server_access_and_event_grants(
    servers: tuple[object, ...], grant: EventGrant, authorized: bool
) -> None:
    prepared: Final = await prepare_event(context(), "ci-build.failed", runtime((server(),)))
    assert isinstance(prepared, Ok)
    assert await still_authorized(context(), prepared.ok, runtime(servers, grant=grant)) is authorized  # pyright: ignore[reportArgumentType]  # servers are MCPServer


@pytest.mark.parametrize("name", ("ci-unknown.event", "ci-build.failed"))
@pytest.mark.asyncio
async def test_a_wrong_mcp_name_is_a_header_mismatch_even_for_an_unknown_event(name: str) -> None:
    admitted: Final = await admit(
        context(), name, ARGS, "poll", {"mcp-name": "something-else"}, SETTINGS, runtime((server(),))
    )
    assert isinstance(admitted, Error) and admitted.error.code is EventsErrorCode.HEADER_MISMATCH
