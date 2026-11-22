import asyncio
from dataclasses import replace
from typing import Final

import pytest

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.mcp_server.events.access import EventGrant
from litellm.proxy.mcp_server.events.catalog import (
    event_servers,
    list_events,
    offered_modes,
    prepare_event,
    resolve,
)
from litellm.proxy.mcp_server.events.config import EventsSettings
from litellm.proxy.mcp_server.events.runtime import DescriptorCache, EventsRuntime
from litellm.proxy.mcp_server.outbound_credentials.result import Error, Ok
from litellm.proxy.mcp_server.subscriptions import SubscriptionTarget
from litellm.types.mcp_server.events import EventsErrorCode, EventType
from litellm.types.mcp_server.mcp_server_manager import MCPServer, MCPServerEventsConfig

from .fakes import BUILD_FAILED, CALLER, JWT, SETTINGS, admitted_subject, context, runtime, server

DEPLOY: Final = EventType(name="deploy.done", delivery=("poll",))
BAD_HEADER: Final = EventType(
    name="bad.header",
    delivery=("poll",),
    input_schema={"type": "object", "properties": {"tags": {"type": "array", "x-mcp-header": "Tags"}}},
)


def test_only_enabled_modern_servers_publish_events() -> None:
    servers: Final = (
        server("a", "a"),
        server("b", "b", MCPServerEventsConfig(enabled=False)),
        server("c", "c", protocol_version="2025-11-25"),
    )
    assert tuple(item.server.server_id for item in event_servers(servers)) == ("a",)


def test_resolution_picks_the_longest_matching_prefix() -> None:
    servers: Final = event_servers((server("short", "ci"), server("long", "ci-prod")))
    resolved: Final = resolve("ci-prod-build.failed", servers)
    assert resolved is not None and (resolved.target.server.server_id, resolved.upstream_name) == (
        "long",
        "build.failed",
    )
    assert resolve("unknown-build.failed", servers) is None


@pytest.mark.parametrize(
    ("settings", "caller", "modes"),
    (
        (SETTINGS, CALLER, ("poll", "push", "webhook")),
        (EventsSettings(enabled=True), CALLER, ("poll", "push")),
        (SETTINGS, UserAPIKeyAuth(), ("poll", "push")),
        (SETTINGS, UserAPIKeyAuth(api_key=JWT, user_id="u"), ("poll", "push")),
        (SETTINGS, admitted_subject("u"), ("poll", "push", "webhook")),
    ),
)
def test_webhook_is_offered_only_with_a_public_url_and_an_authenticated_caller(
    settings: EventsSettings, caller: UserAPIKeyAuth, modes: tuple[str, ...]
) -> None:
    assert offered_modes(BUILD_FAILED, settings, context(caller)) == modes


@pytest.mark.asyncio
async def test_listing_prefixes_names_and_drops_events_the_server_or_caller_cannot_use() -> None:
    servers: Final = (server("srv", "ci", MCPServerEventsConfig(enabled=True, allowed_events=("build.*", "bad.*"))),)
    events: Final = await list_events(
        context(), SETTINGS, runtime(servers, (BUILD_FAILED, DEPLOY, BAD_HEADER), grant=(frozenset({"build.failed"}),))
    )
    assert [(item.name, item.delivery) for item in events] == [("ci-build.failed", ("poll", "push", "webhook"))]
    assert events[0].input_schema == BUILD_FAILED.input_schema


@pytest.mark.asyncio
async def test_a_failing_upstream_does_not_hide_other_servers() -> None:
    healthy: Final = server("ok", "ok")
    broken: Final = server("bad", "bad")
    base: Final = runtime((healthy, broken))

    async def listing(target: object) -> tuple[EventType, ...]:
        if getattr(getattr(target, "server", None), "server_id", None) == "bad":
            raise RuntimeError("upstream down")
        return (BUILD_FAILED,)

    patched: Final = EventsRuntime(
        clock=base.clock,
        allowed_servers=base.allowed_servers,
        grant=base.grant,
        prepare_target=base.prepare_target,
        list_upstream=listing,  # pyright: ignore[reportArgumentType]  # test double
        rate_limited=base.rate_limited,
        call=base.call,
        stream=base.stream,
        screen=base.screen,
        record=base.record,
    )
    assert [item.name for item in await list_events(context(), SETTINGS, patched)] == ["ok-build.failed"]


@pytest.mark.parametrize(
    ("name", "servers", "grant"),
    (
        ("other-build.failed", (server(),), None),
        ("ci-missing.event", (server(),), None),
        ("ci-build.failed", (server(),), (frozenset({"deploy.done"}),)),
        ("ci-build.failed", (server(events=MCPServerEventsConfig(enabled=True, allowed_events=("deploy.*",))),), None),
        ("ci-build.failed", (), None),
    ),
)
@pytest.mark.asyncio
async def test_every_unavailable_event_looks_the_same(
    name: str, servers: tuple[MCPServer, ...], grant: EventGrant
) -> None:
    prepared: Final = await prepare_event(context(), name, runtime(servers, grant=grant))
    assert isinstance(prepared, Error)
    assert (prepared.error.code, prepared.error.data) == (EventsErrorCode.NOT_FOUND, {"kind": "event"})


@pytest.mark.parametrize("grant", (None, (frozenset({"build.failed"}),), (frozenset({"build.*"}), frozenset({"*"}))))
@pytest.mark.asyncio
async def test_a_granted_event_is_prepared(grant: EventGrant) -> None:
    prepared: Final = await prepare_event(context(), "ci-build.failed", runtime((server(),), grant=grant))
    assert isinstance(prepared, Ok)


@pytest.mark.asyncio
async def test_server_catalogs_are_fetched_concurrently_under_one_deadline() -> None:
    second_started: Final = asyncio.Event()
    deploy: Final = BUILD_FAILED.model_copy(update={"name": "deploy.done"})

    async def listing(target: SubscriptionTarget) -> tuple[EventType, ...]:
        if target.server.server_id == "srv-b":
            second_started.set()
            return (deploy,)
        await asyncio.wait_for(second_started.wait(), timeout=5)
        return (BUILD_FAILED,)

    servers: Final = (server(), server(server_id="srv-b", alias="cd"))
    events_runtime: Final = replace(runtime(servers), list_upstream=listing, descriptor_cache=DescriptorCache())
    listed: Final = await list_events(context(), SETTINGS, events_runtime, deadline=1.0)
    assert [item.name for item in listed] == ["cd-deploy.done", "ci-build.failed"]


@pytest.mark.asyncio
async def test_a_granted_event_prepares_with_its_upstream_descriptor() -> None:
    prepared: Final = await prepare_event(context(), "ci-build.failed", runtime((server(),)))
    assert isinstance(prepared, Ok)
    assert (prepared.ok.resolved.upstream_name, prepared.ok.descriptor) == ("build.failed", BUILD_FAILED)


@pytest.mark.asyncio
async def test_a_rate_limited_server_is_not_asked_for_its_catalog() -> None:
    asked: Final[list[str]] = []  # mutable-ok: records upstream listings

    async def listing(target: SubscriptionTarget) -> tuple[EventType, ...]:
        asked.append(target.server.server_id)
        return (BUILD_FAILED,)

    limited: Final = replace(
        runtime((server(),), rate_limited=True), list_upstream=listing, descriptor_cache=DescriptorCache()
    )
    assert (await list_events(context(), SETTINGS, limited), asked) == ((), [])
