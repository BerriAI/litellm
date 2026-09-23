import asyncio
from collections.abc import Iterator, Mapping
from typing import Final

import pytest

import litellm
from litellm.caching import caching_handler
from litellm.caching.caching import Cache
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.integrations.custom_logger import CustomLogger
from litellm.rust_bridge import catalog
from litellm.rust_bridge.catalog import CacheRule, Route, RouteRule
from litellm.rust_bridge.configuration import Rollout
from litellm.types.caching import LiteLLMCacheType
from tests.test_litellm_rust.support.callback_recorder import RecordingLogger, drain_logging
from tests.test_litellm_rust.support.isolation import rebound
from tests.test_litellm_rust.support.recording_server import RecordingServer, ResponseSpec
from tests.test_litellm_rust.support.requests import (
    MESSAGES,
    MESSAGES_EVENTS,
    MESSAGES_MODEL,
    MESSAGES_RESPONSE,
)

pytestmark = pytest.mark.requires_rust_extension

STREAM: Final = ResponseSpec(body=None, events=MESSAGES_EVENTS)

NATIVE_CACHE_RULES: Final = (
    RouteRule(Route.MESSAGES, Rollout.RUST_OPT_IN),
    CacheRule(Rollout.RUST_REQUIRED, backends=frozenset({"local"})),
    *catalog.RULES,
)
ROUTE_ONLY_RULES: Final = (
    RouteRule(Route.MESSAGES, Rollout.RUST_OPT_IN),
    *catalog.RULES,
)


@pytest.fixture
def native_cache_rules() -> Iterator[None]:
    with rebound(catalog, "RULES", NATIVE_CACHE_RULES):
        yield


@pytest.fixture
def route_only_rules() -> Iterator[None]:
    with rebound(catalog, "RULES", ROUTE_ONLY_RULES):
        yield


@pytest.fixture
def local_cache(native_cache_rules: None) -> Iterator[Cache]:
    facade: Final = Cache(type=LiteLLMCacheType.LOCAL)
    with rebound(litellm, "cache", facade):
        yield facade


@pytest.fixture
def messages_server(recording_server: RecordingServer) -> RecordingServer:
    recording_server.default_response = ResponseSpec(body=MESSAGES_RESPONSE)
    return recording_server


def arguments(server: RecordingServer, **kwargs: object) -> dict[str, object]:
    return {
        "model": MESSAGES_MODEL,
        "messages": [dict(message) for message in MESSAGES],
        "max_tokens": 64,
        "api_key": "test-key",
        "api_base": server.base_url,
        **kwargs,
    }


async def await_pending_writes() -> None:
    for _ in range(200):
        pending: Final = tuple(caching_handler._PENDING_CACHE_WRITES)
        if not pending:
            return
        await asyncio.gather(*pending, return_exceptions=True)
        await asyncio.sleep(0.05)
    pytest.fail("pending cache writes never settled")


class DeploymentHookCounter(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    async def async_post_call_success_deployment_hook(self, request_data, response_obj, call_type=None):
        self.calls += 1


@pytest.mark.asyncio
async def test_a_second_identical_call_is_served_from_the_native_response_cache(
    messages_server: RecordingServer, local_cache: Cache
) -> None:
    recorder: Final = RecordingLogger()
    hook_counter: Final = DeploymentHookCounter()
    key: Final = "messages-native-cache"

    with rebound(litellm, "callbacks", [hook_counter]):
        first: Final = await litellm.anthropic.messages.acreate(
            **arguments(messages_server, cache_key=key, callbacks=[recorder])
        )
        await await_pending_writes()
        second: Final = await litellm.anthropic.messages.acreate(
            **arguments(messages_server, cache_key=key, callbacks=[recorder])
        )
        await drain_logging()

    assert len(messages_server.requests) == 1
    assert second == first
    assert local_cache.cache.get_cache(key) is None
    stored: Final = await local_cache.async_get_cache(cache_key=key)
    assert isinstance(stored, Mapping)
    assert stored["content"] == MESSAGES_RESPONSE["content"]
    assert len(recorder.wait_for("log_pre_api_call")) == 1
    success: Final = await recorder.wait_for_async("async_log_success_event", count=2)
    assert len(success) == 2
    assert success[0].kwargs.get("cache_hit") is not True
    assert success[1].kwargs.get("cache_hit") is True
    assert hook_counter.calls == 1


@pytest.mark.asyncio
async def test_a_malformed_cached_message_is_a_miss_for_callbacks(
    messages_server: RecordingServer, local_cache: Cache
) -> None:
    key: Final = "malformed-messages-cache"
    local_cache.add_cache({"id": "incomplete"}, cache_key=key)
    recorder: Final = RecordingLogger()

    response: Final = await litellm.anthropic.messages.acreate(
        **arguments(messages_server, cache_key=key, callbacks=[recorder])
    )
    await drain_logging()

    assert response["content"] == MESSAGES_RESPONSE["content"]
    assert len(messages_server.requests) == 1
    success: Final = await recorder.wait_for_async("async_log_success_event", count=1)
    assert len(success) == 1
    assert success[0].kwargs.get("cache_hit") is not True


@pytest.mark.asyncio
async def test_a_python_selected_cache_stores_in_the_python_backend(
    messages_server: RecordingServer, route_only_rules: None
) -> None:
    facade: Final = Cache(type=LiteLLMCacheType.LOCAL)
    with rebound(litellm, "cache", facade):
        key: Final = "messages-python-cache"
        await litellm.anthropic.messages.acreate(**arguments(messages_server, cache_key=key))
        await await_pending_writes()
        second: Final = await litellm.anthropic.messages.acreate(**arguments(messages_server, cache_key=key))

        assert len(messages_server.requests) == 1
        assert second["content"] == MESSAGES_RESPONSE["content"]
        assert facade.cache.get_cache(key) is not None


@pytest.mark.asyncio
async def test_a_swapped_backend_never_serves_the_stale_native_store(
    messages_server: RecordingServer, local_cache: Cache
) -> None:
    await litellm.anthropic.messages.acreate(**arguments(messages_server, cache_key="first-key"))
    await await_pending_writes()
    stale: Final = local_cache._native_cache
    assert stale is not None

    messages_server.expected_requests = 2
    local_cache.cache = InMemoryCache()
    await litellm.anthropic.messages.acreate(**arguments(messages_server, cache_key="second-key"))
    await await_pending_writes()

    assert len(messages_server.requests) == 2
    assert local_cache.cache.get_cache("second-key") is not None
    assert local_cache.get_cache(cache_key="second-key") is not None
    second: Final = stale.request(local_cache, {"cache_key": "second-key"})
    assert second is not None
    assert await stale.async_lookup(second) is None
    first: Final = stale.request(local_cache, {"cache_key": "first-key"})
    assert first is not None
    assert await stale.async_lookup(first) is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("control", [{"no-cache": True}, {"no-store": True}])
async def test_cache_controls_disable_the_native_lookup_or_store(
    messages_server: RecordingServer, local_cache: Cache, control: dict[str, bool]
) -> None:
    messages_server.expected_requests = 2
    await litellm.anthropic.messages.acreate(**arguments(messages_server, cache=control))
    await await_pending_writes()
    await litellm.anthropic.messages.acreate(**arguments(messages_server, cache=control))
    await await_pending_writes()

    assert len(messages_server.requests) == 2


@pytest.mark.asyncio
async def test_an_unsupported_call_type_skips_the_response_cache(
    messages_server: RecordingServer, local_cache: Cache
) -> None:
    supported: Final = tuple(
        call_type
        for call_type in local_cache.supported_call_types
        if call_type not in {"aanthropic_messages", "anthropic_messages"}
    )
    with rebound(local_cache, "supported_call_types", supported):
        messages_server.expected_requests = 2
        await litellm.anthropic.messages.acreate(**arguments(messages_server))
        await await_pending_writes()
        await litellm.anthropic.messages.acreate(**arguments(messages_server))

    assert len(messages_server.requests) == 2


@pytest.mark.asyncio
async def test_streamed_calls_bypass_the_response_cache(recording_server: RecordingServer, local_cache: Cache) -> None:
    recording_server.default_response = STREAM
    recording_server.expected_requests = 2
    stream: Final = dict(arguments(recording_server, stream=True))

    for _ in range(2):
        response: Final = await litellm.anthropic.messages.acreate(**stream)
        async for _chunk in response:
            pass
    await await_pending_writes()

    assert len(recording_server.requests) == 2


def test_synchronous_calls_share_the_selected_response_cache(
    messages_server: RecordingServer, local_cache: Cache
) -> None:
    kwargs: Final = arguments(messages_server, cache_key="messages-sync-cache")

    first: Final = litellm.anthropic.messages.create(**kwargs)
    second: Final = litellm.anthropic.messages.create(**kwargs)

    assert len(messages_server.requests) == 1
    assert second == first
