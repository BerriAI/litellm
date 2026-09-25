import asyncio
import json
import random
from collections.abc import Callable, Coroutine, Iterable
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import litellm
from litellm.caching.redis_cache import RedisMessage
from litellm.proxy.common_utils.config_sync_pubsub import (
    _CONFIG_SYNCED_TABLE_NAMES,
    _RESYNC_APPLIED_CONFIG_PARAM_NAMES,
    _WRITE_ACTION_NAMES,
    CONFIG_SYNC_CHANNEL,
    CONFIG_SYNC_JITTER_MAX_SECONDS,
    CONFIG_SYNC_MIN_RESYNC_INTERVAL_SECONDS,
    ConfigSyncSubscriber,
    _PublishOnWriteActions,
    publish_config_change,
    wrap_table_actions_for_config_sync,
)

_EXPECTED_WRITE_ACTION_NAMES = (
    "create",
    "create_many",
    "delete",
    "delete_many",
    "update",
    "update_many",
    "upsert",
)

_EXPECTED_CONFIG_SYNCED_TABLE_NAMES = frozenset(
    {
        "litellm_agentstable",
        "litellm_cacheconfig",
        "litellm_configoverrides",
        "litellm_credentialstable",
        "litellm_guardrailstable",
        "litellm_managedvectorstoreindextable",
        "litellm_managedvectorstorestable",
        "litellm_mcpservertable",
        "litellm_policyattachmenttable",
        "litellm_policytable",
        "litellm_prompttable",
        "litellm_proxymodeltable",
        "litellm_searchtoolstable",
        "litellm_ssoconfig",
        "litellm_uisettings",
    }
)

_EXPECTED_RESYNC_APPLIED_CONFIG_PARAM_NAMES = frozenset(
    {
        "anthropic_beta_headers_reload_config",
        "general_settings",
        "litellm_settings",
        "model_cost_map_reload_config",
        "router_settings",
    }
)

_STARTUP_ONLY_CONFIG_PARAM_NAMES = ("environment_variables",)


class _QueuePubSub:
    """A subscription fed from a queue of payloads, standing in for RedisSubscription."""

    def __init__(self, initial_messages: Iterable[str] = ()) -> None:
        self.queue: asyncio.Queue[str] = asyncio.Queue()
        for message in initial_messages:
            self.queue.put_nowait(message)
        self.subscribed_channels: list[str] = []
        self.closed = False

    async def get_message(self, *, timeout: float | None) -> RedisMessage | None:
        if timeout == 0:
            try:
                return self._message(self.queue.get_nowait())
            except asyncio.QueueEmpty:
                return None
        try:
            return self._message(await asyncio.wait_for(self.queue.get(), timeout))
        except asyncio.TimeoutError:
            return None

    def _message(self, payload: str) -> RedisMessage:
        return RedisMessage(channel=self.subscribed_channels[0], payload=payload.encode())

    async def aclose(self) -> None:
        self.closed = True


class _BrokenPubSub(_QueuePubSub):
    async def get_message(self, *, timeout: float | None) -> RedisMessage | None:
        raise ConnectionError("connection lost")


class _CloseFailingBrokenPubSub(_BrokenPubSub):
    async def aclose(self) -> None:
        raise ConnectionError("close failed")


class _EmptyPollsThenMessagePubSub(_QueuePubSub):
    def __init__(self, empty_polls: int, initial_messages: Iterable[str] = ()) -> None:
        super().__init__(initial_messages=initial_messages)
        self.remaining_empty_polls = empty_polls

    async def get_message(self, *, timeout: float | None) -> RedisMessage | None:
        if timeout != 0 and self.remaining_empty_polls > 0:
            self.remaining_empty_polls -= 1
            return None
        return await super().get_message(timeout=timeout)


class _FakeClock:
    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


class _FakeRedisCache:
    """The slice of RedisCache the module uses: namespace, async_publish and async_subscribe."""

    def __init__(
        self,
        subscriptions: Iterable[_QueuePubSub] = (),
        namespace: str | None = None,
        publish_error: Exception | None = None,
    ) -> None:
        self._subscriptions = iter(subscriptions)
        self._publish_error = publish_error
        self.namespace = namespace
        self.published: list[tuple[str, str]] = []

    async def async_publish(self, channel: str, message: str) -> int:
        if self._publish_error is not None:
            raise self._publish_error
        self.published.append((channel, message))
        return 1

    async def async_subscribe(self, *channels: str) -> _QueuePubSub:
        subscription = next(self._subscriptions)
        subscription.subscribed_channels.extend(channels)
        return subscription


class _ClusterRedisCache(_FakeRedisCache):
    """RedisCache over a cluster client raises NotImplementedError for both pub/sub methods."""

    async def async_publish(self, channel: str, message: str) -> int:
        raise NotImplementedError("Redis Cluster clients have no pub/sub support")

    async def async_subscribe(self, *channels: str) -> _QueuePubSub:
        raise NotImplementedError("Redis Cluster clients have no pub/sub support")


def _recording_callback(
    events: list[str], name: str, fired: asyncio.Event
) -> Callable[[], Coroutine[None, None, None]]:
    async def callback() -> None:
        events.append(name)
        fired.set()

    return callback


async def test_publish_noops_when_redis_cache_is_none() -> None:
    await publish_config_change(redis_cache=None, object_type="litellm_proxymodeltable")


async def test_publish_sends_object_type_json_on_channel() -> None:
    cache = _FakeRedisCache()

    await publish_config_change(redis_cache=cache, object_type="litellm_proxymodeltable")

    assert len(cache.published) == 1
    channel, message = cache.published[0]
    assert channel == "litellm_proxy.config_change"
    assert json.loads(message) == {"object_type": "litellm_proxymodeltable"}


async def test_publish_uses_namespaced_channel() -> None:
    cache = _FakeRedisCache(namespace="prod-eu")

    await publish_config_change(redis_cache=cache, object_type="litellm_credentialstable")

    assert cache.published[0][0] == "prod-eu:litellm_proxy.config_change"


async def test_publish_swallows_redis_publish_errors() -> None:
    cache = _FakeRedisCache(publish_error=ConnectionError("redis down"))

    await publish_config_change(redis_cache=cache, object_type="litellm_proxymodeltable")

    assert cache.published == []


async def test_publish_skips_clients_without_pubsub_support() -> None:
    cache = _ClusterRedisCache()

    await publish_config_change(redis_cache=cache, object_type="litellm_proxymodeltable")

    assert cache.published == []


async def test_subscriber_runs_injected_callbacks_in_order_on_message() -> None:
    pubsub = _QueuePubSub()
    cache = _FakeRedisCache(subscriptions=[pubsub])
    events: list[str] = []
    fired = asyncio.Event()
    subscriber = ConfigSyncSubscriber(
        redis_cache=cache,
        resync_callbacks=(
            _recording_callback(events, "add_deployment", asyncio.Event()),
            _recording_callback(events, "get_credentials", fired),
        ),
        debounce_seconds=0.01,
        jitter_max_seconds=0.0,
    )

    subscriber.start()
    pubsub.queue.put_nowait(json.dumps({"object_type": "litellm_proxymodeltable"}))
    await asyncio.wait_for(fired.wait(), timeout=5)
    await subscriber.stop()

    assert events == ["add_deployment", "get_credentials"]
    assert pubsub.subscribed_channels == [CONFIG_SYNC_CHANNEL]
    assert pubsub.closed is True


async def test_burst_within_debounce_window_coalesces_into_one_resync() -> None:
    burst = [json.dumps({"object_type": "litellm_proxymodeltable"}) for _ in range(5)]
    pubsub = _QueuePubSub(initial_messages=burst)
    cache = _FakeRedisCache(subscriptions=[pubsub])
    resyncs: list[str] = []
    fired = asyncio.Event()
    subscriber = ConfigSyncSubscriber(
        redis_cache=cache,
        resync_callbacks=(_recording_callback(resyncs, "resync", fired),),
        debounce_seconds=0.05,
        jitter_max_seconds=0.0,
    )

    subscriber.start()
    await asyncio.wait_for(fired.wait(), timeout=5)
    await asyncio.sleep(0.3)
    await subscriber.stop()

    assert resyncs == ["resync"]
    assert pubsub.queue.empty()


async def test_subscriber_subscribes_on_namespaced_channel_and_resyncs() -> None:
    pubsub = _QueuePubSub()
    cache = _FakeRedisCache(subscriptions=[pubsub], namespace="prod-eu")
    resyncs: list[str] = []
    fired = asyncio.Event()
    subscriber = ConfigSyncSubscriber(
        redis_cache=cache,
        resync_callbacks=(_recording_callback(resyncs, "resync", fired),),
        debounce_seconds=0.01,
        jitter_max_seconds=0.0,
    )

    subscriber.start()
    pubsub.queue.put_nowait(json.dumps({"object_type": "litellm_proxymodeltable"}))
    await asyncio.wait_for(fired.wait(), timeout=5)
    await subscriber.stop()

    assert pubsub.subscribed_channels == ["prod-eu:litellm_proxy.config_change"]
    assert resyncs == ["resync"]


class _MaxJitterRandom(random.Random):
    def uniform(self, a: float, b: float) -> float:
        return b


async def test_debounce_sleep_adds_jitter_from_injected_rng() -> None:
    pubsub = _QueuePubSub(initial_messages=[json.dumps({"object_type": "litellm_proxymodeltable"})])
    cache = _FakeRedisCache(subscriptions=[pubsub])
    sleeps: list[float] = []
    fired = asyncio.Event()

    async def recording_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    subscriber = ConfigSyncSubscriber(
        redis_cache=cache,
        resync_callbacks=(_recording_callback([], "resync", fired),),
        debounce_seconds=1.0,
        jitter_max_seconds=4.0,
        rng=_MaxJitterRandom(),
        sleep=recording_sleep,
    )

    subscriber.start()
    await asyncio.wait_for(fired.wait(), timeout=5)
    await subscriber.stop()

    assert sleeps == [5.0]


def test_default_jitter_window_is_nonzero() -> None:
    assert CONFIG_SYNC_JITTER_MAX_SECONDS > 0


def test_default_min_resync_interval_caps_reload_rate() -> None:
    assert CONFIG_SYNC_MIN_RESYNC_INTERVAL_SECONDS > CONFIG_SYNC_JITTER_MAX_SECONDS


def _throttled_subscriber(
    cache: object,
    events: list[str],
    fired: asyncio.Event,
    clock: _FakeClock,
    min_resync_interval_seconds: float = 10.0,
) -> ConfigSyncSubscriber:
    async def recording_sleep(seconds: float) -> None:
        events.append(f"sleep:{seconds}")
        await asyncio.sleep(0)

    async def resync() -> None:
        events.append("resync")
        fired.set()

    return ConfigSyncSubscriber(
        redis_cache=cache,
        resync_callbacks=(resync,),
        debounce_seconds=0.0,
        jitter_max_seconds=0.0,
        min_resync_interval_seconds=min_resync_interval_seconds,
        sleep=recording_sleep,
        monotonic=clock,
    )


async def test_resync_arriving_inside_min_interval_waits_out_the_remainder() -> None:
    pubsub = _QueuePubSub()
    cache = _FakeRedisCache(subscriptions=[pubsub])
    events: list[str] = []
    fired = asyncio.Event()
    clock = _FakeClock()
    subscriber = _throttled_subscriber(cache=cache, events=events, fired=fired, clock=clock)

    subscriber.start()
    pubsub.queue.put_nowait("change")
    await asyncio.wait_for(fired.wait(), timeout=5)
    fired.clear()
    clock.now += 4.0
    pubsub.queue.put_nowait("change")
    await asyncio.wait_for(fired.wait(), timeout=5)
    await subscriber.stop()

    assert events == ["sleep:0.0", "resync", "sleep:0.0", "sleep:6.0", "resync"]


async def test_resync_after_min_interval_elapsed_is_not_throttled() -> None:
    pubsub = _QueuePubSub()
    cache = _FakeRedisCache(subscriptions=[pubsub])
    events: list[str] = []
    fired = asyncio.Event()
    clock = _FakeClock()
    subscriber = _throttled_subscriber(cache=cache, events=events, fired=fired, clock=clock)

    subscriber.start()
    pubsub.queue.put_nowait("change")
    await asyncio.wait_for(fired.wait(), timeout=5)
    fired.clear()
    clock.now += 30.0
    pubsub.queue.put_nowait("change")
    await asyncio.wait_for(fired.wait(), timeout=5)
    await subscriber.stop()

    assert events == ["sleep:0.0", "resync", "sleep:0.0", "resync"]


async def test_writes_during_the_throttle_wait_collapse_into_the_next_resync() -> None:
    pubsub = _QueuePubSub()
    cache = _FakeRedisCache(subscriptions=[pubsub])
    events: list[str] = []
    fired = asyncio.Event()
    clock = _FakeClock()
    subscriber = _throttled_subscriber(cache=cache, events=events, fired=fired, clock=clock)

    subscriber.start()
    pubsub.queue.put_nowait("change")
    await asyncio.wait_for(fired.wait(), timeout=5)
    fired.clear()
    for _ in range(5):
        pubsub.queue.put_nowait("change")
    await asyncio.wait_for(fired.wait(), timeout=5)
    await asyncio.sleep(0.1)
    await subscriber.stop()

    assert events.count("resync") == 2
    assert pubsub.queue.empty()


async def test_polls_without_messages_do_not_trigger_resyncs() -> None:
    pubsub = _EmptyPollsThenMessagePubSub(empty_polls=3, initial_messages=["change"])
    cache = _FakeRedisCache(subscriptions=[pubsub])
    resyncs: list[str] = []
    fired = asyncio.Event()
    subscriber = ConfigSyncSubscriber(
        redis_cache=cache,
        resync_callbacks=(_recording_callback(resyncs, "resync", fired),),
        debounce_seconds=0.01,
        jitter_max_seconds=0.0,
    )

    subscriber.start()
    await asyncio.wait_for(fired.wait(), timeout=5)
    await asyncio.sleep(0.1)
    await subscriber.stop()

    assert pubsub.remaining_empty_polls == 0
    assert resyncs == ["resync"]


async def test_failing_pubsub_close_still_reconnects() -> None:
    broken = _CloseFailingBrokenPubSub()
    healthy = _QueuePubSub(initial_messages=["change"])
    cache = _FakeRedisCache(subscriptions=[broken, healthy])
    resyncs: list[str] = []
    fired = asyncio.Event()
    subscriber = ConfigSyncSubscriber(
        redis_cache=cache,
        resync_callbacks=(_recording_callback(resyncs, "resync", fired),),
        debounce_seconds=0.01,
        jitter_max_seconds=0.0,
        backoff_initial_seconds=0.02,
        backoff_max_seconds=0.05,
    )

    subscriber.start()
    await asyncio.wait_for(fired.wait(), timeout=5)
    await subscriber.stop()

    assert healthy.subscribed_channels == [CONFIG_SYNC_CHANNEL]
    assert resyncs == ["resync"]


async def test_second_start_does_not_open_a_second_subscription() -> None:
    pubsub = _QueuePubSub()
    cache = _FakeRedisCache(subscriptions=[pubsub])
    subscriber = ConfigSyncSubscriber(redis_cache=cache, resync_callbacks=(), debounce_seconds=0.01)

    subscriber.start()
    task = subscriber._task
    subscriber.start()
    assert task is not None
    assert subscriber._task is task
    await asyncio.sleep(0.05)
    await subscriber.stop()

    assert pubsub.subscribed_channels == [CONFIG_SYNC_CHANNEL]


async def test_redis_error_leads_to_backoff_and_resubscribe() -> None:
    broken = _BrokenPubSub()
    healthy = _QueuePubSub(initial_messages=[json.dumps({"object_type": "litellm_credentialstable"})])
    cache = _FakeRedisCache(subscriptions=[broken, healthy])
    resyncs: list[str] = []
    fired = asyncio.Event()
    subscriber = ConfigSyncSubscriber(
        redis_cache=cache,
        resync_callbacks=(_recording_callback(resyncs, "resync", fired),),
        debounce_seconds=0.01,
        jitter_max_seconds=0.0,
        backoff_initial_seconds=0.02,
        backoff_max_seconds=0.05,
    )

    subscriber.start()
    await asyncio.wait_for(fired.wait(), timeout=5)
    task = subscriber._task
    assert task is not None
    assert task.done() is False
    await subscriber.stop()

    assert broken.subscribed_channels == [CONFIG_SYNC_CHANNEL]
    assert broken.closed is True
    assert healthy.subscribed_channels == [CONFIG_SYNC_CHANNEL]
    assert resyncs == ["resync"]


async def test_failing_resync_callback_does_not_kill_subscriber() -> None:
    pubsub = _QueuePubSub()
    cache = _FakeRedisCache(subscriptions=[pubsub])
    resyncs: list[str] = []
    fired = asyncio.Event()

    async def failing_callback() -> None:
        raise RuntimeError("resync exploded")

    subscriber = ConfigSyncSubscriber(
        redis_cache=cache,
        resync_callbacks=(failing_callback, _recording_callback(resyncs, "resync", fired)),
        debounce_seconds=0.01,
        jitter_max_seconds=0.0,
        min_resync_interval_seconds=0.0,
    )

    subscriber.start()
    pubsub.queue.put_nowait("change")
    await asyncio.wait_for(fired.wait(), timeout=5)
    fired.clear()
    pubsub.queue.put_nowait("change")
    await asyncio.wait_for(fired.wait(), timeout=5)
    await subscriber.stop()

    assert resyncs == ["resync", "resync"]


async def test_stop_cancels_subscriber_cleanly() -> None:
    pubsub = _QueuePubSub()
    cache = _FakeRedisCache(subscriptions=[pubsub])
    subscriber = ConfigSyncSubscriber(redis_cache=cache, resync_callbacks=(), debounce_seconds=0.01)

    subscriber.start()
    await asyncio.sleep(0.05)
    task = subscriber._task
    assert task is not None
    await subscriber.stop()

    assert task.done() is True
    assert subscriber._task is None
    assert pubsub.closed is True
    await subscriber.stop()


async def test_stop_before_start_is_a_noop() -> None:
    cache = _FakeRedisCache(subscriptions=[])
    subscriber = ConfigSyncSubscriber(redis_cache=cache, resync_callbacks=())

    await subscriber.stop()


async def test_subscriber_exits_without_callbacks_when_client_lacks_pubsub() -> None:
    cache = _ClusterRedisCache()
    resyncs: list[str] = []
    subscriber = ConfigSyncSubscriber(
        redis_cache=cache,
        resync_callbacks=(_recording_callback(resyncs, "resync", asyncio.Event()),),
    )

    subscriber.start()
    task = subscriber._task
    assert task is not None
    await asyncio.wait_for(task, timeout=5)

    assert resyncs == []


class _FakeTableActions:
    def __init__(self, calls: list[tuple[str, str]]) -> None:
        self._calls = calls

    async def create(self, **kwargs: object) -> object:
        self._calls.append(("write", "create"))
        return {"id": "m-1"}

    async def find_many(self, **kwargs: object) -> object:
        self._calls.append(("read", "find_many"))
        return []


class _AllWritesTableActions:
    def __init__(self, calls: list[str]) -> None:
        self._calls = calls

    def __getattr__(self, name: str) -> Callable[..., Coroutine[None, None, str]]:
        async def action(*args: object, **kwargs: object) -> str:
            self._calls.append(name)
            return name

        return action


def _recording_publish(calls: list[tuple[str, str]]) -> Callable[[str], Coroutine[None, None, None]]:
    async def publish(object_type: str) -> None:
        calls.append(("publish", object_type))

    return publish


def test_wrapper_passes_through_unsynced_tables() -> None:
    actions = object()

    wrapped = wrap_table_actions_for_config_sync(actions=actions, table_name="litellm_spendlogs")

    assert wrapped is actions


async def test_wrapper_publishes_table_name_after_write() -> None:
    calls: list[tuple[str, str]] = []
    wrapped = wrap_table_actions_for_config_sync(
        actions=_FakeTableActions(calls),
        table_name="litellm_proxymodeltable",
        publish=_recording_publish(calls),
    )

    result = await wrapped.create(data={"model_name": "gpt-5.2"})

    assert result == {"id": "m-1"}
    assert calls == [("write", "create"), ("publish", "litellm_proxymodeltable")]


async def test_wrapper_does_not_publish_on_reads() -> None:
    calls: list[tuple[str, str]] = []
    wrapped = wrap_table_actions_for_config_sync(
        actions=_FakeTableActions(calls),
        table_name="litellm_proxymodeltable",
        publish=_recording_publish(calls),
    )

    result = await wrapped.find_many(where={})

    assert result == []
    assert calls == [("read", "find_many")]


def test_write_action_names_are_pinned() -> None:
    assert _WRITE_ACTION_NAMES == frozenset(_EXPECTED_WRITE_ACTION_NAMES)


def test_config_synced_table_membership_is_pinned() -> None:
    assert _CONFIG_SYNCED_TABLE_NAMES == _EXPECTED_CONFIG_SYNCED_TABLE_NAMES


def test_tool_telemetry_table_writes_pass_through_unwrapped() -> None:
    actions = object()

    wrapped = wrap_table_actions_for_config_sync(actions=actions, table_name="litellm_tooltable")

    assert wrapped is actions


@pytest.mark.parametrize("action_name", _EXPECTED_WRITE_ACTION_NAMES)
async def test_wrapper_publishes_for_every_write_action(action_name: str) -> None:
    write_calls: list[str] = []
    publish_calls: list[tuple[str, str]] = []
    wrapped = wrap_table_actions_for_config_sync(
        actions=_AllWritesTableActions(write_calls),
        table_name="litellm_guardrailstable",
        publish=_recording_publish(publish_calls),
    )

    result = await getattr(wrapped, action_name)(data={})

    assert result == action_name
    assert write_calls == [action_name]
    assert publish_calls == [("publish", "litellm_guardrailstable")]


async def test_model_repository_write_publishes_via_live_coordination_cache() -> None:
    from litellm.proxy import proxy_server
    from litellm.proxy.proxy_server import _set_redis_usage_cache
    from litellm.repositories.model_repository import ModelRepository

    client = _FakeRedisCache()
    prisma_client = MagicMock()
    prisma_client.db.litellm_proxymodeltable.update = AsyncMock(return_value={"model_id": "m-1"})
    repository = ModelRepository(prisma_client)
    table = repository.table
    assert isinstance(table, _PublishOnWriteActions)

    previous_cache = proxy_server.redis_usage_cache
    _set_redis_usage_cache(client)
    try:
        await table.update(where={"model_id": "m-1"}, data={"model_name": "gpt-5.2"})
    finally:
        _set_redis_usage_cache(previous_cache)

    prisma_client.db.litellm_proxymodeltable.update.assert_awaited_once_with(
        where={"model_id": "m-1"}, data={"model_name": "gpt-5.2"}
    )
    assert len(client.published) == 1
    channel, message = client.published[0]
    assert channel == CONFIG_SYNC_CHANNEL
    assert json.loads(message) == {"object_type": "litellm_proxymodeltable"}


async def test_ui_settings_write_publishes_via_live_coordination_cache() -> None:
    from litellm.proxy import proxy_server
    from litellm.proxy.proxy_server import _set_redis_usage_cache
    from litellm.repositories.table_repositories import UISettingsRepository

    client = _FakeRedisCache()
    prisma_client = MagicMock()
    prisma_client.db.litellm_uisettings.upsert = AsyncMock(return_value={"id": "ui_settings"})
    table = UISettingsRepository(prisma_client).table
    assert isinstance(table, _PublishOnWriteActions)

    previous_cache = proxy_server.redis_usage_cache
    _set_redis_usage_cache(client)
    try:
        await table.upsert(
            where={"id": "ui_settings"},
            data={"create": {"id": "ui_settings"}, "update": {"ui_settings": "{}"}},
        )
    finally:
        _set_redis_usage_cache(previous_cache)

    assert len(client.published) == 1
    channel, message = client.published[0]
    assert channel == CONFIG_SYNC_CHANNEL
    assert json.loads(message) == {"object_type": "litellm_uisettings"}


async def _publish_calls_for_invalidated_param(param_name: str) -> list[tuple[str, str]]:
    from litellm.proxy import proxy_server
    from litellm.proxy.proxy_server import _set_redis_usage_cache
    from litellm.proxy.utils import invalidate_config_param

    client = _FakeRedisCache()
    previous_cache = proxy_server.redis_usage_cache
    _set_redis_usage_cache(client)
    try:
        await invalidate_config_param(param_name)
    finally:
        _set_redis_usage_cache(previous_cache)
    return client.published


@pytest.mark.parametrize("param_name", sorted(_EXPECTED_RESYNC_APPLIED_CONFIG_PARAM_NAMES))
async def test_invalidate_config_param_publishes_params_a_resync_applies(param_name: str) -> None:
    published = await _publish_calls_for_invalidated_param(param_name)

    assert len(published) == 1
    channel, message = published[0]
    assert channel == CONFIG_SYNC_CHANNEL
    assert json.loads(message) == {"object_type": param_name}


@pytest.mark.parametrize("param_name", _STARTUP_ONLY_CONFIG_PARAM_NAMES)
async def test_invalidate_config_param_does_not_publish_startup_only_params(param_name: str) -> None:
    published = await _publish_calls_for_invalidated_param(param_name)

    assert published == []


def test_resync_applied_config_param_membership_is_pinned() -> None:
    assert _RESYNC_APPLIED_CONFIG_PARAM_NAMES == _EXPECTED_RESYNC_APPLIED_CONFIG_PARAM_NAMES


async def test_evict_config_param_does_not_publish() -> None:
    from litellm.proxy import proxy_server
    from litellm.proxy.proxy_server import _set_redis_usage_cache
    from litellm.proxy.utils import evict_config_param

    client = _FakeRedisCache()
    previous_cache = proxy_server.redis_usage_cache
    _set_redis_usage_cache(client)
    try:
        await evict_config_param("model_cost_map_reload_config")
    finally:
        _set_redis_usage_cache(previous_cache)

    assert client.published == []


def _reload_config_prisma_client() -> MagicMock:
    config_record = MagicMock()
    config_record.param_value = {"interval_hours": 6, "force_reload": True}
    config_record.reload_revision = 0
    config_record.last_run_at = None
    prisma_client = MagicMock()
    prisma_client.get_generic_data = AsyncMock(return_value=config_record)
    prisma_client.db.litellm_config.find_unique = AsyncMock(return_value=config_record)
    prisma_client.db.litellm_config.upsert = AsyncMock(return_value=config_record)
    prisma_client.db.litellm_config.update_many = AsyncMock(return_value=1)
    return prisma_client


async def test_model_cost_map_reload_does_not_publish_config_change() -> None:
    from litellm.proxy import proxy_server
    from litellm.proxy.proxy_server import ProxyConfig, _set_redis_usage_cache
    from litellm.proxy.utils import litellm_config_cache
    from litellm.utils import _invalidate_model_cost_lowercase_map

    litellm_config_cache.flush_cache()
    prisma_client = _reload_config_prisma_client()
    client = _FakeRedisCache()
    previous_cache = proxy_server.redis_usage_cache
    original_model_cost = litellm.model_cost.copy()
    _set_redis_usage_cache(client)
    try:
        from litellm.litellm_core_utils.get_model_cost_map import ModelCostMapReloaded

        with patch(
            "litellm.litellm_core_utils.get_model_cost_map.refetch_model_cost_map",
            new=AsyncMock(
                return_value=ModelCostMapReloaded(model_cost_map={"gpt-5.2": {"input_cost_per_token": 0.001}})
            ),
        ):
            await ProxyConfig()._check_and_reload_model_cost_map(prisma_client=prisma_client)
    finally:
        litellm.model_cost = original_model_cost
        _invalidate_model_cost_lowercase_map()
        _set_redis_usage_cache(previous_cache)

    prisma_client.db.litellm_config.update_many.assert_awaited_once()
    assert client.published == []


async def test_anthropic_beta_headers_reload_does_not_publish_config_change() -> None:
    from litellm.proxy import proxy_server
    from litellm.proxy.proxy_server import ProxyConfig, _set_redis_usage_cache
    from litellm.proxy.utils import litellm_config_cache

    litellm_config_cache.flush_cache()
    prisma_client = _reload_config_prisma_client()
    client = _FakeRedisCache()
    previous_cache = proxy_server.redis_usage_cache
    _set_redis_usage_cache(client)
    try:
        with patch("litellm.anthropic_beta_headers_manager.reload_beta_headers_config") as mock_reload:
            mock_reload.return_value = {}
            await ProxyConfig()._check_and_reload_anthropic_beta_headers(prisma_client=prisma_client)
    finally:
        _set_redis_usage_cache(previous_cache)

    prisma_client.db.litellm_config.upsert.assert_awaited_once()
    assert client.published == []


class _StopFailingSubscriber(ConfigSyncSubscriber):
    async def stop(self) -> None:
        raise RuntimeError("stop failed")


async def test_proxy_config_subscriber_resyncs_deployments_only() -> None:
    from litellm.proxy.proxy_server import ProxyConfig

    cache = _FakeRedisCache(subscriptions=[_QueuePubSub()])
    config = ProxyConfig()
    prisma_client = MagicMock()
    proxy_logging_obj = MagicMock()
    calls: list[tuple[str, object, object]] = []

    async def fake_add_deployment(prisma_client: object, proxy_logging_obj: object) -> None:
        calls.append(("add_deployment", prisma_client, proxy_logging_obj))

    async def fake_get_credentials(prisma_client: object) -> None:
        calls.append(("get_credentials", prisma_client, None))

    config.add_deployment = fake_add_deployment
    config.get_credentials = fake_get_credentials
    config.start_config_sync_subscriber(
        prisma_client=prisma_client,
        proxy_logging_obj=proxy_logging_obj,
        redis_cache=cache,
    )
    subscriber = config.config_sync_subscriber
    assert subscriber is not None
    for callback in subscriber._resync_callbacks:
        await callback()
    await config.stop_config_sync_subscriber()

    assert calls == [("add_deployment", prisma_client, proxy_logging_obj)]
    assert config.config_sync_subscriber is None
    assert subscriber._task is None


async def test_proxy_config_does_not_start_subscriber_without_coordination_redis() -> None:
    from litellm.proxy.proxy_server import ProxyConfig

    config = ProxyConfig()

    config.start_config_sync_subscriber(
        prisma_client=MagicMock(),
        proxy_logging_obj=MagicMock(),
        redis_cache=None,
    )

    assert config.config_sync_subscriber is None


async def test_proxy_config_keeps_the_first_subscriber_on_repeat_start() -> None:
    from litellm.proxy.proxy_server import ProxyConfig

    cache = _FakeRedisCache(subscriptions=[_QueuePubSub()])
    config = ProxyConfig()

    config.start_config_sync_subscriber(prisma_client=MagicMock(), proxy_logging_obj=MagicMock(), redis_cache=cache)
    first = config.config_sync_subscriber
    config.start_config_sync_subscriber(prisma_client=MagicMock(), proxy_logging_obj=MagicMock(), redis_cache=cache)
    second = config.config_sync_subscriber
    await config.stop_config_sync_subscriber()

    assert first is not None
    assert second is first


async def test_proxy_config_shutdown_survives_a_failing_subscriber_stop() -> None:
    from litellm.proxy.proxy_server import ProxyConfig

    config = ProxyConfig()
    config.config_sync_subscriber = _StopFailingSubscriber(
        redis_cache=_FakeRedisCache(subscriptions=[]),
        resync_callbacks=(),
    )

    await config.stop_config_sync_subscriber()

    assert config.config_sync_subscriber is None
