import asyncio
import json
import random
from typing import Callable, Coroutine, Iterable, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel
from redis.asyncio import Redis

import litellm
from litellm.proxy.common_utils.config_sync_pubsub import (
    CONFIG_SYNC_CHANNEL,
    CONFIG_SYNC_JITTER_MAX_SECONDS,
    CONFIG_SYNC_MIN_RESYNC_INTERVAL_SECONDS,
    ConfigSyncSubscriber,
    _CONFIG_SYNCED_TABLE_NAMES,
    _PublishOnWriteActions,
    _RESYNC_APPLIED_CONFIG_PARAM_NAMES,
    _WRITE_ACTION_NAMES,
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


class _RecordingRedisClient(Redis):
    def __init__(self) -> None:
        self.published: List[Tuple[str, str]] = []

    async def publish(self, channel: str, message: str) -> int:
        self.published.append((channel, message))
        return 1


class _FailingPublishRedisClient(Redis):
    def __init__(self) -> None:
        pass

    async def publish(self, channel: str, message: str) -> int:
        raise ConnectionError("redis down")


class _ScriptedPubSubClient:
    """Pub/sub-capable client that is not a redis.asyncio.Redis.

    Mirrors what RedisCache.init_pubsub_client returns for a cluster backend:
    a node-level client exposing publish/pubsub without being an instance of
    the standalone Redis class.
    """

    def __init__(self, pubsubs: Iterable["_QueuePubSub"]) -> None:
        self._scripted_pubsubs = iter(pubsubs)
        self.published: List[Tuple[str, str]] = []

    async def publish(self, channel: str, message: str) -> int:
        self.published.append((channel, message))
        return 1

    def pubsub(self) -> "_QueuePubSub":
        return next(self._scripted_pubsubs)


class _QueuePubSub:
    def __init__(self, initial_messages: Iterable[object] = ()) -> None:
        self.queue: "asyncio.Queue[object]" = asyncio.Queue()
        for message in initial_messages:
            self.queue.put_nowait(message)
        self.subscribed_channels: List[str] = []
        self.closed = False

    async def subscribe(self, *channels: str) -> None:
        self.subscribed_channels.extend(channels)

    async def get_message(self, *, ignore_subscribe_messages: bool, timeout: float) -> Optional[object]:
        if timeout == 0:
            try:
                return self.queue.get_nowait()
            except asyncio.QueueEmpty:
                return None
        try:
            return await asyncio.wait_for(self.queue.get(), timeout)
        except asyncio.TimeoutError:
            return None

    async def aclose(self) -> None:
        self.closed = True


class _BrokenPubSub(_QueuePubSub):
    async def get_message(self, *, ignore_subscribe_messages: bool, timeout: float) -> Optional[object]:
        raise ConnectionError("connection lost")


class _CloseFailingBrokenPubSub(_BrokenPubSub):
    async def aclose(self) -> None:
        raise ConnectionError("close failed")


class _EmptyPollsThenMessagePubSub(_QueuePubSub):
    def __init__(self, empty_polls: int, initial_messages: Iterable[object] = ()) -> None:
        super().__init__(initial_messages=initial_messages)
        self.remaining_empty_polls = empty_polls

    async def get_message(self, *, ignore_subscribe_messages: bool, timeout: float) -> Optional[object]:
        if timeout != 0 and self.remaining_empty_polls > 0:
            self.remaining_empty_polls -= 1
            return None
        return await super().get_message(ignore_subscribe_messages=ignore_subscribe_messages, timeout=timeout)


class _FakeClock:
    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


class _VirtualTimePubSub(_QueuePubSub):
    """Timed polls move the injected clock forward instead of waiting on the wall clock.

    A delivery scheduled at a clock time is returned by the first poll whose window
    reaches it, with the clock set to that time, so a test asserts when each resync
    and targeted apply ran in subscriber time.
    """

    def __init__(self, clock: _FakeClock, deliveries: Iterable[Tuple[float, object]] = ()) -> None:
        super().__init__()
        self._clock = clock
        self.pending: List[Tuple[float, object]] = sorted(deliveries, key=lambda delivery: delivery[0])

    async def get_message(self, *, ignore_subscribe_messages: bool, timeout: float) -> Optional[object]:
        await asyncio.sleep(0)
        due_by = self._clock.now + timeout
        if self.pending and self.pending[0][0] <= due_by:
            delivered_at, message = self.pending.pop(0)
            self._clock.now = max(self._clock.now, delivered_at)
            return message
        self._clock.now = due_by
        return None


def _change(*model_ids: str) -> str:
    return json.dumps({"object_type": "litellm_proxymodeltable", "model_ids": list(model_ids)})


def _redis_delivery(message: str) -> dict:
    return {"type": "message", "pattern": None, "channel": b"litellm_proxy.config_change", "data": message.encode()}


class _ScriptedPubSubRedisClient(Redis):
    def __init__(self, pubsubs: Iterable[_QueuePubSub]) -> None:
        self._scripted_pubsubs = iter(pubsubs)

    def pubsub(self) -> _QueuePubSub:
        return next(self._scripted_pubsubs)


class _FakeRedisCache:
    def __init__(self, client: object, namespace: Optional[str] = None) -> None:
        self._client = client
        self.namespace = namespace

    def init_pubsub_client(self) -> object:
        return self._client


class _ExplodingRedisCache:
    namespace: Optional[str] = None

    def init_pubsub_client(self) -> object:
        raise ConnectionError("cannot connect")


def _recording_callback(
    events: List[str], name: str, fired: asyncio.Event
) -> Callable[[], Coroutine[None, None, None]]:
    async def callback() -> None:
        events.append(name)
        fired.set()

    return callback


async def test_publish_noops_when_redis_cache_is_none() -> None:
    await publish_config_change(redis_cache=None, object_type="litellm_proxymodeltable")


async def test_publish_sends_object_type_json_on_channel() -> None:
    client = _RecordingRedisClient()
    cache = _FakeRedisCache(client)

    await publish_config_change(redis_cache=cache, object_type="litellm_proxymodeltable")

    assert len(client.published) == 1
    channel, message = client.published[0]
    assert channel == "litellm_proxy.config_change"
    assert json.loads(message) == {"object_type": "litellm_proxymodeltable", "model_ids": []}


async def test_publish_carries_the_written_model_ids() -> None:
    client = _RecordingRedisClient()
    cache = _FakeRedisCache(client)

    await publish_config_change(redis_cache=cache, object_type="litellm_proxymodeltable", model_ids=("m-1", "m-2"))

    assert json.loads(client.published[0][1]) == {"object_type": "litellm_proxymodeltable", "model_ids": ["m-1", "m-2"]}


async def test_publish_uses_namespaced_channel() -> None:
    client = _RecordingRedisClient()
    cache = _FakeRedisCache(client, namespace="prod-eu")

    await publish_config_change(redis_cache=cache, object_type="litellm_credentialstable")

    assert client.published[0][0] == "prod-eu:litellm_proxy.config_change"


async def test_publish_swallows_redis_publish_errors() -> None:
    cache = _FakeRedisCache(_FailingPublishRedisClient())

    await publish_config_change(redis_cache=cache, object_type="litellm_proxymodeltable")


async def test_publish_swallows_client_init_errors() -> None:
    await publish_config_change(redis_cache=_ExplodingRedisCache(), object_type="litellm_proxymodeltable")


async def test_publish_reaches_cluster_derived_pubsub_clients() -> None:
    """LIT-8543: a cluster-backed cache returns a node-level client from
    init_pubsub_client; publishes must go out on it instead of being skipped."""
    client = _ScriptedPubSubClient(pubsubs=[])
    cache = _FakeRedisCache(client)

    await publish_config_change(redis_cache=cache, object_type="litellm_proxymodeltable")

    assert client.published == [
        (CONFIG_SYNC_CHANNEL, json.dumps({"object_type": "litellm_proxymodeltable", "model_ids": []}))
    ]


async def test_subscriber_runs_injected_callbacks_in_order_on_message() -> None:
    pubsub = _QueuePubSub()
    cache = _FakeRedisCache(_ScriptedPubSubRedisClient([pubsub]))
    events: List[str] = []
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
    cache = _FakeRedisCache(_ScriptedPubSubRedisClient([pubsub]))
    resyncs: List[str] = []
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
    cache = _FakeRedisCache(_ScriptedPubSubRedisClient([pubsub]), namespace="prod-eu")
    resyncs: List[str] = []
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


def _virtual_time_subscriber(
    pubsub: _VirtualTimePubSub,
    clock: _FakeClock,
    events: List[str],
    resynced: asyncio.Event,
    *,
    debounce_seconds: float = 0.0,
    jitter_max_seconds: float = 0.0,
    min_resync_interval_seconds: float = 10.0,
    rng: Optional[random.Random] = None,
    expected_resyncs: int = 1,
    apply_error: Optional[Exception] = None,
) -> ConfigSyncSubscriber:
    async def resync() -> None:
        events.append(f"resync@{clock.now}")
        if sum(event.startswith("resync@") for event in events) >= expected_resyncs:
            resynced.set()

    async def apply(model_ids: Tuple[str, ...]) -> None:
        events.append(f"apply:{','.join(model_ids)}@{clock.now}")
        if apply_error is not None:
            raise apply_error

    return ConfigSyncSubscriber(
        redis_cache=_FakeRedisCache(_ScriptedPubSubRedisClient([pubsub])),
        resync_callbacks=(resync,),
        apply_model_ids=apply,
        debounce_seconds=debounce_seconds,
        jitter_max_seconds=jitter_max_seconds,
        min_resync_interval_seconds=min_resync_interval_seconds,
        rng=rng,
        monotonic=clock,
    )


async def _run_until_resynced(subscriber: ConfigSyncSubscriber, resynced: asyncio.Event) -> None:
    subscriber.start()
    try:
        await asyncio.wait_for(resynced.wait(), timeout=5)
    finally:
        await subscriber.stop()


async def test_debounce_waits_out_the_jitter_from_injected_rng() -> None:
    clock = _FakeClock()
    pubsub = _VirtualTimePubSub(clock, deliveries=[(1000.0, _change())])
    events: List[str] = []
    resynced = asyncio.Event()
    subscriber = _virtual_time_subscriber(
        pubsub, clock, events, resynced, debounce_seconds=1.0, jitter_max_seconds=4.0, rng=_MaxJitterRandom()
    )

    await _run_until_resynced(subscriber, resynced)

    assert events == ["resync@1005.0"]


async def test_published_model_ids_apply_on_receipt_while_the_full_resync_still_debounces() -> None:
    clock = _FakeClock()
    pubsub = _VirtualTimePubSub(clock, deliveries=[(1000.0, _change("m-1"))])
    events: List[str] = []
    resynced = asyncio.Event()
    subscriber = _virtual_time_subscriber(
        pubsub, clock, events, resynced, debounce_seconds=1.0, jitter_max_seconds=4.0, rng=_MaxJitterRandom()
    )

    await _run_until_resynced(subscriber, resynced)

    assert events == ["apply:m-1@1000.0", "resync@1005.0"]


async def test_model_ids_published_inside_the_throttle_wait_apply_without_waiting() -> None:
    clock = _FakeClock()
    pubsub = _VirtualTimePubSub(
        clock, deliveries=[(1000.0, _change()), (1004.0, _change("m-2")), (1006.0, _change("m-3", "m-4"))]
    )
    events: List[str] = []
    resynced = asyncio.Event()
    subscriber = _virtual_time_subscriber(pubsub, clock, events, resynced, expected_resyncs=2)

    await _run_until_resynced(subscriber, resynced)

    assert events == ["resync@1000.0", "apply:m-2@1004.0", "apply:m-3,m-4@1006.0", "resync@1010.0"]


async def test_redis_shaped_message_applies_its_model_ids() -> None:
    clock = _FakeClock()
    pubsub = _VirtualTimePubSub(clock, deliveries=[(1000.0, _redis_delivery(_change("m-1")))])
    events: List[str] = []
    resynced = asyncio.Event()
    subscriber = _virtual_time_subscriber(pubsub, clock, events, resynced)

    await _run_until_resynced(subscriber, resynced)

    assert events == ["apply:m-1@1000.0", "resync@1000.0"]


async def test_messages_without_usable_model_ids_only_schedule_the_full_resync() -> None:
    clock = _FakeClock()
    pubsub = _VirtualTimePubSub(
        clock,
        deliveries=[
            (1000.0, "change"),
            (1004.0, json.dumps({"object_type": "litellm_proxymodeltable"})),
            (1005.0, json.dumps({"object_type": "litellm_proxymodeltable", "model_ids": "m-1"})),
            (1006.0, {"type": "message", "pattern": None, "channel": b"c", "data": 7}),
            (1007.0, None),
        ],
    )
    events: List[str] = []
    resynced = asyncio.Event()
    subscriber = _virtual_time_subscriber(pubsub, clock, events, resynced, expected_resyncs=2)

    await _run_until_resynced(subscriber, resynced)

    assert events == ["resync@1000.0", "resync@1010.0"]


async def test_failing_targeted_apply_still_runs_the_full_resync() -> None:
    clock = _FakeClock()
    pubsub = _VirtualTimePubSub(clock, deliveries=[(1000.0, _change("m-1"))])
    events: List[str] = []
    resynced = asyncio.Event()
    subscriber = _virtual_time_subscriber(pubsub, clock, events, resynced, apply_error=RuntimeError("db down"))

    await _run_until_resynced(subscriber, resynced)

    assert events == ["apply:m-1@1000.0", "resync@1000.0"]


async def test_subscriber_without_an_apply_callback_only_resyncs() -> None:
    clock = _FakeClock()
    pubsub = _VirtualTimePubSub(clock, deliveries=[(1000.0, _change("m-1"))])
    resyncs: List[str] = []
    fired = asyncio.Event()
    subscriber = ConfigSyncSubscriber(
        redis_cache=_FakeRedisCache(_ScriptedPubSubRedisClient([pubsub])),
        resync_callbacks=(_recording_callback(resyncs, "resync", fired),),
        debounce_seconds=0.0,
        jitter_max_seconds=0.0,
        monotonic=clock,
    )

    await _run_until_resynced(subscriber, fired)

    assert resyncs == ["resync"]


async def test_peer_holds_every_published_row_before_its_next_full_resync() -> None:
    publisher = _RecordingRedisClient()
    cache = _FakeRedisCache(publisher)
    await publish_config_change(redis_cache=cache, object_type="litellm_proxymodeltable", model_ids=("m-order-1",))
    await publish_config_change(redis_cache=cache, object_type="litellm_proxymodeltable", model_ids=("m-order-2",))
    on_the_wire = [_redis_delivery(message) for _, message in publisher.published]
    clock = _FakeClock()
    pubsub = _VirtualTimePubSub(clock, deliveries=[(1000.0, on_the_wire[0]), (1007.0, on_the_wire[1])])
    db_rows_written_at = {"m-order-1": 1000.0, "m-order-2": 1007.0}
    registry: set = set()
    snapshots: List[Tuple[str, float, frozenset]] = []
    resynced = asyncio.Event()

    async def apply(model_ids: Tuple[str, ...]) -> None:
        registry.update(model_ids)
        snapshots.append(("apply", clock.now, frozenset(registry)))

    async def resync() -> None:
        registry.update(row for row, written_at in db_rows_written_at.items() if written_at <= clock.now)
        snapshots.append(("resync", clock.now, frozenset(registry)))
        if sum(kind == "resync" for kind, _, _ in snapshots) == 2:
            resynced.set()

    subscriber = ConfigSyncSubscriber(
        redis_cache=_FakeRedisCache(_ScriptedPubSubRedisClient([pubsub])),
        resync_callbacks=(resync,),
        apply_model_ids=apply,
        debounce_seconds=1.0,
        jitter_max_seconds=5.0,
        min_resync_interval_seconds=10.0,
        rng=_MaxJitterRandom(),
        monotonic=clock,
    )

    await _run_until_resynced(subscriber, resynced)

    assert snapshots == [
        ("apply", 1000.0, frozenset({"m-order-1"})),
        ("resync", 1006.0, frozenset({"m-order-1"})),
        ("apply", 1007.0, frozenset({"m-order-1", "m-order-2"})),
        ("resync", 1016.0, frozenset({"m-order-1", "m-order-2"})),
    ]


def test_default_jitter_window_is_nonzero() -> None:
    assert CONFIG_SYNC_JITTER_MAX_SECONDS > 0


def test_default_min_resync_interval_caps_reload_rate() -> None:
    assert CONFIG_SYNC_MIN_RESYNC_INTERVAL_SECONDS > CONFIG_SYNC_JITTER_MAX_SECONDS


async def test_resync_arriving_inside_min_interval_waits_out_the_remainder() -> None:
    clock = _FakeClock()
    pubsub = _VirtualTimePubSub(clock, deliveries=[(1000.0, "change"), (1004.0, "change")])
    events: List[str] = []
    resynced = asyncio.Event()
    subscriber = _virtual_time_subscriber(pubsub, clock, events, resynced, expected_resyncs=2)

    await _run_until_resynced(subscriber, resynced)

    assert events == ["resync@1000.0", "resync@1010.0"]


async def test_resync_after_min_interval_elapsed_is_not_throttled() -> None:
    clock = _FakeClock()
    pubsub = _VirtualTimePubSub(clock, deliveries=[(1000.0, "change"), (1030.0, "change")])
    events: List[str] = []
    resynced = asyncio.Event()
    subscriber = _virtual_time_subscriber(pubsub, clock, events, resynced, expected_resyncs=2)

    await _run_until_resynced(subscriber, resynced)

    assert events == ["resync@1000.0", "resync@1030.0"]


async def test_writes_during_the_throttle_wait_collapse_into_the_next_resync() -> None:
    clock = _FakeClock()
    pubsub = _VirtualTimePubSub(
        clock, deliveries=[(1000.0, "change")] + [(1004.0 + offset, "change") for offset in range(5)]
    )
    events: List[str] = []
    resynced = asyncio.Event()
    subscriber = _virtual_time_subscriber(pubsub, clock, events, resynced, expected_resyncs=2)

    await _run_until_resynced(subscriber, resynced)

    assert events == ["resync@1000.0", "resync@1010.0"]
    assert pubsub.pending == []


async def test_polls_without_messages_do_not_trigger_resyncs() -> None:
    pubsub = _EmptyPollsThenMessagePubSub(empty_polls=3, initial_messages=["change"])
    cache = _FakeRedisCache(_ScriptedPubSubRedisClient([pubsub]))
    resyncs: List[str] = []
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
    cache = _FakeRedisCache(_ScriptedPubSubRedisClient([broken, healthy]))
    resyncs: List[str] = []
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
    cache = _FakeRedisCache(_ScriptedPubSubRedisClient([pubsub]))
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
    cache = _FakeRedisCache(_ScriptedPubSubRedisClient([broken, healthy]))
    resyncs: List[str] = []
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
    cache = _FakeRedisCache(_ScriptedPubSubRedisClient([pubsub]))
    resyncs: List[str] = []
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
    cache = _FakeRedisCache(_ScriptedPubSubRedisClient([pubsub]))
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
    cache = _FakeRedisCache(_ScriptedPubSubRedisClient([]))
    subscriber = ConfigSyncSubscriber(redis_cache=cache, resync_callbacks=())

    await subscriber.stop()


async def test_subscriber_subscribes_on_cluster_derived_pubsub_client() -> None:
    """LIT-8543: the subscriber used to disable itself on cluster caches; now it
    subscribes on the node-level client init_pubsub_client returns."""
    pubsub = _QueuePubSub(initial_messages=[json.dumps({"object_type": "litellm_proxymodeltable"})])
    cache = _FakeRedisCache(_ScriptedPubSubClient(pubsubs=[pubsub]))
    resyncs: List[str] = []
    fired = asyncio.Event()
    subscriber = ConfigSyncSubscriber(
        redis_cache=cache,
        resync_callbacks=(_recording_callback(resyncs, "resync", fired),),
        debounce_seconds=0.01,
        jitter_max_seconds=0.0,
    )

    subscriber.start()
    await asyncio.wait_for(fired.wait(), timeout=5)
    await subscriber.stop()

    assert pubsub.subscribed_channels == [CONFIG_SYNC_CHANNEL]
    assert resyncs == ["resync"]


class _FakeTableActions:
    def __init__(self, calls: List[Tuple[str, str]]) -> None:
        self._calls = calls

    async def create(self, **kwargs: object) -> object:
        self._calls.append(("write", "create"))
        return {"id": "m-1"}

    async def find_many(self, **kwargs: object) -> object:
        self._calls.append(("read", "find_many"))
        return []


class _AllWritesTableActions:
    def __init__(self, calls: List[str]) -> None:
        self._calls = calls

    def __getattr__(self, name: str) -> Callable[..., Coroutine[None, None, str]]:
        async def action(*args: object, **kwargs: object) -> str:
            self._calls.append(name)
            return name

        return action


def _recording_publish(calls: List[Tuple[str, str]]) -> Callable[[str, Tuple[str, ...]], Coroutine[None, None, None]]:
    async def publish(object_type: str, model_ids: Tuple[str, ...]) -> None:
        calls.append(("publish", object_type))

    return publish


def _recording_publish_with_ids(
    calls: List[Tuple[str, Tuple[str, ...]]],
) -> Callable[[str, Tuple[str, ...]], Coroutine[None, None, None]]:
    async def publish(object_type: str, model_ids: Tuple[str, ...]) -> None:
        calls.append((object_type, model_ids))

    return publish


class _ModelRow(BaseModel):
    model_id: str
    model_name: str


class _ModelRowTableActions:
    def __getattr__(self, name: str) -> Callable[..., Coroutine[None, None, object]]:
        async def action(*args: object, **kwargs: object) -> object:
            if name.endswith("_many"):
                return 2
            return _ModelRow(model_id="m-1", model_name="gpt-5.2")

        return action


def test_wrapper_passes_through_unsynced_tables() -> None:
    actions = object()

    wrapped = wrap_table_actions_for_config_sync(actions=actions, table_name="litellm_spendlogs")

    assert wrapped is actions


async def test_wrapper_publishes_table_name_after_write() -> None:
    calls: List[Tuple[str, str]] = []
    wrapped = wrap_table_actions_for_config_sync(
        actions=_FakeTableActions(calls),
        table_name="litellm_proxymodeltable",
        publish=_recording_publish(calls),
    )

    result = await wrapped.create(data={"model_name": "gpt-5.2"})

    assert result == {"id": "m-1"}
    assert calls == [("write", "create"), ("publish", "litellm_proxymodeltable")]


async def test_wrapper_does_not_publish_on_reads() -> None:
    calls: List[Tuple[str, str]] = []
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
    write_calls: List[str] = []
    publish_calls: List[Tuple[str, str]] = []
    wrapped = wrap_table_actions_for_config_sync(
        actions=_AllWritesTableActions(write_calls),
        table_name="litellm_guardrailstable",
        publish=_recording_publish(publish_calls),
    )

    result = await getattr(wrapped, action_name)(data={})

    assert result == action_name
    assert write_calls == [action_name]
    assert publish_calls == [("publish", "litellm_guardrailstable")]


@pytest.mark.parametrize(
    "action_name, expected_ids",
    [
        ("create", ("m-1",)),
        ("update", ("m-1",)),
        ("upsert", ("m-1",)),
        ("delete", ()),
        ("create_many", ()),
        ("update_many", ()),
        ("delete_many", ()),
    ],
)
async def test_wrapper_publishes_the_written_model_id_for_row_writes(
    action_name: str, expected_ids: Tuple[str, ...]
) -> None:
    publish_calls: List[Tuple[str, Tuple[str, ...]]] = []
    wrapped = wrap_table_actions_for_config_sync(
        actions=_ModelRowTableActions(),
        table_name="litellm_proxymodeltable",
        publish=_recording_publish_with_ids(publish_calls),
    )

    await getattr(wrapped, action_name)(data={})

    assert publish_calls == [("litellm_proxymodeltable", expected_ids)]


async def test_wrapper_publishes_no_model_ids_for_other_tables() -> None:
    publish_calls: List[Tuple[str, Tuple[str, ...]]] = []
    wrapped = wrap_table_actions_for_config_sync(
        actions=_ModelRowTableActions(),
        table_name="litellm_guardrailstable",
        publish=_recording_publish_with_ids(publish_calls),
    )

    await wrapped.create(data={})

    assert publish_calls == [("litellm_guardrailstable", ())]


async def test_model_repository_write_publishes_via_live_coordination_cache() -> None:
    from litellm.proxy import proxy_server
    from litellm.proxy.proxy_server import _set_redis_usage_cache
    from litellm.repositories.model_repository import ModelRepository

    client = _RecordingRedisClient()
    prisma_client = MagicMock()
    prisma_client.db.litellm_proxymodeltable.update = AsyncMock(return_value={"model_id": "m-1"})
    repository = ModelRepository(prisma_client)
    table = repository.table
    assert isinstance(table, _PublishOnWriteActions)

    previous_cache = proxy_server.redis_usage_cache
    _set_redis_usage_cache(_FakeRedisCache(client))
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
    assert json.loads(message) == {"object_type": "litellm_proxymodeltable", "model_ids": ["m-1"]}


async def test_ui_settings_write_publishes_via_live_coordination_cache() -> None:
    from litellm.proxy import proxy_server
    from litellm.proxy.proxy_server import _set_redis_usage_cache
    from litellm.repositories.table_repositories import UISettingsRepository

    client = _RecordingRedisClient()
    prisma_client = MagicMock()
    prisma_client.db.litellm_uisettings.upsert = AsyncMock(return_value={"id": "ui_settings"})
    table = UISettingsRepository(prisma_client).table
    assert isinstance(table, _PublishOnWriteActions)

    previous_cache = proxy_server.redis_usage_cache
    _set_redis_usage_cache(_FakeRedisCache(client))
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
    assert json.loads(message) == {"object_type": "litellm_uisettings", "model_ids": []}


async def _publish_calls_for_invalidated_param(param_name: str) -> List[Tuple[str, str]]:
    from litellm.proxy import proxy_server
    from litellm.proxy.proxy_server import _set_redis_usage_cache
    from litellm.proxy.utils import invalidate_config_param

    client = _RecordingRedisClient()
    previous_cache = proxy_server.redis_usage_cache
    _set_redis_usage_cache(_FakeRedisCache(client))
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
    assert json.loads(message) == {"object_type": param_name, "model_ids": []}


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

    client = _RecordingRedisClient()
    previous_cache = proxy_server.redis_usage_cache
    _set_redis_usage_cache(_FakeRedisCache(client))
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
    client = _RecordingRedisClient()
    previous_cache = proxy_server.redis_usage_cache
    original_model_cost = litellm.model_cost.copy()
    _set_redis_usage_cache(_FakeRedisCache(client))
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
    client = _RecordingRedisClient()
    previous_cache = proxy_server.redis_usage_cache
    _set_redis_usage_cache(_FakeRedisCache(client))
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

    cache = _FakeRedisCache(_ScriptedPubSubRedisClient([_QueuePubSub()]))
    config = ProxyConfig()
    prisma_client = MagicMock()
    proxy_logging_obj = MagicMock()
    calls: List[Tuple[str, object, object]] = []

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


async def test_proxy_config_subscriber_applies_published_model_ids_from_the_db(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from litellm.proxy import proxy_server
    from litellm.proxy.proxy_server import ProxyConfig

    rows = [MagicMock()]
    prisma_client = MagicMock()
    prisma_client.db.litellm_proxymodeltable.find_many = AsyncMock(return_value=rows)
    router = MagicMock()
    router.get_model_list.return_value = ["deployment"]
    installed: List[object] = []

    def install(db_models: object) -> None:
        installed.append(db_models)

    monkeypatch.setattr(proxy_server, "prisma_client", prisma_client)
    monkeypatch.setattr(proxy_server, "store_model_in_db", True)
    monkeypatch.setattr(proxy_server, "llm_router", router)
    monkeypatch.setattr(proxy_server, "llm_model_list", None)
    monkeypatch.setattr(proxy_server.proxy_config, "get_credentials", AsyncMock())
    monkeypatch.setattr(proxy_server.proxy_config, "_add_deployment", install)
    cache = _FakeRedisCache(_ScriptedPubSubRedisClient([_QueuePubSub()]))
    config = ProxyConfig()

    config.start_config_sync_subscriber(prisma_client=prisma_client, proxy_logging_obj=MagicMock(), redis_cache=cache)
    subscriber = config.config_sync_subscriber
    assert subscriber is not None
    apply = subscriber._apply_model_ids
    assert apply is not None
    await apply(("m-1", "m-2"))
    await config.stop_config_sync_subscriber()

    prisma_client.db.litellm_proxymodeltable.find_many.assert_awaited_once_with(
        where={"model_id": {"in": ["m-1", "m-2"]}}
    )
    assert installed == [rows]
    assert proxy_server.llm_model_list == ["deployment"]


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

    cache = _FakeRedisCache(_ScriptedPubSubRedisClient([_QueuePubSub()]))
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
        redis_cache=_FakeRedisCache(_ScriptedPubSubRedisClient([])),
        resync_callbacks=(),
    )

    await config.stop_config_sync_subscriber()

    assert config.config_sync_subscriber is None
