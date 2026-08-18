import asyncio
import hashlib
import time
from types import SimpleNamespace

import fakeredis.aioredis
import pytest
from redis.crc import key_slot

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.hooks.active_request_registry import ActiveRequestRegistry


class FakePipeline:
    def __init__(self, client):
        self.client = client
        self.operations = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    def set(self, key, value, ex=None):
        self.operations.append(("set", key, value, ex))

    def zadd(self, key, values):
        self.operations.append(("zadd", key, values))

    def expire(self, key, ttl):
        self.operations.append(("expire", key, ttl))

    def delete(self, key):
        self.operations.append(("delete", key))

    def zrem(self, key, value):
        self.operations.append(("zrem", key, value))

    async def execute(self):
        for operation in self.operations:
            if operation[0] == "set":
                self.client.values[operation[1]] = operation[2]
                self.client.expirations[operation[1]] = operation[3]
            elif operation[0] == "zadd":
                self.client.sorted_sets.setdefault(operation[1], {}).update(operation[2])
            elif operation[0] == "delete":
                self.client.values.pop(operation[1], None)
            elif operation[0] == "zrem":
                self.client.sorted_sets.setdefault(operation[1], {}).pop(operation[2], None)
        return []


class FakeRedisClient:
    def __init__(self):
        self.values = {}
        self.sorted_sets = {}
        self.expirations = {}
        self.zrevrange_calls = 0

    def pipeline(self, transaction=False):
        return FakePipeline(self)

    async def zremrangebyscore(self, key, minimum, maximum):
        values = self.sorted_sets.setdefault(key, {})
        for member in [member for member, score in values.items() if score <= maximum]:
            values.pop(member)

    async def zrevrange(self, key, start, end):
        self.zrevrange_calls += 1
        values = self.sorted_sets.setdefault(key, {})
        ordered = [member for member, _ in sorted(values.items(), key=lambda pair: pair[1], reverse=True)]
        return ordered[start : None if end == -1 else end + 1]

    async def zcard(self, key):
        return len(self.sorted_sets.setdefault(key, {}))

    async def mget(self, keys):
        slots = {key_slot(key.encode()) for key in keys}
        if len(slots) > 1:
            raise RuntimeError("CROSSSLOT Keys in request don't hash to the same slot")
        return [self.values.get(key) for key in keys]

    async def zrem(self, key, *members):
        for member in members:
            self.sorted_sets.setdefault(key, {}).pop(member, None)


class FakeRedisCache:
    def __init__(self):
        self.client = FakeRedisClient()

    def init_async_client(self):
        return self.client

    def check_and_fix_namespace(self, key):
        return f"test:{key}"


def make_registry(max_scan_members=None):
    redis_cache = FakeRedisCache()
    usage_cache = SimpleNamespace(dual_cache=SimpleNamespace(redis_cache=redis_cache))
    return ActiveRequestRegistry(usage_cache, max_scan_members=max_scan_members), redis_cache


def make_fakeredis_registry(max_scan_members=None):
    redis_cache = SimpleNamespace(
        init_async_client=lambda: redis_cache.client,
        check_and_fix_namespace=lambda key: f"integration:{key}",
        client=fakeredis.aioredis.FakeRedis(),
    )
    usage_cache = SimpleNamespace(dual_cache=SimpleNamespace(redis_cache=redis_cache))
    return ActiveRequestRegistry(usage_cache, max_scan_members=max_scan_members)


def set_general_settings(monkeypatch, **settings):
    """The registry reads its knobs from general_settings at call time."""
    import litellm.proxy.proxy_server as proxy_server

    monkeypatch.setattr(proxy_server, "general_settings", settings, raising=False)


def test_should_build_identity_record_without_request_content(monkeypatch):
    monkeypatch.setenv("HOSTNAME", "proxy-1")
    import litellm.proxy.proxy_server as proxy_server

    monkeypatch.setattr(proxy_server, "master_key", "master-key", raising=False)
    auth = UserAPIKeyAuth(
        api_key="sk-test-key-value",
        user_id="user-1",
        user_email="user@example.test",
        key_alias="production",
        team_id="team-1",
        team_alias="AI Team",
        org_id="org-1",
        project_id="project-1",
        end_user_id="end-user-1",
        metadata={"organization_alias": "spoofed"},
        organization_metadata={"organization_alias": "Example Org"},
        project_metadata={"project_alias": "Chat Project"},
    )

    record = ActiveRequestRegistry.build_record(
        {"litellm_call_id": "call-1", "model": "gpt-test", "messages": ["secret"]},
        auth,
        "acompletion",
        started_at=100.0,
    )

    assert record["end_user_id"] == "end-user-1"
    assert record["organization_id"] == "org-1"
    assert record["organization_alias"] == "Example Org"
    assert record["project_alias"] == "Chat Project"
    assert record["key_fingerprint"] == hashlib.sha256(f"master-key:{auth.api_key}".encode()).hexdigest()[:12]
    assert record["pod"] == "proxy-1"
    assert "messages" not in record


@pytest.mark.asyncio
async def test_should_register_list_filter_and_remove_requests():
    registry, _ = make_registry()
    auth = UserAPIKeyAuth(
        api_key="sk-test-key",
        user_id="user-1",
        end_user_id="end-user-1",
        org_id="org-1",
        project_id="project-1",
    )
    data = {"litellm_call_id": "same-call-id", "model": "model-a", "stream": True}

    first_id = await registry.register(auth, data, "acompletion")
    second_id = await registry.register(auth, data, "acompletion")

    assert first_id is not None
    assert second_id is not None
    assert first_id != second_id
    result = await registry.list_requests(end_user_id="end-user-1")
    assert result["available"] is True
    assert result["total"] == 2
    assert all(item["streaming"] for item in result["items"])

    await registry.remove(first_id)
    remaining = await registry.list_requests()
    assert remaining["total"] == 1


@pytest.mark.asyncio
async def test_should_update_one_record_when_pre_call_runs_again():
    registry, redis_cache = make_registry()
    auth = UserAPIKeyAuth(api_key="sk-test-key", user_id="user-1")
    started_at = time.time()

    registry_id = await registry.register(
        auth,
        {"litellm_call_id": "client-call-id", "model": "model-a"},
        "acompletion",
        started_at=started_at,
    )
    updated_registry_id = await registry.register(
        auth,
        {"litellm_call_id": "client-call-id", "model": "fallback-model"},
        "acompletion",
        registry_id=registry_id,
        started_at=started_at,
    )

    assert registry_id == updated_registry_id
    assert registry_id is not None
    assert "client-call-id" not in registry_id
    assert len(redis_cache.client.sorted_sets["test:" + registry.INDEX_KEY]) == 1
    result = await registry.list_requests()
    assert result["total"] == 1
    assert result["items"][0]["model"] == "fallback-model"
    assert result["items"][0]["started_at"] == started_at


@pytest.mark.asyncio
async def test_should_bound_untrusted_record_fields():
    registry, _ = make_registry()
    auth = UserAPIKeyAuth(api_key="sk-test-key", metadata={"organization_alias": {"large": "value"}})
    long_value = "x" * 2000

    await registry.register(
        auth,
        {"litellm_call_id": long_value, "model": long_value},
        "acompletion",
    )

    result = await registry.list_requests()
    assert len(result["items"][0]["request_id"]) == registry.MAX_FIELD_LENGTH
    assert len(result["items"][0]["model"]) == registry.MAX_FIELD_LENGTH
    assert result["items"][0]["organization_alias"] is None


@pytest.mark.asyncio
async def test_should_use_real_redis_pipeline_semantics_with_pagination():
    registry = make_fakeredis_registry()
    auth = UserAPIKeyAuth(api_key="sk-test-key", user_id="user-1")
    started_at = time.time()

    registry_ids = await asyncio.gather(
        *[
            registry.register(
                auth,
                {"litellm_call_id": f"call-{index}", "model": "model-a"},
                "acompletion",
                started_at=started_at + index / 1000,
            )
            for index in range(25)
        ]
    )

    first_page = await registry.list_requests(page=1, page_size=10)
    third_page = await registry.list_requests(page=3, page_size=10)
    assert first_page["total"] == 25
    assert len(first_page["items"]) == 10
    assert len(third_page["items"]) == 5

    await asyncio.gather(*(registry.remove(registry_id) for registry_id in registry_ids))
    assert (await registry.list_requests())["total"] == 0


@pytest.mark.asyncio
async def test_should_report_unavailable_without_redis():
    usage_cache = SimpleNamespace(dual_cache=SimpleNamespace(redis_cache=None))
    registry = ActiveRequestRegistry(usage_cache)

    result = await registry.list_requests()

    assert result["available"] is False
    assert result["items"] == ()


def test_should_fall_back_to_safe_ttl_for_invalid_configuration(monkeypatch):
    set_general_settings(monkeypatch, active_request_ttl_seconds="invalid")

    registry, _ = make_registry()

    assert registry.ttl_seconds == registry.DEFAULT_TTL_SECONDS


def test_should_bound_excessive_ttl_configuration(monkeypatch):
    set_general_settings(monkeypatch, active_request_ttl_seconds=999999999)

    registry, _ = make_registry()

    assert registry.ttl_seconds == registry.MAX_TTL_SECONDS


def test_all_multi_key_operations_share_a_redis_cluster_slot():
    registry, redis_cache = make_registry()
    keys = (
        registry._index_key(redis_cache),
        registry._item_key(redis_cache, "first"),
        registry._item_key(redis_cache, "second"),
        registry._cancel_key(redis_cache, "first"),
    )

    assert len({key_slot(key.encode()) for key in keys}) == 1


def test_default_ttl_should_expire_ghost_entries_within_half_an_hour(monkeypatch):
    """A pod killed mid-request leaves entries behind until the TTL expires."""
    set_general_settings(monkeypatch)

    registry, _ = make_registry()

    assert registry.ttl_seconds == 1800


@pytest.mark.asyncio
async def test_should_cap_the_index_scan_when_filters_are_applied():
    registry = make_fakeredis_registry(max_scan_members=2)
    auth = UserAPIKeyAuth(api_key="sk-test-key", user_id="user-1")
    started_at = time.time()

    for index in range(5):
        await registry.register(
            auth,
            {"litellm_call_id": f"call-{index}", "model": "model-a"},
            "acompletion",
            started_at=started_at + index / 1000,
        )

    result = await registry.list_requests(model="model-a", page_size=10)

    assert result["truncated"] is True
    assert len(result["items"]) == 2
    assert result["total"] == 2


@pytest.mark.asyncio
async def test_should_not_report_truncation_below_the_scan_cap():
    registry = make_fakeredis_registry()
    auth = UserAPIKeyAuth(api_key="sk-test-key", user_id="user-1")

    await registry.register(auth, {"litellm_call_id": "call-0", "model": "model-a"}, "acompletion")

    assert (await registry.list_requests(model="model-a"))["truncated"] is False
    assert (await registry.list_requests())["truncated"] is False


def test_should_omit_user_email_by_default(monkeypatch):
    set_general_settings(monkeypatch)

    record = ActiveRequestRegistry.build_record(
        {"litellm_call_id": "call-1"},
        UserAPIKeyAuth(api_key="sk-test-key", user_email="user@example.test"),
        "acompletion",
    )

    assert record["user_email"] is None


def test_should_include_user_email_when_explicitly_enabled(monkeypatch):
    set_general_settings(monkeypatch, active_request_include_user_email=True)

    record = ActiveRequestRegistry.build_record(
        {"litellm_call_id": "call-1"},
        UserAPIKeyAuth(api_key="sk-test-key", user_email="user@example.test"),
        "acompletion",
    )

    assert record["user_email"] == "user@example.test"


@pytest.mark.asyncio
async def test_should_skip_registration_without_redis_or_call_id():
    usage_cache = SimpleNamespace(dual_cache=SimpleNamespace(redis_cache=None))
    without_redis = ActiveRequestRegistry(usage_cache)
    with_redis, _ = make_registry()
    auth = UserAPIKeyAuth(api_key="sk-test-key")

    assert await without_redis.register(auth, {"litellm_call_id": "c"}, "acompletion") is None
    assert await with_redis.register(auth, {}, "acompletion") is None
    await without_redis.remove("registry-id")
    await with_redis.remove(None)


@pytest.mark.asyncio
async def test_should_not_raise_when_redis_writes_fail():
    registry, redis_cache = make_registry()

    def explode():
        raise ConnectionError("redis down")

    redis_cache.init_async_client = explode

    assert (
        await registry.register(
            UserAPIKeyAuth(api_key="sk-test-key"),
            {"litellm_call_id": "call-1"},
            "acompletion",
        )
        is None
    )
    await registry.remove("registry-id")


@pytest.mark.asyncio
async def test_should_not_raise_when_resolving_redis_fails():
    class ExplodingDualCache:
        @property
        def redis_cache(self):
            raise ConnectionError("redis unavailable")

    registry = ActiveRequestRegistry(SimpleNamespace(dual_cache=ExplodingDualCache()))
    auth = UserAPIKeyAuth(api_key="sk-test-key")

    assert await registry.register(auth, {"litellm_call_id": "call-1"}, "acompletion") is None
    await registry.remove("registry-id")


@pytest.mark.asyncio
async def test_registration_sets_the_only_ghost_cleanup_ttl():
    registry, redis_cache = make_registry()

    registry_id = await registry.register(
        UserAPIKeyAuth(api_key="sk-test-key"),
        {"litellm_call_id": "call-1"},
        "acompletion",
    )

    assert redis_cache.client.expirations[registry._item_key(redis_cache, registry_id)] == registry.ttl_seconds


@pytest.mark.asyncio
async def test_remove_drops_the_local_task_even_when_cleanup_is_cancelled(monkeypatch):
    registry, _ = make_registry()
    registry._local_tasks["registry-id"] = asyncio.current_task()

    async def cancel_cleanup(_pipeline):
        raise asyncio.CancelledError

    monkeypatch.setattr(FakePipeline, "execute", cancel_cleanup)

    with pytest.raises(asyncio.CancelledError):
        await registry.remove("registry-id")
    assert "registry-id" not in registry._local_tasks


@pytest.mark.asyncio
async def test_should_drop_index_members_whose_record_expired():
    registry, redis_cache = make_registry()
    auth = UserAPIKeyAuth(api_key="sk-test-key", user_id="user-1")

    started_at = time.time()
    kept = await registry.register(auth, {"litellm_call_id": "call-0"}, "acompletion", started_at=started_at)
    expired = await registry.register(auth, {"litellm_call_id": "call-1"}, "acompletion", started_at=started_at + 1)
    # Simulate the item key hitting its TTL while the index member survives.
    redis_cache.client.values.pop(f"test:{registry.ITEM_KEY_PREFIX}{expired}")

    result = await registry.list_requests()

    assert [item["request_id"] for item in result["items"]] == ["call-0"]
    assert result["total"] == 1
    assert list(redis_cache.client.sorted_sets[f"test:{registry.INDEX_KEY}"]) == [kept]


@pytest.mark.asyncio
async def test_should_refill_a_page_after_dropping_stale_members():
    registry, redis_cache = make_registry()
    auth = UserAPIKeyAuth(api_key="sk-test-key")
    started_at = time.time()
    ids = [
        await registry.register(
            auth,
            {"litellm_call_id": f"call-{index}"},
            "acompletion",
            started_at=started_at + index,
        )
        for index in range(3)
    ]
    redis_cache.client.values.pop(registry._item_key(redis_cache, ids[2]))

    result = await registry.list_requests(page_size=2)

    assert [item["request_id"] for item in result["items"]] == ["call-1", "call-0"]
    assert result["total"] == 2


@pytest.mark.asyncio
async def test_filtered_queries_are_cached_for_one_second():
    registry, redis_cache = make_registry()
    await registry.register(
        UserAPIKeyAuth(api_key="sk-test-key"),
        {"litellm_call_id": "call-1", "model": "model-a"},
        "acompletion",
    )

    await registry.list_requests(model="model-a")
    calls_after_first_query = redis_cache.client.zrevrange_calls
    await registry.list_requests(model="model-a")

    assert redis_cache.client.zrevrange_calls == calls_after_first_query


@pytest.mark.asyncio
async def test_should_match_filters_exactly_not_by_substring():
    registry = make_fakeredis_registry()
    auth = UserAPIKeyAuth(api_key="sk-test-key", user_id="user-1")

    await registry.register(auth, {"litellm_call_id": "call-0", "model": "gpt-4o-mini"}, "acompletion")

    assert (await registry.list_requests(model="gpt-4o-mini"))["total"] == 1
    assert (await registry.list_requests(model="gpt-4o"))["total"] == 0


@pytest.mark.asyncio
async def test_should_return_an_empty_page_past_the_end_without_losing_the_total():
    registry = make_fakeredis_registry()
    auth = UserAPIKeyAuth(api_key="sk-test-key", user_id="user-1")
    started_at = time.time()

    for index in range(3):
        await registry.register(
            auth,
            {"litellm_call_id": f"call-{index}", "model": "model-a"},
            "acompletion",
            started_at=started_at + index / 1000,
        )

    unfiltered = await registry.list_requests(page=9, page_size=10)
    filtered = await registry.list_requests(model="model-a", page=9, page_size=10)

    assert unfiltered["items"] == () and unfiltered["total"] == 3
    assert filtered["items"] == () and filtered["total"] == 3


@pytest.mark.asyncio
async def test_should_treat_an_unreadable_record_as_stale():
    registry, redis_cache = make_registry()
    auth = UserAPIKeyAuth(api_key="sk-test-key")
    started_at = time.time()

    registry_id = await registry.register(auth, {"litellm_call_id": "call-0"}, "acompletion", started_at=started_at)
    redis_cache.client.values[f"test:{registry.ITEM_KEY_PREFIX}{registry_id}"] = "{not json"

    result = await registry.list_requests()

    assert result["items"] == ()
    assert result["total"] == 0
    assert redis_cache.client.sorted_sets[f"test:{registry.INDEX_KEY}"] == {}


@pytest.mark.asyncio
async def test_should_treat_a_non_object_record_as_stale():
    registry, redis_cache = make_registry()
    auth = UserAPIKeyAuth(api_key="sk-test-key")

    registry_id = await registry.register(auth, {"litellm_call_id": "call-0"}, "acompletion", started_at=time.time())
    redis_cache.client.values[f"test:{registry.ITEM_KEY_PREFIX}{registry_id}"] = "[1, 2, 3]"

    assert (await registry.list_requests())["items"] == ()


def test_should_drop_fields_that_are_blank_after_trimming():
    record = ActiveRequestRegistry.build_record(
        {"litellm_call_id": "call-1", "model": "   "},
        UserAPIKeyAuth(api_key="sk-test-key"),
        "acompletion",
    )

    assert record["model"] is None


def test_should_omit_the_key_fingerprint_when_there_is_no_key():
    record = ActiveRequestRegistry.build_record(
        {"litellm_call_id": "call-1"},
        UserAPIKeyAuth(api_key=None),
        "acompletion",
    )

    assert record["key_fingerprint"] is None


@pytest.mark.asyncio
async def test_should_not_publish_a_cancellation_for_an_unknown_request():
    registry = make_fakeredis_registry()

    assert await registry.request_cancel("does-not-exist") is False


@pytest.mark.asyncio
async def test_should_flag_a_running_request_for_cancellation():
    registry = make_fakeredis_registry()
    auth = UserAPIKeyAuth(api_key="sk-test-key")
    started = asyncio.Event()
    ids: list[str] = []

    async def served_request():
        ids.append(await registry.register(auth, {"litellm_call_id": "call-0"}, "acompletion"))
        started.set()
        await asyncio.sleep(30)

    task = asyncio.create_task(served_request())
    await started.wait()

    assert await registry.request_cancel(ids[0]) is True

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=5)


@pytest.mark.asyncio
async def test_should_report_no_cancellation_without_redis():
    usage_cache = SimpleNamespace(dual_cache=SimpleNamespace(redis_cache=None))
    registry = ActiveRequestRegistry(usage_cache)

    assert await registry.request_cancel("anything") is False


@pytest.mark.asyncio
async def test_should_cancel_the_task_this_worker_is_serving():
    registry = make_fakeredis_registry()
    auth = UserAPIKeyAuth(api_key="sk-test-key")
    started = asyncio.Event()
    registry_ids: list[str] = []

    async def served_request():
        registry_id = await registry.register(auth, {"litellm_call_id": "call-0"}, "acompletion")
        registry_ids.append(registry_id)
        started.set()
        await asyncio.sleep(30)

    task = asyncio.create_task(served_request())
    await started.wait()
    await asyncio.sleep(0.2)

    await registry.request_cancel(registry_ids[0])

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=5)


def test_should_ignore_a_cancellation_for_a_request_this_worker_does_not_own():
    registry, _ = make_registry()

    registry._cancel_owned_task("not-a-known-request")


def test_should_leave_a_finished_task_alone():
    registry, _ = make_registry()

    async def already_done():
        return None

    loop = asyncio.new_event_loop()
    try:
        task = loop.create_task(already_done())
        loop.run_until_complete(task)
        registry._local_tasks["done-id"] = task
        registry._cancel_owned_task("done-id")
    finally:
        loop.close()

    assert task.cancelled() is False


@pytest.mark.asyncio
async def test_should_not_raise_when_publishing_a_cancellation_fails():
    registry, redis_cache = make_registry()
    auth = UserAPIKeyAuth(api_key="sk-test-key")
    registry_id = await registry.register(auth, {"litellm_call_id": "call-0"}, "acompletion")

    def explode():
        raise ConnectionError("redis down")

    redis_cache.init_async_client = explode

    assert await registry.request_cancel(registry_id) is False


@pytest.mark.asyncio
async def test_should_start_the_cancel_watcher_once_per_worker():
    registry = make_fakeredis_registry()
    auth = UserAPIKeyAuth(api_key="sk-test-key")

    await registry.register(auth, {"litellm_call_id": "call-0"}, "acompletion")
    first = registry._cancel_watcher
    await registry.register(auth, {"litellm_call_id": "call-1"}, "acompletion")

    assert first is not None
    assert registry._cancel_watcher is first


def test_should_not_start_a_cancel_watcher_without_redis():
    usage_cache = SimpleNamespace(dual_cache=SimpleNamespace(redis_cache=None))
    registry = ActiveRequestRegistry(usage_cache)

    registry._ensure_cancel_watcher()

    assert registry._cancel_watcher is None


@pytest.mark.asyncio
async def test_should_cancel_a_request_another_replica_flagged():
    """The flag is written by whichever replica served the admin call; the owner acts on it."""
    registry = make_fakeredis_registry()
    auth = UserAPIKeyAuth(api_key="sk-test-key")
    started = asyncio.Event()
    ids: list[str] = []

    async def served_request():
        ids.append(await registry.register(auth, {"litellm_call_id": "call-0"}, "acompletion"))
        started.set()
        await asyncio.sleep(30)

    task = asyncio.create_task(served_request())
    await started.wait()

    redis_cache = registry._redis_cache()
    await redis_cache.init_async_client().set(registry._cancel_key(redis_cache, ids[0]), "1")

    await registry._cancel_flagged_requests()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=5)
    assert await redis_cache.init_async_client().get(registry._cancel_key(redis_cache, ids[0])) is None


@pytest.mark.asyncio
async def test_should_ignore_a_poll_without_redis():
    usage_cache = SimpleNamespace(dual_cache=SimpleNamespace(redis_cache=None))

    await ActiveRequestRegistry(usage_cache)._cancel_flagged_requests()


@pytest.mark.asyncio
async def test_should_stop_watching_once_the_worker_has_no_requests_left():
    registry = make_fakeredis_registry()

    await asyncio.wait_for(registry._watch_for_cancellations(), timeout=5)


@pytest.mark.asyncio
async def test_should_keep_watching_after_a_failed_poll():
    registry = make_fakeredis_registry()
    registry._local_tasks["phantom"] = asyncio.current_task()
    calls: list[int] = []

    async def explode():
        calls.append(1)
        if len(calls) > 1:
            registry._local_tasks.clear()
        raise ConnectionError("redis down")

    registry._cancel_flagged_requests = explode

    await asyncio.wait_for(registry._watch_for_cancellations(), timeout=5)

    assert calls == [1, 1]


@pytest.mark.asyncio
async def test_should_let_a_shutdown_cancel_the_watcher():
    registry = make_fakeredis_registry()
    registry._local_tasks["phantom"] = asyncio.current_task()

    watcher = asyncio.create_task(registry._watch_for_cancellations())
    await asyncio.sleep(0)
    watcher.cancel()

    with pytest.raises(asyncio.CancelledError):
        await watcher
