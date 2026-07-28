import asyncio
import json
import time
from types import SimpleNamespace

import pytest

from litellm.constants import CACHE_WARMING_JOB_NAME
from litellm.router_strategy.complexity_router.cache_warming.eligibility import resolve_warm_models
from litellm.router_strategy.complexity_router.cache_warming.refresher import (
    CacheWarmingRefresher,
    collect_warming_enabled_complexity_routers,
    filter_cache_warmable,
)
from litellm.router_strategy.complexity_router.cache_warming.store import CacheWarmingStore
from litellm.router_strategy.complexity_router.cache_warming.types import (
    CACHE_WARMING_RECORD_SCHEMA_VERSION,
    CACHE_WARMING_REPLAY_MARKER_KEY,
    CACHE_WARMING_REPLAY_TAG,
    CacheWarmingAttribution,
    CacheWarmingPayload,
    CacheWarmingRecord,
    compress_payload,
)
from litellm.router_strategy.complexity_router.complexity_router import ComplexityRouter
from litellm.router_strategy.complexity_router.config import ComplexityRouterConfig
from litellm.types.router import TaggedPreRoutingStrategy

from tests.test_litellm.router_strategy.complexity_router.cache_warming.test_store import FakeRedisCache

_DEFAULT_DEPLOYMENTS = {
    "fast-claude": [{"litellm_params": {"model": "anthropic/claude-haiku-4-5"}}],
    "smart-claude": [{"litellm_params": {"model": "anthropic/claude-sonnet-4-5"}}],
    "fast-gpt": [{"litellm_params": {"model": "gpt-5-mini"}}],
    "titan": [{"litellm_params": {"model": "bedrock/amazon.titan-text-express-v1"}}],
    "mystery": [{"litellm_params": {"model": "totally-unknown-model-xyz"}}],
}


class FakeLLMRouter:
    def __init__(
        self,
        redis: FakeRedisCache | None = None,
        enable_tag_filtering: bool = False,
        deployments: dict | None = None,
        replay_delay: float = 0.0,
    ) -> None:
        self.complexity_routers: dict = {}
        self.enable_tag_filtering = enable_tag_filtering
        self.cache = SimpleNamespace(redis_cache=redis)
        self.completion_calls: list[dict] = []
        self.anthropic_calls: list[dict] = []
        self.max_concurrent = 0
        self._in_flight = 0
        self.replay_delay = replay_delay
        self.failing_message_marker: str | None = None
        self._deployments = deployments if deployments is not None else _DEFAULT_DEPLOYMENTS

    def get_model_list(self, model_name: str | None = None, team_id: str | None = None):
        return self._deployments.get(model_name)

    async def acompletion(self, **kwargs: object):
        marker = self.failing_message_marker
        if marker is not None and marker in json.dumps(kwargs.get("messages")):
            raise RuntimeError("provider down")
        self._in_flight += 1
        self.max_concurrent = max(self.max_concurrent, self._in_flight)
        await asyncio.sleep(self.replay_delay)
        self._in_flight -= 1
        self.completion_calls.append(kwargs)

    async def aanthropic_messages(self, **kwargs: object):
        self.anthropic_calls.append(kwargs)


class FakePodLockManager:
    def __init__(self, acquire_result: bool | None = True, redis_cache: object = "attached") -> None:
        self.acquire_result = acquire_result
        self.redis_cache = redis_cache
        self.acquire_calls: list[tuple[str, int | None]] = []
        self.release_calls: list[str] = []

    async def acquire_lock(self, cronjob_id: str, ttl: int | None = None) -> bool | None:
        self.acquire_calls.append((cronjob_id, ttl))
        return self.acquire_result

    async def release_lock(self, cronjob_id: str) -> None:
        self.release_calls.append(cronjob_id)


class FakePrismaClient:
    def __init__(self, rows: tuple = (), raise_error: bool = False) -> None:
        self.queries: list[dict] = []

        async def find_many(where: dict):
            self.queries.append(where)
            if raise_error:
                raise RuntimeError("db down")
            return [row for row in rows if row.token in where["token"]["in"]]

        self.db = SimpleNamespace(litellm_verificationtoken=SimpleNamespace(find_many=find_many))


def _warming_rig(
    redis: FakeRedisCache | None = None,
    enable_tag_filtering: bool = False,
    replay_delay: float = 0.0,
    **cache_warming_overrides: object,
) -> tuple[FakeLLMRouter, FakeRedisCache | None]:
    llm_router = FakeLLMRouter(redis=redis, enable_tag_filtering=enable_tag_filtering, replay_delay=replay_delay)
    strategy = ComplexityRouter(
        model_name="smart-router",
        litellm_router_instance=llm_router,
        complexity_router_config={
            "tiers": {"SIMPLE": ["fast-claude"], "COMPLEX": ["smart-claude"]},
            "cache_warming": {"enabled": True, **cache_warming_overrides},
        },
    )
    llm_router.complexity_routers = {"smart-router": [TaggedPreRoutingStrategy(tags=(), strategy=strategy)]}
    return llm_router, redis


def _seed_session(
    redis: FakeRedisCache,
    session_id: str = "sess-1",
    caller_scope: str = "hash-1",
    served_model: str = "fast-claude",
    last_activity: float | None = None,
    warmth: dict | None = None,
    user_api_key: str | None = "hash-1",
    call_surface: str = "chat_completions",
    content: str = "summarize the deployment policy",
    tools: tuple | None = None,
    tool_choice: object = None,
) -> str:
    payload = CacheWarmingPayload(
        model=served_model,
        messages=({"role": "user", "content": content},),
        system="You are a policy assistant" if call_surface == "anthropic_messages" else None,
        tools=tools,
        tool_choice=tool_choice,
        call_surface=call_surface,
    )
    blob, sha = compress_payload(payload)
    record = CacheWarmingRecord(
        schema_version=CACHE_WARMING_RECORD_SCHEMA_VERSION,
        payload_compressed=blob,
        payload_sha256=sha,
        token_estimate=2048,
        last_activity=last_activity if last_activity is not None else time.time(),
        served_model=served_model,
        attribution=CacheWarmingAttribution(user_api_key=user_api_key),
        auto_router_model_name="smart-router",
    )
    store = CacheWarmingStore(redis_cache=redis, auto_router_model_name="smart-router")
    key = CacheWarmingStore.record_key("smart-router", caller_scope, session_id)
    redis.hashes.setdefault(store.sessions_key(), {})[key] = json.dumps(record.model_dump())
    redis.zsets.setdefault(store.index_key(), {})[key] = time.time() + 3600
    for model_group, stamp in (warmth or {}).items():
        redis.data[CacheWarmingStore.warmth_key(key, model_group)] = json.dumps(stamp)
    return key


def _warmth_stamp(redis: FakeRedisCache, record_key: str, model_group: str) -> float | None:
    raw = redis.data.get(CacheWarmingStore.warmth_key(record_key, model_group))
    return json.loads(raw) if raw is not None else None


async def _tick(llm_router, lock=None, prisma=None, refresher: CacheWarmingRefresher | None = None):
    await (refresher or CacheWarmingRefresher()).run_tick(
        llm_router=llm_router, pod_lock_manager=lock, prisma_client=prisma
    )


def _replayed_models(llm_router: FakeLLMRouter) -> list:
    return [call["model"] for call in llm_router.completion_calls]


# ---------------------------------------------------------------------------
# warm-set resolution + eligibility
# ---------------------------------------------------------------------------


def test_resolve_warm_models_defaults_to_first_member_per_tier_deduped():
    config = ComplexityRouterConfig(
        tiers={"SIMPLE": ["fast-claude", "fast-gpt"], "MEDIUM": "fast-claude", "COMPLEX": ["smart-claude"]}
    )
    assert resolve_warm_models(config) == ("fast-claude", "smart-claude")


def test_resolve_warm_models_prefers_explicit_list():
    config = ComplexityRouterConfig(
        tiers={"SIMPLE": ["fast-claude"], "COMPLEX": ["smart-claude"]},
        cache_warming={"enabled": True, "warm_models": ["smart-claude", "smart-claude", "fast-claude"]},
    )
    assert resolve_warm_models(config) == ("smart-claude", "fast-claude")


def test_filter_cache_warmable_keeps_only_prompt_cacheable_anthropic_bedrock():
    llm_router = FakeLLMRouter()
    groups = ["fast-claude", "smart-claude", "fast-gpt", "titan", "mystery", "absent-group"]
    assert filter_cache_warmable(llm_router, groups) == ("fast-claude", "smart-claude")


def test_filter_cache_warmable_prefers_declared_custom_llm_provider_over_inference():
    deployments = {
        "declared-openai": [{"litellm_params": {"model": "anthropic/claude-sonnet-4-5", "custom_llm_provider": "openai"}}]
    }
    assert filter_cache_warmable(FakeLLMRouter(deployments=deployments), ["declared-openai"]) == ()


def test_collect_warming_enabled_complexity_routers_skips_disabled():
    llm_router, _ = _warming_rig(redis=FakeRedisCache())
    disabled = ComplexityRouter(
        model_name="plain-router",
        litellm_router_instance=llm_router,
        complexity_router_config={"tiers": {"SIMPLE": ["fast-claude"]}},
    )
    llm_router.complexity_routers["plain-router"] = [TaggedPreRoutingStrategy(tags=(), strategy=disabled)]
    collected = collect_warming_enabled_complexity_routers(llm_router)
    assert [strategy.model_name for strategy in collected] == ["smart-router"]


# ---------------------------------------------------------------------------
# pod lock
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_lock_held_by_other_pod_skips_tick_and_never_releases():
    llm_router, redis = _warming_rig(redis=FakeRedisCache())
    _seed_session(redis)
    lock = FakePodLockManager(acquire_result=False)
    await _tick(llm_router, lock=lock)
    assert llm_router.completion_calls == []
    assert lock.release_calls == []


@pytest.mark.asyncio
async def test_lock_acquired_warms_and_releases():
    llm_router, redis = _warming_rig(redis=FakeRedisCache())
    _seed_session(redis)
    lock = FakePodLockManager(acquire_result=True)
    await _tick(llm_router, lock=lock)
    assert sorted(_replayed_models(llm_router)) == ["fast-claude", "smart-claude"]
    assert lock.acquire_calls == [(CACHE_WARMING_JOB_NAME, 60)]
    assert lock.release_calls == [CACHE_WARMING_JOB_NAME]


@pytest.mark.asyncio
async def test_lock_falls_back_to_warming_redis_when_no_injected_manager():
    llm_router, redis = _warming_rig(redis=FakeRedisCache())
    _seed_session(redis)
    await _tick(llm_router, lock=None)
    assert sorted(_replayed_models(llm_router)) == ["fast-claude", "smart-claude"]
    assert any(key.startswith("cronjob_lock:") for key in redis.data)


@pytest.mark.asyncio
async def test_redisless_injected_manager_is_replaced_by_warming_redis_lock():
    llm_router, redis = _warming_rig(redis=FakeRedisCache())
    _seed_session(redis)
    lock = FakePodLockManager(redis_cache=None)
    await _tick(llm_router, lock=lock)
    assert lock.acquire_calls == []
    assert sorted(_replayed_models(llm_router)) == ["fast-claude", "smart-claude"]
    assert any(key.startswith("cronjob_lock:") for key in redis.data)


def test_fallback_lock_manager_is_stable_across_ticks():
    redis = FakeRedisCache()
    refresher = CacheWarmingRefresher()
    first = refresher._resolve_lock_manager(None, redis)
    second = refresher._resolve_lock_manager(None, redis)
    assert first is second
    assert first.redis_cache is redis


@pytest.mark.asyncio
async def test_lock_released_when_tick_raises():
    class ExplodingScriptRedis(FakeRedisCache):
        def async_register_script(self, script: str):
            async def boom(keys: list, args: list):
                raise RuntimeError("redis down")

            return boom

    llm_router, _ = _warming_rig(redis=ExplodingScriptRedis())
    lock = FakePodLockManager(acquire_result=True)
    with pytest.raises(RuntimeError, match="redis down"):
        await _tick(llm_router, lock=lock)
    assert lock.release_calls == [CACHE_WARMING_JOB_NAME]


@pytest.mark.asyncio
async def test_no_warming_routers_never_touches_lock():
    llm_router = FakeLLMRouter(redis=FakeRedisCache())
    lock = FakePodLockManager()
    await _tick(llm_router, lock=lock)
    assert lock.acquire_calls == []


# ---------------------------------------------------------------------------
# session selection: idle skip + interval pacing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_idle_session_not_warmed():
    llm_router, redis = _warming_rig(redis=FakeRedisCache())
    key = _seed_session(redis, last_activity=time.time() - 601)
    sessions_key = CacheWarmingStore(redis_cache=redis, auto_router_model_name="smart-router").sessions_key()
    before = redis.hashes[sessions_key][key]
    await _tick(llm_router)
    assert llm_router.completion_calls == []
    assert redis.hashes[sessions_key][key] == before


@pytest.mark.asyncio
async def test_recently_warmed_model_not_replayed_again():
    llm_router, redis = _warming_rig(redis=FakeRedisCache())
    now = time.time()
    _seed_session(redis, warmth={"fast-claude": now - 10, "smart-claude": now - 300})
    await _tick(llm_router)
    assert _replayed_models(llm_router) == ["smart-claude"]


@pytest.mark.asyncio
async def test_no_redis_store_is_noop():
    llm_router, _ = _warming_rig(redis=None)
    await _tick(llm_router, lock=FakePodLockManager())
    assert llm_router.completion_calls == []


# ---------------------------------------------------------------------------
# replay kwargs
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chat_replay_kwargs_exact():
    llm_router, redis = _warming_rig(redis=FakeRedisCache())
    _seed_session(redis, warmth={"fast-claude": time.time()})
    await _tick(llm_router)
    assert llm_router.completion_calls == [
        {
            "model": "smart-claude",
            "messages": [{"role": "user", "content": "summarize the deployment policy"}],
            "tools": None,
            "tool_choice": None,
            "max_tokens": 1,
            "stream": False,
            "cache": {"no-cache": True},
            "metadata": {
                CACHE_WARMING_REPLAY_MARKER_KEY: True,
                "user_api_key": "hash-1",
                "tags": [CACHE_WARMING_REPLAY_TAG],
            },
        }
    ]


@pytest.mark.asyncio
async def test_anthropic_surface_replays_via_aanthropic_messages_with_system():
    llm_router, redis = _warming_rig(redis=FakeRedisCache())
    _seed_session(
        redis,
        call_surface="anthropic_messages",
        warmth={"fast-claude": time.time()},
        tool_choice={"type": "auto"},
    )
    await _tick(llm_router)
    assert llm_router.completion_calls == []
    assert len(llm_router.anthropic_calls) == 1
    call = llm_router.anthropic_calls[0]
    assert call["model"] == "smart-claude"
    assert call["system"] == "You are a policy assistant"
    assert call["tool_choice"] == {"type": "auto"}
    assert call["max_tokens"] == 1
    assert call["stream"] is False
    assert call["litellm_metadata"][CACHE_WARMING_REPLAY_MARKER_KEY] is True
    assert "metadata" not in call


@pytest.mark.asyncio
async def test_tags_omitted_when_router_tag_filtering_enabled():
    llm_router, redis = _warming_rig(redis=FakeRedisCache(), enable_tag_filtering=True)
    _seed_session(redis, warmth={"fast-claude": time.time()})
    await _tick(llm_router)
    metadata = llm_router.completion_calls[0]["metadata"]
    assert "tags" not in metadata
    assert metadata[CACHE_WARMING_REPLAY_MARKER_KEY] is True


# ---------------------------------------------------------------------------
# failure isolation + attempt timestamping
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_one_session_failure_does_not_block_other_sessions():
    llm_router, redis = _warming_rig(redis=FakeRedisCache())
    llm_router.failing_message_marker = "POISON"
    _seed_session(redis, session_id="sess-bad", content="POISON payload")
    _seed_session(redis, session_id="sess-good")
    await _tick(llm_router)
    assert sorted(_replayed_models(llm_router)) == ["fast-claude", "smart-claude"]
    assert all("POISON" not in json.dumps(call) for call in llm_router.completion_calls)


@pytest.mark.asyncio
async def test_failed_replay_still_stamps_warm_attempt():
    llm_router, redis = _warming_rig(redis=FakeRedisCache())
    llm_router.failing_message_marker = "POISON"
    key = _seed_session(redis, content="POISON payload", warmth={"fast-claude": time.time()})
    await _tick(llm_router)
    stamp = _warmth_stamp(redis, key, "smart-claude")
    assert stamp is not None and stamp > 0


@pytest.mark.asyncio
async def test_successful_replay_stamps_warm_attempt_for_replayed_model_only():
    llm_router, redis = _warming_rig(redis=FakeRedisCache())
    now = time.time()
    key = _seed_session(redis, warmth={"fast-claude": now})
    await _tick(llm_router)
    smart_stamp = _warmth_stamp(redis, key, "smart-claude")
    assert smart_stamp is not None and smart_stamp >= now
    assert _warmth_stamp(redis, key, "fast-claude") == pytest.approx(now)


# ---------------------------------------------------------------------------
# budget stop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_near_budget_key_sessions_are_not_warmed():
    llm_router, redis = _warming_rig(redis=FakeRedisCache())
    _seed_session(redis, session_id="sess-broke", user_api_key="broke-key")
    _seed_session(redis, session_id="sess-rich", caller_scope="hash-2", user_api_key="rich-key")
    prisma = FakePrismaClient(
        rows=(
            SimpleNamespace(token="broke-key", spend=95.0, max_budget=100.0),
            SimpleNamespace(token="rich-key", spend=10.0, max_budget=100.0),
        )
    )
    await _tick(llm_router, prisma=prisma)
    assert sorted(_replayed_models(llm_router)) == ["fast-claude", "smart-claude"]
    assert sorted(prisma.queries[0]["token"]["in"]) == ["broke-key", "rich-key"]
    replayed_keys = {call["metadata"]["user_api_key"] for call in llm_router.completion_calls}
    assert replayed_keys == {"rich-key"}


@pytest.mark.asyncio
async def test_budget_stop_skipped_without_prisma():
    llm_router, redis = _warming_rig(redis=FakeRedisCache())
    _seed_session(redis, user_api_key="broke-key")
    await _tick(llm_router, prisma=None)
    assert sorted(_replayed_models(llm_router)) == ["fast-claude", "smart-claude"]


@pytest.mark.asyncio
async def test_budget_query_error_fails_open():
    llm_router, redis = _warming_rig(redis=FakeRedisCache())
    _seed_session(redis, user_api_key="broke-key")
    await _tick(llm_router, prisma=FakePrismaClient(raise_error=True))
    assert sorted(_replayed_models(llm_router)) == ["fast-claude", "smart-claude"]


@pytest.mark.asyncio
async def test_blocked_and_expired_keys_are_not_warmed():
    from datetime import datetime, timedelta, timezone

    llm_router, redis = _warming_rig(redis=FakeRedisCache())
    _seed_session(redis, session_id="sess-blocked", caller_scope="hash-1", user_api_key="blocked-key")
    _seed_session(redis, session_id="sess-expired", caller_scope="hash-2", user_api_key="expired-key")
    _seed_session(redis, session_id="sess-live", caller_scope="hash-3", user_api_key="live-key")
    prisma = FakePrismaClient(
        rows=(
            SimpleNamespace(token="blocked-key", spend=0.0, max_budget=None, blocked=True, expires=None),
            SimpleNamespace(
                token="expired-key",
                spend=0.0,
                max_budget=None,
                blocked=None,
                expires=datetime.now(timezone.utc) - timedelta(hours=1),
            ),
            SimpleNamespace(
                token="live-key",
                spend=0.0,
                max_budget=None,
                blocked=False,
                expires=datetime.now(timezone.utc) + timedelta(hours=1),
            ),
        )
    )
    await _tick(llm_router, prisma=prisma)
    replayed_keys = {call["metadata"]["user_api_key"] for call in llm_router.completion_calls}
    assert replayed_keys == {"live-key"}


@pytest.mark.asyncio
async def test_deleted_key_sessions_are_not_warmed():
    llm_router, redis = _warming_rig(redis=FakeRedisCache())
    _seed_session(redis, session_id="sess-ghost", user_api_key="ghost-key")
    await _tick(llm_router, prisma=FakePrismaClient(rows=()))
    assert llm_router.completion_calls == []


@pytest.mark.asyncio
async def test_master_key_sessions_warm_without_a_token_row():
    from litellm.constants import LITELLM_PROXY_MASTER_KEY_ALIAS

    llm_router, redis = _warming_rig(redis=FakeRedisCache())
    _seed_session(redis, user_api_key=LITELLM_PROXY_MASTER_KEY_ALIAS)
    await _tick(llm_router, prisma=FakePrismaClient(rows=()))
    assert sorted(_replayed_models(llm_router)) == ["fast-claude", "smart-claude"]


@pytest.mark.asyncio
async def test_key_without_max_budget_is_warmed():
    llm_router, redis = _warming_rig(redis=FakeRedisCache())
    _seed_session(redis, user_api_key="unlimited-key")
    prisma = FakePrismaClient(rows=(SimpleNamespace(token="unlimited-key", spend=10_000.0, max_budget=None),))
    await _tick(llm_router, prisma=prisma)
    assert sorted(_replayed_models(llm_router)) == ["fast-claude", "smart-claude"]


# ---------------------------------------------------------------------------
# concurrency bound
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_replays_bounded_by_semaphore():
    llm_router, redis = _warming_rig(redis=FakeRedisCache(), replay_delay=0.02)
    now = time.time()
    for i in range(6):
        _seed_session(
            redis, session_id=f"sess-{i}", caller_scope=f"hash-{i}", warmth={"fast-claude": now}
        )
    await _tick(llm_router, refresher=CacheWarmingRefresher(max_concurrent_replays=2))
    assert len(llm_router.completion_calls) == 6
    assert llm_router.max_concurrent == 2
