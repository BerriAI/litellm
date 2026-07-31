"""Shared rig for the cache-warming refresher suites.

Everything here is either the production object itself or the narrowest possible stand-in for a backend
the test process has no access to (Redis, the key database, the pod lock).
"""

import asyncio
import json
import os
import time
from contextlib import contextmanager

import litellm
from litellm import Router
from litellm.caching.dual_cache import DualCache
from litellm.proxy.utils import ProxyLogging
from litellm.router_strategy.complexity_router.cache_warming.refresher import CacheWarmingRefresher
from litellm.router_strategy.complexity_router.cache_warming.store import CacheWarmingStore
from litellm.router_strategy.complexity_router.cache_warming.types import (
    CACHE_WARMING_RECORD_SCHEMA_VERSION,
    CacheWarmingAttribution,
    CacheWarmingPayload,
    CacheWarmingRecord,
    WarmthStamp,
    compress_payload,
)
from litellm.router_strategy.complexity_router.complexity_router import ComplexityRouter
from litellm.types.router import TaggedPreRoutingStrategy

from tests.test_litellm.router_strategy.complexity_router.cache_warming.test_store import FakeRedisCache

DEFAULT_MODEL_LIST = [
    {"model_name": "fast-claude", "litellm_params": {"model": "anthropic/claude-haiku-4-5", "api_key": "sk-t"}},
    {"model_name": "smart-claude", "litellm_params": {"model": "anthropic/claude-sonnet-4-5", "api_key": "sk-t"}},
]
UNIFORM_POOL = [
    {"model_name": "uniform", "litellm_params": {"model": "anthropic/claude-sonnet-4-5", "api_key": "sk-t"}},
    {"model_name": "uniform", "litellm_params": {"model": "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0"}},
]
PRICED_TIERS = {"SIMPLE": ["claude-haiku-4-5"], "COMPLEX": ["claude-sonnet-4-5"]}
PRICED_MODEL_LIST = [
    {"model_name": "claude-haiku-4-5", "litellm_params": {"model": "anthropic/claude-haiku-4-5", "api_key": "sk-t"}},
    {"model_name": "claude-sonnet-4-5", "litellm_params": {"model": "anthropic/claude-sonnet-4-5", "api_key": "sk-t"}},
]


class ReplayRouter(Router):
    """The one router double: a real litellm.Router, so every lookup the request path makes on the way to a
    replay is production code (model groups, aliases, pricing via get_model_group_info, tags, deployment
    info, blocked deployments). Only the outbound dispatch is captured instead of sent."""

    def __init__(
        self,
        model_list: list | None = None,
        redis: FakeRedisCache | None = None,
        enable_tag_filtering: bool = False,
        replay_delay: float = 0.0,
    ) -> None:
        super().__init__(
            model_list=[dict(entry) for entry in (model_list if model_list is not None else DEFAULT_MODEL_LIST)],
            enable_tag_filtering=enable_tag_filtering,
        )
        self.cache.redis_cache = redis
        self.completion_calls: list[dict] = []
        self.failed_calls: list[dict] = []
        self.anthropic_calls: list[dict] = []
        self.replay_delay = replay_delay
        self.failing_message_marker: str | None = None
        self.max_concurrent = 0
        self._in_flight = 0
        # Router assigns aanthropic_messages as an instance attribute (router.py:1211), which shadows a
        # subclass method, so the capture is installed after super().__init__ or the replay hits the network
        self.aanthropic_messages = self._capture_anthropic

    async def _capture_anthropic(self, **kwargs: object) -> None:
        self.anthropic_calls.append(kwargs)

    async def acompletion(self, **kwargs: object):  # pyright: ignore[reportIncompatibleMethodOverride]  # test double narrows the overloads
        marker = self.failing_message_marker
        if marker is not None and marker in json.dumps(kwargs.get("messages"), default=str):
            self.failed_calls.append(kwargs)
            raise RuntimeError("provider down")
        self._in_flight += 1
        self.max_concurrent = max(self.max_concurrent, self._in_flight)
        await asyncio.sleep(self.replay_delay)
        self._in_flight -= 1
        self.completion_calls.append(kwargs)


class FakeLeaseLock:
    """acquire/extend/release shaped like RedisDistributedLock."""

    def __init__(self, acquisition: object = None, extend_ok: bool = True) -> None:
        from litellm.proxy._experimental.mcp_server.outbound_credentials.redis_refresh_coordinator import (
            LockAcquisition,
        )

        self.acquisition = acquisition if acquisition is not None else LockAcquisition.ACQUIRED
        self.extend_ok = extend_ok
        self.acquire_calls: list[tuple[str, float]] = []
        self.extend_calls: list[str] = []
        self.release_calls: list[str] = []

    async def acquire(self, key: str, token: str, ttl_seconds: float) -> object:
        self.acquire_calls.append((key, ttl_seconds))
        return self.acquisition

    async def extend(self, key: str, token: str, ttl_seconds: float) -> bool:
        self.extend_calls.append(key)
        return self.extend_ok

    async def release(self, key: str, token: str) -> None:
        self.release_calls.append(key)


class FakeKeyDirectory:
    """get_key_object-shaped resolver: raises for unknown keys, like token_not_found_in_db. With no states
    every looked-up key exists as unlimited, which is the shape most cases want."""

    def __init__(self, states: dict | None = None, raise_all: bool = False) -> None:
        self.states = states
        self.raise_all = raise_all
        self.lookups: list[str] = []

    async def resolve(self, hashed_token: str, prisma_client: object, user_api_key_cache: object):
        self.lookups.append(hashed_token)
        if self.raise_all:
            raise Exception(f"Authentication Error, Invalid proxy server token passed: {hashed_token}")
        if self.states is None:
            return key_state(token=hashed_token)
        if hashed_token not in self.states:
            raise Exception(f"Authentication Error, Invalid proxy server token passed: {hashed_token}")
        return self.states[hashed_token]


def key_state(token: str = "hash-1", **fields: object):
    from litellm.proxy._types import UserAPIKeyAuth

    return UserAPIKeyAuth(token=token, **fields)


def team(team_id: str, **fields: object):
    from litellm.proxy._types import LiteLLM_TeamTableCachedObj

    return LiteLLM_TeamTableCachedObj(team_id=team_id, **fields)


def real_limiter(window_size: int | None = None) -> tuple[object, DualCache]:
    """The production v3 limiter, standalone over an in-memory DualCache, so its own counters are the
    assertion surface. window_size is read from the environment at construction."""
    from litellm.proxy.hooks.parallel_request_limiter_v3 import _PROXY_MaxParallelRequestsHandler_v3
    from litellm.proxy.utils import InternalUsageCache

    previous = os.environ.get("LITELLM_RATE_LIMIT_WINDOW_SIZE")
    if window_size is not None:
        os.environ["LITELLM_RATE_LIMIT_WINDOW_SIZE"] = str(window_size)
    try:
        counters = DualCache()
        return (_PROXY_MaxParallelRequestsHandler_v3(internal_usage_cache=InternalUsageCache(counters)), counters)
    finally:
        if window_size is not None:
            if previous is None:
                del os.environ["LITELLM_RATE_LIMIT_WINDOW_SIZE"]
            else:
                os.environ["LITELLM_RATE_LIMIT_WINDOW_SIZE"] = previous


@contextmanager
def registered_callbacks(*callbacks: object):
    """litellm.callbacks is what ProxyLogging.pre_call_hook walks, so a hook is only reachable through the
    entry point once it is registered there."""
    previous = litellm.callbacks
    litellm.callbacks = list(callbacks)
    try:
        yield
    finally:
        litellm.callbacks = previous


def warming_rig(
    redis: FakeRedisCache | None = None,
    enable_tag_filtering: bool = False,
    replay_delay: float = 0.0,
    tiers: dict | None = None,
    model_list: list | None = None,
    **cache_warming_overrides: object,
) -> tuple[ReplayRouter, FakeRedisCache | None]:
    llm_router = ReplayRouter(
        model_list=model_list, redis=redis, enable_tag_filtering=enable_tag_filtering, replay_delay=replay_delay
    )
    strategy = ComplexityRouter(
        model_name="smart-router",
        litellm_router_instance=llm_router,
        complexity_router_config={
            "tiers": tiers if tiers is not None else {"SIMPLE": ["fast-claude"], "COMPLEX": ["smart-claude"]},
            "cache_warming": {"enabled": True, **cache_warming_overrides},
        },
    )
    llm_router.complexity_routers = {"smart-router": [TaggedPreRoutingStrategy(tags=(), strategy=strategy)]}
    return llm_router, redis


def priced_rig(redis: FakeRedisCache, **overrides: object) -> ReplayRouter:
    return warming_rig(redis=redis, tiers=PRICED_TIERS, model_list=PRICED_MODEL_LIST, **overrides)[0]


def seed_session(
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
    team_id: str | None = None,
    user_id: str | None = None,
    org_id: str | None = None,
    project_id: str | None = None,
    touched: tuple[str, ...] | None = None,
    tags: tuple[str, ...] = (),
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
        session_id=session_id,
        tags=tags,
        attribution=CacheWarmingAttribution(
            user_api_key=user_api_key,
            user_api_key_team_id=team_id,
            user_api_key_user_id=user_id,
            user_api_key_org_id=org_id,
            user_api_key_project_id=project_id,
        ),
        auto_router_model_name="smart-router",
    )
    store = CacheWarmingStore(redis_cache=redis, auto_router_model_name="smart-router")
    record_key = CacheWarmingStore.record_key("smart-router", caller_scope, session_id)
    redis.hashes.setdefault(store.sessions_key(), {})[record_key] = json.dumps(record.model_dump())
    redis.zsets.setdefault(store.index_key(), {})[record_key] = time.time() + 3600
    # capture SADDs every served model, so a seeded session carries at least the one it was served on
    redis.sets.setdefault(CacheWarmingStore.touched_key(record_key), set()).update(
        touched if touched is not None else (served_model,)
    )
    for model_group, stamp in (warmth or {}).items():
        redis.data[CacheWarmingStore.warmth_key(record_key, model_group)] = json.dumps(
            WarmthStamp(at=stamp, warmed=True).model_dump()
        )
    return record_key


def warmth_stamp(redis: FakeRedisCache, record_key: str, model_group: str) -> WarmthStamp | None:
    raw = redis.data.get(CacheWarmingStore.warmth_key(record_key, model_group))
    return WarmthStamp.model_validate_json(raw) if raw is not None else None


def proxy_logging_with_hooks(limiter: object | None = None) -> ProxyLogging:
    """ProxyLogging with the hooks warming consults registered, as the proxy registers them, so a gate that
    reads one is live rather than silently absent. The v3 limiter must be present because warming denies the
    whole tick when any enforcement dependency is missing."""
    from litellm.proxy.hooks.max_iterations_limiter import _PROXY_MaxIterationsHandler
    from litellm.proxy.hooks.parallel_request_limiter_v3 import _PROXY_MaxParallelRequestsHandler_v3
    from litellm.proxy.utils import InternalUsageCache

    logging_obj = ProxyLogging(user_api_key_cache=DualCache())
    logging_obj.proxy_hook_mapping["max_iterations_limiter"] = _PROXY_MaxIterationsHandler(
        internal_usage_cache=InternalUsageCache(DualCache())
    )
    logging_obj.proxy_hook_mapping["parallel_request_limiter"] = limiter or _PROXY_MaxParallelRequestsHandler_v3(
        internal_usage_cache=InternalUsageCache(DualCache())
    )
    return logging_obj


def refresher(
    keys: FakeKeyDirectory | None = None,
    lock: FakeLeaseLock | None = None,
    proxy_logging: ProxyLogging | None = None,
    limiter: object | None = None,
    **kwargs: object,
) -> CacheWarmingRefresher:
    directory = keys if keys is not None else FakeKeyDirectory()
    lease_lock = lock if lock is not None else FakeLeaseLock()
    logging_obj = proxy_logging if proxy_logging is not None else proxy_logging_with_hooks(limiter)
    return CacheWarmingRefresher(
        key_state_resolver=directory.resolve,
        lock_factory=lambda _redis: lease_lock,
        proxy_logging_resolver=lambda: logging_obj,
        **kwargs,
    )


_DB = object()


async def tick(
    llm_router,
    prisma=_DB,
    active: CacheWarmingRefresher | None = None,
    keys: FakeKeyDirectory | None = None,
    user_api_key_cache: DualCache | None = None,
) -> None:
    """Default: a reachable DB whose directory knows every attributed key as unlimited. The key cache is a
    real DualCache because the authorization and budget entry points read through it."""
    await (active if active is not None else refresher(keys=keys)).run_tick(
        llm_router=llm_router,
        prisma_client=object() if prisma is _DB else prisma,
        # a proxy without a database still has an in-memory key cache, so the two dependencies stay
        # independently observable rather than one masking the other
        user_api_key_cache=user_api_key_cache or DualCache(),
    )


def replayed_models(llm_router: ReplayRouter) -> list:
    return [call["model"] for call in llm_router.completion_calls]


def affinity_check(model_group_affinity_config: dict | None = None, session_mode: bool = True):
    from litellm.router_utils.pre_call_checks.deployment_affinity_check import DeploymentAffinityCheck

    return DeploymentAffinityCheck(
        cache=DualCache(),
        ttl_seconds=60,
        enable_user_key_affinity=False,
        enable_responses_api_affinity=False,
        enable_session_id_affinity=session_mode,
        model_group_affinity_config=model_group_affinity_config or {},
    )
