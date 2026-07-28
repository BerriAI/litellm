import asyncio
import time
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import TYPE_CHECKING

from pydantic import BaseModel, TypeAdapter

from litellm._logging import verbose_router_logger
from litellm.constants import CACHE_WARMING_JOB_NAME, LITELLM_PROXY_MASTER_KEY_ALIAS
from litellm.router_strategy.complexity_router.cache_warming.eligibility import resolve_warm_models
from litellm.router_strategy.complexity_router.cache_warming.store import CacheWarmingStore
from litellm.router_strategy.complexity_router.cache_warming.types import (
    CACHE_WARMING_REPLAY_MARKER_KEY,
    CACHE_WARMING_REPLAY_TAG,
    CacheWarmingRecord,
    decompress_payload,
)

if TYPE_CHECKING:
    from litellm.caching.redis_cache import RedisCache
    from litellm.proxy.db.db_transaction_queue.pod_lock_manager import PodLockManager
    from litellm.proxy.utils import PrismaClient
    from litellm.router import Router
    from litellm.router_strategy.complexity_router.complexity_router import ComplexityRouter

CACHE_WARMING_MAX_CONCURRENT_REPLAYS = 10
CACHE_WARMING_LOCK_TTL_SECONDS = 60
CACHE_WARMING_BUDGET_STOP_FRACTION = 0.95

_ATTRIBUTION_ADAPTER: TypeAdapter[Mapping[str, str | None]] = TypeAdapter(Mapping[str, str | None])


class _TokenBudgetRow(BaseModel):
    token: str
    spend: float = 0.0
    max_budget: float | None = None
    blocked: bool | None = None
    expires: datetime | None = None


_TOKEN_ROWS_ADAPTER: TypeAdapter[tuple[_TokenBudgetRow, ...]] = TypeAdapter(tuple[_TokenBudgetRow, ...])


def _excluded_from_warming(row: _TokenBudgetRow, now: float) -> bool:
    if row.blocked is True:
        return True
    if row.expires is not None and row.expires.timestamp() <= now:
        return True
    return row.max_budget is not None and row.spend >= CACHE_WARMING_BUDGET_STOP_FRACTION * row.max_budget


def collect_warming_enabled_complexity_routers(llm_router: "Router") -> tuple["ComplexityRouter", ...]:
    return tuple(
        tagged.strategy
        for tagged_list in llm_router.complexity_routers.values()
        for tagged in tagged_list
        if tagged.strategy.config.cache_warming.enabled
    )


def _deployment_provider(litellm_params: Mapping[str, object], deployment_model: str) -> str | None:
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    declared = litellm_params.get("custom_llm_provider")
    if isinstance(declared, str) and declared:
        return declared
    try:
        _, provider, _, _ = get_llm_provider(model=deployment_model)
    except Exception:  # noqa: BLE001  # unroutable deployment just isn't warmable
        return None
    return provider


def _group_is_cache_warmable(llm_router: "Router", model_group: str) -> bool:
    from litellm.utils import supports_prompt_caching

    deployments = llm_router.get_model_list(model_name=model_group) or []
    for deployment in deployments:
        litellm_params = deployment.get("litellm_params") or {}  # pyright: ignore[reportUnknownMemberType]  # DeploymentTypedDict fields are legacy-untyped
        deployment_model = litellm_params.get("model")  # pyright: ignore[reportUnknownMemberType]  # DeploymentTypedDict fields are legacy-untyped
        if not isinstance(deployment_model, str):
            continue
        provider = _deployment_provider(litellm_params, deployment_model)
        if provider not in ("anthropic", "bedrock"):
            continue
        if supports_prompt_caching(model=deployment_model, custom_llm_provider=provider):
            return True
    return False


def filter_cache_warmable(llm_router: "Router", model_groups: Sequence[str]) -> tuple[str, ...]:
    return tuple(group for group in model_groups if _group_is_cache_warmable(llm_router, group))


class CacheWarmingRefresher:
    def __init__(self, max_concurrent_replays: int = CACHE_WARMING_MAX_CONCURRENT_REPLAYS) -> None:
        self.max_concurrent_replays = max_concurrent_replays
        self._fallback_lock_manager: PodLockManager | None = None

    def _resolve_lock_manager(
        self, injected: "PodLockManager | None", redis_cache: "RedisCache | None"
    ) -> "PodLockManager":
        if injected is not None and injected.redis_cache is not None:
            return injected
        if self._fallback_lock_manager is None or self._fallback_lock_manager.redis_cache is not redis_cache:
            from litellm.proxy.db.db_transaction_queue.pod_lock_manager import PodLockManager

            self._fallback_lock_manager = PodLockManager(redis_cache=redis_cache)
        return self._fallback_lock_manager

    async def run_tick(
        self,
        *,
        llm_router: "Router",
        pod_lock_manager: "PodLockManager | None",
        prisma_client: "PrismaClient | None",
    ) -> None:
        warming_routers = collect_warming_enabled_complexity_routers(llm_router)
        if not warming_routers:
            return
        warmable = tuple(
            (complexity_router, store)
            for complexity_router in warming_routers
            if (store := complexity_router.get_cache_warming_store()) is not None and store.redis_cache is not None
        )
        if not warmable:
            return
        lock_manager = self._resolve_lock_manager(pod_lock_manager, warmable[0][1].redis_cache)
        acquired = await lock_manager.acquire_lock(
            cronjob_id=CACHE_WARMING_JOB_NAME, ttl=CACHE_WARMING_LOCK_TTL_SECONDS
        )
        if not acquired:
            return
        try:
            for complexity_router, store in warmable:
                await self._warm_router_sessions(
                    llm_router=llm_router,
                    complexity_router=complexity_router,
                    store=store,
                    prisma_client=prisma_client,
                )
        finally:
            await lock_manager.release_lock(cronjob_id=CACHE_WARMING_JOB_NAME)

    async def _warm_router_sessions(
        self,
        *,
        llm_router: "Router",
        complexity_router: "ComplexityRouter",
        store: CacheWarmingStore,
        prisma_client: "PrismaClient | None",
    ) -> None:
        config = complexity_router.config.cache_warming
        session_keys = await store.list_session_keys(max_sessions=config.max_sessions)
        if not session_keys:
            return
        if len(session_keys) >= config.max_sessions:
            verbose_router_logger.debug(
                "cache_warming: auto-router %s is at its max_sessions cap (%s); "
                "new sessions are not admitted until existing ones expire",
                complexity_router.model_name,
                config.max_sessions,
            )
        now = time.time()
        records = tuple([(key, await store.get_record(key)) for key in session_keys])
        active = tuple(
            (key, record)
            for key, record in records
            if record is not None and now - record.last_activity <= config.idle_timeout_seconds
        )
        if not active:
            return
        warm_models = filter_cache_warmable(llm_router, resolve_warm_models(complexity_router.config))
        if not warm_models:
            return
        excluded_keys = await self._excluded_key_hashes(
            prisma_client,
            frozenset(
                record.attribution.user_api_key for _, record in active if record.attribution.user_api_key is not None
            ),
        )
        semaphore = asyncio.Semaphore(self.max_concurrent_replays)
        await asyncio.gather(
            *(
                self._warm_session(
                    llm_router=llm_router,
                    store=store,
                    session_key=key,
                    record=record,
                    warm_models=warm_models,
                    refresh_interval_seconds=config.refresh_interval_seconds,
                    session_ttl_seconds=config.session_ttl_seconds,
                    semaphore=semaphore,
                )
                for key, record in active
                if record.attribution.user_api_key not in excluded_keys
            )
        )

    async def _warm_session(
        self,
        *,
        llm_router: "Router",
        store: CacheWarmingStore,
        session_key: str,
        record: CacheWarmingRecord,
        warm_models: tuple[str, ...],
        refresh_interval_seconds: int,
        session_ttl_seconds: int,
        semaphore: asyncio.Semaphore,
    ) -> None:
        warmth = await store.get_warmth(session_key, warm_models)
        now = time.time()
        due_models = tuple(model for model in warm_models if now - warmth.get(model, 0.0) >= refresh_interval_seconds)
        for model_group in due_models:
            async with semaphore:
                attempted_at = time.time()
                try:
                    await self._replay(llm_router=llm_router, record=record, model_group=model_group)
                except Exception:  # noqa: BLE001  # one failing replay must not abort the tick
                    verbose_router_logger.warning(
                        "cache_warming replay failed for session %s model %s",
                        session_key,
                        model_group,
                        exc_info=True,
                    )
                finally:
                    await store.mark_warm_attempt(session_key, model_group, attempted_at, session_ttl_seconds)

    async def _replay(self, *, llm_router: "Router", record: CacheWarmingRecord, model_group: str) -> None:
        payload = decompress_payload(record.payload_compressed)
        attribution = _ATTRIBUTION_ADAPTER.validate_python(record.attribution.model_dump())
        metadata = {  # mutable-ok: request metadata handed to the router call, never retained
            CACHE_WARMING_REPLAY_MARKER_KEY: True,
            **{key: value for key, value in attribution.items() if value is not None},
            **(
                {"tags": [CACHE_WARMING_REPLAY_TAG]} if llm_router.enable_tag_filtering is not True else {}
            ),  # mutable-ok: request metadata, never retained
        }
        messages = [dict(message) for message in payload.messages]  # mutable-ok: router call input, never retained
        if payload.call_surface == "anthropic_messages":
            system = list(payload.system) if isinstance(payload.system, tuple) else payload.system
            await llm_router.aanthropic_messages(  # pyright: ignore[reportUnknownMemberType]  # factory-generated router surface is legacy-untyped
                model=model_group,
                messages=messages,
                system=system,
                tools=list(payload.tools) if payload.tools is not None else None,
                tool_choice=dict(payload.tool_choice)
                if isinstance(payload.tool_choice, Mapping)
                else payload.tool_choice,  # mutable-ok: router call input, never retained
                max_tokens=1,
                stream=False,
                cache={"no-cache": True},  # mutable-ok: router call input, never retained
                litellm_metadata=metadata,
            )
            return
        await llm_router.acompletion(  # pyright: ignore[reportUnknownMemberType, reportCallIssue]  # router overloads are legacy-untyped
            model=model_group,
            messages=messages,  # pyright: ignore[reportArgumentType]  # replay forwards the captured wire shape verbatim
            tools=list(payload.tools) if payload.tools is not None else None,
            tool_choice=dict(payload.tool_choice)
            if isinstance(payload.tool_choice, Mapping)
            else payload.tool_choice,  # mutable-ok: router call input, never retained
            max_tokens=1,
            stream=False,
            cache={"no-cache": True},  # mutable-ok: router call input, never retained
            metadata=metadata,
        )

    @staticmethod
    async def _excluded_key_hashes(prisma_client: "PrismaClient | None", key_hashes: frozenset[str]) -> frozenset[str]:
        if prisma_client is None or not key_hashes:
            return frozenset()
        lookup = frozenset(key for key in key_hashes if key != LITELLM_PROXY_MASTER_KEY_ALIAS)
        if not lookup:
            return frozenset()
        try:
            rows = _TOKEN_ROWS_ADAPTER.validate_python(
                await prisma_client.db.litellm_verificationtoken.find_many(  # pyright: ignore[reportAny]  # prisma client is legacy-untyped
                    where={"token": {"in": list(lookup)}}  # mutable-ok: prisma query input, never retained
                ),
                from_attributes=True,
            )
        except Exception:  # noqa: BLE001  # budget stop fails open; warming continues on query errors
            verbose_router_logger.warning("cache_warming budget check failed; warming continues", exc_info=True)
            return frozenset()
        now = time.time()
        usable = frozenset(row.token for row in rows if not _excluded_from_warming(row, now))
        return lookup - usable
