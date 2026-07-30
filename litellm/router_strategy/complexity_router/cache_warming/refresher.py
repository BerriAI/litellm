import asyncio
import contextlib
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import litellm
from litellm._logging import verbose_router_logger
from litellm.constants import CACHE_WARMING_JOB_NAME, LITELLM_PROXY_MASTER_KEY_ALIAS
from litellm.integrations.anthropic_cache_control_hook import EXPLICIT_PROMPT_CACHING_PROVIDERS
from litellm.router_strategy.complexity_router.cache_warming.eligibility import resolve_warm_models
from litellm.router_strategy.complexity_router.cache_warming.store import CacheWarmingStore
from litellm.router_strategy.complexity_router.cache_warming.types import (
    CACHE_WARMING_REPLAY_MARKER_KEY,
    CACHE_WARMING_REPLAY_TAG,
    CacheWarmingPayload,
    CacheWarmingRecord,
    decompress_payload,
    needs_rewarming,
    warn_once,
)

if TYPE_CHECKING:
    from litellm.caching.dual_cache import DualCache
    from litellm.proxy._experimental.mcp_server.outbound_credentials.redis_distributed_lock import (
        RedisDistributedLock,
    )
    from litellm.proxy._types import (
        LiteLLM_EndUserTable,
        LiteLLM_ProjectTableCachedObj,
        LiteLLM_TeamTable,
        LiteLLM_UserTable,
        UserAPIKeyAuth,
    )
    from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
    from litellm.proxy.utils import PrismaClient, ProxyLogging
    from litellm.router import Router
    from litellm.router_strategy.complexity_router.complexity_router import ComplexityRouter
    from litellm.types.utils import CallTypesLiteral

CACHE_WARMING_MAX_CONCURRENT_REPLAYS = 10
CACHE_WARMING_LOCK_TTL_SECONDS = 60
_RECONSTRUCTED_IDENTITY_FIELDS = ("user_id", "team_id", "org_id", "project_id")

_REPLAY_MAX_OUTPUT_TOKENS = 1


def _replay_surface(payload: CacheWarmingPayload) -> "tuple[str, CallTypesLiteral]":
    """Route and call type as the owning endpoints declare them, so hooks branching on either see the
    surface being replayed."""
    if payload.call_surface == "anthropic_messages":
        return ("/v1/messages", "anthropic_messages")
    return ("/v1/chat/completions", "acompletion")


def _replay_principal(key_state: "UserAPIKeyAuth | None", record: CacheWarmingRecord, route: str) -> "UserAPIKeyAuth":
    """A replay's principal must be COMPLETE, because the authorization enumerations resolve tenancy by id and
    a missing field is an absent ceiling rather than a denial. Three shapes of capture, one reconstruction:

    (a) virtual key: key_state was just read authoritatively from the database, so it is already complete
    (b) keyless proxy caller (JWT, and anything else the proxy authenticated without a virtual key, where
        api_key is None): UserAPIKeyAuth is litellm's principal type for those callers too and the proxy
        stamped their tenancy, which the record carries, so every recorded identity field is restored below and
        the team, user, organization and project gates resolve by id exactly as for a virtual key. Dropping any
        of them is what previously let a JWT caller's replays run against an empty principal and escape every
        tenant control
    (c) direct SDK use with no proxy auth object: nothing was recorded, so there is no tenancy to preserve and
        the principal is genuinely limitless, which is the same unattributed warming as before

    get_key_object fills token but not api_key, and every metadata consumer reads api_key, so it is set from
    the token. end_user_id is request-scoped and survives only on the record."""
    from litellm.proxy._types import UserAPIKeyAuth

    attribution = record.attribution
    base = key_state if key_state is not None else UserAPIKeyAuth(api_key=attribution.user_api_key)
    return base.model_copy(
        update={  # mutable-ok: pydantic model_copy input, never retained
            "api_key": base.api_key or base.token,
            **{
                field: getattr(base, field) or getattr(attribution, f"user_api_key_{field}")
                for field in _RECONSTRUCTED_IDENTITY_FIELDS
            },
            "end_user_id": attribution.user_api_key_end_user_id or base.end_user_id,
            "request_route": route,
            "budget_reservation": None,
        }
    )


def _replay_body(
    payload: CacheWarmingPayload, record: CacheWarmingRecord, model_group: str, route: str
) -> "dict[str, object]":
    """no-cache is only sent when a response cache exists to bypass, because it is a cache control the key
    may be forbidden to set (cache_control_check.py:36) and would otherwise refuse the replay for nothing.
    Anthropic invalidates a cached prefix when tool_choice changes, so it and system ride along there.
    litellm_call_id is per replay because the hanging-request checker keys its cache on it, and the empty
    default would collide every replay onto one entry. The warming marker rides spend_logs_metadata, not
    metadata.tags: tags are an input to deployment selection (enable_tag_filtering makes an unmatched tag
    unroutable) and to policy (_reject_clientside_metadata_tags_check refuses any request carrying them),
    while spend_logs_metadata exists to label spend rows, which is all this marker is for. Fallbacks are disabled at the
    dispatch site rather than in the body so no key-level or pre-call mutation can re-enable them: everything
    warming validated -- prompt-cache eligibility, every-member cacheability, deployment affinity, pricing --
    is a property of the target GROUP, and a fallback substitutes a different group carrying none of it,
    spending the customer's money to warm nothing. router.py:6157 raises before fallbacks,
    context_window_fallbacks and content_policy_fallbacks are consulted, so one flag covers all three; a
    failed replay simply retries next tick."""
    from litellm.litellm_core_utils.core_helpers import get_metadata_variable_name_from_kwargs
    from litellm.proxy.litellm_pre_call_utils import LiteLLMProxyRequestSetup

    data: dict[str, object] = {  # mutable-ok: the request body the proxy entry points mutate in place
        "model": model_group,
        "messages": [dict(message) for message in payload.messages],
        "tools": list(payload.tools) if payload.tools is not None else None,
        "tool_choice": dict(payload.tool_choice) if isinstance(payload.tool_choice, Mapping) else payload.tool_choice,
        "max_tokens": _REPLAY_MAX_OUTPUT_TOKENS,
        "stream": False,
        "litellm_call_id": uuid.uuid4().hex,
        **({"cache": {"no-cache": True}} if litellm.cache is not None else {}),
        **(
            {"system": list(payload.system) if isinstance(payload.system, tuple) else payload.system}
            if payload.call_surface == "anthropic_messages"
            else {}
        ),
    }
    LiteLLMProxyRequestSetup.pre_seed_litellm_metadata_for_route(request_data=data, route=route)
    data[get_metadata_variable_name_from_kwargs(data)] = {  # mutable-ok: request metadata, never retained
        CACHE_WARMING_REPLAY_MARKER_KEY: True,
        **({"session_id": record.session_id} if record.session_id is not None else {}),
        "spend_logs_metadata": {CACHE_WARMING_REPLAY_TAG: "true"},  # mutable-ok: request metadata, never retained
    }
    return data


async def _authorize_replay(
    *,
    principal: "UserAPIKeyAuth",
    data: "dict[str, object]",
    model_group: str,
    route: str,
    team: "LiteLLM_TeamTable | None",
    user: "LiteLLM_UserTable | None",
    end_user: "LiteLLM_EndUserTable | None",
    project: "LiteLLM_ProjectTableCachedObj | None",
    llm_router: "Router",
    prisma_client: "PrismaClient",
    user_api_key_cache: "UserApiKeyCache",
    proxy_logging_obj: "ProxyLogging",
    skip_budget_checks: bool,
) -> None:
    """Authorization is not re-enumerated here. Production has exactly two enumerations of it, and a replay
    runs both: ``_enforce_key_and_fallback_model_access`` for the key-level model allowlist and its fallback
    targets, then ``common_checks`` for everything else (team blocked, team and member and user and project
    model access, every budget scope including organization and tags, the global proxy budget, guardrail
    modification, organization RBAC, vector stores, tool allowlists), plus the per-model budget gates the
    auth builder owns for the key and the end user. The per-model gate is applied without the builder's
    budget_fallbacks rewrite, because rerouting a replay to a different model would warm a cache nothing is
    going to read; over budget means this group is simply not warmed.

    Both accept ``request=None`` by declared contract, because every Request dereference inside them is
    already guarded (``_safe_get_request_headers``, ``_safe_get_request_query_params``,
    ``get_request_route_template``, ``_enforce_user_param_check``). The one exception is the agent trace-id
    header gate, which is reached only for a key whose agent sets ``require_trace_id_on_calls_by_agent`` and
    which raises there; that denies the replay, which is the outcome a replay should get from a gate demanding
    a client header it cannot have.

    Calling the enumerations rather than copying them is the point: three separate review rounds found a gate
    a hand-written list had missed, and a hand-written list will always be one gate behind."""
    from litellm.proxy.auth.auth_checks import common_checks
    from litellm.proxy.auth.user_api_key_auth import (
        _enforce_key_and_fallback_model_access,  # pyright: ignore[reportPrivateUsage]  # the key-level enumeration; its own docstring notes common_checks excludes these
        get_global_proxy_spend,
    )
    from litellm.proxy.proxy_server import general_settings, litellm_proxy_admin_name, model_max_budget_limiter

    await _enforce_key_and_fallback_model_access(
        valid_token=principal,
        request_data=data,
        route=route,
        request=None,
        llm_model_list=llm_router.model_list,
        llm_router=llm_router,
    )
    await model_max_budget_limiter.is_key_within_model_budget(user_api_key_dict=principal, model=model_group)
    end_user_model_budgets = principal.end_user_model_max_budget
    if principal.end_user_id is not None and isinstance(end_user_model_budgets, dict) and end_user_model_budgets:
        await model_max_budget_limiter.is_end_user_within_model_budget(
            end_user_id=principal.end_user_id,
            end_user_model_max_budget=end_user_model_budgets,
            model=model_group,
        )
    await common_checks(
        request_body=data,
        team_object=team,
        user_object=user,
        end_user_object=end_user,
        global_proxy_spend=await get_global_proxy_spend(
            litellm_proxy_admin_name=litellm_proxy_admin_name,
            user_api_key_cache=user_api_key_cache,
            prisma_client=prisma_client,
            token=principal.token or "",
            proxy_logging_obj=proxy_logging_obj,
        ),
        general_settings=general_settings,
        route=route,
        llm_router=llm_router,
        proxy_logging_obj=proxy_logging_obj,
        valid_token=principal,
        request=None,
        skip_budget_checks=skip_budget_checks,
        project_object=project,
    )


def _stamp_identity(data: "dict[str, object]", principal: "UserAPIKeyAuth") -> None:
    """The request path's own three stampers in its order. Load-bearing: add_key_level_controls resets
    data["cache"] and refills it from key metadata, so running after _replay_body lets the key's declared
    controls win over warming's bypass exactly as they win over a caller's, and the reservation must
    already be on the principal because metadata is the channel the cost callback reconciles it through
    (proxy_track_cost_callback.py:446)."""
    from litellm.litellm_core_utils.core_helpers import get_metadata_variable_name_from_kwargs
    from litellm.proxy.litellm_pre_call_utils import LiteLLMProxyRequestSetup
    from litellm.proxy.proxy_server import proxy_config

    metadata_key = get_metadata_variable_name_from_kwargs(data)
    LiteLLMProxyRequestSetup.add_user_api_key_auth_to_request_metadata(
        data=data, user_api_key_dict=principal, _metadata_variable_name=metadata_key
    )
    LiteLLMProxyRequestSetup.add_key_team_project_metadata(
        data=data, user_api_key_dict=principal, _metadata_variable_name=metadata_key
    )
    LiteLLMProxyRequestSetup.apply_dynamic_logging_settings(
        data=data, user_api_key_dict=principal, proxy_config=proxy_config
    )


def _excluded_from_warming(key_state: "UserAPIKeyAuth", now: "datetime", proxy_logging_obj: "ProxyLogging") -> bool:
    """Mirrors the canonical auth checks (user_api_key_auth.py:1717 and :2891) because common_checks owns
    that policy for real traffic but needs a FastAPI Request. datetime.fromisoformat only accepts a
    Z-suffixed expires from 3.11 while requires-python is 3.10, so the offset is normalized and an
    unparseable value excludes rather than admits. A declared max-iterations ceiling excludes too: that
    hook counts every request on a session_id with no read-only mode, so warming would spend the caller's
    own loop budget."""
    from litellm.proxy.hooks.max_iterations_limiter import _PROXY_MaxIterationsHandler

    if key_state.blocked is True:
        return True
    iterations_hook = proxy_logging_obj.get_proxy_hook("max_iterations_limiter")
    if (
        isinstance(iterations_hook, _PROXY_MaxIterationsHandler)
        and iterations_hook._get_max_iterations(key_state) is not None  # pyright: ignore[reportPrivateUsage]  # the hook owns this predicate and exposes no public form
    ):
        warn_once(
            "cache_warming: a key declares max_iterations, whose limiter counts every request on a "
            "session_id and cannot be consulted without incrementing it; warming would spend the "
            "caller's own iteration budget, so sessions on such keys are skipped"
        )
        return True
    expires = key_state.expires
    if expires is None:
        return False
    try:
        expiry = expires if isinstance(expires, datetime) else datetime.fromisoformat(expires.replace("Z", "+00:00"))
    except ValueError:
        return True
    if expiry.tzinfo is None or expiry.tzinfo.utcoffset(expiry) is None:
        expiry = expiry.replace(tzinfo=timezone.utc)
    return expiry < now


def _missing_enforcement_dependency(
    prisma_client: "PrismaClient | None",
    user_api_key_cache: "DualCache | None",
    proxy_logging_obj: "ProxyLogging",
) -> str | None:
    """Names the first thing a replay's admission needs and cannot reach, or None. Every ceiling warming
    claims to respect is enforced by one of these, so losing any single one turns admission into a no-op that
    still spends the customer's money. They are therefore checked once, together, before any session is
    considered, rather than as per-dependency arms that each fail open on their own path."""
    from litellm.proxy.hooks.parallel_request_limiter_v3 import _PROXY_MaxParallelRequestsHandler_v3

    if prisma_client is None:
        return "the database client (key, team, user and project state)"
    if user_api_key_cache is None:
        return "the key cache (authorization objects and budget counters)"
    if not isinstance(
        proxy_logging_obj.get_proxy_hook("parallel_request_limiter"), _PROXY_MaxParallelRequestsHandler_v3
    ):
        return "the v3 parallel-request limiter (rate limits)"
    from litellm.proxy.proxy_server import spend_counter_cache

    if spend_counter_cache is None:
        return "the shared spend counter plane (budget reservation and reconciliation)"
    return None


def _deny_tick(missing: str) -> None:
    warn_once(
        f"cache_warming cannot enforce a replay's ceilings because {missing} is unavailable; no session is "
        "warmed until it is reachable again, because admission without it would spend against limits and "
        "budgets it cannot check"
    )


def _resolve_proxy_logging() -> "ProxyLogging":
    from litellm.proxy.proxy_server import proxy_logging_obj

    return proxy_logging_obj


def _redis_lease_lock(redis_cache: object) -> "RedisDistributedLock":
    """The MCP outbound-credentials distributed lock; promoting it to a neutral home is a follow-up."""
    from litellm.proxy._experimental.mcp_server.outbound_credentials.redis_distributed_lock import (
        RedisDistributedLock,
    )

    return RedisDistributedLock(
        redis_cache.init_async_client(),  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownArgumentType]  # RedisCache is legacy-untyped
        namespace_key=redis_cache.check_and_fix_namespace,  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownArgumentType]  # RedisCache is legacy-untyped
    )


async def _proxy_key_state(
    hashed_token: str, prisma_client: "PrismaClient", user_api_key_cache: "DualCache"
) -> "UserAPIKeyAuth":
    """Read past the local cache tier, not cache-first. Revocation clears the local entry and the Redis entry
    (_delete_cache_key_object, auth_checks.py:1855) but cannot reach another pod's in-memory tier, so a
    cache-first read can still see a key blocked, expired or deleted seconds ago as valid and keep billing it.
    DualCache.async_get_cache offers no local-skip (its local_only flag skips Redis, the opposite), so the
    authoritative row is the way past it; get_key_object writes the result back through _cache_key_object, so
    the platform's cache shape is unchanged. Cost is one query per DISTINCT attributed key per 30-second tick,
    bounded by max_sessions (default 1000). Warming is therefore strictly fresher than real request traffic on
    another pod, which carries the same in-memory staleness."""
    from litellm.proxy.auth.auth_checks import get_key_object

    return await get_key_object(
        hashed_token=hashed_token,
        prisma_client=prisma_client,
        user_api_key_cache=user_api_key_cache,  # pyright: ignore[reportArgumentType]  # UserApiKeyCache is a DualCache alias
        parent_otel_span=None,
        proxy_logging_obj=None,
        check_cache_only=None,
        check_db_only=True,
    )


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


def _session_affinity_active(llm_router: "Router", model_group: str) -> bool:
    from litellm.router_utils.pre_call_checks.deployment_affinity_check import DeploymentAffinityCheck

    return any(
        callback._get_effective_flags(model_group)[2]  # pyright: ignore[reportPrivateUsage]  # no public accessor today; upstream should expose get_effective_flags
        for callback in (llm_router.optional_callbacks or [])  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]  # Router.optional_callbacks is legacy-untyped
        if isinstance(callback, DeploymentAffinityCheck)
    )


def _group_is_cache_warmable(llm_router: "Router", model_group: str) -> bool:
    """Every selectable member must be warmable, because replays route by group name and the Router may
    pick any member."""
    from litellm.utils import supports_prompt_caching

    deployments = llm_router.get_model_list(model_name=model_group) or []
    if not deployments:
        return False
    if len(deployments) > 1 and not _session_affinity_active(llm_router, model_group):
        warn_once(
            f"cache_warming: model group {model_group} has {len(deployments)} deployments and deployment "
            "affinity with session_id is not active for it, so warming is skipped for this group; warming "
            "without affinity cannot produce cache hits reliably because a replay warms one member's cache "
            "while real traffic routes across all of them. Enable it globally with "
            'router_settings.optional_pre_call_checks: ["session_affinity"] or for this group only with '
            f'router_settings.model_group_affinity_config: {{"{model_group}": ["session_affinity"]}}'
        )
        return False
    return all(
        isinstance(model, str)
        and (provider := _deployment_provider(params, model)) in EXPLICIT_PROMPT_CACHING_PROVIDERS
        and supports_prompt_caching(model=model, custom_llm_provider=provider)
        for deployment in deployments
        for params in ((deployment.get("litellm_params") or {}),)  # pyright: ignore[reportUnknownMemberType]  # DeploymentTypedDict fields are legacy-untyped
        for model in (params.get("model"),)  # pyright: ignore[reportUnknownMemberType]  # DeploymentTypedDict fields are legacy-untyped
    )


def filter_cache_warmable(llm_router: "Router", model_groups: Sequence[str]) -> tuple[str, ...]:
    return tuple(group for group in model_groups if _group_is_cache_warmable(llm_router, group))


class CacheWarmingRefresher:
    def __init__(
        self,
        max_concurrent_replays: int = CACHE_WARMING_MAX_CONCURRENT_REPLAYS,
        lock_ttl_seconds: float = CACHE_WARMING_LOCK_TTL_SECONDS,
        lock_factory: 'Callable[[object], "RedisDistributedLock"]' = _redis_lease_lock,
        key_state_resolver: 'Callable[[str, "PrismaClient", "DualCache"], Awaitable["UserAPIKeyAuth"]]' = _proxy_key_state,
        proxy_logging_resolver: 'Callable[[], "ProxyLogging"]' = _resolve_proxy_logging,
    ) -> None:
        self.max_concurrent_replays = max_concurrent_replays
        self.lock_ttl_seconds = lock_ttl_seconds
        self.lock_factory = lock_factory
        self.key_state_resolver = key_state_resolver
        self.proxy_logging_resolver = proxy_logging_resolver

    async def _hold_lease(self, lock: "RedisDistributedLock", token: str, lease_lost: asyncio.Event) -> None:
        while True:
            await asyncio.sleep(self.lock_ttl_seconds / 2)
            if not await lock.extend(CACHE_WARMING_JOB_NAME, token, self.lock_ttl_seconds):
                verbose_router_logger.warning(
                    "cache_warming pod lock was lost mid tick; finishing in-flight replays without admitting new ones"
                )
                lease_lost.set()
                return

    async def run_tick(
        self,
        *,
        llm_router: "Router",
        prisma_client: "PrismaClient | None",
        user_api_key_cache: "DualCache | None" = None,
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
        from litellm.proxy._experimental.mcp_server.outbound_credentials.redis_refresh_coordinator import (
            LockAcquisition,
        )

        redis_cache = warmable[0][1].redis_cache
        if redis_cache is None:
            return
        lock = self.lock_factory(redis_cache)
        token = uuid.uuid4().hex
        acquisition = await lock.acquire(CACHE_WARMING_JOB_NAME, token, self.lock_ttl_seconds)
        if acquisition is not LockAcquisition.ACQUIRED:
            if acquisition is LockAcquisition.ERROR:
                _deny_tick(
                    "the pod lock backend (leader election, without which every pod warms concurrently and "
                    "warmth stamps cannot dedupe because each pod reads them before any pod writes)"
                )
            return
        lease_lost = asyncio.Event()
        lease = asyncio.create_task(self._hold_lease(lock, token, lease_lost))
        try:
            for complexity_router, store in warmable:
                await self._warm_router_sessions(
                    llm_router=llm_router,
                    complexity_router=complexity_router,
                    store=store,
                    prisma_client=prisma_client,
                    user_api_key_cache=user_api_key_cache,
                    lease_lost=lease_lost,
                )
        finally:
            lease.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await lease
            await lock.release(CACHE_WARMING_JOB_NAME, token)

    async def _warm_router_sessions(
        self,
        *,
        llm_router: "Router",
        complexity_router: "ComplexityRouter",
        store: CacheWarmingStore,
        prisma_client: "PrismaClient | None",
        user_api_key_cache: "DualCache | None",
        lease_lost: asyncio.Event,
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
        attributed = frozenset(
            record.attribution.user_api_key
            for _, record in active
            if record.attribution.user_api_key is not None
            and record.attribution.user_api_key != LITELLM_PROXY_MASTER_KEY_ALIAS
        )
        proxy_logging_obj = self.proxy_logging_resolver()
        missing = _missing_enforcement_dependency(prisma_client, user_api_key_cache, proxy_logging_obj)
        if missing is not None:
            _deny_tick(missing)
            return
        key_states = await self._fetch_key_states(prisma_client, user_api_key_cache, attributed)
        if key_states is None:
            return
        checked_at = datetime.now(timezone.utc)
        excluded_keys = frozenset(
            key
            for key in attributed
            if (row := key_states.get(key)) is None or _excluded_from_warming(row, checked_at, proxy_logging_obj)
        )
        semaphore = asyncio.Semaphore(self.max_concurrent_replays)
        outcomes = await asyncio.gather(
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
                    key_state=key_states.get(record.attribution.user_api_key or ""),
                    prisma_client=prisma_client,
                    user_api_key_cache=user_api_key_cache,
                    proxy_logging_obj=proxy_logging_obj,
                    lease_lost=lease_lost,
                )
                for key, record in active
                if record.attribution.user_api_key not in excluded_keys
            ),
            return_exceptions=True,
        )
        for outcome in outcomes:
            if isinstance(outcome, BaseException):
                verbose_router_logger.warning("cache_warming session task failed", exc_info=outcome)

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
        key_state: "UserAPIKeyAuth | None",
        prisma_client: "PrismaClient | None",
        user_api_key_cache: "DualCache | None",
        proxy_logging_obj: "ProxyLogging",
        lease_lost: asyncio.Event,
    ) -> None:
        warmth = await store.get_warmth(session_key, warm_models)
        now = time.time()
        due_models = tuple(
            model
            for model in warm_models
            if needs_rewarming(warmth[model].at if model in warmth else 0.0, now, refresh_interval_seconds)
        )
        if not due_models:
            return
        payload = decompress_payload(record.payload_compressed)
        for model_group in due_models:
            async with semaphore:
                if lease_lost.is_set():
                    return
                attempted_at = time.time()
                warmed = await self._replay_once(
                    llm_router=llm_router,
                    payload=payload,
                    record=record,
                    session_key=session_key,
                    model_group=model_group,
                    key_state=key_state,
                    prisma_client=prisma_client,
                    user_api_key_cache=user_api_key_cache,
                    proxy_logging_obj=proxy_logging_obj,
                )
                await store.mark_warm_attempt(
                    session_key, model_group, attempted_at, session_ttl_seconds, warmed=warmed
                )

    async def _replay_once(
        self,
        *,
        llm_router: "Router",
        payload: CacheWarmingPayload,
        record: CacheWarmingRecord,
        session_key: str,
        model_group: str,
        key_state: "UserAPIKeyAuth | None",
        prisma_client: "PrismaClient | None",
        user_api_key_cache: "DualCache | None",
        proxy_logging_obj: "ProxyLogging",
    ) -> bool:
        """True only when the provider accepted the replay, because a warmth stamp is the claim that the
        provider now holds this session's prefix on this model. A refused or failed replay leaves it cold, so
        it is stamped as attempted-but-cold: the pick keeps treating the model as cold, while the attempt
        still paces the next tick instead of retrying a failing model every 30 seconds."""
        admitted = await self._admit_replay(
            llm_router=llm_router,
            payload=payload,
            record=record,
            model_group=model_group,
            key_state=key_state,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            proxy_logging_obj=proxy_logging_obj,
        )
        if admitted is None:
            return False
        data, principal = admitted
        data["disable_fallbacks"] = True
        try:
            await (
                llm_router.aanthropic_messages(**data)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # factory-generated router surface is legacy-untyped
                if payload.call_surface == "anthropic_messages"
                else llm_router.acompletion(**data)  # pyright: ignore[reportUnknownMemberType, reportCallIssue, reportUnknownVariableType, reportArgumentType]  # router overloads are legacy-untyped
            )
        except Exception as exc:  # noqa: BLE001  # one failing replay must not abort the tick
            verbose_router_logger.warning(
                "cache_warming replay failed for session %s model %s", session_key, model_group, exc_info=True
            )
            await self._report_rejection(data=data, principal=principal, exc=exc, proxy_logging_obj=proxy_logging_obj)
            return False
        await proxy_logging_obj.update_request_status(
            litellm_call_id=str(data.get("litellm_call_id") or ""), status="success"
        )
        return True

    async def _admit_replay(
        self,
        *,
        llm_router: "Router",
        payload: CacheWarmingPayload,
        record: CacheWarmingRecord,
        model_group: str,
        key_state: "UserAPIKeyAuth | None",
        prisma_client: "PrismaClient | None",
        user_api_key_cache: "DualCache | None",
        proxy_logging_obj: "ProxyLogging",
    ) -> "tuple[dict[str, object], UserAPIKeyAuth] | None":
        """A replay is a request, so it enters the same entry points real traffic does and inherits both
        halves of every contract they own rather than reimplementing any of them. Order mirrors the request
        path: budget reservation, identity stamping, the model group's own guardrails
        (common_request_processing.py:1268), pre_call_hook, then the blocked-model gate route_request
        applies before the router call (route_llm_request.py:453). A rejection skips only this replay."""
        from litellm.proxy.route_llm_request import (
            _raise_if_model_fully_blocked,  # pyright: ignore[reportPrivateUsage]  # the gate route_request itself calls
        )
        from litellm.proxy.utils import _check_and_merge_model_level_guardrails

        route, call_type = _replay_surface(payload)
        principal = _replay_principal(key_state, record, route)
        data = _replay_body(payload, record, model_group, route)
        try:
            await self._authorize_and_reserve(
                principal=principal,
                data=data,
                model_group=model_group,
                route=route,
                llm_router=llm_router,
                prisma_client=prisma_client,
                user_api_key_cache=user_api_key_cache,
                proxy_logging_obj=proxy_logging_obj,
            )
            _stamp_identity(data, principal)
            admitted = await proxy_logging_obj.pre_call_hook(
                user_api_key_dict=principal,
                data=_check_and_merge_model_level_guardrails(
                    data=data, llm_router=llm_router, trust_client_model_info=False
                ),
                call_type=call_type,
            )
            _raise_if_model_fully_blocked(llm_router, model_group, principal.team_id)
            return (admitted, principal)
        except Exception as exc:  # noqa: BLE001  # a declined replay skips only itself
            verbose_router_logger.debug(
                "cache_warming: the request path declined this replay on %s: %s", model_group, exc
            )
            await self._report_rejection(data=data, principal=principal, exc=exc, proxy_logging_obj=proxy_logging_obj)
            return None

    @staticmethod
    async def _authorize_and_reserve(
        *,
        principal: "UserAPIKeyAuth",
        data: "dict[str, object]",
        model_group: str,
        route: str,
        llm_router: "Router",
        prisma_client: "PrismaClient | None",
        user_api_key_cache: "DualCache",
        proxy_logging_obj: "ProxyLogging",
    ) -> None:
        """Authorization then reservation, the order and the owners auth itself uses
        (user_api_key_auth.py:2380-2396). The context objects come from the same resolvers because both
        halves are derived from them: the team, user, end-user, organization and project counters, and the
        team, project and per-member authorization gates. Passing None would leave those ceilings unbound and
        those gates unenforced while the cost callback still charged the same scopes. These stay cache-first, so
        they carry the same in-memory staleness a real request on another pod carries; only the key, which is
        the credential warming spends against, is read past the local tier."""
        from litellm.proxy._types import ProxyErrorTypes, ProxyException
        from litellm.proxy.auth.auth_checks import (
            get_end_user_object,
            get_project_object,
            get_team_object,
            get_user_object,
            user_is_scim_deactivated,
        )
        from litellm.proxy.auth.user_api_key_auth import (
            _reserve_budget_after_common_checks,  # pyright: ignore[reportPrivateUsage]  # the flow owner; reserve_budget_for_request alone drops the operator settings
            _should_skip_budget_checks,  # pyright: ignore[reportPrivateUsage]  # same owner, and Request-optional
        )
        from litellm.proxy.proxy_server import general_settings

        cache: UserApiKeyCache = user_api_key_cache  # pyright: ignore[reportAssignmentType]  # UserApiKeyCache is a DualCache alias
        team = (
            await get_team_object(team_id=principal.team_id, prisma_client=prisma_client, user_api_key_cache=cache)
            if principal.team_id is not None
            else None
        )
        user = (
            await get_user_object(
                user_id=principal.user_id, prisma_client=prisma_client, user_api_key_cache=cache, user_id_upsert=False
            )
            if principal.user_id is not None
            else None
        )
        if user_is_scim_deactivated(user):
            raise ProxyException(
                message=f"User={principal.user_id} has been deactivated via SCIM. Keys owned by this user cannot be used.",
                type=ProxyErrorTypes.auth_error,
                param="user_id",
                code=401,
            )
        end_user = (
            await get_end_user_object(
                end_user_id=principal.end_user_id, prisma_client=prisma_client, user_api_key_cache=cache
            )
            if principal.end_user_id is not None
            else None
        )
        project = (
            await get_project_object(
                project_id=principal.project_id, prisma_client=prisma_client, user_api_key_cache=cache
            )
            if principal.project_id is not None
            else None
        )
        skip_budget_checks = _should_skip_budget_checks(
            request_data=data, route=route, request=None, llm_router=llm_router
        )
        await _authorize_replay(
            principal=principal,
            data=data,
            model_group=model_group,
            route=route,
            team=team,
            user=user,
            end_user=end_user,
            project=project,
            llm_router=llm_router,
            prisma_client=prisma_client,
            user_api_key_cache=cache,
            proxy_logging_obj=proxy_logging_obj,
            skip_budget_checks=skip_budget_checks,
        )
        await _reserve_budget_after_common_checks(
            user_api_key_auth_obj=principal,
            request_data=data,
            route=route,
            llm_router=llm_router,
            team_object=team,
            user_object=user,
            prisma_client=prisma_client,
            user_api_key_cache=cache,
            proxy_logging_obj=proxy_logging_obj,
            skip_budget_checks=skip_budget_checks,
            general_settings=general_settings,
            end_user_id=principal.end_user_id,
            end_user_object=end_user,
        )

    @staticmethod
    async def _report_rejection(
        *,
        data: "dict[str, object]",
        principal: "UserAPIKeyAuth",
        exc: Exception,
        proxy_logging_obj: "ProxyLogging",
    ) -> None:
        """The same entry point the proxy calls when a request fails after pre_call_hook
        (common_request_processing.py:2538); it returns the parallel slot, the reserved TPM tokens and the
        budget reservation, which would otherwise be stranded until their TTLs and 429 real traffic.
        Idempotent via the callbacks' own released markers."""
        with contextlib.suppress(Exception):
            await proxy_logging_obj.post_call_failure_hook(
                request_data=data, original_exception=exc, user_api_key_dict=principal
            )

    async def _fetch_key_states(
        self,
        prisma_client: "PrismaClient | None",
        user_api_key_cache: "DualCache | None",
        key_hashes: frozenset[str],
    ) -> "Mapping[str, UserAPIKeyAuth] | None":
        """Cache-aware key state via the proxy resolver; a key whose lookup raises (deleted keys raise
        token_not_found_in_db) is absent from the map and therefore excluded, isolating one bad key."""
        if not key_hashes:
            return {}

        async def _resolve(hashed_token: str) -> "tuple[str, UserAPIKeyAuth | None]":
            try:
                return (hashed_token, await self.key_state_resolver(hashed_token, prisma_client, user_api_key_cache))
            except Exception:  # noqa: BLE001  # an unverifiable key stays fail-closed for that key only
                verbose_router_logger.warning(
                    "cache_warming could not verify a key's state; skipping its sessions this tick", exc_info=True
                )
                return (hashed_token, None)

        resolved = await asyncio.gather(*(_resolve(hashed_token) for hashed_token in key_hashes))
        return {hashed_token: state for hashed_token, state in resolved if state is not None}
