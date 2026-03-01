import asyncio
import logging
import multiprocessing
import os
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException

from litellm._logging import verbose_proxy_logger
from litellm.constants import (
    AIOHTTP_CONNECTOR_LIMIT,
    LITELLM_DETAILED_TIMING,
)
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.performance_endpoints.latency_tracker import (
    latency_tracker,
    per_model_tracker,
)

router = APIRouter()


@router.get(
    "/v1/performance/summary",
    tags=["performance"],
)
async def get_performance_summary(
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> Dict[str, Any]:
    """
    Returns a performance snapshot for diagnosing proxy overhead.

    Sections:
    - debug_flags: is DEBUG logging or detailed timing enabled
    - workers: cpu_count, configured num_workers, cpu_percent
    - connection_pools: in-flight requests, DB pool, Redis pool, HTTP pool
    - latency: rolling avg/p50/p95 from the last 100 requests
    - per_model: per-model overhead/llm_api/total stats
    - issues: ranked list of detected problems with proposed fixes
    """
    if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise HTTPException(
            status_code=403,
            detail="Admin access required to view performance summary",
        )

    # --- Debug flags ---
    from litellm._logging import verbose_proxy_logger as _proxy_logger
    from litellm.proxy.proxy_server import general_settings, prisma_client

    is_detailed_debug = _proxy_logger.isEnabledFor(logging.DEBUG)
    log_level = logging.getLevelName(_proxy_logger.getEffectiveLevel())
    detailed_timing_enabled = LITELLM_DETAILED_TIMING

    # --- Workers / CPU ---
    cpu_count = multiprocessing.cpu_count()
    num_workers = general_settings.get("num_workers", 1)
    cpu_percent = _get_cpu_percent()

    # --- Connection pools ---
    active_asyncio_tasks = _get_in_flight_requests()
    db_pool_info = _get_db_pool_info(general_settings, prisma_client)
    redis_pool_info = _get_redis_pool_info()
    http_pool_info = _get_http_pool_info()

    # --- Latency stats ---
    latency_stats = latency_tracker.stats()
    overhead_pct = _compute_overhead_pct(latency_stats)

    # --- Per-model stats ---
    per_model_stats = per_model_tracker.stats()

    # --- Issues ---
    summary = {
        "debug_flags": {
            "is_detailed_debug": is_detailed_debug,
            "log_level": log_level,
            "detailed_timing_enabled": detailed_timing_enabled,
        },
        "workers": {
            "cpu_count": cpu_count,
            "num_workers": num_workers,
            "cpu_percent": cpu_percent,
        },
        "connection_pools": {
            "in_flight_requests": active_asyncio_tasks,
            "db": db_pool_info,
            "redis": redis_pool_info,
            "http": http_pool_info,
        },
        "latency": {
            **latency_stats,
            "overhead_pct_of_total": overhead_pct,
        },
        "per_model": per_model_stats,
    }
    summary["issues"] = _detect_issues(summary)
    return summary


# ---------------------------------------------------------------------------
# Issue detection
# ---------------------------------------------------------------------------

_SEVERITY_ORDER = {"critical": 0, "warning": 1, "info": 2}


def _detect_issues(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []

    debug_flags = summary["debug_flags"]
    workers = summary["workers"]
    pools = summary["connection_pools"]
    latency = summary["latency"]

    # 1. Debug logging active
    if debug_flags["is_detailed_debug"]:
        issues.append(
            {
                "severity": "warning",
                "title": "Debug logging is active",
                "description": (
                    "LITELLM_LOG=DEBUG adds measurable overhead to every request "
                    "and writes verbose output. Disable in production."
                ),
                "fix": "Set LITELLM_LOG=WARNING (or unset the variable)",
                "fix_snippet": "export LITELLM_LOG=WARNING",
            }
        )

    # 2. Under-provisioned workers
    cpu_count = workers["cpu_count"]
    num_workers = workers["num_workers"]
    if num_workers < cpu_count:
        recommended = 2 * cpu_count + 1
        issues.append(
            {
                "severity": "warning",
                "title": f"Under-provisioned: {num_workers} worker{'s' if num_workers != 1 else ''} for {cpu_count} CPU cores",
                "description": (
                    f"You have {cpu_count} CPU cores but only {num_workers} "
                    f"uvicorn worker{'s' if num_workers != 1 else ''}. "
                    "Additional workers allow more requests to be handled in parallel."
                ),
                "fix": f"Set num_workers: {recommended} in your config (2× CPU + 1)",
                "fix_snippet": f"litellm --num_workers {recommended} --config config.yaml",
            }
        )

    # 3. High overhead percentage
    overhead_pct = latency.get("overhead_pct_of_total")
    overhead = latency.get("overhead")
    if overhead_pct is not None and overhead_pct > 20 and overhead:
        issues.append(
            {
                "severity": "warning",
                "title": f"High LiteLLM overhead: {overhead_pct}% of total request time",
                "description": (
                    f"LiteLLM is adding {overhead['avg_ms']}ms avg overhead "
                    f"(p95: {overhead['p95_ms']}ms). Normal is <5%. "
                    "Common causes: debug logging, DB connection pool exhaustion, or too few workers."
                ),
                "fix": "Check the other issues on this page — high overhead is usually a symptom",
                "fix_snippet": None,
            }
        )

    # 4. High p95 overhead (even if avg looks okay)
    if overhead and overhead.get("p95_ms", 0) > 200:
        issues.append(
            {
                "severity": "warning",
                "title": f"High p95 overhead: {overhead['p95_ms']}ms",
                "description": (
                    "p95 overhead is elevated even if the average looks acceptable. "
                    "This means 1 in 20 requests experiences significant proxy-added latency. "
                    "Likely cause: occasional DB pool queuing or GC pauses."
                ),
                "fix": "Consider increasing database_connection_pool_limit and num_workers",
                "fix_snippet": None,
            }
        )

    # 6. HTTP pool near saturation
    http = pools.get("http", {})
    http_pct = http.get("aiohttp_pct")
    if http_pct is not None and http_pct > 80:
        issues.append(
            {
                "severity": "critical",
                "title": f"HTTP connection pool near capacity: {http_pct}% used",
                "description": (
                    f"aiohttp connector is at {http_pct}% utilization "
                    f"({http.get('aiohttp_active')} / {http.get('aiohttp_limit')} connections). "
                    "New outbound requests will queue until connections free up."
                ),
                "fix": "Increase AIOHTTP_CONNECTOR_LIMIT environment variable",
                "fix_snippet": f"export AIOHTTP_CONNECTOR_LIMIT={http.get('aiohttp_limit', 300) * 2}",
            }
        )

    # Sort: critical first, then warning, then info
    issues.sort(key=lambda i: _SEVERITY_ORDER.get(i["severity"], 99))
    return issues


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_cpu_percent() -> Optional[float]:
    try:
        import psutil

        return psutil.cpu_percent(interval=None)
    except Exception:
        return None


def _get_in_flight_requests() -> Optional[int]:
    try:
        active = sum(1 for t in asyncio.all_tasks() if not t.done())
        return active
    except Exception:
        return None


def _get_db_pool_info(
    general_settings: dict, prisma_client: Any
) -> Dict[str, Any]:
    pool_limit = general_settings.get("database_connection_pool_limit", 10)
    pool_timeout = general_settings.get("database_connection_pool_timeout", 60)
    connected = prisma_client is not None
    return {
        "connected": connected,
        "pool_limit": pool_limit,
        "pool_timeout_seconds": pool_timeout,
    }


def _get_redis_pool_info() -> Dict[str, Any]:
    try:
        from litellm.proxy.proxy_server import redis_usage_cache

        if redis_usage_cache is None:
            return {"enabled": False}
        result: Dict[str, Any] = {"enabled": True}
        try:
            if (
                hasattr(redis_usage_cache, "redis_client")
                and redis_usage_cache.redis_client
            ):
                pool = getattr(
                    redis_usage_cache.redis_client, "connection_pool", None
                )
                if pool is not None:
                    result["max_connections"] = getattr(
                        pool, "max_connections", None
                    )
        except Exception:
            pass
        return result
    except Exception:
        return {"enabled": False}


def _get_http_pool_info() -> Dict[str, Any]:
    """
    Returns aiohttp connector pool stats (configured limit + active connections).
    Active connections are summed across all cached AsyncHTTPHandler clients.
    """
    result: Dict[str, Any] = {
        "aiohttp_limit": AIOHTTP_CONNECTOR_LIMIT,
        "aiohttp_active": None,
        "aiohttp_pct": None,
    }
    try:
        import litellm
        from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

        cache = getattr(litellm, "in_memory_llm_clients_cache", None)
        if cache is None:
            return result

        # LLMClientCache extends InMemoryCache directly — cache_dict is on cache itself
        items = getattr(cache, "cache_dict", {})
        aiohttp_active = 0

        for _key, client in items.items():
            if not isinstance(client, AsyncHTTPHandler):
                continue
            httpx_client = getattr(client, "client", None)
            if httpx_client is None:
                continue
            transport = getattr(httpx_client, "_transport", None)
            if transport is None:
                continue
            # LiteLLMAiohttpTransport stores the aiohttp ClientSession as .client
            session = getattr(transport, "client", None)
            if session is None:
                continue
            connector = getattr(session, "connector", None)
            if connector is None:
                continue
            acquired = getattr(connector, "_acquired", None)
            if acquired is not None:
                aiohttp_active += len(acquired)

        result["aiohttp_active"] = aiohttp_active
        if AIOHTTP_CONNECTOR_LIMIT > 0:
            result["aiohttp_pct"] = round(
                aiohttp_active / AIOHTTP_CONNECTOR_LIMIT * 100, 1
            )
    except Exception as e:
        verbose_proxy_logger.debug(f"Error getting HTTP pool info: {e}")

    return result


def _compute_overhead_pct(latency_stats: dict) -> Optional[float]:
    overhead = latency_stats.get("overhead")
    total = latency_stats.get("total")
    if not overhead or not total:
        return None
    avg_overhead = overhead.get("avg_ms")
    avg_total = total.get("avg_ms")
    if avg_overhead and avg_total and avg_total > 0:
        return round((avg_overhead / avg_total) * 100, 1)
    return None
