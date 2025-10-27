# Start tracing memory allocations
import asyncio
import json
import os
import tracemalloc
from collections import Counter

from fastapi import APIRouter, Depends

from litellm import get_secret_str
from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

router = APIRouter()


@router.get("/debug/asyncio-tasks")
async def get_active_tasks_stats():
    """
    Returns:
      total_active_tasks: int
      by_name: { coroutine_name: count }
    """
    MAX_TASKS_TO_CHECK = 5000
    # Gather all tasks in this event loop (including this endpoint’s own task).
    all_tasks = asyncio.all_tasks()

    # Filter out tasks that are already done.
    active_tasks = [t for t in all_tasks if not t.done()]

    # Count how many active tasks exist, grouped by coroutine function name.
    counter = Counter()
    for idx, task in enumerate(active_tasks):

        # reasonable max circuit breaker
        if idx >= MAX_TASKS_TO_CHECK:
            break
        coro = task.get_coro()
        # Derive a human‐readable name from the coroutine:
        name = (
            getattr(coro, "__qualname__", None)
            or getattr(coro, "__name__", None)
            or repr(coro)
        )
        counter[name] += 1

    return {
        "total_active_tasks": len(active_tasks),
        "by_name": dict(counter),
    }


if os.environ.get("LITELLM_PROFILE", "false").lower() == "true":
    try:
        import objgraph  # type: ignore

        print("growth of objects")  # noqa
        objgraph.show_growth()
        print("\n\nMost common types")  # noqa
        objgraph.show_most_common_types()
        roots = objgraph.get_leaking_objects()
        print("\n\nLeaking objects")  # noqa
        objgraph.show_most_common_types(objects=roots)
    except ImportError:
        raise ImportError(
            "objgraph not found. Please install objgraph to use this feature."
        )

    tracemalloc.start(10)

    @router.get("/memory-usage", include_in_schema=False)
    async def memory_usage():
        # Take a snapshot of the current memory usage
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics("lineno")
        verbose_proxy_logger.debug("TOP STATS: %s", top_stats)

        # Get the top 50 memory usage lines
        top_50 = top_stats[:50]
        result = []
        for stat in top_50:
            result.append(f"{stat.traceback.format(limit=10)}: {stat.size / 1024} KiB")

        return {"top_50_memory_usage": result}


@router.get("/memory-usage-in-mem-cache", include_in_schema=False)
async def memory_usage_in_mem_cache(
    _: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    # returns the size of all in-memory caches on the proxy server
    """
    1. user_api_key_cache
    2. router_cache
    3. proxy_logging_cache
    4. internal_usage_cache
    """
    from litellm.proxy.proxy_server import (
        llm_router,
        proxy_logging_obj,
        user_api_key_cache,
    )

    if llm_router is None:
        num_items_in_llm_router_cache = 0
    else:
        num_items_in_llm_router_cache = len(
            llm_router.cache.in_memory_cache.cache_dict
        ) + len(llm_router.cache.in_memory_cache.ttl_dict)

    num_items_in_user_api_key_cache = len(
        user_api_key_cache.in_memory_cache.cache_dict
    ) + len(user_api_key_cache.in_memory_cache.ttl_dict)

    num_items_in_proxy_logging_obj_cache = len(
        proxy_logging_obj.internal_usage_cache.dual_cache.in_memory_cache.cache_dict
    ) + len(proxy_logging_obj.internal_usage_cache.dual_cache.in_memory_cache.ttl_dict)

    return {
        "num_items_in_user_api_key_cache": num_items_in_user_api_key_cache,
        "num_items_in_llm_router_cache": num_items_in_llm_router_cache,
        "num_items_in_proxy_logging_obj_cache": num_items_in_proxy_logging_obj_cache,
    }


@router.get("/memory-usage-in-mem-cache-items", include_in_schema=False)
async def memory_usage_in_mem_cache_items(
    _: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    # returns the size of all in-memory caches on the proxy server
    """
    1. user_api_key_cache
    2. router_cache
    3. proxy_logging_cache
    4. internal_usage_cache
    """
    from litellm.proxy.proxy_server import (
        llm_router,
        proxy_logging_obj,
        user_api_key_cache,
    )

    if llm_router is None:
        llm_router_in_memory_cache_dict = {}
        llm_router_in_memory_ttl_dict = {}
    else:
        llm_router_in_memory_cache_dict = llm_router.cache.in_memory_cache.cache_dict
        llm_router_in_memory_ttl_dict = llm_router.cache.in_memory_cache.ttl_dict

    return {
        "user_api_key_cache": user_api_key_cache.in_memory_cache.cache_dict,
        "user_api_key_ttl": user_api_key_cache.in_memory_cache.ttl_dict,
        "llm_router_cache": llm_router_in_memory_cache_dict,
        "llm_router_ttl": llm_router_in_memory_ttl_dict,
        "proxy_logging_obj_cache": proxy_logging_obj.internal_usage_cache.dual_cache.in_memory_cache.cache_dict,
        "proxy_logging_obj_ttl": proxy_logging_obj.internal_usage_cache.dual_cache.in_memory_cache.ttl_dict,
    }


@router.get("/otel-spans", include_in_schema=False)
async def get_otel_spans():
    from litellm.proxy.proxy_server import open_telemetry_logger

    if open_telemetry_logger is None:
        return {
            "otel_spans": [],
            "spans_grouped_by_parent": {},
            "most_recent_parent": None,
        }

    otel_exporter = open_telemetry_logger.OTEL_EXPORTER
    if hasattr(otel_exporter, "get_finished_spans"):
        recorded_spans = otel_exporter.get_finished_spans()  # type: ignore
    else:
        recorded_spans = []

    print("Spans: ", recorded_spans)  # noqa

    most_recent_parent = None
    most_recent_start_time = 1000000
    spans_grouped_by_parent = {}
    for span in recorded_spans:
        if span.parent is not None:
            parent_trace_id = span.parent.trace_id
            if parent_trace_id not in spans_grouped_by_parent:
                spans_grouped_by_parent[parent_trace_id] = []
            spans_grouped_by_parent[parent_trace_id].append(span.name)

            # check time of span
            if span.start_time > most_recent_start_time:
                most_recent_parent = parent_trace_id
                most_recent_start_time = span.start_time

    # these are otel spans - get the span name
    span_names = [span.name for span in recorded_spans]
    return {
        "otel_spans": span_names,
        "spans_grouped_by_parent": spans_grouped_by_parent,
        "most_recent_parent": most_recent_parent,
    }


# Helper functions for debugging
def init_verbose_loggers():
    try:
        worker_config = get_secret_str("WORKER_CONFIG")
        # if not, assume it's a json string
        if worker_config is None:
            return
        if os.path.isfile(worker_config):
            return
        _settings = json.loads(worker_config)
        if not isinstance(_settings, dict):
            return

        debug = _settings.get("debug", None)
        detailed_debug = _settings.get("detailed_debug", None)
        if debug is True:  # this needs to be first, so users can see Router init debugg
            import logging

            from litellm._logging import (
                verbose_logger,
                verbose_proxy_logger,
                verbose_router_logger,
            )

            # this must ALWAYS remain logging.INFO, DO NOT MODIFY THIS
            verbose_logger.setLevel(level=logging.INFO)  # sets package logs to info
            verbose_router_logger.setLevel(
                level=logging.INFO
            )  # set router logs to info
            verbose_proxy_logger.setLevel(level=logging.INFO)  # set proxy logs to info
        if detailed_debug is True:
            import logging

            from litellm._logging import (
                verbose_logger,
                verbose_proxy_logger,
                verbose_router_logger,
            )

            verbose_logger.setLevel(level=logging.DEBUG)  # set package log to debug
            verbose_router_logger.setLevel(
                level=logging.DEBUG
            )  # set router logs to debug
            verbose_proxy_logger.setLevel(
                level=logging.DEBUG
            )  # set proxy logs to debug
        elif debug is False and detailed_debug is False:
            # users can control proxy debugging using env variable = 'LITELLM_LOG'
            litellm_log_setting = os.environ.get("LITELLM_LOG", "")
            if litellm_log_setting is not None:
                if litellm_log_setting.upper() == "INFO":
                    import logging

                    from litellm._logging import (
                        verbose_proxy_logger,
                        verbose_router_logger,
                    )

                    # this must ALWAYS remain logging.INFO, DO NOT MODIFY THIS

                    verbose_router_logger.setLevel(
                        level=logging.INFO
                    )  # set router logs to info
                    verbose_proxy_logger.setLevel(
                        level=logging.INFO
                    )  # set proxy logs to info
                elif litellm_log_setting.upper() == "DEBUG":
                    import logging

                    from litellm._logging import (
                        verbose_proxy_logger,
                        verbose_router_logger,
                    )

                    verbose_router_logger.setLevel(
                        level=logging.DEBUG
                    )  # set router logs to info
                    verbose_proxy_logger.setLevel(
                        level=logging.DEBUG
                    )  # set proxy logs to debug
    except Exception as e:
        import logging

        logging.warning(f"Failed to init verbose loggers: {str(e)}")
