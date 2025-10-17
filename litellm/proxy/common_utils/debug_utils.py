# Start tracing memory allocations
import asyncio
import gc
import json
import os
import sys
import tracemalloc
from collections import Counter
from typing import Dict, List, Any, Optional

from fastapi import APIRouter, Depends, Query

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


@router.get("/debug/memory/summary", include_in_schema=False)
async def get_memory_summary(
    _: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> Dict[str, Any]:
    """
        
    This endpoint provides a simplified, actionable view of your proxy's memory usage
    with clear recommendations on what to do if memory is high.
    
    Returns:
    - Current memory usage (MB) and percentage
    - Cache sizes and item counts  
    - Active object counts
    - Health status and recommendations
    
    Example:
    ```bash
    curl http://localhost:4000/debug/memory/summary \\
      -H "Authorization: Bearer sk-1234"
    ```
    
    Next steps based on results:
    - If memory is high: Call POST /debug/memory/gc/collect to free memory
    - For detailed analysis: Call GET /debug/memory/details
    - To clear caches: Use the cache management endpoints
    """
    from litellm.proxy.proxy_server import (
        llm_router,
        proxy_logging_obj,
        user_api_key_cache,
    )
    
    # Get process memory info
    process_memory = {}
    recommendations = []
    health_status = "healthy"
    
    try:
        import psutil
        
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / (1024 * 1024)
        memory_percent = process.memory_percent()
        
        process_memory = {
            "current_usage_mb": round(memory_mb, 2),
            "percent_of_system": round(memory_percent, 2),
            "readable": f"{memory_mb:.1f} MB ({memory_percent:.1f}%)",
        }
        
        # Health assessment
        if memory_percent > 80:
            health_status = "critical"
            recommendations.append("⚠️ Memory usage is critically high! Run POST /debug/memory/gc/collect immediately")
            recommendations.append("Consider restarting the proxy or clearing caches")
        elif memory_percent > 60:
            health_status = "warning"
            recommendations.append("Memory usage is elevated. Consider running garbage collection")
        else:
            health_status = "healthy"
            recommendations.append("✅ Memory usage is normal")
            
    except ImportError:
        process_memory["error"] = "Install psutil for memory monitoring: pip install psutil"
        recommendations.append("Install psutil to monitor process memory: pip install psutil")
    except Exception as e:
        process_memory["error"] = str(e)
    
    # Get cache information
    caches = {}
    total_cache_items = 0
    
    try:
        # User API key cache
        user_cache_items = len(user_api_key_cache.in_memory_cache.cache_dict)
        total_cache_items += user_cache_items
        caches["user_api_keys"] = {
            "items": user_cache_items,
            "description": "Cached API key validations"
        }
        
        # Router cache
        if llm_router is not None:
            router_cache_items = len(llm_router.cache.in_memory_cache.cache_dict)
            total_cache_items += router_cache_items
            caches["llm_responses"] = {
                "items": router_cache_items,
                "description": "Cached LLM responses"
            }
        
        # Proxy logging cache  
        logging_cache_items = len(
            proxy_logging_obj.internal_usage_cache.dual_cache.in_memory_cache.cache_dict
        )
        total_cache_items += logging_cache_items
        caches["usage_tracking"] = {
            "items": logging_cache_items,
            "description": "Cached usage metrics"
        }
        
        if total_cache_items > 10000:
            recommendations.append(f"You have {total_cache_items:,} items in cache. Consider reducing cache TTL settings")
            
    except Exception as e:
        caches["error"] = str(e)
    
    # Get GC stats
    gc_info = {
        "enabled": gc.isenabled(),
        "object_counts": gc.get_count(),
        "uncollectable_objects": len(gc.garbage),
    }
    
    if gc_info["uncollectable_objects"] > 0:
        recommendations.append(f"⚠️ {gc_info['uncollectable_objects']} uncollectable objects detected. This may indicate a memory leak")
    
    return {
        "status": health_status,
        "memory": process_memory,
        "caches": {
            "total_items": total_cache_items,
            "breakdown": caches,
        },
        "garbage_collector": gc_info,
        "recommendations": recommendations if recommendations else ["✅ Everything looks good!"],
        "next_steps": {
            "free_memory": "POST /debug/memory/gc/collect",
            "detailed_stats": "GET /debug/memory/details",
            "cache_contents": "GET /memory-usage-in-mem-cache-items",
        }
    }


@router.get("/debug/memory/details", include_in_schema=False)
async def get_memory_details(
    _: UserAPIKeyAuth = Depends(user_api_key_auth),
    top_n: int = Query(20, description="Number of top object types to return"),
    include_process_info: bool = Query(True, description="Include process memory info"),
) -> Dict[str, Any]:
    """
    🔍 Detailed memory diagnostics for deep debugging
    
    Returns comprehensive information about:
    - Garbage collector statistics (collections per generation, thresholds)
    - Object counts by type (find memory-heavy objects)
    - Process memory breakdown (RSS, VMS, file descriptors)
    - Cache memory usage with exact sizes
    - Uncollectable objects (potential memory leaks)
    
    Query Parameters:
    - top_n: Number of top object types to return (default: 20)
    - include_process_info: Include process-level memory info using psutil (default: true)
    
    Example:
    ```bash
    curl "http://localhost:4000/debug/memory/details?top_n=30" \\
      -H "Authorization: Bearer sk-1234"
    ```
    
    💡 Tip: Call POST /debug/memory/gc/collect first, then check details to see what remains
    """
    from litellm.proxy.proxy_server import (
        llm_router,
        proxy_logging_obj,
        user_api_key_cache,
    )
    
    # Get GC statistics
    gc_stats = {
        "enabled": gc.isenabled(),
        "thresholds": {
            "generation_0": gc.get_threshold()[0],
            "generation_1": gc.get_threshold()[1],
            "generation_2": gc.get_threshold()[2],
            "explanation": "Number of allocations before automatic collection for each generation"
        },
        "current_counts": {
            "generation_0": gc.get_count()[0],
            "generation_1": gc.get_count()[1],
            "generation_2": gc.get_count()[2],
            "explanation": "Current number of allocated objects in each generation"
        },
        "collection_history": [
            {
                "generation": i,
                "total_collections": stat["collections"],
                "total_collected": stat["collected"],
                "uncollectable": stat["uncollectable"],
            }
            for i, stat in enumerate(gc.get_stats())
        ],
    }
    
    # Count objects by type
    type_counts: Counter = Counter()
    total_objects = 0
    
    for obj in gc.get_objects():
        total_objects += 1
        obj_type = type(obj).__name__
        type_counts[obj_type] += 1
    
    # Get top N object types with readable counts
    top_object_types = [
        {
            "type": obj_type, 
            "count": count,
            "count_readable": f"{count:,}"
        }
        for obj_type, count in type_counts.most_common(top_n)
    ]
    
    # Get uncollectable objects (potential memory leaks)
    uncollectable = gc.garbage
    uncollectable_info = {
        "count": len(uncollectable),
        "sample_types": [type(obj).__name__ for obj in uncollectable[:10]],  # First 10
        "warning": "If count > 0, you may have reference cycles preventing garbage collection" if len(uncollectable) > 0 else None,
    }
    
    # Calculate cache memory usage
    cache_stats = {}
    try:
        # User API key cache
        user_cache_size = sys.getsizeof(user_api_key_cache.in_memory_cache.cache_dict)
        user_ttl_size = sys.getsizeof(user_api_key_cache.in_memory_cache.ttl_dict)
        cache_stats["user_api_key_cache"] = {
            "num_items": len(user_api_key_cache.in_memory_cache.cache_dict),
            "cache_dict_size_bytes": user_cache_size,
            "ttl_dict_size_bytes": user_ttl_size,
            "total_size_mb": round((user_cache_size + user_ttl_size) / (1024 * 1024), 2),
        }
        
        # Router cache
        if llm_router is not None:
            router_cache_size = sys.getsizeof(llm_router.cache.in_memory_cache.cache_dict)
            router_ttl_size = sys.getsizeof(llm_router.cache.in_memory_cache.ttl_dict)
            cache_stats["llm_router_cache"] = {
                "num_items": len(llm_router.cache.in_memory_cache.cache_dict),
                "cache_dict_size_bytes": router_cache_size,
                "ttl_dict_size_bytes": router_ttl_size,
                "total_size_mb": round((router_cache_size + router_ttl_size) / (1024 * 1024), 2),
            }
        
        # Proxy logging cache
        logging_cache_size = sys.getsizeof(
            proxy_logging_obj.internal_usage_cache.dual_cache.in_memory_cache.cache_dict
        )
        logging_ttl_size = sys.getsizeof(
            proxy_logging_obj.internal_usage_cache.dual_cache.in_memory_cache.ttl_dict
        )
        cache_stats["proxy_logging_cache"] = {
            "num_items": len(
                proxy_logging_obj.internal_usage_cache.dual_cache.in_memory_cache.cache_dict
            ),
            "cache_dict_size_bytes": logging_cache_size,
            "ttl_dict_size_bytes": logging_ttl_size,
            "total_size_mb": round((logging_cache_size + logging_ttl_size) / (1024 * 1024), 2),
        }
    except Exception as e:
        verbose_proxy_logger.debug(f"Error calculating cache stats: {e}")
        cache_stats["error"] = str(e)
    
    # Process memory info (requires psutil)
    process_info = None
    if include_process_info:
        try:
            import psutil
            
            process = psutil.Process()
            memory_info = process.memory_info()
            process_info = {
                "rss_mb": round(memory_info.rss / (1024 * 1024), 2),  # Resident Set Size (actual RAM usage)
                "vms_mb": round(memory_info.vms / (1024 * 1024), 2),  # Virtual Memory Size
                "percent": round(process.memory_percent(), 2),
                "num_fds": process.num_fds() if hasattr(process, "num_fds") else None,
                "num_threads": process.num_threads(),
                "readable": f"Using {memory_info.rss / (1024 * 1024):.1f} MB of RAM ({process.memory_percent():.1f}% of system)",
            }
        except ImportError:
            process_info = {
                "error": "psutil not installed. Install with: pip install psutil"
            }
        except Exception as e:
            verbose_proxy_logger.debug(f"Error getting process info: {e}")
            process_info = {"error": str(e)}
    
    return {
        "garbage_collector": gc_stats,
        "objects": {
            "total_tracked": total_objects,
            "total_tracked_readable": f"{total_objects:,}",
            "top_types": top_object_types,
        },
        "uncollectable": uncollectable_info,
        "cache_memory": cache_stats,
        "process_memory": process_info,
    }


@router.post("/debug/memory/gc/collect", include_in_schema=False)
async def trigger_gc_collection(
    _: UserAPIKeyAuth = Depends(user_api_key_auth),
    full: bool = Query(True, description="Run full collection (all generations)"),
) -> Dict[str, Any]:
    """
    🧹 Free up memory by running garbage collection
    
    This endpoint manually triggers Python's garbage collector to reclaim memory from
    unreferenced objects. Use this when memory usage is high or growing over time.
    
    Query Parameters:
    - full: Run full GC across all generations (default: true, recommended)
           Set to false for a quick gen-0 only collection
    
    Returns:
    - Number of objects freed
    - Memory usage before/after (if psutil available)
    - Recommendations for next steps
    
    Example:
    ```bash
    # Full collection (recommended for most cases)
    curl -X POST "http://localhost:4000/debug/memory/gc/collect" \\
      -H "Authorization: Bearer sk-1234"
    
    # Quick collection (gen-0 only)
    curl -X POST "http://localhost:4000/debug/memory/gc/collect?full=false" \\
      -H "Authorization: Bearer sk-1234"
    ```
    
    💡 After running this:
    - Check GET /debug/memory/summary to see the impact
    - If memory is still high, check GET /debug/memory/details for what's using memory
    """
    # Get memory before (if psutil available)
    memory_before = None
    memory_after = None
    
    try:
        import psutil
        process = psutil.Process()
        memory_before = {
            "rss_mb": round(process.memory_info().rss / (1024 * 1024), 2),
            "percent": round(process.memory_percent(), 2),
        }
    except:
        pass
    
    # Get counts before collection
    counts_before = gc.get_count()
    
    # Run garbage collection
    if full:
        # Full collection: run all generations
        collected_gen0 = gc.collect(0)
        collected_gen1 = gc.collect(1)
        collected_gen2 = gc.collect(2)
        total_collected = collected_gen0 + collected_gen1 + collected_gen2
        collection_type = "full (all generations)"
    else:
        # Quick collection: just gen-0
        total_collected = gc.collect(0)
        collection_type = "quick (generation 0)"
    
    # Get counts after collection
    counts_after = gc.get_count()
    
    # Get memory after
    try:
        import psutil
        process = psutil.Process()
        memory_after = {
            "rss_mb": round(process.memory_info().rss / (1024 * 1024), 2),
            "percent": round(process.memory_percent(), 2),
        }
    except:
        pass
    
    # Calculate memory freed
    memory_freed_mb = None
    if memory_before and memory_after:
        memory_freed_mb = round(memory_before["rss_mb"] - memory_after["rss_mb"], 2)
    
    # Build response
    result = {
        "success": True,
        "collection_type": collection_type,
        "objects_collected": total_collected,
        "objects_collected_readable": f"{total_collected:,}",
        "object_counts": {
            "before": {
                "generation_0": counts_before[0],
                "generation_1": counts_before[1],
                "generation_2": counts_before[2],
            },
            "after": {
                "generation_0": counts_after[0],
                "generation_1": counts_after[1],
                "generation_2": counts_after[2],
            },
        },
    }
    
    if memory_before and memory_after:
        result["memory_impact"] = {
            "before_mb": memory_before["rss_mb"],
            "after_mb": memory_after["rss_mb"],
            "freed_mb": memory_freed_mb,
            "percent_before": memory_before["percent"],
            "percent_after": memory_after["percent"],
            "readable": f"Freed {abs(memory_freed_mb)} MB" if memory_freed_mb and memory_freed_mb > 0 else "No significant memory freed",
        }
    
    # Add helpful message
    if total_collected == 0:
        result["message"] = "✅ No objects collected - memory is already optimized"
        result["recommendations"] = [
            "If memory is still high, check GET /debug/memory/details to see what's using memory",
            "Large caches may be holding memory - check GET /memory-usage-in-mem-cache"
        ]
    elif total_collected < 100:
        result["message"] = f"✅ Collected {total_collected} objects - minor cleanup"
    elif total_collected < 1000:
        result["message"] = f"✅ Collected {total_collected} objects - moderate cleanup"
    else:
        result["message"] = f"🧹 Collected {total_collected:,} objects - significant cleanup!"
        
    result["next_steps"] = {
        "check_summary": "GET /debug/memory/summary",
        "detailed_analysis": "GET /debug/memory/details",
    }
    
    return result


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
