# Start tracing memory allocations
import asyncio
import gc
import json
import os
import sys
import tracemalloc
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, Depends, HTTPException, Query

from litellm import get_secret_str
from litellm._logging import verbose_proxy_logger
from litellm.constants import PYTHON_GC_THRESHOLD
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

router = APIRouter()

# Configure garbage collection thresholds from environment variables
def configure_gc_thresholds():
    """Configure Python garbage collection thresholds from environment variables."""
    gc_threshold_env = PYTHON_GC_THRESHOLD
    if gc_threshold_env:
        try:
            # Parse threshold string like "1000,50,50"
            thresholds = [int(x.strip()) for x in gc_threshold_env.split(",")]
            if len(thresholds) == 3:
                gc.set_threshold(*thresholds)
                verbose_proxy_logger.info(f"GC thresholds set to: {thresholds}")
            else:
                verbose_proxy_logger.warning(f"GC threshold not set: {gc_threshold_env}. Expected format: 'gen0,gen1,gen2'")
        except ValueError as e:
            verbose_proxy_logger.warning(f"Failed to parse GC threshold: {gc_threshold_env}. Error: {e}")
    
    # Log current thresholds
    current_thresholds = gc.get_threshold()
    verbose_proxy_logger.info(f"Current GC thresholds: gen0={current_thresholds[0]}, gen1={current_thresholds[1]}, gen2={current_thresholds[2]}")

# Initialize GC configuration
configure_gc_thresholds()


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
    Get simplified memory usage summary for the proxy.
    
    Returns:
    - worker_pid: Process ID
    - status: Overall health based on memory usage
    - memory: Process memory usage and RAM info
    - caches: Cache item counts and descriptions
    - garbage_collector: GC status and pending object counts
    
    Example usage:
    curl http://localhost:4000/debug/memory/summary -H "Authorization: Bearer sk-1234"
    
    For detailed analysis, call GET /debug/memory/details
    For cache management, use the cache management endpoints
    """
    from litellm.proxy.proxy_server import (
        llm_router,
        proxy_logging_obj,
        user_api_key_cache,
    )
    
    # Get process memory info
    process_memory = {}
    health_status = "healthy"
    
    try:
        import psutil
        
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / (1024 * 1024)
        memory_percent = process.memory_percent()
        
        process_memory = {
            "summary": f"{memory_mb:.1f} MB ({memory_percent:.1f}% of system memory)",
            "ram_usage_mb": round(memory_mb, 2),
            "system_memory_percent": round(memory_percent, 2),
        }
        
        # Check memory health status
        if memory_percent > 80:
            health_status = "critical"
        elif memory_percent > 60:
            health_status = "warning"
        else:
            health_status = "healthy"
            
    except ImportError:
        process_memory["error"] = "Install psutil for memory monitoring: pip install psutil"
    except Exception as e:
        process_memory["error"] = str(e)
    
    # Get cache information
    caches: Dict[str, Any] = {}
    total_cache_items = 0
    
    try:
        # User API key cache
        user_cache_items = len(user_api_key_cache.in_memory_cache.cache_dict)
        total_cache_items += user_cache_items
        caches["user_api_keys"] = {
            "count": user_cache_items,
            "count_readable": f"{user_cache_items:,}",
            "what_it_stores": "Validated API keys for faster authentication"
        }
        
        # Router cache
        if llm_router is not None:
            router_cache_items = len(llm_router.cache.in_memory_cache.cache_dict)
            total_cache_items += router_cache_items
            caches["llm_responses"] = {
                "count": router_cache_items,
                "count_readable": f"{router_cache_items:,}",
                "what_it_stores": "LLM responses for identical requests"
            }
        
        # Proxy logging cache
        logging_cache_items = len(
            proxy_logging_obj.internal_usage_cache.dual_cache.in_memory_cache.cache_dict
        )
        total_cache_items += logging_cache_items
        caches["usage_tracking"] = {
            "count": logging_cache_items,
            "count_readable": f"{logging_cache_items:,}",
            "what_it_stores": "Usage metrics before database write"
        }
            
    except Exception as e:
        caches["error"] = str(e)
    
    # Get garbage collector stats
    gc_enabled = gc.isenabled()
    objects_pending = gc.get_count()[0]
    uncollectable = len(gc.garbage)
    
    gc_info = {
        "status": "enabled" if gc_enabled else "disabled",
        "objects_awaiting_collection": objects_pending,
    }
    
    # Add warning if garbage collection issues detected
    if uncollectable > 0:
        gc_info["warning"] = f"{uncollectable} uncollectable objects (possible memory leak)"
    
    return {
        "worker_pid": os.getpid(),
        "status": health_status,
        "memory": process_memory,
        "caches": {
            "total_items": total_cache_items,
            "breakdown": caches,
        },
        "garbage_collector": gc_info,
    }


def _get_gc_statistics() -> Dict[str, Any]:
    """Get garbage collector statistics."""
    return {
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


def _get_object_type_counts(top_n: int) -> Tuple[int, List[Dict[str, Any]]]:
    """Count objects by type and return total count and top N types."""
    type_counts: Counter = Counter()
    total_objects = 0
    
    for obj in gc.get_objects():
        total_objects += 1
        obj_type = type(obj).__name__
        type_counts[obj_type] += 1
    
    top_object_types = [
        {
            "type": obj_type, 
            "count": count,
            "count_readable": f"{count:,}"
        }
        for obj_type, count in type_counts.most_common(top_n)
    ]
    
    return total_objects, top_object_types


def _get_uncollectable_objects_info() -> Dict[str, Any]:
    """Get information about uncollectable objects (potential memory leaks)."""
    uncollectable = gc.garbage
    return {
        "count": len(uncollectable),
        "sample_types": [type(obj).__name__ for obj in uncollectable[:10]],
        "warning": "If count > 0, you may have reference cycles preventing garbage collection" if len(uncollectable) > 0 else None,
    }


def _get_cache_memory_stats(user_api_key_cache, llm_router, proxy_logging_obj, redis_usage_cache) -> Dict[str, Any]:
    """Calculate memory usage for all caches."""
    cache_stats: Dict[str, Any] = {}
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
        
        # Redis cache info
        if redis_usage_cache is not None:
            cache_stats["redis_usage_cache"] = {
                "enabled": True,
                "cache_type": type(redis_usage_cache).__name__,
            }
            # Try to get Redis connection pool info if available
            try:
                if hasattr(redis_usage_cache, 'redis_client') and redis_usage_cache.redis_client:
                    if hasattr(redis_usage_cache.redis_client, 'connection_pool'):
                        pool_info = redis_usage_cache.redis_client.connection_pool  # type: ignore
                        cache_stats["redis_usage_cache"]["connection_pool"] = {
                            "max_connections": pool_info.max_connections if hasattr(pool_info, 'max_connections') else None,
                            "connection_class": pool_info.connection_class.__name__ if hasattr(pool_info, 'connection_class') else None,
                        }
            except Exception as e:
                verbose_proxy_logger.debug(f"Error getting Redis pool info: {e}")
        else:
            cache_stats["redis_usage_cache"] = {"enabled": False}
            
    except Exception as e:
        verbose_proxy_logger.debug(f"Error calculating cache stats: {e}")
        cache_stats["error"] = str(e)
    
    return cache_stats


def _get_router_memory_stats(llm_router) -> Dict[str, Any]:
    """Get memory usage statistics for LiteLLM router."""
    litellm_router_memory: Dict[str, Any] = {}
    try:
        if llm_router is not None:
            # Model list memory size
            if hasattr(llm_router, 'model_list') and llm_router.model_list:
                model_list_size = sys.getsizeof(llm_router.model_list)
                litellm_router_memory["model_list"] = {
                    "num_models": len(llm_router.model_list),
                    "size_bytes": model_list_size,
                    "size_mb": round(model_list_size / (1024 * 1024), 4),
                }
                
            # Model names set
            if hasattr(llm_router, 'model_names') and llm_router.model_names:
                model_names_size = sys.getsizeof(llm_router.model_names)
                litellm_router_memory["model_names_set"] = {
                    "num_model_groups": len(llm_router.model_names),
                    "size_bytes": model_names_size,
                    "size_mb": round(model_names_size / (1024 * 1024), 4),
                }
                
            # Deployment names list
            if hasattr(llm_router, 'deployment_names') and llm_router.deployment_names:
                deployment_names_size = sys.getsizeof(llm_router.deployment_names)
                litellm_router_memory["deployment_names"] = {
                    "num_deployments": len(llm_router.deployment_names),
                    "size_bytes": deployment_names_size,
                    "size_mb": round(deployment_names_size / (1024 * 1024), 4),
                }
                
            # Deployment latency map
            if hasattr(llm_router, 'deployment_latency_map') and llm_router.deployment_latency_map:
                latency_map_size = sys.getsizeof(llm_router.deployment_latency_map)
                litellm_router_memory["deployment_latency_map"] = {
                    "num_tracked_deployments": len(llm_router.deployment_latency_map),
                    "size_bytes": latency_map_size,
                    "size_mb": round(latency_map_size / (1024 * 1024), 4),
                }
                
            # Fallback configuration
            if hasattr(llm_router, 'fallbacks') and llm_router.fallbacks:
                fallbacks_size = sys.getsizeof(llm_router.fallbacks)
                litellm_router_memory["fallbacks"] = {
                    "num_fallback_configs": len(llm_router.fallbacks),
                    "size_bytes": fallbacks_size,
                    "size_mb": round(fallbacks_size / (1024 * 1024), 4),
                }
                
            # Total router object size
            router_obj_size = sys.getsizeof(llm_router)
            litellm_router_memory["router_object"] = {
                "size_bytes": router_obj_size,
                "size_mb": round(router_obj_size / (1024 * 1024), 4),
            }
                
        else:
            litellm_router_memory = {"note": "Router not initialized"}
    except Exception as e:
        verbose_proxy_logger.debug(f"Error getting router memory info: {e}")
        litellm_router_memory = {"error": str(e)}
    
    return litellm_router_memory


def _get_process_memory_info(worker_pid: int, include_process_info: bool) -> Optional[Dict[str, Any]]:
    """Get process-level memory information using psutil."""
    if not include_process_info:
        return None
        
    try:
        import psutil
        
        process = psutil.Process()
        memory_info = process.memory_info()
        ram_usage_mb = round(memory_info.rss / (1024 * 1024), 2)
        virtual_memory_mb = round(memory_info.vms / (1024 * 1024), 2)
        memory_percent = round(process.memory_percent(), 2)
        
        return {
            "pid": worker_pid,
            "summary": f"Worker PID {worker_pid} using {ram_usage_mb:.1f} MB of RAM ({memory_percent:.1f}% of system memory)",
            "ram_usage": {
                "megabytes": ram_usage_mb,
                "description": "Actual physical RAM used by this process"
            },
            "virtual_memory": {
                "megabytes": virtual_memory_mb,
                "description": "Total virtual memory allocated (includes swapped memory)"
            },
            "system_memory_percent": {
                "percent": memory_percent,
                "description": "Percentage of total system RAM being used"
            },
            "open_file_handles": {
                "count": process.num_fds() if hasattr(process, "num_fds") else "N/A (Windows)",
                "description": "Number of open file descriptors/handles"
            },
            "threads": {
                "count": process.num_threads(),
                "description": "Number of active threads in this process"
            }
        }
    except ImportError:
        return {
            "pid": worker_pid,
            "error": "psutil not installed. Install with: pip install psutil"
        }
    except Exception as e:
        verbose_proxy_logger.debug(f"Error getting process info: {e}")
        return {"pid": worker_pid, "error": str(e)}


@router.get("/debug/memory/details", include_in_schema=False)
async def get_memory_details(
    _: UserAPIKeyAuth = Depends(user_api_key_auth),
    top_n: int = Query(20, description="Number of top object types to return"),
    include_process_info: bool = Query(True, description="Include process memory info"),
) -> Dict[str, Any]:
    """
    Get detailed memory diagnostics for deep debugging.
    
    Returns:
    - worker_pid: Process ID
    - process_memory: RAM usage, virtual memory, file handles, threads
    - garbage_collector: GC thresholds, counts, collection history
    - objects: Total tracked objects and top object types
    - uncollectable: Objects that can't be garbage collected (potential leaks)
    - cache_memory: Memory usage of user_api_key, router, and logging caches
    - router_memory: Memory usage of router components (model_list, deployment_names, etc.)
    
    Query Parameters:
    - top_n: Number of top object types to return (default: 20)
    - include_process_info: Include process-level memory info using psutil (default: true)
    
    Example usage:
    curl "http://localhost:4000/debug/memory/details?top_n=30" -H "Authorization: Bearer sk-1234"
    
    All memory sizes are reported in both bytes and MB.
    """
    from litellm.proxy.proxy_server import (
        llm_router,
        proxy_logging_obj,
        user_api_key_cache,
        redis_usage_cache,
    )
    
    worker_pid = os.getpid()
    
    # Collect all diagnostics using helper functions
    gc_stats = _get_gc_statistics()
    total_objects, top_object_types = _get_object_type_counts(top_n)
    uncollectable_info = _get_uncollectable_objects_info()
    cache_stats = _get_cache_memory_stats(user_api_key_cache, llm_router, proxy_logging_obj, redis_usage_cache)
    litellm_router_memory = _get_router_memory_stats(llm_router)
    process_info = _get_process_memory_info(worker_pid, include_process_info)
    
    return {
        "worker_pid": worker_pid,
        "process_memory": process_info,
        "garbage_collector": gc_stats,
        "objects": {
            "total_tracked": total_objects,
            "total_tracked_readable": f"{total_objects:,}",
            "top_types": top_object_types,
        },
        "uncollectable": uncollectable_info,
        "cache_memory": cache_stats,
        "router_memory": litellm_router_memory,
    }


@router.post("/debug/memory/gc/configure", include_in_schema=False)
async def configure_gc_thresholds_endpoint(
    _: UserAPIKeyAuth = Depends(user_api_key_auth),
    generation_0: int = Query(700, description="Generation 0 threshold (default: 700)"),
    generation_1: int = Query(10, description="Generation 1 threshold (default: 10)"),
    generation_2: int = Query(10, description="Generation 2 threshold (default: 10)"),
) -> Dict[str, Any]:
    """
    Configure Python garbage collection thresholds.
    
    Lower thresholds mean more frequent GC cycles (less memory, more CPU overhead).
    Higher thresholds mean less frequent GC cycles (more memory, less CPU overhead).
    
    Returns:
    - message: Confirmation message
    - previous_thresholds: Old threshold values
    - new_thresholds: New threshold values
    - objects_awaiting_collection: Current object count in gen-0
    - tip: Hint about when next collection will occur
    
    Query Parameters:
    - generation_0: Number of allocations before gen-0 collection (default: 700)
    - generation_1: Number of gen-0 collections before gen-1 collection (default: 10)  
    - generation_2: Number of gen-1 collections before gen-2 collection (default: 10)
    
    Example for more aggressive collection:
    curl -X POST "http://localhost:4000/debug/memory/gc/configure?generation_0=500" -H "Authorization: Bearer sk-1234"
    
    Example for less aggressive collection:
    curl -X POST "http://localhost:4000/debug/memory/gc/configure?generation_0=1000" -H "Authorization: Bearer sk-1234"
    
    Monitor memory usage with GET /debug/memory/summary after changes.
    """
    # Get current thresholds for logging
    old_thresholds = gc.get_threshold()
    
    # Set new thresholds with error handling
    try:
        gc.set_threshold(generation_0, generation_1, generation_2)
        verbose_proxy_logger.info(
            f"GC thresholds updated from {old_thresholds} to "
            f"({generation_0}, {generation_1}, {generation_2})"
        )
    except Exception as e:
        verbose_proxy_logger.error(f"Failed to set GC thresholds: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to set GC thresholds: {str(e)}"
        )
    
    # Get current object count to show immediate impact
    current_count = gc.get_count()[0]
    
    return {
        "message": "GC thresholds updated",
        "previous_thresholds": f"{old_thresholds[0]}, {old_thresholds[1]}, {old_thresholds[2]}",
        "new_thresholds": f"{generation_0}, {generation_1}, {generation_2}",
        "objects_awaiting_collection": current_count,
        "tip": f"Next collection will run after {generation_0 - current_count} more allocations"
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
