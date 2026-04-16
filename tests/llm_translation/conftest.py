"""Test isolation for tests/llm_translation/.

Design goals
------------
1. Every test starts against a clean litellm state without relying on
   ``importlib.reload(litellm)``. Reloading is expensive (hundreds of
   submodules get rebuilt) and leaks memory because it rebinds module-level
   singletons while their background tasks keep running on the shared event
   loop. With ~2k tests × 8 xdist workers, that leak was driving CircleCI
   workers past their memory limit.

2. Per-test teardown explicitly stops the ``GLOBAL_LOGGING_WORKER`` background
   task so its ``_worker_loop`` does not accumulate across tests. Cached HTTP
   clients are released once at session end (not per-test) because tests and
   SDKs hold references to those clients outside ``LLMClientCache``, and
   closing them mid-run raises "Cannot send a request, as the client has been
   closed." at the next call site.

3. Anything that cannot be restored through a plain attribute assignment is
   reset with a targeted helper (logger levels, asyncio tasks). If a future
   test mutates a new module-level attribute, add it to the relevant baseline
   rather than reaching for ``importlib.reload`` again.
"""

from __future__ import annotations

import asyncio
import copy
import gc
import json
import logging
import os
import sys
import threading
import time
import tracemalloc
from pathlib import Path
from typing import Any, Dict, Iterable

import pytest

sys.path.insert(0, os.path.abspath("../.."))
import litellm  # noqa: E402  (path manipulation must precede import)


# ---------------------------------------------------------------------------
# Diagnostic instrumentation (off by default).
# Enable with ``LLM_TRANSLATION_DIAGNOSTIC=1``.
#
# Per-test JSONL records are appended to
# ``test-results/diagnostic-<worker>.jsonl``. Columns:
#   - ts: wall clock (seconds since epoch)
#   - idx: test index on this worker
#   - phase: "setup" | "teardown"
#   - test: nodeid
#   - rss_mb: process RSS in MiB
#   - fds: open file descriptors
#   - threads: live thread count
#   - tasks: pending asyncio tasks on the session event loop
#   - cache_n: entries in in_memory_llm_clients_cache
#   - azure_closed: bool — whether ANY cached Azure client has is_closed=True
#   - gc_objs: total number of Python objects tracked by gc
#
# One JSONL record makes failures diagnosable: compare the last passing test's
# teardown values to the failing test's setup values. Anything that spikes
# (rss_mb, cache_n, azure_closed flipping True) is the culprit.
# ---------------------------------------------------------------------------

_DIAGNOSTIC_ENABLED = os.environ.get("LLM_TRANSLATION_DIAGNOSTIC") == "1"
_DIAGNOSTIC_DIR = Path(os.environ.get("LLM_TRANSLATION_DIAGNOSTIC_DIR", "test-results"))
_WORKER_ID = os.environ.get("PYTEST_XDIST_WORKER", "main")
_TEST_COUNTER = 0


def _diag_file() -> Path:
    _DIAGNOSTIC_DIR.mkdir(parents=True, exist_ok=True)
    return _DIAGNOSTIC_DIR / f"diagnostic-{_WORKER_ID}.jsonl"


def _proc_rss_mb() -> float:
    """Read RSS from /proc/self/status (Linux); fall back to 0 on other OSes."""
    try:
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    # e.g. "VmRSS:      12345 kB"
                    kb = int(line.split()[1])
                    return round(kb / 1024.0, 1)
    except Exception:
        pass
    return 0.0


def _open_fds() -> int:
    try:
        return len(os.listdir("/proc/self/fd"))
    except Exception:
        return 0


def _count_async_tasks(loop: asyncio.AbstractEventLoop) -> int:
    try:
        return sum(1 for t in asyncio.all_tasks(loop) if not t.done())
    except Exception:
        return 0


def _cache_snapshot() -> tuple[int, bool, list[str], list[str]]:
    """Return (count, any_closed, closed_key_types, all_key_types).

    key_types use the first 40 chars of each cache key, which is enough to
    distinguish "azure-xxx" from "httpx-xxx" from aiohttp-based keys while
    keeping the JSONL small.
    """
    cache = getattr(litellm, "in_memory_llm_clients_cache", None)
    cache_dict = getattr(cache, "cache_dict", None)
    if not isinstance(cache_dict, dict):
        return (0, False, [], [])

    closed_keys: list[str] = []
    all_keys: list[str] = []
    for key, value in cache_dict.items():
        short_key = str(key)[:40]
        all_keys.append(short_key)
        # AsyncAzureOpenAI / AsyncOpenAI wrap httpx in _client; AsyncHTTPHandler
        # uses .client; httpx.AsyncClient itself has .is_closed.
        inner = getattr(value, "_client", None) or getattr(value, "client", None) or value
        try:
            if getattr(inner, "is_closed", False):
                closed_keys.append(short_key)
        except Exception:
            continue
    return (len(cache_dict), bool(closed_keys), closed_keys, all_keys)


def _emit_diag(phase: str, nodeid: str, loop: asyncio.AbstractEventLoop) -> None:
    global _TEST_COUNTER
    if not _DIAGNOSTIC_ENABLED:
        return

    cache_n, any_closed, closed_keys, all_keys = _cache_snapshot()
    record = {
        "ts": time.time(),
        "idx": _TEST_COUNTER,
        "phase": phase,
        "test": nodeid,
        "rss_mb": _proc_rss_mb(),
        "fds": _open_fds(),
        "threads": threading.active_count(),
        "tasks": _count_async_tasks(loop),
        "cache_n": cache_n,
        "azure_closed": any_closed,
        # New: list the exact short cache keys whose client reports
        # is_closed=True. This lets us see which client flipped, and between
        # which two tests. Full key list is for distinguishing azure vs. openai
        # vs. httpx cache entries without needing to know their structure.
        "closed_keys": closed_keys,
        "cache_keys": all_keys,
        "gc_objs": len(gc.get_objects()),
        "worker": _WORKER_ID,
    }
    try:
        with _diag_file().open("a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        # Instrumentation must never fail the test.
        pass


# ---------------------------------------------------------------------------
# Baseline snapshot of mutable module-level litellm state.
# Captured once at conftest import time, before any test runs.
# ---------------------------------------------------------------------------

# Attributes that are lists tests tend to append to. Reset to an empty list
# every test so callback registration from one test does not leak into the
# next one.
_CALLBACK_LIST_ATTRS: tuple[str, ...] = (
    "callbacks",
    "success_callback",
    "failure_callback",
    "service_callback",
    "input_callback",
    "_async_success_callback",
    "_async_failure_callback",
)

# Scalars / simple values tests frequently mutate. Snapshotted from the
# current module state so we restore whatever the library considered the
# default at import time.
_SCALAR_ATTRS: tuple[str, ...] = (
    "num_retries",
    "set_verbose",
    "cache",
    "allowed_fails",
    "disable_aiohttp_transport",
    "force_ipv4",
    "drop_params",
    "modify_params",
    "api_base",
    "api_key",
    "cohere_key",
    "disable_stop_sequence_limit",
    "enable_preview_features",
)

# Provider model collections tests mutate both in-place (``.add(...)``) and
# by reassignment (e.g. ``litellm.vertex_mistral_models = [...]`` in
# test_optional_params.py). Reassignment changes the type from ``set`` to
# ``list``, which then breaks ``litellm.add_known_models()`` in any later
# test that calls it. The old conftest's ``importlib.reload(litellm)`` hid
# this by recreating the module attributes each time; we have to snapshot
# and deep-copy them instead.
_PROVIDER_MODEL_ATTRS: tuple[str, ...] = (
    "open_ai_chat_completion_models",
    "open_ai_text_completion_models",
    "azure_text_models",
    "cohere_models",
    "cohere_chat_models",
    "mistral_chat_models",
    "anthropic_models",
    "empower_models",
    "openrouter_models",
    "vercel_ai_gateway_models",
    "datarobot_models",
    "vertex_text_models",
    "vertex_code_text_models",
    "vertex_language_models",
    "vertex_vision_models",
    "vertex_chat_models",
    "vertex_code_chat_models",
    "vertex_embedding_models",
    "vertex_anthropic_models",
    "vertex_llama3_models",
    "vertex_deepseek_models",
    "vertex_mistral_models",
)


def _snapshot_scalar_baseline() -> Dict[str, Any]:
    """Capture the clean-state value of every tracked attribute."""
    baseline: Dict[str, Any] = {}
    for attr in _CALLBACK_LIST_ATTRS:
        if hasattr(litellm, attr):
            # Store as an empty list; tests expect to start with nothing
            # registered, and the library-side default is always [].
            baseline[attr] = []
    for attr in _SCALAR_ATTRS:
        if hasattr(litellm, attr):
            baseline[attr] = getattr(litellm, attr)
    return baseline


def _snapshot_provider_model_baseline() -> Dict[str, Any]:
    """Deep-copy the original provider model collections.

    Deep-copy so that in-place mutation of the snapshot itself is impossible
    even if a test holds a reference and mutates it later. Restoration
    always hands out a fresh deep-copy for the same reason.
    """
    return {
        attr: copy.deepcopy(getattr(litellm, attr))
        for attr in _PROVIDER_MODEL_ATTRS
        if hasattr(litellm, attr)
    }


_BASELINE: Dict[str, Any] = _snapshot_scalar_baseline()
_PROVIDER_MODEL_BASELINE: Dict[str, Any] = _snapshot_provider_model_baseline()

# LiteLLM's three public loggers. Their level survives module reloads because
# Python's ``logging`` module — not ``litellm`` — owns the Logger objects.
_LITELLM_LOGGERS: tuple[str, ...] = (
    "LiteLLM",
    "LiteLLM Router",
    "LiteLLM Proxy",
)


# ---------------------------------------------------------------------------
# Event loop.
# Session-scoped so long-running asyncio tasks spawned across tests can be
# cancelled centrally. Each test gets its own isolation layer on top.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        yield loop
    finally:
        _drain_pending_tasks(loop)
        loop.close()


def _drain_pending_tasks(loop: asyncio.AbstractEventLoop) -> None:
    """Cancel and await every pending task on the loop. Safe to call repeatedly."""
    pending = [t for t in asyncio.all_tasks(loop) if not t.done()]
    if not pending:
        return
    for task in pending:
        task.cancel()
    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))


# ---------------------------------------------------------------------------
# Reset helpers used by the per-test fixture.
# ---------------------------------------------------------------------------


def _reset_scalar_attributes() -> None:
    """Restore tracked litellm.<attr> values to the captured baseline."""
    for attr, value in _BASELINE.items():
        if isinstance(value, list):
            # Hand each test its own mutable list so mutations stay local.
            setattr(litellm, attr, list(value))
        else:
            setattr(litellm, attr, value)


def _reset_provider_model_collections() -> None:
    """Restore provider model sets/collections to their original type and contents.

    ``test_optional_params.py`` reassigns ``litellm.vertex_mistral_models`` from
    a ``set`` to a ``list``, which then breaks ``litellm.add_known_models()``
    (it calls ``.add(key)``). Restoring from a deep-copy of the original
    snapshot fixes both reassignment and in-place mutation.
    """
    for attr, original in _PROVIDER_MODEL_BASELINE.items():
        setattr(litellm, attr, copy.deepcopy(original))


def _reset_logger_levels() -> None:
    """Clamp the LiteLLM loggers back to WARNING.

    Any test that called ``litellm._turn_on_debug()`` flipped these to DEBUG,
    and that level survives module reloads because the Logger objects live in
    the stdlib ``logging`` module. DEBUG-level logging allocates large debug
    strings per live HTTP request, which is a per-request tax across every
    subsequent test in the worker.
    """
    for name in _LITELLM_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


async def _stop_logging_worker() -> None:
    """Cancel ``GLOBAL_LOGGING_WORKER._worker_task`` on the current loop.

    ``start()`` is idempotent — the next test that enqueues a log message
    will create a fresh worker task. Stopping here prevents the previous
    test's ``_worker_loop`` coroutine from sitting on the session-scoped
    event loop and holding references to old queues/semaphores.
    """
    from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER

    try:
        await GLOBAL_LOGGING_WORKER.stop()
    except Exception:
        # Best-effort; a broken stop() must not break the test suite.
        pass


def _restore_env_vars(initial_env: Dict[str, str], current_env: Iterable[str]) -> None:
    """Restore os.environ to its pre-test state.

    Tests that set env vars without using ``monkeypatch`` would otherwise leak
    API keys, model-cost overrides, and region settings into subsequent tests.
    """
    current_keys = set(current_env)
    for key in current_keys - set(initial_env):
        os.environ.pop(key, None)
    for key, value in initial_env.items():
        if os.environ.get(key) != value:
            os.environ[key] = value


# ---------------------------------------------------------------------------
# Per-test fixture.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def isolate_litellm_state(event_loop, request):
    """Snapshot state, run the test, restore state.

    HTTP client cleanup is deliberately NOT done here:
      - ``close_litellm_async_clients()`` closes clients the OpenAI / Azure
        SDKs and a few passthrough handlers hold references to outside
        ``LLMClientCache``. Closing them mid-run makes subsequent tests
        fail with ``"Cannot send a request, as the client has been closed"``.
      - Instead, client cleanup runs once at session end in
        ``cleanup_http_clients_at_session_end``. Within a single worker the
        client pool is bounded by the number of distinct ``cache_key``
        tuples, not by the test count.
    """
    global _TEST_COUNTER
    _TEST_COUNTER += 1
    nodeid = request.node.nodeid

    # ---- Setup ----
    _emit_diag("setup", nodeid, event_loop)
    _reset_scalar_attributes()
    _reset_provider_model_collections()
    _reset_logger_levels()
    initial_env = dict(os.environ)
    asyncio.set_event_loop(event_loop)

    yield

    # ---- Teardown ----
    _reset_scalar_attributes()
    _reset_provider_model_collections()
    _reset_logger_levels()
    try:
        event_loop.run_until_complete(_stop_logging_worker())
    except RuntimeError:
        # The test may have closed or replaced the loop; ignore and let the
        # session-scoped cleanup catch anything we missed.
        pass
    _drain_pending_tasks(event_loop)
    _restore_env_vars(initial_env, list(os.environ))
    _emit_diag("teardown", nodeid, event_loop)


# ---------------------------------------------------------------------------
# Session-scoped HTTP client cleanup.
# Runs once after all tests in the worker finish. Releases the aiohttp
# sessions / TCP connectors cached in ``LLMClientCache`` plus the global
# ``base_llm_aiohttp_handler`` session. The library also registers an
# atexit hook, but atexit does not fire on SIGKILL (OOM), so doing it
# explicitly here guarantees cleanup under normal exit.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def cleanup_http_clients_at_session_end(event_loop):
    yield

    async def _close_all() -> None:
        try:
            from litellm.llms.custom_httpx.async_client_cleanup import (
                close_litellm_async_clients,
            )

            await close_litellm_async_clients()
        except Exception:
            pass

    try:
        event_loop.run_until_complete(_close_all())
    except RuntimeError:
        pass

    # Drop Python references so GC can reclaim anything the close routine
    # released. Safe because no more tests will run.
    cache = getattr(litellm, "in_memory_llm_clients_cache", None)
    cache_dict = getattr(cache, "cache_dict", None)
    if isinstance(cache_dict, dict):
        cache_dict.clear()


# ---------------------------------------------------------------------------
# Collection ordering.
# custom_logger tests install global logging callbacks; grouping them first
# avoids interleaving them with unrelated tests that would then see leftover
# callbacks from a partially-completed custom_logger test run.
# ---------------------------------------------------------------------------


def pytest_collection_modifyitems(config, items):
    custom_logger_tests = [
        item for item in items if "custom_logger" in item.parent.name
    ]
    other_tests = [
        item for item in items if "custom_logger" not in item.parent.name
    ]

    custom_logger_tests.sort(key=lambda x: x.name)
    other_tests.sort(key=lambda x: x.name)

    items[:] = custom_logger_tests + other_tests
