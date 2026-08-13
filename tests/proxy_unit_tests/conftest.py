# conftest.py

import asyncio
import copy
import inspect
import os
import sys
import warnings

import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

import litellm
import litellm.proxy.proxy_server


# Top-level assignments of these types are the ones importlib.reload(litellm)
# would have effectively reset. We snapshot them at conftest import time and
# deep-copy the snapshot back before every test.
_SNAPSHOT_TYPES = (list, dict, set, tuple, str, int, float, bool, bytes)


def _snapshot_mutable_state(module):
    """Capture a per-module snapshot of primitive and collection attributes."""
    snapshot = {}
    for attr in list(vars(module)):
        if attr.startswith("_"):
            continue
        try:
            value = getattr(module, attr)
        except Exception as exc:
            warnings.warn(
                f"conftest: could not read {module.__name__}.{attr} during snapshot: {exc}",
                stacklevel=2,
            )
            continue
        if value is None or isinstance(value, _SNAPSHOT_TYPES):
            try:
                snapshot[attr] = copy.deepcopy(value)
            except Exception as exc:
                warnings.warn(
                    f"conftest: could not snapshot {module.__name__}.{attr}: {exc}",
                    stacklevel=2,
                )
    return snapshot


def _restore_mutable_state(module, snapshot):
    for attr, default in snapshot.items():
        try:
            setattr(module, attr, copy.deepcopy(default))
        except Exception as exc:
            warnings.warn(
                f"conftest: could not restore {module.__name__}.{attr}: {exc}",
                stacklevel=2,
            )


def _collect_flushable_caches():
    """Return (module, attr) pairs whose values expose flush_cache()."""
    targets = []
    for module in (litellm, litellm.proxy.proxy_server):
        for attr in list(vars(module)):
            if attr.startswith("_"):
                continue
            try:
                value = getattr(module, attr)
            except Exception:
                continue
            # Only instances — a class reference has an unbound flush_cache
            # that can't be called without a self argument.
            if inspect.isclass(value) or inspect.ismodule(value):
                continue
            if callable(getattr(value, "flush_cache", None)):
                targets.append((module, attr))
    return targets


def _flush_caches(targets):
    for module, attr in targets:
        try:
            value = getattr(module, attr)
        except Exception:
            continue
        flush = getattr(value, "flush_cache", None)
        if callable(flush):
            try:
                flush()
            except Exception as exc:
                warnings.warn(
                    f"conftest: flush_cache failed on {module.__name__}.{attr}: {exc}",
                    stacklevel=2,
                )


# Snapshot once at conftest import — these are the "clean" module states.
_LITELLM_STATE = _snapshot_mutable_state(litellm)
_PROXY_SERVER_STATE = _snapshot_mutable_state(litellm.proxy.proxy_server)
_FLUSHABLE_CACHES = _collect_flushable_caches()


@pytest.fixture(scope="function", autouse=True)
def setup_and_teardown():
    """Reset mutable module state on litellm and proxy_server before each test.

    Replaces a previous importlib.reload(litellm) approach that cost ~17s
    per test (re-executing the full litellm __init__ import chain).

    What IS reset:
      - Top-level module attributes of type list / dict / set / tuple
        / str / int / float / bool / bytes, and None-valued attributes.
        These cover callback lists, general_settings, master_key,
        premium_user, prisma_client, etc. — anything the old reload() reset
        by re-executing the module body.
      - Any module-level object instance that exposes flush_cache() (the
        DualCache and LLMClientCache family), which handles cache state
        that can't round-trip through deepcopy because of internal locks.

    What is NOT reset:
      - Class instances without flush_cache() (e.g. ProxyLogging,
        JWTHandler, FastAPI routers, loggers). If a test mutates such an
        instance in-place (setattr on the instance, appending to one of
        its internal lists, etc.), the mutation will leak into later tests.
        Use pytest's monkeypatch.setattr() or a local fixture for those
        cases — don't rely on this autouse fixture to undo them.
    """
    _restore_mutable_state(litellm, _LITELLM_STATE)
    _restore_mutable_state(litellm.proxy.proxy_server, _PROXY_SERVER_STATE)
    _flush_caches(_FLUSHABLE_CACHES)

    loop = asyncio.get_event_loop_policy().new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        yield
    finally:
        loop.close()
        asyncio.set_event_loop(None)


def pytest_collection_modifyitems(config, items):
    # Separate tests in 'test_amazing_proxy_custom_logger.py' and other tests
    custom_logger_tests = [
        item for item in items if "custom_logger" in item.parent.name
    ]
    other_tests = [item for item in items if "custom_logger" not in item.parent.name]

    # Sort tests based on their names
    custom_logger_tests.sort(key=lambda x: x.name)
    other_tests.sort(key=lambda x: x.name)

    # Reorder the items list
    items[:] = custom_logger_tests + other_tests
