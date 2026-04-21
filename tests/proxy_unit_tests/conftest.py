# conftest.py

import asyncio
import copy
import os
import sys

import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

import litellm
import litellm.proxy.proxy_server


def _snapshot_mutable_state(module):
    """Deep-copy every list/dict/set module attribute for later restore.

    Classes, functions, submodules and primitives are skipped — only the
    collections that tests mutate (callbacks, caches, routers, etc.) need
    per-test isolation.
    """
    snapshot = {}
    for attr in list(vars(module)):
        if attr.startswith("_"):
            continue
        try:
            value = getattr(module, attr)
        except Exception:
            continue
        if isinstance(value, (list, dict, set)):
            try:
                snapshot[attr] = copy.deepcopy(value)
            except Exception:
                # Unpickleable collections (e.g. holding open clients) can't
                # round-trip through deepcopy; skip them rather than crash.
                pass
    return snapshot


def _restore_mutable_state(module, snapshot):
    for attr, default in snapshot.items():
        try:
            setattr(module, attr, copy.deepcopy(default))
        except Exception:
            pass


# Snapshot once at conftest import — these are the "clean" module states.
_LITELLM_STATE = _snapshot_mutable_state(litellm)
_PROXY_SERVER_STATE = _snapshot_mutable_state(litellm.proxy.proxy_server)


@pytest.fixture(scope="function", autouse=True)
def setup_and_teardown():
    """
    Reset mutable module state on litellm and proxy_server before every test.

    Replaces a previous importlib.reload(litellm) approach that cost ~17s
    per test (re-executing the full litellm __init__ import chain). The
    snapshot-and-restore below only touches collections that actually leak
    across tests — callbacks, caches, router, etc. — and is effectively
    instantaneous.
    """
    _restore_mutable_state(litellm, _LITELLM_STATE)
    _restore_mutable_state(litellm.proxy.proxy_server, _PROXY_SERVER_STATE)

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
