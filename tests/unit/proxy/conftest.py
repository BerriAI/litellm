# conftest.py

import asyncio
import copy
import inspect
import os
import tempfile
import warnings
from collections.abc import Iterator
from typing import Dict, Optional

import pytest
import yaml
from fastapi.testclient import TestClient
from prisma.errors import ClientNotConnectedError


import litellm
import litellm.proxy.proxy_server
from tests.unit.litellm_core_utils.fake_secret_vault import FakeSecretVault


class StubClientNotConnectedError(ClientNotConnectedError):
    pass


class DisconnectedPrisma:
    def is_connected(self) -> bool:
        return False

    @property
    def _engine(self) -> None:
        raise StubClientNotConnectedError()


@pytest.fixture
def disconnected_prisma() -> DisconnectedPrisma:
    return DisconnectedPrisma()


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
                snapshot[attr] = _restored_value(value)
            except Exception as exc:
                warnings.warn(
                    f"conftest: could not snapshot {module.__name__}.{attr}: {exc}",
                    stacklevel=2,
                )
    return snapshot


_MUTABLE_CONTAINERS = (list, dict, set, bytearray)


def _holds_mutable_container(value) -> bool:
    if isinstance(value, _MUTABLE_CONTAINERS):
        return True
    if isinstance(value, tuple):
        return any(_holds_mutable_container(element) for element in value)
    return False


def _restored_value(value):
    return copy.deepcopy(value) if _holds_mutable_container(value) else value


def _restore_mutable_state(module, snapshot):
    for attr, default in snapshot.items():
        try:
            setattr(module, attr, _restored_value(default))
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


_PROXY_MODULE_GLOBALS_TO_ISOLATE = (
    "master_key",
    "prisma_client",
    "llm_router",
)

_proxy_module_globals_snapshot = pytest.StashKey[Dict[str, object]]()


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_setup(item):
    from litellm.proxy import proxy_server

    item.stash[_proxy_module_globals_snapshot] = {
        name: vars(proxy_server)[name]
        for name in _PROXY_MODULE_GLOBALS_TO_ISOLATE
        if name in vars(proxy_server)
    }
    yield


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_teardown(item, nextitem):
    yield
    snapshot = item.stash.get(_proxy_module_globals_snapshot, None)
    if snapshot is None:
        return
    from litellm.proxy import proxy_server

    for name in _PROXY_MODULE_GLOBALS_TO_ISOLATE:
        if name in snapshot:
            setattr(proxy_server, name, snapshot[name])
        elif name in vars(proxy_server):
            delattr(proxy_server, name)


@pytest.fixture
def secret_vault_factory() -> type[FakeSecretVault]:
    return FakeSecretVault


@pytest.fixture
def httpx_transport(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    litellm.in_memory_llm_clients_cache.flush_cache()
    yield
    litellm.in_memory_llm_clients_cache.flush_cache()


@pytest.fixture(autouse=True)
def _reset_graceful_shutdown_state():
    from litellm.proxy.shutdown.graceful_shutdown_manager import (
        GracefulShutdownManager,
    )

    GracefulShutdownManager.reset()
    yield
    GracefulShutdownManager.reset()


def build_cache_config(enable_cache: bool = True) -> Optional[Dict]:
    """
    Build Redis cache configuration from environment variables.

    Args:
        enable_cache: Whether to enable cache (default: True)

    Returns:
        dict: Cache configuration dict with 'cache' and 'cache_params' keys, or None
    """
    if not enable_cache:
        return None

    redis_host = os.getenv("REDIS_HOST")
    if not redis_host:
        return None

    redis_port = os.getenv("REDIS_PORT", "6379")
    cache_params = {
        "type": "redis",
        "host": redis_host,
        "port": int(redis_port) if redis_port.isdigit() else redis_port,
    }

    redis_password = os.getenv("REDIS_PASSWORD")
    if redis_password:
        cache_params["password"] = redis_password

    return {"cache": True, "cache_params": cache_params}


def build_minimal_proxy_config(
    database_url: Optional[str] = None, **init_options
) -> Dict:
    """
    Build a minimal proxy configuration YAML.

    Args:
        database_url: Optional database URL (falls back to DATABASE_URL env var)
        **init_options: Additional configuration options:
            - master_key: API key for authentication (default: "sk-1234")
            - enable_cache: Whether to enable Redis cache (default: True)
            - success_callback: Callback function for success events

    Returns:
        dict: Configuration dictionary ready to be written as YAML
    """
    config = {
        "general_settings": {"master_key": init_options.get("master_key", "sk-1234")},
        "litellm_settings": {},
    }

    db_url = database_url or os.getenv("DATABASE_URL")
    if db_url:
        config["general_settings"]["database_url"] = db_url

    enable_cache = init_options.get("enable_cache", True)
    cache_config = build_cache_config(enable_cache=enable_cache)
    if cache_config:
        config["litellm_settings"].update(cache_config)

    if init_options.get("success_callback") is not None:
        config["litellm_settings"]["success_callback"] = init_options[
            "success_callback"
        ]

    excluded_keys = {
        "master_key",
        "debug",
        "success_callback",
        "database_url",
        "enable_cache",
    }
    for key, value in init_options.items():
        if key not in excluded_keys and key not in config["litellm_settings"]:
            config["litellm_settings"][key] = value

    return config


def set_proxy_environment_variables(
    monkeypatch, database_url: Optional[str] = None
) -> None:
    """
    Set environment variables for database and Redis.

    Args:
        monkeypatch: pytest monkeypatch fixture
        database_url: Optional database URL (falls back to DATABASE_URL env var)
    """
    db_url = database_url or os.getenv("DATABASE_URL")
    if db_url:
        monkeypatch.setenv("DATABASE_URL", db_url)

    redis_host = os.getenv("REDIS_HOST")
    if redis_host:
        monkeypatch.setenv("REDIS_HOST", redis_host)
        monkeypatch.setenv("REDIS_PORT", os.getenv("REDIS_PORT", "6379"))
        redis_password = os.getenv("REDIS_PASSWORD")
        if redis_password:
            monkeypatch.setenv("REDIS_PASSWORD", redis_password)


def create_proxy_test_client(
    monkeypatch, database_url: Optional[str] = None, **init_options
) -> TestClient:
    """
    Create a proxy TestClient with optional database and Redis cache configuration.

    Args:
        monkeypatch: pytest monkeypatch fixture
        database_url: Optional database URL (falls back to DATABASE_URL env var)
        **init_options: Additional configuration options:
            - master_key: API key for authentication (default: "sk-1234")
            - enable_cache: Whether to enable Redis cache (default: True)
            - success_callback: Callback function for success events
            - debug: Enable debug mode

    Returns:
        TestClient: FastAPI test client for the proxy server
    """
    from litellm.proxy.proxy_server import (
        cleanup_router_config_variables,
        initialize,
        app,
    )

    cleanup_router_config_variables()

    filepath = os.path.dirname(os.path.abspath(__file__))
    default_config_fp = os.path.join(
        filepath, "test_configs", "test_config_hosted_vllm_embedding.yaml"
    )

    enable_cache = init_options.get("enable_cache", True)
    needs_redis = enable_cache and os.getenv("REDIS_HOST") is not None
    needs_db = (database_url or os.getenv("DATABASE_URL")) is not None

    if not os.path.exists(default_config_fp) or needs_redis or needs_db:
        minimal_config = build_minimal_proxy_config(
            database_url=database_url, **init_options
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(minimal_config, f)
            config_fp = f.name
    else:
        config_fp = default_config_fp

    set_proxy_environment_variables(monkeypatch, database_url=database_url)
    monkeypatch.setenv("LITELLM_DANGEROUSLY_PERMIT_WEAK_OR_UNSET_MASTER_KEY", "true")

    asyncio.run(initialize(config=config_fp, debug=init_options.get("debug", False)))
    return TestClient(app)


@pytest.fixture
def fresh_agent_read_through(monkeypatch):
    from litellm.proxy.common_utils import registry_read_through

    read_through = registry_read_through.RegistryReadThrough(
        resync=registry_read_through._resync_agents, is_loaded=registry_read_through._agent_is_loaded
    )
    monkeypatch.setattr(registry_read_through, "agent_registry_read_through", read_through)
    return read_through
