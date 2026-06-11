"""
Shared fixtures and helpers for proxy tests.

This module provides reusable utilities for creating proxy test clients
with database and Redis cache configuration.
"""

import asyncio
import os
import tempfile
from typing import Dict, Optional

import pytest
import yaml
from fastapi.testclient import TestClient

_PROXY_MODULE_GLOBALS_TO_ISOLATE = (
    "master_key",
    "prisma_client",
    "user_config_file_path",
    "general_settings",
    "llm_router",
    "llm_model_list",
    "premium_user",
    "store_model_in_db",
    "proxy_logging_obj",
    "redis_usage_cache",
)

# Known-good defaults to set at the START of every test so a corrupt
# snapshot from a previous worker run cannot propagate further.
_PROXY_MODULE_GLOBALS_RESET = {
    "master_key": None,
    "prisma_client": None,
    "user_config_file_path": None,
    "general_settings": {},
    "llm_router": None,
    "llm_model_list": [],
    "store_model_in_db": False,
    # Prevent _FakeRedisCache leakage from test_redis_auth_cache_flag tests
    # that call _init_cache with a mock Redis and don't restore redis_usage_cache.
    "redis_usage_cache": None,
}


@pytest.fixture(autouse=True)
def _isolate_proxy_module_globals():
    """
    Snapshot and restore module-level globals on litellm.proxy.proxy_server
    that tests sometimes mutate via raw setattr (not monkeypatch).

    We also force a known-good initial state *before* yielding so that a
    corrupt snapshot from a previous worker test cannot cascade into the
    current test (snapshot/restore alone does not help when the pre-test
    state is already wrong).
    """
    from litellm.proxy import proxy_server

    sentinel = object()
    snapshot = {
        name: getattr(proxy_server, name, sentinel)
        for name in _PROXY_MODULE_GLOBALS_TO_ISOLATE
    }

    # Force clean defaults for the subset we know how to reset safely.
    for name, default in _PROXY_MODULE_GLOBALS_RESET.items():
        setattr(proxy_server, name, default)

    # Reset litellm_config_cache redis backend — a _FakeRedisCache from
    # test_redis_auth_cache_flag can leak here via _init_cache's direct
    # litellm_config_cache.redis_cache = redis_usage_cache assignment, even
    # though _init_cache is only called when the config has cache:true.
    try:
        from litellm.proxy.utils import litellm_config_cache
        litellm_config_cache.redis_cache = None
    except Exception:
        pass

    # Also clear any stale FastAPI dependency overrides left by previous tests.
    from litellm.proxy.proxy_server import app
    saved_overrides = dict(app.dependency_overrides)
    app.dependency_overrides.clear()

    try:
        yield
    finally:
        for name, value in snapshot.items():
            if value is sentinel:
                if hasattr(proxy_server, name):
                    delattr(proxy_server, name)
            else:
                setattr(proxy_server, name, value)
        # Restore dependency overrides to pre-test state.
        app.dependency_overrides.clear()
        app.dependency_overrides.update(saved_overrides)


@pytest.fixture(autouse=True)
def _reset_graceful_shutdown_state():
    """Graceful shutdown state is process-scoped; keep it from leaking between tests."""
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

    # Configure database
    db_url = database_url or os.getenv("DATABASE_URL")
    if db_url:
        config["general_settings"]["database_url"] = db_url

    # Configure cache if Redis is available
    enable_cache = init_options.get("enable_cache", True)
    cache_config = build_cache_config(enable_cache=enable_cache)
    if cache_config:
        config["litellm_settings"].update(cache_config)

    # Add success_callback if provided (for realistic readiness endpoint)
    if init_options.get("success_callback") is not None:
        config["litellm_settings"]["success_callback"] = init_options[
            "success_callback"
        ]

    # Add any other litellm_settings from init_options
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
    # Set database URL
    db_url = database_url or os.getenv("DATABASE_URL")
    if db_url:
        monkeypatch.setenv("DATABASE_URL", db_url)

    # Set Redis environment variables if available
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

    # Get config file path
    filepath = os.path.dirname(os.path.abspath(__file__))
    default_config_fp = os.path.join(
        filepath, "test_configs", "test_config_no_auth.yaml"
    )

    # Check if we need to create a minimal config with Redis/database
    enable_cache = init_options.get("enable_cache", True)
    needs_redis = enable_cache and os.getenv("REDIS_HOST") is not None
    needs_db = (database_url or os.getenv("DATABASE_URL")) is not None

    # Create minimal config if:
    # 1. Default config file doesn't exist, OR
    # 2. We need Redis/database config that might not be in the default config
    if not os.path.exists(default_config_fp) or needs_redis or needs_db:
        minimal_config = build_minimal_proxy_config(
            database_url=database_url, **init_options
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(minimal_config, f)
            config_fp = f.name
    else:
        config_fp = default_config_fp

    # Set environment variables
    set_proxy_environment_variables(monkeypatch, database_url=database_url)

    # Initialize proxy
    asyncio.run(initialize(config=config_fp, debug=init_options.get("debug", False)))
    return TestClient(app)
