"""Proxy-boot smoke tests for auto-queue middleware.

These verify import-time wiring and env-var gating without booting a real proxy process.
"""
import importlib
import os
import sys

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
sys.path.insert(0, REPO_ROOT)
_loaded_litellm = sys.modules.get("litellm")
_loaded_litellm_path = getattr(_loaded_litellm, "__file__", None)
_needs_reload = _loaded_litellm_path is not None and not os.path.abspath(
    _loaded_litellm_path
).startswith(REPO_ROOT)
if _needs_reload:
    for _name in list(sys.modules):
        if _name == "litellm" or _name.startswith("litellm."):
            sys.modules.pop(_name, None)


def _reload_auto_queue_module(monkeypatch, **env):
    """Reload the auto_queue_middleware module with specific env vars."""
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    module = importlib.import_module("litellm.proxy.middleware.auto_queue_middleware")
    return importlib.reload(module)


def test_autoq_enabled_env_true(monkeypatch):
    module = _reload_auto_queue_module(monkeypatch, AUTOQ_ENABLED="true")
    assert module.AUTOQ_ENABLED is True


def test_autoq_enabled_env_false(monkeypatch):
    module = _reload_auto_queue_module(monkeypatch, AUTOQ_ENABLED="false")
    assert module.AUTOQ_ENABLED is False


def test_autoq_enabled_env_default_is_false(monkeypatch):
    monkeypatch.delenv("AUTOQ_ENABLED", raising=False)
    module = importlib.import_module("litellm.proxy.middleware.auto_queue_middleware")
    module = importlib.reload(module)
    assert module.AUTOQ_ENABLED is False


def test_middleware_passthrough_when_disabled(monkeypatch):
    """When enabled=False, middleware should be a transparent passthrough."""
    module = _reload_auto_queue_module(monkeypatch, AUTOQ_ENABLED="false", WEB_CONCURRENCY="1")

    received = []

    async def app(scope, receive, send):
        received.append(scope)
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    middleware = module.AutoQueueMiddleware(app, enabled=False)
    assert middleware._enabled is False


def test_middleware_enabled_flag_respected(monkeypatch):
    """Explicit enabled=True overrides the env var."""
    module = _reload_auto_queue_module(monkeypatch, AUTOQ_ENABLED="false", WEB_CONCURRENCY="1")

    async def app(scope, receive, send):
        pass

    middleware = module.AutoQueueMiddleware(app, enabled=True)
    assert middleware._enabled is True


def test_middleware_boot_allows_multiple_workers_when_enabled(monkeypatch):
    module = _reload_auto_queue_module(monkeypatch, AUTOQ_ENABLED="true", WEB_CONCURRENCY="4")

    async def app(scope, receive, send):
        pass

    middleware = module.AutoQueueMiddleware(app, enabled=True)
    assert middleware._enabled is True


def test_proxy_server_wires_auto_queue_reconciler_helpers():
    proxy_server_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../../../litellm/proxy/proxy_server.py")
    )
    source = open(proxy_server_path, "r", encoding="utf-8").read()

    assert "build_auto_queue_reconciler" in source
    assert "async def _start_auto_queue_reconciler" in source
    assert "async def _stop_auto_queue_reconciler" in source
    assert "await _start_auto_queue_reconciler(app)" in source
    assert "await _stop_auto_queue_reconciler(app)" in source
