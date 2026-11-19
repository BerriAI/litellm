"""Regression tests for the SDK-free OTel runtime shim.

The proxy auth hot path calls ``phase_span`` and ``seed_request_identity`` on
every request. These wrappers resolve the SDK-backed implementations with a
lazy import. CPython never caches a failed import, so before memoization an
absent OTel SDK made every request re-scan ``sys.path`` and contend on the
import lock. These tests pin the import to a single resolution.
"""

import builtins
import importlib.abc
import sys
from collections.abc import Sequence
from types import ModuleType
from typing import Final

import pytest

import litellm.integrations.otel.runtime as runtime


def test_logger_not_reimported_after_first_resolution(monkeypatch):
    runtime._otel_runtime.cache_clear()

    counts = {"n": 0}
    real_import = builtins.__import__

    def counting_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "litellm.integrations.otel" and fromlist and "logger" in fromlist:
            counts["n"] += 1
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", counting_import)

    with runtime.phase_span("auth /v1/chat/completions"):
        pass
    after_first = counts["n"]

    for _ in range(49):
        with runtime.phase_span("auth /v1/chat/completions"):
            pass

    assert counts["n"] == after_first, (
        f"otel.logger re-imported {counts['n'] - after_first} times after the first "
        "resolution; it must be memoized so it does not re-scan sys.path per request"
    )

    runtime._otel_runtime.cache_clear()


def test_resolution_is_memoized():
    runtime._otel_runtime.cache_clear()

    for _ in range(25):
        with runtime.phase_span("p"):
            pass

    info = runtime._otel_runtime.cache_info()
    assert info.misses == 1
    assert info.hits >= 24

    runtime._otel_runtime.cache_clear()


def test_wrappers_no_op_when_runtime_absent(monkeypatch):
    monkeypatch.setattr(runtime, "_otel_runtime", lambda: None)

    with runtime.phase_span("auth") as span:
        assert span is None

    assert runtime.seed_request_identity({"token": "sk-x"}, model="gpt-4o") is None


def test_phase_event_no_ops_when_runtime_absent(monkeypatch):
    monkeypatch.setattr(runtime, "_otel_runtime", lambda: None)

    assert runtime.phase_event("litellm.request.body_parsed") is None
    assert runtime.phase_event("litellm.request.body_received", {"litellm.request.body_bytes": 3}) is None


def test_phase_span_does_not_import_the_proxy_in_an_sdk_process(monkeypatch):
    import litellm.proxy

    monkeypatch.delitem(sys.modules, "litellm.proxy.proxy_server", raising=False)
    monkeypatch.delattr(litellm.proxy, "proxy_server", raising=False)
    proxy_imports: list[str] = []

    class _RefuseProxyImport(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname == "litellm.proxy.proxy_server":
                proxy_imports.append(fullname)
                raise ImportError(fullname)
            return None

    monkeypatch.setattr(sys, "meta_path", [_RefuseProxyImport(), *sys.meta_path])

    with runtime.phase_span("route gpt-5-mini") as span:
        assert span is None

    assert runtime.phase_attributes({"litellm.routing.score": 0.25}) is None
    assert proxy_imports == []


def test_phase_attributes_no_op_when_sdk_import_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime._otel_runtime.cache_clear()
    real_import: Final = builtins.__import__

    def without_sdk(
        name: str,
        globals: dict[str, object] | None = None,
        locals: dict[str, object] | None = None,
        fromlist: Sequence[str] = (),
        level: int = 0,
    ) -> ModuleType:
        if name == "litellm.integrations.otel" and "logger" in fromlist:
            raise ImportError("OpenTelemetry SDK is not installed")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", without_sdk)
    try:
        assert runtime.phase_attributes({"litellm.routing.score": 0.25}) is None
        assert runtime._otel_runtime() is None
    finally:
        runtime._otel_runtime.cache_clear()
