"""Regression tests for ``litellm.integrations.otel.runtime``.

These guard the per-request hot path: the SDK-free wrappers must resolve the
optional ``litellm.integrations.otel.logger`` import exactly once and cache the
outcome. Re-attempting the import on every call re-runs the whole import
finder/loader machinery (a failed import is never recorded in ``sys.modules``),
which on a proxy without the OTel SDK cost hundreds of microseconds per request.
"""

import builtins
import contextlib
import sys
import types

import litellm.integrations.otel.runtime as rt


def _reset_runtime_cache():
    rt._resolved = False
    rt._phase_span_impl = None
    rt._seed_request_identity_impl = None


def test_resolves_logger_import_at_most_once(monkeypatch):
    _reset_runtime_cache()

    real_import = builtins.__import__
    attempts = {"n": 0}

    def counting_import(name, *args, **kwargs):
        if name == "litellm.integrations.otel.logger":
            attempts["n"] += 1
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", counting_import)

    for _ in range(25):
        with rt.phase_span("auth /v1/chat/completions"):
            pass
        rt.seed_request_identity(object(), model="gpt-4o")

    assert attempts["n"] <= 1


def test_phase_span_and_seed_noop_when_logger_unavailable():
    rt._resolved = True
    rt._phase_span_impl = None
    rt._seed_request_identity_impl = None

    with rt.phase_span("auth /v1/messages") as span:
        assert span is None

    rt.seed_request_identity(object(), model="claude-3-5")


def test_phase_span_and_seed_use_resolved_impl_when_available():
    calls = {"phase": [], "seed": []}

    @contextlib.contextmanager
    def fake_phase_span(name):
        calls["phase"].append(name)
        yield "live-span"

    def fake_seed(user_api_key_dict, model=None):
        calls["seed"].append((user_api_key_dict, model))

    rt._resolved = True
    rt._phase_span_impl = fake_phase_span
    rt._seed_request_identity_impl = fake_seed

    with rt.phase_span("auth /chat") as span:
        assert span == "live-span"
    rt.seed_request_identity("key-obj", model="gpt-4o")

    assert calls["phase"] == ["auth /chat"]
    assert calls["seed"] == [("key-obj", "gpt-4o")]

    _reset_runtime_cache()


def test_resolve_wires_up_logger_when_importable(monkeypatch):
    """When the OTel logger is importable, ``_resolve`` caches its callables.

    Simulates the SDK-present deployment by injecting a fake logger module, so
    the import-success branch is exercised even though this environment has no
    OTel SDK. Triggering resolution via ``seed_request_identity`` first also
    covers the lazy-resolve path on that entrypoint.
    """
    seen = {}

    @contextlib.contextmanager
    def logger_phase_span(name):
        seen["phase"] = name
        yield "real-span"

    def logger_seed(user_api_key_dict, model=None):
        seen["seed"] = (user_api_key_dict, model)

    fake_logger = types.ModuleType("litellm.integrations.otel.logger")
    fake_logger.phase_span = logger_phase_span
    fake_logger.seed_request_identity = logger_seed
    monkeypatch.setitem(sys.modules, "litellm.integrations.otel.logger", fake_logger)

    _reset_runtime_cache()

    rt.seed_request_identity("key-obj", model="claude-3-5")
    assert rt._resolved is True
    assert rt._phase_span_impl is logger_phase_span
    assert rt._seed_request_identity_impl is logger_seed

    with rt.phase_span("auth /v1/chat/completions") as span:
        assert span == "real-span"

    assert seen == {
        "seed": ("key-obj", "claude-3-5"),
        "phase": "auth /v1/chat/completions",
    }

    _reset_runtime_cache()
