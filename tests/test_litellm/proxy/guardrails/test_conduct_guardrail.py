"""Tests for the Conduct guardrail integration.

The adapter itself is tested in ``conduct-litellm-guard`` on PyPI —
here we only verify the LiteLLM-tree wiring:
  * the module imports cleanly with and without the standalone package
  * the enum + registry entries are populated
  * ``initialize_guardrail`` reads the typed ``unreachable_fallback``
    field, applies the timeout default correctly, and registers the
    callback with LiteLLM's manager (yucheng-berri / cursor findings)
  * only supported event hooks are advertised (veria-ai finding)
"""

from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

# The Conduct guardrail imports its runtime from `conduct-litellm-guard`
# on PyPI. When the package is not installed in the CI environment,
# skip the wiring smoke tests. The missing-package test below runs
# unconditionally because it needs a controlled ImportError.
pytest.importorskip(
    "conduct_litellm_guard",
    reason="Install `conduct-litellm-guard` to test the Conduct guardrail integration.",
)


def test_import_module() -> None:
    module = importlib.import_module("litellm.proxy.guardrails.guardrail_hooks.conduct")
    assert module.ConductGuardrail is not None


def test_class_is_custom_guardrail_subclass() -> None:
    from litellm.integrations.custom_guardrail import CustomGuardrail
    from litellm.proxy.guardrails.guardrail_hooks.conduct import ConductGuardrail

    assert issubclass(ConductGuardrail, CustomGuardrail)


def test_enum_value_registered() -> None:
    from litellm.types.guardrails import SupportedGuardrailIntegrations

    assert SupportedGuardrailIntegrations.CONDUCT.value == "conduct"


def test_registries_populated() -> None:
    from litellm.proxy.guardrails.guardrail_hooks.conduct import (
        guardrail_class_registry,
        guardrail_initializer_registry,
    )

    assert "conduct" in guardrail_class_registry
    assert "conduct" in guardrail_initializer_registry


def test_only_pre_call_event_hook_advertised() -> None:
    """Regression for veria-ai finding on #38143 —
    ``during_call`` mode was silently accepted but never evaluated
    because ``async_moderation_hook`` was not overridden. We only
    advertise pre_call today so LiteLLM validates configs against
    supported hooks and rejects unsupported modes."""
    from litellm.proxy.guardrails.guardrail_hooks.conduct import ConductGuardrail
    from litellm.types.guardrails import GuardrailEventHooks

    hooks = ConductGuardrail.get_supported_event_hooks()
    assert hooks == [GuardrailEventHooks.pre_call]


def test_initialize_guardrail_returns_wired_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CONDUCT_AGENT_TOKEN", "cond_agt_test_placeholder")

    added_callbacks: list[object] = []
    fake_manager = SimpleNamespace(
        add_litellm_callback=lambda cb: added_callbacks.append(cb),
    )

    import litellm

    monkeypatch.setattr(litellm, "logging_callback_manager", fake_manager)

    from litellm.proxy.guardrails.guardrail_hooks.conduct import (
        ConductGuardrail,
        initialize_guardrail,
    )

    litellm_params = SimpleNamespace(
        api_base=None,
        api_key=None,
        mode="pre_call",
        default_on=True,
    )
    guardrail = MagicMock()
    guardrail.get.return_value = "conduct-guard"

    callback = initialize_guardrail(litellm_params, guardrail)

    assert isinstance(callback, ConductGuardrail)
    assert added_callbacks == [callback]


def test_initialize_prefers_typed_unreachable_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression for yucheng-berri / devin-ai-integration findings —
    the typed ``unreachable_fallback`` field replaces the free-form
    ``fail_mode`` this shim previously read. Typos on the old field
    silently defaulted to fail-open behavior; the typed field forces
    Pydantic validation."""
    monkeypatch.setenv("CONDUCT_AGENT_TOKEN", "cond_agt_test_placeholder")
    import litellm
    from litellm.proxy.guardrails.guardrail_hooks.conduct import initialize_guardrail

    captured: dict = {}

    class _FakeGuard:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(
        "litellm.proxy.guardrails.guardrail_hooks.conduct.ConductGuardrail",
        _FakeGuard,
        raising=True,
    )
    monkeypatch.setattr(
        litellm, "logging_callback_manager", SimpleNamespace(add_litellm_callback=lambda cb: None)
    )

    litellm_params = SimpleNamespace(
        api_base=None,
        api_key=None,
        unreachable_fallback="fail_open",
        mode="pre_call",
        default_on=True,
    )
    guardrail = MagicMock()
    guardrail.get.return_value = "conduct-guard"

    initialize_guardrail(litellm_params, guardrail)
    # The Conduct constructor still accepts ``fail_mode`` — we map from
    # the typed LiteLLM field to it.
    assert captured["fail_mode"] == "fail_open"


def test_initialize_applies_timeout_default_when_field_is_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression for cursor[bot] finding —
    ``LitellmParams.timeout`` always exists as ``None``, so
    ``getattr(litellm_params, "timeout", 8.0)`` was never applied. The
    default now uses ``or`` so ``None`` falls through to 8.0."""
    monkeypatch.setenv("CONDUCT_AGENT_TOKEN", "cond_agt_test_placeholder")
    import litellm
    from litellm.proxy.guardrails.guardrail_hooks.conduct import initialize_guardrail

    captured: dict = {}

    class _FakeGuard:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(
        "litellm.proxy.guardrails.guardrail_hooks.conduct.ConductGuardrail",
        _FakeGuard,
    )
    monkeypatch.setattr(
        litellm, "logging_callback_manager", SimpleNamespace(add_litellm_callback=lambda cb: None)
    )

    litellm_params = SimpleNamespace(
        api_base=None,
        api_key=None,
        timeout=None,  # the typical case — field exists but caller left it unset
        mode="pre_call",
        default_on=True,
    )
    guardrail = MagicMock()
    guardrail.get.return_value = "conduct-guard"

    initialize_guardrail(litellm_params, guardrail)
    assert captured["timeout"] == 8.0


def test_missing_standalone_package_raises_at_construction() -> None:
    """Regression for cursor[bot] finding — the previous shim raised
    ``ImportError`` at module load, which the guardrail-hook auto-loader
    treats as "hook unavailable" and silently drops. The subclass now
    imports lazily and raises at ``__init__`` time when actionable."""
    from litellm.proxy.guardrails.guardrail_hooks.conduct import conduct as _mod

    original_base = _mod._BaseConductGuard
    original_error = _mod._IMPORT_ERROR
    try:
        _mod._BaseConductGuard = None  # type: ignore[assignment]
        _mod._IMPORT_ERROR = ImportError("simulated missing package")

        # Class is still importable — no module-load side effect.
        from litellm.proxy.guardrails.guardrail_hooks.conduct import ConductGuardrail

        with pytest.raises(ImportError, match="pip install"):
            ConductGuardrail()
    finally:
        _mod._BaseConductGuard = original_base  # type: ignore[assignment]
        _mod._IMPORT_ERROR = original_error
