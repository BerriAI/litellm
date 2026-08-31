"""Wiring tests for the Conduct guardrail integration.

The real adapter, response-envelope parser, session-ID chain, and
fail-mode logic live in the ``conduct-litellm-guard`` PyPI package and
are tested there. This file only verifies that the LiteLLM-tree
wrapper wires the enum, registries, and initializer correctly.

To keep coverage useful on BerriAI's CI (where the standalone package
is not installed by default), we mock ``conduct_litellm_guard`` into
``sys.modules`` before importing the wrapper. This lets every test run
regardless of whether the real PyPI package is present in the test
environment.
"""

from __future__ import annotations

import importlib
import sys
import types
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Fake standalone package
# ---------------------------------------------------------------------------
#
# The wrapper in ``litellm/proxy/guardrails/guardrail_hooks/conduct/conduct.py``
# imports three symbols:
#
#   from conduct_litellm_guard import ConductGuard
#   from conduct_litellm_guard.guardrail import ConductGuardBlocked, GuardDecision
#
# We stand up a fake package tree in ``sys.modules`` that mirrors that shape
# so imports succeed without the real PyPI package being installed.


def _install_fake_conduct_package() -> type:
    """Register a fake ``conduct_litellm_guard`` package tree in sys.modules.

    Returns the fake ``ConductGuard`` class so tests can assert against it.
    """
    from litellm.integrations.custom_guardrail import CustomGuardrail

    class _FakeConductGuard(CustomGuardrail):
        def __init__(self, **kwargs: Any) -> None:
            self.init_kwargs = kwargs
            super().__init__(guardrail_name=kwargs.get("guardrail_name", ""))

    class _FakeConductGuardBlocked(Exception):
        pass

    class _FakeGuardDecision:
        ALLOW = "allow"
        BLOCK = "block"

    root = types.ModuleType("conduct_litellm_guard")
    root.ConductGuard = _FakeConductGuard  # type: ignore[attr-defined]

    guardrail_mod = types.ModuleType("conduct_litellm_guard.guardrail")
    guardrail_mod.ConductGuardBlocked = _FakeConductGuardBlocked  # type: ignore[attr-defined]
    guardrail_mod.GuardDecision = _FakeGuardDecision  # type: ignore[attr-defined]

    sys.modules["conduct_litellm_guard"] = root
    sys.modules["conduct_litellm_guard.guardrail"] = guardrail_mod

    return _FakeConductGuard


def _forget_wrapper_modules() -> None:
    """Drop cached wrapper imports so subsequent imports re-run the module body."""
    for name in list(sys.modules):
        if name.startswith("litellm.proxy.guardrails.guardrail_hooks.conduct"):
            del sys.modules[name]


@pytest.fixture
def fake_conduct(monkeypatch: pytest.MonkeyPatch) -> type:
    """Install the fake package and reset wrapper imports for each test."""
    _forget_wrapper_modules()
    fake_cls = _install_fake_conduct_package()
    yield fake_cls
    _forget_wrapper_modules()
    sys.modules.pop("conduct_litellm_guard", None)
    sys.modules.pop("conduct_litellm_guard.guardrail", None)


# ---------------------------------------------------------------------------
# Wiring smoke tests
# ---------------------------------------------------------------------------


def test_wrapper_module_imports(fake_conduct: type) -> None:
    """The wrapper module imports cleanly against the fake package."""
    module = importlib.import_module("litellm.proxy.guardrails.guardrail_hooks.conduct")
    assert module.ConductGuardrail is fake_conduct


def test_wrapper_exports_expected_symbols(fake_conduct: type) -> None:
    """Public re-exports match the documented ``__all__``."""
    module = importlib.import_module(
        "litellm.proxy.guardrails.guardrail_hooks.conduct.conduct"
    )
    assert set(module.__all__) == {
        "ConductGuardrail",
        "ConductGuardrailBlocked",
        "GuardDecision",
    }
    assert module.ConductGuardrail is fake_conduct
    assert issubclass(module.ConductGuardrailBlocked, Exception)


def test_class_is_custom_guardrail_subclass(fake_conduct: type) -> None:
    from litellm.integrations.custom_guardrail import CustomGuardrail
    from litellm.proxy.guardrails.guardrail_hooks.conduct import ConductGuardrail

    assert issubclass(ConductGuardrail, CustomGuardrail)


def test_enum_value_registered() -> None:
    """Enum entry exists regardless of standalone package availability."""
    from litellm.types.guardrails import SupportedGuardrailIntegrations

    assert SupportedGuardrailIntegrations.CONDUCT.value == "conduct"


def test_registries_populated(fake_conduct: type) -> None:
    from litellm.proxy.guardrails.guardrail_hooks.conduct import (
        guardrail_class_registry,
        guardrail_initializer_registry,
    )

    assert guardrail_class_registry["conduct"] is fake_conduct
    assert callable(guardrail_initializer_registry["conduct"])


# ---------------------------------------------------------------------------
# initialize_guardrail — cover happy path and populated-params path
# ---------------------------------------------------------------------------


def test_initialize_guardrail_defaults(
    fake_conduct: type,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When LiteLLM params are minimal, defaults are passed to ConductGuard."""
    added_callbacks: list[object] = []
    fake_manager = SimpleNamespace(
        add_litellm_callback=lambda cb: added_callbacks.append(cb),
    )
    import litellm

    monkeypatch.setattr(litellm, "logging_callback_manager", fake_manager)

    from litellm.proxy.guardrails.guardrail_hooks.conduct import (
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

    assert isinstance(callback, fake_conduct)
    assert added_callbacks == [callback]
    kwargs = callback.init_kwargs
    assert kwargs["api_url"] is None
    assert kwargs["agent_token"] is None
    assert kwargs["workspace_id"] is None
    assert kwargs["fail_mode"] == "fail_closed"
    assert kwargs["tool_name"] == "llm_call"
    assert kwargs["timeout"] == 8.0
    assert kwargs["event_hook"] == "pre_call"
    assert kwargs["default_on"] is True


def test_initialize_guardrail_populated_params(
    fake_conduct: type,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-default LiteLLM params flow through to ConductGuard constructor."""
    added_callbacks: list[object] = []
    fake_manager = SimpleNamespace(
        add_litellm_callback=lambda cb: added_callbacks.append(cb),
    )
    import litellm

    monkeypatch.setattr(litellm, "logging_callback_manager", fake_manager)

    from litellm.proxy.guardrails.guardrail_hooks.conduct import (
        initialize_guardrail,
    )

    litellm_params = SimpleNamespace(
        api_base="https://api.conductai.ai/guard",
        api_key="cond_agt_test_placeholder",
        workspace_id="ws_123",
        fail_mode="fail_open",
        tool_name="chat_completion",
        timeout=15.0,
        mode="post_call",
        default_on=False,
    )
    guardrail = MagicMock()
    guardrail.get.return_value = "conduct-prod"

    callback = initialize_guardrail(litellm_params, guardrail)

    assert isinstance(callback, fake_conduct)
    kwargs = callback.init_kwargs
    assert kwargs["api_url"] == "https://api.conductai.ai/guard"
    assert kwargs["agent_token"] == "cond_agt_test_placeholder"
    assert kwargs["workspace_id"] == "ws_123"
    assert kwargs["fail_mode"] == "fail_open"
    assert kwargs["tool_name"] == "chat_completion"
    assert kwargs["timeout"] == 15.0
    assert kwargs["guardrail_name"] == "conduct-prod"
    assert kwargs["event_hook"] == "post_call"
    assert kwargs["default_on"] is False


# ---------------------------------------------------------------------------
# Missing-package branch
# ---------------------------------------------------------------------------


def test_missing_standalone_package_raises_helpful_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The wrapper points users at ``pip install`` when the standalone
    ``conduct-litellm-guard`` package is not installed."""
    _forget_wrapper_modules()
    sys.modules.pop("conduct_litellm_guard", None)
    sys.modules.pop("conduct_litellm_guard.guardrail", None)

    real_import = (
        __builtins__["__import__"]  # type: ignore[index]
        if isinstance(__builtins__, dict)
        else __builtins__.__import__
    )

    def _fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name.startswith("conduct_litellm_guard"):
            raise ImportError("simulated missing package")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _fake_import)

    with pytest.raises(ImportError, match="pip install conduct-litellm-guard"):
        importlib.import_module(
            "litellm.proxy.guardrails.guardrail_hooks.conduct.conduct"
        )

    _forget_wrapper_modules()
