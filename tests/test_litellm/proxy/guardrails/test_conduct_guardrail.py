"""Smoke tests for the Conduct guardrail integration.

The adapter itself is tested in the ``conduct-litellm-guard`` PyPI
package. Here we only verify:
  * the LiteLLM-tree module imports cleanly when the standalone package
    is installed
  * the enum + registry entries are wired
"""

from __future__ import annotations

import importlib
import sys

import pytest

# The Conduct guardrail imports its runtime from `conduct-litellm-guard`
# on PyPI. When the package is not installed in the CI environment,
# skip — the wiring smoke tests only make sense against the real dep.
pytest.importorskip(
    "conduct_litellm_guard",
    reason="Install `conduct-litellm-guard` to test the Conduct guardrail integration.",
)


def test_import_module() -> None:
    """The wrapper module imports without side effects."""
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


def test_initialize_guardrail_returns_wired_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``initialize_guardrail`` maps LiteLLM params to Conduct kwargs and
    registers the callback with ``logging_callback_manager``. This test
    exercises the full function body so coverage reports don't flag it
    as dead code."""
    from types import SimpleNamespace
    from unittest.mock import MagicMock

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


def test_missing_standalone_package_raises_helpful_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When ``conduct-litellm-guard`` is not installed, the import fails
    with a message pointing users at the ``pip install`` command."""
    # Ensure the module is re-imported without the standalone package.
    for name in list(sys.modules):
        if name.startswith(("conduct_litellm_guard", "litellm.proxy.guardrails.guardrail_hooks.conduct")):
            monkeypatch.delitem(sys.modules, name, raising=False)

    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

    def _fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name.startswith("conduct_litellm_guard"):
            raise ImportError("simulated missing package")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _fake_import)

    with pytest.raises(ImportError, match="pip install conduct-litellm-guard"):
        importlib.import_module("litellm.proxy.guardrails.guardrail_hooks.conduct.conduct")
