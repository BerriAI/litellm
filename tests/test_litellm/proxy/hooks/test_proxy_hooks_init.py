"""Regression guard for the enterprise hook registration / import cycle.

Python 3.13 is stricter about partially-initialized modules and surfaces
cycles that Python 3.12 silently tolerated. The previous bug:

  litellm.proxy.hooks.__init__
    -> enterprise.enterprise_hooks
    -> litellm_enterprise.proxy.hooks.managed_files
    -> litellm.llms.base_llm.managed_resources.isolation
    -> litellm.proxy.management_endpoints.common_utils
    -> litellm.proxy.utils  (re-enters litellm.proxy.hooks mid-init)

silently swallowed the ImportError in `hooks/__init__.py`, leaving
``managed_files`` unregistered and the /files endpoint returning 500.
"""

import pytest

from litellm.proxy.hooks import PROXY_HOOKS, get_proxy_hook


def test_managed_files_hook_registered():
    pytest.importorskip("litellm_enterprise")
    assert "managed_files" in PROXY_HOOKS
    hook_cls = get_proxy_hook("managed_files")
    assert hook_cls.__name__ == "_PROXY_LiteLLMManagedFiles"


def test_managed_vector_stores_hook_registered():
    pytest.importorskip("litellm_enterprise")
    assert "managed_vector_stores" in PROXY_HOOKS
    hook_cls = get_proxy_hook("managed_vector_stores")
    assert hook_cls.__name__ == "_PROXY_LiteLLMManagedVectorStores"


def test_isolation_module_does_not_pull_in_proxy_utils():
    """Layering guard: litellm.llms.* must not transitively import
    litellm.proxy.utils, which would reintroduce the import cycle."""
    import importlib
    import sys

    for mod in [
        "litellm.proxy.utils",
        "litellm.proxy.management_endpoints.common_utils",
        "litellm.llms.base_llm.managed_resources.isolation",
    ]:
        sys.modules.pop(mod, None)

    importlib.import_module("litellm.llms.base_llm.managed_resources.isolation")
    assert "litellm.proxy.utils" not in sys.modules
    assert "litellm.proxy.management_endpoints.common_utils" not in sys.modules

def test_enterprise_hooks_import_failure_logs_warning(monkeypatch, caplog):
    """
    If the `enterprise` / `litellm_enterprise` package is unavailable
    (e.g. missing from a hardened non_root Docker image), hooks/__init__.py
    should log a warning and continue, instead of letting the ImportError
    propagate and crash the whole proxy at import time.
    """
    import builtins
    import importlib
    import logging
    import sys

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "enterprise" or name.startswith("enterprise."):
            raise ImportError("No module named 'enterprise'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    sys.modules.pop("litellm.proxy.hooks", None)

    with caplog.at_level(logging.WARNING):
        import litellm.proxy.hooks as hooks_module
        importlib.reload(hooks_module)

    assert any(
        "enterprise hooks" in record.message.lower()
        for record in caplog.records
    ), "Expected a warning to be logged when enterprise hooks import fails"

    monkeypatch.undo()
    sys.modules.pop("litellm.proxy.hooks", None)
    importlib.import_module("litellm.proxy.hooks")
