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

import importlib
import logging
import sys

import pytest

import litellm.proxy
from litellm._logging import verbose_proxy_logger
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


def test_managed_hooks_register_from_litellm_enterprise_without_the_repo_only_enterprise_package(
    monkeypatch: pytest.MonkeyPatch,
):
    pytest.importorskip("litellm_enterprise")
    from litellm_enterprise.proxy.hooks.managed_files import PROXY_LiteLLMManagedFiles
    from litellm_enterprise.proxy.hooks.managed_vector_stores import PROXY_LiteLLMManagedVectorStores

    monkeypatch.setitem(sys.modules, "enterprise", None)
    monkeypatch.setitem(sys.modules, "enterprise.enterprise_hooks", None)
    monkeypatch.delitem(sys.modules, "litellm.proxy.hooks")
    monkeypatch.delattr(litellm.proxy, "hooks")

    wheel_install_hooks = importlib.import_module("litellm.proxy.hooks")

    assert wheel_install_hooks.get_proxy_hook("managed_files") is PROXY_LiteLLMManagedFiles
    assert wheel_install_hooks.get_proxy_hook("managed_vector_stores") is PROXY_LiteLLMManagedVectorStores


def test_proxy_hooks_still_import_and_warn_when_litellm_enterprise_cannot_load(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    monkeypatch.setitem(sys.modules, "litellm_enterprise.proxy.hooks.managed_files", None)
    monkeypatch.delitem(sys.modules, "litellm.proxy.hooks")
    monkeypatch.delattr(litellm.proxy, "hooks")
    verbose_proxy_logger.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.WARNING, logger=verbose_proxy_logger.name):
            hooks_without_enterprise = importlib.import_module("litellm.proxy.hooks")
    finally:
        verbose_proxy_logger.removeHandler(caplog.handler)

    assert "managed_files" not in hooks_without_enterprise.PROXY_HOOKS
    assert "managed_vector_stores" not in hooks_without_enterprise.PROXY_HOOKS
    assert "parallel_request_limiter" in hooks_without_enterprise.PROXY_HOOKS
    assert any(
        "Managed files and managed vector stores are disabled" in record.getMessage() for record in caplog.records
    )
