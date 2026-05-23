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

    # Save and restore the modules we evict so that subsequent tests in the
    # same xdist worker are not left with a freshly-reimported module object.
    # If we leave these entries absent, the next import creates a *new* module
    # object, orphaning any function's __globals__ dict that still points at
    # the old one — which silently breaks unittest.mock.patch on those names.
    saved = {
        mod: sys.modules.get(mod)
        for mod in [
            "litellm.proxy.utils",
            "litellm.proxy.management_endpoints.common_utils",
            "litellm.llms.base_llm.managed_resources.isolation",
        ]
    }
    for mod in saved:
        sys.modules.pop(mod, None)

    try:
        importlib.import_module("litellm.llms.base_llm.managed_resources.isolation")
        assert "litellm.proxy.utils" not in sys.modules
        assert "litellm.proxy.management_endpoints.common_utils" not in sys.modules
    finally:
        # Restore original module objects so later tests see a consistent
        # sys.modules and their pre-imported names remain valid.
        for mod, original in saved.items():
            if original is not None:
                sys.modules[mod] = original
            else:
                sys.modules.pop(mod, None)
