import importlib.util
import os
from pathlib import Path
from typing import Literal, Optional, Union


def _workspace_managed_files_py() -> Optional[Path]:
    try:
        import litellm as _lm

        p = (
            Path(_lm.__file__).resolve().parent.parent
            / "enterprise"
            / "litellm_enterprise"
            / "proxy"
            / "hooks"
            / "managed_files.py"
        )
        return p if p.is_file() else None
    except Exception:
        return None


def _upgrade_managed_files_hook_from_workspace(hooks: dict) -> None:
    """
    Editable / mixed installs often register `litellm_enterprise` via importlib metadata,
    so sys.path cannot shadow site-packages. If the loaded hook is missing newer APIs
    (e.g. `acreate_container`) but the monorepo file defines them, load that module.
    """
    cls = hooks.get("managed_files")
    if cls is None or hasattr(cls, "acreate_container"):
        return
    path = _workspace_managed_files_py()
    if path is None:
        return
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return
    if "async def acreate_container" not in text:
        return
    try:
        mod_name = "_litellm_enterprise_workspace_managed_files"
        spec = importlib.util.spec_from_file_location(mod_name, path)
        if spec is None or spec.loader is None:
            return
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        new_cls = getattr(mod, "_PROXY_LiteLLMManagedFiles", None)
        if new_cls is not None and hasattr(new_cls, "acreate_container"):
            hooks["managed_files"] = new_cls
    except Exception:
        return


from . import *
from .cache_control_check import _PROXY_CacheControlCheck
from .litellm_skills import SkillsInjectionHook
from .max_budget_limiter import _PROXY_MaxBudgetLimiter
from .max_budget_per_session_limiter import _PROXY_MaxBudgetPerSessionHandler
from .max_iterations_limiter import _PROXY_MaxIterationsHandler
from .parallel_request_limiter import _PROXY_MaxParallelRequestsHandler
from .parallel_request_limiter_v3 import _PROXY_MaxParallelRequestsHandler_v3
from .responses_id_security import ResponsesIDSecurity

### CHECK IF ENTERPRISE HOOKS ARE AVAILABLE ####
# Monorepo / editable installs expose `enterprise.enterprise_hooks`.
# PyPI `litellm-enterprise` installs `litellm_enterprise.*` only, so try both.

ENTERPRISE_PROXY_HOOKS: dict = {}
try:
    from enterprise.enterprise_hooks import (
        ENTERPRISE_PROXY_HOOKS as _ENTERPRISE_PROXY_HOOKS,
    )

    ENTERPRISE_PROXY_HOOKS = dict(_ENTERPRISE_PROXY_HOOKS)
except ImportError:
    try:
        from litellm_enterprise.proxy.hooks.managed_files import (
            _PROXY_LiteLLMManagedFiles,
        )
        from litellm_enterprise.proxy.hooks.managed_vector_stores import (
            _PROXY_LiteLLMManagedVectorStores,
        )

        ENTERPRISE_PROXY_HOOKS = {
            "managed_files": _PROXY_LiteLLMManagedFiles,
            "managed_vector_stores": _PROXY_LiteLLMManagedVectorStores,
        }
    except ImportError:
        ENTERPRISE_PROXY_HOOKS = {}

_upgrade_managed_files_hook_from_workspace(ENTERPRISE_PROXY_HOOKS)

# List of all available hooks that can be enabled
PROXY_HOOKS = {
    "max_budget_limiter": _PROXY_MaxBudgetLimiter,
    "parallel_request_limiter": _PROXY_MaxParallelRequestsHandler_v3,
    "cache_control_check": _PROXY_CacheControlCheck,
    "responses_id_security": ResponsesIDSecurity,
    "litellm_skills": SkillsInjectionHook,
    "max_iterations_limiter": _PROXY_MaxIterationsHandler,
    "max_budget_per_session_limiter": _PROXY_MaxBudgetPerSessionHandler,
}

## FEATURE FLAG HOOKS ##
if os.getenv("LEGACY_MULTI_INSTANCE_RATE_LIMITING", "false").lower() == "true":
    PROXY_HOOKS["parallel_request_limiter"] = _PROXY_MaxParallelRequestsHandler


### update PROXY_HOOKS with ENTERPRISE_PROXY_HOOKS ###

PROXY_HOOKS.update(ENTERPRISE_PROXY_HOOKS)


def get_proxy_hook(
    hook_name: Union[
        Literal[
            "max_budget_limiter",
            "managed_files",
            "parallel_request_limiter",
            "cache_control_check",
        ],
        str,
    ],
):
    """
    Factory method to get a proxy hook instance by name
    """
    if hook_name not in PROXY_HOOKS:
        raise ValueError(
            f"Unknown hook: {hook_name}. Available hooks: {list(PROXY_HOOKS.keys())}"
        )
    return PROXY_HOOKS[hook_name]
