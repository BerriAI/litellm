import os
from typing import Literal, Union

from . import *
from .cache_control_check import _PROXY_CacheControlCheck
from .litellm_skills import SkillsInjectionHook
from .max_budget_limiter import _PROXY_MaxBudgetLimiter
from .max_budget_per_session_limiter import _PROXY_MaxBudgetPerSessionHandler
from .max_iterations_limiter import _PROXY_MaxIterationsHandler
from .parallel_request_limiter import _PROXY_MaxParallelRequestsHandler
from .parallel_request_limiter_v3 import _PROXY_MaxParallelRequestsHandler_v3
from .responses_id_security import ResponsesIDSecurity

# List of all available hooks that can be enabled.
# Defined before the enterprise import below so that any module re-imported
# transitively through `enterprise.enterprise_hooks` can resolve `PROXY_HOOKS`
# and `get_proxy_hook` from this partially-initialized module without circling.
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


### CHECK IF ENTERPRISE HOOKS ARE AVAILABLE ####

try:
    from enterprise.enterprise_hooks import ENTERPRISE_PROXY_HOOKS
except ImportError:
    ENTERPRISE_PROXY_HOOKS = {}


### update PROXY_HOOKS with ENTERPRISE_PROXY_HOOKS ###

PROXY_HOOKS.update(ENTERPRISE_PROXY_HOOKS)
