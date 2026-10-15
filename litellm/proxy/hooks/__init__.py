import os
from typing import Final, Literal

from . import *
from .autorouter_baseline_cache import AutoRouterBaselineCache
from .cache_control_check import (  # noqa: F401  # backwards-compatible package export
    PROXY_CacheControlCheck,
    _PROXY_CacheControlCheck,  # pyright: ignore[reportPrivateUsage]  # backwards-compatible package export
)
from .litellm_skills import SkillsInjectionHook
from .max_budget_per_session_limiter import (  # noqa: F401  # backwards-compatible package export
    PROXY_MaxBudgetPerSessionHandler,
    _PROXY_MaxBudgetPerSessionHandler,  # pyright: ignore[reportPrivateUsage]  # backwards-compatible package export
)
from .max_iterations_limiter import (  # noqa: F401  # backwards-compatible package export
    PROXY_MaxIterationsHandler,
    _PROXY_MaxIterationsHandler,  # pyright: ignore[reportPrivateUsage]  # backwards-compatible package export
)
from .parallel_request_limiter import (  # noqa: F401  # backwards-compatible package export
    PROXY_MaxParallelRequestsHandler,
    _PROXY_MaxParallelRequestsHandler,  # pyright: ignore[reportPrivateUsage]  # backwards-compatible package export
)
from .parallel_request_limiter_v3 import (  # noqa: F401  # backwards-compatible package export
    PROXY_MaxParallelRequestsHandler_v3,
    _PROXY_MaxParallelRequestsHandler_v3,  # pyright: ignore[reportPrivateUsage]  # backwards-compatible package export
)
from .prompt_cache_prediction import PromptCacheObserver
from .responses_id_security import ResponsesIDSecurity
from .sensitive_data_routing import (  # noqa: F401  # backwards-compatible package export
    PROXY_SensitiveDataRoutingHandler,
    _PROXY_SensitiveDataRoutingHandler,  # pyright: ignore[reportPrivateUsage]  # backwards-compatible package export
)

# List of all available hooks that can be enabled.
# Defined before the enterprise import below so that any module re-imported
# transitively through `enterprise.enterprise_hooks` can resolve `PROXY_HOOKS`
# and `get_proxy_hook` from this partially-initialized module without circling.
PROXY_HOOKS: Final = {
    "parallel_request_limiter": PROXY_MaxParallelRequestsHandler_v3,
    "cache_control_check": PROXY_CacheControlCheck,
    "responses_id_security": ResponsesIDSecurity,
    "litellm_skills": SkillsInjectionHook,
    "max_iterations_limiter": PROXY_MaxIterationsHandler,
    "max_budget_per_session_limiter": PROXY_MaxBudgetPerSessionHandler,
    "sensitive_data_routing": PROXY_SensitiveDataRoutingHandler,
    "prompt_cache_prediction": PromptCacheObserver,
    "autorouter_baseline_cache": AutoRouterBaselineCache,
}

## FEATURE FLAG HOOKS ##
if os.getenv("LEGACY_MULTI_INSTANCE_RATE_LIMITING", "false").lower() == "true":
    PROXY_HOOKS["parallel_request_limiter"] = PROXY_MaxParallelRequestsHandler


def get_proxy_hook(
    hook_name: Literal["managed_files", "parallel_request_limiter", "cache_control_check"] | str,
):
    """
    Factory method to get a proxy hook instance by name
    """
    if hook_name not in PROXY_HOOKS:
        raise ValueError(f"Unknown hook: {hook_name}. Available hooks: {list(PROXY_HOOKS.keys())}")
    return PROXY_HOOKS[hook_name]


### CHECK IF ENTERPRISE HOOKS ARE AVAILABLE ####

try:
    from enterprise.enterprise_hooks import ENTERPRISE_PROXY_HOOKS
except ImportError:
    ENTERPRISE_PROXY_HOOKS = {}


### update PROXY_HOOKS with ENTERPRISE_PROXY_HOOKS ###

PROXY_HOOKS.update(ENTERPRISE_PROXY_HOOKS)
