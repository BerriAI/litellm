from typing import Literal, Union

from . import *
from .cache_control_check import _PROXY_CacheControlCheck
from .managed_files import _PROXY_LiteLLMManagedFiles
from .max_budget_limiter import _PROXY_MaxBudgetLimiter
from .parallel_request_limiter import _PROXY_MaxParallelRequestsHandler

# List of all available hooks that can be enabled
PROXY_HOOKS = {
    "max_budget_limiter": _PROXY_MaxBudgetLimiter,
    "managed_files": _PROXY_LiteLLMManagedFiles,
    "parallel_request_limiter": _PROXY_MaxParallelRequestsHandler,
    "cache_control_check": _PROXY_CacheControlCheck,
}


def get_proxy_hook(
    hook_name: Union[
        Literal[
            "max_budget_limiter",
            "managed_files",
            "parallel_request_limiter",
            "cache_control_check",
        ],
        str,
    ]
):
    """
    Factory method to get a proxy hook instance by name
    """
    if hook_name not in PROXY_HOOKS:
        raise ValueError(
            f"Unknown hook: {hook_name}. Available hooks: {list(PROXY_HOOKS.keys())}"
        )
    return PROXY_HOOKS[hook_name]
