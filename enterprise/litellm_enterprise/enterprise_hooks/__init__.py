import os
from typing import Dict, Literal, Type, Union

from litellm.integrations.custom_logger import CustomLogger

from .managed_files import _PROXY_LiteLLMManagedFiles
from .parallel_request_limiter_v2 import _PROXY_MaxParallelRequestsHandler

ENTERPRISE_PROXY_HOOKS: Dict[str, Type[CustomLogger]] = {
    "managed_files": _PROXY_LiteLLMManagedFiles,
}


## FEATURE FLAG HOOKS ##

if os.getenv("EXPERIMENTAL_MULTI_INSTANCE_RATE_LIMITING", "false").lower() == "true":
    ENTERPRISE_PROXY_HOOKS["max_parallel_requests"] = _PROXY_MaxParallelRequestsHandler


def get_enterprise_proxy_hook(
    hook_name: Union[
        Literal[
            "managed_files",
            "max_parallel_requests",
        ],
        str,
    ]
):
    """
    Factory method to get a enterprise hook instance by name
    """
    if hook_name not in ENTERPRISE_PROXY_HOOKS:
        raise ValueError(
            f"Unknown hook: {hook_name}. Available hooks: {list(ENTERPRISE_PROXY_HOOKS.keys())}"
        )
    return ENTERPRISE_PROXY_HOOKS[hook_name]
