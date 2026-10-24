import os
from typing import Final, Literal

from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_logger import CustomLogger

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
# Defined before the litellm_enterprise hook imports below so that any module re-imported
# transitively through them can resolve `PROXY_HOOKS`
# and `get_proxy_hook` from this partially-initialized module without circling.
PROXY_HOOKS: Final[dict[str, type[CustomLogger]]] = {  # mutable-ok: shared with cycle importers, extended below
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


try:
    from litellm_enterprise.proxy.hooks.managed_files import PROXY_LiteLLMManagedFiles
    from litellm_enterprise.proxy.hooks.managed_vector_stores import PROXY_LiteLLMManagedVectorStores
except ImportError as litellm_enterprise_import_error:
    verbose_proxy_logger.warning(
        "Managed files and managed vector stores are disabled because litellm_enterprise failed to import: %s",
        litellm_enterprise_import_error,
    )
else:
    PROXY_HOOKS["managed_files"] = PROXY_LiteLLMManagedFiles
    PROXY_HOOKS["managed_vector_stores"] = PROXY_LiteLLMManagedVectorStores
