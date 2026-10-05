from typing import Final

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.llms.anthropic.cache_aware_routing import predict_arm

__all__: Final = ("has_request_transforms", "predict_arm")


def has_request_transforms() -> bool:
    from litellm.proxy.hooks import PROXY_HOOKS

    builtins: Final = frozenset(PROXY_HOOKS.values())
    hooks: Final = ("async_pre_call_hook", "async_pre_request_hook", "async_pre_call_deployment_hook")
    callbacks: Final = litellm.logging_callback_manager.get_custom_loggers_for_type(callback_type=CustomLogger)
    return any(
        type(callback) not in builtins
        and any(getattr(type(callback), hook) is not getattr(CustomLogger, hook) for hook in hooks)
        for callback in callbacks
    )
