from typing import TYPE_CHECKING, Any

import litellm
from litellm._logging import verbose_router_logger
from litellm.integrations.custom_logger import CustomLogger


async def log_success_fallback_event(original_model_group: str, kwargs: dict):
    for _callback in litellm.callbacks:
        if isinstance(_callback, CustomLogger):
            try:
                await _callback.log_success_fallback_event(
                    original_model_group=original_model_group, kwargs=kwargs
                )
            except Exception as e:
                verbose_router_logger.error(
                    f"Error in log_success_fallback_event: {(str(e))}"
                )
                pass


async def log_failure_fallback_event(original_model_group: str, kwargs: dict):
    for _callback in litellm.callbacks:
        if isinstance(_callback, CustomLogger):
            try:
                await _callback.log_failure_fallback_event(
                    original_model_group=original_model_group, kwargs=kwargs
                )
            except Exception as e:
                verbose_router_logger.error(
                    f"Error in log_failure_fallback_event: {(str(e))}"
                )
                pass
