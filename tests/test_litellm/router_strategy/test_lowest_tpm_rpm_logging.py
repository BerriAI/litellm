import time

from litellm.caching.caching import DualCache
from litellm.router_strategy.lowest_tpm_rpm import (
    LowestTPMLoggingHandler as LowestTPMLoggingHandler_v1,
)


def test_lowest_tpm_rpm_v1_model_info_none():
    """log_success_event should not crash when model_info is None (v1 handler)."""
    logger = LowestTPMLoggingHandler_v1(router_cache=DualCache())
    kwargs = {
        "litellm_params": {
            "metadata": {"model_group": "gpt-4"},
            "model_info": None,
        }
    }
    logger.log_success_event(
        kwargs=kwargs, response_obj={}, start_time=time.time(), end_time=time.time()
    )
