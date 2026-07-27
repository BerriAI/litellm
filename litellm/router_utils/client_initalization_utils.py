from typing import TYPE_CHECKING, Any

import anyio

from litellm.utils import calculate_max_parallel_requests

if TYPE_CHECKING:
    from litellm.router import Router as _Router

    LitellmRouter = _Router
else:
    LitellmRouter = Any


class InitalizeCachedClient:
    @staticmethod
    def set_max_parallel_requests_client(litellm_router_instance: LitellmRouter, model: dict):
        litellm_params = model.get("litellm_params", {})
        model_id = model["model_info"]["id"]
        rpm = litellm_params.get("rpm", None)
        tpm = litellm_params.get("tpm", None)
        max_parallel_requests = litellm_params.get("max_parallel_requests", None)
        calculated_max_parallel_requests = calculate_max_parallel_requests(
            rpm=rpm,
            max_parallel_requests=max_parallel_requests,
            tpm=tpm,
            default_max_parallel_requests=litellm_router_instance.default_max_parallel_requests,
        )
        limiter = litellm_router_instance._max_parallel_requests_semaphores.get(model_id)
        if calculated_max_parallel_requests is None or calculated_max_parallel_requests == 0:
            litellm_router_instance._max_parallel_requests_semaphores.pop(model_id, None)
        elif limiter is None:
            litellm_router_instance._max_parallel_requests_semaphores[model_id] = anyio.CapacityLimiter(
                calculated_max_parallel_requests
            )
        else:
            limiter.total_tokens = calculated_max_parallel_requests
