from typing import Any, Dict, List, Optional
import asyncio
import litellm
from litellm._logging import verbose_router_logger

async def routed_acompletion_call(
    router: Any,
    *,
    deployment: Dict[str, Any],
    data: Dict[str, Any],
    messages: List[Dict[str, Any]],
    kwargs: Dict[str, Any],
    parent_otel_span: Optional[Any],
) -> Any:
    """Internal: run the async completion call with pre-call checks and rpm gating.

    This mirrors the legacy Router path but lives here to reduce router.py size.
    Behavior is identical; only used when the internal extracted flag is set.
    """
    model_client = router._get_async_openai_model_client(
        deployment=deployment,
        kwargs=kwargs,
    )
    model_name = deployment["model_name"]
    router.total_calls[model_name] += 1

    input_kwargs = {
        **data,
        "messages": messages,
        "caching": router.cache_responses,
        "client": model_client,
        **kwargs,
    }
    _response = litellm.acompletion(**input_kwargs)

    logging_obj = kwargs.get("litellm_logging_obj", None)

    rpm_semaphore = router._get_client(
        deployment=deployment,
        kwargs=kwargs,
        client_type="max_parallel_requests",
    )
    if rpm_semaphore is not None and isinstance(rpm_semaphore, asyncio.Semaphore):
        async with rpm_semaphore:
            await router.async_routing_strategy_pre_call_checks(
                deployment=deployment,
                logging_obj=logging_obj,
                parent_otel_span=parent_otel_span,
            )
            response = await _response
    else:
        await router.async_routing_strategy_pre_call_checks(
            deployment=deployment,
            logging_obj=logging_obj,
            parent_otel_span=parent_otel_span,
        )
        response = await _response
    return response
