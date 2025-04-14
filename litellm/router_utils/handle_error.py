from typing import TYPE_CHECKING, Any, Optional, Union

from litellm._logging import verbose_router_logger
from litellm.constants import MAX_EXCEPTION_MESSAGE_LENGTH
from litellm.router_utils.cooldown_handlers import (
    _async_get_cooldown_deployments_with_debug_info,
)
from litellm.types.integrations.slack_alerting import AlertType
from litellm.types.router import RouterRateLimitError

if TYPE_CHECKING:
    from opentelemetry.trace import Span as _Span

    from litellm.router import Router as _Router

    LitellmRouter = _Router
    Span = Union[_Span, Any]
else:
    LitellmRouter = Any
    Span = Any


async def send_llm_exception_alert(
    litellm_router_instance: LitellmRouter,
    request_kwargs: dict,
    error_traceback_str: str,
    original_exception,
):
    """
    Only runs if router.slack_alerting_logger is set
    Sends a Slack / MS Teams alert for the LLM API call failure. Only if router.slack_alerting_logger is set.

    Parameters:
        litellm_router_instance (_Router): The LitellmRouter instance.
        original_exception (Any): The original exception that occurred.

    Returns:
        None
    """
    if litellm_router_instance is None:
        return

    if not hasattr(litellm_router_instance, "slack_alerting_logger"):
        return

    if litellm_router_instance.slack_alerting_logger is None:
        return

    if "proxy_server_request" in request_kwargs:
        # Do not send any alert if it's a request from litellm proxy server request
        # the proxy is already instrumented to send LLM API call failures
        return

    litellm_debug_info = getattr(original_exception, "litellm_debug_info", None)
    exception_str = str(original_exception)
    if litellm_debug_info is not None:
        exception_str += litellm_debug_info
    exception_str += f"\n\n{error_traceback_str[:MAX_EXCEPTION_MESSAGE_LENGTH]}"

    await litellm_router_instance.slack_alerting_logger.send_alert(
        message=f"LLM API call failed: `{exception_str}`",
        level="High",
        alert_type=AlertType.llm_exceptions,
        alerting_metadata={},
    )


async def async_raise_no_deployment_exception(
    litellm_router_instance: LitellmRouter, model: str, parent_otel_span: Optional[Span]
):
    """
    Raises a RouterRateLimitError if no deployment is found for the given model.
    """
    verbose_router_logger.info(
        f"get_available_deployment for model: {model}, No deployment available"
    )
    model_ids = litellm_router_instance.get_model_ids(model_name=model)
    _cooldown_time = litellm_router_instance.cooldown_cache.get_min_cooldown(
        model_ids=model_ids, parent_otel_span=parent_otel_span
    )
    _cooldown_list = await _async_get_cooldown_deployments_with_debug_info(
        litellm_router_instance=litellm_router_instance,
        parent_otel_span=parent_otel_span,
    )
    return RouterRateLimitError(
        model=model,
        cooldown_time=_cooldown_time,
        enable_pre_call_checks=litellm_router_instance.enable_pre_call_checks,
        cooldown_list=_cooldown_list,
    )
