from collections.abc import Collection
from typing import TYPE_CHECKING, Any, Final

from litellm._logging import redact_secrets, verbose_router_logger
from litellm.constants import MAX_EXCEPTION_MESSAGE_LENGTH
from litellm.router_deprecated import renamed_router_param
from litellm.router_utils.cooldown_handlers import (
    async_get_cooldown_deployments_with_debug_info,
    get_cooldown_deployment_info,
)
from litellm.types.integrations.slack_alerting import AlertType
from litellm.types.router import RouterRateLimitError

if TYPE_CHECKING:
    from opentelemetry.trace import Span as _Span

    from litellm.router_api import RoutingContext

    Span = _Span
else:
    Span = Any


@renamed_router_param("litellm_router_instance")
async def send_llm_exception_alert(
    ctx: "RoutingContext | None",
    request_kwargs: dict,
    error_traceback_str: str,
    original_exception,
):
    """
    Only runs if router.slack_alerting_logger is set
    Sends a Slack / MS Teams alert for the LLM API call failure. Only if router.slack_alerting_logger is set.

    Parameters:
        ctx (RoutingContext): The router's routing context.
        original_exception (Any): The original exception that occurred.

    Returns:
        None
    """
    if ctx is None:
        return

    slack_alerting_logger: Final = ctx.observers().slack_alerting_logger
    if slack_alerting_logger is None:
        return

    if "proxy_server_request" in request_kwargs:
        # Do not send any alert if it's a request from litellm proxy server request
        # the proxy is already instrumented to send LLM API call failures
        return

    litellm_debug_info: Final = getattr(original_exception, "litellm_debug_info", None)
    exception_str = str(original_exception)
    if litellm_debug_info is not None:
        exception_str += litellm_debug_info
    exception_str += f"\n\n{error_traceback_str[:MAX_EXCEPTION_MESSAGE_LENGTH]}"

    # Redact secrets before sending to external service (Slack / MS Teams)
    exception_str = redact_secrets(exception_str)

    await slack_alerting_logger.send_alert(
        message=f"LLM API call failed: `{exception_str}`",
        level="High",
        alert_type=AlertType.llm_exceptions,
        alerting_metadata={},
    )


@renamed_router_param("litellm_router_instance")
async def async_raise_no_deployment_exception(
    ctx: "RoutingContext",
    model: str,
    parent_otel_span: Span | None,
    route_model_ids: Collection[str] | None = None,
):
    """
    Raises a RouterRateLimitError if no deployment is found for the given model.
    """
    verbose_router_logger.info("get_available_deployment for model: %s, No deployment available", model)
    model_ids: Final = ctx.registry().names(model, kind="ids")
    _cooldown_time: Final = ctx.health().cooldowns.get_min_cooldown(
        model_ids=model_ids, parent_otel_span=parent_otel_span
    )
    _cooldown_list: Final = await async_get_cooldown_deployments_with_debug_info(
        ctx=ctx,
        parent_otel_span=parent_otel_span,
    )
    verbose_router_logger.info(
        "No deployment found for model: %s, cooldown_list with debug info: %s", model, _cooldown_list
    )

    cooldown_list_ids: Final = [cooldown_model[0] for cooldown_model in (_cooldown_list or [])]
    return RouterRateLimitError(
        model=model,
        cooldown_time=_cooldown_time,
        enable_pre_call_checks=ctx.settings().enable_pre_call_checks,
        cooldown_list=cooldown_list_ids,
        model_ids=model_ids,
        cooldown_deployments=(
            None if route_model_ids is None else get_cooldown_deployment_info(ctx, route_model_ids, cooldown_list_ids)
        ),
    )
