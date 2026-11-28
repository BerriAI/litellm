"""
Callbacks triggered on cooling down deployments
"""

import copy
import time
from typing import TYPE_CHECKING, Any, Final

import litellm
from litellm._logging import verbose_logger
from litellm.router_deprecated import renamed_router_param

if TYPE_CHECKING:
    from litellm.integrations.prometheus import PrometheusLogger
    from litellm.router_api import RoutingContext
else:
    PrometheusLogger = Any


def resolved_model_and_provider(model: str, custom_llm_provider: str | None) -> tuple[str, str]:
    try:
        resolved_model, provider, _, _ = litellm.get_llm_provider(model=model, custom_llm_provider=custom_llm_provider)
    except Exception:
        return model, ""
    return resolved_model, provider


def wall_seconds() -> float:
    return time.time()


async def cooldown_seconds_left(ctx: "RoutingContext", deployment_id: str, cooldown_time: float) -> float:
    """The seconds until the router's stored cooldown of the deployment ends, or ``cooldown_time`` when none is read."""
    try:
        active: Final = await ctx.health().cooldowns.async_get_active_cooldowns([deployment_id], parent_otel_span=None)
    except Exception:  # noqa: BLE001  # an unreadable cooldown keeps the length the event carried
        return cooldown_time
    stored: Final = next((cooldown for active_id, cooldown in active if active_id == deployment_id), None)
    return cooldown_time if stored is None else stored["timestamp"] + stored["cooldown_time"] - wall_seconds()


@renamed_router_param("litellm_router_instance")
async def router_cooldown_event_callback(
    ctx: "RoutingContext",
    deployment_id: str,
    exception_status: str | int,
    cooldown_time: float | None,
):
    """
    Callback triggered when a deployment is put into cooldown by litellm

    - Updates deployment state on Prometheus
    - Increments cooldown metric for deployment on Prometheus
    """
    verbose_logger.debug("In router_cooldown_event_callback - updating prometheus")
    _deployment: Final = ctx.registry().deployment(deployment_id)
    if _deployment is None:
        verbose_logger.warning(
            "in router_cooldown_event_callback but _deployment is None for deployment_id=%s. Doing nothing",
            deployment_id,
        )
        return
    _litellm_params: Final = _deployment.litellm_params
    temp_litellm_params = copy.deepcopy(_litellm_params)
    temp_litellm_params = dict(temp_litellm_params)
    _model_name: Final = _deployment.model_name or ""
    _api_base: Final = litellm.get_api_base(model=_model_name, optional_params=temp_litellm_params) or ""
    model_id: Final = _deployment.model_info.id

    litellm_model_name, llm_provider = resolved_model_and_provider(
        temp_litellm_params.get("model") or "", temp_litellm_params.get("custom_llm_provider")
    )

    # get the prometheus logger from in memory loggers
    prometheusLogger: Final[PrometheusLogger | None] = get_prometheus_logger_from_callbacks()

    if prometheusLogger is not None:
        prometheusLogger.set_deployment_complete_outage(
            litellm_model_name=litellm_model_name,
            model_id=model_id,
            api_base=_api_base,
            api_provider=llm_provider,
        )
        cooldown_length: Final = (
            ctx.health().cooldowns.default_cooldown_time if cooldown_time is None else cooldown_time
        )
        prometheusLogger.set_deployment_cooldown_time(model_id=model_id, cooldown_time=cooldown_length)

        prometheusLogger.increment_deployment_cooled_down(
            litellm_model_name=litellm_model_name,
            model_id=str(model_id),
            api_base=_api_base,
            api_provider=llm_provider,
            exception_status=str(exception_status),
        )
        prometheusLogger.set_deployment_cooldown_time(
            model_id=model_id, cooldown_time=await cooldown_seconds_left(ctx, deployment_id, cooldown_length)
        )

    return


def get_prometheus_logger_from_callbacks() -> PrometheusLogger | None:
    """
    Checks if prometheus is a initalized callback, if yes returns it
    """
    from litellm.integrations.prometheus import PrometheusLogger

    if PrometheusLogger is None:
        return None

    for _callback in litellm._async_success_callback:
        if isinstance(_callback, PrometheusLogger):
            return _callback
    for global_callback in litellm.callbacks:
        if isinstance(global_callback, PrometheusLogger):
            return global_callback

    return None


_get_prometheus_logger_from_callbacks = get_prometheus_logger_from_callbacks
