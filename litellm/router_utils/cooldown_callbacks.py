"""
Callbacks triggered on cooling down deployments
"""

from typing import TYPE_CHECKING, Any, Final

import litellm
from litellm._logging import verbose_logger
from litellm.integrations.prometheus_helpers import deployment_identity
from litellm.router_deprecated import renamed_router_param

if TYPE_CHECKING:
    from litellm.integrations.prometheus import PrometheusLogger
    from litellm.router_api import RoutingContext
else:
    PrometheusLogger = Any


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
    identity: Final = deployment_identity.identity_from_deployment(_deployment)

    # get the prometheus logger from in memory loggers
    prometheusLogger: Final[PrometheusLogger | None] = get_prometheus_logger_from_callbacks()

    if prometheusLogger is not None:
        prometheusLogger.set_deployment_complete_outage(
            litellm_model_name=identity.litellm_model_name,
            model_id=identity.model_id,
            api_base=identity.api_base,
            api_provider=identity.api_provider,
        )

        prometheusLogger.increment_deployment_cooled_down(
            litellm_model_name=identity.litellm_model_name,
            model_id=identity.model_id,
            api_base=identity.api_base,
            api_provider=identity.api_provider,
            exception_status=str(exception_status),
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
