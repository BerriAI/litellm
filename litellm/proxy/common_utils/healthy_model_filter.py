"""Opt-in health filtering shared by the model listing endpoints.

`/v1/models`, `GET /v1/models/{id}` and `/v1/model/info` hide models whose
backing deployments are all marked unhealthy by background health checks, either
per request via `healthy_only=true` or proxy-wide via
`general_settings.model_list_healthy_only: true`. Both are opt-in: with neither
set the listings are returned unfiltered and no health lookup runs at all.

The proxy-wide setting is what an operator turns on so every client (UI, SDK,
raw API) sees only reachable models without having to pass the query parameter.
It also makes the background health check loop keep the deployment health cache
populated, so `background_health_checks: true` is the only other setting needed.
The per-request parameter reads that same cache, so on its own it needs the
cache to be filled by either this setting or `enable_health_check_routing`.

Filtering is presentation-only and always fails open: it answers "should this
model be advertised?", never "should a request for it be attempted?". A hidden
model stays callable, and an absent, stale or empty health state hides nothing.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Final

from litellm._logging import verbose_proxy_logger

if TYPE_CHECKING:
    from litellm.router import Router

MODEL_LIST_HEALTHY_ONLY_SETTING: Final = "model_list_healthy_only"


def is_healthy_only_listing_default(general_settings: Mapping[str, object]) -> bool:
    """Whether `model_list_healthy_only` filters every listing on this proxy.

    Only a real `true` counts, so a quoted YAML value never silently starts
    hiding models. This also tells the background health check loop to keep the
    deployment health cache populated, which is the state the filter reads.
    """
    return general_settings.get(MODEL_LIST_HEALTHY_ONLY_SETTING, False) is True


def is_healthy_only_enabled(
    healthy_only: bool | None,
    general_settings: Mapping[str, object],
) -> bool:
    """Whether the health filter applies to this request.

    The per-request `healthy_only=true` and the proxy-wide
    `model_list_healthy_only` setting are independent opt-ins: either one turns
    the filter on, and a request cannot turn the proxy-wide setting back off
    (`healthy_only=false` is the unset default, indistinguishable from absent).
    """
    if healthy_only:
        return True
    return is_healthy_only_listing_default(general_settings)


async def get_hidden_unhealthy_model_names(
    healthy_only: bool | None,
    general_settings: Mapping[str, object],
    llm_router: Router | None,
) -> set[str]:
    """Model names to hide from a listing, empty when the filter is off.

    Empty is also the fail-open answer whenever the router cannot report health
    (no router, no background health checks, stale state, `allowed_fails_policy`
    configured), so callers apply it unconditionally and simply hide nothing.
    """
    if llm_router is None or not is_healthy_only_enabled(healthy_only, general_settings):
        return set()
    unhealthy_names: Final = await llm_router.async_get_fully_unhealthy_model_names()
    if not unhealthy_names:
        verbose_proxy_logger.debug(
            "healthy-only model listing is enabled but no unhealthy deployment state is "
            "available (requires background_health_checks); returning unfiltered model list"
        )
    return unhealthy_names
