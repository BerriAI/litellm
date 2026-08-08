"""
Soft and hard limits on SGR (successful gateway requests).

An allowance can come from two places. An enterprise license may carry
``max_sgr`` and an optional ``sgr_window``, which makes the contracted volume
visible to the deployment without LiteLLM having to receive any telemetry back.
``general_settings`` can also set it directly, which is what lets a customer
self-serve a threshold lower than their contract, and which wins when both are
present. Absent both, the window is a calendar year.

Crossing a threshold alerts, it does not reject: this reads the same
``LiteLLM_DailyGatewayRequests`` rollup the admin UI reads, on a scheduler, so
it is never on the request path and cannot fail a request.
"""

from collections.abc import Mapping
from datetime import datetime, timezone
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, assert_never

from pydantic import BaseModel, Field, TypeAdapter

from litellm._logging import verbose_proxy_logger
from litellm.types.proxy.gateway_requests import (
    SGRLimitConfig,
    SGRLimitState,
    SGRLimitStatus,
    SGRLimitWindow,
)

if TYPE_CHECKING:
    from litellm.proxy._types import EnterpriseLicenseData
    from litellm.proxy.utils import PrismaClient, ProxyLogging

DEFAULT_SGR_SOFT_LIMIT_PERCENT: Final = 0.8
DEFAULT_SGR_LIMIT_WINDOW: Final = SGRLimitWindow.YEAR


class SGRLimitSettings(BaseModel):
    """The `general_settings` keys that configure an SGR allowance."""

    sgr_limit: int | None = Field(
        default=None,
        ge=1,
        description="Hard limit on successful gateway requests per window. Overrides a license's max_sgr",
    )
    sgr_soft_limit_percent: float = Field(
        default=DEFAULT_SGR_SOFT_LIMIT_PERCENT,
        gt=0,
        le=1,
        description="Fraction of the hard limit at which the soft alert fires",
    )
    sgr_limit_window: SGRLimitWindow | None = Field(
        default=None,
        description="Period the limit is counted over, calendar aligned in UTC. Overrides a license's sgr_window",
    )


_SETTINGS_ADAPTER: Final = TypeAdapter(SGRLimitSettings)


def _license_sgr_limit(license_data: "EnterpriseLicenseData | None") -> int | None:
    if license_data is None:
        return None
    max_sgr: Final = license_data.get("max_sgr")
    return max_sgr if isinstance(max_sgr, int) and max_sgr > 0 else None


def _license_sgr_window(license_data: "EnterpriseLicenseData | None") -> SGRLimitWindow | None:
    if license_data is None:
        return None
    window: Final = license_data.get("sgr_window")
    return SGRLimitWindow(window) if window in tuple(member.value for member in SGRLimitWindow) else None


def resolve_sgr_limit(
    *,
    general_settings: Mapping[str, object],
    license_data: "EnterpriseLicenseData | None",
) -> SGRLimitConfig | None:
    """The allowance in force, or None when the deployment has not been given one."""
    try:
        settings: Final = _SETTINGS_ADAPTER.validate_python(
            MappingProxyType(
                {
                    key: value
                    for key, value in general_settings.items()
                    if key in SGRLimitSettings.model_fields and value is not None
                }
            )
        )
    except ValueError:
        verbose_proxy_logger.warning(
            "SGR limit - ignoring invalid sgr_limit settings in general_settings", exc_info=True
        )
        return None

    limit: Final = settings.sgr_limit if settings.sgr_limit is not None else _license_sgr_limit(license_data)
    if limit is None:
        return None

    return SGRLimitConfig(
        limit=limit,
        soft_limit=max(1, int(limit * settings.sgr_soft_limit_percent)),
        window=settings.sgr_limit_window or _license_sgr_window(license_data) or DEFAULT_SGR_LIMIT_WINDOW,
    )


def sgr_window_start(window: SGRLimitWindow, now: datetime) -> str:
    """First day of the current window, as the YYYY-MM-DD the rollup is keyed by."""
    match window:
        case SGRLimitWindow.MONTH:
            return now.strftime("%Y-%m-01")
        case SGRLimitWindow.YEAR:
            return now.strftime("%Y-01-01")
        case _:
            assert_never(window)


def evaluate_sgr_limit(config: SGRLimitConfig, successful_requests: int) -> SGRLimitState:
    if successful_requests >= config.limit:
        return SGRLimitState.HARD_EXCEEDED
    if successful_requests >= config.soft_limit:
        return SGRLimitState.SOFT_EXCEEDED
    return SGRLimitState.UNDER


_WINDOW_TOTAL_SQL: Final = """
    SELECT COALESCE(SUM(successful_requests), 0)::bigint AS successful_requests
    FROM "LiteLLM_DailyGatewayRequests"
    WHERE date >= $1
"""


class _WindowTotalRow(BaseModel):
    successful_requests: int


_ROWS_ADAPTER: Final = TypeAdapter(tuple[_WindowTotalRow, ...])


async def get_sgr_limit_status(
    *,
    prisma_client: "PrismaClient",
    config: SGRLimitConfig,
    now: datetime,
) -> SGRLimitStatus:
    """Sum the window's successful requests and place them against the allowance."""
    window_start: Final = sgr_window_start(config.window, now)
    raw_rows: Final = await prisma_client.db.query_raw(  # pyright: ignore[reportAny]  # untyped prisma client
        _WINDOW_TOTAL_SQL,
        window_start,
    )
    rows: Final = _ROWS_ADAPTER.validate_python(raw_rows or ())
    successful_requests: Final = rows[0].successful_requests if rows else 0

    return SGRLimitStatus(
        limit=config.limit,
        soft_limit=config.soft_limit,
        window=config.window,
        window_start=window_start,
        successful_requests=successful_requests,
        state=evaluate_sgr_limit(config, successful_requests),
    )


async def check_sgr_limit(
    prisma_client: "PrismaClient",
    proxy_logging_obj: "ProxyLogging",
) -> None:
    """
    Scheduler entrypoint. Never raises: a limit check must not kill the job.

    Config is resolved on every run rather than captured once, so a license or
    a `general_settings` value that arrives after startup is picked up.
    """
    from litellm.proxy.proxy_server import _license_check, general_settings

    try:
        config: Final = resolve_sgr_limit(
            general_settings=general_settings,
            license_data=_license_check.airgapped_license_data,
        )
        if config is None:
            return
        status: Final = await get_sgr_limit_status(
            prisma_client=prisma_client,
            config=config,
            now=datetime.now(timezone.utc),
        )
        await proxy_logging_obj.slack_alerting_instance.sgr_limit_alert(status=status)
    except Exception:  # noqa: BLE001  # a failed check must not stop the scheduler
        verbose_proxy_logger.warning("SGR limit - check failed, retrying on the next run", exc_info=True)
