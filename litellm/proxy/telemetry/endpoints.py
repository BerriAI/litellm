from collections.abc import Mapping
from typing import Annotated, Final, Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import CommonProxyErrors, LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.telemetry.runtime import TelemetryRuntime
from litellm.proxy.telemetry.settings import FLUSH_INTERVAL_SECONDS
from litellm.proxy.telemetry.store import StoredReport, TelemetryStore, store_read_errors
from litellm.telemetry.consent import OFF, REQUIRES, TelemetryConsent, parse_consent
from litellm.telemetry.records import TelemetryGroup
from litellm.telemetry.report import report_to_json
from litellm.telemetry.sample import sample_report

router: Final = APIRouter()


class TelemetryReportsResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    reports: tuple[StoredReport, ...]
    next_after: float | None
    next_after_id: str | None


def telemetry_store() -> TelemetryStore | None:
    from litellm.proxy.proxy_server import prisma_client, telemetry_runtime

    known: Final = telemetry_runtime.store
    if known is not None or prisma_client is None:
        return known
    return TelemetryStore(
        getattr(prisma_client.db, "writer", prisma_client.db),  # pyright: ignore[reportArgumentType]  # PrismaWrapper forwards raw queries via __getattr__
        telemetry_runtime.settings.retention_days,
    )


@router.get("/telemetry/reports", tags=["Telemetry"], response_model=TelemetryReportsResponse)
async def export_telemetry_reports(
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
    store: Annotated[TelemetryStore | None, Depends(telemetry_store)],
    after: Annotated[float, Query(description="window_end of the last report already exported")] = 0.0,
    after_id: Annotated[str, Query(description="id of the last report already exported")] = "",
    limit: Annotated[int, Query(ge=1, le=5000)] = 500,
) -> TelemetryReportsResponse:
    """Stored telemetry reports, oldest first, for installs that keep telemetry local instead of sending it.
    Page with ``after=next_after&after_id=next_after_id`` until ``next_after`` is null"""
    if user_api_key_dict.user_role not in (LitellmUserRoles.PROXY_ADMIN, LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY):
        raise HTTPException(status_code=403, detail="Only proxy admin roles can export telemetry reports")
    if store is None:
        raise HTTPException(status_code=500, detail=CommonProxyErrors.db_not_connected_error.value)
    reports: Final = await store.reports_after(after, after_id, limit)
    if len(reports) < limit:
        return TelemetryReportsResponse(reports=reports, next_after=None, next_after_id=None)
    return TelemetryReportsResponse(reports=reports, next_after=reports[-1].window_end, next_after_id=reports[-1].id)


_EVERYTHING: Final = TelemetryConsent(frozenset(TelemetryGroup))
_REPORT_JSON: Final = TypeAdapter(Mapping[str, JsonValue])


class TelemetryGroupInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    group: TelemetryGroup
    requires: TelemetryGroup | None
    enabled: bool


class TelemetrySettingsResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    groups: tuple[TelemetryGroupInfo, ...]
    stored_groups: tuple[TelemetryGroup, ...] | None
    vetoed: bool
    set_by_environment: bool
    environment_variables: tuple[str, ...]
    editable: bool
    destination: Literal["https", "local_table", "none"]
    flush_interval_seconds: float
    retention_days: int
    report: Mapping[str, JsonValue] | None
    report_is_sample: bool


class TelemetrySettingsUpdate(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    groups: tuple[str, ...]


def telemetry_runtime_dependency() -> TelemetryRuntime:
    from litellm.proxy.proxy_server import telemetry_runtime

    return telemetry_runtime


def _destination(runtime: TelemetryRuntime, has_database: bool) -> Literal["https", "local_table", "none"]:
    if runtime.settings.endpoint is not None:
        return "https"
    return "local_table" if has_database else "none"


_STORE_UNAVAILABLE: Final = "Could not reach the stored telemetry settings, try again"


async def _settings_response(
    runtime: TelemetryRuntime, stored: TelemetryConsent | None, has_database: bool, caller: UserAPIKeyAuth
) -> TelemetrySettingsResponse:
    policy: Final = runtime.policy
    effective: Final = policy.effective(stored) if policy is not None else OFF
    last: Final = runtime.last_report
    preview: Final = (
        last
        if last is not None
        else await sample_report(
            effective if effective != OFF else _EVERYTHING, litellm_version=runtime.litellm_version
        )
    )
    return TelemetrySettingsResponse(
        groups=tuple(
            TelemetryGroupInfo(group=group, requires=REQUIRES[group], enabled=group in effective.groups)
            for group in TelemetryGroup
        ),
        stored_groups=tuple(sorted(stored.groups, key=tuple(TelemetryGroup).index)) if stored is not None else None,
        vetoed=policy is None or policy.vetoed,
        set_by_environment=policy is None or policy.vetoed or policy.pinned is not None,
        environment_variables=policy.set_variables if policy is not None else (),
        editable=caller.user_role == LitellmUserRoles.PROXY_ADMIN
        and policy is not None
        and not policy.vetoed
        and policy.pinned is None
        and runtime.store is not None,
        destination=_destination(runtime, has_database),
        flush_interval_seconds=FLUSH_INTERVAL_SECONDS,
        retention_days=runtime.settings.retention_days,
        report=_REPORT_JSON.validate_python(report_to_json(preview)) if preview is not None else None,
        report_is_sample=last is None,
    )


async def _stored(runtime: TelemetryRuntime) -> TelemetryConsent | None:
    try:
        return await runtime.stored_consent()
    except store_read_errors() as e:
        verbose_proxy_logger.warning("telemetry: could not read the stored settings: %s", e)
        raise HTTPException(status_code=503, detail=_STORE_UNAVAILABLE) from e


@router.get("/telemetry/settings", tags=["Telemetry"], response_model=TelemetrySettingsResponse)
async def get_telemetry_settings(
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
    runtime: Annotated[TelemetryRuntime, Depends(telemetry_runtime_dependency)],
    store: Annotated[TelemetryStore | None, Depends(telemetry_store)],
) -> TelemetrySettingsResponse:
    """Which telemetry groups are on, whether env vars control them, where reports go and a last or sample report"""
    if user_api_key_dict.user_role not in (LitellmUserRoles.PROXY_ADMIN, LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY):
        raise HTTPException(status_code=403, detail="Only proxy admin roles can view telemetry settings")
    return await _settings_response(runtime, await _stored(runtime), store is not None, user_api_key_dict)


@router.put("/telemetry/settings", tags=["Telemetry"], response_model=TelemetrySettingsResponse)
async def update_telemetry_settings(
    update: TelemetrySettingsUpdate,
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
    runtime: Annotated[TelemetryRuntime, Depends(telemetry_runtime_dependency)],
) -> TelemetrySettingsResponse:
    """Store the telemetry groups for every worker, applied from the next report window"""
    if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise HTTPException(status_code=403, detail="Only the proxy admin can change telemetry settings")
    policy: Final = runtime.policy
    if policy is None or policy.vetoed or policy.pinned is not None:
        raise HTTPException(status_code=409, detail="Telemetry is set by LITELLM_TELEMETRY_* environment variables")
    store: Final = runtime.store
    if store is None:
        raise HTTPException(status_code=500, detail=CommonProxyErrors.db_not_connected_error.value)
    consent: Final = parse_consent(update.groups)
    if not isinstance(consent, TelemetryConsent):
        raise HTTPException(status_code=400, detail=consent.message())
    try:
        await store.save_consent(consent)
    except store_read_errors() as e:
        verbose_proxy_logger.warning("telemetry: could not save the settings: %s", e)
        raise HTTPException(status_code=503, detail=_STORE_UNAVAILABLE) from e
    verbose_proxy_logger.info(
        "telemetry: %s set groups to %s", user_api_key_dict.user_id, sorted(g.value for g in consent.groups)
    )
    return await _settings_response(runtime, consent, True, user_api_key_dict)
