from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import TYPE_CHECKING, Annotated, Final, Protocol

from fastapi import APIRouter, Depends, HTTPException
from pydantic import JsonValue, TypeAdapter, ValidationError

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.sensitive_data_masker import SensitiveDataMasker
from litellm.proxy._types import CommonProxyErrors, LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_utils.encrypt_decrypt_utils import decrypt_value_helper, encrypt_value_helper
from litellm.proxy.management_endpoints.common_utils import user_api_key_has_admin_view
from litellm.repositories.config_repository import ConfigRepository
from litellm.types.proxy.ternary_endpoints import (
    TernaryDryRunRequest,
    TernaryDryRunResponse,
    TernaryDryRunSummary,
    TernaryExportRequest,
    TernaryExportResponse,
    TernaryMessageResponse,
    TernarySettings,
    TernarySettingsUpdate,
    TernarySettingsView,
)

if TYPE_CHECKING:
    from litellm.integrations.ternary.export_engine import FocusFrame
    from litellm.integrations.ternary.ternary_logger import SettingsTernaryLogger, TernaryLogger
    from litellm.repositories.config_repository import ConfigParam

router: Final = APIRouter()

TERNARY_SETTINGS_PARAM_NAME: Final = "ternary_settings"

_DRY_RUN_SAMPLE_ROWS: Final = 50
_NOT_CONFIGURED: Final = "Ternary settings not found. Initialize them first with /ternary/init"
_RESTART_NOTE: Final = "Restart the proxy to apply the change to scheduled exports"
_CONFIG_FILE_OWNS: Final = (
    'Ternary is configured in the proxy config file (callbacks: ["ternary"]). Change or remove it there'
)

_sensitive_masker: Final = SensitiveDataMasker()
_rows_adapter: Final = TypeAdapter(Sequence[Mapping[str, JsonValue]])


class _ConfigTable(Protocol):
    async def delete(self, *, where: Mapping[str, str]) -> object: ...


class _ConfigStore(Protocol):
    async def get_param(self, param_name: str) -> ConfigParam | None: ...

    async def set_param(self, param_name: str, param_value: object) -> ConfigParam: ...

    @property
    def table(self) -> _ConfigTable: ...


class _ColumnValues(Protocol):
    def to_list(self) -> Sequence[object]: ...


class _Columns(Protocol):
    @property
    def columns(self) -> Sequence[str]: ...

    def get_column(self, name: str) -> _ColumnValues: ...


class _UsageSource(Protocol):
    async def get_usage_data(self, *, limit: int | None) -> FocusFrame: ...


class _Exporter(Protocol):
    def scheduled_window(self, now: datetime) -> _Window: ...

    async def export_days(
        self, *, start_time_utc: datetime, end_time_utc: datetime, limit: int | None
    ) -> Sequence[str]: ...


class _Window(Protocol):
    @property
    def start_time(self) -> datetime: ...

    @property
    def end_time(self) -> datetime: ...


def _http_error(status_code: int, error: str) -> HTTPException:
    return HTTPException(status_code=status_code, detail={"error": error})


class TernarySettingsStore:
    """The ``ternary_settings`` row in ``LiteLLM_Config``, with the API key encrypted at rest."""

    def __init__(self, config: _ConfigStore) -> None:
        self._config: Final = config

    async def get(self) -> TernarySettings | None:
        param: Final = await self._config.get_param(TERNARY_SETTINGS_PARAM_NAME)
        stored_value: Final[object] = None if param is None else param.param_value
        if not stored_value:
            return None
        stored: Final = TernarySettings.model_validate(stored_value)
        api_key: Final = decrypt_value_helper(stored.api_key, key="ternary_api_key", exception_type="error")
        if api_key is None:
            raise _http_error(500, "Failed to decrypt the Ternary API key. Check your salt key configuration.")
        return stored.model_copy(update={"api_key": api_key})

    async def save(self, settings: TernarySettings) -> None:
        encrypted: Final = settings.model_copy(update={"api_key": encrypt_value_helper(settings.api_key)})
        await self._config.set_param(TERNARY_SETTINGS_PARAM_NAME, encrypted.model_dump(mode="json"))

    async def delete(self) -> bool:
        if await self._config.get_param(TERNARY_SETTINGS_PARAM_NAME) is None:
            return False
        await self._config.table.delete(where={"param_name": TERNARY_SETTINGS_PARAM_NAME})
        return True


def ternary_settings_store() -> TernarySettingsStore:
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise _http_error(500, CommonProxyErrors.db_not_connected_error.value)
    return TernarySettingsStore(ConfigRepository(prisma_client))


def ternary_usage_source() -> _UsageSource:
    from litellm.integrations.focus.database import FocusLiteLLMDatabase

    return FocusLiteLLMDatabase()


def _config_file_logger() -> TernaryLogger | None:
    from litellm.integrations.ternary.ternary_logger import SettingsTernaryLogger, TernaryLogger

    registered: Final = litellm.logging_callback_manager.get_custom_loggers_for_type(callback_type=TernaryLogger)
    return next(
        (cb for cb in registered if isinstance(cb, TernaryLogger) and not isinstance(cb, SettingsTernaryLogger)),
        None,
    )


def is_ternary_setup_in_config() -> bool:
    return "ternary" in litellm.callbacks or _config_file_logger() is not None


def _require_settings_owned() -> None:
    if is_ternary_setup_in_config():
        raise _http_error(409, _CONFIG_FILE_OWNS)


def _logger_from_settings(settings: TernarySettings) -> SettingsTernaryLogger:
    from litellm.integrations.ternary.ternary_logger import SettingsTernaryLogger

    return SettingsTernaryLogger(
        api_key=settings.api_key,
        connection_id=settings.connection_id,
        base_url=settings.ternary_base_url,
        frequency=settings.frequency,
        interval_seconds=settings.interval_seconds,
    )


async def ternary_exporter(
    store: Annotated[TernarySettingsStore, Depends(ternary_settings_store)],
) -> _Exporter | None:
    config_logger: Final = _config_file_logger()
    if config_logger is not None:
        return config_logger
    settings: Final = await store.get()
    return _logger_from_settings(settings) if settings is not None else None


def _destination_error(settings: TernarySettings) -> str | None:
    from litellm.integrations.focus.destinations.ternary_destination import FocusTernaryDestination

    try:
        FocusTernaryDestination(
            prefix="ternary_exports",
            config={
                "api_key": settings.api_key,
                "connection_id": settings.connection_id,
                "base_url": settings.ternary_base_url,
            },
        )
    except ValueError as e:
        return str(e)
    return None


async def _save_validated(store: TernarySettingsStore, settings: TernarySettings) -> None:
    error: Final = _destination_error(settings)
    if error is not None:
        raise _http_error(400, error)
    await store.save(settings)


def _require_proxy_admin(user_api_key_dict: UserAPIKeyAuth) -> None:
    if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise _http_error(403, CommonProxyErrors.not_allowed_access.value)


@router.get("/ternary/settings", tags=["Ternary"], response_model=TernarySettingsView)
async def get_ternary_settings(
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
    store: Annotated[TernarySettingsStore, Depends(ternary_settings_store)],
) -> TernarySettingsView:
    """View the Ternary export settings with the API key masked. Proxy Admin or Admin Viewer only."""
    if not user_api_key_has_admin_view(user_api_key_dict):
        raise _http_error(403, CommonProxyErrors.not_allowed_access.value)

    if is_ternary_setup_in_config():
        return TernarySettingsView(status="config_file")
    settings: Final = await store.get()
    if settings is None:
        return TernarySettingsView()

    return TernarySettingsView(
        api_key_masked=_sensitive_masker.mask_dict({"api_key": settings.api_key}).get("api_key"),
        connection_id=settings.connection_id,
        ternary_base_url=settings.ternary_base_url,
        frequency=settings.frequency,
        interval_seconds=settings.interval_seconds,
        status="configured",
    )


@router.post("/ternary/init", tags=["Ternary"], response_model=TernaryMessageResponse)
async def init_ternary_settings(
    request: TernarySettings,
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
    store: Annotated[TernarySettingsStore, Depends(ternary_settings_store)],
) -> TernaryMessageResponse:
    """Store the Ternary export settings, with the API key encrypted. Proxy Admin only."""
    _require_proxy_admin(user_api_key_dict)
    _require_settings_owned()
    await _save_validated(store, request)
    verbose_proxy_logger.info("Ternary settings initialized")
    return TernaryMessageResponse(message=f"Ternary settings saved. {_RESTART_NOTE}", status="success")


@router.put("/ternary/settings", tags=["Ternary"], response_model=TernaryMessageResponse)
async def update_ternary_settings(
    request: TernarySettingsUpdate,
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
    store: Annotated[TernarySettingsStore, Depends(ternary_settings_store)],
) -> TernaryMessageResponse:
    """Update some of the Ternary export settings. Omitted fields keep their stored value. Proxy Admin only."""
    _require_proxy_admin(user_api_key_dict)
    _require_settings_owned()
    changes: Final = request.model_dump(exclude_unset=True, exclude_none=True)
    if not changes:
        raise _http_error(400, "At least one field must be provided for update")

    current: Final = await store.get()
    if current is None:
        raise _http_error(404, _NOT_CONFIGURED)
    if request.ternary_base_url not in (None, current.ternary_base_url) and request.api_key is None:
        raise _http_error(400, "Enter the API key again when changing ternary_base_url")

    try:
        updated: Final = TernarySettings.model_validate({**current.model_dump(), **changes})
    except ValidationError as e:
        raise _http_error(400, str(e)) from e

    await _save_validated(store, updated)
    verbose_proxy_logger.info("Ternary settings updated")
    return TernaryMessageResponse(message=f"Ternary settings updated. {_RESTART_NOTE}", status="success")


@router.delete("/ternary/delete", tags=["Ternary"], response_model=TernaryMessageResponse)
async def delete_ternary_settings(
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
    store: Annotated[TernarySettingsStore, Depends(ternary_settings_store)],
) -> TernaryMessageResponse:
    """Delete the stored Ternary export settings. Proxy Admin only."""
    _require_proxy_admin(user_api_key_dict)
    _require_settings_owned()
    try:
        deleted: Final = await store.delete()
    except Exception as e:
        verbose_proxy_logger.error("Failed to delete Ternary settings: %s", e)
        raise _http_error(500, f"Failed to delete Ternary settings: {e}") from e
    if not deleted:
        raise _http_error(404, "Ternary settings not found")

    from litellm.integrations.ternary.ternary_logger import SettingsTernaryLogger

    litellm.logging_callback_manager.remove_callbacks_by_type(  # pyright: ignore[reportUnknownMemberType]  # untyped shared callback-manager method
        litellm.callbacks, SettingsTernaryLogger
    )
    verbose_proxy_logger.info("Ternary settings deleted")
    return TernaryMessageResponse(message=f"Ternary settings deleted. {_RESTART_NOTE}", status="success")


def _json_scalar(value: object) -> JsonValue:
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, date | datetime):
        return value.isoformat()
    return str(value)


def _column(frame: _Columns, name: str) -> Sequence[object]:
    return frame.get_column(name).to_list() if name in frame.columns else ()


def _numbers(frame: _Columns, name: str) -> tuple[float, ...]:
    return tuple(float(value) for value in _column(frame, name) if isinstance(value, int | float | Decimal))


def _distinct(frame: _Columns, name: str) -> frozenset[object]:
    return frozenset(value for value in _column(frame, name) if value is not None)


def _json_rows(frame: _Columns) -> Sequence[Mapping[str, JsonValue]]:
    columns: Final = tuple(frame.columns)
    values: Final = tuple(tuple(frame.get_column(name).to_list()[:_DRY_RUN_SAMPLE_ROWS]) for name in columns)
    rows: Final = tuple(dict(zip(columns, row)) for row in zip(*values))
    return _rows_adapter.validate_json(json.dumps(rows, default=_json_scalar))


@router.post("/ternary/dry-run", tags=["Ternary"], response_model=TernaryDryRunResponse)
async def ternary_dry_run(
    request: TernaryDryRunRequest,
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
    usage_source: Annotated[_UsageSource, Depends(ternary_usage_source)],
) -> TernaryDryRunResponse:
    """Preview the most recent usage as the FOCUS rows Ternary would receive. Sends nothing. Proxy Admin only."""
    _require_proxy_admin(user_api_key_dict)

    from litellm.integrations.ternary.export_engine import focus_rows_for, usage_days

    data: Final = await usage_source.get_usage_data(limit=request.limit)
    focus_rows: Final = focus_rows_for(data)

    return TernaryDryRunResponse(
        message="Preview only. Nothing was sent to Ternary",
        status="success",
        usage_data=_json_rows(data),
        normalized_data=_json_rows(focus_rows),
        summary=TernaryDryRunSummary(
            total_records=focus_rows.height,
            estimated_spend=sum(_numbers(data, "spend")),
            total_tokens=int(sum(_numbers(data, "prompt_tokens")) + sum(_numbers(data, "completion_tokens"))),
            unique_teams=len(_distinct(data, "team_id")),
            unique_models=len(_distinct(data, "model")),
            days=usage_days(data),
        ),
    )


@router.post("/ternary/export", tags=["Ternary"], response_model=TernaryExportResponse)
async def ternary_export(
    request: TernaryExportRequest,
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
    exporter: Annotated[_Exporter | None, Depends(ternary_exporter)],
) -> TernaryExportResponse:
    """Export usage to Ternary now, one upload per UTC day.

    With no bounds this sends the same window as the next scheduled export. Proxy Admin only.
    """
    _require_proxy_admin(user_api_key_dict)
    if exporter is None:
        raise _http_error(404, _NOT_CONFIGURED)

    scheduled: Final = exporter.scheduled_window(datetime.now(timezone.utc))
    start: Final = request.start_time_utc or scheduled.start_time
    end: Final = request.end_time_utc or scheduled.end_time

    try:
        days: Final = await exporter.export_days(start_time_utc=start, end_time_utc=end, limit=request.limit)
    except (RuntimeError, ValueError) as e:
        verbose_proxy_logger.error("Ternary export failed: %s", e)
        raise _http_error(502, str(e)) from e

    verbose_proxy_logger.info("Ternary export for %s..%s uploaded %d day(s)", start, end, len(days))
    return TernaryExportResponse(
        message=f"Ternary export uploaded {len(days)} day(s)" if days else "No usage to upload in this window",
        status="success",
        start_time_utc=start,
        end_time_utc=end,
        days_uploaded=tuple(days),
    )


async def register_ternary_logger_from_db(store: TernarySettingsStore) -> None:
    """Register a TernaryLogger from the stored settings, unless config already configures one."""
    if is_ternary_setup_in_config():
        return
    try:
        settings: Final = await store.get()
        if settings is None:
            return
        litellm.logging_callback_manager.add_litellm_callback(_logger_from_settings(settings))
    except Exception as e:  # noqa: BLE001  # a bad stored row must not stop proxy startup
        verbose_proxy_logger.warning("Failed to register TernaryLogger from DB settings: %s", e)
