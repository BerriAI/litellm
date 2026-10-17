import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from types import MappingProxyType
from typing import Annotated, Final, Literal, Protocol, TypeAlias

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import AwareDatetime, ConfigDict, Field, JsonValue, TypeAdapter, ValidationError, model_validator

from litellm._logging import verbose_logger, verbose_proxy_logger
from litellm.constants import LITELLM_PROXY_ADMIN_NAME
from litellm.proxy._types import AUDIT_ACTIONS, LitellmTableNames, LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.health_endpoints._health_endpoints import (
    DISABLE_NO_REDIS_WARNING_ENV_VAR,
    env_credential_login_enabled,
    no_redis_on_multiple_workers,
    no_redis_warning_disabled_by_env,
    read_license_data,
)
from litellm.repositories.deployment_warning_ignore_repository import (
    DEPLOYMENT_WARNING_IGNORES_ROW_ID,
    DeploymentWarningIgnoreRepository,
)
from litellm.types.llms.base import LiteLLMBaseModel

router: Final = APIRouter()

ENV_CREDENTIAL_LOGIN_WARNING_ID: Final = "env_credential_login"
NO_REDIS_WARNING_ID: Final = "no_redis"
DETAILED_DEBUG_WARNING_ID: Final = "detailed_debug"
LICENSE_EXPIRY_WARNING_ID_PREFIX: Final = "license_expiry:"
LICENSE_EXPIRY_WARNING_DAYS: Final = 30
LICENSE_EXPIRY_CRITICAL_DAYS: Final = 7
MAX_SNOOZE_SECONDS: Final = 30 * 24 * 60 * 60

_STATIC_WARNING_IDS: Final = frozenset(
    {ENV_CREDENTIAL_LOGIN_WARNING_ID, NO_REDIS_WARNING_ID, DETAILED_DEBUG_WARNING_ID}
)

WarningSeverity: TypeAlias = Literal["critical", "warning"]


@dataclass(frozen=True, slots=True)
class WarningCondition:
    id: str
    severity: WarningSeverity
    env_var_ignore: str | None = None


class AdminIgnore(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    source: Literal["admin"] = "admin"
    ignored_by: str
    ignored_at: AwareDatetime
    ignored_until: AwareDatetime | None = Field(
        default=None,
        description="When the snooze ends. Null means the warning is ignored until an admin restores it.",
    )


class EnvVarIgnore(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    source: Literal["env_var"] = "env_var"
    env_var: str


class DeploymentWarning(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    severity: WarningSeverity
    ignore: Annotated[AdminIgnore | EnvVarIgnore, Field(discriminator="source")] | None = None


class DeploymentWarningsResponse(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    warnings: tuple[DeploymentWarning, ...]
    ignore_available: bool = Field(description="False when no database is connected, so warnings cannot be ignored.")


class DeploymentWarningIgnoreUpdate(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    ignored: bool = Field(description="True ignores the warning for every admin, false restores it.")
    snooze_seconds: int | None = Field(
        default=None,
        gt=0,
        le=MAX_SNOOZE_SECONDS,
        description="How long to ignore the warning, up to 30 days. Leave it out to ignore it forever.",
    )

    @model_validator(mode="after")
    def _snooze_only_when_ignoring(self) -> "DeploymentWarningIgnoreUpdate":
        if self.snooze_seconds is not None and not self.ignored:
            raise ValueError("snooze_seconds is only valid when ignored is true")
        return self


class DeploymentWarningIgnoreResponse(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    ignore: AdminIgnore | None


class DeploymentWarningIgnoreStore(Protocol):
    async def get_raw_ignores(self) -> object: ...

    async def upsert_ignores(self, payload: str) -> None: ...


class ConfigAuditLogger(Protocol):
    async def __call__(
        self,
        param_name: str,
        action: AUDIT_ACTIONS,
        before_value: JsonValue | None,
        after_value: JsonValue | None,
        user_api_key_dict: UserAPIKeyAuth,
        table_name: LitellmTableNames = ...,
    ) -> None: ...


_IGNORES_ADAPTER: Final[TypeAdapter[Mapping[str, AdminIgnore]]] = TypeAdapter(Mapping[str, AdminIgnore])
_JSON_VALUE_ADAPTER: Final[TypeAdapter[JsonValue]] = TypeAdapter(JsonValue)
_AUDIT_ENTRIES_ADAPTER: Final[TypeAdapter[tuple[DeploymentWarningIgnoreResponse, ...]]] = TypeAdapter(
    tuple[DeploymentWarningIgnoreResponse, ...]
)


def license_expiry_condition(expiration_date: str | None, now: datetime) -> WarningCondition | None:
    if not expiration_date:
        return None
    try:
        expires_on: Final = date.fromisoformat(expiration_date)
    except ValueError:
        return None
    days_left: Final = (expires_on - now.date()).days
    if days_left > LICENSE_EXPIRY_WARNING_DAYS:
        return None
    return WarningCondition(
        id=f"{LICENSE_EXPIRY_WARNING_ID_PREFIX}{expires_on.isoformat()}",
        severity="critical" if days_left <= LICENSE_EXPIRY_CRITICAL_DAYS else "warning",
    )


def deployment_warning_conditions(
    *,
    env_credential_login_enabled: bool,
    license_expiration_date: str | None,
    no_redis_on_multiple_workers: bool,
    no_redis_warning_disabled_by_env: bool,
    detailed_debug: bool,
    now: datetime,
) -> tuple[WarningCondition, ...]:
    no_redis_env_var: Final = DISABLE_NO_REDIS_WARNING_ENV_VAR if no_redis_warning_disabled_by_env else None
    candidates: Final = (
        WarningCondition(id=ENV_CREDENTIAL_LOGIN_WARNING_ID, severity="critical")
        if env_credential_login_enabled
        else None,
        license_expiry_condition(license_expiration_date, now),
        WarningCondition(id=NO_REDIS_WARNING_ID, severity="warning", env_var_ignore=no_redis_env_var)
        if no_redis_on_multiple_workers
        else None,
        WarningCondition(id=DETAILED_DEBUG_WARNING_ID, severity="warning") if detailed_debug else None,
    )
    return tuple(condition for condition in candidates if condition is not None)


def is_known_warning_id(warning_id: str) -> bool:
    if warning_id in _STATIC_WARNING_IDS:
        return True
    if not warning_id.startswith(LICENSE_EXPIRY_WARNING_ID_PREFIX):
        return False
    expiration_date: Final = warning_id.removeprefix(LICENSE_EXPIRY_WARNING_ID_PREFIX)
    try:
        return date.fromisoformat(expiration_date).isoformat() == expiration_date
    except ValueError:
        return False


def _parse_stored_ignores(raw: object) -> Mapping[str, AdminIgnore]:
    if raw is None:
        return MappingProxyType({})
    try:
        return MappingProxyType(_IGNORES_ADAPTER.validate_python(json.loads(raw) if isinstance(raw, str) else raw))
    except (json.JSONDecodeError, ValidationError):
        verbose_proxy_logger.warning("Ignoring unreadable %s row", DEPLOYMENT_WARNING_IGNORES_ROW_ID)
        return MappingProxyType({})


async def _read_unexpired_ignores(store: DeploymentWarningIgnoreStore, now: datetime) -> Mapping[str, AdminIgnore]:
    stored: Final = _parse_stored_ignores(await store.get_raw_ignores())
    return MappingProxyType(
        {
            warning_id: ignore
            for warning_id, ignore in stored.items()
            if ignore.ignored_until is None or ignore.ignored_until > now
        }
    )


def _resolve_ignore(
    condition: WarningCondition, ignores: Mapping[str, AdminIgnore]
) -> AdminIgnore | EnvVarIgnore | None:
    if condition.env_var_ignore is not None:
        return EnvVarIgnore(env_var=condition.env_var_ignore)
    return ignores.get(condition.id)


def _ignores_as_audit_entries(ignores: Mapping[str, AdminIgnore]) -> JsonValue:
    entries: Final = tuple(
        DeploymentWarningIgnoreResponse(id=warning_id, ignore=ignores[warning_id]) for warning_id in sorted(ignores)
    )
    return _JSON_VALUE_ADAPTER.validate_json(_AUDIT_ENTRIES_ADAPTER.dump_json(entries))


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def deployment_warning_ignore_store() -> DeploymentWarningIgnoreStore | None:
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        return None
    return DeploymentWarningIgnoreRepository(prisma_client, use_writer=True)


def config_audit_logger() -> ConfigAuditLogger:
    from litellm.proxy.proxy_server import create_config_audit_log

    return create_config_audit_log


async def _deployment_warning_reader(
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
) -> UserAPIKeyAuth:
    if user_api_key_dict.user_role not in (LitellmUserRoles.PROXY_ADMIN, LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only proxy admins and proxy admin viewers can read deployment warnings.",
        )
    return user_api_key_dict


async def _deployment_warning_admin(
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
) -> UserAPIKeyAuth:
    if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only proxy admins can ignore deployment warnings.",
        )
    return user_api_key_dict


async def current_warning_conditions(now: Annotated[datetime, Depends(utc_now)]) -> tuple[WarningCondition, ...]:
    license_data: Final = read_license_data()
    return deployment_warning_conditions(
        env_credential_login_enabled=env_credential_login_enabled(),
        license_expiration_date=None if license_data is None else license_data.get("expiration_date"),
        no_redis_on_multiple_workers=await no_redis_on_multiple_workers(),
        no_redis_warning_disabled_by_env=no_redis_warning_disabled_by_env(),
        detailed_debug=verbose_logger.isEnabledFor(logging.DEBUG),
        now=now,
    )


@router.get(
    "/health/deployment_warnings",
    tags=["health"],
    response_model=DeploymentWarningsResponse,
)
async def get_deployment_warnings(
    _: Annotated[UserAPIKeyAuth, Depends(_deployment_warning_reader)],
    conditions: Annotated[tuple[WarningCondition, ...], Depends(current_warning_conditions)],
    store: Annotated[DeploymentWarningIgnoreStore | None, Depends(deployment_warning_ignore_store)],
    now: Annotated[datetime, Depends(utc_now)],
) -> DeploymentWarningsResponse:
    """
    List the active deployment warnings with their severity and whether an admin ignored or snoozed them.
    Readable by proxy admins and proxy admin viewers.
    """
    ignores: Final[Mapping[str, AdminIgnore]] = (
        MappingProxyType({}) if store is None else await _read_unexpired_ignores(store, now)
    )
    return DeploymentWarningsResponse(
        warnings=tuple(
            DeploymentWarning(id=condition.id, severity=condition.severity, ignore=_resolve_ignore(condition, ignores))
            for condition in conditions
        ),
        ignore_available=store is not None,
    )


@router.patch(
    "/health/deployment_warnings/{warning_id}",
    tags=["health"],
    response_model=DeploymentWarningIgnoreResponse,
)
async def update_deployment_warning_ignore(
    warning_id: str,
    update: DeploymentWarningIgnoreUpdate,
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(_deployment_warning_admin)],
    store: Annotated[DeploymentWarningIgnoreStore | None, Depends(deployment_warning_ignore_store)],
    audit_logger: Annotated[ConfigAuditLogger, Depends(config_audit_logger)],
    now: Annotated[datetime, Depends(utc_now)],
) -> DeploymentWarningIgnoreResponse:
    """
    Ignore, snooze, or restore one deployment warning for every admin of this proxy.
    Only proxy admins can call this, and it needs a database.
    """
    if not is_known_warning_id(warning_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Unknown deployment warning: {warning_id}")
    if store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Ignoring deployment warnings needs a database. Connect one to enable it.",
        )

    before: Final = await _read_unexpired_ignores(store, now)
    ignore: Final = (
        AdminIgnore(
            ignored_by=user_api_key_dict.user_id or LITELLM_PROXY_ADMIN_NAME,
            ignored_at=now,
            ignored_until=None if update.snooze_seconds is None else now + timedelta(seconds=update.snooze_seconds),
        )
        if update.ignored
        else None
    )
    others: Final = {other_id: other for other_id, other in before.items() if other_id != warning_id}
    after: Final = MappingProxyType(others if ignore is None else {**others, warning_id: ignore})

    await store.upsert_ignores(_IGNORES_ADAPTER.dump_json(dict(after)).decode())
    await audit_logger(
        param_name=DEPLOYMENT_WARNING_IGNORES_ROW_ID,
        action="updated",
        before_value=_ignores_as_audit_entries(before),
        after_value=_ignores_as_audit_entries(after),
        user_api_key_dict=user_api_key_dict,
        table_name=LitellmTableNames.UI_SETTINGS_TABLE_NAME,
    )
    return DeploymentWarningIgnoreResponse(id=warning_id, ignore=ignore)
