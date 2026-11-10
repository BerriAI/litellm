"""Shared acknowledgement state for model migration alerts."""

import hashlib
import json
from datetime import date, datetime, timezone
from typing import Annotated, Final

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import TypeAdapter

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.deps.database import DatabaseDep
from litellm.repositories.config_repository import ConfigRepository
from litellm.repositories.daily_activity_buckets import api_key_bucket, api_key_from_bucket
from litellm.repositories.table_repositories import DailyTagSpendRepository
from litellm.types.proxy.auth.user_roles import LitellmUserRoles
from litellm.types.proxy.model_deprecation import DeprecationAlertState, DeprecationAlertUpdate, DeprecationTagUsage

router: Final = APIRouter()


def alert_state_repository(database: DatabaseDep) -> ConfigRepository:
    if database is None:
        raise HTTPException(status_code=503, detail="Alert state requires a database connection")
    return ConfigRepository(database.prisma, use_writer=True)


def alert_tag_repository(database: DatabaseDep) -> DailyTagSpendRepository:
    if database is None:
        raise HTTPException(status_code=503, detail="Model usage tags require a database connection")
    return DailyTagSpendRepository(database.prisma)


def _state_key(model: str, deprecation_date: date) -> str:
    identity: Final = json.dumps([model, deprecation_date.isoformat()])
    digest: Final = hashlib.sha256(identity.encode()).hexdigest()
    return f"model_deprecation_alert:{digest}"


def _require_admin(user: UserAPIKeyAuth, *, write: bool) -> None:
    if user.user_role == LitellmUserRoles.PROXY_ADMIN:
        return
    if not write and user.user_role == LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY:
        return
    raise HTTPException(status_code=403, detail="Only gateway admins can manage model deprecation alerts")


@router.get("/alerting/model-deprecations/state", tags=["alerting"], response_model=DeprecationAlertState)
async def get_deprecation_alert_state(
    model: Annotated[str, Query(min_length=1, max_length=1024)],
    deprecation_date: date,
    user: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
    repository: Annotated[ConfigRepository, Depends(alert_state_repository)],
) -> DeprecationAlertState:
    _require_admin(user, write=False)
    record: Final = await repository.get_param(_state_key(model, deprecation_date))
    if record is None:
        return DeprecationAlertState(model=model, deprecation_date=deprecation_date)
    return DeprecationAlertState.model_validate(record.param_value)


@router.put("/alerting/model-deprecations/state", tags=["alerting"], response_model=DeprecationAlertState)
async def update_deprecation_alert_state(
    data: DeprecationAlertUpdate,
    user: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
    repository: Annotated[ConfigRepository, Depends(alert_state_repository)],
) -> DeprecationAlertState:
    _require_admin(user, write=True)
    state: Final = DeprecationAlertState(
        **data.model_dump(), updated_at=datetime.now(timezone.utc), updated_by=user.user_id
    )
    await repository.set_param(_state_key(data.model, data.deprecation_date), state.model_dump(mode="json"))
    return state


@router.get("/alerting/model-deprecations/tags", tags=["alerting"], response_model=tuple[DeprecationTagUsage, ...])
async def get_deprecation_usage_tags(
    model: Annotated[str, Query(min_length=1, max_length=1024)],
    api_keys: Annotated[tuple[str, ...], Query(min_length=1, max_length=100)],
    start_date: date,
    end_date: date,
    user: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
    repository: Annotated[DailyTagSpendRepository, Depends(alert_tag_repository)],
) -> tuple[DeprecationTagUsage, ...]:
    """Distinct tags observed with this model for a page of affected reporting key IDs."""
    _require_admin(user, write=False)
    if not 0 <= (end_date - start_date).days < 30:
        raise HTTPException(status_code=422, detail="Select up to 30 consecutive calendar dates")
    rows: Final = await repository.table.group_by(
        by=["api_key", "tag"],
        where={
            "api_key": {
                # bounded-ok: query validation limits api_keys to 100 IDs.
                "in": [api_key_from_bucket(key) for key in api_keys]
            },
            "model": model,
            "date": {"gte": start_date.isoformat(), "lte": end_date.isoformat()},
            "api_requests": {"gt": 0},
            "tag": {"not": None},
        },
    )
    tags: Final = TypeAdapter(tuple[DeprecationTagUsage, ...]).validate_python(rows)
    return tuple(
        sorted(
            (DeprecationTagUsage(api_key=api_key_bucket(usage.api_key), tag=usage.tag) for usage in tags),
            key=lambda usage: (usage.api_key, usage.tag),
        )
    )
