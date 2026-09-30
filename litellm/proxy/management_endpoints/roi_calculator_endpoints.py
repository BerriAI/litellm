from collections.abc import Mapping, Sequence
from datetime import date, datetime, timedelta, timezone
from enum import Enum
from types import MappingProxyType
from typing import Annotated, Final, Literal

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field, SecretStr, TypeAdapter, ValidationError

from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.proxy._types import CommonProxyErrors, LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_utils.encrypt_decrypt_utils import decrypt_value_helper, encrypt_value_helper
from litellm.proxy.roi_calculator.analytics import normalize_email, summarize
from litellm.proxy.roi_calculator.estimator import CompletionCaller, EstimatorModel
from litellm.proxy.roi_calculator.github import GitHub, SourceError
from litellm.proxy.roi_calculator.sync import SpendReader, SyncManager, read_spend, spend_prisma_client
from litellm.proxy.roi_calculator.sync_store import SyncStore
from litellm.repositories.config_repository import ConfigRepository
from litellm.types.roi_calculator import (
    DEFAULT_PROMPT,
    ROICompletionRequest,
    ROIIdentityMapResponse,
    ROIIdentityMapUpdate,
    ROIReport,
    ROIReportResponse,
    ROIRepositoriesResponse,
    ROIRepository,
    ROISettings,
    ROISettingsResponse,
    ROISettingsUpdate,
    ROISpendRecord,
    ROISummaryResponse,
    ROISyncStatus,
)

router: Final = APIRouter()
_SETTINGS_KEY: Final = "roi_calculator_settings"
_REPORT_KEY: Final = "roi_calculator_report"
_SYNC_MANAGER: Final = SyncManager()
_ROI_TAGS: Final[list[str | Enum]] = ["roi calculator"]  # mutable-ok: FastAPI requires list-valued route tags


class _StoredSettings(BaseModel):
    model_config = ConfigDict(extra="ignore")

    github_api_url: str = "https://api.github.com"
    github_token: str = ""
    estimator_key: str = ""
    repos: tuple[str, ...] = ()
    estimator_model: str = ""
    estimator_prompt: str = DEFAULT_PROMPT
    backfill_days: int = Field(default=7, ge=1, le=3650)
    update_interval_minutes: float = Field(default=1440, ge=0, le=43200)
    identity_map: Mapping[str, str] = Field(default_factory=lambda: MappingProxyType({}))


class _RouterEstimatorParams(BaseModel):
    model_config = ConfigDict(extra="ignore", from_attributes=True)

    model: str | None = None
    base_model: str | None = None
    custom_llm_provider: str | None = None


class _RouterEstimatorModelInfo(BaseModel):
    model_config = ConfigDict(extra="ignore", from_attributes=True)

    base_model: str | None = None


class _RouterEstimatorDeployment(BaseModel):
    model_config = ConfigDict(extra="ignore", from_attributes=True)

    litellm_params: _RouterEstimatorParams
    model_info: _RouterEstimatorModelInfo | None = None


async def _read_admin(
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
) -> UserAPIKeyAuth:
    if user_api_key_dict.user_role not in (
        LitellmUserRoles.PROXY_ADMIN,
        LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY,
    ):
        raise HTTPException(status_code=403, detail="Only proxy admins can access the ROI Calculator.")
    return user_api_key_dict


async def _write_admin(
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
) -> UserAPIKeyAuth:
    if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise HTTPException(status_code=403, detail="Only proxy admins can change ROI Calculator settings.")
    return user_api_key_dict


async def get_roi_config_repository(
    _user: Annotated[UserAPIKeyAuth, Depends(_read_admin)],
) -> ConfigRepository:
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail=CommonProxyErrors.db_not_connected_error.value,
        )
    return ConfigRepository(prisma_client)


def get_roi_sync_manager() -> SyncManager:
    return _SYNC_MANAGER


def get_github_transport() -> httpx.AsyncBaseTransport | None:
    return None


_ROUTER_ESTIMATOR_DEPLOYMENTS: Final = TypeAdapter(tuple[_RouterEstimatorDeployment, ...])
_MODEL_NAMES: Final = TypeAdapter(tuple[str, ...])


def _estimator_models_from_deployments(deployments: Sequence[object]) -> tuple[EstimatorModel, ...]:
    parsed_deployments: Final = _ROUTER_ESTIMATOR_DEPLOYMENTS.validate_python(deployments)
    return tuple(
        estimator_model
        for deployment in parsed_deployments
        if (estimator_model := _estimator_model(deployment)) is not None
    )


def _estimator_model(deployment: _RouterEstimatorDeployment) -> EstimatorModel | None:
    parameters: Final = deployment.litellm_params
    model: Final = (
        (deployment.model_info.base_model if deployment.model_info is not None else None)
        or parameters.base_model
        or parameters.model
    )
    if model is None:
        return None
    return model, parameters.custom_llm_provider


def _router_estimator_models(model_group: str) -> tuple[EstimatorModel, ...]:
    from litellm.proxy.proxy_server import llm_router

    if llm_router is None:
        return ()
    deployments: Final = llm_router.get_model_list(model_name=model_group) or ()
    return _estimator_models_from_deployments(deployments)


def _router_models() -> tuple[str, ...]:
    from litellm.proxy.proxy_server import llm_router

    if llm_router is None:
        return ()
    return tuple(sorted(frozenset(_MODEL_NAMES.validate_python(llm_router.get_model_names()))))


async def _load_stored_settings(repository: ConfigRepository) -> _StoredSettings:
    parameter: Final = await repository.get_param(_SETTINGS_KEY)
    if parameter is None:
        return _StoredSettings()
    try:
        return _StoredSettings.model_validate(parameter.param_value)
    except ValidationError:
        raise HTTPException(status_code=500, detail="Stored ROI Calculator settings are invalid.") from None


async def _load_settings(repository: ConfigRepository) -> ROISettings:
    stored: Final = await _load_stored_settings(repository)
    token: Final = decrypt_value_helper(stored.github_token, _SETTINGS_KEY) if stored.github_token else ""
    try:
        return ROISettings(
            github_api_url=stored.github_api_url,
            github_token=SecretStr(token or ""),
            estimator_key=SecretStr(decrypt_value_helper(stored.estimator_key, _SETTINGS_KEY) or "")
            if stored.estimator_key
            else SecretStr(""),
            update_interval_minutes=stored.update_interval_minutes,
            repos=stored.repos,
            estimator_model=stored.estimator_model,
            estimator_prompt=stored.estimator_prompt,
            backfill_days=stored.backfill_days,
            identity_map=stored.identity_map,
        )
    except ValidationError:
        raise HTTPException(status_code=500, detail="Stored ROI Calculator settings are invalid.") from None


async def _save_settings(
    repository: ConfigRepository,
    settings: ROISettings,
    encrypted_token: str,
    encrypted_estimator_key: str,
) -> None:
    stored: Final = _StoredSettings(
        github_api_url=settings.github_api_url,
        github_token=encrypted_token,
        estimator_key=encrypted_estimator_key,
        update_interval_minutes=settings.update_interval_minutes,
        repos=settings.repos,
        estimator_model=settings.estimator_model,
        estimator_prompt=settings.estimator_prompt,
        backfill_days=settings.backfill_days,
        identity_map=settings.identity_map,
    )
    await repository.set_param(_SETTINGS_KEY, stored.model_dump(mode="json"))


async def _load_report(repository: ConfigRepository) -> ROIReport | None:
    parameter: Final = await repository.get_param(_REPORT_KEY)
    if parameter is None:
        return None
    try:
        return TypeAdapter(ROIReport).validate_python(parameter.param_value)
    except ValidationError:
        raise HTTPException(status_code=500, detail="Stored ROI Calculator report is invalid.") from None


def _public_settings(settings: ROISettings) -> ROISettingsResponse:
    models: Final = _router_models()
    return ROISettingsResponse(
        github_api_url=settings.github_api_url,
        repos=settings.repos,
        estimator_model=settings.estimator_model,
        estimator_prompt=settings.estimator_prompt,
        backfill_days=settings.backfill_days,
        identity_map=settings.identity_map,
        has_github_token=bool(settings.github_token.get_secret_value()),
        has_estimator_key=bool(settings.estimator_key.get_secret_value()),
        update_interval_minutes=settings.update_interval_minutes,
        default_prompt=DEFAULT_PROMPT,
        available_models=models,
        ready=bool(settings.repos and settings.estimator_model and settings.estimator_model in models),
    )


def _gateway_key(settings: ROISettings) -> str:
    from litellm.proxy.proxy_server import master_key

    credential: Final = settings.estimator_key.get_secret_value() or master_key
    if not credential:
        raise HTTPException(status_code=409, detail="Add an estimator API key in Advanced settings.")
    return credential


def _completion_caller(settings: ROISettings) -> CompletionCaller:
    from litellm.proxy.proxy_server import app

    credential: Final = _gateway_key(settings)
    transport: Final = httpx.ASGITransport(app=app)

    async def complete(request: ROICompletionRequest) -> object:
        client: Final = AsyncHTTPHandler(transport=transport, timeout=180, follow_redirects=False)
        try:
            response: Final = await client.client.post(
                "http://litellm.internal/v1/chat/completions",
                headers=MappingProxyType({"authorization": f"Bearer {credential}", "content-type": "application/json"}),
                content=request.model_dump_json(exclude_none=True),
            )
            response.raise_for_status()
            return TypeAdapter(object).validate_python(response.json())
        finally:
            await client.close()

    return complete


class _GatewayModel(BaseModel):
    id: str


class _GatewayModels(BaseModel):
    data: tuple[_GatewayModel, ...]


async def _test_estimator_access(settings: ROISettings) -> None:
    from litellm.proxy.proxy_server import app

    credential: Final = _gateway_key(settings)
    client: Final = AsyncHTTPHandler(transport=httpx.ASGITransport(app=app), timeout=30, follow_redirects=False)
    try:
        response: Final = await client.client.get(
            "http://litellm.internal/v1/models",
            headers=MappingProxyType({"authorization": f"Bearer {credential}"}),
        )
        response.raise_for_status()
        models: Final = _GatewayModels.model_validate(response.json())
        if not any(model.id == settings.estimator_model for model in models.data):
            raise HTTPException(status_code=409, detail="The estimator key cannot access the selected model.")
    except (httpx.HTTPError, ValidationError):
        raise HTTPException(status_code=409, detail="The estimator key could not connect to the gateway.") from None
    finally:
        await client.close()


def _spend_reader(repository: ConfigRepository) -> SpendReader:
    async def get_spend(start: date, end: date) -> tuple[ROISpendRecord, ...]:
        prisma_client: Final = spend_prisma_client(repository.prisma_client)
        return await read_spend(prisma_client, start, end)

    return get_spend


@router.get(
    "/roi-calculator/settings",
    response_model=ROISettingsResponse,
    tags=_ROI_TAGS,
)
async def get_roi_calculator_settings(
    _user: Annotated[UserAPIKeyAuth, Depends(_read_admin)],
    repository: Annotated[ConfigRepository, Depends(get_roi_config_repository)],
) -> ROISettingsResponse:
    return _public_settings(await _load_settings(repository))


@router.put(
    "/roi-calculator/settings",
    response_model=ROISettingsResponse,
    tags=_ROI_TAGS,
)
async def update_roi_calculator_settings(
    patch: ROISettingsUpdate,
    _user: Annotated[UserAPIKeyAuth, Depends(_write_admin)],
    repository: Annotated[ConfigRepository, Depends(get_roi_config_repository)],
) -> ROISettingsResponse:
    stored: Final = await _load_stored_settings(repository)
    current: Final = await _load_settings(repository)
    if "github_api_url" in patch.model_fields_set and patch.github_api_url is None:
        raise HTTPException(status_code=422, detail="GitHub API URL cannot be null.")
    github_api_url: Final = patch.github_api_url if patch.github_api_url is not None else current.github_api_url
    github_url_changed: Final = github_api_url.rstrip("/") != current.github_api_url.rstrip("/")
    token_was_supplied: Final = "github_token" in patch.model_fields_set
    plaintext_token, encrypted_token = (
        (
            patch.github_token or "",
            TypeAdapter(str).validate_python(encrypt_value_helper(patch.github_token or ""))
            if patch.github_token
            else "",
        )
        if token_was_supplied
        else ("", "")
        if github_url_changed
        else (current.github_token.get_secret_value(), stored.github_token)
    )
    estimator_key: Final = (
        patch.estimator_key or ""
        if "estimator_key" in patch.model_fields_set
        else current.estimator_key.get_secret_value()
    )
    encrypted_estimator_key: Final = (
        TypeAdapter(str).validate_python(encrypt_value_helper(estimator_key)) if estimator_key else ""
    )
    try:
        settings: Final = ROISettings(
            github_api_url=github_api_url,
            github_token=SecretStr(plaintext_token),
            estimator_key=SecretStr(estimator_key),
            update_interval_minutes=patch.update_interval_minutes
            if patch.update_interval_minutes is not None
            else current.update_interval_minutes,
            repos=patch.repos if patch.repos is not None else current.repos,
            estimator_model=(patch.estimator_model if patch.estimator_model is not None else current.estimator_model),
            estimator_prompt=(
                patch.estimator_prompt if patch.estimator_prompt is not None else current.estimator_prompt
            ),
            backfill_days=(patch.backfill_days if patch.backfill_days is not None else current.backfill_days),
            identity_map=current.identity_map,
        )
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors(include_context=False)) from None
    await _save_settings(repository, settings, encrypted_token, encrypted_estimator_key)
    return _public_settings(settings)


@router.get(
    "/roi-calculator/repositories",
    response_model=ROIRepositoriesResponse,
    tags=_ROI_TAGS,
)
async def get_roi_calculator_repositories(
    _user: Annotated[UserAPIKeyAuth, Depends(_read_admin)],
    repository: Annotated[ConfigRepository, Depends(get_roi_config_repository)],
    transport: Annotated[httpx.AsyncBaseTransport | None, Depends(get_github_transport)],
    query: Annotated[str, Query(max_length=200)] = "",
    page: Annotated[int, Query(ge=1, le=1000)] = 1,
) -> ROIRepositoriesResponse:
    github: Final = GitHub(await _load_settings(repository), transport)
    try:
        repos, has_more = await github.repositories(query, page)
    except SourceError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from None
    finally:
        await github.close()
    return ROIRepositoriesResponse(
        repositories=tuple(
            ROIRepository(name=name, visibility=visibility, archived=archived) for name, visibility, archived in repos
        ),
        page=page,
        has_more=has_more,
    )


@router.get(
    "/roi-calculator/sync",
    response_model=ROISyncStatus,
    tags=_ROI_TAGS,
)
async def get_roi_calculator_sync_status(
    _user: Annotated[UserAPIKeyAuth, Depends(_read_admin)],
    repository: Annotated[ConfigRepository, Depends(get_roi_config_repository)],
    manager: Annotated[SyncManager, Depends(get_roi_sync_manager)],
) -> ROISyncStatus:
    status: Final = await SyncStore(repository.prisma_client).status() or manager.status
    settings: Final = await _load_settings(repository)
    report: Final = await _load_report(repository)
    next_update: Final = _next_update(settings, status, report)
    return status.model_copy(update={"next_update": next_update.isoformat() if next_update else None})


@router.post(
    "/roi-calculator/sync",
    response_model=ROISyncStatus,
    status_code=202,
    tags=_ROI_TAGS,
)
async def start_roi_calculator_sync(
    _user: Annotated[UserAPIKeyAuth, Depends(_write_admin)],
    repository: Annotated[ConfigRepository, Depends(get_roi_config_repository)],
    manager: Annotated[SyncManager, Depends(get_roi_sync_manager)],
    transport: Annotated[httpx.AsyncBaseTransport | None, Depends(get_github_transport)],
) -> ROISyncStatus:
    settings: Final = await _load_settings(repository)
    public: Final = _public_settings(settings)
    if not public.ready:
        raise HTTPException(status_code=409, detail="Connect GitHub, select repositories, and choose a router model.")
    if not await manager.start(
        settings,
        repository,
        _spend_reader(repository),
        _completion_caller(settings),
        transport,
        _router_estimator_models(settings.estimator_model),
        SyncStore(repository.prisma_client),
    ):
        raise HTTPException(status_code=409, detail="A sync is already running.")
    return manager.status


@router.delete(
    "/roi-calculator/sync",
    response_model=ROISyncStatus,
    tags=_ROI_TAGS,
)
async def cancel_roi_calculator_sync(
    _user: Annotated[UserAPIKeyAuth, Depends(_write_admin)],
    repository: Annotated[ConfigRepository, Depends(get_roi_config_repository)],
    manager: Annotated[SyncManager, Depends(get_roi_sync_manager)],
) -> ROISyncStatus:
    store: Final = SyncStore(repository.prisma_client)
    await store.cancel()
    await manager.cancel()
    return await store.status() or manager.status


@router.get(
    "/roi-calculator/report",
    response_model=ROIReportResponse,
    tags=_ROI_TAGS,
)
async def get_roi_calculator_report(
    _user: Annotated[UserAPIKeyAuth, Depends(_read_admin)],
    repository: Annotated[ConfigRepository, Depends(get_roi_config_repository)],
    mode: Literal["live", "demo"] = "live",
) -> ROIReportResponse:
    if mode == "demo":
        from litellm.proxy.roi_calculator.sample import sample_report

        sample: Final = summarize(sample_report(datetime.now(timezone.utc)), MappingProxyType({}))
        return ROIReportResponse(report=ROISummaryResponse.model_validate(sample))
    report: Final = await _load_report(repository)
    if report is None:
        return ROIReportResponse(report=None)
    settings: Final = await _load_settings(repository)
    summary: Final = summarize(report, settings.identity_map)
    return ROIReportResponse(report=ROISummaryResponse.model_validate(summary))


@router.put(
    "/roi-calculator/identity-map",
    response_model=ROIIdentityMapResponse,
    tags=_ROI_TAGS,
)
async def update_roi_calculator_identity_map(
    update: ROIIdentityMapUpdate,
    _user: Annotated[UserAPIKeyAuth, Depends(_write_admin)],
    repository: Annotated[ConfigRepository, Depends(get_roi_config_repository)],
) -> ROIIdentityMapResponse:
    login: Final = update.github_login.strip().casefold()
    current: Final = await _load_settings(repository)
    current_stored: Final = await _load_stored_settings(repository)
    new_email: Final = normalize_email(update.email)
    if not login or (update.email is not None and not new_email):
        raise HTTPException(status_code=422, detail="Enter a GitHub login and a valid email address.")
    identity_map: Final[Mapping[str, str]] = MappingProxyType(
        {key: value for key, value in current.identity_map.items() if update.email is None and key != login}
        if update.email is None
        else {**current.identity_map, login: new_email}
    )
    settings: Final = ROISettings(
        github_api_url=current.github_api_url,
        github_token=current.github_token,
        estimator_key=current.estimator_key,
        update_interval_minutes=current.update_interval_minutes,
        repos=current.repos,
        estimator_model=current.estimator_model,
        estimator_prompt=current.estimator_prompt,
        backfill_days=current.backfill_days,
        identity_map=identity_map,
    )
    await _save_settings(repository, settings, current_stored.github_token, current_stored.estimator_key)
    report: Final = await _load_report(repository)
    summary: Final = summarize(report, settings.identity_map) if report is not None else None
    return ROIIdentityMapResponse(
        report=ROISummaryResponse.model_validate(summary) if summary is not None else None,
        identity_map=settings.identity_map,
    )


def _next_update(settings: ROISettings, status: ROISyncStatus, report: ROIReport | None) -> datetime | None:
    if (
        not report
        or not settings.repos
        or not settings.estimator_model
        or not settings.update_interval_minutes
        or status.running
    ):
        return None
    anchor: Final = status.finished_at or status.started_at or report["synced_at"]
    return datetime.fromisoformat(anchor.replace("Z", "+00:00")) + timedelta(minutes=settings.update_interval_minutes)


async def run_scheduled_sync() -> None:
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        return
    repository: Final = ConfigRepository(prisma_client)
    settings: Final = await _load_settings(repository)
    if not settings.update_interval_minutes or not _public_settings(settings).ready:
        return
    store: Final = SyncStore(prisma_client)
    status: Final = await store.status() or _SYNC_MANAGER.status
    report: Final = await _load_report(repository)
    next_update: Final = _next_update(settings, status, report)
    if next_update is None or next_update > datetime.now(timezone.utc):
        return
    await _SYNC_MANAGER.start(
        settings,
        repository,
        _spend_reader(repository),
        _completion_caller(settings),
        estimator_models=_router_estimator_models(settings.estimator_model),
        coordinator=store,
        scheduled_interval=settings.update_interval_minutes,
    )


@router.post("/roi-calculator/connections/test", tags=_ROI_TAGS)
async def test_roi_calculator_connections(
    _user: Annotated[UserAPIKeyAuth, Depends(_write_admin)],
    repository: Annotated[ConfigRepository, Depends(get_roi_config_repository)],
    transport: Annotated[httpx.AsyncBaseTransport | None, Depends(get_github_transport)],
) -> ROISettingsResponse:
    settings: Final = await _load_settings(repository)
    public: Final = _public_settings(settings)
    if not public.ready:
        raise HTTPException(status_code=409, detail="Choose repositories and an available estimator model first.")
    await _test_estimator_access(settings)
    github: Final = GitHub(settings, transport)
    try:
        await github.test_repositories(settings.repos)
    except SourceError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from None
    finally:
        await github.close()
    return public


@router.post("/roi-calculator/setup/reset", tags=_ROI_TAGS)
async def reset_roi_calculator_setup(
    _user: Annotated[UserAPIKeyAuth, Depends(_write_admin)],
    repository: Annotated[ConfigRepository, Depends(get_roi_config_repository)],
) -> ROISettingsResponse:
    from uuid import uuid4

    store: Final = SyncStore(repository.prisma_client)
    owner: Final = str(uuid4())
    status: Final = ROISyncStatus(
        running=True,
        phase="spend",
        stage="Restarting setup",
        done=0,
        total=0,
        estimated=0,
        reused=0,
        needs_attention=0,
        error=None,
    )
    if not await store.acquire(owner, status):
        raise HTTPException(status_code=409, detail="Cancel the running analysis before restarting setup.")
    try:
        current: Final = await _load_settings(repository)
        stored: Final = await _load_stored_settings(repository)
        settings: Final = current.model_copy(update={"repos": ()})
        await _save_settings(repository, settings, stored.github_token, stored.estimator_key)
        await repository.prisma_client.db.execute_raw('DELETE FROM "LiteLLM_Config" WHERE param_name = $1', _REPORT_KEY)
        return _public_settings(settings)
    finally:
        await store.finish(owner, status.model_copy(update={"running": False, "phase": "idle", "stage": "Idle"}))
