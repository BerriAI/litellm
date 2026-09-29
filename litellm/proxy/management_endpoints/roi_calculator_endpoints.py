from collections.abc import Mapping, Sequence
from datetime import date
from enum import Enum
from types import MappingProxyType
from typing import Annotated, Final

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field, SecretStr, TypeAdapter, ValidationError

from litellm.proxy._types import CommonProxyErrors, LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_utils.encrypt_decrypt_utils import decrypt_value_helper, encrypt_value_helper
from litellm.proxy.roi_calculator.analytics import normalize_email, summarize
from litellm.proxy.roi_calculator.estimator import CompletionCaller, EstimatorModel
from litellm.proxy.roi_calculator.github import GitHub, SourceError
from litellm.proxy.roi_calculator.sync import SpendReader, SyncManager, read_spend, spend_prisma_client
from litellm.repositories.config_repository import ConfigRepository
from litellm.types.llms.openai import AllMessageValues
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
    repos: tuple[str, ...] = ()
    estimator_model: str = ""
    estimator_prompt: str = DEFAULT_PROMPT
    backfill_days: int = Field(default=7, ge=1, le=3650)
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
_ROUTER_MESSAGES: Final = TypeAdapter(list[AllMessageValues])


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
) -> None:
    stored: Final = _StoredSettings(
        github_api_url=settings.github_api_url,
        github_token=encrypted_token,
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
        default_prompt=DEFAULT_PROMPT,
        available_models=models,
        ready=bool(settings.repos and settings.estimator_model and settings.estimator_model in models),
    )


def _completion_caller() -> CompletionCaller:
    from litellm.proxy.proxy_server import llm_router

    if llm_router is None:
        raise HTTPException(status_code=503, detail="The proxy model router is not ready.")

    async def complete(request: ROICompletionRequest) -> object:
        messages: Final = _ROUTER_MESSAGES.validate_python(request.messages)
        tags: Final[list[str]] = list(  # mutable-ok: the router requires list-valued tags
            request.metadata["tags"]
        )
        metadata: Final[dict[str, object]] = {  # mutable-ok: the router requires dict metadata
            "tags": tags,
            "litellm_roi_estimator": request.metadata["litellm_roi_estimator"],
        }
        if request.reasoning_effort is None:
            return await llm_router.acompletion(
                model=request.model,
                messages=messages,
                temperature=request.temperature,
                response_format=request.response_format,
                max_tokens=request.max_tokens,
                metadata=metadata,
            )
        response_with_reasoning: Final[object] = await llm_router.acompletion(
            model=request.model,
            messages=messages,
            temperature=request.temperature,
            response_format=request.response_format,
            max_tokens=request.max_tokens,
            metadata=metadata,
            reasoning_effort=request.reasoning_effort,
        )
        return response_with_reasoning

    return complete


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
    try:
        settings: Final = ROISettings(
            github_api_url=github_api_url,
            github_token=SecretStr(plaintext_token),
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
    await _save_settings(repository, settings, encrypted_token)
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
    manager: Annotated[SyncManager, Depends(get_roi_sync_manager)],
) -> ROISyncStatus:
    return manager.status


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
    if not manager.start(
        settings,
        repository,
        _spend_reader(repository),
        _completion_caller(),
        transport,
        _router_estimator_models(settings.estimator_model),
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
    manager: Annotated[SyncManager, Depends(get_roi_sync_manager)],
) -> ROISyncStatus:
    await manager.cancel()
    return manager.status


@router.get(
    "/roi-calculator/report",
    response_model=ROIReportResponse,
    tags=_ROI_TAGS,
)
async def get_roi_calculator_report(
    _user: Annotated[UserAPIKeyAuth, Depends(_read_admin)],
    repository: Annotated[ConfigRepository, Depends(get_roi_config_repository)],
) -> ROIReportResponse:
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
        repos=current.repos,
        estimator_model=current.estimator_model,
        estimator_prompt=current.estimator_prompt,
        backfill_days=current.backfill_days,
        identity_map=identity_map,
    )
    await _save_settings(repository, settings, current_stored.github_token)
    report: Final = await _load_report(repository)
    summary: Final = summarize(report, settings.identity_map) if report is not None else None
    return ROIIdentityMapResponse(
        report=ROISummaryResponse.model_validate(summary) if summary is not None else None,
        identity_map=settings.identity_map,
    )
