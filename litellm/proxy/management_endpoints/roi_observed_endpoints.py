from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from typing import Annotated, Final

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import SecretStr, TypeAdapter, ValidationError

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_utils.encrypt_decrypt_utils import encrypt_value_helper
from litellm.proxy.roi_calculator.analytics import normalize_email
from litellm.proxy.roi_calculator.github import SourceError
from litellm.proxy.roi_calculator.oauth import (
    OAuthConfig,
    Provider,
    begin_authorization,
    connected_settings,
    consume_state,
    exchange_code,
    oauth_config,
    save_grant,
)
from litellm.proxy.roi_calculator.observed_sync import ObservedSyncManager, Progress
from litellm.proxy.roi_calculator.observed_workspace import (
    collect_workspace,
    scoped_data,
    source_details,
    summarize_workspace,
)
from litellm.proxy.roi_calculator.settings import (
    StoredConnection,
    active_connection,
    connection_id,
    enable_observed_reporting,
    get_roi_config_repository,
    load_settings,
    load_stored_settings,
    read_admin,
    save_connection_identities,
    save_settings,
    select_connection,
    stored_connections,
    write_admin,
)
from litellm.proxy.roi_calculator.source import create_source
from litellm.proxy.roi_calculator.sync import read_gateway_user_emails, spend_prisma_client
from litellm.proxy.roi_calculator.sync_store import SyncStore
from litellm.repositories.config_repository import ConfigRepository
from litellm.types.roi_calculator import (
    ROIRepositoriesResponse,
    ROIRepository,
    ROISettings,
    ROISyncStatus,
    normalize_source_login,
)
from litellm.types.roi_observed import (
    ObservedAccount,
    ObservedApp,
    ObservedApps,
    ObservedAuthorization,
    ObservedConnectionIdentities,
    ObservedData,
    ObservedIdentities,
    ObservedIdentityUpdate,
    ObservedReportResponse,
    ObservedSettings,
    ObservedSettingsUpdate,
)

router: Final = APIRouter(prefix="/roi-calculator/observed", tags=["roi calculator"])
_MANAGER: Final = ObservedSyncManager()
_REPORT_KEY: Final = "roi_observed_report"


def get_observed_manager() -> ObservedSyncManager:
    return _MANAGER


def get_observed_transport() -> httpx.AsyncBaseTransport | None:
    return None


def public_settings(settings: ROISettings) -> ObservedSettings:
    return ObservedSettings(
        id=connection_id(settings.source_provider, settings.source_api_url),
        source_provider=settings.source_provider,
        api_url=settings.source_api_url,
        repos=settings.repos,
        has_token=bool(
            (
                settings.gitlab_token if settings.source_provider == "gitlab" else settings.github_token
            ).get_secret_value()
        ),
        update_interval_minutes=settings.update_interval_minutes,
        ready=bool(settings.repos),
        connection_type=settings.connection_type,
    )


async def workspace_settings(repository: ConfigRepository) -> ObservedSettings:
    stored: Final = await load_stored_settings(repository)
    entries: Final = tuple(
        [
            public_settings(await load_settings(repository, select_connection(stored, entry)))
            for entry in stored_connections(stored)
        ]
    )
    current: Final = public_settings(await load_settings(repository, stored))
    return current.model_copy(update={"connections": entries, "ready": any(entry.ready for entry in entries)})


async def _data(repository: ConfigRepository) -> ObservedData | None:
    saved: Final = await repository.get_param(_REPORT_KEY)
    if saved is None or saved.param_value is None:
        return None
    try:
        data: Final = ObservedData.model_validate(saved.param_value)
    except ValidationError:
        raise HTTPException(500, "The saved report is invalid. Sync again to rebuild it.") from None
    if data.connections:
        return data
    settings: Final = ROISettings.model_validate(
        {
            "source_provider": data.source_provider,
            "repos": data.repos,
            ("gitlab_api_url" if data.source_provider == "gitlab" else "github_api_url"): data.source_api_url,
        }
    )
    return scoped_data(data, source_details(settings))


@router.get("/settings", response_model=ObservedSettings)
async def get_observed_settings(
    _user: Annotated[UserAPIKeyAuth, Depends(read_admin)],
    repository: Annotated[ConfigRepository, Depends(get_roi_config_repository)],
) -> ObservedSettings:
    return await workspace_settings(repository)


@router.put("/settings", response_model=ObservedSettings)
async def save_observed_settings(
    patch: ObservedSettingsUpdate,
    _user: Annotated[UserAPIKeyAuth, Depends(write_admin)],
    repository: Annotated[ConfigRepository, Depends(get_roi_config_repository)],
    transport: Annotated[httpx.AsyncBaseTransport | None, Depends(get_observed_transport)],
) -> ObservedSettings:
    if (status := await SyncStore(repository.prisma_client, "roi_observed").status()) and status.running:
        raise HTTPException(409, "Cancel the running sync before changing the connection.")
    original: Final = await load_stored_settings(repository)
    target_id: Final = connection_id(patch.source_provider, patch.api_url)
    selected_id: Final = patch.connection_id or target_id
    selected: Final = next((entry for entry in stored_connections(original) if entry.id == selected_id), None)
    if patch.connection_id and selected is None:
        raise HTTPException(404, "This connection no longer exists. Reload Connections.")
    if (
        patch.connection_id
        and selected_id != target_id
        and any(entry.id == target_id for entry in stored_connections(original))
    ):
        raise HTTPException(409, "This provider and host are already connected. Edit that connection instead.")
    if patch.token is None and selected:
        await connected_settings(repository, transport, selected_id)
    refreshed: Final = await load_stored_settings(repository)
    saved_connection: Final = next((entry for entry in stored_connections(refreshed) if entry.id == selected_id), None)
    stored: Final = select_connection(refreshed, saved_connection) if saved_connection else refreshed
    current: Final = await load_settings(repository, stored)
    changed: Final = target_id != connection_id(current.source_provider, current.source_api_url)
    existing_token: Final = current.gitlab_token if patch.source_provider == "gitlab" else current.github_token
    token: Final = patch.token if patch.token is not None else "" if changed else existing_token.get_secret_value()
    updates: Final[Mapping[str, object]] = {
        "report_mode": "observed",
        "source_provider": patch.source_provider,
        "repos": patch.repos,
        "update_interval_minutes": patch.update_interval_minutes
        if patch.update_interval_minutes is not None
        else current.update_interval_minutes,
        "identity_map": {} if changed else current.identity_map,
        "ignored_logins": () if changed else current.ignored_logins,
        "connection_type": "token" if patch.token is not None or changed else current.connection_type,
        "oauth_refresh_token": SecretStr("") if patch.token is not None or changed else current.oauth_refresh_token,
        "oauth_expires_at": None if patch.token is not None or changed else current.oauth_expires_at,
        ("gitlab_api_url" if patch.source_provider == "gitlab" else "github_api_url"): patch.api_url,
        ("gitlab_token" if patch.source_provider == "gitlab" else "github_token"): SecretStr(token),
    }
    try:
        settings: Final = ROISettings.model_validate({**current.model_dump(), **updates})
    except ValidationError as exc:
        raise HTTPException(422, exc.errors(include_context=False, include_input=False)) from None
    source: Final = create_source(settings, transport)
    try:
        if settings.repos:
            await source.test_repositories(settings.repos)
        elif token:
            await source.repositories(page=1)
    except SourceError as exc:
        raise HTTPException(502, str(exc)) from None
    finally:
        await source.close()
    encrypted: Final = TypeAdapter(str).validate_python(encrypt_value_helper(token)) if token else ""
    await save_settings(
        repository,
        settings,
        encrypted if settings.source_provider == "github" else stored.github_token,
        stored.estimator_key,
        encrypted if settings.source_provider == "gitlab" else stored.gitlab_token,
        revision=stored.revision,
        replace_connection_id=patch.connection_id,
    )
    return await workspace_settings(repository)


@router.get("/report", response_model=ObservedReportResponse)
async def get_observed_report(
    _user: Annotated[UserAPIKeyAuth, Depends(read_admin)],
    repository: Annotated[ConfigRepository, Depends(get_roi_config_repository)],
) -> ObservedReportResponse:
    stored: Final = await load_stored_settings(repository)
    data: Final = await _data(repository)
    return ObservedReportResponse(report=summarize_workspace(data, stored_connections(stored)) if data else None)


@router.get("/identities", response_model=ObservedIdentities)
async def get_observed_identities(
    _user: Annotated[UserAPIKeyAuth, Depends(read_admin)],
    repository: Annotated[ConfigRepository, Depends(get_roi_config_repository)],
) -> ObservedIdentities:
    stored: Final = await load_stored_settings(repository)
    data: Final = await _data(repository)
    report: Final = summarize_workspace(data, stored_connections(stored)) if data else None
    return ObservedIdentities(
        gateway_emails=tuple(sorted(await read_gateway_user_emails(spend_prisma_client(repository.prisma_client)))),
        identity_map=stored.identity_map,
        unmatched_logins=report.unmatched_logins if report else (),
        connections=tuple(
            ObservedConnectionIdentities(
                id=entry.id,
                source_provider=entry.source_provider,
                api_url=entry.api_url,
                repos=entry.repos,
                identity_map=entry.identity_map,
                unmatched_logins=tuple(
                    login.split(":", 1)[-1] for login in report.unmatched_logins if login.startswith(entry.id + ":")
                )
                if report
                else (),
            )
            for entry in stored_connections(stored)
        ),
    )


@router.put("/identities", response_model=ObservedReportResponse)
async def save_observed_identities(
    patch: ObservedIdentityUpdate,
    _user: Annotated[UserAPIKeyAuth, Depends(write_admin)],
    repository: Annotated[ConfigRepository, Depends(get_roi_config_repository)],
) -> ObservedReportResponse:
    email: Final = normalize_email(patch.email)
    emails: Final = await read_gateway_user_emails(spend_prisma_client(repository.prisma_client))
    if email not in emails:
        raise HTTPException(422, "Choose an existing internal user email.")
    stored: Final = await load_stored_settings(repository)
    connections: Final = stored_connections(stored)
    if not connections:
        raise HTTPException(409, "Connect a repository before linking accounts.")
    accounts: Final = (
        patch.accounts
        if patch.accounts is not None
        else tuple(ObservedAccount(connection_id=active_connection(stored).id, login=login) for login in patch.logins)
    )
    if any(account.connection_id not in {entry.id for entry in connections} for account in accounts):
        raise HTTPException(422, "Choose an existing connection.")
    data: Final = await _data(repository)
    report: Final = summarize_workspace(data, connections) if data else None
    existing: Final = next((person.accounts for person in report.people if person.email == email), ()) if report else ()

    def update(entry: StoredConnection) -> StoredConnection:
        if patch.accounts is None and entry.id != active_connection(stored).id:
            return entry
        try:
            logins: Final = tuple(
                normalize_source_login(account.login, entry.source_provider)
                for account in accounts
                if account.connection_id == entry.id
            )
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None
        if any(login in entry.identity_map and entry.identity_map[login] != email for login in logins):
            raise HTTPException(409, "An account is already linked to another email. Unlink it first.")
        old: Final = frozenset(
            (
                *(login for login, address in entry.identity_map.items() if address == email),
                *(account.login for account in existing if account.connection_id == entry.id),
            )
        )
        return entry.model_copy(
            update={
                "identity_map": {
                    **{login: address for login, address in entry.identity_map.items() if address != email},
                    **dict.fromkeys(logins, email),
                },
                "ignored_logins": tuple(sorted((frozenset(entry.ignored_logins) | old) - frozenset(logins))),
            }
        )

    updated: Final = tuple(update(entry) for entry in connections)
    await save_connection_identities(repository, stored, updated)
    return ObservedReportResponse(report=summarize_workspace(data, updated) if data else None)


@router.get("/sync", response_model=ROISyncStatus)
async def get_observed_sync(
    _user: Annotated[UserAPIKeyAuth, Depends(read_admin)],
    repository: Annotated[ConfigRepository, Depends(get_roi_config_repository)],
    manager: Annotated[ObservedSyncManager, Depends(get_observed_manager)],
) -> ROISyncStatus:
    return await SyncStore(repository.prisma_client, "roi_observed").status() or manager.status


async def _report_days(repository: ConfigRepository, days: int | None) -> int:
    if days is not None:
        return days
    saved: Final = await repository.get_param(_REPORT_KEY)
    if saved is None:
        return 28
    try:
        data: Final = ObservedData.model_validate(saved.param_value)
    except ValidationError:
        return 28
    return (data.current.window.end - data.current.window.start).days + 1


async def _start_sync(
    repository: ConfigRepository,
    manager: ObservedSyncManager,
    transport: httpx.AsyncBaseTransport | None,
    scheduled_interval: float = 0,
    days: int | None = None,
) -> bool:
    from litellm.proxy.management_endpoints.roi_calculator_endpoints import branch_spend_reader

    stored: Final = await load_stored_settings(repository)
    reporting_days: Final = await _report_days(repository, days)
    entries: Final = tuple(entry for entry in stored_connections(stored) if entry.repos)
    if not entries:
        raise HTTPException(409, "Select at least one repository.")
    settings: Final = tuple([await connected_settings(repository, transport, entry.id) for entry in entries])
    await enable_observed_reporting(repository)

    async def build(progress: Progress) -> ObservedData:
        from litellm.proxy.management_endpoints.roi_calculator_endpoints import gateway_user_reader, spend_reader

        data: Final = await collect_workspace(
            tuple((entry, branch_spend_reader(repository, entry)) for entry in settings),
            spend_reader(repository),
            gateway_user_reader(repository),
            datetime.now(timezone.utc),
            progress,
            transport,
            days=reporting_days,
        )
        current: Final = tuple(
            entry for entry in stored_connections(await load_stored_settings(repository)) if entry.repos
        )
        if tuple((entry.id, entry.repos) for entry in current) != tuple((entry.id, entry.repos) for entry in entries):
            raise SourceError("The connection changed during sync. Sync again with the current repositories.")
        return data

    return await manager.start(build, SyncStore(repository.prisma_client, "roi_observed"), scheduled_interval)


@router.post("/sync", response_model=ROISyncStatus, status_code=202)
async def start_observed_sync(
    _user: Annotated[UserAPIKeyAuth, Depends(write_admin)],
    repository: Annotated[ConfigRepository, Depends(get_roi_config_repository)],
    manager: Annotated[ObservedSyncManager, Depends(get_observed_manager)],
    transport: Annotated[httpx.AsyncBaseTransport | None, Depends(get_observed_transport)],
    days: Annotated[int | None, Query(ge=1, le=366)] = None,
) -> ROISyncStatus:
    if not await _start_sync(repository, manager, transport, days=days):
        raise HTTPException(409, "A sync is already running.")
    return manager.status


@router.delete("/sync", response_model=ROISyncStatus)
async def cancel_observed_sync(
    _user: Annotated[UserAPIKeyAuth, Depends(write_admin)],
    repository: Annotated[ConfigRepository, Depends(get_roi_config_repository)],
    manager: Annotated[ObservedSyncManager, Depends(get_observed_manager)],
) -> ROISyncStatus:
    store: Final = SyncStore(repository.prisma_client, "roi_observed")
    await store.cancel()
    await manager.cancel()
    return await store.status() or manager.status


async def run_observed_schedule() -> None:
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        return
    repository: Final = ConfigRepository(prisma_client, use_writer=True)
    settings: Final = await load_settings(repository)
    if (
        not any(entry.repos for entry in stored_connections(await load_stored_settings(repository)))
        or not settings.update_interval_minutes
    ):
        return
    status: Final = await SyncStore(prisma_client, "roi_observed").status()
    if status and status.running:
        return
    anchor: Final = status.finished_at if status else None
    if anchor and datetime.fromisoformat(anchor) + timedelta(minutes=settings.update_interval_minutes) > datetime.now(
        timezone.utc
    ):
        return
    await _start_sync(repository, _MANAGER, None, settings.update_interval_minutes)


@router.get("/repositories", response_model=ROIRepositoriesResponse)
async def observed_repositories(
    _user: Annotated[UserAPIKeyAuth, Depends(read_admin)],
    repository: Annotated[ConfigRepository, Depends(get_roi_config_repository)],
    transport: Annotated[httpx.AsyncBaseTransport | None, Depends(get_observed_transport)],
    connection: Annotated[str | None, Query(max_length=100)] = None,
    query: Annotated[str, Query(max_length=200)] = "",
    page: Annotated[int, Query(ge=1, le=1000)] = 1,
) -> ROIRepositoriesResponse:
    source: Final = create_source(await connected_settings(repository, transport, connection), transport)
    try:
        repositories, more = await source.repositories(query, page)
    except SourceError as exc:
        raise HTTPException(502, str(exc)) from None
    finally:
        await source.close()
    return ROIRepositoriesResponse(
        repositories=tuple(
            ROIRepository(name=name, visibility=visibility, archived=archived)
            for name, visibility, archived in repositories
        ),
        page=page,
        has_more=more,
    )


@router.get("/apps", response_model=ObservedApps)
async def observed_apps(_user: Annotated[UserAPIKeyAuth, Depends(read_admin)]) -> ObservedApps:
    def details(provider: Provider) -> ObservedApp:
        config: Final = oauth_config(provider)
        return ObservedApp(
            configured=config is not None,
            can_install=bool(config and config.installation_url),
            api_url=config.api_url if config else None,
            callback_url=config.redirect_uri if config else None,
        )

    return ObservedApps(github=details("github"), gitlab=details("gitlab"))


@router.post("/oauth/{provider}/start", response_model=ObservedAuthorization)
async def start_observed_authorization(
    provider: Provider,
    _user: Annotated[UserAPIKeyAuth, Depends(write_admin)],
    repository: Annotated[ConfigRepository, Depends(get_roi_config_repository)],
    install: bool = False,
) -> JSONResponse:
    config: Final = oauth_config(provider)
    if config is None:
        raise HTTPException(409, "Configure the provider app client ID, client secret, and PROXY_BASE_URL first.")
    if install and not config.installation_url:
        raise HTTPException(409, "Configure the GitHub app slug to manage repository access.")
    url, nonce = await begin_authorization(repository, config, install=install)
    response: Final = JSONResponse(ObservedAuthorization(url=url).model_dump(mode="json"))
    response.set_cookie(
        "litellm_roi_oauth",
        nonce,
        httponly=True,
        secure=config.proxy_url.startswith("https://"),
        samesite="lax",
        max_age=600,
        path=config.cookie_path,
    )
    response.headers["Cache-Control"] = "no-store"
    return response


def get_oauth_repository() -> ConfigRepository:
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(503, "The database is unavailable. Try connecting again later.")
    return ConfigRepository(prisma_client, use_writer=True)


def _authorization_redirect(config: OAuthConfig, query: str) -> RedirectResponse:
    response: Final = RedirectResponse(config.proxy_url + "/ui/roi-calculator/?" + query, status_code=303)
    response.delete_cookie("litellm_roi_oauth", path=config.cookie_path)
    response.headers["Cache-Control"] = "no-store"
    response.headers["Referrer-Policy"] = "no-referrer"
    return response


@router.get("/oauth/{provider}/callback", include_in_schema=False)
async def observed_authorization_callback(
    provider: Provider,
    request: Request,
    repository: Annotated[ConfigRepository, Depends(get_oauth_repository)],
    transport: Annotated[httpx.AsyncBaseTransport | None, Depends(get_observed_transport)],
    state: str = "",
    code: str = "",
    error: str = "",
) -> RedirectResponse:
    config: Final = oauth_config(provider)
    if config is None:
        raise HTTPException(409, "The provider app is not configured.")
    try:
        return await _complete_authorization(config, request, repository, transport, state, code, error)
    except HTTPException:
        return _authorization_redirect(config, "connection_failed=1")


async def _complete_authorization(
    config: OAuthConfig,
    request: Request,
    repository: ConfigRepository,
    transport: httpx.AsyncBaseTransport | None,
    state: str,
    code: str,
    error: str,
) -> RedirectResponse:
    verified: Final = await consume_state(repository, state, request.cookies.get("litellm_roi_oauth", ""), config)
    if error or not code:
        return _authorization_redirect(config, "connection_cancelled=1")
    if (status := await SyncStore(repository.prisma_client, "roi_observed").status()) and status.running:
        raise HTTPException(409, "Cancel the running sync before changing the connection.")
    grant: Final = await exchange_code(config, verified, code, transport)
    validation_settings: Final = ROISettings.model_validate(
        {
            "source_provider": config.provider,
            "connection_type": "app",
            ("github_api_url" if config.provider == "github" else "gitlab_api_url"): config.api_url,
            ("github_token" if config.provider == "github" else "gitlab_token"): grant.access_token,
        }
    )
    source: Final = create_source(validation_settings, transport)
    try:
        await source.repositories(page=1)
    except SourceError as exc:
        raise HTTPException(502, str(exc)) from None
    finally:
        await source.close()
    await save_grant(repository, config, grant, revision=verified.settings_revision)
    return _authorization_redirect(config, "connected=" + config.provider)


@router.get("/oauth/github/installed", include_in_schema=False)
async def observed_installation_callback(
    request: Request,
    repository: Annotated[ConfigRepository, Depends(get_oauth_repository)],
    state: str = "",
) -> RedirectResponse:
    config: Final = oauth_config("github")
    if config is None or config.installation_url is None:
        raise HTTPException(409, "The GitHub app is not configured.")
    try:
        verified: Final = await consume_state(
            repository, state, request.cookies.get("litellm_roi_oauth", ""), config, flow="install"
        )
        if (await load_stored_settings(repository)).revision != verified.settings_revision:
            raise HTTPException(409, "The connection changed during installation. Start again from Connections.")
        url, nonce = await begin_authorization(repository, config)
    except HTTPException:
        return _authorization_redirect(config, "connection_failed=1")
    response: Final = RedirectResponse(url, status_code=303)
    response.set_cookie(
        "litellm_roi_oauth",
        nonce,
        httponly=True,
        secure=config.proxy_url.startswith("https://"),
        samesite="lax",
        max_age=600,
        path=config.cookie_path,
    )
    response.headers["Cache-Control"] = "no-store"
    response.headers["Referrer-Policy"] = "no-referrer"
    return response
