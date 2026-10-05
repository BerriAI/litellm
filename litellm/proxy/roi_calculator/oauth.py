import asyncio
import hashlib
import json
import os
import re
import secrets
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Final, Literal, Protocol, TypeAlias, cast
from urllib.parse import parse_qsl, urlencode, urlsplit

import httpx
from fastapi import HTTPException
from oauthlib.oauth2 import WebApplicationClient
from pydantic import BaseModel, ConfigDict, Field, SecretStr, TypeAdapter

from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,  # pyright: ignore[reportUnknownVariableType]  # shared client factory has untyped params
)
from litellm.proxy.common_utils.encrypt_decrypt_utils import decrypt_value_helper, encrypt_value_helper
from litellm.proxy.roi_calculator.settings import (
    active_connection,
    connection_id,
    load_settings,
    load_stored_settings,
    save_settings,
    select_connection,
    stored_connections,
)
from litellm.proxy.roi_calculator.sync_store import SyncStore
from litellm.repositories.config_repository import ConfigRepository
from litellm.types.llms.custom_http import httpxSpecialProvider
from litellm.types.roi_calculator import ROISettings, ROISyncStatus

Provider: TypeAlias = Literal["github", "gitlab"]
_STATE_PREFIX: Final = "roi_oauth_state_"


@dataclass(frozen=True, slots=True)
class OAuthConfig:
    provider: Provider
    api_url: str
    base_url: str
    client_id: str
    client_secret: SecretStr
    proxy_url: str
    app_slug: str = ""

    @property
    def cookie_path(self) -> str:
        return urlsplit(self.proxy_url).path + "/roi-calculator/observed/oauth"

    @property
    def installation_url(self) -> str | None:
        if self.provider != "github" or not self.app_slug:
            return None
        path: Final = "apps" if self.api_url == "https://api.github.com" else "github-apps"
        return f"{self.base_url}/{path}/{self.app_slug}/installations/new"

    @property
    def redirect_uri(self) -> str:
        return f"{self.proxy_url}/roi-calculator/observed/oauth/{self.provider}/callback"

    @property
    def authorize_url(self) -> str:
        return self.base_url + ("/login/oauth/authorize" if self.provider == "github" else "/oauth/authorize")

    @property
    def token_url(self) -> str:
        return self.base_url + ("/login/oauth/access_token" if self.provider == "github" else "/oauth/token")


def oauth_config(provider: Provider) -> OAuthConfig | None:
    prefix: Final = f"LITELLM_ROI_{provider.upper()}_"
    client_id: Final = os.environ.get(prefix + "CLIENT_ID", "")
    client_secret: Final = os.environ.get(prefix + "CLIENT_SECRET", "")
    proxy_url: Final = os.environ.get("PROXY_BASE_URL", "").rstrip("/")
    if not client_id or not client_secret or not proxy_url:
        return None
    try:
        parsed: Final = urlsplit(proxy_url)
    except ValueError:
        return None
    if parsed.username or parsed.password or parsed.query or parsed.fragment or not parsed.hostname:
        return None
    if parsed.scheme != "https" and not (parsed.scheme == "http" and parsed.hostname in ("localhost", "127.0.0.1")):
        return None
    base: Final = os.environ.get(
        prefix + "URL", "https://github.com" if provider == "github" else "https://gitlab.com"
    ).rstrip("/")
    api_url: Final = (
        "https://api.github.com"
        if base == "https://github.com"
        else base + ("/api/v3" if provider == "github" else "/api/v4")
    )
    try:
        validated: Final = ROISettings.model_validate(
            {
                "source_provider": provider,
                ("github_api_url" if provider == "github" else "gitlab_api_url"): api_url,
            }
        )
    except ValueError:
        return None
    app_slug: Final = os.environ.get(prefix + "APP_SLUG", "")
    if app_slug and not re.fullmatch(r"[A-Za-z0-9-]+", app_slug):
        return None
    return OAuthConfig(
        provider, validated.source_api_url, base, client_id, SecretStr(client_secret), proxy_url, app_slug
    )


class OAuthState(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: Provider
    client_id: str
    api_url: str
    browser_nonce: SecretStr
    verifier: SecretStr
    expires_at: datetime
    settings_revision: int
    flow: Literal["authorize", "install"] = "authorize"


class _Envelope(BaseModel):
    payload: str


class _StateRow(BaseModel):
    param_value: _Envelope


class _Database(Protocol):
    async def query_raw(self, query: str, *args: object) -> object: ...
    async def execute_raw(self, query: str, *args: object) -> int: ...


def _state_key(state: str) -> str:
    return _STATE_PREFIX + hashlib.sha256(state.encode()).hexdigest()


async def begin_authorization(
    repository: ConfigRepository, config: OAuthConfig, *, install: bool = False
) -> tuple[str, str]:
    client: Final = WebApplicationClient(config.client_id)
    verifier: Final = TypeAdapter(str).validate_python(client.create_code_verifier(64))
    challenge: Final = TypeAdapter(str).validate_python(client.create_code_challenge(verifier, "S256"))
    state: Final = secrets.token_urlsafe(32)
    nonce: Final = secrets.token_urlsafe(32)
    stored: Final = await load_stored_settings(repository)
    value: Final = OAuthState(
        settings_revision=stored.revision,
        flow="install" if install else "authorize",
        provider=config.provider,
        client_id=config.client_id,
        api_url=config.api_url,
        browser_nonce=SecretStr(nonce),
        verifier=SecretStr(verifier),
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=10),
    )
    encoded: Final = json.dumps({**value.model_dump(mode="json"), "browser_nonce": nonce, "verifier": verifier})
    payload: Final = TypeAdapter(str).validate_python(encrypt_value_helper(encoded))
    await repository.set_param(_state_key(state), _Envelope(payload=payload).model_dump(mode="json"))
    delegate: Final = repository.prisma_client.writer_db
    database: Final = cast(_Database, delegate)  # cast-ok: Prisma delegates database methods dynamically
    await database.execute_raw(
        "DELETE FROM \"LiteLLM_Config\" WHERE starts_with(param_name, $1) AND last_run_at < NOW() - INTERVAL '20 minutes'",
        _STATE_PREFIX,
    )
    url: Final = TypeAdapter(str).validate_python(
        client.prepare_request_uri(  # pyright: ignore[reportUnknownMemberType]  # oauthlib leaves extension kwargs untyped
            config.authorize_url,
            redirect_uri=config.redirect_uri,
            scope="read_api read_user" if config.provider == "gitlab" else None,
            state=state,
            code_challenge=challenge,
            code_challenge_method="S256",
        )
    )
    if install and config.installation_url:
        return config.installation_url + "?" + urlencode({"state": state}), nonce
    return url, nonce


async def consume_state(
    repository: ConfigRepository,
    state: str,
    nonce: str,
    config: OAuthConfig,
    *,
    flow: Literal["authorize", "install"] = "authorize",
) -> OAuthState:
    if not state or not nonce or len(state) > 200:
        raise HTTPException(400, "The connection expired. Start again from the ROI Calculator.")
    delegate: Final = repository.prisma_client.writer_db
    database: Final = cast(_Database, delegate)  # cast-ok: Prisma delegates database methods dynamically
    rows: Final = TypeAdapter(tuple[_StateRow, ...]).validate_python(
        await database.query_raw(
            'DELETE FROM "LiteLLM_Config" WHERE param_name = $1 RETURNING param_value',
            _state_key(state),
        )
    )
    if not rows:
        raise HTTPException(400, "This connection was already used or expired. Start again.")
    plaintext: Final = decrypt_value_helper(rows[0].param_value.payload, _state_key(state))
    if plaintext is None:
        raise HTTPException(400, "Could not verify the connection. Start again.")
    value: Final = OAuthState.model_validate_json(plaintext)
    if (
        value.flow != flow
        or not secrets.compare_digest(value.browser_nonce.get_secret_value(), nonce)
        or value.expires_at < datetime.now(timezone.utc)
        or (value.provider, value.client_id, value.api_url) != (config.provider, config.client_id, config.api_url)
    ):
        raise HTTPException(400, "Could not verify the connection. Start again in the same browser.")
    return value


class TokenGrant(BaseModel):
    access_token: SecretStr
    token_type: str = "bearer"
    refresh_token: SecretStr = SecretStr("")
    expires_in: int | None = Field(default=None, gt=0)


async def _token(config: OAuthConfig, body: str, transport: httpx.AsyncBaseTransport | None) -> TokenGrant:
    client: Final = get_async_httpx_client(
        llm_provider=httpxSpecialProvider.ROICalculator,
        params={"timeout": 30, "follow_redirects": False, "transport": transport},
    ).client
    try:
        response: Final = await client.post(
            config.token_url, data=dict(parse_qsl(body)), headers={"Accept": "application/json"}
        )
    except httpx.RequestError:
        raise HTTPException(502, "Could not reach the provider. Try connecting again.") from None
    finally:
        if transport is not None:
            await client.aclose()
    if response.status_code != 200:
        raise HTTPException(502, "The provider rejected the connection. Try connecting again.")
    try:
        result: Final = TokenGrant.model_validate(response.json())
    except ValueError:
        raise HTTPException(502, "The provider did not return a valid token. Try connecting again.") from None
    if not result.access_token.get_secret_value() or result.token_type.casefold() != "bearer":
        raise HTTPException(502, "The provider returned an unsupported token.")
    return result


async def exchange_code(
    config: OAuthConfig, state: OAuthState, code: str, transport: httpx.AsyncBaseTransport | None = None
) -> TokenGrant:
    client: Final = WebApplicationClient(config.client_id)
    body: Final = TypeAdapter(str).validate_python(
        client.prepare_request_body(  # pyright: ignore[reportUnknownMemberType]  # oauthlib leaves extension kwargs untyped
            code=code,
            redirect_uri=config.redirect_uri,
            code_verifier=state.verifier.get_secret_value(),
            client_secret=config.client_secret.get_secret_value(),
        )
    )
    return await _token(config, body, transport)


async def save_grant(
    repository: ConfigRepository,
    config: OAuthConfig,
    grant: TokenGrant,
    *,
    revision: int | None = None,
    previous: ROISettings | None = None,
    attempt: int = 0,
) -> ROISettings:
    stored: Final = await load_stored_settings(repository)
    selected: Final = next(
        (entry for entry in stored_connections(stored) if entry.id == connection_id(config.provider, config.api_url)),
        None,
    )
    scoped: Final = select_connection(stored, selected) if selected else stored
    current: Final = await load_settings(repository, scoped)
    if revision is not None and stored.revision != revision:
        raise HTTPException(409, "The connection changed during authorization. Start again from Connections.")
    if previous is not None and (
        current.source_provider,
        current.source_api_url,
        current.connection_type,
        current.gitlab_token if current.source_provider == "gitlab" else current.github_token,
        current.oauth_refresh_token,
    ) != (
        previous.source_provider,
        previous.source_api_url,
        previous.connection_type,
        previous.gitlab_token if previous.source_provider == "gitlab" else previous.github_token,
        previous.oauth_refresh_token,
    ):
        return current
    changed: Final = (current.source_provider, current.source_api_url) != (config.provider, config.api_url)
    refresh_token: Final = (
        grant.refresh_token
        if grant.refresh_token.get_secret_value() or previous is None
        else previous.oauth_refresh_token
    )
    fields: Final[Mapping[str, object]] = {
        "report_mode": "observed" if previous is None else current.report_mode,
        "source_provider": config.provider,
        "connection_type": "app",
        "repos": () if changed else current.repos,
        "identity_map": {} if changed else current.identity_map,
        "ignored_logins": () if changed else current.ignored_logins,
        "oauth_refresh_token": refresh_token,
        "oauth_expires_at": datetime.now(timezone.utc) + timedelta(seconds=grant.expires_in)
        if grant.expires_in
        else None,
        ("github_token" if config.provider == "github" else "gitlab_token"): grant.access_token,
        ("github_api_url" if config.provider == "github" else "gitlab_api_url"): config.api_url,
    }
    settings: Final = ROISettings.model_validate({**current.model_dump(), **fields})
    encrypted: Final = TypeAdapter(str).validate_python(encrypt_value_helper(grant.access_token.get_secret_value()))
    try:
        await save_settings(
            repository,
            settings,
            encrypted if config.provider == "github" else scoped.github_token,
            stored.estimator_key,
            encrypted if config.provider == "gitlab" else scoped.gitlab_token,
            revision=stored.revision,
        )
        return settings
    except HTTPException as exc:
        if exc.status_code != 409 or previous is None or attempt == 2:
            raise

    return await save_grant(repository, config, grant, revision=revision, previous=previous, attempt=attempt + 1)


def _expired(settings: ROISettings) -> bool:
    return bool(
        settings.connection_type == "app"
        and settings.oauth_expires_at is not None
        and settings.oauth_expires_at <= datetime.now(timezone.utc) + timedelta(minutes=5)
    )


async def connected_settings(
    repository: ConfigRepository, transport: httpx.AsyncBaseTransport | None = None, selected_id: str | None = None
) -> ROISettings:
    initial: Final = await load_settings(repository, selected_id=selected_id)
    if not _expired(initial):
        return initial
    selected: Final = selected_id or active_connection(await load_stored_settings(repository)).id
    store: Final = SyncStore(repository.prisma_client, "roi_oauth_refresh_" + selected)
    owner: Final = secrets.token_urlsafe(24)
    status: Final = ROISyncStatus(
        running=True,
        phase="spend",
        stage="Refreshing connection",
        done=0,
        total=0,
        estimated=0,
        reused=0,
        needs_attention=0,
        error=None,
    )

    async def wait_for_connection(attempt: int) -> ROISettings:
        if attempt >= 100:
            raise HTTPException(409, "The connection is refreshing. Try again shortly.")
        settings: Final = await load_settings(repository, selected_id=selected)
        if not _expired(settings):
            return settings
        if not await store.acquire(owner, status):
            await asyncio.sleep(0.1)
            return await wait_for_connection(attempt + 1)
        try:
            current: Final = await load_settings(repository, selected_id=selected)
            if not _expired(current):
                return current
            config: Final = oauth_config(current.source_provider)
            if (
                config is None
                or config.api_url != current.source_api_url
                or not current.oauth_refresh_token.get_secret_value()
            ):
                raise HTTPException(409, "The app connection expired. Reconnect from Connections.")
            client: Final = WebApplicationClient(config.client_id)
            body: Final = TypeAdapter(str).validate_python(
                client.prepare_refresh_body(  # pyright: ignore[reportUnknownMemberType]  # oauthlib leaves extension kwargs untyped
                    refresh_token=current.oauth_refresh_token.get_secret_value(),
                    client_id=config.client_id,
                    client_secret=config.client_secret.get_secret_value(),
                )
            )
            grant: Final = await _token(config, body, transport)
            return await save_grant(repository, config, grant, previous=current)
        finally:
            await store.finish(owner, status.model_copy(update={"running": False, "phase": "complete"}))

    return await wait_for_connection(0)
