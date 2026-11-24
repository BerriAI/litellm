import asyncio
from collections.abc import AsyncIterator
from datetime import date, datetime, timezone
from typing import Final
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI, HTTPException
from pydantic import SecretStr

from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.roi_calculator.github import SourceError
from litellm.proxy.roi_calculator.oauth import (
    OAuthConfig,
    TokenGrant,
    begin_authorization,
    connected_settings,
    consume_state,
    exchange_code,
    save_grant,
)
from litellm.proxy.roi_calculator.observed_sync import ObservedSyncManager, Progress
from litellm.proxy.roi_calculator.settings import load_settings, load_stored_settings, save_settings
from litellm.proxy.roi_calculator.sync_store import SyncStore
from litellm.proxy.utils import PrismaClient, ProxyLogging
from litellm.repositories.config_repository import ConfigRepository
from litellm.types.roi_observed import ObservedData, ObservedPeriodData, ObservedWindow
from tests.integration._support.database import scratch_database, write_rows


@pytest_asyncio.fixture(loop_scope="function")
async def repository(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[ConfigRepository]:
    with scratch_database() as url:
        write_rows(
            'CREATE TABLE "LiteLLM_Config" (param_name TEXT PRIMARY KEY, param_value JSONB NOT NULL, '
            "last_run_at TIMESTAMP NOT NULL DEFAULT NOW(), reload_revision BIGINT NOT NULL DEFAULT 0)",
            (),
            database_url=url,
        )
        monkeypatch.setenv("DATABASE_URL", url)
        monkeypatch.delenv("DATABASE_URL_READ_REPLICA", raising=False)
        monkeypatch.setenv("LITELLM_SALT_KEY", "test-only-observed-roi-salt-0123456789")
        client: Final = PrismaClient(url, ProxyLogging(UserApiKeyCache()))
        await client.connect()
        try:
            yield ConfigRepository(client, use_writer=True)
        finally:
            await client.disconnect()


def _config(provider: str = "github") -> OAuthConfig:
    from pydantic import TypeAdapter

    from litellm.proxy.roi_calculator.oauth import Provider

    selected: Final = TypeAdapter(Provider).validate_python(provider)
    return OAuthConfig(
        selected,
        "https://api.github.com" if provider == "github" else "https://gitlab.com/api/v4",
        f"https://{provider}.com",
        "test-client",
        SecretStr("test-client-secret"),
        "https://gateway.example.test",
    )


def _report() -> ObservedData:
    period: Final = ObservedPeriodData(
        window=ObservedWindow(start=date(2026, 9, 1), end=date(2026, 9, 28)), pulls=(), issues=(), spend=()
    )
    return ObservedData(
        source_provider="github",
        source_api_url="https://api.github.com",
        repos=("org/repo",),
        captured_at=datetime.now(timezone.utc),
        gateway_emails=(),
        current=period,
        previous=period,
        last_year=period,
    )


@pytest.mark.asyncio
async def test_settings_compare_and_swap_rejects_stale_writers(repository: ConfigRepository) -> None:
    assert await repository.set_param_if_revision("settings", {"revision": 1, "account": "ari"}, 0)
    assert not await repository.set_param_if_revision("settings", {"revision": 1, "account": "bea"}, 0)
    writes: Final = await asyncio.gather(
        *(repository.set_param_if_revision("settings", {"revision": 2, "account": name}, 1) for name in ("bea", "cam"))
    )
    assert sum(writes) == 1
    saved: Final = await repository.get_param("settings")
    assert saved is not None and saved.param_value in (
        {"revision": 2, "account": "bea"},
        {"revision": 2, "account": "cam"},
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ("github", "gitlab"))
async def test_authorization_uses_pkce_single_use_state_and_encrypted_credentials(
    repository: ConfigRepository, provider: str
) -> None:
    config: Final = _config(provider)
    url, nonce = await begin_authorization(repository, config)
    params: Final = parse_qs(urlsplit(url).query)
    assert params["code_challenge_method"] == ["S256"]
    assert params["redirect_uri"] == [config.redirect_uri]
    state: Final = await consume_state(repository, params["state"][0], nonce, config)
    assert state.verifier.get_secret_value() != "**********"

    def respond(request: httpx.Request) -> httpx.Response:
        body: Final = parse_qs(request.content.decode())
        assert body["code_verifier"] == [state.verifier.get_secret_value()]
        assert body["code"] == ["test-code"]
        assert body["client_secret"] == ["test-client-secret"]
        assert body["redirect_uri"] == [config.redirect_uri]
        return httpx.Response(
            200, json={"access_token": "test-access", "refresh_token": "test-refresh", "expires_in": 3600}
        )

    grant: Final = await exchange_code(config, state, "test-code", httpx.MockTransport(respond))
    await save_grant(repository, config, grant, revision=state.settings_revision)
    stored: Final = await load_stored_settings(repository)
    assert "test-access" not in stored.model_dump_json() and "test-refresh" not in stored.model_dump_json()
    connected: Final = await load_settings(repository)
    assert (
        connected.github_token if provider == "github" else connected.gitlab_token
    ).get_secret_value() == "test-access"
    assert connected.oauth_refresh_token.get_secret_value() == "test-refresh"
    with pytest.raises(HTTPException, match="already used"):
        await consume_state(repository, params["state"][0], nonce, config)


@pytest.mark.asyncio
async def test_authorization_rejects_a_different_browser_and_changed_settings(repository: ConfigRepository) -> None:
    config: Final = _config()
    url, nonce = await begin_authorization(repository, config)
    state: Final = parse_qs(urlsplit(url).query)["state"][0]
    with pytest.raises(HTTPException, match="same browser"):
        await consume_state(repository, state, "another-browser", config)
    new_url, new_nonce = await begin_authorization(repository, config)
    verified: Final = await consume_state(repository, parse_qs(urlsplit(new_url).query)["state"][0], new_nonce, config)
    await save_grant(repository, config, TokenGrant(access_token=SecretStr("first")))
    with pytest.raises(HTTPException, match="changed during authorization"):
        await save_grant(
            repository, config, TokenGrant(access_token=SecretStr("stale")), revision=verified.settings_revision
        )
    assert (await load_settings(repository)).github_token.get_secret_value() == "first"
    assert nonce != new_nonce


@pytest.mark.asyncio
async def test_concurrent_app_refresh_rotates_once_and_preserves_account_links(
    repository: ConfigRepository, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LITELLM_ROI_GITHUB_CLIENT_ID", "test-client")
    monkeypatch.setenv("LITELLM_ROI_GITHUB_CLIENT_SECRET", "test-client-secret")
    monkeypatch.setenv("PROXY_BASE_URL", "https://gateway.example.test")
    await save_grant(
        repository,
        _config(),
        TokenGrant(access_token=SecretStr("old-access"), refresh_token=SecretStr("old-refresh"), expires_in=1),
    )
    stored: Final = await load_stored_settings(repository)
    settings: Final = (await load_settings(repository)).model_copy(update={"identity_map": {"ari": "ari@example.test"}})
    await save_settings(repository, settings, stored.github_token, stored.estimator_key, revision=stored.revision)
    requests: Final[asyncio.Queue[httpx.Request]] = asyncio.Queue()

    def respond(request: httpx.Request) -> httpx.Response:
        requests.put_nowait(request)
        assert parse_qs(request.content.decode())["refresh_token"] == ["old-refresh"]
        return httpx.Response(
            200, json={"access_token": "new-access", "refresh_token": "new-refresh", "expires_in": 3600}
        )

    results: Final = await asyncio.gather(
        *(connected_settings(repository, httpx.MockTransport(respond)) for _ in range(12))
    )
    assert requests.qsize() == 1
    assert all(result.github_token.get_secret_value() == "new-access" for result in results)
    assert all(result.identity_map == {"ari": "ari@example.test"} for result in results)
    assert (await load_settings(repository)).oauth_refresh_token.get_secret_value() == "new-refresh"


@pytest.mark.asyncio
async def test_refresh_cannot_restore_a_connection_replaced_while_the_provider_responds(
    repository: ConfigRepository, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LITELLM_ROI_GITHUB_CLIENT_ID", "test-client")
    monkeypatch.setenv("LITELLM_ROI_GITHUB_CLIENT_SECRET", "test-client-secret")
    monkeypatch.setenv("PROXY_BASE_URL", "https://gateway.example.test")
    await save_grant(
        repository,
        _config(),
        TokenGrant(access_token=SecretStr("old"), refresh_token=SecretStr("refresh"), expires_in=1),
    )
    started: Final = asyncio.Event()
    release: Final = asyncio.Event()

    async def respond(request: httpx.Request) -> httpx.Response:
        started.set()
        await release.wait()
        return httpx.Response(200, json={"access_token": "late-token", "expires_in": 3600})

    pending: Final = asyncio.create_task(connected_settings(repository, httpx.MockTransport(respond)))
    await asyncio.wait_for(started.wait(), 2)
    await save_grant(repository, _config(), TokenGrant(access_token=SecretStr("replacement"), expires_in=3600))
    release.set()
    result: Final = await asyncio.wait_for(pending, 2)
    assert result.github_token.get_secret_value() == "replacement"
    assert (await load_settings(repository)).github_token.get_secret_value() == "replacement"


@pytest.mark.asyncio
async def test_failed_cancelled_and_stale_workers_cannot_replace_the_published_report(
    repository: ConfigRepository,
) -> None:
    store: Final = SyncStore(repository.prisma_client, "roi_observed")
    manager: Final = ObservedSyncManager()
    report: Final = _report()
    await repository.set_param("roi_observed_report", report.model_dump(mode="json"))
    release: Final = asyncio.Event()

    async def build(progress: Progress) -> ObservedData:
        await release.wait()
        raise SourceError("source failure")

    assert await manager.start(build, store)
    assert not await ObservedSyncManager().start(build, store)
    await store.cancel()
    release.set()
    await manager.cancel()
    published: Final = await repository.get_param("roi_observed_report")
    assert published is not None and ObservedData.model_validate(published.param_value) == report

    async def complete(progress: Progress) -> ObservedData:
        return report

    assert await manager.start(complete, store)

    async def finished() -> None:
        for _ in range(200):
            status: Final = await store.status()
            if status is not None and not status.running:
                assert status.phase == "complete", status
                return
            await asyncio.sleep(0.01)
        pytest.fail("Observed report did not publish")

    await asyncio.wait_for(finished(), 5)
    saved: Final = await repository.get_param("roi_observed_report")
    assert saved is not None and ObservedData.model_validate(saved.param_value) == report


@pytest.mark.asyncio
@pytest.mark.parametrize("provider,install_first", (("github", False), ("github", True), ("gitlab", False)))
async def test_app_callback_round_trip_and_admin_authorization(
    repository: ConfigRepository, monkeypatch: pytest.MonkeyPatch, provider: str, install_first: bool
) -> None:
    from fastapi import FastAPI

    from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
    from litellm.proxy.management_endpoints.roi_observed_endpoints import (
        get_oauth_repository,
        get_observed_transport,
        router,
    )
    from litellm.proxy.roi_calculator.settings import get_roi_config_repository

    config: Final = _config(provider)
    monkeypatch.setenv(f"LITELLM_ROI_{provider.upper()}_CLIENT_ID", config.client_id)
    monkeypatch.setenv(f"LITELLM_ROI_{provider.upper()}_CLIENT_SECRET", config.client_secret.get_secret_value())
    monkeypatch.setenv("PROXY_BASE_URL", config.proxy_url)
    monkeypatch.setenv("LITELLM_ROI_GITHUB_APP_SLUG", "example-roi")

    def respond(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            assert str(request.url) == config.token_url
            if parse_qs(request.content.decode()).get("code") == ["rejected-code"]:
                return httpx.Response(400, json={"error": "invalid_grant"})
            return httpx.Response(
                200,
                json={
                    "access_token": "provider-test-access",
                    "refresh_token": "provider-test-refresh",
                    "expires_in": 3600,
                },
            )
        assert request.headers["Authorization"] == "Bearer provider-test-access"
        assert "PRIVATE-TOKEN" not in request.headers
        return httpx.Response(200, json=[])

    app: Final = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_roi_config_repository] = lambda: repository
    app.dependency_overrides[get_oauth_repository] = lambda: repository
    app.dependency_overrides[get_observed_transport] = lambda: httpx.MockTransport(respond)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=config.proxy_url) as client:
        for role, expected in (
            (LitellmUserRoles.INTERNAL_USER, 403),
            (LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY, 403),
            (LitellmUserRoles.PROXY_ADMIN, 200),
        ):
            app.dependency_overrides[user_api_key_auth] = lambda role=role: UserAPIKeyAuth(user_role=role)
            response: Final = await client.post(f"/roi-calculator/observed/oauth/{provider}/start")
            assert response.status_code == expected, response.text
        successful: Final = await client.post(
            f"/roi-calculator/observed/oauth/{provider}/start", params={"install": str(install_first).lower()}
        )
        assert "HttpOnly" in successful.headers["set-cookie"] and "Secure" in successful.headers["set-cookie"]
        initial_state: Final = parse_qs(urlsplit(successful.json()["url"]).query)["state"][0]
        installed: Final = (
            await client.get("/roi-calculator/observed/oauth/github/installed", params={"state": initial_state})
            if install_first
            else None
        )
        if install_first:
            assert urlsplit(successful.json()["url"]).path == "/apps/example-roi/installations/new"
            assert installed is not None
            assert installed.status_code == 303, installed.text
            assert urlsplit(installed.headers["location"]).path == "/login/oauth/authorize"
            assert parse_qs(urlsplit(installed.headers["location"]).query)["state"][0] != initial_state
        state: Final = (
            parse_qs(urlsplit(installed.headers["location"]).query)["state"][0] if installed else initial_state
        )
        callback: Final = await client.get(
            f"/roi-calculator/observed/oauth/{provider}/callback", params={"state": state, "code": "test-code"}
        )
        assert callback.status_code == 303, callback.text
        assert callback.headers["location"] == config.proxy_url + f"/ui/roi-calculator/?connected={provider}"
        saved: Final = await client.get("/roi-calculator/observed/settings")
        assert saved.json()["has_token"] is True and saved.json()["connection_type"] == "app"
        assert "provider-test-access" not in saved.text and "provider-test-refresh" not in saved.text
        replay: Final = await client.get(
            f"/roi-calculator/observed/oauth/{provider}/callback", params={"state": state, "code": "test-code"}
        )
        assert replay.status_code == 303
        assert replay.headers["location"] == config.proxy_url + "/ui/roi-calculator/?connection_failed=1"
        restart: Final = await client.post(f"/roi-calculator/observed/oauth/{provider}/start")
        denied_state: Final = parse_qs(urlsplit(restart.json()["url"]).query)["state"][0]
        denied: Final = await client.get(
            f"/roi-calculator/observed/oauth/{provider}/callback",
            params={"state": denied_state, "error": "access_denied"},
        )
        assert denied.status_code == 303
        assert denied.headers["location"] == config.proxy_url + "/ui/roi-calculator/?connection_cancelled=1"
        unchanged: Final = await client.get("/roi-calculator/observed/settings")
        assert unchanged.json() == saved.json()
        expired_installation: Final = (
            await client.get("/roi-calculator/observed/oauth/github/installed", params={"state": initial_state})
            if install_first
            else None
        )
        if expired_installation is not None:
            assert expired_installation.status_code == 303
            assert expired_installation.headers["location"].endswith("?connection_failed=1")
        retry: Final = await client.post(f"/roi-calculator/observed/oauth/{provider}/start")
        rejected: Final = await client.get(
            f"/roi-calculator/observed/oauth/{provider}/callback",
            params={"state": parse_qs(urlsplit(retry.json()["url"]).query)["state"][0], "code": "rejected-code"},
        )
        assert rejected.status_code == 303
        assert rejected.headers["location"] == config.proxy_url + "/ui/roi-calculator/?connection_failed=1"
        assert "litellm_roi_oauth" not in client.cookies
        assert (await client.get("/roi-calculator/observed/settings")).json() == saved.json()


@pytest.mark.asyncio
async def test_identity_api_combines_accounts_and_removes_automatic_links_without_resync(
    repository: ConfigRepository,
) -> None:
    from fastapi import FastAPI

    from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
    from litellm.proxy.management_endpoints.roi_observed_endpoints import router
    from litellm.proxy.roi_calculator.settings import get_roi_config_repository
    from litellm.types.roi_observed import ObservedPull

    write_rows('CREATE TABLE "LiteLLM_UserTable" (user_id TEXT PRIMARY KEY, user_email TEXT)', ())
    write_rows(
        'INSERT INTO "LiteLLM_UserTable" VALUES (%s,%s),(%s,%s)', ("ari", "ari@example.test", "bea", "bea@example.test")
    )
    settings: Final = (await load_settings(repository)).model_copy(update={"repos": ("org/repo",)})
    await save_settings(repository, settings, "", "")
    base: Final = _report()
    pulls: Final = tuple(
        ObservedPull(
            repo="org/repo",
            number=index,
            title="Change",
            url=f"https://github.com/org/repo/pull/{index}",
            author=login,
            profile_email="ari@example.test" if login == "ari" else "",
            merged_at=base.captured_at,
        )
        for index, login in enumerate(("ari", "old-ari"))
    )
    data: Final = base.model_copy(
        update={
            "gateway_emails": ("ari@example.test", "bea@example.test"),
            "current": base.current.model_copy(update={"pulls": pulls}),
        }
    )
    await repository.set_param("roi_observed_report", data.model_dump(mode="json"))
    app: Final = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_roi_config_repository] = lambda: repository
    app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="https://gateway.example.test"
    ) as client:
        linked: Final = await client.put(
            "/roi-calculator/observed/identities", json={"email": "ari@example.test", "logins": ["ari", "old-ari"]}
        )
        assert linked.status_code == 200, linked.text
        person: Final = linked.json()["report"]["people"][0]
        assert person["logins"] == ["ari", "old-ari"] and person["periods"]["current"]["merged_prs"] == 2
        conflict: Final = await client.put(
            "/roi-calculator/observed/identities", json={"email": "bea@example.test", "logins": ["old-ari"]}
        )
        assert conflict.status_code == 409
        unknown: Final = await client.put(
            "/roi-calculator/observed/identities", json={"email": "missing@example.test", "logins": []}
        )
        assert unknown.status_code == 422
        removed: Final = await client.put(
            "/roi-calculator/observed/identities", json={"email": "ari@example.test", "logins": []}
        )
        assert removed.json()["report"]["people"] == []
        assert [login.split(":", 1)[-1] for login in removed.json()["report"]["unmatched_logins"]] == ["ari", "old-ari"]
        app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(
            user_role=LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY
        )
        readonly: Final = await client.get("/roi-calculator/observed/report")
        assert readonly.status_code == 200
        denied: Final = await client.put(
            "/roi-calculator/observed/identities", json={"email": "ari@example.test", "logins": []}
        )
        assert denied.status_code == 403


@pytest.mark.asyncio
async def test_both_app_connections_keep_repositories_credentials_and_account_links(
    repository: ConfigRepository,
) -> None:
    from litellm.proxy.roi_calculator.settings import connection_id, stored_connections

    await save_grant(
        repository,
        _config(),
        TokenGrant(access_token=SecretStr("github-access"), refresh_token=SecretStr("github-refresh")),
    )
    stored: Final = await load_stored_settings(repository)
    github: Final = (await load_settings(repository)).model_copy(
        update={"repos": ("org/service", "org/docs"), "identity_map": {"ari": "ari@example.test"}}
    )
    await save_settings(repository, github, stored.github_token, stored.estimator_key, revision=stored.revision)
    await save_grant(
        repository,
        _config("gitlab"),
        TokenGrant(access_token=SecretStr("gitlab-access"), refresh_token=SecretStr("gitlab-refresh")),
    )
    gitlab_id: Final = connection_id("gitlab", _config("gitlab").api_url)
    github_id: Final = connection_id("github", _config().api_url)
    first: Final = await load_settings(repository, selected_id=github_id)
    second: Final = await load_settings(repository, selected_id=gitlab_id)
    assert first.repos == ("org/service", "org/docs")
    assert first.identity_map == {"ari": "ari@example.test"}
    assert first.github_token.get_secret_value() == "github-access"
    assert first.oauth_refresh_token.get_secret_value() == "github-refresh"
    assert second.gitlab_token.get_secret_value() == "gitlab-access"
    assert second.oauth_refresh_token.get_secret_value() == "gitlab-refresh"
    assert second.identity_map == {} and second.repos == ()
    await save_grant(repository, _config(), TokenGrant(access_token=SecretStr("github-rotated")), previous=first)
    preserved: Final = await load_settings(repository, selected_id=gitlab_id)
    assert preserved.model_dump(exclude={"github_token", "github_api_url"}) == second.model_dump(
        exclude={"github_token", "github_api_url"}
    )
    assert (await load_settings(repository, selected_id=github_id)).github_token.get_secret_value() == "github-rotated"
    persisted: Final = await load_stored_settings(repository)
    assert len(stored_connections(persisted)) == 2
    assert all(
        token not in persisted.model_dump_json()
        for token in ("github-access", "github-rotated", "gitlab-access", "github-refresh", "gitlab-refresh")
    )


@pytest.mark.asyncio
async def test_cross_provider_account_form_saves_atomically_without_legacy_logins(repository: ConfigRepository) -> None:
    from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
    from litellm.proxy.management_endpoints.roi_observed_endpoints import router
    from litellm.proxy.roi_calculator.settings import get_roi_config_repository, stored_connections, write_admin

    write_rows('CREATE TABLE "LiteLLM_UserTable" (user_id TEXT PRIMARY KEY, user_email TEXT)', ())
    write_rows(
        'INSERT INTO "LiteLLM_UserTable" VALUES (%s, %s), (%s, %s)',
        ("ari", "ari@example.test", "bea", "bea@example.test"),
    )
    await save_grant(repository, _config("github"), TokenGrant(access_token=SecretStr("github-token")))
    await save_grant(repository, _config("gitlab"), TokenGrant(access_token=SecretStr("gitlab-token")))
    connections: Final = stored_connections(await load_stored_settings(repository))
    accounts: Final = tuple({"connection_id": entry.id, "login": "ari"} for entry in connections)
    app: Final = FastAPI()
    app.include_router(router)

    async def config_repository() -> ConfigRepository:
        return repository

    async def admin() -> UserAPIKeyAuth:
        return UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN)

    app.dependency_overrides[get_roi_config_repository] = config_repository
    app.dependency_overrides[write_admin] = admin
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gateway.test") as client:
        saved: Final = await client.put(
            "/roi-calculator/observed/identities", json={"email": "ari@example.test", "accounts": accounts}
        )
        assert saved.status_code == 200, saved.text
        after: Final = stored_connections(await load_stored_settings(repository))
        assert all(entry.identity_map == {"ari": "ari@example.test"} for entry in after)
        conflicting: Final = ({"connection_id": connections[0].id, "login": "bea"}, accounts[1])
        rejected: Final = await client.put(
            "/roi-calculator/observed/identities", json={"email": "bea@example.test", "accounts": conflicting}
        )
        assert rejected.status_code == 409, rejected.text
        assert stored_connections(await load_stored_settings(repository)) == after


@pytest.mark.asyncio
async def test_http_sync_combines_providers_retains_period_and_recovers_invalid_reports(
    repository: ConfigRepository,
) -> None:
    from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
    from litellm.proxy.management_endpoints.roi_observed_endpoints import (
        get_observed_manager,
        get_observed_transport,
        router,
    )
    from litellm.proxy.roi_calculator.settings import get_roi_config_repository
    from litellm.types.roi_observed import ObservedReportResponse, ObservedSettings

    write_rows('CREATE TABLE "LiteLLM_UserTable" (user_id TEXT PRIMARY KEY, user_email TEXT)', ())
    write_rows(
        'CREATE TABLE "LiteLLM_DailyUserSpend" (user_id TEXT, date TEXT, spend DOUBLE PRECISION, api_requests INTEGER)',
        (),
    )
    write_rows(
        'CREATE TABLE "LiteLLM_SpendLogs" (spend DOUBLE PRECISION, request_tags JSONB, metadata JSONB, "startTime" TIMESTAMP)',
        (),
    )

    def respond(request: httpx.Request) -> httpx.Response:
        if request.headers.get("Authorization") == "Bearer rejected":
            return httpx.Response(401, json={"message": "Bad credentials"})
        if request.url.path == "/graphql":
            return httpx.Response(
                200, json={"data": {"search": {"issueCount": 0, "nodes": [], "pageInfo": {"hasNextPage": False}}}}
            )
        if request.url.path.startswith("/repos/"):
            return httpx.Response(200, json={"full_name": "org/service", "has_issues": True})
        if request.url.path in ("/user/repos", "/api/v4/projects"):
            return httpx.Response(200, json=[])
        if request.url.path == "/api/v4/projects/org/service":
            return httpx.Response(200, json={"id": 1, "path_with_namespace": "org/service"})
        if request.url.path.endswith(("/merge_requests", "/issues")):
            return httpx.Response(200, json=[])
        raise AssertionError(str(request.url))

    manager: Final = ObservedSyncManager()
    app: Final = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_roi_config_repository] = lambda: repository
    app.dependency_overrides[get_observed_manager] = lambda: manager
    app.dependency_overrides[get_observed_transport] = lambda: httpx.MockTransport(respond)
    app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN)

    async def finish() -> None:
        for _ in range(200):
            status: Final = await SyncStore(repository.prisma_client, "roi_observed").status()
            if status and not status.running:
                assert status.phase == "complete", status.error
                return
            await asyncio.sleep(0.01)
        pytest.fail("Sync did not finish")

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gateway.test") as client:
        assert (await client.get("/roi-calculator/observed/report")).json() == {"report": None}
        assert (await client.post("/roi-calculator/observed/sync")).status_code == 409
        for provider, url in (("github", "https://api.github.com"), ("gitlab", "https://gitlab.com/api/v4")):
            result: Final = await client.put(
                "/roi-calculator/observed/settings",
                json={
                    "source_provider": provider,
                    "api_url": url,
                    "token": "test-token",
                    "repos": ["org/service"],
                    "update_interval_minutes": 0,
                },
            )
            assert result.status_code == 200, result.text
        settings: Final = ObservedSettings.model_validate(
            (await client.get("/roi-calculator/observed/settings")).json()
        )
        assert len(settings.connections) == 2 and all(entry.has_token for entry in settings.connections)
        for entry in settings.connections:
            repositories: Final = await client.get(
                "/roi-calculator/observed/repositories", params={"connection": entry.id}
            )
            assert repositories.status_code == 200, repositories.text
        rejected: Final = await client.put(
            "/roi-calculator/observed/settings",
            json={
                "source_provider": "github",
                "api_url": "https://api.github.com",
                "token": "rejected",
                "repos": ["org/service"],
            },
        )
        assert rejected.status_code == 502, rejected.text
        assert (
            ObservedSettings.model_validate((await client.get("/roi-calculator/observed/settings")).json()) == settings
        )
        for params in ({"days": 7}, {}):
            started: Final = await client.post("/roi-calculator/observed/sync", params=params)
            assert started.status_code == 202, started.text
            await asyncio.wait_for(finish(), 5)
            response: Final = await client.get("/roi-calculator/observed/report")
            report: Final = ObservedReportResponse.model_validate(response.json()).report
            assert report is not None and report.source_provider == "mixed"
            assert len(report.connections) == 2 and report.periods.current.merged_prs == 0
            assert all(
                (period.window.end - period.window.start).days == 6
                for period in (report.periods.current, report.periods.previous, report.periods.last_year)
            )
        await repository.set_param("roi_observed_report", {"invalid": True})
        assert (await client.get("/roi-calculator/observed/report")).status_code == 500
        recovered: Final = await client.post("/roi-calculator/observed/sync")
        assert recovered.status_code == 202, recovered.text
        await asyncio.wait_for(finish(), 5)
        rebuilt: Final = ObservedReportResponse.model_validate(
            (await client.get("/roi-calculator/observed/report")).json()
        ).report
        assert (
            rebuilt is not None
            and (rebuilt.periods.current.window.end - rebuilt.periods.current.window.start).days == 27
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ("github", "gitlab"))
async def test_host_whitespace_preserves_app_credentials_and_account_matches(
    repository: ConfigRepository, provider: str
) -> None:
    from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
    from litellm.proxy.management_endpoints.roi_observed_endpoints import get_observed_transport, router
    from litellm.proxy.roi_calculator.settings import connection_id, get_roi_config_repository

    config: Final = _config(provider)
    granted: Final = await save_grant(
        repository,
        config,
        TokenGrant(access_token=SecretStr("test-access"), refresh_token=SecretStr("test-refresh"), expires_in=3600),
    )
    stored: Final = await load_stored_settings(repository)
    before: Final = granted.model_copy(
        update={"repos": ("org/service",), "identity_map": {"ari": "ari@example.test"}, "ignored_logins": ("bea",)}
    )
    await save_settings(
        repository, before, stored.github_token, stored.estimator_key, stored.gitlab_token, revision=stored.revision
    )

    def respond(request: httpx.Request) -> httpx.Response:
        assert request.headers.get("Authorization") == "Bearer test-access"
        if "/repos/" in request.url.path:
            return httpx.Response(200, json={"full_name": "org/service"})
        if request.url.path.endswith("/merge_requests"):
            return httpx.Response(200, json=[])
        return httpx.Response(200, json={"id": 1, "path_with_namespace": "org/service"})

    app: Final = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_roi_config_repository] = lambda: repository
    app.dependency_overrides[get_observed_transport] = lambda: httpx.MockTransport(respond)
    app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gateway.test") as client:
        response: Final = await client.put(
            "/roi-calculator/observed/settings",
            json={
                "connection_id": connection_id(provider, config.api_url),
                "source_provider": provider,
                "api_url": f"  {config.api_url}/  ",
                "repos": ["org/service"],
            },
        )
        assert response.status_code == 200, response.text
    assert (await load_settings(repository)) == before


@pytest.mark.asyncio
async def test_connection_edits_replace_only_the_selected_host_and_keep_the_workspace_schedule(
    repository: ConfigRepository,
) -> None:
    from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
    from litellm.proxy.management_endpoints.roi_observed_endpoints import get_observed_transport, router
    from litellm.proxy.roi_calculator.settings import get_roi_config_repository, stored_connections
    from litellm.types.roi_calculator import ROISettings
    from litellm.types.roi_observed import ObservedSettings

    legacy: Final = ROISettings(repos=("org/service",), estimator_model="legacy-model", update_interval_minutes=60)
    await save_settings(repository, legacy, "", "")

    def respond(request: httpx.Request) -> httpx.Response:
        if "/repos/" in request.url.path:
            return httpx.Response(200, json={"full_name": "org/service"})
        if "/projects/" in request.url.path:
            return httpx.Response(200, json={"id": 1, "path_with_namespace": "org/service"})
        raise AssertionError(str(request.url))

    app: Final = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_roi_config_repository] = lambda: repository
    app.dependency_overrides[get_observed_transport] = lambda: httpx.MockTransport(respond)
    app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gateway.test") as client:
        assert (await client.get("/roi-calculator/observed/settings")).status_code == 200
        assert (await load_settings(repository)) == legacy
        github: Final = await client.put(
            "/roi-calculator/observed/settings",
            json={
                "source_provider": "github",
                "api_url": "https://api.github.com",
                "token": "github-token",
                "repos": ["org/service"],
                "update_interval_minutes": 30,
            },
        )
        assert github.status_code == 200, github.text
        gitlab: Final = await client.put(
            "/roi-calculator/observed/settings",
            json={
                "source_provider": "gitlab",
                "api_url": "https://gitlab.com/api/v4",
                "token": "gitlab-token",
                "repos": ["org/service"],
            },
        )
        assert gitlab.status_code == 200, gitlab.text
        before: Final = stored_connections(await load_stored_settings(repository))
        edited: Final = await client.put(
            "/roi-calculator/observed/settings",
            json={
                "connection_id": github.json()["id"],
                "source_provider": "github",
                "api_url": "https://git.example.test/api/v3",
                "token": "enterprise-token",
                "repos": ["org/service"],
            },
        )
        assert edited.status_code == 200, edited.text
        settings: Final = ObservedSettings.model_validate(edited.json())
        assert len(settings.connections) == 2
        assert {entry.api_url for entry in settings.connections} == {
            "https://git.example.test/api/v3",
            "https://gitlab.com/api/v4",
        }
        assert all(entry.update_interval_minutes == 30 for entry in settings.connections)
        after: Final = stored_connections(await load_stored_settings(repository))
        assert next(entry for entry in after if entry.source_provider == "gitlab") == next(
            entry for entry in before if entry.source_provider == "gitlab"
        )
        assert (await load_settings(repository)).report_mode == "observed"
        stale: Final = await client.put(
            "/roi-calculator/observed/settings",
            json={
                "connection_id": github.json()["id"],
                "source_provider": "github",
                "api_url": "https://api.github.com",
                "repos": [],
            },
        )
        assert stale.status_code == 404, stale.text
        duplicate: Final = await client.put(
            "/roi-calculator/observed/settings",
            json={
                "connection_id": settings.id,
                "source_provider": "gitlab",
                "api_url": "https://gitlab.com/api/v4",
                "repos": [],
            },
        )
        assert duplicate.status_code == 409, duplicate.text
        assert stored_connections(await load_stored_settings(repository)) == after
        manual: Final = await client.put(
            "/roi-calculator/observed/settings",
            json={
                "source_provider": "gitlab",
                "api_url": "https://gitlab.com/api/v4",
                "repos": ["org/service"],
                "update_interval_minutes": 0,
            },
        )
        assert manual.status_code == 200, manual.text
        reselected: Final = await client.put(
            "/roi-calculator/observed/settings",
            json={
                "connection_id": settings.id,
                "source_provider": "github",
                "api_url": "https://git.example.test/api/v3",
                "repos": ["org/service"],
            },
        )
        assert reselected.status_code == 200, reselected.text
        assert all(
            entry.update_interval_minutes == 0
            for entry in ObservedSettings.model_validate(reselected.json()).connections
        )
