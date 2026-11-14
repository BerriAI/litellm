from collections.abc import Mapping
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Final
from unittest.mock import AsyncMock, MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from litellm.models.deployment_warning import AdminIgnore
from litellm.models.gateway_activity import GatewayActivity
from litellm.models.user_banner import UserBanner
from litellm.proxy.deps.database import get_database
from litellm.proxy.deps.repositories import (
    DeploymentWarningIgnoreRepositoryDep,
    GatewayActivityRepositoryDep,
    UserBannerRepositoryDep,
)
from litellm.repositories import database as database_runtime
from tests._support.native_database import native_database

NATIVE: Final = native_database()


def _app(database: database_runtime.Database | None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield {"database": database}

    app: Final = FastAPI(lifespan=lifespan)

    @app.get("/banner")
    async def banner(banners: UserBannerRepositoryDep) -> UserBanner:
        return UserBanner() if banners is None else await banners.get_banner()

    @app.get("/ignores")
    async def ignores(ignores: DeploymentWarningIgnoreRepositoryDep) -> Mapping[str, AdminIgnore] | None:
        return None if ignores is None else await ignores.get_ignores()

    @app.get("/activity")
    async def activity(activity: GatewayActivityRepositoryDep) -> tuple[GatewayActivity, ...] | None:
        return None if activity is None else await activity.get_activity("2026-01-01", "2026-01-31")

    return app


def test_lifespan_database_is_shared_until_shutdown():
    prisma: Final = MagicMock()
    prisma.db.litellm_uisettings.find_unique = AsyncMock(return_value=None)
    app: Final = _app(database_runtime.Database(prisma=prisma, native=NATIVE))
    with TestClient(app) as client:
        assert client.get("/banner").json() == UserBanner().model_dump()
        prisma.db.litellm_uisettings.find_unique.return_value = SimpleNamespace(
            ui_settings={"enabled": True, "message": "published", "revision": "revision"}
        )
        assert (
            client.get("/banner").json()
            == UserBanner(enabled=True, message="published", revision="revision").model_dump()
        )


def test_both_repositories_use_the_lifespan_database():
    prisma: Final = MagicMock()
    prisma.db.litellm_uisettings.find_unique = AsyncMock(
        return_value=SimpleNamespace(ui_settings={"message": "stored banner"})
    )
    stored_ignores: Final = {"detailed_debug": {"ignored_by": "admin", "ignored_at": "2026-10-14T12:00:00Z"}}
    prisma.writer_db.litellm_uisettings.find_unique = AsyncMock(
        return_value=SimpleNamespace(ui_settings=stored_ignores)
    )
    with TestClient(_app(database_runtime.Database(prisma=prisma, native=NATIVE))) as client:
        assert client.get("/banner").json() == UserBanner(message="stored banner").model_dump()
        assert client.get("/ignores").json() == {
            "detailed_debug": {**stored_ignores["detailed_debug"], "source": "admin", "ignored_until": None}
        }


def test_no_database_state_preserves_optional_repository():
    with TestClient(_app(None)) as client:
        response: Final = client.get("/banner")
        assert response.status_code == 200
        assert response.json() == UserBanner().model_dump()


def test_dependency_override_changes_next_request_without_changing_lifespan_owner(monkeypatch):
    prisma: Final = MagicMock()
    prisma.db.litellm_uisettings.find_unique = AsyncMock(
        return_value=SimpleNamespace(ui_settings={"message": "stored"})
    )
    app: Final = _app(database_runtime.Database(prisma=prisma, native=NATIVE))
    with TestClient(app) as client:
        assert client.get("/banner").json() == UserBanner(message="stored").model_dump()
        monkeypatch.setitem(app.dependency_overrides, get_database, lambda: None)
        assert client.get("/banner").json() == UserBanner().model_dump()


def test_gateway_activity_reads_the_lifespan_reader_not_the_writer() -> None:
    prisma: Final = MagicMock()
    row: Final = {"kind": "status", "status_code": 429, "failed_requests": 3}
    prisma.db.query_raw = AsyncMock(return_value=[row])
    prisma.writer_db.query_raw = AsyncMock(return_value=[{**row, "failed_requests": 99}])
    with TestClient(_app(database_runtime.Database(prisma=prisma, native=NATIVE))) as client:
        response: Final = client.get("/activity")
        assert response.status_code == 200
        assert response.json() == [row]


def test_gateway_activity_without_database_is_unavailable() -> None:
    with TestClient(_app(None)) as client:
        response: Final = client.get("/activity")
        assert response.status_code == 200
        assert response.json() is None
