from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Final
from unittest.mock import AsyncMock, MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from litellm.models.user_banner import UserBanner
from litellm.proxy.deps.database import get_database
from litellm.proxy.deps.repositories import UserBannerRepositoryDep
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
