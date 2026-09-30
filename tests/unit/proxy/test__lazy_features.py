import sys
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from types import ModuleType
from typing import Final

import pytest
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient

from litellm.proxy._lazy_features import (
    LazyFeature,
    LazyFeatureMiddleware,
    attach_lazy_features,
    lazy_tag_to_prefix,
    loaded_lazy_modules,
)

FLAG: Final = "LITELLM_DISABLE_LAZY_ROUTES"
WARMUP_PATH: Final = "/lazy/warm/{name}"


def _feature_module(monkeypatch: pytest.MonkeyPatch, name: str, path: str) -> LazyFeature:
    async def served() -> dict[str, str]:
        return {"feature": name}

    router: Final = APIRouter()
    router.add_api_route(path, served, methods=["GET"])
    module: Final = ModuleType(f"tests.unit.proxy.lazy_fixture_{name}")
    module.router = router  # pyright: ignore[reportAttributeAccessIssue]  # fixture module built at test time
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return LazyFeature(name=name, module_path=module.__name__, path_prefixes=(path,))


def _paths(app: FastAPI) -> tuple[str, ...]:
    return tuple(str(getattr(route, "path", "")) for route in app.routes)


def _has_lazy_middleware(app: FastAPI) -> bool:
    return any(middleware.cls is LazyFeatureMiddleware for middleware in app.user_middleware)


@pytest.mark.parametrize("value", ("1", "true", "TRUE", "yes", "on"))
def test_flag_registers_every_feature_at_startup(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv(FLAG, value)
    features: Final = (
        _feature_module(monkeypatch, "alpha", "/alpha/list"),
        _feature_module(monkeypatch, "beta", "/beta/list"),
    )
    app: Final = FastAPI()

    attach_lazy_features(app, features)

    assert WARMUP_PATH not in _paths(app)
    assert not _has_lazy_middleware(app)
    assert loaded_lazy_modules(app) == set()
    with TestClient(app) as client:
        at_startup: Final = _paths(app)
        assert {"/alpha/list", "/beta/list"} <= set(at_startup)
        assert loaded_lazy_modules(app) == {features[0].module_path, features[1].module_path}
        assert client.get("/beta/list").json() == {"feature": "beta"}
        assert client.post("/lazy/warm/alpha").status_code == 404
        assert _paths(app) == at_startup, "first feature request changed the table"


def test_flag_registers_before_the_inner_lifespan_and_after_late_routes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(FLAG, "true")
    features: Final = (_feature_module(monkeypatch, "epsilon", "/epsilon/{name}"),)
    seen_by_inner_lifespan: Final[list[tuple[str, ...]]] = []  # mutable-ok: captured from inside the lifespan

    @asynccontextmanager
    async def inner_lifespan(app_: FastAPI) -> AsyncGenerator[None]:
        seen_by_inner_lifespan.append(_paths(app_))
        yield

    async def late() -> dict[str, str]:
        return {"feature": "late"}

    app: Final = FastAPI(lifespan=inner_lifespan)
    attach_lazy_features(app, features)
    app.add_api_route("/epsilon/list", late, methods=["GET"])

    with TestClient(app) as client:
        assert client.get("/epsilon/list").json() == {"feature": "late"}, "late eager route must win, as in lazy mode"
        assert client.get("/epsilon/x").json() == {"feature": "epsilon"}
    assert seen_by_inner_lifespan == [_paths(app)], "startup hooks inside the proxy lifespan must see the full table"


@pytest.mark.parametrize("value", (None, "", "0", "false", "off"))
def test_without_the_flag_features_still_mount_on_first_request(
    monkeypatch: pytest.MonkeyPatch, value: str | None
) -> None:
    if value is None:
        monkeypatch.delenv(FLAG, raising=False)
    else:
        monkeypatch.setenv(FLAG, value)
    features: Final = (_feature_module(monkeypatch, "gamma", "/gamma/list"),)
    app: Final = FastAPI()

    attach_lazy_features(app, features)

    assert "/gamma/list" not in _paths(app)
    assert WARMUP_PATH in _paths(app)
    assert _has_lazy_middleware(app)
    with TestClient(app) as client:
        assert client.get("/gamma/list").json() == {"feature": "gamma"}
    assert "/gamma/list" in _paths(app)


def test_flag_keeps_registering_after_one_feature_fails_to_import(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(FLAG, "true")
    broken: Final = LazyFeature(
        name="broken", module_path="tests.unit.proxy.lazy_fixture_does_not_exist", path_prefixes=("/broken",)
    )
    healthy: Final = _feature_module(monkeypatch, "delta", "/delta/list")
    app: Final = FastAPI()

    attach_lazy_features(app, (broken, healthy))

    with TestClient(app) as client:
        assert "/delta/list" in _paths(app)
        assert loaded_lazy_modules(app) == {broken.module_path, healthy.module_path}
        assert client.get("/delta/list").json() == {"feature": "delta"}
        assert client.get("/broken").status_code == 404


def test_flag_hides_the_swagger_warmup_plugin(monkeypatch: pytest.MonkeyPatch) -> None:
    import litellm.proxy._lazy_openapi_snapshot as snapshot

    monkeypatch.setattr(snapshot, "SNAPSHOT_FILE", snapshot.SNAPSHOT_FILE.with_name("missing-snapshot.json"))
    monkeypatch.setenv(FLAG, "false")
    assert lazy_tag_to_prefix() != {}, "control: without the flag and without a snapshot the plugin has tags"
    monkeypatch.setenv(FLAG, "true")
    assert lazy_tag_to_prefix() == {}
