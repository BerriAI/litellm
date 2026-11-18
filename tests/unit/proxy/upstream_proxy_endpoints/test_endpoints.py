from dataclasses import dataclass, field
from typing import Final

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from litellm.llms.custom_httpx.upstream_proxy import UPSTREAM_PROXY_REGISTRY, resolve_upstream_proxy
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.upstream_proxy_endpoints.endpoints import (
    ProxyReferences,
    UpstreamProxyList,
    config_repository,
    current_references,
)
from litellm.proxy.upstream_proxy_endpoints.endpoints import router as upstream_proxy_router
from litellm.repositories.config_repository import ConfigParam

_JP: Final = {"name": "jp", "protocol": "http", "host": "10.80.0.64", "port": 3128}


@dataclass
class _MemoryConfigRow:
    value: object = None
    revision: int = 0
    lose_next_write: bool = False
    writes: int = 0

    async def get_param(self, param_name: str) -> ConfigParam | None:
        return None if self.value is None else ConfigParam(param_name=param_name, param_value=self.value)

    async def set_param_if_revision(self, param_name: str, param_value: object, revision: int) -> bool:
        if self.lose_next_write:
            return False
        if self.revision != revision:
            return False
        self.value = param_value
        self.revision += 1
        self.writes += 1
        return True


@dataclass
class _Harness:
    row: _MemoryConfigRow = field(default_factory=_MemoryConfigRow)
    references: ProxyReferences = field(default_factory=lambda: ProxyReferences(models=(), credentials=()))
    role: LitellmUserRoles = LitellmUserRoles.PROXY_ADMIN

    def client(self) -> TestClient:
        app: Final = FastAPI()
        app.include_router(upstream_proxy_router)
        app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(user_role=self.role, api_key="sk-1")
        app.dependency_overrides[config_repository] = lambda: self.row
        app.dependency_overrides[current_references] = lambda: self.references
        return TestClient(app)


@pytest.fixture(autouse=True)
def _salt_and_empty_registry(monkeypatch: pytest.MonkeyPatch) -> None:  # pyright: ignore[reportUnusedFunction]  # autouse fixture
    monkeypatch.setenv("LITELLM_SALT_KEY", "sk-test-salt")
    UPSTREAM_PROXY_REGISTRY.replace(())


def test_created_proxy_resolves_by_name_with_its_credentials_and_stores_the_password_encrypted() -> None:
    harness: Final = _Harness()

    response: Final = harness.client().post("/upstream_proxies", json={**_JP, "username": "u", "password": "p@ss"})

    assert response.status_code == 200
    assert response.json()["proxies"][0]["has_password"] is True
    assert "p@ss" not in str(harness.row.value)
    assert resolve_upstream_proxy("jp", UPSTREAM_PROXY_REGISTRY) == "http://u:p%40ss@10.80.0.64:3128"


def test_patch_keeps_an_omitted_password_and_clears_an_empty_one() -> None:
    harness: Final = _Harness()
    client: Final = harness.client()
    client.post("/upstream_proxies", json={**_JP, "username": "u", "password": "p"})

    client.patch("/upstream_proxies/jp", json={"port": 8080})
    assert resolve_upstream_proxy("jp", UPSTREAM_PROXY_REGISTRY) == "http://u:p@10.80.0.64:8080"

    client.patch("/upstream_proxies/jp", json={"username": "", "password": ""})
    assert resolve_upstream_proxy("jp", UPSTREAM_PROXY_REGISTRY) == "http://10.80.0.64:8080"


def test_patch_with_an_invalid_host_is_rejected_and_leaves_the_proxy_unchanged() -> None:
    harness: Final = _Harness()
    client: Final = harness.client()
    client.post("/upstream_proxies", json=_JP)

    response: Final = client.patch("/upstream_proxies/jp", json={"host": "bad host/x"})

    assert response.status_code == 422
    assert resolve_upstream_proxy("jp", UPSTREAM_PROXY_REGISTRY) == "http://10.80.0.64:3128"


@pytest.mark.parametrize("protocol", ("socks5", "socks5h", "ftp"))
def test_only_protocols_the_transport_can_apply_are_accepted(protocol: str) -> None:
    response: Final = _Harness().client().post("/upstream_proxies", json={**_JP, "protocol": protocol})

    assert response.status_code == 422


def test_a_proxy_still_referenced_cannot_be_deleted_and_an_unreferenced_one_stops_resolving() -> None:
    harness: Final = _Harness(
        references=ProxyReferences(models=(("named-jp", "jp"), ("other", None)), credentials=(("jp-account", "jp"),))
    )
    client: Final = harness.client()
    client.post("/upstream_proxies", json=_JP)
    client.post("/upstream_proxies", json={**_JP, "name": "eu", "host": "10.34.6.157"})

    blocked: Final = client.delete("/upstream_proxies/jp")
    deleted: Final = client.delete("/upstream_proxies/eu")

    assert blocked.status_code == 409
    assert "named-jp" in blocked.json()["detail"]
    assert "jp-account" in blocked.json()["detail"]
    assert deleted.status_code == 200
    assert [proxy.name for proxy in UpstreamProxyList.model_validate(deleted.json()).proxies] == ["jp"]
    with pytest.raises(ValueError, match="names no proxy"):
        resolve_upstream_proxy("eu", UPSTREAM_PROXY_REGISTRY)


def test_a_write_racing_another_admin_is_refused_instead_of_overwriting_it() -> None:
    harness: Final = _Harness()
    client: Final = harness.client()
    client.post("/upstream_proxies", json=_JP)
    harness.row.lose_next_write = True

    response: Final = client.patch("/upstream_proxies/jp", json={"port": 9})

    assert response.status_code == 409
    assert resolve_upstream_proxy("jp", UPSTREAM_PROXY_REGISTRY) == "http://10.80.0.64:3128"


def test_duplicate_names_are_refused() -> None:
    client: Final = _Harness().client()
    client.post("/upstream_proxies", json=_JP)

    assert client.post("/upstream_proxies", json={**_JP, "port": 1}).status_code == 409


def test_view_only_admin_can_list_but_not_change_proxies() -> None:
    harness: Final = _Harness(role=LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY)
    client: Final = harness.client()

    assert client.get("/upstream_proxies").status_code == 200
    assert client.post("/upstream_proxies", json=_JP).status_code == 403
    assert harness.row.writes == 0


def test_internal_user_cannot_read_proxies() -> None:
    harness: Final = _Harness(role=LitellmUserRoles.INTERNAL_USER)

    assert harness.client().get("/upstream_proxies").status_code == 403
