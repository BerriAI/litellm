import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.container_endpoints import endpoints, ownership
from litellm.types.containers.main import ContainerListResponse, ContainerObject

PROXY_SERVER_STUB = SimpleNamespace(
    general_settings={},
    prisma_client=None,
    llm_router=None,
    proxy_config=None,
    proxy_logging_obj=None,
    select_data_generator=None,
    user_api_base=None,
    user_max_tokens=None,
    user_model=None,
    user_request_timeout=None,
    user_temperature=None,
    version="test",
)
ADMIN = UserAPIKeyAuth(user_id="admin-1", user_role=LitellmUserRoles.PROXY_ADMIN)
NON_ADMIN = UserAPIKeyAuth(user_id="user-1")


@pytest.fixture(autouse=True)
def clear_allowed_container_ids_cache():
    ownership._ALLOWED_CONTAINER_IDS_CACHE.cache_dict.clear()
    ownership._ALLOWED_CONTAINER_IDS_CACHE.ttl_dict.clear()
    yield
    ownership._ALLOWED_CONTAINER_IDS_CACHE.cache_dict.clear()
    ownership._ALLOWED_CONTAINER_IDS_CACHE.ttl_dict.clear()


def _client(auth: UserAPIKeyAuth) -> TestClient:
    app = FastAPI()
    app.include_router(endpoints.router)
    app.dependency_overrides[user_api_key_auth] = lambda: auth
    return TestClient(app)


def _container(container_id: str) -> ContainerObject:
    return ContainerObject(id=container_id, object="container", created_at=1, status="active")


def _page(*container_ids: str, has_more: bool) -> ContainerListResponse:
    return ContainerListResponse(
        object="list",
        data=[_container(container_id) for container_id in container_ids],
        has_more=has_more,
    )


def _upstream_pages(monkeypatch, pages_by_after) -> MagicMock:
    processor_cls = MagicMock(
        side_effect=lambda data: SimpleNamespace(
            base_process_llm_request=AsyncMock(return_value=pages_by_after[data["after"]])
        )
    )
    monkeypatch.setattr(endpoints, "ProxyBaseLLMRequestProcessing", processor_cls)
    return processor_cls


def _forwarded_pages(processor_cls: MagicMock):
    return [(call.kwargs["data"]["after"], call.kwargs["data"]["limit"]) for call in processor_cls.call_args_list]


def test_list_containers_forwards_typed_pagination_params_for_admins(monkeypatch):
    monkeypatch.setitem(sys.modules, "litellm.proxy.proxy_server", PROXY_SERVER_STUB)
    processor_cls = _upstream_pages(monkeypatch, {"cntr_prev": _page("cntr_next", has_more=True)})

    response = _client(ADMIN).get(
        "/v1/containers",
        params={"limit": "1", "order": "desc", "after": "cntr_prev"},
        headers={"Authorization": "Bearer sk-test"},
    )

    assert response.status_code == 200
    assert [item["id"] for item in response.json()["data"]] == ["cntr_next"]
    assert response.json()["has_more"] is True
    assert _forwarded_pages(processor_cls) == [("cntr_prev", 1)]
    assert processor_cls.call_args.kwargs["data"]["order"] == "desc"


def test_list_containers_rejects_a_non_integer_limit(monkeypatch):
    monkeypatch.setitem(sys.modules, "litellm.proxy.proxy_server", PROXY_SERVER_STUB)
    processor_cls = _upstream_pages(monkeypatch, {})

    response = _client(ADMIN).get(
        "/v1/containers",
        params={"limit": "abc"},
        headers={"Authorization": "Bearer sk-test"},
    )

    assert response.status_code == 422
    processor_cls.assert_not_called()


def test_list_containers_pages_upstream_until_non_admin_keys_see_their_containers(monkeypatch):
    monkeypatch.setitem(sys.modules, "litellm.proxy.proxy_server", PROXY_SERVER_STUB)
    table = AsyncMock()
    table.find_many.return_value = [SimpleNamespace(model_object_id="container:openai:cntr_owned")]
    monkeypatch.setattr(
        ownership,
        "_get_prisma_client",
        AsyncMock(return_value=SimpleNamespace(db=SimpleNamespace(litellm_managedobjecttable=table))),
    )
    processor_cls = _upstream_pages(
        monkeypatch,
        {
            None: _page("cntr_other", has_more=True),
            "cntr_other": _page("cntr_owned", has_more=False),
        },
    )

    response = _client(NON_ADMIN).get(
        "/v1/containers",
        params={"limit": "1"},
        headers={"Authorization": "Bearer sk-test"},
    )

    assert response.status_code == 200
    body = response.json()
    assert [item["id"] for item in body["data"]] == ["cntr_owned"]
    assert body["first_id"] == "cntr_owned"
    assert body["last_id"] == "cntr_owned"
    assert body["has_more"] is False
    assert _forwarded_pages(processor_cls) == [(None, 100), ("cntr_other", 100)]
