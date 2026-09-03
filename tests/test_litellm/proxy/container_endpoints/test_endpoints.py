import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.container_endpoints import endpoints
from litellm.types.containers.main import ContainerListResponse

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


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(endpoints.router)
    app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(user_id="user-1")
    return TestClient(app)


def test_list_containers_forwards_typed_pagination_params(monkeypatch):
    monkeypatch.setitem(sys.modules, "litellm.proxy.proxy_server", PROXY_SERVER_STUB)
    upstream = ContainerListResponse(object="list", data=[], has_more=True)
    captured = {}

    class FakeProcessor:
        def __init__(self, data):
            captured["data"] = data

        async def base_process_llm_request(self, **kwargs):
            return upstream

        async def _handle_llm_api_exception(self, **kwargs):
            raise kwargs["e"]

    monkeypatch.setattr(endpoints, "ProxyBaseLLMRequestProcessing", FakeProcessor)
    monkeypatch.setattr(endpoints, "filter_container_list_response", AsyncMock(return_value=upstream))

    response = _client().get(
        "/v1/containers",
        params={"limit": "1", "order": "desc", "after": "cntr_prev"},
        headers={"Authorization": "Bearer sk-test"},
    )

    assert response.status_code == 200
    assert response.json()["has_more"] is True
    assert captured["data"]["limit"] == 1
    assert captured["data"]["order"] == "desc"
    assert captured["data"]["after"] == "cntr_prev"
