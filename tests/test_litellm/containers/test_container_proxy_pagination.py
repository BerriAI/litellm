from typing import Any, Dict, List

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.container_endpoints.pagination import (
    parse_container_list_query_params,
)


@pytest.fixture
def captured_data(monkeypatch) -> List[Dict[str, Any]]:
    from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing

    captured: List[Dict[str, Any]] = []

    async def fake_base_process_llm_request(self, **kwargs):
        captured.append(self.data)
        return {"object": "list", "data": []}

    monkeypatch.setattr(
        ProxyBaseLLMRequestProcessing,
        "base_process_llm_request",
        fake_base_process_llm_request,
    )
    return captured


@pytest.fixture
def client(monkeypatch) -> TestClient:
    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
    from litellm.proxy.container_endpoints import endpoints, ownership

    async def fake_auth():
        return UserAPIKeyAuth(user_id="user-1", user_role=LitellmUserRoles.PROXY_ADMIN)

    async def fake_assert_user_can_access_container(container_id, user_api_key_dict, custom_llm_provider):
        return container_id, custom_llm_provider

    async def fake_get_container_forwarding_params(container_id, original_container_id, custom_llm_provider):
        return {"container_id": container_id, "custom_llm_provider": custom_llm_provider}

    async def fake_filter_container_list_response(response, user_api_key_dict, custom_llm_provider):
        return response

    for module in (endpoints, ownership):
        monkeypatch.setattr(
            module,
            "assert_user_can_access_container",
            fake_assert_user_can_access_container,
            raising=False,
        )
        monkeypatch.setattr(
            module,
            "get_container_forwarding_params",
            fake_get_container_forwarding_params,
            raising=False,
        )
    monkeypatch.setattr(
        endpoints,
        "filter_container_list_response",
        fake_filter_container_list_response,
    )

    from litellm.proxy.container_endpoints import handler_factory

    monkeypatch.setattr(
        handler_factory,
        "assert_user_can_access_container",
        fake_assert_user_can_access_container,
    )
    monkeypatch.setattr(
        handler_factory,
        "get_container_forwarding_params",
        fake_get_container_forwarding_params,
    )

    app = FastAPI()
    app.include_router(endpoints.router)
    app.dependency_overrides[user_api_key_auth] = fake_auth
    return TestClient(app)


def test_should_forward_container_list_pagination_as_top_level_params(client, captured_data):
    response = client.get("/v1/containers?after=cntr_cursor&limit=20&order=desc")

    assert response.status_code == 200
    assert captured_data[0]["after"] == "cntr_cursor"
    assert captured_data[0]["limit"] == 20
    assert captured_data[0]["order"] == "desc"
    assert "query_params" not in captured_data[0]


def test_should_forward_container_file_list_pagination_as_top_level_params(client, captured_data):
    response = client.get("/v1/containers/cntr_123/files?after=cfile_cursor&limit=5&order=asc")

    assert response.status_code == 200
    assert captured_data[0]["after"] == "cfile_cursor"
    assert captured_data[0]["limit"] == 5
    assert captured_data[0]["order"] == "asc"
    assert captured_data[0]["container_id"] == "cntr_123"
    assert "query_params" not in captured_data[0]


def test_should_not_forward_pagination_params_to_non_list_container_routes(client, captured_data):
    response = client.get("/v1/containers/cntr_123/files/cfile_1?limit=5")

    assert response.status_code == 200
    assert "limit" not in captured_data[0]


def test_should_reject_invalid_container_list_pagination_params(client, captured_data):
    assert client.get("/v1/containers?order=sideways").status_code == 400
    assert client.get("/v1/containers?limit=abc").status_code == 400
    assert client.get("/v1/containers?limit=0").status_code == 400
    assert captured_data == []


def test_should_ignore_non_pagination_query_params():
    parsed = parse_container_list_query_params({"custom_llm_provider": "azure", "limit": "3"})

    assert parsed == {"limit": 3}
