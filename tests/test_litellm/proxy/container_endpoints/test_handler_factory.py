import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.container_endpoints import endpoints, handler_factory

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


def test_list_container_files_forwards_declared_query_params(monkeypatch):
    monkeypatch.setitem(sys.modules, "litellm.proxy.proxy_server", PROXY_SERVER_STUB)
    monkeypatch.setattr(
        handler_factory,
        "assert_user_can_access_container",
        AsyncMock(return_value=("cntr_123", "openai")),
    )
    processor_cls = MagicMock()
    processor_cls.return_value.base_process_llm_request = AsyncMock(
        return_value={"object": "list", "data": [], "has_more": True}
    )
    monkeypatch.setattr(handler_factory, "ProxyBaseLLMRequestProcessing", processor_cls)

    response = _client().get(
        "/v1/containers/cntr_123/files",
        params={"limit": "1", "order": "desc", "after": "cfile_prev", "unknown": "x"},
        headers={"Authorization": "Bearer sk-test"},
    )

    assert response.status_code == 200
    assert response.json()["has_more"] is True
    assert processor_cls.return_value.base_process_llm_request.await_args.kwargs["route_type"] == "alist_container_files"
    forwarded = processor_cls.call_args.kwargs["data"]
    assert forwarded["container_id"] == "cntr_123"
    assert forwarded["limit"] == "1"
    assert forwarded["order"] == "desc"
    assert forwarded["after"] == "cfile_prev"
    assert "unknown" not in forwarded
