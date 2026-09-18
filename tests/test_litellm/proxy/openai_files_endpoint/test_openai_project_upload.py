from typing import Final
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient
from starlette.requests import Request

import litellm
from litellm.caching.caching import DualCache
from litellm.proxy import proxy_server
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.openai_files_endpoints import files_endpoints
from litellm.proxy.openai_files_endpoints.common_utils import add_openai_project_header
from litellm.proxy.utils import ProxyLogging
from litellm.router import Router
from litellm.types.llms.openai import OpenAIFileObject

VALID_BATCH_LINE = (
    b'{"custom_id":"req-1","method":"POST","url":"/v1/chat/completions",'
    b'"body":{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}]}}\n'
)


@pytest.mark.parametrize(
    ("form_fields", "expected_project"),
    [
        ({}, "proj-request"),
        ({"project": "proj-form"}, "proj-form"),
        ({"model": "gpt-4o"}, "proj-request"),
    ],
)
def test_upload_forwards_openai_project(
    monkeypatch: pytest.MonkeyPatch,
    form_fields: dict[str, str],
    expected_project: str,
) -> None:
    router = Router(
        model_list=[
            {
                "model_name": "gpt-4o",
                "litellm_params": {"model": "openai/gpt-4o", "api_key": "sk-test"},
            }
        ]
    )
    proxy_logging_obj = ProxyLogging(user_api_key_cache=DualCache(default_in_memory_ttl=1))
    proxy_logging_obj._add_proxy_hooks(router)
    proxy_logging_obj.update_request_status = AsyncMock()
    proxy_logging_obj.post_call_failure_hook = AsyncMock()
    monkeypatch.setattr(proxy_server, "proxy_logging_obj", proxy_logging_obj)
    monkeypatch.setattr(proxy_server, "llm_router", router)
    monkeypatch.setattr(proxy_server, "master_key", None)
    monkeypatch.setattr(proxy_server, "prisma_client", None)
    monkeypatch.setattr(files_endpoints, "files_config", [])

    captured_kwargs: dict[str, object] = {}

    async def fake_acreate_file(**kwargs: object) -> OpenAIFileObject:
        captured_kwargs.update(kwargs)
        return OpenAIFileObject(
            id="file-openai-123",
            object="file",
            bytes=2,
            created_at=1234567890,
            filename="batch.jsonl",
            purpose="batch",
            status="uploaded",
        )

    monkeypatch.setattr(litellm, "acreate_file", fake_acreate_file)
    proxy_server.app.dependency_overrides[proxy_server.user_api_key_auth] = lambda: UserAPIKeyAuth(
        api_key="test-key",
        user_role=LitellmUserRoles.PROXY_ADMIN,
        user_id="test-user",
    )

    try:
        response = TestClient(proxy_server.app).post(
            "/v1/files",
            files={"file": ("batch.jsonl", VALID_BATCH_LINE, "application/jsonl")},
            data={"purpose": "batch", **form_fields},
            headers={"Authorization": "Bearer test-key", "OpenAI-Project": "proj-request"},
        )
    finally:
        proxy_server.app.dependency_overrides.pop(proxy_server.user_api_key_auth, None)

    assert response.status_code == 200, response.text
    assert captured_kwargs["extra_headers"] == {"OpenAI-Project": expected_project}
    proxy_logging_obj.post_call_failure_hook.assert_not_called()


def test_project_header_preserves_other_configured_headers() -> None:
    request: Final = Request({"type": "http", "headers": [(b"openai-project", b"proj-request")], "query_string": b""})
    data: Final[dict[str, object]] = {"extra_headers": {"X-Custom": "keep", "openai-project": "proj-config"}}

    add_openai_project_header(data, request, "openai")

    assert data["extra_headers"] == {"X-Custom": "keep", "OpenAI-Project": "proj-config"}
