"""Regression tests: the OpenAI `project` parameter must reach the SDK client
for the Batches and Files APIs (BerriAI/litellm#41803).

Project-scoped API keys were failing with 401s because `project` was accepted
by litellm's batches/files functions but never forwarded to the OpenAI client,
so the `OpenAI-Project` header was never sent.
"""

from typing import Any, Final

import httpx
import pytest
import respx

import litellm
from litellm.batches import main as batches_main
from litellm.llms.openai.common_utils import get_openai_credentials
from litellm.llms.openai.openai import OpenAIBatchesAPI, OpenAIFilesAPI
from litellm.types.llms.openai import CreateBatchRequest, CreateFileRequest
from litellm.types.router import GenericLiteLLMParams

_API_BASE: Final = "https://api.openai.com/v1"


def _batch_json() -> dict[str, Any]:
    return {
        "id": "batch_123",
        "object": "batch",
        "endpoint": "/v1/chat/completions",
        "input_file_id": "file-123",
        "completion_window": "24h",
        "status": "validating",
        "created_at": 1234567890,
    }


def _file_json() -> dict[str, Any]:
    return {
        "id": "file-123",
        "object": "file",
        "bytes": 2,
        "created_at": 1234567890,
        "filename": "test.jsonl",
        "purpose": "batch",
        "status": "processed",
    }


def _client_kwargs(**overrides: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "api_key": "sk-test",
        "api_base": _API_BASE,
        "timeout": 10.0,
        "max_retries": 0,
        "organization": "org-123",
        "project": "proj-abc",
    }
    kwargs.update(overrides)
    return kwargs


def test_files_get_openai_client_sets_project_header():
    client = OpenAIFilesAPI().get_openai_client(**_client_kwargs())
    assert client is not None
    assert client.project == "proj-abc"
    assert client.default_headers["OpenAI-Project"] == "proj-abc"
    assert client.organization == "org-123"


def test_batches_get_openai_client_sets_project_header():
    client = OpenAIBatchesAPI().get_openai_client(**_client_kwargs())
    assert client is not None
    assert client.project == "proj-abc"
    assert client.default_headers["OpenAI-Project"] == "proj-abc"


def test_get_openai_client_omits_project_when_none():
    client = OpenAIFilesAPI().get_openai_client(**_client_kwargs(organization=None, project=None))
    assert client is not None
    assert client.project is None


def test_create_batch_sends_project_header(respx_mock: respx.MockRouter):
    route: Final = respx_mock.post(f"{_API_BASE}/batches").mock(return_value=httpx.Response(200, json=_batch_json()))
    OpenAIBatchesAPI().create_batch(
        _is_async=False,
        create_batch_data=CreateBatchRequest(
            input_file_id="file-123",
            endpoint="/v1/chat/completions",
            completion_window="24h",
        ),
        **_client_kwargs(organization=None, project="proj-xyz"),
    )
    assert route.calls.last.request.headers["OpenAI-Project"] == "proj-xyz"


def test_create_file_sends_project_header(respx_mock: respx.MockRouter):
    route: Final = respx_mock.post(f"{_API_BASE}/files").mock(return_value=httpx.Response(200, json=_file_json()))
    OpenAIFilesAPI().create_file(
        _is_async=False,
        create_file_data=CreateFileRequest(
            file=("test.jsonl", b"{}", "application/json"),
            purpose="batch",
        ),
        **_client_kwargs(organization=None, project="proj-xyz"),
    )
    assert route.calls.last.request.headers["OpenAI-Project"] == "proj-xyz"


def test_get_openai_credentials_resolves_project(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("OPENAI_PROJECT", raising=False)
    monkeypatch.setattr(litellm, "project", None)
    assert get_openai_credentials(project="proj-param").project == "proj-param"
    monkeypatch.setattr(litellm, "project", "proj-global")
    assert get_openai_credentials().project == "proj-global"
    monkeypatch.setattr(litellm, "project", None)
    monkeypatch.setenv("OPENAI_PROJECT", "proj-env")
    assert get_openai_credentials().project == "proj-env"
    monkeypatch.delenv("OPENAI_PROJECT", raising=False)
    assert get_openai_credentials().project is None


def test_resolve_openai_project_precedence(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("OPENAI_PROJECT", raising=False)
    monkeypatch.setattr(litellm, "project", None)
    params: Final = GenericLiteLLMParams()
    assert batches_main._resolve_openai_project(params, {"OpenAI-Project": "proj-header"}) == "proj-header"
    param_project: Final = GenericLiteLLMParams(project="proj-param")
    assert batches_main._resolve_openai_project(param_project, {"OpenAI-Project": "proj-header"}) == "proj-param"
    monkeypatch.setattr(litellm, "project", "proj-global")
    monkeypatch.setenv("OPENAI_PROJECT", "proj-env")
    assert batches_main._resolve_openai_project(params, None) == "proj-global"
    monkeypatch.setattr(litellm, "project", None)
    assert batches_main._resolve_openai_project(params, None) == "proj-env"
    monkeypatch.delenv("OPENAI_PROJECT", raising=False)
    assert batches_main._resolve_openai_project(params, None) is None


@pytest.mark.asyncio
async def test_acreate_batch_forwards_project_header(respx_mock: respx.MockRouter):
    """End to end: project kwarg and OpenAI-Project header reach the wire."""
    route: Final = respx_mock.post(f"{_API_BASE}/batches").mock(return_value=httpx.Response(200, json=_batch_json()))
    await batches_main.acreate_batch(
        completion_window="24h",
        endpoint="/v1/chat/completions",
        input_file_id="file-123",
        custom_llm_provider="openai",
        api_key="sk-test",
        project="proj-explicit",
    )
    assert route.calls.last.request.headers["OpenAI-Project"] == "proj-explicit"

    await batches_main.acreate_batch(
        completion_window="24h",
        endpoint="/v1/chat/completions",
        input_file_id="file-123",
        custom_llm_provider="openai",
        api_key="sk-test",
        extra_headers={"OpenAI-Project": "proj-header"},
    )
    assert route.calls.last.request.headers["OpenAI-Project"] == "proj-header"
