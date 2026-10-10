from typing import Final

import pytest
import respx
from httpx import Response
from openai import DEFAULT_MAX_RETRIES, AsyncOpenAI, OpenAI

import litellm
from litellm.llms.openai.fine_tuning.handler import OpenAIFineTuningAPI
from litellm.types.utils import LiteLLMFineTuningJob

_API_BASE: Final = "https://example.test/v1"
_FINE_TUNING_JOB: Final = {
    "id": "ftjob-openai-create-123",
    "object": "fine_tuning.job",
    "created_at": 1735689600,
    "model": "gpt-5.4-nano",
    "status": "queued",
    "fine_tuned_model": None,
    "training_file": "file-abc123",
    "hyperparameters": {"n_epochs": "auto"},
    "organization_id": "org-test",
    "result_files": [],
    "validation_file": None,
    "trained_tokens": None,
    "seed": 0,
    "error": None,
}


@pytest.mark.parametrize(("is_async", "expected_class"), [(False, OpenAI), (True, AsyncOpenAI)])
def test_get_openai_client_ignores_azure_only_params(
    is_async: bool, expected_class: type[OpenAI] | type[AsyncOpenAI]
) -> None:
    client: Final = OpenAIFineTuningAPI().get_openai_client(
        api_key="sk-test",
        api_base=_API_BASE,
        timeout=12.5,
        max_retries=3,
        organization="org-test",
        _is_async=is_async,
        api_version="2024-10-21",
        litellm_params={"api_version": "2024-10-21"},
    )
    assert isinstance(client, expected_class)
    assert str(client.base_url) == f"{_API_BASE}/"
    assert client.timeout == 12.5
    assert client.max_retries == 3
    assert client.organization == "org-test"


def test_get_openai_client_without_max_retries_keeps_the_sdk_default() -> None:
    client: Final = OpenAIFineTuningAPI().get_openai_client(
        api_key="sk-test", api_base=None, timeout=10.0, max_retries=None, organization=None
    )
    assert isinstance(client, OpenAI)
    assert client.max_retries == DEFAULT_MAX_RETRIES


def _create_route(respx_mock: respx.MockRouter) -> respx.Route:
    return respx_mock.post(f"{_API_BASE}/fine_tuning/jobs").mock(return_value=Response(200, json=_FINE_TUNING_JOB))


def test_create_fine_tuning_job_with_api_version_reaches_openai(respx_mock: respx.MockRouter) -> None:
    route: Final = _create_route(respx_mock)
    response: Final = litellm.create_fine_tuning_job(
        model="gpt-5.4-nano",
        training_file="file-abc123",
        custom_llm_provider="openai",
        api_key="sk-test",
        api_base=_API_BASE,
        api_version="2024-10-21",
    )
    assert isinstance(response, LiteLLMFineTuningJob)
    assert response.id == _FINE_TUNING_JOB["id"]
    assert "api-version" not in route.calls.last.request.url.params


@pytest.mark.asyncio
async def test_acreate_fine_tuning_job_with_api_version_reaches_openai(respx_mock: respx.MockRouter) -> None:
    route: Final = _create_route(respx_mock)
    response: Final = await litellm.acreate_fine_tuning_job(
        model="gpt-5.4-nano",
        training_file="file-abc123",
        custom_llm_provider="openai",
        api_key="sk-test",
        api_base=_API_BASE,
        api_version="2024-10-21",
    )
    assert response.id == _FINE_TUNING_JOB["id"]
    assert "api-version" not in route.calls.last.request.url.params
