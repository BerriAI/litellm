import json
from typing import Final, Literal

import httpx
import pytest
import respx

import litellm
from litellm.types.utils import LiteLLMFineTuningJob

API_BASE: Final = "https://fine-tuning.test"
BASE_MODEL: Final = "gpt-4.1-nano-2025-04-14"
PROVIDER_JOB: Final = {
    "id": "ftjob-abc123",
    "object": "fine_tuning.job",
    "created_at": 1700000000,
    "error": None,
    "fine_tuned_model": None,
    "finished_at": None,
    "hyperparameters": {"n_epochs": 3},
    "model": BASE_MODEL,
    "organization_id": "org-test",
    "result_files": [],
    "seed": 42,
    "status": "validating_files",
    "trained_tokens": None,
    "training_file": "file-abc123",
    "validation_file": None,
}


@pytest.mark.asyncio
@respx.mock
@pytest.mark.parametrize(
    ("custom_llm_provider", "route"),
    [("openai", f"{API_BASE}/v1/fine_tuning/jobs"), ("azure", f"{API_BASE}/openai/fine_tuning/jobs")],
)
@pytest.mark.parametrize("prefixed", [True, False], ids=["provider-prefixed", "bare"])
async def test_create_sends_the_model_without_the_provider_prefix(
    monkeypatch: pytest.MonkeyPatch, custom_llm_provider: str, route: str, prefixed: bool
) -> None:
    sent: Final = respx.post(url__startswith=route).mock(return_value=httpx.Response(200, json=PROVIDER_JOB))

    async with httpx.AsyncClient() as session:
        monkeypatch.setattr(litellm, "aclient_session", session)
        await litellm.acreate_fine_tuning_job(
            model=f"{custom_llm_provider}/{BASE_MODEL}" if prefixed else BASE_MODEL,
            training_file="file-abc123",
            custom_llm_provider=custom_llm_provider,
            api_key="sk-test",
            api_base=f"{API_BASE}/v1" if custom_llm_provider == "openai" else API_BASE,
            **({"api_version": "2024-10-21"} if custom_llm_provider == "azure" else {}),
        )

    assert json.loads(sent.calls.last.request.content)["model"] == BASE_MODEL


@pytest.mark.asyncio
@respx.mock
@pytest.mark.parametrize(
    ("custom_llm_provider", "api_base", "route", "sent_api_version"),
    [
        ("openai", f"{API_BASE}/v1", f"{API_BASE}/v1/fine_tuning/jobs", None),
        ("azure", API_BASE, f"{API_BASE}/openai/fine_tuning/jobs", "2024-10-21"),
    ],
)
@pytest.mark.parametrize("is_async", [True, False], ids=["async", "sync"])
async def test_create_with_api_version_reaches_the_provider_and_only_azure_gets_it(
    monkeypatch: pytest.MonkeyPatch,
    custom_llm_provider: Literal["openai", "azure"],
    api_base: str,
    route: str,
    sent_api_version: str | None,
    is_async: bool,
) -> None:
    sent: Final = respx.post(url__startswith=route).mock(return_value=httpx.Response(200, json=PROVIDER_JOB))
    request: Final = {
        "model": BASE_MODEL,
        "training_file": "file-abc123",
        "custom_llm_provider": custom_llm_provider,
        "api_key": "sk-test",
        "api_base": api_base,
        "api_version": "2024-10-21",
    }

    async with httpx.AsyncClient() as session:
        monkeypatch.setattr(litellm, "aclient_session", session)
        job: Final = (
            await litellm.acreate_fine_tuning_job(**request) if is_async else litellm.create_fine_tuning_job(**request)
        )

    assert isinstance(job, LiteLLMFineTuningJob)
    assert job.id == PROVIDER_JOB["id"]
    assert sent.calls.last.request.url.params.get("api-version") == sent_api_version
