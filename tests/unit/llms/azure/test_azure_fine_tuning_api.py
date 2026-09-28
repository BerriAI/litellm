import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from openai import AsyncAzureOpenAI

import litellm
from litellm.llms.azure.fine_tuning.handler import AzureOpenAIFineTuningAPI


def _expected_dir() -> Path:
    return Path(__file__).resolve().parent.parent.parent / "expected_fine_tuning_api"


def _load_json(file_name: str) -> dict:
    path = _expected_dir() / file_name
    assert path.exists(), f"Expected fixture file not found: {path}"
    with open(path) as f:
        return json.load(f)


class _MockSDKResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def model_dump(self) -> dict:
        return self._payload


def _mock_azure_client(
    create_payload: dict | None = None,
    list_payload: dict | None = None,
    cancel_payload: dict | None = None,
):
    client = AsyncAzureOpenAI(
        api_key="test-key",
        api_version="2024-10-21",
        azure_endpoint="https://exampleopenaiendpoint-production.up.railway.app",
    )
    client.fine_tuning.jobs.create = AsyncMock(
        return_value=(
            _MockSDKResponse(create_payload) if create_payload is not None else None
        )
    )  # type: ignore[method-assign]
    client.fine_tuning.jobs.list = AsyncMock(
        return_value=list_payload
    )  # type: ignore[method-assign]
    client.fine_tuning.jobs.cancel = AsyncMock(
        return_value=(
            _MockSDKResponse(cancel_payload) if cancel_payload is not None else None
        )
    )  # type: ignore[method-assign]
    return client


@pytest.mark.asyncio
async def test_azure_acreate_fine_tuning_job_request_and_output_match_expected_json():
    expected_request = _load_json("azure_create_request.json")
    raw_response = _load_json("azure_create_raw_response.json")
    expected_output = _load_json("azure_create_expected_output.json")

    mock_client = _mock_azure_client(create_payload=raw_response)

    with patch.object(
        AzureOpenAIFineTuningAPI, "get_openai_client", return_value=mock_client
    ):
        response = await litellm.acreate_fine_tuning_job(
            model="gpt-35-turbo-1106",
            training_file="file-5e4b20ecbd724182b9964f3cd2ab7212",
            custom_llm_provider="azure",
            api_base="https://exampleopenaiendpoint-production.up.railway.app",
            api_key="test-key",
            api_version="2024-10-21",
        )

    request_kwargs = mock_client.fine_tuning.jobs.create.call_args.kwargs
    assert request_kwargs == expected_request

    response_dict = response.model_dump(exclude={"_hidden_params"})
    for key, expected_value in expected_output.items():
        assert key in response_dict, f"Missing key in response: {key}"
        assert response_dict[key] == expected_value

    assert response.id is not None
    assert response.model == "davinci-002"


@pytest.mark.asyncio
async def test_azure_alist_fine_tuning_jobs_request_matches_expected_json():
    expected_request = _load_json("azure_list_request.json")
    raw_list_response = _load_json("azure_list_raw_response.json")

    mock_client = _mock_azure_client(list_payload=raw_list_response)

    with patch.object(
        AzureOpenAIFineTuningAPI, "get_openai_client", return_value=mock_client
    ):
        response = await litellm.alist_fine_tuning_jobs(
            after=expected_request["after"],
            limit=expected_request["limit"],
            custom_llm_provider="azure",
            api_base="https://exampleopenaiendpoint-production.up.railway.app",
            api_key="test-key",
            api_version="2024-10-21",
        )

    request_kwargs = mock_client.fine_tuning.jobs.list.call_args.kwargs
    assert request_kwargs == expected_request
    assert response == raw_list_response


@pytest.mark.asyncio
async def test_azure_acancel_fine_tuning_job_request_and_output_match_expected_json():
    expected_request = _load_json("azure_cancel_request.json")
    raw_response = _load_json("azure_cancel_raw_response.json")
    expected_output = _load_json("azure_cancel_expected_output.json")

    mock_client = _mock_azure_client(cancel_payload=raw_response)

    with patch.object(
        AzureOpenAIFineTuningAPI, "get_openai_client", return_value=mock_client
    ):
        response = await litellm.acancel_fine_tuning_job(
            fine_tuning_job_id=expected_request["fine_tuning_job_id"],
            custom_llm_provider="azure",
            api_base="https://exampleopenaiendpoint-production.up.railway.app",
            api_key="test-key",
            api_version="2024-10-21",
        )

    request_kwargs = mock_client.fine_tuning.jobs.cancel.call_args.kwargs
    assert request_kwargs == expected_request

    response_dict = response.model_dump(exclude={"_hidden_params"})
    for key, expected_value in expected_output.items():
        assert key in response_dict, f"Missing key in response: {key}"
        assert response_dict[key] == expected_value

    assert response.status == "cancelled"


def test_azure_trainingtype_defaults_to_one():
    handler = AzureOpenAIFineTuningAPI()
    create_data = {"model": "gpt-4o-mini", "training_file": "file-test"}

    handler._ensure_training_type(create_data)

    assert "extra_body" in create_data
    assert create_data["extra_body"]["trainingType"] == 1
