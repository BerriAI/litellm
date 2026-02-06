import os
import sys
import traceback
import json
import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
from openai import APITimeoutError as Timeout

import litellm

litellm.num_retries = 0
import asyncio
import logging
from typing import Optional
import openai
from test_openai_batches_and_files import load_vertex_ai_credentials

from litellm import create_fine_tuning_job
from litellm._logging import verbose_logger
from litellm.llms.vertex_ai.fine_tuning.handler import (
    FineTuningJobCreate,
    VertexFineTuningAPI,
)
from litellm.types.llms.openai import Hyperparameters
from litellm.integrations.custom_logger import CustomLogger
from litellm.types.utils import StandardLoggingPayload
from unittest.mock import patch, MagicMock, AsyncMock

vertex_finetune_api = VertexFineTuningAPI()


class TestCustomLogger(CustomLogger):
    def __init__(self):
        super().__init__()
        self.standard_logging_object: Optional[StandardLoggingPayload] = None

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        print(
            "Success event logged with kwargs=",
            kwargs,
            "and response_obj=",
            response_obj,
        )
        self.standard_logging_object = kwargs["standard_logging_object"]


@pytest.mark.asyncio
async def test_create_fine_tune_jobs_async():
    try:
        custom_logger = TestCustomLogger()
        litellm.callbacks = ["datadog", custom_logger]
        verbose_logger.setLevel(logging.DEBUG)
        file_name = "openai_batch_completions.jsonl"
        _current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(_current_dir, file_name)

        file_obj = await litellm.acreate_file(
            file=open(file_path, "rb"),
            purpose="fine-tune",
            custom_llm_provider="openai",
        )
        print("Response from creating file=", file_obj)

        create_fine_tuning_response = await litellm.acreate_fine_tuning_job(
            model="gpt-3.5-turbo-0125",
            training_file=file_obj.id,
        )

        print(
            "response from litellm.create_fine_tuning_job=", create_fine_tuning_response
        )

        assert create_fine_tuning_response.id is not None
        assert create_fine_tuning_response.model == "gpt-3.5-turbo-0125"

        await asyncio.sleep(2)
        _logged_standard_logging_object = custom_logger.standard_logging_object
        assert _logged_standard_logging_object is not None
        print(
            "custom_logger.standard_logging_object=",
            json.dumps(_logged_standard_logging_object, indent=4),
        )
        assert _logged_standard_logging_object["model"] == "gpt-3.5-turbo-0125"
        assert _logged_standard_logging_object["id"] == create_fine_tuning_response.id

        # list fine tuning jobs
        print("listing ft jobs")
        ft_jobs = await litellm.alist_fine_tuning_jobs(limit=2)
        print("response from litellm.list_fine_tuning_jobs=", ft_jobs)
        assert len(list(ft_jobs)) > 0

        # retrieve fine tuning job
        response = await litellm.aretrieve_fine_tuning_job(
            fine_tuning_job_id=create_fine_tuning_response.id,
        )
        print("response from litellm.retrieve_fine_tuning_job=", response)

        # delete file

        await litellm.afile_delete(
            file_id=file_obj.id,
        )

        # cancel ft job
        response = await litellm.acancel_fine_tuning_job(
            fine_tuning_job_id=create_fine_tuning_response.id,
        )

        print("response from litellm.cancel_fine_tuning_job=", response)

        assert response.status == "cancelled"
        assert response.id == create_fine_tuning_response.id
    except openai.RateLimitError:
        pass
    except Exception as e:
        if "Job has already completed" in str(e):
            return
        else:
            pytest.fail(f"Error occurred: {e}")
    pass


@pytest.mark.asyncio
async def test_azure_create_fine_tune_jobs_async():
    try:
        verbose_logger.setLevel(logging.DEBUG)
        file_name = "azure_fine_tune.jsonl"
        _current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(_current_dir, file_name)

        file_id = "file-5e4b20ecbd724182b9964f3cd2ab7212"

        create_fine_tuning_response = await litellm.acreate_fine_tuning_job(
            model="gpt-35-turbo-1106",
            training_file=file_id,
            custom_llm_provider="azure",
            api_base="https://exampleopenaiendpoint-production.up.railway.app",
        )

        print(
            "response from litellm.create_fine_tuning_job=", create_fine_tuning_response
        )

        assert create_fine_tuning_response.id is not None

        # response from Example/mocked endpoint
        assert create_fine_tuning_response.model == "davinci-002"

        # list fine tuning jobs
        print("listing ft jobs")
        ft_jobs = await litellm.alist_fine_tuning_jobs(
            limit=2,
            custom_llm_provider="azure",
            api_base="https://exampleopenaiendpoint-production.up.railway.app",
        )
        print("response from litellm.list_fine_tuning_jobs=", ft_jobs)

        # cancel ft job
        response = await litellm.acancel_fine_tuning_job(
            fine_tuning_job_id=create_fine_tuning_response.id,
            custom_llm_provider="azure",
            api_key=os.getenv("AZURE_SWEDEN_API_KEY"),
            api_base="https://exampleopenaiendpoint-production.up.railway.app",
        )

        print("response from litellm.cancel_fine_tuning_job=", response)

        assert response.status == "cancelled"
        assert response.id == create_fine_tuning_response.id
    except openai.RateLimitError:
        pass
    except Exception as e:
        if "Job has already completed" in str(e):
            pass
        else:
            pytest.fail(f"Error occurred: {e}")
    pass


@pytest.mark.asyncio()
async def test_create_vertex_fine_tune_jobs_mocked():
    load_vertex_ai_credentials()
    # Define reusable variables for the test
    project_id = "633608382793"
    location = "us-central1"
    job_id = "3978211980451250176"
    base_model = "gemini-1.0-pro-002"
    tuned_model_name = f"{base_model}-f9259f2c-3fdf-4dd3-9413-afef2bfd24f5"
    training_file = (
        "gs://cloud-samples-data/ai-platform/generative_ai/sft_train_data.jsonl"
    )
    create_time = "2024-12-31T22:40:20.211140Z"

    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json = MagicMock(
        return_value={
            "name": f"projects/{project_id}/locations/{location}/tuningJobs/{job_id}",
            "tunedModelDisplayName": tuned_model_name,
            "baseModel": base_model,
            "supervisedTuningSpec": {"trainingDatasetUri": training_file},
            "state": "JOB_STATE_PENDING",
            "createTime": create_time,
            "updateTime": create_time,
        }
    )

    # Save original callbacks to restore later
    original_callbacks = litellm.callbacks
    # Disable callbacks to avoid Datadog logging interfering with the mock
    litellm.callbacks = []
    
    try:
        with patch(
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
            return_value=mock_response,
        ) as mock_post:
            create_fine_tuning_response = await litellm.acreate_fine_tuning_job(
                model=base_model,
                custom_llm_provider="vertex_ai",
                training_file=training_file,
                vertex_project=project_id,
                vertex_location=location,
            )

            # Verify the request
            mock_post.assert_called_once()

            # Validate the request
            assert mock_post.call_args.kwargs["json"] == {
                "baseModel": base_model,
                "supervisedTuningSpec": {"training_dataset_uri": training_file},
                "tunedModelDisplayName": None,
            }

            # Verify the response
            response_json = json.loads(create_fine_tuning_response.model_dump_json())
            assert (
                response_json["id"]
                == f"projects/{project_id}/locations/{location}/tuningJobs/{job_id}"
            )
            assert response_json["model"] == base_model
            assert response_json["object"] == "fine_tuning.job"
            assert response_json["fine_tuned_model"] == tuned_model_name
            assert response_json["status"] == "queued"
            assert response_json["training_file"] == training_file
            assert (
                response_json["created_at"] == 1735684820
            )  # Unix timestamp for create_time
            assert response_json["error"] is None
            assert response_json["finished_at"] is None
            assert response_json["validation_file"] is None
            assert response_json["trained_tokens"] is None
            assert response_json["estimated_finish"] is None
            assert response_json["integrations"] == []
    finally:
        # Restore original callbacks
        litellm.callbacks = original_callbacks


@pytest.mark.asyncio()
async def test_create_vertex_fine_tune_jobs_mocked_with_hyperparameters():
    load_vertex_ai_credentials()
    # Define reusable variables for the test
    project_id = "633608382793"
    location = "us-central1"
    job_id = "3978211980451250176"
    base_model = "gemini-1.0-pro-002"
    tuned_model_name = f"{base_model}-f9259f2c-3fdf-4dd3-9413-afef2bfd24f5"
    training_file = (
        "gs://cloud-samples-data/ai-platform/generative_ai/sft_train_data.jsonl"
    )
    create_time = "2024-12-31T22:40:20.211140Z"

    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json = MagicMock(
        return_value={
            "name": f"projects/{project_id}/locations/{location}/tuningJobs/{job_id}",
            "tunedModelDisplayName": tuned_model_name,
            "baseModel": base_model,
            "supervisedTuningSpec": {"trainingDatasetUri": training_file},
            "state": "JOB_STATE_PENDING",
            "createTime": create_time,
            "updateTime": create_time,
        }
    )

    # Save original callbacks to restore later
    original_callbacks = litellm.callbacks
    # Disable callbacks to avoid Datadog logging interfering with the mock
    litellm.callbacks = []
    
    try:
        with patch(
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
            return_value=mock_response,
        ) as mock_post:
            create_fine_tuning_response = await litellm.acreate_fine_tuning_job(
                model=base_model,
                custom_llm_provider="vertex_ai",
                training_file=training_file,
                vertex_project=project_id,
                vertex_location=location,
                hyperparameters={
                    "n_epochs": 5,
                    "learning_rate_multiplier": 0.2,
                    "adapter_size": "SMALL",
                },
            )

            # Verify the request
            mock_post.assert_called_once()

            # Validate the request
            assert mock_post.call_args.kwargs["json"] == {
                "baseModel": base_model,
                "supervisedTuningSpec": {
                    "training_dataset_uri": training_file,
                    "hyperParameters": {
                        "epoch_count": 5,
                        "learning_rate_multiplier": 0.2,
                        "adapter_size": "SMALL",
                    },
                },
                "tunedModelDisplayName": None,
            }

            # Verify the response
            response_json = json.loads(create_fine_tuning_response.model_dump_json())
            assert (
                response_json["id"]
                == f"projects/{project_id}/locations/{location}/tuningJobs/{job_id}"
            )
            assert response_json["model"] == base_model
            assert response_json["object"] == "fine_tuning.job"
            assert response_json["fine_tuned_model"] == tuned_model_name
            assert response_json["status"] == "queued"
            assert response_json["training_file"] == training_file
            assert (
                response_json["created_at"] == 1735684820
            )  # Unix timestamp for create_time
            assert response_json["error"] is None
            assert response_json["finished_at"] is None
            assert response_json["validation_file"] is None
            assert response_json["trained_tokens"] is None
            assert response_json["estimated_finish"] is None
            assert response_json["integrations"] == []
    finally:
        # Restore original callbacks
        litellm.callbacks = original_callbacks


# Testing OpenAI -> Vertex AI param mapping


def test_convert_openai_request_to_vertex_basic():
    openai_data = FineTuningJobCreate(
        training_file="gs://bucket/train.jsonl",
        validation_file="gs://bucket/val.jsonl",
        model="text-davinci-002",
        hyperparameters={"n_epochs": 3, "learning_rate_multiplier": 0.1},
        suffix="my_fine_tuned_model",
    )

    result = vertex_finetune_api.convert_openai_request_to_vertex(openai_data)

    print("converted vertex ai result=", json.dumps(result, indent=4))

    assert result["baseModel"] == "text-davinci-002"
    assert result["tunedModelDisplayName"] == "my_fine_tuned_model"
    assert (
        result["supervisedTuningSpec"]["training_dataset_uri"]
        == "gs://bucket/train.jsonl"
    )
    assert (
        result["supervisedTuningSpec"]["validation_dataset"] == "gs://bucket/val.jsonl"
    )
    assert result["supervisedTuningSpec"]["hyperParameters"]["epoch_count"] == 3
    assert (
        result["supervisedTuningSpec"]["hyperParameters"]["learning_rate_multiplier"]
        == 0.1
    )


def test_convert_openai_request_to_vertex_with_adapter_size():
    original_hyperparameters = {
        "n_epochs": 5,
        "learning_rate_multiplier": 0.2,
        "adapter_size": "SMALL",
    }
    openai_data = FineTuningJobCreate(
        training_file="gs://bucket/train.jsonl",
        model="text-davinci-002",
        hyperparameters=Hyperparameters(**original_hyperparameters),
        suffix="custom_model",
    )

    result = vertex_finetune_api.convert_openai_request_to_vertex(
        openai_data, original_hyperparameters=original_hyperparameters
    )

    print("converted vertex ai result=", json.dumps(result, indent=4))

    assert result["baseModel"] == "text-davinci-002"
    assert result["tunedModelDisplayName"] == "custom_model"
    assert (
        result["supervisedTuningSpec"]["training_dataset_uri"]
        == "gs://bucket/train.jsonl"
    )
    assert result["supervisedTuningSpec"]["hyperParameters"]["epoch_count"] == 5
    assert (
        result["supervisedTuningSpec"]["hyperParameters"]["learning_rate_multiplier"]
        == 0.2
    )
    assert result["supervisedTuningSpec"]["hyperParameters"]["adapter_size"] == "SMALL"


def test_convert_basic_openai_request_to_vertex_request():
    openai_data = FineTuningJobCreate(
        training_file="gs://bucket/train.jsonl",
        model="gemini-1.0-pro-002",
    )

    result = vertex_finetune_api.convert_openai_request_to_vertex(
        openai_data,
    )

    print("converted vertex ai result=", json.dumps(result, indent=4))

    assert result["baseModel"] == "gemini-1.0-pro-002"
    assert result["tunedModelDisplayName"] == None
    assert (
        result["supervisedTuningSpec"]["training_dataset_uri"]
        == "gs://bucket/train.jsonl"
    )


@pytest.mark.asyncio()
@pytest.mark.skip(reason="skipping - we run mock tests for vertex ai")
async def test_create_vertex_fine_tune_jobs():
    verbose_logger.setLevel(logging.DEBUG)
    # load_vertex_ai_credentials()

    vertex_credentials = os.getenv("GCS_PATH_SERVICE_ACCOUNT")
    print("creating fine tuning job")
    create_fine_tuning_response = await litellm.acreate_fine_tuning_job(
        model="gemini-1.0-pro-002",
        custom_llm_provider="vertex_ai",
        training_file="gs://cloud-samples-data/ai-platform/generative_ai/sft_train_data.jsonl",
        vertex_project="pathrise-convert-1606954137718",
        vertex_location="us-central1",
        vertex_credentials=vertex_credentials,
    )
    print("vertex ai create fine tuning response=", create_fine_tuning_response)

    assert create_fine_tuning_response.id is not None
    assert create_fine_tuning_response.model == "gemini-1.0-pro-002"
    assert create_fine_tuning_response.object == "fine_tuning.job"


@pytest.mark.asyncio
async def test_mock_openai_create_fine_tune_job():
    """Test that create_fine_tuning_job sends correct parameters to OpenAI"""
    from openai import AsyncOpenAI
    from openai.types.fine_tuning.fine_tuning_job import FineTuningJob, Hyperparameters

    client = AsyncOpenAI(api_key="fake-api-key")

    with patch.object(client.fine_tuning.jobs, "create") as mock_create:
        mock_create.return_value = FineTuningJob(
            id="ft-123",
            model="gpt-3.5-turbo-0125",
            created_at=1677610602,
            status="validating_files",
            fine_tuned_model="ft:gpt-3.5-turbo-0125:org:custom_suffix:id",
            object="fine_tuning.job",
            hyperparameters=Hyperparameters(
                n_epochs=3,
            ),
            organization_id="org-123",
            seed=42,
            training_file="file-123",
            result_files=[],
        )

        response = await litellm.acreate_fine_tuning_job(
            model="gpt-3.5-turbo-0125",
            training_file="file-123",
            hyperparameters={"n_epochs": 3},
            suffix="custom_suffix",
            client=client,
        )

        # Verify the request
        mock_create.assert_called_once()
        request_params = mock_create.call_args.kwargs

        assert request_params["model"] == "gpt-3.5-turbo-0125"
        assert request_params["training_file"] == "file-123"
        assert request_params["hyperparameters"] == {"n_epochs": 3}
        assert request_params["suffix"] == "custom_suffix"

        # Verify the response
        assert response.id == "ft-123"
        assert response.model == "gpt-3.5-turbo-0125"
        assert response.status == "validating_files"
        assert response.fine_tuned_model == "ft:gpt-3.5-turbo-0125:org:custom_suffix:id"


@pytest.mark.asyncio
async def test_mock_openai_list_fine_tune_jobs():
    """Test that list_fine_tuning_jobs sends correct parameters to OpenAI"""
    from openai import AsyncOpenAI
    from unittest.mock import AsyncMock

    client = AsyncOpenAI(api_key="fake-api-key")

    with patch.object(
        client.fine_tuning.jobs, "list", new_callable=AsyncMock
    ) as mock_list:
        # Simple mock return value - actual structure doesn't matter for this test
        mock_list.return_value = []

        await litellm.alist_fine_tuning_jobs(limit=2, after="ft-000", client=client)

        # Only verify that the client was called with correct parameters
        mock_list.assert_called_once()
        request_params = mock_list.call_args.kwargs

        assert request_params["limit"] == 2
        assert request_params["after"] == "ft-000"


@pytest.mark.asyncio
async def test_mock_openai_cancel_fine_tune_job():
    """Test that cancel_fine_tuning_job sends correct parameters to OpenAI"""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key="fake-api-key")

    with patch.object(client.fine_tuning.jobs, "cancel") as mock_cancel:
        try:
            await litellm.acancel_fine_tuning_job(
                fine_tuning_job_id="ft-123", client=client
            )
        except Exception as e:
            print("error=", e)

        # Only verify that the client was called with correct parameters
        mock_cancel.assert_called_once_with(fine_tuning_job_id="ft-123")


@pytest.mark.asyncio
async def test_mock_openai_retrieve_fine_tune_job():
    """Test that retrieve_fine_tuning_job sends correct parameters to OpenAI"""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key="fake-api-key")

    with patch.object(client.fine_tuning.jobs, "retrieve") as mock_retrieve:
        try:
            response = await litellm.aretrieve_fine_tuning_job(
                fine_tuning_job_id="ft-123", client=client
            )
        except Exception as e:
            print("error=", e)


        # Verify the request
        mock_retrieve.assert_called_once_with(fine_tuning_job_id="ft-123")
