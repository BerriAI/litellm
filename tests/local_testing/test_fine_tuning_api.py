import os
import sys
import traceback

import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
from openai import APITimeoutError as Timeout

import litellm

litellm.num_retries = 0
import asyncio
import logging

import openai
from test_gcs_bucket import load_vertex_ai_credentials

from litellm import create_fine_tuning_job
from litellm._logging import verbose_logger
from litellm.llms.fine_tuning_apis.vertex_ai import (
    FineTuningJobCreate,
    VertexFineTuningAPI,
)

vertex_finetune_api = VertexFineTuningAPI()


def test_create_fine_tune_job():
    try:
        verbose_logger.setLevel(logging.DEBUG)
        file_name = "openai_batch_completions.jsonl"
        _current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(_current_dir, file_name)

        file_obj = litellm.create_file(
            file=open(file_path, "rb"),
            purpose="fine-tune",
            custom_llm_provider="openai",
        )
        print("Response from creating file=", file_obj)

        create_fine_tuning_response = litellm.create_fine_tuning_job(
            model="gpt-3.5-turbo-0125",
            training_file=file_obj.id,
        )

        print(
            "response from litellm.create_fine_tuning_job=", create_fine_tuning_response
        )

        assert create_fine_tuning_response.id is not None
        assert create_fine_tuning_response.model == "gpt-3.5-turbo-0125"

        # list fine tuning jobs
        print("listing ft jobs")
        ft_jobs = litellm.list_fine_tuning_jobs(limit=2)
        print("response from litellm.list_fine_tuning_jobs=", ft_jobs)

        assert len(list(ft_jobs)) > 0

        # delete file

        litellm.file_delete(
            file_id=file_obj.id,
        )

        # cancel ft job
        response = litellm.cancel_fine_tuning_job(
            fine_tuning_job_id=create_fine_tuning_response.id,
        )

        print("response from litellm.cancel_fine_tuning_job=", response)

        assert response.status == "cancelled"
        assert response.id == create_fine_tuning_response.id
        pass
    except openai.RateLimitError:
        pass
    except Exception as e:
        if "Job has already completed" in str(e):
            return
        else:
            pytest.fail(f"Error occurred: {e}")


@pytest.mark.asyncio
async def test_create_fine_tune_jobs_async():
    try:
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

        # list fine tuning jobs
        print("listing ft jobs")
        ft_jobs = await litellm.alist_fine_tuning_jobs(limit=2)
        print("response from litellm.list_fine_tuning_jobs=", ft_jobs)
        assert len(list(ft_jobs)) > 0

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
@pytest.mark.skip(reason="skipping until we can cancel fine tuning jobs")
async def test_create_vertex_fine_tune_jobs():
    try:
        verbose_logger.setLevel(logging.DEBUG)
        load_vertex_ai_credentials()

        vertex_credentials = os.getenv("GCS_PATH_SERVICE_ACCOUNT")
        print("creating fine tuning job")
        create_fine_tuning_response = await litellm.acreate_fine_tuning_job(
            model="gemini-1.0-pro-002",
            custom_llm_provider="vertex_ai",
            training_file="gs://cloud-samples-data/ai-platform/generative_ai/sft_train_data.jsonl",
            vertex_project="adroit-crow-413218",
            vertex_location="us-central1",
            vertex_credentials=vertex_credentials,
        )
        print("vertex ai create fine tuning response=", create_fine_tuning_response)

        assert create_fine_tuning_response.id is not None
        assert create_fine_tuning_response.model == "gemini-1.0-pro-002"
        assert create_fine_tuning_response.object == "fine_tuning.job"
    except:
        pass


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

    print("converted vertex ai result=", result)

    assert result["baseModel"] == "text-davinci-002"
    assert result["tunedModelDisplayName"] == "my_fine_tuned_model"
    assert (
        result["supervisedTuningSpec"]["training_dataset_uri"]
        == "gs://bucket/train.jsonl"
    )
    assert (
        result["supervisedTuningSpec"]["validation_dataset"] == "gs://bucket/val.jsonl"
    )
    assert result["supervisedTuningSpec"]["epoch_count"] == 3
    assert result["supervisedTuningSpec"]["learning_rate_multiplier"] == 0.1


def test_convert_openai_request_to_vertex_with_adapter_size():
    openai_data = FineTuningJobCreate(
        training_file="gs://bucket/train.jsonl",
        model="text-davinci-002",
        hyperparameters={"n_epochs": 5, "learning_rate_multiplier": 0.2},
        suffix="custom_model",
    )

    result = vertex_finetune_api.convert_openai_request_to_vertex(
        openai_data, adapter_size="SMALL"
    )

    print("converted vertex ai result=", result)

    assert result["baseModel"] == "text-davinci-002"
    assert result["tunedModelDisplayName"] == "custom_model"
    assert (
        result["supervisedTuningSpec"]["training_dataset_uri"]
        == "gs://bucket/train.jsonl"
    )
    assert result["supervisedTuningSpec"]["validation_dataset"] is None
    assert result["supervisedTuningSpec"]["epoch_count"] == 5
    assert result["supervisedTuningSpec"]["learning_rate_multiplier"] == 0.2
    assert result["supervisedTuningSpec"]["adapter_size"] == "SMALL"
