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

from litellm import create_fine_tuning_job
from litellm._logging import verbose_logger


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
            api_key=os.getenv("AZURE_SWEDEN_API_KEY"),
            api_base="https://my-endpoint-sweden-berri992.openai.azure.com/",
        )

        print(
            "response from litellm.create_fine_tuning_job=", create_fine_tuning_response
        )

        assert create_fine_tuning_response.id is not None
        assert create_fine_tuning_response.model == "gpt-35-turbo-1106"

        # list fine tuning jobs
        print("listing ft jobs")
        ft_jobs = await litellm.alist_fine_tuning_jobs(
            limit=2,
            custom_llm_provider="azure",
            api_key=os.getenv("AZURE_SWEDEN_API_KEY"),
            api_base="https://my-endpoint-sweden-berri992.openai.azure.com/",
        )
        print("response from litellm.list_fine_tuning_jobs=", ft_jobs)

        # cancel ft job
        response = await litellm.acancel_fine_tuning_job(
            fine_tuning_job_id=create_fine_tuning_response.id,
            custom_llm_provider="azure",
            api_key=os.getenv("AZURE_SWEDEN_API_KEY"),
            api_base="https://my-endpoint-sweden-berri992.openai.azure.com/",
        )

        print("response from litellm.cancel_fine_tuning_job=", response)

        assert response.status == "cancelled"
        assert response.id == create_fine_tuning_response.id
    except openai.RateLimitError:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")
    pass
