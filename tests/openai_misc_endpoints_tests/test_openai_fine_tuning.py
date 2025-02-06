from openai import AsyncOpenAI
import os
import pytest
import asyncio


@pytest.mark.asyncio
async def test_openai_fine_tuning():
    """
    [PROD Test] e2e tests for /fine_tuning/jobs endpoints
    """
    client = AsyncOpenAI(api_key="sk-1234", base_url="http://0.0.0.0:4000")

    file_name = "openai_fine_tuning.jsonl"
    _current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(_current_dir, file_name)

    response = await client.files.create(
        extra_body={"custom_llm_provider": "azure"},
        file=open(file_path, "rb"),
        purpose="fine-tune",
    )

    print("response from files.create: {}".format(response))

    await asyncio.sleep(5)

    # create fine tuning job

    ft_job = await client.fine_tuning.jobs.create(
        model="gpt-35-turbo-0613",
        training_file=response.id,
        extra_body={"custom_llm_provider": "azure"},
    )

    print("response from ft job={}".format(ft_job))

    # response from example endpoint
    assert ft_job.id is not None

    # list all fine tuning jobs
    list_ft_jobs = await client.fine_tuning.jobs.list(
        extra_query={"custom_llm_provider": "azure"}
    )

    print("list of ft jobs={}".format(list_ft_jobs))

    # cancel specific fine tuning job
    cancel_ft_job = await client.fine_tuning.jobs.cancel(
        fine_tuning_job_id=ft_job.id,
        extra_body={"custom_llm_provider": "azure"},
    )

    print("response from cancel ft job={}".format(cancel_ft_job))

    assert cancel_ft_job.id is not None

    # delete OG file
    await client.files.delete(
        file_id=response.id,
        extra_body={"custom_llm_provider": "azure"},
    )
