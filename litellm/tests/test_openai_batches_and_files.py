# What is this?
## Unit Tests for OpenAI Batches API
import asyncio
import json
import os
import sys
import traceback

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import asyncio
import logging
import time

import pytest

import litellm
from litellm import create_batch, create_file


@pytest.mark.parametrize("provider", ["openai", "azure"])
def test_create_batch(provider):
    """
    1. Create File for Batch completion
    2. Create Batch Request
    3. Retrieve the specific batch
    """
    file_name = "openai_batch_completions.jsonl"
    _current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(_current_dir, file_name)

    file_obj = litellm.create_file(
        file=open(file_path, "rb"),
        purpose="batch",
        custom_llm_provider=provider,
    )
    print("Response from creating file=", file_obj)

    batch_input_file_id = file_obj.id
    assert (
        batch_input_file_id is not None
    ), "Failed to create file, expected a non null file_id but got {batch_input_file_id}"

    time.sleep(5)
    create_batch_response = litellm.create_batch(
        completion_window="24h",
        endpoint="/v1/chat/completions",
        input_file_id=batch_input_file_id,
        custom_llm_provider=provider,
        metadata={"key1": "value1", "key2": "value2"},
    )

    print("response from litellm.create_batch=", create_batch_response)

    assert (
        create_batch_response.id is not None
    ), f"Failed to create batch, expected a non null batch_id but got {create_batch_response.id}"
    assert (
        create_batch_response.endpoint == "/v1/chat/completions"
        or create_batch_response.endpoint == "/chat/completions"
    ), f"Failed to create batch, expected endpoint to be /v1/chat/completions but got {create_batch_response.endpoint}"
    assert (
        create_batch_response.input_file_id == batch_input_file_id
    ), f"Failed to create batch, expected input_file_id to be {batch_input_file_id} but got {create_batch_response.input_file_id}"

    retrieved_batch = litellm.retrieve_batch(
        batch_id=create_batch_response.id, custom_llm_provider=provider
    )
    print("retrieved batch=", retrieved_batch)
    # just assert that we retrieved a non None batch

    assert retrieved_batch.id == create_batch_response.id

    # list all batches
    list_batches = litellm.list_batches(custom_llm_provider=provider, limit=2)
    print("list_batches=", list_batches)

    file_content = litellm.file_content(
        file_id=batch_input_file_id, custom_llm_provider=provider
    )

    result = file_content.content

    result_file_name = "batch_job_results_furniture.jsonl"

    with open(result_file_name, "wb") as file:
        file.write(result)

    pass


@pytest.mark.parametrize("provider", ["openai", "azure"])
@pytest.mark.asyncio()
@pytest.mark.flaky(retries=3, delay=1)
async def test_async_create_batch(provider):
    """
    1. Create File for Batch completion
    2. Create Batch Request
    3. Retrieve the specific batch
    """
    print("Testing async create batch")

    file_name = "openai_batch_completions.jsonl"
    _current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(_current_dir, file_name)
    file_obj = await litellm.acreate_file(
        file=open(file_path, "rb"),
        purpose="batch",
        custom_llm_provider=provider,
    )
    print("Response from creating file=", file_obj)

    await asyncio.sleep(5)
    batch_input_file_id = file_obj.id
    assert (
        batch_input_file_id is not None
    ), "Failed to create file, expected a non null file_id but got {batch_input_file_id}"

    create_batch_response = await litellm.acreate_batch(
        completion_window="24h",
        endpoint="/v1/chat/completions",
        input_file_id=batch_input_file_id,
        custom_llm_provider=provider,
        metadata={"key1": "value1", "key2": "value2"},
    )

    print("response from litellm.create_batch=", create_batch_response)

    assert (
        create_batch_response.id is not None
    ), f"Failed to create batch, expected a non null batch_id but got {create_batch_response.id}"
    assert (
        create_batch_response.endpoint == "/v1/chat/completions"
        or create_batch_response.endpoint == "/chat/completions"
    ), f"Failed to create batch, expected endpoint to be /v1/chat/completions but got {create_batch_response.endpoint}"
    assert (
        create_batch_response.input_file_id == batch_input_file_id
    ), f"Failed to create batch, expected input_file_id to be {batch_input_file_id} but got {create_batch_response.input_file_id}"

    await asyncio.sleep(1)

    retrieved_batch = await litellm.aretrieve_batch(
        batch_id=create_batch_response.id, custom_llm_provider=provider
    )
    print("retrieved batch=", retrieved_batch)
    # just assert that we retrieved a non None batch

    assert retrieved_batch.id == create_batch_response.id

    # list all batches
    list_batches = await litellm.alist_batches(custom_llm_provider=provider, limit=2)
    print("list_batches=", list_batches)

    # try to get file content for our original file

    file_content = await litellm.afile_content(
        file_id=batch_input_file_id, custom_llm_provider=provider
    )

    print("file content = ", file_content)

    # file obj
    file_obj = await litellm.afile_retrieve(
        file_id=batch_input_file_id, custom_llm_provider=provider
    )
    print("file obj = ", file_obj)
    assert file_obj.id == batch_input_file_id

    # delete file
    delete_file_response = await litellm.afile_delete(
        file_id=batch_input_file_id, custom_llm_provider=provider
    )

    print("delete file response = ", delete_file_response)

    assert delete_file_response.id == batch_input_file_id

    all_files_list = await litellm.afile_list(
        custom_llm_provider=provider,
    )

    print("all_files_list = ", all_files_list)

    # # write this file content to a file
    # with open("file_content.json", "w") as f:
    #     json.dump(file_content, f)


def test_retrieve_batch():
    pass


def test_cancel_batch():
    pass


def test_list_batch():
    pass
