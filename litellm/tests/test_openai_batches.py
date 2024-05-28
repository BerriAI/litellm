# What is this?
## Unit Tests for OpenAI Batches API
import sys, os, json
import traceback
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import pytest, logging, asyncio
import litellm
from litellm import (
    create_batch,
    create_file,
)
import time


def test_create_batch():
    """
    1. Create File for Batch completion
    2. Create Batch Request
    3. Retrieve the specific batch
    """
    file_obj = litellm.create_file(
        file=open("openai_batch_completions.jsonl", "rb"),
        purpose="batch",
        custom_llm_provider="openai",
    )
    print("Response from creating file=", file_obj)

    batch_input_file_id = file_obj.id
    assert (
        batch_input_file_id is not None
    ), "Failed to create file, expected a non null file_id but got {batch_input_file_id}"

    create_batch_response = litellm.create_batch(
        completion_window="24h",
        endpoint="/v1/chat/completions",
        input_file_id=batch_input_file_id,
        custom_llm_provider="openai",
        metadata={"key1": "value1", "key2": "value2"},
    )

    print("response from litellm.create_batch=", create_batch_response)

    assert (
        create_batch_response.id is not None
    ), f"Failed to create batch, expected a non null batch_id but got {create_batch_response.id}"
    assert (
        create_batch_response.endpoint == "/v1/chat/completions"
    ), f"Failed to create batch, expected endpoint to be /v1/chat/completions but got {create_batch_response.endpoint}"
    assert (
        create_batch_response.input_file_id == batch_input_file_id
    ), f"Failed to create batch, expected input_file_id to be {batch_input_file_id} but got {create_batch_response.input_file_id}"

    time.sleep(30)

    retrieved_batch = litellm.retrieve_batch(
        batch_id=create_batch_response.id, custom_llm_provider="openai"
    )
    print("retrieved batch=", retrieved_batch)
    # just assert that we retrieved a non None batch

    assert retrieved_batch.id == create_batch_response.id

    pass


def test_retrieve_batch():
    pass


def test_cancel_batch():
    pass


def test_list_batch():
    pass
