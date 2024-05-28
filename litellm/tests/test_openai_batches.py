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


def test_create_batch():
    """
    1. Create File for Batch completion
    2. Create Batch Request
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

    print("response from creating file=", file_obj)
    # response = create_batch(
    #     completion_window="24h",
    #     endpoint="/v1/chat/completions",
    #     input_file_id="1",
    #     custom_llm_provider="openai",
    #     metadata={"key1": "value1", "key2": "value2"},
    # )

    print("response")
    pass


def test_retrieve_batch():
    pass


def test_cancel_batch():
    pass


def test_list_batch():
    pass
