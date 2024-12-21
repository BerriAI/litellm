# What this tests ?
## Tests /batches endpoints
import pytest
import asyncio
import aiohttp, openai
from openai import OpenAI, AsyncOpenAI
from typing import Optional, List, Union
from test_openai_files_endpoints import upload_file, delete_file
import os
import sys
import time


BASE_URL = "http://localhost:4000"  # Replace with your actual base URL
API_KEY = "sk-1234"  # Replace with your actual API key


async def create_batch(session, input_file_id, endpoint, completion_window):
    url = f"{BASE_URL}/v1/batches"
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {
        "input_file_id": input_file_id,
        "endpoint": endpoint,
        "completion_window": completion_window,
    }

    async with session.post(url, headers=headers, json=payload) as response:
        assert response.status == 200, f"Expected status 200, got {response.status}"
        result = await response.json()
        print(f"Batch creation successful. Batch ID: {result.get('id', 'N/A')}")
        return result


async def get_batch_by_id(session, batch_id):
    url = f"{BASE_URL}/v1/batches/{batch_id}"
    headers = {"Authorization": f"Bearer {API_KEY}"}

    async with session.get(url, headers=headers) as response:
        if response.status == 200:
            result = await response.json()
            return result
        else:
            print(f"Error: Failed to get batch. Status code: {response.status}")
            return None


async def list_batches(session):
    url = f"{BASE_URL}/v1/batches"
    headers = {"Authorization": f"Bearer {API_KEY}"}

    async with session.get(url, headers=headers) as response:
        if response.status == 200:
            result = await response.json()
            return result
        else:
            print(f"Error: Failed to get batch. Status code: {response.status}")
            return None


@pytest.mark.asyncio
async def test_batches_operations():
    async with aiohttp.ClientSession() as session:
        # Test file upload and get file_id
        file_id = await upload_file(session, purpose="batch")

        create_batch_response = await create_batch(
            session, file_id, "/v1/chat/completions", "24h"
        )
        batch_id = create_batch_response.get("id")
        assert batch_id is not None

        # Test get batch
        get_batch_response = await get_batch_by_id(session, batch_id)
        print("response from get batch", get_batch_response)

        assert get_batch_response["id"] == batch_id
        assert get_batch_response["input_file_id"] == file_id

        # test LIST Batches
        list_batch_response = await list_batches(session)
        print("response from list batch", list_batch_response)

        assert list_batch_response is not None
        assert len(list_batch_response["data"]) > 0

        element_0 = list_batch_response["data"][0]
        assert element_0["id"] is not None

        # Test delete file
        await delete_file(session, file_id)


from openai import OpenAI

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)


def create_batch_oai_sdk(filepath) -> str:
    batch_input_file = client.files.create(file=open(filepath, "rb"), purpose="batch")
    batch_input_file_id = batch_input_file.id

    rq = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": filepath,
        },
    )

    print(f"Batch submitted. ID: {rq.id}")
    return rq.id


def await_batch_completion(batch_id: str):
    while True:
        batch = client.batches.retrieve(batch_id)
        if batch.status == "completed":
            print(f"Batch {batch_id} completed.")
            return

        print("waiting for batch to complete...")
        time.sleep(10)


def write_content_to_file(batch_id: str, output_path: str) -> str:
    batch = client.batches.retrieve(batch_id)
    content = client.files.content(batch.output_file_id)
    print("content from files.content", content.content)
    content.write_to_file(output_path)


import jsonlines


def read_jsonl(filepath: str):
    results = []
    with jsonlines.open(filepath) as f:
        for line in f:
            results.append(line)

    for item in results:
        print(item)
        custom_id = item["custom_id"]
        print(custom_id)


def test_e2e_batches_files():
    """
    [PROD Test] Ensures OpenAI Batches + files work with OpenAI SDK
    """
    input_path = "input.jsonl"
    output_path = "out.jsonl"

    _current_dir = os.path.dirname(os.path.abspath(__file__))
    input_file_path = os.path.join(_current_dir, input_path)
    output_file_path = os.path.join(_current_dir, output_path)

    batch_id = create_batch_oai_sdk(input_file_path)
    await_batch_completion(batch_id)
    write_content_to_file(batch_id, output_file_path)
    read_jsonl(output_file_path)


@pytest.mark.skip(reason="Local only test to verify if things work well")
def test_vertex_batches_endpoint():
    """
    Test VertexAI Batches Endpoint
    """
    import os

    oai_client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    file_name = "local_testing/vertex_batch_completions.jsonl"
    _current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(_current_dir, file_name)
    file_obj = oai_client.files.create(
        file=open(file_path, "rb"),
        purpose="batch",
        extra_body={"custom_llm_provider": "vertex_ai"},
    )
    print("Response from creating file=", file_obj)

    batch_input_file_id = file_obj.id
    assert (
        batch_input_file_id is not None
    ), f"Failed to create file, expected a non null file_id but got {batch_input_file_id}"

    create_batch_response = oai_client.batches.create(
        completion_window="24h",
        endpoint="/v1/chat/completions",
        input_file_id=batch_input_file_id,
        extra_body={"custom_llm_provider": "vertex_ai"},
        metadata={"key1": "value1", "key2": "value2"},
    )
    print("response from create batch", create_batch_response)
    pass
