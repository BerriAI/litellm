# What this tests ?
## Tests /batches endpoints
import pytest
import asyncio
import aiohttp, openai
from openai import OpenAI, AsyncOpenAI
from typing import Optional, List, Union
from test_openai_files_endpoints import upload_file, delete_file


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
