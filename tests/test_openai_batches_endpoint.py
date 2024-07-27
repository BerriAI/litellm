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


@pytest.mark.asyncio
async def test_file_operations():
    async with aiohttp.ClientSession() as session:
        # Test file upload and get file_id
        file_id = await upload_file(session, purpose="batch")

        batch_id = await create_batch(session, file_id, "/v1/chat/completions", "24h")
        assert batch_id is not None

        # Test delete file
        await delete_file(session, file_id)
