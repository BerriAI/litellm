# What this tests ?
## Tests /chat/completions by generating a key and then making a chat completions request
import pytest
import asyncio
import aiohttp, openai
from openai import OpenAI, AsyncOpenAI
from typing import Optional, List, Union


BASE_URL = "http://localhost:4000"  # Replace with your actual base URL
API_KEY = "sk-1234"  # Replace with your actual API key


@pytest.mark.asyncio
async def test_file_operations():
    async with aiohttp.ClientSession() as session:
        # Test file upload and get file_id
        file_id = await upload_file(session)

        # Test list files
        await list_files(session)

        # Test get file
        await get_file(session, file_id)

        # Test get file content
        await get_file_content(session, file_id)

        # Test delete file
        await delete_file(session, file_id)


async def upload_file(session, purpose="fine-tune"):
    url = f"{BASE_URL}/v1/files"
    headers = {"Authorization": f"Bearer {API_KEY}"}
    data = aiohttp.FormData()
    data.add_field("purpose", purpose)
    data.add_field(
        "file", b'{"prompt": "Hello", "completion": "Hi"}', filename="mydata.jsonl"
    )

    async with session.post(url, headers=headers, data=data) as response:
        assert response.status == 200
        result = await response.json()
        assert "id" in result
        print(f"File upload successful. File ID: {result['id']}")
        return result["id"]


async def list_files(session):
    url = f"{BASE_URL}/v1/files"
    headers = {"Authorization": f"Bearer {API_KEY}"}

    async with session.get(url, headers=headers) as response:
        assert response.status == 200
        result = await response.json()
        assert "data" in result
        print("List files successful")


async def get_file(session, file_id):
    url = f"{BASE_URL}/v1/files/{file_id}"
    headers = {"Authorization": f"Bearer {API_KEY}"}

    async with session.get(url, headers=headers) as response:
        assert response.status == 200
        result = await response.json()
        assert result["id"] == file_id
        assert result["object"] == "file"
        assert "bytes" in result
        assert "created_at" in result
        assert "filename" in result
        assert result["purpose"] == "fine-tune"
        print(f"Get file successful for file ID: {file_id}")


async def get_file_content(session, file_id):
    url = f"{BASE_URL}/v1/files/{file_id}/content"
    headers = {"Authorization": f"Bearer {API_KEY}"}

    async with session.get(url, headers=headers) as response:
        assert response.status == 200
        content = await response.text()
        assert content  # Check if content is not empty
        print(f"Get file content successful for file ID: {file_id}")


async def delete_file(session, file_id):
    url = f"{BASE_URL}/v1/files/{file_id}"
    headers = {"Authorization": f"Bearer {API_KEY}"}

    async with session.delete(url, headers=headers) as response:
        assert response.status == 200
        result = await response.json()
        assert "deleted" in result
        assert result["id"] == file_id
        print(f"Delete file successful for file ID: {file_id}")
