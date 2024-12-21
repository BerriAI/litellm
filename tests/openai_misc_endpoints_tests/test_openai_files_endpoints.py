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
    openai_client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)
    file_content = b'{"prompt": "Hello", "completion": "Hi"}'
    uploaded_file = await openai_client.files.create(
        purpose="fine-tune",
        file=file_content,
    )
    list_files = await openai_client.files.list()
    print("list_files=", list_files)

    get_file = await openai_client.files.retrieve(file_id=uploaded_file.id)
    print("get_file=", get_file)

    get_file_content = await openai_client.files.content(file_id=uploaded_file.id)
    print("get_file_content=", get_file_content.content)

    assert get_file_content.content == file_content
    # try get_file_content.write_to_file
    get_file_content.write_to_file("get_file_content.jsonl")

    delete_file = await openai_client.files.delete(file_id=uploaded_file.id)
    print("delete_file=", delete_file)


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
        print("content from /files/{file_id}/content=", content)
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
