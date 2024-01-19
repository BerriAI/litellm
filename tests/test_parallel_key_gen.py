# What this tests ?
## Tests /key/generate by making 10 parallel requests, and asserting all are successful
import pytest
import asyncio
import aiohttp


async def generate_key(session, i):
    url = "http://0.0.0.0:4000/key/generate"
    headers = {"Authorization": "Bearer sk-1234", "Content-Type": "application/json"}
    data = {
        "models": ["azure-models"],
        "aliases": {"mistral-7b": "gpt-3.5-turbo"},
        "duration": None,
    }

    async with session.post(url, headers=headers, json=data) as response:
        status = response.status
        response_text = await response.text()

        print(f"Response {i} (Status code: {status}):")
        print(response_text)
        print()

        if status != 200:
            raise Exception(f"Request {i} did not return a 200 status code: {status}")


@pytest.mark.asyncio
async def test_key_gen():
    async with aiohttp.ClientSession() as session:
        tasks = [generate_key(session, i) for i in range(1, 11)]
        await asyncio.gather(*tasks)
