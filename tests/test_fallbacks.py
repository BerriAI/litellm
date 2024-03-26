# What is this?
## This tests if the proxy fallbacks work as expected
import pytest
import asyncio
import aiohttp
from large_text import text


async def chat_completion(session, key: str, model: str, messages: list):
    url = "http://0.0.0.0:4000/chat/completions"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    data = {
        "model": model,
        "messages": messages,
    }

    async with session.post(url, headers=headers, json=data) as response:
        status = response.status
        response_text = await response.text()

        print(response_text)
        print()

        if status != 200:
            raise Exception(f"Request did not return a 200 status code: {status}")
        return await response.json()


@pytest.mark.asyncio
async def test_chat_completion():
    """
    make chat completion call with prompt > context window. expect it to work with fallback
    """
    async with aiohttp.ClientSession() as session:
        model = "gpt-3.5-turbo"
        messages = [
            {"role": "system", "content": text},
            {"role": "user", "content": "Who was Alexander?"},
        ]
        await chat_completion(
            session=session, key="sk-1234", model=model, messages=messages
        )
