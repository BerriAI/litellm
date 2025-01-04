"""
Test Basic LLM Calling

on a production proxy config
"""

from typing import List, Dict

import aiohttp
import pytest


async def generate_key(session):
    url = "http://0.0.0.0:4000/key/generate"
    headers = {"Authorization": "Bearer sk-1234", "Content-Type": "application/json"}
    async with session.post(url, headers=headers, json={}) as response:
        if response.status != 200:
            raise Exception(
                f"Key generation failed with status {response.status}: {await response.text()}"
            )
        data = await response.json()
        return data["key"]


async def chat_completion(
    session, key: str, model: str, messages: List[Dict[str, str]]
):
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
        if response.status != 200:
            raise Exception(
                f"Request failed with status {response.status}: {await response.text()}"
            )
        return await response.json()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model",
    [
        "gpt-3.5-turbo",
        "openai/gpt-4o",
        "azure/chatgpt-v-2",
        "anthropic/claude-3-5-sonnet-20241022",
        "claude-3-5-sonnet-20241022",
    ],
)
async def test_basic_llm_calling(model):
    async with aiohttp.ClientSession() as session:
        # Generate a new key
        try:
            key = await generate_key(session)
            print(f"Generated new key: {key}")
        except Exception as e:
            pytest.fail(f"Failed to generate key: {str(e)}")

        try:
            response = await chat_completion(
                session,
                key,
                model,
                [{"role": "user", "content": "Hello, how are you?"}],
            )
            print(f"Response for {model}:", response)
            assert "choices" in response
            assert len(response["choices"]) > 0
            assert "message" in response["choices"][0]
            assert "content" in response["choices"][0]["message"]
            print(f"Test passed for {model}")
        except Exception as e:
            pytest.fail(f"Error testing {model}: {str(e)}")
