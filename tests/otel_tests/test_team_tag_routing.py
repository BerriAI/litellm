# What this tests ?
## Set tags on a team and then make a request to /chat/completions
import pytest
import asyncio
import aiohttp, openai
from openai import OpenAI, AsyncOpenAI
from typing import Optional, List, Union
from litellm._uuid import uuid

LITELLM_MASTER_KEY = "sk-1234"


async def chat_completion(
    session, key, model: Union[str, List] = "fake-openai-endpoint"
):
    url = "http://0.0.0.0:4000/chat/completions"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    print("headers=", headers)
    data = {
        "model": model,
        "messages": [
            {"role": "user", "content": f"Hello! {str(uuid.uuid4())}"},
        ],
    }

    async with session.post(url, headers=headers, json=data) as response:
        status = response.status
        response_text = await response.text()

        if status != 200:
            raise Exception(response_text)

        return await response.json(), response.headers


async def model_info_get_call(session, key, model_id: str):
    # make get call pass "litellm_model_id" in query params
    url = f"http://0.0.0.0:4000/model/info?litellm_model_id={model_id}"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    async with session.get(url, headers=headers) as response:
        status = response.status
        response_text = await response.text()

        if status != 200:
            raise Exception(response_text)

        return await response.json()


@pytest.mark.asyncio()
async def test_chat_completion_with_no_tags():
    async with aiohttp.ClientSession() as session:
        key = LITELLM_MASTER_KEY
        response, headers = await chat_completion(session, key)
        headers = dict(headers)
        print(response)
        print(headers)
        assert response is not None
