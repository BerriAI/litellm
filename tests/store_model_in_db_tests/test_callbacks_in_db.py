"""
PROD TEST - DO NOT Delete this Test

e2e test for langfuse callback in DB 
- Add langfuse callback to DB - with /config/update
- wait 20 seconds for the callback to be loaded into the instance 
- Make a /chat/completions request to the proxy
- Check if the request is logged in Langfuse
"""

import pytest
import asyncio
import aiohttp
import os
import dotenv
from dotenv import load_dotenv
import pytest
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion

load_dotenv()

# used for testing
LANGFUSE_BASE_URL = "https://exampleopenaiendpoint-production-c715.up.railway.app"


async def config_update(session, routing_strategy=None):
    url = "http://0.0.0.0:4000/config/update"
    headers = {"Authorization": "Bearer sk-1234", "Content-Type": "application/json"}
    print("routing_strategy: ", routing_strategy)
    data = {
        "litellm_settings": {"success_callback": ["langfuse"]},
        "environment_variables": {
            "LANGFUSE_PUBLIC_KEY": "any-public-key",
            "LANGFUSE_SECRET_KEY": "any-secret-key",
            "LANGFUSE_HOST": LANGFUSE_BASE_URL,
        },
    }

    async with session.post(url, headers=headers, json=data) as response:
        status = response.status
        response_text = await response.text()

        print(response_text)
        print("status: ", status)

        if status != 200:
            raise Exception(f"Request did not return a 200 status code: {status}")
        return await response.json()


async def check_langfuse_request(response_id: str):
    async with aiohttp.ClientSession() as session:
        url = f"{LANGFUSE_BASE_URL}/langfuse/trace/{response_id}"
        async with session.get(url) as response:
            response_json = await response.json()
            assert response.status == 200, f"Expected status 200, got {response.status}"
            assert (
                response_json["exists"] == True
            ), f"Request {response_id} not found in Langfuse traces"
            assert response_json["request_id"] == response_id, f"Request ID mismatch"


async def make_chat_completions_request() -> ChatCompletion:
    client = AsyncOpenAI(api_key="sk-1234", base_url="http://0.0.0.0:4000")
    response = await client.chat.completions.create(
        model="fake-openai-endpoint",
        messages=[{"role": "user", "content": "Hello, world!"}],
    )
    print(response)
    return response


@pytest.mark.asyncio
async def test_e2e_langfuse_callbacks_in_db():

    session = aiohttp.ClientSession()

    # add langfuse callback to DB
    await config_update(session)

    # wait 20 seconds for the callback to be loaded into the instance
    await asyncio.sleep(20)

    # make a /chat/completions request to the proxy
    response = await make_chat_completions_request()
    print(response)
    response_id = response.id
    print("response_id: ", response_id)

    await asyncio.sleep(11)
    # check if the request is logged in Langfuse
    await check_langfuse_request(response_id)
